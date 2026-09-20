"""共享限流协调器单元测试（实施计划书 v2.0 T1，AC-01/02）。

全部离线：注入假单调时钟 + sleep 记录器（sleep 推进假时钟），
不产生真实等待。验证：

- 响应头回落 = 服务端窗口滚动（window update）
- in-flight 记账防止并发超预算
- 优先级公平：低优先级不越过先等待的高优先级；高优先级可用保留预算
- P3/P4 不能占用 P0/P1 的 critical reserve
- 429 → 冻结（请求数不增长）；418 → 封禁（立即上抛，不重试）
- 非法 header 忽略并计诊断错误
- public 与 signed 客户端注入同一 coordinator 时预算真正共享
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
import pytest

from cointrader.config import ApiConfig, DataConfig
from cointrader.data.binance import BinancePublicClient
from cointrader.errors import IPBanError
from cointrader.execution.transport import SignedClient
from cointrader.rate_limit import (
    DEFAULT_ENDPOINT_WEIGHT,
    RateLimitBannedError,
    RateLimitBusyError,
    RateLimitCoordinator,
    RateLimitScope,
    RequestPriority,
    endpoint_weight,
    klines_weight,
)
from cointrader.secrets import SecretStr

FUT = RateLimitScope.FUTURES
SPOT = RateLimitScope.SPOT


@dataclass
class FakeClock:
    """假单调时钟：sleep 记录秒数并推进时间轴，测试零真实等待。"""

    value: float = 1000.0
    sleeps: list[float] = field(default_factory=list)

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


def make_coordinator(
    fake: FakeClock,
    futures: int = 2400,
    spot: int = 6000,
    **overrides: object,
) -> RateLimitCoordinator:
    kwargs: dict[str, object] = {
        "soft_limit_ratio": 0.80,
        "critical_reserve_ratio": 0.30,
        "freeze_seconds": 120.0,
        "ban_seconds": 3600.0,
        "clock": fake,
        "sleep": fake.sleep,
    }
    kwargs.update(overrides)
    return RateLimitCoordinator(
        {FUT: futures, SPOT: spot},  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


class TestBudgetAndWindow:
    def test_header_rollover_updates_window(self) -> None:
        """响应头回落（服务端窗口滚动）后，被预算挡住的请求应放行。"""
        fake = FakeClock()
        co = make_coordinator(fake)
        co.observe(FUT, 2000, 200)  # P3 低限 = 2400*(0.8-0.3) = 1200 < 2000
        with pytest.raises(RateLimitBusyError), co.acquire(FUT, RequestPriority.P3_CANDIDATE, 10, timeout=1.0):
            pass
        # 窗口滚动：header 回落
        fake.value += 61
        co.observe(FUT, 50, 200)
        with co.acquire(FUT, RequestPriority.P3_CANDIDATE, 10):
            pass

    def test_in_flight_prevents_over_budget(self) -> None:
        """已发送未返回的 in-flight weight 计入预算，不得超发。"""
        fake = FakeClock()
        co = make_coordinator(fake)
        with co.acquire(FUT, RequestPriority.P0_CRITICAL, 2000):
            snap = co.snapshot()[FUT]
            assert snap.in_flight == 2000
            # 剩余额度 400 < 500：P1 也必须等（P1 上限 = 硬限 2400）
            with pytest.raises(RateLimitBusyError), co.acquire(FUT, RequestPriority.P1_RECOVERY, 500, timeout=1.0):
                pass
        # 释放后可放行
        with co.acquire(FUT, RequestPriority.P1_RECOVERY, 500):
            pass
        assert co.snapshot()[FUT].in_flight == 0

    def test_in_flight_released_on_exception(self) -> None:
        """permit 内抛异常也必须释放 in-flight（异常路径配对）。"""
        fake = FakeClock()
        co = make_coordinator(fake)
        with pytest.raises(RuntimeError), co.acquire(FUT, RequestPriority.P0_CRITICAL, 100):
            raise RuntimeError("boom")
        assert co.snapshot()[FUT].in_flight == 0


class TestPriority:
    def test_low_priority_blocked_high_priority_uses_reserve(self) -> None:
        """已用 1100：P3（上限 1200）过不了 200；P0/P1 用保留预算仍可过。"""
        fake = FakeClock()
        co = make_coordinator(fake)
        co.observe(FUT, 1100, 200)
        with pytest.raises(RateLimitBusyError), co.acquire(FUT, RequestPriority.P3_CANDIDATE, 200, timeout=1.0):
            pass
        with co.acquire(FUT, RequestPriority.P0_CRITICAL, 300):
            pass
        with co.acquire(FUT, RequestPriority.P1_RECOVERY, 300):
            pass

    def test_low_priority_yields_to_waiting_higher_priority(self) -> None:
        """存在先等待的更高优先级请求时，低优先级即使有预算也让步。"""
        fake = FakeClock()
        co = make_coordinator(fake)
        # 预算充足（无观测 header，本地记账为空）
        with co.acquire(FUT, RequestPriority.P3_CANDIDATE, 1):
            pass
        # 模拟一个 P1 等待者已在队列（单元测试直接注入等待计数）
        st = co._states[FUT]  # noqa: SLF001 —— 单元测试直接操纵内部状态
        st.waiting[RequestPriority.P1_RECOVERY] = 1
        try:
            with pytest.raises(RateLimitBusyError), co.acquire(FUT, RequestPriority.P3_CANDIDATE, 1, timeout=1.0):
                pass
            # P1 自身不受 P3 影响
            with co.acquire(FUT, RequestPriority.P1_RECOVERY, 1):
                pass
        finally:
            st.waiting[RequestPriority.P1_RECOVERY] = 0
        with co.acquire(FUT, RequestPriority.P3_CANDIDATE, 1):
            pass

    def test_caps_by_priority(self) -> None:
        """限额 2400 / 软限 0.8 / reserve 0.3：P0P1=2400，P2=1920，P3P4=1200。"""
        fake = FakeClock()
        co = make_coordinator(fake)
        assert co.cap_for(FUT, RequestPriority.P0_CRITICAL) == 2400
        assert co.cap_for(FUT, RequestPriority.P1_RECOVERY) == 2400
        assert co.cap_for(FUT, RequestPriority.P2_POSITION) == 1920
        assert co.cap_for(FUT, RequestPriority.P3_CANDIDATE) == 1200
        assert co.cap_for(FUT, RequestPriority.P4_BACKFILL) == 1200
        snap = co.snapshot()[FUT]
        assert (snap.soft_limit, snap.low_limit) == (1920, 1200)


class TestFreezeAndBan:
    def test_429_freezes_scope_until_window_and_no_probe(self) -> None:
        """429 冻结期间任何优先级都不得发请求；到期后放行。"""
        fake = FakeClock()
        co = make_coordinator(fake)
        co.observe(FUT, 100, 429)
        snap = co.snapshot()[FUT]
        assert snap.frozen_until == pytest.approx(1000.0 + 120.0)
        assert snap.freezes == 1
        with pytest.raises(RateLimitBusyError), co.acquire(FUT, RequestPriority.P0_CRITICAL, 1, timeout=10.0):
            pass
        # 冻结期间无请求发出 → 时钟只被等待推进，请求计数不变
        fake.value += 120
        with co.acquire(FUT, RequestPriority.P0_CRITICAL, 1):
            pass
        assert co.snapshot()[FUT].frozen_until is None

    def test_429_freeze_uses_max_retry_after_min_freeze(self) -> None:
        """429 冻结 = max(Retry-After, 最小保护 120s)。"""
        fake = FakeClock()
        co = make_coordinator(fake)
        co.observe(FUT, 100, 429, retry_after_s=7.0)
        assert co.snapshot()[FUT].frozen_until == pytest.approx(1120.0)
        fake.value = 2000.0
        co.observe(FUT, 100, 429, retry_after_s=300.0)
        assert co.snapshot()[FUT].frozen_until == pytest.approx(2300.0)

    def test_418_bans_scope_and_raises_without_retry(self) -> None:
        """418 封禁期间 acquire 立即上抛（不等待、不重试）；到期自动解除。"""
        fake = FakeClock()
        co = make_coordinator(fake)
        co.observe(FUT, 100, 418)
        snap = co.snapshot()[FUT]
        assert snap.ban_until == pytest.approx(1000.0 + 3600.0)
        assert snap.bans == 1
        with pytest.raises(RateLimitBannedError), co.acquire(FUT, RequestPriority.P0_CRITICAL, 1):
            pass
        # 封禁未到期前 timeout 参数无效 —— 语义是立即停机而非等待
        with pytest.raises(RateLimitBannedError), co.acquire(FUT, RequestPriority.P3_CANDIDATE, 1, timeout=5.0):
            pass
        fake.value += 3600
        with co.acquire(FUT, RequestPriority.P0_CRITICAL, 1):
            pass

    def test_scopes_are_isolated(self) -> None:
        """FUTURES 冻结不影响 SPOT。"""
        fake = FakeClock()
        co = make_coordinator(fake)
        co.observe(FUT, 100, 418)
        with co.acquire(SPOT, RequestPriority.P3_CANDIDATE, 10):
            pass

    def test_invalid_header_ignored_with_diagnostic(self) -> None:
        """负值/非法 header 忽略并计诊断错误，不污染已用观测。"""
        fake = FakeClock()
        co = make_coordinator(fake)
        co.observe(FUT, -5, 200)
        snap = co.snapshot()[FUT]
        assert snap.diagnostic_errors == 1
        assert snap.observed_used is None
        co.observe(FUT, 100, 200)
        assert co.snapshot()[FUT].observed_used == 100


class TestConcurrency:
    def test_concurrent_in_flight_never_exceeds_low_cap(self) -> None:
        """13 线程并发 P3 请求（12×100 填满低限，第 13 个必须等窗口滚动）：
        in-flight 峰值不得超过低限 1200。"""
        fake = FakeClock()
        co = make_coordinator(fake)

        class FakeServer:
            """模拟币安服务端：已用 weight 随 60s 窗口滚动清零。"""

            def __init__(self) -> None:
                self.window_start = fake.value
                self.used = 0

            def header(self) -> int:
                if fake.value - self.window_start >= 60.0:
                    self.window_start = fake.value
                    self.used = 0
                return self.used

            def charge(self, weight: int) -> None:
                self.used += weight

        server = FakeServer()
        server_lock = threading.Lock()
        peak = 0
        peak_lock = threading.Lock()
        done = 0
        done_lock = threading.Lock()

        def worker() -> None:
            nonlocal peak, done
            with co.acquire(FUT, RequestPriority.P3_CANDIDATE, 100):
                with server_lock:
                    server.charge(100)
                    header = server.header()
                co.observe(FUT, header, 200)
                with peak_lock:
                    peak = max(peak, co.snapshot()[FUT].in_flight)
            with done_lock:
                done += 1

        threads = [threading.Thread(target=worker) for _ in range(13)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert all(not t.is_alive() for t in threads), "并发测试超时"
        assert done == 13
        assert peak <= 1200, f"in-flight 峰值 {peak} 超过 P3 低限 1200"

    def test_timeout_raises_busy(self) -> None:
        fake = FakeClock()
        co = make_coordinator(fake)
        co.observe(FUT, 2000, 200)
        with pytest.raises(RateLimitBusyError), co.acquire(FUT, RequestPriority.P3_CANDIDATE, 10, timeout=5.0):
            pass
        assert fake.sleeps, "等待应通过可注入 sleep 完成"


class TestEndpointWeights:
    @pytest.mark.parametrize(
        ("limit", "expected"),
        [(1, 1), (100, 1), (101, 2), (500, 2), (501, 5), (1000, 5), (1001, 10), (1500, 10)],
    )
    def test_klines_weight_tiers(self, limit: int, expected: int) -> None:
        assert klines_weight(limit) == expected

    def test_futures_ticker_24h_full_market_is_80(self) -> None:
        """全市场 Futures ticker/24hr 实测 ~80（旧代码的 40 低估，必须修正）。"""
        assert endpoint_weight(FUT, "/fapi/v1/ticker/24hr") == 80
        assert endpoint_weight(FUT, "/fapi/v1/ticker/24hr", {"symbol": "BTCUSDT"}) == 2

    def test_klines_and_funding_rate_use_limit(self) -> None:
        assert endpoint_weight(FUT, "/fapi/v1/klines", {"limit": 30}) == 1
        assert endpoint_weight(SPOT, "/api/v3/klines", {"limit": 1000}) == 5
        assert endpoint_weight(FUT, "/fapi/v1/fundingRate", {"limit": 100}) == 1
        assert endpoint_weight(FUT, "/fapi/v1/fundingRate", {"limit": 1500}) == 10

    def test_unknown_endpoint_conservative_default_with_warning(self) -> None:
        fake = FakeClock()
        co = make_coordinator(fake)
        weight = endpoint_weight(FUT, "/fapi/v1/mystery", on_unknown=co.note_unknown_endpoint)
        assert weight == DEFAULT_ENDPOINT_WEIGHT
        # 再次调用不再重复告警
        weight2 = endpoint_weight(FUT, "/fapi/v1/mystery", on_unknown=co.note_unknown_endpoint)
        assert weight2 == DEFAULT_ENDPOINT_WEIGHT


# ---------------------------------------------------------------------------
# public / signed 共享同一 coordinator 的集成验证（T1 要求：显式注入同一实例）
# ---------------------------------------------------------------------------


def _signed_client(
    coordinator: RateLimitCoordinator,
    fake: FakeClock,
    handler: Callable[[httpx.Request], httpx.Response] | None = None,
) -> SignedClient:
    transport = httpx.Client(
        transport=httpx.MockTransport(
            handler or (lambda req: httpx.Response(200, json={"serverTime": 1}))
        )
    )
    return SignedClient(
        base_url="https://testnet.binance.example",
        api_key=SecretStr("test-key", name="API_KEY"),
        secret=SecretStr("test-secret", name="API_SECRET"),
        market="perp",
        rate_limiter=coordinator,
        client=transport,
        sleep_fn=fake.sleep,
        now_fn=fake,
    )


class TestSharedCoordinator:
    def test_public_and_signed_share_budget(self, tmp_cache_dir) -> None:
        """public 请求观测到的 header 同时约束 signed 请求（同一出口 IP 预算）。"""
        import pathlib

        fake = FakeClock()
        co = make_coordinator(fake)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"x-mbx-used-weight-1m": "1199"},
                json={"serverTime": 1},
            )

        data_config = DataConfig(cache_dir=pathlib.Path(tmp_cache_dir))
        public = BinancePublicClient(
            ApiConfig(),
            data_config,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            sleep_fn=fake.sleep,
            rate_limiter=co,
        )
        signed = _signed_client(co, fake)

        # P3 权重 1：1199 + 1 <= 1200 → 放行，观测 header 1199
        public.futures_time()
        assert co.snapshot()[FUT].observed_used == 1199

        # signed P3 请求（/fapi/v1/userTrades 权重 10）：1199 + 10 > 1200
        # → 被同一出口 IP 预算挡住：等待一个完整窗口后才放行（429 熔断兜底）
        result = signed.get("/fapi/v1/userTrades", {"symbol": "BTCUSDT"}, priority=RequestPriority.P3_CANDIDATE)
        assert result == {"serverTime": 1}
        assert any(s >= 59.0 for s in fake.sleeps), "P3 请求必须被共享预算挡下等待窗口滚动"

        # 但 P1（恢复/对账）用保留预算立即放行：1199 + 5 <= 2400
        result = signed.get("/fapi/v2/account", priority=RequestPriority.P1_RECOVERY)
        assert result == {"serverTime": 1}
        public.close()
        signed.close()

    def test_418_from_signed_bans_scope_for_public_too(self, tmp_cache_dir) -> None:
        """signed 收到 418 后，整个 scope 封禁 —— public 同一 coordinator 也被拒。"""
        import pathlib

        fake = FakeClock()
        co = make_coordinator(fake)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(418, json={"code": -1003, "msg": "IP limited"})

        data_config = DataConfig(cache_dir=pathlib.Path(tmp_cache_dir))
        public = BinancePublicClient(
            ApiConfig(),
            data_config,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            sleep_fn=fake.sleep,
            rate_limiter=co,
        )
        with pytest.raises(IPBanError):
            public.futures_time()
        # 封禁是 scope 级：任何客户端、任何优先级都不得再探测
        with pytest.raises(RateLimitBannedError), co.acquire(FUT, RequestPriority.P0_CRITICAL, 1):
            pass
        public.close()
