"""transport 传输层单元测试（开发设计文档 §3.5 / §11.1）。

用 httpx.MockTransport 注入假响应，禁止真实网络。
重点：错误分类、critical 请求永不重试、限流退避、敏感信息不落日志/异常。
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from cointrader.errors import (
    AuthError,
    BinanceError,
    ClockError,
    IPBanError,
    NetworkError,
    OrderRejected,
    RateLimitError,
    UnknownSubmission,
)
from cointrader.execution.transport import SignedClient
from cointrader.rate_limit import (
    RateLimitBannedError,
    RateLimitCoordinator,
    RateLimitScope,
    RequestPriority,
)
from cointrader.secrets import SecretStr

SECRET = "test-secret-do-not-log"
KEY = "test-api-key"


def make_client(
    handler: Callable[[httpx.Request], httpx.Response], **overrides: object
) -> tuple[SignedClient, _RecordingSleep]:
    transport = httpx.MockTransport(handler)
    sleeper = _RecordingSleep()
    kwargs: dict[str, object] = {
        "base_url": "https://testnet.binance.example",
        "api_key": SecretStr(KEY, name="API_KEY"),
        "secret": SecretStr(SECRET, name="API_SECRET"),
        "market": "spot",
        "client": httpx.Client(transport=transport),
        "sleep_fn": sleeper,
        "now_fn": sleeper.now,
    }
    kwargs.update(overrides)
    return SignedClient(**kwargs), sleeper  # type: ignore[arg-type]


class _RecordingSleep:
    """记录退避调用，避免测试真实休眠；同时推进假时钟（与注入的 coordinator 共享）。"""

    def __init__(self) -> None:
        self.calls: list[float] = []
        self.value: float = 1_700_000_000.0

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.value += seconds

    def now(self) -> float:
        return self.value


def make_coordinator(sleeper: _RecordingSleep, **overrides: object) -> RateLimitCoordinator:
    """测试用共享 coordinator：时钟/sleep 与 _RecordingSleep 同步。"""
    kwargs: dict[str, object] = {
        "soft_limit_ratio": 0.80,
        "critical_reserve_ratio": 0.30,
        "freeze_seconds": 120.0,
        "ban_seconds": 3600.0,
        "clock": sleeper.now,
        "sleep": sleeper,
    }
    kwargs.update(overrides)
    return RateLimitCoordinator(
        {RateLimitScope.SPOT: 2400, RateLimitScope.FUTURES: 2400},  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _resp(status: int, body: dict | None = None, headers: dict[str, str] | None = None) -> httpx.Response:
    import json

    return httpx.Response(
        status,
        content=json.dumps(body or {}).encode(),
        headers=headers or {},
    )


class TestErrorClassification:
    def test_400_order_rejected_code_gives_order_rejected(self) -> None:
        c, _ = make_client(lambda req: _resp(400, {"code": -2010, "msg": "New order rejected"}))
        with pytest.raises(OrderRejected) as exc_info:
            c.get("/api/v3/order")
        assert exc_info.value.code == -2010

    def test_400_unknown_code_gives_binance_error(self) -> None:
        c, _ = make_client(lambda req: _resp(400, {"code": -1111, "msg": "weird"}))
        with pytest.raises(BinanceError) as exc_info:
            c.get("/api/v3/order")
        assert not isinstance(exc_info.value, OrderRejected)

    def test_401_gives_auth_error_and_counts_stat(self) -> None:
        c, _ = make_client(lambda req: _resp(401, {"code": -2014, "msg": "Bad API key"}))
        with pytest.raises(AuthError):
            c.get("/api/v3/account")
        assert c.stats.auth_errors == 1

    def test_403_gives_auth_error(self) -> None:
        c, _ = make_client(lambda req: _resp(403, {"code": -2015, "msg": "no perm"}))
        with pytest.raises(AuthError):
            c.get("/api/v3/account")

    def test_minus_1021_gives_clock_error(self) -> None:
        c, _ = make_client(lambda req: _resp(400, {"code": -1021, "msg": "timestamp outside recvWindow"}))
        with pytest.raises(ClockError) as exc_info:
            c.get("/api/v3/account")
        assert exc_info.value.code == -1021

    def test_minus_1022_gives_auth_error(self) -> None:
        c, _ = make_client(lambda req: _resp(400, {"code": -1022, "msg": "Bad signature"}))
        with pytest.raises(AuthError):
            c.get("/api/v3/account")

    def test_418_gives_ip_ban_error(self) -> None:
        c, _ = make_client(lambda req: _resp(418, {"code": -1003, "msg": "IP limited"}))
        with pytest.raises(IPBanError):
            c.get("/api/v3/account")

    def test_200_returns_body(self) -> None:
        c, _ = make_client(lambda req: _resp(200, {"serverTime": 123}))
        assert c.get("/api/v3/time") == {"serverTime": 123}


class TestCriticalNeverRetries:
    def test_5xx_on_place_order_raises_unknown_submission_once(self) -> None:
        calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            return _resp(500, {"code": -1003, "msg": "server down"})

        c, _ = make_client(handler)
        with pytest.raises(UnknownSubmission):
            c.place_order("/api/v3/order", {"symbol": "BTCUSDT"}, client_order_id="ct-test-1")
        assert len(calls) == 1, "critical 请求禁止自动重发（可能已成交）"
        assert c.stats.unknown_submissions == 1

    def test_408_on_critical_raises_unknown_submission(self) -> None:
        c, _ = make_client(lambda req: _resp(408, {}))
        with pytest.raises(UnknownSubmission):
            c.execute_critical("POST", "/fapi/v1/order", {"symbol": "BTCUSDT"}, client_order_id="ct-1")

    def test_timeout_on_critical_raises_unknown_submission_not_network(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("timeout")

        c, _ = make_client(handler)
        with pytest.raises(UnknownSubmission) as exc_info:
            c.place_order("/api/v3/order", {"symbol": "BTCUSDT"}, client_order_id="ct-timeout")
        assert exc_info.value.client_order_id == "ct-timeout"

    def test_429_on_critical_raises_rate_limit_error_without_retry(self) -> None:
        calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            return _resp(429, {"code": -1003, "msg": "too many"}, headers={"Retry-After": "2"})

        c, _ = make_client(handler)
        with pytest.raises(RateLimitError):
            c.place_order("/api/v3/order", {"symbol": "BTCUSDT"}, client_order_id="ct-rl")
        assert len(calls) == 1, "429 = 请求未提交，但 critical 仍不自动重试，上抛决策"

    def test_cancel_order_critical_5xx_is_unknown(self) -> None:
        c, _ = make_client(lambda req: _resp(502, {}))
        with pytest.raises(UnknownSubmission):
            c.cancel_order("/api/v3/order", {"symbol": "BTCUSDT"}, client_order_id="ct-c")


class TestReadRequestRetry:
    def test_5xx_retries_then_succeeds(self) -> None:
        calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            if len(calls) < 3:
                return _resp(503, {})
            return _resp(200, {"ok": True})

        c, _ = make_client(handler, read_max_retries=3)
        assert c.get("/api/v3/time") == {"ok": True}
        assert len(calls) == 3
        assert len(c._sleep.calls) == 2, "每次重试前应退避"  # noqa: SLF001

    def test_5xx_retries_exhausted_raises(self) -> None:
        calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            return _resp(500, {})

        c, _ = make_client(handler, read_max_retries=2)
        with pytest.raises(BinanceError):
            c.get("/api/v3/time")
        assert len(calls) == 3

    def test_429_respects_retry_after_header(self) -> None:
        calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            if len(calls) == 1:
                return _resp(429, {}, headers={"Retry-After": "7"})
            return _resp(200, {"ok": 1})

        c, sleeper = make_client(handler, read_max_retries=2)
        c.coordinator = make_coordinator(sleeper)
        c.get("/api/v3/time")
        assert sleeper.calls[0] == 7.0, "必须尊重 Retry-After 头"
        assert c.stats.rate_limit_hits == 1

    def test_429_freezes_coordinator_for_min_protection(self) -> None:
        """429 后 coordinator 冻结至少 120s；重试必须等冻结解除后才会发。"""
        calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            if len(calls) == 1:
                return _resp(429, {})
            return _resp(200, {"ok": 1})

        c, sleeper = make_client(handler, read_max_retries=2)
        c.coordinator = make_coordinator(sleeper)
        c.get("/api/v3/time")
        # 重试确实等到冻结解除后才发出（共两次请求，中间等待 >= 119s 假时间）
        assert len(calls) == 2
        assert sum(sleeper.calls) >= 119.0, f"冻结期内不得立即重试: {sleeper.calls}"
        assert c.coordinator.snapshot()[RateLimitScope.SPOT].freezes == 1

    def test_network_timeout_retries_then_network_error(self) -> None:
        calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            raise httpx.ReadTimeout("read timeout")

        c, _ = make_client(handler, read_max_retries=1)
        with pytest.raises(NetworkError):
            c.get("/api/v3/time")
        assert len(calls) == 2


class TestWeightThrottle:
    def test_p1_can_use_reserve_but_full_limit_blocks(self) -> None:
        """P1（恢复/对账）可用保留预算；已达硬限时才等待。"""
        def handler(req: httpx.Request) -> httpx.Response:
            return _resp(200, {"ok": 1}, headers={"x-mbx-used-weight-1m": "2390"})

        c, sleeper = make_client(handler, weight_limit=2400, soft_limit_ratio=0.80)
        c.coordinator = make_coordinator(sleeper)
        c.get("/api/v3/time")  # 观测 2390
        c.get("/api/v3/time")  # 2390 + 1 <= 2400（P1 硬限）→ 不等待
        assert c.stats.throttled_seconds == 0

    def test_hard_limit_triggers_wait(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return _resp(200, {"ok": 1}, headers={"x-mbx-used-weight-1m": "2400"})

        c, sleeper = make_client(handler, weight_limit=2400)
        c.coordinator = make_coordinator(sleeper)
        c.get("/api/v3/time")  # 观测 2400
        c.get("/api/v3/time")  # 2400 + 1 > 2400 → 等待窗口滚动（假时钟推进）
        assert c.stats.throttled_seconds > 0, "达硬限时应等待"

    def test_low_priority_cannot_use_reserve(self) -> None:
        """已用 1500：P1 可过（保留预算），P4 回填被挡（上限 1200）。"""
        def handler(req: httpx.Request) -> httpx.Response:
            return _resp(200, {"ok": 1}, headers={"x-mbx-used-weight-1m": "1500"})

        c, sleeper = make_client(handler, weight_limit=2400)
        c.coordinator = make_coordinator(sleeper)
        c.get("/api/v3/time")  # P1 默认：1500+1 <= 2400 放行，观测 1500
        c.get("/api/v3/time", priority=RequestPriority.P1_RECOVERY)  # 仍放行
        # P4 回填：1500 + 1 > 1200 → 等待（假时钟推进后强放行，不断言等待时长）
        c.get("/api/v3/time", priority=RequestPriority.P4_BACKFILL)
        assert any(s >= 59.0 for s in sleeper.calls), "P4 不得占用保留预算"

    def test_418_bans_scope_across_clients(self) -> None:
        """418 后同 scope 的 coordinator 被封禁：任何新请求立即上抛。"""
        c, _ = make_client(lambda req: _resp(418, {"code": -1003, "msg": "IP limited"}))
        with pytest.raises(IPBanError):
            c.get("/api/v3/account")
        with pytest.raises(RateLimitBannedError), c.coordinator.acquire(
            RateLimitScope.SPOT, RequestPriority.P0_CRITICAL, 1
        ):
            pass

    def test_low_weight_does_not_throttle(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return _resp(200, {"ok": 1}, headers={"x-mbx-used-weight-1m": "100"})

        c, _ = make_client(handler, weight_limit=2400)
        c.get("/api/v3/time")
        c.get("/api/v3/time")
        assert c.stats.throttled_seconds == 0


class TestSecretRedaction:
    def test_secret_never_in_url_or_exception(self) -> None:
        seen_urls: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen_urls.append(str(req.url))
            return _resp(401, {"code": -2014, "msg": "bad key"})

        c, _ = make_client(handler)
        with pytest.raises(AuthError) as exc_info:
            c.get("/api/v3/account")
        for url in seen_urls:
            assert SECRET not in url, "Secret 不得出现在请求 URL"
            assert KEY not in url, "API Key 走 header，不得进 URL query"
        text = str(exc_info.value)
        assert SECRET not in text and KEY not in text, "异常信息不得含凭证"

    def test_signature_present_in_query_but_not_secret(self) -> None:
        seen: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(str(req.url))
            return _resp(200, {"ok": 1})

        c, _ = make_client(handler)
        c.get("/api/v3/account")
        assert "signature=" in seen[0]
        assert SECRET not in seen[0]
