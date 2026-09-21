"""数据层测试 —— 缓存、限流、解析。

全部离线运行：用 ``httpx.MockTransport`` 替换真实网络。
这样测试不依赖币安可用性，也不需要网络权限，且能精确构造边界场景
（限流、418、格式异常），这些在真实环境里很难复现。
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from cointrader.config import ApiConfig, DataConfig, RateLimitConfig
from cointrader.data.binance import BinancePublicClient
from cointrader.data.cache import CACHE_FORMAT_VERSION, DiskCache, make_key
from cointrader.data.funding import (
    FundingIntervals,
    annualize_rate,
    consecutive_negative_streak,
    consecutive_positive_streak,
    deannualize_rate,
    fetch_funding_history,
    fetch_funding_intervals,
    normalize_funding_to_8h,
    summarize_funding,
)
from cointrader.data.klines import (
    historical_quote_volume_3d_avg,
    historical_quote_volume_24h,
    klines_to_frame,
    parse_symbol_rules,
    tradable_perpetuals,
)
from cointrader.errors import (
    DataUnavailableError,
    InsufficientDataError,
    IPBanError,
    ParseError,
    RateLimitError,
)
from cointrader.rate_limit import (
    RateLimitCoordinator,
    RateLimitScope,
)
from conftest import make_funding_series


class TestFundingNormalization:
    """跨周期资金费统一到 8h，且不能把未来事件带入更早桶。"""

    def test_normalizes_4h_events_to_right_closed_8h_buckets(self) -> None:
        index = pd.date_range("2025-01-01", periods=4, freq="4h", tz="UTC")
        rates = pd.Series([0.1, 0.2, 0.3, 0.4], index=index, dtype=float)

        normalized = normalize_funding_to_8h(rates, 4)

        assert normalized.index.tolist() == list(
            pd.to_datetime(["2025-01-01 00:00", "2025-01-01 08:00", "2025-01-01 16:00"], utc=True)
        )
        assert normalized.iloc[0] == pytest.approx(0.1)
        assert normalized.iloc[1] == pytest.approx(0.5)
        assert normalized.iloc[2] == pytest.approx(0.4)


class TestHistoricalVolume:
    """历史成交量只在 K 线闭合后可见。"""

    def test_rolling_24h_volume_uses_six_closed_4h_bars(self) -> None:
        index = pd.date_range("2025-01-01", periods=7, freq="4h", tz="UTC")
        frame = pd.DataFrame({"quote_volume": range(1, 8)}, index=index)

        volume = historical_quote_volume_24h(frame)

        assert len(volume) == 2
        assert volume.iloc[0] == pytest.approx(sum(range(1, 7)))
        assert volume.index[0] == pd.Timestamp("2025-01-02 00:00", tz="UTC")

    def test_rolling_3d_average_daily_volume_uses_18_closed_4h_bars(self) -> None:
        index = pd.date_range("2025-01-01", periods=19, freq="4h", tz="UTC")
        frame = pd.DataFrame({"quote_volume": range(1, 20)}, index=index)

        volume = historical_quote_volume_3d_avg(frame)

        assert len(volume) == 2
        assert volume.iloc[0] == pytest.approx(sum(range(1, 19)) / 3)
        assert volume.index[0] == pd.Timestamp("2025-01-04 00:00", tz="UTC")





class TestDiskCache:
    """磁盘缓存。"""

    def test_put_and_get_roundtrip(self, tmp_cache_dir: Path) -> None:
        cache = DiskCache(tmp_cache_dir)
        cache.put("test_ns", "key1", {"value": [1, 2, 3]}, ttl=3600)

        entry = cache.get("test_ns", "key1")
        assert entry is not None
        assert entry.data == {"value": [1, 2, 3]}
        assert entry.is_fresh()

    def test_ttl_expiry(self, tmp_cache_dir: Path) -> None:
        """TTL 过期后 get_fresh 应返回 None。"""
        cache = DiskCache(tmp_cache_dir)
        cache.put("test_ns", "key1", {"v": 1}, ttl=1)

        assert cache.get_fresh("test_ns", "key1") is not None

        # 手工把创建时间往前挪，模拟过期
        path = tmp_cache_dir / "test_ns" / "key1.json"
        payload = json.loads(path.read_text())
        payload["created_at"] = time.time() - 10
        path.write_text(json.dumps(payload))

        assert cache.get_fresh("test_ns", "key1") is None
        # 但原始 get 仍能拿到（只是不新鲜）
        assert cache.get("test_ns", "key1") is not None

    def test_none_ttl_never_expires(self, tmp_cache_dir: Path) -> None:
        """ttl=None 表示永不过期（用于历史数据）。"""
        cache = DiskCache(tmp_cache_dir)
        cache.put("test_ns", "key1", {"v": 1}, ttl=None)

        path = tmp_cache_dir / "test_ns" / "key1.json"
        payload = json.loads(path.read_text())
        payload["created_at"] = time.time() - 10_000_000
        path.write_text(json.dumps(payload))

        assert cache.get_fresh("test_ns", "key1") is not None

    def test_corrupted_cache_returns_none_and_self_heals(self, tmp_cache_dir: Path) -> None:
        """损坏的缓存应被当作未命中并删除，而不是抛异常中断流程。"""
        cache = DiskCache(tmp_cache_dir)
        cache.put("test_ns", "key1", {"v": 1}, ttl=3600)

        path = tmp_cache_dir / "test_ns" / "key1.json"
        path.write_text("{ 这不是合法 JSON")

        assert cache.get("test_ns", "key1") is None
        assert not path.exists(), "损坏的缓存文件应被删除"

    def test_version_mismatch_invalidates(self, tmp_cache_dir: Path) -> None:
        """缓存格式版本不符时应失效（数据结构变更后的自我保护）。"""
        cache = DiskCache(tmp_cache_dir)
        cache.put("test_ns", "key1", {"v": 1}, ttl=3600)

        path = tmp_cache_dir / "test_ns" / "key1.json"
        payload = json.loads(path.read_text())
        payload["v"] = CACHE_FORMAT_VERSION + 999
        path.write_text(json.dumps(payload))

        assert cache.get("test_ns", "key1") is None

    def test_atomic_write_leaves_no_temp_files(self, tmp_cache_dir: Path) -> None:
        """原子写不应留下临时文件。"""
        cache = DiskCache(tmp_cache_dir)
        for i in range(10):
            cache.put("test_ns", f"key{i}", {"v": i}, ttl=3600)

        leftovers = list((tmp_cache_dir / "test_ns").glob("*.tmp"))
        assert not leftovers, f"留下了临时文件: {leftovers}"

    def test_namespace_traversal_rejected(self, tmp_cache_dir: Path) -> None:
        """缓存 namespace/key 不得包含路径穿越字符。"""
        cache = DiskCache(tmp_cache_dir)

        with pytest.raises(ValueError, match="namespace"):
            cache.put("../escape", "key", {})

        with pytest.raises(ValueError, match="key"):
            cache.put("ns", "../../etc/passwd", {})

    def test_disabled_cache_is_noop(self, tmp_cache_dir: Path) -> None:
        cache = DiskCache(tmp_cache_dir, enabled=False)
        cache.put("ns", "key", {"v": 1})
        assert cache.get("ns", "key") is None

    def test_purge_namespace(self, tmp_cache_dir: Path) -> None:
        cache = DiskCache(tmp_cache_dir)
        for i in range(5):
            cache.put("ns", f"key{i}", {"v": i})

        assert cache.purge_namespace("ns") == 5
        assert cache.stats().get("ns", 0) == 0

    def test_make_key_is_order_independent(self) -> None:
        """参数顺序不同但内容相同的调用应生成同一个键。"""
        assert make_key("a", 1, "b") != make_key("a", "b", 1)
        assert make_key({"x": 1, "y": 2}) == make_key({"y": 2, "x": 1})


# ---------------------------------------------------------------------------
# 客户端：限流与错误处理
# ---------------------------------------------------------------------------


def make_client(
    handler: Any,
    *,
    data_config: DataConfig | None = None,
    api_config: ApiConfig | None = None,
    sleep_calls: list[float] | None = None,
) -> BinancePublicClient:
    """构造一个用 mock transport 的客户端（不发真实网络请求）。

    Args:
        handler: MockTransport 的处理器。
        data_config: 数据配置。默认使用极短的退避时间，避免测试变慢。
        api_config: API 配置。
        sleep_calls: 传入列表则记录所有 sleep 调用（而不是真的睡）。
    """
    config = data_config or DataConfig(
        cache_dir=Path(tempfile.mkdtemp(prefix="cointrader_test_")),
        rate_limit=RateLimitConfig(
            base_backoff_seconds=0.001, max_backoff_seconds=0.01, max_retries=2
        ),
    )
    return BinancePublicClient(
        api_config or ApiConfig(),
        config,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep_fn=(sleep_calls.append if sleep_calls is not None else lambda _: None),
    )


class _FakeClock:
    """假单调时钟：sleep 记录并推进时间轴，避免测试真实等待。"""

    def __init__(self) -> None:
        self.value = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


class TestClientErrorHandling:
    """客户端错误分类与重试。"""

    def test_418_raises_ipban_and_does_not_retry(self, tmp_cache_dir: Path) -> None:
        """HTTP 418 必须立即抛 IPBanError，且**不得重试**。

        继续请求会延长封禁时间。这是区别于其他错误码的关键行为。
        """
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            return httpx.Response(418, json={"code": -1003, "msg": "IP banned"})

        data_config = DataConfig(
            cache_dir=tmp_cache_dir,
            rate_limit=RateLimitConfig(max_retries=5, base_backoff_seconds=0.001),
        )
        client = BinancePublicClient(ApiConfig(), data_config, client=httpx.Client(transport=httpx.MockTransport(handler)))

        with pytest.raises(IPBanError):
            client.futures_time()

        assert attempts["count"] == 1, f"418 不应重试，实际请求了 {attempts['count']} 次"
        client.close()

    def test_429_retries_with_backoff(self, tmp_cache_dir: Path) -> None:
        """HTTP 429 应重试，并尊重 Retry-After 头；coordinator 冻结由假时钟推进。"""
        attempts = {"count": 0}
        fake = _FakeClock()
        coordinator = RateLimitCoordinator(
            {RateLimitScope.SPOT: 6000, RateLimitScope.FUTURES: 2400},
            clock=fake, sleep=fake.sleep,
        )

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] < 3:
                return httpx.Response(429, headers={"Retry-After": "7"}, json={"code": -1003})
            return httpx.Response(200, json={"serverTime": 1789560670000})

        data_config = DataConfig(
            cache_dir=tmp_cache_dir,
            rate_limit=RateLimitConfig(max_retries=5, base_backoff_seconds=0.001),
        )
        client = BinancePublicClient(
            ApiConfig(),
            data_config,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            sleep_fn=fake.sleep,
            rate_limiter=coordinator,
        )

        result = client.futures_time()

        assert result == 1789560670000
        assert attempts["count"] == 3
        # 前两次 429 后应分别尊重 Retry-After: 7（客户端退避），随后 coordinator 冻结等待
        retry_after_sleeps = [s for s in fake.sleeps if s == 7.0]
        assert len(retry_after_sleeps) == 2, f"应尊重两次 Retry-After: {fake.sleeps}"
        freeze_waits = [s for s in fake.sleeps if s >= 60.0]
        assert freeze_waits, f"冻结期应等待而非立即重试: {fake.sleeps}"
        client.close()

    def test_429_exhausts_retries_and_raises(self, tmp_cache_dir: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"code": -1003})

        fake = _FakeClock()
        coordinator = RateLimitCoordinator(
            {RateLimitScope.SPOT: 6000, RateLimitScope.FUTURES: 2400},
            clock=fake, sleep=fake.sleep,
        )
        data_config = DataConfig(
            cache_dir=tmp_cache_dir,
            rate_limit=RateLimitConfig(max_retries=2, base_backoff_seconds=0.001),
        )
        client = BinancePublicClient(
            ApiConfig(),
            data_config,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            sleep_fn=fake.sleep,
            rate_limiter=coordinator,
        )

        with pytest.raises(RateLimitError):
            client.futures_time()
        assert fake.sleeps, "冻结等待应经可注入 sleep 完成"
        client.close()

    def test_5xx_retries_then_succeeds(self, tmp_cache_dir: Path) -> None:
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"serverTime": 1})

        data_config = DataConfig(
            cache_dir=tmp_cache_dir,
            rate_limit=RateLimitConfig(max_retries=3, base_backoff_seconds=0.001),
        )
        client = BinancePublicClient(
            ApiConfig(),
            data_config,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            sleep_fn=lambda _: None,
        )

        assert client.futures_time() == 1
        assert attempts["count"] == 2
        client.close()

    def test_invalid_symbol_raises_data_unavailable(self, tmp_cache_dir: Path) -> None:
        """币安错误码 -1121（无效交易对）应转为 DataUnavailableError，且不重试。"""
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            return httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."})

        data_config = DataConfig(cache_dir=tmp_cache_dir, rate_limit=RateLimitConfig(max_retries=3))
        client = BinancePublicClient(
            ApiConfig(),
            data_config,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            sleep_fn=lambda _: None,
        )

        with pytest.raises(DataUnavailableError):
            client.funding_history("NONEXISTENT")

        assert attempts["count"] == 1, "参数错误不应重试"
        client.close()

    def test_timeout_retries(self, tmp_cache_dir: Path) -> None:
        """网络超时应重试。"""
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise httpx.ConnectTimeout("模拟超时")
            return httpx.Response(200, json={"serverTime": 42})

        data_config = DataConfig(
            cache_dir=tmp_cache_dir,
            rate_limit=RateLimitConfig(max_retries=3, base_backoff_seconds=0.001),
        )
        client = BinancePublicClient(
            ApiConfig(),
            data_config,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            sleep_fn=lambda _: None,
        )

        assert client.futures_time() == 42
        client.close()


class TestRateLimiting:
    """限流控制。"""

    def test_weight_header_is_tracked(self, tmp_cache_dir: Path) -> None:
        """应从响应头更新已用权重。"""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"x-mbx-used-weight-1m": "1200"},
                json={"serverTime": 1},
            )

        data_config = DataConfig(cache_dir=tmp_cache_dir)
        client = BinancePublicClient(
            ApiConfig(), data_config, client=httpx.Client(transport=httpx.MockTransport(handler))
        )

        client.futures_time()
        assert (
            client.coordinator.snapshot()[RateLimitScope.FUTURES].observed_used == 1200
        ), "未读取权重响应头"
        client.close()

    def test_throttles_when_approaching_limit(self, tmp_cache_dir: Path) -> None:
        """权重超过 P3 候选预算时应等待窗口滚动。

        不主动降速会一路撞到 429，然后被 418 封 IP。
        """
        def handler(request: httpx.Request) -> httpx.Response:
            # 每次都返回一个很高的已用权重（超过 80% 的 2400）
            return httpx.Response(
                200,
                headers={"x-mbx-used-weight-1m": "2100"},
                json={"serverTime": 1},
            )

        fake = _FakeClock()
        coordinator = RateLimitCoordinator(
            {RateLimitScope.SPOT: 6000, RateLimitScope.FUTURES: 2400},
            clock=fake, sleep=fake.sleep,
        )
        data_config = DataConfig(
            cache_dir=tmp_cache_dir,
            rate_limit=RateLimitConfig(futures_weight_per_min=2400, soft_limit_ratio=0.80),
        )
        client = BinancePublicClient(
            ApiConfig(),
            data_config,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            sleep_fn=fake.sleep,
            rate_limiter=coordinator,
        )

        client.futures_time()  # 首次：观测 2100 > P3 低限 1200
        client.futures_time()  # 第二次：被共享预算挡住，等待窗口滚动（假时钟）

        assert fake.sleeps, "权重超过候选预算时应等待窗口滚动"
        assert client.stats.throttled_seconds > 0
        client.close()


class TestCachingIntegration:
    """缓存与请求的集成。"""

    def test_second_call_hits_cache(self, tmp_cache_dir: Path) -> None:
        """相同参数的第二次调用应命中缓存，不产生新请求。"""
        requests = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            requests["count"] += 1
            return httpx.Response(200, json={"symbols": []})

        data_config = DataConfig(cache_dir=tmp_cache_dir)
        client = BinancePublicClient(
            ApiConfig(), data_config, client=httpx.Client(transport=httpx.MockTransport(handler))
        )

        client.futures_exchange_info()
        client.futures_exchange_info()
        client.futures_exchange_info()

        assert requests["count"] == 1, f"应只请求 1 次，实际 {requests['count']} 次"
        assert client.stats.cache_hits == 2
        client.close()


# ---------------------------------------------------------------------------
# 资金费
# ---------------------------------------------------------------------------


class TestFundingIntervals:
    """结算周期处理 —— 本项目最易出错的地方。"""

    def test_funding_info_only_lists_non_default(self, clean_env, tmp_cache_dir: Path) -> None:
        """fundingInfo 只返回非默认周期的合约，未列出的应视为 8h。

        这是币安的实际行为，误以为是"全部合约列表"会导致大量币种
        被误判为未知周期。
        """
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[
                {"symbol": "LPTUSDT", "fundingIntervalHours": 4},
                {"symbol": "ARKUSDT", "fundingIntervalHours": 1},
            ])

        data_config = DataConfig(cache_dir=tmp_cache_dir)
        client = BinancePublicClient(
            ApiConfig(), data_config, client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        intervals = fetch_funding_intervals(client)

        assert intervals.get("LPTUSDT") == 4
        assert intervals.get("ARKUSDT") == 1
        # BTCUSDT 未出现在响应中 → 应为默认 8h
        assert intervals.get("BTCUSDT") == 8
        client.close()

    def test_periods_per_year_by_interval(self) -> None:
        """每年结算次数必须按各自周期算。

        手工验算::

            8h → 24/8 × 365 = 1095 次/年
            4h → 24/4 × 365 = 2190 次/年
            1h → 24   × 365 = 8760 次/年
        """
        intervals = FundingIntervals({"AUSDT": 8, "BUSDT": 4, "CUSDT": 1})

        assert intervals.periods_per_year("AUSDT") == pytest.approx(1095.0)
        assert intervals.periods_per_year("BUSDT") == pytest.approx(2190.0)
        assert intervals.periods_per_year("CUSDT") == pytest.approx(8760.0)


class TestAnnualization:
    """年化换算 —— 这个公式错了整个项目就错了。"""

    def test_annualize_8h(self) -> None:
        """手工验算: 0.0001 × (24/8) × 365 = 0.1095 = 10.95%"""
        assert annualize_rate(0.0001, 8) == pytest.approx(0.1095, abs=1e-12)

    def test_annualize_4h_is_double_of_8h(self) -> None:
        """**关键检查**：同样单期费率下，4h 结算的年化是 8h 的两倍。

        如果错误地对所有币套用 8h 公式，4h 结算的币年化会被**低估一半**，
        直接导致筛选环节漏掉最好的标的。
        """
        rate = 0.0001
        ann_8h = annualize_rate(rate, 8)
        ann_4h = annualize_rate(rate, 4)

        assert ann_4h == pytest.approx(ann_8h * 2, rel=1e-12)

    def test_annualize_1h_is_eight_times_8h(self) -> None:
        rate = 0.0001
        assert annualize_rate(rate, 1) == pytest.approx(annualize_rate(rate, 8) * 8, rel=1e-12)

    def test_deannualize_is_inverse(self) -> None:
        """is_inverse 关系必须成立（出场阈值换算依赖它）。"""
        for interval in (1, 4, 8):
            original = 0.0003
            ann = annualize_rate(original, interval)
            assert deannualize_rate(ann, interval) == pytest.approx(original, rel=1e-12)

    def test_annualize_rejects_nonpositive_interval(self) -> None:
        with pytest.raises(ValueError, match="结算周期"):
            annualize_rate(0.0001, 0)

        with pytest.raises(ValueError, match="结算周期"):
            annualize_rate(0.0001, -8)

    def test_annualize_accepts_series(self) -> None:
        """应支持 Series 输入（回测里大量使用）。"""
        import pandas as pd

        series = pd.Series([0.0001, 0.0002])
        result = annualize_rate(series, 8)

        assert isinstance(result, pd.Series)
        assert result.iloc[0] == pytest.approx(0.1095, abs=1e-12)
        assert result.iloc[1] == pytest.approx(0.2190, abs=1e-12)


class TestStreaks:
    """连续正/负期数统计。"""

    def test_consecutive_positive_streak(self) -> None:
        rates = make_funding_series([0.0001, 0.0001, -0.0001, 0.0001, 0.0001, 0.0001])
        streak = consecutive_positive_streak(rates)

        assert list(streak) == [1, 2, 0, 1, 2, 3]

    def test_consecutive_negative_streak(self) -> None:
        rates = make_funding_series([-0.0001, -0.0001, 0.0001, -0.0001])
        streak = consecutive_negative_streak(rates)

        assert list(streak) == [1, 2, 0, 1]

    def test_zero_rate_breaks_streak(self) -> None:
        """**零费率不算正** —— 这是刻意的语义。

        零费率下持有没有收益，但成本照付。把它算作"正"会让策略
        在无利可图的时期持续持仓。
        """
        rates = make_funding_series([0.0001, 0.0, 0.0001])
        streak = consecutive_positive_streak(rates)

        assert list(streak) == [1, 0, 1], f"零费率应打断连续计数，实际 {list(streak)}"


class TestFundingHistoryParsing:
    """资金费历史解析。"""

    def test_parses_records(self, tmp_cache_dir: Path, sample_funding_history) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=sample_funding_history)

        data_config = DataConfig(cache_dir=tmp_cache_dir)
        client = BinancePublicClient(
            ApiConfig(), data_config, client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        frame = fetch_funding_history(client, "BTCUSDT")

        assert len(frame) == 3
        assert list(frame.columns) == ["funding_rate", "mark_price"]
        assert frame["funding_rate"].dtype == float
        # 时间应升序
        assert frame.index.is_monotonic_increasing
        # 索引应为 UTC
        assert str(frame.index.tz) == "UTC"
        client.close()

    def test_empty_history_raises(self, tmp_cache_dir: Path) -> None:
        """空历史必须抛异常，而非静默返回空 —— 静默会导致选币结论污染。"""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[])

        data_config = DataConfig(cache_dir=tmp_cache_dir)
        client = BinancePublicClient(
            ApiConfig(), data_config, client=httpx.Client(transport=httpx.MockTransport(handler))
        )

        with pytest.raises(InsufficientDataError):
            fetch_funding_history(client, "EMPTYUSDT")
        client.close()

    def test_malformed_record_raises_parse_error(self, tmp_cache_dir: Path) -> None:
        """字段缺失时应抛 ParseError，而不是静默跳过。"""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[{"symbol": "BTCUSDT"}])   # 缺少 fundingTime

        data_config = DataConfig(cache_dir=tmp_cache_dir)
        client = BinancePublicClient(
            ApiConfig(), data_config, client=httpx.Client(transport=httpx.MockTransport(handler))
        )

        with pytest.raises(ParseError):
            fetch_funding_history(client, "BTCUSDT")
        client.close()


class TestSummarize:
    """资金费摘要统计。"""

    def test_summary_fields(self) -> None:
        rates = make_funding_series([0.0002] * 50 + [-0.0002] * 10)
        frame = rates.to_frame("funding_rate")
        frame["mark_price"] = 1.0

        summary = summarize_funding(frame, "TESTUSDT", 8)

        assert summary["symbol"] == "TESTUSDT"
        assert summary["periods"] == 60
        assert summary["interval_hours"] == 8
        assert summary["positive_ratio"] == pytest.approx(50 / 60, rel=1e-9)
        assert summary["longest_positive_streak"] == 50
        assert summary["longest_negative_streak"] == 10

    def test_empty_frame_raises(self) -> None:
        import pandas as pd

        with pytest.raises(InsufficientDataError):
            summarize_funding(pd.DataFrame(), "X", 8)


# ---------------------------------------------------------------------------
# K 线与交易规则
# ---------------------------------------------------------------------------


class TestKlineParsing:
    """K 线解析 —— 字符串转数值是最容易静默出错的地方。"""

    def test_parses_string_numbers_to_float(self, sample_klines) -> None:
        frame = klines_to_frame(sample_klines, "BTCUSDT")

        assert len(frame) == 2
        for column in ("open", "high", "low", "close", "volume", "quote_volume"):
            assert frame[column].dtype == float, f"{column} 未转为 float，实际 {frame[column].dtype}"

        assert frame["open"].iloc[0] == pytest.approx(75599.90)
        assert frame["close"].iloc[1] == pytest.approx(76100.00)
        assert frame["trades"].iloc[0] == 780094

    def test_index_is_utc_datetime(self, sample_klines) -> None:
        frame = klines_to_frame(sample_klines, "BTCUSDT")
        assert str(frame.index.tz) == "UTC"
        assert frame.index.is_monotonic_increasing

    def test_empty_klines_raises(self) -> None:
        with pytest.raises(ParseError):
            klines_to_frame([], "BTCUSDT")

    def test_malformed_kline_raises(self) -> None:
        """字段数不足时必须抛异常 —— 静默补零会污染回测。"""
        with pytest.raises(ParseError, match="字段数异常"):
            klines_to_frame([[1, 2, 3]], "BTCUSDT")

    def test_non_numeric_price_raises(self) -> None:
        bad = [[1789516800000, "NOT_A_NUMBER", "1", "1", "1", "1", 1, "1", 1, "1", "1", "0"]]
        with pytest.raises(ParseError):
            klines_to_frame(bad, "BTCUSDT")


class TestSymbolRules:
    """交易规则解析。"""

    def test_parse_btc_rules(self, sample_exchange_info_spot) -> None:
        rules = parse_symbol_rules(sample_exchange_info_spot, "BTCUSDT")

        assert rules.symbol == "BTCUSDT"
        assert rules.tick_size == pytest.approx(0.01)
        assert rules.step_size == pytest.approx(0.00001)
        assert rules.min_qty == pytest.approx(0.00001)
        assert rules.min_notional == pytest.approx(5.0)
        assert rules.quantity_precision == 5   # "0.00001000" → 5 位有效小数

    def test_parse_shib_rules(self, sample_exchange_info_spot) -> None:
        """SHIB 的 stepSize 是 1（整数个），精度应为 0。

        字符串推断精度比浮点 log10 可靠 —— 不会有浮点误差差一位。
        """
        rules = parse_symbol_rules(sample_exchange_info_spot, "SHIBUSDT")
        assert rules.step_size == pytest.approx(1.0)
        assert rules.quantity_precision == 0

    def test_missing_symbol_raises(self, sample_exchange_info_spot) -> None:
        with pytest.raises(DataUnavailableError, match="不存在"):
            parse_symbol_rules(sample_exchange_info_spot, "FAKEUSDT")

    def test_round_qty_down_is_conservative(self, sample_exchange_info_spot) -> None:
        """向下取整必须真的向下（宁可少买不可多买）。"""
        rules = parse_symbol_rules(sample_exchange_info_spot, "BTCUSDT")

        # stepSize = 0.00001，0.00123456 应向下对齐到 0.00123
        assert rules.round_qty(0.00123456, mode="down") == pytest.approx(0.00123, abs=1e-9)

        # 向上取整
        assert rules.round_qty(0.00123456, mode="up") == pytest.approx(0.00124, abs=1e-9)

    def test_round_qty_below_min_returns_zero(self, sample_exchange_info_spot) -> None:
        """低于最小数量时返回 0（表示无法下单）。"""
        rules = parse_symbol_rules(sample_exchange_info_spot, "BTCUSDT")
        assert rules.round_qty(0.000001) == 0.0

    def test_is_tradeable_respects_min_notional(self, sample_exchange_info_spot) -> None:
        """名义额低于 minNotional 时必须判定为不可下单。

        这是"回测跑得通、实盘跑不通"的经典来源：
        回测里下 0.00002 BTC（约 1.5 USDT）没问题，实盘直接被拒。
        """
        rules = parse_symbol_rules(sample_exchange_info_spot, "BTCUSDT")

        # 0.0001 BTC × 70000 = 7 USDT ≥ 5，可下单
        ok, reason = rules.is_tradeable(0.0001, 70000.0)
        assert ok, reason

        # 0.00005 BTC × 70000 = 3.5 USDT < 5，不可下单
        ok, reason = rules.is_tradeable(0.00005, 70000.0)
        assert not ok
        assert "minNotional" in reason

    def test_round_price_aligns_to_tick(self, sample_exchange_info_spot) -> None:
        rules = parse_symbol_rules(sample_exchange_info_spot, "BTCUSDT")
        # tickSize = 0.01
        assert rules.round_price(75599.987, mode="down") == pytest.approx(75599.98)


class TestTradablePerpetuals:
    """永续合约筛选。"""

    def test_filters_non_perpetual(self, sample_exchange_info_futures) -> None:
        """季度交割合约必须被排除 —— 它们会到期，不适合长期持有。"""
        symbols = tradable_perpetuals(sample_exchange_info_futures)

        assert "BTCUSDT_250926" not in symbols, "季度合约不应在列表中"

    def test_filters_non_trading(self, sample_exchange_info_futures) -> None:
        """非 TRADING 状态的合约必须被排除。"""
        symbols = tradable_perpetuals(sample_exchange_info_futures)
        assert "DELISTEDUSDT" not in symbols

    def test_excludes_configured_bases(self, sample_exchange_info_futures) -> None:
        """配置里的排除列表应生效（稳定币对）。"""
        symbols = tradable_perpetuals(
            sample_exchange_info_futures, exclude_bases=("USDC", "FDUSD")
        )
        assert "USDCUSDT" not in symbols

    def test_includes_valid_perpetuals(self, sample_exchange_info_futures) -> None:
        symbols = tradable_perpetuals(
            sample_exchange_info_futures, exclude_bases=("USDC",)
        )
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols
        assert symbols == sorted(symbols), "结果应已排序"


# ---------------------------------------------------------------------------
# 分页
# ---------------------------------------------------------------------------


class TestPagination:
    """分页逻辑 —— 分页写错会导致数据静默缺失。"""

    def test_funding_history_paginates(self, tmp_cache_dir: Path) -> None:
        """多页拉取必须取全；单页 limit 固定 FUNDING_PAGE_SIZE（大 limit 会被 WAF 403）。"""
        from cointrader.data.binance import FUNDING_PAGE_SIZE

        total = 250  # 需要 3 页（100+100+50）
        base_time = 1_700_000_000_000
        step = 28_800_000
        seen_limits: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            params = dict(request.url.params)
            start = int(params.get("startTime", 0))
            limit = int(params.get("limit", FUNDING_PAGE_SIZE))
            seen_limits.append(limit)
            all_records: list[dict[str, str | int]] = [
                {
                    "symbol": "BTCUSDT",
                    "fundingTime": base_time + i * step,
                    "fundingRate": "0.00010000",
                    "markPrice": "70000.0",
                }
                for i in range(total)
            ]
            window = [r for r in all_records if int(r["fundingTime"]) >= start][:limit]
            return httpx.Response(200, json=window)

        data_config = DataConfig(cache_dir=tmp_cache_dir)
        client = BinancePublicClient(
            ApiConfig(), data_config, client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        records = client.funding_history("BTCUSDT", limit=total)

        assert len(records) == total, f"分页应取回全部 {total} 条，实际 {len(records)} 条"
        times = [r["fundingTime"] for r in records]
        assert times == sorted(times), "结果应按时间升序"
        assert len(set(times)) == total, "不应有重复记录"
        assert len(seen_limits) == 3, f"应分 3 页，实际 {len(seen_limits)} 次请求"
        assert all(v == FUNDING_PAGE_SIZE for v in seen_limits), "单页 limit 必须 ≤ 100（防 WAF 403）"
        client.close()

    def test_pagination_terminates_on_no_progress(self, tmp_cache_dir: Path) -> None:
        """服务端忽略 startTime 时必须能终止，不能死循环。

        币安在某些边界条件下会返回同一页。分页循环里的"无进展"检测
        是防止挂死的关键保护 —— 没有它，一次边界请求会让扫描任务永久卡住。
        """
        def handler(request: httpx.Request) -> httpx.Response:
            # 永远返回同一页（忽略 startTime）
            return httpx.Response(200, json=[
                {"symbol": "X", "fundingTime": 1000, "fundingRate": "0.0001", "markPrice": "1"},
                {"symbol": "X", "fundingTime": 2000, "fundingRate": "0.0001", "markPrice": "1"},
                {"symbol": "X", "fundingTime": 3000, "fundingRate": "0.0001", "markPrice": "1"},
            ])

        data_config = DataConfig(cache_dir=tmp_cache_dir)
        client = BinancePublicClient(
            ApiConfig(), data_config, client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        # 若没有终止保护，这个调用会挂死
        records = client.funding_history("X", limit=3)

        assert len(records) == 3
        client.close()

    def test_funding_history_terminates_when_cursor_past_end(self, tmp_cache_dir: Path) -> None:
        """带上 endTime 且已取完时，应立即终止而不是继续请求。"""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            params = dict(request.url.params)
            start = int(params.get("startTime", 0))
            # 只有三条数据，起始时间分别是 1000/2000/3000
            all_records: list[dict[str, str | int]] = [
                {"symbol": "X", "fundingTime": t, "fundingRate": "0.0001", "markPrice": "1"}
                for t in (1000, 2000, 3000)
            ]
            limit = int(params.get("limit", 1000))
            window = [r for r in all_records if int(r["fundingTime"]) >= start][:limit]
            return httpx.Response(200, json=window)

        data_config = DataConfig(cache_dir=tmp_cache_dir)
        client = BinancePublicClient(
            ApiConfig(), data_config, client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        records = client.funding_history("X", end_ms=3000, limit=10)

        assert len(records) == 3
        # 取完后应停止，不应无限请求
        assert calls["n"] == 1, f"请求次数异常: {calls['n']}"
        client.close()

    def test_kline_pagination(self, tmp_cache_dir: Path) -> None:
        total = 7
        page_size = 3
        base_time = 1_700_000_000_000
        step = 28_800_000

        def handler(request: httpx.Request) -> httpx.Response:
            params = dict(request.url.params)
            start = int(params.get("startTime", 0))
            limit = int(params.get("limit", 1500))
            all_candles: list[list[str | int]] = [
                [
                    base_time + i * step, "100", "110", "90", "105", "1000",
                    base_time + i * step + step - 1, "100000", 10, "500", "50000", "0",
                ]
                for i in range(total)
            ]
            window = [c for c in all_candles if int(c[0]) >= start][:limit]
            return httpx.Response(200, json=window)

        data_config = DataConfig(cache_dir=tmp_cache_dir)
        client = BinancePublicClient(
            ApiConfig(), data_config, client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        raw = client.futures_klines("BTCUSDT", "8h", limit=page_size)

        assert len(raw) == total
        frame = klines_to_frame(raw, "BTCUSDT")
        assert len(frame) == total
        client.close()


__all__: list[str] = []
