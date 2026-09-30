"""历史区间覆盖索引与缺口复用测试（实施计划书 v5.0 T1，AC-01/02/03）。

全部离线：``httpx.MockTransport`` 伪造交易所（K 线 / 资金费 / 服务器时间），
``tmp_cache_dir`` 隔离磁盘缓存，服务器时间固定为常量（无真实时钟依赖）。

覆盖点：
- 重复整段命中零新请求（含跨进程模拟：新客户端实例同缓存目录）；
- 单缺口只请求该缺口（逐请求断言 startTime/endTime 参数）；
- 跨页分页、相邻/重叠边界、半开区间右端排除；
- 未闭合 K 线不进入历史覆盖（closed 边界收缩 + INCOMPLETE）；
- 部分失败（429/500/解析错误/无进展）绝不推进覆盖；断点后重读；
- 损坏索引/损坏段文件按 miss 重拉（含 checksum 校验失败可见）;
- funding 空区间记 complete 空段；limit 上限截断 + 续拉；
- 线程并发 single-flight 不产生重复拉取或假完整；
- 不同 symbol/interval 的 cache-key 隔离。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from cointrader.config import ApiConfig, DataConfig, RateLimitConfig, UniverseConfig
from cointrader.data.binance import BinancePublicClient
from cointrader.data.coverage import (
    COVERAGE_FORMAT_VERSION,
    CoverageKey,
    HistoricalDatasetResult,
    canonical_checksum,
    clip_to_requested,
    funding_identity,
    kline_identity,
    merge_intervals,
    subtract_intervals,
)
from cointrader.data.venue import Venue
from cointrader.domain.market import DataQuality
from cointrader.errors import BinanceError, ParseError, RateLimitError
from cointrader.rate_limit import RateLimitCoordinator, RateLimitScope

H8 = 8 * 3_600_000  # 8h（毫秒）

# 8h 对齐的基准时间（28_800_000 * 59028），保证 candle open 落在周期网格上
BASE = 1_700_006_400_000


# ---------------------------------------------------------------------------
# 假交易所
# ---------------------------------------------------------------------------


def make_candle(open_ms: int) -> list[Any]:
    """构造一根 12 字段 K 线（价格/数量为字符串，贴近币安真实响应）。"""
    return [
        open_ms,
        "100.5", "110.5", "90.5", "105.5", "1000.5",
        open_ms + H8 - 1,
        "100000.5", 10,
        "500.5", "50000.5", "0",
    ]


def make_funding_event(funding_time: int, rate: str = "0.00010000") -> dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "fundingTime": funding_time,
        "fundingRate": rate,
        "markPrice": "70000.0",
    }


class FakeHistoryServer:
    """内存假交易所：/fapi/v1/time、/fapi/v1/klines、/fapi/v1/fundingRate。

    - ``server_now_ms`` 固定（可测试中推进，模拟时间流逝）；
    - ``klines[(symbol, interval)]`` / ``funding[symbol]`` 为全集数据，
      按 startTime/endTime/limit 窗口化返回（模拟币安分页语义）；
    - ``fail_klines_from`` 给定时，startTime >= 该值的 K 线页返回 500；
    - ``no_progress_klines`` 给定时，K 线页永远返回同一页（忽略 startTime）；
    - ``delay_s`` 每页前 sleep，用于放大并发竞态窗口。
    """

    def __init__(
        self,
        *,
        server_now_ms: int,
        klines: dict[tuple[str, str], list[list[Any]]] | None = None,
        funding: dict[str, list[dict[str, Any]]] | None = None,
        delay_s: float = 0.0,
        fail_klines_from: int | None = None,
        no_progress_klines: bool = False,
        malformed_kline_from: int | None = None,
    ) -> None:
        self.server_now_ms = server_now_ms
        self.klines = klines or {}
        self.funding = funding or {}
        self.delay_s = delay_s
        self.fail_klines_from = fail_klines_from
        self.no_progress_klines = no_progress_klines
        self.malformed_kline_from = malformed_kline_from
        self.requests: list[tuple[str, dict[str, str]]] = []

    # -- 断言辅助 -------------------------------------------------------------

    def kline_pages(self) -> list[dict[str, str]]:
        return [p for path, p in self.requests if path == "/fapi/v1/klines"]

    def funding_pages(self) -> list[dict[str, str]]:
        return [p for path, p in self.requests if path == "/fapi/v1/fundingRate"]

    def data_requests(self) -> list[str]:
        return [path for path, _ in self.requests if path in ("/fapi/v1/klines", "/fapi/v1/fundingRate")]

    def reset(self) -> None:
        self.requests.clear()

    # -- MockTransport handler -------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = dict(request.url.params)
        self.requests.append((path, params))
        if self.delay_s:
            time.sleep(self.delay_s)

        if path == "/fapi/v1/time":
            return httpx.Response(200, json={"serverTime": self.server_now_ms})

        if path == "/fapi/v1/klines":
            symbol = params["symbol"]
            interval = params["interval"]
            start = int(params.get("startTime", 0))
            end = int(params.get("endTime", 10**16))
            limit = int(params.get("limit", 1500))
            if self.fail_klines_from is not None and start >= self.fail_klines_from:
                return httpx.Response(500)
            candles = self.klines.get((symbol, interval), [])
            if self.no_progress_klines:
                return httpx.Response(200, json=candles[:limit])
            window = [c for c in candles if start <= int(c[0]) <= end][:limit]
            if (
                self.malformed_kline_from is not None
                and window
                and int(window[-1][0]) >= self.malformed_kline_from
            ):
                window = [c[:3] for c in window]  # 字段数异常
            return httpx.Response(200, json=window)

        if path == "/fapi/v1/fundingRate":
            symbol = params["symbol"]
            start = int(params.get("startTime", 0))
            end = int(params.get("endTime", 10**16))
            limit = int(params.get("limit", 100))
            records = self.funding.get(symbol, [])
            funding_window = [r for r in records if start <= int(r["fundingTime"]) <= end][:limit]
            return httpx.Response(200, json=funding_window)

        return httpx.Response(404, json={"code": -1, "msg": f"unknown path {path}"})


def make_client(
    handler: Any,
    tmp_cache_dir: Path,
    *,
    sleep_calls: list[float] | None = None,
    funding_page_size: int | None = None,
) -> BinancePublicClient:
    """MockTransport 客户端（短退避、不真睡），缓存目录为临时目录。

    ``funding_page_size``：资金费分页单页条数（None = 用配置默认值）。
    验证分页行为时显式传小页，不把测试钉在默认值上。
    """
    data_config = DataConfig(
        cache_dir=tmp_cache_dir,
        rate_limit=RateLimitConfig(
            base_backoff_seconds=0.001, max_backoff_seconds=0.01, max_retries=2
        ),
    )
    # 页大小现在按 venue 配置（主网/ demo 的 WAF 阈值不同）
    universe = UniverseConfig()
    if funding_page_size is not None:
        universe = UniverseConfig(
            mainnet_funding_page_size=funding_page_size,
            demo_funding_page_size=funding_page_size,
        )
    return BinancePublicClient(
        ApiConfig(),
        data_config,
        universe=universe,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep_fn=(sleep_calls.append if sleep_calls is not None else lambda _s: None),
    )


class _FakeClock:
    """假单调时钟：sleep 记录并推进时间轴，让 coordinator 冻结期瞬间通过。"""

    def __init__(self) -> None:
        self.value = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


def make_flaky_client(handler: Any, tmp_cache_dir: Path) -> tuple[BinancePublicClient, _FakeClock]:
    """429/限流测试专用：注入假时钟 + 共享 coordinator。

    429 会触发 coordinator 冻结（默认 120s）；若用真实时钟 + no-op sleep，
    冻结等待会空转真实 120s。假时钟让冻结等待瞬间推进（同 test_data 模式）。
    """
    fake = _FakeClock()
    coordinator = RateLimitCoordinator(
        {RateLimitScope.SPOT: 6000, RateLimitScope.FUTURES: 2400},
        clock=fake, sleep=fake.sleep,
    )
    data_config = DataConfig(
        cache_dir=tmp_cache_dir,
        rate_limit=RateLimitConfig(
            base_backoff_seconds=0.001, max_backoff_seconds=0.01, max_retries=2
        ),
    )
    client = BinancePublicClient(
        ApiConfig(),
        data_config,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep_fn=fake.sleep,
        rate_limiter=coordinator,
    )
    return client, fake


def read_index(tmp_cache_dir: Path, key: CoverageKey) -> dict[str, Any] | None:
    """直接读磁盘上的覆盖索引（验证元数据/checksum/版本可校验）。"""
    path = tmp_cache_dir / "coverage_v1_mainnet" / f"{key.key_hash()}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload["data"]
    if not isinstance(data, dict):
        return None
    return data


#: 测试客户端的 venue（BinancePublicClient 默认 MAINNET）；
#: CoverageKey.venue 必须与之一致，否则 key_hash 失配、读不到索引。
_VENUE = Venue.MAINNET.value


def klines_key(symbol: str, interval: str = "8h") -> CoverageKey:
    return CoverageKey(
        venue=_VENUE, market="futures", dataset="futures_klines",
        symbol=symbol, interval=interval,
    )


def funding_key(symbol: str) -> CoverageKey:
    return CoverageKey(venue=_VENUE, market="futures", dataset="funding_history", symbol=symbol)


# ---------------------------------------------------------------------------
# 纯函数：区间运算 / 键 / checksum
# ---------------------------------------------------------------------------


class TestIntervalMath:
    def test_merge_overlapping_adjacent_contained(self) -> None:
        assert merge_intervals([(1, 5), (4, 7), (8, 9), (0, 2), (20, 25), (25, 30)]) == [
            (0, 7), (8, 9), (20, 30),
        ]

    def test_merge_unsorted_and_degenerate(self) -> None:
        assert merge_intervals([(5, 9), (1, 3), (3, 5), (7, 7), (0, 0)]) == [(1, 9)]

    def test_merge_empty(self) -> None:
        assert merge_intervals([]) == []

    def test_subtract_partial_and_full(self) -> None:
        assert subtract_intervals((0, 10), [(0, 3), (5, 8)]) == [(3, 5), (8, 10)]
        assert subtract_intervals((0, 10), [(-5, 15)]) == []
        assert subtract_intervals((2, 5), [(0, 3), (4, 6)]) == [(3, 4)]
        assert subtract_intervals((2, 5), [(0, 5)]) == []
        assert subtract_intervals((0, 10), []) == [(0, 10)]

    def test_subtract_clips_out_of_range_cover(self) -> None:
        # covered 超出 requested 的部分自动裁剪
        assert subtract_intervals((4, 6), [(0, 5), (10, 20)]) == [(5, 6)]

    def test_clip(self) -> None:
        assert clip_to_requested((2, 8), (0, 5)) == (2, 5)
        assert clip_to_requested((9, 12), (0, 5)) is None

    def test_invalid_requested_raises(self) -> None:
        with pytest.raises(ValueError):
            subtract_intervals((5, 5), [])


class TestCoverageKeyAndChecksum:
    def test_key_hash_stable_and_distinct(self) -> None:
        a = klines_key("BTCUSDT", "8h")
        b = CoverageKey(
            venue=_VENUE, market="futures", dataset="futures_klines",
            symbol="BTCUSDT", interval="8h",
        )
        assert a.key_hash() == b.key_hash()
        assert len(a.key_hash()) == 32
        assert a.key_hash() != klines_key("ETHUSDT", "8h").key_hash()
        assert a.key_hash() != klines_key("BTCUSDT", "4h").key_hash()
        assert a.key_hash() != funding_key("BTCUSDT").key_hash()
        # hex-only：可安全用作文件 key，不含原始 symbol
        int(a.key_hash(), 16)

    def test_key_validation(self) -> None:
        with pytest.raises(ValueError):
            CoverageKey(venue=_VENUE, market="futures", dataset="futures_klines",
                       symbol="BTCUSDT", interval="bogus")
        with pytest.raises(ValueError):
            CoverageKey(venue=_VENUE, market="futures", dataset="futures_klines",
                       symbol="", interval="8h")
        with pytest.raises(ValueError):
            CoverageKey(venue=_VENUE, market="futures", dataset="nope",
                       symbol="BTCUSDT", interval="")
        with pytest.raises(ValueError):
            CoverageKey(venue=_VENUE, market="futures", dataset="funding_history",
                       symbol="BTCUSDT", interval="8h")

    def test_checksum_stable_and_sensitive(self) -> None:
        records = [make_funding_event(1000), make_funding_event(2000)]
        assert canonical_checksum(records) == canonical_checksum(list(records))
        assert canonical_checksum(records) != canonical_checksum(
            [make_funding_event(1000, "0.00020000"), make_funding_event(2000)]
        )
        assert canonical_checksum([]) != canonical_checksum([make_funding_event(1000)])

    def test_identities(self) -> None:
        assert kline_identity(make_candle(123)) == 123
        with pytest.raises(ParseError):
            kline_identity([1, 2, 3])
        with pytest.raises(ParseError):
            kline_identity("not a candle")
        assert funding_identity(make_funding_event(456)) == 456
        with pytest.raises(ParseError):
            funding_identity({"fundingRate": "0.0001"})
        with pytest.raises(ParseError):
            funding_identity(["not", "a", "record"])


# ---------------------------------------------------------------------------
# K 线覆盖
# ---------------------------------------------------------------------------


class TestCoverageKlines:
    """futures_klines 双边界路径：覆盖索引 + 只拉缺口 + 仅已闭合。"""

    def test_full_range_fetch_then_zero_rerun(self, tmp_cache_dir: Path) -> None:
        """重复整段命中：第二次（含新客户端实例）零网络请求。"""
        now = BASE + 10 * H8 + 2 * 3_600_000  # 第 10 根 candle 未闭合
        candles = [make_candle(BASE + i * H8) for i in range(10)]
        server = FakeHistoryServer(server_now_ms=now, klines={("BTCUSDT", "8h"): candles})
        client = make_client(server.handle, tmp_cache_dir)

        start, end = BASE, BASE + 6 * H8
        result = client.historical_repository.fetch_futures_klines(
            "BTCUSDT", "8h", start, end, limit=1500
        )
        assert [int(c[0]) for c in result.records] == [BASE + i * H8 for i in range(6)]
        assert result.quality is DataQuality.FRESH
        assert result.requested_range == (start, end)
        assert result.covered_ranges == [(start, end)]
        assert result.missing_ranges == []
        assert result.network_pages == 1
        assert result.cache_hits == 0
        # 首拉：1 页 K 线 + 1 次交易所时间（闭合边界）
        assert len(server.kline_pages()) == 1
        assert sum(1 for p, _ in server.requests if p == "/fapi/v1/time") == 1

        # 第二次：整段命中 → 零数据请求、零时间请求
        server.reset()
        result2 = client.historical_repository.fetch_futures_klines(
            "BTCUSDT", "8h", start, end, limit=1500
        )
        assert [int(c[0]) for c in result2.records] == [int(c[0]) for c in result.records]
        assert result2.quality is DataQuality.FRESH
        assert result2.cache_hits == 1
        assert result2.network_pages == 0
        assert server.data_requests() == []
        assert client.stats.historical_cache_hits == 1
        assert client.stats.cache_hits == 0  # 与短 TTL 实时缓存分开计数

        # 跨进程模拟：新客户端实例、同缓存目录 → 同样零网络
        server.reset()
        client2 = make_client(server.handle, tmp_cache_dir)
        result3 = client2.historical_repository.fetch_futures_klines(
            "BTCUSDT", "8h", start, end, limit=1500
        )
        assert [int(c[0]) for c in result3.records] == [int(c[0]) for c in result.records]
        assert server.data_requests() == []
        client.close()
        client2.close()

    def test_gap_only_requested(self, tmp_cache_dir: Path) -> None:
        """先覆盖前 4 根，再请求 6 根 → 只请求 [4h..6h) 缺口。"""
        now = BASE + 10 * H8
        candles = [make_candle(BASE + i * H8) for i in range(10)]
        server = FakeHistoryServer(server_now_ms=now, klines={("BTCUSDT", "8h"): candles})
        client = make_client(server.handle, tmp_cache_dir)
        repo = client.historical_repository

        first = repo.fetch_futures_klines("BTCUSDT", "8h", BASE, BASE + 4 * H8, limit=1500)
        assert len(first.records) == 4 and first.quality is DataQuality.FRESH

        server.reset()
        result = repo.fetch_futures_klines("BTCUSDT", "8h", BASE, BASE + 6 * H8, limit=1500)
        assert len(result.records) == 6
        assert result.quality is DataQuality.FRESH
        # 恰好一个缺口页，且参数严格落在缺口内
        pages = server.kline_pages()
        assert len(pages) == 1
        assert int(pages[0]["startTime"]) == BASE + 4 * H8
        assert int(pages[0]["endTime"]) == BASE + 6 * H8 - 1
        client.close()

    def test_cross_page_pagination(self, tmp_cache_dir: Path) -> None:
        """limit=2 跨 3 页取回 6 根；cursor 严格推进、无重复。"""
        now = BASE + 10 * H8
        candles = [make_candle(BASE + i * H8) for i in range(10)]
        server = FakeHistoryServer(server_now_ms=now, klines={("BTCUSDT", "8h"): candles})
        client = make_client(server.handle, tmp_cache_dir)

        result = client.historical_repository.fetch_futures_klines(
            "BTCUSDT", "8h", BASE, BASE + 5 * H8, limit=2
        )
        opens = [int(c[0]) for c in result.records]
        assert opens == [BASE + i * H8 for i in range(5)]
        assert len(set(opens)) == 5
        pages = server.kline_pages()
        assert len(pages) == 3
        cursors = [int(p["startTime"]) for p in pages]
        assert cursors == [BASE, BASE + H8 + 1, BASE + 3 * H8 + 1]
        assert all(int(p["endTime"]) == BASE + 5 * H8 - 1 for p in pages)
        assert result.network_pages == 3
        client.close()

    def test_adjacent_and_overlapping_ranges(self, tmp_cache_dir: Path) -> None:
        """相邻/重叠范围复用覆盖；半开区间右端的 candle 不进入结果。"""
        now = BASE + 12 * H8
        candles = [make_candle(BASE + i * H8) for i in range(12)]
        server = FakeHistoryServer(server_now_ms=now, klines={("BTCUSDT", "8h"): candles})
        client = make_client(server.handle, tmp_cache_dir)
        repo = client.historical_repository

        a = repo.fetch_futures_klines("BTCUSDT", "8h", BASE, BASE + 4 * H8, limit=1500)
        assert len(a.records) == 4

        server.reset()
        b = repo.fetch_futures_klines("BTCUSDT", "8h", BASE + 4 * H8, BASE + 8 * H8, limit=1500)
        assert len(b.records) == 4  # 右端 BASE+8H 的 candle 不含（半开）
        pages = server.kline_pages()
        assert len(pages) == 1
        assert int(pages[0]["startTime"]) == BASE + 4 * H8

        # 并集请求：两段全覆盖 → 零网络
        server.reset()
        union = repo.fetch_futures_klines("BTCUSDT", "8h", BASE, BASE + 8 * H8, limit=1500)
        assert len(union.records) == 8
        assert server.data_requests() == []
        assert union.quality is DataQuality.FRESH

        # 重叠请求 [4h, 12h)：只拉 [8h, 12h)
        server.reset()
        overlap = repo.fetch_futures_klines("BTCUSDT", "8h", BASE + 4 * H8, BASE + 12 * H8, limit=1500)
        assert len(overlap.records) == 8
        pages = server.kline_pages()
        assert len(pages) == 1
        assert int(pages[0]["startTime"]) == BASE + 8 * H8
        client.close()

    def test_unclosed_candle_excluded_then_completed(self, tmp_cache_dir: Path) -> None:
        """未闭合 candle 不进入历史覆盖；时间推进后只补拉缺口。"""
        now = BASE + 4 * H8 - 1  # 前 3 根（open<BASE+3H）已闭合，第 3 根（open=BASE+3H）未闭合
        candles = [make_candle(BASE + i * H8) for i in range(10)]
        server = FakeHistoryServer(server_now_ms=now, klines={("BTCUSDT", "8h"): candles})
        client = make_client(server.handle, tmp_cache_dir)
        repo = client.historical_repository

        result = repo.fetch_futures_klines("BTCUSDT", "8h", BASE, BASE + 4 * H8, limit=1500)
        assert [int(c[0]) for c in result.records] == [BASE, BASE + H8, BASE + 2 * H8]
        assert result.quality is DataQuality.INCOMPLETE
        assert result.covered_ranges == [(BASE, BASE + 3 * H8)]
        assert result.missing_ranges == [(BASE + 3 * H8, BASE + 4 * H8)]
        # 请求只到闭合边界（exclusive = now - 8h + 1 = BASE+3H），未拉未闭合 candle
        assert int(server.kline_pages()[0]["endTime"]) == BASE + 3 * H8 - 1

        # 交易所时间推进过闭合点：只补拉 [3H, 4H) 缺口
        server.server_now_ms = BASE + 5 * H8
        server.reset()
        result2 = repo.fetch_futures_klines("BTCUSDT", "8h", BASE, BASE + 4 * H8, limit=1500)
        assert len(result2.records) == 4
        assert result2.quality is DataQuality.FRESH
        pages = server.kline_pages()
        assert len(pages) == 1
        assert int(pages[0]["startTime"]) == BASE + 3 * H8
        client.close()

    def test_future_range_not_fetched(self, tmp_cache_dir: Path) -> None:
        """整个范围都在未来（未闭合区）：零 K 线请求、INCOMPLETE。"""
        now = BASE + 5 * H8
        candles = [make_candle(BASE + i * H8) for i in range(12)]
        server = FakeHistoryServer(server_now_ms=now, klines={("BTCUSDT", "8h"): candles})
        client = make_client(server.handle, tmp_cache_dir)

        result = client.historical_repository.fetch_futures_klines(
            "BTCUSDT", "8h", BASE + 6 * H8, BASE + 8 * H8, limit=1500
        )
        assert result.records == []
        assert result.quality is DataQuality.INCOMPLETE
        assert result.missing_ranges == [(BASE + 6 * H8, BASE + 8 * H8)]
        assert server.kline_pages() == []  # 只问了时间，没拉 K 线
        client.close()

    def test_range_validation(self, tmp_cache_dir: Path) -> None:
        """repository 入口要求 start < end（半开区间合法边界）。"""
        server = FakeHistoryServer(server_now_ms=BASE + 10 * H8)
        client = make_client(server.handle, tmp_cache_dir)
        repo = client.historical_repository
        with pytest.raises(ValueError):
            repo.fetch_futures_klines("BTCUSDT", "8h", BASE + 5, BASE, limit=1500)
        with pytest.raises(ValueError):
            repo.fetch_futures_klines("BTCUSDT", "8h", BASE, BASE, limit=1500)
        with pytest.raises(ValueError):
            repo.fetch_funding_history("BTCUSDT", BASE, BASE, limit=10)
        client.close()

    def test_429_does_not_advance_coverage(self, tmp_cache_dir: Path) -> None:
        """429 重试耗尽：抛 RateLimitError，覆盖索引无任何段。"""
        now = BASE + 10 * H8
        candles = [make_candle(BASE + i * H8) for i in range(10)]
        server = FakeHistoryServer(server_now_ms=now, klines={("BTCUSDT", "8h"): candles})
        client = make_client(server.handle, tmp_cache_dir)

        def handler_429(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/fapi/v1/time":
                return httpx.Response(200, json={"serverTime": now})
            return httpx.Response(429, headers={"Retry-After": "1"}, json={"code": -1003})

        flaky, fake_clock = make_flaky_client(handler_429, tmp_cache_dir)
        with pytest.raises(RateLimitError):
            flaky.historical_repository.fetch_futures_klines(
                "BTCUSDT", "8h", BASE, BASE + 4 * H8, limit=1500
            )
        assert read_index(tmp_cache_dir, klines_key("BTCUSDT")) is None
        assert list((tmp_cache_dir / "hist_segments_v1_mainnet").glob("*.json")) == []
        # 冻结等待经假时钟完成（非真实等待）
        assert fake_clock.sleeps
        flaky.close()

        # 恢复正常后完整重拉（startTime 仍是范围起点，无假覆盖）
        result = client.historical_repository.fetch_futures_klines(
            "BTCUSDT", "8h", BASE, BASE + 4 * H8, limit=1500
        )
        assert len(result.records) == 4 and result.quality is DataQuality.FRESH
        assert int(server.kline_pages()[0]["startTime"]) == BASE
        client.close()

    def test_malformed_page_raises_and_no_coverage(self, tmp_cache_dir: Path) -> None:
        """解析错误（字段数异常）：抛 ParseError，覆盖不前进。"""
        now = BASE + 10 * H8
        candles = [make_candle(BASE + i * H8) for i in range(10)]
        server = FakeHistoryServer(
            server_now_ms=now,
            klines={("BTCUSDT", "8h"): candles},
            malformed_kline_from=BASE + 2 * H8,
        )
        client = make_client(server.handle, tmp_cache_dir)
        with pytest.raises(ParseError):
            client.historical_repository.fetch_futures_klines(
                "BTCUSDT", "8h", BASE, BASE + 4 * H8, limit=1500
            )
        assert read_index(tmp_cache_dir, klines_key("BTCUSDT")) is None
        client.close()

    def test_no_progress_raises_and_no_coverage(self, tmp_cache_dir: Path) -> None:
        """服务端忽略 startTime（永远同一页）：ParseError，覆盖不前进。"""
        now = BASE + 10 * H8
        candles = [make_candle(BASE + i * H8) for i in range(2)]
        server = FakeHistoryServer(
            server_now_ms=now, klines={("BTCUSDT", "8h"): candles}, no_progress_klines=True
        )
        client = make_client(server.handle, tmp_cache_dir)
        # limit=2 使单页“满页”，范围 4 根宽 → 第二页无进展
        with pytest.raises(ParseError, match="无进展"):
            client.historical_repository.fetch_futures_klines(
                "BTCUSDT", "8h", BASE, BASE + 4 * H8, limit=2
            )
        assert read_index(tmp_cache_dir, klines_key("BTCUSDT")) is None
        assert list((tmp_cache_dir / "hist_segments_v1_mainnet").glob("*.json")) == []
        client.close()

    def test_partial_failure_checkpoint_and_resume(self, tmp_cache_dir: Path) -> None:
        """两个缺口：第一个成功后第二个 500 → 第一段已落盘（断点），
        恢复后只补拉第二个缺口。"""
        now = BASE + 10 * H8
        candles = [make_candle(BASE + i * H8) for i in range(10)]
        # 预覆盖中间段 [2H, 4H)，制造两个独立缺口 [0,2H) 与 [4H,6H)
        server = FakeHistoryServer(server_now_ms=now, klines={("BTCUSDT", "8h"): candles})
        client = make_client(server.handle, tmp_cache_dir)
        repo = client.historical_repository
        pre = repo.fetch_futures_klines("BTCUSDT", "8h", BASE + 2 * H8, BASE + 4 * H8, limit=1500)
        assert len(pre.records) == 2

        # 第二个缺口（startTime >= BASE+4H）返回 500
        server.fail_klines_from = BASE + 4 * H8
        server.reset()
        with pytest.raises(BinanceError):
            repo.fetch_futures_klines("BTCUSDT", "8h", BASE, BASE + 6 * H8, limit=1500)

        index = read_index(tmp_cache_dir, klines_key("BTCUSDT"))
        assert index is not None
        ranges = sorted((s["start_ms"], s["end_ms"]) for s in index["segments"])
        # [0,2H) 已成功发布；[4H,6H) 失败不覆盖
        assert ranges == [(BASE, BASE + 2 * H8), (BASE + 2 * H8, BASE + 4 * H8)]

        # 恢复：只补拉 [4H, 6H)
        server.fail_klines_from = None
        server.reset()
        result = repo.fetch_futures_klines("BTCUSDT", "8h", BASE, BASE + 6 * H8, limit=1500)
        assert len(result.records) == 6 and result.quality is DataQuality.FRESH
        pages = server.kline_pages()
        assert len(pages) == 1
        assert int(pages[0]["startTime"]) == BASE + 4 * H8
        client.close()

    def test_corrupt_index_refetches_everything(self, tmp_cache_dir: Path) -> None:
        """索引文件损坏 → 整体按 miss 重拉，结果仍正确。"""
        now = BASE + 10 * H8
        candles = [make_candle(BASE + i * H8) for i in range(10)]
        server = FakeHistoryServer(server_now_ms=now, klines={("BTCUSDT", "8h"): candles})
        client = make_client(server.handle, tmp_cache_dir)
        repo = client.historical_repository
        first = repo.fetch_futures_klines("BTCUSDT", "8h", BASE, BASE + 4 * H8, limit=1500)
        assert len(first.records) == 4

        index_path = tmp_cache_dir / "coverage_v1_mainnet" / f"{klines_key('BTCUSDT').key_hash()}.json"
        index_path.write_text("{ 这不是合法 JSON", encoding="utf-8")

        server.reset()
        result = repo.fetch_futures_klines("BTCUSDT", "8h", BASE, BASE + 4 * H8, limit=1500)
        assert [int(c[0]) for c in result.records] == [BASE + i * H8 for i in range(4)]
        assert len(server.kline_pages()) == 1  # 全部重拉
        assert result.quality in (DataQuality.FRESH, DataQuality.DEGRADED)
        client.close()

    def test_corrupt_segment_refetches_that_range_only(self, tmp_cache_dir: Path) -> None:
        """段文件损坏 → 读回 checksum 失败 → 仅该段范围重拉，计数器可见。"""
        now = BASE + 10 * H8
        candles = [make_candle(BASE + i * H8) for i in range(10)]
        server = FakeHistoryServer(server_now_ms=now, klines={("BTCUSDT", "8h"): candles})
        client = make_client(server.handle, tmp_cache_dir)
        repo = client.historical_repository
        repo.fetch_futures_klines("BTCUSDT", "8h", BASE, BASE + 2 * H8, limit=1500)
        repo.fetch_futures_klines("BTCUSDT", "8h", BASE + 2 * H8, BASE + 4 * H8, limit=1500)

        key = klines_key("BTCUSDT")
        index = read_index(tmp_cache_dir, key)
        assert index is not None and len(index["segments"]) == 2
        seg_path = tmp_cache_dir / "hist_segments_v1_mainnet" / f"{index['segments'][0]['segment_key']}.json"
        payload = json.loads(seg_path.read_text(encoding="utf-8"))
        payload["data"][0][4] = "999999"  # 篡改收盘价
        seg_path.write_text(json.dumps(payload), encoding="utf-8")

        server.reset()
        result = repo.fetch_futures_klines("BTCUSDT", "8h", BASE, BASE + 4 * H8, limit=1500)
        assert len(result.records) == 4
        assert client.stats.cache_read_failures == 1
        assert result.quality is DataQuality.DEGRADED  # 已补齐但发生过读失败
        pages = server.kline_pages()
        assert len(pages) == 1  # 只重拉损坏段
        assert int(pages[0]["startTime"]) == BASE
        client.close()

    def test_empty_range_records_complete_empty_segment(self, tmp_cache_dir: Path) -> None:
        """空响应（该窗口无 K 线）= complete 空段：第二次零网络。"""
        now = BASE + 10 * H8
        server = FakeHistoryServer(server_now_ms=now, klines={})
        client = make_client(server.handle, tmp_cache_dir)
        repo = client.historical_repository

        result = repo.fetch_futures_klines("EMPTYUSDT", "8h", BASE, BASE + 4 * H8, limit=1500)
        assert result.records == []
        assert result.quality is DataQuality.FRESH
        assert result.covered_ranges == [(BASE, BASE + 4 * H8)]
        assert result.missing_ranges == []

        index = read_index(tmp_cache_dir, klines_key("EMPTYUSDT"))
        assert index is not None
        assert index["segments"][0]["record_count"] == 0

        server.reset()
        result2 = repo.fetch_futures_klines("EMPTYUSDT", "8h", BASE, BASE + 4 * H8, limit=1500)
        assert server.data_requests() == []
        assert result2.records == []
        client.close()

    def test_index_metadata_and_checksum_verifiable(self, tmp_cache_dir: Path) -> None:
        """索引/段元数据：版本号、key、checksum 与段文件可互相验证。"""
        now = BASE + 10 * H8
        candles = [make_candle(BASE + i * H8) for i in range(6)]
        server = FakeHistoryServer(server_now_ms=now, klines={("BTCUSDT", "8h"): candles})
        client = make_client(server.handle, tmp_cache_dir)
        repo = client.historical_repository
        repo.fetch_futures_klines("BTCUSDT", "8h", BASE, BASE + 6 * H8, limit=1500)

        key = klines_key("BTCUSDT")
        index = read_index(tmp_cache_dir, key)
        assert index is not None
        assert index["format_version"] == COVERAGE_FORMAT_VERSION
        assert index["key"] == key.to_dict()
        assert len(index["segments"]) == 1
        seg_meta = index["segments"][0]
        assert seg_meta["start_ms"] == BASE
        assert seg_meta["end_ms"] == BASE + 6 * H8
        assert seg_meta["record_count"] == 6
        assert seg_meta["complete"] is True
        assert seg_meta["format_version"] == COVERAGE_FORMAT_VERSION

        seg_path = tmp_cache_dir / "hist_segments_v1_mainnet" / f"{seg_meta['segment_key']}.json"
        seg_data = json.loads(seg_path.read_text(encoding="utf-8"))["data"]
        assert canonical_checksum(seg_data) == seg_meta["checksum"]
        client.close()

    def test_key_isolation_between_symbols_and_intervals(self, tmp_cache_dir: Path) -> None:
        """不同 symbol / interval 不串数据：各自完整索引、独立拉取。"""
        now = BASE + 10 * H8
        candles_a = [make_candle(BASE + i * H8) for i in range(6)]
        candles_b = [make_candle(BASE + i * H8 + 1) for i in range(6)]  # open 各偏 1ms
        server = FakeHistoryServer(
            server_now_ms=now,
            klines={
                ("AAAUSDT", "8h"): candles_a,
                ("BBBUSDT", "8h"): candles_b,
                ("AAAUSDT", "4h"): [make_candle(BASE + i * 14_400_000) for i in range(12)],
            },
        )
        client = make_client(server.handle, tmp_cache_dir)
        repo = client.historical_repository

        ra = repo.fetch_futures_klines("AAAUSDT", "8h", BASE, BASE + 4 * H8, limit=1500)
        rb = repo.fetch_futures_klines("BBBUSDT", "8h", BASE, BASE + 4 * H8, limit=1500)
        assert [int(c[0]) for c in ra.records] == [BASE + i * H8 for i in range(4)]
        assert [int(c[0]) for c in rb.records] == [BASE + i * H8 + 1 for i in range(4)]

        # 同 symbol 不同 interval：不共享覆盖
        rc = repo.fetch_futures_klines("AAAUSDT", "4h", BASE, BASE + 4 * H8, limit=1500)
        assert len(rc.records) == 8  # [BASE, BASE+32h) 含 8 根 4h K 线

        index_files = list((tmp_cache_dir / "coverage_v1_mainnet").glob("*.json"))
        assert len(index_files) == 3  # AAA-8h / BBB-8h / AAA-4h
        client.close()

    def test_concurrent_single_flight_no_duplicate_fetch(self, tmp_cache_dir: Path) -> None:
        """8 线程并发同 key：网络页数 = 单飞一份，无假完整，结果一致。"""
        now = BASE + 10 * H8
        candles = [make_candle(BASE + i * H8) for i in range(10)]
        server = FakeHistoryServer(
            server_now_ms=now, klines={("BTCUSDT", "8h"): candles}, delay_s=0.02
        )
        client = make_client(server.handle, tmp_cache_dir)

        results: list[HistoricalDatasetResult] = []
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                results.append(
                    client.historical_repository.fetch_futures_klines(
                        "BTCUSDT", "8h", BASE, BASE + 4 * H8, limit=2
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - 收集线程异常供断言
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert errors == []
        assert len(results) == 8

        expected = [BASE + i * H8 for i in range(4)]
        for r in results:
            assert [int(c[0]) for c in r.records] == expected
            assert r.quality is DataQuality.FRESH
            assert r.missing_ranges == []
        # limit=2 → 单飞需要 2 满页 + 1 确认空页 = 3 页 + 1 次时间；
        # 8 线程绝不能产生 8× 页
        assert len(server.kline_pages()) == 3
        assert sum(1 for p, _ in server.requests if p == "/fapi/v1/time") == 1

        # 索引一致性：单段、checksum 与段文件一致
        index = read_index(tmp_cache_dir, klines_key("BTCUSDT"))
        assert index is not None and len(index["segments"]) == 1
        seg_meta = index["segments"][0]
        seg_path = tmp_cache_dir / "hist_segments_v1_mainnet" / f"{seg_meta['segment_key']}.json"
        seg_data = json.loads(seg_path.read_text(encoding="utf-8"))["data"]
        assert canonical_checksum(seg_data) == seg_meta["checksum"]
        assert seg_meta["record_count"] == 4
        client.close()


# ---------------------------------------------------------------------------
# Funding 覆盖
# ---------------------------------------------------------------------------


class TestCoverageFunding:
    """funding_history 双边界路径：fundingTime 身份、半开区间、limit 上限。"""

    def test_roundtrip_and_zero_rerun(self, tmp_cache_dir: Path) -> None:
        now = BASE + 8 * H8 + 3_600_000
        events = [make_funding_event(BASE + i * H8) for i in range(8)]
        server = FakeHistoryServer(server_now_ms=now, funding={"BTCUSDT": events})
        client = make_client(server.handle, tmp_cache_dir, funding_page_size=100)
        repo = client.historical_repository

        result = repo.fetch_funding_history("BTCUSDT", BASE, BASE + 6 * H8, limit=1000)
        assert [r["fundingTime"] for r in result.records] == [BASE + i * H8 for i in range(6)]
        assert result.quality is DataQuality.FRESH
        assert result.covered_ranges == [(BASE, BASE + 6 * H8)]
        assert len(server.funding_pages()) == 1
        # 单页 limit 等于配置的 funding_page_size
        assert server.funding_pages()[0]["limit"] == "100"

        server.reset()
        result2 = repo.fetch_funding_history("BTCUSDT", BASE, BASE + 6 * H8, limit=1000)
        assert [r["fundingTime"] for r in result2.records] == [r["fundingTime"] for r in result.records]
        assert server.data_requests() == []
        assert client.stats.historical_cache_hits == 1
        client.close()

    def test_default_page_size_is_500(self, tmp_cache_dir: Path) -> None:
        """默认分页 500 条：一年 8h 结算（1095 条）只需 3 页。

        回归：旧值 100 需 11 页，而 2026-09-28 实测 500 与 100 的 RTT 几乎相同
        （0.51s vs 0.49s）——多出的 8 次请求是纯浪费。同时防止 coverage 的
        page_limit 与客户端 limit 不一致而把满页误判为短页。
        """
        total = 1095
        now = BASE + (total + 2) * H8 + 3_600_000
        events = [make_funding_event(BASE + i * H8) for i in range(total)]
        server = FakeHistoryServer(server_now_ms=now, funding={"BTCUSDT": events})
        client = make_client(server.handle, tmp_cache_dir)  # 用配置默认页大小
        repo = client.historical_repository

        result = repo.fetch_funding_history(
            "BTCUSDT", BASE, BASE + total * H8, limit=total + 10
        )
        assert len(result.records) == total
        assert result.quality is DataQuality.FRESH
        pages = server.funding_pages()
        assert pages, "应有网络请求"
        assert all(p["limit"] == "500" for p in pages), "默认单页必须是 500"
        assert len(pages) == 3, f"1095 条按 500/页应 3 页，实际 {len(pages)}"
        client.close()

    def test_half_open_boundary(self, tmp_cache_dir: Path) -> None:
        """end_ms 处的记录被排除（半开）；扩大范围后纳入。"""
        now = BASE + 8 * H8 + 3_600_000
        events = [make_funding_event(BASE + i * H8) for i in range(8)]
        server = FakeHistoryServer(server_now_ms=now, funding={"BTCUSDT": events})
        client = make_client(server.handle, tmp_cache_dir)
        repo = client.historical_repository

        r1 = repo.fetch_funding_history("BTCUSDT", BASE, BASE + 6 * H8, limit=1000)
        assert r1.records[-1]["fundingTime"] == BASE + 5 * H8
        assert BASE + 6 * H8 not in [r["fundingTime"] for r in r1.records]

        r2 = repo.fetch_funding_history("BTCUSDT", BASE, BASE + 7 * H8, limit=1000)
        assert r2.records[-1]["fundingTime"] == BASE + 6 * H8
        client.close()

    def test_limit_cap_truncates_and_resumes(self, tmp_cache_dir: Path) -> None:
        """limit=150 截断（2 页即停，同 legacy 请求数）；覆盖只前进到已取子区间；
        下次加大 limit 只续拉尾部缺口。"""
        h4 = 4 * 3_600_000
        total = 250
        events = [make_funding_event(BASE + i * h4) for i in range(total)]
        end = BASE + total * h4 + h4  # 比最后一条事件多一个步长
        now = end + 3_600_000
        server = FakeHistoryServer(server_now_ms=now, funding={"BTCUSDT": events})
        client = make_client(server.handle, tmp_cache_dir, funding_page_size=100)
        repo = client.historical_repository

        r1 = repo.fetch_funding_history("BTCUSDT", BASE, end, limit=150)
        assert len(r1.records) == 150
        assert r1.records[-1]["fundingTime"] == BASE + 149 * h4
        assert len(server.funding_pages()) == 2  # 100 + 100（第 2 页取到即停）
        assert r1.quality is DataQuality.INCOMPLETE
        # 覆盖只前进到已完整取回的子区间
        assert r1.covered_ranges[0][0] == BASE
        assert r1.covered_ranges[0][1] == BASE + 199 * h4 + 1
        assert r1.missing_ranges == [(BASE + 199 * h4 + 1, end)]

        server.reset()
        r2 = repo.fetch_funding_history("BTCUSDT", BASE, end, limit=300)
        assert len(r2.records) == total
        assert r2.quality is DataQuality.FRESH
        assert r2.missing_ranges == []
        pages = server.funding_pages()
        assert len(pages) == 1  # 只续拉尾部
        assert int(pages[0]["startTime"]) == BASE + 199 * h4 + 1
        client.close()

    def test_empty_range_complete_empty_segment(self, tmp_cache_dir: Path) -> None:
        now = BASE + 10 * H8
        server = FakeHistoryServer(server_now_ms=now, funding={"NONEUSDT": []})
        client = make_client(server.handle, tmp_cache_dir)
        repo = client.historical_repository

        result = repo.fetch_funding_history("NONEUSDT", BASE, BASE + 4 * H8, limit=100)
        assert result.records == []
        assert result.quality is DataQuality.FRESH
        assert result.covered_ranges == [(BASE, BASE + 4 * H8)]

        server.reset()
        result2 = repo.fetch_funding_history("NONEUSDT", BASE, BASE + 4 * H8, limit=100)
        assert result2.records == []
        assert server.data_requests() == []
        client.close()

    def test_future_events_not_fetched(self, tmp_cache_dir: Path) -> None:
        """fundingTime >= 交易所当前时间的事件不存在：尾段留在 missing。"""
        now = BASE + 4 * H8
        events = [make_funding_event(BASE + i * H8) for i in range(8)]
        server = FakeHistoryServer(server_now_ms=now, funding={"BTCUSDT": events})
        client = make_client(server.handle, tmp_cache_dir)

        result = client.historical_repository.fetch_funding_history(
            "BTCUSDT", BASE, BASE + 8 * H8, limit=1000
        )
        assert [r["fundingTime"] for r in result.records] == [BASE + i * H8 for i in range(4)]
        assert result.quality is DataQuality.INCOMPLETE
        assert result.missing_ranges == [(BASE + 4 * H8, BASE + 8 * H8)]
        client.close()

    def test_429_does_not_advance_coverage(self, tmp_cache_dir: Path) -> None:
        now = BASE + 8 * H8
        events = [make_funding_event(BASE + i * H8) for i in range(8)]
        server = FakeHistoryServer(server_now_ms=now, funding={"BTCUSDT": events})
        client = make_client(server.handle, tmp_cache_dir)

        def handler_429(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/fapi/v1/time":
                return httpx.Response(200, json={"serverTime": now})
            return httpx.Response(429, json={"code": -1003})

        flaky, _fake = make_flaky_client(handler_429, tmp_cache_dir)
        with pytest.raises(RateLimitError):
            flaky.historical_repository.fetch_funding_history(
                "BTCUSDT", BASE, BASE + 4 * H8, limit=100
            )
        assert read_index(tmp_cache_dir, funding_key("BTCUSDT")) is None
        flaky.close()

        result = client.historical_repository.fetch_funding_history(
            "BTCUSDT", BASE, BASE + 4 * H8, limit=100
        )
        assert len(result.records) == 4 and result.quality is DataQuality.FRESH
        client.close()

    def test_limit_below_1_rejected(self, tmp_cache_dir: Path) -> None:
        server = FakeHistoryServer(server_now_ms=BASE + 10 * H8)
        client = make_client(server.handle, tmp_cache_dir)
        with pytest.raises(ValueError):
            client.funding_history("BTCUSDT", start_ms=BASE, end_ms=BASE + H8, limit=0)
        client.close()


# ---------------------------------------------------------------------------
# 客户端公开方法兼容性（返回类型/签名不变）
# ---------------------------------------------------------------------------


class TestClientCompatibility:
    def test_futures_klines_returns_raw_lists(self, tmp_cache_dir: Path) -> None:
        now = BASE + 10 * H8
        candles = [make_candle(BASE + i * H8) for i in range(6)]
        server = FakeHistoryServer(server_now_ms=now, klines={("BTCUSDT", "8h"): candles})
        client = make_client(server.handle, tmp_cache_dir)
        raw = client.futures_klines("BTCUSDT", "8h", start_ms=BASE, end_ms=BASE + 4 * H8)
        assert isinstance(raw, list)
        assert all(isinstance(c, list) and len(c) == 12 for c in raw)
        assert [int(c[0]) for c in raw] == [BASE + i * H8 for i in range(4)]
        client.close()

    def test_funding_history_returns_raw_dicts(self, tmp_cache_dir: Path) -> None:
        now = BASE + 10 * H8
        events = [make_funding_event(BASE + i * H8) for i in range(6)]
        server = FakeHistoryServer(server_now_ms=now, funding={"BTCUSDT": events})
        client = make_client(server.handle, tmp_cache_dir)
        records = client.funding_history("BTCUSDT", start_ms=BASE, end_ms=BASE + 4 * H8)
        assert isinstance(records, list)
        assert all(isinstance(r, dict) for r in records)
        assert [r["fundingTime"] for r in records] == [BASE + i * H8 for i in range(4)]
        client.close()

    def test_single_bound_calls_stay_legacy(self, tmp_cache_dir: Path) -> None:
        """只给一个边界（或都不给）→ 原 legacy 路径（短 TTL 缓存），不建覆盖索引。"""
        now = BASE + 10 * H8
        candles = [make_candle(BASE + i * H8) for i in range(6)]
        server = FakeHistoryServer(server_now_ms=now, klines={("BTCUSDT", "8h"): candles})
        client = make_client(server.handle, tmp_cache_dir)

        raw = client.futures_klines("BTCUSDT", "8h", end_ms=BASE + 4 * H8)
        # legacy 语义：无 startTime → 假服务器返回 open <= end 的全部 5 根（含端）
        assert len(raw) == 5
        # legacy 路径走短 TTL namespace，不产生覆盖索引
        assert list((tmp_cache_dir / "coverage_v1_mainnet").glob("*.json")) == []
        assert (tmp_cache_dir / "futures_klines").is_dir()
        client.close()


class TestT4CrossFault:
    """v5.0 T4：交叉故障组合与内存资源边界（AC-02/AC-08）。"""

    def test_repeated_and_overlapping_requests_keep_metadata_bounded(
        self, tmp_cache_dir: Path
    ) -> None:
        """T4.2：重复/重叠请求下覆盖元数据不无限增长（段去重、索引不重复写文件）。"""
        now = BASE + 20 * H8
        candles = [make_candle(BASE + i * H8) for i in range(12)]
        server = FakeHistoryServer(server_now_ms=now, klines={("BTCUSDT", "8h"): candles})
        client = make_client(server.handle, tmp_cache_dir)

        def counts() -> tuple[int, int, int]:
            idx_files = len(list((tmp_cache_dir / "coverage_v1_mainnet").glob("*.json")))
            seg_files = len(list((tmp_cache_dir / "hist_segments_v1_mainnet").glob("*.json")))
            index = read_index(tmp_cache_dir, klines_key("BTCUSDT"))
            seg_count = len(index["segments"]) if index else 0
            return (idx_files, seg_files, seg_count)

        client.futures_klines("BTCUSDT", "8h", start_ms=BASE, end_ms=BASE + 8 * H8)
        client.futures_klines("BTCUSDT", "8h", start_ms=BASE + H8, end_ms=BASE + 10 * H8)
        baseline = counts()
        assert baseline[2] > 0, "前置：应已建立覆盖段"

        for _ in range(3):
            client.futures_klines("BTCUSDT", "8h", start_ms=BASE, end_ms=BASE + 8 * H8)
            client.futures_klines("BTCUSDT", "8h", start_ms=BASE + H8, end_ms=BASE + 10 * H8)

        assert counts() == baseline, "重复/重叠请求不得增长索引/段文件数或段数"
        index = read_index(tmp_cache_dir, klines_key("BTCUSDT"))
        ranges = [(s["start_ms"], s["end_ms"]) for s in index["segments"]]  # type: ignore[index]
        assert len(ranges) == len(set(ranges)), "索引段不得重复"
        client.close()

    def test_hit_then_coordinator_rate_limit_keeps_coverage_valid(
        self, tmp_cache_dir: Path
    ) -> None:
        """T4.1 组合：历史命中零网络；后续缺口拉取被 coordinator 限流（429）时，
        未完成 range 不写 coverage，已完成的段保持有效。"""
        now = BASE + 20 * H8
        candles = [make_candle(BASE + i * H8) for i in range(12)]
        server = FakeHistoryServer(server_now_ms=now, klines={("BTCUSDT", "8h"): candles})
        client = make_client(server.handle, tmp_cache_dir)

        # 第一阶段：前 8 根完整入缓存（命中后零请求）
        client.futures_klines("BTCUSDT", "8h", start_ms=BASE, end_ms=BASE + 8 * H8)
        pages_before = len(server.kline_pages())
        assert len(client.futures_klines("BTCUSDT", "8h", start_ms=BASE, end_ms=BASE + 8 * H8)) == 8
        assert len(server.kline_pages()) == pages_before, "整段命中不得再发请求"

        # 第二阶段：扩展范围触发缺口拉取，但 klines 页全部 429
        def handler_429(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/fapi/v1/time":
                return httpx.Response(200, json={"serverTime": now})
            return httpx.Response(429, headers={"Retry-After": "1"}, json={"code": -1003})

        flaky, fake_clock = make_flaky_client(handler_429, tmp_cache_dir)
        with pytest.raises(RateLimitError):
            flaky.futures_klines("BTCUSDT", "8h", start_ms=BASE, end_ms=BASE + 12 * H8)
        assert fake_clock.sleeps, "coordinator 冻结等待应经可注入 sleep 完成"

        index = read_index(tmp_cache_dir, klines_key("BTCUSDT"))
        assert index is not None, "已完成段必须保留"
        ranges = [(s["start_ms"], s["end_ms"]) for s in index["segments"]]  # type: ignore[index]
        assert (BASE, BASE + 8 * H8) in ranges, "已完成段不得被失败请求破坏"
        assert (BASE + 8 * H8, BASE + 12 * H8) not in ranges, "限流未完成区间不得记覆盖"

        # 恢复后只补缺口
        client2 = make_client(server.handle, tmp_cache_dir)
        server.reset()
        result = client2.historical_repository.fetch_futures_klines(
            "BTCUSDT", "8h", BASE, BASE + 12 * H8, limit=1500
        )
        assert len(result.records) == 12 and result.quality is DataQuality.FRESH
        gap_pages = server.kline_pages()
        assert gap_pages, "恢复后应补拉缺口"
        assert all(int(p["startTime"]) >= BASE + 8 * H8 for p in gap_pages), "只打缺口，不重复已完成段"
        flaky.close()
        client2.close()
        client.close()


__all__: list[str] = []
