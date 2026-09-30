"""venue（交易场地）隔离测试。

背景（2026-09 实测）：主网与 Demo Trading 的合约清单**不同** ——
主网 527 个可交易 USDT 永续，demo 528 个；仅主网有 65 个（PUMPUSDT 等），
仅 demo 有 66 个（TONUSDT 等）。venue 必须显式携带，且缓存键与历史覆盖
namespace 都要按 venue 隔离，否则两个场地互相污染（先写的一方胜出），
表现为“换了 venue 却拿到另一边的数据”。

另外验证回测契约：回测恒用 mainnet 历史，不受 execution.mode 影响。

全部离线：MockTransport，不发真实请求。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from cointrader.config import ApiConfig, DataConfig, UniverseConfig
from cointrader.data.binance import BinancePublicClient
from cointrader.data.venue import (
    MAINNET_VENUE,
    SUPPORTED_VENUES,
    Venue,
    venue_for_execution_mode,
)
from cointrader.errors import ConfigError


def make_client(
    tmp_cache_dir: Path,
    *,
    handler: Any = None,
    venue: Venue = Venue.MAINNET,
    universe: UniverseConfig | None = None,
) -> BinancePublicClient:
    if handler is None:

        def handler(_req: httpx.Request) -> httpx.Response:  # noqa: ARG001
            return httpx.Response(200, json={"symbols": []})

    return BinancePublicClient(
        ApiConfig(),
        DataConfig(cache_dir=tmp_cache_dir),
        universe=universe,
        venue=venue,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


class TestVenueDerivation:
    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("paper", Venue.DEMO),
            ("testnet", Venue.DEMO),
            ("shadow", Venue.DEMO),
            ("live", Venue.MAINNET),
            ("LIVE", Venue.MAINNET),  # 大小写不敏感
        ],
    )
    def test_mode_maps_to_venue(self, mode: str, expected: Venue) -> None:
        assert venue_for_execution_mode(mode) is expected

    def test_unknown_mode_raises(self) -> None:
        """未知 mode 必须报错，绝不能静默退回主网。"""
        with pytest.raises(ValueError, match="未知 execution.mode"):
            venue_for_execution_mode("nope")

    def test_mainnet_is_the_backtest_venue(self) -> None:
        """回测契约：历史数据的 venue 恒为主网。"""
        assert MAINNET_VENUE is Venue.MAINNET
        assert Venue.MAINNET in SUPPORTED_VENUES


class TestEndpointSelection:
    def test_demo_and_mainnet_use_different_bases(self, tmp_cache_dir: Path) -> None:
        main = make_client(tmp_cache_dir, venue=Venue.MAINNET)
        demo = make_client(tmp_cache_dir, venue=Venue.DEMO)

        assert main.futures_base == "https://fapi.binance.com"
        assert main.spot_base == "https://api.binance.com"
        assert demo.futures_base == "https://demo-fapi.binance.com"
        assert demo.spot_base == "https://demo-api.binance.com"

    def test_exchange_info_hits_venue_endpoint(self, tmp_cache_dir: Path) -> None:
        """exchangeInfo 必须打到本 venue 的域名（币池来源不能串）。"""
        seen: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(f"{req.url.host}{req.url.path}")
            return httpx.Response(200, json={"symbols": []})
        demo = make_client(tmp_cache_dir, handler=handler, venue=Venue.DEMO)
        demo.futures_exchange_info()
        demo.spot_exchange_info()

        hosts = [s.split("/", 1)[0] for s in seen]
        assert "demo-fapi.binance.com" in hosts
        assert "demo-api.binance.com" in hosts
        assert "fapi.binance.com" not in hosts, f"demo 不得打到主网端点: {hosts}"
        assert "api.binance.com" not in hosts, f"demo 不得打到主网端点: {hosts}"

    def test_custom_universe_endpoints_are_honored(self, tmp_cache_dir: Path) -> None:
        """universe 节的端点可覆盖（兼容经典 testnet 域名）。"""
        universe = UniverseConfig(
            demo_futures_base="https://testnet.binancefuture.com",
            demo_spot_base="https://testnet.binance.vision",
        )
        demo = make_client(tmp_cache_dir, venue=Venue.DEMO, universe=universe)
        assert demo.futures_base == "https://testnet.binancefuture.com"
        assert demo.spot_base == "https://testnet.binance.vision"

    def test_universe_rejects_plain_http(self) -> None:
        with pytest.raises(ConfigError, match="必须使用 https"):
            UniverseConfig(demo_futures_base="http://demo-fapi.binance.com")


class TestCacheIsolation:
    def test_ckey_includes_venue(self, tmp_cache_dir: Path) -> None:
        main = make_client(tmp_cache_dir, venue=Venue.MAINNET)
        demo = make_client(tmp_cache_dir, venue=Venue.DEMO)
        assert main.ckey("futures_exchange_info") != demo.ckey("futures_exchange_info")

    def test_same_venue_same_key(self, tmp_cache_dir: Path) -> None:
        a = make_client(tmp_cache_dir, venue=Venue.DEMO)
        b = make_client(tmp_cache_dir, venue=Venue.DEMO)
        assert a.ckey("x", 1) == b.ckey("x", 1)

    def test_exchange_info_cache_not_shared_across_venues(self, tmp_cache_dir: Path) -> None:
        """核心回归：缓存**不能**跨 venue 复用。

        旧实现 key 只有 topic（``make_key("futures_exchange_info")``），
        先跑 demo 再跑主网会直接拿到另一边的合约清单 —— 币池静默错位。
        """
        calls: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req.url.host)
            symbol = "DEMOONLYUSDT" if "demo" in req.url.host else "MAINONLYUSDT"
            return httpx.Response(
                200,
                json={"symbols": [{"symbol": symbol, "status": "TRADING"}]},
            )

        # 同一个进程/缓存目录，两个 venue 交替请求
        demo = make_client(tmp_cache_dir, handler=handler, venue=Venue.DEMO)
        main = make_client(tmp_cache_dir, handler=handler, venue=Venue.MAINNET)

        demo_first = demo.futures_exchange_info()
        main_first = main.futures_exchange_info()

        assert demo_first["symbols"][0]["symbol"] == "DEMOONLYUSDT"
        assert main_first["symbols"][0]["symbol"] == "MAINONLYUSDT", (
            "主网请求拿到了 demo 的缓存 = 缓存键缺少 venue 隔离"
        )
        assert len(calls) == 2, "两个 venue 各应发一次真实请求"

        # 再读一次应命中各自的缓存，不产生新请求
        assert demo.futures_exchange_info()["symbols"][0]["symbol"] == "DEMOONLYUSDT"
        assert main.futures_exchange_info()["symbols"][0]["symbol"] == "MAINONLYUSDT"
        assert len(calls) == 2, "缓存未命中，重复发请求"


class TestCoverageNamespaceIsolation:
    def test_namespaces_carry_venue(self, tmp_cache_dir: Path) -> None:
        main = make_client(tmp_cache_dir, venue=Venue.MAINNET)
        demo = make_client(tmp_cache_dir, venue=Venue.DEMO)

        assert main.historical_repository.index_namespace == "coverage_v1_mainnet"
        assert main.historical_repository.segment_namespace == "hist_segments_v1_mainnet"
        assert demo.historical_repository.index_namespace == "coverage_v1_demo"
        assert demo.historical_repository.segment_namespace == "hist_segments_v1_demo"

    def test_coverage_key_hash_differs_by_venue(self, tmp_cache_dir: Path) -> None:
        """覆盖索引文件 key = CoverageKey.key_hash()，venue 必须参与哈希，
        否则 demo 拉过的历史区间会被主网当成“已覆盖”而跳过。"""
        from cointrader.data.coverage import CoverageKey

        main = make_client(tmp_cache_dir, venue=Venue.MAINNET)
        demo = make_client(tmp_cache_dir, venue=Venue.DEMO)

        def key_for(client: BinancePublicClient) -> str:
            return CoverageKey(
                venue=client.venue_name,
                market="futures",
                dataset="funding_history",
                symbol="BTCUSDT",
            ).key_hash()

        assert key_for(main) != key_for(demo)


class TestFundingPageSizeConfig:
    """页大小按 venue 配置：主网 500，demo 200。

    2026-09-28 实测 demo 的 /fapi/v1/fundingRate 在 limit>=210 时稳定 403，
    而主网 limit=500 正常。用一个全局值必然踩雷。
    """

    def test_mainnet_and_demo_have_different_page_sizes(self, tmp_cache_dir: Path) -> None:
        main = make_client(tmp_cache_dir, venue=Venue.MAINNET)
        demo = make_client(tmp_cache_dir, venue=Venue.DEMO)
        assert main.funding_page_size == 500
        assert demo.funding_page_size == 200

    def test_page_size_follows_universe_config(self, tmp_cache_dir: Path) -> None:
        universe = UniverseConfig(mainnet_funding_page_size=300, demo_funding_page_size=120)
        main = make_client(tmp_cache_dir, venue=Venue.MAINNET, universe=universe)
        demo = make_client(tmp_cache_dir, venue=Venue.DEMO, universe=universe)
        assert main.funding_page_size == 300
        assert demo.funding_page_size == 120

    def test_page_size_actually_sent_as_limit(self, tmp_cache_dir: Path) -> None:
        """配置的页大小必须真的出现在请求参数里（不能只改属性）。"""
        seen: list[int] = []

        def handler(req: httpx.Request) -> httpx.Response:
            if req.url.path.endswith("/fundingRate"):
                seen.append(int(req.url.params.get("limit", 0)))
                return httpx.Response(200, json=[])
            return httpx.Response(200, json={"symbols": []})

        demo = make_client(tmp_cache_dir, handler=handler, venue=Venue.DEMO)
        demo.funding_history("BTCUSDT", limit=100)
        assert seen == [200], f"demo 必须用 200，实际 {seen}"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("mainnet_funding_page_size", 0),
            ("mainnet_funding_page_size", 501),
            ("demo_funding_page_size", 0),
            ("demo_funding_page_size", 201),
        ],
    )
    def test_out_of_range_page_size_rejected(self, field: str, value: int) -> None:
        with pytest.raises(ConfigError, match="funding_page_size"):
            if field == "mainnet_funding_page_size":
                UniverseConfig(mainnet_funding_page_size=value)
            else:
                UniverseConfig(demo_funding_page_size=value)
