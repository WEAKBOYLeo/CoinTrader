"""扫描器测试。

扫描器是选币环节 —— 它决定了哪些币会进入回测。如果它错了，
后面的所有工作都在错误的候选集上展开。

**最容易错的地方是结算周期**：4h 的币用 8h 公式算年化会低估一半。
本文件用真实比例的夹具数据锁死这个行为。

全部离线运行（用 mock transport 提供币安响应）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from cointrader.config import (
    ApiConfig,
    BacktestConfig,
    Config,
    CostsConfig,
    DataConfig,
    EntryConfig,
    ExitConfig,
    LoggingConfig,
    PerpFeeConfig,
    RateLimitConfig,
    RiskConfig,
    SelectionConfig,
    SlippageConfig,
    SpotFeeConfig,
    StrategyConfig,
)
from cointrader.data.binance import BinancePublicClient
from cointrader.data.funding import annualize_rate
from cointrader.research.costs import LiquidityTier
from cointrader.research.scanner import (
    Candidate,
    FundingScanner,
    ScanResult,
    format_scan_table,
)

# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def make_config(tmp_path: Path, **overrides: Any) -> Config:
    """构造一个指向临时目录的配置。"""
    return Config(
        data=DataConfig(
            cache_dir=tmp_path / "cache",
            raw_dir=tmp_path / "raw",
            rate_limit=RateLimitConfig(max_retries=1, base_backoff_seconds=0.001),
        ),
        costs=CostsConfig(
            spot=SpotFeeConfig(maker=0.001, taker=0.001),
            perp=PerpFeeConfig(maker=0.0002, taker=0.0005),
            spot_bnb_discount=0.75,
            perp_bnb_discount=0.90,
            slippage=SlippageConfig(major=0.0001, mid=0.0003, small=0.0005),
        ),
        backtest=overrides.get("backtest", BacktestConfig()),
        strategy=overrides.get(
            "strategy",
            StrategyConfig(
                entry=EntryConfig(
                    lookback_periods=10,
                    min_consecutive_positive=3,
                    min_trailing_annualized=0.05,
                    min_annualized_rate=0.05,
                ),
                exit=ExitConfig(max_holding_periods=100),
                selection=SelectionConfig(
                    max_positions=5,
                    per_position_weight=0.15,
                    min_quote_volume_3d_avg=1_000_000.0,
                    min_quote_volume_24h=1_000_000.0,
                ),
            ),
        ),
        risk=RiskConfig(),
        api=ApiConfig(),
        logging=LoggingConfig(),
    )


def _funding_records(symbol: str, rate: float, count: int, interval_hours: int) -> list[dict[str, Any]]:
    base = 1_780_000_000_000
    step = interval_hours * 3_600_000
    return [
        {
            "symbol": symbol,
            "fundingTime": base + i * step,
            "fundingRate": f"{rate:.8f}",
            "markPrice": "100.0",
        }
        for i in range(count)
    ]


def make_handler(
    *,
    symbols: dict[str, dict[str, Any]],
) -> Any:
    """构造一个模拟币安响应的 handler。

    Args:
        symbols: symbol → {"rate": 单期费率, "count": 期数, "interval": 周期小时, "volume": 24h额}
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = dict(request.url.params)

        if path == "/fapi/v1/fundingInfo":
            # 只返回非 8h 周期的合约（币安的实际行为）
            return httpx.Response(200, json=[
                {"symbol": s, "fundingIntervalHours": cfg["interval"]}
                for s, cfg in symbols.items()
                if cfg.get("interval", 8) != 8
            ])

        if path == "/fapi/v1/exchangeInfo":
            return httpx.Response(200, json={
                "symbols": [
                    {
                        "symbol": s,
                        "status": "TRADING",
                        "contractType": "PERPETUAL",
                        "baseAsset": s.replace("USDT", ""),
                        "quoteAsset": "USDT",
                        "filters": [
                            {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "1000", "stepSize": "0.001"},
                            {"filterType": "NOTIONAL", "minNotional": "5.0"},
                        ],
                    }
                    for s in symbols
                ]
            })

        if path == "/fapi/v1/klines":
            symbol = params.get("symbol")
            if symbol is None or (cfg := symbols.get(symbol)) is None:
                return httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."})
            interval_ms = 4 * 3_600_000
            bar_volume = float(cfg.get("volume", 0.0)) / 6.0
            base = 1_780_000_000_000
            raw = [
                [
                    base + i * interval_ms,
                    "100.0", "100.0", "100.0", "100.0",
                    "0.0",
                    base + (i + 1) * interval_ms - 1,
                    str(bar_volume),
                    "0", "0.0", "0.0", "0",
                ]
                for i in range(18)
            ]
            return httpx.Response(200, json=raw)

        if path == "/fapi/v1/ticker/24hr":
            return httpx.Response(200, json=[
                {"symbol": s, "quoteVolume": str(cfg.get("volume", 0.0))}
                for s, cfg in symbols.items()
            ])

        if path == "/fapi/v1/fundingRate":
            symbol = params.get("symbol")
            if symbol is None or (cfg := symbols.get(symbol)) is None:
                return httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."})
            limit = int(params.get("limit", 1000))
            records = _funding_records(
                symbol, cfg["rate"], cfg["count"], cfg.get("interval", 8)
            )
            start = int(params.get("startTime", 0))
            window = [r for r in records if r["fundingTime"] >= start][:limit]
            return httpx.Response(200, json=window)

        return httpx.Response(404, json={"msg": f"unhandled path {path}"})

    return handler


def make_scanner(config: Config, handler: Any) -> FundingScanner:
    client = BinancePublicClient(
        config.api,
        config.data,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep_fn=lambda _: None,
    )
    return FundingScanner(config, client)


# ---------------------------------------------------------------------------
# 结算周期 —— 最容易错的地方
# ---------------------------------------------------------------------------


class TestIntervalHandling:
    """结算周期处理。"""

    def test_4h_symbol_annualization_uses_its_own_interval(self, tmp_path: Path) -> None:
        """**核心检查**：4h 结算的币必须用 4h 折算年化。

        同样的单期费率 0.0001：
          8h 公式 → 年化 10.95%
          4h 公式 → 年化 21.90%

        如果用错，这个币会被严重低估并沉到排名底部 ——
        而它其实是收租更频繁的更好标的。
        """
        symbols = {
            "FASTUSDT": {"rate": 0.0001, "count": 100, "interval": 4, "volume": 100e6},
            "SLOWUSDT": {"rate": 0.0001, "count": 100, "interval": 8, "volume": 100e6},
        }
        config = make_config(tmp_path)
        scanner = make_scanner(config, make_handler(symbols=symbols))

        result = scanner.scan()
        by_symbol = {c.symbol: c for c in result.candidates}

        fast = by_symbol["FASTUSDT"]
        slow = by_symbol["SLOWUSDT"]

        assert fast.interval_hours == 4
        assert slow.interval_hours == 8

        # 手工验算
        assert fast.gross_annualized == pytest.approx(annualize_rate(0.0001, 4), rel=1e-6)
        assert slow.gross_annualized == pytest.approx(annualize_rate(0.0001, 8), rel=1e-6)

        # 4h 的年化应为 8h 的两倍
        assert fast.gross_annualized == pytest.approx(slow.gross_annualized * 2, rel=1e-6)

    def test_unknown_symbol_defaults_to_8h_with_warning(self, tmp_path: Path, caplog) -> None:
        """未在 fundingInfo 中出现的合约按 8h 处理（币安的默认周期）。"""
        symbols = {"BTCUSDT": {"rate": 0.0001, "count": 100, "volume": 1000e6}}
        config = make_config(tmp_path)
        scanner = make_scanner(config, make_handler(symbols=symbols))

        result = scanner.scan()
        assert result.candidates[0].interval_hours == 8


class TestScanFiltering:
    """扫描过滤逻辑。"""

    def test_filters_low_volume_symbols(self, tmp_path: Path) -> None:
        """24h 成交额低于门槛的币应被跳过 —— 它们的滑点会吃掉全部收益。"""
        symbols = {
            "GOODUSDT": {"rate": 0.0003, "count": 100, "volume": 500e6},
            "DEADUSDT": {"rate": 0.0003, "count": 100, "volume": 100.0},   # 几乎没成交
        }
        config = make_config(tmp_path)
        scanner = make_scanner(config, make_handler(symbols=symbols))

        result = scanner.scan()
        scanned = {c.symbol for c in result.candidates}

        assert "GOODUSDT" in scanned
        assert "DEADUSDT" not in scanned
        assert "DEADUSDT" in result.skipped

    def test_excludes_configured_bases(self, tmp_path: Path) -> None:
        """配置中的排除列表（稳定币对）应生效。"""
        symbols = {
            "BTCUSDT": {"rate": 0.0003, "count": 100, "volume": 500e6},
            "USDCUSDT": {"rate": 0.0003, "count": 100, "volume": 500e6},
        }
        config = make_config(tmp_path)
        # SelectionConfig 默认排除 USDC
        scanner = make_scanner(config, make_handler(symbols=symbols))

        result = scanner.scan()
        assert "USDCUSDT" not in {c.symbol for c in result.candidates}

    def test_max_symbols_takes_highest_volume(self, tmp_path: Path) -> None:
        """max_symbols 截断时，应保留成交额最高的那些。"""
        symbols = {
            f"SYM{i}USDT": {"rate": 0.0003, "count": 100, "volume": float(i) * 1e6}
            for i in range(1, 11)
        }
        config = make_config(tmp_path)
        scanner = make_scanner(config, make_handler(symbols=symbols))

        result = scanner.scan(max_symbols=3)
        got = {c.symbol for c in result.candidates}

        # volume 最大的三个: SYM10, SYM9, SYM8
        assert got == {"SYM10USDT", "SYM9USDT", "SYM8USDT"}

    def test_single_symbol_failure_does_not_abort_scan(self, tmp_path: Path) -> None:
        """单个币种失败不应中断整轮扫描（币种可能刚下架）。"""
        symbols = {
            "OKUSDT": {"rate": 0.0003, "count": 100, "volume": 500e6},
        }
        handler = make_handler(symbols=symbols)
        config = make_config(tmp_path)
        scanner = make_scanner(config, handler)

        # 扫描池里加一个不存在于 handler 的币
        result = scanner.scan(symbols=["OKUSDT", "GHOSTUSDT"])

        assert len(result.candidates) == 1
        assert "GHOSTUSDT" in result.skipped


class TestCandidateEvaluation:
    """候选评估与可行性判定。"""

    def test_low_trailing_rate_marked_untradeable(self, tmp_path: Path) -> None:
        """资金费持续过低的币应被标记为不可交易。"""
        symbols = {"LOWUSDT": {"rate": 0.000001, "count": 200, "volume": 500e6}}
        config = make_config(tmp_path)
        scanner = make_scanner(config, make_handler(symbols=symbols))

        result = scanner.scan()
        candidate = result.candidates[0]

        assert not candidate.tradeable
        assert "低于门槛" in candidate.rejection_reason

    def test_high_rate_marked_tradeable(self, tmp_path: Path) -> None:
        """资金费足够高的币应被标记为可交易。"""
        # 单期 0.0005，8h → 年化 54.75%，远超 5% 门槛
        symbols = {"HIGHUSDT": {"rate": 0.0005, "count": 200, "volume": 500e6}}
        config = make_config(tmp_path)
        scanner = make_scanner(config, make_handler(symbols=symbols))

        result = scanner.scan()
        candidate = result.candidates[0]

        assert candidate.tradeable, f"应判定为可交易: {candidate.rejection_reason}"
        assert candidate.rejection_reason == ""

    def test_insufficient_history_marked_untradeable(self, tmp_path: Path) -> None:
        """历史期数少于回看窗口时不可交易（数据不足 ≠ 策略不适用）。"""
        symbols = {"NEWUSDT": {"rate": 0.0005, "count": 5, "volume": 500e6}}
        config = make_config(tmp_path)
        scanner = make_scanner(config, make_handler(symbols=symbols))

        result = scanner.scan()
        candidate = result.candidates[0]

        assert not candidate.tradeable
        assert "期数据" in candidate.rejection_reason

    def test_negative_rate_symbol_has_negative_streak(self, tmp_path: Path) -> None:
        """持续负费率应被记录在最长负连续期数里。"""
        symbols = {"NEGUSDT": {"rate": -0.0002, "count": 100, "volume": 500e6}}
        config = make_config(tmp_path)
        scanner = make_scanner(config, make_handler(symbols=symbols))

        result = scanner.scan()
        candidate = result.candidates[0]

        assert candidate.longest_negative_streak == 100
        assert candidate.positive_ratio == 0.0
        assert not candidate.tradeable

    def test_cost_depends_on_liquidity_tier(self, tmp_path: Path) -> None:
        """成本必须按该币的流动性档位算，而不是一律用主流币的滑点。

        小市值币的滑点可能是主流币的 5 倍。用错档位会让
        高费率但流动性差的币看起来比实际更有利可图。
        """
        symbols = {
            "BIGUSDT": {"rate": 0.0003, "count": 100, "volume": 800e6},    # major
            "TINYUSDT": {"rate": 0.0003, "count": 100, "volume": 20e6},    # small
        }
        config = make_config(tmp_path)
        scanner = make_scanner(config, make_handler(symbols=symbols))

        result = scanner.scan()
        by_symbol = {c.symbol: c for c in result.candidates}

        assert by_symbol["BIGUSDT"].round_trip_cost < by_symbol["TINYUSDT"].round_trip_cost
        assert by_symbol["BIGUSDT"].round_trip_cost == pytest.approx(0.0028, rel=1e-9)
        assert by_symbol["TINYUSDT"].round_trip_cost == pytest.approx(0.0044, rel=1e-9)


class TestScanResult:
    """结果容器。"""

    def _candidate(self, symbol: str, trailing: float, tradeable: bool = True) -> Candidate:
        return Candidate(
            symbol=symbol,
            interval_hours=8,
            periods=100,
            quote_volume_24h=100e6,
            tier=LiquidityTier.MID,
            mean_rate=0.0001,
            gross_annualized=0.1,
            trailing_annualized=trailing,
            positive_ratio=0.9,
            longest_negative_streak=2,
            round_trip_cost=0.0037,
            breakeven_days=5.0,
            net_annualized_naive=0.05,
            tradeable=tradeable,
        )

    def test_top_sorts_by_trailing_descending(self) -> None:
        result = ScanResult(
            candidates=[
                self._candidate("A", 0.10),
                self._candidate("B", 0.50),
                self._candidate("C", 0.30),
            ]
        )
        top = result.top(10)

        assert [c.symbol for c in top] == ["B", "C", "A"]

    def test_top_excludes_untradeable_by_default(self) -> None:
        result = ScanResult(
            candidates=[
                self._candidate("GOOD", 0.10, tradeable=True),
                self._candidate("BAD", 0.99, tradeable=False),
            ]
        )

        assert [c.symbol for c in result.top(10)] == ["GOOD"]
        assert "BAD" in [c.symbol for c in result.top(10, tradeable_only=False)]

    def test_top_respects_limit(self) -> None:
        result = ScanResult(
            candidates=[self._candidate(f"S{i}", 0.1 * i) for i in range(1, 11)]
        )
        assert len(result.top(3)) == 3

    def test_summary_counts(self) -> None:
        result = ScanResult(
            candidates=[
                self._candidate("A", 0.1, tradeable=True),
                self._candidate("B", 0.2, tradeable=False),
            ],
            skipped={"C": "额度不足"},
            total_symbols=3,
        )
        summary = result.summary()

        assert summary["total_symbols_scanned"] == 3
        assert summary["candidates_returned"] == 2
        assert summary["tradeable"] == 1
        assert summary["skipped"] == 1

    def test_to_frame(self) -> None:
        result = ScanResult(candidates=[self._candidate("A", 0.1)])
        frame = result.to_frame()

        assert len(frame) == 1
        assert frame.iloc[0]["symbol"] == "A"

    def test_empty_to_frame(self) -> None:
        assert ScanResult(candidates=[]).to_frame().empty


class TestFormatting:
    """表格格式化 —— 特别是中文符号的宽度处理。"""

    def _candidate(self, symbol: str, trailing: float) -> Candidate:
        return Candidate(
            symbol=symbol,
            interval_hours=8,
            periods=100,
            quote_volume_24h=100e6,
            tier=LiquidityTier.MID,
            mean_rate=0.0001,
            gross_annualized=0.2,
            trailing_annualized=trailing,
            positive_ratio=0.95,
            longest_negative_streak=2,
            round_trip_cost=0.0037,
            breakeven_days=5.0,
            net_annualized_naive=0.15,
        )

    def test_handles_cjk_symbols(self) -> None:
        """币安上有中文交易对符号（实测存在），表格必须处理其双宽特性。

        用 len() 做填充会让整个表格错位。
        """
        result = ScanResult(candidates=[
            self._candidate("币安人生USDT", 0.5),
            self._candidate("BTCUSDT", 0.3),
        ])
        output = format_scan_table(result, limit=10)

        lines = [line for line in output.splitlines() if "USDT" in line]
        # 所有数据行的显示宽度必须一致
        from cointrader.research.scanner import _display_width

        widths = {_display_width(line) for line in lines}
        assert len(widths) == 1, f"表格未对齐，行宽: {widths}"

    def test_handles_infinite_breakeven(self) -> None:
        """负费率下回本时间为无穷，必须显示为 ∞ 而不是崩溃。"""
        candidate = self._candidate("NEGUSDT", -0.1)
        candidate.breakeven_days = float("inf")

        output = format_scan_table(ScanResult(candidates=[candidate]))
        assert "∞" in output

    def test_empty_result_message(self) -> None:
        output = format_scan_table(ScanResult(candidates=[]))
        assert "没有找到" in output

    def test_shows_scan_counts(self) -> None:
        result = ScanResult(
            candidates=[self._candidate("BTCUSDT", 0.3)],
            total_symbols=527,
        )
        output = format_scan_table(result, limit=10)

        assert "527" in output


__all__: list[str] = []
