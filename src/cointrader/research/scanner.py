"""候选币种扫描 —— 全市场资金费横向比较。

## 这个模块回答的问题

「**哪些币**的资金费水平，在扣除真实成本后仍值得建仓？」

这是把整个框架的零件串起来的第一步。它做的事：

1. 拉全市场永续合约列表（过滤掉不可交易的）
2. 拉最近 3 天平均日成交额（用于流动性分档、过滤死币）
3. **逐币**拉资金费历史 —— 注意是逐币，因为每个币的结算周期不同
4. 用该币**自己的周期**折算年化
5. 扣除成本算出净收益，排序

## 为什么必须逐币处理周期

实测数据：782 个永续合约中，4h 结算的 466 个、8h 的 312 个、1h 的 4 个。

用统一的 8h 公式会给 4h 的币算出一个**只有真实值一半**的年化，
让它们全部沉到排名底部 —— 而它们恰恰是收租更频繁、回本更快的标的。
反之亦然。这是本模块唯一不能妥协的地方。

## 关于速率

全市场扫描要发起几百次请求。虽然每次都有磁盘缓存，首次全量扫描
仍可能耗时数分钟。``max_symbols`` 参数用于限制范围（调试时很有用）。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..config import Config
from ..data.binance import BinancePublicClient
from ..data.funding import (
    FundingIntervals,
    annualize_rate,
    consecutive_positive_streak,
    fetch_funding_history,
    fetch_funding_intervals,
    trailing_annualized,
)
from ..data.klines import fetch_historical_quote_volume_3d_avg, tradable_perpetuals
from ..errors import CoinTraderError, InsufficientDataError
from ..research.costs import CostModel, LiquidityTier, classify_liquidity

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Candidate:
    """一个候选币种的扫描结果。"""

    symbol: str
    interval_hours: int
    periods: int
    quote_volume_24h: float  # 实际语义：最近 3 天平均日成交额
    tier: LiquidityTier

    # -- 毛收益口径（未扣费）--
    mean_rate: float                 # 单期平均费率
    gross_annualized: float          # 全期平均年化
    trailing_annualized: float       # 最近窗口的年化（决策用）
    positive_ratio: float            # 费率为正的期数占比
    longest_negative_streak: int     # 最长连续负费率期数

    # -- 成本与净收益 --
    round_trip_cost: float           # 往返成本（占名义额）
    breakeven_days: float            # 按 trailing 年化算的回本天数
    net_annualized_naive: float      # 未扣除回本期的朴素净年化（仅供排序）

    # -- 可行性 --
    tradeable: bool = True
    rejection_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval_hours": self.interval_hours,
            "periods": self.periods,
            "quote_volume_3d_avg": round(self.quote_volume_24h, 2),
            "tier": self.tier.value,
            "mean_rate": self.mean_rate,
            "gross_annualized": round(self.gross_annualized, 4),
            "trailing_annualized": round(self.trailing_annualized, 4),
            "positive_ratio": round(self.positive_ratio, 4),
            "longest_negative_streak": self.longest_negative_streak,
            "round_trip_cost": round(self.round_trip_cost, 6),
            "breakeven_days": (
                round(self.breakeven_days, 2)
                if self.breakeven_days != float("inf")
                else None
            ),
            "net_annualized_naive": (
                round(self.net_annualized_naive, 4)
                if self.net_annualized_naive != float("-inf")
                else None
            ),
            "tradeable": self.tradeable,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(slots=True)
class ScanResult:
    """一次扫描的完整结果。"""

    candidates: list[Candidate]
    skipped: dict[str, str] = field(default_factory=dict)   # symbol → 跳过原因
    total_symbols: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    def top(self, n: int = 20, *, tradeable_only: bool = True) -> list[Candidate]:
        """按 trailing 年化取前 n 名。"""
        pool = [c for c in self.candidates if c.tradeable] if tradeable_only else self.candidates
        return sorted(pool, key=lambda c: c.trailing_annualized, reverse=True)[:n]

    def summary(self) -> dict[str, Any]:
        tradeable = [c for c in self.candidates if c.tradeable]
        return {
            "total_symbols_scanned": self.total_symbols,
            "candidates_returned": len(self.candidates),
            "tradeable": len(tradeable),
            "skipped": len(self.skipped),
            "client_stats": self.stats,
        }

    def to_frame(self) -> pd.DataFrame:
        """转为 DataFrame，便于排序、过滤、导出 CSV。"""
        if not self.candidates:
            return pd.DataFrame()
        return pd.DataFrame([c.as_dict() for c in self.candidates])


class FundingScanner:
    """资金费候选扫描器。

    Args:
        config: 完整配置。
        client: 数据客户端。为 None 时自动构造。
    """

    def __init__(self, config: Config, client: BinancePublicClient | None = None) -> None:
        self.config = config
        self.client = client or BinancePublicClient(config.api, config.data)
        self.cost_model = CostModel(config.costs)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> FundingScanner:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------

    def scan(
        self,
        *,
        symbols: Iterable[str] | None = None,
        max_symbols: int | None = None,
        lookback_days: int | None = None,
        min_quote_volume: float | None = None,
    ) -> ScanResult:
        """扫描全市场（或指定子集）。

        Args:
            symbols: 指定币种列表。为 None 时扫描全市场可交易永续。
            max_symbols: 最多扫描多少个币（按成交额降序取前 N）。
            lookback_days: 回看天数。None 表示取全部可得历史。
            min_quote_volume: 最小 3 天平均日成交额过滤。None 时用配置值。

        Returns:
            ScanResult。
        """
        sel = self.config.strategy.selection
        min_volume = (
            min_quote_volume
            if min_quote_volume is not None
            else sel.min_quote_volume_3d_avg
        )

        logger.info("开始扫描资金费候选...")
        intervals = fetch_funding_intervals(self.client)
        futures_info = self.client.futures_exchange_info()

        # 1. 确定候选池
        if symbols is not None:
            universe = sorted(set(symbols))
        else:
            universe = tradable_perpetuals(
                futures_info,
                exclude_bases=sel.exclude_bases,
            )
        logger.info("候选池: %d 个永续合约", len(universe))

        # 2. 成交额过滤（用于剔除死币 —— 它们的滑点会吃掉全部收益）
        volumes = self._fetch_volumes(universe)
        filtered: list[str] = []
        skipped: dict[str, str] = {}

        for symbol in universe:
            volume = volumes.get(symbol, 0.0)
            if volume < min_volume:
                skipped[symbol] = f"3天平均日成交额 {volume:,.0f} < {min_volume:,.0f}"
                continue
            filtered.append(symbol)

        # 按成交额降序，保证 max_symbols 截断时保留流动性最好的
        filtered.sort(key=lambda s: volumes.get(s, 0.0), reverse=True)
        if max_symbols is not None:
            for symbol in filtered[max_symbols:]:
                skipped[symbol] = f"超出 max_symbols={max_symbols} 限制"
            filtered = filtered[:max_symbols]

        logger.info("通过成交额过滤: %d 个（跳过 %d 个）", len(filtered), len(skipped))

        # 3. 逐币扫描
        candidates: list[Candidate] = []
        for index, symbol in enumerate(filtered, start=1):
            if index % 25 == 0:
                logger.info("扫描进度: %d/%d", index, len(filtered))

            try:
                candidate = self._scan_symbol(symbol, intervals, volumes, lookback_days)
            except InsufficientDataError as exc:
                skipped[symbol] = f"数据不足: {exc}"
                continue
            except CoinTraderError as exc:
                # 单个币失败不应中断整轮扫描（币种可能刚下架）
                skipped[symbol] = f"{type(exc).__name__}: {exc}"
                continue

            if candidate is not None:
                candidates.append(candidate)

        logger.info("扫描完成: 得到 %d 个候选", len(candidates))

        return ScanResult(
            candidates=candidates,
            skipped=skipped,
            total_symbols=len(universe),
            stats=self.client.stats.as_dict(),
        )

    # ------------------------------------------------------------------

    def _fetch_volumes(self, symbols: Iterable[str]) -> dict[str, float]:
        """拉取每个币最近 3 天平均日成交额。

        只使用已收盘的 4h K 线；缺少完整 72h 数据的币种不通过准入。
        """
        import time

        volumes: dict[str, float] = {}
        start_ms = int((time.time() - 5 * 86_400) * 1000)
        for symbol in symbols:
            try:
                series = fetch_historical_quote_volume_3d_avg(
                    self.client, symbol, start_ms=start_ms
                )
                if not series.empty and pd.notna(series.iloc[-1]):
                    volumes[symbol] = float(series.iloc[-1])
            except CoinTraderError as exc:
                logger.debug("%s 3天成交额获取失败: %s", symbol, exc)
        return volumes

    def _scan_symbol(
        self,
        symbol: str,
        intervals: FundingIntervals,
        volumes: dict[str, float],
        lookback_days: int | None,
    ) -> Candidate | None:
        """扫描单个币种。"""
        interval_hours = intervals.get(symbol)

        start_ms = None
        if lookback_days is not None:
            import time

            start_ms = int((time.time() - lookback_days * 86_400) * 1000)

        frame = fetch_funding_history(self.client, symbol, start_ms=start_ms)
        rates = frame["funding_rate"]

        if rates.empty:
            return None

        # 该币**自己的**周期折算（不是硬编码 8）
        annualized = annualize_rate(rates, interval_hours)

        # 决策用的 trailing 年化：最近 lookback_periods 期
        window = min(self.config.strategy.entry.lookback_periods, len(rates))
        trailing_series = trailing_annualized(rates, interval_hours, window=window)
        trailing = float(trailing_series.iloc[-1]) if pd.notna(trailing_series.iloc[-1]) else 0.0

        volume = volumes.get(symbol, 0.0)
        tier = classify_liquidity(volume)

        # 成本按该币的流动性档位算
        round_trip = self.cost_model.entry_cost(tier) + self.cost_model.exit_cost(tier)
        breakeven_days = self.cost_model.breakeven_days(
            float(rates.mean()), interval_hours, tier
        )

        # 朴素净年化：仅供排序。用 trailing 年化减去「摊销后的年化成本」。
        # 摊销期数取 max_holding_periods（假设按最长持有期摊薄成本）。
        # 注意：这不是最终收益预期，只是为了给候选排个序。
        holding_periods = self.config.strategy.exit.max_holding_periods
        amortized_cost_annual = (
            round_trip / holding_periods * (24.0 / interval_hours) * 365.0
        )
        net_annualized_naive = trailing - amortized_cost_annual

        # 连续为正的最长段（用于判断费率是否稳定）
        streak_series = consecutive_positive_streak(rates)
        longest_pos = int(streak_series.max()) if len(streak_series) else 0

        candidate = Candidate(
            symbol=symbol,
            interval_hours=interval_hours,
            periods=len(rates),
            quote_volume_24h=volume,
            tier=tier,
            mean_rate=float(rates.mean()),
            gross_annualized=float(annualized.mean()),
            trailing_annualized=trailing,
            positive_ratio=float((rates > 0).mean()),
            longest_negative_streak=self._longest_negative(rates),
            round_trip_cost=round_trip,
            breakeven_days=breakeven_days,
            net_annualized_naive=net_annualized_naive,
        )

        # 可行性判定
        min_periods = self.config.strategy.entry.lookback_periods
        if len(rates) < min_periods:
            candidate.tradeable = False
            candidate.rejection_reason = f"仅 {len(rates)} 期数据，少于回看窗口 {min_periods}"
        elif trailing < self.config.strategy.entry.min_trailing_annualized:
            candidate.tradeable = False
            candidate.rejection_reason = (
                f"trailing 年化 {trailing:.2%} 低于门槛 "
                f"{self.config.strategy.entry.min_trailing_annualized:.2%}"
            )
        elif longest_pos < self.config.strategy.entry.min_consecutive_positive:
            candidate.tradeable = False
            candidate.rejection_reason = (
                f"最长连续正费率 {longest_pos} 期，少于要求 "
                f"{self.config.strategy.entry.min_consecutive_positive}"
            )

        return candidate

    @staticmethod
    def _longest_negative(rates: pd.Series) -> int:
        """最长连续负费率期数。决定资金需要能撑多久。"""
        is_negative = rates < 0
        longest = current = 0
        for flag in is_negative.to_numpy():
            if flag:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return int(longest)


def _display_width(text: str) -> int:
    """计算字符串在等宽终端里的显示宽度。

    币安上真实存在非 ASCII 的交易对符号（实测有 ``币安人生USDT``、
    ``龙虾USDT``、``我踏马来了USDT`` 等）。这些字符在等宽字体里占**两个**
    字符宽度，用 ``len()`` 做填充会让整个表格错位。
    """
    import unicodedata

    width = 0
    for char in text:
        # East Asian Wide / Fullwidth 占两格
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def _pad(text: str, width: int, *, align: str = "left") -> str:
    """按**显示宽度**填充字符串。"""
    padding = max(0, width - _display_width(text))
    if align == "right":
        return " " * padding + text
    if align == "center":
        left = padding // 2
        return " " * left + text + " " * (padding - left)
    return text + " " * padding


def format_scan_table(result: ScanResult, *, limit: int = 20) -> str:
    """把扫描结果格式化为终端表格（纯文本，不依赖 rich）。

    列宽按显示宽度计算，正确处理中文符号（占两格）。
    """
    top = result.top(limit)
    if not top:
        return "没有找到满足条件的候选币种。"

    # 列宽：取表头与所有数据行的最大显示宽度
    symbol_width = max([_display_width("币种"), *(_display_width(c.symbol) for c in top)])

    header_cells = [
        _pad("币种", symbol_width),
        _pad("周期", 5, align="right"),
        _pad("年化(毛)", 10, align="right"),
        _pad("年化(近)", 10, align="right"),
        _pad("正费率", 8, align="right"),
        _pad("回本(天)", 9, align="right"),
        _pad("档位", 6, align="right"),
        _pad("3天均额(百万)", 13, align="right"),
    ]
    header = "  ".join(header_cells)
    lines = [header, "-" * _display_width(header)]

    for candidate in top:
        breakeven = (
            f"{candidate.breakeven_days:.1f}"
            if candidate.breakeven_days != float("inf")
            else "∞"
        )
        row = [
            _pad(candidate.symbol, symbol_width),
            _pad(f"{candidate.interval_hours}h", 5, align="right"),
            _pad(f"{candidate.gross_annualized:.2%}", 10, align="right"),
            _pad(f"{candidate.trailing_annualized:.2%}", 10, align="right"),
            _pad(f"{candidate.positive_ratio:.1%}", 8, align="right"),
            _pad(breakeven, 9, align="right"),
            _pad(candidate.tier.value, 6, align="right"),
            _pad(f"{candidate.quote_volume_24h / 1e6:,.1f}", 13, align="right"),
        ]
        lines.append("  ".join(row))

    lines.append("")
    lines.append(
        f"共扫描 {result.total_symbols} 个合约，{len(result.candidates)} 个有数据，"
        f"其中 {len([c for c in result.candidates if c.tradeable])} 个满足条件。"
    )
    if len(top) < len([c for c in result.candidates if c.tradeable]):
        lines.append(f"（下表仅展示前 {len(top)} 名，用 --top 调整）")
    lines.append("提示: '年化(毛)' 为全历史平均，'年化(近)' 为最近窗口 —— 决策应以后者为准。")
    lines.append("      扣费后的单币结论需运行滚动回测（cointrader backtest <SYMBOL>）。")
    lines.append("      多币组合请用 portfolio；它会按历史事件动态选币，不能回填未来赢家。")

    return "\n".join(lines)


__all__ = ["Candidate", "FundingScanner", "ScanResult", "format_scan_table"]
