"""资金费率数据 —— 拉取、归一化、年化。

本模块处理本项目**最关键的数据正确性问题**：结算周期不统一。

币安永续合约的资金费结算周期实测有三种：

===========  ======
周期          合约数
===========  ======
4 小时        466
8 小时        312
1 小时        4
===========  ======

如果用统一的 8 小时公式折算年化，4 小时结算的币会被**低估一半**，
直接导致筛选环节漏掉最好的标的。反之若统一按 4 小时折算 8 小时的币，
则会**高估一倍**，把不该进的仓位放进来。

因此本模块的硬规则是：**年化必须按每个合约自己的周期折算**，
周期从 ``/fapi/v1/fundingInfo`` 拉取，绝不硬编码。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, overload

import pandas as pd

from ..errors import InsufficientDataError, ParseError
from .binance import BinancePublicClient

logger = logging.getLogger(__name__)

#: 拿不到 fundingInfo 时的兜底周期（小时）。仅作为最后手段，会打警告。
FALLBACK_INTERVAL_HOURS = 8

#: 资金费历史的保留期限（天）。币安只保留约一年，影响回测跨度上限。
FUNDING_HISTORY_RETENTION_DAYS = 365


@dataclass(frozen=True, slots=True)
class FundingIntervals:
    """合约 → 结算周期（小时）的映射。"""

    mapping: dict[str, int]

    def get(self, symbol: str) -> int:
        """取某合约的结算周期（小时）。

        缺失时返回兜底值并打警告 —— 但**不静默**，因为静默兜底
        正是年化算错的根源。
        """
        interval = self.mapping.get(symbol)
        if interval is None:
            logger.warning(
                "合约 %s 无结算周期记录，兜底按 %dh 折算。这可能导致年化不准，请核查。",
                symbol,
                FALLBACK_INTERVAL_HOURS,
            )
            return FALLBACK_INTERVAL_HOURS
        return interval

    def periods_per_year(self, symbol: str) -> float:
        """某合约每年的结算次数。"""
        return (24.0 / self.get(symbol)) * 365.0

    def __len__(self) -> int:
        return len(self.mapping)


def fetch_funding_intervals(client: BinancePublicClient) -> FundingIntervals:
    """拉取全市场资金费结算周期。

    Returns:
        FundingIntervals 对象。

    Notes:
        ``/fapi/v1/fundingInfo`` **只返回非默认周期的合约**（即只列出 4h/1h 的，
        8h 的不出现在响应里）。因此未出现在响应中的合约都应视为 8h 默认值。
        这一点如果理解错，会把绝大多数合约误判为未知周期。
    """
    raw = client.funding_info()

    mapping: dict[str, int] = {}
    for item in raw:
        symbol = item.get("symbol")
        hours = item.get("fundingIntervalHours")
        if symbol is None or hours is None:
            continue
        try:
            mapping[str(symbol)] = int(hours)
        except (TypeError, ValueError):
            logger.warning("结算周期字段异常，跳过: %s", item)
            continue

    logger.info("已获取 %d 个非默认周期的合约（其余按 %dh 处理）", len(mapping), FALLBACK_INTERVAL_HOURS)
    return FundingIntervals(mapping=mapping)


def fetch_funding_history(
    client: BinancePublicClient,
    symbol: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> pd.DataFrame:
    """拉取单个合约的资金费历史并规范化为 DataFrame。

    Args:
        client: 只读客户端。
        symbol: 合约符号。
        start_ms: 起始时间（毫秒）。
        end_ms: 结束时间（毫秒）。

    Returns:
        DataFrame，索引为 UTC 时间戳，列为 ``funding_rate`` / ``mark_price``。
        按时间升序，无重复。

    Raises:
        InsufficientDataError: 该合约没有任何资金费记录。
        ParseError: 字段缺失或格式异常。
    """
    records = client.funding_history(symbol, start_ms=start_ms, end_ms=end_ms)
    if not records:
        raise InsufficientDataError(f"合约 {symbol} 无资金费历史记录")

    normalized: list[dict[str, Any]] = []
    for record in records:
        try:
            normalized.append(
                {
                    "timestamp": int(record["fundingTime"]),
                    "funding_rate": float(record["fundingRate"]),
                    "mark_price": float(record.get("markPrice") or "nan"),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ParseError(f"资金费记录字段异常 {symbol}: {record!r} ({exc})") from exc

    frame = pd.DataFrame(normalized)

    # 币安的 fundingTime 偶尔有 ±几毫秒误差（实测见 1789488000001），
    # 直接 drop_duplicates 会漏掉真正的重复。round 到秒再判重。
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.drop_duplicates(subset="timestamp", keep="last")
    frame = frame.sort_values("timestamp").set_index("timestamp")

    return frame[["funding_rate", "mark_price"]]


def normalize_funding_to_8h(rates: pd.Series, interval_hours: int) -> pd.Series:
    """把资金费事件聚合到统一 8h 结算桶。

    桶右端是可见时点。每个事件只归入它所在的 8h 结束桶，聚合后的费率
    只包含该桶结束时刻及之前已经发生的资金费事件。
    """
    if interval_hours <= 0:
        raise ValueError(f"结算周期必须为正，当前 {interval_hours}")
    if rates.empty:
        return rates.astype(float).rename("funding_rate")
    if not rates.index.is_monotonic_increasing:
        raise ValueError("资金费率序列必须按时间升序")

    index = pd.DatetimeIndex(rates.index)
    # 8h 整点事件属于刚刚结束的桶；4h/1h 事件向上归入最近的 8h 结束点。
    bucket = index.ceil("8h")
    normalized = rates.astype(float).groupby(bucket).sum().sort_index()
    start = normalized.index[0]
    end = normalized.index[-1]
    full_index = pd.date_range(start=start, end=end, freq="8h", tz=start.tz)
    normalized = normalized.reindex(full_index, fill_value=0.0)
    normalized.name = rates.name or "funding_rate"
    return normalized


@overload
def annualize_rate(rate: float, interval_hours: int) -> float: ...


@overload
def annualize_rate(rate: pd.Series, interval_hours: int) -> pd.Series: ...


def annualize_rate(rate: float | pd.Series, interval_hours: int) -> float | pd.Series:
    """把单期资金费率折算为年化。

    公式::

        年化 = rate * (24 / interval_hours) * 365

    这是本项目最重要的一行数学。用错周期的后果见模块文档。

    Args:
        rate: 单期费率（小数，如 0.0001 表示 0.01%）。标量或 Series。
        interval_hours: 结算周期（小时）。必须来自 ``fundingInfo``，不可硬编码。

    Returns:
        年化费率。输入 Series 则返回 Series（保持索引）。

    Note:
        有两个 ``@overload`` 声明，让类型检查器在传入 Series 时
        正确推断出返回 Series —— 否则调用方拿到 Series 却被告知是 float，
        后续的 ``.mean()`` 之类调用会被误报为类型错误。
    """
    if interval_hours <= 0:
        raise ValueError(f"结算周期必须为正，当前 {interval_hours}")
    return rate * (24.0 / interval_hours) * 365.0


def deannualize_rate(annualized: float, interval_hours: int) -> float:
    """年化 → 单期费率的反函数。用于把策略阈值换算回单期口径。"""
    if interval_hours <= 0:
        raise ValueError(f"结算周期必须为正，当前 {interval_hours}")
    return annualized / ((24.0 / interval_hours) * 365.0)


def trailing_annualized(
    rates: pd.Series,
    interval_hours: int,
    *,
    window: int,
) -> pd.Series:
    """滚动窗口的年化资金费率。

    用于策略信号：判断"最近的资金费水平"而非"全历史平均"。
    全历史平均会把三个月前的行情混进今天的决策。

    Args:
        rates: 单期费率序列（升序）。
        interval_hours: 结算周期。
        window: 回看期数。

    Returns:
        与输入等长的 Series。前 ``window - 1`` 个位置为 NaN
        （数据不足，不填充 —— 填充会制造虚假信号）。
    """
    if window <= 0:
        raise ValueError(f"window 必须为正，当前 {window}")
    rolling_mean = rates.rolling(window=window, min_periods=window).mean()
    return annualize_rate(rolling_mean, interval_hours)


def consecutive_positive_streak(rates: pd.Series) -> pd.Series:
    """每个位置上「截至当前连续为正」的期数。

    用于进场条件：只有资金费连续为正足够久，才说明这不是噪音。

    Returns:
        与输入等长的整数 Series。当前值为负时记 0。
    """
    is_positive = rates > 0
    # 分组技巧：每次遇到 False 就开一个新组，组内计数即为连续为正的长度
    groups = (~is_positive).cumsum()
    return is_positive.groupby(groups).cumsum().astype(int)


def consecutive_negative_streak(rates: pd.Series) -> pd.Series:
    """每个位置上「截至当前连续为负」的期数。用于出场条件。"""
    is_negative = rates < 0
    groups = (~is_negative).cumsum()
    return is_negative.groupby(groups).cumsum().astype(int)


def summarize_funding(
    frame: pd.DataFrame,
    symbol: str,
    interval_hours: int,
) -> dict[str, Any]:
    """对单个合约的资金费历史做摘要统计。

    这些数字是选币环节的直接依据，也是回测报告的基础。

    Returns:
        含以下键的字典：

        - ``symbol`` / ``interval_hours`` / ``periods`` / ``start`` / ``end``
        - ``mean_rate`` / ``mean_annualized`` —— 全期平均
        - ``positive_ratio`` —— 费率为正的比例（决定了策略的胜率上限）
        - ``max_annualized`` / ``min_annualized`` —— 极值
        - ``longest_negative_streak`` —— 最长连续负费率期数（决定资金能撑多久）
    """
    if frame.empty:
        raise InsufficientDataError(f"{symbol} 资金费数据为空")

    rates = frame["funding_rate"]
    periods = len(rates)
    annualized = annualize_rate(rates, interval_hours)

    return {
        "symbol": symbol,
        "interval_hours": interval_hours,
        "periods": periods,
        "start": frame.index[0].isoformat(),
        "end": frame.index[-1].isoformat(),
        "mean_rate": float(rates.mean()),
        "mean_annualized": float(annualized.mean()),
        "median_annualized": float(annualized.median()),
        "positive_ratio": float((rates > 0).mean()),
        "max_annualized": float(annualized.max()),
        "min_annualized": float(annualized.min()),
        "std_annualized": float(annualized.std()) if periods > 1 else 0.0,
        "longest_negative_streak": int(consecutive_negative_streak(rates).max()),
        "longest_positive_streak": int(consecutive_positive_streak(rates).max()),
    }


__all__ = [
    "FALLBACK_INTERVAL_HOURS",
    "FUNDING_HISTORY_RETENTION_DAYS",
    "FundingIntervals",
    "annualize_rate",
    "consecutive_negative_streak",
    "consecutive_positive_streak",
    "deannualize_rate",
    "fetch_funding_history",
    "normalize_funding_to_8h",
    "fetch_funding_intervals",
    "summarize_funding",
    "trailing_annualized",
]
