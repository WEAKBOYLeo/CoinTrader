"""绩效指标 —— 回测的结论性数字。

本模块的设计原则：**只输出能被证伪的数字。**

具体体现在三点：

1. **成本必须显式摊开。** 报告"毛收益"和"净收益"两个数字，
   以及"费用占毛收益比例"。这个比例超过 50% 时策略在给交易所打工。

2. **样本内外必须分开。** 时间序列不能随机分割（会泄漏未来信息）。
   本模块用前推式分割，并且**样本外只允许看一次**。

3. **必须有安慰剂对照。** ``placebo_test`` 把收益序列随机打乱后重跑。
   如果打乱后依然"赚钱"，说明策略的收益来自结构泄漏或恒定漂移，
   而不是择时能力 —— 这正是回测自欺欺人的典型形态。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..errors import InsufficientDataError

#: 年化时使用的交易日/日历日基准
DEFAULT_DAYS_PER_YEAR = 365


@dataclass(slots=True)
class PerformanceMetrics:
    """一组绩效指标。所有 *_pct 字段均为小数（0.01 = 1%）。"""

    periods: int
    total_periods_available: int

    gross_return: float          # 未扣费的总收益（资金费累积）
    net_return: float            # 扣费后的总收益
    cost_paid: float             # 总成本（占初始资金比例）

    net_annualized: float        # 年化净收益
    gross_annualized: float      # 年化毛收益

    max_drawdown: float
    max_drawdown_duration: int   # 最大回撤持续期数

    win_rate: float              # 盈利期数占比
    profit_factor: float         # 总盈利 / 总亏损
    sharpe: float                # 年化夏普（无风险利率按 0）

    longest_negative_streak: int # 最长连续负收益期数
    worst_period: float          # 最差单期收益
    best_period: float           # 最好单期收益

    n_trades: int = 0            # 完成的往返交易次数
    n_round_trips: int = 0

    #: 平均仓位权重（已投入名义额 / 总资金）。用于区分
    #: "策略不赚钱" 与 "策略赚钱但资金没用满" —— 这是两个完全不同的问题。
    avg_exposure: float = 0.0
    #: 处于持仓状态的期数占比
    time_in_market: float = 0.0

    @property
    def return_on_deployed(self) -> float:
        """**已投入资金**口径的年化收益。

        为什么需要这个指标：策略按总资金算年化 5% 时，可能是
        （a）策略本身只有 5% 的收益率，也可能是
        （b）策略收益率很高但只用了 20% 的资金。

        这两种情况的应对完全不同：(a) 应放弃，(b) 应提高仓位或增加币种。

        若 ``avg_exposure`` 为 0（从未持仓），返回 0。
        """
        if self.avg_exposure <= 0:
            return 0.0
        return self.net_annualized / self.avg_exposure

    @property
    def cost_to_gross_ratio(self) -> float:
        """费用占毛收益的比例。

        超过 0.5 意味着超过一半的收益交给了交易所。
        毛收益 <= 0 时返回 inf（没有收益可分，比例无意义）。
        """
        if self.gross_return <= 0:
            return float("inf")
        return self.cost_paid / self.gross_return

    @property
    def is_profitable(self) -> bool:
        return self.net_return > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "periods": self.periods,
            "total_periods_available": self.total_periods_available,
            "gross_return": round(self.gross_return, 6),
            "net_return": round(self.net_return, 6),
            "cost_paid": round(self.cost_paid, 6),
            "cost_to_gross_ratio": (
                round(self.cost_to_gross_ratio, 4)
                if math.isfinite(self.cost_to_gross_ratio)
                else None
            ),
            "net_annualized": round(self.net_annualized, 6),
            "gross_annualized": round(self.gross_annualized, 6),
            "max_drawdown": round(self.max_drawdown, 6),
            "max_drawdown_duration": self.max_drawdown_duration,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": (
                round(self.profit_factor, 4) if math.isfinite(self.profit_factor) else None
            ),
            "sharpe": round(self.sharpe, 4),
            "longest_negative_streak": self.longest_negative_streak,
            "worst_period": round(self.worst_period, 6),
            "best_period": round(self.best_period, 6),
            "n_trades": self.n_trades,
            "n_round_trips": self.n_round_trips,
            "avg_exposure": round(self.avg_exposure, 4),
            "time_in_market": round(self.time_in_market, 4),
            "return_on_deployed": round(self.return_on_deployed, 6),
            "is_profitable": self.is_profitable,
        }


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """最大回撤及其持续期数。

    Args:
        equity: 净值曲线（累计值，非收益率）。

    Returns:
        ``(最大回撤比例, 持续期数)``。最大回撤为正数（0.15 表示跌了 15%）。
    """
    if len(equity) < 2:
        return 0.0, 0

    running_max = equity.cummax()
    drawdown = (running_max - equity) / running_max.replace(0, np.nan)
    drawdown = drawdown.fillna(0.0)

    peak_dd = float(drawdown.max())
    if peak_dd <= 0:
        return 0.0, 0

    # 回撤持续期：从最高点到最后一次恢复到该最高点的期数
    longest = 0
    current = 0
    for value in drawdown.to_numpy():
        if value > 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return peak_dd, int(longest)


def sharpe_ratio(returns: pd.Series, days_per_year: int = DEFAULT_DAYS_PER_YEAR) -> float:
    """年化夏普比率（无风险利率按 0）。

    对资金费套利这类低波动策略，夏普通常很高，但**不要据此判断策略好坏**：
    高夏普可能只是因为收益序列被人为平滑（比如按结算周期而非按市值计价）。
    """
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std == 0 or not math.isfinite(std):
        return 0.0
    return float(returns.mean()) / std * math.sqrt(days_per_year)


def profit_factor(returns: pd.Series) -> float:
    """盈亏比 = 总盈利 / 总亏损。无亏损时返回 inf。"""
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def longest_streak(returns: pd.Series, *, negative: bool = True) -> int:
    """最长连续同号期数。"""
    if returns.empty:
        return 0
    mask = (returns < 0) if negative else (returns > 0)
    longest = 0
    current = 0
    for flag in mask.to_numpy():
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def compute_metrics(
    period_returns: pd.Series,
    *,
    gross_returns: pd.Series | None = None,
    cost_returns: pd.Series | None = None,
    periods_per_year: float,
    n_round_trips: int = 0,
    total_periods_available: int | None = None,
    exposure: pd.Series | None = None,
) -> PerformanceMetrics:
    """由逐期收益序列计算全部绩效指标。

    Args:
        period_returns: 逐期**净**收益（已扣费）。
        gross_returns: 逐期毛收益（未扣费）。为 None 时视作等于净收益。
        cost_returns: 逐期成本。为 None 时视作零。
        periods_per_year: 每年结算期数（由结算周期换算）。
        n_round_trips: 完成的往返交易次数。
        total_periods_available: 原始可用期数（用于报告覆盖率）。
        exposure: 逐期仓位权重（已投入名义额/总资金）。用于计算
            ``return_on_deployed``。为 None 时按 0 处理。

    Returns:
        PerformanceMetrics。

    Raises:
        InsufficientDataError: 收益序列为空或不足两期。
    """
    if period_returns.empty:
        raise InsufficientDataError("收益序列为空，无法计算绩效")
    if len(period_returns) < 2:
        raise InsufficientDataError(f"收益序列仅 {len(period_returns)} 期，不足以计算绩效")

    net = period_returns.astype(float)
    gross = (gross_returns if gross_returns is not None else net).astype(float)
    costs = (cost_returns if cost_returns is not None else pd.Series(0.0, index=net.index)).astype(float)

    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year 必须为正，当前 {periods_per_year}")

    # 仓位权重统计
    if exposure is not None and not exposure.empty:
        exposure_series = exposure.astype(float)
        avg_exposure = float(exposure_series.mean())
        time_in_market = float((exposure_series > 0).mean())
    else:
        avg_exposure = 0.0
        time_in_market = 0.0

    # 复利累计。用 cumprod 而非 sum —— 对逐期比例收益，复利才是真实净值。
    net_equity = (1.0 + net).cumprod()
    total_net_return = float(net_equity.iloc[-1] - 1.0)

    # 毛收益同样按复利累计，但成本单独累加（成本是线性支出，不复利）
    total_gross_return = float((1.0 + gross).prod() - 1.0)
    total_cost = float(costs.sum())

    n_periods = len(net)
    years = n_periods / periods_per_year
    if years > 0 and (1.0 + total_net_return) > 0:
        annualized_net = (1.0 + total_net_return) ** (1.0 / years) - 1.0
    else:
        annualized_net = float("nan")

    if years > 0 and (1.0 + total_gross_return) > 0:
        annualized_gross = (1.0 + total_gross_return) ** (1.0 / years) - 1.0
    else:
        annualized_gross = float("nan")

    peak_dd, dd_duration = max_drawdown(net_equity)

    return PerformanceMetrics(
        periods=n_periods,
        total_periods_available=total_periods_available or n_periods,
        gross_return=total_gross_return,
        net_return=total_net_return,
        cost_paid=total_cost,
        net_annualized=annualized_net,
        gross_annualized=annualized_gross,
        max_drawdown=peak_dd,
        max_drawdown_duration=dd_duration,
        win_rate=float((net > 0).mean()),
        profit_factor=profit_factor(net),
        sharpe=sharpe_ratio(net),
        longest_negative_streak=longest_streak(net, negative=True),
        worst_period=float(net.min()),
        best_period=float(net.max()),
        n_round_trips=n_round_trips,
        avg_exposure=avg_exposure,
        time_in_market=time_in_market,
    )


# ---------------------------------------------------------------------------
# 前推式分割
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SplitResult:
    """样本内/外分割结果。"""

    in_sample: pd.Series
    out_of_sample: pd.Series
    split_index: int

    @property
    def in_sample_end(self) -> Any:
        return self.in_sample.index[-1] if len(self.in_sample) else None

    @property
    def out_of_sample_start(self) -> Any:
        return self.out_of_sample.index[0] if len(self.out_of_sample) else None

    def describe(self) -> dict[str, Any]:
        return {
            "in_sample_periods": len(self.in_sample),
            "out_of_sample_periods": len(self.out_of_sample),
            "in_sample_end": str(self.in_sample_end),
            "out_of_sample_start": str(self.out_of_sample_start),
        }


def forward_split(series: pd.Series, *, in_sample_ratio: float = 0.70) -> SplitResult:
    """按时间**前推式**分割（不随机打乱）。

    为什么不能随机分割：时间序列的相邻点高度相关。随机打乱会让
    "明天的数据"出现在训练集里，模型/参数就学到了未来 —— 样本外指标
    会好看得离谱，实盘则一塌糊涂。

    Args:
        series: 按时间升序的序列。
        in_sample_ratio: 样本内占比。

    Returns:
        SplitResult。

    Raises:
        InsufficientDataError: 任一侧为空。
    """
    if not 0.0 < in_sample_ratio < 1.0:
        raise ValueError(f"in_sample_ratio 必须在 (0,1) 内，当前 {in_sample_ratio}")

    split_index = int(len(series) * in_sample_ratio)
    if split_index <= 0 or split_index >= len(series):
        raise InsufficientDataError(
            f"序列长度 {len(series)} 无法按 {in_sample_ratio} 分割出非空的两部分"
        )

    return SplitResult(
        in_sample=series.iloc[:split_index],
        out_of_sample=series.iloc[split_index:],
        split_index=split_index,
    )


# ---------------------------------------------------------------------------
# 安慰剂对照
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PlaceboResult:
    """安慰剂检验结果。"""

    real_net_return: float
    shuffled_mean_return: float
    shuffled_std_return: float
    shuffled_p95_return: float
    n_trials: int
    p_value: float

    @property
    def real_beats_placebo(self) -> bool:
        """真实策略是否显著优于随机排列（p < 0.05）。"""
        return self.p_value < 0.05

    def describe(self) -> dict[str, Any]:
        return {
            "real_net_return": round(self.real_net_return, 6),
            "shuffled_mean_return": round(self.shuffled_mean_return, 6),
            "shuffled_std_return": round(self.shuffled_std_return, 6),
            "shuffled_p95_return": round(self.shuffled_p95_return, 6),
            "n_trials": self.n_trials,
            "p_value": round(self.p_value, 4),
            "real_beats_placebo": self.real_beats_placebo,
        }


def placebo_test(
    returns: pd.Series,
    *,
    n_trials: int = 1000,
    seed: int = 20260916,
) -> PlaceboResult:
    """安慰剂对照：打乱收益顺序后重跑，看真实策略是否更好。

    **这个测试抓的是什么：**

    如果一个策略只是"长期持有 + 稳定收租"，那么打乱顺序不改变总收益，
    检验会显示 p 值很大 —— 说明收益不来自**择时**，而来自**持仓**。
    这对资金费套利是**正常且预期**的结果，不一定是坏事。

    真正需要警惕的是两种情况：

    1. 打乱后收益**变高** → 真实策略的择时是负贡献，在破坏收益
    2. 收益集中在极少数几期 → 去掉那几期就不赚钱，说明是运气不是策略

    Args:
        returns: 逐期收益序列。
        n_trials: 打乱次数。
        seed: 随机种子（固定以保证可复现）。

    Returns:
        PlaceboResult。
    """
    if returns.empty:
        raise InsufficientDataError("收益序列为空，无法做安慰剂检验")

    values = returns.to_numpy(dtype=float)
    real_total = float(np.prod(1.0 + values) - 1.0)

    rng = np.random.default_rng(seed)
    shuffled_totals = np.empty(n_trials, dtype=float)

    for trial in range(n_trials):
        shuffled = rng.permutation(values)
        shuffled_totals[trial] = float(np.prod(1.0 + shuffled) - 1.0)

    # 单侧 p 值：随机排列中收益 >= 真实值的比例
    p_value = float((shuffled_totals >= real_total).mean())

    return PlaceboResult(
        real_net_return=real_total,
        shuffled_mean_return=float(shuffled_totals.mean()),
        shuffled_std_return=float(shuffled_totals.std(ddof=1)) if n_trials > 1 else 0.0,
        shuffled_p95_return=float(np.percentile(shuffled_totals, 95)),
        n_trials=n_trials,
        p_value=p_value,
    )


def concentration_check(returns: pd.Series, *, top_n: int = 5) -> dict[str, Any]:
    """收益集中度检查 —— 去掉最好的几期后还剩多少。

    这是抓"运气策略"最直接的方法。如果一个策略的全部收益来自
    5 个结算期，那它不是策略，是抽奖。

    Returns:
        含 ``full_return`` / ``return_without_top`` / ``top_contribution``
        / ``is_concentrated`` 的字典。
    """
    if returns.empty:
        raise InsufficientDataError("收益序列为空")

    values = returns.astype(float).sort_values(ascending=False)
    full = float((1.0 + returns.astype(float)).prod() - 1.0)

    dropped = returns.astype(float).drop(values.index[:top_n])
    without_top = float((1.0 + dropped).prod() - 1.0) if len(dropped) else 0.0

    # "贡献"定义为毛收益中被最好的 top_n 期占掉的比例
    contribution = (full - without_top) / full if full > 0 else float("nan")

    return {
        "full_return": full,
        "return_without_top": without_top,
        "top_n": top_n,
        "top_contribution": contribution,
        # 去掉最好的 5 期后收益腰斩 → 判定为高度集中
        "is_concentrated": bool(full > 0 and without_top < full * 0.5),
    }


__all__ = [
    "DEFAULT_DAYS_PER_YEAR",
    "PerformanceMetrics",
    "PlaceboResult",
    "SplitResult",
    "compute_metrics",
    "concentration_check",
    "forward_split",
    "longest_streak",
    "max_drawdown",
    "placebo_test",
    "profit_factor",
    "sharpe_ratio",
]
