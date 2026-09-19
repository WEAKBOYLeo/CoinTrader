"""回测引擎 —— 资金费套利的最小化、无前瞻模拟。

## 模型说明（先读这段再看代码）

**头寸结构**：现货多头 + 永续空头，名义额相等，做 delta 中性。

**逐期收益组成**（每期 = 一次资金费结算）::

    period_return[i] = funding_received[i]
                       - costs_executed[i]
                       + basis_adjustment[i]

其中：

- ``funding_rate`` 按**多头付空头**约定。现货多头 + 永续空头 ⇒
  费率 > 0 时**收取**，因此组合的多头直接等于 ``funding_rate``。
- 信号只看固定滚动窗口内、截至当前时点的数据；订单延迟到未来 bar
  执行，执行前信号失效则撤单。
- 手续费和滑点在实际进/出场成交时点入账，不把成本事后摊到未知的持有期。
- ``basis_adjustment`` 在实际建仓和平仓成交时点发生，见下。

**基差处理**：默认 ``zero_basis``（保守）。假定调仓时白送一次
``max_basis_adverse`` 的损失。理由见 docs/ARCHITECTURE.md §2.3：
持有期间现货与永续报价会漂移，这段损益是真实风险，
而它恰好在牛市中**对空永续方不利**（永续溢价扩大）。悲观假设比乐观假设诚实。

## 无前瞻（lookahead）保证

这是整个回测里最容易骗自己的地方，本引擎用**结构性**手段而非纪律来保证：

1. **信号与执行分离。** 第 ``i`` 期的信号只读取固定窗口内的
   ``rates[max(0, i-window+1):i+1]``（含当期，因为当期资金费在结算时点已知），
   成交价使用第 ``i + lag`` 期，``lag >= 1`` 由配置强制。
2. **订单状态真实推进。** 未成交建仓不会收资金费；若执行前信号转差则撤销。
   平仓订单执行前仍保留仓位，数据结束时才按最后可见时点强制平仓。
3. **组合动态选币。** 多币事件按时间推进，只在当前事件可见的候选中排名；
   已持仓和待成交订单都占用 ``max_positions``。
4. **结构性证伪测试。** ``tests/test_no_lookahead.py`` 会截断/篡改未来数据，
   并验证早期信号与早期账本不变。

**前视陷阱的反例**（本引擎刻意不这么写）::

    # ❌ 错误：用当期的 close 决定当期成交
    signal = rates[i] > threshold
    pnl = signal * returns[i]

    # ✅ 正确：信号在 i 产生，成交在 i+lag
    signal = rates[:i+1] > threshold   # 只看历史
    pnl = signal * returns[i + lag]    # 未来才成交
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import BacktestConfig, StrategyConfig
from ..data.funding import (
    annualize_rate,
    consecutive_negative_streak,
    consecutive_positive_streak,
    normalize_funding_to_8h,
)
from ..errors import InsufficientDataError
from ..research.costs import CostModel, LiquidityTier, classify_liquidity
from ..research.metrics import PerformanceMetrics, compute_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TradePeriodRecord:
    """一笔交易在单个 8h 周期的可对账明细。"""

    trade_id: int
    symbol: str
    period_index: int
    period_time: Any
    funding_rate_8h: float
    notional_usdt: float
    funding_pnl_usdt: float
    entry_cost_usdt: float
    exit_cost_usdt: float
    basis_pnl_usdt: float
    net_pnl_usdt: float
    cumulative_net_pnl_usdt: float
    cumulative_return_on_notional: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "period_index": self.period_index,
            "period_time": str(self.period_time),
            "funding_rate_8h": round(self.funding_rate_8h, 8),
            "notional_usdt": round(self.notional_usdt, 2),
            "funding_pnl_usdt": round(self.funding_pnl_usdt, 6),
            "entry_cost_usdt": round(self.entry_cost_usdt, 6),
            "exit_cost_usdt": round(self.exit_cost_usdt, 6),
            "basis_pnl_usdt": round(self.basis_pnl_usdt, 6),
            "net_pnl_usdt": round(self.net_pnl_usdt, 6),
            "cumulative_net_pnl_usdt": round(self.cumulative_net_pnl_usdt, 6),
            "cumulative_return_on_notional": round(self.cumulative_return_on_notional, 8),
        }


@dataclass(slots=True)
class TradeRecord:
    """一笔完整的往返交易。"""

    trade_id: int
    symbol: str
    entry_index: int
    exit_index: int
    entry_time: Any
    exit_time: Any
    holding_periods: int
    notional: float

    gross_funding: float      # 累计收到的资金费（占名义额比例）
    entry_cost: float
    exit_cost: float
    basis_adjustment: float
    net_return: float         # 占**投入资金**的比例
    exit_reason: str
    periods: list[TradePeriodRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "entry_time": str(self.entry_time),
            "exit_time": str(self.exit_time),
            "holding_periods": self.holding_periods,
            "notional": round(self.notional, 2),
            "gross_funding": round(self.gross_funding, 6),
            "entry_cost": round(self.entry_cost, 6),
            "exit_cost": round(self.exit_cost, 6),
            "basis_adjustment": round(self.basis_adjustment, 6),
            "net_return": round(self.net_return, 6),
            "exit_reason": self.exit_reason,
            "periods": [period.as_dict() for period in self.periods],
        }


@dataclass(slots=True)
class BacktestResult:
    """单个币种的回测结果。"""

    symbol: str
    interval_hours: int
    metrics: PerformanceMetrics
    period_returns: pd.Series
    gross_returns: pd.Series
    cost_returns: pd.Series
    equity_curve: pd.Series
    #: 逐期仓位权重（已投入名义额 / 总资金）。组合回测需要它来正确聚合
    #: 「资金使用率」——不保存这个序列会导致 run_portfolio 只能把 exposure
    #: 当 0 处理，进而让 avg_exposure / return_on_deployed 恒为 0，
    #: 即使 net_annualized 本身是对的，这两个字段也会静默失真。
    exposure: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: list[TradeRecord] = field(default_factory=list)
    #: 每笔交易逐 8h 的资金费、成本和累计收益。
    period_ledger: list[TradePeriodRecord] = field(default_factory=list)
    entry_signal_count: int = 0
    rolling_window_periods: int = 0
    exit_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def periods_per_year(self) -> float:
        return (24.0 / self.interval_hours) * 365.0

    def summary(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval_hours": self.interval_hours,
            **self.metrics.as_dict(),
            "n_trades": len(self.trades),
            "n_traded_periods": len(self.period_ledger),
            "trades": [trade.as_dict() for trade in self.trades],
            "period_ledger": [period.as_dict() for period in self.period_ledger],
            "entry_signal_count": self.entry_signal_count,
            "rolling_window_periods": self.rolling_window_periods,
            "exit_reasons": dict(self.exit_reasons),
        }


def _replacement_premium(holding_periods: int, strategy: StrategyConfig) -> float:
    """返回该持仓年龄要求的替换相对优势。"""
    exit_cfg = strategy.exit
    if holding_periods <= 60:
        return float("inf")
    if holding_periods <= 120:
        return exit_cfg.replacement_premium_under_120
    return exit_cfg.replacement_premium_over_120





@dataclass(frozen=True, slots=True)
class Signals:
    """预计算的信号序列。每个元素只用截至该位置的数据。"""

    entry: np.ndarray        # bool: 是否应建仓
    exit: np.ndarray         # bool: 是否应平仓
    trailing_ann: np.ndarray # float: 入场/排名用滚动年化
    exit_trailing_ann: np.ndarray # float: 出场用 30 期滚动年化
    pos_streak: np.ndarray   # int: 连续正费率期数
    neg_streak: np.ndarray   # int: 连续负费率期数

    def __len__(self) -> int:
        return len(self.entry)


def build_signals(
    rates: pd.Series,
    interval_hours: int,
    strategy: StrategyConfig,
    *,
    history_window: int | None = None,
) -> Signals:
    """由单期费率序列构建进出场信号。

    所有滚动统计的窗口右端都是当前位置，**不包含未来**。
    ``history_window`` 非空时，信号只允许使用最近这段历史；窗口未满前
    不产生信号，连续正/负费率计数也不会继承窗口外的数据。

    Args:
        rates: 单期资金费率，按时间升序。
        interval_hours: 结算周期。
        strategy: 策略参数。
        history_window: 固定滚动历史窗口。为 None 时保留完整的因果历史，
            主要用于独立的信号诊断；正式回测由配置传入固定窗口。

    Returns:
        Signals。

    Note:
        信号在位置 ``i`` 的含义是「基于截至第 i 期结算的信息，
        决定在第 i+lag 期建仓」。引擎负责施加 lag。
    """
    n = len(rates)
    if history_window is not None:
        if history_window <= 0:
            raise ValueError(f"history_window 必须为正，当前 {history_window}")
        if history_window < strategy.entry.lookback_periods:
            raise ValueError(
                "history_window 不能小于 entry.lookback_periods: "
                f"{history_window} < {strategy.entry.lookback_periods}"
            )

    if n == 0:
        empty_b = np.zeros(0, dtype=bool)
        return Signals(
            entry=empty_b,
            exit=empty_b,
            trailing_ann=np.zeros(0),
            exit_trailing_ann=np.zeros(0),
            pos_streak=np.zeros(0, dtype=int),
            neg_streak=np.zeros(0, dtype=int),
        )

    entry_cfg = strategy.entry

    # 所有策略判断先使用最近 N 期滑动平均费率；生产配置 N=10。
    # 所有策略判断先使用最近 N 期滑动平均费率；生产配置 N=10。
    smoothed_rates = rates.rolling(
        window=entry_cfg.lookback_periods,
        min_periods=entry_cfg.lookback_periods,
    ).mean()
    trailing = annualize_rate(smoothed_rates, interval_hours)
    pos_streak = consecutive_positive_streak(smoothed_rates).to_numpy(dtype=int)
    neg_streak = consecutive_negative_streak(smoothed_rates).to_numpy(dtype=int)

    # 固定滚动窗口不能使用窗口外的连续正/负费率；窗口未满时不决策。
    if history_window is not None:
        available = np.minimum(np.arange(n, dtype=int) + 1, history_window)
        pos_streak = np.minimum(pos_streak, available)
        neg_streak = np.minimum(neg_streak, available)
        history_ready = (np.arange(n, dtype=int) + 1) >= history_window
    else:
        history_ready = np.ones(n, dtype=bool)

    # 当前期判断也使用同一条最近 N 期均值，避免 entry 的两个阈值口径不一致。
    current_ann = annualize_rate(smoothed_rates, interval_hours)

    # 进场：滚动年化达标 + 连续为正足够久
    entry = (
        history_ready
        & trailing.notna().to_numpy()
        & (trailing.to_numpy() >= entry_cfg.min_trailing_annualized)
        & (current_ann.to_numpy() >= entry_cfg.min_annualized_rate)
        & (pos_streak >= entry_cfg.min_consecutive_positive)
    )

    exit_smoothed_rates = rates.rolling(
        window=strategy.exit.exit_lookback_periods,
        min_periods=strategy.exit.exit_lookback_periods,
    ).mean()
    exit_trailing = annualize_rate(exit_smoothed_rates, interval_hours)
    exit_ready = history_ready & exit_trailing.notna().to_numpy()

    # 资金费 30 期均值转负，下一可执行期退出。年龄只影响替换，不影响该止损。
    exit_signal = exit_ready & (exit_trailing.to_numpy() < 0.0)

    return Signals(
        entry=entry,
        exit=exit_signal,
        trailing_ann=trailing.to_numpy(dtype=float),
        exit_trailing_ann=exit_trailing.to_numpy(dtype=float),
        pos_streak=pos_streak,
        neg_streak=neg_streak,
    )


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------


class FundingCarryBacktester:
    """资金费套利回测引擎。

    Args:
        backtest: 回测配置（``execution_lag_bars`` 强制 >= 1）。
        strategy: 策略配置。
        cost_model: 成本模型。
        tier: 该币种的流动性档位（决定滑点）。
    """

    def __init__(
        self,
        backtest: BacktestConfig,
        strategy: StrategyConfig,
        cost_model: CostModel,
        *,
        tier: LiquidityTier | str = LiquidityTier.MAJOR,
        basis_adverse_pct: float = 0.0,
    ) -> None:
        if backtest.execution_lag_bars < 1:
            # 配置层已经卡过一道，这里是第二道 —— 双保险
            raise ValueError(
                f"execution_lag_bars 必须 >= 1（当前 {backtest.execution_lag_bars}），"
                "否则会引入前瞻偏差"
            )
        self.backtest = backtest
        self.strategy = strategy
        if backtest.rolling_window_periods < strategy.entry.lookback_periods:
            raise ValueError(
                "rolling_window_periods 不能小于 entry.lookback_periods: "
                f"{backtest.rolling_window_periods} < {strategy.entry.lookback_periods}"
            )
        self.cost_model = cost_model
        self.tier = LiquidityTier(tier)
        self.basis_adverse_pct = basis_adverse_pct

    # -- 主流程 -------------------------------------------------------------

    def run(
        self,
        symbol: str,
        rates: pd.Series,
        interval_hours: int,
        *,
        entry_allowed: pd.Series | np.ndarray | None = None,
        exit_allowed: pd.Series | np.ndarray | None = None,
        quote_volume_24h: pd.Series | None = None,
    ) -> BacktestResult:
        """对单个币种运行固定窗口、事件驱动回测。

        每个时点 ``i`` 只读取截至 ``i`` 的资金费率。信号在 ``i`` 产生，
        订单最早在 ``i + execution_lag_bars`` 执行；在成交前信号失效时，
        未成交的建仓会被撤销。交易成本记在实际成交时点，不再事后摊销。

        ``entry_allowed`` 给组合回测使用：它是一个只允许在对应信号时点
        建仓的布尔掩码，通常由按时间推进的动态选币器生成。

        Args:
            symbol: 合约符号。
            rates: 单期资金费率序列（升序）。索引为结算时间。
            interval_hours: 结算周期小时数。
            entry_allowed: 可选的外部建仓许可掩码，长度必须与 ``rates`` 相同。

        Returns:
            BacktestResult。

        Raises:
            InsufficientDataError: 数据不足以完成固定窗口预热。
        """
        if interval_hours <= 0:
            raise ValueError(f"结算周期必须为正，当前 {interval_hours}")

        rates = normalize_funding_to_8h(rates.astype(float), interval_hours)
        interval_hours = 8
        if not rates.index.is_monotonic_increasing:
            raise ValueError("资金费率序列必须按时间升序")

        rolling_window = self.backtest.rolling_window_periods
        min_history = max(self.strategy.entry.lookback_periods, rolling_window)
        min_periods = min_history + self.backtest.execution_lag_bars + 1
        if len(rates) < min_periods:
            raise InsufficientDataError(
                f"{symbol} 仅 {len(rates)} 期数据，至少需要 {min_periods} 期"
                f"（滚动窗口 {rolling_window} + lag "
                f"{self.backtest.execution_lag_bars} + 1）"
            )

        rates_arr = rates.to_numpy()
        n = len(rates_arr)

        if entry_allowed is None:
            entry_allowed_arr = np.ones(n, dtype=bool)
        elif isinstance(entry_allowed, pd.Series):
            if len(entry_allowed) != n:
                raise ValueError("entry_allowed 长度必须与 rates 相同")
            entry_allowed_arr = entry_allowed.to_numpy(dtype=bool)
        else:
            entry_allowed_arr = np.asarray(entry_allowed, dtype=bool)
            if len(entry_allowed_arr) != n:
                raise ValueError("entry_allowed 长度必须与 rates 相同")

        if exit_allowed is None:
            exit_allowed_arr: np.ndarray | None = None
        elif isinstance(exit_allowed, pd.Series):
            if len(exit_allowed) != n:
                raise ValueError("exit_allowed 长度必须与 rates 相同")
            exit_allowed_arr = exit_allowed.to_numpy(dtype=bool)
        else:
            exit_allowed_arr = np.asarray(exit_allowed, dtype=bool)
            if len(exit_allowed_arr) != n:
                raise ValueError("exit_allowed 长度必须与 rates 相同")

        # 历史成交量只允许使用事件时点以前已经闭合的 4h K 线结果。
        if quote_volume_24h is None:
            tier_by_index = np.full(n, self.tier, dtype=object)
        else:
            volume = quote_volume_24h.astype(float).sort_index()
            if not volume.index.is_monotonic_increasing:
                raise ValueError("历史成交量序列必须按时间升序")
            aligned_volume = volume.reindex(rates.index, method="ffill")
            tier_by_index = np.array(
                [
                    classify_liquidity(float(value)) if pd.notna(value) else self.tier
                    for value in aligned_volume.to_numpy()
                ],
                dtype=object,
            )
        signals = build_signals(
            rates,
            interval_hours,
            self.strategy,
            history_window=rolling_window,
        )

        # 回测配置覆盖按流动性档位的默认滑点；当前基准为每腿 0.15%。
        per_position_weight = self.strategy.selection.per_position_weight
        periods_per_year = (24.0 / interval_hours) * 365.0

        net_returns = np.zeros(n)
        gross_returns = np.zeros(n)
        cost_returns = np.zeros(n)
        # 逐期仓位权重（已投入名义额 / 总资金）。用于区分
        # "策略不赚钱" 与 "策略赚钱但资金没用满"。
        exposure = np.zeros(n)

        trades: list[TradeRecord] = []
        period_ledger: list[TradePeriodRecord] = []
        exit_reasons: dict[str, int] = {}

        # 状态机：信号时点和成交时点严格分离。
        lag = self.backtest.execution_lag_bars
        in_position = False
        entry_signal_index: int | None = None
        entry_exec_index: int | None = None
        pending_entry_signal_index: int | None = None
        pending_entry_exec_index: int | None = None
        pending_exit_signal_index: int | None = None
        pending_exit_exec_index: int | None = None
        pending_exit_reason: str | None = None
        entry_tier: LiquidityTier = self.tier

        #: 当前持仓实际收到资金费的期数与总额。只在事件循环中累积，
        #: 不在平仓时根据索引事后推算。
        accrued_periods = 0
        accrued_funding = 0.0

        for i in range(n):
            # ---- 执行已排队的平仓/建仓 ----
            # 平仓先于建仓，避免同一时点同时占用旧仓和新仓。
            if pending_exit_exec_index == i:
                if not in_position or entry_signal_index is None or entry_exec_index is None:
                    raise RuntimeError("回测状态异常：平仓订单没有对应持仓")
                self._close_trade(
                    symbol=symbol,
                    rates=rates,
                    entry_signal_index=entry_signal_index,
                    entry_exec_index=entry_exec_index,
                    exit_signal_index=pending_exit_signal_index if pending_exit_signal_index is not None else i,
                    exit_exec_index=i,
                    weight=per_position_weight,
                    accrued_periods=accrued_periods,
                    accrued_funding=accrued_funding,
                    entry_cost_rate=self.cost_model.entry_cost(
                        entry_tier, slippage_per_leg=self.backtest.slippage_per_leg
                    ),
                    exit_cost_rate=self.cost_model.exit_cost(
                        tier_by_index[i], slippage_per_leg=self.backtest.slippage_per_leg
                    ),
                    net_returns=net_returns,
                    gross_returns=gross_returns,
                    cost_returns=cost_returns,
                    trades=trades,
                    period_ledger=period_ledger,
                    exit_reason=pending_exit_reason or "signal",
                )
                closed_reason = pending_exit_reason or "signal"
                exit_reasons[closed_reason] = exit_reasons.get(closed_reason, 0) + 1
                in_position = False
                entry_signal_index = None
                entry_exec_index = None
                pending_exit_signal_index = None
                pending_exit_exec_index = None
                pending_exit_reason = None
                accrued_periods = 0
                accrued_funding = 0.0

            if pending_entry_exec_index == i:
                if pending_entry_signal_index is None:
                    raise RuntimeError("回测状态异常：建仓订单缺少信号索引")
                entry_tier = tier_by_index[i]
                in_position = True
                entry_signal_index = pending_entry_signal_index
                entry_exec_index = i
                pending_entry_signal_index = None
                pending_entry_exec_index = None
                accrued_periods = 0
                accrued_funding = 0.0

            # ---- 结算：持仓且已过建仓成交时点，本期收取资金费 ----
            if in_position:
                exposure[i] = per_position_weight
                if entry_exec_index is not None and i > entry_exec_index:
                    # 现货多头 + 永续空头在费率为正时收取资金费。
                    funding = rates_arr[i] * per_position_weight
                    net_returns[i] += funding
                    gross_returns[i] += funding
                    accrued_periods += 1
                    accrued_funding += rates_arr[i]   # 未乘权重

            # ---- 决策：只基于当前位置可见的滚动信号 ----
            can_execute = (i + lag) < n

            # 订单执行前必须仍满足进场条件；转差到出场条件则同样撤单。
            if pending_entry_exec_index is not None:
                if (
                    (not signals.entry[i] or signals.exit[i])
                    and i < pending_entry_exec_index
                ):
                    pending_entry_signal_index = None
                    pending_entry_exec_index = None
                continue

            if not in_position:
                if (
                    signals.entry[i]
                    and entry_allowed_arr[i]
                    and not signals.exit[i]
                    and can_execute
                ):
                    pending_entry_signal_index = i
                    pending_entry_exec_index = i + lag
                continue

            if entry_exec_index is None:
                raise RuntimeError("回测状态异常：持仓缺少建仓成交索引")
            holding = i - entry_exec_index
            should_exit = (
                pending_exit_exec_index is None
                and can_execute
                and (
                    signals.exit[i]
                    or (
                        exit_allowed_arr[i]
                        if exit_allowed_arr is not None
                        else False
                    )
                    or holding >= self.strategy.exit.max_holding_periods
                )
            )
            if should_exit:
                replacement_exit = bool(
                    exit_allowed_arr[i] if exit_allowed_arr is not None else False
                )
                reason = (
                    "max_holding"
                    if holding >= self.strategy.exit.max_holding_periods
                    and not signals.exit[i]
                    and not replacement_exit
                    else "negative_smoothed_funding"
                    if signals.exit[i]
                    else "replacement_or_floor"
                )
                pending_exit_signal_index = i
                pending_exit_exec_index = i + lag
                pending_exit_reason = reason
                # reason 保留到实际执行；若数据在执行前结束，归为 end_of_data.

        # 未成交的入场订单没有交易记录，也没有成本。
        # 已成交但尚未完成平仓的仓位，在最后一个可见时点强制平仓，避免
        # 只保留盈利交易造成幸存者偏差。
        if in_position:
            if entry_signal_index is None or entry_exec_index is None:
                raise RuntimeError("回测状态异常：尾部持仓缺少建仓信息")
            final_index = n - 1
            self._close_trade(
                symbol=symbol,
                rates=rates,
                entry_signal_index=entry_signal_index,
                entry_exec_index=entry_exec_index,
                exit_signal_index=pending_exit_signal_index if pending_exit_signal_index is not None else final_index,
                exit_exec_index=final_index,
                weight=per_position_weight,
                accrued_periods=accrued_periods,
                accrued_funding=accrued_funding,
                entry_cost_rate=self.cost_model.entry_cost(
                    entry_tier, slippage_per_leg=self.backtest.slippage_per_leg
                ),
                exit_cost_rate=self.cost_model.exit_cost(
                    tier_by_index[final_index], slippage_per_leg=self.backtest.slippage_per_leg
                ),
                net_returns=net_returns,
                gross_returns=gross_returns,
                cost_returns=cost_returns,
                trades=trades,
                period_ledger=period_ledger,
                exit_reason="end_of_data",
            )
            exit_reasons["end_of_data"] = exit_reasons.get("end_of_data", 0) + 1

        net_series = pd.Series(net_returns, index=rates.index, name="net")
        gross_series = pd.Series(gross_returns, index=rates.index, name="gross")
        cost_series = pd.Series(cost_returns, index=rates.index, name="cost")
        exposure_series = pd.Series(exposure, index=rates.index, name="exposure")
        equity = (1.0 + net_series).cumprod()

        metrics = compute_metrics(
            net_series,
            gross_returns=gross_series,
            cost_returns=cost_series,
            periods_per_year=periods_per_year,
            n_round_trips=len(trades),
            total_periods_available=n,
            exposure=exposure_series,
        )

        return BacktestResult(
            symbol=symbol,
            interval_hours=interval_hours,
            metrics=metrics,
            period_returns=net_series,
            gross_returns=gross_series,
            cost_returns=cost_series,
            equity_curve=equity,
            exposure=exposure_series,
            trades=trades,
            period_ledger=period_ledger,
            entry_signal_count=int(np.count_nonzero(signals.entry & entry_allowed_arr)),
            rolling_window_periods=rolling_window,
            exit_reasons=exit_reasons,
        )

    # -- 平仓记账 -----------------------------------------------------------

    def _close_trade(
        self,
        *,
        symbol: str,
        rates: pd.Series,
        entry_signal_index: int,
        entry_exec_index: int,
        exit_signal_index: int,
        exit_exec_index: int,
        weight: float,
        accrued_periods: int,
        accrued_funding: float,
        entry_cost_rate: float,
        exit_cost_rate: float,
        net_returns: np.ndarray,
        gross_returns: np.ndarray,
        cost_returns: np.ndarray,
        trades: list[TradeRecord],
        period_ledger: list[TradePeriodRecord],
        exit_reason: str,
    ) -> None:
        """在实际平仓成交时点完成一笔交易记账。

        成本和基差损失记在真实的进/出场成交位置，不根据未来索引摊销。
        资金费仍由主循环逐期累积传入，保证交易记录与逐期账本一致。

        ``entry_signal_index`` / ``exit_signal_index`` 只用于保留事件语义；
        价格成交索引始终由 ``entry_exec_index`` / ``exit_exec_index`` 明确给出。
        """
        del entry_signal_index, exit_signal_index, gross_returns

        if not (0 <= entry_exec_index < len(rates)):
            raise ValueError(f"建仓成交索引越界: {entry_exec_index}")
        if not (0 <= exit_exec_index < len(rates)):
            raise ValueError(f"平仓成交索引越界: {exit_exec_index}")
        if exit_exec_index < entry_exec_index:
            raise ValueError("平仓成交不能早于建仓成交")

        # 实际收到的资金费（由主循环累积，非索引推算）。
        gross_funding = float(accrued_funding)

        # 成本占**投入资金**的比例。
        entry_cost = entry_cost_rate * weight
        exit_cost = exit_cost_rate * weight
        basis_leg = -abs(self.basis_adverse_pct) * weight
        basis_adjustment = basis_leg * 2.0

        total_cost = entry_cost + exit_cost
        net = gross_funding * weight - total_cost + basis_adjustment

        # 逐 8h ledger：成交成本位于真实成交期，资金费从建仓成交后的下一期开始。
        trade_id = len(trades) + 1
        notional_usdt = weight * self.backtest.initial_capital
        cumulative = 0.0
        periods: list[TradePeriodRecord] = []
        for period_index in range(entry_exec_index, exit_exec_index + 1):
            funding_rate = float(rates.iloc[period_index])
            funding_received = period_index > entry_exec_index and (
                exit_reason == "end_of_data" or period_index < exit_exec_index
            )
            funding_pnl = funding_rate * notional_usdt if funding_received else 0.0
            entry_pnl = -entry_cost * self.backtest.initial_capital if period_index == entry_exec_index else 0.0
            exit_pnl = -exit_cost * self.backtest.initial_capital if period_index == exit_exec_index else 0.0
            basis_pnl = basis_leg * self.backtest.initial_capital if period_index in (entry_exec_index, exit_exec_index) else 0.0
            period_net = funding_pnl + entry_pnl + exit_pnl + basis_pnl
            cumulative += period_net
            periods.append(
                TradePeriodRecord(
                    trade_id=trade_id,
                    symbol=symbol,
                    period_index=period_index,
                    period_time=rates.index[period_index],
                    funding_rate_8h=funding_rate,
                    notional_usdt=notional_usdt,
                    funding_pnl_usdt=funding_pnl,
                    entry_cost_usdt=-entry_pnl,
                    exit_cost_usdt=-exit_pnl,
                    basis_pnl_usdt=basis_pnl,
                    net_pnl_usdt=period_net,
                    cumulative_net_pnl_usdt=cumulative,
                    cumulative_return_on_notional=cumulative / notional_usdt if notional_usdt else 0.0,
                )
            )
        period_ledger.extend(periods)

        net_returns[entry_exec_index] -= entry_cost
        cost_returns[entry_exec_index] += entry_cost
        net_returns[exit_exec_index] -= exit_cost
        cost_returns[exit_exec_index] += exit_cost
        if basis_leg != 0.0:
            net_returns[entry_exec_index] += basis_leg
            net_returns[exit_exec_index] += basis_leg

        trades.append(
            TradeRecord(
                trade_id=trade_id,
                symbol=symbol,
                entry_index=entry_exec_index,
                exit_index=exit_exec_index,
                entry_time=rates.index[entry_exec_index],
                exit_time=rates.index[exit_exec_index],
                # 用实际收租期数而非信号索引差，保证交易归因与逐期账本一致。
                holding_periods=max(1, accrued_periods),
                notional=weight,
                gross_funding=gross_funding,
                entry_cost=entry_cost,
                exit_cost=exit_cost,
                basis_adjustment=basis_adjustment,
                net_return=net,
                exit_reason=exit_reason,
                periods=periods,
            )
        )



# ---------------------------------------------------------------------------
# 组合回测
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PortfolioResult:
    """多币种组合回测结果。"""

    per_symbol: dict[str, BacktestResult]
    portfolio_returns: pd.Series
    metrics: PerformanceMetrics
    weight_per_position: float
    portfolio_gross_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    portfolio_cost_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    portfolio_exposure: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    max_positions: int = 0
    dynamic_selection: bool = True

    def summary(self) -> dict[str, Any]:
        return {
            "n_symbols": len(self.per_symbol),
            "n_trade_periods": sum(len(res.period_ledger) for res in self.per_symbol.values()),
            "weight_per_position": self.weight_per_position,
            "max_positions": self.max_positions,
            "dynamic_selection": self.dynamic_selection,
            "portfolio": self.metrics.as_dict(),
            "per_symbol": {sym: res.summary() for sym, res in self.per_symbol.items()},
        }


@dataclass(slots=True)
class _SelectionState:
    """动态选币器的单币种状态。"""

    in_position: bool = False
    entry_exec_index: int | None = None
    pending_entry_exec_index: int | None = None
    pending_exit_exec_index: int | None = None


def _build_dynamic_masks(
    per_symbol_rates: dict[str, pd.Series],
    intervals: dict[str, int],
    backtest: BacktestConfig,
    strategy: StrategyConfig,
    symbols: list[str],
    max_positions: int,
    quote_volumes_24h: dict[str, pd.Series] | None = None,
    min_quote_volume_3d_avg: float = 0.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """按时间推进动态选币，并生成因果的进场/替换出场掩码。

    每个事件先执行已排队订单，再用当前可见的 10 期均值比较候选。
    持仓资金费均值转负立即退出；没有足够优势的替换候选时，才使用
    持仓年龄对应的 3%/10% 保留阈值。
    """
    signals_by_symbol: dict[str, Signals] = {}
    entry_masks = {
        symbol: np.zeros(len(per_symbol_rates[symbol]), dtype=bool) for symbol in symbols
    }
    exit_masks = {
        symbol: np.zeros(len(per_symbol_rates[symbol]), dtype=bool) for symbol in symbols
    }
    states = {symbol: _SelectionState() for symbol in symbols}
    events: dict[Any, list[tuple[str, int]]] = {}

    for symbol in symbols:
        rates = per_symbol_rates[symbol]
        signals_by_symbol[symbol] = build_signals(
            rates,
            intervals[symbol],
            strategy,
            history_window=backtest.rolling_window_periods,
        )
        for index, timestamp in enumerate(rates.index):
            events.setdefault(timestamp, []).append((symbol, index))

    lag = backtest.execution_lag_bars
    for timestamp in sorted(events):
        event_group = sorted(events[timestamp], key=lambda item: item[0])

        # 先处理真实执行事件。平仓释放名额后，同一时点才允许新建仓。
        for symbol, index in event_group:
            state = states[symbol]
            if state.pending_exit_exec_index == index:
                state.in_position = False
                state.entry_exec_index = None
                state.pending_exit_exec_index = None

            if state.pending_entry_exec_index == index:
                state.in_position = True
                state.entry_exec_index = index
                state.pending_entry_exec_index = None

        candidates: list[tuple[float, str, int]] = []
        eligible_scores: dict[str, float] = {}
        volumes_ok: dict[str, bool] = {}

        for symbol, index in event_group:
            state = states[symbol]
            signals = signals_by_symbol[symbol]
            can_execute = index + lag < len(per_symbol_rates[symbol])
            score = float(signals.exit_trailing_ann[index])
            volume_ok = True
            if quote_volumes_24h is not None:
                volume_series = quote_volumes_24h.get(symbol)
                volume = (
                    float(volume_series.asof(timestamp))
                    if volume_series is not None and not volume_series.empty
                    else float("nan")
                )
                volume_ok = np.isfinite(volume) and volume >= min_quote_volume_3d_avg
            volumes_ok[symbol] = volume_ok
            if (
                signals.entry[index]
                and not signals.exit[index]
                and volume_ok
                and can_execute
            ):
                state = states[symbol]
                if not state.in_position and state.pending_entry_exec_index is None:
                    eligible_scores[symbol] = float(signals.exit_trailing_ann[index])

        best_score = max(eligible_scores.values(), default=float("-inf"))
        for symbol, index in event_group:
            state = states[symbol]
            signals = signals_by_symbol[symbol]
            can_execute = index + lag < len(per_symbol_rates[symbol])

            # 成交前必须仍满足进场条件；转差或失去流动性则撤销未成交建仓。
            if state.pending_entry_exec_index is not None:
                if (
                    (not signals.entry[index] or signals.exit[index] or not volumes_ok[symbol])
                    and index < state.pending_entry_exec_index
                ):
                    state.pending_entry_exec_index = None
                continue

            if state.in_position:
                if state.pending_exit_exec_index is not None or not can_execute:
                    continue
                if state.entry_exec_index is None:
                    raise RuntimeError("动态选币状态异常：持仓缺少建仓索引")
                holding = index - state.entry_exec_index
                current_score = float(signals.exit_trailing_ann[index])
                better_exists = np.isfinite(best_score) and (
                    not np.isfinite(current_score) or best_score > current_score
                )
                premium = _replacement_premium(holding, strategy)
                floor_exit = False
                replacement = (
                    holding > 60
                    and better_exists
                    and np.isfinite(current_score)
                    and best_score >= current_score * (1.0 + premium)
                )
                if signals.exit[index] or replacement or floor_exit or (
                    holding >= strategy.exit.max_holding_periods
                ):
                    exit_masks[symbol][index] = True
                    state.pending_exit_exec_index = index + lag
                continue

            if signals.entry[index] and not signals.exit[index] and volumes_ok[symbol] and can_execute:
                score = signals.exit_trailing_ann[index]
                candidates.append(
                    (float(score) if np.isfinite(score) else float("-inf"), symbol, index)
                )

        occupied = sum(
            int(
                (state.in_position and state.pending_exit_exec_index is None)
                or state.pending_entry_exec_index is not None
            )
            for state in states.values()
        )
        slots = max(0, max_positions - occupied)
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for _, symbol, index in candidates[:slots]:
            entry_masks[symbol][index] = True
            states[symbol].pending_entry_exec_index = index + lag

    return entry_masks, exit_masks


def run_portfolio(
    per_symbol_rates: dict[str, pd.Series],
    intervals: dict[str, int],
    backtest: BacktestConfig,
    strategy: StrategyConfig,
    cost_model: CostModel,
    *,
    tiers: dict[str, LiquidityTier] | None = None,
    quote_volumes_24h: dict[str, pd.Series] | None = None,
    basis_adverse_pct: float = 0.0,
    max_positions: int | None = None,
) -> PortfolioResult:
    """跑固定滚动窗口、动态选币的多币种组合。

    每个资金费事件只使用该时点以前的数据计算信号。候选在事件发生时按
    trailing 年化排名，且已持仓/已排队订单都会占用 ``max_positions`` 名额。
    因此组合不会把未来赢家回填到过去，也不会把独立单币结果简单相加。

    交易成本和收益由单币事件引擎按真实执行时点记账，再按时间轴合成。
    ``max_positions × per_position_weight`` 超过 1 时直接拒绝，避免组合
    在名义资金上超配后仍被误读为回测结论。

    Raises:
        InsufficientDataError: 全部币种都跑失败。
        ValueError: 组合目标仓位超过 100% 总资金。
    """
    tiers = tiers or {}
    limit = max_positions if max_positions is not None else strategy.selection.max_positions
    weight = strategy.selection.per_position_weight

    if limit <= 0:
        raise ValueError(f"max_positions 必须为正，当前 {limit}")
    if limit * weight > 1.0 + 1e-9:
        raise ValueError(
            f"max_positions({limit}) × per_position_weight({weight:.2f}) > 1.0，"
            "真实组合回测拒绝超配"
        )

    eligible_rates: dict[str, pd.Series] = {}
    eligible_intervals: dict[str, int] = {}
    eligible_volumes: dict[str, pd.Series] = {}
    failures: dict[str, str] = {}
    min_history = max(strategy.entry.lookback_periods, backtest.rolling_window_periods)
    min_periods = min_history + backtest.execution_lag_bars + 1

    for symbol, raw_rates in per_symbol_rates.items():
        interval = intervals.get(symbol)
        if interval is None:
            failures[symbol] = "缺少结算周期"
            continue
        rates = normalize_funding_to_8h(raw_rates.astype(float), interval)
        if len(rates) < min_periods:
            failures[symbol] = (
                f"{symbol} 仅 {len(rates)} 期数据，至少需要 {min_periods} 期"
                f"（滚动窗口 {min_history} + lag {backtest.execution_lag_bars} + 1）"
            )
            continue
        eligible_rates[symbol] = rates
        eligible_intervals[symbol] = 8
        if quote_volumes_24h is not None and symbol in quote_volumes_24h:
            eligible_volumes[symbol] = quote_volumes_24h[symbol].astype(float).sort_index()

    if not eligible_rates:
        raise InsufficientDataError(
            f"全部 {len(per_symbol_rates)} 个币种回测失败。失败原因: {failures}"
        )

    if failures:
        logger.warning("%d 个币种因数据不足被跳过: %s", len(failures), list(failures)[:10])

    symbols = list(eligible_rates)
    entry_masks, exit_masks = _build_dynamic_masks(
        eligible_rates,
        eligible_intervals,
        backtest,
        strategy,
        symbols,
        limit,
        quote_volumes_24h=eligible_volumes if quote_volumes_24h is not None else None,
        min_quote_volume_3d_avg=strategy.selection.min_quote_volume_3d_avg,
    )

    per_symbol: dict[str, BacktestResult] = {}
    for symbol in symbols:
        engine = FundingCarryBacktester(
            backtest,
            strategy,
            cost_model,
            tier=tiers.get(symbol, LiquidityTier.SMALL),
            basis_adverse_pct=basis_adverse_pct,
        )
        try:
            per_symbol[symbol] = engine.run(
                symbol,
                eligible_rates[symbol],
                eligible_intervals[symbol],
                entry_allowed=entry_masks[symbol],
                exit_allowed=exit_masks[symbol],
                quote_volume_24h=eligible_volumes.get(symbol),
            )
        except InsufficientDataError as exc:
            failures[symbol] = str(exc)

    if not per_symbol:
        raise InsufficientDataError(
            f"全部 {len(per_symbol_rates)} 个币种回测失败。失败原因: {failures}"
        )

    # 各币种事件时间可能不同，使用有序外连接；缺失事件代表该币当期无收益。
    aligned = pd.DataFrame({sym: res.period_returns for sym, res in per_symbol.items()})
    aligned = aligned.sort_index().fillna(0.0)
    portfolio_returns = aligned.sum(axis=1)

    gross_aligned = pd.DataFrame({sym: res.gross_returns for sym, res in per_symbol.items()})
    gross_aligned = gross_aligned.reindex(aligned.index).fillna(0.0)
    portfolio_gross = gross_aligned.sum(axis=1)

    cost_aligned = pd.DataFrame({sym: res.cost_returns for sym, res in per_symbol.items()})
    cost_aligned = cost_aligned.reindex(aligned.index).fillna(0.0)
    portfolio_cost = cost_aligned.sum(axis=1)

    exposure_aligned = pd.DataFrame({sym: res.exposure for sym, res in per_symbol.items()})
    exposure_aligned = exposure_aligned.reindex(aligned.index).fillna(0.0)
    portfolio_exposure = exposure_aligned.sum(axis=1)

    if float(portfolio_exposure.max()) > 1.0 + 1e-9:
        raise RuntimeError("动态选币产生了超过 100% 的组合敞口")

    # 混合周期没有单一自然频率；沿用最密周期作为保守的年化近似。
    max_ppy = max((24.0 / res.interval_hours) * 365.0 for res in per_symbol.values())

    metrics = compute_metrics(
        portfolio_returns,
        gross_returns=portfolio_gross,
        cost_returns=portfolio_cost,
        periods_per_year=max_ppy,
        n_round_trips=sum(len(res.trades) for res in per_symbol.values()),
        total_periods_available=len(portfolio_returns),
        exposure=portfolio_exposure,
    )

    return PortfolioResult(
        per_symbol=per_symbol,
        portfolio_returns=portfolio_returns,
        metrics=metrics,
        weight_per_position=weight,
        portfolio_gross_returns=portfolio_gross,
        portfolio_cost_returns=portfolio_cost,
        portfolio_exposure=portfolio_exposure,
        max_positions=limit,
        dynamic_selection=True,
    )



__all__ = [
    "BacktestResult",
    "FundingCarryBacktester",
    "PortfolioResult",
    "Signals",
    "TradeRecord",
    "build_signals",
    "run_portfolio",
]
