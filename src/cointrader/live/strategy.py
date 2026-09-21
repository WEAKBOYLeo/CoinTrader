"""实时策略协调器（开发文档 §7.1 / §7.2 / §7.3 / §7.4）。

职责边界（硬约束）：

1. **低频刷新**候选指标缓存（资金费历史/周期/流动性），主循环不重复拉历史。
2. 主循环每轮 ``evaluate()``：读缓存 + 新鲜报价（由 service 注入）→
   每个候选产出可持久化的 ``StrategyDecision``（拒绝也落盘）。
3. 策略条件严格复用配置与回测语义（§4：滑窗年化、连续正费率、
   负均值退出、最长持仓、换仓 premium），计算公式与 ``backtest.engine``
   一致（同一条「最近 N 期滑动平均费率」口径）。
4. 本模块不做任何 IO 之外的交易动作；下单由 ``LiveService`` 调
   ``PairExecutor``。``build_signal()`` 只负责交易意图前的市场数据/
   名义额检查（portfolio.py），资金费策略判断在本模块。
5. 去重统一入口 ``can_open()``：实际持仓 / 非终态 pair / intent / order /
   本轮已提交，任一成立禁止重复开仓（§7.3）。
6. **判断逻辑委托**（实施计划书 3.0 T2）：指标计算与入场/退出/排序判断
   全部在纯 evaluator（``cointrader.strategy.funding_carry.FundingCarryEvaluator``）
   中执行；本类只保留 IO（候选刷新、去重查询、报价获取/校验）与
   ``StrategyDecision`` 组装，公开调用签名保持不变。

无未来数据：所有窗口右端都是当前点，只用已结算（已可见）资金费。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol

from ..config import Config
from ..errors import LiveGateBlocked
from ..execution.store import StateStore
from ..strategy.funding_carry import (
    CandidateInput,
    CarryEvaluation,
    EvalContext,
    EvalKind,
    FundingCarryEvaluator,
    HeldInput,
    QuoteInput,
)
from . import portfolio
from .decisions import DecisionKind, ReasonCode, StrategyDecision
from .market_sync import MarketDataSynchronizer, ScanEpoch

logger = logging.getLogger(__name__)

__all__ = [
    "CandidateCache",
    "HeldPosition",
    "LiveContext",
    "LiveStrategy",
    "Quote",
    "StrategyDataProvider",
]


# ---------------------------------------------------------------------------
# 数据注入
# ---------------------------------------------------------------------------


class StrategyDataProvider(Protocol):
    """策略数据源（外部注入，单测用 fake，不访问真实交易所）。

    ``funding_rates`` / ``quote_volume_3d_avg`` 支持可选 ``end_ms``：scan epoch
    用它把窗口固定到统一 cutoff 前（只用已闭合/已发生的数据）。
    """

    def funding_rates(
        self, symbol: str, periods: int, *, end_ms: int | None = None
    ) -> list[tuple[int, Decimal, Decimal]]:
        """升序 (结算时间戳 ms, 单期费率, 结算标记价)；``end_ms`` 给定只返回 <= end_ms。"""

    def funding_interval_hours(self, symbol: str) -> int:
        """该合约的资金费结算周期（小时），必须来自 fundingInfo。"""

    def quote_volume_3d_avg(self, symbol: str, *, end_ms: int | None = None) -> Decimal:
        """最近 3 天平均日成交额（USDT）；``end_ms`` 给定只使用闭合于其前的 4h K 线。"""

    def tradable_universe(self) -> tuple[str, ...]:
        """全部可交易 USDT 永续 symbol（已排除 exclude_bases）。仅动态候选池模式使用。"""

    def quote_volume_24h(self) -> dict[str, float]:
        """全市场各 symbol 的 24h 成交额（USDT）。仅动态候选池模式使用。"""


@dataclass(slots=True)
class CandidateCache:
    """单个候选的指标缓存（低频刷新，主循环只读）。"""

    symbol: str
    rates: list[Decimal] = field(default_factory=list)  # 单期费率，升序
    mark_prices: list[Decimal] = field(default_factory=list)  # 结算标记价（与 rates 对齐）
    timestamps: list[int] = field(default_factory=list)  # 结算时间 ms
    interval_hours: int = 8
    volume_3d_avg: Decimal = Decimal("0")
    refreshed_ts_ms: int = 0
    error: str = ""


@dataclass(frozen=True, slots=True)
class Quote:
    """一次新鲜报价（提交开仓前必须重新获取）。"""

    spot_price: Decimal
    perp_price: Decimal
    ts_ms: int  # 本地接收时间（UTC ms）
    spot_ts_ms: int = 0  # 现货报价接收时间（0 = 用 ts_ms）
    perp_ts_ms: int = 0  # 永续报价接收时间（0 = 用 ts_ms）


@dataclass(frozen=True, slots=True)
class HeldPosition:
    """交易所确认的实际持仓（对账/快照来源，不是本地 pair 状态）。"""

    symbol: str
    spot_qty: Decimal
    perp_qty: Decimal  # 负数 = 空头
    opened_ms: int = 0  # 开仓时间（0 = 未知）


@dataclass(slots=True)
class LiveContext:
    """一轮 evaluate() 的输入（由 LiveService 构造）。"""

    now_ms: int
    run_id: str
    total_capital: Decimal  # 真实账户资金；0 = 未知 → 禁止开仓
    held: dict[str, HeldPosition]
    quotes: dict[str, Quote]
    submitted_this_run: frozenset[str] = frozenset()
    account_ok: bool = True
    reconcile_ok: bool = True
    # scan epoch（实施计划书 v2.0 T2）：无 synchronizer 的旧路径默认 True
    scan_epoch_id: str = ""
    decision_cutoff_ms: int = 0
    market_data_ready: bool = True


class _StoreDedupProbe:
    """账本只读去重查询 → 纯 evaluator 的 ``DedupProbe``。"""

    def __init__(self, store: StateStore) -> None:
        self._store = store

    def active_pair(self, symbol: str) -> bool:
        return self._store.active_pair_for_symbol(symbol) is not None

    def has_open_intent(self, symbol: str) -> bool:
        return bool(self._store.has_open_intent(symbol))

    def has_open_order(self, symbol: str) -> bool:
        return bool(self._store.has_open_order(symbol))


# ---------------------------------------------------------------------------
# 协调器
# ---------------------------------------------------------------------------


class LiveStrategy:
    """实时策略协调器（兼容 facade）。

    判断逻辑委托 ``FundingCarryEvaluator``（纯计算）；本类保留：
    - IO：候选池/指标刷新、scan epoch 触发、store 去重查询；
    - 决策组装：评估结果 → ``StrategyDecision``（字段/原因码逐一对应）。

    Args:
        config: 顶层配置（门槛唯一来源 = strategy 段 + execution.canary_notional）。
        data: 数据源（注入，单测用 fake）。
        store: 账本（去重查询用；只读方法）。
        strategy_version: 策略版本标签。
        config_hash: 配置摘要 hash。
        now_fn: 可注入时钟（秒）。
        synchronizer: 可选的 MarketDataSynchronizer；注入后所有排名/开仓/换仓
            绑定其最近 READY epoch（同一 cutoff 横截面），market_data_ready
            未就绪时未持仓候选统一 SKIP MARKET_DATA_NOT_READY。
    """

    def __init__(
        self,
        *,
        config: Config,
        data: StrategyDataProvider,
        store: StateStore,
        strategy_version: str = "funding_carry-1.0",
        config_hash: str = "",
        now_fn: Callable[[], float] = time.time,
        synchronizer: MarketDataSynchronizer | None = None,
    ) -> None:
        self.config = config
        self.data = data
        self.store = store
        self.strategy_version = strategy_version
        self.config_hash = config_hash
        self._now = now_fn
        self._synchronizer = synchronizer
        self.candidates: dict[str, CandidateCache] = {}
        self._universe: list[str] = []
        self._universe_ts_ms = 0
        self._epoch_id: str = ""
        self._evaluator = FundingCarryEvaluator(
            config=config,
            strategy_version=strategy_version,
            config_hash=config_hash,
            now_fn=now_fn,
        )
        self._dedup = _StoreDedupProbe(store)

    # -- 低频刷新 -------------------------------------------------------------

    @property
    def dynamic_pool(self) -> bool:
        """``live_symbols`` 为空 = 动态候选池（回测同口径：可交易永续 + 成交额过滤 + top N）。"""
        return not self.config.execution.live_symbols

    @property
    def candidate_symbols(self) -> tuple[str, ...]:
        """当前候选池。epoch 模式 = 最近 READY epoch 的已完成候选；
        固定模式 = 配置的 ``live_symbols``；legacy 动态模式 = 最近一次
        ``refresh_universe()`` 的结果（首刷前为空，无开仓评估）。"""
        if self._synchronizer is not None:
            return tuple(self.candidates.keys())
        if self.dynamic_pool:
            return tuple(self._universe)
        return tuple(self.config.execution.live_symbols)

    def _current_ready_epoch(self) -> ScanEpoch | None:
        if self._synchronizer is None:
            return None
        return self._synchronizer.latest_ready()

    def refresh_universe(self) -> None:
        """刷新候选池（动态模式）：可交易永续 → 24h 成交额门槛 → 按量 top N。

        固定模式只重置池时间戳（零请求）。动态模式刷新失败保留旧池；
        池内 symbol 指标缓存的时效由 ``max_candidate_data_age_seconds`` 把关。
        """
        if not self.dynamic_pool:
            self._universe = list(self.config.execution.live_symbols)
            self._universe_ts_ms = int(self._now() * 1000)
            return
        now_ms = int(self._now() * 1000)
        interval_ms = int(self.config.execution.universe_refresh_seconds * 1000)
        if self._universe and now_ms - self._universe_ts_ms < interval_ms:
            return
        try:
            symbols = tuple(self.data.tradable_universe())
            volumes = {str(k): float(v) for k, v in self.data.quote_volume_24h().items()}
        except Exception as exc:  # noqa: BLE001 —— 保留旧池（可能为空），不中断主循环
            logger.warning("动态候选池刷新失败（保留旧池 %d 个）: %s", len(self._universe), exc)
            return
        min_volume = Decimal(str(self.config.strategy.selection.min_quote_volume_3d_avg))
        pool = [s for s in symbols if Decimal(str(volumes.get(s, 0.0))) >= min_volume]
        pool.sort(key=lambda s: volumes.get(s, 0.0), reverse=True)
        max_n = int(self.config.execution.candidate_pool_max_symbols)
        if max_n > 0:
            pool = pool[:max_n]
        self._universe = pool
        self._universe_ts_ms = now_ms
        logger.info(
            "动态候选池刷新: %d 个 symbol（universe=%d，24h 成交额>= %.0f，top %s）",
            len(pool), len(symbols), float(min_volume), max_n if max_n > 0 else "不限",
        )

    def prune_universe(self, allowed: set[str]) -> int:
        """从池中剔除不在 ``allowed`` 内的 symbol（无规则/非 TRADING/最小名义额不满足等）。"""
        removed = [s for s in self._universe if s not in allowed]
        self._universe = [s for s in self._universe if s in allowed]
        if removed:
            logger.info("候选池剔除 %d 个不可开仓 symbol: %s", len(removed), removed)
        return len(removed)

    def refresh_candidates(self) -> None:
        """刷新候选指标（由 service 每个 tick 调用）。

        - epoch 模式：触发/推进 scan epoch 构建；只把最近 READY epoch 的快照
          写入 ``self.candidates``（同一 cutoff 横截面，READY 前绝不发布）。
        - legacy 模式：各币按自己结算周期判超龄后刷新（失败保留旧缓存）。
        """
        if self._synchronizer is not None:
            try:
                self._synchronizer.trigger_refresh()
            except Exception as exc:  # noqa: BLE001 —— universe 失败保留旧 READY，不中断主循环
                logger.warning("scan epoch 触发失败（保留旧 READY）: %s", exc)
            epoch = self._current_ready_epoch()
            if epoch is not None and epoch.epoch_id != self._epoch_id:
                self._epoch_id = epoch.epoch_id
                self.candidates = {
                    snap.symbol: CandidateCache(
                        symbol=snap.symbol,
                        rates=list(snap.rates),
                        mark_prices=list(snap.mark_prices),
                        timestamps=list(snap.timestamps),
                        interval_hours=snap.interval_hours,
                        volume_3d_avg=snap.quote_volume_3d_avg,
                        refreshed_ts_ms=snap.fetched_ms,
                        error="",
                    )
                    for snap in self._synchronizer.snapshots_for(epoch.epoch_id).values()
                }
            return
        self._refresh_candidates_legacy()

    def _refresh_candidates_legacy(self) -> None:
        """legacy 刷新：各币刷新间隔 = 该币自己的资金费结算周期（两次结算之间
        费率不变）；失败保留旧缓存并标记 error/数据年龄。

        ⚠️ 旧版「每分钟 N 个 symbol」预算已被共享 weight 调度取代（实施计划书
        v2.0）：预算由 RateLimitCoordinator 按 scope/优先级统一控制。
        """
        self.refresh_universe()
        entry = self.config.strategy.entry
        exit_cfg = self.config.strategy.exit
        # 需要覆盖：入场窗口 + 连续正计数 + 退出窗口
        periods = max(
            entry.lookback_periods + entry.min_consecutive_positive,
            exit_cfg.exit_lookback_periods,
        ) + 5

        now = self._now()
        min_refetch_ms = int(self.config.execution.candidate_refresh_seconds * 1000)
        due: list[tuple[int, str]] = []
        for symbol in self.candidate_symbols:
            cache = self.candidates.get(symbol)
            if cache is None:
                due.append((0, symbol))
                continue
            interval_ms = int(cache.interval_hours) * 3600 * 1000
            age_ms = int(now * 1000) - cache.refreshed_ts_ms
            if cache.error or age_ms >= max(interval_ms, min_refetch_ms):
                due.append((cache.refreshed_ts_ms, symbol))
        due.sort(key=lambda item: item[0])  # 最旧的先刷

        for _ts, symbol in due:
            old = self.candidates.get(symbol)
            try:
                raw = self.data.funding_rates(symbol, periods)
                interval = int(self.data.funding_interval_hours(symbol))
                volume = self.data.quote_volume_3d_avg(symbol)
            except Exception as exc:  # noqa: BLE001
                # 刷新失败：保留旧缓存（带年龄），本轮拒绝开仓
                if old is not None:
                    old.error = f"{type(exc).__name__}: {exc}"
                else:
                    self.candidates[symbol] = CandidateCache(
                        symbol=symbol,
                        interval_hours=8,
                        refreshed_ts_ms=int(self._now() * 1000),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                logger.warning("候选 %s 指标刷新失败（保留旧缓存）: %s", symbol, exc)
                continue
            if not raw:
                self.candidates[symbol] = CandidateCache(
                    symbol=symbol,
                    refreshed_ts_ms=int(self._now() * 1000),
                    error="无资金费历史",
                )
                continue
            ts_list = [int(ts) for ts, _, _ in raw]
            self.candidates[symbol] = CandidateCache(
                symbol=symbol,
                rates=[Decimal(str(rate)) for _, rate, _ in raw],
                mark_prices=[Decimal(str(mark)) for _, _, mark in raw],
                timestamps=ts_list,
                interval_hours=interval,
                volume_3d_avg=Decimal(str(volume)),
                refreshed_ts_ms=int(self._now() * 1000),
                error="",
            )

    # -- 指标兼容转发（旧私有方法 → 纯 evaluator；仅测试/诊断使用） -------------

    def _entry_metrics(self, cache: CandidateCache) -> tuple[Decimal, int]:
        """兼容转发：与旧实现同口径（委托 ``FundingCarryEvaluator.entry_metrics``）。"""
        candidate = CandidateInput(
            symbol=cache.symbol,
            rates=tuple(cache.rates),
            mark_prices=tuple(cache.mark_prices),
            timestamps=tuple(cache.timestamps),
            interval_hours=cache.interval_hours,
            volume_3d_avg=cache.volume_3d_avg,
            refreshed_ts_ms=cache.refreshed_ts_ms,
            error=cache.error,
        )
        return self._evaluator.entry_metrics(candidate)

    # -- 纯评估上下文构造 -------------------------------------------------------

    def _eval_candidates(self) -> dict[str, CandidateInput]:
        """可变缓存 → 纯 evaluator 的不可变输入。"""
        return {
            symbol: CandidateInput(
                symbol=cache.symbol,
                rates=tuple(cache.rates),
                mark_prices=tuple(cache.mark_prices),
                timestamps=tuple(cache.timestamps),
                interval_hours=cache.interval_hours,
                volume_3d_avg=cache.volume_3d_avg,
                refreshed_ts_ms=cache.refreshed_ts_ms,
                error=cache.error,
            )
            for symbol, cache in self.candidates.items()
        }

    def _eval_context(self, ctx: LiveContext) -> EvalContext:
        """LiveContext → 纯 EvalContext；有 synchronizer 时绑定 READY epoch
        并同步回填 LiveContext 的 epoch 审计字段（与旧行为一致）。"""
        held = {
            symbol: HeldInput(
                symbol=position.symbol,
                spot_qty=position.spot_qty,
                perp_qty=position.perp_qty,
                opened_ms=position.opened_ms,
            )
            for symbol, position in ctx.held.items()
        }
        quotes = {
            symbol: QuoteInput(
                spot_price=quote.spot_price,
                perp_price=quote.perp_price,
                ts_ms=quote.ts_ms,
                spot_ts_ms=quote.spot_ts_ms,
                perp_ts_ms=quote.perp_ts_ms,
            )
            for symbol, quote in ctx.quotes.items()
        }
        if self._synchronizer is None:
            return EvalContext(
                now_ms=ctx.now_ms,
                total_capital=ctx.total_capital,
                held=held,
                quotes=quotes,
                submitted_this_run=ctx.submitted_this_run,
                account_ok=ctx.account_ok,
                reconcile_ok=ctx.reconcile_ok,
            )
        epoch = self._current_ready_epoch()
        if epoch is not None:
            ctx.market_data_ready = True
            ctx.scan_epoch_id = epoch.epoch_id
            ctx.decision_cutoff_ms = epoch.decision_cutoff_ms
        return EvalContext(
            now_ms=ctx.now_ms,
            total_capital=ctx.total_capital,
            held=held,
            quotes=quotes,
            submitted_this_run=ctx.submitted_this_run,
            account_ok=ctx.account_ok,
            reconcile_ok=ctx.reconcile_ok,
            epoch_active=True,
            epoch_id=epoch.epoch_id if epoch is not None else None,
            decision_cutoff_ms=epoch.decision_cutoff_ms if epoch is not None else 0,
            epoch_excluded=dict(epoch.excluded) if epoch is not None else {},
            expected_symbols=self._non_held_expected_symbols(ctx.held)
            if epoch is None
            else (),
        )

    # -- 主入口 ---------------------------------------------------------------

    def evaluate(self, ctx: LiveContext) -> list[StrategyDecision]:
        """评估全部候选与持仓。每个 symbol 恰好一条决策。

        判断逻辑（scan epoch 闸门、退出/换仓、入场门槛、槽位排序）在纯
        evaluator 中执行；本方法只负责上下文构造与决策组装。
        """
        eval_ctx = self._eval_context(ctx)
        candidates = self._eval_candidates()
        evaluations = self._evaluator.evaluate(
            eval_ctx, self.candidate_symbols, candidates, self._dedup
        )
        return [self._to_decision(ev, ctx) for ev in evaluations]

    def _non_held_expected_symbols(self, held: dict[str, HeldPosition]) -> tuple[str, ...]:
        """无 READY 时仍需产出决策的候选集合（每个 symbol 恰好一条决策）。"""
        if self._synchronizer is not None:
            expected = self._synchronizer.expected_symbols()
            excluded = set()
            latest = self._synchronizer.latest()
            if latest is not None:
                excluded = set(latest.excluded)
            return tuple(s for s in expected if s not in held and s not in excluded)
        return tuple(s for s in self.candidate_symbols if s not in held)

    def _epoch_stamp(self) -> tuple[str | None, int | None]:
        """当前 READY epoch 的审计戳（(epoch_id, cutoff_ms)）；无则 (None, None)。"""
        epoch = self._current_ready_epoch()
        if epoch is None:
            return (None, None)
        return (epoch.epoch_id, epoch.decision_cutoff_ms)

    # -- 决策组装（评估结果 → StrategyDecision） --------------------------------

    def _decision_skip(
        self, ctx: LiveContext, ev: CarryEvaluation, **fields: object
    ) -> StrategyDecision:
        scan_epoch_id, decision_cutoff_ms = self._epoch_stamp()
        base: dict[str, object] = {
            "symbol": ev.symbol,
            "run_id": ctx.run_id,
            "ts_ms": ctx.now_ms,
            "decision_kind": (
                DecisionKind.PENDING_QUOTE if ev.kind == EvalKind.PENDING_QUOTE else DecisionKind.SKIP
            ),
            "allowed": False,
            "reason_code": ev.reason_code,
            "reason_text": ev.reason_text,
            "strategy_version": self.strategy_version,
            "config_hash": self.config_hash,
            "funding_interval_hours": ev.funding_interval_hours,
            "trailing_annualized": ev.trailing_annualized,
            "exit_average_annualized": ev.exit_average_annualized,
            "consecutive_positive_periods": ev.consecutive_positive_periods,
            "quote_volume_3d_avg": ev.quote_volume_3d_avg,
            "entry_threshold": ev.entry_threshold,
            "exit_threshold": ev.exit_threshold,
            "position_age_periods": ev.position_age_periods,
            "spot_price": ev.spot_price,
            "perp_price": ev.perp_price,
            "quote_ts_ms": ev.quote_ts_ms,
            "requested_notional": ev.requested_notional,
            "scan_epoch_id": scan_epoch_id,
            "decision_cutoff_ms": decision_cutoff_ms,
            "metrics": dict(ev.metrics),
        }
        base.update(fields)
        return StrategyDecision(**base)  # type: ignore[arg-type]

    def _to_decision(self, ev: CarryEvaluation, ctx: LiveContext) -> StrategyDecision:
        scan_epoch_id, decision_cutoff_ms = self._epoch_stamp()

        if ev.kind == EvalKind.OPEN:
            if (
                ev.spot_price is None
                or ev.perp_price is None
                or ev.quote_ts_ms is None
                or ev.requested_notional is None
            ):
                return self._decision_skip(
                    ctx,
                    CarryEvaluation(
                        symbol=ev.symbol,
                        kind=EvalKind.SKIP,
                        allowed=False,
                        reason_code=ReasonCode.STALE_QUOTE,
                        reason_text="评估结果缺少开仓必要字段（内部错误）",
                    ),
                )
            # build_signal 保留为本地拦截兼容层（与 evaluator 同口径双保险）
            try:
                signal = portfolio.build_signal(
                    ev.symbol,
                    spot_price=ev.spot_price,
                    perp_price=ev.perp_price,
                    quote_ts_ms=ev.quote_ts_ms,
                    requested_notional=ev.requested_notional,
                    config=self.config,
                    reason="funding_carry",
                    strategy_version=self.strategy_version,
                    now_ms=ctx.now_ms,
                )
            except LiveGateBlocked as exc:
                return self._decision_skip(
                    ctx,
                    CarryEvaluation(
                        symbol=ev.symbol,
                        kind=EvalKind.SKIP,
                        allowed=False,
                        reason_code=ReasonCode.STALE_QUOTE,
                        reason_text=str(exc),
                    ),
                )
            if signal is None:
                return self._decision_skip(
                    ctx,
                    CarryEvaluation(
                        symbol=ev.symbol,
                        kind=EvalKind.SKIP,
                        allowed=False,
                        reason_code=ReasonCode.NOTIONAL_TOO_SMALL,
                        reason_text="build_signal 本地拒绝（名义额/价格非法）",
                    ),
                )
            return StrategyDecision(
                symbol=ev.symbol,
                run_id=ctx.run_id,
                ts_ms=ctx.now_ms,
                decision_kind=DecisionKind.OPEN,
                allowed=True,
                reason_code=ReasonCode.ENTRY_OK,
                reason_text="策略条件全部满足，允许开仓",
                strategy_version=self.strategy_version,
                config_hash=self.config_hash,
                funding_interval_hours=ev.funding_interval_hours,
                trailing_annualized=ev.trailing_annualized,
                consecutive_positive_periods=ev.consecutive_positive_periods,
                quote_volume_3d_avg=ev.quote_volume_3d_avg,
                entry_threshold=ev.entry_threshold,
                spot_price=ev.spot_price,
                perp_price=ev.perp_price,
                quote_ts_ms=ev.quote_ts_ms,
                requested_notional=signal.target_notional,
                scan_epoch_id=scan_epoch_id,
                decision_cutoff_ms=decision_cutoff_ms,
                metrics=dict(ev.metrics),
            )

        if ev.kind in (EvalKind.EXIT, EvalKind.REPLACE, EvalKind.HOLD):
            metrics: dict[str, object] = {}
            if ev.kind == EvalKind.REPLACE and ev.replacement_symbol is not None:
                metrics["replacement_symbol"] = ev.replacement_symbol
                if ev.premium is not None:
                    metrics["premium"] = str(ev.premium)
            return StrategyDecision(
                symbol=ev.symbol,
                run_id=ctx.run_id,
                ts_ms=ctx.now_ms,
                decision_kind=ev.kind,
                allowed=True,
                reason_code=ev.reason_code,
                reason_text=ev.reason_text,
                strategy_version=self.strategy_version,
                config_hash=self.config_hash,
                funding_interval_hours=ev.funding_interval_hours,
                trailing_annualized=ev.trailing_annualized,
                exit_average_annualized=ev.exit_average_annualized,
                quote_volume_3d_avg=ev.quote_volume_3d_avg,
                exit_threshold=ev.exit_threshold,
                position_age_periods=ev.position_age_periods,
                scan_epoch_id=scan_epoch_id,
                decision_cutoff_ms=decision_cutoff_ms,
                metrics=metrics,
            )

        # SKIP / PENDING_QUOTE
        return self._decision_skip(ctx, ev)

    # -- 单 symbol 入口（service 兼容） ------------------------------------------

    def can_open(self, symbol: str, ctx: LiveContext) -> StrategyDecision:
        """统一开仓判断入口（§7.3）。返回 SKIP/OPEN/PENDING_QUOTE 决策，本身不下单。

        前置门槛（步骤 1-12）通过后若本轮上下文无该 symbol 新鲜报价，返回
        ``PENDING_QUOTE`` 中间态：LiveService 按需获取报价后调 ``complete_open``
        产出最终决策（动态候选池池子大，每轮只为通过前置检查的少数候选拉报价）。
        """
        ev = self._evaluator.can_open(
            symbol, self._eval_context(ctx), self._eval_candidates(), self._dedup
        )
        return self._to_decision(ev, ctx)

    def complete_open(
        self, symbol: str, ctx: LiveContext, quote: Quote | None
    ) -> StrategyDecision:
        """第二阶段：LiveService 获取报价后定案（步骤 13-15）。获取失败 = STALE_QUOTE。"""
        quote_input = (
            None
            if quote is None
            else QuoteInput(
                spot_price=quote.spot_price,
                perp_price=quote.perp_price,
                ts_ms=quote.ts_ms,
                spot_ts_ms=quote.spot_ts_ms,
                perp_ts_ms=quote.perp_ts_ms,
            )
        )
        ev = self._evaluator.complete_open(
            symbol, self._eval_context(ctx), self._eval_candidates(), quote_input
        )
        return self._to_decision(ev, ctx)


# ---------------------------------------------------------------------------
# 真实数据 provider（live 层可依赖 data 层；§4.3 依赖方向）
# ---------------------------------------------------------------------------


class PublicDataStrategyProvider:
    """基于 ``BinancePublicClient`` 的策略数据源（只读公开接口）。"""

    def __init__(self, config: Config, client: Any | None = None,
                 now_fn: Callable[[], float] = time.time) -> None:
        from ..data.binance import BinancePublicClient
        from ..data.funding import FundingIntervals, fetch_funding_intervals

        self.config = config
        self._now = now_fn
        if client is None:
            client = BinancePublicClient(config.api, config.data)
        self.client: BinancePublicClient = client  # type: ignore[assignment]
        self._intervals: FundingIntervals | None = None
        self._intervals_fn = fetch_funding_intervals

    def tradable_universe(self) -> tuple[str, ...]:
        """全部可交易 USDT 永续（排除 exclude_bases；与回测 scanner 同口径）。"""
        from ..data.klines import tradable_perpetuals

        info = self.client.futures_exchange_info()
        return tuple(
            tradable_perpetuals(
                info, exclude_bases=self.config.strategy.selection.exclude_bases
            )
        )

    def quote_volume_24h(self) -> dict[str, float]:
        """全市场 24h 成交额（单次 ticker 请求，不做逐币请求）。"""
        out: dict[str, float] = {}
        for item in self.client.futures_tickers_24h():
            symbol = str(item.get("symbol") or "")
            if not symbol:
                continue
            try:
                out[symbol] = float(item.get("quoteVolume") or 0.0)
            except (TypeError, ValueError):
                continue
        return out

    def close(self) -> None:
        self.client.close()

    def funding_interval_hours(self, symbol: str) -> int:
        if self._intervals is None:
            self._intervals = self._intervals_fn(self.client)
        return int(self._intervals.get(symbol))

    def funding_rates(
        self, symbol: str, periods: int, *, end_ms: int | None = None
    ) -> list[tuple[int, Decimal, Decimal]]:
        from ..data.funding import fetch_funding_history

        interval = self.funding_interval_hours(symbol)
        # start 锚到 8h 网格：同一 8h 窗口内缓存键稳定，epoch 重试/重建命中
        # 磁盘缓存，避免每 pass 重复 3 页拉取触发 WAF 速率拦截
        span_ms = (periods + 5) * interval * 3600 * 1000
        _grid = 8 * 3600 * 1000
        start_ms = (int(self._now() * 1000) - span_ms) // _grid * _grid
        frame = fetch_funding_history(self.client, symbol, start_ms=start_ms)
        rates = frame["funding_rate"]
        marks = frame["mark_price"]
        out: list[tuple[int, Decimal, Decimal]] = []
        for ts, value, mark in zip(rates.index, rates, marks, strict=False):
            import math

            ts_ms = int(ts.timestamp() * 1000)
            if end_ms is not None and ts_ms > end_ms:
                continue  # epoch 不变量：只用 cutoff 前已发生的结算
            mark_value = float(mark)
            if math.isnan(mark_value):
                mark_value = 0.0
            out.append((ts_ms, Decimal(str(value)), Decimal(str(mark_value))))
        return out

    def quote_volume_3d_avg(self, symbol: str, *, end_ms: int | None = None) -> Decimal:
        import math

        if end_ms is not None:
            # epoch 模式：固定到「闭合时间 <= end_ms 的最后一根 4h K 线」，
            # 只请求最近 18 根闭合 K 线（weight 分档 1~2，不用 limit=1500 耗 10）。
            bar_ms = 4 * 3600 * 1000
            close_ms = (end_ms // bar_ms + 1) * bar_ms
            if close_ms > end_ms:
                close_ms -= bar_ms
            start_ms = close_ms - 18 * bar_ms
            klines = self.client.futures_klines(
                symbol, "4h", start_ms=start_ms, end_ms=end_ms, limit=20
            )
            closed = [k for k in klines if int(k[0]) + bar_ms <= end_ms][-18:]
            if len(closed) < 18:
                raise ValueError(f"{symbol} 闭合 4h K 线不足 18 根（window end={close_ms}）")
            day_totals: dict[int, Decimal] = {}
            for k in closed:
                open_ms = int(k[0])
                day = open_ms // (86_400 * 1000)
                day_totals[day] = day_totals.get(day, Decimal("0")) + Decimal(str(k[7]))
            days = sorted(day_totals)
            if len(days) < 3:
                raise ValueError(f"{symbol} 成交额窗口不足 3 天")
            avg = sum((day_totals[d] for d in days[-3:]), Decimal("0")) / 3
            if not avg.is_finite() or avg <= 0:
                raise ValueError(f"{symbol} 3 天平均日成交额无效")
            return avg

        from ..data.klines import fetch_historical_quote_volume_3d_avg

        start_ms = int(self._now() * 1000) - 5 * 86_400 * 1000
        series = fetch_historical_quote_volume_3d_avg(self.client, symbol, start_ms=start_ms)
        value = series.iloc[-1]
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            raise ValueError(f"{symbol} 3 天平均日成交额无效")
        return Decimal(str(value))
