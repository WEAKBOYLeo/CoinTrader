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
from . import portfolio
from .decisions import DecisionKind, ReasonCode, StrategyDecision

logger = logging.getLogger(__name__)

__all__ = [
    "CandidateCache",
    "HeldPosition",
    "LiveContext",
    "LiveStrategy",
    "Quote",
    "StrategyDataProvider",
]

_HOURS_PER_YEAR = Decimal(24 * 365)


# ---------------------------------------------------------------------------
# 数据注入
# ---------------------------------------------------------------------------


class StrategyDataProvider(Protocol):
    """策略数据源（外部注入，单测用 fake，不访问真实交易所）。"""

    def funding_rates(self, symbol: str, periods: int) -> list[tuple[int, Decimal, Decimal]]:
        """升序 (结算时间戳 ms, 单期费率, 结算标记价) 列表，至少 periods 期（若可得）。"""

    def funding_interval_hours(self, symbol: str) -> int:
        """该合约的资金费结算周期（小时），必须来自 fundingInfo。"""

    def quote_volume_3d_avg(self, symbol: str) -> Decimal:
        """最近 3 天平均日成交额（USDT）。"""

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


# ---------------------------------------------------------------------------
# 协调器
# ---------------------------------------------------------------------------


class LiveStrategy:
    """实时策略协调器。

    Args:
        config: 顶层配置（门槛唯一来源 = strategy 段 + execution.canary_notional）。
        data: 数据源（注入，单测用 fake）。
        store: 账本（去重查询用；只读方法）。
        strategy_version: 策略版本标签。
        config_hash: 配置摘要 hash。
        now_fn: 可注入时钟（秒）。
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
    ) -> None:
        self.config = config
        self.data = data
        self.store = store
        self.strategy_version = strategy_version
        self.config_hash = config_hash
        self._now = now_fn
        self.candidates: dict[str, CandidateCache] = {}
        self._universe: list[str] = []
        self._universe_ts_ms = 0
        self._refetch_win_start = 0.0
        self._refetch_win_count = 0

    # -- 低频刷新 -------------------------------------------------------------

    @property
    def dynamic_pool(self) -> bool:
        """``live_symbols`` 为空 = 动态候选池（回测同口径：可交易永续 + 成交额过滤 + top N）。"""
        return not self.config.execution.live_symbols

    @property
    def candidate_symbols(self) -> tuple[str, ...]:
        """当前候选池。固定模式 = 配置的 ``live_symbols``；动态模式 = 最近一次
        ``refresh_universe()`` 的结果（首刷前为空，无开仓评估）。"""
        if self.dynamic_pool:
            return tuple(self._universe)
        return tuple(self.config.execution.live_symbols)

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
        """刷新候选指标缓存（由 service 每个 tick 调用，内部判断超龄才拉取）。

        每币刷新间隔 = 该币自己的资金费结算周期（两次结算之间费率不变，
        提前刷无新信息）；每 60 秒窗口限量 ``candidate_refetch_per_minute``
        个 symbol，把 00/04/08/12/16/20 UTC 结算边界的全员到点摊开，
        不撞 API 按分钟计的限流。失败保留旧缓存并标记 error/数据年龄。
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
        # 每 60s 窗口限流（币安限流按分钟计，边界全员到点不能一次全拉）
        if now - self._refetch_win_start >= 60.0:
            self._refetch_win_start = now
            self._refetch_win_count = 0
        budget = int(self.config.execution.candidate_refetch_per_minute) - self._refetch_win_count
        if budget <= 0:
            return
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
        self._refetch_win_count += min(len(due), budget)

        for _ts, symbol in due[:budget]:
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

    def _cache_age_ms(self, cache: CandidateCache) -> int:
        return self._ctx_now_ms() - cache.refreshed_ts_ms

    def _ctx_now_ms(self) -> int:
        return int(self._now() * 1000)

    # -- 指标（纯函数，复用 backtest 语义） ------------------------------------

    @staticmethod
    def _annualized(period_rate: Decimal, interval_hours: int) -> Decimal:
        return period_rate * _HOURS_PER_YEAR / Decimal(interval_hours)

    def _entry_metrics(self, cache: CandidateCache) -> tuple[Decimal, int]:
        """(最近 lookback 期滑动平均的年化, 连续正滑动平均期数)。

        与 ``backtest.engine.build_signals`` 同口径：
        先对原始费率做 lookback 期滑动平均，再按该币自己的周期年化。
        """
        rates = cache.rates
        lookback = self.config.strategy.entry.lookback_periods
        if len(rates) < lookback:
            return Decimal("0"), 0
        window = rates[-lookback:]
        mean = sum(window, Decimal("0")) / Decimal(lookback)
        annualized = self._annualized(mean, cache.interval_hours)

        # 连续正：从当前位置往回数「滑动平均为正」的期数（窗口未满的位置不算正）
        streak = 0
        for pos in range(len(rates) - 1, lookback - 2, -1):
            w = rates[pos - lookback + 1 : pos + 1]
            if sum(w, Decimal("0")) / Decimal(lookback) > 0:
                streak += 1
            else:
                break
        return annualized, streak

    def _exit_average_annualized(self, cache: CandidateCache) -> Decimal | None:
        """最近 exit_lookback_periods 期平均费率年化；窗口未满返回 None。"""
        rates = cache.rates
        window = self.config.strategy.exit.exit_lookback_periods
        if len(rates) < window:
            return None
        mean = sum(rates[-window:], Decimal("0")) / Decimal(window)
        return self._annualized(mean, cache.interval_hours)

    def _position_age_periods(self, held: HeldPosition, interval_hours: int) -> int:
        if held.opened_ms <= 0:
            return 0
        period_ms = Decimal(interval_hours) * 3600 * 1000
        return int((self._ctx_now_ms() - held.opened_ms) // period_ms)

    # -- 主入口 ---------------------------------------------------------------

    def evaluate(self, ctx: LiveContext) -> list[StrategyDecision]:
        """评估全部候选与持仓。每个 symbol 恰好一条决策。"""
        decisions: list[StrategyDecision] = []

        # 1) 持仓 symbol：退出/换仓评估（异常退出路径由 service 优先处理）
        for symbol in sorted(ctx.held):
            held = ctx.held[symbol]
            decisions.append(self._evaluate_exit(symbol, held, ctx))

        # 2) 未持仓候选：开仓评估
        for symbol in self.candidate_symbols:
            if symbol in ctx.held:
                continue
            decisions.append(self.can_open(symbol, ctx))

        return decisions

    # -- 开仓（§7.2 判断顺序，尽早拒绝并记录原因） -----------------------------

    def _make_skip(self, symbol: str, ctx: LiveContext) -> Callable[..., StrategyDecision]:
        """构造拒绝/中间态决策的闭包（§7.2 判断顺序各处复用同一形状）。"""

        def skip(code: str, text: str, kind: str = DecisionKind.SKIP,
                 **metrics: object) -> StrategyDecision:
            # 已知审计字段提升到顶层列（DB 可直接查询），其余进 metrics
            lift_keys = (
                "funding_interval_hours", "trailing_annualized", "exit_average_annualized",
                "consecutive_positive_periods", "quote_volume_3d_avg", "entry_threshold",
                "exit_threshold", "position_age_periods", "spot_price", "perp_price",
                "quote_ts_ms", "requested_notional",
            )
            lifted = {k: metrics.pop(k) for k in lift_keys if k in metrics}
            return StrategyDecision(
                symbol=symbol,
                run_id=ctx.run_id,
                ts_ms=ctx.now_ms,
                decision_kind=kind,
                allowed=False,
                reason_code=code,
                reason_text=text,
                strategy_version=self.strategy_version,
                config_hash=self.config_hash,
                metrics=dict(metrics),
                **lifted,  # type: ignore[arg-type]
            )

        return skip

    def can_open(self, symbol: str, ctx: LiveContext) -> StrategyDecision:
        """统一开仓判断入口（§7.3）。返回 SKIP/OPEN/PENDING_QUOTE 决策，本身不下单。

        前置门槛（步骤 1-12）通过后若本轮上下文无该 symbol 新鲜报价，返回
        ``PENDING_QUOTE`` 中间态：LiveService 按需获取报价后调 ``complete_open``
        产出最终决策（动态候选池池子大，每轮只为通过前置检查的少数候选拉报价）。
        """
        skip = self._make_skip(symbol, ctx)
        entry = self.config.strategy.entry
        selection = self.config.strategy.selection
        base = symbol.replace("USDT", "")
        cache = self.candidates.get(symbol)

        # 前置状态（service 也已把关，这里记录原因保证决策链完整）
        if not ctx.reconcile_ok:
            return skip(ReasonCode.RECONCILIATION_BLOCKED, "最近对账未通过，禁止新增风险")
        if not ctx.account_ok or ctx.total_capital <= 0:
            return skip(ReasonCode.ACCOUNT_STATE_UNKNOWN, "账户快照缺失/过期/资金为零，禁止开仓")

        # 5. 排除列表
        if base.upper() in {b.upper() for b in selection.exclude_bases}:
            return skip(ReasonCode.EXCLUDED_ASSET, f"{base} 在排除列表中")
        if not symbol.endswith("USDT"):
            return skip(ReasonCode.INVALID_SYMBOL, f"{symbol} 不是 USDT 永续对")

        # 6/7/8/9. 数据窗口与策略门槛
        if cache is None:
            return skip(ReasonCode.INSUFFICIENT_HISTORY, "候选指标缓存不存在（刷新失败）")
        if cache.error:
            return skip(ReasonCode.STALE_DATA, f"候选指标缓存异常: {cache.error}")
        # 数据年龄上限 = 该币结算周期 + 宽限（两次结算间费率不变，超龄=刷新掉链）
        stale_after_ms = (
            int(cache.interval_hours) * 3600 * 1000
            + int(self.config.execution.max_candidate_data_age_seconds * 1000)
        )
        if self._cache_age_ms(cache) > stale_after_ms:
            return skip(
                ReasonCode.STALE_DATA,
                f"候选指标数据年龄 {self._cache_age_ms(cache)}ms 超过 {stale_after_ms}ms",
            )
        if len(cache.rates) < entry.lookback_periods:
            return skip(
                ReasonCode.INSUFFICIENT_HISTORY,
                f"资金费历史 {len(cache.rates)} 期 < 入场窗口 {entry.lookback_periods} 期",
            )

        trailing, streak = self._entry_metrics(cache)
        min_trailing = Decimal(str(entry.min_trailing_annualized))
        min_rate = Decimal(str(entry.min_annualized_rate))
        if trailing < max(min_trailing, min_rate):
            return skip(
                ReasonCode.TRAILING_RATE_BELOW_THRESHOLD,
                f"最近 {entry.lookback_periods} 期滑动平均年化 {trailing} 低于门槛 "
                f"{max(min_trailing, min_rate)}",
                trailing_annualized=trailing,
                consecutive_positive_periods=streak,
                funding_interval_hours=cache.interval_hours,
            )
        if streak < entry.min_consecutive_positive:
            return skip(
                ReasonCode.CONSECUTIVE_POSITIVE_TOO_SHORT,
                f"连续正滑动平均 {streak} 期 < 要求 {entry.min_consecutive_positive} 期",
                trailing_annualized=trailing,
                consecutive_positive_periods=streak,
                funding_interval_hours=cache.interval_hours,
            )
        min_volume = Decimal(str(selection.min_quote_volume_3d_avg))
        if cache.volume_3d_avg < min_volume:
            return skip(
                ReasonCode.LOW_LIQUIDITY,
                f"3 天平均日成交额 {cache.volume_3d_avg} < 门槛 {min_volume}",
                quote_volume_3d_avg=cache.volume_3d_avg,
            )

        # 10-12. 去重（§7.3 三层 + 本轮已提交 + 最大持仓数）
        if symbol in ctx.held:
            return skip(ReasonCode.ALREADY_HELD, "交易所存在实际持仓")
        if symbol in ctx.submitted_this_run:
            return skip(ReasonCode.ALREADY_SUBMITTED, "本轮已提交相同 symbol 的开仓")
        if self.store.active_pair_for_symbol(symbol):
            return skip(ReasonCode.ACTIVE_ORDER, "存在同 symbol 非终态 pair")
        if self.store.has_open_intent(symbol):
            return skip(ReasonCode.ACTIVE_INTENT, "存在同 symbol 未终态 intent")
        if self.store.has_open_order(symbol):
            return skip(ReasonCode.ACTIVE_ORDER, "存在同 symbol 未终态 order")
        if len(ctx.held) >= selection.max_positions:
            return skip(
                ReasonCode.MAX_POSITIONS,
                f"当前持仓 {len(ctx.held)} 已达上限 {selection.max_positions}",
            )

        # 13. 新鲜报价（动态候选池：上下文未含该 symbol 报价 → PENDING 中间态）
        quote = ctx.quotes.get(symbol)
        if quote is None:
            return skip(
                ReasonCode.PENDING_QUOTE,
                "前置门槛通过，等待新鲜报价（service 按需获取）",
                kind=DecisionKind.PENDING_QUOTE,
            )
        return self._open_with_quote(symbol, ctx, quote, skip)

    def complete_open(
        self, symbol: str, ctx: LiveContext, quote: Quote | None
    ) -> StrategyDecision:
        """第二阶段：LiveService 获取报价后定案（步骤 13-15）。获取失败 = STALE_QUOTE。"""
        skip = self._make_skip(symbol, ctx)
        if quote is None:
            return skip(ReasonCode.STALE_QUOTE, "无新鲜报价（本轮获取失败）")
        return self._open_with_quote(symbol, ctx, quote, skip)

    def _open_with_quote(
        self, symbol: str, ctx: LiveContext, quote: Quote,
        skip: Callable[..., StrategyDecision],
    ) -> StrategyDecision:
        """开仓第二阶段（§7.2 步骤 13-15）：报价新鲜度 → 名义额/基差 → build_signal。"""
        selection = self.config.strategy.selection
        cache = self.candidates.get(symbol)
        if cache is None:
            return skip(ReasonCode.INSUFFICIENT_HISTORY, "候选指标缓存缺失（前置检查后异常）")
        trailing, streak = self._entry_metrics(cache)
        min_trailing = Decimal(str(self.config.strategy.entry.min_trailing_annualized))
        max_quote_age_ms = int(self.config.execution.max_market_data_age_seconds * 1000)
        age = ctx.now_ms - quote.ts_ms
        if age < 0 or age > max_quote_age_ms:
            return skip(ReasonCode.STALE_QUOTE, f"报价年龄 {age}ms > {max_quote_age_ms}ms")
        if quote.spot_price <= 0 or quote.perp_price <= 0:
            return skip(ReasonCode.STALE_QUOTE, "报价非正数")

        # 14. 名义额与意图前市场检查（build_signal 只负责这层，§7.1 第 6 条）
        requested = (ctx.total_capital * Decimal(str(selection.per_position_weight))).quantize(
            Decimal("0.01")
        )
        if requested <= 0:
            return skip(ReasonCode.NOTIONAL_TOO_SMALL, "目标名义额非正")
        basis = (quote.perp_price - quote.spot_price) / quote.spot_price
        if basis < 0 and abs(basis) > Decimal(str(self.config.execution.hedge_tolerance_pct)):
            return skip(
                ReasonCode.BASIS_DISCOUNT,
                f"永续深度贴水 {basis}，开空头不利，等待收敛",
                spot_price=quote.spot_price,
                perp_price=quote.perp_price,
                quote_ts_ms=quote.ts_ms,
            )
        try:
            signal = portfolio.build_signal(
                symbol,
                spot_price=quote.spot_price,
                perp_price=quote.perp_price,
                quote_ts_ms=quote.ts_ms,
                requested_notional=requested,
                config=self.config,
                reason="funding_carry",
                strategy_version=self.strategy_version,
                now_ms=ctx.now_ms,
            )
        except LiveGateBlocked as exc:
            return skip(ReasonCode.STALE_QUOTE, str(exc))
        if signal is None:
            return skip(ReasonCode.NOTIONAL_TOO_SMALL, "build_signal 本地拒绝（名义额/价格非法）")

        # 15. 通过 → OPEN（intent 与下单由 service 执行）
        return StrategyDecision(
            symbol=symbol,
            run_id=ctx.run_id,
            ts_ms=ctx.now_ms,
            decision_kind=DecisionKind.OPEN,
            allowed=True,
            reason_code=ReasonCode.ENTRY_OK,
            reason_text="策略条件全部满足，允许开仓",
            strategy_version=self.strategy_version,
            config_hash=self.config_hash,
            funding_interval_hours=cache.interval_hours,
            trailing_annualized=trailing,
            consecutive_positive_periods=streak,
            quote_volume_3d_avg=cache.volume_3d_avg,
            entry_threshold=min_trailing,
            spot_price=quote.spot_price,
            perp_price=quote.perp_price,
            quote_ts_ms=quote.ts_ms,
            requested_notional=signal.target_notional,
            metrics={
                "quote_source": "public",
                "requested_notional_raw": str(requested),
            },
        )

    # -- 退出 / 换仓（§7.4） ----------------------------------------------------

    def _evaluate_exit(self, symbol: str, held: HeldPosition, ctx: LiveContext) -> StrategyDecision:
        exit_cfg = self.config.strategy.exit
        cache = self.candidates.get(symbol)
        interval = cache.interval_hours if cache else 8
        age_periods = self._position_age_periods(held, interval)
        exit_avg = self._exit_average_annualized(cache) if cache else None

        common: dict[str, object] = {
            "run_id": ctx.run_id,
            "ts_ms": ctx.now_ms,
            "symbol": symbol,
            "strategy_version": self.strategy_version,
            "config_hash": self.config_hash,
            "position_age_periods": age_periods,
            "exit_average_annualized": exit_avg,
            "exit_threshold": Decimal("0"),
            "funding_interval_hours": interval,
        }
        if cache is not None:
            trailing, _ = self._entry_metrics(cache)
            common["trailing_annualized"] = trailing
            common["quote_volume_3d_avg"] = cache.volume_3d_avg

        # 1. 负资金费退出（最近 exit_lookback 期均值转负；窗口未满不判定）
        if exit_avg is not None and exit_avg < 0:
            return StrategyDecision(
                decision_kind=DecisionKind.EXIT,
                allowed=True,
                reason_code=ReasonCode.NEGATIVE_EXIT_AVG,
                reason_text=(
                    f"最近 {exit_cfg.exit_lookback_periods} 期平均资金费率年化 {exit_avg} 转负，策略退出"
                ),
                **common,  # type: ignore[arg-type]
            )

        # 2. 最长持仓（结算周期数）
        if age_periods >= exit_cfg.max_holding_periods:
            return StrategyDecision(
                decision_kind=DecisionKind.EXIT,
                allowed=True,
                reason_code=ReasonCode.MAX_HOLDING,
                reason_text=f"持仓 {age_periods} 期达到最长 {exit_cfg.max_holding_periods} 期，强制退出",
                **common,  # type: ignore[arg-type]
            )

        # 3. 换仓：新候选相对优势超过按年龄分段的 premium（先平旧，后开新）
        if cache is not None and age_periods > 60:
            premium = (
                Decimal(str(exit_cfg.replacement_premium_under_120))
                if age_periods <= 120
                else Decimal(str(exit_cfg.replacement_premium_over_120))
            )
            held_trailing = self._entry_metrics(cache)[0]
            best_symbol, best_trailing = self._best_replacement_candidate(
                symbol, ctx, premium, held_trailing
            )
            if best_symbol is not None:
                return StrategyDecision(
                    decision_kind=DecisionKind.REPLACE,
                    allowed=True,
                    reason_code=ReasonCode.REPLACEMENT,
                    reason_text=(
                        f"{best_symbol} trailing {best_trailing} 超过持仓 "
                        f"{held_trailing} × premium {premium}"
                    ),
                    metrics={"replacement_symbol": best_symbol, "premium": str(premium)},
                    **common,  # type: ignore[arg-type]
                )

        # 4. 否则 HOLD
        return StrategyDecision(
            decision_kind=DecisionKind.HOLD,
            allowed=True,
            reason_code=ReasonCode.HOLD_OK,
            reason_text="未触发退出/换仓条件，继续持有",
            **common,  # type: ignore[arg-type]
        )

    def _best_replacement_candidate(
        self,
        held_symbol: str,
        ctx: LiveContext,
        premium: Decimal,
        held_trailing: Decimal,
    ) -> tuple[str | None, Decimal | None]:
        """找一个未持仓候选，trailing >= 持仓 trailing × premium。"""
        entry = self.config.strategy.entry
        best: tuple[str, Decimal] | None = None
        for symbol in self.candidate_symbols:
            if symbol == held_symbol or symbol in ctx.held:
                continue
            cache = self.candidates.get(symbol)
            if cache is None or cache.error or len(cache.rates) < entry.lookback_periods:
                continue
            trailing, streak = self._entry_metrics(cache)
            if streak < entry.min_consecutive_positive:
                continue
            if trailing >= held_trailing * premium and (best is None or trailing > best[1]):
                best = (symbol, trailing)
        if best is None:
            return None, None
        return best[0], best[1]


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

    def funding_rates(self, symbol: str, periods: int) -> list[tuple[int, Decimal, Decimal]]:
        from ..data.funding import fetch_funding_history

        interval = self.funding_interval_hours(symbol)
        start_ms = int(self._now() * 1000) - (periods + 5) * interval * 3600 * 1000
        frame = fetch_funding_history(self.client, symbol, start_ms=start_ms)
        rates = frame["funding_rate"]
        marks = frame["mark_price"]
        out: list[tuple[int, Decimal, Decimal]] = []
        for ts, value, mark in zip(rates.index, rates, marks, strict=False):
            import math

            mark_value = float(mark)
            if math.isnan(mark_value):
                mark_value = 0.0
            out.append((int(ts.timestamp() * 1000), Decimal(str(value)), Decimal(str(mark_value))))
        return out

    def quote_volume_3d_avg(self, symbol: str) -> Decimal:
        import math

        from ..data.klines import fetch_historical_quote_volume_3d_avg

        start_ms = int(self._now() * 1000) - 5 * 86_400 * 1000
        series = fetch_historical_quote_volume_3d_avg(self.client, symbol, start_ms=start_ms)
        value = series.iloc[-1]
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            raise ValueError(f"{symbol} 3 天平均日成交额无效")
        return Decimal(str(value))

