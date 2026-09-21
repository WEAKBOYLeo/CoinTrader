"""资金费率 carry 纯策略 evaluator（实施计划书 3.0 T2）。

职责边界（硬约束）：

1. **纯计算**：相同输入与同一时钟输出相同结果；不访问网络、不读数据库、
   不创建订单、不写账本。IO（候选刷新、去重查询、报价获取）全部外置：
   去重经 ``DedupProbe`` 协议注入，时钟经 ``now_fn`` 注入。
2. 策略条件与 ``backtest.engine`` 同口径（滑窗年化、连续正费率、负均值退出、
   最长持仓、换仓 premium），计算公式从 ``live.strategy.LiveStrategy`` 原样迁移，
   不改变任何数值语义。
3. 输出中性 ``CarryEvaluation``（kind/reason 字符串与 ``live.decisions`` 的
   ``DecisionKind``/``ReasonCode`` 常量逐值相等 —— 由
   ``tests/test_strategy_contracts.py`` 断言，本模块刻意不 import
   ``cointrader.live``：``live/__init__`` 会拉起 service→strategy 的循环依赖）。
4. 无未来数据：所有窗口右端都是当前点，只用已结算（已可见）资金费；
   scan epoch 绑定（cutoff 横截面）由调用方经 ``EvalContext`` 传入。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol

from ..config import Config

__all__ = [
    "CandidateInput",
    "CarryEvaluation",
    "DedupProbe",
    "EvalContext",
    "EvalKind",
    "FundingCarryEvaluator",
    "HeldInput",
    "QuoteInput",
    "Reason",
]

_HOURS_PER_YEAR = Decimal(24 * 365)


class EvalKind:
    """评估结果类型。值必须与 ``live.decisions.DecisionKind`` 相等（测试强制）。"""

    OPEN = "OPEN"
    HOLD = "HOLD"
    EXIT = "EXIT"
    REPLACE = "REPLACE"
    SKIP = "SKIP"
    PENDING_QUOTE = "PENDING_QUOTE"


class Reason:
    """原因码。值必须与 ``live.decisions.ReasonCode`` 相等（测试强制）。"""

    RECONCILIATION_BLOCKED = "RECONCILIATION_BLOCKED"
    ACCOUNT_STATE_UNKNOWN = "ACCOUNT_STATE_UNKNOWN"
    EXCLUDED_ASSET = "EXCLUDED_ASSET"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    STALE_DATA = "STALE_DATA"
    TRAILING_RATE_BELOW_THRESHOLD = "TRAILING_RATE_BELOW_THRESHOLD"
    CONSECUTIVE_POSITIVE_TOO_SHORT = "CONSECUTIVE_POSITIVE_TOO_SHORT"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    ALREADY_HELD = "ALREADY_HELD"
    ACTIVE_INTENT = "ACTIVE_INTENT"
    ACTIVE_ORDER = "ACTIVE_ORDER"
    MAX_POSITIONS = "MAX_POSITIONS"
    STALE_QUOTE = "STALE_QUOTE"
    BASIS_DISCOUNT = "BASIS_DISCOUNT"
    INVALID_SYMBOL = "INVALID_SYMBOL"
    NOTIONAL_TOO_SMALL = "NOTIONAL_TOO_SMALL"
    ALREADY_SUBMITTED = "ALREADY_SUBMITTED"
    PENDING_QUOTE = "PENDING_QUOTE"
    SETTLEMENT_LAG = "SETTLEMENT_LAG"
    MARKET_DATA_NOT_READY = "MARKET_DATA_NOT_READY"
    RANKED_OUT = "RANKED_OUT"
    NEGATIVE_EXIT_AVG = "NEGATIVE_EXIT_AVG"
    MAX_HOLDING = "MAX_HOLDING"
    REPLACEMENT = "REPLACEMENT"
    ENTRY_OK = "ENTRY_OK"
    HOLD_OK = "HOLD_OK"


# ---------------------------------------------------------------------------
# 纯输入
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateInput:
    """单个候选的指标横截面（由调用方从低频缓存/epoch 快照转换而来）。"""

    symbol: str
    rates: tuple[Decimal, ...]  # 单期费率，升序
    mark_prices: tuple[Decimal, ...]  # 结算标记价（与 rates 对齐）
    timestamps: tuple[int, ...]  # 结算时间 ms
    interval_hours: int = 8
    volume_3d_avg: Decimal = Decimal("0")
    refreshed_ts_ms: int = 0
    error: str = ""


@dataclass(frozen=True)
class HeldInput:
    """交易所确认的实际持仓（对账/快照来源，不是本地 pair 状态）。"""

    symbol: str
    spot_qty: Decimal
    perp_qty: Decimal  # 负数 = 空头
    opened_ms: int = 0  # 开仓时间（0 = 未知）


@dataclass(frozen=True)
class QuoteInput:
    """一次新鲜报价（提交开仓前必须重新获取）。"""

    spot_price: Decimal
    perp_price: Decimal
    ts_ms: int  # 本地接收时间（UTC ms）
    spot_ts_ms: int = 0  # 现货报价接收时间（0 = 用 ts_ms）
    perp_ts_ms: int = 0  # 永续报价接收时间（0 = 用 ts_ms）


class DedupProbe(Protocol):
    """开仓去重查询（§7.3 三层）。由调用方注入（生产=账本只读方法，测试=fake）。"""

    def active_pair(self, symbol: str) -> bool:
        """存在同 symbol 非终态 pair。"""

    def has_open_intent(self, symbol: str) -> bool:
        """存在同 symbol 未终态 intent。"""

    def has_open_order(self, symbol: str) -> bool:
        """存在同 symbol 未终态 order。"""


@dataclass(frozen=True)
class EvalContext:
    """一轮评估的输入（由 LiveService/LiveStrategy 构造，本身不含 IO）。"""

    now_ms: int
    total_capital: Decimal  # 真实账户资金；0 = 未知 → 禁止开仓
    held: Mapping[str, HeldInput]
    quotes: Mapping[str, QuoteInput]
    submitted_this_run: frozenset[str] = frozenset()
    account_ok: bool = True
    reconcile_ok: bool = True
    # scan epoch 绑定：epoch_active=True 表示调用方接入了 MarketDataSynchronizer；
    # epoch_id=None 且 epoch_active=True → 无 READY epoch（未持仓候选禁止新开仓）
    epoch_active: bool = False
    epoch_id: str | None = None
    decision_cutoff_ms: int = 0
    epoch_excluded: Mapping[str, str] = field(default_factory=dict)
    # 无 READY 时仍需产出决策的候选集合（每个 symbol 恰好一条决策）
    expected_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class CarryEvaluation:
    """单 symbol 评估结果（中性契约；LiveStrategy 映射为 StrategyDecision）。

    策略只输出目标名义额与证据，不含订单参数、不含 API client。
    """

    symbol: str
    kind: str
    allowed: bool
    reason_code: str
    reason_text: str
    requested_notional: Decimal | None = None
    spot_price: Decimal | None = None
    perp_price: Decimal | None = None
    quote_ts_ms: int | None = None
    trailing_annualized: Decimal | None = None
    consecutive_positive_periods: int | None = None
    exit_average_annualized: Decimal | None = None
    position_age_periods: int | None = None
    funding_interval_hours: int | None = None
    quote_volume_3d_avg: Decimal | None = None
    entry_threshold: Decimal | None = None
    exit_threshold: Decimal | None = None
    replacement_symbol: str | None = None
    premium: Decimal | None = None
    metrics: Mapping[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------


class FundingCarryEvaluator:
    """资金费率 carry 策略的纯计算核心。

    Args:
        config: 顶层配置（门槛唯一来源 = strategy 段 + execution.canary_notional）。
        strategy_version: 策略版本标签。
        config_hash: 配置摘要 hash。
        now_fn: 可注入时钟（秒）；数据年龄判定用它，保证「同一输入+时钟=同结果」。
    """

    def __init__(
        self,
        *,
        config: Config,
        strategy_version: str = "funding_carry-1.0",
        config_hash: str = "",
        now_fn: Callable[[], float] = lambda: 0.0,
    ) -> None:
        self.config = config
        self.strategy_version = strategy_version
        self.config_hash = config_hash
        self._now = now_fn

    # -- 指标（纯函数，复用 backtest 语义） ------------------------------------

    def _ctx_now_ms(self) -> int:
        return int(self._now() * 1000)

    @staticmethod
    def _annualized(period_rate: Decimal, interval_hours: int) -> Decimal:
        return period_rate * _HOURS_PER_YEAR / Decimal(interval_hours)

    def entry_metrics(self, candidate: CandidateInput) -> tuple[Decimal, int]:
        """(最近 lookback 期滑动平均的年化, 连续正滑动平均期数)。

        与 ``backtest.engine.build_signals`` 同口径：
        先对原始费率做 lookback 期滑动平均，再按该币自己的周期年化。
        """
        rates = candidate.rates
        lookback = self.config.strategy.entry.lookback_periods
        if len(rates) < lookback:
            return Decimal("0"), 0
        window = rates[-lookback:]
        mean = sum(window, Decimal("0")) / Decimal(lookback)
        annualized = self._annualized(mean, candidate.interval_hours)

        # 连续正：从当前位置往回数「滑动平均为正」的期数（窗口未满的位置不算正）
        streak = 0
        for pos in range(len(rates) - 1, lookback - 2, -1):
            w = rates[pos - lookback + 1 : pos + 1]
            if sum(w, Decimal("0")) / Decimal(lookback) > 0:
                streak += 1
            else:
                break
        return annualized, streak

    def exit_average_annualized(self, candidate: CandidateInput) -> Decimal | None:
        """最近 exit_lookback_periods 期平均费率年化；窗口未满返回 None。"""
        rates = candidate.rates
        window = self.config.strategy.exit.exit_lookback_periods
        if len(rates) < window:
            return None
        mean = sum(rates[-window:], Decimal("0")) / Decimal(window)
        return self._annualized(mean, candidate.interval_hours)

    def _position_age_periods(self, held: HeldInput, interval_hours: int) -> int:
        if held.opened_ms <= 0:
            return 0
        period_ms = Decimal(interval_hours) * 3600 * 1000
        return int((self._ctx_now_ms() - held.opened_ms) // period_ms)

    def _cache_age_ms(self, candidate: CandidateInput) -> int:
        return self._ctx_now_ms() - candidate.refreshed_ts_ms

    # -- 主入口 ---------------------------------------------------------------

    def evaluate(
        self,
        ctx: EvalContext,
        candidate_symbols: Sequence[str],
        candidates: Mapping[str, CandidateInput],
        dedup: DedupProbe,
    ) -> list[CarryEvaluation]:
        """评估全部候选与持仓。每个 symbol 恰好一条决策。

        开仓槽位按 trailing 年化从高到低分配：只放行 top (max_positions -
        已持仓)，落选记 ``RANKED_OUT``（防止单 tick 内 held 未更新导致开超
        上限，也防止低收益币抢先占用槽位）。
        """
        evaluations: list[CarryEvaluation] = []
        held = dict(ctx.held)

        # 0) scan epoch 闸门：无 READY epoch 时未持仓候选统一 SKIP
        # MARKET_DATA_NOT_READY；持仓评估（风险降低）不被阻止。
        if ctx.epoch_active and ctx.epoch_id is None:
            for symbol in sorted(held):
                evaluations.append(self.exit_eval(symbol, held[symbol], ctx, candidates))
            for symbol in ctx.expected_symbols:
                if symbol in held:
                    continue
                evaluations.append(
                    self._skip(
                        symbol,
                        Reason.MARKET_DATA_NOT_READY,
                        "无 READY scan epoch（市场数据不完整），禁止新开仓",
                    )
                )
            return evaluations

        # 1) 持仓 symbol：退出/换仓评估（异常退出路径由 service 优先处理）
        for symbol in sorted(held):
            evaluations.append(self.exit_eval(symbol, held[symbol], ctx, candidates))

        # 2) 未持仓候选：开仓评估；通过前置门槛的按收益率排序分配槽位
        pending: list[tuple[Decimal, str, CarryEvaluation]] = []
        for symbol in candidate_symbols:
            if symbol in held:
                continue
            evaluation = self.can_open(symbol, ctx, candidates, dedup)
            if evaluation.kind == EvalKind.PENDING_QUOTE:
                candidate = candidates[symbol]
                trailing, _ = self.entry_metrics(candidate)
                pending.append((trailing, symbol, evaluation))
            else:
                evaluations.append(evaluation)

        slots = max(int(self.config.strategy.selection.max_positions) - len(held), 0)
        pending.sort(key=lambda item: (-item[0], item[1]))
        for rank, (trailing, symbol, evaluation) in enumerate(pending):
            if rank < slots:
                evaluations.append(evaluation)
                continue
            evaluations.append(
                self._skip(
                    symbol,
                    Reason.RANKED_OUT,
                    f"本轮通过门槛 {len(pending)} 个，槽位 {slots} 个；"
                    f"{symbol} 年化 {trailing} 排名第 {rank + 1}，未进 top {slots}",
                    trailing_annualized=trailing,
                )
            )

        return evaluations

    # -- 开仓（§7.2 判断顺序，尽早拒绝并记录原因） -----------------------------

    def _skip(
        self,
        symbol: str,
        code: str,
        text: str,
        *,
        kind: str = EvalKind.SKIP,
        **fields: object,
    ) -> CarryEvaluation:
        base: dict[str, object] = {
            "symbol": symbol,
            "kind": kind,
            "allowed": False,
            "reason_code": code,
            "reason_text": text,
        }
        base.update(fields)
        return CarryEvaluation(**base)  # type: ignore[arg-type]

    def can_open(
        self,
        symbol: str,
        ctx: EvalContext,
        candidates: Mapping[str, CandidateInput],
        dedup: DedupProbe,
    ) -> CarryEvaluation:
        """统一开仓判断入口（§7.3）。返回 SKIP/OPEN/PENDING_QUOTE，本身不下单。"""
        entry = self.config.strategy.entry
        selection = self.config.strategy.selection
        base = symbol.replace("USDT", "")
        candidate = candidates.get(symbol)
        held = dict(ctx.held)

        # 前置状态（service 也已把关，这里记录原因保证决策链完整）
        if not ctx.reconcile_ok:
            return self._skip(symbol, Reason.RECONCILIATION_BLOCKED, "最近对账未通过，禁止新增风险")
        if not ctx.account_ok or ctx.total_capital <= 0:
            return self._skip(
                symbol, Reason.ACCOUNT_STATE_UNKNOWN, "账户快照缺失/过期/资金为零，禁止开仓"
            )

        # scan epoch 闸门：只允许对 READY 同一 cutoff 横截面内的候选开仓
        if ctx.epoch_active:
            if ctx.epoch_id is None:
                return self._skip(
                    symbol, Reason.MARKET_DATA_NOT_READY, "无 READY scan epoch，禁止新开仓"
                )
            if symbol in ctx.epoch_excluded:
                return self._skip(
                    symbol,
                    Reason.EXCLUDED_ASSET,
                    f"{symbol} 在 epoch {ctx.epoch_id} 中确定性排除: {ctx.epoch_excluded[symbol]}",
                )
            if symbol not in candidates:
                return self._skip(
                    symbol,
                    Reason.MARKET_DATA_NOT_READY,
                    f"{symbol} 不在 READY epoch {ctx.epoch_id} 的已完成快照内",
                )

        # 5. 排除列表
        if base.upper() in {b.upper() for b in selection.exclude_bases}:
            return self._skip(symbol, Reason.EXCLUDED_ASSET, f"{base} 在排除列表中")
        if not symbol.endswith("USDT"):
            return self._skip(symbol, Reason.INVALID_SYMBOL, f"{symbol} 不是 USDT 永续对")

        # 6/7/8/9. 数据窗口与策略门槛
        if candidate is None:
            return self._skip(symbol, Reason.INSUFFICIENT_HISTORY, "候选指标缓存不存在（刷新失败）")
        if candidate.error:
            return self._skip(symbol, Reason.STALE_DATA, f"候选指标缓存异常: {candidate.error}")
        # 数据年龄上限 = 该币结算周期 + 宽限（两次结算间费率不变，超龄=刷新掉链）
        stale_after_ms = (
            int(candidate.interval_hours) * 3600 * 1000
            + int(self.config.execution.max_candidate_data_age_seconds * 1000)
        )
        if self._cache_age_ms(candidate) > stale_after_ms:
            return self._skip(
                symbol,
                Reason.STALE_DATA,
                f"候选指标数据年龄 {self._cache_age_ms(candidate)}ms 超过 {stale_after_ms}ms",
            )
        # 结算滞后门：最新一期结算已发生但缓存未刷新 → 本轮禁止开仓。
        if candidate.timestamps:
            last_settle_ms = candidate.timestamps[-1] + int(candidate.interval_hours) * 3600 * 1000
            if last_settle_ms <= ctx.now_ms:
                return self._skip(
                    symbol,
                    Reason.SETTLEMENT_LAG,
                    f"最新结算（{last_settle_ms}）已发生但缓存未刷新，等待数据补齐",
                )
        if len(candidate.rates) < entry.lookback_periods:
            return self._skip(
                symbol,
                Reason.INSUFFICIENT_HISTORY,
                f"资金费历史 {len(candidate.rates)} 期 < 入场窗口 {entry.lookback_periods} 期",
            )

        trailing, streak = self.entry_metrics(candidate)
        min_trailing = Decimal(str(entry.min_trailing_annualized))
        min_rate = Decimal(str(entry.min_annualized_rate))
        if trailing < max(min_trailing, min_rate):
            return self._skip(
                symbol,
                Reason.TRAILING_RATE_BELOW_THRESHOLD,
                f"最近 {entry.lookback_periods} 期滑动平均年化 {trailing} 低于门槛 "
                f"{max(min_trailing, min_rate)}",
                trailing_annualized=trailing,
                consecutive_positive_periods=streak,
                funding_interval_hours=candidate.interval_hours,
            )
        if streak < entry.min_consecutive_positive:
            return self._skip(
                symbol,
                Reason.CONSECUTIVE_POSITIVE_TOO_SHORT,
                f"连续正滑动平均 {streak} 期 < 要求 {entry.min_consecutive_positive} 期",
                trailing_annualized=trailing,
                consecutive_positive_periods=streak,
                funding_interval_hours=candidate.interval_hours,
            )
        min_volume = Decimal(str(selection.min_quote_volume_3d_avg))
        if candidate.volume_3d_avg < min_volume:
            return self._skip(
                symbol,
                Reason.LOW_LIQUIDITY,
                f"3 天平均日成交额 {candidate.volume_3d_avg} < 门槛 {min_volume}",
                quote_volume_3d_avg=candidate.volume_3d_avg,
            )

        # 10-12. 去重（§7.3 三层 + 本轮已提交 + 最大持仓数）
        if symbol in held:
            return self._skip(symbol, Reason.ALREADY_HELD, "交易所存在实际持仓")
        if symbol in ctx.submitted_this_run:
            return self._skip(symbol, Reason.ALREADY_SUBMITTED, "本轮已提交相同 symbol 的开仓")
        if dedup.active_pair(symbol):
            return self._skip(symbol, Reason.ACTIVE_ORDER, "存在同 symbol 非终态 pair")
        if dedup.has_open_intent(symbol):
            return self._skip(symbol, Reason.ACTIVE_INTENT, "存在同 symbol 未终态 intent")
        if dedup.has_open_order(symbol):
            return self._skip(symbol, Reason.ACTIVE_ORDER, "存在同 symbol 未终态 order")
        if len(held) >= selection.max_positions:
            return self._skip(
                symbol,
                Reason.MAX_POSITIONS,
                f"当前持仓 {len(held)} 已达上限 {selection.max_positions}",
            )

        # 13. 新鲜报价（动态候选池：上下文未含该 symbol 报价 → PENDING 中间态）
        quote = ctx.quotes.get(symbol)
        if quote is None:
            return self._skip(
                symbol,
                Reason.PENDING_QUOTE,
                "前置门槛通过，等待新鲜报价（service 按需获取）",
                kind=EvalKind.PENDING_QUOTE,
            )
        return self._open_with_quote(symbol, ctx, quote, candidate, trailing, streak)

    def complete_open(
        self,
        symbol: str,
        ctx: EvalContext,
        candidates: Mapping[str, CandidateInput],
        quote: QuoteInput | None,
    ) -> CarryEvaluation:
        """第二阶段：调用方获取报价后定案（步骤 13-15）。获取失败 = STALE_QUOTE。"""
        if quote is None:
            return self._skip(symbol, Reason.STALE_QUOTE, "无新鲜报价（本轮获取失败）")
        candidate = candidates.get(symbol)
        if candidate is None:
            return self._skip(symbol, Reason.INSUFFICIENT_HISTORY, "候选指标缓存缺失（前置检查后异常）")
        trailing, streak = self.entry_metrics(candidate)
        return self._open_with_quote(symbol, ctx, quote, candidate, trailing, streak)

    def _open_with_quote(
        self,
        symbol: str,
        ctx: EvalContext,
        quote: QuoteInput,
        candidate: CandidateInput,
        trailing: Decimal,
        streak: int,
    ) -> CarryEvaluation:
        """开仓第二阶段（§7.2 步骤 13-15）：报价新鲜度 → 名义额/基差。

        与 legacy ``build_signal`` 同一本地拦截口径；策略只输出目标名义额，
        订单参数由 execution 层生成。
        """
        selection = self.config.strategy.selection
        entry = self.config.strategy.entry
        min_trailing = Decimal(str(entry.min_trailing_annualized))
        max_quote_age_ms = int(self.config.execution.max_market_data_age_seconds * 1000)
        age = ctx.now_ms - quote.ts_ms
        if age < 0 or age > max_quote_age_ms:
            return self._skip(symbol, Reason.STALE_QUOTE, f"报价年龄 {age}ms > {max_quote_age_ms}ms")
        if quote.spot_price <= 0 or quote.perp_price <= 0:
            return self._skip(symbol, Reason.STALE_QUOTE, "报价非正数")

        # 两市场报价接收时间偏差（AC-05：未同步 Spot/Futures 价格不得生成交易意图）
        spot_ts = quote.spot_ts_ms or quote.ts_ms
        perp_ts = quote.perp_ts_ms or quote.ts_ms
        skew = abs(spot_ts - perp_ts)
        max_skew_ms = int(self.config.execution.max_quote_skew_ms)
        if skew > max_skew_ms:
            return self._skip(
                symbol,
                Reason.STALE_QUOTE,
                f"Spot/Futures 报价接收时间偏差 {skew}ms > {max_skew_ms}ms（两市场不同步）",
                spot_price=quote.spot_price,
                perp_price=quote.perp_price,
                quote_ts_ms=quote.ts_ms,
            )

        # 14. 名义额与意图前市场检查（名义额 = 资本 × 权重，canary 上限截断）
        requested = (ctx.total_capital * Decimal(str(selection.per_position_weight))).quantize(
            Decimal("0.01")
        )
        if requested <= 0:
            return self._skip(symbol, Reason.NOTIONAL_TOO_SMALL, "目标名义额非正")
        basis = (quote.perp_price - quote.spot_price) / quote.spot_price
        if basis < 0 and abs(basis) > Decimal(str(self.config.execution.hedge_tolerance_pct)):
            return self._skip(
                symbol,
                Reason.BASIS_DISCOUNT,
                f"永续深度贴水 {basis}，开空头不利，等待收敛",
                spot_price=quote.spot_price,
                perp_price=quote.perp_price,
                quote_ts_ms=quote.ts_ms,
            )
        cap = Decimal(str(self.config.execution.canary_notional))
        target = min(requested, cap)

        # 15. 通过 → OPEN（intent 与下单由 service 执行）
        return CarryEvaluation(
            symbol=symbol,
            kind=EvalKind.OPEN,
            allowed=True,
            reason_code=Reason.ENTRY_OK,
            reason_text="策略条件全部满足，允许开仓",
            requested_notional=target,
            spot_price=quote.spot_price,
            perp_price=quote.perp_price,
            quote_ts_ms=quote.ts_ms,
            trailing_annualized=trailing,
            consecutive_positive_periods=streak,
            funding_interval_hours=candidate.interval_hours,
            quote_volume_3d_avg=candidate.volume_3d_avg,
            entry_threshold=min_trailing,
            metrics={
                "quote_source": "public",
                "requested_notional_raw": str(requested),
            },
        )

    # -- 退出 / 换仓（§7.4） ----------------------------------------------------

    def exit_eval(
        self,
        symbol: str,
        held: HeldInput,
        ctx: EvalContext,
        candidates: Mapping[str, CandidateInput],
        candidate_symbols: Sequence[str] | None = None,
    ) -> CarryEvaluation:
        """单持仓退出评估：负资金费 → 最长持仓 → 换仓 → HOLD。"""
        exit_cfg = self.config.strategy.exit
        candidate = candidates.get(symbol)
        interval = candidate.interval_hours if candidate else 8
        age_periods = self._position_age_periods(held, interval)
        exit_avg = self.exit_average_annualized(candidate) if candidate else None

        common: dict[str, object] = {
            "position_age_periods": age_periods,
            "exit_average_annualized": exit_avg,
            "exit_threshold": Decimal("0"),
            "funding_interval_hours": interval,
        }
        if candidate is not None:
            trailing, _ = self.entry_metrics(candidate)
            common["trailing_annualized"] = trailing
            common["quote_volume_3d_avg"] = candidate.volume_3d_avg

        # 1. 负资金费退出（最近 exit_lookback 期均值转负；窗口未满不判定）
        if exit_avg is not None and exit_avg < 0:
            return CarryEvaluation(
                symbol=symbol,
                kind=EvalKind.EXIT,
                allowed=True,
                reason_code=Reason.NEGATIVE_EXIT_AVG,
                reason_text=(
                    f"最近 {exit_cfg.exit_lookback_periods} 期平均资金费率年化 {exit_avg} 转负，策略退出"
                ),
                **common,  # type: ignore[arg-type]
            )

        # 2. 最长持仓（结算周期数）
        if age_periods >= exit_cfg.max_holding_periods:
            return CarryEvaluation(
                symbol=symbol,
                kind=EvalKind.EXIT,
                allowed=True,
                reason_code=Reason.MAX_HOLDING,
                reason_text=f"持仓 {age_periods} 期达到最长 {exit_cfg.max_holding_periods} 期，强制退出",
                **common,  # type: ignore[arg-type]
            )

        # 3. 换仓：新候选相对优势超过按年龄分段的 premium（先平旧，后开新）
        if candidate is not None and age_periods > 60:
            premium = (
                Decimal(str(exit_cfg.replacement_premium_under_120))
                if age_periods <= 120
                else Decimal(str(exit_cfg.replacement_premium_over_120))
            )
            held_trailing = self.entry_metrics(candidate)[0]
            best_symbol, best_trailing = self._best_replacement_candidate(
                symbol,
                ctx,
                premium,
                held_trailing,
                candidate_symbols if candidate_symbols is not None else tuple(candidates),
                candidates,
            )
            if best_symbol is not None:
                return CarryEvaluation(
                    symbol=symbol,
                    kind=EvalKind.REPLACE,
                    allowed=True,
                    reason_code=Reason.REPLACEMENT,
                    reason_text=(
                        f"{best_symbol} trailing {best_trailing} 超过持仓 "
                        f"{held_trailing} × premium {premium}"
                    ),
                    replacement_symbol=best_symbol,
                    premium=premium,
                    **common,  # type: ignore[arg-type]
                )

        # 4. 否则 HOLD
        return CarryEvaluation(
            symbol=symbol,
            kind=EvalKind.HOLD,
            allowed=True,
            reason_code=Reason.HOLD_OK,
            reason_text="未触发退出/换仓条件，继续持有",
            **common,  # type: ignore[arg-type]
        )

    def _best_replacement_candidate(
        self,
        held_symbol: str,
        ctx: EvalContext,
        premium: Decimal,
        held_trailing: Decimal,
        candidate_symbols: Sequence[str],
        candidates: Mapping[str, CandidateInput],
    ) -> tuple[str | None, Decimal | None]:
        """找一个未持仓候选，trailing >= 持仓 trailing × premium。

        无 READY epoch 时不发策略性换仓（数据不足；风险降低型退出不受影响）。
        """
        entry = self.config.strategy.entry
        if ctx.epoch_active and ctx.epoch_id is None:
            return None, None
        best: tuple[str, Decimal] | None = None
        for symbol in candidate_symbols:
            if symbol == held_symbol or symbol in ctx.held:
                continue
            candidate = candidates.get(symbol)
            if candidate is None or candidate.error or len(candidate.rates) < entry.lookback_periods:
                continue
            trailing, streak = self.entry_metrics(candidate)
            if streak < entry.min_consecutive_positive:
                continue
            if trailing >= held_trailing * premium and (best is None or trailing > best[1]):
                best = (symbol, trailing)
        if best is None:
            return None, None
        return best[0], best[1]
