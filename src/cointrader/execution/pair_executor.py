"""双腿执行器 —— 开仓/平仓状态机、UNKNOWN 恢复、补偿（开发设计文档 §6）。

执行规则（硬约束）：

1. **先开永续，后开现货**。永续腿更容易失败，让它先失败可以避免裸多头。
2. **Spot 数量取 Futures 实际 ``executedQty``** 再按 Spot 规则向下归一化。
3. **下单重试上限为零**。超时/5xx → ``UNKNOWN_SUBMISSION`` →
   按 clientOrderId 查询一次 → 确认已成交则继续、确认未成交则停止并告警。
4. 另一腿失败 → 只执行**减风险**补偿；无法补偿 → HALT + CRITICAL 告警。
5. 对冲完成判定靠两腿终态 + 实际成交量 + 持仓一致性，不靠「两个 200」。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any, cast

from ..errors import BinanceError, OrderRejected, UnknownSubmission
from .futures import FuturesAdapter
from .guard import AuditLog, Market, OrderSide, OrderType
from .metrics import Metrics
from .models import (
    Fill,
    Order,
    OrderIntent,
    OrderRequest,
    PairExecution,
    PairStatus,
    new_client_order_id,
    new_id,
)
from .risk import RiskState
from .risk_gate import RiskGate
from .rules import RuleError, check_notional, floor_to_step, format_decimal
from .spot import SpotAdapter
from .store import StateStore

logger = logging.getLogger(__name__)

__all__ = ["PairExecutor", "PairPrecheckFailed"]


class PairPrecheckFailed(Exception):
    """开仓前检查未通过（未提交任何订单）。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _now_ms() -> int:
    return int(time.time() * 1000)


class PairExecutor:
    """资金费套利的双腿执行器。"""

    def __init__(
        self,
        *,
        spot: SpotAdapter,
        futures: FuturesAdapter,
        store: StateStore,
        gate: RiskGate,
        metrics: Metrics | None = None,
        audit: AuditLog | None = None,
        on_alert: Callable[[str, str], None] | None = None,
        strategy_version: str = "funding_carry-1.0",
        max_market_data_age_ms: int = 5000,
        max_leg_slippage_pct: Decimal = Decimal("0.001"),
        hedge_tolerance_pct: Decimal = Decimal("0.005"),
        order_ack_timeout_seconds: float = 3.0,
        poll_interval_seconds: float = 0.25,
        now_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.spot = spot
        self.futures = futures
        self.store = store
        self.gate = gate
        self.metrics = metrics or Metrics()
        self.audit = audit
        self.on_alert = on_alert or (lambda kind, msg: logger.critical("【告警】%s: %s", kind, msg))
        self.strategy_version = strategy_version
        self.max_market_data_age_ms = max_market_data_age_ms
        self.max_leg_slippage_pct = max_leg_slippage_pct
        self.hedge_tolerance_pct = hedge_tolerance_pct
        self.order_ack_timeout_seconds = order_ack_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._now = now_fn
        self._sleep = sleep_fn

    # ------------------------------------------------------------------
    # 开仓
    # ------------------------------------------------------------------

    def open_pair(
        self,
        symbol: str,
        target_notional: Decimal,
        *,
        spot_price: Decimal,
        perp_price: Decimal,
        quote_ts_ms: int,
        state: RiskState,
        reason: str = "",
        run_id: str = "",
        signal_decision_id: str = "",
        decision_ts_ms: int = 0,
    ) -> PairExecution:
        """开一对对冲头寸（永续空 + 现货多）。返回终态 PairExecution。"""
        pair = PairExecution(
            pair_execution_id=new_id("pair"),
            symbol=symbol,
            target_notional=target_notional,
            strategy_version=self.strategy_version,
            kind="open",
            created_ms=_now_ms(),
            updated_ms=_now_ms(),
            run_id=run_id,
            open_or_close_reason=reason,
            signal_decision_id=signal_decision_id,
            signal_ts_ms=quote_ts_ms,
            decision_ts_ms=decision_ts_ms,
        )

        pair.touches(PairStatus.PRECHECK.value)
        try:
            self._precheck_open(pair, symbol, target_notional, spot_price, quote_ts_ms, state, reason)
        except PairPrecheckFailed as exc:
            pair.touches(PairStatus.FAILED.value, error=exc.reason)
            self._persist(pair)
            return pair

        pair.touches(PairStatus.SUBMIT_PERP.value)
        pair.submit_ts_ms = _now_ms()
        perp_order = self._submit_perp_open(pair, perp_price)
        if perp_order is None:
            self._persist(pair)
            return pair
        executed = perp_order.executed_qty
        if executed <= 0:
            pair.touches(PairStatus.FAILED.value, error="永续腿无成交（撤单后执行量为 0）")
            self._persist(pair)
            return pair
        pair.touches(PairStatus.PERP_FILLED.value)
        self._check_leg_slippage(perp_order, perp_price, "perp")

        pair.touches(PairStatus.SUBMIT_SPOT.value)
        spot_order = self._submit_spot_leg(pair, executed, spot_price)
        if spot_order is None:
            self._persist(pair)
            return pair
        pair.touches(PairStatus.SPOT_FILLED.value)
        self._check_leg_slippage(spot_order, spot_price, "spot")

        spot_exec = spot_order.executed_qty
        # 残量 = 现货目标量 - 现货实际成交量。
        # 不能用永续成交量算：若 Spot 归一化已 reduce 过 dust，再按 executed 算会重复
        # reduce，把永续超平成反向裸头寸。
        residual = spot_order.quantity - spot_exec
        if residual > 0:
            residual = self._reduce_perp_residual(pair, residual)
        pair.residual_qty = residual

        pair.touches(PairStatus.HEDGE_VERIFIED.value)
        if pair.perp is not None and pair.spot is not None:
            pair.actual_perp_notional = (
                pair.perp.executed_qty * pair.perp.avg_price if pair.perp.avg_price else None
            )
            pair.actual_spot_notional = (
                pair.spot.executed_qty * pair.spot.avg_price if pair.spot.avg_price else None
            )
        verified, error = self._verify_hedge(pair)
        if verified:
            pair.touches(PairStatus.COMPLETE.value)
            pair.completed_ts_ms = _now_ms()
        else:
            pair.touches(PairStatus.COMPENSATED.value, error=error or "")
            pair.completed_ts_ms = _now_ms()
        self._sync_fills(symbol, pair.created_ms)
        self._persist(pair)
        return pair

    def _precheck_open(
        self,
        pair: PairExecution,
        symbol: str,
        target_notional: Decimal,
        spot_price: Decimal,
        quote_ts_ms: int,
        state: RiskState,
        reason: str,
    ) -> None:
        decision = self.gate.allow_open(symbol, float(target_notional), state)
        if not decision.allowed:
            raise PairPrecheckFailed(f"风控闸门拒绝: {decision.reason}")

        age_ms = _now_ms() - quote_ts_ms
        if age_ms > self.max_market_data_age_ms:
            raise PairPrecheckFailed(f"行情数据过期: {age_ms}ms > {self.max_market_data_age_ms}ms")

        for label, adapter in (("spot", self.spot), ("perp", self.futures)):
            try:
                rule = adapter.rule(symbol)
            except KeyError as exc:
                raise PairPrecheckFailed(str(exc)) from exc
            if rule.status != "TRADING":
                raise PairPrecheckFailed(f"{label} 规则状态非 TRADING: {rule.status}")
            if target_notional < rule.min_notional:
                raise PairPrecheckFailed(f"{label} 目标名义额 {target_notional} 低于最小 {rule.min_notional}")

        if self.store.active_pair_for_symbol(symbol):
            raise PairPrecheckFailed(f"{symbol} 已有未完结 pair，禁止重复开仓")

        intent = OrderIntent(
            symbol=symbol,
            side=OrderSide.SELL,
            market=Market.PERP,
            order_type=OrderType.MARKET,
            quantity=target_notional / Decimal(str(1)),  # 名义额/参考价在 intent 里仅作参考
            price=spot_price,
            reason=reason,
            strategy_version=self.strategy_version,
            signal_time_ms=quote_ts_ms,
            pair_execution_id=pair.pair_execution_id,
        )
        if not self.store.record_intent(intent):
            raise PairPrecheckFailed(f"重复 intent {intent.intent_id}")
        pair.intent_id = intent.intent_id
        self._persist(pair)

    def _submit_perp_open(self, pair: PairExecution, perp_price: Decimal) -> Order | None:
        rule = self.futures.rule(pair.symbol)
        qty = floor_to_step(pair.target_notional / perp_price, rule.step_size)
        if qty <= 0:
            pair.touches(PairStatus.FAILED.value, error="归一化后永续数量为 0")
            self._halt(f"{pair.symbol} 开仓数量归一化为 0")
            return None
        check_notional(qty, perp_price, rule)

        req = OrderRequest(
            client_order_id=new_client_order_id(self.strategy_version, pair.pair_execution_id, "perp"),
            symbol=pair.symbol,
            side=OrderSide.SELL,
            market=Market.PERP,
            order_type=OrderType.MARKET,
            quantity=qty,
            price=perp_price,
        )
        pair.perp_request = req
        pair.perp = self._local_order(req, "NEW", pair)
        self.store.upsert_order(pair.perp)
        self._persist(pair)

        order = self._submit_with_recovery(self.futures, req, pair)
        if order is None:
            return None
        pair.perp = order
        self.store.upsert_order(order)

        order = self._wait_terminal(self.futures, order)
        if order is None:
            pair.touches(PairStatus.FAILED.value, error="永续腿终态未确认，已停止并停机")
            self._halt(f"{pair.symbol} 永续腿状态未确认")
            return None
        pair.perp = order
        pair.exchange_confirmed_ts_ms = _now_ms()
        self.store.upsert_order(order)
        self._persist(pair)
        return order

    def _submit_spot_leg(self, pair: PairExecution, perp_executed: Decimal, spot_price: Decimal) -> Order | None:
        rule = self.spot.rule(pair.symbol)
        spot_qty = floor_to_step(perp_executed, rule.step_size)
        try:
            if spot_qty <= 0:
                raise RuleError(f"现货归一化数量为 0（perp 成交 {perp_executed}）")
            check_notional(spot_qty, spot_price, rule)
        except RuleError as exc:
            return self._compensate_flatten_perp(pair, perp_executed, reason=str(exc))

        # 超出容差的残量先 reduce 掉
        if perp_executed > spot_qty:
            dust = perp_executed - spot_qty
            if dust / perp_executed > self.hedge_tolerance_pct:
                left = self._reduce_perp_residual(pair, dust)
                if left > 0:
                    pair.touches(PairStatus.FAILED.value, error=f"永续残量 {dust} 无法完全 reduce（剩 {left}）")
                    self._halt(f"{pair.symbol} 永续残量 reduce 失败")
                    return None

        req = OrderRequest(
            client_order_id=new_client_order_id(self.strategy_version, pair.pair_execution_id, "spot"),
            symbol=pair.symbol,
            side=OrderSide.BUY,
            market=Market.SPOT,
            order_type=OrderType.MARKET,
            quantity=spot_qty,
            price=spot_price,
        )
        pair.spot_request = req
        pair.spot = self._local_order(req, "NEW", pair)
        self.store.upsert_order(pair.spot)
        self._persist(pair)

        order = self._submit_with_recovery(self.spot, req, pair)
        if order is None:
            if pair.status == PairStatus.UNKNOWN_SUBMISSION.value or pair.status == PairStatus.SUBMIT_SPOT.value:
                return self._compensate_flatten_perp(
                    pair, pair.perp.executed_qty if pair.perp else perp_executed,
                    reason=f"现货腿失败: {pair.error or '未知'}",
                )
            # 现货腿被明确拒绝（非未知状态）且永续已成交 → 必须平掉永续，
            # 不能留下裸空腿（文档 §6.1：另一腿失败 → COMPENSATE_OR_FLATTEN）
            if pair.perp is not None and pair.perp.executed_qty > 0:
                return self._compensate_flatten_perp(
                    pair, pair.perp.executed_qty,
                    reason=f"现货腿失败: {pair.error or '被拒绝'}",
                )
            return None
        pair.spot = order
        self.store.upsert_order(order)

        order = self._wait_terminal(self.spot, order)
        if order is None:
            return self._compensate_flatten_perp(pair, perp_executed, reason="现货腿等待终态超时")
        pair.spot = order
        self.store.upsert_order(order)
        self._persist(pair)
        return order

    # ------------------------------------------------------------------
    # 平仓
    # ------------------------------------------------------------------

    def close_pair(
        self, symbol: str, *, reason: str = "", run_id: str = "",
        signal_decision_id: str = "", decision_ts_ms: int = 0,
    ) -> PairExecution:
        """按交易所实际数量平仓（先现货后永续 reduce-only）。"""
        pair = PairExecution(
            pair_execution_id=new_id("close"),
            symbol=symbol,
            target_notional=Decimal("0"),
            strategy_version=self.strategy_version,
            kind="close",
            created_ms=_now_ms(),
            updated_ms=_now_ms(),
            run_id=run_id,
            open_or_close_reason=reason,
            signal_decision_id=signal_decision_id,
            decision_ts_ms=decision_ts_ms,
        )
        pair.touches(PairStatus.PRECHECK_CLOSE.value)

        decision = self.gate.allow_reduce()
        if not decision.allowed:
            pair.touches(PairStatus.FAILED.value, error=decision.reason)
            self._persist(pair)
            return pair

        try:
            spot_qty = self.spot.balances().get(symbol.replace("USDT", ""), Decimal("0"))
        except Exception as exc:  # noqa: BLE001
            self._fail_close(pair, f"现货余额查询失败: {exc}")
            return pair
        try:
            perp_qty = self.futures.position_qty(symbol)
        except Exception as exc:  # noqa: BLE001
            self._fail_close(pair, f"永续持仓查询失败: {exc}")
            return pair

        if spot_qty <= 0 and abs(perp_qty) <= 0:
            pair.touches(PairStatus.COMPLETE.value)
            pair.completed_ts_ms = _now_ms()
            self._persist(pair)
            return pair

        pair.touches(PairStatus.FILL_TRACKING.value)
        perp_short = -perp_qty if perp_qty < 0 else Decimal("0")

        if spot_qty > 0:
            spot_rule = self.spot.rule(symbol)
            # 现货平仓用 MARKET 单，数量必须对齐 LOT_SIZE step。
            # 注意：不能用 normalize_qty(is_market=True) —— 币安 MARKET_LOT_SIZE
            # stepSize 为 0，会报「step 必须为正」；且开仓成交含手续费扣减后
            # 余额本身不保证对齐 step（如 0.0007 扣费后 0.00069930），
            # 直接原样卖出会被 -1013 LOT_SIZE 拒绝（实盘 demo 实测）。
            sell_qty = floor_to_step(spot_qty, spot_rule.step_size)
            if sell_qty < spot_rule.min_qty:
                logger.warning(
                    "【平仓】%s 现货余额 %s 对齐 step 后 %s 低于最小数量 %s，"
                    "不可卖出，按灰尘处理",
                    symbol, format_decimal(spot_qty), format_decimal(sell_qty),
                    format_decimal(spot_rule.min_qty),
                )
                sell_qty = Decimal("0")
            if sell_qty > 0:
                req = OrderRequest(
                    client_order_id=new_client_order_id(self.strategy_version, pair.pair_execution_id, "spot"),
                    symbol=symbol,
                    side=OrderSide.SELL,
                    market=Market.SPOT,
                    order_type=OrderType.MARKET,
                    quantity=sell_qty,
                )
                order = self._submit_with_recovery(self.spot, req, pair)
                pair.spot = order
                if order is None:
                    self._fail_close(pair, f"现货平仓失败: {pair.error}")
                    return pair
                self.store.upsert_order(order)

        if perp_short > 0:
            cid = new_client_order_id(self.strategy_version, pair.pair_execution_id, "perp")
            try:
                order = self.futures.reduce_only_close(symbol, perp_short, client_order_id=cid)
            except UnknownSubmission:
                order = self._recover_once(self.futures, symbol, cid)
                if order is None:
                    self._fail_close(pair, f"永续平仓结果未知且查询失败，剩 {perp_short}，转人工")
                    return pair
            except (OrderRejected, BinanceError) as exc:
                self._fail_close(pair, f"永续平仓被拒: {exc}，剩 {perp_short}，转人工")
                return pair
            pair.perp = order
            self.store.upsert_order(order)

        pair.touches(PairStatus.RESIDUAL_CHECK.value)
        residual_spot = self._spot_balance(symbol)
        residual_perp = self.futures.position_qty(symbol)
        # 灰尘容忍：低于一个 step 的余额无法再下最小单（step 以下不可卖），
        # 不触发停机；达到一个 step 以上的残量才是真事故。
        spot_step = self.spot.rule(symbol).step_size
        perp_step = self.futures.rule(symbol).step_size
        spot_dust = Decimal("0") < residual_spot < spot_step
        perp_dust = Decimal("0") < abs(residual_perp) < perp_step
        if (residual_spot > 0 and not spot_dust) or (abs(residual_perp) > 0 and not perp_dust):
            pair.touches(PairStatus.COMPENSATE_RESIDUAL.value,
                         error=f"平仓后残量 spot={residual_spot} perp={residual_perp}")
            self.on_alert("RESIDUAL_AFTER_CLOSE",
                          f"{symbol} 平仓后残留 spot={residual_spot} perp={residual_perp}，需人工处理")
            self.gate.halt(f"{symbol} 平仓后存在残量")
            self._persist(pair)
            return pair
        if spot_dust or perp_dust:
            logger.warning(
                "【平仓】%s 平仓后留下低于 step 的灰尘 spot=%s perp=%s（不可再卖，忽略）",
                symbol, format_decimal(residual_spot), format_decimal(residual_perp),
            )

        pair.touches(PairStatus.COMPLETE.value)
        pair.completed_ts_ms = _now_ms()
        self._sync_fills(symbol, pair.created_ms)
        self._persist(pair)
        return pair

    def _spot_balance(self, symbol: str) -> Decimal:
        try:
            return self.spot.balances().get(symbol.replace("USDT", ""), Decimal("0"))
        except Exception:  # noqa: BLE001
            return Decimal("0")

    def _fail_close(self, pair: PairExecution, error: str) -> None:
        pair.touches(PairStatus.FAILED.value, error=error)
        self.on_alert("CLOSE_FAILED", f"{pair.symbol}: {error}")
        self._halt(f"平仓失败: {error}")
        self._persist(pair)

    # ------------------------------------------------------------------
    # 提交与恢复
    # ------------------------------------------------------------------

    def _local_order(self, req: OrderRequest, state: str, pair: PairExecution | None = None) -> Order:
        order = Order(
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            market=req.market,
            side=req.side,
            order_type=req.order_type,
            quantity=req.quantity,
            price=req.price,
            reduce_only=req.reduce_only,
            state=state,
            updated_ms=_now_ms(),
            submit_ts_ms=_now_ms(),
            ack_ts_ms=_now_ms(),
        )
        self._stamp_pair(order, pair)
        return order

    @staticmethod
    def _stamp_pair(order: Order, pair: PairExecution | None) -> None:
        """把关联链字段（run/pair/intent）从 pair 同步到订单。"""
        if pair is None:
            return
        order.run_id = pair.run_id
        order.pair_execution_id = pair.pair_execution_id
        order.intent_id = pair.intent_id
        if order.terminal_ts_ms is None and order.is_terminal:
            order.terminal_ts_ms = order.updated_ms or _now_ms()
        if order.terminal_ts_ms is None and order.is_terminal and order.state in (
            "REJECTED", "CANCELED", "EXPIRED"
        ):
            order.failure_class = order.state
        if order.is_terminal and order.failure_class is None and order.state in (
            "REJECTED", "CANCELED", "EXPIRED"
        ) and order.executed_qty <= 0:
            order.failure_class = order.state

    def _submit_with_recovery(self, adapter: Any, req: OrderRequest, pair: PairExecution) -> Order | None:
        """提交 + UNKNOWN_SUBMISSION 单次恢复。返回已确认 Order 或 None。"""
        try:
            order: Order = adapter.place_order(req)
        except UnknownSubmission:
            pair.touches(PairStatus.UNKNOWN_SUBMISSION.value, error=f"提交结果未知: {req.client_order_id}")
            self._persist(pair)
            recovered = self._recover_once(adapter, req.symbol, req.client_order_id)
            if recovered is None:
                pair.touches(PairStatus.HALTED.value,
                             error=f"{req.client_order_id} 提交未知且查询确认未成交/查询失败，不重下，转人工")
                self.on_alert("UNKNOWN_SUBMISSION", f"{req.market.value} {req.symbol} {req.client_order_id} 结果未知，已停止")
                self._halt(f"{req.client_order_id} 结果未知")
                return None
            order = recovered
        except (OrderRejected, BinanceError) as exc:
            pair.touches(PairStatus.FAILED.value, error=f"{req.client_order_id} 被拒绝: {exc}")
            self.on_alert("ORDER_REJECTED", f"{req.market.value} {req.symbol}: {exc}")
            self.metrics.inc("order_rejection_total", labels={"market": req.market.value})
            return None
        except Exception as exc:  # noqa: BLE001
            # 认证/时钟/IP 封禁等：不允许继续
            pair.touches(PairStatus.FAILED.value, error=f"{req.client_order_id} 提交异常: {exc}")
            self.on_alert("ORDER_SUBMIT_ERROR", f"{req.market.value} {req.symbol}: {exc}")
            self._halt(f"{req.client_order_id} 提交异常: {exc}")
            return None

        self._stamp_pair(order, pair)
        self.metrics.inc("order_submit_total",
                         labels={"market": req.market.value, "side": req.side.value,
                                 "result": order.state.lower()})
        self._persist(pair)
        return order

    def _recover_once(self, adapter: Any, symbol: str, client_order_id: str) -> Order | None:
        """UNKNOWN 恢复的唯一动作：按 clientOrderId 查询一次。"""
        try:
            order = adapter.query_by_client_order_id(symbol, client_order_id)
        except Exception as exc:  # noqa: BLE001
            self.on_alert("RECOVERY_QUERY_FAILED", f"{client_order_id} 恢复查询失败: {exc}")
            return None
        return cast("Order | None", order)

    def _wait_terminal(self, adapter: Any, order: Order) -> Order | None:
        """轮询等待终态；超时后撤单并再查一次。

        Returns:
            终态/部分成交 Order；None 表示撤单后确认无成交（可安全放弃）。
        """
        deadline = self._now() + self.order_ack_timeout_seconds
        current = order
        while not current.is_terminal and self._now() < deadline:
            self._sleep(self.poll_interval_seconds)
            try:
                q = adapter.query_by_client_order_id(current.symbol, current.client_order_id)
            except Exception:  # noqa: BLE001
                q = None
            if q is not None:
                current = q
        if current.is_terminal:
            self._mark_terminal_order(current)
            return current

        # 超时未终态 → 撤单（减风险）
        try:
            adapter.cancel_order(current.symbol, current.client_order_id)
        except Exception as exc:  # noqa: BLE001
            self.on_alert("CANCEL_TIMEOUT_ORDER_FAILED",
                          f"{current.client_order_id} 超时撤单失败: {exc}，已停止")
            self._halt(f"{current.client_order_id} 超时且撤单失败")
            return None
        try:
            final: Order | None = adapter.query_by_client_order_id(current.symbol, current.client_order_id)
        except Exception:  # noqa: BLE001
            final = None
        if final is None or final.executed_qty <= 0:
            return None
        return final

    @staticmethod
    def _mark_terminal_order(order: Order) -> None:
        if order.terminal_ts_ms is None:
            order.terminal_ts_ms = order.updated_ms or _now_ms()
        if order.state in ("REJECTED", "CANCELED", "EXPIRED") and order.executed_qty <= 0:
            order.failure_class = order.state

    # ------------------------------------------------------------------
    # 补偿 / 校验
    # ------------------------------------------------------------------

    def _reduce_perp_residual(self, pair: PairExecution, qty: Decimal) -> Decimal:
        """reduce-only 平掉永续残量。返回未能平掉的数量（0 = 完全成功）。"""
        try:
            self.futures.reduce_only_close(pair.symbol, qty,
                                           client_order_id=new_client_order_id(
                                               self.strategy_version, pair.pair_execution_id, "red"))
            return Decimal("0")
        except (OrderRejected, BinanceError, UnknownSubmission) as exc:
            self.on_alert("REDUCE_FAILED", f"{pair.symbol} 永续残量 reduce 失败: {exc}")
            return qty

    def _compensate_flatten_perp(self, pair: PairExecution, perp_qty: Decimal, *, reason: str) -> Order | None:
        """现货腿不可执行时，平掉永续腿（唯一安全补偿）。"""
        if perp_qty <= 0:
            pair.touches(PairStatus.COMPLETE.value)
            return None
        left = self._reduce_perp_residual(pair, perp_qty)
        if left <= 0:
            pair.touches(PairStatus.COMPENSATED.value, error=f"已补偿（永续腿已平）: {reason}")
            return None
        pair.touches(PairStatus.HALTED.value,
                     error=f"补偿失败: 永续腿剩 {left} 无法平掉（{reason}）")
        self.on_alert("COMPENSATION_FAILED",
                      f"{pair.symbol} 永续腿剩 {left} 无法平掉，裸头寸风险！原因: {reason}")
        self._halt(f"{pair.symbol} 补偿失败，剩 {left}")
        return None

    def _check_leg_slippage(self, order: Order, reference: Decimal, label: str) -> None:
        if order.avg_price is None or reference <= 0:
            return
        slip = abs(order.avg_price - reference) / reference
        if slip > self.max_leg_slippage_pct:
            self.on_alert("SLIPPAGE_EXCEEDED",
                          f"{order.symbol} {label} 腿滑点 {slip:.4%} > {self.max_leg_slippage_pct:.2%}")
            self._halt(f"{order.symbol} {label} 腿滑点超限")

    def _verify_hedge(self, pair: PairExecution) -> tuple[bool, str | None]:
        if pair.perp is None or pair.spot is None:
            return False, "缺少腿订单"
        perp_qty = pair.perp.executed_qty
        spot_qty = pair.spot.executed_qty
        if perp_qty <= 0 or spot_qty <= 0:
            return False, f"存在零成交腿 (perp={perp_qty}, spot={spot_qty})"
        diff_pct = abs(perp_qty - spot_qty) / perp_qty
        if diff_pct > self.hedge_tolerance_pct:
            return False, f"两腿数量差 {diff_pct:.4%} 超容差 {self.hedge_tolerance_pct:.2%}"
        return True, None

    # ------------------------------------------------------------------
    # 基础设施
    # ------------------------------------------------------------------

    def _sync_fills(self, symbol: str, since_ms: int) -> None:
        """把交易所成交明细同步到账本（幂等）。"""
        # client_order_id → (run_id, pair_execution_id, intent_id) 关联映射
        order_meta: dict[str, tuple[str, str, str]] = {}
        try:
            for row in self.store.orders(limit=10000):
                if row.get("symbol") != symbol or not row.get("client_order_id"):
                    continue
                order_meta[str(row["client_order_id"])] = (
                    str(row.get("run_id") or ""),
                    str(row.get("pair_execution_id") or ""),
                    str(row.get("intent_id") or ""),
                )
        except Exception:  # noqa: BLE001
            logger.warning("读取订单关联信息失败", exc_info=True)

        def _meta(client_order_id: str) -> tuple[str, str, str]:
            return order_meta.get(str(client_order_id), ("", "", ""))

        # Futures：userTrades 带 clientOrderId
        try:
            for t in self.futures.user_trades(symbol):
                if int(t.get("time") or 0) < since_ms:
                    continue
                run_id, pair_id, intent_id = _meta(t.get("clientOrderId") or "")
                qty = Decimal(str(t.get("qty") or 0))
                price = Decimal(str(t.get("price") or 0))
                fill = Fill(
                    fill_id=str(t.get("id")),
                    client_order_id=str(t.get("clientOrderId") or ""),
                    exchange_order_id=str(t.get("orderId") or ""),
                    symbol=symbol,
                    market=Market.PERP,
                    side=OrderSide.BUY if t.get("isBuyer") else OrderSide.SELL,
                    quantity=qty,
                    price=price,
                    fee_asset=str(t.get("commissionAsset") or ""),
                    fee_amount=Decimal(str(t.get("commission") or 0)),
                    ts_ms=int(t.get("time") or 0),
                    run_id=run_id,
                    pair_execution_id=pair_id or None,
                    intent_id=intent_id or None,
                    exchange_trade_id=str(t.get("id")),
                    quote_qty=qty * price,
                    maker_taker="MAKER" if t.get("isBuyerMaker") else "TAKER",
                    exchange_ts_ms=int(t.get("time") or 0),
                    received_ts_ms=_now_ms(),
                )
                self.store.record_fill(fill)
        except Exception:  # noqa: BLE001
            logger.warning("同步永续成交失败", exc_info=True)

        # Spot：myTrades 只有 orderId，通过 exchange_order_id 映射
        try:
            order_ids = {
                str(row.get("exchange_order_id")): row["client_order_id"]
                for row in self.store.orders(limit=10000)
                if row.get("exchange_order_id") and row["symbol"] == symbol
            }
            for t in self.spot.my_trades(symbol):
                if int(t.get("time") or 0) < since_ms:
                    continue
                oid = str(t.get("orderId") or "")
                run_id, pair_id, intent_id = _meta(order_ids.get(oid, ""))
                qty = Decimal(str(t.get("qty") or 0))
                price = Decimal(str(t.get("price") or 0))
                fill = Fill(
                    fill_id=str(t.get("id")),
                    client_order_id=order_ids.get(oid, ""),
                    exchange_order_id=oid,
                    symbol=symbol,
                    market=Market.SPOT,
                    side=OrderSide.BUY if t.get("isBuyer") else OrderSide.SELL,
                    quantity=qty,
                    price=price,
                    fee_asset=str(t.get("commissionAsset") or ""),
                    fee_amount=Decimal(str(t.get("commission") or 0)),
                    ts_ms=int(t.get("time") or 0),
                    run_id=run_id,
                    pair_execution_id=pair_id or None,
                    intent_id=intent_id or None,
                    exchange_trade_id=str(t.get("id")),
                    quote_qty=qty * price,
                    maker_taker="MAKER" if t.get("isBuyerMaker") else "TAKER",
                    exchange_ts_ms=int(t.get("time") or 0),
                    received_ts_ms=_now_ms(),
                )
                self.store.record_fill(fill)
        except Exception:  # noqa: BLE001
            logger.warning("同步现货成交失败", exc_info=True)

    def _halt(self, reason: str) -> None:
        self.gate.halt(reason)

    def _persist(self, pair: PairExecution) -> None:
        pair.updated_ms = _now_ms()
        self.store.upsert_pair(pair)
