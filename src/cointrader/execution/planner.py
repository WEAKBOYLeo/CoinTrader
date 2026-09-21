"""OrderPlanner：``ApprovedIntent`` → 领域 ``ExecutionPlan``（T3，AC-06）。

只规划，不执行：数量按 Spot/Futures 规则向下归一化（``execution.rules``），
clientOrderId 复用现有生成规则；计划有效期不晚于审批有效期；
平仓/关闭语义下所有订单 ``reduce_only=True``（领域层强制）。
**不创建 transport、不发起网络请求。** legacy ``PairExecutor`` 的状态机
（部分成交/UNKNOWN/补偿）不变，T4 通过 adapter 消费本计划。
"""

from __future__ import annotations

import time
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from ..domain.execution import (
    ExecutionPlan,
    PlanOrder,
    PlanOrderType,
    PlanSide,
)
from ..domain.market import MarketKind
from ..domain.portfolio import IntentAction  # noqa: F401  (re-export 便利)
from ..domain.risk import ApprovedIntent
from ..execution.models import new_client_order_id, new_id
from ..execution.rules import RuleError, SymbolRules, floor_to_step, format_decimal

__all__ = ["PlanError", "OrderPlanner"]


class PlanError(Exception):
    """规划失败（未产生任何可执行订单）。"""


def _now_ms() -> int:
    return int(time.time() * 1000)


class OrderPlanner:
    """双腿计划生成器（纯计算）。

    Args:
        strategy_version: 策略版本（用于 clientOrderId）。
    """

    def __init__(self, strategy_version: str = "fc3") -> None:
        self.strategy_version = strategy_version

    def plan(
        self,
        approved: ApprovedIntent,
        *,
        spot_price: Decimal,
        perp_price: Decimal,
        spot_rules: SymbolRules,
        perp_rules: SymbolRules,
        close_spot_qty: Decimal | None = None,
        close_perp_qty: Decimal | None = None,
        now_ms: int | None = None,
        plan_id: str | None = None,
    ) -> ExecutionPlan:
        """把审批通过的意图转成可执行计划。

        开仓：现货 BUY + 永续 SELL，数量 = 批准名义额 / 现报价 向下归一化。
        平仓：现货 SELL + 永续 BUY（全部 reduce-only），数量必须显式传入
        （``close_spot_qty``/``close_perp_qty`` = 交易所实际持仓量；领域层
        要求计划订单数量为正，不能用名义额 0 推导）。
        """
        now_ms = _now_ms() if now_ms is None else now_ms
        plan_id = plan_id or new_id("plan")
        if approved.is_expired(now_ms):
            raise PlanError(
                f"ApprovedIntent 已过期（valid_until={approved.valid_until_ms}, "
                f"now={now_ms}），拒绝规划（过期计划不可提交）"
            )
        closing = approved.is_closing
        symbol = approved.intent.symbol

        spot_side = PlanSide.SELL if closing else PlanSide.BUY
        perp_side = PlanSide.BUY if closing else PlanSide.SELL
        reduce_only = closing

        if closing:
            spot_qty = (
                self._quantity_from_qty(close_spot_qty, spot_price, spot_rules)
                if close_spot_qty is not None
                else Decimal("0")
            )
            perp_qty = (
                self._quantity_from_qty(close_perp_qty, perp_price, perp_rules)
                if close_perp_qty is not None
                else Decimal("0")
            )
            if spot_qty <= 0 or perp_qty <= 0:
                raise PlanError("平仓计划必须提供两腿实际持仓量（close_spot_qty/close_perp_qty）")
        else:
            spot_qty = self._quantity(approved.approved_spot_notional, spot_price, spot_rules)
            perp_qty = self._quantity(approved.approved_perp_notional, perp_price, perp_rules)
            if spot_qty <= 0 or perp_qty <= 0:
                raise PlanError(
                    f"归一化后存在零量腿（spot={format_decimal(spot_qty)} "
                    f"perp={format_decimal(perp_qty)}），拒绝规划"
                )

        orders: list[PlanOrder] = [
            PlanOrder(
                client_order_id=new_client_order_id(self.strategy_version, plan_id, "p"),
                market=MarketKind.FUTURES,
                symbol=symbol,
                side=perp_side,
                order_type=PlanOrderType.MARKET,
                quantity=perp_qty,
                price=None,
                reduce_only=reduce_only,
            ),
            PlanOrder(
                client_order_id=new_client_order_id(self.strategy_version, plan_id, "s"),
                market=MarketKind.SPOT,
                symbol=symbol,
                side=spot_side,
                order_type=PlanOrderType.MARKET,
                quantity=spot_qty,
                price=None,
                reduce_only=reduce_only,
            ),
        ]
        return ExecutionPlan(
            plan_id=plan_id,
            approved_intent=approved,
            orders=tuple(orders),
            plan_created_at_ms=now_ms,
            expires_at_ms=min(approved.valid_until_ms, now_ms + 30_000),
        )

    def _quantity_from_qty(
        self, qty: Decimal, price: Decimal, rules: SymbolRules
    ) -> Decimal:
        """平仓量：按规则步长归一（向上取整以尽量全平，但不超过实际量）。

        平仓必须平掉实际持仓：向下归一会产生残量，因此这里向上对齐步长；
        交易所 reduce-only 订单以持仓为上限，不会产生反向开仓。
        """
        if qty <= 0:
            return Decimal("0")
        if price <= 0:
            raise PlanError(f"非法价格 {price}")
        step = rules.qty_step(is_market=True)
        raw = qty / step
        if raw != raw.to_integral_value(rounding=ROUND_FLOOR):
            qty = (raw.to_integral_value(rounding=ROUND_CEILING) * step).normalize()
        if qty < rules.min_qty_for(is_market=True):
            raise PlanError(f"平仓量 {format_decimal(qty)} 低于最小值")
        max_qty = rules.max_qty_for(is_market=True)
        if max_qty > 0 and qty > max_qty:
            raise PlanError(f"平仓量 {format_decimal(qty)} 超过最大值")
        return qty

    def _quantity(self, notional: Decimal, price: Decimal, rules: SymbolRules) -> Decimal:
        """名义额 → 数量（向下归一化 + 最小/最大校验）。"""
        if price <= 0:
            raise PlanError(f"非法价格 {price}")
        if notional <= 0:
            return Decimal("0")
        raw = notional / price
        try:
            qty = floor_to_step(raw, rules.qty_step(is_market=True))
            min_qty = rules.min_qty_for(is_market=True)
            if qty < min_qty:
                raise RuleError(f"数量 {raw} 归一化后 {format_decimal(qty)} 低于最小值 {format_decimal(min_qty)}")
            max_qty = rules.max_qty_for(is_market=True)
            if max_qty > 0 and qty > max_qty:
                raise RuleError(f"数量 {format_decimal(qty)} 超过最大值 {format_decimal(max_qty)}")
            return qty
        except RuleError as exc:
            raise PlanError(str(exc)) from exc
