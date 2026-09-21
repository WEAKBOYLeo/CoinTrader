"""执行领域契约：执行计划、计划订单与 Pair 状态。

硬约束（实施计划书 §5/§9）：

- ``ExecutionPlan`` 由 OrderPlanner 创建、PairExecutor 只读消费；
  所有数量/价格一律 ``Decimal``，不合法数量在发单前拒绝。
- ``client_order_id`` 唯一且稳定：同一意图不可生成第二个有效提交。
- ``PairStatus`` 非法状态迁移抛领域异常并保留旧状态（迁移判定由
  execution 层状态机执行，这里只定义状态集合与终态/兼容状态）。
- 本模块无 IO、无网络、无 broker 依赖。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from .common import (
    InvalidDomainValue,
    parse_bool,
    parse_decimal,
    parse_enum,
    parse_int_ms,
    parse_optional,
    parse_str,
)
from .market import MarketKind
from .risk import ApprovedIntent

__all__ = [
    "ExecutionPlan",
    "PairStatus",
    "PAIR_TERMINAL_STATES",
    "PlanOrder",
    "PlanOrderType",
    "PlanSide",
]


class PlanSide(str, Enum):
    """计划订单方向。"""

    BUY = "BUY"
    SELL = "SELL"


class PlanOrderType(str, Enum):
    """计划订单类型（实际下单参数由现有 rules/adapter 归一化）。"""

    LIMIT = "LIMIT"
    MARKET = "MARKET"


class PairStatus(str, Enum):
    """Pair 状态机状态。

    与 ``execution.models.PairStatus`` 现有状态一一对应（迁移期兼容），
    领域侧只声明状态集合，不实现迁移规则。
    """

    # 开仓
    NEW_INTENT = "NEW_INTENT"
    PRECHECK = "PRECHECK"
    SUBMIT_PERP = "SUBMIT_PERP"
    PERP_ACKNOWLEDGED = "PERP_ACKNOWLEDGED"
    PERP_FILLED = "PERP_FILLED"
    SUBMIT_SPOT = "SUBMIT_SPOT"
    SPOT_FILLED = "SPOT_FILLED"
    HEDGE_VERIFIED = "HEDGE_VERIFIED"
    COMPLETE = "COMPLETE"
    # 异常/补偿
    UNKNOWN_SUBMISSION = "UNKNOWN_SUBMISSION"
    COMPENSATING = "COMPENSATING"
    COMPENSATED = "COMPENSATED"
    FLATTENED = "FLATTENED"
    # 终态
    FAILED = "FAILED"
    HALTED = "HALTED"
    # 平仓
    CLOSE_INTENT = "CLOSE_INTENT"
    PRECHECK_CLOSE = "PRECHECK_CLOSE"
    FILL_TRACKING = "FILL_TRACKING"
    RESIDUAL_CHECK = "RESIDUAL_CHECK"
    COMPENSATE_RESIDUAL = "COMPENSATE_RESIDUAL"


#: 终态集合（与 execution 层保持一致；终态不可再迁移）。
PAIR_TERMINAL_STATES = frozenset(
    {
        PairStatus.COMPLETE.value,
        PairStatus.COMPENSATED.value,
        PairStatus.FLATTENED.value,
        PairStatus.FAILED.value,
        PairStatus.HALTED.value,
    }
)


@dataclass(frozen=True)
class PlanOrder:
    """执行计划中的单条订单意图参数。

    数量/价格已由调用方按规则归一化（stepSize/minNotional 等）；
    本对象只做结构性校验，不访问交易所。
    """

    client_order_id: str
    market: MarketKind
    symbol: str
    side: PlanSide
    order_type: PlanOrderType
    quantity: Decimal
    price: Decimal | None
    reduce_only: bool = False

    def __post_init__(self) -> None:
        if not self.client_order_id:
            raise InvalidDomainValue("PlanOrder.client_order_id 不能为空")
        if not self.symbol:
            raise InvalidDomainValue("PlanOrder.symbol 不能为空")
        if self.quantity <= 0:
            raise InvalidDomainValue(
                f"计划订单数量必须为正（发单前拒绝）: {self.symbol} {self.quantity}"
            )
        if self.price is not None and self.price <= 0:
            raise InvalidDomainValue(f"计划订单价格必须为正: {self.symbol} {self.price}")
        if self.order_type is PlanOrderType.LIMIT and self.price is None:
            raise InvalidDomainValue(f"LIMIT 计划订单必须有价格: {self.symbol}")

    def to_dict(self) -> dict[str, object]:
        return {
            "client_order_id": self.client_order_id,
            "market": self.market.value,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": str(self.quantity),
            "price": None if self.price is None else str(self.price),
            "reduce_only": self.reduce_only,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PlanOrder:
        return cls(
            client_order_id=parse_str(
                data.get("client_order_id"), "PlanOrder.client_order_id"
            ),
            market=parse_enum(MarketKind, data.get("market"), "PlanOrder.market"),
            symbol=parse_str(data.get("symbol"), "PlanOrder.symbol"),
            side=parse_enum(PlanSide, data.get("side"), "PlanOrder.side"),
            order_type=parse_enum(
                PlanOrderType, data.get("order_type"), "PlanOrder.order_type"
            ),
            quantity=parse_decimal(data.get("quantity"), "PlanOrder.quantity"),
            price=parse_optional(data.get("price"), "PlanOrder.price", parse_decimal),
            reduce_only=parse_bool(data.get("reduce_only"), "PlanOrder.reduce_only"),
        )


@dataclass(frozen=True)
class ExecutionPlan:
    """一次批准的 ApprovedIntent 对应的稳定执行计划。

    - 过期计划不可执行（``is_expired``）。
    - ``reduce_only`` 语义由计划内所有订单的 ``reduce_only`` 承载，
      下游不得清除。
    """

    plan_id: str
    approved_intent: ApprovedIntent
    orders: tuple[PlanOrder, ...]
    plan_created_at_ms: int
    expires_at_ms: int

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise InvalidDomainValue("ExecutionPlan.plan_id 不能为空")
        if self.expires_at_ms < self.plan_created_at_ms:
            raise InvalidDomainValue(
                "expires_at_ms 不得早于 plan_created_at_ms"
                f"（{self.expires_at_ms} < {self.plan_created_at_ms}）"
            )
        if not self.orders:
            raise InvalidDomainValue("ExecutionPlan 至少包含一条计划订单")
        if self.approved_intent.is_closing and not all(
            order.reduce_only for order in self.orders
        ):
            raise InvalidDomainValue(
                "平仓/关闭语义的执行计划所有订单必须 reduce_only"
            )

    def is_expired(self, now_ms: int) -> bool:
        return now_ms >= self.expires_at_ms

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "approved_intent": self.approved_intent.to_dict(),
            "orders": [o.to_dict() for o in self.orders],
            "plan_created_at_ms": self.plan_created_at_ms,
            "expires_at_ms": self.expires_at_ms,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ExecutionPlan:
        intent_raw = data.get("approved_intent")
        if not isinstance(intent_raw, Mapping):
            raise InvalidDomainValue("ExecutionPlan.approved_intent 必须是对象")
        orders_raw = data.get("orders")
        if not isinstance(orders_raw, list) or not orders_raw:
            raise InvalidDomainValue("ExecutionPlan.orders 必须是非空列表")
        orders = []
        for item in orders_raw:
            if not isinstance(item, Mapping):
                raise InvalidDomainValue("ExecutionPlan.orders 元素必须是对象")
            orders.append(PlanOrder.from_dict(item))
        return cls(
            plan_id=parse_str(data.get("plan_id"), "ExecutionPlan.plan_id"),
            approved_intent=ApprovedIntent.from_dict(intent_raw),
            orders=tuple(orders),
            plan_created_at_ms=parse_int_ms(
                data.get("plan_created_at_ms"), "ExecutionPlan.plan_created_at_ms"
            ),
            expires_at_ms=parse_int_ms(data.get("expires_at_ms"), "ExecutionPlan.expires_at_ms"),
        )
