"""订单状态机。

规则（开发设计文档 §2.2/§6.1）：

1. 只有**非终态**订单（NEW / PARTIALLY_FILLED / UNKNOWN）可以被刷新。
2. 终态（FILLED / CANCELED / REJECTED / EXPIRED）不可再变。
   交易所若对终态订单再发事件，属于异常，按「忽略 + 告警」处理。
3. 任何**未知/无法识别**的状态一律映射为 ``UNKNOWN``（状态不可信），
   触发 REST 查询，而不是猜测。
4. 所有转移必须先过 ``validate_transition``，非法转移抛
   ``IllegalStateTransition`` —— 静默接受非法转移是账本失真的根源。
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from ..errors import IllegalStateTransition

__all__ = ["OrderState", "TERMINAL_STATES", "apply_update", "map_exchange_status", "validate_transition"]


class OrderState(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    #: 状态不可信（本地新建未确认 / 无法识别 / 查询失败）。
    UNKNOWN = "UNKNOWN"


TERMINAL_STATES = frozenset(
    {
        OrderState.FILLED.value,
        OrderState.CANCELED.value,
        OrderState.REJECTED.value,
        OrderState.EXPIRED.value,
    }
)

#: 交易所状态字符串 → 本地状态。无法识别的返回 UNKNOWN。
_EXCHANGE_STATUS_MAP: dict[str, OrderState] = {
    "NEW": OrderState.NEW,
    "NEW_GUARANTEE": OrderState.NEW,
    "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
    "FILLED": OrderState.FILLED,
    "DONE": OrderState.FILLED,
    "CANCELED": OrderState.CANCELED,
    "CANCELLED": OrderState.CANCELED,
    "PENDING_CANCEL": OrderState.CANCELED,
    "EXPIRING": OrderState.EXPIRED,
    "EXPIRED": OrderState.EXPIRED,
    "REJECTED": OrderState.REJECTED,
    "PENDING": OrderState.UNKNOWN,
}


def map_exchange_status(raw_status: str | None) -> OrderState:
    """把交易所状态字符串映射为本地状态；无法识别 → UNKNOWN。"""
    if not raw_status:
        return OrderState.UNKNOWN
    return _EXCHANGE_STATUS_MAP.get(str(raw_status).upper(), OrderState.UNKNOWN)


def validate_transition(current: str, new: OrderState) -> None:
    """校验状态转移合法性，非法时抛 IllegalStateTransition。"""
    cur = current if isinstance(current, OrderState) else OrderState(current)
    if cur in TERMINAL_STATES and cur != new:
        raise IllegalStateTransition(
            f"终态订单不可再变: {cur.value} → {new.value}（交易所重复/异常事件，应忽略并告警）"
        )
    # 其余转移（非终态 → 任意，含 UNKNOWN 恢复）均允许
    _ = cur


def apply_update(
    order_state: str,
    raw_status: str | None,
    *,
    executed_qty: Decimal | None = None,
    avg_price: Decimal | None = None,
    exchange_order_id: str | None = None,
) -> tuple[str, Decimal, Decimal | None, str | None]:
    """受控地用交易所状态刷新本地订单。

    Returns:
        (new_state, new_executed_qty, new_avg_price, new_exchange_order_id)

    Raises:
        IllegalStateTransition: 非法转移。
    """
    new_state = map_exchange_status(raw_status)
    validate_transition(order_state, new_state)
    return (
        new_state.value,
        executed_qty if executed_qty is not None else Decimal("0"),
        avg_price,
        exchange_order_id,
    )
