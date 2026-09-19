"""订单状态机单元测试（开发设计文档 §2.2 / §6.1 / §11.1）。

核心不变量：非法转移必须抛异常，静默接受非法转移是账本失真的根源。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cointrader.errors import IllegalStateTransition
from cointrader.execution.order_state import (
    TERMINAL_STATES,
    OrderState,
    apply_update,
    map_exchange_status,
    validate_transition,
)


class TestMapExchangeStatus:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("NEW", OrderState.NEW),
            ("new", OrderState.NEW),  # 大小写不敏感
            ("PARTIALLY_FILLED", OrderState.PARTIALLY_FILLED),
            ("FILLED", OrderState.FILLED),
            ("DONE", OrderState.FILLED),
            ("CANCELED", OrderState.CANCELED),
            ("CANCELLED", OrderState.CANCELED),
            ("PENDING_CANCEL", OrderState.CANCELED),
            ("EXPIRED", OrderState.EXPIRED),
            ("REJECTED", OrderState.REJECTED),
        ],
    )
    def test_known_statuses(self, raw: str, expected: OrderState) -> None:
        assert map_exchange_status(raw) is expected

    @pytest.mark.parametrize("raw", ["FOO", "", None, "PENDING", "MAGIC_STATE"])
    def test_unknown_becomes_unknown(self, raw: str | None) -> None:
        """未知状态一律映射 UNKNOWN（状态不可信），触发 REST 查询而不是猜测。"""
        assert map_exchange_status(raw) is OrderState.UNKNOWN


class TestTerminalStates:
    def test_terminal_set_contents(self) -> None:
        assert {
            OrderState.FILLED.value,
            OrderState.CANCELED.value,
            OrderState.REJECTED.value,
            OrderState.EXPIRED.value,
        } == TERMINAL_STATES
        assert OrderState.NEW.value not in TERMINAL_STATES
        assert OrderState.UNKNOWN.value not in TERMINAL_STATES


class TestValidateTransition:
    @pytest.mark.parametrize(
        ("cur", "new"),
        [
            (OrderState.NEW, OrderState.PARTIALLY_FILLED),
            (OrderState.NEW, OrderState.FILLED),
            (OrderState.NEW, OrderState.CANCELED),
            (OrderState.NEW, OrderState.REJECTED),
            (OrderState.PARTIALLY_FILLED, OrderState.FILLED),
            (OrderState.PARTIALLY_FILLED, OrderState.CANCELED),
            (OrderState.UNKNOWN, OrderState.FILLED),  # UNKNOWN 恢复
            (OrderState.UNKNOWN, OrderState.UNKNOWN),
            (OrderState.FILLED, OrderState.FILLED),  # 终态→自身（重复事件）允许
        ],
    )
    def test_legal_transitions(self, cur: OrderState, new: OrderState) -> None:
        validate_transition(cur.value, new)

    @pytest.mark.parametrize(
        ("cur", "new"),
        [
            (OrderState.FILLED, OrderState.NEW),
            (OrderState.FILLED, OrderState.PARTIALLY_FILLED),
            (OrderState.FILLED, OrderState.CANCELED),
            (OrderState.CANCELED, OrderState.FILLED),
            (OrderState.REJECTED, OrderState.NEW),
            (OrderState.EXPIRED, OrderState.FILLED),
        ],
    )
    def test_terminal_to_other_rejected(self, cur: OrderState, new: OrderState) -> None:
        """终态订单不可再变：交易所重复/异常事件应忽略并告警。"""
        with pytest.raises(IllegalStateTransition, match="终态订单不可再变"):
            validate_transition(cur.value, new)

    def test_accepts_plain_string_current(self) -> None:
        validate_transition("NEW", OrderState.FILLED)


class TestApplyUpdate:
    def test_returns_full_tuple(self) -> None:
        state, qty, price, oid = apply_update(
            "NEW",
            "FILLED",
            executed_qty=Decimal("1.5"),
            avg_price=Decimal("100"),
            exchange_order_id="42",
        )
        assert state == "FILLED"
        assert qty == Decimal("1.5")
        assert price == Decimal("100")
        assert oid == "42"

    def test_defaults_when_fields_absent(self) -> None:
        state, qty, price, oid = apply_update("NEW", "PARTIALLY_FILLED")
        assert state == "PARTIALLY_FILLED"
        assert qty == Decimal("0")
        assert price is None
        assert oid is None

    def test_illegal_transition_raises(self) -> None:
        with pytest.raises(IllegalStateTransition):
            apply_update("FILLED", "CANCELED")
