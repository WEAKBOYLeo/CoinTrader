"""组合规划领域契约：当前组合视图与组合意图。

硬约束（实施计划书 §5）：

- ``PortfolioView`` 是 planner 读取的当前持仓只读视图（来自 ledger current
  projection），不代表策略决定。
- ``PortfolioIntent`` 由 planner 创建、先落 ledger 再交风险层；
  确定性幂等指纹用于去重 —— 相同语义输入必须产生相同 ``fingerprint``。
- intent 不直接含交易所订单参数（订单参数是 execution 层 ``OrderPlanner`` 的职责）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from .common import (
    InvalidDomainValue,
    parse_decimal,
    parse_enum,
    parse_int_ms,
    parse_optional,
    parse_str,
)

__all__ = [
    "CurrentPosition",
    "IntentAction",
    "PortfolioIntent",
    "PortfolioView",
]


@dataclass(frozen=True)
class CurrentPosition:
    """当前组合中单个 symbol 的两腿名义额（只读事实）。"""

    symbol: str
    spot_notional: Decimal
    perp_notional: Decimal
    updated_at_ms: int

    def __post_init__(self) -> None:
        if not self.symbol:
            raise InvalidDomainValue("CurrentPosition.symbol 不能为空")
        if self.spot_notional < 0 or self.perp_notional < 0:
            raise InvalidDomainValue(f"当前名义额不得为负: {self.symbol}")


@dataclass(frozen=True)
class PortfolioView:
    """planner 的输入：截至 ``as_of_ms`` 的当前组合视图。"""

    snapshot_id: str
    as_of_ms: int
    entries: tuple[CurrentPosition, ...] = ()

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise InvalidDomainValue("PortfolioView.snapshot_id 不能为空")
        seen: set[str] = set()
        for entry in self.entries:
            if entry.symbol in seen:
                raise InvalidDomainValue(f"组合视图含重复 symbol: {entry.symbol}")
            seen.add(entry.symbol)

    def get(self, symbol: str) -> CurrentPosition | None:
        for entry in self.entries:
            if entry.symbol == symbol:
                return entry
        return None


class IntentAction(str, Enum):
    """组合意图动作。"""

    OPEN = "OPEN"
    RESIZE = "RESIZE"
    CLOSE = "CLOSE"
    HOLD = "HOLD"
    REPLACE = "REPLACE"


def intent_fingerprint(
    action: IntentAction,
    symbol: str,
    target_spot_notional: Decimal,
    target_perp_notional: Decimal,
    snapshot_id: str,
    decision_cutoff_ms: int,
) -> str:
    """确定性幂等指纹：只取语义字段（不含 id/时间戳/关联 id）。"""

    def canon(value: Decimal) -> str:
        return format(value, "f")

    payload = "|".join(
        [
            action.value,
            symbol,
            canon(target_spot_notional),
            canon(target_perp_notional),
            snapshot_id,
            str(decision_cutoff_ms),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class PortfolioIntent:
    """从目标组合到当前组合的差异。

    - 确定性幂等指纹：``fingerprint()`` 只取语义字段
      （action/symbol/两腿目标名义额/输入快照/决策截止），
      不含 ``created_at_ms``/``correlation_id``/``causation_id``，
      因此重复输入不产生新指纹，重复 intent 可去重。
    - ``CLOSE``/平仓语义下两腿目标名义额必须为 0（由 ``__post_init__`` 保证）。
    """

    intent_id: str
    action: IntentAction
    symbol: str
    target_spot_notional: Decimal
    target_perp_notional: Decimal
    reason: str
    snapshot_id: str
    decision_cutoff_ms: int
    created_at_ms: int
    correlation_id: str | None = None
    causation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.intent_id:
            raise InvalidDomainValue("PortfolioIntent.intent_id 不能为空")
        if not self.symbol:
            raise InvalidDomainValue("PortfolioIntent.symbol 不能为空")
        if self.target_spot_notional < 0 or self.target_perp_notional < 0:
            raise InvalidDomainValue(f"目标名义额不得为负: {self.symbol}")
        if self.action is IntentAction.CLOSE and (
            self.target_spot_notional != 0 or self.target_perp_notional != 0
        ):
            raise InvalidDomainValue(f"CLOSE 意图的两腿目标名义额必须为 0: {self.symbol}")

    @property
    def is_closing(self) -> bool:
        """平仓/降为零的意图：``reduce_only`` 路径标记来源。"""
        return self.action is IntentAction.CLOSE

    def fingerprint(self) -> str:
        """确定性幂等指纹（sha256 前 32 位十六进制）。"""
        return intent_fingerprint(
            self.action,
            self.symbol,
            self.target_spot_notional,
            self.target_perp_notional,
            self.snapshot_id,
            self.decision_cutoff_ms,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "intent_id": self.intent_id,
            "action": self.action.value,
            "symbol": self.symbol,
            "target_spot_notional": str(self.target_spot_notional),
            "target_perp_notional": str(self.target_perp_notional),
            "reason": self.reason,
            "snapshot_id": self.snapshot_id,
            "decision_cutoff_ms": self.decision_cutoff_ms,
            "created_at_ms": self.created_at_ms,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "fingerprint": self.fingerprint(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PortfolioIntent:
        intent = cls(
            intent_id=parse_str(data.get("intent_id"), "PortfolioIntent.intent_id"),
            action=parse_enum(IntentAction, data.get("action"), "PortfolioIntent.action"),
            symbol=parse_str(data.get("symbol"), "PortfolioIntent.symbol"),
            target_spot_notional=parse_decimal(
                data.get("target_spot_notional"), "PortfolioIntent.target_spot_notional"
            ),
            target_perp_notional=parse_decimal(
                data.get("target_perp_notional"), "PortfolioIntent.target_perp_notional"
            ),
            reason=parse_str(data.get("reason"), "PortfolioIntent.reason"),
            snapshot_id=parse_str(data.get("snapshot_id"), "PortfolioIntent.snapshot_id"),
            decision_cutoff_ms=parse_int_ms(
                data.get("decision_cutoff_ms"), "PortfolioIntent.decision_cutoff_ms"
            ),
            created_at_ms=parse_int_ms(
                data.get("created_at_ms"), "PortfolioIntent.created_at_ms"
            ),
            correlation_id=parse_optional(
                data.get("correlation_id"), "PortfolioIntent.correlation_id", parse_str
            ),
            causation_id=parse_optional(
                data.get("causation_id"), "PortfolioIntent.causation_id", parse_str
            ),
        )
        stored = data.get("fingerprint")
        if stored is not None and stored != intent.fingerprint():
            raise InvalidDomainValue(
                "PortfolioIntent.fingerprint 与语义字段不一致"
                f"（stored={stored}，computed={intent.fingerprint()}）"
            )
        return intent
