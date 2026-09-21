"""风险审批领域契约：规则证据、审批决定与已批准意图。

硬约束（实施计划书 §5）：

- 新风险没有 ``ALLOW``/``RESIZE`` 决定不得进入 execution。
- 审批输入（快照 ids、决定时点、有效期）必须可追溯；
  过期或输入变更必须重新审批。
- ``ApprovedIntent`` 只可从有效 ``RiskDecision`` 创建
  （经 ``ApprovedIntent.from_decision`` 强制校验）。
- 风险审批只收紧不放宽：批准名义额不得超过请求名义额。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from .common import (
    ExpiredDomainObject,
    InvalidDomainValue,
    RiskNotApproved,
    parse_bool,
    parse_decimal,
    parse_enum,
    parse_int_ms,
    parse_str,
)
from .portfolio import PortfolioIntent

__all__ = [
    "ApprovedIntent",
    "RiskDecision",
    "RiskDecisionKind",
    "RuleEvidence",
]


class RiskDecisionKind(str, Enum):
    """风险审批结果。仅 ``ALLOW``/``RESIZE`` 放行执行。"""

    ALLOW = "ALLOW"
    RESIZE = "RESIZE"
    CLOSE_ONLY = "CLOSE_ONLY"
    REJECT = "REJECT"
    HALT = "HALT"


@dataclass(frozen=True)
class RuleEvidence:
    """单条风控规则的判定证据（可回放审计）。"""

    rule: str
    passed: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.rule:
            raise InvalidDomainValue("RuleEvidence.rule 不能为空")

    def to_dict(self) -> dict[str, object]:
        return {"rule": self.rule, "passed": self.passed, "detail": self.detail}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RuleEvidence:
        return cls(
            rule=parse_str(data.get("rule"), "RuleEvidence.rule"),
            passed=parse_bool(data.get("passed"), "RuleEvidence.passed"),
            detail=parse_str(data.get("detail"), "RuleEvidence.detail"),
        )


@dataclass(frozen=True)
class RiskDecision:
    """风险内核对单个 intent 的同步审批结果。

    - ``approved_*_notional``：允许执行的名义额。``ALLOW`` 时等于请求值，
      ``RESIZE`` 时小于请求值，其余决定下无执行语义（字段仍给出，便于审计）。
    - ``snapshot_ids``：审批依据的全部输入快照 id（市场/账户）。
    - ``valid_until_ms``：决定过期时点；过期必须重新审批。
    """

    decision_id: str
    decision: RiskDecisionKind
    symbol: str
    requested_spot_notional: Decimal
    requested_perp_notional: Decimal
    approved_spot_notional: Decimal
    approved_perp_notional: Decimal
    snapshot_ids: tuple[str, ...]
    rules: tuple[RuleEvidence, ...]
    decided_at_ms: int
    valid_until_ms: int
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise InvalidDomainValue("RiskDecision.decision_id 不能为空")
        if not self.symbol:
            raise InvalidDomainValue("RiskDecision.symbol 不能为空")
        if not self.snapshot_ids:
            raise InvalidDomainValue("RiskDecision 必须记录输入快照 ids")
        if self.valid_until_ms < self.decided_at_ms:
            raise InvalidDomainValue(
                "valid_until_ms 不得早于 decided_at_ms"
                f"（{self.valid_until_ms} < {self.decided_at_ms}）"
            )
        if self.approved_spot_notional > self.requested_spot_notional:
            raise InvalidDomainValue(
                "批准名义额不得超过请求（风险只收紧不放宽）: spot"
            )
        if self.approved_perp_notional > self.requested_perp_notional:
            raise InvalidDomainValue(
                "批准名义额不得超过请求（风险只收紧不放宽）: perp"
            )

    @property
    def allows_execution(self) -> bool:
        return self.decision in (RiskDecisionKind.ALLOW, RiskDecisionKind.RESIZE)

    def is_expired(self, now_ms: int) -> bool:
        return now_ms >= self.valid_until_ms

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "decision": self.decision.value,
            "symbol": self.symbol,
            "requested_spot_notional": str(self.requested_spot_notional),
            "requested_perp_notional": str(self.requested_perp_notional),
            "approved_spot_notional": str(self.approved_spot_notional),
            "approved_perp_notional": str(self.approved_perp_notional),
            "snapshot_ids": list(self.snapshot_ids),
            "rules": [r.to_dict() for r in self.rules],
            "decided_at_ms": self.decided_at_ms,
            "valid_until_ms": self.valid_until_ms,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RiskDecision:
        snapshot_ids_raw = data.get("snapshot_ids")
        if not isinstance(snapshot_ids_raw, list) or any(
            not isinstance(s, str) for s in snapshot_ids_raw
        ):
            raise InvalidDomainValue("RiskDecision.snapshot_ids 必须是字符串列表")
        rules_raw = data.get("rules")
        if not isinstance(rules_raw, list):
            raise InvalidDomainValue("RiskDecision.rules 必须是列表")
        rules = []
        for item in rules_raw:
            if not isinstance(item, Mapping):
                raise InvalidDomainValue("RiskDecision.rules 元素必须是对象")
            rules.append(RuleEvidence.from_dict(item))
        return cls(
            decision_id=parse_str(data.get("decision_id"), "RiskDecision.decision_id"),
            decision=parse_enum(
                RiskDecisionKind, data.get("decision"), "RiskDecision.decision"
            ),
            symbol=parse_str(data.get("symbol"), "RiskDecision.symbol"),
            requested_spot_notional=parse_decimal(
                data.get("requested_spot_notional"), "RiskDecision.requested_spot_notional"
            ),
            requested_perp_notional=parse_decimal(
                data.get("requested_perp_notional"), "RiskDecision.requested_perp_notional"
            ),
            approved_spot_notional=parse_decimal(
                data.get("approved_spot_notional"), "RiskDecision.approved_spot_notional"
            ),
            approved_perp_notional=parse_decimal(
                data.get("approved_perp_notional"), "RiskDecision.approved_perp_notional"
            ),
            snapshot_ids=tuple(snapshot_ids_raw),
            rules=tuple(rules),
            decided_at_ms=parse_int_ms(data.get("decided_at_ms"), "RiskDecision.decided_at_ms"),
            valid_until_ms=parse_int_ms(
                data.get("valid_until_ms"), "RiskDecision.valid_until_ms"
            ),
            reason=parse_str(data.get("reason"), "RiskDecision.reason"),
        )


@dataclass(frozen=True)
class ApprovedIntent:
    """已批准的组合意图：intent + 风险批准证据。

    execution 只消费本对象；``is_closing``/``reduce_only`` 不得被下游清除。
    快照与版本可追溯：``decision_id`` + intent 的 ``snapshot_id``/``decision_cutoff_ms``。
    """

    intent: PortfolioIntent
    decision_id: str
    approved_spot_notional: Decimal
    approved_perp_notional: Decimal
    decided_at_ms: int
    valid_until_ms: int
    reduce_only: bool = False
    is_closing: bool = False

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise InvalidDomainValue("ApprovedIntent.decision_id 不能为空")
        if self.approved_spot_notional < 0 or self.approved_perp_notional < 0:
            raise InvalidDomainValue("批准名义额不得为负")
        if self.valid_until_ms < self.decided_at_ms:
            raise InvalidDomainValue(
                "valid_until_ms 不得早于 decided_at_ms"
                f"（{self.valid_until_ms} < {self.decided_at_ms}）"
            )
        if self.is_closing != self.intent.is_closing:
            raise InvalidDomainValue(
                "ApprovedIntent.is_closing 必须与意图平仓语义一致（不可被清除或伪造）"
            )
        if self.approved_spot_notional > self.intent.target_spot_notional:
            raise InvalidDomainValue("批准名义额不得超过意图目标（风险只收紧不放宽）: spot")
        if self.approved_perp_notional > self.intent.target_perp_notional:
            raise InvalidDomainValue("批准名义额不得超过意图目标（风险只收紧不放宽）: perp")

    @classmethod
    def from_decision(
        cls,
        intent: PortfolioIntent,
        decision: RiskDecision,
        *,
        now_ms: int,
    ) -> ApprovedIntent:
        """只可从有效 RiskDecision 创建：ALLOW/RESIZE、未过期、symbol 一致，
        且审批必须引用意图输入快照、审批时点不得早于意图创建/决策截止（因果与无前瞻）。"""
        if decision.symbol != intent.symbol:
            raise InvalidDomainValue(
                f"审批决定与意图 symbol 不一致: {decision.symbol} != {intent.symbol}"
            )
        if intent.snapshot_id not in decision.snapshot_ids:
            raise InvalidDomainValue(
                f"审批决定缺少意图输入快照 id（输入快照不一致）: {intent.snapshot_id}"
            )
        if decision.decided_at_ms < intent.created_at_ms:
            raise InvalidDomainValue(
                "审批时点早于意图创建时点（因果链断裂）: "
                f"decided={decision.decided_at_ms} < created={intent.created_at_ms}"
            )
        if decision.decided_at_ms < intent.decision_cutoff_ms:
            raise InvalidDomainValue(
                "审批时点早于决策截止（无前瞻被破坏）: "
                f"decided={decision.decided_at_ms} < cutoff={intent.decision_cutoff_ms}"
            )
        if not decision.allows_execution:
            raise RiskNotApproved(
                f"风险审批未放行（{decision.decision.value}）: {intent.symbol}"
            )
        if decision.is_expired(now_ms):
            raise ExpiredDomainObject(
                f"风险审批已过期（valid_until={decision.valid_until_ms}, now={now_ms}）: "
                f"{intent.symbol}"
            )
        return cls(
            intent=intent,
            decision_id=decision.decision_id,
            approved_spot_notional=decision.approved_spot_notional,
            approved_perp_notional=decision.approved_perp_notional,
            decided_at_ms=decision.decided_at_ms,
            valid_until_ms=decision.valid_until_ms,
            reduce_only=decision.decision is RiskDecisionKind.RESIZE
            and (
                decision.approved_spot_notional < intent.target_spot_notional
                or decision.approved_perp_notional < intent.target_perp_notional
            ),
            is_closing=intent.is_closing,
        )

    def is_expired(self, now_ms: int) -> bool:
        return now_ms >= self.valid_until_ms

    def to_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent.to_dict(),
            "decision_id": self.decision_id,
            "approved_spot_notional": str(self.approved_spot_notional),
            "approved_perp_notional": str(self.approved_perp_notional),
            "decided_at_ms": self.decided_at_ms,
            "valid_until_ms": self.valid_until_ms,
            "reduce_only": self.reduce_only,
            "is_closing": self.is_closing,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ApprovedIntent:
        intent_raw = data.get("intent")
        if not isinstance(intent_raw, Mapping):
            raise InvalidDomainValue("ApprovedIntent.intent 必须是对象")
        return cls(
            intent=PortfolioIntent.from_dict(intent_raw),
            decision_id=parse_str(data.get("decision_id"), "ApprovedIntent.decision_id"),
            approved_spot_notional=parse_decimal(
                data.get("approved_spot_notional"), "ApprovedIntent.approved_spot_notional"
            ),
            approved_perp_notional=parse_decimal(
                data.get("approved_perp_notional"), "ApprovedIntent.approved_perp_notional"
            ),
            decided_at_ms=parse_int_ms(
                data.get("decided_at_ms"), "ApprovedIntent.decided_at_ms"
            ),
            valid_until_ms=parse_int_ms(
                data.get("valid_until_ms"), "ApprovedIntent.valid_until_ms"
            ),
            reduce_only=parse_bool(data.get("reduce_only"), "ApprovedIntent.reduce_only"),
            is_closing=parse_bool(data.get("is_closing"), "ApprovedIntent.is_closing"),
        )
