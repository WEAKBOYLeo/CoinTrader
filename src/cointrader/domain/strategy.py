"""策略领域契约：目标组合与每轮纯计算提案。

硬约束（实施计划书 §5）：

- 策略只返回可序列化的 ``StrategyProposal``/``TargetPortfolio``；
  不含订单字段、不含 API client、不创建订单、不访问网络、不写账本。
- 相同输入与同一时钟下输出必须确定。
- ``TargetPortfolio`` 两腿目标方向固定为 Spot 多、Perp 空，
  因此名义额只记录大小（非负），不记录方向。
- 过期（``now >= valid_until_ms``）的提案不可转 intent。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from .common import (
    InvalidDomainValue,
    parse_decimal,
    parse_enum,
    parse_int_ms,
    parse_str,
)

__all__ = [
    "PositionReason",
    "StrategyAction",
    "StrategyProposal",
    "TargetPortfolio",
    "TargetPosition",
]


class StrategyAction(str, Enum):
    """单个持仓上的策略动作（不含订单参数）。"""

    OPEN = "OPEN"
    HOLD = "HOLD"
    EXIT = "EXIT"
    REPLACE = "REPLACE"
    SKIP = "SKIP"


@dataclass(frozen=True)
class TargetPosition:
    """单个 symbol 的两腿目标名义额。

    方向固定：Spot 多头 + Perp 空头；名义额只表示大小。
    """

    symbol: str
    spot_notional: Decimal
    perp_notional: Decimal

    def __post_init__(self) -> None:
        if not self.symbol:
            raise InvalidDomainValue("TargetPosition.symbol 不能为空")
        if self.spot_notional < 0 or self.perp_notional < 0:
            raise InvalidDomainValue(f"目标名义额不得为负: {self.symbol}")


@dataclass(frozen=True)
class TargetPortfolio:
    """策略输出的完整目标组合。空组合合法（= 全部平仓）。

    不可变映射：symbol -> 两腿目标名义额。
    """

    entries: tuple[TargetPosition, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for entry in self.entries:
            if entry.symbol in seen:
                raise InvalidDomainValue(f"目标组合含重复 symbol: {entry.symbol}")
            seen.add(entry.symbol)

    def get(self, symbol: str) -> TargetPosition | None:
        for entry in self.entries:
            if entry.symbol == symbol:
                return entry
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": [
                {
                    "symbol": e.symbol,
                    "spot_notional": str(e.spot_notional),
                    "perp_notional": str(e.perp_notional),
                }
                for e in self.entries
            ]
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TargetPortfolio:
        raw = data.get("entries")
        if not isinstance(raw, list):
            raise InvalidDomainValue("TargetPortfolio.entries 必须是列表")
        entries = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise InvalidDomainValue("TargetPortfolio.entries 元素必须是对象")
            entries.append(
                TargetPosition(
                    symbol=parse_str(item.get("symbol"), "TargetPosition.symbol"),
                    spot_notional=parse_decimal(
                        item.get("spot_notional"), "TargetPosition.spot_notional"
                    ),
                    perp_notional=parse_decimal(
                        item.get("perp_notional"), "TargetPosition.perp_notional"
                    ),
                )
            )
        return cls(entries=tuple(entries))


@dataclass(frozen=True)
class PositionReason:
    """单个 symbol 的策略理由与证据（可回放审计）。"""

    symbol: str
    action: StrategyAction
    reason: str
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class StrategyProposal:
    """一轮策略纯计算的输出。

    记录输入 snapshot id、策略版本、配置哈希与有效期；
    过期不可提交（由 ``is_expired`` 判定，调用方负责检查）。
    """

    proposal_id: str
    snapshot_id: str
    decision_cutoff_ms: int
    strategy_version: str
    config_hash: str
    valid_until_ms: int
    target: TargetPortfolio
    reasons: tuple[PositionReason, ...] = ()

    def __post_init__(self) -> None:
        if not self.proposal_id:
            raise InvalidDomainValue("StrategyProposal.proposal_id 不能为空")
        if not self.snapshot_id:
            raise InvalidDomainValue("StrategyProposal.snapshot_id 不能为空")
        if not self.strategy_version:
            raise InvalidDomainValue("StrategyProposal.strategy_version 不能为空")
        if self.valid_until_ms < self.decision_cutoff_ms:
            raise InvalidDomainValue(
                "valid_until_ms 不得早于 decision_cutoff_ms"
                f"（{self.valid_until_ms} < {self.decision_cutoff_ms}）"
            )

    def is_expired(self, now_ms: int) -> bool:
        return now_ms >= self.valid_until_ms

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "snapshot_id": self.snapshot_id,
            "decision_cutoff_ms": self.decision_cutoff_ms,
            "strategy_version": self.strategy_version,
            "config_hash": self.config_hash,
            "valid_until_ms": self.valid_until_ms,
            "target": self.target.to_dict(),
            "reasons": [
                {
                    "symbol": r.symbol,
                    "action": r.action.value,
                    "reason": r.reason,
                    "evidence": list(r.evidence),
                }
                for r in self.reasons
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> StrategyProposal:
        target_raw = data.get("target")
        if not isinstance(target_raw, Mapping):
            raise InvalidDomainValue("StrategyProposal.target 必须是对象")
        reasons_raw = data.get("reasons")
        if not isinstance(reasons_raw, list):
            raise InvalidDomainValue("StrategyProposal.reasons 必须是列表")
        reasons = []
        for item in reasons_raw:
            if not isinstance(item, Mapping):
                raise InvalidDomainValue("StrategyProposal.reasons 元素必须是对象")
            evidence_raw = item.get("evidence")
            if not isinstance(evidence_raw, list) or any(
                not isinstance(e, str) for e in evidence_raw
            ):
                raise InvalidDomainValue("PositionReason.evidence 必须是字符串列表")
            reasons.append(
                PositionReason(
                    symbol=parse_str(item.get("symbol"), "PositionReason.symbol"),
                    action=parse_enum(
                        StrategyAction, item.get("action"), "PositionReason.action"
                    ),
                    reason=parse_str(item.get("reason"), "PositionReason.reason"),
                    evidence=tuple(evidence_raw),
                )
            )
        return cls(
            proposal_id=parse_str(data.get("proposal_id"), "StrategyProposal.proposal_id"),
            snapshot_id=parse_str(data.get("snapshot_id"), "StrategyProposal.snapshot_id"),
            decision_cutoff_ms=parse_int_ms(
                data.get("decision_cutoff_ms"), "StrategyProposal.decision_cutoff_ms"
            ),
            strategy_version=parse_str(
                data.get("strategy_version"), "StrategyProposal.strategy_version"
            ),
            config_hash=parse_str(data.get("config_hash"), "StrategyProposal.config_hash"),
            valid_until_ms=parse_int_ms(
                data.get("valid_until_ms"), "StrategyProposal.valid_until_ms"
            ),
            target=TargetPortfolio.from_dict(target_raw),
            reasons=tuple(reasons),
        )
