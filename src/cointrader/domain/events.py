"""事件领域契约：事件 envelope、健康事件与订单事件。

硬约束（实施计划书 §5）：

- ``schema_version`` 初始为 1；未知版本拒绝解析，不静默猜字段。
- 事件只保留脱敏摘要和必要关联字段（correlation/causation），
  不复制原始大型 payload，不出现密钥/签名/完整认证 URL。
- payload 为字符串键值对（脱敏摘要），结构稳定、序列化无歧义。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from .common import (
    InvalidDomainValue,
    UnknownSchemaVersion,
    parse_decimal,
    parse_int_ms,
    parse_str,
    reject_unknown_keys,
)

__all__ = [
    "EventEnvelope",
    "HealthEvent",
    "HealthKind",
    "OrderEvent",
    "Payload",
    "SUPPORTED_SCHEMA_VERSIONS",
]

#: 本实现支持解析的 schema 版本集合。
SUPPORTED_SCHEMA_VERSIONS = (1,)

#: payload 类型：字符串键 -> 字符串值（脱敏摘要）。
Payload = tuple[tuple[str, str], ...]


def _parse_payload(value: object, field: str) -> Payload:
    if not isinstance(value, list):
        raise InvalidDomainValue(f"{field} 必须是键值对列表")
    result: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise InvalidDomainValue(f"{field} 元素必须是 [key, value] 键值对")
        key, val = item
        if not isinstance(key, str) or not isinstance(val, str):
            raise InvalidDomainValue(f"{field} 键值必须都是字符串: {item!r}")
        result.append((key, val))
    return tuple(result)


@dataclass(frozen=True)
class EventEnvelope:
    """领域事件 envelope：关联 id、schema version 和事件元数据。

    ``payload`` 只允许字符串键值对（脱敏摘要），避免事件携带
    原始大型 JSON 或敏感字段。
    """

    event_id: str
    event_type: str
    schema_version: int
    occurred_at_ms: int
    correlation_id: str | None
    causation_id: str | None
    payload: Payload = ()

    def __post_init__(self) -> None:
        if not self.event_id:
            raise InvalidDomainValue("EventEnvelope.event_id 不能为空")
        if not self.event_type:
            raise InvalidDomainValue("EventEnvelope.event_type 不能为空")
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise UnknownSchemaVersion(
                f"未知 schema_version={self.schema_version}，"
                f"支持: {', '.join(str(v) for v in SUPPORTED_SCHEMA_VERSIONS)}"
            )

    @staticmethod
    def payload_to_dict(payload: Payload) -> list[list[str]]:
        return [list(pair) for pair in payload]

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "schema_version": self.schema_version,
            "occurred_at_ms": self.occurred_at_ms,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "payload": self.payload_to_dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EventEnvelope:
        version = data.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise InvalidDomainValue("EventEnvelope.schema_version 必须是 int")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise UnknownSchemaVersion(
                f"未知 schema_version={version}，"
                f"支持: {', '.join(str(v) for v in SUPPORTED_SCHEMA_VERSIONS)}"
            )
        reject_unknown_keys(
            data,
            frozenset(
                {
                    "event_id",
                    "event_type",
                    "schema_version",
                    "occurred_at_ms",
                    "correlation_id",
                    "causation_id",
                    "payload",
                }
            ),
            "EventEnvelope",
        )
        correlation_raw = data.get("correlation_id")
        causation_raw = data.get("causation_id")
        return cls(
            event_id=parse_str(data.get("event_id"), "EventEnvelope.event_id"),
            event_type=parse_str(data.get("event_type"), "EventEnvelope.event_type"),
            schema_version=version,
            occurred_at_ms=parse_int_ms(
                data.get("occurred_at_ms"), "EventEnvelope.occurred_at_ms"
            ),
            correlation_id=(
                parse_str(correlation_raw, "EventEnvelope.correlation_id")
                if correlation_raw is not None
                else None
            ),
            causation_id=(
                parse_str(causation_raw, "EventEnvelope.causation_id")
                if causation_raw is not None
                else None
            ),
            payload=_parse_payload(data.get("payload"), "EventEnvelope.payload"),
        )


class HealthKind(str, Enum):
    """健康事件类型（observability 消费）。"""

    OK = "OK"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"
    RECOVERED = "RECOVERED"


@dataclass(frozen=True)
class HealthEvent:
    """健康/告警摘要事件（不含原始大 payload）。"""

    source: str
    kind: HealthKind
    message: str
    details: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source:
            raise InvalidDomainValue("HealthEvent.source 不能为空")
        if not self.message:
            raise InvalidDomainValue("HealthEvent.message 不能为空")

    def to_payload(self) -> Payload:
        payload: list[tuple[str, str]] = [
            ("source", self.source),
            ("kind", self.kind.value),
            ("message", self.message),
        ]
        payload.extend((f"detail.{i}", detail) for i, detail in enumerate(self.details))
        return tuple(payload)


@dataclass(frozen=True)
class OrderEvent:
    """订单/成交事实摘要事件。

    ``executed_qty`` 是已确认成交数量（部分成交恢复的依据）。
    """

    client_order_id: str
    symbol: str
    status: str
    executed_qty: Decimal
    event_time_ms: int

    def __post_init__(self) -> None:
        if not self.client_order_id:
            raise InvalidDomainValue("OrderEvent.client_order_id 不能为空")
        if self.executed_qty < 0:
            raise InvalidDomainValue(f"OrderEvent.executed_qty 不得为负: {self.client_order_id}")

    def to_payload(self) -> Payload:
        return (
            ("client_order_id", self.client_order_id),
            ("symbol", self.symbol),
            ("status", self.status),
            ("executed_qty", format(self.executed_qty, "f")),
            ("event_time_ms", str(self.event_time_ms)),
        )

    @classmethod
    def from_payload(cls, payload: Payload) -> OrderEvent:
        mapping = dict(payload)
        missing = [
            key
            for key in ("client_order_id", "symbol", "status", "executed_qty", "event_time_ms")
            if key not in mapping
        ]
        if missing:
            raise InvalidDomainValue(f"OrderEvent payload 缺字段: {', '.join(missing)}")
        time_raw = mapping["event_time_ms"]
        if not isinstance(time_raw, str) or not time_raw.lstrip("-").isdigit() or int(time_raw) < 0:
            raise InvalidDomainValue(f"OrderEvent event_time_ms 非法: {time_raw!r}")
        return cls(
            client_order_id=parse_str(mapping["client_order_id"], "client_order_id"),
            symbol=parse_str(mapping["symbol"], "symbol"),
            status=parse_str(mapping["status"], "status"),
            executed_qty=parse_decimal(mapping["executed_qty"], "executed_qty"),
            event_time_ms=int(time_raw),
        )
