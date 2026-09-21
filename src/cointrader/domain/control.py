"""控制面领域契约：控制命令与安全状态。

硬约束（实施计划书 §4/§5）：

- ``ControlCommand`` 只能进入状态机，不直接包含任意下单调用
  （结构上无 broker 字段）。
- ``SafetyState`` 由 ControlPlane 唯一拥有；单向升级，
  降级（恢复）必须显式恢复（重新预检+对账+显式确认，由 application 层执行）；
  kill switch 不能被代码忽略。
- 只有 ``RUNNING`` 允许新增风险（``allows_new_risk``）；
  ``can_open`` 还须叠加数据新鲜/账户完整/对账/账本同步/RiskGate 等
  其它闸门（由 application 层计算，本枚举只是必要条件）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from .common import (
    InvalidDomainValue,
    parse_enum,
    parse_int_ms,
    parse_optional,
    parse_str,
)

__all__ = [
    "CommandKind",
    "ControlCommand",
    "SafetyState",
    "SafetyStateKind",
]


class CommandKind(str, Enum):
    """控制命令类型。"""

    HALT_NEW_RISK = "HALT_NEW_RISK"
    RECOVERY = "RECOVERY"
    CLOSE_ONLY = "CLOSE_ONLY"
    EMERGENCY_FLATTEN = "EMERGENCY_FLATTEN"
    RESUME_AFTER_CHECKS = "RESUME_AFTER_CHECKS"


@dataclass(frozen=True)
class ControlCommand:
    """控制面命令：由 watchdog/对账/风控/人工入口创建。"""

    command: CommandKind
    reason: str
    source: str
    issued_at_ms: int
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.reason:
            raise InvalidDomainValue("ControlCommand.reason 不能为空")
        if not self.source:
            raise InvalidDomainValue("ControlCommand.source 不能为空")
        if self.command is CommandKind.EMERGENCY_FLATTEN and self.source != "manual":
            raise InvalidDomainValue(
                "EMERGENCY_FLATTEN 仅允许人工明确触发（source 必须为 'manual'）"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "command": self.command.value,
            "reason": self.reason,
            "source": self.source,
            "issued_at_ms": self.issued_at_ms,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ControlCommand:
        return cls(
            command=parse_enum(CommandKind, data.get("command"), "ControlCommand.command"),
            reason=parse_str(data.get("reason"), "ControlCommand.reason"),
            source=parse_str(data.get("source"), "ControlCommand.source"),
            issued_at_ms=parse_int_ms(
                data.get("issued_at_ms"), "ControlCommand.issued_at_ms"
            ),
            correlation_id=parse_optional(
                data.get("correlation_id"), "ControlCommand.correlation_id", parse_str
            ),
        )


class SafetyStateKind(str, Enum):
    """运行安全状态（展示用大写枚举；外部配置运行模式仍为小写）。"""

    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    RECOVERY = "RECOVERY"
    CLOSE_ONLY = "CLOSE_ONLY"
    HALTED = "HALTED"
    EMERGENCY_FLATTEN = "EMERGENCY_FLATTEN"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class SafetyState:
    """当前安全状态：枚举 + 原因 + 变更时间 + 来源。

    状态迁移规则（单向升级、显式恢复）由 ``SafetyStateMachine``（risk 包）
    在 T3 实现；本对象只承载状态事实。
    """

    state: SafetyStateKind
    reason: str
    changed_at_ms: int
    source: str = "system"

    def __post_init__(self) -> None:
        if not self.reason:
            raise InvalidDomainValue("SafetyState.reason 不能为空")

    @property
    def allows_new_risk(self) -> bool:
        """新增风险的状态必要条件（非充分条件）。"""
        return self.state is SafetyStateKind.RUNNING

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "reason": self.reason,
            "changed_at_ms": self.changed_at_ms,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SafetyState:
        return cls(
            state=parse_enum(SafetyStateKind, data.get("state"), "SafetyState.state"),
            reason=parse_str(data.get("reason"), "SafetyState.reason"),
            changed_at_ms=parse_int_ms(
                data.get("changed_at_ms"), "SafetyState.changed_at_ms"
            ),
            source=parse_str(data.get("source"), "SafetyState.source"),
        )
