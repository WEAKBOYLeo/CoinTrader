"""安全状态机：合法迁移 + 恢复前提（实施计划书 3.0 T3，AC-05）。

核心规则（开发设计文档 §7.3）：

- 加严（HALT_NEW_RISK / EMERGENCY_FLATTEN）任意状态都允许；放宽
  （回到 RUNNING）必须走显式 ``RECOVERY → RESUME_AFTER_CHECKS`` 两步，
  ``RESUME_AFTER_CHECKS`` 只能从 ``RECOVERY`` 发出（代表重新预检+对账
  已通过）。
- 删除停机文件不会自动恢复交易（状态机没有任何自动降级路径）。
- 触发 ``EMERGENCY_FLATTEN`` 必须 ``source == "manual"``（领域层与
  状态机双重强制）。
- 非法迁移抛出异常并保持原状态（状态对象不可变，不就地变更）。
"""

from __future__ import annotations

from ..domain.common import InvalidDomainValue
from ..domain.control import CommandKind, ControlCommand, SafetyState, SafetyStateKind

__all__ = ["SafetyStateMachine", "LEGAL_TRANSITIONS"]

#: 目标状态允许从哪些源状态迁移。
LEGAL_TRANSITIONS: dict[SafetyStateKind, frozenset[SafetyStateKind]] = {
    SafetyStateKind.STARTING: frozenset(
        {SafetyStateKind.STARTING, SafetyStateKind.STOPPED}
    ),
    SafetyStateKind.RUNNING: frozenset(
        {
            SafetyStateKind.STARTING,
            SafetyStateKind.RUNNING,
            SafetyStateKind.DEGRADED,
            SafetyStateKind.RECOVERY,
        }
    ),
    SafetyStateKind.DEGRADED: frozenset(
        {
            SafetyStateKind.STARTING,
            SafetyStateKind.RUNNING,
            SafetyStateKind.DEGRADED,
            SafetyStateKind.RECOVERY,
        }
    ),
    SafetyStateKind.RECOVERY: frozenset(
        {
            SafetyStateKind.STARTING,
            SafetyStateKind.RUNNING,
            SafetyStateKind.DEGRADED,
            SafetyStateKind.RECOVERY,
            SafetyStateKind.CLOSE_ONLY,
        }
    ),
    SafetyStateKind.CLOSE_ONLY: frozenset(
        {
            SafetyStateKind.STARTING,
            SafetyStateKind.RUNNING,
            SafetyStateKind.DEGRADED,
            SafetyStateKind.RECOVERY,
            SafetyStateKind.CLOSE_ONLY,
            SafetyStateKind.HALTED,
        }
    ),
    SafetyStateKind.HALTED: frozenset(
        {
            SafetyStateKind.RUNNING,
            SafetyStateKind.DEGRADED,
            SafetyStateKind.RECOVERY,
            SafetyStateKind.CLOSE_ONLY,
            SafetyStateKind.HALTED,
        }
    ),
    SafetyStateKind.EMERGENCY_FLATTEN: frozenset(
        {
            SafetyStateKind.STARTING,
            SafetyStateKind.RUNNING,
            SafetyStateKind.DEGRADED,
            SafetyStateKind.RECOVERY,
            SafetyStateKind.CLOSE_ONLY,
            SafetyStateKind.HALTED,
            SafetyStateKind.STOPPED,
            SafetyStateKind.EMERGENCY_FLATTEN,
        }
    ),
    SafetyStateKind.STOPPED: frozenset(
        {SafetyStateKind.STARTING, SafetyStateKind.RUNNING, SafetyStateKind.DEGRADED}
    ),
}

#: 控制命令 → 目标状态。
_COMMAND_TARGET: dict[CommandKind, SafetyStateKind] = {
    CommandKind.HALT_NEW_RISK: SafetyStateKind.CLOSE_ONLY,
    CommandKind.CLOSE_ONLY: SafetyStateKind.CLOSE_ONLY,
    CommandKind.RECOVERY: SafetyStateKind.RECOVERY,
    CommandKind.RESUME_AFTER_CHECKS: SafetyStateKind.RUNNING,
    CommandKind.EMERGENCY_FLATTEN: SafetyStateKind.EMERGENCY_FLATTEN,
}


class SafetyStateMachine:
    """安全状态迁移器（纯状态，事件通过 ``apply`` 注入）。"""

    def __init__(self, initial: SafetyState) -> None:
        self._state = initial

    @property
    def state(self) -> SafetyState:
        return self._state

    def can_transition(self, target: SafetyStateKind) -> bool:
        return self._state.state in LEGAL_TRANSITIONS.get(target, frozenset())

    def apply(self, command: ControlCommand, *, now_ms: int) -> SafetyState:
        """应用控制命令，返回新状态（非法迁移抛异常，状态不变）。"""
        target = _COMMAND_TARGET.get(command.command)
        if target is None:
            raise InvalidDomainValue(f"未知控制命令: {command.command}")
        if command.command is CommandKind.RESUME_AFTER_CHECKS and self._state.state is not SafetyStateKind.RECOVERY:
            raise InvalidDomainValue(
                "RESUME_AFTER_CHECKS 只能从 RECOVERY 状态发出（需先完成重新预检+对账）"
            )
        if command.command is CommandKind.EMERGENCY_FLATTEN and command.source != "manual":
            raise InvalidDomainValue("EMERGENCY_FLATTEN 必须由人工确认触发（source=manual）")
        if not self.can_transition(target):
            raise InvalidDomainValue(
                f"非法状态迁移: {self._state.state.value} → {target.value}"
                f"（reason={command.reason or 'n/a'}）"
            )
        self._state = SafetyState(
            target,
            reason=command.reason,
            changed_at_ms=now_ms,
            source=command.source,
        )
        return self._state

    def shutdown(self, *, reason: str, now_ms: int) -> SafetyState:
        """进程优雅停机（T4）：停机后状态机不再参与迁移。

        停机是进程级动作，不走控制命令通道；任何状态都可停机
        （STOPPED 幂等）。RECOVERY/HALTED 停机后不会自动降级为 RUNNING
        —— 进程退出后新进程从 RECOVERY 重新起步（启动预检+对账通过才 RUNNING）。
        """
        if self._state.state is SafetyStateKind.STOPPED:
            return self._state
        self._state = SafetyState(
            SafetyStateKind.STOPPED, reason=reason, changed_at_ms=now_ms, source="system"
        )
        return self._state
