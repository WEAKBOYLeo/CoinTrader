"""T3：安全状态机 + Control Plane 契约测试（AC-05）。

覆盖：合法迁移、非法迁移拒绝（状态不变）、恢复前提（RECOVERY→RESUME）、
紧急平仓必须人工、ControlPublisher/HealthPublisher 只发布不执行。
"""

from __future__ import annotations

import pytest

from cointrader.domain.common import InvalidDomainValue
from cointrader.domain.control import CommandKind, ControlCommand, SafetyState, SafetyStateKind
from cointrader.domain.events import HealthEvent, HealthKind
from cointrader.observability.control import ControlPublisher
from cointrader.observability.events import HealthPublisher
from cointrader.risk.state_machine import SafetyStateMachine

NOW = 1_800_001_000_000


def state(kind: SafetyStateKind) -> SafetyState:
    return SafetyState(kind, reason=f"seed-{kind.value}", changed_at_ms=NOW, source="system")


def cmd(kind: CommandKind, source: str = "system") -> ControlCommand:
    return ControlCommand(command=kind, reason="t3", source=source, issued_at_ms=NOW)


def test_escalation_from_any_state_is_legal():
    for start in (
        SafetyStateKind.STARTING,
        SafetyStateKind.RUNNING,
        SafetyStateKind.DEGRADED,
        SafetyStateKind.RECOVERY,
        SafetyStateKind.CLOSE_ONLY,
        SafetyStateKind.HALTED,
    ):
        sm = SafetyStateMachine(state(start))
        sm.apply(cmd(CommandKind.HALT_NEW_RISK), now_ms=NOW)
        assert sm.state.state is SafetyStateKind.CLOSE_ONLY


def test_resume_only_from_recovery():
    sm = SafetyStateMachine(state(SafetyStateKind.CLOSE_ONLY))
    with pytest.raises(InvalidDomainValue):
        sm.apply(cmd(CommandKind.RESUME_AFTER_CHECKS), now_ms=NOW)
    assert sm.state.state is SafetyStateKind.CLOSE_ONLY  # 状态不变

    sm.apply(cmd(CommandKind.RECOVERY), now_ms=NOW)
    assert sm.state.state is SafetyStateKind.RECOVERY
    sm.apply(cmd(CommandKind.RESUME_AFTER_CHECKS), now_ms=NOW)
    assert sm.state.state is SafetyStateKind.RUNNING
    assert sm.state.reason == "t3"


def test_emergency_flatten_requires_manual():
    sm = SafetyStateMachine(state(SafetyStateKind.RUNNING))
    with pytest.raises(InvalidDomainValue):
        ControlCommand(
            command=CommandKind.EMERGENCY_FLATTEN,
            reason="auto",
            source="watchdog",
            issued_at_ms=NOW,
        )
    sm.apply(cmd(CommandKind.EMERGENCY_FLATTEN, source="manual"), now_ms=NOW)
    assert sm.state.state is SafetyStateKind.EMERGENCY_FLATTEN
    assert sm.state.source == "manual"


def test_stopped_cannot_deescalate():
    sm = SafetyStateMachine(state(SafetyStateKind.STOPPED))
    with pytest.raises(InvalidDomainValue):
        sm.apply(cmd(CommandKind.HALT_NEW_RISK), now_ms=NOW)
    with pytest.raises(InvalidDomainValue):
        sm.apply(cmd(CommandKind.RESUME_AFTER_CHECKS), now_ms=NOW)
    assert sm.state.state is SafetyStateKind.STOPPED


def test_no_auto_recovery_path_exists():
    """删除停机文件不自动恢复：没有任何命令能跳过 RECOVERY 直接回 RUNNING。"""
    sm = SafetyStateMachine(state(SafetyStateKind.HALTED))
    with pytest.raises(InvalidDomainValue):
        sm.apply(cmd(CommandKind.RESUME_AFTER_CHECKS), now_ms=NOW)


def test_can_transition_reports_targets():
    sm = SafetyStateMachine(state(SafetyStateKind.RUNNING))
    assert sm.can_transition(SafetyStateKind.CLOSE_ONLY)
    assert sm.can_transition(SafetyStateKind.RECOVERY)
    assert not sm.can_transition(SafetyStateKind.STARTING)


# -- ControlPublisher ---------------------------------------------------------


def test_publisher_routes_commands_to_state_machine():
    sm = SafetyStateMachine(state(SafetyStateKind.RUNNING))
    seen: list[ControlCommand] = []
    pub = ControlPublisher(subscribers=[lambda c: (sm.apply(c, now_ms=NOW), seen.append(c))])
    pub.halt_new_risk(reason="对账不一致", source="reconcile")
    assert sm.state.state is SafetyStateKind.CLOSE_ONLY
    assert len(seen) == 1
    assert seen[0].command is CommandKind.HALT_NEW_RISK
    assert pub.recent[0].source == "reconcile"


def test_publisher_emergency_force_manual():
    pub = ControlPublisher()
    cmd_obj = pub.emergency_flatten(reason="人工紧急平仓")
    assert cmd_obj.source == "manual"
    assert cmd_obj.command is CommandKind.EMERGENCY_FLATTEN


def test_publisher_bad_subscriber_does_not_break_others():
    got: list[ControlCommand] = []

    def bad(_c: ControlCommand) -> None:
        raise RuntimeError("subscriber failure")

    pub = ControlPublisher(subscribers=[bad, got.append])
    pub.recovery(reason="重新预检", source="manual")
    assert len(got) == 1  # 异常订阅者不阻塞后续订阅者


# -- HealthPublisher ----------------------------------------------------------


def test_health_publisher_only_publishes():
    got: list[HealthEvent] = []
    pub = HealthPublisher(subscribers=[got.append])
    ev = pub.publish(
        HealthKind.DEGRADED,
        source="reconcile",
        message="对账差异: BTCUSDT",
        details=("diff", "0.01"),
    )
    assert got == [ev]
    assert ev.kind is HealthKind.DEGRADED
    # 发布器本身没有任何下单/执行接口
    assert not hasattr(pub, "place_order")
    assert not hasattr(pub, "submit")
