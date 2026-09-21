"""T4.6：控制面与生命周期集成测试（计划 4.0 T4）。

覆盖生产装配面（LiveService + ServiceRunner + SafetyStateMachine）：

- 启动顺序 trace（application.lifecycle 固化阶段全绿 → RUNNING）；
- 风控闸门 halt（含 KILL_SWITCH 语义）→ ControlPublisher → 状态机
  CLOSE_ONLY → can_open=False、开仓被拒、无自动恢复；
- EMERGENCY_FLATTEN 仅人工（source=manual）；
- 闸门恢复 = 显式两步（CLOSE_ONLY → RECOVERY → RESUME_AFTER_CHECKS → RUNNING）；
- 用户流断线 → RECOVERY；流恢复 + 对账 → 自动回 RUNNING。

状态机本身的迁移矩阵见 tests/test_control_state.py（T1 领域层）。
"""

from __future__ import annotations

from typing import Any

from cointrader.domain.control import SafetyStateKind
from cointrader.execution.risk_gate import HaltState
from cointrader.live.service import ServiceState
from live_helpers import (
    LiveFakeData,
    make_live_rates,
    make_service,
    seed_current_positions,
)

SYMBOL = "BTCUSDT"


def _env(tmp_path: Any) -> dict[str, Any]:
    data = LiveFakeData({SYMBOL: make_live_rates(20, "0.0005")})
    env = make_service(tmp_path, data)
    env["svc"].run_id = "run-t4"
    return env


class TestT4ControlIntegration:
    def test_gate_halt_enter_close_only_and_blocks_new_risk(self, tmp_path: Any) -> None:
        env = _env(tmp_path)
        svc = env["svc"]

        r0 = svc.run_once()
        assert r0["state"] == "RUNNING"
        assert len(env["executor"].open_calls) == 1

        # 闸门停机（等价 KILL_SWITCH 文件存在：RiskGate.state 强制 HALT_NEW_RISK）
        svc.gate.halt("kill switch engaged")
        r1 = svc.run_once()
        assert r1["state"] == "HALTED"
        assert svc.safety_state.state is SafetyStateKind.CLOSE_ONLY
        assert svc.safety_state.allows_new_risk is False
        assert svc.can_open is False
        assert svc.state is ServiceState.HALTED

        # 无自动恢复：下一 tick 仍 HALTED（删除停机文件也必须显式 recover+预检）
        r2 = svc.run_once()
        assert r2["state"] == "HALTED"
        assert len(env["executor"].open_calls) == 1, "HALTED 期间不得开新仓"

    def test_gate_recovery_is_two_step_resume(self, tmp_path: Any) -> None:
        env = _env(tmp_path)
        svc = env["svc"]
        svc.run_once()  # RUNNING + 开仓一次

        svc.gate.halt("incident")
        svc.run_once()  # → CLOSE_ONLY（service HALTED）
        assert svc.safety_state.state is SafetyStateKind.CLOSE_ONLY

        # 显式恢复（recover 已含对账+预检证明）→ 两步回 RUNNING
        svc.gate.recover(reconciliation_ok=True, preflight_ok=True)
        r = svc.run_once()
        assert r["state"] == "RUNNING"
        assert svc.safety_state.state is SafetyStateKind.RUNNING
        # 控制面审计：RECOVERY 与 RESUME_AFTER_CHECKS 均留痕
        kinds = [c.command.value for c in svc._control.recent]  # noqa: SLF001
        assert "RECOVERY" in kinds and "RESUME_AFTER_CHECKS" in kinds

    def test_emergency_flatten_requires_manual_source(self, tmp_path: Any) -> None:
        env = _env(tmp_path)
        svc = env["svc"]
        # 紧急平仓态只可能由人工触发（领域层双重强制）
        svc.gate.state = HaltState.EMERGENCY_FLATTEN  # type: ignore[method-assign]
        r = svc.run_once()
        assert r["state"] == "HALTED"
        assert svc.safety_state.state is SafetyStateKind.EMERGENCY_FLATTEN
        assert env["executor"].open_calls == []
        flatten_cmds = [
            c for c in svc._control.recent if c.command.value == "EMERGENCY_FLATTEN"  # noqa: SLF001
        ]
        assert flatten_cmds and all(c.source == "manual" for c in flatten_cmds)

    def test_stream_disconnect_recovery_and_auto_clear(self, tmp_path: Any) -> None:
        from decimal import Decimal

        env = _env(tmp_path)
        svc = env["svc"]
        spot = env["spot"]
        futures = env["futures"]

        r0 = svc.run_once()
        assert r0["state"] == "RUNNING"

        # 用户流断线 → RECOVERY（禁止开新仓）
        class _StaleStream:
            market = "spot"
            generation = 1
            is_fresh = False

        svc.attach_streams([_StaleStream()])  # type: ignore[list-item]
        r1 = svc.run_once()
        assert r1["state"] == "RECOVERY"
        assert svc.safety_state.state is SafetyStateKind.RECOVERY

        # 流恢复 + 对账通过 → 自动回 RUNNING（恢复前提：对账+闸门）
        class _FreshStream:
            market = "spot"
            generation = 2
            is_fresh = True

        svc.attach_streams([_FreshStream()])  # type: ignore[list-item]
        svc._last_reconcile_ms = 0  # noqa: SLF001 测试时钟固定，强制触发 recovery_check
        # 模拟交易所确认持仓（避免恢复轮重复开仓）
        spot.balances_map["BTC"] = Decimal("0.01")
        futures.position_amt = Decimal("-0.01")
        seed_current_positions(env["store"], SYMBOL, "0.01", "-0.01")
        r2 = svc.run_once()
        assert r2["state"] == "RUNNING"
        assert svc.safety_state.state is SafetyStateKind.RUNNING

    def test_ledger_write_failure_enters_recovery_no_new_risk(self, tmp_path: Any) -> None:
        """账本写失败 → RECOVERY（fail closed），禁止继续新增风险。"""
        from cointrader.execution.store import StoreError

        env = _env(tmp_path)
        svc = env["svc"]
        r0 = svc.run_once()
        assert r0["state"] == "RUNNING"

        def boom(*a: Any, **k: Any) -> None:
            raise StoreError("模拟磁盘写失败")

        svc.store.record_signal_decision = boom  # type: ignore[method-assign]
        r1 = svc.run_once()
        assert r1["state"] == "RECOVERY"
        assert svc.safety_state.state is SafetyStateKind.RECOVERY
        assert len(env["executor"].open_calls) == 1, "RECOVERY 不得开新仓"

    def test_stop_marks_safety_stopped(self, tmp_path: Any) -> None:
        env = _env(tmp_path)
        svc = env["svc"]
        svc.run_once()
        svc.stop()
        assert svc.state is ServiceState.STOPPED
        assert svc.safety_state.state is SafetyStateKind.STOPPED
        assert svc.can_open is False
