"""T3：RiskGate 兼容闸门 + RiskGateAdapter 测试（AC-05/07）。

覆盖：开仓/减仓权限分离、停机只升不降、恢复必须预检+对账、
停机文件语义、adapter 状态映射与委托。
"""

from __future__ import annotations

import pytest

from cointrader.config import RiskConfig
from cointrader.domain.common import InvalidDomainValue
from cointrader.domain.control import SafetyStateKind
from cointrader.errors import LiveGateBlocked
from cointrader.execution.risk import Position, RiskManager, RiskState
from cointrader.execution.risk_gate import HaltState, RiskGate
from cointrader.risk.adapter import RiskGateAdapter

NOW = 1_800_003_000.0


def make_gate(kill_check=None) -> RiskGate:
    return RiskGate(RiskManager(RiskConfig()), kill_check=kill_check)


def make_state() -> RiskState:
    return RiskState(total_capital=10_000.0, available_balance=5_000.0)


def test_normal_allows_open_and_reduce():
    gate = make_gate()
    assert gate.allow_open("BTCUSDT", 100.0, make_state(), now=NOW).allowed
    assert gate.allow_reduce(now=NOW).allowed


def test_halt_blocks_open_but_not_reduce():
    gate = make_gate()
    gate.halt("对账不一致")
    assert gate.state is HaltState.HALT_NEW_RISK
    assert not gate.allow_open("BTCUSDT", 100.0, make_state(), now=NOW).allowed
    assert gate.allow_reduce(now=NOW).allowed  # 事故时减仓必须可行


def test_halt_only_escalates():
    gate = make_gate()
    gate.halt("限流", emergency=True)
    assert gate.state is HaltState.EMERGENCY_FLATTEN
    gate.halt("再限流")  # 不能降级
    assert gate.state is HaltState.EMERGENCY_FLATTEN


def test_recover_requires_preflight_and_reconciliation():
    gate = make_gate()
    gate.halt("对账不一致")
    with pytest.raises(LiveGateBlocked):
        gate.recover(reconciliation_ok=True, preflight_ok=False)
    with pytest.raises(LiveGateBlocked):
        gate.recover(reconciliation_ok=False, preflight_ok=True)
    gate.recover(reconciliation_ok=True, preflight_ok=True)
    assert gate.state is HaltState.NORMAL
    assert gate.is_recovered()
    assert gate.allow_open("BTCUSDT", 100.0, make_state(), now=NOW).allowed


def test_kill_file_forces_halt_and_blocks_recovery():
    gate = make_gate(kill_check=lambda: True)
    assert gate.state is HaltState.HALT_NEW_RISK  # 文件存在 = 至少 HALT_NEW_RISK
    with pytest.raises(LiveGateBlocked):
        gate.recover(reconciliation_ok=True, preflight_ok=True)


def test_unhedged_position_blocks_new_open():
    gate = make_gate()
    state = make_state()
    state.positions["BTCUSDT"] = Position(
        symbol="BTCUSDT", spot_qty=1.0, perp_qty=0.5, spot_price=100.0, perp_price=100.0
    )
    decision = gate.allow_open("BTCUSDT", 100.0, state, now=NOW)
    assert not decision.allowed
    assert gate.allow_reduce(now=NOW).allowed  # 但减仓/补腿仍可行


def test_adapter_state_mapping():
    for safety, halt in [
        (SafetyStateKind.RUNNING, HaltState.NORMAL),
        (SafetyStateKind.DEGRADED, HaltState.NORMAL),
        (SafetyStateKind.CLOSE_ONLY, HaltState.HALT_NEW_RISK),
        (SafetyStateKind.HALTED, HaltState.HALT_NEW_RISK),
        (SafetyStateKind.RECOVERY, HaltState.HALT_NEW_RISK),
        (SafetyStateKind.EMERGENCY_FLATTEN, HaltState.EMERGENCY_FLATTEN),
        (SafetyStateKind.STOPPED, HaltState.HALT_NEW_RISK),
    ]:
        assert RiskGateAdapter.to_halt(safety) is halt


def test_adapter_delegates():
    gate = make_gate()
    adapter = RiskGateAdapter(gate)
    assert adapter.safety_kind is SafetyStateKind.RUNNING
    assert adapter.allow_open("BTCUSDT", 100.0, make_state(), now=NOW).allowed
    gate.halt("测试停机")
    assert adapter.safety_kind is SafetyStateKind.CLOSE_ONLY
    assert not adapter.allow_open("BTCUSDT", 100.0, make_state(), now=NOW).allowed
    assert adapter.allow_reduce(now=NOW).allowed
    with pytest.raises(LiveGateBlocked):
        adapter.recover(reconciliation_ok=False, preflight_ok=True)
    adapter.recover(reconciliation_ok=True, preflight_ok=True)
    assert adapter.safety_kind is SafetyStateKind.RUNNING


def test_emergency_flatten_blocks_open():
    gate = make_gate()
    gate.halt("人工紧急平仓", emergency=True)
    assert not gate.allow_open("BTCUSDT", 100.0, make_state(), now=NOW).allowed
    assert gate.allow_reduce(now=NOW).allowed


def test_invalid_domain_value_type():
    """InvalidDomainValue 是领域异常（状态机/领域对象共用）。"""
    assert issubclass(InvalidDomainValue, Exception)
