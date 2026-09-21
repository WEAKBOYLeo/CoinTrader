"""RiskGate 兼容适配器：领域安全状态 ↔ legacy ``HaltState``（T3）。

审批链（开发设计文档 §7.3）：``RiskKernel``（新内核，规则证据）→
``RiskGate``（最终兼容闸门，开仓/减仓权限）→ ``guard``（认证/授权）→
Broker。新开风险必须同时通过内核与闸门；本适配器只做状态映射与
委托，不新增任何规则。
"""

from __future__ import annotations

from typing import cast

from ..domain.control import SafetyStateKind
from ..execution.risk import RiskManager, RiskState
from ..execution.risk_gate import GateDecision, HaltState, RiskGate
from .kernel import RiskExposure

__all__ = ["RiskGateAdapter", "RiskRulesAdapter"]

#: 领域安全状态 → legacy 闸门状态。
_TO_HALT: dict[SafetyStateKind, HaltState] = {
    SafetyStateKind.STARTING: HaltState.NORMAL,
    SafetyStateKind.RUNNING: HaltState.NORMAL,
    SafetyStateKind.DEGRADED: HaltState.NORMAL,
    SafetyStateKind.RECOVERY: HaltState.HALT_NEW_RISK,
    SafetyStateKind.CLOSE_ONLY: HaltState.HALT_NEW_RISK,
    SafetyStateKind.HALTED: HaltState.HALT_NEW_RISK,
    SafetyStateKind.EMERGENCY_FLATTEN: HaltState.EMERGENCY_FLATTEN,
    SafetyStateKind.STOPPED: HaltState.HALT_NEW_RISK,
}

#: legacy 闸门状态 → 领域安全状态（只读投影）。
_TO_SAFETY: dict[HaltState, SafetyStateKind] = {
    HaltState.NORMAL: SafetyStateKind.RUNNING,
    HaltState.HALT_NEW_RISK: SafetyStateKind.CLOSE_ONLY,
    HaltState.EMERGENCY_FLATTEN: SafetyStateKind.EMERGENCY_FLATTEN,
}


class RiskGateAdapter:
    """包裹 legacy ``RiskGate`` 的最终兼容闸门（委托，不改语义）。

    Args:
        gate: 已构造的 ``RiskGate``（持有 RiskManager/停机文件/审计）。
    """

    def __init__(self, gate: RiskGate) -> None:
        self.gate = gate

    @property
    def halt_state(self) -> HaltState:
        return self.gate.state

    @property
    def safety_kind(self) -> SafetyStateKind:
        return _TO_SAFETY[self.gate.state]

    @staticmethod
    def to_halt(state: SafetyStateKind) -> HaltState:
        """领域状态 → 闸门状态（供新 Control Plane 同步旧闸门）。"""
        return _TO_HALT[state]

    def allow_open(
        self,
        symbol: str,
        notional: float,
        state: RiskState,
        *,
        is_closing: bool = False,
        now: float | None = None,
    ) -> GateDecision:
        return self.gate.allow_open(
            symbol, notional, state, is_closing=is_closing, now=now
        )

    def allow_reduce(self, *, now: float | None = None) -> GateDecision:
        return self.gate.allow_reduce(now=now)

    def recover(self, *, reconciliation_ok: bool, preflight_ok: bool) -> None:
        self.gate.recover(reconciliation_ok=reconciliation_ok, preflight_ok=preflight_ok)


class RiskRulesAdapter:
    """``RiskRulesPort`` 的 execution 侧实现（包裹 ``RiskManager``）。

    内核只依赖端口（config + domain）；本适配器把 ``RiskManager`` 的
    限额/日亏/对冲/基差检查投影成纯数据 ``(规则名, 是否通过, 原因)``。
    """

    def __init__(self, manager: RiskManager) -> None:
        self._manager = manager

    def preflight_all(self, state: object) -> tuple[tuple[str, bool, str], ...]:
        verdicts = self._manager.preflight_all(cast(RiskState, state))
        all_ok = all(v.allowed for v in verdicts)
        worst = next((v.reason for v in verdicts if not v.allowed), "全部数值风控检查通过")
        return (("risk_preflight", all_ok, worst),)

    def check_order(self, symbol: str, notional: float, state: object) -> tuple[bool, str]:
        v = self._manager.check_order(symbol, notional, cast(RiskState, state))
        return (v.allowed, v.reason)

    def exposure(self, state: object) -> RiskExposure:
        st = cast(RiskState, state)
        return RiskExposure(
            total_exposure=st.total_exposure,
            symbol_exposure={s: p.notional for s, p in st.positions.items()},
        )
