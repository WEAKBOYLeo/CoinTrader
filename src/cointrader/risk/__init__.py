"""风险内核与安全状态机（T3）。

- ``RiskKernel``：同步审批，输出带规则证据的 ``RiskDecision``（AC-04）。
- ``SafetyStateMachine``：合法迁移与恢复前提（AC-05）。
- ``RiskGateAdapter``：legacy ``RiskGate`` 兼容适配器（最终兼容闸门）。

禁止：访问网络、写账本、持有 transport。
"""

from .adapter import RiskGateAdapter, RiskRulesAdapter
from .kernel import RiskExposure, RiskKernel
from .state_machine import LEGAL_TRANSITIONS, SafetyStateMachine

__all__ = [
    "LEGAL_TRANSITIONS",
    "RiskExposure",
    "RiskGateAdapter",
    "RiskKernel",
    "RiskRulesAdapter",
    "SafetyStateMachine",
]
