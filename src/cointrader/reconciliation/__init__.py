"""对账门面（T4，AC-07）。

``ReconciliationFacade`` 统一封装 ``ExchangeStateSynchronizer``（交易所事实
capture + fills/funding 增量同步）与 ``Reconciler``（对账判定）。对账结果
是**应用 gate 输入**（RUNNING 门禁 / RECOVERY 恢复条件），不被 WebUI 或
PairExecutor 私藏。
"""

from .facade import GateOutcome, ReconciliationFacade

__all__ = ["GateOutcome", "ReconciliationFacade"]
