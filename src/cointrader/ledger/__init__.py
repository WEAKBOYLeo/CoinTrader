"""账本端口与查询服务（T4，AC-07/08）。

- ``LedgerPort``：StateStore（SQLite）提供的账本/读模型能力的最小协议面。
- ``LedgerQueryService``：WebUI/CLI/reporting 统一的**只读** read model
  入口；查询失败（``StoreError``）向上传播，由调用方做故障隔离
  （查询故障不得影响执行主循环）。
- 事件、current projection、read model 与唯一键继续使用同一事务
  （由 StateStore 保证，本层不拆事务）。
"""

from .ports import LedgerPort
from .service import LedgerQueryService

__all__ = ["LedgerPort", "LedgerQueryService"]
