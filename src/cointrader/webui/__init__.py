"""实时交易 WebUI（只读仪表盘，运行在 live run 进程内的独立守护线程）。

硬约束：
- 只读：只查询状态账本（独立 SQLite 连接）与 LiveService 内存快照，
  绝不调用下单、撤单、杠杆变更或任何写账本接口。
- 故障隔离：WebUI 线程中任何异常都被捕获并降级为 HTTP 500/告警日志，
  不允许影响主循环。
"""

from .server import LiveWebUI, build_payload

__all__ = ["LiveWebUI", "build_payload"]
