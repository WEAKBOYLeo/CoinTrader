"""应用编排层（T4，AC-07）。

``ServiceRunner`` 是 LiveService 生命周期（tick/恢复/心跳/看门狗/停机）
的唯一编排入口；查询层（WebUI/CLI/reporting）只经 ledger read model
读取 current projection。
"""

from .runner import ServiceRunner

__all__ = ["ServiceRunner"]
