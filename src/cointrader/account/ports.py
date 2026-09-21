"""账户状态端口（实施计划书 3.0 T2）。

- ``AccountQueryPort``：REST capture 端口。必须带
  ``snapshot_id/capture_start_ms/capture_end_ms/complete``（领域
  ``AccountSnapshot`` 契约）；不完整 bundle 不更新 current projection。
- ``UserStreamStatusPort``：用户流新鲜度端口（只读状态，不持有连接逻辑）。
"""

from __future__ import annotations

from typing import Protocol

from ..domain.account import AccountSnapshot

__all__ = ["AccountQueryPort", "UserStreamStatusPort"]


class AccountQueryPort(Protocol):
    """私有账户事实端口：一次 REST capture → 领域 ``AccountSnapshot``；
    另暴露最近一次完整 capture 的 current 投影（只读）。"""

    def capture(self) -> AccountSnapshot:
        """capture 失败不得抛裸异常后猜值；返回 ``complete=False`` 的领域错误快照。"""
        ...

    def current(self) -> AccountSnapshot | None:
        """最近一次完整 capture 的投影；``None`` = 尚无可信状态（未知状态）。

        查询层只读 current；返回 ``None``/过期时调用方必须按「未知状态」
        处理，不得用旧值/默认值放行新风险。
        """
        ...


class UserStreamStatusPort(Protocol):
    """用户流状态（断线 → 控制面降级，不直接下单/查询）。"""

    def is_connected(self) -> bool:
        ...

    def last_event_ms(self) -> int:
        ...
