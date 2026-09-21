"""账户状态投影：capture → current projection（实施计划书 3.0 T2）。

硬约束（AC-08）：

- **完整 capture 才更新 current projection**；不完整（``complete=False``）
  不覆盖可信状态，只作为本轮「账户未知」事实返回。
- 写入账本经 ``LedgerWriter`` 注入（生产 = ``StateStore`` v2 单事务
  snapshot group + current projection，由 application 层组装）；
  本模块不 import 数据库/执行层。
- 查询层只读 ``current``；账户失败时调用方必须按「未知状态」处理，
  不得用旧值/默认值放行新风险。
"""

from __future__ import annotations

from typing import Protocol

from ..domain.account import AccountSnapshot
from .ports import AccountQueryPort

__all__ = ["AccountProjector", "LedgerWriter"]


class LedgerWriter(Protocol):
    """账户快照账本写入端（单事务 group + current projection）。"""

    def save_complete(self, snapshot: AccountSnapshot) -> None:
        ...


class AccountProjector:
    """以同 snapshot id 更新 current projection 的账户状态机。"""

    def __init__(
        self,
        query: AccountQueryPort,
        writer: LedgerWriter | None = None,
    ) -> None:
        self._query = query
        self._writer = writer
        self._current: AccountSnapshot | None = None

    def refresh(self) -> AccountSnapshot:
        """执行一次 capture。完整 → 更新 current projection 并写账本；
        不完整 → 原样返回（不覆盖可信状态、不写账本）。"""
        snapshot = self._query.capture()
        if not snapshot.complete:
            return snapshot
        self._current = snapshot
        if self._writer is not None:
            self._writer.save_complete(snapshot)
        return snapshot

    @property
    def current(self) -> AccountSnapshot | None:
        """当前可信投影；None = 尚无完整 capture（未知状态）。"""
        return self._current

    def freshness_ok(self, now_ms: int, max_age_ms: int) -> bool:
        """current projection 新鲜度：capture 完成时间距 now 不超过上限。"""
        current = self._current
        if current is None or max_age_ms < 0:
            return False
        return 0 <= now_ms - current.capture_end_ms <= max_age_ms
