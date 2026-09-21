"""LedgerPort：账本读模型协议面（T4）。

结构化协议：``execution.store.StateStore`` 天然满足（方法签名兼容）。
本模块不 import execution（保持 ledger 包对实现零依赖，实现由注入）。
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any, Protocol


class LedgerPort(Protocol):
    """账本/读模型最小能力面（current projection + 历史 read model）。"""

    def current_positions(self, *, include_tombstones: bool = False) -> list[dict[str, Any]]: ...

    def open_pairs(self) -> list[dict[str, Any]]: ...

    def orders_in_states(self, states: tuple[str, ...] | list[str]) -> list[dict[str, Any]]: ...

    def signal_decisions(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        symbol: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def run_sessions(self, limit: int = 10) -> list[dict[str, Any]]: ...

    def latest_run_session(self) -> dict[str, Any] | None: ...

    def run_session(self, run_id: str) -> dict[str, Any] | None: ...

    def online_stats(self, now_ms: int, *, heartbeat_fresh_ms: int = 120_000) -> dict[str, Any]: ...

    def current_account(self) -> dict[str, Any] | None: ...

    def runtime_state(self) -> dict[str, dict[str, Any]]: ...

    def expected_positions(self) -> dict[str, dict[str, Decimal]]: ...

    def fills_for_orders(self, client_order_ids: Sequence[str]) -> list[dict[str, Any]]: ...

    def get_pair(self, pair_execution_id: str) -> dict[str, Any] | None: ...
