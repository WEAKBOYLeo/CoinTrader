"""LedgerQueryService：只读 read model（T4）。

WebUI（``/api/state``、``/healthz``）、CLI（``live status/positions/
orders/trades``）与报告统一从这里读取 current projection；默认**隐藏
tombstone 旧持仓**。本服务不写账本、不访问交易所。
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from .ports import LedgerPort

__all__ = ["LedgerQueryService"]


class LedgerQueryService:
    """账本查询门面（注入 ``LedgerPort``，默认实现 = ``StateStore``）。"""

    def __init__(self, ledger: LedgerPort) -> None:
        self._ledger = ledger

    # -- current projection -------------------------------------------------

    def positions(self, *, include_tombstones: bool = False) -> list[dict[str, Any]]:
        """当前持仓投影。tombstone（已平/已废弃）默认不显示。"""
        return self._ledger.current_positions(include_tombstones=include_tombstones)

    def current_positions(self, *, include_tombstones: bool = False) -> list[dict[str, Any]]:
        """与 ``positions`` 等价的账本原名直通（T4.4：WebUI/CLI/reporting 用）。"""
        return self._ledger.current_positions(include_tombstones=include_tombstones)

    def open_pairs(self) -> list[dict[str, Any]]:
        return self._ledger.open_pairs()

    def expected_positions(self) -> dict[str, dict[str, Decimal]]:
        return self._ledger.expected_positions()

    def current_account(self) -> dict[str, Any] | None:
        return self._ledger.current_account()

    def runtime_state(self) -> dict[str, dict[str, Any]]:
        return self._ledger.runtime_state()

    # -- 历史 read model -----------------------------------------------------

    def orders_in_states(self, states: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
        """按状态集合查订单（状态集非空；空集无意义，调用方应显式指定）。"""
        if not states:
            raise ValueError("orders_in_states(states) 需要非空状态集合")
        return self._ledger.orders_in_states(states)

    def signal_decisions(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        symbol: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._ledger.signal_decisions(
            since_ms=since_ms, until_ms=until_ms, symbol=symbol, run_id=run_id
        )

    def run_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        return self._ledger.run_sessions(limit=limit)

    def latest_run_session(self) -> dict[str, Any] | None:
        return self._ledger.latest_run_session()

    def run_session(self, run_id: str) -> dict[str, Any] | None:
        return self._ledger.run_session(run_id)

    def online_stats(self, now_ms: int, *, heartbeat_fresh_ms: int = 120_000) -> dict[str, Any]:
        return self._ledger.online_stats(now_ms, heartbeat_fresh_ms=heartbeat_fresh_ms)

    def fills_for_orders(self, client_order_ids: Sequence[str]) -> list[dict[str, Any]]:
        return self._ledger.fills_for_orders(client_order_ids)

    def get_pair(self, pair_execution_id: str) -> dict[str, Any] | None:
        return self._ledger.get_pair(pair_execution_id)

    # -- T4.4：历史表只读查询面（WebUI/CLI/reporting 唯一查询端口） ----------

    def orders(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        symbol: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        return self._ledger.orders(
            since_ms=since_ms, until_ms=until_ms, symbol=symbol, limit=limit
        )

    def fills(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        symbol: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        return self._ledger.fills(
            since_ms=since_ms, until_ms=until_ms, symbol=symbol, limit=limit
        )

    def funding_cashflows(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        symbol: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        return self._ledger.funding_cashflows(
            since_ms=since_ms, until_ms=until_ms, symbol=symbol, limit=limit
        )

    def position_snapshots(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        symbol: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        return self._ledger.position_snapshots(
            since_ms=since_ms, until_ms=until_ms, symbol=symbol, limit=limit
        )

    def account_snapshots(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        return self._ledger.account_snapshots(
            since_ms=since_ms, until_ms=until_ms, limit=limit
        )

    def reconciliation_runs(
        self, *, since_ms: int | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        return self._ledger.reconciliation_runs(since_ms=since_ms, limit=limit)

    def exchange_events(
        self,
        *,
        since_ms: int | None = None,
        market: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        return self._ledger.exchange_events(
            since_ms=since_ms, market=market, limit=limit
        )

    def lease_holders(self) -> list[dict[str, Any]]:
        return self._ledger.lease_holders()

    def position_opened_ms(self, symbol: str) -> int | None:
        return self._ledger.position_opened_ms(symbol)

    def scan_epochs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return self._ledger.scan_epochs(limit=limit)

    def candidate_snapshots(
        self, epoch_id: str, *, limit: int = 1000
    ) -> list[dict[str, Any]]:
        return self._ledger.candidate_snapshots(epoch_id, limit=limit)

    def sync_cursors(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        return self._ledger.sync_cursors(limit=limit)

    def schema_version(self) -> int:
        return self._ledger.schema_version()

    def orders_for_pair(self, pair_execution_id: str) -> list[dict[str, Any]]:
        return self._ledger.orders_for_pair(pair_execution_id)

    def pair_executions(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        symbol: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        return self._ledger.pair_executions(
            since_ms=since_ms, until_ms=until_ms, symbol=symbol, limit=limit
        )

    def close(self) -> None:
        """释放底层存储（若实现有 close）；查询服务本身无状态。"""
        close = getattr(self._ledger, "close", None)
        if callable(close):
            close()
