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

    def open_pairs(self) -> list[dict[str, Any]]:
        return self._ledger.open_pairs()

    def expected_positions(self) -> dict[str, dict[str, Decimal]]:
        return self._ledger.expected_positions()

    def current_account(self) -> dict[str, Any] | None:
        return self._ledger.current_account()

    def runtime_state(self) -> dict[str, dict[str, Any]]:
        return self._ledger.runtime_state()

    # -- 历史 read model -----------------------------------------------------

    def orders(self, states: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
        """按状态集合查订单（状态集非空；空集无意义，调用方应显式指定）。"""
        if not states:
            raise ValueError("orders(states) 需要非空状态集合")
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
