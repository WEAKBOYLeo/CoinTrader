"""SQLite 事件账本与快照（开发设计文档 §5.2）。

原则：

1. **先写本地 intent，再提交交易所订单**。本地账本是唯一恢复依据。
2. 每次状态变更**追加事件**（exchange_events），并更新当前快照表。
3. 交易所订单 id、client order id、fill id 建唯一索引 ——
   重复事件/重复下单在数据库层面就被挡住。
4. 所有时间 UTC 毫秒。Decimal 存 TEXT。
5. **数据库不保存 API Secret**；日志不保存签名和完整认证请求。
6. 单实例锁通过 ``service_leases`` 表实现，防止两个进程同时管理同一账户。

线程安全：单连接 + 锁。本系统只有一个执行进程，WAL 模式足够。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import Any

from .migrations import (
    SCHEMA_VERSION,
    MigrationError,
    apply_migrations,
    schema_version_of,
)
from .models import (
    AccountSnapshot,
    Fill,
    Order,
    OrderIntent,
    PairExecution,
    PositionSnapshot,
    ReconciliationResult,
)

logger = logging.getLogger(__name__)

__all__ = ["StateStore", "StoreError", "LeaseConflict", "SCHEMA_VERSION"]


class StoreError(Exception):
    """账本写入/读取失败。**写入失败必须进入 RECOVERY**（文档 §8.2）。"""


class LeaseConflict(Exception):
    """单实例锁被其他持有者占用。"""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS execution_intents (
    intent_id TEXT PRIMARY KEY,
    pair_execution_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    market TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quantity TEXT NOT NULL,
    price TEXT,
    reduce_only INTEGER NOT NULL DEFAULT 0,
    is_closing INTEGER NOT NULL DEFAULT 0,
    reason TEXT NOT NULL DEFAULT '',
    strategy_version TEXT NOT NULL DEFAULT '',
    signal_time_ms INTEGER,
    created_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS pair_executions (
    pair_execution_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    target_notional TEXT NOT NULL,
    hedge_ratio TEXT,
    residual_qty TEXT,
    error TEXT NOT NULL DEFAULT '',
    strategy_version TEXT NOT NULL DEFAULT '',
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pair_status ON pair_executions(status);

CREATE TABLE IF NOT EXISTS orders (
    client_order_id TEXT PRIMARY KEY,
    exchange_order_id TEXT,
    symbol TEXT NOT NULL,
    market TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quantity TEXT NOT NULL,
    executed_qty TEXT NOT NULL DEFAULT '0',
    avg_price TEXT,
    state TEXT NOT NULL,
    reduce_only INTEGER NOT NULL DEFAULT 0,
    updated_ms INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_exchange_id
    ON orders(exchange_order_id) WHERE exchange_order_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_orders_state ON orders(state);

CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    client_order_id TEXT NOT NULL,
    exchange_order_id TEXT,
    symbol TEXT NOT NULL,
    market TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity TEXT NOT NULL,
    price TEXT NOT NULL,
    fee_asset TEXT NOT NULL,
    fee_amount TEXT NOT NULL,
    ts_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(client_order_id);

CREATE TABLE IF NOT EXISTS position_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    spot_qty TEXT NOT NULL,
    perp_qty TEXT NOT NULL,
    spot_price TEXT,
    perp_price TEXT,
    basis_pct TEXT,
    source TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS account_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    source TEXT NOT NULL,
    available_balance TEXT,
    wallet_balance TEXT,
    equity TEXT,
    unrealized_pnl TEXT,
    margin_ratio TEXT
);

CREATE TABLE IF NOT EXISTS exchange_events (
    event_fp TEXT PRIMARY KEY,
    market TEXT NOT NULL,
    event_type TEXT NOT NULL,
    exchange_ts INTEGER,
    recv_ts INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON exchange_events(recv_ts);

CREATE TABLE IF NOT EXISTS risk_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    allowed INTEGER NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    ts_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS reconciliation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    consistent INTEGER NOT NULL,
    can_open INTEGER NOT NULL,
    mismatches TEXT NOT NULL DEFAULT '',
    repaired TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    details TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS service_leases (
    lease_name TEXT PRIMARY KEY,
    pid INTEGER NOT NULL,
    holder TEXT NOT NULL,
    acquired_ms INTEGER NOT NULL,
    expires_ms INTEGER NOT NULL
);
"""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _dec(value: Decimal | int | float | str | None) -> str | None:
    if value is None:
        return None
    return format(Decimal(str(value)), "f")


#: 含 symbol 列的表（只读查询过滤器使用）
_TABLES_WITH_SYMBOL = frozenset({
    "execution_intents",
    "pair_executions",
    "orders",
    "fills",
    "position_snapshots",
    "funding_cashflows",
    "signal_decisions",
})

#: 含 run_id 列的表
_TABLES_WITH_RUN_ID = frozenset({
    "pair_executions",
    "orders",
    "fills",
    "position_snapshots",
    "account_snapshots",
    "funding_cashflows",
    "pnl_ledger",
    "signal_decisions",
})


@dataclass(frozen=True, slots=True)
class Lease:
    name: str
    holder: str
    pid: int


class StateStore:
    """SQLite 事件账本。

    Args:
        db_path: 数据库文件路径（父目录自动创建）。
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        try:
            self._conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
            # 版本化迁移（§8.2）：旧库无损升级；失败拒绝启动
            with self._lock:
                apply_migrations(self._conn)
        except MigrationError as exc:
            raise StoreError(f"Schema 迁移失败，拒绝启动: {exc}") from exc
        except sqlite3.Error as exc:
            raise StoreError(f"无法初始化账本 {self.db_path}: {exc}") from exc

    # -- 生命周期 -----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> StateStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- 内部 ---------------------------------------------------------------

    def _execute(self, sql: str, args: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        try:
            with self._lock:
                return self._conn.execute(sql, args)
        except sqlite3.IntegrityError:
            # 唯一约束冲突是幂等语义（record_fill/record_event 依赖它判断重复事件），
            # 必须原样上抛，不能包成 StoreError
            raise
        except sqlite3.Error as exc:
            raise StoreError(f"账本写入失败: {exc}\nSQL: {sql}") from exc

    # -- 意图 ---------------------------------------------------------------

    def record_intent(self, intent: OrderIntent) -> bool:
        """记录策略意图。**先于交易所提交**。返回是否为新记录。"""
        try:
            with self._lock:
                existing = self._conn.execute(
                    "SELECT 1 FROM execution_intents WHERE intent_id = ?", (intent.intent_id,)
                ).fetchone()
            if existing:
                return False
            self._execute(
                "INSERT INTO execution_intents (intent_id, pair_execution_id, symbol, side, market,"
                " order_type, quantity, price, reduce_only, is_closing, reason, strategy_version,"
                " signal_time_ms, created_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    intent.intent_id,
                    intent.pair_execution_id,
                    intent.symbol,
                    intent.side.value,
                    intent.market.value,
                    intent.order_type.value,
                    _dec(intent.quantity),
                    _dec(intent.price),
                    int(intent.reduce_only),
                    int(intent.is_closing),
                    intent.reason,
                    intent.strategy_version,
                    intent.signal_time_ms,
                    _now_ms(),
                ),
            )
            return True
        except StoreError:
            raise
        except sqlite3.Error as exc:
            raise StoreError(f"意图写入失败: {exc}") from exc

    def open_intent_count(self) -> int:
        """未被处理的意图数（用于重复意图检查的辅助指标）。"""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM execution_intents").fetchone()
        return int(row[0])

    # -- Pair 执行 ----------------------------------------------------------

    def upsert_pair(self, pair: PairExecution) -> None:
        self._execute(
            "INSERT INTO pair_executions (pair_execution_id, symbol, kind, status, target_notional,"
            " hedge_ratio, residual_qty, error, strategy_version, created_ms, updated_ms,"
            " run_id, open_or_close_reason, signal_decision_id, signal_ts_ms, decision_ts_ms,"
            " submit_ts_ms, exchange_confirmed_ts_ms, completed_ts_ms,"
            " actual_spot_notional, actual_perp_notional, funding_pnl, fee_pnl, basis_pnl,"
            " realized_pnl, unrealized_pnl, net_pnl)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(pair_execution_id) DO UPDATE SET"
            " status=excluded.status, hedge_ratio=excluded.hedge_ratio,"
            " residual_qty=excluded.residual_qty, error=excluded.error, updated_ms=excluded.updated_ms,"
            " run_id=excluded.run_id, open_or_close_reason=excluded.open_or_close_reason,"
            " signal_decision_id=excluded.signal_decision_id,"
            " completed_ts_ms=COALESCE(excluded.completed_ts_ms, completed_ts_ms),"
            " funding_pnl=excluded.funding_pnl, fee_pnl=excluded.fee_pnl,"
            " basis_pnl=excluded.basis_pnl, realized_pnl=excluded.realized_pnl,"
            " unrealized_pnl=excluded.unrealized_pnl, net_pnl=excluded.net_pnl",
            (
                pair.pair_execution_id,
                pair.symbol,
                pair.kind,
                pair.status,
                _dec(pair.target_notional),
                _dec(pair.hedge_ratio),
                _dec(pair.residual_qty),
                pair.error,
                pair.strategy_version,
                pair.created_ms,
                pair.updated_ms,
                pair.run_id or None,
                pair.open_or_close_reason or None,
                pair.signal_decision_id or None,
                pair.signal_ts_ms or None,
                pair.decision_ts_ms or None,
                pair.submit_ts_ms or None,
                pair.exchange_confirmed_ts_ms or None,
                pair.completed_ts_ms or None,
                _dec(pair.actual_spot_notional),
                _dec(pair.actual_perp_notional),
                _dec(pair.funding_pnl),
                _dec(pair.fee_pnl),
                _dec(pair.basis_pnl),
                _dec(pair.realized_pnl),
                _dec(pair.unrealized_pnl),
                _dec(pair.net_pnl),
            ),
        )

    def update_pair_pnl(self, pair_execution_id: str, **pnl: Decimal | None) -> None:
        """报告层回填 pair 的 PnL 字段（幂等覆盖）。"""
        sets: list[str] = []
        args: list[Any] = []
        for name in ("funding_pnl", "fee_pnl", "basis_pnl", "realized_pnl",
                     "unrealized_pnl", "net_pnl"):
            if name in pnl:
                sets.append(f"{name}=?")
                args.append(_dec(pnl[name]))  # type: ignore[arg-type]
        if not sets:
            return
        args.append(pair_execution_id)
        self._execute(
            f"UPDATE pair_executions SET {', '.join(sets)} WHERE pair_execution_id = ?",  # noqa: S608
            tuple(args),
        )

    def active_pair_for_symbol(self, symbol: str) -> str | None:
        """该 symbol 是否存在未终态的 pair（防止同一 symbol 重复开仓）。"""
        terminal = ("COMPLETE", "COMPENSATED", "FLATTENED", "FAILED", "HALTED")
        placeholders = ",".join("?" * len(terminal))
        with self._lock:
            row = self._conn.execute(
                f"SELECT pair_execution_id FROM pair_executions"  # noqa: S608
                # 仅占位符由 ? 拼接，值全部参数绑定，无注入面
                f" WHERE symbol = ? AND status NOT IN ({placeholders}) LIMIT 1",
                (symbol, *terminal),
            ).fetchone()
        return row[0] if row else None

    # -- 订单 ---------------------------------------------------------------

    def upsert_order(self, order: Order) -> bool:
        """记录/更新订单。返回是否为**新** client_order_id。

        重复 clientOrderId 直接更新 —— 这是「重复 intent 不产生重复订单」
        的数据库层保证。
        """
        try:
            with self._lock:
                existing = self._conn.execute(
                    "SELECT 1 FROM orders WHERE client_order_id = ?", (order.client_order_id,)
                ).fetchone()
            if existing:
                # 交易所订单 id 变化时用 COALESCE 保留已有值
                self._execute(
                    "UPDATE orders SET exchange_order_id = COALESCE(?, exchange_order_id),"
                    " executed_qty = ?, avg_price = ?, state = ?, updated_ms = ?,"
                    " terminal_ts_ms = COALESCE(?, terminal_ts_ms),"
                    " exchange_update_ts_ms = COALESCE(?, exchange_update_ts_ms),"
                    " failure_class = COALESCE(?, failure_class),"
                    " raw_status_summary = COALESCE(?, raw_status_summary)"
                    " WHERE client_order_id = ?",
                    (
                        order.exchange_order_id,
                        _dec(order.executed_qty),
                        _dec(order.avg_price),
                        order.state,
                        order.updated_ms,
                        order.terminal_ts_ms,
                        order.updated_ms or None,
                        order.failure_class or None,
                        order.raw_status_summary or None,
                        order.client_order_id,
                    ),
                )
                return False
            self._execute(
                "INSERT INTO orders (client_order_id, exchange_order_id, symbol, market, side,"
                " order_type, quantity, executed_qty, avg_price, state, reduce_only, updated_ms,"
                " run_id, pair_execution_id, intent_id, submit_ts_ms, ack_ts_ms, terminal_ts_ms,"
                " exchange_update_ts_ms, failure_class, raw_status_summary)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    order.client_order_id,
                    order.exchange_order_id,
                    order.symbol,
                    order.market.value,
                    order.side.value,
                    order.order_type.value,
                    _dec(order.quantity),
                    _dec(order.executed_qty),
                    _dec(order.avg_price),
                    order.state,
                    int(order.reduce_only),
                    order.updated_ms,
                    order.run_id or None,
                    order.pair_execution_id or None,
                    order.intent_id or None,
                    order.submit_ts_ms or None,
                    order.ack_ts_ms or None,
                    order.terminal_ts_ms,
                    order.updated_ms or None,
                    order.failure_class or None,
                    order.raw_status_summary or None,
                ),
            )
            return True
        except sqlite3.IntegrityError as exc:
            # exchange_order_id 唯一索引冲突：该交易所订单已被别的 client id 记录
            raise StoreError(f"交易所订单 id 冲突（client={order.client_order_id}）: {exc}") from exc

    def get_order(self, client_order_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._conn.row_factory = None
            row = self._conn.execute(
                "SELECT client_order_id, exchange_order_id, symbol, market, side, order_type,"
                " quantity, executed_qty, avg_price, state, reduce_only, updated_ms"
                " FROM orders WHERE client_order_id = ?",
                (client_order_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(zip(
            ("client_order_id", "exchange_order_id", "symbol", "market", "side", "order_type",
             "quantity", "executed_qty", "avg_price", "state", "reduce_only", "updated_ms"),
            row, strict=False,
        ))

    def orders_in_states(self, states: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
        placeholders = ",".join("?" * len(states))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT client_order_id, exchange_order_id, symbol, market, state"  # noqa: S608
                # 仅占位符由 ? 拼接，值全部参数绑定，无注入面
                f" FROM orders WHERE state IN ({placeholders})",
                tuple(states),
            ).fetchall()
        return [
            {"client_order_id": r[0], "exchange_order_id": r[1], "symbol": r[2], "market": r[3], "state": r[4]}
            for r in rows
        ]

    def expected_positions(self) -> dict[str, dict[str, Decimal]]:
        """从 fills 汇总本地期望持仓：{symbol: {SPOT: qty, PERP: qty(负=空)}}。"""
        with self._lock:
            rows = self._conn.execute("SELECT symbol, market, side, quantity FROM fills").fetchall()
        out: dict[str, dict[str, Decimal]] = {}
        for symbol, market, side, qty_text in rows:
            qty = Decimal(str(qty_text))
            if side == "SELL":
                qty = -qty
            slot = out.setdefault(symbol, {"SPOT": Decimal("0"), "PERP": Decimal("0")})
            slot[market] = slot.get(market, Decimal("0")) + qty
        return out

    # -- 成交 ---------------------------------------------------------------

    def record_fill(self, fill: Fill) -> bool:
        """记录成交。fill_id 唯一 —— 重复事件返回 False（幂等，§8.3）。"""
        try:
            self._execute(
                "INSERT INTO fills (fill_id, client_order_id, exchange_order_id, symbol, market,"
                " side, quantity, price, fee_asset, fee_amount, ts_ms, run_id, pair_execution_id,"
                " intent_id, exchange_trade_id, quote_qty, maker_taker, exchange_ts_ms,"
                " received_ts_ms)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    fill.fill_id,
                    fill.client_order_id,
                    fill.exchange_order_id,
                    fill.symbol,
                    fill.market.value,
                    fill.side.value,
                    _dec(fill.quantity),
                    _dec(fill.price),
                    fill.fee_asset,
                    _dec(fill.fee_amount),
                    fill.ts_ms,
                    fill.run_id or None,
                    fill.pair_execution_id or None,
                    fill.intent_id or None,
                    fill.exchange_trade_id or None,
                    _dec(fill.quote_qty),
                    fill.maker_taker or None,
                    fill.exchange_ts_ms or None,
                    fill.received_ts_ms or None,
                ),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    # -- 快照 ---------------------------------------------------------------

    def record_position_snapshot(self, snap: PositionSnapshot) -> None:
        self._execute(
            "INSERT INTO position_snapshots (ts_ms, symbol, spot_qty, perp_qty, spot_price, perp_price,"
            " basis_pct, source, run_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                snap.ts_ms,
                snap.symbol,
                _dec(snap.spot_qty),
                _dec(snap.perp_qty),
                _dec(snap.spot_price),
                _dec(snap.perp_price),
                _dec(snap.basis_pct),
                snap.source,
                snap.run_id or None,
            ),
        )

    def record_account_snapshot(self, snap: AccountSnapshot) -> None:
        self._execute(
            "INSERT INTO account_snapshots (ts_ms, source, available_balance, wallet_balance,"
            " equity, unrealized_pnl, margin_ratio, run_id) VALUES (?,?,?,?,?,?,?,?)",
            (
                snap.ts_ms,
                snap.source,
                _dec(snap.available_balance),
                _dec(snap.wallet_balance),
                _dec(snap.equity),
                _dec(snap.unrealized_pnl),
                _dec(snap.margin_ratio),
                snap.run_id or None,
            ),
        )

    # -- 运行会话（§6.1 关联链起点） --------------------------------------

    def start_run_session(self, *, run_id: str, started_ms: int, mode: str,
                          strategy_version: str, config_hash: str, code_revision: str,
                          spot_endpoint: str, futures_endpoint: str,
                          user_stream_mode: str, timezone: str = "UTC") -> None:
        self._execute(
            "INSERT INTO run_sessions (run_id, started_ms, status, stop_reason, code_revision,"
            " strategy_version, config_hash, mode, spot_endpoint, futures_endpoint,"
            " user_stream_mode, timezone) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(run_id) DO NOTHING",
            (run_id, started_ms, "RUNNING", "", code_revision, strategy_version,
             config_hash, mode, spot_endpoint, futures_endpoint, user_stream_mode, timezone),
        )

    def update_run_session(self, run_id: str, *, status: str | None = None,
                           stop_reason: str | None = None) -> None:
        if status is not None:
            self._execute(
                "UPDATE run_sessions SET status = ? WHERE run_id = ?", (status, run_id)
            )
        if stop_reason is not None:
            self._execute(
                "UPDATE run_sessions SET stop_reason = ? WHERE run_id = ?", (stop_reason, run_id)
            )

    def end_run_session(self, run_id: str, *, ended_ms: int, status: str, stop_reason: str) -> None:
        self._execute(
            "UPDATE run_sessions SET ended_ms = ?, status = ?, stop_reason = ?"
            " WHERE run_id = ? AND (ended_ms IS NULL OR ? >= ended_ms)",
            (ended_ms, status, stop_reason, run_id, ended_ms),
        )

    def mark_interrupted_sessions(self, *, now_ms: int) -> int:
        """把非优雅退出（kill -9/断电）遗留的会话补记为 INTERRUPTED（§断点重连）。

        只处理 ``ended_ms IS NULL`` 且状态仍为 RUNNING/RECOVERY 的行；
        幂等（重复调用返回 0）；返回被标记的行数。
        """
        cur = self._execute(
            "UPDATE run_sessions SET ended_ms = ?, status = 'INTERRUPTED',"
            " stop_reason = 'process_exited_without_graceful_stop'"
            " WHERE ended_ms IS NULL AND status IN ('RUNNING', 'RECOVERY')",
            (now_ms,),
        )
        return cur.rowcount

    def run_session(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT run_id, started_ms, ended_ms, status, stop_reason, code_revision,"
                " strategy_version, config_hash, mode, spot_endpoint, futures_endpoint,"
                " user_stream_mode, timezone FROM run_sessions WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(zip(
            ("run_id", "started_ms", "ended_ms", "status", "stop_reason", "code_revision",
             "strategy_version", "config_hash", "mode", "spot_endpoint", "futures_endpoint",
             "user_stream_mode", "timezone"),
            row, strict=False,
        ))

    def latest_run_session(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT run_id FROM run_sessions ORDER BY started_ms DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return self.run_session(str(row[0]))

    def run_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        """按开始时间倒序的最近 N 个 run_session（只读诊断用）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id FROM run_sessions ORDER BY started_ms DESC LIMIT ?", (limit,)
            ).fetchall()
        return [r for r in (self.run_session(str(row[0])) for row in rows) if r is not None]

    def lease_holders(self) -> list[dict[str, Any]]:
        """当前 service_leases 表内容（只读诊断用）。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT lease_name, pid, holder, acquired_ms, expires_ms FROM service_leases"
            )
            rows = cur.fetchall()
            names = [d[0] for d in cur.description]
        return [dict(zip(names, row, strict=False)) for row in rows]

    # -- 策略决策（每个候选必须可解释，拒绝也落盘） ---------------------------

    def record_signal_decision(self, decision: Any) -> bool:
        """写入一条 signal_decision。decision_id 唯一 → 重复返回 False。"""
        try:
            self._execute(
                "INSERT INTO signal_decisions (decision_id, run_id, ts_ms, symbol, decision_kind,"
                " allowed, reason_code, reason_text, strategy_version, config_hash,"
                " funding_interval_hours, trailing_annualized, exit_average_annualized,"
                " consecutive_positive_periods, quote_volume_3d_avg, entry_threshold,"
                " exit_threshold, position_age_periods, spot_price, perp_price, quote_ts_ms,"
                " requested_notional, metrics_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    decision.decision_id,
                    decision.run_id,
                    decision.ts_ms,
                    decision.symbol,
                    decision.decision_kind,
                    int(decision.allowed),
                    decision.reason_code,
                    decision.reason_text,
                    decision.strategy_version,
                    decision.config_hash,
                    decision.funding_interval_hours,
                    _dec(decision.trailing_annualized),
                    _dec(decision.exit_average_annualized),
                    decision.consecutive_positive_periods,
                    _dec(decision.quote_volume_3d_avg),
                    _dec(decision.entry_threshold),
                    _dec(decision.exit_threshold),
                    decision.position_age_periods,
                    _dec(decision.spot_price),
                    _dec(decision.perp_price),
                    decision.quote_ts_ms,
                    _dec(decision.requested_notional),
                    json.dumps(decision.metrics, ensure_ascii=False, default=str),
                ),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def signal_decisions(self, *, since_ms: int | None = None, until_ms: int | None = None,
                         symbol: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        return self._query_table(
            "signal_decisions", since_ms=since_ms, until_ms=until_ms, symbol=symbol,
            run_id=run_id,
        )

    # -- 资金费现金流（§8.1） ------------------------------------------------

    def record_funding_cashflow(self, *, cashflow_id: str, symbol: str, funding_ts_ms: int,
                                funding_rate: Decimal, interval_hours: int, amount: Decimal,
                                source: str, run_id: str = "",
                                pair_execution_id: str | None = None,
                                asset: str = "USDT", raw_summary: str = "{}") -> bool:
        """记录一条已确认资金费。``(symbol, funding_ts_ms)`` 唯一 → 重复结算事件幂等。"""
        try:
            self._execute(
                "INSERT INTO funding_cashflows (cashflow_id, run_id, pair_execution_id, symbol,"
                " funding_ts_ms, received_ts_ms, funding_rate, interval_hours, asset, amount,"
                " source, reconciled, raw_summary) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (cashflow_id, run_id, pair_execution_id, symbol, funding_ts_ms, _now_ms(),
                 _dec(funding_rate), interval_hours, asset, _dec(amount), source, 0, raw_summary),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    # -- PnL 账本（§8.4） ---------------------------------------------------

    def record_ledger_entry(self, *, ledger_id: str, kind: str, amount: Decimal, ts_ms: int,
                            source: str = "", run_id: str = "", pair_execution_id: str | None = None,
                            asset: str = "USDT", period_start_ms: int | None = None,
                            period_end_ms: int | None = None,
                            calculation_version: str = "") -> bool:
        try:
            self._execute(
                "INSERT INTO pnl_ledger (ledger_id, run_id, pair_execution_id, ts_ms, kind, amount,"
                " asset, source, period_start_ms, period_end_ms, reconciled, calculation_version)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (ledger_id, run_id, pair_execution_id, ts_ms, kind, _dec(amount), asset,
                 source, period_start_ms, period_end_ms, 0, calculation_version),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    # -- 运行时状态（只读 CLI 用；服务每轮轻量更新） -------------------------

    def set_runtime_state(self, key: str, value: str) -> None:
        self._execute(
            "INSERT INTO runtime_state (key, value, updated_ms) VALUES (?,?,?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
            " updated_ms = excluded.updated_ms",
            (key, value, _now_ms()),
        )

    def runtime_state(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value, updated_ms FROM runtime_state"
            ).fetchall()
        return {row[0]: {"value": row[1], "updated_ms": int(row[2])} for row in rows}

    # -- 去重辅助（§7.3 统一入口的底层查询） ---------------------------------

    def has_open_intent(self, symbol: str) -> bool:
        """同 symbol 是否存在未终态 intent（其关联 pair 非终态，或无关联 pair）。"""
        terminal = ("COMPLETE", "COMPENSATED", "FLATTENED", "FAILED", "HALTED")
        placeholders = ",".join("?" * len(terminal))
        with self._lock:
            row = self._conn.execute(
                f"SELECT 1 FROM execution_intents i"  # noqa: S608
                f" LEFT JOIN pair_executions p ON p.pair_execution_id = i.pair_execution_id"
                f" WHERE i.symbol = ? AND i.is_closing = 0"
                f" AND (p.pair_execution_id IS NULL OR p.status NOT IN ({placeholders}))"  # noqa: S608
                f" LIMIT 1",
                (symbol, *terminal),
            ).fetchone()
        return row is not None

    def has_open_order(self, symbol: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM orders WHERE symbol = ? AND state IN ('NEW','PARTIALLY_FILLED','UNKNOWN')"
                " LIMIT 1",
                (symbol,),
            ).fetchone()
        return row is not None

    def get_pair(self, pair_execution_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM pair_executions WHERE pair_execution_id = ?",
                (pair_execution_id,),
            )
            row = cur.fetchone()
            names = [d[0] for d in cur.description]
        if row is None:
            return None
        return dict(zip(names, row, strict=False))

    def orders_for_pair(self, pair_execution_id: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM orders WHERE pair_execution_id = ? ORDER BY updated_ms",
                (pair_execution_id,),
            )
            rows = cur.fetchall()
            names = [d[0] for d in cur.description]
        return [dict(zip(names, row, strict=False)) for row in rows]

    def fills_for_orders(self, client_order_ids: Sequence[str]) -> list[dict[str, Any]]:
        """按 client order id 集合查成交。"""
        ids = [i for i in dict.fromkeys(client_order_ids) if i]
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            cur = self._conn.execute(
                f"SELECT * FROM fills WHERE client_order_id IN ({placeholders}) ORDER BY ts_ms",  # noqa: S608
                tuple(ids),
            )
            rows = cur.fetchall()
            names = [d[0] for d in cur.description]
        return [dict(zip(names, row, strict=False)) for row in rows]

    def open_pairs(self) -> list[dict[str, Any]]:
        """所有非终态 pair。"""
        return self._query_table(
            "pair_executions",
            where=("status NOT IN ('COMPLETE','COMPENSATED','FLATTENED','FAILED','HALTED')",),
        )

    def position_opened_ms(self, symbol: str) -> int | None:
        """该 symbol 当前持仓的开仓时间（最早一个非终态/open 成功 pair；无则最早成交）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT MIN(created_ms) FROM pair_executions"
                " WHERE symbol = ? AND kind = 'open'"
                " AND status NOT IN ('FAILED','HALTED')",
                (symbol,),
            ).fetchone()
            opened = row[0] if row and row[0] is not None else None
            if opened is None:
                row = self._conn.execute(
                    "SELECT MIN(ts_ms) FROM fills WHERE symbol = ? AND market = 'SPOT' AND side = 'BUY'",
                    (symbol,),
                ).fetchone()
                opened = row[0] if row and row[0] is not None else None
        return int(opened) if opened is not None else None

    def schema_version(self) -> int:
        with self._lock:
            return schema_version_of(self._conn)

    # -- 只读查询（CLI/报告导出；绝不触发写交易接口） --------------------------

    def _query_table(
        self,
        table: str,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        symbol: str | None = None,
        run_id: str | None = None,
        where: tuple[str, ...] = (),
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        """通用只读查询：按表结构自动带时间 / symbol / run_id 过滤。"""
        ts_col = {"orders": "updated_ms", "pair_executions": "created_ms",
                  "run_sessions": "started_ms", "funding_cashflows": "funding_ts_ms"}.get(table, "ts_ms")
        conditions: list[str] = list(where)
        args: list[Any] = []
        if since_ms is not None:
            conditions.append(f"{ts_col} >= ?")
            args.append(since_ms)
        if until_ms is not None:
            conditions.append(f"{ts_col} <= ?")
            args.append(until_ms)
        if symbol is not None and table in _TABLES_WITH_SYMBOL:
            conditions.append("symbol = ?")
            args.append(symbol)
        if run_id is not None and table in _TABLES_WITH_RUN_ID:
            conditions.append("run_id = ?")
            args.append(run_id)
        where_sql = f" WHERE {' AND '.join(conditions)}" if conditions else ""  # noqa: S608
        sql = (
            f"SELECT * FROM {table}{where_sql} ORDER BY {ts_col} DESC LIMIT {int(limit)}"  # noqa: S608
        )
        with self._lock:
            cur = self._conn.execute(sql, tuple(args))
            rows = cur.fetchall()
            names = [d[0] for d in cur.description]
        return [dict(zip(names, row, strict=False)) for row in rows]

    def orders(self, *, since_ms: int | None = None, until_ms: int | None = None,
               symbol: str | None = None, limit: int = 5000) -> list[dict[str, Any]]:
        return self._query_table("orders", since_ms=since_ms, until_ms=until_ms,
                                 symbol=symbol, limit=limit)

    def fills(self, *, since_ms: int | None = None, until_ms: int | None = None,
              symbol: str | None = None, limit: int = 5000) -> list[dict[str, Any]]:
        return self._query_table("fills", since_ms=since_ms, until_ms=until_ms,
                                 symbol=symbol, limit=limit)

    def pair_executions(self, *, since_ms: int | None = None, until_ms: int | None = None,
                        symbol: str | None = None, limit: int = 5000) -> list[dict[str, Any]]:
        return self._query_table("pair_executions", since_ms=since_ms, until_ms=until_ms,
                                 symbol=symbol, limit=limit)

    def funding_cashflows(self, *, since_ms: int | None = None, until_ms: int | None = None,
                          symbol: str | None = None, limit: int = 5000) -> list[dict[str, Any]]:
        return self._query_table("funding_cashflows", since_ms=since_ms, until_ms=until_ms,
                                 symbol=symbol, limit=limit)

    def pnl_ledger(self, *, since_ms: int | None = None, until_ms: int | None = None,
                   limit: int = 5000) -> list[dict[str, Any]]:
        return self._query_table("pnl_ledger", since_ms=since_ms, until_ms=until_ms, limit=limit)

    def position_snapshots(self, *, since_ms: int | None = None, until_ms: int | None = None,
                           symbol: str | None = None, limit: int = 5000) -> list[dict[str, Any]]:
        return self._query_table("position_snapshots", since_ms=since_ms, until_ms=until_ms,
                                 symbol=symbol, limit=limit)

    def account_snapshots(self, *, since_ms: int | None = None, until_ms: int | None = None,
                          limit: int = 5000) -> list[dict[str, Any]]:
        return self._query_table("account_snapshots", since_ms=since_ms, until_ms=until_ms,
                                 limit=limit)

    def reconciliation_runs(self, *, since_ms: int | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        return self._query_table("reconciliation_runs", since_ms=since_ms, limit=limit)

    def risk_decisions(self, *, since_ms: int | None = None, limit: int = 5000) -> list[dict[str, Any]]:
        return self._query_table("risk_decisions", since_ms=since_ms, limit=limit)

    def exchange_events(self, *, since_ms: int | None = None, market: str | None = None,
                        limit: int = 5000) -> list[dict[str, Any]]:
        conditions: list[str] = []
        args: list[Any] = []
        if since_ms is not None:
            conditions.append("recv_ts >= ?")
            args.append(since_ms)
        if market is not None:
            conditions.append("market = ?")
            args.append(market)
        where_sql = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM exchange_events{where_sql} ORDER BY recv_ts DESC LIMIT {int(limit)}"  # noqa: S608
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
            names = [d[0] for d in self._conn.execute(sql, tuple(args)).description]
        return [dict(zip(names, row, strict=False)) for row in rows]

    def realized_pnl_since(self, since_ms: int) -> Decimal:
        """过去 N 小时已实现 PnL 合计（风控 24h 亏损检查用）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(CAST(amount AS REAL)), 0) FROM pnl_ledger"
                " WHERE kind = 'REALIZED' AND ts_ms >= ?",
                (since_ms,),
            ).fetchone()
        return Decimal(str(row[0])) if row else Decimal("0")

    # -- 事件 ---------------------------------------------------------------

    def record_event(
        self,
        *,
        fingerprint: str,
        market: str,
        event_type: str,
        exchange_ts: int | None,
        payload: dict[str, Any],
        generation: int = 0,
    ) -> bool:
        """追加事件。fingerprint 唯一 —— 重复事件返回 False。"""
        try:
            self._execute(
                "INSERT INTO exchange_events (event_fp, market, event_type, exchange_ts, recv_ts,"
                " generation, payload) VALUES (?,?,?,?,?,?,?)",
                (
                    fingerprint,
                    market,
                    event_type,
                    exchange_ts,
                    _now_ms(),
                    generation,
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    # -- 风控决策 / 对账 ------------------------------------------------------

    def record_risk_decision(self, kind: str, allowed: bool, reason: str = "") -> None:
        self._execute(
            "INSERT INTO risk_decisions (kind, allowed, reason, ts_ms) VALUES (?,?,?,?)",
            (kind, int(allowed), reason, _now_ms()),
        )

    def record_reconciliation(self, result: ReconciliationResult, *, reason: str = "") -> None:
        self._execute(
            "INSERT INTO reconciliation_runs (ts_ms, consistent, can_open, mismatches, repaired, reason, details)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                result.ts_ms,
                int(result.consistent),
                int(result.can_open),
                ",".join(result.mismatches),
                ",".join(result.repaired),
                reason,
                json.dumps(result.details, ensure_ascii=False, default=str),
            ),
        )

    # -- 单实例锁 -----------------------------------------------------------

    def acquire_lease(self, name: str, holder: str | None = None, *, ttl_seconds: float = 30.0) -> Lease:
        """获取单实例锁。已有**未过期**的其他持有者时抛 LeaseConflict。"""
        holder = holder or f"pid-{os.getpid()}"
        now = _now_ms()
        expires = now + int(ttl_seconds * 1000)
        try:
            self._execute(
                "INSERT INTO service_leases (lease_name, pid, holder, acquired_ms, expires_ms)"
                " VALUES (?,?,?,?,?)"
                " ON CONFLICT(lease_name) DO UPDATE SET pid=excluded.pid,"
                " holder=excluded.holder, acquired_ms=excluded.acquired_ms, expires_ms=excluded.expires_ms"
                " WHERE service_leases.expires_ms < excluded.acquired_ms"
                " OR service_leases.holder = excluded.holder",
                (name, os.getpid(), holder, now, expires),
            )
        except sqlite3.Error as exc:
            raise StoreError(f"锁写入失败: {exc}") from exc

        with self._lock:
            row = self._conn.execute(
                "SELECT holder, pid FROM service_leases WHERE lease_name = ?", (name,)
            ).fetchone()
        if row is None or row[0] != holder:
            current_holder = row[0] if row else "<none>"
            raise LeaseConflict(
                f"单实例锁 '{name}' 已被 {current_holder} 持有。"
                "同一账户禁止两个执行进程并发管理（文档 §4.2）。"
            )
        return Lease(name=name, holder=holder, pid=os.getpid())

    def refresh_lease(self, name: str, holder: str, *, ttl_seconds: float = 30.0) -> None:
        self._execute(
            "UPDATE service_leases SET expires_ms = ? WHERE lease_name = ? AND holder = ?",
            (_now_ms() + int(ttl_seconds * 1000), name, holder),
        )

    def release_lease(self, name: str, holder: str) -> None:
        self._execute(
            "DELETE FROM service_leases WHERE lease_name = ? AND holder = ?", (name, holder)
        )


__all__ = ["Lease", "LeaseConflict", "StateStore", "StoreError"]
