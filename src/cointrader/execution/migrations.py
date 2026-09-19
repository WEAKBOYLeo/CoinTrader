"""SQLite Schema 版本迁移（开发文档 §8.1/§8.2）。

迁移规则（硬约束）：

1. 打开数据库 → 读 ``schema_meta.schema_version`` → 顺序执行未完成迁移。
2. 每个迁移在**单个事务**中完成；成功后更新版本号。
3. 迁移失败 → 抛 ``MigrationError`` → 拒绝启动，不继续交易。
4. 不允许靠「删除数据库重建」升级：既有库必须从旧版本无损升级，
   已有 orders/fills/pair_executions 数据不丢失。
5. 版本高于本代码支持的版本 → 拒绝启动（防止旧代码写坏新 schema）。

迁移内容全部保持幂等（IF NOT EXISTS / 列存在性检查），
这样「事务中途断电后再次启动」也能继续完成。
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from collections.abc import Callable

__all__ = ["MigrationError", "SCHEMA_VERSION", "MIGRATIONS", "apply_migrations"]


class MigrationError(Exception):
    """Schema 迁移失败。启动必须中止。"""


#: 本代码支持的 schema 版本
SCHEMA_VERSION = 1


def _now_ms() -> int:
    return int(time.time() * 1000)


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """为既有表补齐缺失列（幂等）。"""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}  # noqa: S608
    for name, decl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")  # noqa: S608


def _split_sql(script: str) -> list[str]:
    """把一段 DDL 按顶层分号拆成单条语句（忽略引号内分号）。"""
    statements: list[str] = []
    buffer: list[str] = []
    in_quote = False
    for ch in script:
        if ch == "'":
            in_quote = not in_quote
        if ch == ";" and not in_quote:
            stmt = "".join(buffer).strip()
            if stmt:
                statements.append(stmt)
            buffer = []
        else:
            buffer.append(ch)
    tail = "".join(buffer).strip()
    if tail:
        statements.append(tail)
    return statements


def _migrate_1(conn: sqlite3.Connection) -> None:
    """v0（既有账本）→ v1：补关联字段 + 新表。"""
    # -- 新表（逐条 execute，保持在外层事务内；executescript 会隐式 COMMIT） --
    _ddl = """
    CREATE TABLE IF NOT EXISTS schema_meta (
        schema_version INTEGER NOT NULL,
        migrated_at_ms INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS run_sessions (
        run_id TEXT PRIMARY KEY,
        started_ms INTEGER NOT NULL,
        ended_ms INTEGER,
        status TEXT NOT NULL DEFAULT 'RUNNING',
        stop_reason TEXT NOT NULL DEFAULT '',
        code_revision TEXT NOT NULL DEFAULT '',
        strategy_version TEXT NOT NULL DEFAULT '',
        config_hash TEXT NOT NULL DEFAULT '',
        mode TEXT NOT NULL DEFAULT '',
        spot_endpoint TEXT NOT NULL DEFAULT '',
        futures_endpoint TEXT NOT NULL DEFAULT '',
        user_stream_mode TEXT NOT NULL DEFAULT '',
        timezone TEXT NOT NULL DEFAULT 'UTC'
    );
    CREATE TABLE IF NOT EXISTS signal_decisions (
        decision_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL DEFAULT '',
        ts_ms INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        decision_kind TEXT NOT NULL,
        allowed INTEGER NOT NULL,
        reason_code TEXT NOT NULL DEFAULT '',
        reason_text TEXT NOT NULL DEFAULT '',
        strategy_version TEXT NOT NULL DEFAULT '',
        config_hash TEXT NOT NULL DEFAULT '',
        funding_interval_hours INTEGER,
        trailing_annualized TEXT,
        exit_average_annualized TEXT,
        consecutive_positive_periods INTEGER,
        quote_volume_3d_avg TEXT,
        entry_threshold TEXT,
        exit_threshold TEXT,
        position_age_periods INTEGER,
        spot_price TEXT,
        perp_price TEXT,
        quote_ts_ms INTEGER,
        requested_notional TEXT,
        metrics_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_signal_decisions_run
        ON signal_decisions(run_id, ts_ms);
    CREATE TABLE IF NOT EXISTS funding_cashflows (
        cashflow_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL DEFAULT '',
        pair_execution_id TEXT,
        symbol TEXT NOT NULL,
        funding_ts_ms INTEGER NOT NULL,
        received_ts_ms INTEGER NOT NULL,
        funding_rate TEXT NOT NULL,
        interval_hours INTEGER NOT NULL,
        asset TEXT NOT NULL DEFAULT 'USDT',
        amount TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT '',
        reconciled INTEGER NOT NULL DEFAULT 0,
        raw_summary TEXT NOT NULL DEFAULT '{}',
        UNIQUE(symbol, funding_ts_ms)
    );
    CREATE TABLE IF NOT EXISTS pnl_ledger (
        ledger_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL DEFAULT '',
        pair_execution_id TEXT,
        ts_ms INTEGER NOT NULL,
        kind TEXT NOT NULL,
        amount TEXT NOT NULL,
        asset TEXT NOT NULL DEFAULT 'USDT',
        source TEXT NOT NULL DEFAULT '',
        period_start_ms INTEGER,
        period_end_ms INTEGER,
        reconciled INTEGER NOT NULL DEFAULT 0,
        calculation_version TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_pnl_ledger_pair
        ON pnl_ledger(pair_execution_id);
    CREATE TABLE IF NOT EXISTS runtime_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_ms INTEGER NOT NULL
    );
    """
    for statement in _split_sql(_ddl):
        conn.execute(statement)

    # -- 既有表补列（幂等） ---------------------------------------------------
    _ensure_columns(conn, "pair_executions", {
        "run_id": "TEXT",
        "open_or_close_reason": "TEXT",
        "signal_decision_id": "TEXT",
        "signal_ts_ms": "INTEGER",
        "decision_ts_ms": "INTEGER",
        "submit_ts_ms": "INTEGER",
        "exchange_confirmed_ts_ms": "INTEGER",
        "completed_ts_ms": "INTEGER",
        "actual_spot_notional": "TEXT",
        "actual_perp_notional": "TEXT",
        "funding_pnl": "TEXT",
        "fee_pnl": "TEXT",
        "basis_pnl": "TEXT",
        "realized_pnl": "TEXT",
        "unrealized_pnl": "TEXT",
        "net_pnl": "TEXT",
    })
    _ensure_columns(conn, "orders", {
        "run_id": "TEXT",
        "pair_execution_id": "TEXT",
        "intent_id": "TEXT",
        "submit_ts_ms": "INTEGER",
        "ack_ts_ms": "INTEGER",
        "terminal_ts_ms": "INTEGER",
        "exchange_update_ts_ms": "INTEGER",
        "failure_class": "TEXT",
        "raw_status_summary": "TEXT",
    })
    _ensure_columns(conn, "fills", {
        "run_id": "TEXT",
        "pair_execution_id": "TEXT",
        "intent_id": "TEXT",
        "exchange_trade_id": "TEXT",
        "quote_qty": "TEXT",
        "maker_taker": "TEXT",
        "exchange_ts_ms": "INTEGER",
        "received_ts_ms": "INTEGER",
    })
    # 不同市场的交易所 fill/trade id 可能相同：唯一约束必须带 market
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_fills_market_trade"
        " ON fills(market, exchange_trade_id)"
        " WHERE exchange_trade_id IS NOT NULL"
    )
    _ensure_columns(conn, "position_snapshots", {"run_id": "TEXT"})
    _ensure_columns(conn, "account_snapshots", {"run_id": "TEXT"})

    # -- 版本号 --------------------------------------------------------------
    conn.execute(
        "INSERT INTO schema_meta (schema_version, migrated_at_ms) VALUES (?, ?)"
        " ON CONFLICT DO UPDATE SET"
        " schema_version = excluded.schema_version,"
        " migrated_at_ms = excluded.migrated_at_ms",
        (1, _now_ms()),
    )


#: 版本号 → 迁移函数（顺序执行）
MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {1: _migrate_1}


def schema_version_of(conn: sqlite3.Connection) -> int:
    """读取当前 schema 版本；无 schema_meta 视为 0（既有旧库）。"""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_meta'"
    ).fetchone()
    if row is None:
        return 0
    version = conn.execute("SELECT schema_version FROM schema_meta LIMIT 1").fetchone()
    return int(version[0]) if version else 0


def apply_migrations(conn: sqlite3.Connection) -> int:
    """把 ``conn`` 升级到 ``SCHEMA_VERSION``。返回最终版本。

    Raises:
        MigrationError: 版本超前，或任何迁移执行失败。
    """
    current = schema_version_of(conn)
    if current > SCHEMA_VERSION:
        raise MigrationError(
            f"数据库 schema 版本 {current} 高于本代码支持的最大版本 {SCHEMA_VERSION}，"
            "拒绝启动（禁止旧代码写新 schema）。"
        )
    for target in sorted(MIGRATIONS):
        if target <= current:
            continue
        try:
            conn.execute("BEGIN")
            MIGRATIONS[target](conn)
            conn.execute("COMMIT")
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise MigrationError(f"迁移 v{current} → v{target} 失败: {exc}") from exc
        current = target
    return current
