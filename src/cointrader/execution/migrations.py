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
SCHEMA_VERSION = 2


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


def _migrate_2(conn: sqlite3.Connection) -> None:
    """v1 → v2：heartbeat 截断、scan epoch、同步游标、账户快照组、
    current projection、funding authority（实施计划书 v2.0 T3，AC-06/08/09/11）。

    全部语句幂等（IF NOT EXISTS / 列存在性检查），单事务内完成；
    中途断电后重启可续跑。旧 funding 行统一 authority='ESTIMATED'，
    不静默改写为交易所事实。
    """
    # 1) run_sessions.last_heartbeat_ms：异常会话下次接管按 heartbeat 截断，
    #    不把停机间隔计入在线时长（部分迁移残留的残表不阻塞续跑）
    _ensure_columns(conn, "run_sessions", {"last_heartbeat_ms": "INTEGER"})
    if {"ended_ms", "started_ms"} <= _columns(conn, "run_sessions"):
        conn.execute(
            "UPDATE run_sessions SET last_heartbeat_ms = COALESCE(ended_ms, started_ms)"
            " WHERE last_heartbeat_ms IS NULL"
        )

    # 2) scan epoch 与候选快照（T2 契约持久化）
    conn.execute(
        "CREATE TABLE IF NOT EXISTS scan_epochs ("
        " epoch_id TEXT PRIMARY KEY,"
        " universe_snapshot_ts_ms INTEGER NOT NULL,"
        " decision_cutoff_ms INTEGER NOT NULL,"
        " status TEXT NOT NULL,"
        " expected_json TEXT NOT NULL DEFAULT '[]',"
        " excluded_json TEXT NOT NULL DEFAULT '{}',"
        " failed_json TEXT NOT NULL DEFAULT '{}',"
        " created_ms INTEGER NOT NULL,"
        " completed_ms INTEGER,"
        " expires_ms INTEGER NOT NULL,"
        " error TEXT NOT NULL DEFAULT '')"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS candidate_snapshots ("
        " epoch_id TEXT NOT NULL,"
        " symbol TEXT NOT NULL,"
        " interval_hours INTEGER NOT NULL,"
        " rates_json TEXT NOT NULL DEFAULT '[]',"
        " mark_prices_json TEXT NOT NULL DEFAULT '[]',"
        " timestamps_json TEXT NOT NULL DEFAULT '[]',"
        " expected_last_funding_ms INTEGER NOT NULL,"
        " volume_window_end_ms INTEGER NOT NULL,"
        " quote_volume_3d_avg TEXT,"
        " fetched_ms INTEGER NOT NULL,"
        " error TEXT NOT NULL DEFAULT '',"
        " PRIMARY KEY (epoch_id, symbol))"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_candidate_snapshots_epoch"
        " ON candidate_snapshots(epoch_id)"
    )

    # 3) 同步游标：(scope, stream, symbol_key) 只前进不后退
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sync_cursors ("
        " scope TEXT NOT NULL,"
        " stream TEXT NOT NULL,"
        " symbol_key TEXT NOT NULL,"
        " last_time_ms INTEGER NOT NULL DEFAULT 0,"
        " last_id TEXT NOT NULL DEFAULT '',"
        " updated_ms INTEGER NOT NULL,"
        " PRIMARY KEY (scope, stream, symbol_key))"
    )

    # 4) 账户快照组与资产明细；既有快照表加 nullable snapshot_id（旧行保持可读）
    conn.execute(
        "CREATE TABLE IF NOT EXISTS account_snapshot_groups ("
        " snapshot_id TEXT PRIMARY KEY,"
        " ts_ms INTEGER NOT NULL,"
        " capture_start_ms INTEGER NOT NULL,"
        " capture_end_ms INTEGER NOT NULL,"
        " source TEXT NOT NULL DEFAULT '',"
        " spot_equity_usdt TEXT,"
        " futures_equity_usdt TEXT,"
        " total_equity_usdt TEXT,"
        " available_balance_usdt TEXT,"
        " complete INTEGER NOT NULL DEFAULT 0,"
        " run_id TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS account_assets ("
        " snapshot_id TEXT NOT NULL,"
        " market TEXT NOT NULL,"
        " asset TEXT NOT NULL,"
        " free_qty TEXT NOT NULL DEFAULT '0',"
        " locked_qty TEXT NOT NULL DEFAULT '0',"
        " total_qty TEXT NOT NULL DEFAULT '0',"
        " price_usdt TEXT,"
        " value_usdt TEXT,"
        " PRIMARY KEY (snapshot_id, market, asset))"
    )
    _ensure_columns(conn, "account_snapshots", {"snapshot_id": "TEXT"})
    _ensure_columns(conn, "position_snapshots", {"snapshot_id": "TEXT"})

    # 5) current projection：只能由完整交易所 snapshot 事务更新
    conn.execute(
        "CREATE TABLE IF NOT EXISTS current_account ("
        " id INTEGER PRIMARY KEY CHECK (id = 1),"
        " snapshot_id TEXT NOT NULL,"
        " observed_at_ms INTEGER NOT NULL,"
        " source TEXT NOT NULL DEFAULT '',"
        " spot_equity_usdt TEXT,"
        " futures_equity_usdt TEXT,"
        " total_equity_usdt TEXT,"
        " available_balance_usdt TEXT,"
        " complete INTEGER NOT NULL DEFAULT 0,"
        " updated_ms INTEGER NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS current_positions ("
        " symbol TEXT PRIMARY KEY,"
        " spot_qty TEXT NOT NULL,"
        " perp_qty TEXT NOT NULL,"
        " spot_price TEXT,"
        " perp_price TEXT,"
        " snapshot_id TEXT,"
        " observed_at_ms INTEGER NOT NULL,"
        " source TEXT NOT NULL DEFAULT '',"
        " reconciled INTEGER NOT NULL DEFAULT 0,"
        " tombstone INTEGER NOT NULL DEFAULT 0)"
    )

    # 6) funding_cashflows 重建：移除旧 UNIQUE(symbol, funding_ts_ms)，
    #    新增 market/exchange_income_id/authority/observed_ms；
    #    (market, exchange_income_id) 部分唯一索引挡住重复回补；旧行 → ESTIMATED
    has_old = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'funding_cashflows'"
    ).fetchone()
    if has_old is not None:
        has_authority = "authority" in _columns(conn, "funding_cashflows")
        if not has_authority:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS funding_cashflows_v2 ("
                " cashflow_id TEXT PRIMARY KEY,"
                " run_id TEXT NOT NULL DEFAULT '',"
                " pair_execution_id TEXT,"
                " market TEXT NOT NULL DEFAULT 'PERP',"
                " symbol TEXT NOT NULL,"
                " funding_ts_ms INTEGER NOT NULL,"
                " received_ts_ms INTEGER NOT NULL,"
                " funding_rate TEXT NOT NULL,"
                " interval_hours INTEGER NOT NULL,"
                " asset TEXT NOT NULL DEFAULT 'USDT',"
                " amount TEXT NOT NULL,"
                " source TEXT NOT NULL DEFAULT '',"
                " reconciled INTEGER NOT NULL DEFAULT 0,"
                " raw_summary TEXT NOT NULL DEFAULT '{}',"
                " exchange_income_id TEXT,"
                " authority TEXT NOT NULL DEFAULT 'ESTIMATED',"
                " observed_ms INTEGER NOT NULL DEFAULT 0)"
            )
            conn.execute(
                "INSERT INTO funding_cashflows_v2 (cashflow_id, run_id, pair_execution_id,"
                " market, symbol, funding_ts_ms, received_ts_ms, funding_rate,"
                " interval_hours, asset, amount, source, reconciled, raw_summary,"
                " authority) SELECT cashflow_id, run_id, pair_execution_id, 'PERP',"
                " symbol, funding_ts_ms, received_ts_ms, funding_rate, interval_hours,"
                " asset, amount, source, reconciled, raw_summary, 'ESTIMATED'"
                " FROM funding_cashflows"
            )
            conn.execute("DROP TABLE funding_cashflows")
            conn.execute("ALTER TABLE funding_cashflows_v2 RENAME TO funding_cashflows")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_funding_market_income"
        " ON funding_cashflows(market, exchange_income_id)"
        " WHERE exchange_income_id IS NOT NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_funding_symbol_ts"
        " ON funding_cashflows(symbol, funding_ts_ms)"
    )

    # 7) 版本号（幂等：多行时只升不降，读取用 MAX）
    conn.execute(
        "INSERT INTO schema_meta (schema_version, migrated_at_ms) VALUES (?, ?)"
        " ON CONFLICT DO UPDATE SET"
        " schema_version = excluded.schema_version,"
        " migrated_at_ms = excluded.migrated_at_ms",
        (2, _now_ms()),
    )
    conn.execute(
        "UPDATE schema_meta SET schema_version = ?, migrated_at_ms = ? WHERE schema_version < ?",
        (2, _now_ms(), 2),
    )


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}  # noqa: S608


#: 版本号 → 迁移函数（顺序执行）
MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {1: _migrate_1, 2: _migrate_2}


def schema_version_of(conn: sqlite3.Connection) -> int:
    """读取当前 schema 版本；无 schema_meta 视为 0（既有旧库）。"""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_meta'"
    ).fetchone()
    if row is None:
        return 0
    version = conn.execute("SELECT MAX(schema_version) FROM schema_meta").fetchone()
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
