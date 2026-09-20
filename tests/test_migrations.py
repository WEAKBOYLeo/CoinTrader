"""Schema 迁移测试（开发文档 §8.1/§8.2：旧库无损升级、中断可续、拒绝越级）。"""

from __future__ import annotations

import sqlite3
import time

import pytest

from cointrader.execution import migrations as migrations_mod
from cointrader.execution import store as store_mod
from cointrader.execution.migrations import (
    SCHEMA_VERSION,
    MigrationError,
    apply_migrations,
    schema_version_of,
)
from cointrader.execution.store import StateStore, StoreError

NOW_MS = int(time.time() * 1000)


def _make_v0_db(path) -> None:
    """按迁移前的既有 schema 建库并写入历史数据（模拟旧版本账本）。"""
    conn = sqlite3.connect(path)
    conn.executescript(store_mod._SCHEMA)  # noqa: SLF001
    conn.execute(
        "INSERT INTO pair_executions (pair_execution_id, symbol, kind, status,"
        " target_notional, error, strategy_version, created_ms, updated_ms)"
        " VALUES ('pair-legacy-1', 'BTCUSDT', 'open', 'COMPLETE', '100', '', 'v0', ?, ?)",
        (NOW_MS - 1000, NOW_MS - 1000),
    )
    conn.execute(
        "INSERT INTO orders (client_order_id, exchange_order_id, symbol, market, side,"
        " order_type, quantity, state, updated_ms)"
        " VALUES ('co-legacy-1', 'ex-legacy-1', 'BTCUSDT', 'SPOT', 'BUY', 'MARKET', '0.01', 'FILLED', ?)",
        (NOW_MS - 1000,),
    )
    conn.execute(
        "INSERT INTO fills (fill_id, client_order_id, symbol, market, side, quantity, price,"
        " fee_asset, fee_amount, ts_ms)"
        " VALUES ('f-legacy-1', 'co-legacy-1', 'BTCUSDT', 'SPOT', 'BUY', '0.01', '100', 'USDT', '0.01', ?)",
        (NOW_MS - 1000,),
    )
    conn.commit()
    conn.close()


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}  # noqa: S608


class TestMigrations:
    def test_fresh_db_reaches_current_version(self, tmp_path):
        store = StateStore(tmp_path / "t.sqlite3")
        assert schema_version_of(store._conn) == SCHEMA_VERSION  # noqa: SLF001
        tables = _tables(store._conn)
        for required in ("schema_meta", "run_sessions", "signal_decisions",
                         "funding_cashflows", "pnl_ledger", "runtime_state"):
            assert required in tables, f"缺少新表 {required}"
        store.close()

    def test_v0_db_upgrades_without_data_loss(self, tmp_path):
        db = tmp_path / "old.sqlite3"
        _make_v0_db(db)
        store = StateStore(db)
        assert schema_version_of(store._conn) == SCHEMA_VERSION  # noqa: SLF001
        # 历史数据无损
        pairs = store.pair_executions(limit=10)
        assert [p["pair_execution_id"] for p in pairs] == ["pair-legacy-1"]
        orders = store.orders(limit=10)
        assert [o["client_order_id"] for o in orders] == ["co-legacy-1"]
        fills = store.fills(limit=10)
        assert [f["fill_id"] for f in fills] == ["f-legacy-1"]
        # 新列补齐
        assert {"run_id", "signal_decision_id", "completed_ts_ms", "net_pnl"} <= _columns(
            store._conn, "pair_executions")  # noqa: SLF001
        assert {"pair_execution_id", "intent_id", "run_id"} <= _columns(store._conn, "orders")  # noqa: SLF001
        assert {"exchange_trade_id", "run_id"} <= _columns(store._conn, "fills")  # noqa: SLF001
        store.close()

    def test_upgrade_is_idempotent(self, tmp_path):
        db = tmp_path / "old.sqlite3"
        _make_v0_db(db)
        store = StateStore(db)
        conn = store._conn  # noqa: SLF001
        assert apply_migrations(conn) == SCHEMA_VERSION  # 再跑一遍不报错
        assert schema_version_of(conn) == SCHEMA_VERSION
        # 第三次：仍然 OK
        assert apply_migrations(conn) == SCHEMA_VERSION
        store.close()

    def test_interrupted_migration_recovers_on_restart(self, tmp_path, monkeypatch):
        """迁移中途失败（模拟断电）→ 拒绝启动；再次启动 → 幂等续跑完成。"""
        db = tmp_path / "old.sqlite3"
        _make_v0_db(db)
        real_ensure = migrations_mod._ensure_columns  # noqa: SLF001

        def boom(conn, table, columns):
            if table == "fills":
                raise sqlite3.OperationalError("simulated power loss")
            real_ensure(conn, table, columns)

        monkeypatch.setattr(migrations_mod, "_ensure_columns", boom)  # noqa: SLF001
        with pytest.raises(StoreError, match="迁移失败"):
            StateStore(db)
        # 断电点之后未 bump 版本 → 重启续跑
        monkeypatch.undo()
        store = StateStore(db)
        assert schema_version_of(store._conn) == SCHEMA_VERSION  # noqa: SLF001
        assert {"exchange_trade_id"} <= _columns(store._conn, "fills")  # noqa: SLF001
        # 历史数据仍在
        assert [p["pair_execution_id"] for p in store.pair_executions(limit=10)] == ["pair-legacy-1"]
        store.close()

    def test_partial_state_without_version_recovers(self, tmp_path):
        """新表已建但版本未 bump（事务提交前断电）→ 重启补齐。"""
        db = tmp_path / "old.sqlite3"
        _make_v0_db(db)
        conn = sqlite3.connect(db)
        # 模拟 _migrate_1 执行到一半：部分新表已建，版本未更新
        conn.execute("CREATE TABLE run_sessions (run_id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()
        assert schema_version_of(sqlite3.connect(db)) == 0
        store = StateStore(db)
        assert schema_version_of(store._conn) == SCHEMA_VERSION  # noqa: SLF001
        assert "runtime_state" in _tables(store._conn)
        store.close()

    def test_newer_schema_version_refused(self, tmp_path):
        db = tmp_path / "new.sqlite3"
        conn = sqlite3.connect(db)
        conn.executescript(store_mod._SCHEMA)  # noqa: SLF001
        conn.execute("CREATE TABLE schema_meta (schema_version INTEGER, migrated_at_ms INTEGER)")
        conn.execute("INSERT INTO schema_meta VALUES (99, ?)", (NOW_MS,))
        conn.commit()
        conn.close()
        with pytest.raises(StoreError, match="高于本代码"):
            StateStore(db)

    def test_apply_migrations_direct_error(self, tmp_path):
        conn = sqlite3.connect(tmp_path / "x.sqlite3")
        conn.execute("CREATE TABLE schema_meta (schema_version INTEGER, migrated_at_ms INTEGER)")
        conn.execute("INSERT INTO schema_meta VALUES (99, 0)")
        with pytest.raises(MigrationError):
            apply_migrations(conn)
        conn.close()


def _make_v1_db(path) -> None:
    """真实 v1 账本 fixture：两个 run、非终态订单、fills、估算 funding、
    position/account snapshots、pnl ledger（T3 实施动作 1）。"""
    conn = sqlite3.connect(path)
    conn.executescript(store_mod._SCHEMA)  # noqa: SLF001
    conn.commit()
    conn.close()
    conn = sqlite3.connect(path)
    migrations_mod._migrate_1(conn)  # noqa: SLF001
    conn.commit()
    conn.close()

    conn = sqlite3.connect(path)
    for _i, (run, started) in enumerate(
        (("run-1", NOW_MS - 2 * 86_400_000), ("run-2", NOW_MS - 86_400_000))
    ):
        conn.execute(
            "INSERT INTO run_sessions (run_id, started_ms, ended_ms, status, stop_reason,"
            " code_revision, strategy_version, config_hash, mode, spot_endpoint,"
            " futures_endpoint, user_stream_mode) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run, started, NOW_MS - 1000, "STOPPED", "graceful_stop", "rev", "v1", "h",
             "testnet", "e", "f", "poll"),
        )
    conn.execute(
        "INSERT INTO pair_executions (pair_execution_id, symbol, kind, status, target_notional,"
        " error, strategy_version, created_ms, updated_ms, run_id) VALUES"
        " ('pair-1', 'BTCUSDT', 'open', 'COMPLETE', '100', '', 'v1', ?, ?, 'run-1'),"
        " ('pair-2', 'ETHUSDT', 'open', 'SUBMIT_SPOT', '50', '', 'v1', ?, ?, 'run-2')",
        (NOW_MS - 1000, NOW_MS - 900, NOW_MS - 500, NOW_MS - 400),
    )
    conn.execute(
        "INSERT INTO orders (client_order_id, exchange_order_id, symbol, market, side,"
        " order_type, quantity, state, updated_ms, run_id, pair_execution_id)"
        " VALUES ('co-1', 'ex-1', 'BTCUSDT', 'SPOT', 'BUY', 'MARKET', '0.01', 'FILLED', ?, 'run-1', 'pair-1'),"
        " ('co-2', 'ex-2', 'ETHUSDT', 'PERP', 'SELL', 'MARKET', '0.5', 'NEW', ?, 'run-2', 'pair-2')",
        (NOW_MS - 1000, NOW_MS - 400),
    )
    for i, (fid, cid, market, ts) in enumerate((
        ("f-1", "co-1", "SPOT", NOW_MS - 990),
        ("f-2", "co-2", "PERP", NOW_MS - 390),
    )):
        conn.execute(
            "INSERT INTO fills (fill_id, client_order_id, exchange_order_id, symbol, market, side,"
            " quantity, price, fee_asset, fee_amount, ts_ms, run_id, exchange_trade_id)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (fid, cid, f"ex-{i + 1}", "BTCUSDT" if i == 0 else "ETHUSDT", market,
             "BUY" if i == 0 else "SELL", "0.01" if i == 0 else "0.5",
             "100" if i == 0 else "2000", "USDT", "0.01", ts, "run-1" if i == 0 else "run-2",
             f"trade-{i + 1}"),
        )
    for i, ts in enumerate((NOW_MS - 3 * 86_400_000, NOW_MS - 2 * 86_400_000, NOW_MS - 86_400_000)):
        conn.execute(
            "INSERT INTO funding_cashflows (cashflow_id, run_id, symbol, funding_ts_ms,"
            " received_ts_ms, funding_rate, interval_hours, asset, amount, source, reconciled)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (f"fcf-{i}", "run-1", "BTCUSDT", ts, ts + 60_000, "0.0001", 8, "USDT",
             "0.008", "funding_history", 1),
        )
    conn.execute(
        "INSERT INTO position_snapshots (ts_ms, symbol, spot_qty, perp_qty, spot_price,"
        " perp_price, basis_pct, source, run_id) VALUES (?,?,?,?,?,?,?,?,?)",
        (NOW_MS - 500, "BTCUSDT", "0.01", "-0.01", "100", "100.5", "0.005", "reconcile", "run-1"),
    )
    conn.execute(
        "INSERT INTO account_snapshots (ts_ms, source, available_balance, wallet_balance,"
        " equity, unrealized_pnl, margin_ratio, run_id) VALUES (?,?,?,?,?,?,?,?)",
        (NOW_MS - 500, "PERP", "10", "20", "20.5", "0.5", "0.1", "run-1"),
    )
    conn.execute(
        "INSERT INTO pnl_ledger (ledger_id, run_id, pair_execution_id, ts_ms, kind, amount,"
        " asset, source, period_start_ms, period_end_ms, reconciled, calculation_version)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("led-1", "run-1", "pair-1", NOW_MS - 100, "REALIZED", "1.23", "USDT",
         "pnl_aggregator", NOW_MS - 1000, NOW_MS - 100, 1, "pnl-v1"),
    )
    conn.commit()
    conn.close()


class TestMigrate2:
    """v1 → v2（AC-11：数据无损、estimated 标记、索引、幂等）。"""

    def test_v1_db_upgrades_without_data_loss(self, tmp_path):
        db = tmp_path / "v1.sqlite3"
        _make_v1_db(db)
        store = StateStore(db)
        conn = store._conn  # noqa: SLF001
        assert schema_version_of(conn) == 2
        # 行数/关联字段不丢
        assert conn.execute("SELECT COUNT(*) FROM run_sessions").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM pair_executions").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM position_snapshots").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM account_snapshots").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM pnl_ledger").fetchone()[0] == 1
        assert [f["fill_id"] for f in store.fills(limit=10)] == ["f-2", "f-1"]
        # 非终态订单保留（恢复要能接住）
        assert store.orders_in_states(("NEW",))
        # 旧 funding 统一 ESTIMATED + market PERP，不被改写
        rows = store.funding_cashflows(limit=10)
        assert len(rows) == 3
        assert all(r["authority"] == "ESTIMATED" and r["market"] == "PERP" for r in rows)
        assert all(r["amount"] == "0.008" for r in rows)
        # heartbeat 初始化 = COALESCE(ended_ms, started_ms)
        row = conn.execute("SELECT last_heartbeat_ms FROM run_sessions WHERE run_id='run-1'").fetchone()
        assert row[0] == NOW_MS - 1000
        # 新表存在
        tables = _tables(conn)
        for required in ("scan_epochs", "candidate_snapshots", "sync_cursors",
                         "account_snapshot_groups", "account_assets",
                         "current_account", "current_positions"):
            assert required in tables
        # 快照表 nullable snapshot_id（旧行保持可读）
        assert "snapshot_id" in _columns(conn, "account_snapshots")
        assert "snapshot_id" in _columns(conn, "position_snapshots")
        # 旧 fills 唯一索引仍在
        idx = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert "idx_fills_market_trade" in idx
        assert "idx_funding_market_income" in idx
        store.close()

    def test_v2_migration_idempotent(self, tmp_path):
        db = tmp_path / "v1.sqlite3"
        _make_v1_db(db)
        store = StateStore(db)
        conn = store._conn  # noqa: SLF001
        assert apply_migrations(conn) == 2
        assert apply_migrations(conn) == 2
        assert conn.execute("SELECT COUNT(*) FROM funding_cashflows").fetchone()[0] == 3
        store.close()

    def test_v2_migration_failure_keeps_v1(self, tmp_path, monkeypatch):
        """v2 迁移中途失败 → 事务回滚，schema 仍为 v1，重试可完成。"""
        db = tmp_path / "v1.sqlite3"
        _make_v1_db(db)
        real_ensure = migrations_mod._ensure_columns  # noqa: SLF001

        def boom(conn, table, columns):
            if table == "run_sessions":
                raise sqlite3.OperationalError("simulated power loss in v2")
            real_ensure(conn, table, columns)

        monkeypatch.setattr(migrations_mod, "_ensure_columns", boom)  # noqa: SLF001
        with pytest.raises(StoreError, match="迁移失败"):
            StateStore(db)
        # 回滚后仍是 v1 且数据完好
        assert schema_version_of(sqlite3.connect(db)) == 1
        n = sqlite3.connect(db).execute("SELECT COUNT(*) FROM funding_cashflows").fetchone()[0]
        assert n == 3
        monkeypatch.undo()
        store = StateStore(db)
        assert schema_version_of(store._conn) == 2  # noqa: SLF001
        assert store._conn.execute("SELECT COUNT(*) FROM funding_cashflows").fetchone()[0] == 3  # noqa: SLF001
        store.close()

    def test_schema_ahead_v99_refused(self, tmp_path):
        db = tmp_path / "v99.sqlite3"
        conn = sqlite3.connect(db)
        conn.executescript(store_mod._SCHEMA)  # noqa: SLF001
        conn.execute("CREATE TABLE schema_meta (schema_version INTEGER, migrated_at_ms INTEGER)")
        conn.execute("INSERT INTO schema_meta VALUES (99, ?)", (NOW_MS,))
        conn.commit()
        conn.close()
        with pytest.raises(StoreError, match="高于本代码"):
            StateStore(db)
