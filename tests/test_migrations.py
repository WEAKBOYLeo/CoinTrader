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
        assert schema_version_of(store._conn) == 1  # noqa: SLF001
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
        assert apply_migrations(conn) == 1  # 再跑一遍不报错
        assert schema_version_of(conn) == 1
        # 第三次：仍然 OK
        assert apply_migrations(conn) == 1
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
        assert schema_version_of(store._conn) == 1  # noqa: SLF001
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
        assert schema_version_of(store._conn) == 1  # noqa: SLF001
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
        conn.execute("INSERT INTO schema_meta VALUES (2, 0)")
        with pytest.raises(MigrationError):
            apply_migrations(conn)
        conn.close()
