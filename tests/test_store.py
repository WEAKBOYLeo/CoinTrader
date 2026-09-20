"""SQLite 事件账本单元测试（开发设计文档 §5.2 / §11.1）。

核心不变量：重复 clientOrderId / fill_id 不产生重复记录；
单实例锁阻止两个执行进程并发管理同一账户。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from decimal import Decimal

import pytest

from cointrader.execution.guard import Market, OrderSide, OrderType
from cointrader.execution.models import (
    AccountSnapshot,
    Fill,
    Order,
    OrderIntent,
    PairExecution,
    PositionSnapshot,
    ReconciliationResult,
)
from cointrader.execution.store import LeaseConflict, StateStore


@pytest.fixture
def store(tmp_path) -> Iterator[StateStore]:
    s = StateStore(tmp_path / "trading.sqlite3")
    yield s
    s.close()


def make_order(client_id: str, *, state: str = "NEW", qty: str = "1") -> Order:
    return Order(
        client_order_id=client_id,
        symbol="BTCUSDT",
        market=Market.SPOT,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal(qty),
        state=state,
        updated_ms=int(time.time() * 1000),
    )


def make_fill(fill_id: str, *, client_id: str = "ct-x") -> Fill:
    return Fill(
        fill_id=fill_id,
        client_order_id=client_id,
        symbol="BTCUSDT",
        market=Market.PERP,
        side=OrderSide.SELL,
        quantity=Decimal("0.5"),
        price=Decimal("100"),
        fee_asset="USDT",
        fee_amount=Decimal("0.05"),
        ts_ms=123456,
    )


class TestOrderIdempotency:
    def test_duplicate_client_order_id_is_update_not_insert(self, store: StateStore) -> None:
        o1 = make_order("ct-1", state="NEW")
        assert store.upsert_order(o1) is True, "首次写入应为新订单"

        o2 = make_order("ct-1", state="FILLED")
        o2.executed_qty = Decimal("1")
        o2.avg_price = Decimal("100")
        assert store.upsert_order(o2) is False, "重复 clientOrderId 必须走更新，不得重复插入"

        row = store.get_order("ct-1")
        assert row is not None
        assert row["state"] == "FILLED"
        assert Decimal(row["executed_qty"]) == Decimal("1")

    def test_distinct_client_order_ids_both_stored(self, store: StateStore) -> None:
        store.upsert_order(make_order("ct-a"))
        store.upsert_order(make_order("ct-b"))
        assert store.get_order("ct-a") is not None
        assert store.get_order("ct-b") is not None

    def test_get_order_unknown_returns_none(self, store: StateStore) -> None:
        assert store.get_order("ct-nope") is None


class TestFillIdempotency:
    def test_duplicate_fill_id_rejected(self, store: StateStore) -> None:
        assert store.record_fill(make_fill("f-1")) is True
        assert store.record_fill(make_fill("f-1")) is False, "重复 fill_id 必须幂等拒绝"
        assert store.record_fill(make_fill("f-2")) is True

    def test_expected_positions_aggregates_by_side(self, store: StateStore) -> None:
        # PERP SELL = 空头（负）
        store.record_fill(make_fill("f-1"))
        # SPOT BUY
        spot_fill = Fill(
            fill_id="f-2",
            client_order_id="ct-x",
            symbol="BTCUSDT",
            market=Market.SPOT,
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
            price=Decimal("100"),
            fee_asset="USDT",
            fee_amount=Decimal("0"),
            ts_ms=1,
        )
        store.record_fill(spot_fill)
        expected = store.expected_positions()
        assert expected["BTCUSDT"]["SPOT"] == Decimal("0.5")
        assert expected["BTCUSDT"]["PERP"] == Decimal("-0.5")


class TestIntentIdempotency:
    def test_duplicate_intent_rejected(self, store: StateStore) -> None:
        intent = OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            market=Market.PERP,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=Decimal("100"),
            intent_id="int-fixed-1",
        )
        assert store.record_intent(intent) is True
        assert store.record_intent(intent) is False, "重复 intent 不得重复记录"
        assert store.open_intent_count() == 1


class TestPairLifecycle:
    def test_active_pair_blocks_second_open(self, store: StateStore) -> None:
        pair = PairExecution(
            pair_execution_id="pair-1",
            symbol="BTCUSDT",
            target_notional=Decimal("100"),
            strategy_version="v1",
            created_ms=1,
            updated_ms=1,
        )
        store.upsert_pair(pair)
        assert store.active_pair_for_symbol("BTCUSDT") == "pair-1"

    def test_terminal_pair_frees_symbol(self, store: StateStore) -> None:
        pair = PairExecution(
            pair_execution_id="pair-2",
            symbol="BTCUSDT",
            target_notional=Decimal("100"),
            strategy_version="v1",
            status="COMPLETE",
            created_ms=1,
            updated_ms=1,
        )
        store.upsert_pair(pair)
        assert store.active_pair_for_symbol("BTCUSDT") is None, "终态 pair 不得阻止重新开仓"

    def test_upsert_updates_status(self, store: StateStore) -> None:
        pair = PairExecution(
            pair_execution_id="pair-3",
            symbol="ETHUSDT",
            target_notional=Decimal("50"),
            strategy_version="v1",
            created_ms=1,
            updated_ms=1,
        )
        store.upsert_pair(pair)
        pair.touches("FILLED")
        store.upsert_pair(pair)
        assert store.active_pair_for_symbol("ETHUSDT") == "pair-3"


class TestLease:
    def test_second_holder_conflicts(self, store: StateStore) -> None:
        lease = store.acquire_lease("executor", holder="process-A")
        assert lease.holder == "process-A"
        with pytest.raises(LeaseConflict, match="已被 process-A 持有"):
            store.acquire_lease("executor", holder="process-B")

    def test_same_holder_reacquires(self, store: StateStore) -> None:
        store.acquire_lease("executor", holder="process-A")
        lease = store.acquire_lease("executor", holder="process-A")
        assert lease.holder == "process-A"

    def test_expired_lease_can_be_taken(self, store: StateStore) -> None:
        store.acquire_lease("executor", holder="process-A", ttl_seconds=0.05)
        time.sleep(0.1)
        lease = store.acquire_lease("executor", holder="process-B")
        assert lease.holder == "process-B", "过期锁必须可被新进程接管"

    def test_release_frees_lease(self, store: StateStore) -> None:
        store.acquire_lease("executor", holder="process-A")
        store.release_lease("executor", "process-A")
        lease = store.acquire_lease("executor", holder="process-B")
        assert lease.holder == "process-B"

    def test_release_only_by_holder(self, store: StateStore) -> None:
        store.acquire_lease("executor", holder="process-A")
        store.release_lease("executor", "intruder")  # 不属于自己，释放无效
        with pytest.raises(LeaseConflict):
            store.acquire_lease("executor", holder="process-B")


class TestRunSessionInterruption:
    """断点重连：非优雅退出会话的 INTERRUPTED 标记（计划 1.0 T1）。"""

    def _start(self, store: StateStore, run_id: str, started_ms: int) -> None:
        store.start_run_session(
            run_id=run_id, started_ms=started_ms, mode="testnet",
            strategy_version="t", config_hash="h", code_revision="rev",
            spot_endpoint="e", futures_endpoint="f", user_stream_mode="poll",
        )

    def test_marks_running_and_recovery_null_sessions(self, store: StateStore) -> None:
        self._start(store, "run-a", 1000)
        self._start(store, "run-b", 2000)
        store.update_run_session("run-a", status="RECOVERY")  # 模拟处于 RECOVERY 时进程被杀
        # T3 v2：run-b 有心跳 → 按 heartbeat 截断；run-a 无心跳 → 截断到 started_ms
        store.update_run_heartbeat("run-b", now_ms=4000)

        n = store.mark_interrupted_sessions(now_ms=9000)
        assert n == 2
        row_a = store.run_session("run-a")
        assert row_a is not None
        assert row_a["status"] == "INTERRUPTED"
        assert row_a["ended_ms"] == 1000  # 无 heartbeat → started_ms（停机间隔不计在线）
        assert row_a["stop_reason"] == "process_exited_without_graceful_stop"
        row_b = store.run_session("run-b")
        assert row_b is not None
        assert row_b["status"] == "INTERRUPTED"
        assert row_b["ended_ms"] == 4000  # 有 heartbeat → 按心跳截断（不是重启时刻 9000）

    def test_closed_sessions_untouched(self, store: StateStore) -> None:
        self._start(store, "run-a", 1000)
        store.end_run_session("run-a", ended_ms=5000, status="STOPPED", stop_reason="graceful_stop")
        self._start(store, "run-b", 6000)

        n = store.mark_interrupted_sessions(now_ms=9000)
        assert n == 1
        closed = store.run_session("run-a")
        assert closed is not None
        assert closed["status"] == "STOPPED" and closed["ended_ms"] == 5000

    def test_idempotent_second_call_returns_zero(self, store: StateStore) -> None:
        self._start(store, "run-a", 1000)
        assert store.mark_interrupted_sessions(now_ms=9000) == 1
        assert store.mark_interrupted_sessions(now_ms=9100) == 0


class TestReconciliationRecord:
    def test_record_reconciliation_roundtrip(self, store: StateStore) -> None:
        result = ReconciliationResult(
            ts_ms=1,
            consistent=True,
            mismatches=(),
            repaired=("r1",),
            can_open=True,
        )
        store.record_reconciliation(result)
        assert result.fingerprint()

    def test_inconsistent_result_requires_mismatches(self) -> None:
        with pytest.raises(ValueError, match="逻辑矛盾"):
            ReconciliationResult(
                ts_ms=1,
                consistent=True,
                mismatches=("x",),
                repaired=(),
                can_open=True,
            )


class TestSnapshots:
    def test_position_and_account_snapshots(self, store: StateStore) -> None:
        store.record_position_snapshot(
            PositionSnapshot(
                ts_ms=1,
                symbol="BTCUSDT",
                spot_qty=Decimal("1"),
                perp_qty=Decimal("-1"),
                spot_price=Decimal("100"),
                perp_price=Decimal("100.5"),
                basis_pct=Decimal("0.005"),
            )
        )
        store.record_account_snapshot(
            AccountSnapshot(
                ts_ms=1,
                source="SPOT",
                available_balance=Decimal("1000"),
                wallet_balance=Decimal("1000"),
            )
        )
        # 写入成功即可（无异常），快照是事件历史
        store.record_risk_decision("ALLOW_OPEN", True, "通过")
        assert store.record_event(
            fingerprint="fp-1", market="spot", event_type="executionReport",
            exchange_ts=1, payload={"k": "v"},
        )
        assert not store.record_event(
            fingerprint="fp-1", market="spot", event_type="executionReport",
            exchange_ts=1, payload={"k": "v"},
        ), "重复指纹事件必须幂等拒绝"

    def test_no_secret_columns_in_schema(self, store: StateStore) -> None:
        with store._lock:  # noqa: SLF001
            cols = [r[1] for r in store._conn.execute("PRAGMA table_info(orders)")]  # noqa: SLF001
        assert not any("secret" in c or "signature" in c for c in cols), "账本不得保存签名/密钥"


class TestOnlineStats:
    """v2 在线/中断时长口径（T4/AC-09，固定时钟手算）。

    规则：已结束 run 计 [started, ended]（INTERRUPTED 按 heartbeat 截断）；
    未结束 run 计到 min(now, last_heartbeat + freshness)；重叠/负值截断；
    重启不清零累计。
    """

    FRESH = 120_000  # heartbeat 新鲜度裕量（与 online_stats 默认一致）

    def _start(self, store: StateStore, run_id: str, started_ms: int) -> None:
        store.start_run_session(
            run_id=run_id, started_ms=started_ms, mode="testnet",
            strategy_version="t", config_hash="h", code_revision="rev",
            spot_endpoint="e", futures_endpoint="f", user_stream_mode="poll",
        )

    def test_graceful_stop_counts_full_duration(self, store: StateStore) -> None:
        self._start(store, "run-1", 1_000)
        store.end_run_session("run-1", ended_ms=61_000, status="STOPPED",
                              stop_reason="graceful_stop")
        stats = store.online_stats(now_ms=1_000_000)
        assert stats["run_count"] == 1
        assert stats["total_online_ms"] == 60_000
        assert stats["first_start_ms"] == 1_000
        assert stats["last_end_ms"] == 61_000
        assert stats["current_run_online_ms"] is None
        # span = 首启 → now（含当前停机段）；downtime = span - 在线
        assert stats["span_ms"] == 1_000_000 - 1_000
        assert stats["total_downtime_ms"] == stats["span_ms"] - 60_000

    def test_kill_then_late_restart_counts_downtime_not_online(
        self, store: StateStore
    ) -> None:
        """kill 后 5 小时才重启：停机 5h 不计在线，累计不重置。"""
        self._start(store, "run-1", 1_000)
        store.update_run_heartbeat("run-1", now_ms=361_000)  # 在线 6 分钟后被杀
        store.mark_interrupted_sessions(now_ms=361_000 + 1)  # ended 按 heartbeat 截断
        self._start(store, "run-2", 361_000 + 5 * 3_600_000)  # 5 小时后重启
        store.update_run_heartbeat("run-2", now_ms=361_000 + 5 * 3_600_000 + 60_000)
        now = 361_000 + 5 * 3_600_000 + 60_000
        stats = store.online_stats(now_ms=now)
        assert stats["run_count"] == 2
        # run-1 在线 6min（heartbeat 截断）+ run-2 在线 60s
        assert stats["total_online_ms"] == 360_000 + 60_000
        assert stats["current_run_online_ms"] == 60_000
        assert stats["span_ms"] == now - 1_000
        assert stats["total_downtime_ms"] == stats["span_ms"] - (360_000 + 60_000)

    def test_open_run_without_fresh_heartbeat_not_counted_past_allowance(
        self, store: StateStore
    ) -> None:
        """心跳停更 10 分钟的未结束会话：在线只计到 heartbeat+freshness。"""
        self._start(store, "run-1", 1_000)
        store.update_run_heartbeat("run-1", now_ms=121_000)
        # now 比心跳晚 10 分钟（远超 2 分钟裕量）
        stats = store.online_stats(now_ms=121_000 + 600_000)
        expected_end = 121_000 + self.FRESH
        assert stats["total_online_ms"] == expected_end - 1_000
        assert stats["current_run_online_ms"] == expected_end - 1_000

    def test_overlapping_and_negative_sessions_clamped(self, store: StateStore) -> None:
        """重叠区间只计一次；ended < started 的脏数据不产生负在线。"""
        self._start(store, "run-1", 1_000)
        store.end_run_session("run-1", ended_ms=61_000, status="STOPPED",
                              stop_reason="graceful_stop")
        self._start(store, "run-2", 30_000)  # 与 run-1 重叠 30s
        store.end_run_session("run-2", ended_ms=91_000, status="STOPPED",
                              stop_reason="graceful_stop")
        self._start(store, "run-3", 200_000)
        store.end_run_session("run-3", ended_ms=150_000, status="STOPPED",  # 负时长脏数据
                              stop_reason="graceful_stop")
        stats = store.online_stats(now_ms=1_000_000)
        # run-1+run-2 合并为 [1_000, 91_000] = 90s；run-3 = 0
        assert stats["total_online_ms"] == 90_000
        assert stats["total_downtime_ms"] >= 0

    def test_three_runs_cumulative_never_resets(self, store: StateStore) -> None:
        """三次 run：累计 = 各段在线之和（重启不清零）。"""
        for i, (start, end) in enumerate(((1_000, 101_000), (200_000, 251_000),
                                          (300_000, 331_000)), start=1):
            self._start(store, f"run-{i}", start)
            store.end_run_session(f"run-{i}", ended_ms=end, status="STOPPED",
                                  stop_reason="graceful_stop")
        stats = store.online_stats(now_ms=1_000_000)
        assert stats["run_count"] == 3
        assert stats["total_online_ms"] == 100_000 + 51_000 + 31_000
        assert stats["first_start_ms"] == 1_000
        assert stats["span_ms"] == 1_000_000 - 1_000
        assert stats["total_downtime_ms"] == stats["span_ms"] - stats["total_online_ms"]
