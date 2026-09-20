"""kill/restart 恢复集成测试（实施计划书 v2.0 T3，AC-06/07）。

全部离线：fake adapter + 真实 StateStore + 真实 ExchangeStateSynchronizer +
真实 LiveService（make_service）。覆盖：

- 候选池外已有仓位出现在 current projection（恢复不受候选池限制）；
- 连续两次启动恢复不改变 facts（fills/income 幂等、游标只前进）；
- 账户接口失败 → capture 失败 → RECOVERING/RECOVERY，can_open=false；
- 对账不一致闸门：can_open=false 直到通过。
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from cointrader.execution.sync import ExchangeStateSynchronizer
from live_helpers import (
    NOW,
    FakeServiceAdapter,
    LiveFakeData,
    make_live_config,
    make_live_rates,
    make_service,
)

SYMBOL = "BTCUSDT"
OUT_OF_POOL = "DOGEUSDT"


def _attach_startup_fakes(svc) -> None:
    """fake 适配器补 startup 预检接口（calibrate / load_rules）。"""
    from live_helpers import make_symbol_rules

    spot, futures = svc.spot, svc.futures
    spot.calibrate = lambda samples=5: 0  # type: ignore[method-assign]
    futures.calibrate = lambda samples=5: 0  # type: ignore[method-assign]
    spot.load_rules = lambda: {SYMBOL: make_symbol_rules("spot")}  # type: ignore[method-assign]
    futures.load_rules = lambda: {SYMBOL: make_symbol_rules("perp")}  # type: ignore[method-assign]


def _tick_config() -> Any:
    return make_live_config(
        live_symbols=(SYMBOL,),
        exec_overrides={
            "reconciliation_interval_seconds": 3600,
            "snapshot_interval_seconds": 3600,
            "time_resync_seconds": 3600,
            "recovery_backfill_days": 30,
            "exchange_snapshot_reuse_seconds": 0.001,
        },
    )


class _RecoveryFuturesAdapter(FakeServiceAdapter):
    """带开放订单/成交/income 的期货 fake（恢复路径所需只读接口）。"""

    def __init__(self, position_symbol: str = "BTCUSDT") -> None:
        super().__init__("perp")
        self.position_symbol = position_symbol
        self.open_orders_list: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.income: list[dict[str, Any]] = []
        self.account_fail = False

    def account(self) -> dict[str, Any]:
        if self.account_fail:
            raise RuntimeError("futures account api down")
        return super().account()

    def positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if self.position_amt == 0:
            return []
        return [{
            "symbol": self.position_symbol, "positionSide": "BOTH",
            "positionAmt": str(self.position_amt), "entryPrice": "100",
        }]

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return list(self.open_orders_list)

    def user_trades(self, symbol: str, *, from_id=None, start_ms=None, limit=100) -> list[dict[str, Any]]:
        rows = [t for t in self.trades if t.get("symbol") == symbol]
        if start_ms is not None:
            rows = [r for r in rows if r["time"] >= start_ms]
        return rows[:limit]

    def income_history(
        self, *, income_type: str = "FUNDING_FEE", start_ms=None, end_ms=None, limit=1000
    ) -> list[dict[str, Any]]:
        rows = list(self.income)
        if start_ms is not None:
            rows = [r for r in rows if r["time"] >= start_ms]
        return rows[:limit]


class _RecoverySpotAdapter(FakeServiceAdapter):
    def __init__(self) -> None:
        super().__init__("spot")
        self.trades: list[dict[str, Any]] = []

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return []

    def my_trades(self, symbol: str, *, from_id=None, start_ms=None, limit=100) -> list[dict[str, Any]]:
        rows = [t for t in self.trades if t.get("symbol") == symbol]
        if start_ms is not None:
            rows = [r for r in rows if r["time"] >= start_ms]
        return rows[:limit]


def _seed_exchange_fakes(
    spot: _RecoverySpotAdapter,
    fut: _RecoveryFuturesAdapter,
    *,
    out_of_pool_position: bool = False,
) -> None:
    fut.position_amt = Decimal("-0.01")
    fut.position_symbol = OUT_OF_POOL if out_of_pool_position else SYMBOL
    spot.balances_map["BTC"] = Decimal("0.01")
    for i in range(5):
        fut.trades.append({
            "id": 500 + i, "clientOrderId": f"ct-seed-{i}", "orderId": 9000 + i,
            "price": "100", "qty": "0.01", "quoteQty": "1", "commission": "0.001",
            "commissionAsset": "USDT", "time": int(NOW * 1000) - (5 - i) * 60_000,
            "buyer": False, "seller": True, "symbol": SYMBOL,
        })
    for i in range(3):
        fut.income.append({
            "id": 700 + i, "incomeType": "FUNDING_FEE",
            "time": int(NOW * 1000) - (3 - i) * 8 * 3600 * 1000,
            "symbol": SYMBOL, "asset": "USDT", "income": "0.008",
        })


def _make_env(tmp_path, *, out_of_pool: bool = False, account_fail: bool = False) -> dict:
    data = LiveFakeData({SYMBOL: make_live_rates(20, "0.0005")})
    spot = _RecoverySpotAdapter()
    fut = _RecoveryFuturesAdapter()
    fut.account_fail = account_fail
    _seed_exchange_fakes(spot, fut, out_of_pool_position=out_of_pool)
    cfg = _tick_config()
    env = make_service(
        tmp_path, data, config=cfg, spot=spot, futures=fut,  # type: ignore[arg-type]
    )
    env["svc"].exch_sync = ExchangeStateSynchronizer(
        store=env["store"], spot=spot, futures=fut, config=cfg, now_fn=lambda: NOW
    )
    env["svc"]._ledger_sync_ok = False  # noqa: SLF001
    _attach_startup_fakes(env["svc"])
    env["spot"] = spot
    env["futures"] = fut
    return env


class TestRestartRecovery:
    def test_out_of_pool_position_restored_in_current_projection(self, tmp_path) -> None:
        """候选池外仓位（DOGE）：恢复不受候选池限制，出现在 current projection。"""
        env = _make_env(tmp_path, out_of_pool=True)
        svc = env["svc"]
        svc._now = lambda: NOW  # noqa: SLF001
        svc.run_id = ""
        report = svc.startup()
        assert report.mode == "paper"
        rows = env["store"].current_positions(include_tombstones=True)
        symbols = {r["symbol"] for r in rows}
        assert OUT_OF_POOL in symbols, "候选池外已有仓位必须出现在 current projection"
        doge = next(r for r in rows if r["symbol"] == OUT_OF_POOL)
        assert doge["perp_qty"] == "-0.01"
        acct = env["store"].current_account()
        assert acct is not None and acct["complete"] == 1

    def test_repeated_recovery_is_idempotent(self, tmp_path) -> None:
        """两次启动恢复（kill -9 后 systemd 拉起）：facts/游标/PnL 数量不变。"""
        env = _make_env(tmp_path)
        svc = env["svc"]
        svc._now = lambda: NOW  # noqa: SLF001
        svc.run_id = ""
        svc.startup()
        store = env["store"]
        fills_after_1 = len(store.fills(limit=1000))
        income_after_1 = len(store.funding_cashflows(limit=1000))
        cursors_after_1 = {
            (r[0], r[1], r[2]): (r[3], r[4])
            for r in store._conn.execute(  # noqa: SLF001
                "SELECT scope, stream, symbol_key, last_time_ms, last_id FROM sync_cursors"
            )
        }

        # 第二次「重启」：同一 store 的新服务实例（旧会话遗留 → 心跳截断）
        env2 = _make_env(tmp_path, out_of_pool=False)
        env2["store"].close()
        env2["store"] = store
        svc2 = env2["svc"]
        svc2.store = store
        # 其余组件仍持有 env2 已关闭的 store → 全部指回同一账本
        for comp in (svc2, svc2.executor, svc2.account_builder, svc2.reconciler,
                     svc2.strategy, svc2.gate):
            if comp is not None and hasattr(comp, "store"):
                comp.store = store
        svc2.exch_sync = ExchangeStateSynchronizer(
            store=store, spot=env2["spot"], futures=env2["futures"],
            config=env2["config"], now_fn=lambda: NOW,
        )
        svc2._now = lambda: NOW  # noqa: SLF001
        svc2.run_id = ""
        svc2.startup()

        fills_after_2 = len(store.fills(limit=1000))
        income_after_2 = len(store.funding_cashflows(limit=1000))
        assert fills_after_2 == fills_after_1, "重复恢复不得重复入账成交"
        assert income_after_2 == income_after_1, "重复恢复不得重复入账资金费"
        cursors_after_2 = {
            (r[0], r[1], r[2]): (r[3], r[4])
            for r in store._conn.execute(  # noqa: SLF001
                "SELECT scope, stream, symbol_key, last_time_ms, last_id FROM sync_cursors"
            )
        }
        assert cursors_after_2 == cursors_after_1, "游标幂等（重复回补不改变事实与游标）"
        # 旧会话被 heartbeat 截断标记
        old = store.run_session(svc.run_id)
        assert old is not None and old["status"] == "INTERRUPTED"

    def test_account_failure_blocks_can_open(self, tmp_path) -> None:
        """账户接口失败 → capture 失败 → RECOVERING（禁止开仓），不得用旧值放行。"""
        env = _make_env(tmp_path, account_fail=True)
        svc = env["svc"]
        svc._now = lambda: NOW  # noqa: SLF001
        svc.run_id = ""
        svc.startup()
        assert svc.state.value == "RECOVERY"
        assert svc.can_open is False
        assert "capture 失败" in svc._recovery_reason  # noqa: SLF001
        # 账户 current projection 未被不完整快照污染
        assert env["store"].current_account() is None

    def test_reconcile_gate_blocks_can_open_until_pass(self, tmp_path) -> None:
        """对账不一致闸门：RECOVERY 期间 can_open=false；对账通过 + 账本同步后回 RUNNING。"""
        env = _make_env(tmp_path)
        svc = env["svc"]
        svc._now = lambda: NOW  # noqa: SLF001
        svc.run_id = ""
        # 先制造不一致（本地期望 1 BTC 现货，交易所 0）
        env["reconciler"].run = lambda *, reason="periodic", snapshot=None: (  # type: ignore[method-assign]
            __import__("cointrader.execution.models", fromlist=["ReconciliationResult"])
            .ReconciliationResult(
                ts_ms=int(NOW * 1000), consistent=False, can_open=False,
                mismatches=("BTC: 不一致",), repaired=(),
            )
        )
        svc.startup()
        assert svc.state.value == "RECOVERY"
        assert svc.can_open is False

        from cointrader.execution.models import ReconciliationResult

        env["reconciler"].run = lambda *, reason="periodic", snapshot=None: ReconciliationResult(  # type: ignore[method-assign]
            ts_ms=int(NOW * 1000), consistent=True, can_open=True,
            mismatches=(), repaired=(),
        )
        # 流恢复新鲜后一次恢复对账 → 回到 RUNNING
        class _FreshStream:
            market = "spot"
            generation = 1
            is_fresh = True
            def start(self) -> None: ...
            def stop(self) -> None: ...

        svc.attach_streams([_FreshStream(), _FreshStream()])  # type: ignore[list-item]
        svc._last_reconcile_ms = 0  # noqa: SLF001
        result = svc.run_once()
        assert result["state"] == "RUNNING"
        assert svc._ledger_sync_ok is True  # noqa: SLF001
