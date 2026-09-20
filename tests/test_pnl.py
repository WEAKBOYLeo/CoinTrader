"""PnL 聚合测试（开发文档 §8.4：手工计算固定 fixture 验证分项口径）。

口径 pnl-v1：
    funding_pnl  = Σ 已确认资金费
    trading_fee  = -Σ 两腿手续费价值
    basis_pnl    = qty × (开仓基差 − 平仓基差)
    realized     = funding + fee + basis
"""

from __future__ import annotations

from decimal import Decimal

from cointrader.execution.guard import Market, OrderSide, OrderType
from cointrader.execution.models import Fill, Order, PairExecution
from cointrader.execution.store import StateStore
from cointrader.reporting.pnl import CALC_VERSION, PnlAggregator

T0 = 1_790_000_000_000
T1 = T0 + 10 * 8 * 3600 * 1000


def _order(cid: str, pair_id: str, market: Market, side: OrderSide) -> Order:
    return Order(
        client_order_id=cid,
        symbol="BTCUSDT",
        market=market,
        side=side,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.01"),
        state="FILLED",
        executed_qty=Decimal("0.01"),
        pair_execution_id=pair_id,
        updated_ms=T0,
    )


def _fill(fid: str, cid: str, market: Market, side: OrderSide,
          price: str, fee: str, ts: int) -> Fill:
    return Fill(
        fill_id=fid,
        client_order_id=cid,
        symbol="BTCUSDT",
        market=market,
        side=side,
        quantity=Decimal("0.01"),
        price=Decimal(price),
        fee_asset="USDT",
        fee_amount=Decimal(fee),
        ts_ms=ts,
        exchange_trade_id=f"ex-{fid}",
    )


def _seed_round_trip(store: StateStore, *,
                     spot_entry: str = "100", perp_entry: str = "100",
                     spot_exit: str = "100", perp_exit: str = "100",
                     funding: str | None = "0.5",
                     run_id: str = "run-pnl") -> tuple[dict, dict]:
    """开 0.01 BTC（spot 买 + perp 卖），10 期后平（spot 卖 + perp 买）。"""
    open_pair = PairExecution(
        pair_execution_id="pair-open-1", symbol="BTCUSDT",
        target_notional=Decimal("1"), status="COMPLETE", kind="open",
        strategy_version="t", run_id=run_id,
        created_ms=T0, updated_ms=T0, completed_ts_ms=T0,
    )
    close_pair = PairExecution(
        pair_execution_id="pair-close-1", symbol="BTCUSDT",
        target_notional=Decimal("0"), status="COMPLETE", kind="close",
        strategy_version="t", run_id=run_id,
        created_ms=T1, updated_ms=T1, completed_ts_ms=T1,
    )
    store.upsert_pair(open_pair)
    store.upsert_pair(close_pair)

    orders = [
        _order("co-open-spot", "pair-open-1", Market.SPOT, OrderSide.BUY),
        _order("co-open-perp", "pair-open-1", Market.PERP, OrderSide.SELL),
        _order("co-close-spot", "pair-close-1", Market.SPOT, OrderSide.SELL),
        _order("co-close-perp", "pair-close-1", Market.PERP, OrderSide.BUY),
    ]
    for o in orders:
        store.upsert_order(o)

    fills = [
        _fill("f1", "co-open-spot", Market.SPOT, OrderSide.BUY, spot_entry, "0.001", T0),
        _fill("f2", "co-open-perp", Market.PERP, OrderSide.SELL, perp_entry, "0.0005", T0),
        _fill("f3", "co-close-spot", Market.SPOT, OrderSide.SELL, spot_exit, "0.001", T1),
        _fill("f4", "co-close-perp", Market.PERP, OrderSide.BUY, perp_exit, "0.0005", T1),
    ]
    for f in fills:
        assert store.record_fill(f) is True

    if funding is not None:
        assert store.record_funding_cashflow(
            cashflow_id="cf-1", symbol="BTCUSDT",
            funding_ts_ms=T0 + 8 * 3600 * 1000,
            funding_rate=Decimal("0.0005"), interval_hours=8,
            amount=Decimal(funding), source="test", run_id=run_id,
        ) is True

    open_row = store.get_pair("pair-open-1")
    close_row = store.get_pair("pair-close-1")
    assert open_row is not None and close_row is not None
    return open_row, close_row


class TestForTrip:
    def test_flat_prices_funding_only(self, tmp_path):
        """零基差变化：PnL = 资金费 − 手续费（手工口径）。"""
        store = StateStore(tmp_path / "t.sqlite3")
        open_row, close_row = _seed_round_trip(store)
        pnl = PnlAggregator(store, now_fn=lambda: T1 / 1000).for_trip(open_row, close_row)

        assert pnl.funding_pnl == Decimal("0.5")
        # 0.001 + 0.0005 + 0.001 + 0.0005 = 0.003
        assert pnl.trading_fee == -Decimal("0.003")
        assert pnl.basis_pnl == Decimal("0")
        assert pnl.realized_pnl == Decimal("0.497")
        assert pnl.unrealized_pnl == Decimal("0")
        assert pnl.net_pnl == Decimal("0.497")
        assert pnl.calc_version == CALC_VERSION
        assert pnl.kind == "round_trip"
        assert pnl.status == "COMPLETE"

    def test_basis_gain_counted(self, tmp_path):
        """开仓基差 +1、平仓基差 +0.5 → 基差收益 0.01 × 0.5 = 0.005。"""
        store = StateStore(tmp_path / "t.sqlite3")
        open_row, close_row = _seed_round_trip(
            store, perp_entry="101", spot_exit="100", perp_exit="100.5", funding=None)
        pnl = PnlAggregator(store, now_fn=lambda: T1 / 1000).for_trip(open_row, close_row)
        assert pnl.funding_pnl == Decimal("0")
        assert pnl.basis_pnl == Decimal("0.005")
        assert pnl.realized_pnl == Decimal("0.005") - Decimal("0.003")

    def test_basis_loss_counted(self, tmp_path):
        """开仓基差 0、平仓基差 +0.5 → 基差损失 −0.005（卖出时付出溢价）。"""
        store = StateStore(tmp_path / "t.sqlite3")
        open_row, close_row = _seed_round_trip(
            store, spot_entry="100", perp_entry="100", perp_exit="100.5", funding=None)
        pnl = PnlAggregator(store, now_fn=lambda: T1 / 1000).for_trip(open_row, close_row)
        assert pnl.basis_pnl == -Decimal("0.005")

    def test_open_only_is_unrealized(self, tmp_path):
        """未平仓：funding+fee 计入未实现，不用未确认报价冒充已实现。"""
        store = StateStore(tmp_path / "t.sqlite3")
        open_pair = PairExecution(
            pair_execution_id="pair-open-2", symbol="BTCUSDT",
            target_notional=Decimal("1"), status="COMPLETE", kind="open",
            strategy_version="t", run_id="run-pnl",
            created_ms=T0, updated_ms=T0, completed_ts_ms=T0,
        )
        store.upsert_pair(open_pair)
        for o in (_order("co-open-spot", "pair-open-2", Market.SPOT, OrderSide.BUY),
                  _order("co-open-perp", "pair-open-2", Market.PERP, OrderSide.SELL)):
            store.upsert_order(o)
        store.record_fill(_fill("f1", "co-open-spot", Market.SPOT, OrderSide.BUY, "100", "0.001", T0))
        store.record_fill(_fill("f2", "co-open-perp", Market.PERP, OrderSide.SELL, "100", "0.0005", T0))
        store.record_funding_cashflow(
            cashflow_id="cf-1", symbol="BTCUSDT",
            funding_ts_ms=T0 + 8 * 3600 * 1000,
            funding_rate=Decimal("0.0005"), interval_hours=8,
            amount=Decimal("0.5"), source="test",
        )
        open_row = store.get_pair("pair-open-2")
        agg = PnlAggregator(store, now_fn=lambda: T1 / 1000)
        pnl = agg.for_trip(open_row, None)
        assert pnl.kind == "open"
        assert pnl.realized_pnl == Decimal("0")
        # 无报价 → 基差不估值（notes 标记），只确认 funding − fee
        assert pnl.unrealized_pnl == Decimal("0.5") - Decimal("0.0015")
        assert any("报价" in n for n in pnl.notes)

    def test_funding_outside_window_excluded(self, tmp_path):
        store = StateStore(tmp_path / "t.sqlite3")
        _seed_round_trip(store, funding=None)
        # 平仓之后的资金费不属于本 round trip
        store.record_funding_cashflow(
            cashflow_id="cf-late", symbol="BTCUSDT",
            funding_ts_ms=T1 + 8 * 3600 * 1000,
            funding_rate=Decimal("0.0005"), interval_hours=8,
            amount=Decimal("9.9"), source="test",
        )
        open_row, close_row = store.get_pair("pair-open-1"), store.get_pair("pair-close-1")
        pnl = PnlAggregator(store, now_fn=lambda: T1 / 1000).for_trip(open_row, close_row)
        assert pnl.funding_pnl == Decimal("0")


class TestForRun:
    def test_run_aggregates_round_trips(self, tmp_path):
        store = StateStore(tmp_path / "t.sqlite3")
        open_row, close_row = _seed_round_trip(store, run_id="run-a")
        # 第二笔（亏损）：无资金费 + 费用
        store.upsert_pair(PairExecution(
            pair_execution_id="pair-open-2", symbol="BTCUSDT",
            target_notional=Decimal("1"), status="COMPLETE", kind="open",
            strategy_version="t", run_id="run-a",
            created_ms=T1, updated_ms=T1, completed_ts_ms=T1,
        ))
        store.upsert_pair(PairExecution(
            pair_execution_id="pair-close-2", symbol="BTCUSDT",
            target_notional=Decimal("0"), status="COMPLETE", kind="close",
            strategy_version="t", run_id="run-a",
            created_ms=T1 + 8 * 3600 * 1000, updated_ms=T1 + 8 * 3600 * 1000,
            completed_ts_ms=T1 + 8 * 3600 * 1000,
        ))
        for o in (_order("co2-open-spot", "pair-open-2", Market.SPOT, OrderSide.BUY),
                  _order("co2-open-perp", "pair-open-2", Market.PERP, OrderSide.SELL),
                  _order("co2-close-spot", "pair-close-2", Market.SPOT, OrderSide.SELL),
                  _order("co2-close-perp", "pair-close-2", Market.PERP, OrderSide.BUY)):
            store.upsert_order(o)
        store.record_fill(_fill("g1", "co2-open-spot", Market.SPOT, OrderSide.BUY, "100", "0.001", T1))
        store.record_fill(_fill("g2", "co2-open-perp", Market.PERP, OrderSide.SELL, "100", "0.0005", T1))
        store.record_fill(_fill("g3", "co2-close-spot", Market.SPOT, OrderSide.SELL, "100", "0.001", T1 + 8 * 3600 * 1000))
        store.record_fill(_fill("g4", "co2-close-perp", Market.PERP, OrderSide.BUY, "100", "0.0005", T1 + 8 * 3600 * 1000))

        summary = PnlAggregator(store, now_fn=lambda: T1 / 1000).for_run("run-a")
        assert len(summary.per_pair) == 2
        # 第一笔 0.497，第二笔 −0.003
        assert summary.net_pnl == Decimal("0.497") - Decimal("0.003")
        assert summary.funding_pnl == Decimal("0.5")
        assert summary.trading_fee == -Decimal("0.006")
        # 无账户快照 → cash_delta 无法交叉验证（显式标记，不猜）
        assert summary.cash_delta is None
        assert any("cash_delta" in d for d in summary.differences)

    def test_run_filters_by_run_id(self, tmp_path):
        store = StateStore(tmp_path / "t.sqlite3")
        _seed_round_trip(store, run_id="run-a")
        summary = PnlAggregator(store, now_fn=lambda: T1 / 1000).for_run("run-other")
        assert summary.per_pair == []
        assert summary.net_pnl == Decimal("0")


def _seed_closed_trip_b(store: StateStore) -> None:
    """run-b 的已平仓 round trip：零基差、无资金费 → net = −fee = −0.003。"""
    store.upsert_pair(PairExecution(
        pair_execution_id="pair-open-3", symbol="BTCUSDT",
        target_notional=Decimal("1"), status="COMPLETE", kind="open",
        strategy_version="t", run_id="run-b",
        created_ms=T1, updated_ms=T1, completed_ts_ms=T1,
    ))
    store.upsert_pair(PairExecution(
        pair_execution_id="pair-close-3", symbol="BTCUSDT",
        target_notional=Decimal("0"), status="COMPLETE", kind="close",
        strategy_version="t", run_id="run-b",
        created_ms=T1 + 8 * 3600 * 1000, updated_ms=T1 + 8 * 3600 * 1000,
        completed_ts_ms=T1 + 8 * 3600 * 1000,
    ))
    for o in (_order("co3-open-spot", "pair-open-3", Market.SPOT, OrderSide.BUY),
              _order("co3-open-perp", "pair-open-3", Market.PERP, OrderSide.SELL),
              _order("co3-close-spot", "pair-close-3", Market.SPOT, OrderSide.SELL),
              _order("co3-close-perp", "pair-close-3", Market.PERP, OrderSide.BUY)):
        store.upsert_order(o)
    store.record_fill(_fill("h1", "co3-open-spot", Market.SPOT, OrderSide.BUY, "100", "0.001", T1))
    store.record_fill(_fill("h2", "co3-open-perp", Market.PERP, OrderSide.SELL, "100", "0.0005", T1))
    store.record_fill(_fill("h3", "co3-close-spot", Market.SPOT, OrderSide.SELL, "100", "0.001", T1 + 8 * 3600 * 1000))
    store.record_fill(_fill("h4", "co3-close-perp", Market.PERP, OrderSide.BUY, "100", "0.0005", T1 + 8 * 3600 * 1000))


def _seed_open_eth_trip(store: StateStore) -> None:
    """run-a 开仓后一直未平（跨 run 未平仓）：funding − fee 计入未实现。"""
    open_pair = PairExecution(
        pair_execution_id="pair-open-4", symbol="ETHUSDT",
        target_notional=Decimal("1"), status="COMPLETE", kind="open",
        strategy_version="t", run_id="run-a",
        created_ms=T0, updated_ms=T0, completed_ts_ms=T0,
    )
    store.upsert_pair(open_pair)
    for o in (_order("co4-open-spot", "pair-open-4", Market.SPOT, OrderSide.BUY),
              _order("co4-open-perp", "pair-open-4", Market.PERP, OrderSide.SELL)):
        store.upsert_order(o)
    store.record_fill(_fill("i1", "co4-open-spot", Market.SPOT, OrderSide.BUY, "100", "0.001", T0))
    store.record_fill(_fill("i2", "co4-open-perp", Market.PERP, OrderSide.SELL, "100", "0.0005", T0))
    store.record_funding_cashflow(
        cashflow_id="cf-eth", symbol="ETHUSDT",
        funding_ts_ms=T0 + 8 * 3600 * 1000,
        funding_rate=Decimal("0.0005"), interval_hours=8,
        amount=Decimal("0.3"), source="test", run_id="run-a",
    )


class TestForAllRuns:
    """跨 run 聚合（断点重连后的全局视图，计划 1.0 T2 / AC-04）。"""

    def test_all_runs_equals_sum_of_run_views(self, tmp_path):
        store = StateStore(tmp_path / "t.sqlite3")
        _seed_round_trip(store, run_id="run-a")
        _seed_closed_trip_b(store)
        _seed_open_eth_trip(store)
        agg = PnlAggregator(store, now_fn=lambda: T1 / 1000)

        sa, sb = agg.for_run("run-a"), agg.for_run("run-b")
        total = agg.for_all_runs()

        assert len(total.per_pair) == 3
        assert {p.symbol for p in total.per_pair} == {"BTCUSDT", "ETHUSDT"}
        assert sa.net_pnl + sb.net_pnl == Decimal("0.7925")
        assert total.net_pnl == sa.net_pnl + sb.net_pnl
        assert total.funding_pnl == sa.funding_pnl + sb.funding_pnl
        assert total.trading_fee == sa.trading_fee + sb.trading_fee
        assert total.realized_pnl == sa.realized_pnl + sb.realized_pnl
        assert total.unrealized_pnl == sa.unrealized_pnl + sb.unrealized_pnl

    def test_all_runs_empty_store(self, tmp_path):
        store = StateStore(tmp_path / "t.sqlite3")
        total = PnlAggregator(store, now_fn=lambda: T1 / 1000).for_all_runs()
        assert total.per_pair == []
        assert total.net_pnl == Decimal("0")


class TestPersist:
    def test_persist_round_trip_writes_ledger(self, tmp_path):
        store = StateStore(tmp_path / "t.sqlite3")
        open_row, close_row = _seed_round_trip(store, run_id="run-p")
        agg = PnlAggregator(store, now_fn=lambda: T1 / 1000)
        pnl = agg.persist_round_trip(open_row, close_row)
        rows = [r for r in store.pnl_ledger(limit=100)
                if r.get("pair_execution_id") == "pair-open-1"]
        kinds = {r["kind"] for r in rows}
        assert kinds == {"FUNDING", "FEE", "BASIS", "REALIZED"}
        assert all(r["calculation_version"] == CALC_VERSION for r in rows)
        assert all(r["run_id"] == "run-p" for r in rows)
        # pair 行回填 PnL
        closed = store.get_pair("pair-close-1")
        assert Decimal(str(closed["realized_pnl"])) == pnl.realized_pnl

    def test_persist_idempotent(self, tmp_path):
        store = StateStore(tmp_path / "t.sqlite3")
        open_row, close_row = _seed_round_trip(store)
        agg = PnlAggregator(store, now_fn=lambda: T1 / 1000)
        agg.persist_round_trip(open_row, close_row)
        agg.persist_round_trip(open_row, close_row)  # 重复不新增
        rows = [r for r in store.pnl_ledger(limit=100)
                if r.get("pair_execution_id") == "pair-open-1"]
        assert len(rows) == 4

    def test_persist_requires_closed_trip(self, tmp_path):
        store = StateStore(tmp_path / "t.sqlite3")
        open_row, _ = _seed_round_trip(store)
        agg = PnlAggregator(store, now_fn=lambda: T1 / 1000)
        try:
            agg.persist_round_trip(open_row, None)
            raise AssertionError("未平仓不得写 REALIZED 账本")
        except ValueError:
            pass
