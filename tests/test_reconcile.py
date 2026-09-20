"""对账器单元测试（开发设计文档 §8 / §11.1）。

核心不变量：对账不一致 → can_open=False；本地状态滞后按交易所修复；
对账查询失败 = 状态不可信。
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

import pytest

from cointrader.execution.guard import Market, OrderSide, OrderType
from cointrader.execution.models import Fill, Order
from cointrader.execution.reconcile import Reconciler
from cointrader.execution.store import StateStore


class FakeAdapter:
    def __init__(self, rules: dict[str, Any] | None = None) -> None:
        self.rules = rules or {}
        self.open_orders_list: list[dict[str, Any]] = []
        self.order_results: dict[str, Order | None] = {}
        self.order_query_error: Exception | None = None
        self.balances_map: dict[str, Decimal] = {}
        self.positions_list: list[dict[str, Any]] = []
        self.query_calls: list[str] = []

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:  # noqa: ARG002
        if self.order_query_error is not None:
            raise self.order_query_error
        return list(self.open_orders_list)

    def query_by_client_order_id(self, symbol: str, client_order_id: str) -> Order | None:  # noqa: ARG001
        self.query_calls.append(client_order_id)
        return self.order_results.get(client_order_id)

    def balances(self) -> dict[str, Decimal]:
        return dict(self.balances_map)

    def positions(self, symbol: str | None = None) -> list[dict[str, Any]]:  # noqa: ARG002
        return list(self.positions_list)

    def rule(self, symbol: str) -> Any:
        if symbol not in self.rules:
            raise KeyError(symbol)
        return self.rules[symbol]


@pytest.fixture
def env(tmp_path) -> dict[str, Any]:
    store = StateStore(tmp_path / "trading.sqlite3")
    spot = FakeAdapter()
    perp = FakeAdapter()
    reconciler = Reconciler(store, spot, perp)  # type: ignore[arg-type]  # 结构化 FakeAdapter
    return {"store": store, "spot": spot, "perp": perp, "reconciler": reconciler}


def order(client_id: str, *, state: str = "FILLED") -> Order:
    return Order(
        client_order_id=client_id,
        symbol="BTCUSDT",
        market=Market.SPOT,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1"),
        state=state,
        updated_ms=int(time.time() * 1000),
    )


def spot_fill(symbol: str, qty: str) -> Fill:
    return Fill(
        fill_id=f"fill-{symbol}-{qty}",
        client_order_id="ct-x",
        symbol=symbol,
        market=Market.SPOT,
        side=OrderSide.BUY,
        quantity=Decimal(qty),
        price=Decimal("100"),
        fee_asset="USDT",
        fee_amount=Decimal("0"),
        ts_ms=1,
    )


class TestConsistency:
    def test_fully_consistent_allows_open(self, env: dict) -> None:
        result = env["reconciler"].run(reason="startup")
        assert result.consistent is True
        assert result.can_open is True
        assert result.mismatches == ()

    def test_unknown_open_order_on_exchange_blocks_open(self, env: dict) -> None:
        env["spot"].open_orders_list.append(
            {"clientOrderId": "ct-ghost", "symbol": "BTCUSDT", "origQty": "1", "type": "MARKET"}
        )
        result = env["reconciler"].run()
        assert result.consistent is False
        assert result.can_open is False, "交易所存在本地未知订单时必须禁止开新仓"
        assert any("ct-ghost" in m for m in result.mismatches)

    def test_position_mismatch_blocks_open(self, env: dict) -> None:
        env["store"].record_fill(spot_fill("BTCUSDT", "1"))
        env["spot"].balances_map["BTC"] = Decimal("0.5")
        result = env["reconciler"].run()
        assert result.consistent is False
        assert result.can_open is False
        assert any("Spot 余额不一致" in m for m in result.mismatches)

    def test_perp_position_mismatch_blocks_open(self, env: dict) -> None:
        env["store"].record_fill(
            Fill(
                fill_id="f-perp", client_order_id="ct-x", symbol="BTCUSDT", market=Market.PERP,
                side=OrderSide.SELL, quantity=Decimal("1"), price=Decimal("100"),
                fee_asset="USDT", fee_amount=Decimal("0"), ts_ms=1,
            )
        )
        env["perp"].positions_list.append({"symbol": "BTCUSDT", "positionSide": "BOTH", "positionAmt": "-0.4"})
        result = env["reconciler"].run()
        assert result.can_open is False
        assert any("永续持仓不一致" in m for m in result.mismatches)

    def test_dust_positions_within_epsilon_are_consistent(self, env: dict) -> None:
        env["spot"].balances_map["BTC"] = Decimal("0.0000001")  # < QTY_EPSILON
        result = env["reconciler"].run()
        assert result.consistent is True, "灰尘仓位在容差内不算不一致"

    def test_fee_dust_below_step_is_consistent(self, env: dict) -> None:
        """手续费在现货腿以基础币扣减，余额低于 step 但高于 QTY_EPSILON（demo 实测 0.0000086）。

        该残灰不可再下最小单，对账不得判为不一致，否则服务永久卡 RECOVERY。
        """
        from cointrader.execution.rules import SymbolRules
        rule = SymbolRules(
            symbol="BTCUSDT", market="spot", status="TRADING", base_asset="BTC",
            quote_asset="USDT", tick_size=Decimal("0.01"), min_qty=Decimal("0.00001"),
            max_qty=Decimal("1000"), step_size=Decimal("0.00001"), min_notional=Decimal("5"),
        )
        env["spot"].rules["BTCUSDT"] = rule
        env["spot"].balances_map["BTC"] = Decimal("0.0000086")  # < step 0.00001
        result = env["reconciler"].run()
        assert result.consistent is True, f"sub-step 灰尘应容忍: {result.mismatches}"

    def test_real_residual_above_step_still_mismatches(self, env: dict) -> None:
        """达到一个 step 以上的残量仍是事故，必须报不一致。"""
        from cointrader.execution.rules import SymbolRules
        rule = SymbolRules(
            symbol="BTCUSDT", market="spot", status="TRADING", base_asset="BTC",
            quote_asset="USDT", tick_size=Decimal("0.01"), min_qty=Decimal("0.00001"),
            max_qty=Decimal("1000"), step_size=Decimal("0.00001"), min_notional=Decimal("5"),
        )
        env["spot"].rules["BTCUSDT"] = rule
        env["spot"].balances_map["BTC"] = Decimal("0.00002")  # 2 个 step
        result = env["reconciler"].run()
        assert result.consistent is False
        assert any("Spot 余额不一致" in m for m in result.mismatches)

class TestRepair:
    def test_local_new_order_confirmed_filled_on_exchange_is_repaired(self, env: dict) -> None:
        store: StateStore = env["store"]
        store.upsert_order(order("ct-a", state="NEW"))
        env["spot"].order_results["ct-a"] = order("ct-a", state="FILLED")
        result = env["reconciler"].run()
        assert result.consistent is True, "事件丢失导致的滞后属于修复，不算 mismatch"
        assert any("ct-a" in r for r in result.repaired)
        row = store.get_order("ct-a")
        assert row is not None and row["state"] == "FILLED"

    def test_local_new_order_absent_on_exchange_marked_canceled(self, env: dict) -> None:
        store: StateStore = env["store"]
        store.upsert_order(order("ct-b", state="NEW"))
        # 交易所查不到该订单 → 明确未接单 → 本地标 CANCELED
        result = env["reconciler"].run()
        assert result.consistent is True
        row = store.get_order("ct-b")
        assert row is not None and row["state"] == "CANCELED"

    def test_partial_fill_order_absent_on_exchange_is_mismatch(self, env: dict) -> None:
        store: StateStore = env["store"]
        # 本地认为部分成交，交易所却查无此单 → 不能擅自标 CANCELED（已有成交！），必须报不一致
        store.upsert_order(order("ct-d", state="PARTIALLY_FILLED"))
        result = env["reconciler"].run()
        assert result.consistent is False
        assert result.can_open is False
        assert any("ct-d" in m for m in result.mismatches)


class TestQueryFailure:
    def test_open_orders_query_failure_marks_untrusted(self, env: dict) -> None:
        env["perp"].order_query_error = RuntimeError("connection reset")
        result = env["reconciler"].run()
        assert result.consistent is False
        assert result.can_open is False
        assert any("PERP" in m and "对账查询失败" in m for m in result.mismatches)


class TestReconciliationRecorded:
    def test_run_records_result_to_store(self, env: dict) -> None:
        env["reconciler"].run(reason="unit-test")
        with env["store"]._lock:  # noqa: SLF001
            row = env["store"]._conn.execute(  # noqa: SLF001
                "SELECT consistent, can_open, reason FROM reconciliation_runs"
            ).fetchall()
        assert len(row) == 1
        assert row[0][0] == 1 and row[0][1] == 1
        assert row[0][2] == "unit-test", "对账原因必须入账本（审计可追溯）"


class TestIgnoreAssets:
    """demo 平台发放资产（USDC）不参与对账；本地有期望持仓时忽略失效。"""

    def test_ignored_asset_without_expected_is_skipped(self, env: dict) -> None:
        env["spot"].balances_map = {"BTC": Decimal("1"), "USDC": Decimal("5000")}
        reconciler = Reconciler(env["store"], env["spot"], env["perp"], ignore_assets=["USDC"])
        result = reconciler.run(reason="ignore-test")
        assert any("BTC" in m for m in result.mismatches), "非忽略资产差异仍须上报"
        assert not any("USDC" in m for m in result.mismatches), "平台发放 USDC 应被忽略"

    def test_ignored_asset_still_checked_when_bot_trades_it(self, env: dict) -> None:
        # 本地有 USDCUSDT 成交记录 → 该资产受管理，忽略失效
        env["store"].record_fill(spot_fill("USDCUSDT", "1"))
        env["spot"].balances_map = {"USDC": Decimal("1")}
        reconciler = Reconciler(env["store"], env["spot"], env["perp"], ignore_assets=["USDC"])
        result = reconciler.run(reason="ignore-test-2")
        assert result.consistent is True, "期望 1 与实际 1 一致"
        env["spot"].balances_map = {"USDC": Decimal("2")}
        result = reconciler.run(reason="ignore-test-3")
        assert any("USDC" in m for m in result.mismatches), "受管理资产差异不得被忽略"


def _bundle(*, spot_orders=(), perp_orders=(), positions=(), balances: dict | None = None,
            complete: bool = True) -> Any:
    from cointrader.execution.models import ExchangeSnapshotBundle

    return ExchangeSnapshotBundle(
        snapshot_id="snap-test-1",
        capture_start_ms=1,
        capture_end_ms=2,
        spot_account={"balances": [
            {"asset": a, "free": str(v), "locked": "0"} for a, v in (balances or {}).items()
        ]},
        futures_account={
            "totalWalletBalance": "10000",
            "availableBalance": "9999",
            "totalUnrealizedProfit": "0",
        },
        positions=list(positions),
        spot_open_orders=list(spot_orders),
        perp_open_orders=list(perp_orders),
        source="test",
        complete=complete,
    )


class TestBundle:
    """T3（AC-06/07）：live 路径消费同一 capture bundle，不重复拉 API。"""

    def test_bundle_consumed_without_adapter_api_calls(self, env: dict) -> None:
        spot: FakeAdapter = env["spot"]
        perp: FakeAdapter = env["perp"]
        # 若对账器重复拉 API（open_orders/balances/positions），fake 会抛错
        spot.order_query_error = RuntimeError("open_orders must not be called with bundle")
        perp.order_query_error = RuntimeError("open_orders must not be called with bundle")

        def boom_balances() -> dict[str, Decimal]:
            raise RuntimeError("balances must not be called with bundle")

        def boom_positions(symbol: str | None = None) -> list[dict[str, Any]]:
            raise RuntimeError("positions must not be called with bundle")

        spot.balances = boom_balances  # type: ignore[method-assign]
        perp.balances = boom_balances  # type: ignore[method-assign]
        spot.positions = boom_positions  # type: ignore[method-assign]
        perp.positions = boom_positions  # type: ignore[method-assign]

        result = env["reconciler"].run(
            reason="bundle-test",
            snapshot=_bundle(),
        )
        assert result.consistent is True
        assert result.details.get("snapshot_id") == "snap-test-1"

    def test_incomplete_bundle_blocks_open(self, env: dict) -> None:
        result = env["reconciler"].run(
            reason="incomplete", snapshot=_bundle(complete=False)
        )
        assert result.consistent is False
        assert result.can_open is False
        assert any("bundle 不完整" in m for m in result.mismatches)

    def test_bundle_unknown_order_blocks_open(self, env: dict) -> None:
        result = env["reconciler"].run(
            snapshot=_bundle(spot_orders=[
                {"clientOrderId": "ct-ghost", "symbol": "BTCUSDT", "origQty": "1"}
            ])
        )
        assert result.can_open is False
        assert any("ct-ghost" in m for m in result.mismatches)

    def test_bundle_position_mismatch_blocks_open(self, env: dict) -> None:
        env["store"].record_fill(spot_fill("BTCUSDT", "1"))
        result = env["reconciler"].run(
            snapshot=_bundle(balances={"BTC": Decimal("0.5")})
        )
        assert result.can_open is False
        assert any("Spot 余额不一致" in m for m in result.mismatches)
