"""开仓去重测试（开发文档 §7.3：统一入口 + 三层去重 + 重启恢复）。

覆盖：
- store 层去重查询（非终态 intent / order / pair、position_opened_ms）
- service 层：已有实际持仓不重复开仓；本轮已提交不重复
- 重启：同一账本 + 交易所仍有持仓 → 新进程不得再开
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

from cointrader.execution.guard import Market, OrderSide, OrderType
from cointrader.execution.models import Fill, Order, OrderIntent, PairExecution
from cointrader.execution.store import StateStore
from live_helpers import NOW, FakeServiceAdapter, LiveFakeData, make_live_rates, make_service

SYMBOL = "BTCUSDT"


def _intent(symbol: str = SYMBOL, pair_id: str | None = None) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=OrderSide.BUY,
        market=Market.SPOT,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.01"),
        price=Decimal("100"),
        pair_execution_id=pair_id,
    )


def _order(symbol: str, state: str, client_id: str = "co-1") -> Order:
    return Order(
        client_order_id=client_id,
        symbol=symbol,
        market=Market.SPOT,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.01"),
        state=state,
        updated_ms=int(time.time() * 1000),
    )


def _pair(pair_id: str, symbol: str, status: str, kind: str = "open") -> PairExecution:
    return PairExecution(
        pair_execution_id=pair_id,
        symbol=symbol,
        target_notional=Decimal("100"),
        status=status,
        kind=kind,
        strategy_version="t",
        created_ms=int(time.time() * 1000),
        updated_ms=int(time.time() * 1000),
    )


class TestStoreDedup:
    def test_active_pair_blocks(self, tmp_path):
        store = StateStore(tmp_path / "t.sqlite3")
        assert store.active_pair_for_symbol(SYMBOL) is None
        store.upsert_pair(_pair("p1", SYMBOL, "OPENING"))
        assert store.active_pair_for_symbol(SYMBOL) == "p1"
        store.upsert_pair(_pair("p1", SYMBOL, "COMPLETE"))
        assert store.active_pair_for_symbol(SYMBOL) is None

    def test_has_open_intent(self, tmp_path):
        store = StateStore(tmp_path / "t.sqlite3")
        # 无关联 pair 的开仓 intent → 存在
        assert store.has_open_intent(SYMBOL) is False
        assert store.record_intent(_intent(pair_id=None)) is True
        assert store.has_open_intent(SYMBOL) is True
        # 关联 pair 终态 → 不再算未终态 intent
        store.record_intent(_intent("ETHUSDT", pair_id="p-term"))
        store.upsert_pair(_pair("p-term", "ETHUSDT", "COMPLETE"))
        assert store.has_open_intent("ETHUSDT") is False
        # 平仓 intent 不算
        store.record_intent(OrderIntent(
            symbol="SOLUSDT", side=OrderSide.SELL, market=Market.SPOT,
            order_type=OrderType.MARKET, quantity=Decimal("0.01"),
            price=Decimal("100"), is_closing=True,
        ))
        assert store.has_open_intent("SOLUSDT") is False

    def test_has_open_order(self, tmp_path):
        store = StateStore(tmp_path / "t.sqlite3")
        assert store.has_open_order(SYMBOL) is False
        store.upsert_order(_order(SYMBOL, "FILLED"))
        assert store.has_open_order(SYMBOL) is False
        store.upsert_order(_order(SYMBOL, "NEW", "co-2"))
        assert store.has_open_order(SYMBOL) is True

    def test_position_opened_ms(self, tmp_path):
        store = StateStore(tmp_path / "t.sqlite3")
        assert store.position_opened_ms(SYMBOL) is None
        store.upsert_pair(_pair("p1", SYMBOL, "COMPLETE", kind="open"))
        opened = store.position_opened_ms(SYMBOL)
        assert opened is not None and opened > 0


def _env(tmp_path, **kw: Any) -> dict[str, Any]:
    data = LiveFakeData({SYMBOL: make_live_rates(20, "0.0005")})
    return make_service(tmp_path, data, **kw)


class TestServiceDedup:
    def test_no_duplicate_open_when_position_exists(self, tmp_path):
        """真实持仓存在 → 即使策略信号通过也不得再次开仓。"""
        env = _env(tmp_path)
        spot: FakeServiceAdapter = env["spot"]
        futures: FakeServiceAdapter = env["futures"]
        svc = env["svc"]
        svc.run_id = "run-dedup"
        spot.balances_map["BTC"] = Decimal("0.01")
        futures.position_amt = Decimal("-0.01")

        r1 = svc.run_once()
        assert r1["state"] == "RUNNING"
        assert env["executor"].open_calls == [], "已有实际持仓时不得开仓"
        # 持仓决策应为退出评估（HOLD），而不是 OPEN
        decisions = env["store"].signal_decisions(run_id="run-dedup")
        kinds = {d["decision_kind"] for d in decisions}
        assert "OPEN" not in kinds

    def test_open_then_second_round_no_reopen(self, tmp_path):
        """开仓成功 → 下轮（对账确认真实持仓后）重复信号不产生第二笔开仓。"""
        clock = {"t": NOW}
        env = _env(tmp_path, now_fn=lambda: clock["t"])
        spot: FakeServiceAdapter = env["spot"]
        futures: FakeServiceAdapter = env["futures"]
        svc = env["svc"]
        svc.run_id = "run-dedup2"

        r1 = svc.run_once()
        assert r1["state"] == "RUNNING"
        assert len(env["executor"].open_calls) == 1, f"第一轮应开仓一次: {r1}"
        # 模拟交易所确认持仓
        spot.balances_map["BTC"] = Decimal("0.01")
        futures.position_amt = Decimal("-0.01")
        # 推进时钟 > 对账周期（30s），让第二轮重新对账并以真实持仓为准
        clock["t"] = NOW + 31

        r2 = svc.run_once()
        assert r2["state"] == "RUNNING"
        assert len(env["executor"].open_calls) == 1, f"第二轮不得重复开仓: {r2}"
        assert len(env["executor"].close_calls) == 0
        decisions = env["store"].signal_decisions(run_id="run-dedup2")
        second_round = [d for d in decisions if d["ts_ms"] >= int((NOW + 31) * 1000)]
        assert {d["decision_kind"] for d in second_round} == {"HOLD"}

    def test_restart_no_reopen_with_existing_position(self, tmp_path):
        """重启（新 service + 同一账本）且交易所仍有持仓 → 不得再开仓。"""
        data = LiveFakeData({SYMBOL: make_live_rates(20, "0.0005")})
        env1 = make_service(tmp_path, data)
        svc1 = env1["svc"]
        svc1.run_id = "run-restart"
        svc1.run_once()
        assert len(env1["executor"].open_calls) == 1

        # 交易所持仓仍在（平仓没发生）
        env1["spot"].balances_map["BTC"] = Decimal("0.01")
        env1["futures"].position_amt = Decimal("-0.01")
        env2 = make_service(
            tmp_path / "restart", data,
            config=env1["config"],
            spot=env1["spot"], futures=env1["futures"], store=env1["store"],
        )
        svc2 = env2["svc"]
        svc2.run_id = "run-restart2"
        r = svc2.run_once()
        assert r["state"] == "RUNNING"
        assert env2["executor"].open_calls == [], "重启后持仓仍在，不得再开仓"
        # 账本中该 symbol 的开仓 pair 只有一条
        pairs = env2["store"].pair_executions(symbol=SYMBOL, limit=100)
        opens = [p for p in pairs if p.get("kind") == "open"]
        assert len(opens) == 1


class TestFillIdempotency:
    def test_duplicate_fill_not_duplicated(self, tmp_path):
        """重复 fill 事件（同 exchange_trade_id）在账本层不重复 —— PnL 幂等基础。"""
        store = StateStore(tmp_path / "t.sqlite3")

        def fill(i: str, trade_id: str = "ex-trade-1") -> Fill:
            return Fill(
                fill_id=i,
                client_order_id="co-1",
                symbol=SYMBOL,
                market=Market.SPOT,
                side=OrderSide.BUY,
                quantity=Decimal("0.01"),
                price=Decimal("100"),
                fee_asset="USDT",
                fee_amount=Decimal("0.01"),
                ts_ms=int(time.time() * 1000),
                exchange_trade_id=trade_id,
            )

        assert store.record_fill(fill("f1")) is True
        assert store.record_fill(fill("f1-dup")) is False, "同 (market, exchange_trade_id) 不得重复入账"
