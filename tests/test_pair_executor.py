"""双腿执行器单元测试（开发设计文档 §6 / §11.1）。

核心不变量：

- 永不盲目重发：每腿 place 至多一次（补偿订单除外，且补偿只减仓）。
- Spot 数量 = Futures 实际 executedQty（归一化后），不是原始目标量。
- UNKNOWN_SUBMISSION 只按 clientOrderId 查询一次。
- 另一腿明确失败 → 平掉已成交腿，不得留下裸腿。
- 平仓 = reduce-only，不能反向开仓。
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

import pytest

from cointrader.config import RiskConfig
from cointrader.errors import OrderRejected, UnknownSubmission
from cointrader.execution.guard import Market, OrderSide, OrderType
from cointrader.execution.models import Order, OrderRequest
from cointrader.execution.pair_executor import PairExecutor
from cointrader.execution.risk import RiskManager, RiskState
from cointrader.execution.risk_gate import HaltState, RiskGate
from cointrader.execution.rules import SymbolRules
from cointrader.execution.store import StateStore

PRICE = Decimal("100")


def make_rule(market: str) -> SymbolRules:
    return SymbolRules(
        symbol="BTCUSDT",
        market=market,
        status="TRADING",
        base_asset="BTC",
        quote_asset="USDT",
        tick_size=Decimal("0.1"),
        min_qty=Decimal("0.001"),
        max_qty=Decimal("100"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("10"),
    )


class FakeExchange:
    """按脚本出牌的假交易所（Spot/Futures 通用）。"""

    def __init__(self, market: str) -> None:
        self.market_name = market
        self._rule = make_rule(market)
        self._counter = 0
        self.place_results: list[Order | Exception] = []
        self.query_results: dict[str, Order | None] = {}
        self.place_calls: list[OrderRequest] = []
        self.query_calls: list[str] = []
        self.cancel_calls: list[str] = []
        self.reduce_calls: list[tuple[str, Decimal, str]] = []
        self.balances_map: dict[str, Decimal] = {}
        self.position_amt: Decimal = Decimal("0")
        self.user_trade_rows: list[dict[str, Any]] = []
        self.my_trade_rows: list[dict[str, Any]] = []

    # -- 被 pair_executor 调用的接口 --------------------------------------

    def rule(self, symbol: str) -> SymbolRules:  # noqa: ARG002
        return self._rule

    def place_order(self, req: OrderRequest) -> Order:
        self.place_calls.append(req)
        if self.place_results:
            result = self.place_results.pop(0)
            if callable(result):
                result = result(req)
        else:
            result = self._default_fill(req)
        if isinstance(result, Exception):
            raise result
        self._apply_fill(req, result)
        return result

    def query_by_client_order_id(self, symbol: str, client_order_id: str) -> Order | None:  # noqa: ARG002
        self.query_calls.append(client_order_id)
        return self.query_results.get(client_order_id)

    def cancel_order(self, symbol: str, *, client_order_id: str | None = None, order_id: str | None = None) -> None:  # noqa: ARG002
        self.cancel_calls.append(client_order_id or "")

    def reduce_only_close(self, symbol: str, quantity: Decimal, *, client_order_id: str) -> Order:
        self.reduce_calls.append((symbol, quantity, client_order_id))
        req = OrderRequest(
            client_order_id=client_order_id,
            symbol=symbol,
            side=OrderSide.BUY,  # 平空 = 买入
            market=Market.PERP,
            order_type=OrderType.MARKET,
            quantity=quantity,
            reduce_only=True,
        )
        self.place_calls.append(req)
        order = self._default_fill(req)
        self._apply_fill(req, order)
        return order

    def position_qty(self, symbol: str) -> Decimal:  # noqa: ARG002
        return self.position_amt

    def balances(self) -> dict[str, Decimal]:
        return dict(self.balances_map)

    def user_trades(self, symbol: str, **kw: Any) -> list[dict[str, Any]]:  # noqa: ARG002
        return self.user_trade_rows

    def my_trades(self, symbol: str, **kw: Any) -> list[dict[str, Any]]:  # noqa: ARG002
        return self.my_trade_rows

    # -- 内部 ---------------------------------------------------------------

    def _default_fill(self, req: OrderRequest) -> Order:
        self._counter += 1
        return self._make_order(req, req.quantity, PRICE)

    def _make_order(self, req: OrderRequest, executed: Decimal, avg: Decimal) -> Order:
        self._counter += 1
        return Order(
            client_order_id=req.client_order_id,
            exchange_order_id=f"ex-{self.market_name}-{self._counter}",
            symbol=req.symbol,
            market=req.market,
            side=req.side,
            order_type=req.order_type,
            quantity=req.quantity,
            state="FILLED",
            executed_qty=executed,
            avg_price=avg,
            price=req.price,
            reduce_only=req.reduce_only,
            updated_ms=int(time.time() * 1000),
        )

    def _apply_fill(self, req: OrderRequest, order: Order) -> None:
        base = req.symbol.replace("USDT", "")
        if req.market is Market.SPOT:
            delta = order.executed_qty
            if req.side is OrderSide.SELL:
                delta = -delta
            self.balances_map[base] = self.balances_map.get(base, Decimal("0")) + delta
        else:
            delta = order.executed_qty
            if req.side is OrderSide.SELL:
                delta = -delta
            self.position_amt += delta


@pytest.fixture
def env(tmp_path) -> dict[str, Any]:
    spot = FakeExchange("spot")
    perp = FakeExchange("perp")
    store = StateStore(tmp_path / "trading.sqlite3")
    gate = RiskGate(_make_risk_manager())
    alerts: list[tuple[str, str]] = []
    executor = PairExecutor(
        spot=spot,
        futures=perp,
        store=store,
        gate=gate,
        on_alert=lambda kind, msg: alerts.append((kind, msg)),
        order_ack_timeout_seconds=3.0,
        poll_interval_seconds=0.01,
        sleep_fn=lambda s: None,
    )
    return {"spot": spot, "perp": perp, "store": store, "gate": gate, "alerts": alerts, "executor": executor}


def _make_risk_manager() -> RiskManager:
    return RiskManager(RiskConfig())


@pytest.fixture
def risk_state() -> RiskState:
    return RiskState(total_capital=10_000.0)


def open_kwargs(quote_ts_ms: int) -> dict[str, Any]:
    return dict(
        spot_price=PRICE,
        perp_price=PRICE,
        quote_ts_ms=quote_ts_ms,
    )


class TestOpenPair:
    def test_full_fill_completes_hedge(self, env: dict, risk_state: RiskState) -> None:
        executor: PairExecutor = env["executor"]
        pair = executor.open_pair("BTCUSDT", Decimal("100"), state=risk_state, **open_kwargs(int(time.time() * 1000)))
        assert pair.status == "COMPLETE", f"全成交应完成对冲，实际 {pair.status}: {pair.error}"
        perp_req = env["perp"].place_calls[0]
        spot_req = env["spot"].place_calls[0]
        assert perp_req.side is OrderSide.SELL, "永续腿必须先开空"
        assert spot_req.side is OrderSide.BUY, "现货腿必须买入"
        assert perp_req.quantity == Decimal("1")
        # Spot 数量 = 永续实际成交量
        assert spot_req.quantity == perp_req.quantity == Decimal("1")

    def test_spot_qty_follows_partial_perp_fill(self, env: dict, risk_state: RiskState) -> None:
        """部分成交：Spot 单量必须按 Futures 实际 executedQty，而不是原始目标量。"""
        executor: PairExecutor = env["executor"]
        perp: FakeExchange = env["perp"]
        # 永续目标 1.0（100 USDT / 100 价），实际只成交 0.6
        perp.place_results.append(
            Order(
                client_order_id="set-later",  # 占位，不影响数量断言
                exchange_order_id="ex-p",
                symbol="BTCUSDT",
                market=Market.PERP,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
                state="FILLED",
                executed_qty=Decimal("0.6"),
                avg_price=PRICE,
                price=PRICE,
            )
        )
        pair = executor.open_pair("BTCUSDT", Decimal("100"), state=risk_state, **open_kwargs(int(time.time() * 1000)))
        assert pair.status == "COMPLETE", f"部分成交应对实际成交量完成对冲，实际 {pair.status}: {pair.error}"
        spot_req = env["spot"].place_calls[0]
        assert spot_req.quantity == Decimal("0.6"), "现货数量必须等于永续实际成交量（归一化后），而不是原始目标 1.0"
        # 每腿只下一单：无补偿
        assert len(perp.place_calls) == 1
        assert len(env["spot"].place_calls) == 1
        assert perp.reduce_calls == []

    def test_perp_partial_leaves_dust_residue_flattened(self, env: dict, risk_state: RiskState) -> None:
        """Spot 归一化后与永续成交量差超出容差 → 残量必须 reduce-only 平掉。"""
        executor: PairExecutor = env["executor"]
        perp: FakeExchange = env["perp"]
        # 永续成交 0.6，但 Spot 规则 step 大（0.25）→ 只能买 0.5，残量 0.1 超容差
        env["spot"]._rule = SymbolRules(  # noqa: SLF001
            symbol="BTCUSDT", market="spot", status="TRADING", base_asset="BTC", quote_asset="USDT",
            tick_size=Decimal("0.1"), min_qty=Decimal("0.25"), max_qty=Decimal("100"),
            step_size=Decimal("0.25"), min_notional=Decimal("10"),
        )
        perp.place_results.append(
            Order(
                client_order_id="set-later", exchange_order_id="ex-p2", symbol="BTCUSDT",
                market=Market.PERP, side=OrderSide.SELL, order_type=OrderType.MARKET,
                quantity=Decimal("1"), state="FILLED", executed_qty=Decimal("0.6"),
                avg_price=PRICE, price=PRICE,
            )
        )
        pair = executor.open_pair("BTCUSDT", Decimal("100"), state=risk_state, **open_kwargs(int(time.time() * 1000)))
        assert pair.status in ("COMPLETE", "COMPENSATED"), f"{pair.status}: {pair.error}"
        total_reduced = sum((q for _, q, _ in perp.reduce_calls), Decimal("0"))
        assert len(perp.reduce_calls) == 1, f"残量只应 reduce 一次，实际: {perp.reduce_calls}"
        assert total_reduced == Decimal("0.1"), f"reduce 总量应为 0.1，实际 {total_reduced}"
        assert perp.position_amt == Decimal("-0.5"), "reduce 后永续持仓必须与现货余额对齐，不得超平"

    def test_unknown_submission_queries_once_never_resubmits(self, env: dict, risk_state: RiskState) -> None:
        """提交结果未知：只按 clientOrderId 查询一次；确认未成交 → 补偿平永续，绝不重下。"""
        executor: PairExecutor = env["executor"]
        perp: FakeExchange = env["perp"]
        perp.place_results.append(UnknownSubmission("timeout", client_order_id="?", market="perp"))
        pair = executor.open_pair("BTCUSDT", Decimal("100"), state=risk_state, **open_kwargs(int(time.time() * 1000)))
        # 查询确认未成交（query_results 为空 → None）→ 永续腿按 0 成交处理
        assert pair.status in ("COMPENSATED", "HALTED", "FAILED"), pair.status
        perp_cids = [r.client_order_id for r in perp.place_calls if r.side is OrderSide.SELL]
        assert len(perp_cids) == 1, "未知结果禁止重发同一永续订单"
        # 查询至多一次
        assert len(perp.query_calls) <= 1, "UNKNOWN 恢复只允许查询一次"
        kinds = [k for k, _ in env["alerts"]]
        assert "UNKNOWN_SUBMISSION" in kinds, "未知提交必须告警"

    def test_unknown_submission_recovers_confirmed_fill(self, env: dict, risk_state: RiskState) -> None:
        """提交超时但查询确认已成交：继续状态机，不重下。"""
        executor: PairExecutor = env["executor"]
        perp: FakeExchange = env["perp"]

        confirmed: dict[str, Order] = {}

        def query_fn(symbol: str, cid: str) -> Order:  # noqa: ARG001
            return confirmed[cid]

        perp.place_results.append(UnknownSubmission("timeout", client_order_id="?", market="perp"))
        # 用查询结果钩子：在 place 抛出前先不知道 cid —— 通过包装 query 动态生成
        original_query = perp.query_by_client_order_id

        def dynamic_query(symbol: str, cid: str) -> Order | None:
            if cid in perp.query_results:
                original_query(symbol, cid)
                return perp.query_results[cid]
            order = Order(
                client_order_id=cid,
                exchange_order_id="ex-recovered",
                symbol=symbol,
                market=Market.PERP,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
                state="FILLED",
                executed_qty=Decimal("1"),
                avg_price=PRICE,
                price=PRICE,
            )
            original_query(symbol, cid)
            return order

        perp.query_by_client_order_id = dynamic_query  # type: ignore[method-assign]
        pair = executor.open_pair("BTCUSDT", Decimal("100"), state=risk_state, **open_kwargs(int(time.time() * 1000)))
        assert pair.status == "COMPLETE", f"查询确认成交后应继续完成，实际 {pair.status}: {pair.error}"
        perp_cids = [r.client_order_id for r in perp.place_calls if r.side is OrderSide.SELL]
        assert len(perp_cids) == 1, "确认已成交后禁止重下"
        assert len(env["spot"].place_calls) == 1

    def test_spot_rejection_flattens_perp_no_naked_leg(self, env: dict, risk_state: RiskState) -> None:
        """现货腿被明确拒单 → 已成交的永续腿必须平掉，不得留裸腿。"""
        executor: PairExecutor = env["executor"]
        spot: FakeExchange = env["spot"]
        perp: FakeExchange = env["perp"]
        spot.place_results.append(OrderRejected("insufficient balance", code=-2019, status=400))
        pair = executor.open_pair("BTCUSDT", Decimal("100"), state=risk_state, **open_kwargs(int(time.time() * 1000)))
        assert pair.status in ("COMPENSATED", "HALTED"), f"拒单后必须补偿，实际 {pair.status}: {pair.error}"
        assert len(perp.reduce_calls) == 1, "永续腿必须被 reduce-only 平掉"
        assert perp.position_amt == Decimal("0"), "补偿后永续持仓必须归零"
        kinds = [k for k, _ in env["alerts"]]
        assert "ORDER_REJECTED" in kinds

    def test_stale_market_data_rejected_locally(self, env: dict, risk_state: RiskState) -> None:
        executor: PairExecutor = env["executor"]
        old_ts = int(time.time() * 1000) - 60_000
        pair = executor.open_pair("BTCUSDT", Decimal("100"), state=risk_state, **open_kwargs(old_ts))
        assert pair.status == "FAILED"
        assert "过期" in pair.error
        assert env["perp"].place_calls == [], "数据过期必须在本地拒绝，不得发单"

    def test_duplicate_active_pair_blocked(self, env: dict, risk_state: RiskState) -> None:
        """已有未完结 pair 的 symbol 禁止重复开仓。"""
        executor: PairExecutor = env["executor"]
        perp: FakeExchange = env["perp"]
        # 永续腿永不终态：place 返回 NEW，查询也返回 NEW → 超时撤单
        def always_new(req: OrderRequest) -> Order:
            return Order(
                client_order_id=req.client_order_id,
                exchange_order_id="ex-new",
                symbol=req.symbol,
                market=Market.PERP,
                side=req.side,
                order_type=req.order_type,
                quantity=req.quantity,
                state="NEW",
                executed_qty=Decimal("0"),
            )

        perp.place_results.append(always_new)
        # 用会推进的假时钟让 _wait_terminal 超时（3 秒 ack 超时）
        clock = {"t": 0.0}
        executor._now = lambda: clock["t"]  # noqa: SLF001
        executor._sleep = lambda s: clock.update(t=clock["t"] + s)  # noqa: SLF001
        pair1 = executor.open_pair("BTCUSDT", Decimal("100"), state=risk_state, **open_kwargs(int(time.time() * 1000)))
        assert pair1.status in ("FAILED", "COMPENSATED", "HALTED"), pair1.status
        # 恢复闸门，单独验证 active_pair 重复开仓防护
        env["gate"].recover(reconciliation_ok=True, preflight_ok=True)
        # 第二笔：若第一笔留下未终态 pair 必须被拒；若第一笔已终态（撤单/补偿）则允许
        # 这里验证 active_pair 逻辑本身
        from cointrader.execution.models import PairExecution

        active = PairExecution(
            pair_execution_id="pair-active-x",
            symbol="BTCUSDT",
            target_notional=Decimal("100"),
            strategy_version="v1",
            created_ms=1,
            updated_ms=1,
        )
        env["store"].upsert_pair(active)
        pair2 = executor.open_pair("BTCUSDT", Decimal("100"), state=risk_state, **open_kwargs(int(time.time() * 1000)))
        assert pair2.status == "FAILED"
        assert "未完结" in pair2.error

    def test_slippage_exceeds_halts_gate(self, env: dict, risk_state: RiskState) -> None:
        executor: PairExecutor = env["executor"]
        perp: FakeExchange = env["perp"]
        # 永续成交价 110 vs 参考价 100 → 滑点 10% ≫ 0.1%
        perp.place_results.append(
            Order(
                client_order_id="x",
                exchange_order_id="ex-slip",
                symbol="BTCUSDT",
                market=Market.PERP,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
                state="FILLED",
                executed_qty=Decimal("1"),
                avg_price=Decimal("110"),
                price=PRICE,
            )
        )
        executor.open_pair("BTCUSDT", Decimal("100"), state=risk_state, **open_kwargs(int(time.time() * 1000)))
        kinds = [k for k, _ in env["alerts"]]
        assert "SLIPPAGE_EXCEEDED" in kinds
        assert env["gate"].state is HaltState.HALT_NEW_RISK, "滑点超限必须停机"


class TestClosePair:
    def test_close_is_reduce_only_and_zeroes_positions(self, env: dict, risk_state: RiskState) -> None:
        executor: PairExecutor = env["executor"]
        spot: FakeExchange = env["spot"]
        perp: FakeExchange = env["perp"]
        spot.balances_map["BTC"] = Decimal("1")
        perp.position_amt = Decimal("-1")
        pair = executor.close_pair("BTCUSDT", reason="test")
        assert pair.status == "COMPLETE", f"平仓应完成，实际 {pair.status}: {pair.error}"
        # 现货腿：卖出全部余额
        spot_req = spot.place_calls[0]
        assert spot_req.side is OrderSide.SELL
        assert spot_req.quantity == Decimal("1")
        # 永续腿：买入 reduce-only 平空
        assert len(perp.reduce_calls) == 1
        symbol, qty, _cid = perp.reduce_calls[0]
        assert qty == Decimal("1")
        perp_close_req = perp.place_calls[-1]
        assert perp_close_req.reduce_only is True, "永续平仓必须 reduce-only"
        assert perp_close_req.side is OrderSide.BUY, "平空 = 买入"
        assert spot.balances_map["BTC"] == Decimal("0")
        assert perp.position_amt == Decimal("0")

    def test_close_no_position_is_noop(self, env: dict, risk_state: RiskState) -> None:
        executor: PairExecutor = env["executor"]
        pair = executor.close_pair("BTCUSDT")
        assert pair.status == "COMPLETE"
        assert env["spot"].place_calls == []
        assert env["perp"].place_calls == []


class TestHaltSemantics:
    def test_halt_blocks_open_but_allows_close(self, env: dict, risk_state: RiskState) -> None:
        """HALT_NEW_RISK：开仓拒绝，reduce-only 平仓放行（文档 §7.3）。"""
        executor: PairExecutor = env["executor"]
        spot: FakeExchange = env["spot"]
        perp: FakeExchange = env["perp"]
        env["gate"].halt("test incident")
        # 开仓必须被拒绝且不产生任何订单
        pair = executor.open_pair("BTCUSDT", Decimal("100"), state=risk_state, **open_kwargs(int(time.time() * 1000)))
        assert pair.status == "FAILED"
        assert env["perp"].place_calls == [] and env["spot"].place_calls == []
        # 平仓（减仓）必须放行
        spot.balances_map["BTC"] = Decimal("0.5")
        perp.position_amt = Decimal("-0.5")
        close = executor.close_pair("BTCUSDT", reason="incident flatten")
        assert close.status == "COMPLETE"
        assert len(perp.reduce_calls) == 1

    def test_recover_requires_reconciliation_and_preflight(self, env: dict) -> None:
        gate: RiskGate = env["gate"]
        gate.halt("boom")
        with pytest.raises(Exception, match="恢复条件不满足|停机文件"):
            gate.recover(reconciliation_ok=False, preflight_ok=True)
        gate.recover(reconciliation_ok=True, preflight_ok=True)
        assert gate.state is HaltState.NORMAL

    def test_kill_file_forces_halt_and_blocks_recover(self, env: dict) -> None:
        gate: RiskGate = env["gate"]
        gate2 = RiskGate(_make_risk_manager(), kill_check=lambda: True)
        assert gate2.state is HaltState.HALT_NEW_RISK, "停机文件存在 = 强制 HALT_NEW_RISK"
        gate2._state = HaltState.NORMAL  # noqa: SLF001
        with pytest.raises(Exception, match="停机文件仍存在"):
            gate2.recover(reconciliation_ok=True, preflight_ok=True)
        assert gate.state is not None
