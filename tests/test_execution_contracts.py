"""T3：OrderPlanner + PairExecutor 计划入口契约测试（AC-06）。

覆盖：开仓/平仓计划生成、数量归一化、零量/非法价拒绝、平仓 reduce-only
不变量、计划过期拒绝执行、过期计划不触碰 adapter。
"""

from __future__ import annotations

import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from cointrader.config import RiskConfig
from cointrader.domain.execution import ExecutionPlan, PlanOrder, PlanOrderType, PlanSide
from cointrader.domain.market import MarketKind
from cointrader.domain.portfolio import IntentAction, PortfolioIntent
from cointrader.domain.risk import ApprovedIntent
from cointrader.execution.pair_executor import PairPrecheckFailed
from cointrader.execution.planner import OrderPlanner, PlanError
from cointrader.execution.risk import RiskManager, RiskState
from cointrader.execution.risk_gate import RiskGate
from cointrader.execution.store import StateStore
from test_pair_executor import FakeExchange, make_rule

NOW = 1_800_002_000_000
PRICE = Decimal("100")


def make_intent(action: IntentAction = IntentAction.OPEN) -> PortfolioIntent:
    spot = "0" if action is IntentAction.CLOSE else "100"
    perp = "0" if action is IntentAction.CLOSE else "100"
    return PortfolioIntent(
        intent_id="it-plan",
        action=action,
        symbol="BTCUSDT",
        target_spot_notional=Decimal(spot),
        target_perp_notional=Decimal(perp),
        reason="t3-plan",
        snapshot_id="snap-1",
        decision_cutoff_ms=NOW - 1000,
        created_at_ms=NOW,
        reduces_risk=action is IntentAction.CLOSE,
    )


def make_approved(intent: PortfolioIntent) -> ApprovedIntent:
    spot = Decimal("0") if intent.action is IntentAction.CLOSE else Decimal("100")
    perp = Decimal("0") if intent.action is IntentAction.CLOSE else Decimal("100")
    return ApprovedIntent(
        intent=intent,
        decision_id="rd-1",
        approved_spot_notional=spot,
        approved_perp_notional=perp,
        decided_at_ms=NOW,
        valid_until_ms=NOW + 30_000,
        is_closing=intent.is_closing,
    )


@pytest.fixture
def planner() -> OrderPlanner:
    return OrderPlanner("fc3")


# -- 开仓计划 ----------------------------------------------------------------


def test_open_plan_two_legs_normalized(planner):
    plan = planner.plan(
        make_approved(make_intent()),
        spot_price=PRICE,
        perp_price=PRICE,
        spot_rules=make_rule("spot"),
        perp_rules=make_rule("perp"),
        now_ms=NOW,
        plan_id="plan-1",
    )
    assert len(plan.orders) == 2
    perp, spot = plan.orders
    assert perp.market is MarketKind.FUTURES
    assert perp.side is PlanSide.SELL
    assert perp.quantity == Decimal("1.000")  # 100/100，step 0.001
    assert spot.market is MarketKind.SPOT
    assert spot.side is PlanSide.BUY
    assert spot.quantity == Decimal("1.000")
    assert not perp.reduce_only and not spot.reduce_only
    assert plan.expires_at_ms <= plan.approved_intent.valid_until_ms
    assert perp.client_order_id.startswith("ct-fc3")
    assert perp.client_order_id != spot.client_order_id


def test_open_plan_rejects_zero_qty_leg(planner):
    approved = make_approved(make_intent())
    half = ApprovedIntent(
        intent=approved.intent,
        decision_id="rd-3",
        approved_spot_notional=Decimal("0"),
        approved_perp_notional=Decimal("100"),
        decided_at_ms=NOW,
        valid_until_ms=NOW + 30_000,
    )
    with pytest.raises(PlanError):
        planner.plan(
            half,
            spot_price=PRICE,
            perp_price=PRICE,
            spot_rules=make_rule("spot"),
            perp_rules=make_rule("perp"),
            now_ms=NOW,
            plan_id="plan-zero",
        )


def test_open_plan_min_notional_below_rejects(planner):
    # 名义额 100 → 数量 1.0 正常；构造极小名义额触发 min_qty 拒绝
    approved = make_approved(make_intent())
    tiny = ApprovedIntent(
        intent=approved.intent,
        decision_id="rd-2",
        approved_spot_notional=Decimal("0.0001"),
        approved_perp_notional=Decimal("0.0001"),
        decided_at_ms=NOW,
        valid_until_ms=NOW + 30_000,
    )
    with pytest.raises(PlanError):
        planner.plan(
            tiny,
            spot_price=PRICE,
            perp_price=PRICE,
            spot_rules=make_rule("spot"),
            perp_rules=make_rule("perp"),
            now_ms=NOW,
            plan_id="plan-tiny",
        )


def test_open_plan_rejects_bad_price(planner):
    with pytest.raises(PlanError):
        planner.plan(
            make_approved(make_intent()),
            spot_price=Decimal("0"),
            perp_price=PRICE,
            spot_rules=make_rule("spot"),
            perp_rules=make_rule("perp"),
            now_ms=NOW,
        )


def test_expired_approved_intent_rejected_before_planning(planner):
    approved = make_approved(make_intent())
    expired = ApprovedIntent(
        intent=approved.intent,
        decision_id=approved.decision_id,
        approved_spot_notional=approved.approved_spot_notional,
        approved_perp_notional=approved.approved_perp_notional,
        decided_at_ms=NOW - 60_000,
        valid_until_ms=NOW - 30_000,  # 已过期
    )
    with pytest.raises(PlanError, match="过期"):
        planner.plan(
            expired,
            spot_price=PRICE,
            perp_price=PRICE,
            spot_rules=make_rule("spot"),
            perp_rules=make_rule("perp"),
            now_ms=NOW,
            plan_id="plan-expired-approved",
        )


# -- 平仓计划 ----------------------------------------------------------------


def test_close_plan_all_reduce_only(planner):
    plan = planner.plan(
        make_approved(make_intent(IntentAction.CLOSE)),
        spot_price=PRICE,
        perp_price=PRICE,
        spot_rules=make_rule("spot"),
        perp_rules=make_rule("perp"),
        close_spot_qty=Decimal("1.000"),
        close_perp_qty=Decimal("1.000"),
        now_ms=NOW,
        plan_id="plan-close",
    )
    assert all(o.reduce_only for o in plan.orders)
    perp, spot = plan.orders
    assert perp.side is PlanSide.BUY
    assert spot.side is PlanSide.SELL


def test_close_plan_rounds_qty_up_to_step(planner):
    plan = planner.plan(
        make_approved(make_intent(IntentAction.CLOSE)),
        spot_price=PRICE,
        perp_price=PRICE,
        spot_rules=make_rule("spot"),
        perp_rules=make_rule("perp"),
        close_spot_qty=Decimal("0.9994"),
        close_perp_qty=Decimal("0.9994"),
        now_ms=NOW,
        plan_id="plan-close2",
    )
    # 平仓向上对齐步长：0.9994 → 1.000（避免残量）
    assert all(o.quantity == Decimal("1.000") for o in plan.orders)


def test_close_plan_requires_actual_qty(planner):
    with pytest.raises(PlanError):
        planner.plan(
            make_approved(make_intent(IntentAction.CLOSE)),
            spot_price=PRICE,
            perp_price=PRICE,
            spot_rules=make_rule("spot"),
            perp_rules=make_rule("perp"),
            now_ms=NOW,
        )


# -- 计划 → 执行入口 ----------------------------------------------------------


def test_expired_plan_rejected_before_adapter_touch(pair_env: dict[str, Any]):
    executor = pair_env["executor"]
    spot_ex, perp_ex = pair_env["spot"], pair_env["perp"]
    planner = OrderPlanner("fc3")
    real_now = int(time.time() * 1000)
    plan = planner.plan(
        make_approved(make_intent()),
        spot_price=PRICE,
        perp_price=PRICE,
        spot_rules=make_rule("spot"),
        perp_rules=make_rule("perp"),
        now_ms=real_now,
        plan_id="plan-exp",
    )
    old_plan = ExecutionPlan(
        plan_id=plan.plan_id,
        approved_intent=plan.approved_intent,
        orders=plan.orders,
        plan_created_at_ms=real_now - 40_000,
        expires_at_ms=real_now - 10_000,  # 已过期
    )
    with pytest.raises(PairPrecheckFailed, match="过期"):
        executor.open_pair_from_plan(
            old_plan,
            spot_price=PRICE,
            perp_price=PRICE,
            quote_ts_ms=int(time.time() * 1000),
            state=env_state(),
        )
    assert spot_ex.place_calls == [] and perp_ex.place_calls == []


def env_state() -> RiskState:
    return RiskState(total_capital=10_000.0, available_balance=5_000.0)


@pytest.fixture
def pair_env(tmp_path: Path) -> dict[str, Any]:
    spot = FakeExchange("spot")
    perp = FakeExchange("perp")
    store = StateStore(tmp_path / "trading.sqlite3")
    gate = RiskGate(RiskManager(RiskConfig()))
    from cointrader.execution.pair_executor import PairExecutor

    executor = PairExecutor(
        spot=spot,  # type: ignore[arg-type]
        futures=perp,  # type: ignore[arg-type]
        store=store,
        gate=gate,
        order_ack_timeout_seconds=3.0,
        poll_interval_seconds=0.01,
        sleep_fn=lambda s: None,
    )
    return {"spot": spot, "perp": perp, "store": store, "gate": gate, "executor": executor}


def test_plan_execution_happy_path_uses_plan_ids(pair_env: dict[str, Any]):
    executor = pair_env["executor"]
    planner = OrderPlanner("fc3")
    plan = planner.plan(
        make_approved(make_intent()),
        spot_price=PRICE,
        perp_price=PRICE,
        spot_rules=make_rule("spot"),
        perp_rules=make_rule("perp"),
        now_ms=int(time.time() * 1000),
        plan_id="plan-ok",
    )
    pair = executor.open_pair_from_plan(
        plan,
        spot_price=PRICE,
        perp_price=PRICE,
        quote_ts_ms=int(time.time() * 1000),
        state=env_state(),
    )
    assert pair.status == "COMPLETE"
    planned_ids = {o.client_order_id for o in plan.orders}
    placed_ids = {req.client_order_id for req in pair_env["perp"].place_calls + pair_env["spot"].place_calls}
    assert planned_ids <= placed_ids


def test_missing_spot_leg_rejected(pair_env: dict[str, Any]):
    executor = pair_env["executor"]
    planner = OrderPlanner("fc3")
    full = planner.plan(
        make_approved(make_intent()),
        spot_price=PRICE,
        perp_price=PRICE,
        spot_rules=make_rule("spot"),
        perp_rules=make_rule("perp"),
        now_ms=int(time.time() * 1000),
        plan_id="plan-nos",
    )
    only_perp = ExecutionPlan(
        plan_id=full.plan_id,
        approved_intent=full.approved_intent,
        orders=(full.orders[0],),
        plan_created_at_ms=full.plan_created_at_ms,
        expires_at_ms=full.expires_at_ms,
    )
    with pytest.raises(PairPrecheckFailed, match="现货腿"):
        executor.open_pair_from_plan(
            only_perp,
            spot_price=PRICE,
            perp_price=PRICE,
            quote_ts_ms=int(time.time() * 1000),
            state=env_state(),
        )


def test_domain_plan_order_requires_positive_quantity():
    with pytest.raises(Exception):
        PlanOrder(
            client_order_id="ct-x",
            market=MarketKind.SPOT,
            symbol="BTCUSDT",
            side=PlanSide.SELL,
            order_type=PlanOrderType.MARKET,
            quantity=Decimal("0"),
            price=None,
            reduce_only=True,
        )


# -- v5.0 T2 边界：planner 是 consumer，不是 freshness 权威 -------------------


def test_planner_does_not_enforce_quote_freshness(planner):
    """OrderPlanner 只消费给定的价格/规则，不校验报价年龄（AC-04 边界）。

    报价新鲜度 gate 在上游（runner/service 的 ``evaluate_quote_gate``）：
    越期报价在到达 planner 之前即被拒绝（broker 调用 0）。这里锁定
    planner 不因「报价陈旧」而改变规划行为，职责不重叠。
    """
    approved = make_approved(make_intent())
    # now 距报价产生远超 max_market_data_age（5s），但在审批有效期内
    plan = planner.plan(
        approved,
        spot_price=PRICE,
        perp_price=PRICE,
        spot_rules=make_rule("spot"),
        perp_rules=make_rule("perp"),
        now_ms=NOW + 20_000,
        plan_id="plan-stale-quote",
    )
    assert len(plan.orders) == 2
    assert all(o.quantity > 0 for o in plan.orders)
