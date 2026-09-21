"""T3：风险内核 RiskKernel 契约测试（AC-04）。

覆盖：ALLOW / RESIZE / CLOSE_ONLY / HALT / REJECT 五条路径、
规则证据完整性、确定性（同输入同决定）、平仓放行、无前瞻保护。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cointrader.domain.account import AccountSnapshot
from cointrader.domain.common import InvalidDomainValue
from cointrader.domain.control import SafetyState, SafetyStateKind
from cointrader.domain.market import DataQuality, InstrumentQuote, MarketKind, MarketSnapshot
from cointrader.domain.portfolio import IntentAction, PortfolioIntent
from cointrader.domain.risk import ApprovedIntent, RiskDecisionKind
from cointrader.execution.risk import Position, RiskManager, RiskState
from cointrader.risk.adapter import RiskRulesAdapter
from cointrader.risk.kernel import RiskExposure, RiskKernel
from live_helpers import NOW_MS, make_live_config

NOW = NOW_MS


def make_intent(action: IntentAction = IntentAction.OPEN, spot: str = "100", perp: str = "100") -> PortfolioIntent:
    return PortfolioIntent(
        intent_id="it-1",
        action=action,
        symbol="BTCUSDT",
        target_spot_notional=Decimal(spot),
        target_perp_notional=Decimal(perp),
        reason="t3-kernel",
        snapshot_id="snap-1",
        decision_cutoff_ms=NOW - 1000,
        created_at_ms=NOW,
        reduces_risk=action is IntentAction.CLOSE,
    )


def make_market(quality: DataQuality = DataQuality.FRESH, *, gen_ms: int = NOW) -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id="snap-m",
        generated_at_ms=gen_ms,
        decision_cutoff_ms=NOW - 1000,
        quality=quality,
        quotes=(
            InstrumentQuote(
                symbol="BTCUSDT",
                market=MarketKind.SPOT,
                price=Decimal("100"),
                funding_rate_8h=None,
                funding_cutoff_ms=None,
                quote_time_ms=NOW - 500,
            ),
        ),
    )


def make_account(complete: bool = True) -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id="snap-a",
        capture_start_ms=NOW - 2000,
        capture_end_ms=NOW - 500,
        complete=complete,
        equity=Decimal("10000"),
        available=Decimal("5000"),
    )


def make_safety(kind: SafetyStateKind = SafetyStateKind.RUNNING) -> SafetyState:
    return SafetyState(kind, reason=f"t3-{kind.value}", changed_at_ms=NOW)


def make_risk_state() -> RiskState:
    return RiskState(total_capital=10_000.0, available_balance=5_000.0)


@pytest.fixture
def kernel() -> RiskKernel:
    cfg = make_live_config()
    return RiskKernel(cfg, rules=RiskRulesAdapter(RiskManager(cfg.risk)))


def _approve(kernel, intent, **kw):
    defaults = dict(
        now_ms=NOW,
        safety=make_safety(),
        market=make_market(),
        account=make_account(),
        risk_state=make_risk_state(),
    )
    defaults.update(kw)
    return kernel.approve(intent, **defaults)


# -- ALLOW -------------------------------------------------------------------


def test_allows_open_when_all_rules_pass(kernel):
    d = _approve(kernel, make_intent())
    assert d.decision is RiskDecisionKind.ALLOW
    assert d.approved_spot_notional == Decimal("100")
    assert d.approved_perp_notional == Decimal("100")
    assert d.valid_until_ms > d.decided_at_ms
    assert "snap-m" in d.snapshot_ids and "snap-a" in d.snapshot_ids


def test_rule_evidence_covers_all_rule_ids(kernel):
    d = _approve(kernel, make_intent())
    ids = [r.rule for r in d.rules]
    for expected in (
        "notional_valid",
        "safety_state",
        "market_fresh",
        "account_complete",
        "risk_preflight",
        "risk_limits",
    ):
        assert expected in ids
    assert all(r.passed for r in d.rules)


def test_decision_is_deterministic(kernel):
    a = _approve(kernel, make_intent())
    b = _approve(kernel, make_intent())
    assert a.decision_id == b.decision_id
    assert a.approved_spot_notional == b.approved_spot_notional
    assert [r.to_dict() for r in a.rules] == [r.to_dict() for r in b.rules]


def test_approved_intent_roundtrip(kernel):
    intent = make_intent()
    d = _approve(kernel, intent)
    approved = ApprovedIntent.from_decision(intent, d, now_ms=NOW)
    assert approved.approved_spot_notional == Decimal("100")
    assert approved.is_closing is False
    assert not approved.is_expired(NOW)


# -- 平仓 --------------------------------------------------------------------


def test_close_intent_allowed_with_zero_approved(kernel):
    intent = make_intent(IntentAction.CLOSE, spot="0", perp="0")
    d = _approve(kernel, intent)
    assert d.decision is RiskDecisionKind.ALLOW
    assert d.approved_spot_notional == Decimal("0")
    assert d.approved_perp_notional == Decimal("0")
    # 平仓不要求市场/账户快照
    d2 = _approve(kernel, intent, market=None, account=None, risk_state=None)
    assert d2.decision is RiskDecisionKind.ALLOW


def test_close_intent_rejected_when_stopped(kernel):
    intent = make_intent(IntentAction.CLOSE, spot="0", perp="0")
    d = _approve(kernel, intent, safety=make_safety(SafetyStateKind.STOPPED))
    assert d.decision is RiskDecisionKind.REJECT


# -- 降级路径 ----------------------------------------------------------------


def test_degraded_safety_gives_close_only(kernel):
    d = _approve(kernel, make_intent(), safety=make_safety(SafetyStateKind.DEGRADED))
    assert d.decision is RiskDecisionKind.CLOSE_ONLY
    assert d.approved_spot_notional == Decimal("0")
    failed = [r for r in d.rules if r.rule == "safety_state"]
    assert failed and not failed[0].passed


def test_halted_safety_gives_halt(kernel):
    d = _approve(kernel, make_intent(), safety=make_safety(SafetyStateKind.HALTED))
    assert d.decision is RiskDecisionKind.HALT


def test_stale_market_gives_close_only(kernel):
    d = _approve(kernel, make_intent(), market=make_market(DataQuality.STALE))
    assert d.decision is RiskDecisionKind.CLOSE_ONLY


def test_missing_market_gives_close_only(kernel):
    d = _approve(kernel, make_intent(), market=None)
    assert d.decision is RiskDecisionKind.CLOSE_ONLY


def test_incomplete_account_gives_close_only(kernel):
    d = _approve(kernel, make_intent(), account=make_account(complete=False))
    assert d.decision is RiskDecisionKind.CLOSE_ONLY


def test_missing_risk_state_gives_close_only(kernel):
    d = _approve(kernel, make_intent(), risk_state=None)
    assert d.decision is RiskDecisionKind.CLOSE_ONLY


# -- 硬拒绝 ------------------------------------------------------------------


def test_invalid_notional_rejected(kernel):
    d = _approve(kernel, make_intent(spot="0", perp="0"))
    assert d.decision is RiskDecisionKind.REJECT


def test_daily_loss_rejects(kernel):
    state = make_risk_state()
    state.realized_pnl_today = -300.0  # 3% > 2% 停机线
    d = _approve(kernel, make_intent(), risk_state=state)
    assert d.decision is RiskDecisionKind.REJECT
    assert any(not r.passed for r in d.rules if r.rule == "risk_preflight")


def test_market_time_in_future_rejected(kernel):
    d = _approve(kernel, make_intent(), market=make_market(gen_ms=NOW + 60_000))
    assert d.decision is RiskDecisionKind.REJECT


# -- RESIZE ------------------------------------------------------------------


def test_resize_when_per_symbol_exposure_exceeded(kernel):
    state = make_risk_state()
    state.positions["BTCUSDT"] = Position(
        symbol="BTCUSDT", spot_qty=1.9, perp_qty=1.9, spot_price=100.0, perp_price=100.0
    )  # 当前敞口 380；单币种上限 500 → 可用 120
    d = _approve(kernel, make_intent(), risk_state=state)
    assert d.decision is RiskDecisionKind.RESIZE
    total = d.approved_spot_notional + d.approved_perp_notional
    assert total == Decimal("120")
    assert total < Decimal("200")
    assert all(r.passed for r in d.rules)


def test_close_only_when_no_headroom(kernel):
    state = make_risk_state()
    state.positions["BTCUSDT"] = Position(
        symbol="BTCUSDT", spot_qty=5.0, perp_qty=5.0, spot_price=100.0, perp_price=100.0
    )  # 当前敞口 1000 > 单币种上限 500 → 可用 0
    d = _approve(kernel, make_intent(), risk_state=state)
    assert d.decision is RiskDecisionKind.CLOSE_ONLY
    assert d.approved_spot_notional == Decimal("0")


# -- 领域联动 ----------------------------------------------------------------


def test_rejected_decision_cannot_become_approved_intent(kernel):
    intent = make_intent()
    d = _approve(kernel, intent, safety=make_safety(SafetyStateKind.CLOSE_ONLY))
    with pytest.raises(Exception):
        ApprovedIntent.from_decision(intent, d, now_ms=NOW)


def test_safety_state_requires_reason():
    with pytest.raises(InvalidDomainValue):
        SafetyState(SafetyStateKind.RUNNING, reason="", changed_at_ms=NOW)


# -- T1：全链路 Decimal 端口 + 快照一致性 ------------------------------------


class _RecordingRules:
    """记录端口金额类型的 fake ``RiskRulesPort``（验证内核不做 float 转换）。"""

    def __init__(self, *, allow: bool = True) -> None:
        self.allow = allow
        self.notional_types: list[type] = []
        self._exp = RiskExposure(Decimal("0"), {})

    def preflight_all(self, state: object) -> tuple[tuple[str, bool, str], ...]:
        return (("risk_preflight", True, "ok"),)

    def check_order(self, symbol: str, notional: Decimal, state: object) -> tuple[bool, str]:
        self.notional_types.append(type(notional))
        return (self.allow, "ok" if self.allow else "单币种限额超限")

    def exposure(self, state: object) -> RiskExposure:
        return self._exp


def test_kernel_passes_decimal_notional_to_rules_port():
    rules = _RecordingRules()
    kernel = RiskKernel(make_live_config(), rules=rules)
    d = _approve(kernel, make_intent())
    assert d.decision is RiskDecisionKind.ALLOW
    assert rules.notional_types == [Decimal]


def test_success_decision_includes_intent_snapshot_id(kernel):
    d = _approve(kernel, make_intent())
    # 审批必须引用意图输入快照（ApprovedIntent.from_decision 可静态校验）
    assert "snap-1" in d.snapshot_ids


def test_resize_path_all_decimal():
    """RESIZE 收紧路径全 Decimal：fake 规则拒绝后，批准额 = 可用额度，
    且端口收到的金额类型均为 Decimal（内核不做 float 转换）。"""
    cfg = make_live_config()
    rules = _RecordingRules(allow=False)
    rules._exp = RiskExposure(Decimal("380"), {"BTCUSDT": Decimal("380")})
    kernel = RiskKernel(cfg, rules=rules)
    state = make_risk_state()
    state.positions["BTCUSDT"] = Position(
        symbol="BTCUSDT", spot_qty=1.9, perp_qty=1.9, spot_price=100.0, perp_price=100.0
    )  # 当前敞口 380；单币种上限 500 → 可用 120
    d = _approve(kernel, make_intent(), risk_state=state)
    assert d.decision is RiskDecisionKind.RESIZE
    assert d.approved_spot_notional + d.approved_perp_notional == Decimal("120")
    assert rules.notional_types == [Decimal]
    # 适配器路径：legacy float 事实 → Decimal 端口（一次性边界转换）
    adapter = RiskRulesAdapter(RiskManager(cfg.risk))
    port_exp = adapter.exposure(state)
    assert isinstance(port_exp.total_exposure, Decimal)
    assert all(isinstance(v, Decimal) for v in port_exp.symbol_exposure.values())
