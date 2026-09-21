"""domain 契约测试。

覆盖（AC-01/02/04）：

- 不可变性（frozen dataclass）；
- 金额/数量/费率一律 Decimal，反序列化拒绝二进制浮点数；
- 序列化 round-trip 保真；
- 非法值（非法枚举、负数名义额、时间逆序、无前瞻破坏）显式拒绝；
- 未知 schema version 拒绝解析；
- 风险审批链：无 ALLOW/RESIZE 不得创建 ApprovedIntent；过期拒绝；
  批准名义额不得超过请求。
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from cointrader.domain import (
    AccountSnapshot,
    ApprovedIntent,
    AssetBalance,
    CommandKind,
    ControlCommand,
    DataQuality,
    DomainError,
    EventEnvelope,
    ExecutionPlan,
    ExpiredDomainObject,
    HealthEvent,
    HealthKind,
    InstrumentQuote,
    IntentAction,
    InvalidDomainValue,
    MarketKind,
    MarketSnapshot,
    OrderEvent,
    PairStatus,
    PlanOrder,
    PlanOrderType,
    PlanSide,
    PortfolioIntent,
    PositionReason,
    PositionSnapshot,
    RiskDecision,
    RiskDecisionKind,
    RiskNotApproved,
    RuleEvidence,
    SafetyState,
    SafetyStateKind,
    StrategyAction,
    StrategyProposal,
    TargetPortfolio,
    TargetPosition,
    UnknownSchemaVersion,
)

DEC = Decimal


def _quote(**kwargs: object) -> InstrumentQuote:
    base: dict[str, object] = {
        "symbol": "BTCUSDT",
        "market": MarketKind.FUTURES,
        "price": DEC("100.5"),
        "funding_rate_8h": DEC("0.0001"),
        "funding_cutoff_ms": 1000,
        "quote_time_ms": 900,
    }
    base.update(kwargs)
    return InstrumentQuote(**base)  # type: ignore[arg-type]


def _snapshot(**kwargs: object) -> MarketSnapshot:
    base: dict[str, object] = {
        "snapshot_id": "s1",
        "generated_at_ms": 1000,
        "decision_cutoff_ms": 1000,
        "quality": DataQuality.FRESH,
        "quotes": (_quote(),),
    }
    base.update(kwargs)
    return MarketSnapshot(**base)  # type: ignore[arg-type]


def _intent(**kwargs: object) -> PortfolioIntent:
    base: dict[str, object] = {
        "intent_id": "i1",
        "action": IntentAction.OPEN,
        "symbol": "BTCUSDT",
        "target_spot_notional": DEC("100"),
        "target_perp_notional": DEC("99"),
        "reason": "carry",
        "snapshot_id": "s1",
        "decision_cutoff_ms": 1000,
        "created_at_ms": 1000,
    }
    base.update(kwargs)
    return PortfolioIntent(**base)  # type: ignore[arg-type]


def _decision(**kwargs: object) -> RiskDecision:
    base: dict[str, object] = {
        "decision_id": "d1",
        "decision": RiskDecisionKind.ALLOW,
        "symbol": "BTCUSDT",
        "requested_spot_notional": DEC("100"),
        "requested_perp_notional": DEC("99"),
        "approved_spot_notional": DEC("100"),
        "approved_perp_notional": DEC("99"),
        "snapshot_ids": ("s1",),
        "rules": (RuleEvidence("exposure_limit", True, "ok"),),
        "decided_at_ms": 1000,
        "valid_until_ms": 2000,
    }
    base.update(kwargs)
    return RiskDecision(**base)  # type: ignore[arg-type]


class TestImmutability:
    """契约对象不可变：状态变化 = 新对象。"""

    def test_market_snapshot_frozen(self) -> None:
        snap = _snapshot()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            snap.quality = DataQuality.STALE  # type: ignore[misc]

    def test_intent_frozen(self) -> None:
        intent = _intent()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            intent.reason = "changed"  # type: ignore[misc]

    def test_risk_decision_frozen(self) -> None:
        decision = _decision()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            decision.decision = RiskDecisionKind.REJECT  # type: ignore[misc]


class TestMarketSnapshot:
    def test_round_trip(self) -> None:
        snap = _snapshot()
        restored = MarketSnapshot.from_dict(json.loads(json.dumps(snap.to_dict())))
        assert restored == snap
        assert restored.quote("BTCUSDT", MarketKind.FUTURES) is not None
        assert restored.quote("BTCUSDT", MarketKind.SPOT) is None

    def test_rejects_float_price(self) -> None:
        data = _snapshot().to_dict()
        data["quotes"][0]["price"] = 100.5  # type: ignore[index]
        with pytest.raises(InvalidDomainValue, match="price"):
            MarketSnapshot.from_dict(data)

    def test_rejects_unknown_quality(self) -> None:
        data = _snapshot().to_dict()
        data["quality"] = "SUPER_FRESH"
        with pytest.raises(InvalidDomainValue, match="quality"):
            MarketSnapshot.from_dict(data)

    def test_rejects_future_quote(self) -> None:
        quote = _quote(quote_time_ms=1001)
        with pytest.raises(DomainError, match="无前瞻"):
            _snapshot(quotes=(quote,))

    def test_rejects_funding_cutoff_after_decision_cutoff(self) -> None:
        quote = _quote(funding_cutoff_ms=1001)
        with pytest.raises(DomainError, match="无前瞻"):
            _snapshot(quotes=(quote,))

    def test_rejects_cutoff_after_generated(self) -> None:
        with pytest.raises(DomainError):
            _snapshot(decision_cutoff_ms=1001)

    def test_rejects_non_positive_price(self) -> None:
        with pytest.raises(InvalidDomainValue):
            _quote(price=DEC("0"))
        with pytest.raises(InvalidDomainValue):
            _quote(price=DEC("-1"))

    def test_rejects_missing_field(self) -> None:
        data = _snapshot().to_dict()
        del data["snapshot_id"]
        with pytest.raises(InvalidDomainValue, match="snapshot_id"):
            MarketSnapshot.from_dict(data)


class TestAccountSnapshot:
    def _account(self, **kwargs: object) -> AccountSnapshot:
        base: dict[str, object] = {
            "snapshot_id": "a1",
            "capture_start_ms": 1000,
            "capture_end_ms": 1100,
            "complete": True,
            "equity": DEC("1000"),
            "available": DEC("800"),
            "balances": (AssetBalance("USDT", DEC("1000"), DEC("800")),),
        }
        base.update(kwargs)
        return AccountSnapshot(**base)  # type: ignore[arg-type]

    def test_round_trip_with_incomplete_flag(self) -> None:
        snap = self._account(complete=False)
        restored = AccountSnapshot.from_dict(json.loads(json.dumps(snap.to_dict())))
        assert restored == snap
        assert restored.complete is False

    def test_rejects_time_inversion(self) -> None:
        with pytest.raises(InvalidDomainValue, match="capture_end"):
            self._account(capture_end_ms=999)

    def test_rejects_negative_equity(self) -> None:
        with pytest.raises(InvalidDomainValue):
            self._account(equity=DEC("-1"))

    def test_position_tombstone_allowed(self) -> None:
        tomb = PositionSnapshot("BTCUSDT", MarketKind.SPOT, DEC("0"), DEC("0"), 5)
        assert tomb.is_tombstone
        live = PositionSnapshot("BTCUSDT", MarketKind.SPOT, DEC("1"), DEC("100.5"), 5)
        assert not live.is_tombstone
        with pytest.raises(InvalidDomainValue):
            PositionSnapshot("BTCUSDT", MarketKind.SPOT, DEC("-1"), DEC("0"), 5)


class TestStrategyProposal:
    def _proposal(self, **kwargs: object) -> StrategyProposal:
        base: dict[str, object] = {
            "proposal_id": "p1",
            "snapshot_id": "s1",
            "decision_cutoff_ms": 1000,
            "strategy_version": "funding-carry-1",
            "config_hash": "deadbeef",
            "valid_until_ms": 2000,
            "target": TargetPortfolio(entries=(TargetPosition("BTCUSDT", DEC("100"), DEC("99")),)),
            "reasons": (
                PositionReason(
                    "BTCUSDT", StrategyAction.OPEN, "high funding", ("rate=0.0001",)
                ),
            ),
        }
        base.update(kwargs)
        return StrategyProposal(**base)  # type: ignore[arg-type]

    def test_round_trip(self) -> None:
        proposal = self._proposal()
        restored = StrategyProposal.from_dict(json.loads(json.dumps(proposal.to_dict())))
        assert restored == proposal

    def test_empty_target_portfolio_legal(self) -> None:
        proposal = self._proposal(target=TargetPortfolio())
        assert proposal.target.entries == ()

    def test_target_rejects_negative_notional(self) -> None:
        with pytest.raises(InvalidDomainValue):
            TargetPosition("BTCUSDT", DEC("-1"), DEC("0"))

    def test_target_rejects_duplicate_symbol(self) -> None:
        with pytest.raises(InvalidDomainValue, match="重复"):
            TargetPortfolio(
                entries=(
                    TargetPosition("BTCUSDT", DEC("1"), DEC("1")),
                    TargetPosition("BTCUSDT", DEC("2"), DEC("2")),
                )
            )

    def test_expired_proposal_detected(self) -> None:
        proposal = self._proposal()
        assert proposal.is_expired(2000) is True
        assert proposal.is_expired(1999) is False

    def test_rejects_valid_until_before_cutoff(self) -> None:
        with pytest.raises(InvalidDomainValue, match="valid_until"):
            self._proposal(valid_until_ms=999)


class TestPortfolioIntent:
    def test_fingerprint_deterministic_and_idempotent(self) -> None:
        a = _intent(created_at_ms=1000, correlation_id=None)
        b = _intent(created_at_ms=9999, correlation_id="different")
        assert a.fingerprint() == b.fingerprint()

    def test_fingerprint_changes_with_semantics(self) -> None:
        a = _intent()
        b = _intent(symbol="ETHUSDT")
        assert a.fingerprint() != b.fingerprint()

    def test_close_intent_requires_zero_targets(self) -> None:
        with pytest.raises(InvalidDomainValue, match="CLOSE"):
            _intent(action=IntentAction.CLOSE, target_spot_notional=DEC("1"), reduces_risk=True)
        close = _intent(
            action=IntentAction.CLOSE, target_spot_notional=DEC("0"), target_perp_notional=DEC("0"), reduces_risk=True
        )
        assert close.is_closing

    def test_close_intent_requires_reduces_risk(self) -> None:
        with pytest.raises(InvalidDomainValue, match="reduces_risk"):
            _intent(
                action=IntentAction.CLOSE,
                target_spot_notional=DEC("0"),
                target_perp_notional=DEC("0"),
                reduces_risk=False,
            )

    def test_reduces_risk_round_trip_and_legacy_compat(self) -> None:
        intent = _intent(reduces_risk=True)
        data = intent.to_dict()
        restored = PortfolioIntent.from_dict(json.loads(json.dumps(data)))
        assert restored.reduces_risk is True
        # 旧数据（无 reduces_risk 字段）：CLOSE 按语义推导为 True，保持兼容
        close = _intent(
            action=IntentAction.CLOSE,
            target_spot_notional=DEC("0"),
            target_perp_notional=DEC("0"),
            reduces_risk=True,
        )
        legacy_data = close.to_dict()
        del legacy_data["reduces_risk"]
        assert PortfolioIntent.from_dict(legacy_data).reduces_risk is True
        # 旧数据非 CLOSE → 默认 False
        legacy_data2 = _intent().to_dict()
        del legacy_data2["reduces_risk"]
        assert PortfolioIntent.from_dict(legacy_data2).reduces_risk is False
        # 新数据显式 False + CLOSE → 拒绝
        bad = close.to_dict()
        bad["reduces_risk"] = False
        with pytest.raises(InvalidDomainValue, match="reduces_risk"):
            PortfolioIntent.from_dict(bad)

    def test_round_trip_verifies_stored_fingerprint(self) -> None:
        intent = _intent()
        data = intent.to_dict()
        restored = PortfolioIntent.from_dict(json.loads(json.dumps(data)))
        assert restored == intent
        data["fingerprint"] = "deadbeef"
        with pytest.raises(InvalidDomainValue, match="fingerprint"):
            PortfolioIntent.from_dict(data)


class TestRiskDecisionChain:
    def test_only_allow_resize_executable(self) -> None:
        for kind in RiskDecisionKind:
            decision = _decision(decision=kind)
            assert decision.allows_execution is (kind in (RiskDecisionKind.ALLOW, RiskDecisionKind.RESIZE))

    def test_approved_cannot_exceed_requested(self) -> None:
        with pytest.raises(InvalidDomainValue, match="收紧"):
            _decision(approved_spot_notional=DEC("101"))
        with pytest.raises(InvalidDomainValue, match="收紧"):
            _decision(approved_perp_notional=DEC("100"))

    def test_requires_snapshot_ids(self) -> None:
        with pytest.raises(InvalidDomainValue, match="快照"):
            _decision(snapshot_ids=())

    def test_from_decision_allows(self) -> None:
        approved = ApprovedIntent.from_decision(_intent(), _decision(), now_ms=1500)
        assert approved.decision_id == "d1"
        assert approved.approved_spot_notional == DEC("100")
        assert approved.is_closing is False

    def test_from_decision_rejects_non_allow(self) -> None:
        with pytest.raises(RiskNotApproved, match="REJECT"):
            ApprovedIntent.from_decision(_intent(), _decision(decision=RiskDecisionKind.REJECT), now_ms=1500)
        with pytest.raises(RiskNotApproved, match="HALT"):
            ApprovedIntent.from_decision(_intent(), _decision(decision=RiskDecisionKind.HALT), now_ms=1500)
        with pytest.raises(RiskNotApproved):
            ApprovedIntent.from_decision(
                _intent(), _decision(decision=RiskDecisionKind.CLOSE_ONLY), now_ms=1500
            )

    def test_from_decision_rejects_expired(self) -> None:
        with pytest.raises(ExpiredDomainObject, match="过期"):
            ApprovedIntent.from_decision(_intent(), _decision(), now_ms=2000)

    def test_from_decision_rejects_symbol_mismatch(self) -> None:
        with pytest.raises(InvalidDomainValue, match="symbol"):
            ApprovedIntent.from_decision(_intent(symbol="ETHUSDT"), _decision(), now_ms=1500)

    def test_resize_marks_reduce_only(self) -> None:
        decision = _decision(
            decision=RiskDecisionKind.RESIZE,
            approved_spot_notional=DEC("50"),
        )
        approved = ApprovedIntent.from_decision(_intent(), decision, now_ms=1500)
        assert approved.reduce_only is True
        assert approved.approved_spot_notional == DEC("50")

    def test_close_intent_cannot_lose_is_closing(self) -> None:
        close = _intent(
            action=IntentAction.CLOSE,
            target_spot_notional=DEC("0"),
            target_perp_notional=DEC("0"),
            reduces_risk=True,
        )
        decision = _decision(
            requested_spot_notional=DEC("0"),
            requested_perp_notional=DEC("0"),
            approved_spot_notional=DEC("0"),
            approved_perp_notional=DEC("0"),
        )
        approved = ApprovedIntent.from_decision(close, decision, now_ms=1500)
        assert approved.is_closing is True
        with pytest.raises(InvalidDomainValue, match="is_closing"):
            ApprovedIntent(
                intent=close,
                decision_id="d1",
                approved_spot_notional=DEC("0"),
                approved_perp_notional=DEC("0"),
                decided_at_ms=1000,
                valid_until_ms=2000,
                is_closing=False,
            )

    def test_decision_round_trip(self) -> None:
        restored = RiskDecision.from_dict(json.loads(json.dumps(_decision().to_dict())))
        assert restored == _decision()
        assert restored.rules[0].passed is True

    # -- T1：ApprovedIntent 契约强化 -------------------------------------------

    def test_is_closing_cannot_be_forged(self) -> None:
        with pytest.raises(InvalidDomainValue, match="is_closing"):
            ApprovedIntent(
                intent=_intent(),
                decision_id="d1",
                approved_spot_notional=DEC("100"),
                approved_perp_notional=DEC("99"),
                decided_at_ms=1000,
                valid_until_ms=2000,
                is_closing=True,
            )

    def test_approved_cannot_exceed_intent_target(self) -> None:
        with pytest.raises(InvalidDomainValue, match="收紧"):
            ApprovedIntent(
                intent=_intent(),
                decision_id="d1",
                approved_spot_notional=DEC("101"),
                approved_perp_notional=DEC("99"),
                decided_at_ms=1000,
                valid_until_ms=2000,
            )
        with pytest.raises(InvalidDomainValue, match="收紧"):
            ApprovedIntent(
                intent=_intent(),
                decision_id="d1",
                approved_spot_notional=DEC("100"),
                approved_perp_notional=DEC("100"),
                decided_at_ms=1000,
                valid_until_ms=2000,
            )

    def test_from_decision_requires_intent_snapshot(self) -> None:
        with pytest.raises(InvalidDomainValue, match="快照"):
            ApprovedIntent.from_decision(
                _intent(), _decision(snapshot_ids=("other",)), now_ms=1500
            )

    def test_from_decision_rejects_causality_violation(self) -> None:
        with pytest.raises(InvalidDomainValue, match="因果"):
            ApprovedIntent.from_decision(
                _intent(), _decision(decided_at_ms=999, valid_until_ms=2000), now_ms=1500
            )

    def test_from_decision_rejects_lookahead(self) -> None:
        # 审批时点 >= 意图创建但早于决策截止 → 无前瞻被破坏
        early_intent = _intent(created_at_ms=900)
        with pytest.raises(InvalidDomainValue, match="无前瞻"):
            ApprovedIntent.from_decision(
                early_intent,
                _decision(decided_at_ms=950, valid_until_ms=2000),
                now_ms=1500,
            )

    def test_correlation_fields_survive_serialization(self) -> None:
        intent = _intent(correlation_id="corr-9", causation_id="cause-9")
        approved = ApprovedIntent.from_decision(intent, _decision(), now_ms=1500)
        restored = ApprovedIntent.from_dict(json.loads(json.dumps(approved.to_dict())))
        assert restored == approved
        assert restored.intent.correlation_id == "corr-9"
        assert restored.intent.causation_id == "cause-9"
        assert restored.intent.fingerprint() == intent.fingerprint()


class TestExecutionPlan:
    def _plan(self, **kwargs: object) -> ExecutionPlan:
        intent = _intent()
        base: dict[str, object] = {
            "plan_id": "pl1",
            "approved_intent": ApprovedIntent.from_decision(intent, _decision(), now_ms=1500),
            "orders": (
                PlanOrder(
                    client_order_id="cid-1",
                    market=MarketKind.SPOT,
                    symbol="BTCUSDT",
                    side=PlanSide.BUY,
                    order_type=PlanOrderType.LIMIT,
                    quantity=DEC("0.01"),
                    price=DEC("100.5"),
                ),
            ),
            "plan_created_at_ms": 1000,
            "expires_at_ms": 2000,
        }
        base.update(kwargs)
        return ExecutionPlan(**base)  # type: ignore[arg-type]

    def test_round_trip(self) -> None:
        plan = self._plan()
        restored = ExecutionPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        assert restored == plan

    def test_rejects_non_positive_quantity(self) -> None:
        with pytest.raises(InvalidDomainValue, match="数量"):
            PlanOrder(
                client_order_id="c",
                market=MarketKind.SPOT,
                symbol="S",
                side=PlanSide.BUY,
                order_type=PlanOrderType.LIMIT,
                quantity=DEC("0"),
                price=DEC("1"),
            )

    def test_limit_requires_price(self) -> None:
        with pytest.raises(InvalidDomainValue, match="LIMIT"):
            PlanOrder(
                client_order_id="c",
                market=MarketKind.SPOT,
                symbol="S",
                side=PlanSide.BUY,
                order_type=PlanOrderType.LIMIT,
                quantity=DEC("1"),
                price=None,
            )

    def test_market_order_without_price_allowed(self) -> None:
        order = PlanOrder(
            client_order_id="c",
            market=MarketKind.FUTURES,
            symbol="S",
            side=PlanSide.SELL,
            order_type=PlanOrderType.MARKET,
            quantity=DEC("1"),
            price=None,
            reduce_only=True,
        )
        assert order.price is None

    def test_rejects_expires_before_created(self) -> None:
        with pytest.raises(InvalidDomainValue):
            self._plan(expires_at_ms=999)

    def test_closing_plan_requires_reduce_only_orders(self) -> None:
        close = _intent(
            action=IntentAction.CLOSE,
            target_spot_notional=DEC("0"),
            target_perp_notional=DEC("0"),
            reduces_risk=True,
        )
        decision = _decision(
            requested_spot_notional=DEC("0"),
            requested_perp_notional=DEC("0"),
            approved_spot_notional=DEC("0"),
            approved_perp_notional=DEC("0"),
        )
        approved = ApprovedIntent.from_decision(close, decision, now_ms=1500)
        with pytest.raises(InvalidDomainValue, match="reduce_only"):
            self._plan(approved_intent=approved)
        good_orders = (
            PlanOrder(
                client_order_id="cid-1",
                market=MarketKind.SPOT,
                symbol="BTCUSDT",
                side=PlanSide.SELL,
                order_type=PlanOrderType.LIMIT,
                quantity=DEC("0.01"),
                price=DEC("100.5"),
                reduce_only=True,
            ),
        )
        plan = self._plan(approved_intent=approved, orders=good_orders)
        assert plan.is_expired(2000) is True


class TestControlAndSafety:
    def test_all_states_round_trip(self) -> None:
        for kind in SafetyStateKind:
            state = SafetyState(kind, "reason", 1000)
            restored = SafetyState.from_dict(json.loads(json.dumps(state.to_dict())))
            assert restored == state

    def test_only_running_allows_new_risk(self) -> None:
        for kind in SafetyStateKind:
            state = SafetyState(kind, "r", 1)
            assert state.allows_new_risk == (kind is SafetyStateKind.RUNNING)

    def test_control_command_round_trip(self) -> None:
        cmd = ControlCommand(
            CommandKind.HALT_NEW_RISK,
            "db write failure",
            "watchdog",
            1000,
            correlation_id="corr-1",
        )
        restored = ControlCommand.from_dict(json.loads(json.dumps(cmd.to_dict())))
        assert restored == cmd

    def test_emergency_flatten_requires_manual_source(self) -> None:
        with pytest.raises(InvalidDomainValue, match="manual"):
            ControlCommand(CommandKind.EMERGENCY_FLATTEN, "panic", "watchdog", 1000)
        cmd = ControlCommand(CommandKind.EMERGENCY_FLATTEN, "panic", "manual", 1000)
        assert cmd.command is CommandKind.EMERGENCY_FLATTEN


class TestEvents:
    def test_envelope_round_trip(self) -> None:
        env = EventEnvelope(
            event_id="e1",
            event_type="order.filled",
            schema_version=1,
            occurred_at_ms=1000,
            correlation_id="c1",
            causation_id="c0",
            payload=(("executed_qty", "0.5"),),
        )
        restored = EventEnvelope.from_dict(json.loads(json.dumps(env.to_dict())))
        assert restored == env

    def test_unknown_schema_version_rejected(self) -> None:
        env = EventEnvelope("e1", "t", 1, 0, None, None)
        data = env.to_dict()
        data["schema_version"] = 99
        with pytest.raises(UnknownSchemaVersion, match="99"):
            EventEnvelope.from_dict(data)
        with pytest.raises(UnknownSchemaVersion):
            EventEnvelope(event_id="e1", event_type="t", schema_version=99,
                          occurred_at_ms=0, correlation_id=None, causation_id=None)

    def test_rejects_unknown_keys_in_envelope(self) -> None:
        data = EventEnvelope("e1", "t", 1, 0, None, None).to_dict()
        data["mystery_field"] = "x"
        with pytest.raises(InvalidDomainValue, match="mystery_field"):
            EventEnvelope.from_dict(data)

    def test_health_event_payload(self) -> None:
        event = HealthEvent("market_data", HealthKind.DEGRADED, "epoch expired", ("detail-a",))
        payload = dict(event.to_payload())
        assert payload["kind"] == "DEGRADED"
        assert payload["detail.0"] == "detail-a"

    def test_order_event_payload_round_trip(self) -> None:
        event = OrderEvent("cid-1", "BTCUSDT", "FILLED", DEC("0.25"), 1234)
        restored = OrderEvent.from_payload(event.to_payload())
        assert restored == event
        with pytest.raises(InvalidDomainValue, match="executed_qty"):
            OrderEvent.from_payload((("client_order_id", "cid"),))

    def test_serialized_decimals_are_strings(self) -> None:
        raw = json.dumps(_snapshot().to_dict())
        assert '"price": "100.5"' in raw

    def test_pair_status_matches_execution_layer(self) -> None:
        """domain PairStatus 与 execution 层状态一一对应（迁移期兼容）。"""
        from cointrader.execution.models import PairStatus as ExecPairStatus

        domain_values = {s.value for s in PairStatus}
        exec_values = {s.value for s in ExecPairStatus}
        assert domain_values == exec_values
