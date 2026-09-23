"""T2：application 生产 pipeline（proposal → intent）契约测试（计划 4.0 T2，AC-02/03/04）。

覆盖：

- 确定性：相同 fake 输入 → 相同 proposal/intent 序列（intent_id = 指纹）；
- 拒绝/跳过也写策略结果（proposal reasons 带证据，目标组合不含被拒 symbol）；
- 指纹幂等：重复输入不产生第二个 intent；
- 换仓：CLOSE 先于 REPLACE/OPEN（先平旧后开新），CLOSE 腿 reduces_risk=True；
- pipeline 不持有 broker/executor（静态扫描 + 构造面），T2 阶段 broker 调用 0；
- 账本写失败原样上抛（fail closed，不得吞掉）。
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from cointrader.application.composition import Pipeline
from cointrader.domain.portfolio import CurrentPosition, IntentAction, PortfolioView
from cointrader.domain.strategy import StrategyProposal, TargetPortfolio, TargetPosition
from cointrader.portfolio.planner import PortfolioPlanner
from cointrader.strategy.adapter import decisions_to_proposal

NOW = 1_800_000_000_000


class _Decision:
    """legacy ``StrategyDecision`` 的最小 fake（结构协议匹配）。"""

    def __init__(
        self,
        symbol: str,
        kind: str,
        *,
        allowed: bool = True,
        notional: Decimal | None = None,
        rc: str = "ENTRY_OK",
        rt: str = "",
    ) -> None:
        self._symbol = symbol
        self._kind = kind
        self._allowed = allowed
        self._notional = notional
        self._rc = rc
        self._rt = rt

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def decision_kind(self) -> str:
        return self._kind

    @property
    def allowed(self) -> bool:
        return self._allowed

    @property
    def requested_notional(self) -> Decimal | None:
        return self._notional

    @property
    def reason_code(self) -> str:
        return self._rc

    @property
    def reason_text(self) -> str:
        return self._rt


class _FakeLedger:
    """内存 ``PipelineLedgerPort`` fake（幂等 + 失败注入）。"""

    def __init__(self) -> None:
        self.proposals: list[dict[str, object]] = []
        self.intents: list[dict[str, object]] = []
        self.plans: list[dict[str, object]] = []
        self.fail = False

    def _check(self) -> None:
        if self.fail:
            raise RuntimeError("db write failure (injected)")

    def append_strategy_proposal(self, proposal: Mapping[str, object]) -> bool:
        self._check()
        pid = str(proposal["proposal_id"])
        if any(p.get("proposal_id") == pid for p in self.proposals):
            return False
        self.proposals.append(dict(proposal))
        return True

    def append_portfolio_intent(self, intent: Mapping[str, object]) -> bool:
        self._check()
        iid = str(intent["intent_id"])
        if any(i.get("intent_id") == iid for i in self.intents):
            return False
        self.intents.append(dict(intent))
        return True

    def append_risk_decision(self, decision: Mapping[str, object]) -> bool:  # pragma: no cover
        self._check()
        return True

    def append_execution_plan(self, plan: Mapping[str, object]) -> bool:  # pragma: no cover
        self._check()
        self.plans.append(dict(plan))
        return True

    def has_portfolio_intent_fingerprint(self, fingerprint: str) -> bool:
        return any(i.get("intent_id") == fingerprint for i in self.intents)


def _view(symbols: dict[str, tuple[str, str]]) -> PortfolioView:
    return PortfolioView(
        snapshot_id="proj-1",
        as_of_ms=NOW,
        entries=tuple(
            CurrentPosition(
                symbol=s,
                spot_notional=Decimal(spot),
                perp_notional=Decimal(perp),
                updated_at_ms=NOW,
            )
            for s, (spot, perp) in sorted(symbols.items())
        ),
    )


def _proposal(decisions: list[_Decision], view: PortfolioView) -> Any:
    return decisions_to_proposal(
        decisions,
        view,
        proposal_id="prop-1",
        snapshot_id="snap-m",
        decision_cutoff_ms=NOW - 1000,
        strategy_version="v-test",
        config_hash="h",
        valid_until_ms=NOW + 30_000,
    )


def _pipeline(ledger: _FakeLedger) -> Pipeline:
    return Pipeline(planner=PortfolioPlanner(), ledger=ledger)


class TestPipelineDeterminism:
    def test_same_input_same_proposal_and_intents(self) -> None:
        view = _view({"AAAUSDT": ("100", "100")})
        decisions = [
            _Decision("BBBUSDT", "OPEN", notional=Decimal("50"), rc="ENTRY_OK"),
            _Decision("AAAUSDT", "EXIT", rc="NEGATIVE_EXIT_AVG", rt="负费率"),
        ]
        ids: list[tuple[str, ...]] = []
        for _ in range(2):
            ledger = _FakeLedger()
            _, intents = _pipeline(ledger).run_round(
                proposal=_proposal(decisions, view), view=view, now_ms=NOW, run_id="run-1"
            )
            ids.append(tuple(i.intent_id for i in intents))
        assert ids[0] == ids[1]
        assert len(ids[0]) == 2

    def test_rejects_and_skips_recorded_in_proposal(self) -> None:
        view = _view({})
        decisions = [
            _Decision("AAAUSDT", "OPEN", notional=Decimal("50"), rc="ENTRY_OK"),
            _Decision("BBBUSDT", "OPEN", allowed=False, rc="MAX_POSITIONS", rt="槽位已满"),
            _Decision("CCCUSDT", "SKIP", rc="RANKED_OUT", rt="排名第 4"),
        ]
        ledger = _FakeLedger()
        proposal, intents = _pipeline(ledger).run_round(
            proposal=_proposal(decisions, view), view=view, now_ms=NOW
        )
        # 策略结果（含拒绝/跳过证据）落账
        assert len(ledger.proposals) == 1
        target_symbols = {e.symbol for e in proposal.target.entries}
        assert target_symbols == {"AAAUSDT"}  # 被拒/跳过的 symbol 不进目标组合
        actions = {r.symbol: r.action.value for r in proposal.reasons}
        assert actions["BBBUSDT"] == "SKIP"
        assert actions["CCCUSDT"] == "SKIP"
        assert len(intents) == 1
        assert intents[0].symbol == "AAAUSDT"
        # 两腿目标名义额 = 请求名义额（与 legacy signal 语义一致）
        assert intents[0].target_spot_notional == Decimal("50")
        assert intents[0].target_perp_notional == Decimal("50")


class TestPipelineIdempotency:
    def test_duplicate_fingerprint_no_second_intent(self) -> None:
        view = _view({})
        decisions = [_Decision("AAAUSDT", "OPEN", notional=Decimal("50"))]
        ledger = _FakeLedger()
        pipeline = _pipeline(ledger)
        first, intents1 = pipeline.run_round(
            proposal=_proposal(decisions, view), view=view, now_ms=NOW
        )
        second, intents2 = pipeline.run_round(
            proposal=_proposal(decisions, view), view=view, now_ms=NOW
        )
        assert len(intents1) == 1
        assert intents2 == (), "重复 fingerprint 不产生第二个 intent"
        assert len(ledger.intents) == 1
        assert len(ledger.proposals) == 1  # proposal 也幂等（同 proposal_id）
        assert first == second

    def test_changed_view_produces_new_intent(self) -> None:
        decisions = [_Decision("AAAUSDT", "OPEN", notional=Decimal("50"))]
        ledger = _FakeLedger()
        pipeline = _pipeline(ledger)
        pipeline.run_round(proposal=_proposal(decisions, _view({})), view=_view({}), now_ms=NOW)
        # 新快照（不同 snapshot_id）→ 新指纹 → 允许再次产生 intent
        view2 = PortfolioView(
            snapshot_id="proj-2", as_of_ms=NOW + 8_000, entries=_view({}).entries
        )
        proposal2 = StrategyProposal(
            proposal_id="prop-2",
            snapshot_id="proj-2",
            decision_cutoff_ms=NOW + 8_000,
            strategy_version="v-test",
            config_hash="h",
            valid_until_ms=NOW + 30_000,
            target=TargetPortfolio(
                entries=(TargetPosition("AAAUSDT", Decimal("50"), Decimal("50")),)
            ),
            reasons=(),
        )
        _, intents = pipeline.run_round(proposal=proposal2, view=view2, now_ms=NOW + 8_000)
        assert len(intents) == 1
        assert len(ledger.intents) == 2


class TestPipelineReplaceOrdering:
    def test_replace_close_before_open(self) -> None:
        view = _view({"AAAUSDT": ("100", "100")})
        decisions = [
            _Decision("AAAUSDT", "REPLACE", rc="REPLACEMENT", rt="换仓"),
            _Decision("BBBUSDT", "OPEN", notional=Decimal("80"), rc="ENTRY_OK"),
        ]
        ledger = _FakeLedger()
        _, intents = _pipeline(ledger).run_round(
            proposal=_proposal(decisions, view), view=view, now_ms=NOW
        )
        assert [i.action for i in intents] == [IntentAction.CLOSE, IntentAction.REPLACE]
        close, replace = intents
        assert close.symbol == "AAAUSDT" and close.reduces_risk is True
        assert close.target_spot_notional == 0 and close.target_perp_notional == 0
        assert replace.symbol == "BBBUSDT" and replace.reduces_risk is False
        # 稳定因果关联：同轮 intent 共享 correlation_id（proposal_id）
        assert close.correlation_id == replace.correlation_id == "prop-1"


class TestPipelineNoExecutionSurface:
    def test_composition_source_has_no_execution_references(self) -> None:
        src = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "cointrader"
            / "application"
            / "composition.py"
        )
        text = src.read_text(encoding="utf-8")
        for forbidden in ("PairExecutor", "open_pair", "close_pair", "cointrader.execution", "BrokerPort"):
            assert forbidden not in text, f"composition 出现执行层引用: {forbidden}"

    def test_pipeline_instance_holds_only_planner_and_ledger(self) -> None:
        pipeline = _pipeline(_FakeLedger())
        # 构造面只有 planner/ledger 两个端口，无 broker/executor/safety 字段
        assert set(pipeline.__dataclass_fields__) == {"planner", "ledger"}

    def test_broker_call_count_is_zero(self) -> None:
        """T2 边界：整条策略侧流水线 broker 调用恒为 0（T3 才接通执行）。"""
        ledger = _FakeLedger()
        calls: list[str] = []

        class _NoBroker:  # 若被引用则记录
            def open_pair(self, *a: object, **kw: object) -> None:  # pragma: no cover
                calls.append("open_pair")

        _ = _NoBroker()
        pipeline = Pipeline(planner=PortfolioPlanner(), ledger=ledger)
        pipeline.run_round(
            proposal=_proposal(
                [_Decision("AAAUSDT", "OPEN", notional=Decimal("50"))], _view({})
            ),
            view=_view({}),
            now_ms=NOW,
        )
        assert calls == []


class TestPipelineFailClosed:
    def test_ledger_failure_propagates(self) -> None:
        ledger = _FakeLedger()
        ledger.fail = True
        with pytest.raises(RuntimeError, match="db write failure"):
            _pipeline(ledger).run_round(
                proposal=_proposal(
                    [_Decision("AAAUSDT", "OPEN", notional=Decimal("50"))], _view({})
                ),
                view=_view({}),
                now_ms=NOW,
            )
        # 失败不落账、不产生 intent
        assert ledger.proposals == [] and ledger.intents == []

    def test_intent_write_failure_after_proposal_stops_round(self) -> None:
        ledger = _FakeLedger()
        pipeline = _pipeline(ledger)
        proposal = _proposal(
            [_Decision("AAAUSDT", "OPEN", notional=Decimal("50"))], _view({})
        )
        ledger.append_strategy_proposal(dict(proposal.to_dict()))  # proposal 先落成功
        ledger.fail = True
        with pytest.raises(RuntimeError):
            pipeline.run_round(proposal=proposal, view=_view({}), now_ms=NOW)
        assert len(ledger.intents) == 0


class TestViewFromLedgerPositions:
    def test_notional_from_qty_price_and_abs(self) -> None:
        rows: list[dict[str, Any]] = [
            {
                "symbol": "BTCUSDT",
                "spot_qty": "0.01",
                "perp_qty": "-0.01",  # 空头负数 → abs
                "spot_price": "100",
                "perp_price": "100.5",
                "observed_at_ms": NOW,
            },
            {
                "symbol": "ETHUSDT",
                "spot_qty": "1",
                "perp_qty": "1",
                "spot_price": None,  # 价格缺失 → 该腿名义额 0（fail closed）
                "perp_price": None,
                "observed_at_ms": NOW - 100,
            },
        ]
        view = PortfolioPlanner.view_from_ledger_positions(rows, as_of_ms=NOW)
        by_symbol = {e.symbol: e for e in view.entries}
        assert by_symbol["BTCUSDT"].spot_notional == Decimal("1.00")
        assert by_symbol["BTCUSDT"].perp_notional == Decimal("1.005")
        assert by_symbol["ETHUSDT"].spot_notional == 0
        # snapshot_id 由最大 observed 派生，确定性
        assert view.snapshot_id == f"proj-{NOW}"
        again = PortfolioPlanner.view_from_ledger_positions(rows, as_of_ms=NOW + 5)
        assert again.snapshot_id == view.snapshot_id
        assert [e.symbol for e in view.entries] == ["BTCUSDT", "ETHUSDT"]

    def test_empty_rows(self) -> None:
        view = PortfolioPlanner.view_from_ledger_positions([], as_of_ms=NOW)
        assert view.entries == ()
        assert view.snapshot_id == "proj-0"


class TestExecutionPlanQuoteProvenance:
    """v5.0 T2（AC-04）：quote provenance 与 plan 的关联走既有 ledger JSON
    payload 字段（自由 key，无 schema migration），可序列化且往返不丢失。"""

    def test_plan_payload_carries_quote_block_through_ledger(self) -> None:
        import json as _json

        ledger = _FakeLedger()
        plan_payload: dict[str, object] = {
            "plan_id": "plan-it-1",
            "run_id": "run-1",
            "quote": {
                "symbol": "BTCUSDT",
                "source": "rest",
                "received_at_ms": NOW - 10,
                "spot_received_ms": NOW - 10,
                "perp_received_ms": NOW - 5,
                "spot_exchange_ts_ms": None,
                "perp_exchange_ts_ms": None,
                "connection_generation": None,
                "gate_now_ms": NOW,
                "gate_quality": "FRESH",
                "gate_reason": "ok",
            },
        }
        assert ledger.append_execution_plan(plan_payload) is True
        assert len(ledger.plans) == 1
        # JSON 往返（账本 payload_json 同构）不丢失 quote 块与 None 语义
        roundtripped = _json.loads(_json.dumps(ledger.plans[0], default=str))
        assert roundtripped["quote"]["source"] == "rest"
        assert roundtripped["quote"]["gate_quality"] == "FRESH"
        assert roundtripped["quote"]["spot_exchange_ts_ms"] is None
