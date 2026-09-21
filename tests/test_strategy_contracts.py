"""策略纯度与契约测试（实施计划书 3.0 T2，AC-02）。

覆盖：

- 纯 evaluator 相同输入与同一时钟输出确定（无隐藏状态）；
- 策略只输出目标名义额/证据，无订单参数、无网络、无账本写入能力；
- 中性 kind/reason 字符串与 ``live.decisions`` 常量逐值等价（单一事实源由本测试强制）;
- 入场门槛/去重/epoch 闸门/退出/换仓/槽位排序的最小行为面；
- ``to_strategy_proposal`` 目标组合推导与可序列化性。
"""

from __future__ import annotations

from dataclasses import fields
from decimal import Decimal

from cointrader.domain.portfolio import PortfolioView
from cointrader.live.decisions import DecisionKind, ReasonCode
from cointrader.strategy.adapter import to_strategy_proposal
from cointrader.strategy.funding_carry import (
    CandidateInput,
    CarryEvaluation,
    EvalContext,
    EvalKind,
    FundingCarryEvaluator,
    HeldInput,
    QuoteInput,
    Reason,
)
from live_helpers import NOW, NOW_MS, make_live_config

SYMBOL = "BTCUSDT"
RATE_OK = "0.0005"  # 年化 0.0005*1095 ≈ 0.548 > 0.30
PERIOD_MS = 8 * 3600 * 1000


def _cand(symbol: str = SYMBOL, rate: str = RATE_OK, n: int = 20, **kw: object) -> CandidateInput:
    end = NOW_MS - 30 * 60 * 1000  # 末期结算已发生且未触发结算滞后门
    ts = tuple(end - (n - 1 - i) * PERIOD_MS for i in range(n))
    base: dict[str, object] = {
        "symbol": symbol,
        "rates": tuple(Decimal(rate) for _ in range(n)),
        "mark_prices": tuple(Decimal("100") for _ in range(n)),
        "timestamps": ts,
        "interval_hours": 8,
        "volume_3d_avg": Decimal("10000000"),
        "refreshed_ts_ms": NOW_MS,
    }
    base.update(kw)
    return CandidateInput(**base)  # type: ignore[arg-type]


def _quote(ts_ms: int = NOW_MS, spot: str = "100", perp: str = "100") -> QuoteInput:
    return QuoteInput(
        spot_price=Decimal(spot), perp_price=Decimal(perp), ts_ms=ts_ms
    )


class _NoDedup:
    def active_pair(self, symbol: str) -> bool:  # noqa: ARG002
        return False

    def has_open_intent(self, symbol: str) -> bool:  # noqa: ARG002
        return False

    def has_open_order(self, symbol: str) -> bool:  # noqa: ARG002
        return False


def _ctx(**kw: object) -> EvalContext:
    base: dict[str, object] = {
        "now_ms": NOW_MS,
        "total_capital": Decimal("10000"),
        "held": {},
        "quotes": {},
    }
    base.update(kw)
    return EvalContext(**base)  # type: ignore[arg-type]


def _evaluator(**kw: object) -> FundingCarryEvaluator:
    base: dict[str, object] = {
        "config": make_live_config(),
        "strategy_version": "pure-test",
        "config_hash": "cfg-pure",
        "now_fn": lambda: NOW,
    }
    base.update(kw)
    return FundingCarryEvaluator(**base)  # type: ignore[arg-type]


class TestPurityAndDeterminism:
    def test_same_inputs_same_clock_same_output(self) -> None:
        cands = {SYMBOL: _cand()}
        ctx = _ctx(quotes={SYMBOL: _quote()})
        first = _evaluator().evaluate(ctx, (SYMBOL,), cands, _NoDedup())
        second = _evaluator().evaluate(ctx, (SYMBOL,), cands, _NoDedup())
        assert first == second

    def test_no_hidden_state_between_calls(self) -> None:
        cands = {SYMBOL: _cand()}
        ctx = _ctx(quotes={SYMBOL: _quote()})
        evaluator = _evaluator()
        assert evaluator.evaluate(ctx, (SYMBOL,), cands, _NoDedup()) == evaluator.evaluate(
            ctx, (SYMBOL,), cands, _NoDedup()
        )

    def test_evaluation_carries_no_order_capability(self) -> None:
        """策略输出不得含订单参数（订单参数是 execution 层职责）。"""
        names = {f.name for f in fields(CarryEvaluation)}
        forbidden = {
            "client_order_id",
            "order_type",
            "order_side",
            "side",
            "quantity",
            "price",
            "leverage",
        }
        assert names.isdisjoint(forbidden)

    def test_evaluator_holds_no_io_handles(self) -> None:
        evaluator = _evaluator()
        attrs = set(vars(evaluator))
        assert attrs.isdisjoint({"store", "client", "broker", "data"})


class TestLegacyParity:
    def test_kind_values_match_decision_kind(self) -> None:
        for name in ("OPEN", "HOLD", "EXIT", "REPLACE", "SKIP", "PENDING_QUOTE"):
            assert getattr(EvalKind, name) == getattr(DecisionKind, name)

    def test_reason_values_match_reason_code(self) -> None:
        for name, value in vars(Reason).items():
            if name.startswith("_"):
                continue
            assert getattr(ReasonCode, name) == value, f"Reason.{name} 与 ReasonCode 不一致"


class TestEntryGate:
    def test_open_with_canary_truncation(self) -> None:
        ctx = _ctx(quotes={SYMBOL: _quote()})
        result = _evaluator().evaluate(ctx, (SYMBOL,), {SYMBOL: _cand()}, _NoDedup())
        assert len(result) == 1
        evaluation = result[0]
        assert evaluation.kind == EvalKind.OPEN
        assert evaluation.allowed
        assert evaluation.reason_code == Reason.ENTRY_OK
        # 10000 × 0.3 = 3000 请求 → canary_notional=10 截断
        assert evaluation.requested_notional == Decimal("10")

    def test_trailing_below_threshold(self) -> None:
        ctx = _ctx(quotes={SYMBOL: _quote()})
        result = _evaluator().can_open(SYMBOL, ctx, {SYMBOL: _cand(rate="0.0001")}, _NoDedup())
        assert result.kind == EvalKind.SKIP
        assert result.reason_code == Reason.TRAILING_RATE_BELOW_THRESHOLD

    def test_no_quote_pending(self) -> None:
        ctx = _ctx(quotes={})
        result = _evaluator().can_open(SYMBOL, ctx, {SYMBOL: _cand()}, _NoDedup())
        assert result.kind == EvalKind.PENDING_QUOTE
        # complete_open 获取失败 = STALE_QUOTE
        failed = _evaluator().complete_open(SYMBOL, ctx, {SYMBOL: _cand()}, None)
        assert failed.reason_code == Reason.STALE_QUOTE
        # 获取成功 → OPEN
        ok = _evaluator().complete_open(SYMBOL, ctx, {SYMBOL: _cand()}, _quote())
        assert ok.kind == EvalKind.OPEN

    def test_stale_quote(self) -> None:
        ctx = _ctx(quotes={SYMBOL: _quote(ts_ms=NOW_MS - 60 * 1000)})
        result = _evaluator().can_open(SYMBOL, ctx, {SYMBOL: _cand()}, _NoDedup())
        assert result.reason_code == Reason.STALE_QUOTE

    def test_quote_skew_rejected(self) -> None:
        ctx = _ctx(
            quotes={
                SYMBOL: QuoteInput(
                    spot_price=Decimal("100"),
                    perp_price=Decimal("100"),
                    ts_ms=NOW_MS,
                    spot_ts_ms=NOW_MS - 2000,
                    perp_ts_ms=NOW_MS,
                )
            }
        )
        result = _evaluator().can_open(SYMBOL, ctx, {SYMBOL: _cand()}, _NoDedup())
        assert result.reason_code == Reason.STALE_QUOTE

    def test_reconcile_and_account_gates(self) -> None:
        cands = {SYMBOL: _cand()}
        blocked = _evaluator().can_open(
            SYMBOL, _ctx(reconcile_ok=False, quotes={SYMBOL: _quote()}), cands, _NoDedup()
        )
        assert blocked.reason_code == Reason.RECONCILIATION_BLOCKED
        unknown = _evaluator().can_open(
            SYMBOL, _ctx(account_ok=False, quotes={SYMBOL: _quote()}), cands, _NoDedup()
        )
        assert unknown.reason_code == Reason.ACCOUNT_STATE_UNKNOWN

    def test_dedup_probe_blocks(self) -> None:
        class _ActivePair(_NoDedup):
            def active_pair(self, symbol: str) -> bool:  # noqa: ARG002
                return True

        result = _evaluator().can_open(
            SYMBOL, _ctx(quotes={SYMBOL: _quote()}), {SYMBOL: _cand()}, _ActivePair()
        )
        assert result.reason_code == Reason.ACTIVE_ORDER

    def test_excluded_symbol_in_epoch(self) -> None:
        ctx = _ctx(
            quotes={SYMBOL: _quote()},
            epoch_active=True,
            epoch_id="ep1",
            decision_cutoff_ms=NOW_MS,
            epoch_excluded={SYMBOL: "no spot leg"},
        )
        result = _evaluator().can_open(SYMBOL, ctx, {SYMBOL: _cand()}, _NoDedup())
        assert result.reason_code == Reason.EXCLUDED_ASSET

    def test_no_ready_epoch_blocks_new_risk_but_not_exit(self) -> None:
        held = {
            SYMBOL: HeldInput(
                symbol=SYMBOL, spot_qty=Decimal("0.01"), perp_qty=Decimal("-0.01")
            )
        }
        cands = {
            SYMBOL: _cand(rate="-0.0005"),
            "ETHUSDT": _cand(symbol="ETHUSDT"),
        }
        ctx = _ctx(
            held=held,
            epoch_active=True,
            epoch_id=None,
            expected_symbols=("ETHUSDT",),
        )
        result = _evaluator().evaluate(ctx, (), cands, _NoDedup())
        by_symbol = {e.symbol: e for e in result}
        # 未持仓候选：禁止新开仓
        assert by_symbol["ETHUSDT"].reason_code == Reason.MARKET_DATA_NOT_READY
        # 持仓（风险降低）：负均值退出仍放行
        assert by_symbol[SYMBOL].kind == EvalKind.EXIT
        assert by_symbol[SYMBOL].reason_code == Reason.NEGATIVE_EXIT_AVG


class TestExitAndReplacement:
    def test_negative_exit_avg(self) -> None:
        held = {
            SYMBOL: HeldInput(
                symbol=SYMBOL, spot_qty=Decimal("0.01"), perp_qty=Decimal("-0.01")
            )
        }
        ctx = _ctx(held=held)
        result = _evaluator().evaluate(ctx, (), {SYMBOL: _cand(rate="-0.0005")}, _NoDedup())
        assert result[0].kind == EvalKind.EXIT
        assert result[0].reason_code == Reason.NEGATIVE_EXIT_AVG

    def test_max_holding_exit(self) -> None:
        held = {
            SYMBOL: HeldInput(
                symbol=SYMBOL,
                spot_qty=Decimal("0.01"),
                perp_qty=Decimal("-0.01"),
                opened_ms=NOW_MS - 10 * PERIOD_MS,  # max_holding_periods=10
            )
        }
        ctx = _ctx(held=held)
        result = _evaluator().evaluate(ctx, (), {SYMBOL: _cand()}, _NoDedup())
        assert result[0].reason_code == Reason.MAX_HOLDING

    def test_replacement_when_better_candidate(self) -> None:
        held = {
            SYMBOL: HeldInput(
                symbol=SYMBOL,
                spot_qty=Decimal("0.01"),
                perp_qty=Decimal("-0.01"),
                opened_ms=NOW_MS - 70 * PERIOD_MS,  # > 60 期，eligible
            )
        }
        cands = {
            SYMBOL: _cand(rate="0.0002"),  # 持仓 trailing ≈ 0.219
            "ETHUSDT": _cand(symbol="ETHUSDT", rate="0.0005"),  # trailing ≈ 0.548
        }
        ctx = _ctx(held=held)
        evaluator = _evaluator(config=make_live_config(exit_overrides={"max_holding_periods": 200}))
        result = evaluator.evaluate(ctx, ("ETHUSDT",), cands, _NoDedup())
        by_symbol = {e.symbol: e for e in result}
        assert by_symbol[SYMBOL].kind == EvalKind.REPLACE
        assert by_symbol[SYMBOL].replacement_symbol == "ETHUSDT"

    def test_ranked_out_slot_allocation(self) -> None:
        """max_positions=2、已持仓 1 → 只有 1 个新槽位。

        与 legacy 行为一致：槽位分配作用于 PENDING_QUOTE 候选（ctx 无报价）；
        落选者 SKIP RANKED_OUT，入选者保持 PENDING_QUOTE 待 service 取报价。
        """
        held = {
            "ETHUSDT": HeldInput(
                symbol="ETHUSDT", spot_qty=Decimal("0.01"), perp_qty=Decimal("-0.01")
            )
        }
        cands = {
            "AAAUSDT": _cand(symbol="AAAUSDT", rate="0.0005"),
            "BBBUSDT": _cand(symbol="BBBUSDT", rate="0.0003"),  # trailing ≈ 0.328
        }
        ctx = _ctx(held=held, quotes={})
        result = _evaluator().evaluate(ctx, ("AAAUSDT", "BBBUSDT"), cands, _NoDedup())
        by_symbol = {e.symbol: e for e in result}
        assert by_symbol["AAAUSDT"].kind == EvalKind.PENDING_QUOTE
        assert by_symbol["BBBUSDT"].kind == EvalKind.SKIP
        assert by_symbol["BBBUSDT"].reason_code == Reason.RANKED_OUT


class TestProposal:
    def test_to_strategy_proposal_targets(self) -> None:
        from cointrader.domain.portfolio import CurrentPosition

        view = PortfolioView(
            snapshot_id="s1",
            as_of_ms=NOW_MS,
            entries=(
                CurrentPosition(
                    symbol=SYMBOL,
                    spot_notional=Decimal("100"),
                    perp_notional=Decimal("100"),
                    updated_at_ms=NOW_MS,
                ),
                CurrentPosition(
                    symbol="ETHUSDT",
                    spot_notional=Decimal("50"),
                    perp_notional=Decimal("50"),
                    updated_at_ms=NOW_MS,
                ),
            ),
        )
        evaluations = [
            CarryEvaluation(
                symbol=SYMBOL,
                kind=EvalKind.EXIT,
                allowed=True,
                reason_code=Reason.NEGATIVE_EXIT_AVG,
                reason_text="退出",
            ),
            CarryEvaluation(
                symbol="SOLUSDT",
                kind=EvalKind.OPEN,
                allowed=True,
                reason_code=Reason.ENTRY_OK,
                reason_text="开仓",
                requested_notional=Decimal("10"),
            ),
        ]
        proposal = to_strategy_proposal(
            evaluations,
            view,
            proposal_id="p1",
            snapshot_id="s1",
            decision_cutoff_ms=NOW_MS,
            strategy_version="pure-test",
            config_hash="cfg-pure",
            valid_until_ms=NOW_MS + 60_000,
        )
        # 目标 = 当前(ETH 持有) − 退出(BTC) + 新开(SOL 10)
        assert proposal.target.get(SYMBOL) is None
        assert proposal.target.get("ETHUSDT") is not None
        sol = proposal.target.get("SOLUSDT")
        assert sol is not None and sol.spot_notional == Decimal("10")
        assert proposal.is_expired(NOW_MS + 60_001) is True
        # 可序列化（round-trip 保真）
        import json

        from cointrader.domain.strategy import StrategyProposal

        restored = StrategyProposal.from_dict(json.loads(json.dumps(proposal.to_dict())))
        assert restored == proposal
