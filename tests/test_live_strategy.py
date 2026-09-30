"""实时策略开仓判断测试（开发文档 §7.2 判断顺序 / §7.3 去重）。

覆盖：门槛通过/拒绝、边界值、数据窗口、流动性、排除列表、过期数据、
未来数据防护、报价新鲜度、去重（held/submitted/max positions/account/reconcile）。
"""

from __future__ import annotations

from decimal import Decimal

from cointrader.live.decisions import DecisionKind, ReasonCode
from cointrader.live.strategy import HeldPosition
from conftest import FakeStrategyData
from live_helpers import (
    NOW,
    NOW_MS,
    LiveFakeData,
    make_context,
    make_live_config,
    make_live_rates,
    make_quote,
    make_strategy,
)

SYMBOL = "BTCUSDT"
RATE_OK = "0.0005"  # 年化 0.0005*1095 ≈ 0.548 > 0.30


def _data() -> FakeStrategyData:
    return LiveFakeData({SYMBOL: make_live_rates(20, RATE_OK)})


def _open_decision(tmp_path, data: FakeStrategyData, **ctx_kw):
    strat, _store = make_strategy(tmp_path, data)
    strat.refresh_candidates()
    ctx_kw.setdefault("quotes", {SYMBOL: make_quote()})
    ctx = make_context(**ctx_kw)
    decisions = strat.evaluate(ctx)
    assert len(decisions) == 1
    return decisions[0]


class TestEntry:
    def test_all_conditions_pass_opens(self, tmp_path):
        decision = _open_decision(tmp_path, _data())
        assert decision.decision_kind is DecisionKind.OPEN
        assert decision.allowed
        assert decision.reason_code is ReasonCode.ENTRY_OK
        # canary 上限截断：10000 × 0.3 请求 → 10（execution.canary_notional）
        assert decision.requested_notional == Decimal("10")
        assert decision.trailing_annualized is not None
        assert float(decision.trailing_annualized) > 0.30

    def test_trailing_below_threshold_skips(self, tmp_path):
        data = LiveFakeData({SYMBOL: make_live_rates(20, "0.0001")})  # 年化 0.1095
        decision = _open_decision(tmp_path, data)
        assert decision.decision_kind is DecisionKind.SKIP
        assert decision.reason_code is ReasonCode.TRAILING_RATE_BELOW_THRESHOLD

    def test_trailing_at_boundary_passes(self, tmp_path):
        # 0.000274 × 1095 = 0.30003 ≥ 0.30
        data = LiveFakeData({SYMBOL: make_live_rates(20, "0.000274")})
        decision = _open_decision(tmp_path, data)
        assert decision.decision_kind is DecisionKind.OPEN

    def test_trailing_just_below_boundary_skips(self, tmp_path):
        # 0.0002739 × 1095 = 0.2999157 < 0.30
        data = LiveFakeData({SYMBOL: make_live_rates(20, "0.0002739")})
        decision = _open_decision(tmp_path, data)
        assert decision.reason_code is ReasonCode.TRAILING_RATE_BELOW_THRESHOLD

    def test_consecutive_positive_too_short(self, tmp_path):
        # 尾部 4 期全正（trailing 达标），但往回数只有 2 个滑动均值为正
        tail_start = NOW_MS - 30 * 60 * 1000 - 19 * 8 * 3600 * 1000  # 末期结算 = NOW-30min
        values = ["-0.001"] * 16 + [RATE_OK] * 4
        rates = [(tail_start + i * 8 * 3600 * 1000, Decimal(v), Decimal("100"))
                 for i, v in enumerate(values)]
        data = LiveFakeData({SYMBOL: rates})
        decision = _open_decision(tmp_path, data)
        assert decision.decision_kind is DecisionKind.SKIP
        assert decision.reason_code is ReasonCode.CONSECUTIVE_POSITIVE_TOO_SHORT
        assert decision.consecutive_positive_periods == 2

    def test_insufficient_history(self, tmp_path):
        data = LiveFakeData({SYMBOL: make_live_rates(3, RATE_OK)})
        decision = _open_decision(tmp_path, data)
        assert decision.reason_code is ReasonCode.INSUFFICIENT_HISTORY

    def test_low_liquidity_skips(self, tmp_path):
        data = LiveFakeData(
            {SYMBOL: make_live_rates(20, RATE_OK)},
            volumes={SYMBOL: Decimal("500000")},
        )
        decision = _open_decision(tmp_path, data)
        assert decision.reason_code is ReasonCode.LOW_LIQUIDITY

    def test_excluded_asset_skips(self, tmp_path):
        cfg = make_live_config(live_symbols=("USDCUSDT",))
        data = LiveFakeData({"USDCUSDT": make_live_rates(20, RATE_OK)})
        strat, _ = make_strategy(tmp_path, data, config=cfg)
        strat.refresh_candidates()
        decision = strat.can_open("USDCUSDT", make_context())
        assert decision.reason_code is ReasonCode.EXCLUDED_ASSET

    def test_stale_candidate_data_skips(self, tmp_path):
        clock = {"t": NOW}
        cfg = make_live_config()
        data = LiveFakeData({SYMBOL: make_live_rates(20, RATE_OK)})
        strat, _ = make_strategy(tmp_path, data, config=cfg, now_fn=lambda: clock["t"])
        strat.refresh_candidates()
        # 超龄上限 = 结算周期 8h + max_candidate_data_age 30min；超过 9h 才拒绝
        clock["t"] = NOW + 9 * 3600
        decision = strat.can_open(SYMBOL, make_context())
        assert decision.reason_code is ReasonCode.STALE_DATA

    def test_candidate_refresh_failure_keeps_old_cache_and_flags_error(self, tmp_path):
        clock = {"t": NOW}
        data = LiveFakeData({SYMBOL: make_live_rates(20, RATE_OK)})
        strat, _ = make_strategy(tmp_path, data, now_fn=lambda: clock["t"])
        strat.refresh_candidates()
        # 刷新失败 → 保留旧缓存但标记 error，本轮拒绝
        data.set_rates(SYMBOL, [])

        class _Boom(FakeStrategyData):
            def funding_rates(self, symbol: str, periods: int, *, end_ms: int | None = None):
                raise RuntimeError("api down")

        strat.data = _Boom({})  # type: ignore[assignment]
        clock["t"] = NOW + 8 * 3600 + 60  # 跨过结算周期 → 触发重刷 → 失败
        strat.refresh_candidates()
        decision = strat.can_open(SYMBOL, make_context(now_ms=clock["t"]))
        assert decision.reason_code is ReasonCode.STALE_DATA
        assert "api down" in decision.reason_text

    def test_no_future_data_used(self, tmp_path):
        """未来时间戳（未结算）的费率必须被数据源丢弃，不影响决策。"""
        base = make_live_rates(20, RATE_OK)
        future = [
            (NOW_MS + 8 * 3600 * 1000, Decimal("-0.01"), Decimal("100"))
        ]
        data_no_future = LiveFakeData({SYMBOL: base})
        data_with_future = LiveFakeData({SYMBOL: base + future})
        d1 = _open_decision(tmp_path, data_no_future)
        d2 = _open_decision(tmp_path, data_with_future)
        assert d1.decision_kind is DecisionKind.OPEN
        assert d2.decision_kind is DecisionKind.OPEN, "未来（未结算）费率不得改变当前决策"
        assert d2.trailing_annualized == d1.trailing_annualized


class TestEntryPreconditions:
    def test_reconciliation_blocked(self, tmp_path):
        ctx = make_context(quotes={SYMBOL: make_quote()})
        ctx.reconcile_ok = False
        strat, _ = make_strategy(tmp_path, _data())
        strat.refresh_candidates()
        assert strat.can_open(SYMBOL, ctx).reason_code is ReasonCode.RECONCILIATION_BLOCKED

    def test_account_unknown_blocks(self, tmp_path):
        ctx = make_context(quotes={SYMBOL: make_quote()}, total_capital=Decimal("0"))
        strat, _ = make_strategy(tmp_path, _data())
        strat.refresh_candidates()
        assert strat.can_open(SYMBOL, ctx).reason_code is ReasonCode.ACCOUNT_STATE_UNKNOWN
        ctx2 = make_context(quotes={SYMBOL: make_quote()})
        ctx2.account_ok = False
        assert strat.can_open(SYMBOL, ctx2).reason_code is ReasonCode.ACCOUNT_STATE_UNKNOWN

    def test_basis_discount_blocks_open(self, tmp_path):
        # 永续贴水 -0.6% < -0.5% 容差 → 开空不利
        ctx = make_context(quotes={SYMBOL: make_quote(spot="100", perp="99.4")})
        strat, _ = make_strategy(tmp_path, _data())
        strat.refresh_candidates()
        assert strat.can_open(SYMBOL, ctx).reason_code is ReasonCode.BASIS_DISCOUNT


class TestQuoteFreshness:
    def test_no_quote_pending(self, tmp_path):
        """前置通过但无报价 → PENDING_QUOTE 中间态（service 按需拉报价后定案）。"""
        decision = _open_decision(tmp_path, _data(), quotes={})
        assert decision.decision_kind is DecisionKind.PENDING_QUOTE
        assert decision.reason_code is ReasonCode.PENDING_QUOTE
        assert not decision.allowed

    def test_complete_open_after_pending(self, tmp_path):
        strat, _ = make_strategy(tmp_path, _data())
        strat.refresh_candidates()
        ctx = make_context()  # 无报价
        pending = strat.can_open(SYMBOL, ctx)
        assert pending.decision_kind is DecisionKind.PENDING_QUOTE
        decision = strat.complete_open(SYMBOL, ctx, make_quote())
        assert decision.decision_kind is DecisionKind.OPEN
        assert decision.reason_code is ReasonCode.ENTRY_OK

    def test_complete_open_quote_failed(self, tmp_path):
        strat, _ = make_strategy(tmp_path, _data())
        strat.refresh_candidates()
        ctx = make_context()
        decision = strat.complete_open(SYMBOL, ctx, None)
        assert decision.decision_kind is DecisionKind.SKIP
        assert decision.reason_code is ReasonCode.STALE_QUOTE

    def test_stale_quote_skips(self, tmp_path):
        old = make_quote(ts=NOW - 30)  # 30s 前 > 5s 上限
        decision = _open_decision(tmp_path, _data(), quotes={SYMBOL: old})
        assert decision.reason_code is ReasonCode.STALE_QUOTE

    def test_future_quote_skips(self, tmp_path):
        future = make_quote(ts=NOW + 60)
        decision = _open_decision(tmp_path, _data(), quotes={SYMBOL: future})
        assert decision.reason_code is ReasonCode.STALE_QUOTE

    def test_slow_tick_fresh_quote_not_rejected(self, tmp_path):
        """慢 tick：tick 开场 ctx.now 之后 60s 才拿到报价（限流/403 重试）。

        报价相对**定案时刻**（gate_now_ms）是新鲜的 → 不得误判"未来时间戳"；
        回归 VPS/本地实测：top3 候选每轮 STALE_QUOTE（报价年龄 -54s）永不     开仓。
        """
        strat, _ = make_strategy(tmp_path, _data())
        strat.refresh_candidates()
        ctx = make_context()
        late = make_quote(ts=NOW + 60)  # 相对 ctx.now（tick 开场）是"未来"
        # 闸门时刻 = 报价刚接收后的当前时间 → 放行
        decision = strat.complete_open(SYMBOL, ctx, late, gate_now_ms=late.ts_ms)
        assert decision.decision_kind is DecisionKind.OPEN
        assert decision.reason_code is ReasonCode.ENTRY_OK
        # 闸门时刻仍早于报价接收时间（真时钟回拨）→ 仍拒
        early = make_quote(ts=NOW + 60)
        decision2 = strat.complete_open(
            SYMBOL, ctx, early, gate_now_ms=int(NOW * 1000) - 1000
        )
        assert decision2.decision_kind is DecisionKind.SKIP
        assert decision2.reason_code is ReasonCode.STALE_QUOTE


class TestDedup:
    def test_already_held(self, tmp_path):
        ctx = make_context(
            held={SYMBOL: HeldPosition(SYMBOL, Decimal("0.01"), Decimal("-0.01"))},
            quotes={SYMBOL: make_quote()},
        )
        strat, _ = make_strategy(tmp_path, _data())
        strat.refresh_candidates()
        decisions = strat.evaluate(ctx)
        # held symbol → 走退出评估（HOLD），不再有开仓决策
        assert all(d.symbol == SYMBOL for d in decisions)
        assert all(d.decision_kind is not DecisionKind.OPEN for d in decisions)
        # can_open 直接调用 → ALREADY_HELD
        assert strat.can_open(SYMBOL, ctx).reason_code is ReasonCode.ALREADY_HELD

    def test_submitted_this_run(self, tmp_path):
        ctx = make_context(quotes={SYMBOL: make_quote()})
        ctx.submitted_this_run = frozenset({SYMBOL})
        strat, _ = make_strategy(tmp_path, _data())
        strat.refresh_candidates()
        assert strat.can_open(SYMBOL, ctx).reason_code is ReasonCode.ALREADY_SUBMITTED

    def test_max_positions(self, tmp_path):
        cfg = make_live_config(live_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
                               selection_overrides={"max_positions": 2})
        data = LiveFakeData({s: make_live_rates(20, RATE_OK)
                                 for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")})
        strat, _ = make_strategy(tmp_path, data, config=cfg)
        strat.refresh_candidates()
        held = {
            "BTCUSDT": HeldPosition("BTCUSDT", Decimal("0.01"), Decimal("-0.01")),
            "ETHUSDT": HeldPosition("ETHUSDT", Decimal("0.01"), Decimal("-0.01")),
        }
        ctx = make_context(held=held, quotes={s: make_quote() for s in held})
        decision = strat.can_open("SOLUSDT", ctx)
        assert decision.reason_code is ReasonCode.MAX_POSITIONS


class TestQuoteProvenanceGateCompat:
    """v5.0 T2（AC-04）：共享测试 fixture 与订单前 gate 的兼容性。

    策略层数学不检查 source/symbol provenance（由 runner 的 gate 负责）；
    这里锁定 fixture 默认 provenance（source="fake"）能通过 gate，避免
    fixture 回归时 runner 开仓路径被 fail-closed 误拒。
    """

    def test_fixture_quote_passes_entry_gate(self):
        from cointrader.domain.market import DataQuality
        from cointrader.live.strategy import evaluate_quote_gate

        verdict = evaluate_quote_gate(
            make_quote(),
            intent_symbol=SYMBOL,
            now_ms=NOW_MS,
            max_age_ms=5_000,
            max_skew_ms=500,
        )
        assert verdict.quality is DataQuality.FRESH

    def test_fixture_quote_symbol_follows_intent(self):
        from cointrader.domain.market import DataQuality
        from cointrader.live.strategy import evaluate_quote_gate

        verdict = evaluate_quote_gate(
            make_quote(symbol="ETHUSDT"),
            intent_symbol="ETHUSDT",
            now_ms=NOW_MS,
            max_age_ms=5_000,
            max_skew_ms=500,
        )
        assert verdict.quality is DataQuality.FRESH
        # 同一 fixture 报价与不匹配 intent → 拒绝
        wrong = evaluate_quote_gate(
            make_quote(symbol="ETHUSDT"),
            intent_symbol=SYMBOL,
            now_ms=NOW_MS,
            max_age_ms=5_000,
            max_skew_ms=500,
        )
        assert wrong.quality is not DataQuality.FRESH

    def test_unregistered_source_fixture_quote_rejected(self):
        from cointrader.domain.market import DataQuality
        from cointrader.live.strategy import evaluate_quote_gate

        # 未知来源（空串）不得 FRESH：这是 fixture 必须带 source="fake" 的原因
        verdict = evaluate_quote_gate(
            make_quote(source=""),
            intent_symbol=SYMBOL,
            now_ms=NOW_MS,
            max_age_ms=5_000,
            max_skew_ms=500,
        )
        assert verdict.quality is DataQuality.INCOMPLETE
