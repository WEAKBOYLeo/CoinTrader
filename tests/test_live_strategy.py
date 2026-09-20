"""实时策略开仓判断测试（开发文档 §7.2 判断顺序 / §7.3 去重）。

覆盖：门槛通过/拒绝、边界值、数据窗口、流动性、排除列表、过期数据、
未来数据防护、报价新鲜度、去重（held/submitted/max positions/account/reconcile）。
"""

from __future__ import annotations

import time
from decimal import Decimal

from cointrader.live.decisions import DecisionKind, ReasonCode
from cointrader.live.strategy import HeldPosition
from conftest import FakeStrategyData, make_rate_series
from live_helpers import (
    NOW,
    make_context,
    make_live_config,
    make_quote,
    make_strategy,
)

SYMBOL = "BTCUSDT"
RATE_OK = "0.0005"  # 年化 0.0005*1095 ≈ 0.548 > 0.30


def _data() -> FakeStrategyData:
    return FakeStrategyData({SYMBOL: make_rate_series(20, RATE_OK)})


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
        data = FakeStrategyData({SYMBOL: make_rate_series(20, "0.0001")})  # 年化 0.1095
        decision = _open_decision(tmp_path, data)
        assert decision.decision_kind is DecisionKind.SKIP
        assert decision.reason_code is ReasonCode.TRAILING_RATE_BELOW_THRESHOLD

    def test_trailing_at_boundary_passes(self, tmp_path):
        # 0.000274 × 1095 = 0.30003 ≥ 0.30
        data = FakeStrategyData({SYMBOL: make_rate_series(20, "0.000274")})
        decision = _open_decision(tmp_path, data)
        assert decision.decision_kind is DecisionKind.OPEN

    def test_trailing_just_below_boundary_skips(self, tmp_path):
        # 0.0002739 × 1095 = 0.2999157 < 0.30
        data = FakeStrategyData({SYMBOL: make_rate_series(20, "0.0002739")})
        decision = _open_decision(tmp_path, data)
        assert decision.reason_code is ReasonCode.TRAILING_RATE_BELOW_THRESHOLD

    def test_consecutive_positive_too_short(self, tmp_path):
        # 尾部 4 期全正（trailing 达标），但往回数只有 2 个滑动均值为正
        tail_start = int(time.time() * 1000) - 20 * 8 * 3600 * 1000
        values = ["-0.001"] * 16 + [RATE_OK] * 4
        rates = [(tail_start + i * 8 * 3600 * 1000, Decimal(v), Decimal("100"))
                 for i, v in enumerate(values)]
        data = FakeStrategyData({SYMBOL: rates})
        decision = _open_decision(tmp_path, data)
        assert decision.decision_kind is DecisionKind.SKIP
        assert decision.reason_code is ReasonCode.CONSECUTIVE_POSITIVE_TOO_SHORT
        assert decision.consecutive_positive_periods == 2

    def test_insufficient_history(self, tmp_path):
        data = FakeStrategyData({SYMBOL: make_rate_series(3, RATE_OK)})
        decision = _open_decision(tmp_path, data)
        assert decision.reason_code is ReasonCode.INSUFFICIENT_HISTORY

    def test_low_liquidity_skips(self, tmp_path):
        data = FakeStrategyData(
            {SYMBOL: make_rate_series(20, RATE_OK)},
            volumes={SYMBOL: Decimal("500000")},
        )
        decision = _open_decision(tmp_path, data)
        assert decision.reason_code is ReasonCode.LOW_LIQUIDITY

    def test_excluded_asset_skips(self, tmp_path):
        cfg = make_live_config(live_symbols=("USDCUSDT",))
        data = FakeStrategyData({"USDCUSDT": make_rate_series(20, RATE_OK)})
        strat, _ = make_strategy(tmp_path, data, config=cfg)
        strat.refresh_candidates()
        decision = strat.can_open("USDCUSDT", make_context())
        assert decision.reason_code is ReasonCode.EXCLUDED_ASSET

    def test_stale_candidate_data_skips(self, tmp_path):
        clock = {"t": NOW}
        cfg = make_live_config()
        data = FakeStrategyData({SYMBOL: make_rate_series(20, RATE_OK)})
        strat, _ = make_strategy(tmp_path, data, config=cfg, now_fn=lambda: clock["t"])
        strat.refresh_candidates()
        clock["t"] = NOW + 2000  # 超过 max_candidate_data_age_seconds=1800
        decision = strat.can_open(SYMBOL, make_context())
        assert decision.reason_code is ReasonCode.STALE_DATA

    def test_candidate_refresh_failure_keeps_old_cache_and_flags_error(self, tmp_path):
        data = FakeStrategyData({SYMBOL: make_rate_series(20, RATE_OK)})
        strat, _ = make_strategy(tmp_path, data)
        strat.refresh_candidates()
        # 刷新失败 → 保留旧缓存但标记 error，本轮拒绝
        data.set_rates(SYMBOL, [])

        class _Boom(FakeStrategyData):
            def funding_rates(self, symbol: str, periods: int):
                raise RuntimeError("api down")

        strat.data = _Boom({})  # type: ignore[assignment]
        strat.refresh_candidates()
        decision = strat.can_open(SYMBOL, make_context())
        assert decision.reason_code is ReasonCode.STALE_DATA
        assert "api down" in decision.reason_text

    def test_no_future_data_used(self, tmp_path):
        """未来时间戳（未结算）的费率必须被数据源丢弃，不影响决策。"""
        base = make_rate_series(20, RATE_OK)
        future = [
            (int(time.time() * 1000) + 8 * 3600 * 1000, Decimal("-0.01"), Decimal("100"))
        ]
        data_no_future = FakeStrategyData({SYMBOL: base})
        data_with_future = FakeStrategyData({SYMBOL: base + future})
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
        data = FakeStrategyData({s: make_rate_series(20, RATE_OK)
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
