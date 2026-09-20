"""实时策略退出/换仓测试（开发文档 §7.4）。"""

from __future__ import annotations

from decimal import Decimal

from cointrader.live.decisions import DecisionKind, ReasonCode
from conftest import FakeStrategyData  # noqa: F401  # 类型标注用
from live_helpers import (
    NOW_MS,
    LiveFakeData,
    make_context,
    make_held,
    make_live_config,
    make_quote,
    make_strategy,
)

SYMBOL = "BTCUSDT"
RATE_OK = "0.0005"
INTERVAL_MS = 8 * 3600 * 1000


def _rates(*segments: tuple[int, str]) -> list[tuple[int, Decimal, Decimal]]:
    """按 (条数, 费率值) 段拼接连续时间戳的费率序列（全部在过去）。"""
    n = sum(count for count, _ in segments)
    start = NOW_MS - 30 * 60 * 1000 - n * INTERVAL_MS
    out: list[tuple[int, Decimal, Decimal]] = []
    i = 0
    for count, value in segments:
        for _ in range(count):
            out.append((start + i * INTERVAL_MS, Decimal(value), Decimal("100")))
            i += 1
    return out


def _exit_decision(tmp_path, rates, held, *, config=None, quotes=None):
    data = LiveFakeData({SYMBOL: rates},
                            volumes={SYMBOL: Decimal("10000000")})
    if config is not None:
        strat, _ = make_strategy(tmp_path, data, config=config)
    else:
        strat, _ = make_strategy(tmp_path, data)
    strat.refresh_candidates()
    ctx = make_context(
        held={SYMBOL: held},
        quotes=quotes or {SYMBOL: make_quote()},
    )
    decisions = strat.evaluate(ctx)
    assert len(decisions) == 1
    return decisions[0]


class TestExit:
    def test_negative_exit_average_exits(self, tmp_path):
        # 最近 6 期（退出窗口）均值转负
        rates = _rates((14, RATE_OK), (6, "-0.0005"))
        decision = _exit_decision(tmp_path, rates, make_held(age_periods=2))
        assert decision.decision_kind is DecisionKind.EXIT
        assert decision.reason_code is ReasonCode.NEGATIVE_EXIT_AVG
        assert float(decision.exit_average_annualized or 0) < 0

    def test_exit_window_not_full_no_negative_exit(self, tmp_path):
        # 只有 5 期历史 < 退出窗口 6 → 窗口未满不判定负均值（也不触发其他退出）
        rates = _rates((5, "-0.0005"))
        decision = _exit_decision(tmp_path, rates, make_held(age_periods=1))
        assert decision.decision_kind is DecisionKind.HOLD

    def test_max_holding_exits(self, tmp_path):
        # 全正费率（不触发负均值），持仓 10 期 = max_holding_periods=10
        rates = _rates((20, RATE_OK))
        decision = _exit_decision(tmp_path, rates, make_held(age_periods=10))
        assert decision.decision_kind is DecisionKind.EXIT
        assert decision.reason_code is ReasonCode.MAX_HOLDING

    def test_normal_hold(self, tmp_path):
        rates = _rates((20, RATE_OK))
        decision = _exit_decision(tmp_path, rates, make_held(age_periods=2))
        assert decision.decision_kind is DecisionKind.HOLD
        assert decision.reason_code is ReasonCode.HOLD_OK

    def test_exit_takes_priority_over_replacement(self, tmp_path):
        """负均值退出优先于换仓。"""
        cfg = make_live_config(live_symbols=("BTCUSDT", "ETHUSDT"))
        data = LiveFakeData({
            "BTCUSDT": _rates((14, RATE_OK), (6, "-0.0005")),
            "ETHUSDT": _rates((20, "0.0006")),
        })
        strat, _ = make_strategy(tmp_path, data, config=cfg)
        strat.refresh_candidates()
        held = make_held(age_periods=70)
        ctx = make_context(held={"BTCUSDT": held},
                           quotes={"BTCUSDT": make_quote()})
        decision = next(d for d in strat.evaluate(ctx) if d.symbol == SYMBOL)
        assert decision.decision_kind is DecisionKind.EXIT
        assert decision.reason_code is ReasonCode.NEGATIVE_EXIT_AVG


class TestReplacement:
    def _two_symbol_env(self, tmp_path, btc_value: str, eth_value: str):
        # max_holding 调大，避免持仓年龄先触发 MAX_HOLDING 干扰换仓判定
        cfg = make_live_config(live_symbols=("BTCUSDT", "ETHUSDT"),
                               exit_overrides={"max_holding_periods": 300})
        data = LiveFakeData({
            "BTCUSDT": _rates((20, btc_value)),
            "ETHUSDT": _rates((20, eth_value)),
        })
        strat, _ = make_strategy(tmp_path, data, config=cfg)
        strat.refresh_candidates()
        return strat

    def test_replacement_when_better_candidate(self, tmp_path):
        strat = self._two_symbol_env(tmp_path, RATE_OK, "0.0006")
        held = make_held(age_periods=70)
        ctx = make_context(held={"BTCUSDT": held},
                           quotes={"BTCUSDT": make_quote()})
        decision = next(d for d in strat.evaluate(ctx) if d.symbol == SYMBOL)
        assert decision.decision_kind is DecisionKind.REPLACE
        assert decision.reason_code is ReasonCode.REPLACEMENT
        assert decision.metrics.get("replacement_symbol") == "ETHUSDT"

    def test_no_replacement_under_60_periods(self, tmp_path):
        strat = self._two_symbol_env(tmp_path, RATE_OK, "0.0006")
        held = make_held(age_periods=50)
        ctx = make_context(held={"BTCUSDT": held},
                           quotes={"BTCUSDT": make_quote()})
        decision = next(d for d in strat.evaluate(ctx) if d.symbol == SYMBOL)
        assert decision.decision_kind is DecisionKind.HOLD

    def test_no_replacement_below_premium_over_120(self, tmp_path):
        # 持仓 200 期 → premium 0.70：ETH trailing 必须 ≥ BTC trailing × 0.70 才换
        strat = self._two_symbol_env(tmp_path, RATE_OK, "0.0003")  # ETH 更差（年化 0.328 < 0.548×0.7）
        held = make_held(age_periods=200)
        ctx = make_context(held={"BTCUSDT": held},
                           quotes={"BTCUSDT": make_quote()})
        decision = next(d for d in strat.evaluate(ctx) if d.symbol == SYMBOL)
        assert decision.decision_kind is DecisionKind.HOLD

    def test_replacement_picks_best_candidate(self, tmp_path):
        cfg = make_live_config(live_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
                               exit_overrides={"max_holding_periods": 300})
        data = LiveFakeData({
            "BTCUSDT": _rates((20, RATE_OK)),
            "ETHUSDT": _rates((20, "0.0006")),
            "SOLUSDT": _rates((20, "0.0009")),  # 最佳
        })
        strat, _ = make_strategy(tmp_path, data, config=cfg)
        strat.refresh_candidates()
        held = make_held(age_periods=70)
        ctx = make_context(held={"BTCUSDT": held},
                           quotes={"BTCUSDT": make_quote()})
        decision = next(d for d in strat.evaluate(ctx) if d.symbol == SYMBOL)
        assert decision.decision_kind is DecisionKind.REPLACE
        assert decision.metrics.get("replacement_symbol") == "SOLUSDT"
