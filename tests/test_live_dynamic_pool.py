"""动态候选池测试（开发文档 §7.1 第 7 条：live_symbols 空 = 回测同口径动态池）。

覆盖：池选择（成交额过滤 + top N）、刷新节流、失败保留旧池、prune、
PENDING_QUOTE 中间态 → complete_open 定案、service run_once 端到端解析。
"""

from __future__ import annotations

from decimal import Decimal  # noqa: F401  (保持与兄弟测试一致的类型习惯)

from cointrader.live.decisions import DecisionKind, ReasonCode
from conftest import FakeStrategyData, make_rate_series
from live_helpers import (
    NOW,
    make_context,
    make_live_config,
    make_quote,
    make_service,
    make_strategy,
)

UNIVERSE = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT")
VOLUMES_24H = {
    "AAAUSDT": 50e6,
    "BBBUSDT": 40e6,
    "CCCUSDT": 30e6,
    "DDDUSDT": 10e6,
    "EEEUSDT": 0.5e6,  # 低于 min_quote_volume_3d_avg=1M
}
RATE_OK = "0.0005"  # 年化 0.548 > 0.30


def _data() -> FakeStrategyData:
    return FakeStrategyData(
        {s: make_rate_series(20, RATE_OK) for s in UNIVERSE},
        universe=UNIVERSE,
        volumes_24h=VOLUMES_24H,
    )


def _dyn_config(**exec_overrides: object):
    overrides = {"candidate_pool_max_symbols": 3, **exec_overrides}
    return make_live_config(live_symbols=(), exec_overrides=dict(overrides))


class TestDynamicPoolSelection:
    def test_top_n_by_volume(self, tmp_path):
        strat, _ = make_strategy(tmp_path, _data(), config=_dyn_config())
        assert strat.dynamic_pool
        strat.refresh_universe()
        assert list(strat.candidate_symbols) == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]

    def test_volume_below_threshold_excluded(self, tmp_path):
        # EEE 24h 成交额 0.5M < 1M；即使放开 top N 也不进池
        strat, _ = make_strategy(
            tmp_path, _data(), config=_dyn_config(candidate_pool_max_symbols=5)
        )
        strat.refresh_universe()
        assert "EEEUSDT" not in strat.candidate_symbols
        assert len(strat.candidate_symbols) == 4

    def test_universe_refresh_failure_keeps_old_pool(self, tmp_path):
        data = _data()
        strat, _ = make_strategy(tmp_path, data, config=_dyn_config())
        strat.refresh_universe()
        old = strat.candidate_symbols
        data.set_universe_fail(True)
        strat.refresh_universe()
        assert strat.candidate_symbols == old

    def test_universe_not_refetched_within_interval(self, tmp_path):
        clock = {"t": NOW}
        data = _data()
        strat, _ = make_strategy(
            tmp_path, data, config=_dyn_config(universe_refresh_seconds=1800.0),
            now_fn=lambda: clock["t"],
        )
        strat.refresh_universe()
        calls_after_first = data.universe_calls
        assert calls_after_first == 1
        clock["t"] = NOW + 60
        strat.refresh_universe()
        assert data.universe_calls == calls_after_first
        clock["t"] = NOW + 1801
        strat.refresh_universe()
        assert data.universe_calls == calls_after_first + 1

    def test_prune_universe_drops_unruleable_symbols(self, tmp_path):
        strat, _ = make_strategy(tmp_path, _data(), config=_dyn_config())
        strat.refresh_universe()
        pruned = strat.prune_universe({"AAAUSDT"})
        assert pruned == 2
        assert list(strat.candidate_symbols) == ["AAAUSDT"]

    def test_refresh_candidates_builds_pool_first(self, tmp_path):
        strat, _ = make_strategy(tmp_path, _data(), config=_dyn_config())
        assert strat.candidate_symbols == ()  # 首刷前无候选
        strat.refresh_candidates()
        assert strat.candidate_symbols == ("AAAUSDT", "BBBUSDT", "CCCUSDT")

    def test_fixed_pool_ignores_universe(self, tmp_path):
        data = _data()
        strat, _ = make_strategy(tmp_path, data, config=make_live_config())  # 固定 BTCUSDT
        assert not strat.dynamic_pool
        strat.refresh_universe()
        assert data.universe_calls == 0
        assert strat.candidate_symbols == ("BTCUSDT",)


class TestDynamicOpenFlow:
    def test_pending_then_entry_ok(self, tmp_path):
        strat, _ = make_strategy(tmp_path, _data(), config=_dyn_config())
        strat.refresh_candidates()
        ctx = make_context()  # 动态池上下文：无池报价
        decisions = strat.evaluate(ctx)
        assert len(decisions) == 3
        assert all(d.decision_kind is DecisionKind.PENDING_QUOTE for d in decisions)
        assert all(not d.allowed for d in decisions)
        finals = [strat.complete_open(d.symbol, ctx, make_quote()) for d in decisions]
        assert all(
            d.decision_kind is DecisionKind.OPEN and d.reason_code is ReasonCode.ENTRY_OK
            for d in finals
        )

    def test_service_run_once_resolves_pending_and_opens(self, tmp_path):
        env = make_service(tmp_path, _data(), config=_dyn_config())
        env["svc"].run_once()
        # 3 个池 symbol 全部通过门槛 → PENDING → 按需报价 → OPEN
        assert [c["symbol"] for c in env["executor"].open_calls] == [
            "AAAUSDT", "BBBUSDT", "CCCUSDT",
        ]
        rows = env["store"].signal_decisions()
        assert all(r["decision_kind"] != DecisionKind.PENDING_QUOTE for r in rows)
        opens = [r for r in rows if r["reason_code"] == ReasonCode.ENTRY_OK]
        assert len(opens) == 3

    def test_service_quote_failure_records_stale_quote(self, tmp_path):
        env = make_service(tmp_path, _data(), config=_dyn_config())
        env["svc"]._quote_fetcher = lambda _s: None  # noqa: SLF001
        env["svc"].run_once()
        assert env["executor"].open_calls == []
        rows = env["store"].signal_decisions()
        assert all(r["reason_code"] == ReasonCode.STALE_QUOTE for r in rows)
        assert len(rows) == 3
