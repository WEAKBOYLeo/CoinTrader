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
    LiveFakeData,
    make_context,
    make_live_config,
    make_live_rates,
    make_quote,
    make_service,
    make_strategy,
)

UNIVERSE = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT")
SYMBOL = "BTCUSDT"
VOLUMES_24H = {
    "AAAUSDT": 50e6,
    "BBBUSDT": 40e6,
    "CCCUSDT": 30e6,
    "DDDUSDT": 10e6,
    "EEEUSDT": 0.5e6,  # 低于 min_quote_volume_3d_avg=1M
}
RATE_OK = "0.0005"  # 年化 0.548 > 0.30


def _data() -> FakeStrategyData:
    return LiveFakeData(
        {s: make_live_rates(20, RATE_OK) for s in UNIVERSE},
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


class TestRefetchRateLimit:
    """结算边界全员到点：每 60s 窗口限量，不能一次全拉（API 按分钟限流）。"""

    def _ten_symbol_env(self, tmp_path):
        clock = {"t": NOW}
        universe = tuple(f"S{i:02d}USDT" for i in range(10))
        data = LiveFakeData(
            {s: make_live_rates(20, RATE_OK) for s in universe},
            universe=universe,
            volumes_24h={s: 50e6 for s in universe},
        )
        cfg = _dyn_config(candidate_pool_max_symbols=10, candidate_refetch_per_minute=3)
        strat, _ = make_strategy(tmp_path, data, config=cfg, now_fn=lambda: clock["t"])
        return strat, data, clock

    def test_boundary_burst_spread_over_windows(self, tmp_path):
        strat, data, clock = self._ten_symbol_env(tmp_path)
        strat.refresh_candidates()
        assert data.funding_calls == 3  # 首批只刷 3 个
        strat.refresh_candidates()  # 同窗口内：预算耗尽，不再拉
        assert data.funding_calls == 3
        clock["t"] = NOW + 61
        strat.refresh_candidates()
        assert data.funding_calls == 6
        clock["t"] = NOW + 122
        strat.refresh_candidates()
        assert data.funding_calls == 9
        clock["t"] = NOW + 183
        strat.refresh_candidates()
        assert data.funding_calls == 10

    def test_fresh_cache_not_refetched_before_boundary(self, tmp_path):
        strat, data, clock = self._ten_symbol_env(tmp_path)
        clock["t"] = NOW + 600  # 10 分钟：全部刷完（5 窗口 × 3）
        for _ in range(5):
            strat.refresh_candidates()
            clock["t"] += 61
        assert data.funding_calls == 10
        strat.refresh_candidates()  # 刚刷过（< 8h 周期）：不重拉
        assert data.funding_calls == 10
        clock["t"] = NOW + 7 * 3600  # 7h < 8h 周期：仍未到点
        strat.refresh_candidates()
        assert data.funding_calls == 10
        clock["t"] = NOW + 8 * 3600 + 1000  # 跨过各自 fetch 时间 + 结算周期：重拉
        strat.refresh_candidates()
        assert data.funding_calls > 10


class TestEntryGates:
    def test_settlement_lag_blocks_entry(self, tmp_path):
        """最新结算已发生但缓存未刷新 → 禁止开仓（防拿旧数据比收益率）。"""
        clock = {"t": NOW}
        data = LiveFakeData({SYMBOL: make_live_rates(40, RATE_OK)})
        strat, _ = make_strategy(tmp_path, data, now_fn=lambda: clock["t"])
        strat.refresh_candidates()
        clock["t"] = NOW + 8 * 3600 + 60  # 结算边界已过 1 分钟，未重刷
        ctx = make_context(now_ms=clock["t"])
        decision = strat.can_open(SYMBOL, ctx)
        assert decision.decision_kind is DecisionKind.SKIP
        assert decision.reason_code is ReasonCode.SETTLEMENT_LAG
        # 重刷后解除（数据补齐）
        data.now_ms = int(clock["t"] * 1000)
        data.set_rates(
            SYMBOL,
            make_rate_series(40, RATE_OK, end_ms=data.now_ms - 30 * 60 * 1000 + 8 * 3600 * 1000),
        )
        strat.refresh_candidates()
        decision2 = strat.can_open(SYMBOL, ctx)
        assert decision2.decision_kind is DecisionKind.PENDING_QUOTE


class TestDynamicOpenFlow:
    def test_pending_then_entry_ok(self, tmp_path):
        strat, _ = make_strategy(tmp_path, _data(), config=_dyn_config())
        strat.refresh_candidates()
        ctx = make_context()  # 动态池上下文：无池报价
        decisions = strat.evaluate(ctx)
        assert len(decisions) == 3
        # 3 个通过前置，max_positions=2 → 按收益率取 top 2，第 3 个 RANKED_OUT
        by_symbol = {d.symbol: d for d in decisions}
        assert by_symbol["AAAUSDT"].decision_kind is DecisionKind.PENDING_QUOTE
        assert by_symbol["BBBUSDT"].decision_kind is DecisionKind.PENDING_QUOTE
        assert by_symbol["CCCUSDT"].reason_code is ReasonCode.RANKED_OUT
        finals = [
            strat.complete_open(s, ctx, make_quote())
            for s in ("AAAUSDT", "BBBUSDT")
        ]
        assert all(
            d.decision_kind is DecisionKind.OPEN and d.reason_code is ReasonCode.ENTRY_OK
            for d in finals
        )

    def test_top_n_by_trailing_gets_slots(self, tmp_path):
        """收益率高者得槽位：低收益币不得抢先占用。"""
        universe = ("LOWUSDT", "HIGHUSDT", "MIDUSDT")
        data = LiveFakeData(
            {
                "LOWUSDT": make_live_rates(20, "0.000275"),   # 年化 ~0.301
                "HIGHUSDT": make_live_rates(20, "0.001"),     # 年化 ~1.095
                "MIDUSDT": make_live_rates(20, "0.0005"),     # 年化 ~0.548
            },
            universe=universe,
            volumes_24h={s: 50e6 for s in universe},
        )
        strat, _ = make_strategy(tmp_path, data, config=_dyn_config())
        strat.refresh_candidates()
        decisions = {d.symbol: d for d in strat.evaluate(make_context())}
        assert decisions["HIGHUSDT"].decision_kind is DecisionKind.PENDING_QUOTE
        assert decisions["MIDUSDT"].decision_kind is DecisionKind.PENDING_QUOTE
        assert decisions["LOWUSDT"].reason_code is ReasonCode.RANKED_OUT
        assert "排名第 3" in decisions["LOWUSDT"].reason_text

    def test_service_run_once_resolves_pending_and_opens(self, tmp_path):
        env = make_service(tmp_path, _data(), config=_dyn_config())
        env["svc"].run_once()
        # 3 个池 symbol 全部通过门槛，槽位 2 → 按池序（收益率相同）取前 2 开仓
        assert [c["symbol"] for c in env["executor"].open_calls] == ["AAAUSDT", "BBBUSDT"]
        rows = env["store"].signal_decisions()
        assert all(r["decision_kind"] != DecisionKind.PENDING_QUOTE for r in rows)
        opens = [r for r in rows if r["reason_code"] == ReasonCode.ENTRY_OK]
        assert len(opens) == 2
        ranked = [r for r in rows if r["reason_code"] == ReasonCode.RANKED_OUT]
        assert [r["symbol"] for r in ranked] == ["CCCUSDT"]

    def test_service_quote_failure_records_stale_quote(self, tmp_path):
        env = make_service(tmp_path, _data(), config=_dyn_config())
        env["svc"]._quote_fetcher = lambda _s: None  # noqa: SLF001
        env["svc"].run_once()
        assert env["executor"].open_calls == []
        rows = env["store"].signal_decisions()
        stale = [r for r in rows if r["reason_code"] == ReasonCode.STALE_QUOTE]
        assert len(stale) == 2
        assert any(r["reason_code"] == ReasonCode.RANKED_OUT for r in rows)
