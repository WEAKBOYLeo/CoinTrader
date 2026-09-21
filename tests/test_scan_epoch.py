"""Scan epoch 与数据完整性闸门测试（实施计划书 v2.0 T2，AC-03/04/05）。

核心命题：只有「全部可交易候选 + 同一 decision_cutoff」的横截面才能
READY 并参与横向排名；失败不得当 excluded；旧 epoch（cutoff 后出现新
结算/K 线闭合）必须 EXPIRED 且不得新增风险；固定等待时间在完整性上没有
任何地位。

全部离线：逻辑时钟 + conftest FakeStrategyData 派生 fake。
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from cointrader.config import Config
from cointrader.execution.store import StateStore
from cointrader.live.decisions import DecisionKind, ReasonCode
from cointrader.live.market_sync import MarketDataSynchronizer, ScanEpochStatus
from cointrader.live.strategy import LiveStrategy, Quote
from conftest import FakeStrategyData, make_rate_series
from live_helpers import (
    NOW,
    NOW_MS,
    make_context,
    make_held,
    make_live_config,
    make_live_rates,
    make_quote,
)

RATE_OK = "0.0005"  # 年化 ~0.548 > 0.30
RATE_HIGH = "0.001"  # 年化 ~1.095
RATE_LOW = "0.000275"  # 年化 ~0.301


class EpochFakeData(FakeStrategyData):
    """可控制「哪些 symbol 本轮失败」的数据源（failed，不是 excluded）。"""

    def __init__(self, *args, fail: set[str] | None = None, **kw) -> None:
        super().__init__(*args, **kw)
        self._fail: set[str] = set(fail or ())

    def set_fail(self, symbols: set[str]) -> None:
        self._fail = set(symbols)

    def funding_rates(
        self, symbol: str, periods: int, *, end_ms: int | None = None
    ) -> list[tuple[int, Decimal, Decimal]]:
        if symbol in self._fail:
            raise RuntimeError("rate limited (fake 429)")
        return super().funding_rates(symbol, periods, end_ms=end_ms)


def _cfg(**exec_overrides: object):
    overrides = {
        "candidate_pool_max_symbols": 10,
        "universe_refresh_seconds": 1.0,  # 测试里快速重建
        "scan_epoch_deadline_seconds": 2.0,
        "candidate_refresh_concurrency": 4,
        **exec_overrides,
    }
    return make_live_config(live_symbols=(), exec_overrides=dict(overrides))


def _exit_cfg(**exit_overrides: object):
    overrides = {"exit_lookback_periods": 6, "max_holding_periods": 10, **exit_overrides}
    return make_live_config(
        live_symbols=(),
        exec_overrides={
            "candidate_pool_max_symbols": 10,
            "universe_refresh_seconds": 1.0,
            "scan_epoch_deadline_seconds": 2.0,
            "candidate_refresh_concurrency": 4,
        },
        exit_overrides=dict(overrides),
    )


def _env(
    tmp_path: Path,
    rates: dict[str, str],
    *,
    fail: set[str] | None = None,
    excluded: dict[str, str] | None = None,
    exclusion_fn: object | None = None,
    config: Config | None = None,
    **exec_overrides: object,
) -> tuple[LiveStrategy, MarketDataSynchronizer, EpochFakeData, dict]:
    clock = {"t": NOW}
    universe = tuple(rates)
    data = EpochFakeData(
        {s: make_live_rates(20, r) for s, r in rates.items()},
        universe=universe,
        volumes_24h={s: 50e6 for s in universe},
        fail=fail,
        now_ms=NOW_MS,
    )
    cfg = config if config is not None else _cfg(**exec_overrides)
    sync = MarketDataSynchronizer(
        config=cfg,
        data=data,
        server_time_fn=lambda: int(clock["t"] * 1000),
        now_fn=lambda: clock["t"],
        excluded_symbols=excluded,
        exclusion_fn=exclusion_fn,  # type: ignore[arg-type]
    )
    store = StateStore(tmp_path / "trading.sqlite3")
    strat = LiveStrategy(
        config=cfg,
        data=data,
        store=store,
        strategy_version="epoch-test",
        config_hash="cfg-epoch",
        now_fn=lambda: clock["t"],
        synchronizer=sync,
    )
    return strat, sync, data, clock


def _ctx(clock: dict, held=None):
    return make_context(held=held or {}, now_ms=clock["t"])


class TestEpochCompleteness:
    def test_ready_requires_full_cross_section_and_ranks_true_top(
        self, tmp_path
    ) -> None:
        """100 候选场景中收益最高的 symbol 最后到达：READY 前零开仓，
        完整后排名选中真正的 top。"""
        rates = {f"S{i:02d}USDT": RATE_OK for i in range(100)}
        rates["TOPUSDT"] = RATE_HIGH  # 收益最高，最后到达
        strat, sync, data, clock = _env(
            tmp_path, rates, fail={"TOPUSDT"}, candidate_pool_max_symbols=200
        )

        # 第一轮构建：TOP 失败 → 不能 READY
        sync.build_once()
        assert sync.latest_ready() is None, "最高收益候选未到时不得 READY"
        strat.refresh_candidates()
        decisions = strat.evaluate(_ctx(clock))
        assert all(
            d.decision_kind is DecisionKind.SKIP
            and d.reason_code is ReasonCode.MARKET_DATA_NOT_READY
            for d in decisions
        ), "READY 前未持仓候选必须全部 SKIP MARKET_DATA_NOT_READY"
        assert len(decisions) == 101  # 每个 expected symbol 恰好一条

        # TOP 到达（同 builder 内重试成功）→ 完整横截面 → READY
        data.set_fail(set())
        clock["t"] = NOW + 0.5
        sync.build_once()
        strat.refresh_candidates()
        epoch = sync.latest_ready()
        assert epoch is not None
        assert "TOPUSDT" in epoch.expected_symbols
        assert strat.candidate_symbols == epoch.expected_symbols

        strat.refresh_candidates()
        decisions = strat.evaluate(_ctx(clock))
        pending = {
            d.symbol: d for d in decisions if d.decision_kind is DecisionKind.PENDING_QUOTE
        }
        # 槽位 max_positions=2（live_helpers 默认）：top 2 进 PENDING，TOPUSDT 必须在其中
        assert len(pending) == 2
        assert "TOPUSDT" in pending
        open_decisions = [
            strat.complete_open(s, _ctx(clock), make_quote()) for s in pending
        ]
        assert all(
            d.decision_kind is DecisionKind.OPEN and d.scan_epoch_id == epoch.epoch_id
            for d in open_decisions
        )
        assert all(d.decision_cutoff_ms == epoch.decision_cutoff_ms for d in open_decisions)

    def test_failed_symbol_is_failed_not_excluded(self, tmp_path) -> None:
        """网络/限流失败的 symbol 必须记 failed，不得当 excluded 静默排除。"""
        rates = {f"S{i:02d}USDT": RATE_OK for i in range(4)}
        strat, sync, data, clock = _env(tmp_path, rates, fail={"S01USDT"})

        sync.build_once()
        clock["t"] = NOW + 3  # 超过 deadline → 封存 DEGRADED
        sync.build_once()
        epoch = sync.latest()
        assert epoch is not None
        assert epoch.status is ScanEpochStatus.DEGRADED
        assert "S01USDT" in epoch.failed, "失败必须是 failed（带原因）"
        assert "S01USDT" not in epoch.excluded, "failed 不得混入 excluded"
        assert sync.latest_ready() is None
        readiness = sync.readiness(now_ms=int(clock["t"] * 1000))
        assert readiness is not None and not readiness.can_rank
        assert readiness.failed_count == 1
        assert "S01USDT" in readiness.failed

    def test_deterministic_excluded_counts_as_coverage(self, tmp_path) -> None:
        """确定性排除（无共同规则等）带 reason 参与 expected 覆盖证明。"""
        rates = {"AAAUSDT": RATE_OK, "BBBUSDT": RATE_OK, "XXXUSDT": RATE_OK}
        strat, sync, data, clock = _env(
            tmp_path, rates, excluded={"XXXUSDT": "NO_COMMON_RULES"}
        )
        sync.build_once()
        epoch = sync.latest_ready()
        assert epoch is not None, "其余候选完整时 excluded 不应阻止 READY"
        assert "XXXUSDT" in epoch.excluded
        assert epoch.excluded["XXXUSDT"] == "NO_COMMON_RULES"
        assert "XXXUSDT" not in epoch.expected_symbols or "XXXUSDT" in epoch.excluded
        assert "XXXUSDT" not in sync.snapshots_for(epoch.epoch_id)

    def test_settlement_lag_is_failed(self, tmp_path) -> None:
        """cutoff 前应有的最后一次结算未到账 → 该候选 failed（不是 excluded）。"""
        rates = {
            "OKUSDT": RATE_OK,
            "LAGUSDT": RATE_OK,
        }
        strat, sync, data, clock = _env(tmp_path, rates)
        data.set_rates(
            "LAGUSDT",
            make_rate_series(
                20, RATE_OK, end_ms=NOW_MS - 30 * 60 * 1000
            ),
        )
        sync.build_once()
        assert sync.latest_ready() is None
        # 等 deadline → DEGRADED 且 failed 原因含结算滞后
        clock["t"] = NOW + 3
        sync.build_once()
        epoch = sync.latest()
        assert epoch is not None and epoch.status is ScanEpochStatus.DEGRADED
        assert "LAGUSDT" in epoch.failed
        assert "OKUSDT" not in epoch.failed

    def test_trigger_respects_universe_interval(self, tmp_path) -> None:
        """READY 有效期内反复触发不重建（universe 请求不浪费）。"""
        rates = {f"S{i:02d}USDT": RATE_OK for i in range(3)}
        strat, sync, data, clock = _env(tmp_path, rates, universe_refresh_seconds=60.0)
        first = sync.build_once()
        epoch = sync.latest_ready()
        assert epoch is not None and epoch.epoch_id == first
        calls = data.universe_calls
        assert sync.trigger_refresh() == first, "未到期不重建"
        assert data.universe_calls == calls
        clock["t"] = NOW + 61
        second = sync.trigger_refresh()
        assert second != first, "universe 到期后重建新 epoch"
        assert data.universe_calls == calls + 1


class TestEpochExpiry:
    def test_expired_after_new_kline_close_blocks_new_risk(self, tmp_path) -> None:
        """cutoff 后 4h K 线闭合 → 旧 epoch EXPIRED：禁开仓，退出仍可执行。"""
        rates = {"AAAUSDT": RATE_OK}
        strat, sync, data, clock = _env(tmp_path, rates)
        sync.build_once()
        epoch = sync.latest_ready()
        assert epoch is not None
        strat.refresh_candidates()

        # 持仓 70 期（> max_holding=10 → 触发 EXIT；风险降低不受 epoch 闸门影响）
        held = {"AAAUSDT": make_held("AAAUSDT", age_periods=70)}
        clock["t"] = NOW + 4 * 3600 + 60  # 跨过 4h K 线闭合
        assert sync.latest_ready() is None, "新 4h K 线闭合后旧 epoch 必须 EXPIRED"
        latest = sync.latest()
        assert latest is not None
        assert latest.status is ScanEpochStatus.EXPIRED

        decisions = strat.evaluate(_ctx(clock, held=held))
        by_symbol = {d.symbol: d for d in decisions}
        assert by_symbol["AAAUSDT"].decision_kind is DecisionKind.EXIT, (
            "EXPIRED 不得阻止已有仓位的风险降低退出"
        )

    def test_expired_epoch_allows_rebuild_then_open_again(self, tmp_path) -> None:
        """EXPIRED 后新 epoch 完整重建 → 恢复 READY → 重新可开仓。"""
        rates = {"AAAUSDT": RATE_OK}
        strat, sync, data, clock = _env(tmp_path, rates)
        first = sync.build_once()
        assert sync.latest_ready() is not None

        clock["t"] = NOW + 4 * 3600 + 60
        data.now_ms = int(clock["t"] * 1000)
        data.set_rates("AAAUSDT", make_live_rates_at(clock["t"], 20, RATE_OK))
        epoch2 = sync.trigger_refresh()
        assert epoch2 != first
        sync.build_once()  # worker 在 service 里执行 fill；测试中同步完成
        latest = sync.latest()
        assert latest is not None and latest.status is ScanEpochStatus.READY
        strat.refresh_candidates()
        decisions = strat.evaluate(_ctx(clock))
        assert any(
            d.decision_kind is DecisionKind.PENDING_QUOTE for d in decisions
        ), "新 READY epoch 应恢复开仓评估"


def make_live_rates_at(t: float, n: int, rate: str) -> list[tuple[int, Decimal, Decimal]]:
    """锚定到任意逻辑时钟 t 的费率序列（末期结算在 t-30min）。"""
    return make_rate_series(
        n, rate, end_ms=int(t * 1000) - 30 * 60 * 1000 + 8 * 3600 * 1000
    )


class TestReplacementGate:
    def test_replacement_blocked_without_ready_epoch(self, tmp_path) -> None:
        """数据不足（无 READY epoch）时不得发策略性换仓。"""
        cfg = _exit_cfg(max_holding_periods=270)
        strat, sync, data, clock = _env(
            tmp_path,
            {"AAAUSDT": RATE_LOW, "BBBUSDT": RATE_HIGH},
            fail={"BBBUSDT"},
            config=cfg,
        )
        held = {"AAAUSDT": make_held("AAAUSDT", age_periods=70)}
        sync.build_once()
        clock["t"] = NOW + 3
        sync.build_once()  # BBB 持续失败 → DEGRADED，无 READY
        assert sync.latest_ready() is None
        decisions = {d.symbol: d for d in strat.evaluate(_ctx(clock, held=held))}
        # 持仓评估发生，但不得出现 REPLACE
        assert all(d.decision_kind is not DecisionKind.REPLACE for d in decisions.values())

    def test_replacement_allowed_from_same_ready_epoch(self, tmp_path) -> None:
        """READY 时：同 epoch 内高收益候选可触发换仓。"""
        cfg = _exit_cfg(max_holding_periods=270)
        strat, sync, data, clock = _env(
            tmp_path,
            {"AAAUSDT": RATE_LOW, "BBBUSDT": RATE_HIGH},
            config=cfg,
        )
        held = {"AAAUSDT": make_held("AAAUSDT", age_periods=70)}
        sync.build_once()
        assert sync.latest_ready() is not None
        strat.refresh_candidates()
        decisions = {d.symbol: d for d in strat.evaluate(_ctx(clock, held=held))}
        assert decisions["AAAUSDT"].decision_kind is DecisionKind.REPLACE
        assert decisions["AAAUSDT"].metrics.get("replacement_symbol") == "BBBUSDT"


class TestQuoteSkew:
    def test_quote_skew_over_limit_blocks_open(self, tmp_path) -> None:
        """Spot/Futures 报价接收时间偏差超限 → 不得生成交易意图。"""
        rates = {"AAAUSDT": RATE_OK}
        strat, sync, data, clock = _env(tmp_path, rates)
        sync.build_once()
        strat.refresh_candidates()
        ctx = _ctx(clock)
        base_ts = int(clock["t"] * 1000)
        skewed = Quote(
            spot_price=Decimal("100"),
            perp_price=Decimal("100"),
            ts_ms=base_ts,
            spot_ts_ms=base_ts,
            perp_ts_ms=base_ts + 2000,  # > max_quote_skew_ms=500
        )
        decision = strat.complete_open("AAAUSDT", ctx, skewed)
        assert decision.decision_kind is DecisionKind.SKIP
        assert decision.reason_code is ReasonCode.STALE_QUOTE
        # 同步报价正常放行
        aligned = Quote(
            spot_price=Decimal("100"),
            perp_price=Decimal("100"),
            ts_ms=base_ts,
            spot_ts_ms=base_ts,
            perp_ts_ms=base_ts + 100,
        )
        ok = strat.complete_open("AAAUSDT", ctx, aligned)
        assert ok.decision_kind is DecisionKind.OPEN


class TestOldEpochNoMix:
    def test_new_epoch_replaces_candidates_atomically(self, tmp_path) -> None:
        """新 epoch READY 后候选表整体切换到新横截面；旧数据不混入。"""
        rates = {"AAAUSDT": RATE_OK, "BBBUSDT": RATE_OK}
        strat, sync, data, clock = _env(tmp_path, rates)
        first = sync.build_once()
        assert sync.latest_ready() is not None
        strat.refresh_candidates()
        old_id = strat._epoch_id  # noqa: SLF001 —— 单元测试读内部状态
        assert old_id == first

        # 新 epoch：BBB 收益翻倍，数据重锚到新时钟
        clock["t"] = NOW + 4 * 3600 + 60
        data.now_ms = int(clock["t"] * 1000)
        data.set_rates("BBBUSDT", make_live_rates_at(clock["t"], 20, RATE_HIGH))
        second = sync.trigger_refresh()
        assert second != first
        sync.build_once()
        strat.refresh_candidates()
        assert strat._epoch_id == second  # noqa: SLF001
        # 新横截面下 BBB 才是 top 1
        decisions = {d.symbol: d for d in strat.evaluate(_ctx(clock))}
        assert decisions["BBBUSDT"].decision_kind is DecisionKind.PENDING_QUOTE
        assert decisions["BBBUSDT"].scan_epoch_id == second
        pending = [s for s, d in decisions.items() if d.decision_kind is DecisionKind.PENDING_QUOTE]
        assert pending[0] == "BBBUSDT", "排名必须基于新 epoch 横截面（BBB 最高收益）"


class TestPairExclusion:
    """可开仓对预筛（exclusion_fn）：无现货/合约腿的币在 epoch 构建时
    直接 excluded，不进排名、不占开仓槽位。

    回归：demo 受限对（BR/BTW/PIEVERSE 无 demo 现货交易对）资金费排名最高
    → 永久占满 top3 槽位 → 报价永远失败（STALE_QUOTE）→ 10+ 小时零开仓。
    """

    def test_excluded_pair_does_not_occupy_open_slot(self, tmp_path) -> None:
        rates = {"BRUSDT": RATE_HIGH, "XMRUSDT": RATE_OK}
        strat, sync, data, clock = _env(
            tmp_path,
            rates,
            exclusion_fn=lambda s: "无现货交易对" if s == "BRUSDT" else None,
        )
        sync.build_once()
        epoch = sync.latest_ready()
        assert epoch is not None and epoch.status is ScanEpochStatus.READY
        assert epoch.excluded.get("BRUSDT") == "无现货交易对"
        assert "XMRUSDT" in epoch.funding_last_ts_ms

        strat.refresh_candidates()
        decisions = {d.symbol: d for d in strat.evaluate(_ctx(clock))}
        # excluded 币不参与评估（不在候选内）；有现货腿的 XMR 正常进排名，
        # 全闸通过（PENDING_QUOTE = 等新鲜报价，ctx 未带报价）
        assert "BRUSDT" not in decisions
        assert decisions["XMRUSDT"].decision_kind in (
            DecisionKind.OPEN,
            DecisionKind.PENDING_QUOTE,
        )

    def test_exclusion_fn_exception_keeps_symbol_in_expected(self, tmp_path) -> None:
        """fn 异常 ≠ 确定性排除：symbol 留在 expected，交给后续闸门。"""

        def boom(_s: str) -> str | None:
            raise RuntimeError("rules unavailable (fake)")

        rates = {"AUSDT": RATE_OK}
        _strat, sync, _data, _clock = _env(tmp_path, rates, exclusion_fn=boom)
        sync.build_once()
        epoch = sync.latest_ready()
        assert epoch is not None and epoch.status is ScanEpochStatus.READY
        assert "AUSDT" in epoch.funding_last_ts_ms
        assert epoch.excluded == {}
