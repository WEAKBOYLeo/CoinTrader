"""T2：订单前双腿报价来源与 freshness gate（实施计划书 5.0 T2，AC-04）。

覆盖：

- ``evaluate_quote_gate`` 纯函数（固定时钟）：stale / future / 缺接收时间 /
  两腿 skew 超限 / symbol 不匹配 / 来源未知 / 非正价格 / 边界（含端点）；
- REST adapter（``_make_quote_fetcher``）provenance：``source="rest"``、symbol、
  两腿本地接收时间、交易所时间戳为 None（不得伪造）；
- runner 集成（time-of-check == time-of-use）：风险审批时报价尚新鲜、
  plan 前重验越期 → 不建 plan、broker（executor）调用 0、拒绝原因可见；
- 正常 quote 等价性：开仓一次、执行计划落账且携带 quote provenance；
- CLOSE/reduce-only 不被 entry freshness gate 阻塞（陈旧报价仍可平仓，
  且计划 payload 记录报价质量）；
- quote snapshot 与 intent/plan 关联走既有 ledger JSON payload 字段
  （无 schema migration）。

全部离线：fake 数据源/时钟/执行器，不联网、不触碰真实账本。
"""

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from cointrader.domain.market import DataQuality
from cointrader.live.service import _make_quote_fetcher
from cointrader.live.strategy import (
    KNOWN_QUOTE_SOURCES,
    Quote,
    QuoteGateVerdict,
    evaluate_quote_gate,
)
from conftest import FakeStrategyData
from live_helpers import (
    NOW,
    NOW_MS,
    LiveFakeData,
    make_live_config,
    make_live_rates,
    make_quote,
    make_service,
    seed_current_positions,
)

SYMBOL = "BTCUSDT"
MAX_AGE_MS = 5_000  # execution.max_market_data_age_seconds 默认 5.0
MAX_SKEW_MS = 500  # execution.max_quote_skew_ms 默认 500


def _gate(
    quote: Quote,
    *,
    now_ms: int = NOW_MS,
    intent_symbol: str = SYMBOL,
    max_age_ms: int = MAX_AGE_MS,
    max_skew_ms: int = MAX_SKEW_MS,
) -> QuoteGateVerdict:
    return evaluate_quote_gate(
        quote,
        intent_symbol=intent_symbol,
        now_ms=now_ms,
        max_age_ms=max_age_ms,
        max_skew_ms=max_skew_ms,
    )


# ---------------------------------------------------------------------------
# 纯函数 gate
# ---------------------------------------------------------------------------


class TestQuoteGatePure:
    """evaluate_quote_gate：fail-closed 判定顺序与边界（注入时钟）。"""

    def test_fresh_quote_passes(self) -> None:
        verdict = _gate(make_quote())
        assert verdict.quality is DataQuality.FRESH
        assert verdict.spot_age_ms == 0 and verdict.perp_age_ms == 0
        assert verdict.max_leg_age_ms == 0 and verdict.skew_ms == 0
        assert "FRESH" not in verdict.reason or "通过" in verdict.reason

    def test_source_is_finite_registered_set(self) -> None:
        # 生产 REST 与测试 fake 必须是已注册来源（有限集）
        assert {"rest", "stream", "fake"} <= KNOWN_QUOTE_SOURCES

    def test_stale_quote_rejected(self) -> None:
        verdict = _gate(make_quote(ts=NOW - 30))  # 30s 前 > 5s 上限
        assert verdict.quality is DataQuality.STALE
        assert "过期" in verdict.reason
        assert verdict.max_leg_age_ms == 30_000

    def test_boundary_age_inclusive_then_rejected(self) -> None:
        # 边界含端点（与 portfolio/pair_executor 的 age > max 口径一致）：
        # age == max_age_ms 仍 FRESH，age == max_age_ms + 1 拒绝
        at_boundary = _gate(make_quote(ts=NOW - 5))
        assert at_boundary.quality is DataQuality.FRESH
        just_expired = _gate(make_quote(ts=NOW - 5 - 0.001))
        assert just_expired.quality is DataQuality.STALE
        assert just_expired.max_leg_age_ms == MAX_AGE_MS + 1

    def test_future_timestamp_rejected(self) -> None:
        verdict = _gate(make_quote(ts=NOW + 60))
        assert verdict.quality is DataQuality.INVALID
        assert "未来" in verdict.reason

    def test_missing_receive_timestamp_rejected(self) -> None:
        quote = Quote(
            spot_price=Decimal("100"),
            perp_price=Decimal("100"),
            ts_ms=0,  # 缺少接收时间
            symbol=SYMBOL,
            source="fake",
        )
        verdict = _gate(quote)
        assert verdict.quality is DataQuality.INVALID
        assert "接收时间" in verdict.reason

    def test_future_exchange_timestamp_rejected(self) -> None:
        quote = Quote(
            spot_price=Decimal("100"),
            perp_price=Decimal("100"),
            ts_ms=NOW_MS - 1000,
            symbol=SYMBOL,
            source="rest",
            spot_exchange_ts_ms=NOW_MS + 10,  # 交易所时间晚于当前 → 不可信
        )
        verdict = _gate(quote)
        assert verdict.quality is DataQuality.INVALID
        assert "交易所时间戳" in verdict.reason

    def test_missing_exchange_timestamp_is_credible_for_rest(self) -> None:
        # REST 报价不提供交易所时间戳是合法口径：新鲜度以本地接收年龄判定，
        # 不得因 exchange ts 缺失而默认 FRESH 之外的「现在 fresh」伪造
        verdict = _gate(make_quote())  # exchange ts 均为 None
        assert verdict.quality is DataQuality.FRESH

    def test_skew_over_limit_rejected(self) -> None:
        # 两腿均为过去时间、接收时刻相差 2000ms > 500ms → 不同步
        quote = Quote(
            spot_price=Decimal("100"),
            perp_price=Decimal("100"),
            ts_ms=NOW_MS - 2000,
            spot_ts_ms=NOW_MS - 4000,
            perp_ts_ms=NOW_MS - 2000,
            symbol=SYMBOL,
            source="fake",
        )
        verdict = _gate(quote)
        assert verdict.quality is DataQuality.STALE
        assert "偏差" in verdict.reason
        assert verdict.skew_ms == 2000

    def test_future_leg_timestamp_rejected_as_invalid(self) -> None:
        # 任一腿接收时间为未来（负年龄）→ INVALID，不按 skew 放行
        quote = Quote(
            spot_price=Decimal("100"),
            perp_price=Decimal("100"),
            ts_ms=NOW_MS,
            spot_ts_ms=NOW_MS,
            perp_ts_ms=NOW_MS + 2000,  # 永续腿晚于当前 → 不可信
            symbol=SYMBOL,
            source="fake",
        )
        verdict = _gate(quote)
        assert verdict.quality is DataQuality.INVALID

    def test_skew_boundary_inclusive_then_rejected(self) -> None:
        at = Quote(
            spot_price=Decimal("100"), perp_price=Decimal("100"),
            ts_ms=NOW_MS, spot_ts_ms=NOW_MS - MAX_SKEW_MS, perp_ts_ms=NOW_MS,
            symbol=SYMBOL, source="fake",
        )
        assert _gate(at).quality is DataQuality.FRESH
        over = Quote(
            spot_price=Decimal("100"), perp_price=Decimal("100"),
            ts_ms=NOW_MS, spot_ts_ms=NOW_MS - (MAX_SKEW_MS + 1), perp_ts_ms=NOW_MS,
            symbol=SYMBOL, source="fake",
        )
        assert _gate(over).quality is DataQuality.STALE

    def test_symbol_mismatch_rejected(self) -> None:
        verdict = _gate(make_quote(symbol="ETHUSDT"))
        assert verdict.quality is DataQuality.INVALID
        assert "不一致" in verdict.reason

    def test_missing_symbol_cannot_verify(self) -> None:
        quote = Quote(
            spot_price=Decimal("100"), perp_price=Decimal("100"),
            ts_ms=NOW_MS, symbol="", source="fake",
        )
        verdict = _gate(quote)
        assert verdict.quality is DataQuality.INCOMPLETE
        assert "symbol" in verdict.reason

    def test_unknown_source_not_fresh(self) -> None:
        for source in ("", "mystery", "REST"):
            verdict = _gate(make_quote(source=source))
            assert verdict.quality is not DataQuality.FRESH, f"source={source!r}"
            assert verdict.quality is DataQuality.INCOMPLETE

    def test_nonpositive_price_rejected(self) -> None:
        assert _gate(make_quote(spot="0")).quality is DataQuality.INVALID
        assert _gate(make_quote(perp="-1")).quality is DataQuality.INVALID

    def test_leg_received_ms_fallback_to_ts_ms(self) -> None:
        quote = Quote(
            spot_price=Decimal("100"), perp_price=Decimal("100"), ts_ms=NOW_MS,
            symbol=SYMBOL, source="fake",
        )
        assert quote.spot_received_ms == NOW_MS  # spot_ts_ms=0 → 回退 ts_ms
        assert quote.perp_received_ms == NOW_MS
        assert quote.received_at_ms == NOW_MS


# ---------------------------------------------------------------------------
# REST adapter provenance
# ---------------------------------------------------------------------------


class _FakePublic:
    """最小公开行情 fake（与真实 premiumIndex 字段一致，无 lastPrice）。"""

    def spot_price(self, symbol: str) -> str:  # noqa: ARG002
        return "81000.00"

    def premium_index(self, symbol: str) -> dict[str, Any]:  # noqa: ARG002
        return {
            "symbol": "BTCUSDT",
            "markPrice": "81044.91",
            "indexPrice": "81070.59",
            "lastFundingRate": "0.00008906",
            "time": 1789801254000,
        }


class TestRestAdapterProvenance:
    def test_fetch_carries_source_symbol_and_receive_times(self) -> None:
        import time

        fetch = _make_quote_fetcher(_FakePublic())
        quote = fetch("BTCUSDT")
        assert quote is not None
        assert quote.source == "rest"
        assert quote.symbol == "BTCUSDT"
        # 两腿本地接收时间均存在且为最近时刻（不冒充交易所时刻）
        wall_now_ms = int(time.time() * 1000)
        assert 0 < quote.spot_received_ms <= wall_now_ms
        assert 0 < quote.perp_received_ms <= wall_now_ms
        assert quote.received_at_ms > 0
        # REST 不提供交易所时间戳 → None（不得用本地时刻伪造）
        assert quote.spot_exchange_ts_ms is None
        assert quote.perp_exchange_ts_ms is None
        assert quote.connection_generation is None
        # REST 报价在接收当下通过 gate（新鲜度 = 本地接收年龄口径）
        verdict = evaluate_quote_gate(
            quote,
            intent_symbol="BTCUSDT",
            now_ms=wall_now_ms,
            max_age_ms=MAX_AGE_MS,
            max_skew_ms=MAX_SKEW_MS,
        )
        assert verdict.quality is DataQuality.FRESH

    def test_fetch_failure_returns_none(self) -> None:
        class _Boom:
            def spot_price(self, symbol: str) -> str:  # noqa: ARG002
                raise RuntimeError("api down")

        assert _make_quote_fetcher(_Boom())("BTCUSDT") is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# runner 集成：plan 前最后一刻 gate（broker 调用 0 / 等价性 / 平仓不阻塞）
# ---------------------------------------------------------------------------


def _env(tmp_path: Path) -> dict[str, Any]:
    cfg = make_live_config()
    cfg = replace(cfg, execution=replace(cfg.execution,
                                         candidate_refresh_seconds=5,
                                         reconciliation_interval_seconds=5,
                                         snapshot_interval_seconds=5))
    data = LiveFakeData({SYMBOL: make_live_rates(20, "0.0005")})
    return make_service(tmp_path, data, config=cfg)


def _plan_payloads(store: Any) -> list[dict[str, Any]]:
    return [json.loads(x["payload_json"]) for x in store.pipeline_records("execution_plan")]


class TestRunnerQuoteGate:
    def test_fresh_quote_opens_and_plan_carries_provenance(self, tmp_path) -> None:
        """正常 quote 等价性：开仓一次；执行计划 payload 携带 quote provenance
        并与 intent 关联（plan_id = plan-<intent_id>），走既有 JSON 字段。"""
        env = _env(tmp_path)
        svc = env["svc"]
        svc.run_id = "run-t2-fresh"

        r = svc.run_once()
        assert r["state"] == "RUNNING"
        assert [c["symbol"] for c in env["executor"].open_calls] == [SYMBOL]
        assert len(r["intents"]) == 1

        plans = _plan_payloads(env["store"])
        assert len(plans) == 1
        plan = plans[0]
        quote_block = plan["quote"]
        assert quote_block["source"] == "fake"
        assert quote_block["symbol"] == SYMBOL
        assert quote_block["received_at_ms"] > 0
        assert quote_block["spot_received_ms"] > 0
        assert quote_block["perp_received_ms"] > 0
        assert quote_block["spot_exchange_ts_ms"] is None
        assert quote_block["gate_quality"] == "FRESH"
        assert quote_block["gate_now_ms"] >= quote_block["received_at_ms"]
        # 与 intent 关联：plan_id = plan-<intent_id>
        assert plan["plan_id"] == f"plan-{r['intents'][0]}"

    def test_quote_stale_before_plan_zero_broker_calls(self, tmp_path) -> None:
        """时间间隙场景：风险审批时报价新鲜（放行），plan 前重验已越期 →
        不建执行计划、broker 调用 0、拒绝原因可见（source/time/reason）。"""
        env = _env(tmp_path)
        svc = env["svc"]
        svc.run_id = "run-t2-tou"
        alerts: list[tuple[str, str]] = []
        svc._on_alert = lambda kind, msg: alerts.append((kind, str(msg)))  # noqa: SLF001

        # 首轮 run_once 的 quote 获取顺序（固定池单 symbol）：
        # 1 周期采样 → 2 _build_context → 3 _market_snapshot →
        # 4 _market_snapshot_for（风险审批）→ 5 plan 前重取。
        # 前 4 次返回新鲜报价（风险审批放行），第 5 次起越期 → runner gate 拒绝。
        calls = {"n": 0}

        def fetch(_sym: str) -> Quote:
            calls["n"] += 1
            if calls["n"] <= 4:
                return make_quote(symbol=_sym)
            return make_quote(ts=NOW - 30, symbol=_sym)  # 30s 前 → STALE

        svc._quote_fetcher = fetch  # noqa: SLF001

        r = svc.run_once()
        assert r["state"] == "RUNNING"
        assert env["executor"].open_calls == [], "越期报价不得进入 plan/executor"
        assert env["executor"].close_calls == []
        assert _plan_payloads(env["store"]) == [], "拒绝后不得创建执行计划"
        assert any("入场报价被拒" in s and SYMBOL in s for s in r["skipped"]), r["skipped"]
        kinds = [k for k, _ in alerts]
        assert "QUOTE_GATE_REJECTED" in kinds
        detail = next(m for k, m in alerts if k == "QUOTE_GATE_REJECTED")
        assert "STALE" in detail and "source='fake'" in detail

    def test_close_not_blocked_by_entry_gate_and_quality_recorded(self, tmp_path) -> None:
        """CLOSE/reduce-only 不被 entry freshness gate 阻塞：陈旧报价下仍平仓，
        且计划 payload 记录报价质量（STALE）。"""
        env = _env(tmp_path)
        svc = env["svc"]
        spot = env["spot"]
        futures = env["futures"]
        svc.run_id = "run-t2-close"
        data: FakeStrategyData = svc.strategy.data  # noqa: SLF001

        # 1) 正常开仓一次
        svc.run_once()
        assert len(env["executor"].open_calls) == 1

        # 2) 持仓确认 + 尾部费率转负 → EXIT → CLOSE intent
        spot.balances_map["BTC"] = Decimal("0.01")
        futures.position_amt = Decimal("-0.01")
        seed_current_positions(env["store"], SYMBOL, "0.01", "-0.01")
        values = ["0.0005"] * 14 + ["-0.0005"] * 6
        start = NOW_MS - 30 * 60 * 1000 - 20 * 8 * 3600 * 1000
        data.set_rates(SYMBOL, [(start + i * 8 * 3600 * 1000, Decimal(v), Decimal("100"))
                                for i, v in enumerate(values)])
        clock = {"t": NOW + 8 * 3600 + 62}
        svc._now = lambda: clock["t"]  # noqa: SLF001
        svc.strategy._now = lambda: clock["t"]  # noqa: SLF001

        # 3) 本轮全部报价越期：平仓不得被 entry gate 阻塞
        def stale_fetch(_sym: str) -> Quote:
            return make_quote(ts=NOW - 30, symbol=_sym)

        svc._quote_fetcher = stale_fetch  # noqa: SLF001

        r2 = svc.run_once()
        assert r2["state"] == "RUNNING"
        assert [c["symbol"] for c in env["executor"].close_calls] == [SYMBOL], \
            "陈旧报价不得阻塞 reduce-only 平仓"
        plans = _plan_payloads(env["store"])
        # 平仓计划 = 全部订单 reduce_only（与 r1 的 OPEN plan 区分）
        close_plans = [
            p for p in plans
            if p["orders"] and all(o["reduce_only"] for o in p["orders"])
        ]
        assert len(close_plans) == 1
        close_plan = close_plans[0]
        # 平仓计划仍携带报价 provenance，且记录了非 FRESH 质量
        assert close_plan["quote"]["gate_quality"] == "STALE"
        assert close_plan["quote"]["source"] == "fake"

    def test_unknown_source_zero_broker_calls(self, tmp_path) -> None:
        """来源未知（空 source）的报价在 plan 前重验被拒：broker 调用 0。"""
        env = _env(tmp_path)
        svc = env["svc"]
        svc.run_id = "run-t2-src"

        calls = {"n": 0}

        def fetch(_sym: str) -> Quote:
            calls["n"] += 1
            if calls["n"] <= 4:
                return make_quote(symbol=_sym)
            return Quote(
                spot_price=Decimal("100"), perp_price=Decimal("100"),
                ts_ms=int(NOW * 1000), symbol=_sym, source="",  # 未知来源
            )

        svc._quote_fetcher = fetch  # noqa: SLF001
        r = svc.run_once()
        assert env["executor"].open_calls == []
        assert _plan_payloads(env["store"]) == []
        assert any("入场报价被拒" in s for s in r["skipped"]), r["skipped"]
