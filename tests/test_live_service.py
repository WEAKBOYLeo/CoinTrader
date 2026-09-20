"""LiveService 全流程测试（开发文档 §7.7：完整周期 + 重启行为）。

固定费率场景：
  1) 正费率 → 开仓一次
  2) 持仓确认 → 重复信号不重复开仓
  3) 尾部费率转负 → 策略退出 → 平仓一次
  4) 平仓后持仓归零 → 不再开新仓
  5) 重启（新进程实例 + 同一账本）→ 持仓/决策链完整，不重复开仓
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from cointrader.live.decisions import DecisionKind
from conftest import FakeStrategyData
from live_helpers import NOW, NOW_MS, LiveFakeData, make_live_config, make_live_rates, make_service

SYMBOL = "BTCUSDT"
INTERVAL_MS = 8 * 3600 * 1000


def _neg_tail_rates(n: int = 20) -> list[tuple[int, Decimal, Decimal]]:
    """前 14 期正、最近 6 期负（退出窗口均值转负）。"""
    start = NOW_MS - 30 * 60 * 1000 - n * INTERVAL_MS
    values = ["0.0005"] * 14 + ["-0.0005"] * 6
    return [(start + i * INTERVAL_MS, Decimal(v), Decimal("100")) for i, v in enumerate(values)]


def _env(tmp_path: Path) -> dict[str, Any]:
    cfg = make_live_config()
    # 缩短周期，让测试时钟 31s 步进能触发对账/快照/候选刷新
    cfg = replace(cfg, execution=replace(cfg.execution,
                                         candidate_refresh_seconds=5,
                                         reconciliation_interval_seconds=5,
                                         snapshot_interval_seconds=5))
    data = LiveFakeData({SYMBOL: make_live_rates(20, "0.0005")})
    return make_service(tmp_path, data, config=cfg)


class TestServiceCycle:
    def test_full_open_hold_exit_close_restart_cycle(self, tmp_path):
        env = _env(tmp_path)
        svc = env["svc"]
        spot = env["spot"]
        futures = env["futures"]
        data: FakeStrategyData = env["svc"].strategy.data  # noqa: SLF001
        clock = {"t": NOW}
        svc._now = lambda: clock["t"]  # noqa: SLF001
        svc.strategy._now = lambda: clock["t"]  # noqa: SLF001  # 刷新超龄判断用策略时钟
        svc.run_id = "run-cycle"
        env["store"].start_run_session(
            run_id="run-cycle", started_ms=int(NOW * 1000), mode="demo",
            strategy_version="t", config_hash="h", code_revision="rev",
            spot_endpoint="e", futures_endpoint="f", user_stream_mode="poll",
        )

        # 1) 正费率 → 开仓一次
        r1 = svc.run_once()
        assert r1["state"] == "RUNNING"
        assert [c["symbol"] for c in env["executor"].open_calls] == [SYMBOL]
        assert env["executor"].close_calls == []

        # 2) 交易所确认持仓 → 重复信号不重复开仓
        spot.balances_map["BTC"] = Decimal("0.01")
        futures.position_amt = Decimal("-0.01")
        clock["t"] = NOW + 31
        r2 = svc.run_once()
        assert r2["state"] == "RUNNING"
        assert len(env["executor"].open_calls) == 1, f"不得重复开仓: {r2}"
        assert env["executor"].close_calls == []

        # 3) 尾部费率转负（新结算期到点后才可见）→ 策略退出 → 平仓一次
        data.set_rates(SYMBOL, _neg_tail_rates())
        clock["t"] = NOW + 8 * 3600 + 62
        r3 = svc.run_once()
        assert r3["state"] == "RUNNING"
        assert [c["symbol"] for c in env["executor"].close_calls] == [SYMBOL]
        close_reason = env["executor"].close_calls[0]["reason"]
        assert "NEGATIVE_EXIT_AVG" in close_reason

        # 4) 平仓成交 → 交易所持仓归零 → 下一轮不再开新仓
        spot.balances_map["BTC"] = Decimal("0")
        futures.position_amt = Decimal("0")
        clock["t"] = NOW + 8 * 3600 + 93
        r4 = svc.run_once()
        assert r4["state"] == "RUNNING"
        assert len(env["executor"].open_calls) == 1, f"平仓后费率仍为负，不得开新仓: {r4}"
        assert len(env["executor"].close_calls) == 1

        # 决策链完整：OPEN → HOLD → EXIT → SKIP（拒绝原因落盘）
        decisions = env["store"].signal_decisions(run_id="run-cycle")
        decisions = sorted(decisions, key=lambda d: d["ts_ms"])
        kinds = [d["decision_kind"] for d in decisions]
        assert kinds[0] == DecisionKind.OPEN
        assert DecisionKind.EXIT in kinds
        skips = [d for d in decisions if d["decision_kind"] == DecisionKind.SKIP]
        assert skips, "平仓后负费率候选应有 SKIP 决策并带原因码"
        assert all(d["reason_code"] for d in skips)

        # PnL 已归集（round trip 写入 pnl_ledger，带 run_id）
        ledger = env["store"].pnl_ledger(limit=100)
        assert any(row.get("run_id") == "run-cycle" for row in ledger)

        # 5) 重启：新 service 实例 + 同一账本/交易所（无持仓）→ 不重复开仓
        env2 = make_service(
            tmp_path / "restart", data,
            config=env["config"],
            spot=spot, futures=futures, store=env["store"],
        )
        svc2 = env2["svc"]
        svc2.run_id = "run-cycle-restart"
        r5 = svc2.run_once()
        assert r5["state"] == "RUNNING"
        assert env2["executor"].open_calls == [], "重启后费率仍为负，不得开新仓"
        pairs = env2["store"].pair_executions(symbol=SYMBOL, limit=100)
        assert [p["kind"] for p in pairs].count("open") == 1
        assert [p["kind"] for p in pairs].count("close") == 1

    def test_exit_failure_enters_recovery_and_no_new_risk(self, tmp_path):
        """平仓失败 → RECOVERY：禁止开新仓（不得带着未知状态继续）。"""
        env = _env(tmp_path)
        svc = env["svc"]
        spot = env["spot"]
        futures = env["futures"]
        svc.run_id = "run-exitfail"
        data: FakeStrategyData = svc.strategy.data  # noqa: SLF001

        r1 = svc.run_once()
        assert len(env["executor"].open_calls) == 1
        spot.balances_map["BTC"] = Decimal("0.01")
        futures.position_amt = Decimal("-0.01")

        data.set_rates(SYMBOL, _neg_tail_rates())
        clock_t = {"t": NOW}
        svc._now = lambda: clock_t["t"]  # noqa: SLF001
        svc.strategy._now = lambda: clock_t["t"]  # noqa: SLF001  # 刷新超龄判断用策略时钟
        clock_t["t"] = NOW + 8 * 3600 + 31  # 跨过结算周期，新（负）费率可见

        # 让平仓返回失败
        def failing_close(symbol: str, **kw: Any) -> Any:
            from cointrader.execution.models import PairExecution

            pair = PairExecution(
                pair_execution_id="pair-close-fail", symbol=symbol,
                target_notional=Decimal("0"), status="FAILED", kind="close",
                strategy_version="t", error="reduce-only rejected",
                created_ms=int(NOW * 1000), updated_ms=int(NOW * 1000),
            )
            env["executor"].close_calls.append(dict(symbol=symbol, reason=kw.get("reason", ""), run_id=kw.get("run_id", "")))
            return pair

        env["executor"].close_pair = failing_close  # type: ignore[method-assign]
        r2 = svc.run_once()
        assert r2["state"] == "RECOVERY"
        # RECOVERY 期间再跑：不开新仓
        r3 = svc.run_once()
        assert r3["state"] in ("RECOVERY", "HALTED")
        assert len(env["executor"].open_calls) == 1, "RECOVERY 禁止开新仓"
        _ = r1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))


class TestQuoteFetcher:
    def test_fetch_uses_premium_index_mark_price(self):
        """premiumIndex 无 lastPrice 字段；必须用 markPrice（回归 2026-09-19 实跑 KeyError）。"""
        from cointrader.live.service import _make_quote_fetcher

        class _FakePublic:
            def spot_price(self, symbol: str) -> str:  # noqa: ARG002
                return "81000.00"

            def premium_index(self, symbol: str) -> dict:  # noqa: ARG002
                # 真实 premiumIndex 响应字段（无 lastPrice）
                return {
                    "symbol": "BTCUSDT",
                    "markPrice": "81044.91",
                    "indexPrice": "81070.59",
                    "lastFundingRate": "0.00008906",
                    "time": 1789801254000,
                }

        fetch = _make_quote_fetcher(_FakePublic())
        quote = fetch("BTCUSDT")
        assert quote is not None
        assert quote.spot_price == Decimal("81000.00")
        assert quote.perp_price == Decimal("81044.91")

    def test_fetch_failure_returns_none(self):
        from cointrader.live.service import _make_quote_fetcher

        class _Boom:
            def spot_price(self, symbol: str):  # noqa: ARG002
                raise RuntimeError("api down")

        assert _make_quote_fetcher(_Boom())("BTCUSDT") is None


class TestTimeResync:
    def test_calibrate_uses_min_rtt_sample(self, tmp_path):
        """多点采样校准：取 RTT 最小样本（误差下界），不被抖动样本带偏。"""
        from cointrader.execution.spot import SpotAdapter

        # _now 返回秒（calibrate 内部 ×1000）；true offset = 100ms
        samples_now = iter([0.0, 0.3, 1.0, 1.1, 2.0, 2.05])
        servers: Iterator[int] = iter([250, 1200, 2125])

        class _FakeSpot(SpotAdapter):
            def server_time_ms(self) -> int:
                return next(servers)

        class _Client:
            time_offset_ms = 0

        adapter = _FakeSpot(client=_Client(), now_fn=lambda: next(samples_now))  # type: ignore[arg-type]
        # s1: rtt 300 → offset 100；s2: rtt 100 但时钟漂移样本 → offset 150；s3: rtt 50 → offset 100
        offset = adapter.calibrate(samples=3)
        assert offset == 100
        assert adapter.client.time_offset_ms == 100  # type: ignore[attr-defined]

    def test_periodic_resync_runs_and_updates_offset(self, tmp_path):
        env = _env(tmp_path)
        svc = env["svc"]
        spot, futures = env["spot"], env["futures"]
        calls = {"n": 0}

        def fake_calibrate(samples: int = 5) -> int:
            calls["n"] += 1
            return -120

        spot.calibrate = fake_calibrate  # type: ignore[method-assign]
        futures.calibrate = fake_calibrate  # type: ignore[method-assign]
        clock = {"t": NOW}
        svc._now = lambda: clock["t"]  # noqa: SLF001
        calls["n"] = 0
        clock["t"] = NOW + 400  # > time_resync_seconds=300
        r = svc.run_once()
        assert r["state"] == "RUNNING"
        assert calls["n"] == 2, "每周期应对 spot 与 perp 各重校准一次"

    def test_resync_offset_beyond_limit_enters_recovery(self, tmp_path):
        """重校准发现偏移超限（时钟真坏了）→ RECOVERY，禁止开新仓。"""
        env = _env(tmp_path)
        svc = env["svc"]

        def bad_calibrate(samples: int = 5) -> int:
            return -9000

        env["spot"].calibrate = bad_calibrate  # type: ignore[method-assign]
        env["futures"].calibrate = bad_calibrate  # type: ignore[method-assign]
        clock = {"t": NOW}
        svc._now = lambda: clock["t"]  # noqa: SLF001
        clock["t"] = NOW + 400
        r = svc.run_once()
        assert r["state"] == "RECOVERY"
        assert "偏移" in svc._recovery_reason  # noqa: SLF001


class TestT4WebSnapshot:
    """T4/AC-10：web_snapshot 带限流/市场就绪/新鲜度，且 JSON 可序列化。"""

    def test_web_snapshot_includes_t4_blocks_json_serializable(self, tmp_path):
        import json

        env = _env(tmp_path)
        snap = env["svc"].web_snapshot()
        assert "rate_limits" in snap and "market_data" in snap and "freshness" in snap
        assert snap["freshness"]["ledger_sync_ok"] is True
        assert "account_age_ms" in snap["freshness"]
        assert "heartbeat" not in str(type(snap))  # 无异常对象混入
        json.dumps(snap)  # 不含异常对象/凭据，全可序列化

    def test_web_snapshot_market_data_readiness(self, tmp_path):
        from cointrader.live.market_sync import DataReadiness, ScanEpochStatus

        env = _env(tmp_path)
        svc = env["svc"]

        class _StubSync:
            def readiness(self, now_ms: int | None = None) -> DataReadiness:
                return DataReadiness(
                    epoch_id="ep-1", status=ScanEpochStatus.READY,
                    expected=3, completed=3, excluded_count=0, failed_count=0,
                    missing=(), failed={}, age_ms=1000, reason="",
                )

            def latest_ready(self) -> object:
                return None

        svc.synchronizer = _StubSync()  # type: ignore[assignment]
        snap = svc.web_snapshot()
        assert snap["market_data"]["status"] == "READY"
        assert snap["market_data"]["can_rank"] is True
        assert snap["market_data"]["expected"] == 3
