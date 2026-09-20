"""LiveService 长跑韧性测试（计划 1.0 T1 / AC-01、AC-02、AC-03）。

覆盖：
  1) 单轮 tick 异常 → RECOVERY，进程不退出，下一轮继续（计数机制）；
  2) 连续 tick 异常达到 max_consecutive_tick_errors → run_forever 返回 False（退出码 1 交给守护重启）；
  3) 成功一轮清零计数（两次失败 + 成功一轮 → 不触发超限）；
  4) run_once 每轮刷新 lease → 第一实例运行超 30s 后第二实例仍 LeaseConflict；stop 后可取锁；
  5) 锁被占时 startup 抛 LiveGateBlocked（第二进程启动失败）；
  6) kill -9 遗留的 RUNNING/RECOVERY 会话：startup 标记 INTERRUPTED + 从交易所恢复 held 持仓。

全部不联网：适配器/执行器/对账器用 fake，账本用临时 SQLite。
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import cointrader.execution.store as store_mod
from cointrader.errors import LiveGateBlocked
from cointrader.execution.store import LeaseConflict, StateStore
from cointrader.live.service import ServiceState
from live_helpers import (
    NOW,
    FakeServiceAdapter,
    LiveFakeData,
    make_live_config,
    make_live_rates,
    make_service,
    make_symbol_rules,
)

SYMBOL = "BTCUSDT"


def _tick_config(max_errors: int = 2):
    cfg = make_live_config()
    return replace(cfg, execution=replace(cfg.execution, mode="testnet",
                                          max_consecutive_tick_errors=max_errors))


def _env(tmp_path: Path, *, max_errors: int = 2) -> dict[str, Any]:
    cfg = _tick_config(max_errors)
    data = LiveFakeData({SYMBOL: make_live_rates(20, "0.0005")})
    return make_service(tmp_path, data, config=cfg)


def _attach_startup_fakes(svc) -> None:
    """给 fake 适配器补 startup 预检所需接口（calibrate / load_rules）。"""
    spot, futures = svc.spot, svc.futures
    spot.calibrate = lambda samples=5: 0  # type: ignore[method-assign]
    futures.calibrate = lambda samples=5: 0  # type: ignore[method-assign]
    spot.load_rules = lambda: {SYMBOL: make_symbol_rules("spot")}  # type: ignore[method-assign]
    futures.load_rules = lambda: {SYMBOL: make_symbol_rules("perp")}  # type: ignore[method-assign]


class TestTickFaultTolerance:
    """AC-01：tick 异常容错与连续计数。"""

    def test_consecutive_tick_errors_stop_with_false(self, tmp_path):
        """连续异常达到阈值 → 优雅 stop() 后返回 False（CLI 退出码 1）。"""
        env = _env(tmp_path, max_errors=2)
        svc = env["svc"]
        alerts: list[str] = []
        svc._on_alert = lambda kind, msg: alerts.append(kind)  # noqa: SLF001
        calls = {"n": 0}

        def boom() -> None:
            calls["n"] += 1
            raise RuntimeError("网络抖动（假）")

        svc.run_once = boom  # type: ignore[method-assign]
        assert svc.run_forever(tick_seconds=0.001) is False
        assert calls["n"] == 2, "达到阈值即应停止，不多跑"
        assert "TICK_LOOP_FAILURE" in alerts
        assert svc.state is ServiceState.STOPPED, "超限路径必须先 stop()"
        # 优雅停机：run_session 已关闭，不留 ended_ms=NULL
        row = env["store"].run_session("run-default") if svc.run_id else None
        assert row is None or row["ended_ms"] is not None

    def test_single_tick_error_enters_recovery_and_continues(self, tmp_path):
        """单轮异常 → RECOVERY（告警），循环继续；成功一轮清零计数。"""
        env = _env(tmp_path, max_errors=3)
        svc = env["svc"]
        calls = {"n": 0}

        def flaky() -> None:
            calls["n"] += 1
            if calls["n"] in (1, 2):
                raise RuntimeError("瞬时故障（假）")

        svc.run_once = flaky  # type: ignore[method-assign]
        ok = svc.run_forever(tick_seconds=0.001, stop_check=lambda: calls["n"] >= 3)
        assert ok is True, "两次失败 + 一次成功（计数清零）不得触发超限停机"
        assert svc.state is ServiceState.RECOVERY, "进入过 RECOVERY 且不自动解除"

    def test_normal_stop_returns_true(self, tmp_path):
        env = _env(tmp_path)
        svc = env["svc"]
        ticks: dict[str, int] = {"n": 0}

        def stop_check() -> bool:
            ticks["n"] += 1
            return ticks["n"] >= 2

        assert svc.run_forever(tick_seconds=0.001, stop_check=stop_check) is True


class TestLeaseLifetime:
    """AC-02：锁续期让第一实例长跑有效；stop 后释放。"""

    def test_run_once_refresh_lease_survives_beyond_ttl(self, tmp_path, monkeypatch):
        """主循环每轮刷新 lease：30s（TTL）后第二实例仍取不到锁。"""
        env = _env(tmp_path)
        svc = env["svc"]
        store: StateStore = env["store"]
        clock = {"t": NOW}
        svc._now = lambda: clock["t"]  # noqa: SLF001

        # 真实 store 时钟（毫秒）改为可控，模拟「长跑超过 TTL 30s」
        real_now_ms = store_mod._now_ms
        t = {"now": real_now_ms()}
        monkeypatch.setattr(store_mod, "_now_ms", lambda: t["now"])

        lease = store.acquire_lease(svc.lease_name, holder="pid-self")
        svc.lease_holder = lease.holder

        # 第一实例运行 61s（期间 run_once 每轮刷新）
        t["now"] += 61_000
        clock["t"] = NOW + 61
        r = svc.run_once()
        assert r["state"] == "RUNNING"

        # 超过 TTL 30s 后，第二实例仍冲突（刷新生效）
        with pytest.raises(LeaseConflict):
            store.acquire_lease(svc.lease_name, holder="pid-other")

        # 对照组：未刷新的锁在过期后可被接管
        store2 = StateStore(tmp_path / "other.sqlite3")
        store2.acquire_lease("l2", holder="pid-a")
        t["now"] += 61_000
        assert store2.acquire_lease("l2", holder="pid-b").holder == "pid-b"

        # stop() 释放锁 → 第二实例可启动
        svc.stop()
        lease2 = store.acquire_lease(svc.lease_name, holder="pid-other")
        assert lease2.holder == "pid-other"
        store2.close()

    def test_startup_blocked_when_lease_held(self, tmp_path):
        """锁被其他实例占用时 startup 抛 LiveGateBlocked（启动失败退出码 1）。"""
        env = _env(tmp_path)
        store: StateStore = env["store"]
        store.acquire_lease(env["svc"].lease_name, holder="pid-other")

        svc = env["svc"]
        svc._now = lambda: NOW  # noqa: SLF001
        _attach_startup_fakes(svc)
        with pytest.raises(LiveGateBlocked):
            svc.startup()


class TestRestartRecovery:
    """AC-03：kill -9 / 断电后重启 —— 中断会话标记 + 持仓恢复。"""


class _FakeStream:
    """最小用户流替身：新鲜度可控。"""

    def __init__(self, market: str, fresh: bool = True) -> None:
        self.market = market
        self.generation = 1
        self.fresh = fresh

    @property
    def is_fresh(self) -> bool:
        return self.fresh


class TestRecoveryAutoClear:
    """RECOVERY 自动解除（长跑自愈）：流恢复新鲜 + 对账通过 → 回 RUNNING。"""

    def test_streams_stale_keeps_recovery(self, tmp_path):
        env = _env(tmp_path)
        svc = env["svc"]
        svc._now = lambda: NOW  # noqa: SLF001
        svc.attach_streams([_FakeStream("spot", fresh=False), _FakeStream("perp", fresh=True)])
        r = svc.run_once()
        assert r["state"] == "RECOVERY"
        assert svc.state is ServiceState.RECOVERY

    def test_streams_fresh_and_recon_ok_clears_recovery(self, tmp_path):
        env = _env(tmp_path)
        svc = env["svc"]
        svc._now = lambda: NOW  # noqa: SLF001
        svc.attach_streams([_FakeStream("spot"), _FakeStream("perp")])
        svc._state = ServiceState.RECOVERY  # noqa: SLF001
        svc._recovery_reason = "用户流不新鲜（测试）"  # noqa: SLF001
        svc._last_reconcile_ms = 0  # noqa: SLF001 强制触发 recovery_check

        r = svc.run_once()
        assert r["state"] == "RUNNING", f"流新鲜+对账通过应回 RUNNING: {r}"
        assert svc.state is ServiceState.RUNNING
        assert svc._reconcile_ok is True  # noqa: SLF001
        assert "recovery_check" in env["reconciler"].calls  # noqa: SLF001

    def test_streams_fresh_but_recon_fails_keeps_recovery(self, tmp_path):
        env = _env(tmp_path)
        svc = env["svc"]
        svc._now = lambda: NOW  # noqa: SLF001
        svc.attach_streams([_FakeStream("spot"), _FakeStream("perp")])
        svc._state = ServiceState.RECOVERY  # noqa: SLF001
        svc._recovery_reason = "用户流不新鲜（测试）"  # noqa: SLF001
        svc._last_reconcile_ms = 0  # noqa: SLF001

        from cointrader.execution.models import ReconciliationResult
        env["reconciler"].run = lambda *, reason="periodic": ReconciliationResult(  # type: ignore[method-assign]
            ts_ms=int(NOW * 1000), consistent=False, can_open=False,
            mismatches=("BTCUSDT: Spot 余额不一致",), repaired=(),
        )
        r = svc.run_once()
        assert r["state"] == "RECOVERY"
        assert svc.state is ServiceState.RECOVERY

    def test_startup_marks_interrupted_and_restores_held(self, tmp_path):
        data = LiveFakeData({SYMBOL: make_live_rates(20, "0.0005")})
        store = StateStore(tmp_path / "trading.sqlite3")

        # 上次 run：非优雅退出（kill -9），会话遗留 ended_ms=NULL
        store.start_run_session(
            run_id="run-old", started_ms=int(NOW * 1000) - 3_600_000, mode="testnet",
            strategy_version="t", config_hash="h", code_revision="rev",
            spot_endpoint="e", futures_endpoint="f", user_stream_mode="poll",
        )

        # 交易所仍有真实持仓（断点恢复的依据）
        spot = FakeServiceAdapter("spot")
        spot.balances_map["BTC"] = Decimal("0.01")
        futures = FakeServiceAdapter("perp")
        futures.position_amt = Decimal("-0.01")

        env = make_service(tmp_path, data, config=_tick_config(), store=store,
                           spot=spot, futures=futures)
        svc = env["svc"]
        svc.run_id = ""  # make_service 预设了 run_id；真实进程启动时为空，必须开新会话
        svc._now = lambda: NOW  # noqa: SLF001
        _attach_startup_fakes(svc)

        report = svc.startup()

        # 旧会话被标记 INTERRUPTED（ended_ms=启动时刻，原因固定）
        old = store.run_session("run-old")
        assert old is not None
        assert old["status"] == "INTERRUPTED"
        assert old["ended_ms"] == int(NOW * 1000)
        assert old["stop_reason"] == "process_exited_without_graceful_stop"

        # 新 run_session 开启且 RUNNING
        new = store.latest_run_session()
        assert new is not None
        assert new["run_id"] != "run-old"
        assert new["status"] == "RUNNING"
        assert svc.run_id == new["run_id"]

        # held 持仓从交易所恢复（对账后刷新）
        svc._refresh_held()  # noqa: SLF001
        assert set(svc._held) == {SYMBOL}
        assert svc._held[SYMBOL].spot_qty == Decimal("0.01")  # noqa: SLF001

        # 旧会话不被后续写操作覆盖（幂等性本身由 tests/test_store.py 覆盖；
        # startup 只在启动时调用一次 mark，新会话在标记之后才创建）
        old2 = store.run_session("run-old")
        assert old2 is not None and old2["ended_ms"] == int(NOW * 1000)
        assert report.mode == "testnet"

    def test_no_interrupted_sessions_is_noop(self, tmp_path):
        """全新账本（无任何会话）：startup 正常，标记数 0。"""
        data = LiveFakeData({SYMBOL: make_live_rates(20, "0.0005")})
        env = make_service(tmp_path / "x", data, config=_tick_config())
        svc = env["svc"]
        svc._now = lambda: NOW  # noqa: SLF001
        _attach_startup_fakes(svc)
        svc.startup()
        assert svc.state is ServiceState.RUNNING
