"""T4：应用编排（application/ledger/reconciliation）生命周期测试（AC-07/08）。

覆盖：
  1) LiveService.run_once/run_forever 委托到 ServiceRunner（行为等价，
     monkeypatch svc.run_once 的异常预算路径仍生效）；
  2) runner tick 体：闸门 HALT 同步与闸门恢复回 RUNNING；
  3) LedgerQueryService 只读 read model：tombstone 默认隐藏、跨重启
     读取一致 current projection、空状态集拒绝；
  4) ReconciliationFacade gate 组合：capture 失败/同步失败/对账失败 →
     ok=False，全通过 + gate_allows → ok=True；legacy 无同步器路径。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cointrader.execution.models import ReconciliationResult
from cointrader.execution.store import StateStore
from cointrader.execution.sync import SyncCaptureError
from cointrader.ledger import LedgerQueryService
from cointrader.live.service import ServiceState
from cointrader.reconciliation import GateOutcome, ReconciliationFacade
from live_helpers import (
    LiveFakeData,
    make_live_config,
    make_live_rates,
    make_service,
    make_symbol_rules,
)

SYMBOL = "BTCUSDT"


def _env(tmp_path: Path) -> dict[str, Any]:
    data = LiveFakeData({SYMBOL: make_live_rates(20, "0.0005")})
    return make_service(tmp_path, data)


def _attach_startup_fakes(svc) -> None:
    spot, futures = svc.spot, svc.futures
    spot.calibrate = lambda samples=5: 0  # type: ignore[method-assign]
    futures.calibrate = lambda samples=5: 0  # type: ignore[method-assign]
    spot.load_rules = lambda: {SYMBOL: make_symbol_rules("spot")}  # type: ignore[method-assign]
    futures.load_rules = lambda: {SYMBOL: make_symbol_rules("perp")}  # type: ignore[method-assign]


class TestRunnerDelegation:
    """run_once/run_forever 委托到 application runner（行为等价）。"""

    def test_run_once_delegates_to_runner(self, tmp_path: Path) -> None:
        svc = _env(tmp_path)["svc"]
        from cointrader.application.runner import ServiceRunner

        out = ServiceRunner(svc).run_once()
        assert isinstance(out, dict)
        assert "state" in out

    def test_monkeypatched_tick_still_governs_forever(self, tmp_path: Path) -> None:
        cfg = make_live_config()
        cfg = replace(cfg, execution=replace(cfg.execution, mode="testnet", max_consecutive_tick_errors=2))
        env = make_service(tmp_path, LiveFakeData({SYMBOL: make_live_rates(20, "0.0005")}), config=cfg)
        svc = env["svc"]
        alerts: list[str] = []
        svc._on_alert = lambda kind, msg: alerts.append(kind)  # noqa: SLF001
        calls = {"n": 0}

        def boom() -> None:
            calls["n"] += 1
            raise RuntimeError("tick 失败（假）")

        svc.run_once = boom  # type: ignore[method-assign]
        assert svc.run_forever(tick_seconds=0.001) is False
        assert calls["n"] == 2
        assert "TICK_LOOP_FAILURE" in alerts
        assert svc.state is ServiceState.STOPPED

    def test_normal_stop_returns_true(self, tmp_path: Path) -> None:
        env = _env(tmp_path)
        svc = env["svc"]
        ticks = {"n": 0}

        def tick() -> dict[str, Any]:
            ticks["n"] += 1
            return {"state": "RUNNING"}

        svc.run_once = tick  # type: ignore[method-assign]
        assert svc.run_forever(tick_seconds=0.001, stop_check=lambda: ticks["n"] >= 3) is True
        assert ticks["n"] >= 3


class TestRecoveryStuckEscalation:
    """RECOVERY 卡死升级：启动后（曾达 RUNNING）持续 RECOVERY 超过
    ``recovery_escalation_seconds`` → 优雅停机退出码 1，交 systemd 重启。

    背景缺陷：stale 用户流分支每 tick 提前 return，自动解除分支永远跑不到；
    进程会无限等待且 can_open=0（VPS 实测卡 14.5h）。升级后重启重新预检+对账。
    启动阶段的长 RECOVERY（补账/epoch 构建）不计入，不得误杀。
    """

    def _env(self, tmp_path: Path, escalation: float = 1.0) -> Any:
        cfg = make_live_config()
        cfg = replace(
            cfg, execution=replace(cfg.execution, recovery_escalation_seconds=escalation)
        )
        return make_service(
            tmp_path, LiveFakeData({SYMBOL: make_live_rates(20, "0.0005")}), config=cfg
        )["svc"]

    def test_elapsed_tracking_across_transitions(self, tmp_path: Path) -> None:
        svc = self._env(tmp_path)
        # 初始 RECOVERY（从未 RUNNING）：慢启动合法，不计时
        assert svc.recovery_elapsed_ms() is None
        svc.resume_after_checks("测试预检通过", source="test")
        assert svc._state is ServiceState.RUNNING  # noqa: SLF001
        assert svc.recovery_elapsed_ms() is None
        svc.enter_recovery("用户流不新鲜（卡死场景）")
        assert svc._state is ServiceState.RECOVERY  # noqa: SLF001
        assert svc.recovery_elapsed_ms() is not None
        svc.resume_after_checks("恢复", source="test")
        assert svc.recovery_elapsed_ms() is None

    def test_stuck_recovery_escalates_to_stop(self, tmp_path: Path) -> None:
        svc = self._env(tmp_path, escalation=1.0)
        alerts: list[tuple[str, str]] = []
        svc._on_alert = lambda kind, msg: alerts.append((kind, msg))  # noqa: SLF001
        svc.resume_after_checks("测试预检通过", source="test")
        svc.enter_recovery("用户流不新鲜/不可信: perp（卡死）")
        # 时间跳前 2s > 阈值 1s（白盒注入门控时间戳）
        svc._recovery_since_ms = int(svc._now() * 1000) - 2000  # noqa: SLF001
        svc.run_once = lambda: {"state": "RECOVERY"}  # type: ignore[method-assign]
        assert svc.run_forever(tick_seconds=0.001) is False
        assert any(k == "RECOVERY_STUCK" for k, _ in alerts)
        assert svc.state is ServiceState.STOPPED

    def test_short_recovery_not_escalated(self, tmp_path: Path) -> None:
        svc = self._env(tmp_path, escalation=900.0)
        svc.resume_after_checks("测试预检通过", source="test")
        svc.enter_recovery("瞬时用户流抖动")
        svc.run_once = lambda: {"state": "RECOVERY"}  # type: ignore[method-assign]
        ticks = {"n": 0}

        def stop_check() -> bool:
            ticks["n"] += 1
            return ticks["n"] >= 3

        assert svc.run_forever(tick_seconds=0.001, stop_check=stop_check) is True
        assert ticks["n"] >= 3
        assert svc.state is not ServiceState.STOPPED

    def test_initial_recovery_before_running_not_escalated(self, tmp_path: Path) -> None:
        # 启动阶段长 RECOVERY（补账/epoch 构建合法）：未达过 RUNNING 不升级
        svc = self._env(tmp_path, escalation=1.0)
        svc.run_once = lambda: {"state": "RECOVERY"}  # type: ignore[method-assign]
        ticks = {"n": 0}

        def stop_check() -> bool:
            ticks["n"] += 1
            return ticks["n"] >= 3

        assert svc.run_forever(tick_seconds=0.001, stop_check=stop_check) is True
        assert svc.state is not ServiceState.STOPPED


class TestTickLifecycle:
    """runner tick 体：闸门 HALT 同步与恢复。"""

    def test_gate_halt_marks_service_halted(self, tmp_path: Path) -> None:
        svc = _env(tmp_path)["svc"]
        _attach_startup_fakes(svc)
        svc.startup()  # 失败会抛异常，能走到这里 = 启动成功
        svc.gate.halt("测试停机")
        out = svc.run_once()
        assert out["state"] == "HALTED"
        assert svc.state is ServiceState.HALTED

    def test_gate_recovery_returns_to_running(self, tmp_path: Path) -> None:
        svc = _env(tmp_path)["svc"]
        _attach_startup_fakes(svc)
        svc.startup()
        svc.gate.halt("测试停机")
        svc.run_once()
        # FakeGate.recover 是 no-op；直接置 NORMAL 模拟真实 recover 成功
        from cointrader.execution.risk_gate import HaltState

        svc.gate.state = HaltState.NORMAL  # type: ignore[method-assign]
        out = svc.run_once()
        assert out["state"] == "RUNNING"


class TestLedgerQueryService:
    """只读 read model：tombstone 隐藏 + 跨重启一致。"""

    def test_tombstones_hidden_by_default(self, tmp_path: Path) -> None:
        svc = _env(tmp_path)["svc"]
        _attach_startup_fakes(svc)
        svc.startup()
        svc.run_once()  # 至少跑一轮（RUNNING tick 或受控 RECOVERY）

        q = LedgerQueryService(svc.store)
        visible = q.positions()
        all_rows = q.positions(include_tombstones=True)
        assert len(visible) <= len(all_rows)
        for row in visible:
            assert row["tombstone"] == 0

    def test_current_projection_survives_restart(self, tmp_path: Path) -> None:
        """重启后（新 store 句柄）读同一 DB：current projection 一致。"""
        env = _env(tmp_path)
        svc = env["svc"]
        _attach_startup_fakes(svc)
        svc.startup()
        before = LedgerQueryService(svc.store).positions()
        sessions_before = LedgerQueryService(svc.store).run_sessions()

        # 模拟重启：关闭旧句柄，重开同一 DB
        svc.stop()
        svc.store.close()
        reopened = StateStore(tmp_path / "trading.sqlite3")
        try:
            after = LedgerQueryService(reopened).positions()
            sessions_after = LedgerQueryService(reopened).run_sessions()
        finally:
            reopened.close()
        assert after == before
        assert len(sessions_after) == len(sessions_before)

    def test_empty_states_rejected(self, tmp_path: Path) -> None:
        svc = _env(tmp_path)["svc"]
        q = LedgerQueryService(svc.store)
        with pytest.raises(ValueError):
            q.orders_in_states(())


class _FakeSync:
    """ExchangeStateSynchronizer 的最小 fake。"""

    def __init__(self, *, capture_error: bool = False) -> None:
        self._capture_error = capture_error
        self.synced: list[list[str]] = []

    def capture(self):
        if self._capture_error:
            raise SyncCaptureError("capture 失败（假）")
        return SimpleNamespace(complete=True)

    def sync_fills(self, symbols: list[str]) -> list:
        self.synced.append(list(symbols))
        return [
            SimpleNamespace(market="spot", stream="fills", symbol=s, error=None, complete=True)
            for s in symbols
        ]

    def sync_funding_income(self, symbols: list[str]) -> list:
        return [
            SimpleNamespace(market="perp", stream="income", symbol=s, error=None, complete=True)
            for s in symbols
        ]


class _FakeReconciler:
    def __init__(self, can_open: bool = True) -> None:
        self._can_open = can_open

    def run(self, *, reason: str = "periodic", snapshot: Any = None) -> ReconciliationResult:
        mismatches = () if self._can_open else ("mismatch-1",)
        return ReconciliationResult(
            ts_ms=1,
            consistent=self._can_open,
            mismatches=mismatches,
            repaired=(),
            can_open=self._can_open,
        )


class TestReconciliationFacade:
    def test_all_good_gives_ok(self) -> None:
        sync = _FakeSync()
        rec = _FakeReconciler(can_open=True)
        facade = ReconciliationFacade(
            exch_sync=sync,  # type: ignore[arg-type]
            reconciler=rec,  # type: ignore[arg-type]
            ledger_symbols_fn=lambda: ["BTCUSDT"],
            gate_allows=lambda: True,
        )
        out = facade.evaluate(reason="periodic")
        assert isinstance(out, GateOutcome)
        assert out.ok is True
        assert out.ledger_sync_ok is True
        assert out.ledger_sync_error == ""
        assert sync.synced == [["BTCUSDT"]]

    def test_capture_failure_fails_gate(self) -> None:
        sync = _FakeSync(capture_error=True)
        rec = _FakeReconciler(can_open=True)
        facade = ReconciliationFacade(
            exch_sync=sync,  # type: ignore[arg-type]
            reconciler=rec,  # type: ignore[arg-type]
            ledger_symbols_fn=lambda: ["BTCUSDT"],
        )
        out = facade.evaluate()
        assert out.ok is False
        assert "capture" in out.ledger_sync_error

    def test_mismatch_fails_gate(self) -> None:
        sync = _FakeSync()
        rec = _FakeReconciler(can_open=False)
        facade = ReconciliationFacade(
            exch_sync=sync,  # type: ignore[arg-type]
            reconciler=rec,  # type: ignore[arg-type]
        )
        out = facade.evaluate()
        assert out.ok is False
        assert out.result.mismatches

    def test_gate_allows_blocks_even_if_consistent(self) -> None:
        sync = _FakeSync()
        rec = _FakeReconciler(can_open=True)
        facade = ReconciliationFacade(
            exch_sync=sync,  # type: ignore[arg-type]
            reconciler=rec,  # type: ignore[arg-type]
            gate_allows=lambda: False,
        )
        assert facade.evaluate().ok is False

    def test_legacy_without_sync_is_ok(self) -> None:
        rec = _FakeReconciler(can_open=True)
        facade = ReconciliationFacade(exch_sync=None, reconciler=rec)  # type: ignore[arg-type]
        out = facade.evaluate()
        assert out.ok is True
        assert out.bundle is None
