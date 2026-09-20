"""实盘编排层单元测试（开发设计文档 §4.2 / §8 / §11.1）。

覆盖：Signal 生成规则、启动顺序（时钟/杠杆/对账/单实例锁）、
RECOVERY 进入条件、SHADOW 不下单、HALT 语义。
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

import pytest

from cointrader.config import Config, ExecutionConfig
from cointrader.errors import ClockError, LiveGateBlocked
from cointrader.execution.reconcile import Reconciler
from cointrader.execution.store import StateStore
from cointrader.live.portfolio import build_signal
from cointrader.live.service import LiveService, ServiceState

NOW_MS = int(time.time() * 1000)


def make_config(**exc_overrides: Any) -> Config:
    from cointrader.config import (
        ApiConfig,
        BacktestConfig,
        CostsConfig,
        DataConfig,
        LoggingConfig,
        RiskConfig,
        StrategyConfig,
    )

    base = ExecutionConfig(**exc_overrides)
    return Config(
        data=DataConfig(),
        costs=CostsConfig(),
        backtest=BacktestConfig(),
        strategy=StrategyConfig(),
        risk=RiskConfig(),
        api=ApiConfig(),
        logging=LoggingConfig(),
        execution=base,
    )


class TestBuildSignal:
    def test_basic_signal(self) -> None:
        sig = build_signal(
            "BTCUSDT", spot_price=100, perp_price=100.5, quote_ts_ms=NOW_MS, now_ms=NOW_MS,
            requested_notional=50, config=make_config(canary_notional=200),
        )
        assert sig is not None
        assert sig.symbol == "BTCUSDT"
        assert sig.target_notional == Decimal("50")
        assert sig.basis_pct > 0, "永续溢价应为正基差"

    def test_notional_capped_at_canary(self) -> None:
        sig = build_signal(
            "BTCUSDT", spot_price=100, perp_price=100, quote_ts_ms=NOW_MS, now_ms=NOW_MS,
            requested_notional=5000, config=make_config(canary_notional=10),
        )
        assert sig is not None
        assert sig.target_notional == Decimal("10"), "名义额必须截断到 canary 上限"

    def test_stale_quote_rejected(self) -> None:
        with pytest.raises(LiveGateBlocked, match="过期"):
            build_signal(
                "BTCUSDT", spot_price=100, perp_price=100,
                quote_ts_ms=NOW_MS - 60_000, now_ms=NOW_MS, requested_notional=10,
                config=make_config(),
            )

    def test_future_quote_rejected(self) -> None:
        with pytest.raises(LiveGateBlocked):
            build_signal(
                "BTCUSDT", spot_price=100, perp_price=100,
                quote_ts_ms=NOW_MS + 60_000, now_ms=NOW_MS, requested_notional=10,
                config=make_config(),
            )

    def test_deep_discount_skipped(self) -> None:
        """深度贴水（perp < spot 超容差）：开空头不利，跳过。"""
        sig = build_signal(
            "BTCUSDT", spot_price=100, perp_price=98, quote_ts_ms=NOW_MS, now_ms=NOW_MS,
            requested_notional=10, config=make_config(hedge_tolerance_pct=0.005),
        )
        assert sig is None

    def test_invalid_inputs(self) -> None:
        cfg = make_config()
        assert build_signal("BTCDOM", spot_price=100, perp_price=100, quote_ts_ms=NOW_MS, now_ms=NOW_MS,
                            requested_notional=10, config=cfg) is None
        assert build_signal("BTCUSDT", spot_price=0, perp_price=100, quote_ts_ms=NOW_MS, now_ms=NOW_MS,
                            requested_notional=10, config=cfg) is None
        assert build_signal("BTCUSDT", spot_price=100, perp_price=100, quote_ts_ms=NOW_MS, now_ms=NOW_MS,
                            requested_notional=0, config=cfg) is None


# ---------------------------------------------------------------------------
# LiveService
# ---------------------------------------------------------------------------


class FakeServiceAdapter:
    """服务层 fake：calibrate/load_rules/leverage_and_margin + 对账接口。"""

    def __init__(self, *, time_offset: int = 50, leverage: int = 1, margin: str = "isolated",
                 symbols: tuple[str, ...] = ("BTCUSDT",)) -> None:
        self._time_offset = time_offset
        self._leverage = leverage
        self._margin = margin
        self._symbols = symbols
        self.open_orders_list: list[dict[str, Any]] = []
        self.order_results: dict[str, Any] = {}
        self.balances_map: dict[str, Decimal] = {}
        self.positions_list: list[dict[str, Any]] = []
        self.calibrated = 0

    def calibrate(self) -> int:
        self.calibrated += 1
        return self._time_offset

    def load_rules(self) -> dict[str, Any]:
        return {s: object() for s in self._symbols}

    def leverage_and_margin(self, symbol: str) -> tuple[int, str]:  # noqa: ARG002
        return self._leverage, self._margin

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:  # noqa: ARG002
        return list(self.open_orders_list)

    def query_by_client_order_id(self, symbol: str, client_order_id: str) -> Any:  # noqa: ARG001
        return self.order_results.get(client_order_id)

    def balances(self) -> dict[str, Decimal]:
        return dict(self.balances_map)

    def positions(self, symbol: str | None = None) -> list[dict[str, Any]]:  # noqa: ARG002
        return list(self.positions_list)


class FakeStream:
    def __init__(self, market: str, fresh: bool = True) -> None:
        self.market = market
        self._fresh = fresh
        self.generation = 1
        self.started = False
        self.stopped = False

    @property
    def is_fresh(self) -> bool:
        return self._fresh

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakePair:
    def __init__(self, status: str = "COMPLETE", error: str = "") -> None:
        self.status = status
        self.error = error
        self.pair_execution_id = "pair-fake"


class FakeExecutor:
    def __init__(self, status: str = "COMPLETE") -> None:
        self.status = status
        self.calls: list[dict[str, Any]] = []

    def open_pair(self, symbol: str, notional: Decimal, **kw: Any) -> FakePair:
        self.calls.append({"symbol": symbol, "notional": notional, **kw})
        return FakePair(self.status)


@pytest.fixture
def service_env(tmp_path) -> dict[str, Any]:
    from cointrader.execution.risk import RiskManager
    from cointrader.execution.risk_gate import RiskGate

    store = StateStore(tmp_path / "trading.sqlite3")
    spot = FakeServiceAdapter()
    perp = FakeServiceAdapter()
    gate = RiskGate(RiskManager(make_config().risk))
    executor = FakeExecutor()
    reconciler = Reconciler(store, spot, perp)  # type: ignore[arg-type]
    alerts: list[tuple[str, str]] = []
    service = LiveService(
        config=make_config(),
        spot=spot,  # type: ignore[arg-type]
        futures=perp,  # type: ignore[arg-type]
        store=store,
        gate=gate,
        executor=executor,  # type: ignore[arg-type]
        reconciler=reconciler,
        on_alert=lambda k, m: alerts.append((k, m)),
    )
    return {"store": store, "spot": spot, "perp": perp, "gate": gate,
            "executor": executor, "service": service, "alerts": alerts}


class TestStartup:
    def test_happy_path_reaches_running(self, service_env: dict) -> None:
        service: LiveService = service_env["service"]
        s_spot, s_perp = FakeStream("spot"), FakeStream("perp")
        service.attach_streams([s_spot, s_perp])  # type: ignore[list-item]  # 结构化 fake
        report = service.startup()
        assert service.state is ServiceState.RUNNING
        assert report.reconciliation_consistent is True
        assert report.symbols == ("BTCUSDT",)
        assert s_spot.started and s_perp.started, "对账通过后才启动用户流"

    def test_clock_offset_exceeds_limit_fails(self, service_env: dict) -> None:
        service: LiveService = service_env["service"]
        service_env["spot"]._time_offset = 9999  # noqa: SLF001
        service.config = make_config(server_time_offset_limit_ms=3000)
        with pytest.raises(ClockError, match="偏移"):
            service.startup()
        # 失败后锁必须释放
        lease = service.store.acquire_lease("live-executor", holder="other")
        assert lease.holder == "other"

    def test_leverage_mismatch_refuses_startup(self, service_env: dict) -> None:
        service: LiveService = service_env["service"]
        service.config = make_config(leverage=2)
        with pytest.raises(LiveGateBlocked, match="杠杆"):
            service.startup()

    def test_no_common_symbols_refuses_startup(self, service_env: dict) -> None:
        service: LiveService = service_env["service"]
        service_env["perp"]._symbols = ("ETHUSDT",)  # noqa: SLF001
        with pytest.raises(LiveGateBlocked, match="共同"):
            service.startup()

    def test_lease_conflict_refuses_startup(self, service_env: dict) -> None:
        service: LiveService = service_env["service"]
        service.store.acquire_lease("live-executor", holder="other-process")
        with pytest.raises(LiveGateBlocked, match="锁"):
            service.startup()
        kinds = [k for k, _ in service_env["alerts"]]
        assert "LEASE_CONFLICT" in kinds

    def test_startup_reconciliation_mismatch_enters_recovery(self, service_env: dict) -> None:
        service: LiveService = service_env["service"]
        # 交易所存在本地未知开放订单
        service_env["spot"].open_orders_list.append(
            {"clientOrderId": "ct-ghost", "symbol": "BTCUSDT", "origQty": "1", "type": "MARKET"}
        )
        s_spot, s_perp = FakeStream("spot"), FakeStream("perp")
        service.attach_streams([s_spot, s_perp])  # type: ignore[list-item]  # 结构化 fake
        report = service.startup()
        assert service.state is ServiceState.RECOVERY
        assert report.reconciliation_consistent is False
        kinds = [k for k, _ in service_env["alerts"]]
        assert "RECOVERY" in kinds


class TestRunLoop:
    def _started(self, service_env: dict) -> LiveService:
        service: LiveService = service_env["service"]
        service.attach_streams([FakeStream("spot"), FakeStream("perp")])  # type: ignore[list-item]
        service.startup()
        return service

    def test_running_executes_signals(self, service_env: dict) -> None:
        service = self._started(service_env)
        sig = build_signal("BTCUSDT", spot_price=100, perp_price=100.1,
                           quote_ts_ms=NOW_MS, now_ms=NOW_MS, requested_notional=10, config=service.config)
        assert sig is not None
        service._signal_provider = lambda: [sig]  # type: ignore[method-assign]  # noqa: SLF001
        result = service.run_once()
        assert result["state"] == "RUNNING"
        assert result["opened"] == ["BTCUSDT"]
        assert len(service_env["executor"].calls) == 1

    def test_shadow_mode_records_but_never_orders(self, service_env: dict) -> None:
        service = self._started(service_env)
        service.config = make_config(mode="shadow")
        service.mode = "shadow"
        sig = build_signal("BTCUSDT", spot_price=100, perp_price=100.1,
                           quote_ts_ms=NOW_MS, now_ms=NOW_MS, requested_notional=10, config=service.config)
        assert sig is not None
        service._signal_provider = lambda: [sig]  # type: ignore[method-assign]  # noqa: SLF001
        result = service.run_once()
        assert result["state"] == "RUNNING"
        assert service_env["executor"].calls == [], "SHADOW 模式禁止下单"
        kinds = [k for k, _ in service_env["alerts"]]
        assert "SHADOW_INTENT" in kinds

    def test_stale_stream_refuses_startup(self, service_env: dict) -> None:
        """启动前流未新鲜 → 拒绝启动（避免首轮误入不可自动解除的 RECOVERY）。"""
        service: LiveService = service_env["service"]
        s_spot = FakeStream("spot", fresh=False)
        service.attach_streams([s_spot, FakeStream("perp")])  # type: ignore[list-item]
        service.fresh_wait_seconds = 0.2
        with pytest.raises(LiveGateBlocked, match="新鲜"):
            service.startup()

    def test_stale_stream_after_startup_enters_recovery(self, service_env: dict) -> None:
        service: LiveService = service_env["service"]
        s_spot = FakeStream("spot")
        service.attach_streams([s_spot, FakeStream("perp")])  # type: ignore[list-item]
        service.fresh_wait_seconds = 0.2
        service.startup()
        s_spot._fresh = False  # noqa: SLF001
        result = service.run_once()
        assert service.state is ServiceState.RECOVERY
        assert result["state"] == "RECOVERY"

    def test_halted_gate_blocks_open(self, service_env: dict) -> None:
        service = self._started(service_env)
        service.gate.halt("incident")
        sig = build_signal("BTCUSDT", spot_price=100, perp_price=100.1,
                           quote_ts_ms=NOW_MS, now_ms=NOW_MS, requested_notional=10, config=service.config)
        assert sig is not None
        service._signal_provider = lambda: [sig]  # type: ignore[method-assign]  # noqa: SLF001
        result = service.run_once()
        assert result["state"] == "HALTED"
        assert service_env["executor"].calls == [], "HALT 下禁止开仓"

    def test_recovered_gate_restores_running(self, service_env: dict) -> None:
        service = self._started(service_env)
        service.gate.halt("incident")
        service.run_once()
        service.gate.recover(reconciliation_ok=True, preflight_ok=True)
        result = service.run_once()
        assert result["state"] == "RUNNING"

    def test_open_failure_is_reported_not_retried(self, service_env: dict) -> None:
        service = self._started(service_env)
        service_env["executor"].status = "FAILED"
        sig = build_signal("BTCUSDT", spot_price=100, perp_price=100.1,
                           quote_ts_ms=NOW_MS, now_ms=NOW_MS, requested_notional=10, config=service.config)
        provider = {"calls": 0}

        def once() -> list:
            provider["calls"] += 1
            return [sig]

        service._signal_provider = once  # type: ignore[method-assign]  # noqa: SLF001
        result = service.run_once()
        assert result["state"] == "RUNNING"
        assert result["opened"] == []
        assert len(service_env["executor"].calls) == 1
        kinds = [k for k, _ in service_env["alerts"]]
        assert "OPEN_FAILED" in kinds

    def test_default_risk_state_unknown_capital_refuses_open(self, service_env: dict) -> None:
        """默认 RiskState 资金未知：即使走到 executor，preflight 也应拒绝。

        这里用真 PairExecutor + 假 adapter 太重，直接验证默认 risk_state_fn 语义：
        total_capital=0 → RiskManager.preflight 拒绝。
        """
        from cointrader.execution.risk import RiskManager

        service: LiveService = service_env["service"]
        state = service._default_risk_state()  # noqa: SLF001
        assert state.total_capital == 0.0
        verdict = RiskManager(make_config().risk).preflight(state)
        assert verdict.allowed is False, "资金未知时禁止开仓"


class TestStop:
    def test_stop_closes_streams_and_releases_lease(self, service_env: dict) -> None:
        service: LiveService = service_env["service"]
        s_spot, s_perp = FakeStream("spot"), FakeStream("perp")
        service.attach_streams([s_spot, s_perp])  # type: ignore[list-item]  # 结构化 fake
        service.startup()
        service.stop()
        assert s_spot.stopped and s_perp.stopped
        assert service.state is ServiceState.STOPPED
        # 锁已释放：其他进程可接管
        lease = service.store.acquire_lease("live-executor", holder="other")
        assert lease.holder == "other"

    def test_recovery_does_not_auto_recover(self, service_env: dict) -> None:
        """RECOVERY 不会自动消失：必须对账通过 + gate 恢复（§7.3 精神）。"""
        service = TestRunLoop()._started(service_env)
        service.enter_recovery("manual test")
        assert service.state is ServiceState.RECOVERY
        # 单纯再跑一轮不会自动恢复（用户流仍假设为新鲜但对账未过 —— 这里模拟对账失败）
        service_env["spot"].open_orders_list.append(
            {"clientOrderId": "ct-late-ghost", "symbol": "BTCUSDT", "origQty": "1", "type": "MARKET"}
        )
        # 强制触发对账（把 _last_reconcile_ms 清零）
        service._last_reconcile_ms = 0  # noqa: SLF001
        result = service.run_once()
        assert service.state is ServiceState.RECOVERY
        assert result["state"] == "RECOVERY"
