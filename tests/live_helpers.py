"""实时 Demo 交易（开发文档 §7）测试辅助：LiveStrategy / LiveService 装配。

所有测试不依赖网络：策略数据源、账户快照、行情、执行器全部用 fake。
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from cointrader.config import (
    ApiConfig,
    BacktestConfig,
    Config,
    CostsConfig,
    DataConfig,
    EntryConfig,
    ExecutionConfig,
    ExitConfig,
    LoggingConfig,
    RiskConfig,
    SelectionConfig,
    StrategyConfig,
)
from cointrader.execution.models import PairExecution, ReconciliationResult
from cointrader.execution.risk_gate import HaltState
from cointrader.execution.rules import SymbolRules
from cointrader.execution.store import StateStore
from cointrader.live.account_state import AccountStateBuilder
from cointrader.live.strategy import HeldPosition, LiveContext, LiveStrategy, Quote
from conftest import FakeStrategyData

NOW = 1_800_000_000.0  # 固定逻辑时钟（秒）


def make_live_config(
    live_symbols: tuple[str, ...] = ("BTCUSDT",),
    entry_overrides: dict[str, Any] | None = None,
    exit_overrides: dict[str, Any] | None = None,
    selection_overrides: dict[str, Any] | None = None,
    exec_overrides: dict[str, Any] | None = None,
) -> Config:
    """默认 8h 周期下的紧凑阈值：回看 4 期、连续 3 期、年化 0.30（≈0.000274/期）。"""
    entry: dict[str, Any] = dict(min_annualized_rate=0.30, min_consecutive_positive=3,
                 lookback_periods=4, min_trailing_annualized=0.30)
    entry.update(entry_overrides or {})
    exit_: dict[str, Any] = dict(exit_lookback_periods=6, max_holding_periods=10)
    exit_.update(exit_overrides or {})
    selection: dict[str, Any] = dict(max_positions=2, per_position_weight=0.3,
                     min_quote_volume_3d_avg=1_000_000)
    selection.update(selection_overrides or {})
    return Config(
        data=DataConfig(),
        costs=CostsConfig(),
        backtest=BacktestConfig(),
        strategy=StrategyConfig(
            entry=EntryConfig(**entry),
            exit=ExitConfig(**exit_),
            selection=SelectionConfig(**selection),
        ),
        risk=RiskConfig(),
        api=ApiConfig(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(live_symbols=tuple(live_symbols), **(exec_overrides or {})),
    )


def make_quote(spot: str = "100", perp: str = "100", ts: float | None = None) -> Quote:
    return Quote(
        spot_price=Decimal(spot),
        perp_price=Decimal(perp),
        ts_ms=int((ts or NOW) * 1000),
    )


def make_held(symbol: str = "BTCUSDT", *, age_periods: int = 0,
              interval_ms: int = 8 * 3600 * 1000) -> HeldPosition:
    opened_ms = int(NOW * 1000) - age_periods * interval_ms
    return HeldPosition(
        symbol=symbol,
        spot_qty=Decimal("0.01"),
        perp_qty=Decimal("-0.01"),
        opened_ms=opened_ms,
    )


def make_strategy(tmp_path: Path, data: FakeStrategyData, *,
                  config: Config | None = None,
                  strategy_version: str = "strat-test-v1",
                  now_fn: Any = None) -> tuple[LiveStrategy, StateStore]:
    store = StateStore(tmp_path / "trading.sqlite3")
    cfg = config or make_live_config(live_symbols=tuple(data._rates.keys()) or ("BTCUSDT",))  # noqa: SLF001
    strat = LiveStrategy(
        config=cfg,
        data=data,
        store=store,
        strategy_version=strategy_version,
        config_hash="cfg-test",
        now_fn=now_fn or (lambda: NOW),
    )
    return strat, store


def make_context(held: dict[str, HeldPosition] | None = None,
                 quotes: dict[str, Quote] | None = None,
                 total_capital: Decimal = Decimal("10000"),
                 run_id: str = "run-test",
                 now_ms: int | None = None) -> LiveContext:
    return LiveContext(
        now_ms=int((now_ms or NOW) * 1000),
        run_id=run_id,
        total_capital=total_capital,
        held=held or {},
        quotes=quotes or {},
    )


# ---------------------------------------------------------------------------
# 服务级 fake（开发文档 §7.7 全流程）
# ---------------------------------------------------------------------------


def make_symbol_rules(market: str) -> SymbolRules:
    return SymbolRules(
        symbol="BTCUSDT",
        market=market,
        status="TRADING",
        base_asset="BTC",
        quote_asset="USDT",
        tick_size=Decimal("0.1"),
        min_qty=Decimal("0.001"),
        max_qty=Decimal("100"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("1"),
    )


class FakeServiceAdapter:
    """Spot/Futures 账户接口 fake（余额 + 持仓 + 规则 + 杠杆）。"""

    def __init__(self, market: str) -> None:
        self.market = market
        self.balances_map: dict[str, Decimal] = {}
        self.position_amt: Decimal = Decimal("0")
        self._rule = make_symbol_rules(market)

    def rule(self, symbol: str) -> SymbolRules:  # noqa: ARG002
        return self._rule

    def account(self) -> dict[str, Any]:
        if self.market == "spot":
            assets: dict[str, Decimal] = {"USDT": Decimal("9999")}
            assets.update(self.balances_map)
            return {
                "balances": [
                    {"asset": asset, "free": str(v), "locked": "0"}
                    for asset, v in assets.items()
                ]
            }
        return {
            "totalWalletBalance": "10000",
            "availableBalance": "9999",
            "totalUnrealizedProfit": "0",
        }

    def positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if self.market == "spot":
            return []
        if self.position_amt == 0:
            return []
        return [
            {"symbol": "BTCUSDT", "positionSide": "BOTH",
             "positionAmt": str(self.position_amt), "entryPrice": "100"}
        ]

    def balances(self) -> dict[str, Decimal]:
        if self.market == "spot":
            return dict(self.balances_map)
        return {
            "USDT": Decimal("9999"),
            "USDT_available": Decimal("9999"),
            "futures_wallet": Decimal("10000"),
            "futures_available_balance": Decimal("9999"),
            "futures_unrealized_pnl": Decimal("0"),
            "futures_cross_pnl": Decimal("0"),
        }

    def position_qty(self, symbol: str) -> Decimal:  # noqa: ARG002
        return self.position_amt

    def leverage_and_margin(self, symbol: str) -> tuple[int, str]:  # noqa: ARG002
        return 1, "isolated"

    def ensure_leverage_and_margin(self, symbol: str, lev: int, margin: str) -> None:
        pass

    def mark_price(self, symbol: str) -> Decimal:  # noqa: ARG002
        return Decimal("100")


class FakeGate:
    """与 RiskGate 相同属性面的假闸门（恒 NORMAL）。"""

    def __init__(self) -> None:
        self.state = HaltState.NORMAL

    def reconcile(self, **kw: Any) -> ReconciliationResult:
        return _ok_recon()

    def recover(self, reconciliation_ok: bool, preflight_ok: bool) -> None:
        pass

    def halt(self, reason: str) -> None:
        self.state = HaltState.HALT_NEW_RISK


def _ok_recon() -> ReconciliationResult:
    return ReconciliationResult(
        ts_ms=int(NOW * 1000), consistent=True, mismatches=(), repaired=(), can_open=True,
    )


class FakeExecutor:
    """记录开/平仓调用的假执行器（写 pair 到 store，模拟真实状态机终态）。"""

    def __init__(self, store: StateStore) -> None:
        self.store = store
        self.open_calls: list[dict[str, Any]] = []
        self.close_calls: list[dict[str, Any]] = []
        self.strategy_version = "strat-test-v1"
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return str(self._counter)

    def open_pair(self, symbol: str, notional: Decimal, *,
                  spot_price: Decimal, perp_price: Decimal, quote_ts_ms: int,
                  state: Any, reason: str = "", run_id: str = "",
                  signal_decision_id: str = "", decision_ts_ms: int = 0,
                  **kw: Any) -> PairExecution:
        ts = decision_ts_ms or int(NOW * 1000)
        pair = PairExecution(
            pair_execution_id=f"pair-open-{self._next_id()}",
            symbol=symbol,
            target_notional=notional,
            status="COMPLETE",
            kind="open",
            strategy_version=self.strategy_version,
            run_id=run_id,
            signal_decision_id=signal_decision_id,
            signal_ts_ms=ts,
            decision_ts_ms=ts,
            open_or_close_reason=reason,
            hedge_ratio=Decimal("1"),
            created_ms=ts,
            updated_ms=ts,
            completed_ts_ms=ts,
            actual_spot_notional=notional,
            actual_perp_notional=notional,
        )
        self.store.upsert_pair(pair)
        self.open_calls.append(dict(symbol=symbol, notional=notional, run_id=run_id))
        return pair

    def close_pair(self, symbol: str, *, reason: str = "", run_id: str = "",
                   signal_decision_id: str = "", decision_ts_ms: int = 0,
                   **kw: Any) -> PairExecution:
        ts = decision_ts_ms or int(NOW * 1000)
        pair = PairExecution(
            pair_execution_id=f"pair-close-{self._next_id()}",
            symbol=symbol,
            target_notional=Decimal("0"),
            status="COMPLETE",
            kind="close",
            strategy_version=self.strategy_version,
            run_id=run_id,
            signal_decision_id=signal_decision_id,
            decision_ts_ms=ts,
            open_or_close_reason=reason,
            created_ms=ts,
            updated_ms=ts,
            completed_ts_ms=ts,
        )
        self.store.upsert_pair(pair)
        self.close_calls.append(dict(symbol=symbol, reason=reason, run_id=run_id))
        return pair

    def reconcile(self) -> None:
        pass


class FakeReconciler:
    """恒一致的对账器（§7.7：对账通过才允许开仓/刷新持仓）。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, *, reason: str = "periodic") -> ReconciliationResult:
        self.calls.append(reason)
        return _ok_recon()


def make_service(tmp_path: Path, data: FakeStrategyData, *,
                 config: Config | None = None,
                 spot: FakeServiceAdapter | None = None,
                 futures: FakeServiceAdapter | None = None,
                 store: StateStore | None = None,
                 now_fn: Any = None) -> dict[str, Any]:
    """装配一个不联网的 LiveService（strategy 路径）。"""
    from cointrader.live.service import LiveService

    cfg = config or make_live_config(live_symbols=tuple(data._rates.keys()) or ("BTCUSDT",))  # noqa: SLF001
    store = store or StateStore(tmp_path / "trading.sqlite3")
    spot = spot or FakeServiceAdapter("spot")
    futures = futures or FakeServiceAdapter("perp")
    executor = FakeExecutor(store)
    reconciler = FakeReconciler()
    strategy, _ = make_strategy(tmp_path / "s", data, config=cfg)
    clock = now_fn or (lambda: NOW)
    builder = AccountStateBuilder(
        config=cfg,
        store=store,
        candidate_symbols=tuple(cfg.execution.live_symbols),
        now_fn=clock,
    )
    svc = LiveService(
        config=cfg,
        store=store,
        gate=FakeGate(),  # type: ignore[arg-type]  # 结构化 fake，接口一致
        executor=executor,  # type: ignore[arg-type]
        reconciler=reconciler,  # type: ignore[arg-type]
        strategy=strategy,
        account_builder=builder,
        now_fn=clock,
        spot=spot,  # type: ignore[arg-type]
        futures=futures,  # type: ignore[arg-type]
        quote_fetcher=lambda _sym: make_quote(),
        on_alert=lambda kind, msg: None,
    )
    # 测试直接驱动 run_once：跳过 start() 的完整预检，置为 RUNNING
    from cointrader.live.service import ServiceState

    svc._state = ServiceState.RUNNING  # noqa: SLF001
    svc.run_id = "run-default"
    return {"svc": svc, "spot": spot, "futures": futures, "executor": executor,
            "store": store, "config": cfg, "reconciler": reconciler}
