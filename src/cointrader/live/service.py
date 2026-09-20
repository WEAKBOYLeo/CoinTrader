"""实盘生命周期与主循环（开发设计文档 §4.2 / §8）。

启动顺序（§8.1，硬约束）：

1. 加载配置 → 2. 确定模式（默认 PAPER，LIVE 需双重开关）→
3. 校准 server time → 4. 加载规则 → 5. 账户/持仓快照与杠杆/保证金预检 →
6. 取单实例锁 → 7. 启动对账 → 8. 全部通过才启动用户数据流 →
9. 用户流稳定后才允许产生开仓意图。

任何对账不一致、用户流不可信、停机文件、风控停机 → 进入 RECOVERY：
禁止开新仓，只允许 reduce-only 平仓路径（由 ``RiskGate`` 放行）。
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

from ..config import Config
from ..errors import BinanceError, ClockError, LiveGateBlocked
from ..execution.futures import FuturesAdapter
from ..execution.guard import AuditLog
from ..execution.metrics import Metrics
from ..execution.models import PairExecution, PositionSnapshot, new_id
from ..execution.pair_executor import PairExecutor
from ..execution.reconcile import Reconciler
from ..execution.risk import Position, RiskManager, RiskState
from ..execution.risk_gate import HaltState, RiskGate
from ..execution.spot import SpotAdapter
from ..execution.store import LeaseConflict, StateStore, StoreError
from ..execution.user_stream import PollingUserStream, UserStream
from ..reporting.pnl import PnlAggregator
from .account_state import AccountStateBuilder, AccountStateError, AccountStateResult
from .decisions import DecisionKind
from .portfolio import Signal
from .strategy import (
    HeldPosition,
    LiveContext,
    LiveStrategy,
    PublicDataStrategyProvider,
    Quote,
)

#: 用户流对象：WebSocket 流或 demo 用的 REST 轮询流（共享 market/is_fresh/start/stop 接口）
UserStreamLike = UserStream | PollingUserStream

logger = logging.getLogger(__name__)

__all__ = ["LiveService", "ServiceState", "StartupReport", "build_live_context"]


def _code_revision() -> str:
    """当前代码 revision（git HEAD）；不可用时返回 unknown。"""
    import shutil
    import subprocess  # 延迟导入，避免非 git 环境报错

    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        # 参数为常量（rev-parse HEAD），git 路径来自 shutil.which，无不可信输入
        proc = subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        revision = proc.stdout.strip()
        return revision or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


class ServiceState(str, Enum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    RECOVERY = "RECOVERY"
    HALTED = "HALTED"
    STOPPED = "STOPPED"


@dataclass(slots=True)
class StartupReport:
    """启动预检结果（审计用）。"""

    mode: str
    time_offset_spot_ms: int = 0
    time_offset_perp_ms: int = 0
    symbols: tuple[str, ...] = ()
    leverage: int = 0
    margin_type: str = ""
    reconciliation_consistent: bool = False
    reconciliation_mismatches: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


class LiveService:
    """实盘编排器。本类不直接调用 REST —— 全部经由 execution 层适配器。

    Args:
        config: 顶层配置（execution.mode 决定 SHADOW 是否只生成意图）。
        spot/futures: 市场适配器（真实或 fake）。
        store: 事件账本。
        gate: 风控闸门（开仓/减仓权限分离）。
        executor: 双腿执行器。
        reconciler: 对账器。
        lease_name: 单实例锁名（同一账户禁止两个执行进程）。
        risk_state_fn: 每轮构造 RiskState（默认：账本持仓 + 资金未知 → 禁止开仓）。
        signal_provider: 每轮返回 Signal 列表（研究层/组合层注入）。
    """

    def __init__(
        self,
        *,
        config: Config,
        spot: SpotAdapter,
        futures: FuturesAdapter,
        store: StateStore,
        gate: RiskGate,
        executor: PairExecutor,
        reconciler: Reconciler,
        streams: list[UserStreamLike] | None = None,
        lease_name: str = "live-executor",
        risk_state_fn: Callable[[], RiskState] | None = None,
        signal_provider: Callable[[], list[Signal]] | None = None,
        on_alert: Callable[[str, str], None] | None = None,
        now_fn: Callable[[], float] = time.time,
        fresh_wait_seconds: float = 30.0,
        strategy: LiveStrategy | None = None,
        account_builder: AccountStateBuilder | None = None,
        quote_fetcher: Callable[[str], Quote | None] | None = None,
        config_hash: str = "",
        code_revision: str = "",
        spot_endpoint: str = "",
        futures_endpoint: str = "",
    ) -> None:
        self.config = config
        self.mode = config.execution.mode
        self.fresh_wait_seconds = fresh_wait_seconds
        self.spot = spot
        self.futures = futures
        self.store = store
        self.gate = gate
        self.executor = executor
        self.reconciler = reconciler
        self.streams: list[UserStreamLike] = list(streams or [])
        self.lease_name = lease_name
        self.lease_holder: str | None = None
        self.metrics = Metrics()
        self.strategy = strategy
        self.account_builder = account_builder
        self._quote_fetcher = quote_fetcher or (lambda _s: None)
        self.config_hash = config_hash
        self.code_revision = code_revision or _code_revision()
        self.spot_endpoint = spot_endpoint
        self.futures_endpoint = futures_endpoint
        self.pnl = PnlAggregator(store, now_fn=now_fn)
        # 运行会话与实时状态（重启后从 DB/交易所恢复）
        self.run_id = ""
        self._submitted_this_run: set[str] = set()
        self._account_result: AccountStateResult | None = None
        self._held: dict[str, HeldPosition] = {}
        self._last_candidate_refresh_ms = 0
        self._last_snapshot_ms = 0
        self._reconcile_ok = False
        self._last_resync_ms = 0
        self._on_alert = on_alert or (lambda kind, msg: logger.critical("【告警】%s: %s", kind, msg))
        self._risk_state_fn = risk_state_fn or self._risk_state_from_account
        self._signal_provider = signal_provider or (lambda: [])
        self._now = now_fn
        self._state = ServiceState.STARTING
        self._recovery_reason = ""
        self._last_reconcile_ms = 0
        self._started = False

    # -- 状态 ---------------------------------------------------------------

    @property
    def state(self) -> ServiceState:
        if self.gate.state is not HaltState.NORMAL and self._state is ServiceState.RUNNING:
            return ServiceState.HALTED
        return self._state

    def enter_recovery(self, reason: str) -> None:
        """进入 RECOVERY：禁止开新仓。幂等。"""
        if self._state is ServiceState.RECOVERY:
            return
        if self._state is ServiceState.STOPPED:
            return
        self._state = ServiceState.RECOVERY
        self._recovery_reason = reason
        self._on_alert("RECOVERY", reason)
        self._persist_runtime_state()
        logger.error("【进入 RECOVERY】%s（禁止开新仓，等待对账通过后恢复）", reason)

    def web_snapshot(self) -> dict[str, Any]:
        """WebUI 用的内存状态快照（只读、无锁）。

        只读简单属性（GIL 原子读），不取任何写锁，不会阻塞/干扰主循环。
        """
        now_ms = int(self._now() * 1000)
        acct = self._account_result
        held = sorted(
            (
                {"symbol": h.symbol,
                 "spot_qty": str(h.spot_qty),
                 "perp_qty": str(h.perp_qty),
                 "opened_ms": h.opened_ms or None}
                for h in self._held.values()
            ),
            key=lambda r: str(r["symbol"]),
        )
        streams: list[dict[str, Any]] = []
        for s in self.streams:
            try:
                streams.append({
                    "market": str(getattr(s, "market", "?")),
                    "fresh": bool(getattr(s, "is_fresh", False)),
                    "generation": getattr(s, "generation", None),
                })
            except Exception:  # noqa: BLE001
                streams.append({"market": "?", "fresh": False, "generation": None})
        try:
            gate_state = self.gate.state.value
        except Exception:  # noqa: BLE001
            gate_state = None
        return {
            "state": self.state.value,
            "recovery_reason": self._recovery_reason,
            "run_id": self.run_id or None,
            "mode": self.mode,
            "can_open": self._reconcile_ok and self.state is ServiceState.RUNNING,
            "gate_state": gate_state,
            "held": held,
            "account": None if acct is None else {
                "total_capital": str(acct.total_capital),
                "spot_available": str(acct.spot_available),
                "futures_wallet": str(acct.futures_wallet),
                "futures_available": str(acct.futures_available),
                "futures_unrealized": str(acct.futures_unrealized),
            },
            "user_streams": streams,
            "last_reconcile_age_ms": (
                now_ms - self._last_reconcile_ms if self._last_reconcile_ms else None
            ),
            "code_revision": self.code_revision,
            "metrics": self.metrics.snapshot(),
        }

    def _check_leverage_margin(
        self, symbol: str, want_lev: int, want_margin: str
    ) -> tuple[int, str, bool]:
        """单个 symbol 的杠杆/保证金检查（开发文档 §7.6）。

        Returns:
            (leverage, margin_type, read_available)。
            Demo 无杠杆读接口（-5000）时：显式设置并采用，
            ``read_available=False`` —— 不能写成「读取验证成功」。
        """
        try:
            lev, margin = self.futures.leverage_and_margin(symbol)
            return int(lev), str(margin), True
        except BinanceError as exc:
            if exc.code == -5000:
                logger.warning(
                    "杠杆/保证金读接口不可用（binance_code=%s，symbol=%s），"
                    "已显式设置 (%d, %s) 并采用（未读回验证）",
                    exc.code, symbol, want_lev, want_margin,
                )
                self.futures.ensure_leverage_and_margin(symbol, want_lev, want_margin)
                return want_lev, want_margin, False
            raise

    def _ensure_run_session(self) -> None:
        """创建/恢复 run_session，并保存脱敏配置快照与 hash。"""
        if not self.config_hash and self.config.source_path is not None:
            try:
                self.config_hash = hashlib.sha256(
                    self.config.source_path.read_bytes()
                ).hexdigest()
            except OSError:
                self.config_hash = ""
        if self.run_id:
            return
        self.run_id = new_id("run")
        strategy_version = (
            self.strategy.strategy_version if self.strategy is not None else "funding_carry-1.0"
        )
        self.store.start_run_session(
            run_id=self.run_id,
            started_ms=int(self._now() * 1000),
            mode=self.mode,
            strategy_version=strategy_version,
            config_hash=self.config_hash,
            code_revision=self.code_revision,
            spot_endpoint=self.spot_endpoint,
            futures_endpoint=self.futures_endpoint,
            user_stream_mode=self.config.execution.user_stream_mode,
        )
        logger.info("【run_session 开始】run_id=%s config_hash=%s revision=%s",
                    self.run_id, self.config_hash, self.code_revision)

    # -- 账户状态 / 持仓 / 快照（§7.5 / §8.5） -------------------------------

    def _refresh_account_state(self, source: str) -> None:
        if self.account_builder is None:
            return
        result = self.account_builder.snapshot(
            spot=self.spot, futures=self.futures, source=source, run_id=self.run_id
        )
        self._account_result = result

    def _account_fresh(self, now_ms: int) -> bool:
        if self._account_result is None:
            return False
        stale_ms = int(self.config.execution.user_stream_fresh_seconds * 2000)
        return (now_ms - self._account_result.ts_ms) <= stale_ms

    def _risk_state_from_account(self) -> RiskState:
        """真实 RiskState（§7.5）。账户快照缺失/过期 → 资金未知（total_capital=0）→ 拒绝开仓。"""
        now_ms = int(self._now() * 1000)
        result = self._account_result
        if result is None or not self._account_fresh(now_ms):
            self.store.record_risk_decision(
                "ACCOUNT_STATE", False,
                "账户快照缺失/过期，资金未知，禁止开仓（不得用默认值放行）",
            )
            return RiskState(positions={}, total_capital=0.0)
        return result.state

    def _refresh_held(self) -> None:
        """以交易所对账后的真实持仓为准刷新 held 集合（§7.3）。"""
        try:
            spot_balances = self.spot.balances()
        except Exception as exc:  # noqa: BLE001
            self.store.record_risk_decision("HELD_STATE", False, f"现货余额查询失败: {exc}")
            return
        current: dict[str, HeldPosition] = {}
        for symbol in self.config.execution.live_symbols:
            base = symbol.replace("USDT", "")
            spot_qty = spot_balances.get(base, Decimal("0"))
            try:
                perp_qty = self.futures.position_qty(symbol)
            except Exception as exc:  # noqa: BLE001
                self.store.record_risk_decision("HELD_STATE", False, f"{symbol} 永续持仓查询失败: {exc}")
                continue
            if spot_qty > 0 or abs(perp_qty) > 0:
                current[symbol] = HeldPosition(
                    symbol=symbol,
                    spot_qty=spot_qty,
                    perp_qty=perp_qty,
                    opened_ms=self.store.position_opened_ms(symbol) or 0,
                )
        if current != self._held:
            if current:
                logger.info("【实际持仓】%s", {s: (p.spot_qty, p.perp_qty) for s, p in current.items()})
            self._held = current

    def _fetch_quotes(self) -> dict[str, Quote]:
        quotes: dict[str, Quote] = {}
        for symbol in self.strategy.candidate_symbols if self.strategy else ():
            quote = self._quote_fetcher(symbol)
            if quote is not None:
                quotes[symbol] = quote
        return quotes

    def _sample_position_snapshots(self, quotes: dict[str, Quote]) -> None:
        for symbol, held in self._held.items():
            quote = quotes.get(symbol)
            spot_price = quote.spot_price if quote else None
            perp_price = quote.perp_price if quote else None
            basis: Decimal | None = None
            if spot_price is not None and perp_price is not None and spot_price > 0:
                basis = (perp_price - spot_price) / spot_price
            self.store.record_position_snapshot(
                PositionSnapshot(
                    ts_ms=int(self._now() * 1000),
                    symbol=symbol,
                    spot_qty=held.spot_qty,
                    perp_qty=held.perp_qty,
                    spot_price=spot_price,
                    perp_price=perp_price,
                    basis_pct=basis,
                    source="periodic",
                    run_id=self.run_id,
                )
            )

    def _collect_funding_cashflows(self) -> None:
        """资金费结算窗口后拉取实际资金费并写账本（§8.5，幂等）。"""
        if self.strategy is None:
            return
        for symbol, held in self._held.items():
            cache = self.strategy.candidates.get(symbol)
            if cache is None or len(cache.rates) == 0:
                continue
            rows = self.store.funding_cashflows(symbol=symbol, limit=1000)
            last_ts = max((int(r.get("funding_ts_ms") or 0) for r in rows), default=0)
            now_ms = int(self._now() * 1000)
            open_pairs = [
                p for p in self.store.open_pairs()
                if str(p.get("symbol")) == symbol and str(p.get("kind")) == "open"
            ]
            pair_id = str(open_pairs[0]["pair_execution_id"]) if open_pairs else None
            short_qty = abs(held.perp_qty)
            if short_qty <= 0:
                continue
            for ts, rate, mark in zip(
                cache.timestamps, cache.rates, cache.mark_prices, strict=False
            ):
                if ts <= last_ts or ts > now_ms:
                    continue
                if mark <= 0:
                    continue
                # 我们是永续空头：收到 = -rate × 空单名义额（rate>0 → 正收益）
                amount = -rate * short_qty * mark
                added = self.store.record_funding_cashflow(
                    cashflow_id=new_id("fcf"),
                    symbol=symbol,
                    funding_ts_ms=ts,
                    funding_rate=rate,
                    interval_hours=cache.interval_hours,
                    amount=amount.quantize(Decimal("0.000001")),
                    source="funding_history",
                    run_id=self.run_id,
                    pair_execution_id=pair_id,
                    raw_summary=(
                        f'{{"mark_price": "{mark}", "short_qty": "{short_qty}", '
                        f'"price_source": "funding_history.mark_price"}}'
                    ),
                )
                if added:
                    logger.info("【资金费入账】%s ts=%s rate=%s amount=%s", symbol, ts, rate, amount)

    def _post_close(self, symbol: str, pair: PairExecution) -> None:
        """平仓完成后：PnL 归集 + 再次对账确认两腿归零（§7.4）。"""
        try:
            close_row = self.store.get_pair(pair.pair_execution_id)
            opens = [
                r for r in self.store.pair_executions(symbol=symbol, limit=1000)
                if str(r.get("kind")) == "open"
            ]
            open_row = opens[-1] if opens else None
            if close_row is not None:
                self.pnl.persist_round_trip(open_row, close_row)
            result = self.reconciler.run(reason="post_close")
            self._last_reconcile_ms = int(self._now() * 1000)
            if not result.consistent:
                self.enter_recovery(f"平仓后对账不一致: {list(result.mismatches)}")
        except StoreError as exc:
            self._on_alert("POST_CLOSE_FAILED", f"{symbol}: {exc}")
            self.enter_recovery(f"平仓后 PnL 归集/对账失败: {exc}")

    def _persist_runtime_state(self) -> None:
        try:
            self.store.set_runtime_state("run_id", self.run_id)
            self.store.set_runtime_state("service_state", self.state.value)
            self.store.set_runtime_state("recovery_reason", self._recovery_reason)
            self.store.set_runtime_state("last_reconcile_ms", str(self._last_reconcile_ms))
            self.store.set_runtime_state(
                "account_snapshot_ms", str(self._account_result.ts_ms if self._account_result else 0)
            )
            self.store.set_runtime_state("can_open", "1" if (self._reconcile_ok and self._state is ServiceState.RUNNING) else "0")
            self.store.set_runtime_state("mode", self.mode)
            if self._account_result is not None:
                self.store.set_runtime_state(
                    "total_capital", str(self._account_result.total_capital)
                )
            if self.run_id:
                self.store.update_run_session(self.run_id, status=self.state.value)
        except StoreError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("运行时状态写账本失败", exc_info=True)

    # -- 启动（§8.1） --------------------------------------------------------

    def startup(self) -> StartupReport:
        """按 §8.1 顺序执行启动预检。失败抛异常（启动时崩溃是特性）。"""
        report = StartupReport(mode=self.mode)

        # 6. 单实例锁（先取锁，防止对账期间另一进程写账本）
        try:
            lease = self.store.acquire_lease(self.lease_name)
        except LeaseConflict as exc:
            self._on_alert("LEASE_CONFLICT", str(exc))
            raise LiveGateBlocked(str(exc)) from exc
        self.lease_holder = lease.holder

        try:
            # 重启恢复（§断点重连）：把上次非优雅退出（kill -9/断电）遗留的
            # RUNNING/RECOVERY 会话补记为 INTERRUPTED。标记失败（StoreError）视为
            # 账本硬错误，走下方 except 路径（释放锁、关流）拒绝启动。
            interrupted = self.store.mark_interrupted_sessions(now_ms=int(self._now() * 1000))
            if interrupted > 0:
                logger.info("【重启恢复】已标记 %d 个中断会话", interrupted)

            # 4. 校准 server time（§3.1/-1021 防线）
            report.time_offset_spot_ms = int(self.spot.calibrate())
            report.time_offset_perp_ms = int(self.futures.calibrate())
            limit = self.config.execution.server_time_offset_limit_ms
            for label, offset in (("spot", report.time_offset_spot_ms), ("perp", report.time_offset_perp_ms)):
                if abs(offset) > limit:
                    raise ClockError(
                        f"{label} server time 偏移 {offset}ms 超过阈值 ±{limit}ms。"
                        "停止交易，重新同步系统 NTP（-1021 防线，§3.5）。"
                    )

            # 5. 规则与杠杆/保证金预检（§3.2 / 开发文档 §7.6：
            #    对**所有**允许交易的目标 symbol 逐个检查，不只检查第一个）
            spot_rules = self.spot.load_rules()
            perp_rules = self.futures.load_rules()
            common = sorted(set(spot_rules) & set(perp_rules))
            report.symbols = tuple(common)
            if not common:
                raise LiveGateBlocked("Spot 与 Futures 无共同可交易 symbol，拒绝启动")
            want_lev = self.config.execution.leverage
            want_margin = self.config.execution.margin_type
            symbol_detail: dict[str, Any] = {}
            for symbol in self.config.execution.live_symbols:
                if symbol not in spot_rules:
                    raise LiveGateBlocked(f"启动预检失败：{symbol} 缺少 Spot 规则")
                if symbol not in perp_rules:
                    raise LiveGateBlocked(f"启动预检失败：{symbol} 缺少 Futures 规则")
                info: dict[str, Any] = {"symbol": symbol}
                for label, rule in (("spot", spot_rules[symbol]), ("perp", perp_rules[symbol])):
                    status = getattr(rule, "status", "")
                    if status and status != "TRADING":
                        raise LiveGateBlocked(
                            f"启动预检失败：{symbol} {label} 状态 {status} 非 TRADING"
                        )
                    min_notional = getattr(rule, "min_notional", None)
                    info[f"{label}_min_notional"] = str(min_notional) if min_notional is not None else None
                    canary = Decimal(str(self.config.execution.canary_notional))
                    if min_notional is not None and canary < Decimal(str(min_notional)):
                        raise LiveGateBlocked(
                            f"启动预检失败：{symbol} {label} 最小名义额 {min_notional} "
                            f"高于 canary_notional {canary}，无法合法开仓"
                        )
                lev, margin, read_available = self._check_leverage_margin(symbol, want_lev, want_margin)
                info["leverage"] = lev
                info["margin_type"] = margin
                info["leverage_read_available"] = read_available
                report.leverage = lev
                report.margin_type = margin
                if (lev, margin) != (want_lev, want_margin):
                    raise LiveGateBlocked(
                        f"杠杆/保证金模式不符：{symbol} 当前 ({lev}, {margin})，"
                        f"期望 ({want_lev}, {want_margin})。启动预检失败（§3.2），拒绝启动。"
                    )
                symbol_detail[symbol] = info
            report.extra["symbols"] = symbol_detail

            # 7. 启动对账（§8.1 第 8 步）
            result = self.reconciler.run(reason="startup")
            self._last_reconcile_ms = int(self._now() * 1000)
            report.reconciliation_consistent = result.consistent
            report.reconciliation_mismatches = result.mismatches

            # 8. 用户数据流（全部通过后才启动）
            for stream in self.streams:
                stream.start()
            # 等待全部流达到新鲜状态再进入主循环：
            # poll 模式首轮 REST 查询 <2s；stream 模式需首次连接成功。
            # 避免主循环第一轮因「尚未可观测」误入 RECOVERY（且 RECOVERY 不自动解除）。
            fresh_deadline = self._now() + self.fresh_wait_seconds
            while any(not s.is_fresh for s in self.streams):
                if self._now() >= fresh_deadline:
                    raise LiveGateBlocked(
                        f"用户流 {self.fresh_wait_seconds:.0f} 秒内未达新鲜状态，拒绝启动"
                    )
                time.sleep(0.05)

            self._started = True

            # 启动 run_session（§6.1 关联链起点）+ 脱敏配置摘要
            self._ensure_run_session()

            # 首次账户快照（失败不拒绝启动，但开仓会被 ACCOUNT_STATE_UNKNOWN 拒绝）
            if self.account_builder is not None:
                try:
                    self._refresh_account_state(source="startup")
                except AccountStateError as exc:
                    self.store.record_risk_decision("ACCOUNT_STATE", False, str(exc))
                    self._on_alert("ACCOUNT_STATE_UNKNOWN", str(exc))

            if not result.can_open:
                self.enter_recovery(f"启动对账不一致: {list(result.mismatches)}")
            else:
                self._state = ServiceState.RUNNING
                self._persist_runtime_state()
                logger.info(
                    "【实盘服务启动完成】mode=%s run_id=%s symbols=%s offset=%d/%dms",
                    self.mode, self.run_id, list(report.symbols),
                    report.time_offset_spot_ms, report.time_offset_perp_ms,
                )
        except Exception:
            # 启动失败：释放锁、关流，不留半成品
            self._teardown()
            raise
        return report

    # -- 主循环 --------------------------------------------------------------

    def run_once(self) -> dict[str, Any]:
        """执行一轮主循环。可独立单测（不阻塞、不 sleep）。"""
        if self._state is ServiceState.STOPPED:
            return {"state": "STOPPED"}

        # 单实例锁续期（长跑韧性）：TTL 30s、主循环 5s 间隔，裕量 6 倍。
        # 刷新失败只告警不崩溃（DB 写坏会在后续步骤自然暴露）。
        if self.lease_holder is not None:
            try:
                self.store.refresh_lease(self.lease_name, self.lease_holder)
            except StoreError as exc:
                logger.warning("单实例锁刷新失败（后续写账本会自然暴露）: %s", exc)

        # 风控闸门状态同步
        gate_state = self.gate.state
        if gate_state is not HaltState.NORMAL and self._state is not ServiceState.RECOVERY:
            self._state = ServiceState.HALTED
            self._persist_runtime_state()
            self._on_alert("GATE_HALT", f"风控闸门状态 {gate_state.value}，禁止开新仓")
            return {"state": "HALTED"}
        if self._state is ServiceState.HALTED and gate_state is HaltState.NORMAL:
            # 闸门恢复（recover 已含对账+预检证明）且用户流仍新鲜 → 回到 RUNNING
            self._state = ServiceState.RUNNING
            logger.warning("【闸门恢复】回到 RUNNING")

        # 用户流新鲜度（§3.4：断线/过期 = 状态不可信）
        now_ms = int(self._now() * 1000)
        stale = [s for s in self.streams if not s.is_fresh]
        if stale:
            detail = ", ".join(f"{s.market}（代次 {s.generation}）" for s in stale)
            self.enter_recovery(f"用户流不新鲜/不可信: {detail}")
            return {"state": "RECOVERY", "reason": self._recovery_reason}

        # RECOVERY 自动解除（长跑自愈）：全部用户流恢复新鲜后做一次恢复对账，
        # 通过且风控闸门 NORMAL → 回 RUNNING，无需重启进程。
        # 对账按周期节流，避免 RECOVERY 期间每个 tick 全量对账。
        if self._state is ServiceState.RECOVERY:
            interval_ms = int(self.config.execution.reconciliation_interval_seconds * 1000)
            if now_ms - self._last_reconcile_ms < interval_ms:
                return {"state": "RECOVERY", "reason": self._recovery_reason}
            result = self.reconciler.run(reason="recovery_check")
            self._last_reconcile_ms = now_ms
            if not (result.can_open and self.gate.state is HaltState.NORMAL):
                self._recovery_reason = (
                    f"流恢复后对账/闸门未通过: {list(result.mismatches)}"
                    if not result.can_open
                    else f"风控闸门未恢复: {self.gate.state.value}"
                )
                self._persist_runtime_state()
                return {"state": "RECOVERY", "reason": self._recovery_reason}
            self._reconcile_ok = True
            self._recovery_reason = ""
            self._state = ServiceState.RUNNING
            self._refresh_held()
            self._persist_runtime_state()
            logger.warning("【RECOVERY 解除】用户流全部新鲜 + 对账通过 + 闸门 NORMAL，回到 RUNNING")

        if self._state is not ServiceState.RUNNING:
            return {"state": "RECOVERY", "reason": self._recovery_reason}

        opened: list[str] = []
        skipped: list[str] = []
        closed: list[str] = []

        # 周期性重校准 server time（§3.5）：代理 RTT 漂移会让启动时偏移过期→ -1021。
        # 校准失败只告警保留旧偏移；偏移超限才进 RECOVERY（时钟真的坏了）。
        resync_ms = int(self.config.execution.time_resync_seconds * 1000)
        if now_ms - self._last_resync_ms >= resync_ms:
            self._last_resync_ms = now_ms
            offset_limit = self.config.execution.server_time_offset_limit_ms
            for label, adapter in (("spot", self.spot), ("perp", self.futures)):
                try:
                    offset = adapter.calibrate()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("重校准 server time 失败（%s），保留旧偏移: %s", label, exc)
                    continue
                if abs(offset) > offset_limit:
                    self.enter_recovery(f"{label} server time 偏移 {offset}ms 超阈值 ±{offset_limit}ms（-1021 防线）")
                    self._persist_runtime_state()
                    return {"state": "RECOVERY", "reason": self._recovery_reason}

        # 周期性对账（§8.5）
        interval_ms = int(self.config.execution.reconciliation_interval_seconds * 1000)
        if now_ms - self._last_reconcile_ms >= interval_ms:
            result = self.reconciler.run(reason="periodic")
            self._last_reconcile_ms = now_ms
            self._reconcile_ok = result.can_open
            if not result.can_open:
                self.enter_recovery(f"周期对账不一致: {list(result.mismatches)}")
                self._persist_runtime_state()
                return {"state": "RECOVERY", "reason": self._recovery_reason}
            # 对账通过后以交易所真实持仓为准（§7.3 去重基础）
            self._refresh_held()

        # 周期性采样：账户快照 / 持仓快照 / 资金费（§8.5 至少 30s）
        snapshot_ms = int(self.config.execution.snapshot_interval_seconds * 1000)
        if self.strategy is not None and now_ms - self._last_snapshot_ms >= snapshot_ms:
            self._last_snapshot_ms = now_ms
            try:
                self._refresh_account_state(source="periodic")
            except (AccountStateError, StoreError) as exc:
                self._account_result = None
                self.store.record_risk_decision("ACCOUNT_STATE", False, str(exc))
                self._on_alert("ACCOUNT_STATE_UNKNOWN", f"账户快照失败: {exc}（开仓将被拒绝）")
            quotes = self._fetch_quotes()
            try:
                self._sample_position_snapshots(quotes)
                self._collect_funding_cashflows()
            except StoreError as exc:
                self.enter_recovery(f"账本写入失败: {exc}")
                return {"state": "RECOVERY", "reason": self._recovery_reason}

        # 低频候选刷新（§7.1：不在 5s 主循环拉全市场历史）
        if self.strategy is not None:
            refresh_ms = int(self.config.execution.candidate_refresh_seconds * 1000)
            if now_ms - self._last_candidate_refresh_ms >= refresh_ms:
                self.strategy.refresh_candidates()
                self._last_candidate_refresh_ms = now_ms

        # 无策略协调器：兼容旧 signal_provider 路径（仅测试/过渡期）
        if self.strategy is None:
            for signal in self._signal_provider():
                if self.store.active_pair_for_symbol(signal.symbol):
                    skipped.append(f"{signal.symbol}: 已有未完结 pair")
                    continue
                if self.mode == "shadow":
                    self._on_alert(
                        "SHADOW_INTENT",
                        f"shadow 模式只记录意图: {signal.symbol} notional={signal.target_notional}",
                    )
                    self.metrics.inc("strategy_signal_count", labels={"symbol": signal.symbol})
                    continue
                pair = self.executor.open_pair(
                    signal.symbol,
                    signal.target_notional,
                    spot_price=signal.spot_price,
                    perp_price=signal.perp_price,
                    quote_ts_ms=signal.quote_ts_ms,
                    state=self._risk_state_fn(),
                    reason=signal.reason,
                )
                if pair.status == "COMPLETE":
                    opened.append(signal.symbol)
                else:
                    skipped.append(f"{signal.symbol}: {pair.status} {pair.error}")
                    self._on_alert("OPEN_FAILED", f"{signal.symbol}: {pair.status} {pair.error}")
                self.metrics.inc("strategy_signal_count", labels={"symbol": signal.symbol})
            self._persist_runtime_state()
            return {"state": "RUNNING", "opened": opened, "skipped": skipped, "closed": closed}

        # ---- 实时策略路径（工作包一） ----
        ctx = self._build_context(now_ms)
        try:
            decisions = self.strategy.evaluate(ctx)
            for decision in decisions:
                self.store.record_signal_decision(decision)
        except StoreError as exc:
            self.enter_recovery(f"signal_decision 写账本失败: {exc}")
            return {"state": "RECOVERY", "reason": self._recovery_reason}

        # 1) 退出 / 换仓（先写决策再平仓，§7.4）
        for decision in decisions:
            if decision.decision_kind not in (DecisionKind.EXIT, DecisionKind.REPLACE):
                continue
            if self.mode == "shadow":
                self._on_alert(
                    "SHADOW_EXIT",
                    f"shadow 模式只记录退出意图: {decision.symbol} {decision.reason_code}",
                )
                continue
            pair = self.executor.close_pair(
                decision.symbol,
                reason=f"{decision.reason_code}: {decision.reason_text}",
                run_id=self.run_id,
                signal_decision_id=decision.decision_id,
                decision_ts_ms=decision.ts_ms,
            )
            if pair.status == "COMPLETE":
                closed.append(decision.symbol)
                self._post_close(decision.symbol, pair)
                if self._state is not ServiceState.RUNNING:
                    self._persist_runtime_state()
                    return {"state": self.state.value, "reason": self._recovery_reason}
            else:
                self._on_alert("EXIT_FAILED", f"{decision.symbol}: {pair.status} {pair.error}")
                self.enter_recovery(f"平仓失败: {decision.symbol} {pair.status} {pair.error}")
                self._persist_runtime_state()
                return {"state": self.state.value, "reason": self._recovery_reason}

        # 2) 开仓（决策已含全部门槛判断；执行层仍会再次检查，不能只信上层）
        for decision in decisions:
            if decision.decision_kind is not DecisionKind.OPEN or not decision.allowed:
                continue
            symbol = decision.symbol
            if self.mode == "shadow":
                self._on_alert(
                    "SHADOW_INTENT",
                    f"shadow 模式只记录意图: {symbol} notional={decision.requested_notional}",
                )
                self.metrics.inc("strategy_signal_count", labels={"symbol": symbol})
                continue
            if symbol in self._submitted_this_run:
                skipped.append(f"{symbol}: 本轮已提交")
                continue
            # 执行前重新确认报价新鲜（决策到下单之间可能已过期）
            if decision.quote_ts_ms is None or decision.spot_price is None or decision.perp_price is None:
                skipped.append(f"{symbol}: 决策缺少报价")
                continue
            quote = self._quote_fetcher(symbol)
            if quote is None:
                skipped.append(f"{symbol}: 提交前报价获取失败")
                continue
            self._submitted_this_run.add(symbol)
            state = self._risk_state_fn()
            pair = self.executor.open_pair(
                symbol,
                decision.requested_notional or Decimal("0"),
                spot_price=quote.spot_price,
                perp_price=quote.perp_price,
                quote_ts_ms=quote.ts_ms,
                state=state,
                reason=f"{decision.reason_code}: {decision.reason_text}",
                run_id=self.run_id,
                signal_decision_id=decision.decision_id,
                decision_ts_ms=decision.ts_ms,
            )
            if pair.status == "COMPLETE":
                opened.append(symbol)
            else:
                skipped.append(f"{symbol}: {pair.status} {pair.error}")
                self._on_alert("OPEN_FAILED", f"{symbol}: {pair.status} {pair.error}")
            self.metrics.inc("strategy_signal_count", labels={"symbol": symbol})

        self._persist_runtime_state()
        return {"state": "RUNNING", "opened": opened, "skipped": skipped, "closed": closed}

    def _build_context(self, now_ms: int) -> LiveContext:
        """构造本轮策略评估输入（报价每轮重新获取，§6.2）。"""
        account_ok = self._account_fresh(now_ms)
        total = self._account_result.total_capital if (account_ok and self._account_result) else Decimal("0")
        return LiveContext(
            now_ms=now_ms,
            run_id=self.run_id,
            total_capital=total,
            held=dict(self._held),
            quotes=self._fetch_quotes(),
            submitted_this_run=frozenset(self._submitted_this_run),
            account_ok=account_ok,
            reconcile_ok=self._reconcile_ok,
        )

    def run_forever(self, *, tick_seconds: float = 5.0, stop_check: Callable[[], bool] | None = None) -> bool:
        """阻塞主循环（CLI 用）。stop_check 返回 True 时退出。

        长跑容错（§断点重连）：单轮 ``run_once()`` 异常只进 RECOVERY 并计数，
        不崩溃进程；连续异常达到 ``execution.max_consecutive_tick_errors``
        时优雅停机并返回 False（CLI 以退出码 1 结束，交给 systemd 重启）。
        任一轮成功则计数清零。

        Returns:
            True = 正常停止（stop_check/外部信号）；False = 连续 tick 失败超限。
        """
        max_errors = self.config.execution.max_consecutive_tick_errors
        consecutive_errors = 0
        from .watchdog import LoopWatchdog

        watchdog = LoopWatchdog(
            timeout_seconds=self.config.execution.watchdog_timeout_seconds,
        )
        watchdog.start()
        try:
            while not (stop_check and stop_check()):
                # 心跳在轮首打点：单轮正常耗时（对账/候选刷新）不触发误杀
                watchdog.beat()
                try:
                    self.run_once()
                except Exception as exc:  # noqa: BLE001 tick 级容错是长跑设计：瞬时故障（网络抖动/瞬时 DB 错误）不得崩溃进程；KeyboardInterrupt 等 BaseException 不被捕获
                    consecutive_errors += 1
                    logger.error("【主循环 tick 异常】连续 %d/%d: %s", consecutive_errors, max_errors, exc)
                    self.enter_recovery(f"主循环 tick 异常: {exc}")
                    if consecutive_errors >= max_errors:
                        self._on_alert("TICK_LOOP_FAILURE",
                                       f"连续 {consecutive_errors} 轮 tick 异常，优雅停机等待守护进程重启")
                        self.stop()
                        return False
                else:
                    consecutive_errors = 0
                time.sleep(tick_seconds)
        finally:
            watchdog.stop()
        return True

    # -- 停机 --------------------------------------------------------------

    def _teardown(self) -> None:
        for stream in self.streams:
            try:
                stream.stop()
            except Exception:  # noqa: BLE001
                logger.warning("关闭用户流失败", exc_info=True)
        if self.lease_holder:
            try:
                self.store.release_lease(self.lease_name, self.lease_holder)
            except Exception:  # noqa: BLE001
                logger.warning("释放单实例锁失败", exc_info=True)
            self.lease_holder = None

    def stop(self) -> None:
        """优雅停机：关流 + 释放锁 + 关闭 run_session + 标记 STOPPED。

        RECOVERY 不自动解除（§7.3）：重启后必须重新预检 + 对账。
        """
        self._state = ServiceState.STOPPED
        if self.run_id:
            try:
                self.store.end_run_session(
                    self.run_id,
                    ended_ms=int(self._now() * 1000),
                    status="STOPPED",
                    stop_reason="graceful_stop",
                )
            except Exception:  # noqa: BLE001
                logger.warning("关闭 run_session 失败", exc_info=True)
        self._teardown()
        logger.warning("【实盘服务已停止】mode=%s", self.mode)

    # -- 辅助 --------------------------------------------------------------

    def attach_streams(self, streams: list[UserStreamLike]) -> None:
        """工厂装配用：服务先建（流的 on_untrusted 回调指向本服务）。"""
        self.streams = list(streams)

    def _default_risk_state(self) -> RiskState:
        """从账本持仓构造 RiskState。

        ⚠️ total_capital 默认 0 = 资金未知 → preflight 拒绝开仓。
        实盘部署必须注入带真实资金/已实现盈亏的 risk_state_fn，
        而不是让服务在未知资金下裸奔。
        """
        positions: dict[str, Position] = {}
        for symbol, legs in self.store.expected_positions().items():
            positions[symbol] = Position(
                symbol=symbol,
                spot_qty=float(legs.get("SPOT", 0)),
                perp_qty=float(-legs.get("PERP", 0)),  # PERP 负 = 空 → Position 正 = 空
            )
        return RiskState(positions=positions, total_capital=0.0)


# ---------------------------------------------------------------------------
# 上下文工厂（真实装配）
# ---------------------------------------------------------------------------

_WS_BASE = {
    ("spot", True): "wss://stream.binance.com:9443",
    ("spot", False): "wss://stream.binance.com:9443",
    ("perp", True): "wss://stream.binancefuture.com",
    ("perp", False): "wss://fstream.binance.com",
}


def _make_quote_fetcher(public_client: Any) -> Callable[[str], Quote | None]:
    """新鲜报价获取器：提交开仓前必须重新获取（§6.2）。

    行情源为公开只读接口；失败返回 None（本轮 STALE_QUOTE，不下单）。
    """

    def fetch(symbol: str) -> Quote | None:
        try:
            spot_price = public_client.spot_price(symbol)
            perp_payload = public_client.premium_index(symbol)
            # premiumIndex 无 lastPrice；用永续标记价（markPrice，标准参考价）
            perp_price = Decimal(str(perp_payload["markPrice"]))
            return Quote(
                spot_price=Decimal(str(spot_price)),
                perp_price=perp_price,
                ts_ms=int(time.time() * 1000),
            )
        except Exception:  # noqa: BLE001
            logger.warning("报价获取失败: %s", symbol, exc_info=True)
            return None

    return fetch


def build_live_context(  # noqa: PLR0913
    *,
    config: Config,
    spot_key: Any,
    spot_secret: Any,
    futures_key: Any,
    futures_secret: Any,
    is_testnet: bool,
    spot_base: str | None = None,
    futures_base: str | None = None,
) -> LiveService:
    """从凭证构建完整的实盘上下文（文档 §4.1 分层装配）。

    凭证只从这里注入，不进配置、不进账本。
    """
    from ..secrets import SecretStr  # 本地 import，避免顶层循环

    api = config.api
    exc = config.execution
    spot_url = spot_base or (api.spot_testnet_base if is_testnet else api.spot_base)
    futures_url = futures_base or (api.futures_testnet_base if is_testnet else api.futures_base)

    from ..data.binance import BinancePublicClient

    public_client = BinancePublicClient(api, config.data)
    config_hash = ""
    if config.source_path is not None:
        try:
            config_hash = hashlib.sha256(config.source_path.read_bytes()).hexdigest()
        except OSError:
            config_hash = ""

    from ..execution.transport import SignedClient

    spot_client = SignedClient(
        base_url=spot_url,
        api_key=SecretStr(spot_key, name="SPOT_KEY") if isinstance(spot_key, str) else spot_key,
        secret=SecretStr(spot_secret, name="SPOT_SECRET") if isinstance(spot_secret, str) else spot_secret,
        market="spot",
        recv_window_ms=exc.recv_window_ms,
        timeout_seconds=exc.request_timeout_seconds,
        read_max_retries=5,  # 代理链路偶发 SSL 中断，读请求多一层重试裕量
    )
    futures_client = SignedClient(
        base_url=futures_url,
        api_key=SecretStr(futures_key, name="FUT_KEY") if isinstance(futures_key, str) else futures_key,
        secret=SecretStr(futures_secret, name="FUT_SECRET") if isinstance(futures_secret, str) else futures_secret,
        market="perp",
        recv_window_ms=exc.recv_window_ms,
        timeout_seconds=exc.request_timeout_seconds,
        read_max_retries=5,  # 代理链路偶发 SSL 中断，读请求多一层重试裕量
    )
    spot = SpotAdapter(spot_client)
    futures = FuturesAdapter(futures_client)

    state_db = config.resolved_path(exc.state_db)
    store = StateStore(state_db)
    risk_manager = RiskManager(config.risk)
    audit = AuditLog(config.resolved_path(exc.audit_path))
    gate = RiskGate(risk_manager, kill_check=_kill_switch_check(), audit=audit)
    executor = PairExecutor(
        spot=spot,
        futures=futures,
        store=store,
        gate=gate,
        audit=audit,
        strategy_version="funding_carry-1.0",
        max_market_data_age_ms=int(exc.max_market_data_age_seconds * 1000),
        max_leg_slippage_pct=Decimal(str(exc.max_leg_slippage_pct)),
        hedge_tolerance_pct=Decimal(str(exc.hedge_tolerance_pct)),
        order_ack_timeout_seconds=exc.order_ack_timeout_seconds,
        poll_interval_seconds=0.25,
    )
    reconciler = Reconciler(store, spot, futures, ignore_assets=exc.reconcile_ignore_assets)

    service = LiveService(
        config=config,
        spot=spot,
        futures=futures,
        store=store,
        gate=gate,
        executor=executor,
        reconciler=reconciler,
        strategy=LiveStrategy(
            config=config,
            data=PublicDataStrategyProvider(config, client=public_client),
            store=store,
            strategy_version=executor.strategy_version,
            config_hash=config_hash,
        ),
        account_builder=AccountStateBuilder(
            config=config, store=store, candidate_symbols=tuple(exc.live_symbols)
        ),
        quote_fetcher=_make_quote_fetcher(public_client),
        config_hash=config_hash,
        spot_endpoint=spot_url,
        futures_endpoint=futures_url,
    )

    # 测试网/demo 的 user stream 端点可在 config 中覆盖（demo trading 域名不同）
    ws_base_cfg = {
        "spot": config.api.spot_testnet_ws_base if is_testnet else _WS_BASE[("spot", False)],
        "perp": config.api.futures_testnet_ws_base if is_testnet else _WS_BASE[("perp", False)],
    }
    streams: list[UserStreamLike]
    if exc.user_stream_mode == "poll":
        # demo trading 用户流不可用：REST 轮询替代新鲜度判定
        streams = [
            PollingUserStream(
                market=market,
                adapter=spot if market == "spot" else futures,
                poll_seconds=exc.user_stream_poll_seconds,
                fresh_seconds=exc.user_stream_fresh_seconds,
                on_untrusted=service.enter_recovery,
            )
            for market in ("spot", "perp")
        ]
    else:
        streams = [
            UserStream(
                market=market,
                adapter=spot if market == "spot" else futures,
                ws_base=ws_base_cfg[market],
                on_event=_event_sink(store),
                on_untrusted=service.enter_recovery,
                keepalive_seconds=exc.user_stream_keepalive_seconds,
                staleness_seconds=30.0,
            )
            for market in ("spot", "perp")
        ]
    service.attach_streams(streams)
    return service


def _kill_switch_check() -> Callable[[], bool]:
    from ..secrets import is_kill_switch_engaged

    return is_kill_switch_engaged


def _event_sink(store: StateStore) -> Callable[[Any], None]:
    from ..execution.user_stream import StreamEvent

    def sink(event: StreamEvent) -> None:
        try:
            store.record_event(
                fingerprint=event.fingerprint,
                market=event.market,
                event_type=event.event_type,
                exchange_ts=event.exchange_ts_ms,
                payload=event.payload,
                generation=event.generation,
            )
        except Exception:  # noqa: BLE001
            logger.warning("事件入账本失败", exc_info=True)

    return sink
