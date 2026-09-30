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
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

from ..config import Config
from ..domain.account import AccountSnapshot
from ..domain.common import InvalidDomainValue
from ..domain.control import (
    CommandKind,
    ControlCommand,
    SafetyState,
    SafetyStateKind,
)
from ..domain.market import DataQuality, MarketSnapshot
from ..domain.portfolio import PortfolioView
from ..domain.strategy import StrategyProposal
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
from ..execution.sync import ExchangeStateSynchronizer, SyncCaptureError
from ..execution.user_stream import PollingUserStream, UserStream
from ..market_data.service import MarketDataService
from ..observability.control import ControlPublisher
from ..observability.recovery import classify_recovery_reason
from ..portfolio.planner import PortfolioPlanner
from ..rate_limit import rate_limit_diagnostics_to_dict
from ..reporting.pnl import PnlAggregator
from ..risk.state_machine import SafetyStateMachine
from ..strategy.adapter import decisions_to_proposal
from ..strategy.funding_carry import CandidateInput, FundingCarryEvaluator
from .account_state import AccountStateBuilder, AccountStateError, AccountStateResult
from .market_sync import MarketDataSynchronizer
from .portfolio import Signal
from .strategy import (
    HeldPosition,
    LiveContext,
    LiveStrategy,
    PublicDataStrategyProvider,
    Quote,
    QuoteGateVerdict,
    evaluate_quote_gate,
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


#: T4：安全状态机（owner）→ 旧 ServiceState 兼容映射。旧状态不再独立决定
#: ``can_open``，只用于旧 API/日志的展示兼容。
_SAFETY_TO_SERVICE: dict[SafetyStateKind, ServiceState] = {
    SafetyStateKind.STARTING: ServiceState.STARTING,
    SafetyStateKind.RUNNING: ServiceState.RUNNING,
    SafetyStateKind.DEGRADED: ServiceState.RUNNING,
    SafetyStateKind.RECOVERY: ServiceState.RECOVERY,
    SafetyStateKind.CLOSE_ONLY: ServiceState.HALTED,
    SafetyStateKind.HALTED: ServiceState.HALTED,
    SafetyStateKind.EMERGENCY_FLATTEN: ServiceState.HALTED,
    SafetyStateKind.STOPPED: ServiceState.STOPPED,
}


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
        synchronizer: MarketDataSynchronizer | None = None,
        exch_sync: ExchangeStateSynchronizer | None = None,
        pair_exclusion_fn: Callable[[str], str | None] | None = None,
    ) -> None:
        self.config = config
        self.mode = config.execution.mode
        self.fresh_wait_seconds = fresh_wait_seconds
        self.spot = spot
        self.futures = futures
        self._pair_exclusion_fn = pair_exclusion_fn
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
        self._last_snapshot_ms = 0
        self._leverage_checked: set[str] = set()
        self._reconcile_ok = False
        self._last_resync_ms = 0
        self._on_alert = on_alert or (lambda kind, msg: logger.critical("【告警】%s: %s", kind, msg))
        self._risk_state_fn = risk_state_fn or self._risk_state_from_account
        self._signal_provider = signal_provider or (lambda: [])
        self._now = now_fn
        # T4：安全状态机 = 生产安全状态唯一 owner（fail closed：初始 RECOVERY，
        # 启动预检+对账通过才 RUNNING）；所有 watchdog/用户流/对账/限流/DB
        # 失败/人工命令经 ControlPublisher 进入状态机。
        self._safety_sm = SafetyStateMachine(
            SafetyState(
                SafetyStateKind.RECOVERY,
                reason="初始化：等待启动预检+对账通过（fail closed）",
                changed_at_ms=0,
                source="system",
            )
        )
        self._control = ControlPublisher(subscribers=(self._apply_control_command,))
        self._state = ServiceState.RECOVERY
        self._recovery_reason = ""
        self._recovery_entered_ms: int | None = None
        self._recovery_last_retry_ms: int | None = None
        self._recovery_retry_count = 0
        self._recovery_diagnostic: dict[str, Any] = {}
        # RECOVERY 卡死升级：只在曾达 RUNNING 后开始计时（启动预检/补账的
        # 长 RECOVERY 是合法的，不得误杀）；每次 RECOVERY→非 RECOVERY→RECOVERY
        # 重新起算，RUNNING 时清零。
        self._recovery_since_ms: int | None = None
        self._reached_running = False
        self._last_reconcile_ms = 0
        self._started = False
        # T3：scan epoch 同步器（选币闸门）与交易所事实同步器（账本恢复）
        self.synchronizer = synchronizer
        self.exch_sync = exch_sync
        # 账本同步闸门：无事实同步器时视为 legacy 路径（True）；
        # 有同步器时首次同步通过前 False（禁止开仓）
        self._ledger_sync_ok = exch_sync is None
        self._ledger_sync_error = ""  # T4：最后一次 facts 同步失败原因（Web 展示）

    # -- 状态 ---------------------------------------------------------------

    @property
    def state(self) -> ServiceState:
        """兼容映射（T4）：安全状态机是 owner；闸门非 NORMAL 时叠加 HALTED
        展示（下一 tick 经 ControlPublisher 同步进状态机）。"""
        if self.gate.state is not HaltState.NORMAL and self._state is ServiceState.RUNNING:
            return ServiceState.HALTED
        return self._state

    def recovery_elapsed_ms(self) -> int | None:
        """启动后（曾达 RUNNING）持续停留在 RECOVERY 的毫秒数；未计时返回 None。

        供 runner 的 RECOVERY 卡死升级判定（``recovery_escalation_seconds``）：
        自愈路径失效（如用户流 untrusted 卡死）时不再无限等待，停机交 systemd 重启。
        """
        if not self._reached_running or self._recovery_since_ms is None:
            return None
        return max(0, int(self._now() * 1000) - self._recovery_since_ms)

    # -- 控制面（T4：SafetyStateMachine 唯一写入者） ------------------------

    def _apply_control_command(self, cmd: ControlCommand) -> None:
        """ControlPublisher 订阅者：应用命令到状态机并同步兼容状态。"""
        try:
            self._safety_sm.apply(cmd, now_ms=cmd.issued_at_ms)
        except InvalidDomainValue as exc:
            logger.warning("控制命令被状态机拒绝（保持原状态）: %s", exc)
            return
        self._sync_service_state()

    def _sync_service_state(self) -> None:
        """安全状态机 → 旧 ServiceState（只读兼容投影，不再反向写入状态机）。"""
        mapped = _SAFETY_TO_SERVICE[self._safety_sm.state.state]
        if mapped is not self._state:
            self._state = mapped
            if mapped is ServiceState.RUNNING:
                self._reached_running = True
                self._recovery_since_ms = None
            elif mapped is ServiceState.RECOVERY:
                if self._reached_running:
                    self._recovery_since_ms = int(self._now() * 1000)
            self._persist_runtime_state()

    def _publish_recovery(self, reason: str, *, source: str = "service") -> None:
        self._control.recovery(reason=reason or "未指定原因（fail closed）", source=source)

    def _update_recovery_diagnostic(
        self,
        reason: str | None = None,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        retry: bool = True,
    ) -> None:
        """更新当前 Recovery 错误块；不改变安全状态迁移。"""
        now_ms = int(self._now() * 1000)
        if self._recovery_entered_ms is None:
            self._recovery_entered_ms = now_ms
        if retry:
            self._recovery_last_retry_ms = now_ms
            self._recovery_retry_count += 1
        message = reason or self._recovery_reason or "未指定原因（fail closed）"
        self._recovery_diagnostic = {
            "code": code or classify_recovery_reason(message),
            "message": message,
            "entered_at_ms": self._recovery_entered_ms,
            "last_retry_at_ms": self._recovery_last_retry_ms,
            "retry_count": self._recovery_retry_count,
            "details": details or {},
        }

    def _record_recovery_retry(self, now_ms: int | None = None) -> None:
        """记录一次恢复检查尝试，供 WebUI/CLI 展示最近重试时间。"""
        if self._recovery_entered_ms is None:
            self._recovery_entered_ms = now_ms or int(self._now() * 1000)
        self._recovery_last_retry_ms = now_ms or int(self._now() * 1000)
        self._recovery_retry_count += 1
        if self._recovery_diagnostic:
            self._recovery_diagnostic["last_retry_at_ms"] = self._recovery_last_retry_ms
            self._recovery_diagnostic["retry_count"] = self._recovery_retry_count

    def enter_recovery(self, reason: str) -> None:
        """进入 RECOVERY：禁止开新仓。幂等；经 ControlPublisher → 状态机。"""
        if self._safety_sm.state.state is SafetyStateKind.STOPPED:
            return
        self._recovery_reason = reason
        already_recovering = self._safety_sm.state.state is SafetyStateKind.RECOVERY
        self._update_recovery_diagnostic(reason, retry=not already_recovering)
        if not already_recovering:
            self._publish_recovery(reason)
        self._on_alert("RECOVERY", reason)
        self._persist_runtime_state()
        logger.error("【进入 RECOVERY】%s（禁止开新仓，等待对账通过后恢复）", reason)

    def resume_after_checks(self, reason: str, *, source: str = "service") -> None:
        """显式恢复：RECOVERY →（必要时两步）→ RUNNING。

        放宽必须走显式 RECOVERY → RESUME_AFTER_CHECKS（领域规则）；
        非 RECOVERY 态先经 RECOVERY 再 RESUME（CLOSE_ONLY/HALTED 回 RUNNING
        同样先入 RECOVERY）。STOPPED 不恢复。
        """
        current = self._safety_sm.state.state
        if current is SafetyStateKind.STOPPED:
            return
        if current is not SafetyStateKind.RECOVERY:
            self._control.recovery(
                reason=reason or "恢复前确认停机已解除", source=source
            )
        self._control.resume_after_checks(reason=reason or "预检+对账通过", source=source)
        self._recovery_reason = ""
        self._recovery_entered_ms = None
        self._recovery_last_retry_ms = None
        self._recovery_retry_count = 0
        self._recovery_diagnostic = {}
        self._persist_runtime_state()

    def apply_gate_state(self) -> None:
        """风控闸门（含 KILL_SWITCH 文件）→ 控制面命令（加严方向）。

        只加严不放宽：闸门恢复仍由显式 ``recover()`` + ``resume_after_checks``
        完成。EMERGENCY_FLATTEN 只可能由人工触发（领域层强制 source=manual）。
        """
        kind = self._safety_sm.state.state
        if kind is SafetyStateKind.STOPPED or kind is SafetyStateKind.EMERGENCY_FLATTEN:
            return
        if self.gate.state is HaltState.NORMAL:
            return
        now_ms = int(self._now() * 1000)
        if self.gate.state is HaltState.EMERGENCY_FLATTEN:
            self._control.publish(
                CommandKind.EMERGENCY_FLATTEN,
                reason=getattr(self.gate, "halt_reason", "") or "emergency flatten",
                source="manual",
                now_ms=now_ms,
            )
        else:
            if kind is SafetyStateKind.CLOSE_ONLY or kind is SafetyStateKind.RECOVERY:
                return  # 已加严/已恢复中，不重复发布
            self._control.halt_new_risk(
                reason=getattr(self.gate, "halt_reason", "") or f"gate {self.gate.state.value}",
                source="risk_gate",
                now_ms=now_ms,
            )

    @property
    def safety_state(self) -> SafetyState:
        """生产安全状态（owner：状态机；旧 gate/_state 只做兼容投影）。"""
        return self._safety_sm.state

    @property
    def _market_data_ready(self) -> bool:
        """scan epoch 闸门（T2/T3）：无同步器时 legacy 路径视为就绪。"""
        if self.synchronizer is None:
            return True
        return self.synchronizer.latest_ready() is not None

    @property
    def can_open(self) -> bool:
        """开仓权限 = 安全状态机允许新增风险 + 全部事实闸门通过（T4）：
        状态机 RUNNING && state==RUNNING && streams 新鲜 && 账户完整 &&
        账本同步 && 对账 && epoch READY && 风控闸门 NORMAL。
        任一 false 禁止新增风险。旧 ServiceState 不再独立决定开仓。"""
        if not self.config.execution.order_submission_enabled:
            return False
        if not self._safety_sm.state.allows_new_risk:
            return False
        if self._state is not ServiceState.RUNNING:
            return False
        if self.gate.state is not HaltState.NORMAL:
            return False
        if not self._reconcile_ok or not self._ledger_sync_ok:
            return False
        if not self._market_data_ready:
            return False
        if self._account_result is None or not self._account_result.complete:
            return False
        return all(s.is_fresh for s in self.streams)

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
        # T4：限流器快照（JSON 可序列化，不含 URL/密钥）；
        # v5.0 T3：source/as_of/quality 信封 + 同出口 IP 不确定性显式呈现
        rate_limits: dict[str, Any] | None = None
        if self.exch_sync is not None:
            rate_limits = self._rate_limits_snapshot()
        rate_limits_envelope: dict[str, Any] | None = None
        if rate_limits is not None:
            qualities = [str(v.get("quality")) for v in rate_limits.values()]
            if any(q == "OK" for q in qualities):
                rate_block_quality = "OK"
            elif any(q == "STALE" for q in qualities):
                rate_block_quality = "STALE"
            else:
                rate_block_quality = "UNKNOWN"
            rate_limits_envelope = {
                "source": "RateLimitCoordinator.snapshot（服务内存，仅诊断，非事实源）",
                "as_of_ms": now_ms,
                "quality": rate_block_quality,
            }
        # T4：市场数据就绪度（epoch status/coverage/cutoff）
        market_data: dict[str, Any] | None = None
        if self.synchronizer is not None:
            market_data = self._market_data_readiness(now_ms)
        market_data_envelope: dict[str, Any] | None = None
        if market_data is not None:
            market_data_envelope = {
                "source": "MarketDataSynchronizer.readiness（服务内存，仅诊断）",
                "as_of_ms": now_ms,
                "quality": {
                    "READY": "OK",
                    "STALE": "STALE",
                }.get(str(market_data.get("status")), "DEGRADED"),
            }
        # v5.0 T3：数据层 cache 健康（ClientStats；写失败不得计作保存成功）
        cache_stats, cache_stats_envelope = self._cache_stats_snapshot(now_ms)
        # T4：新鲜度块（账户/对账/心跳年龄，供 Web/CLI 统一展示）
        acct_ts = None if acct is None else int(getattr(acct, "ts_ms", 0) or 0)
        freshness = {
            "account_ts_ms": acct_ts,
            "account_age_ms": (now_ms - acct_ts) if acct_ts else None,
            "account_complete": bool(acct is not None and acct.complete),
            "reconcile_age_ms": (
                now_ms - self._last_reconcile_ms if self._last_reconcile_ms else None
            ),
            "reconcile_ok": self._reconcile_ok,
            "ledger_sync_ok": self._ledger_sync_ok,
            "ledger_sync_error": self._ledger_sync_error,
        }
        return {
            "state": self.state.value,
            "recovery_reason": self._recovery_reason,
            "recovery_error": dict(self._recovery_diagnostic) if self._recovery_diagnostic else None,
            "run_id": self.run_id or None,
            "mode": self.mode,
            "can_open": self.can_open,
            "order_submission_enabled": self.config.execution.order_submission_enabled,
            "observation_only": not self.config.execution.order_submission_enabled,
            "gates": {
                "running": self._state is ServiceState.RUNNING,
                "gate_normal": self.gate.state is HaltState.NORMAL,
                "reconcile_ok": self._reconcile_ok,
                "ledger_sync_ok": self._ledger_sync_ok,
                "market_data_ready": self._market_data_ready,
                "account_complete": (
                    self._account_result is not None and self._account_result.complete
                ),
            },
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
            "rate_limits": rate_limits,
            "rate_limits_envelope": rate_limits_envelope,
            "market_data": market_data,
            "market_data_envelope": market_data_envelope,
            "pool": self._pool_snapshot(now_ms),
            "freshness": freshness,
            "freshness_envelope": {
                "source": "LiveService.web_snapshot（服务内存，仅诊断）",
                "as_of_ms": now_ms,
                "quality": (
                    "UNKNOWN"
                    if acct_ts is None
                    else (
                        "DEGRADED"
                        if (not self._reconcile_ok or self._ledger_sync_error)
                        else "OK"
                    )
                ),
            },
            "cache_stats": cache_stats,
            "cache_stats_envelope": cache_stats_envelope,
            # T4：安全状态机（owner）+ 控制命令审计（状态迁移可追溯）
            "safety": self._safety_sm.state.to_dict(),
            "control_events": [c.to_dict() for c in self._control.recent[-20:]],
            "code_revision": self.code_revision,
            "metrics": self.metrics.snapshot(),
        }

    def _rate_limits_snapshot(self) -> dict[str, Any] | None:
        """共享限流协调器的只读快照（v5.0 T3：服务端观测 vs 本地估算分离 +
        header 年龄 + 同出口 IP 不确定性；不触碰调度语义）。"""
        try:
            client = getattr(self.spot, "client", None)
            limiter = getattr(client, "coordinator", None)
            if limiter is None or not hasattr(limiter, "snapshot"):
                return None
            t = float(limiter.now())
            snap = limiter.snapshot()
            out: dict[str, Any] = {}
            for scope, s in snap.items():
                out[str(getattr(scope, "value", scope))] = rate_limit_diagnostics_to_dict(
                    s, now_mono=t
                )
            return out or None
        except Exception:  # noqa: BLE001 —— 诊断块失败不影响快照主体
            logger.debug("限流器快照失败", exc_info=True)
            return None

    def _cache_stats_snapshot(
        self, now_ms: int
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """数据层 cache 健康（v5.0 T3）：复用 T1 的 ClientStats 计数。

        Returns:
            ``(cache_stats, envelope)``；客户端/缓存不可用 → 双 None
            （展示层渲染 UNKNOWN/null，不得渲染为 0/healthy）。
            写/读失败计数 > 0 → quality DEGRADED（失败不得计作保存成功）。
        """
        try:
            client = getattr(self.spot, "client", None)
            stats = getattr(client, "stats", None)
            cache = getattr(client, "cache", None)
            if stats is None or cache is None:
                return None, None
            data: dict[str, Any] = stats.as_dict()
            data["cache_enabled"] = bool(getattr(cache, "enabled", False))
            try:
                data["namespaces"] = cache.stats()
                data["total_bytes"] = int(cache.total_bytes())
            except Exception:  # noqa: BLE001 —— 磁盘统计失败不阻塞诊断主体
                logger.debug("cache 磁盘统计失败", exc_info=True)
            quality = "OK"
            if int(data.get("cache_write_failures", 0)) > 0 or int(
                data.get("cache_read_failures", 0)
            ) > 0:
                quality = "DEGRADED"
            if not data["cache_enabled"]:
                quality = "UNKNOWN"
            envelope = {
                "source": "BinancePublicClient.stats（服务内存，仅诊断）",
                "as_of_ms": now_ms,
                "quality": quality,
            }
            return data, envelope
        except Exception:  # noqa: BLE001
            logger.debug("cache 统计快照失败", exc_info=True)
            return None, None

    def _market_data_readiness(self, now_ms: int) -> dict[str, Any] | None:
        """epoch 就绪度 → 可序列化 dict（T4：Web 数据状态区）。"""
        try:
            synchronizer = self.synchronizer
            if synchronizer is None:
                return None
            rd = synchronizer.readiness(now_ms)
            if rd is None:
                return None
            cutoff_ms: int | None = None
            try:
                epochs = self.store.scan_epochs(limit=1)
                if epochs:
                    cutoff_ms = epochs[0].get("decision_cutoff_ms")
            except Exception:  # noqa: BLE001
                cutoff_ms = None
            stage_reader = getattr(synchronizer, "stage_readiness", None)
            stages = stage_reader(now_ms) if callable(stage_reader) else {
                "stages": [],
                "as_of_ms": now_ms,
                "age_ms": None,
                "freshness": "UNKNOWN",
            }
            return {
                "epoch_id": str(rd.epoch_id),
                "status": str(rd.status.value),
                "can_rank": bool(rd.can_rank),
                "expected": int(rd.expected),
                "completed": int(rd.completed),
                "excluded_count": int(rd.excluded_count),
                "failed_count": int(rd.failed_count),
                "failed": {str(k): str(v) for k, v in (rd.failed or {}).items()},
                "age_ms": int(rd.age_ms),
                "reason": str(rd.reason or ""),
                "decision_cutoff_ms": cutoff_ms,
                "stages": stages.get("stages", []),
                "stages_as_of_ms": stages.get("as_of_ms"),
                "stages_age_ms": stages.get("age_ms"),
                "stages_freshness": stages.get("freshness", "UNKNOWN"),
            }
        except Exception:  # noqa: BLE001
            logger.debug("epoch 就绪度快照失败", exc_info=True)
            return None

    def _pool_snapshot(self, now_ms: int) -> dict[str, Any] | None:
        """币池页面只读 projection：epoch、端点健康、阶段和最终候选。"""
        synchronizer = self.synchronizer
        if synchronizer is None:
            return None
        try:
            from ..data.venue import venue_for_execution_mode

            data = getattr(self.strategy, "data", None)
            venue = ""
            venue_fn = getattr(data, "venue_name", None)
            if callable(venue_fn):
                venue = str(venue_fn())
            if not venue:
                venue = venue_for_execution_mode(self.config.execution.mode).value

            readiness = synchronizer.readiness(now_ms)
            latest = synchronizer.latest()
            if readiness is None:
                return {
                    "as_of_ms": now_ms,
                    "venue": venue,
                    "epoch": None,
                    "api_sources": self._pool_api_sources(now_ms),
                    "stages": [],
                    "candidates": [],
                    "quality": "UNKNOWN",
                    "error": "尚无 scan epoch",
                    "order_submission_enabled": self.config.execution.order_submission_enabled,
                    "observation_only": not self.config.execution.order_submission_enabled,
                }
            epoch_status = readiness.status.value
            epoch = {
                "id": readiness.epoch_id,
                "status": epoch_status,
                "created_ms": latest.created_ms if latest and latest.epoch_id == readiness.epoch_id else None,
                "completed_ms": latest.completed_ms if latest and latest.epoch_id == readiness.epoch_id else None,
                "decision_cutoff_ms": latest.decision_cutoff_ms if latest and latest.epoch_id == readiness.epoch_id else None,
                "age_ms": readiness.age_ms,
                "reason": readiness.reason,
                "expected": readiness.expected,
                "completed": readiness.completed,
                "excluded_count": readiness.excluded_count,
                "failed_count": readiness.failed_count,
            }
            stage_data = synchronizer.stage_readiness(now_ms)
            candidates = self._pool_candidates(synchronizer, readiness, now_ms)
            quality = "OK" if readiness.can_rank else (
                "STALE" if epoch_status == "EXPIRED" else "DEGRADED"
            )
            return {
                "as_of_ms": now_ms,
                "venue": venue,
                "epoch": epoch,
                "api_sources": self._pool_api_sources(now_ms),
                "stages": stage_data.get("stages", []),
                "stages_as_of_ms": stage_data.get("as_of_ms"),
                "stages_age_ms": stage_data.get("age_ms"),
                "stages_freshness": stage_data.get("freshness", "UNKNOWN"),
                "candidates": candidates,
                "quality": quality,
                "error": readiness.reason if not readiness.can_rank else "",
                "order_submission_enabled": self.config.execution.order_submission_enabled,
                "observation_only": not self.config.execution.order_submission_enabled,
            }
        except Exception as exc:  # noqa: BLE001 —— 币池块局部降级
            logger.debug("币池 projection 构建失败", exc_info=True)
            return {
                "as_of_ms": now_ms,
                "venue": "",
                "epoch": None,
                "api_sources": None,
                "stages": [],
                "candidates": [],
                "quality": "UNKNOWN",
                "error": f"币池诊断不可用: {type(exc).__name__}",
                "order_submission_enabled": self.config.execution.order_submission_enabled,
                "observation_only": not self.config.execution.order_submission_enabled,
            }

    def _pool_api_sources(self, now_ms: int) -> list[dict[str, Any]] | None:
        data = getattr(self.strategy, "data", None)
        client = getattr(data, "client", None)
        stats = getattr(client, "stats", None)
        reader = getattr(stats, "endpoint_health", None)
        if not callable(reader):
            return None
        try:
            result = reader(now_ms=now_ms)
            return result if isinstance(result, list) else None
        except Exception:  # noqa: BLE001
            return None

    def _pool_candidates(
        self, synchronizer: MarketDataSynchronizer, readiness: Any, now_ms: int
    ) -> list[dict[str, Any]]:
        if not readiness.can_rank:
            return []
        snapshots = synchronizer.snapshots_for(readiness.epoch_id)
        premium = synchronizer.premium_snapshots()
        evaluator = FundingCarryEvaluator(config=self.config, now_fn=self._now)
        entry = self.config.strategy.entry
        selection = self.config.strategy.selection
        threshold = max(
            Decimal(str(entry.min_trailing_annualized)),
            Decimal(str(entry.min_annualized_rate)),
        )
        rows: list[dict[str, Any]] = []
        for symbol, snap in sorted(snapshots.items()):
            if snap.error or snap.volume_status != "FETCHED":
                continue
            if snap.quote_volume_3d_avg < Decimal(str(selection.min_quote_volume_3d_avg)):
                continue
            candidate = CandidateInput(
                symbol=symbol,
                rates=snap.rates,
                mark_prices=snap.mark_prices,
                timestamps=snap.timestamps,
                interval_hours=snap.interval_hours,
                volume_3d_avg=snap.quote_volume_3d_avg,
                refreshed_ts_ms=snap.fetched_ms,
            )
            trailing, streak = evaluator.entry_metrics(candidate)
            if trailing < threshold or streak < entry.min_consecutive_positive:
                continue
            avg = (
                sum(snap.rates[-entry.lookback_periods:], Decimal("0"))
                / Decimal(entry.lookback_periods)
                if len(snap.rates) >= entry.lookback_periods else None
            )
            live = premium.get(symbol, {})
            blockers = self._pool_open_blockers()
            if self.can_open:
                blockers.extend(("仍需提交前双腿报价 freshness", "持仓槽位/去重条件需在决策时确认"))
            current_epoch = synchronizer.latest()
            cutoff_ms = (
                current_epoch.decision_cutoff_ms
                if current_epoch is not None and current_epoch.epoch_id == readiness.epoch_id
                else None
            )
            rows.append({
                "symbol": symbol,
                "pool_status": "READY",
                "current_funding_rate": live.get("current_funding_rate"),
                "latest_settled_rate": str(snap.latest_settled_rate) if snap.latest_settled_rate is not None else None,
                "average_rate_8h": str(avg) if avg is not None else None,
                "trailing_annualized": str(trailing),
                "threshold_min_trailing_annualized": str(entry.min_trailing_annualized),
                "threshold_min_annualized_rate": str(entry.min_annualized_rate),
                "effective_threshold": str(threshold),
                "threshold_distance": str(max(threshold - trailing, Decimal("0"))),
                "threshold_excess": str(max(trailing - threshold, Decimal("0"))),
                "consecutive_positive_periods": streak,
                "required_consecutive_positive": entry.min_consecutive_positive,
                "quote_volume_3d_avg": str(snap.quote_volume_3d_avg),
                "quote_volume_threshold": str(selection.min_quote_volume_3d_avg),
                "volume_status": snap.volume_status,
                "open_qualification": "POOL_PASS_NOT_OPEN_GUARANTEE",
                "open_blockers": blockers,
                "funding_updated_ms": snap.fetched_ms,
                "latest_settled_ms": snap.latest_settled_ms,
                "premium_fetched_ms": live.get("fetched_ms"),
                "next_funding_time_ms": live.get("next_funding_time_ms"),
                "volume_window_end_ms": snap.volume_window_end_ms,
                "settle_interval_hours": snap.settle_interval_hours,
                "epoch_id": readiness.epoch_id,
                "decision_cutoff_ms": cutoff_ms,
            })
        rows.sort(key=lambda row: (-Decimal(row["trailing_annualized"]), row["symbol"]))
        return rows

    def _pool_open_blockers(self) -> list[str]:
        blockers: list[str] = []
        if self.state is not ServiceState.RUNNING:
            blockers.append(f"服务状态 {self.state.value}")
        if not self._reconcile_ok:
            blockers.append("最近对账未通过")
        if not self._ledger_sync_ok:
            blockers.append("账本事实同步未通过")
        if self._account_result is None or not self._account_result.complete:
            blockers.append("账户快照未知/不完整")
        if self.gate.state is not HaltState.NORMAL:
            blockers.append(f"风控闸门 {self.gate.state.value}")
        if not self.config.execution.order_submission_enabled:
            blockers.append("当前为只观测模式，订单提交已暂停")
        if not all(s.is_fresh for s in self.streams):
            blockers.append("用户流/账户数据不新鲜")
        return blockers

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

    def _refresh_account_state(self, source: str, *, bundle: Any | None = None) -> None:
        if self.account_builder is None:
            return
        if self.strategy is not None and self.strategy.dynamic_pool:
            # 动态候选池：风险快照的持仓范围跟随当前池（builder 构造时池可能为空）
            self.account_builder.candidate_symbols = tuple(self.strategy.candidate_symbols)
        asset_prices = self._managed_asset_prices()
        result = self.account_builder.snapshot(
            spot=self.spot, futures=self.futures, source=source, run_id=self.run_id,
            bundle=bundle, asset_prices=asset_prices,
        )
        self._account_result = result

    def _managed_asset_prices(self) -> dict[str, Decimal]:
        """受管理现货资产的估值价格（无新鲜报价 → 该资产无法估值 → 快照不完整）。

        只对当前有余额的非 USDT 资产取价；无余额资产不参与估值。
        价格取自公共行情 quote_fetcher（现货价）。
        """
        prices: dict[str, Decimal] = {}
        try:
            balances = self.spot.balances()
        except Exception:  # noqa: BLE001 —— 余额查不到时不估值（快照会因缺失而不完整或被拒）
            return prices
        for asset, qty in balances.items():
            if asset == "USDT" or qty <= 0:
                continue
            quote = self._quote_fetcher(f"{asset}USDT")
            if quote is not None and quote.spot_price > 0:
                prices[asset] = quote.spot_price
        return prices

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

    def _ledger_sync_symbols(self) -> list[str]:
        """事实回补覆盖的 symbol：持仓 + 未完结 pair + 历史成交。

        不含候选池：候选币无账本事实可补，全池同步 = 每周期 100+ 次
        myTrades/income 请求（weight 爆炸）且 demo 受限交易对必返 -1121。
        候选池外已有仓位仍由持仓/账本集合覆盖（不受当前池限制）。
        """
        symbols: set[str] = set(self._held)
        try:
            for pair in self.store.open_pairs():
                sym = pair.get("symbol")
                if sym:
                    symbols.add(str(sym))
            for fill in self.store.fills(limit=1000):
                sym = fill.get("symbol")
                if sym:
                    symbols.add(str(sym))
        except Exception:  # noqa: BLE001 —— 账本读失败不阻断（同步本身会报错）
            logger.debug("事实回补 symbol 集账本查询失败", exc_info=True)
        return sorted(symbols)

    def _periodic_reconcile(self, now_ms: int) -> bool:
        """周期/恢复对账：T3 路径复用短期 capture bundle（poll/account/
        reconcile 单飞），并定期增量同步 facts。返回 True = 本轮对账通过。"""
        bundle = None
        if self.exch_sync is not None:
            try:
                bundle = self.exch_sync.capture()
            except SyncCaptureError as exc:
                self.enter_recovery(f"capture 失败: {exc}")
                return False
            if self._ledger_sync_ok:
                # 增量 facts 同步（节流 = 对账周期）
                fill_results = self.exch_sync.sync_fills(self._ledger_sync_symbols())
                income_results = self.exch_sync.sync_funding_income(self._ledger_sync_symbols())
                bad = [
                    f"{r.market}/{r.stream}/{r.symbol}: {r.error or '不完整'}"
                    for r in (*fill_results, *income_results)
                    if r.error or not r.complete
                ]
                if bad:
                    self._ledger_sync_ok = False
                    self._ledger_sync_error = "; ".join(bad[:3])
                    self.enter_recovery(f"事实同步失败: {'; '.join(bad[:3])}")
                    return False
                self._ledger_sync_error = ""
        result = self.reconciler.run(reason="periodic", snapshot=bundle)
        self._last_reconcile_ms = now_ms
        self._reconcile_ok = result.can_open
        if not result.can_open:
            reason = f"周期对账不一致: {list(result.mismatches)}"
            self.enter_recovery(reason)
            self._update_recovery_diagnostic(
                reason,
                code="RECONCILIATION_MISMATCH",
                details={"mismatches": list(result.mismatches)},
                retry=False,
            )
            self._persist_runtime_state()
            return False
        return True

    def _refresh_held(self) -> None:
        """以交易所对账后的真实持仓为准刷新 held 集合（§7.3）。"""
        try:
            spot_balances = self.spot.balances()
        except Exception as exc:  # noqa: BLE001
            self.store.record_risk_decision("HELD_STATE", False, f"现货余额查询失败: {exc}")
            return
        current: dict[str, HeldPosition] = {}
        symbols = set(self.config.execution.live_symbols)
        if self.strategy is not None and self.strategy.dynamic_pool:
            # 动态候选池：池内 + 已跟踪持仓（池轮换时不丢跟踪）
            symbols.update(self.strategy.candidate_symbols)
            symbols.update(self._held)
        for symbol in sorted(symbols):
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
        """本轮批量报价：持仓恒拉；固定候选池全拉；动态候选池只拉持仓
        （池内通过前置检查的 symbol 在 PENDING_QUOTE 解析时按需获取）。"""
        symbols: set[str] = set(self._held)
        if self.strategy is not None and not self.strategy.dynamic_pool:
            symbols.update(self.strategy.candidate_symbols)
        quotes: dict[str, Quote] = {}
        for symbol in symbols:
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
                    market="PERP",
                    authority="ESTIMATED",  # T3：估算口径，不得冒充交易所事实
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
            self.store.set_runtime_state(
                "recovery_diagnostic",
                json.dumps(self._recovery_diagnostic, ensure_ascii=False, default=str),
            )
            self.store.set_runtime_state(
                "recovery_last_retry_ms",
                str(self._recovery_last_retry_ms or ""),
            )
            self.store.set_runtime_state("recovery_retry_count", str(self._recovery_retry_count))
            self.store.set_runtime_state("last_reconcile_ms", str(self._last_reconcile_ms))
            self.store.set_runtime_state(
                "account_snapshot_ms", str(self._account_result.ts_ms if self._account_result else 0)
            )
            self.store.set_runtime_state("can_open", "1" if self.can_open else "0")
            self.store.set_runtime_state(
                "order_submission_enabled",
                "1" if self.config.execution.order_submission_enabled else "0",
            )
            if self._recovery_since_ms is not None:
                self.store.set_runtime_state("recovery_since_ms", str(self._recovery_since_ms))
            else:
                self.store.set_runtime_state("recovery_since_ms", "")
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
        """按 §8.1 顺序执行启动预检（T4.3：顺序固化在
        ``application.lifecycle.run_startup``，本方法保留兼容 facade）。

        硬失败（锁冲突/时钟超阈/规则不符/流未新鲜/账本硬错误）抛异常
        （启动时崩溃是特性）；软闸门（对账/账户/事实同步/市场 READY）
        失败 → 默认 RECOVERY（禁止开仓，主循环闸门全绿后自动回 RUNNING）。
        """
        from ..application.lifecycle import run_startup

        return run_startup(self)

    # -- T4.3：启动阶段（顺序由 application.lifecycle 固化） --------------

    def _stage_mode(self) -> str:
        """1. config→mode：execution.mode 决定 SHADOW/paper/live。"""
        if not self.mode:
            raise LiveGateBlocked("execution.mode 未配置，拒绝启动")
        return self.mode

    def _stage_server_time(self, report: StartupReport) -> str:
        """2. 校准 server time（§3.1/-1021 防线；偏移超阈 = fatal）。"""
        report.time_offset_spot_ms = int(self.spot.calibrate())
        report.time_offset_perp_ms = int(self.futures.calibrate())
        limit = self.config.execution.server_time_offset_limit_ms
        for label, offset in (("spot", report.time_offset_spot_ms), ("perp", report.time_offset_perp_ms)):
            if abs(offset) > limit:
                raise ClockError(
                    f"{label} server time 偏移 {offset}ms 超过阈值 ±{limit}ms。"
                    "停止交易，重新同步系统 NTP（-1021 防线，§3.5）。"
                )
        return f"offset spot={report.time_offset_spot_ms}ms perp={report.time_offset_perp_ms}ms"

    def _stage_rules(self, report: StartupReport) -> str:
        """3. 规则与杠杆/保证金预检（§3.2 / 开发文档 §7.6，fatal）。"""
        spot_rules = self.spot.load_rules()
        perp_rules = self.futures.load_rules()
        common = sorted(set(spot_rules) & set(perp_rules))
        if not common:
            raise LiveGateBlocked("Spot 与 Futures 无共同可交易 symbol，拒绝启动")
        want_lev = self.config.execution.leverage
        want_margin = self.config.execution.margin_type
        # 动态候选池：先构建初始池，剔除不可开仓 symbol（无规则/非 TRADING/
        # 最小名义额高于 canary）——单个 pool symbol 不健康不应阻断整个服务；
        # 杠杆/保证金推迟到开仓时按需验证。
        dynamic_pool = self.strategy is not None and self.strategy.dynamic_pool
        if dynamic_pool and self.strategy is not None:
            self.strategy.refresh_universe()
            if self._pair_exclusion_fn is not None:
                allowed = {
                    s for s in self.strategy.candidate_symbols
                    if self._pair_exclusion_fn(s) is None
                }
            else:
                canary = Decimal(str(self.config.execution.canary_notional))
                allowed = set()
                for symbol in self.strategy.candidate_symbols:
                    spot_rule = spot_rules.get(symbol)
                    perp_rule = perp_rules.get(symbol)
                    if spot_rule is None or perp_rule is None:
                        continue
                    healthy = True
                    for rule in (spot_rule, perp_rule):
                        status = getattr(rule, "status", "")
                        if status and status != "TRADING":
                            healthy = False
                            break
                        min_notional = getattr(rule, "min_notional", None)
                        if min_notional is not None and canary < Decimal(str(min_notional)):
                            healthy = False
                            break
                    if healthy:
                        allowed.add(symbol)
            pruned = self.strategy.prune_universe(allowed)
            report.symbols = tuple(self.strategy.candidate_symbols)
            logger.info(
                "【动态候选池】启动初始池 %d 个 symbol（剔除不可开仓 %d 个）；"
                "杠杆/保证金开仓时按需验证",
                len(report.symbols), pruned,
            )
        else:
            report.symbols = tuple(common)
        symbol_detail: dict[str, Any] = {}
        for symbol in report.symbols:
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
            if dynamic_pool:
                # 动态池：开仓前 _check_leverage_margin 逐个验证（首次开仓该 symbol 时）
                info["leverage"] = want_lev
                info["margin_type"] = want_margin
                info["leverage_read_available"] = False
                info["leverage_checked_at"] = "on_open"
            else:
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
        return f"{len(report.symbols)} symbols"

    def _stage_lease(self) -> str:
        """4. 单实例锁 + 中断会话补记（fatal）。锁在 capture/对账前取得，
        防止对账期间另一进程写账本。"""
        try:
            lease = self.store.acquire_lease(self.lease_name)
        except LeaseConflict as exc:
            self._on_alert("LEASE_CONFLICT", str(exc))
            raise LiveGateBlocked(str(exc)) from exc
        self.lease_holder = lease.holder
        # 重启恢复（§断点重连）：把上次非优雅退出（kill -9/断电）遗留的
        # RUNNING/RECOVERY 会话补记为 INTERRUPTED。标记失败（StoreError）视为
        # 账本硬错误 → fatal。
        interrupted = self.store.mark_interrupted_sessions(now_ms=int(self._now() * 1000))
        if interrupted > 0:
            logger.info("【重启恢复】已标记 %d 个中断会话", interrupted)
        return lease.holder

    def _stage_capture(self, ctx: Any) -> str:
        """5. capture 交易所快照束（soft：失败进闸门 → 默认 RECOVERY）。"""
        if self.exch_sync is None:
            return "no-exch-sync"
        try:
            bundle = self.exch_sync.capture()
        except SyncCaptureError as exc:
            ctx.gates.append(f"capture 失败: {exc}")  # type: ignore[attr-defined]
            return "capture-failed"
        ctx.bundle = bundle  # type: ignore[attr-defined]
        return "captured"

    def _stage_facts_sync(self, ctx: Any) -> str:
        """6. facts 同步（fills + funding income，游标分页幂等；soft）。"""
        if self.exch_sync is None or ctx.bundle is None:  # type: ignore[attr-defined]
            return "skip"
        symbols = self._ledger_sync_symbols()
        fill_results = self.exch_sync.sync_fills(symbols)
        income_results = self.exch_sync.sync_funding_income(symbols)
        bad = [
            f"{r.market}/{r.stream}/{r.symbol}: {r.error or '不完整'}"
            for r in (*fill_results, *income_results)
            if r.error or not r.complete
        ]
        self._ledger_sync_ok = not bad
        self._ledger_sync_error = "; ".join(bad[:3])
        if bad:
            ctx.gates.append(f"事实同步未完成: {'; '.join(bad[:3])}")  # type: ignore[attr-defined]
            return "partial"
        return "ok"

    def _stage_reconcile(self, ctx: Any) -> str:
        """7. 启动对账（消费同一 bundle；soft：不一致 → 闸门）。"""
        result = self.reconciler.run(reason="startup", snapshot=ctx.bundle)  # type: ignore[arg-type]
        self._last_reconcile_ms = int(self._now() * 1000)
        ctx.report.reconciliation_consistent = result.consistent  # type: ignore[attr-defined]
        ctx.report.reconciliation_mismatches = result.mismatches  # type: ignore[attr-defined]
        ctx.reconcile_result = result  # type: ignore[attr-defined]
        return f"consistent={result.consistent}"

    def _stage_streams(self) -> str:
        """8. 用户数据流（全部达到新鲜前拒绝启动，fatal 超时）。"""
        for stream in self.streams:
            stream.start()
        fresh_deadline = self._now() + self.fresh_wait_seconds
        while any(not s.is_fresh for s in self.streams):
            if self._now() >= fresh_deadline:
                raise LiveGateBlocked(
                    f"用户流 {self.fresh_wait_seconds:.0f} 秒内未达新鲜状态，拒绝启动"
                )
            time.sleep(0.05)
        return f"{len(self.streams)} streams fresh"

    def _stage_session(self) -> str:
        """9. run_session（§6.1 关联链起点）+ heartbeat（DB 写失败 = fatal）。"""
        self._started = True
        self._ensure_run_session()
        if self.run_id:
            try:
                self.store.update_run_heartbeat(self.run_id, now_ms=int(self._now() * 1000))
            except StoreError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("启动 heartbeat 写入失败", exc_info=True)
        return self.run_id or "no-run-id"

    def _stage_account(self, ctx: Any) -> str:
        """10. 首次账户快照（消费同一 bundle；soft：失败 → 开仓被拒）。"""
        if self.account_builder is None:
            return "no-account-builder"
        try:
            self._refresh_account_state(source="startup", bundle=ctx.bundle)  # type: ignore[arg-type]
        except AccountStateError as exc:
            self.store.record_risk_decision("ACCOUNT_STATE", False, str(exc))
            self._on_alert("ACCOUNT_STATE_UNKNOWN", str(exc))
            return "failed"
        return "ok"

    def _stage_final_gates(self, ctx: Any) -> str:
        """11. 启动闸门（T4）：对账/账户完整/市场 READY/风控闸门全绿 →
        RESUME_AFTER_CHECKS → RUNNING；任一 false → 默认 RECOVERY。"""
        result = ctx.reconcile_result  # type: ignore[attr-defined]
        if result is not None and not result.can_open:
            ctx.gates.append(f"启动对账不一致: {list(result.mismatches)}")  # type: ignore[attr-defined]
        acct = self._account_result
        if self.account_builder is not None and (
            acct is None or not self._account_fresh(int(self._now() * 1000)) or not acct.complete
        ):
            ctx.gates.append("账户快照缺失/不完整（资金未知，禁止新增风险）")  # type: ignore[attr-defined]
        if not self._market_data_ready:
            ctx.gates.append("市场数据 epoch 未 READY（选币闸门）")  # type: ignore[attr-defined]
        if self.gate.state is not HaltState.NORMAL:
            ctx.gates.append(f"风控闸门 {self.gate.state.value}")  # type: ignore[attr-defined]
        if ctx.gates:  # type: ignore[attr-defined]
            self.enter_recovery("; ".join(ctx.gates))  # type: ignore[arg-type]
            return "recovery"
        self.resume_after_checks("启动预检+对账通过", source="startup")
        logger.info(
            "【实盘服务启动完成】mode=%s run_id=%s symbols=%s offset=%d/%dms",
            self.mode, self.run_id, list(ctx.report.symbols),  # type: ignore[attr-defined]
            ctx.report.time_offset_spot_ms,  # type: ignore[attr-defined]
            ctx.report.time_offset_perp_ms,  # type: ignore[attr-defined]
        )
        return "running"

    # -- 主循环 --------------------------------------------------------------

    def run_once(self) -> dict[str, Any]:
        """执行一轮主循环（T4：编排逻辑已迁移至 ``application.runner.ServiceRunner``，
        本方法保留兼容 facade，行为不变）。"""
        from ..application.runner import ServiceRunner

        return ServiceRunner(self).run_once()

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

    # ---- 领域生产流水线（实施计划书 4.0 T2） ------------------------------

    @property
    def _pipeline(self):
        """延迟组装的 ``application.composition.Pipeline``（单一 composition 入口）。

        T2 边界：pipeline 只做 proposal/intent 落账与 diff，不触 broker；
        T3 在此追加 RiskKernel → ApprovedIntent → ExecutionPlan → submit。
        """
        obj = self.__dict__.get("_pipeline_obj")
        if obj is None:
            from ..application.composition import Pipeline

            obj = Pipeline(planner=PortfolioPlanner(), ledger=self.store)
            self.__dict__["_pipeline_obj"] = obj
        return obj

    @property
    def _risk_kernel(self):
        """T3：同步风险内核（RiskRulesAdapter 复用 legacy RiskManager 数值规则）。"""
        obj = self.__dict__.get("_risk_kernel_obj")
        if obj is None:
            from ..risk.adapter import RiskRulesAdapter
            from ..risk.kernel import RiskKernel

            obj = RiskKernel(self.config, rules=RiskRulesAdapter(RiskManager(self.config.risk)))
            self.__dict__["_risk_kernel_obj"] = obj
        return obj

    @property
    def _order_planner(self):
        """T3：OrderPlanner（ApprovedIntent → ExecutionPlan）。"""
        obj = self.__dict__.get("_order_planner_obj")
        if obj is None:
            from ..execution.planner import OrderPlanner

            obj = OrderPlanner(
                strategy_version=(
                    self.strategy.strategy_version if self.strategy is not None else "funding_carry-1.0"
                )
            )
            self.__dict__["_order_planner_obj"] = obj
        return obj

    def _safety_state(self, now_ms: int) -> SafetyState:
        """T4：生产安全状态（owner：SafetyStateMachine）；风险内核据此审批。"""
        return self._safety_sm.state

    def _account_snapshot(self, now_ms: int) -> AccountSnapshot:
        """T3：账户状态 → 领域 AccountSnapshot（不完整/过期 → complete=False，fail closed）。"""
        res = self._account_result
        if res is None or not self._account_fresh(now_ms):
            return AccountSnapshot(
                snapshot_id="unknown",
                capture_start_ms=now_ms,
                capture_end_ms=now_ms,
                complete=False,
                equity=Decimal("0"),
                available=Decimal("0"),
            )
        return AccountSnapshot(
            snapshot_id=res.snapshot_id or f"acct-{res.ts_ms}",
            capture_start_ms=res.ts_ms,
            capture_end_ms=res.ts_ms,
            complete=res.complete,
            equity=res.total_equity,
            available=res.spot_available + res.futures_available,
        )

    def _entry_quote_verdict(self, quote: Quote, symbol: str) -> QuoteGateVerdict:
        """v5.0 T2（AC-04）：双腿报价 gate（纯函数；边界取自 execution 配置，
        时钟 = ``self._now()``，即下单前最后一刻的当前时间）。"""
        exc = self.config.execution
        return evaluate_quote_gate(
            quote,
            intent_symbol=symbol,
            now_ms=int(self._now() * 1000),
            max_age_ms=int(exc.max_market_data_age_seconds * 1000),
            max_skew_ms=int(exc.max_quote_skew_ms),
        )

    def _market_snapshot_for(self, symbol: str, now_ms: int) -> MarketSnapshot:
        """单 symbol 市场事实（T3：新增风险审批用）。

        报价缺失/过期/未来时间 → INCOMPLETE（风险层禁止新增风险）；
        不伪造价格。v5.0 T2（AC-04）：报价经 gate 判非 FRESH（陈旧/来源
        未知/skew 超限）时不得映射为 FRESH 市场事实（no stale-as-fresh）。
        """
        from ..domain.market import InstrumentQuote, MarketKind

        try:
            quote = self._quote_fetcher(symbol)
        except Exception:  # noqa: BLE001 - 报价获取失败 = 市场事实不可信
            quote = None
        # 新鲜度用**当前时钟**判定（非调用方传入的 tick 时刻）：报价在取数期间
        # 可能耗时 10-70s（限流排队/403 重试），用取数前的 now 比较会把刚接收的
        # 报价误判为"未来时间戳"。now_ms 只用于快照元数据。
        gate_now_ms = int(self._now() * 1000)
        if (
            quote is None
            or quote.ts_ms > gate_now_ms
            or quote.spot_price is None
            or quote.perp_price is None
        ):
            return MarketSnapshot(
                snapshot_id=f"no-quote-{symbol}",
                generated_at_ms=now_ms,
                decision_cutoff_ms=0,
                quality=DataQuality.INCOMPLETE,
                quotes=(),
            )
        verdict = self._entry_quote_verdict(quote, symbol)
        if verdict.quality is not DataQuality.FRESH:
            # 陈旧/不可证明新鲜的报价只能降低风险，不得作为 FRESH 市场事实
            return MarketSnapshot(
                snapshot_id=f"quote-rejected-{symbol}-{quote.ts_ms}",
                generated_at_ms=now_ms,
                decision_cutoff_ms=0,
                quality=verdict.quality,
                quotes=(),
            )
        return MarketSnapshot(
            snapshot_id=f"quote-{symbol}-{quote.ts_ms}",
            # 不变量 decision_cutoff_ms <= generated_at_ms：快照生成时刻用取数后
            # 时钟（gate_now_ms ≥ quote.ts_ms）；用 tick 开场的 now_ms 会在慢取数
            # 下被 quote.ts_ms 反超 → 审批异常（实测 RISK_ERROR）。
            generated_at_ms=gate_now_ms,
            decision_cutoff_ms=quote.ts_ms,
            quality=DataQuality.FRESH,
            quotes=(
                InstrumentQuote(
                    symbol=symbol,
                    market=MarketKind.SPOT,
                    price=quote.spot_price,
                    funding_rate_8h=None,
                    funding_cutoff_ms=None,
                    quote_time_ms=quote.ts_ms,
                ),
                InstrumentQuote(
                    symbol=symbol,
                    market=MarketKind.FUTURES,
                    price=quote.perp_price,
                    funding_rate_8h=None,
                    funding_cutoff_ms=None,
                    quote_time_ms=quote.ts_ms,
                ),
            ),
        )

    def _market_snapshot(self, now_ms: int) -> MarketSnapshot:
        """当前市场事实横截面。

        - 动态候选池：synchronizer 的 READY scan epoch → ``MarketDataService``
          （FRESH/STALE/…由 epoch 状态决定）。
        - 固定池（无 synchronizer）：以本轮刚拉取的报价构造 FRESH 快照；
          任一报价缺失/未来时间（无前瞻）→ INCOMPLETE（fail closed，
          风险层据此禁止新增风险）。
        """
        if self.synchronizer is not None:
            return MarketDataService(self.synchronizer, now_fn=self._now).snapshot()
        from ..domain.market import InstrumentQuote, MarketKind

        try:
            quotes = self._fetch_quotes()
        except Exception:  # noqa: BLE001 - 报价拉取失败 = 市场事实不可信
            quotes = {}
        if not quotes:
            return MarketSnapshot(
                snapshot_id="no-quotes",
                generated_at_ms=now_ms,
                decision_cutoff_ms=0,
                quality=DataQuality.INCOMPLETE,
                quotes=(),
            )
        instrument_quotes: list[InstrumentQuote] = []
        bad = False
        for symbol in sorted(quotes):
            q = quotes[symbol]
            if q is None or q.ts_ms > now_ms or q.spot_price is None or q.perp_price is None:
                bad = True  # 缺失/未来报价：不伪造价格
                continue
            instrument_quotes.append(
                InstrumentQuote(
                    symbol=symbol,
                    market=MarketKind.SPOT,
                    price=q.spot_price,
                    funding_rate_8h=None,
                    funding_cutoff_ms=None,
                    quote_time_ms=q.ts_ms,
                )
            )
            instrument_quotes.append(
                InstrumentQuote(
                    symbol=symbol,
                    market=MarketKind.FUTURES,
                    price=q.perp_price,
                    funding_rate_8h=None,
                    funding_cutoff_ms=None,
                    quote_time_ms=q.ts_ms,
                )
            )
        quality = DataQuality.FRESH if (instrument_quotes and not bad) else DataQuality.INCOMPLETE
        cutoff = min(q.ts_ms for q in quotes.values() if q is not None and q.ts_ms <= now_ms) if instrument_quotes else 0
        return MarketSnapshot(
            snapshot_id=f"quotes-{now_ms}",
            generated_at_ms=now_ms,
            decision_cutoff_ms=cutoff,
            quality=quality,
            quotes=tuple(instrument_quotes),
        )

    def _portfolio_view(self, now_ms: int) -> PortfolioView:
        """ledger current projection → 只读 PortfolioView（查询事实源）。"""
        return PortfolioPlanner.view_from_ledger_positions(
            self.store.current_positions(), as_of_ms=now_ms
        )

    def _build_strategy_proposal(
        self,
        decisions,
        view: PortfolioView,
        market_snap: MarketSnapshot,
        now_ms: int,
    ) -> StrategyProposal:
        """legacy 策略决策 → 领域 StrategyProposal（数值语义不变，adapter 转换）。"""
        run_id = self.run_id or ""
        return decisions_to_proposal(
            decisions,
            view,
            proposal_id=f"{run_id}:{market_snap.snapshot_id}:{market_snap.decision_cutoff_ms}",
            snapshot_id=market_snap.snapshot_id,
            decision_cutoff_ms=market_snap.decision_cutoff_ms,
            strategy_version=(
                self.strategy.strategy_version if self.strategy is not None else "funding_carry-1.0"
            ),
            config_hash=self.config_hash,
            valid_until_ms=now_ms + 30_000,
        )

    def run_forever(self, *, tick_seconds: float = 5.0, stop_check: Callable[[], bool] | None = None) -> bool:
        """阻塞主循环（T4：watchdog/异常预算/停机逻辑已迁移至
        ``application.runner.ServiceRunner.run_forever``，本方法保留兼容 facade）。"""
        from ..application.runner import ServiceRunner

        return ServiceRunner(self).run_forever(tick_seconds=tick_seconds, stop_check=stop_check)

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
        self._safety_sm.shutdown(reason="graceful_stop", now_ms=int(self._now() * 1000))
        self._state = ServiceState.STOPPED
        if self.synchronizer is not None:
            try:
                self.synchronizer.stop()
            except Exception:  # noqa: BLE001 —— 超时只告警，不阻止安全停机
                logger.warning("market-data-sync 停止超时（继续停机）")
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

    v5.0 T2（AC-04）provenance：两腿各自的本地接收时间 + symbol +
    ``source="rest"``；REST 报价不提供交易所时间戳 → 记 None（不得用本地
    时刻冒充）。
    """

    def fetch(symbol: str) -> Quote | None:
        from ..rate_limit import RequestPriority

        # 提交前报价 = 下单关键路径：P0 优先级，不与 P3 候选刷新抢预算排队
        try:
            t0 = time.time()
            spot_price = public_client.spot_price(
                symbol, priority=RequestPriority.P0_CRITICAL
            )
            t1 = time.time()
            spot_received_ms = int(time.time() * 1000)
            perp_payload = public_client.premium_index(
                symbol, priority=RequestPriority.P0_CRITICAL
            )
            # premiumIndex 无 lastPrice；用永续标记价（markPrice，标准参考价）
            perp_price = Decimal(str(perp_payload["markPrice"]))
            perp_received_ms = int(time.time() * 1000)
            total = time.time() - t0
            if total > 5.0:
                logger.warning(
                    "报价获取耗时 %.1fs（spot %.2fs / perp %.2fs）: %s",
                    total, t1 - t0, time.time() - t1, symbol,
                )
            return Quote(
                spot_price=Decimal(str(spot_price)),
                perp_price=perp_price,
                ts_ms=spot_received_ms,
                spot_ts_ms=spot_received_ms,
                perp_ts_ms=perp_received_ms,
                symbol=symbol,
                source="rest",
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
    from ..data.venue import venue_for_execution_mode
    from ..rate_limit import RateLimitCoordinator, RateLimitScope

    # T1/T3：同一进程全部 Spot/Futures 请求共享一个限流协调器
    #（公开行情/候选 P3，账户/恢复 P1，下单 P0 保留预算）
    data_cfg = config.data.rate_limit
    coordinator = RateLimitCoordinator(
        {
            RateLimitScope.SPOT: int(data_cfg.spot_weight_per_min),
            RateLimitScope.FUTURES: int(data_cfg.futures_weight_per_min),
        },
        soft_limit_ratio=float(data_cfg.soft_limit_ratio),
        critical_reserve_ratio=float(data_cfg.critical_reserve_ratio),
        freeze_seconds=float(data_cfg.rate_limit_freeze_seconds),
        ban_seconds=float(data_cfg.ip_ban_seconds),
    )
    public_client = BinancePublicClient(
        api,
        config.data,
        universe=config.universe,
        # 显式按实际签名端点推导 venue，而不是只看 exc.mode：
        # 调用方可传入自定义 base（如经典 testnet），端点与 mode 必须一致。
        venue=venue_for_execution_mode(exc.mode),
        rate_limiter=coordinator,
    )
    logger.info(
        "公开数据 venue=%s（futures=%s spot=%s，is_testnet=%s）",
        public_client.venue_name,
        public_client.futures_base,
        public_client.spot_base,
        is_testnet,
    )
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
        rate_limiter=coordinator,
    )
    futures_client = SignedClient(
        base_url=futures_url,
        api_key=SecretStr(futures_key, name="FUT_KEY") if isinstance(futures_key, str) else futures_key,
        secret=SecretStr(futures_secret, name="FUT_SECRET") if isinstance(futures_secret, str) else futures_secret,
        market="perp",
        recv_window_ms=exc.recv_window_ms,
        timeout_seconds=exc.request_timeout_seconds,
        read_max_retries=5,  # 代理链路偶发 SSL 中断，读请求多一层重试裕量
        rate_limiter=coordinator,
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
    reconciler = Reconciler(
        store,
        spot,
        futures,
        ignore_assets=exc.reconcile_ignore_assets,
        relative_qty_tolerance=Decimal(str(exc.reconciliation_qty_tolerance_pct)),
    )
    exch_sync = ExchangeStateSynchronizer(
        store=store, spot=spot, futures=futures, config=config
    )

    provider = PublicDataStrategyProvider(config, client=public_client)
    # 可开仓对预筛（§启动预检同一口径）：无现货/合约交易对、非 TRADING、
    # 最小名义额超 canary → epoch 构建时直接 excluded，不进入后续筛选。
    # 规则 5 分钟刷新（exchangeInfo 变化不频繁）；规则不可用时返回 None
    # （≠确定性排除，放行后由开仓闸门再次拦截）。
    canary_notional = Decimal(str(config.execution.canary_notional))
    _rules_cache: dict[str, Any] = {"ts": 0.0, "spot": {}, "perp": {}}

    def _pair_exclusion_reason(symbol: str) -> str | None:
        if time.monotonic() - float(_rules_cache["ts"]) > 300:
            try:
                _rules_cache["spot"] = spot.load_rules()
                _rules_cache["perp"] = futures.load_rules()
                _rules_cache["ts"] = time.monotonic()
            except Exception:  # noqa: BLE001 —— 刷新失败保留旧规则
                logger.warning("现货/合约规则刷新失败（保留旧规则）", exc_info=True)
        spot_map = _rules_cache["spot"]
        perp_map = _rules_cache["perp"]
        if not spot_map or not perp_map:
            return None
        spot_rule = spot_map.get(symbol)
        if spot_rule is None:
            return "无现货交易对"
        perp_rule = perp_map.get(symbol)
        if perp_rule is None:
            return "无合约交易对"
        for name, rule in (("现货", spot_rule), ("合约", perp_rule)):
            status = getattr(rule, "status", "")
            if status and status != "TRADING":
                return f"{name}非TRADING({status})"
            min_notional = getattr(rule, "min_notional", None)
            if min_notional is not None and canary_notional < Decimal(str(min_notional)):
                return f"{name}最小名义额{min_notional}超canary"
        return None

    def _long_history_symbols() -> set[str]:
        """返回仍需退出指标的 symbol；失败时保守回退空集。"""
        symbols: set[str] = set()
        try:
            symbols.update(
                str(row["symbol"])
                for row in store.current_positions()
                if row.get("symbol")
            )
            symbols.update(str(symbol) for symbol in store.expected_positions())
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取持仓长历史 symbol 失败: %s", exc)
        return symbols

    # T2/T3：scan epoch 同步器（后台有界并发，交易主循环只读 READY 快照）
    def _persist_scan_epoch(epoch: Any, snapshots: Any) -> None:
        evaluator = FundingCarryEvaluator(config=config)
        entry_cfg = config.strategy.entry
        min_volume = Decimal(str(config.strategy.selection.min_quote_volume_3d_avg))
        threshold = max(
            Decimal(str(entry_cfg.min_trailing_annualized)),
            Decimal(str(entry_cfg.min_annualized_rate)),
        )
        final_rows = []
        for snap in snapshots.values():
            if snap.error or snap.volume_status != "FETCHED" or snap.quote_volume_3d_avg < min_volume:
                continue
            candidate = CandidateInput(
                symbol=snap.symbol,
                rates=snap.rates,
                mark_prices=snap.mark_prices,
                timestamps=snap.timestamps,
                interval_hours=snap.interval_hours,
                volume_3d_avg=snap.quote_volume_3d_avg,
                refreshed_ts_ms=snap.fetched_ms,
            )
            trailing, streak = evaluator.entry_metrics(candidate)
            if trailing < threshold or streak < entry_cfg.min_consecutive_positive:
                continue
            final_rows.append(snap)
        snapshot_rows = [
            {
                "symbol": snap.symbol,
                "interval_hours": snap.interval_hours,
                "rates": snap.rates,
                "mark_prices": snap.mark_prices,
                "timestamps": snap.timestamps,
                "expected_last_funding_ms": snap.expected_last_funding_ms,
                "volume_window_end_ms": snap.volume_window_end_ms,
                "quote_volume_3d_avg": snap.quote_volume_3d_avg,
                "fetched_ms": snap.fetched_ms,
                "error": snap.error,
                "volume_status": snap.volume_status,
                "settle_interval_hours": snap.settle_interval_hours,
                "latest_settled_rate": snap.latest_settled_rate,
                "latest_settled_ms": snap.latest_settled_ms,
            }
            for snap in final_rows
        ]
        store.record_scan_epoch(
            epoch_id=epoch.epoch_id,
            universe_snapshot_ts_ms=epoch.universe_snapshot_ts_ms,
            decision_cutoff_ms=epoch.decision_cutoff_ms,
            status=epoch.status.value,
            expected=epoch.expected_symbols,
            excluded=dict(epoch.excluded),
            failed=dict(epoch.failed),
            created_ms=epoch.created_ms,
            completed_ms=epoch.completed_ms,
            expires_ms=epoch.expires_ms,
            error=epoch.error,
            snapshots=snapshot_rows,
        )

    synchronizer = MarketDataSynchronizer(
        config=config,
        data=provider,
        server_time_fn=public_client.futures_time,
        exclusion_fn=_pair_exclusion_reason,
        long_history_symbols_fn=_long_history_symbols,
        epoch_persist_fn=_persist_scan_epoch,
    )

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
            data=provider,
            store=store,
            strategy_version=executor.strategy_version,
            config_hash=config_hash,
            synchronizer=synchronizer,
            exclusion_fn=_pair_exclusion_reason,
        ),
        account_builder=AccountStateBuilder(
            config=config, store=store, candidate_symbols=tuple(exc.live_symbols)
        ),
        quote_fetcher=_make_quote_fetcher(public_client),
        config_hash=config_hash,
        spot_endpoint=spot_url,
        futures_endpoint=futures_url,
        synchronizer=synchronizer,
        exch_sync=exch_sync,
        pair_exclusion_fn=_pair_exclusion_reason,
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
    synchronizer.start()
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
