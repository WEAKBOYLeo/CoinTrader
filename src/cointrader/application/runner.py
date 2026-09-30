"""应用编排 runner：tick 循环、恢复、心跳、看门狗与停机（T4，AC-07）。

``ServiceRunner`` 是**唯一装配/编排层**的生命周期核心：

- ``run_once()``：一轮主循环（心跳/租约续期 → 闸门同步 → 用户流新鲜度 →
  RECOVERY 恢复 gate → 周期对账/采样 → 策略评估 → 先平仓后开仓）。
  代码自 ``LiveService.run_once`` 逐行迁移（``self.`` → ``svc.``），行为不变。
- ``run_forever()``：阻塞主循环（LoopWatchdog + tick 异常预算 + 优雅停机），
  自 ``LiveService.run_forever`` 逐行迁移。

``LiveService`` 保留兼容 facade：其 ``run_once``/``run_forever`` 委托到
本 runner。查询层（WebUI/CLI/reporting）只读 ledger read model。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from ..domain.market import DataQuality
from ..domain.risk import ApprovedIntent
from ..execution.pair_executor import PairPrecheckFailed
from ..execution.planner import PlanError
from ..execution.risk_gate import HaltState
from ..execution.store import StoreError
from ..execution.sync import SyncCaptureError
from ..live.account_state import AccountStateError
from ..live.decisions import DecisionKind, StrategyDecision
from ..live.service import ServiceState

if TYPE_CHECKING:  # pragma: no cover - 仅类型
    from ..live.service import LiveService

logger = logging.getLogger(__name__)

__all__ = ["ServiceRunner"]


class _PhaseTimer:
    """tick 分阶段耗时诊断：单阶段 >10s 告警（限流排队/403 重试/网络卡点定位）。"""

    def __init__(self, name: str, now_fn: Callable[[], float]) -> None:
        self._name = name
        self._now = now_fn
        self._t0 = 0.0

    def __enter__(self) -> _PhaseTimer:
        self._t0 = self._now()
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        elapsed = self._now() - self._t0
        if elapsed > 10.0:
            logger.warning("【tick 阶段耗时】%s: %.1fs", self._name, elapsed)
        return False


class ServiceRunner:
    """LiveService 生命周期编排（runner 自身无可变状态）。"""

    def __init__(self, service: LiveService) -> None:
        self._service = service

    def run_once(self) -> dict[str, Any]:
        """执行一轮主循环。可独立单测（不阻塞、不 sleep）。"""
        svc = self._service

        if svc._state is ServiceState.STOPPED:
            return {"state": "STOPPED"}

        # v2 心跳：每轮（成功或受控 RECOVERY）更新；异常会话下次接管按
        # heartbeat 截断，停机间隔不计入在线时长（T3/AC-09）。
        if svc.run_id:
            try:
                svc.store.update_run_heartbeat(svc.run_id, now_ms=int(svc._now() * 1000))
            except StoreError as exc:
                logger.warning("heartbeat 写入失败: %s", exc)
            except Exception:  # noqa: BLE001
                logger.warning("heartbeat 写入异常", exc_info=True)

        # 单实例锁续期（长跑韧性）：TTL 30s、主循环 5s 间隔，裕量 6 倍。
        # 刷新失败只告警不崩溃（DB 写坏会在后续步骤自然暴露）。
        if svc.lease_holder is not None:
            try:
                svc.store.refresh_lease(svc.lease_name, svc.lease_holder)
            except StoreError as exc:
                logger.warning("单实例锁刷新失败（后续写账本会自然暴露）: %s", exc)

        # 风控闸门状态同步（T4：经 ControlPublisher 进入安全状态机，只加严）
        gate_state = svc.gate.state
        if gate_state is not HaltState.NORMAL and svc._state is not ServiceState.RECOVERY:
            svc.apply_gate_state()
            svc._persist_runtime_state()
            svc._on_alert("GATE_HALT", f"风控闸门状态 {gate_state.value}，禁止开新仓")
            return {"state": "HALTED"}
        if svc._state is ServiceState.HALTED and gate_state is HaltState.NORMAL:
            # 闸门恢复（recover 已含对账+预检证明）且用户流仍新鲜 → 回 RUNNING。
            # 状态机在 CLOSE_ONLY 时 service 态 = HALTED；恢复 = 显式两步
            # （CLOSE_ONLY → RECOVERY → RESUME_AFTER_CHECKS，由 resume 完成）。
            svc.resume_after_checks("闸门恢复（recover 已含对账+预检证明）", source="gate_recover")
            logger.warning("【闸门恢复】回到 RUNNING")

        # 用户流新鲜度（§3.4：断线/过期 = 状态不可信）
        now_ms = int(svc._now() * 1000)
        stale = [s for s in svc.streams if not s.is_fresh]
        if stale:
            detail = ", ".join(f"{s.market}（代次 {s.generation}）" for s in stale)
            reason = f"用户流不新鲜/不可信: {detail}"
            svc.enter_recovery(reason)
            svc._update_recovery_diagnostic(
                reason,
                code="USER_STREAM_UNTRUSTED",
                details={"streams": [{"market": s.market, "generation": s.generation} for s in stale]},
                retry=False,
            )
            svc._persist_runtime_state()
            return {"state": "RECOVERY", "reason": svc._recovery_reason}

        # RECOVERY 自动解除（长跑自愈）：全部用户流恢复新鲜后做一次恢复对账，
        # 通过且风控闸门 NORMAL → 回 RUNNING，无需重启进程。
        # 对账按周期节流，避免 RECOVERY 期间每个 tick 全量对账。
        if svc._state is ServiceState.RECOVERY:
            interval_ms = int(svc.config.execution.reconciliation_interval_seconds * 1000)
            if now_ms - svc._last_reconcile_ms < interval_ms:
                return {"state": "RECOVERY", "reason": svc._recovery_reason}
            svc._record_recovery_retry(now_ms)
            svc._persist_runtime_state()
            bundle = None
            if svc.exch_sync is not None:
                try:
                    bundle = svc.exch_sync.capture()
                    fill_results = svc.exch_sync.sync_fills(svc._ledger_sync_symbols())
                    income_results = svc.exch_sync.sync_funding_income(svc._ledger_sync_symbols())
                    bad = [
                        f"{r.market}/{r.stream}/{r.symbol}: {r.error or '不完整'}"
                        for r in (*fill_results, *income_results)
                        if r.error or not r.complete
                    ]
                    svc._ledger_sync_ok = not bad and bundle.complete
                    svc._ledger_sync_error = "; ".join(bad[:3])
                except SyncCaptureError as exc:
                    svc._recovery_reason = f"capture 失败: {exc}"
                    svc._update_recovery_diagnostic(
                        svc._recovery_reason,
                        code="EXCHANGE_CAPTURE_FAILED",
                        retry=False,
                    )
                    svc._persist_runtime_state()
                    return {"state": "RECOVERY", "reason": svc._recovery_reason}
            result = svc.reconciler.run(reason="recovery_check", snapshot=bundle)
            svc._last_reconcile_ms = now_ms
            if not (result.can_open and svc._ledger_sync_ok and svc.gate.state is HaltState.NORMAL):
                svc._recovery_reason = (
                    f"流恢复后对账/闸门未通过: {list(result.mismatches)}"
                    if not result.can_open
                    else (
                        "账本同步未通过"
                        if not svc._ledger_sync_ok
                        else f"风控闸门未恢复: {svc.gate.state.value}"
                    )
                )
                code = (
                    "RECONCILIATION_MISMATCH"
                    if not result.can_open
                    else "LEDGER_SYNC_FAILED"
                    if not svc._ledger_sync_ok
                    else "RISK_GATE"
                )
                svc._update_recovery_diagnostic(
                    svc._recovery_reason,
                    code=code,
                    details={"mismatches": list(result.mismatches)},
                    retry=False,
                )
                svc._persist_runtime_state()
                return {"state": "RECOVERY", "reason": svc._recovery_reason}
            svc._reconcile_ok = True
            svc._recovery_reason = ""
            svc.resume_after_checks(
                "用户流全部新鲜 + 对账通过 + 闸门 NORMAL", source="recovery_check"
            )
            svc._refresh_held()
            svc._persist_runtime_state()
            logger.warning("【RECOVERY 解除】用户流全部新鲜 + 对账通过 + 闸门 NORMAL，回到 RUNNING")

        if svc._state is not ServiceState.RUNNING:
            return {"state": "RECOVERY", "reason": svc._recovery_reason}

        opened: list[str] = []
        skipped: list[str] = []
        closed: list[str] = []

        # 周期性重校准 server time（§3.5）：代理 RTT 漂移会让启动时偏移过期→ -1021。
        # 校准失败只告警保留旧偏移；偏移超限才进 RECOVERY（时钟真的坏了）。
        resync_ms = int(svc.config.execution.time_resync_seconds * 1000)
        if now_ms - svc._last_resync_ms >= resync_ms:
            svc._last_resync_ms = now_ms
            offset_limit = svc.config.execution.server_time_offset_limit_ms
            for label, adapter in (("spot", svc.spot), ("perp", svc.futures)):
                try:
                    offset = adapter.calibrate()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("重校准 server time 失败（%s），保留旧偏移: %s", label, exc)
                    continue
                if abs(offset) > offset_limit:
                    svc.enter_recovery(f"{label} server time 偏移 {offset}ms 超阈值 ±{offset_limit}ms（-1021 防线）")
                    svc._persist_runtime_state()
                    return {"state": "RECOVERY", "reason": svc._recovery_reason}

        # 周期性对账（§8.5；T3：复用短期 capture bundle + 增量 facts 同步）
        interval_ms = int(svc.config.execution.reconciliation_interval_seconds * 1000)
        if now_ms - svc._last_reconcile_ms >= interval_ms:
            with _PhaseTimer("reconcile", svc._now):
                if not svc._periodic_reconcile(now_ms):
                    svc._persist_runtime_state()
                    return {"state": "RECOVERY", "reason": svc._recovery_reason}
            # 对账通过后以交易所真实持仓为准（§7.3 去重基础）
            with _PhaseTimer("refresh_held", svc._now):
                svc._refresh_held()

        # 周期性采样：账户快照 / 持仓快照 / 资金费（§8.5 至少 30s）
        snapshot_ms = int(svc.config.execution.snapshot_interval_seconds * 1000)
        if svc.strategy is not None and now_ms - svc._last_snapshot_ms >= snapshot_ms:
            with _PhaseTimer("snapshot", svc._now):
                svc._last_snapshot_ms = now_ms
                bundle = None
                if svc.exch_sync is not None:
                    try:
                        # 复用窗口内返回同一 bundle（single-flight），不重复拉 API
                        bundle = svc.exch_sync.capture()
                    except SyncCaptureError:
                        bundle = None
                try:
                    svc._refresh_account_state(source="periodic", bundle=bundle)
                except (AccountStateError, StoreError) as exc:
                    svc._account_result = None
                    svc.store.record_risk_decision("ACCOUNT_STATE", False, str(exc))
                    svc._on_alert("ACCOUNT_STATE_UNKNOWN", f"账户快照失败: {exc}（开仓将被拒绝）")
                quotes = svc._fetch_quotes()
                try:
                    svc._sample_position_snapshots(quotes)
                    svc._collect_funding_cashflows()
                except StoreError as exc:
                    svc.enter_recovery(f"账本写入失败: {exc}")
                    return {"state": "RECOVERY", "reason": svc._recovery_reason}

        # 候选刷新：每 tick 调用；strategy 内部按各币结算周期判超龄、按分钟限流量，
        # 无超龄币时纯内存判断（§7.1）
        if svc.strategy is not None:
            with _PhaseTimer("refresh_candidates", svc._now):
                svc.strategy.refresh_candidates()

        # 无策略协调器：兼容旧 signal_provider 路径（仅测试/过渡期）
        if svc.strategy is None:
            for signal in svc._signal_provider():
                if svc.store.active_pair_for_symbol(signal.symbol):
                    skipped.append(f"{signal.symbol}: 已有未完结 pair")
                    continue
                # 计划 4.0 T2（T2.4）：raw signal → executor.open_pair 直调路径已移除。
                # 信号必须经领域 pipeline（strategy → proposal → intent → T3 风险/执行），
                # legacy signal_provider 不再直接下单。
                svc._on_alert(
                    "SIGNAL_PATH_DISABLED",
                    f"legacy signal_provider 路径不直接下单（领域 pipeline 待接入）: "
                    f"{signal.symbol} notional={signal.target_notional}",
                )
                svc.metrics.inc("strategy_signal_count", labels={"symbol": signal.symbol})
            svc._persist_runtime_state()
            return {"state": "RUNNING", "opened": opened, "skipped": skipped, "closed": closed}

        # ---- 实时策略路径（工作包一） ----
        with _PhaseTimer("build_context", svc._now):
            ctx = svc._build_context(now_ms)
        try:
            with _PhaseTimer("evaluate", svc._now):
                decisions = svc.strategy.evaluate(ctx)
            # 动态候选池：PENDING_QUOTE 中间态 → 按需获取新鲜报价后定案；
            # 中间态本身不落账本，只记录最终决策（每 symbol 每轮仍恰好一条）。
            resolved: list[StrategyDecision] = []
            with _PhaseTimer("pending_quotes", svc._now):
                for decision in decisions:
                    if decision.decision_kind is DecisionKind.PENDING_QUOTE:
                        quote = svc._quote_fetcher(decision.symbol)
                        # 闸门时刻 = 定案时的当前时间（非 tick 开场的 ctx.now）：
                        # 慢 tick（限流/403 重试）下刚接收的报价不得被误判未来时间戳
                        decision = svc.strategy.complete_open(
                            decision.symbol, ctx, quote,
                            gate_now_ms=int(svc._now() * 1000),
                        )
                    resolved.append(decision)
            for decision in resolved:
                svc.store.record_signal_decision(decision)
            decisions = resolved
        except StoreError as exc:
            svc.enter_recovery(f"signal_decision 写账本失败: {exc}")
            return {"state": "RECOVERY", "reason": svc._recovery_reason}

        # 1) 领域生产流水线（计划 4.0 T2/T3）：
        #    proposal → intent（先落账 + 指纹幂等）→ RiskKernel 审批（RiskDecision
        #    先落账）→ ApprovedIntent → ExecutionPlan（落账）→ 计划化执行入口
        #    ``open_pair_from_plan``（开仓/平仓统一；平仓全 reduce-only）。
        #    生产路径不存在 raw ``executor.open_pair``/``executor.close_pair`` 直调。
        view = svc._portfolio_view(now_ms)
        market_snap = svc._market_snapshot(now_ms)
        proposal = svc._build_strategy_proposal(decisions, view, market_snap, now_ms)
        _, new_intents = svc._pipeline.run_round(
            proposal=proposal, view=view, now_ms=now_ms, run_id=svc.run_id or ""
        )
        intent_ids: list[str] = []
        exit_reasons = {
            d.symbol: (d.reason_code, d.reason_text) for d in decisions
            if d.decision_kind in (DecisionKind.EXIT, DecisionKind.REPLACE)
        }
        for intent in new_intents:
            intent_ids.append(intent.intent_id)
            svc.metrics.inc(
                "pipeline_intent_count",
                labels={"action": intent.action.value, "symbol": intent.symbol},
            )
            symbol = intent.symbol
            if svc.mode == "shadow":
                svc._on_alert(
                    "SHADOW_INTENT",
                    f"shadow 模式只记录意图: {symbol} {intent.action.value} "
                    f"spot={format(intent.target_spot_notional, 'f')} "
                    f"perp={format(intent.target_perp_notional, 'f')}",
                )
                continue

            # -- 风险审批（先写 RiskDecision，fail closed） --
            try:
                # 先取市场快照（内部取报价，可能耗时 10-70s），**之后**再取当前时间：
                # 参数求值顺序若先算 now_ms 会早于快照生成时刻 → 误判无前瞻违规
                market = None if intent.is_closing else svc._market_snapshot_for(
                    symbol, int(svc._now() * 1000)
                )
                decision = svc._risk_kernel.approve(
                    intent,
                    now_ms=int(svc._now() * 1000),  # 审批时刻 = 快照生成后的当前
                    safety=svc._safety_state(now_ms),
                    market=market,
                    account=None if intent.is_closing else svc._account_snapshot(now_ms),
                    risk_state=svc._risk_state_fn(),
                )
            except Exception as exc:  # noqa: BLE001 - 审批异常不得当通过
                svc._on_alert("RISK_ERROR", f"风险审批异常: {symbol}: {exc}")
                skipped.append(f"{symbol}: 风险审批异常")
                continue
            decision_payload = dict(decision.to_dict())
            decision_payload["run_id"] = svc.run_id or ""
            svc.store.append_risk_decision(decision_payload)
            if not decision.allows_execution:
                skipped.append(f"{symbol}: 风险 {decision.decision.value}: {decision.reason}")
                svc._on_alert(
                    "RISK_BLOCKED",
                    f"风险审批未放行: {symbol} {decision.decision.value} {decision.reason}",
                )
                continue
            try:
                approved = ApprovedIntent.from_decision(intent, decision, now_ms=now_ms)
            except Exception as exc:  # noqa: BLE001 - 审批证据异常不得进入执行
                svc._on_alert("RISK_ERROR", f"ApprovedIntent 构造失败: {symbol}: {exc}")
                skipped.append(f"{symbol}: 审批证据无效")
                continue

            # -- 执行计划（执行前重新获取新鲜报价；平仓取交易所实际持仓量） --
            quote = svc._quote_fetcher(symbol)
            if quote is None:
                skipped.append(f"{symbol}: 提交前报价获取失败")
                continue
            # v5.0 T2（AC-04）：OrderPlanner/submit 紧邻处重验双腿报价
            # （time-of-check == time-of-use，消除检查/使用间隙）。
            # OPEN/增加风险：非 FRESH 不建 plan、不调用 executor（broker 提交 0）；
            # CLOSE/reduce-only：不被 entry freshness gate 阻塞，仅记录报价质量。
            verdict = svc._entry_quote_verdict(quote, symbol)
            if not intent.is_closing and verdict.quality is not DataQuality.FRESH:
                skipped.append(
                    f"{symbol}: 入场报价被拒 ({verdict.quality.value}): {verdict.reason}"
                )
                svc._on_alert(
                    "QUOTE_GATE_REJECTED",
                    f"{symbol} 入场报价被拒 ({verdict.quality.value}, "
                    f"source={quote.source!r}, {verdict.reason})",
                )
                continue
            if intent.is_closing and verdict.quality is not DataQuality.FRESH:
                # 平仓不受 entry gate 阻塞；保留可观测性（计划 payload 同步记录）
                logger.info(
                    "平仓报价非 FRESH（不阻塞 reduce-only）: %s %s: %s",
                    symbol, verdict.quality.value, verdict.reason,
                )
            close_spot_qty: Decimal | None = None
            close_perp_qty: Decimal | None = None
            if intent.is_closing:
                try:
                    close_spot_qty = svc.spot.balances().get(
                        symbol.replace("USDT", ""), Decimal("0")
                    )
                    close_perp_qty = abs(svc.futures.position_qty(symbol))
                except Exception as exc:  # noqa: BLE001
                    svc.enter_recovery(f"平仓前持仓查询失败: {symbol}: {exc}")
                    svc._persist_runtime_state()
                    return {"state": svc.state.value, "reason": svc._recovery_reason}
                if close_spot_qty <= 0 and close_perp_qty <= 0:
                    # 交易所无实际持仓：无可平，跳过（不造 plan）
                    skipped.append(f"{symbol}: 无实际持仓，跳过平仓")
                    continue
            try:
                plan = svc._order_planner.plan(
                    approved,
                    spot_price=quote.spot_price,
                    perp_price=quote.perp_price,
                    spot_rules=svc.spot.rule(symbol),
                    perp_rules=svc.futures.rule(symbol),
                    close_spot_qty=close_spot_qty,
                    close_perp_qty=close_perp_qty,
                    now_ms=int(svc._now() * 1000),
                    plan_id=f"plan-{intent.intent_id}",
                )
            except PlanError as exc:
                skipped.append(f"{symbol}: 规划失败: {exc}")
                svc._on_alert("PLAN_FAILED", f"{symbol}: {exc}")
                continue
            plan_payload = dict(plan.to_dict())
            plan_payload["run_id"] = svc.run_id or ""
            # v5.0 T2（AC-04）：quote snapshot provenance 与 intent/plan 关联
            # （plan_id = plan-<intent_id> 已绑定 intent；payload 为自由 JSON，
            # 无需 schema migration）。
            plan_payload["quote"] = {
                "symbol": quote.symbol,
                "source": quote.source,
                "received_at_ms": quote.received_at_ms,
                "spot_received_ms": quote.spot_received_ms,
                "perp_received_ms": quote.perp_received_ms,
                "spot_exchange_ts_ms": quote.spot_exchange_ts_ms,
                "perp_exchange_ts_ms": quote.perp_exchange_ts_ms,
                "connection_generation": quote.connection_generation,
                "gate_now_ms": int(svc._now() * 1000),
                "gate_quality": verdict.quality.value,
                "gate_reason": verdict.reason,
            }
            svc.store.append_execution_plan(plan_payload)

            # 杠杆/保证金：固定池启动预检已验证；动态池 symbol 首次开仓时验证一次
            if not intent.is_closing and symbol not in svc._leverage_checked:
                lev, margin, _read = svc._check_leverage_margin(
                    symbol, svc.config.execution.leverage, svc.config.execution.margin_type
                )
                if (lev, margin) != (svc.config.execution.leverage, svc.config.execution.margin_type):
                    skipped.append(f"{symbol}: 杠杆/保证金 ({lev}, {margin}) 与配置不符，放弃开仓")
                    svc._on_alert(
                        "OPEN_FAILED", f"{symbol} 杠杆/保证金模式不符: ({lev}, {margin})"
                    )
                    continue
                svc._leverage_checked.add(symbol)

            # -- 计划化执行（开仓/平仓统一入口；raw open_pair/close_pair 已禁用） --
            try:
                if intent.is_closing:
                    code, text = exit_reasons.get(symbol, ("PLAN_CLOSE", ""))
                    close_reason = f"{code}: {text}".rstrip(": ")
                else:
                    close_reason = f"{intent.action.value}:{symbol}"
                pair = svc.executor.open_pair_from_plan(
                    plan,
                    spot_price=quote.spot_price,
                    perp_price=quote.perp_price,
                    quote_ts_ms=quote.ts_ms,
                    state=svc._risk_state_fn(),
                    reason=close_reason,
                    run_id=svc.run_id or "",
                )
            except PairPrecheckFailed as exc:
                skipped.append(f"{symbol}: 计划预检失败: {exc}")
                svc._on_alert("PLAN_PRECHECK_FAILED", f"{symbol}: {exc}")
                continue
            svc.metrics.inc("strategy_signal_count", labels={"symbol": symbol})
            if intent.is_closing:
                if pair.status == "COMPLETE":
                    closed.append(symbol)
                    svc._post_close(symbol, pair)
                    if svc._state is not ServiceState.RUNNING:
                        svc._persist_runtime_state()
                        return {"state": svc.state.value, "reason": svc._recovery_reason}
                else:
                    svc._on_alert("EXIT_FAILED", f"{symbol}: {pair.status} {pair.error}")
                    svc.enter_recovery(f"平仓失败: {symbol} {pair.status} {pair.error}")
                    svc._persist_runtime_state()
                    return {"state": svc.state.value, "reason": svc._recovery_reason}
            elif pair.status == "COMPLETE":
                opened.append(symbol)
            else:
                skipped.append(f"{symbol}: {pair.status} {pair.error}")
                svc._on_alert("OPEN_FAILED", f"{symbol}: {pair.status} {pair.error}")

        svc._persist_runtime_state()
        return {
            "state": "RUNNING",
            "opened": opened,
            "skipped": skipped,
            "closed": closed,
            "intents": intent_ids,
        }


    def run_forever(self, *, tick_seconds: float = 5.0, stop_check: Callable[[], bool] | None = None) -> bool:
        """阻塞主循环（CLI 用）。stop_check 返回 True 时退出。

        长跑容错（§断点重连）：单轮 ``run_once()`` 异常只进 RECOVERY 并计数，
        不崩溃进程；连续异常达到 ``execution.max_consecutive_tick_errors``
        时优雅停机并返回 False（CLI 以退出码 1 结束，交给 systemd 重启）。
        任一轮成功则计数清零。

        RECOVERY 卡死升级（``execution.recovery_escalation_seconds``）：启动后
        （曾达 RUNNING）持续停留在 RECOVERY 超过阈值 = 自愈路径失效（如用户流
        untrusted 卡死短路了自动解除），优雅停机并以退出码 1 结束，systemd
        重启后重新预检+对账。启动阶段的长 RECOVERY（补账/epoch 构建）不计入。

        Returns:
            True = 正常停止（stop_check/外部信号）；False = 连续 tick 失败超限。
        """
        svc = self._service
        max_errors = svc.config.execution.max_consecutive_tick_errors
        escalation_ms = int(svc.config.execution.recovery_escalation_seconds * 1000)
        consecutive_errors = 0
        from ..live.watchdog import LoopWatchdog

        watchdog = LoopWatchdog(
            timeout_seconds=svc.config.execution.watchdog_timeout_seconds,
        )
        watchdog.start()
        try:
            while not (stop_check and stop_check()):
                # 心跳在轮首打点：单轮正常耗时（对账/候选刷新）不触发误杀
                watchdog.beat()
                try:
                    # 调用 service 的公开 tick（可被测试 monkeypatch；正常路径
                    # service.run_once 委托回 ServiceRunner.run_once 的 tick 体）
                    svc.run_once()
                except Exception as exc:  # noqa: BLE001 tick 级容错是长跑设计：瞬时故障（网络抖动/瞬时 DB 错误）不得崩溃进程；KeyboardInterrupt 等 BaseException 不被捕获
                    consecutive_errors += 1
                    logger.error("【主循环 tick 异常】连续 %d/%d: %s", consecutive_errors, max_errors, exc)
                    svc.enter_recovery(f"主循环 tick 异常: {exc}")
                    if consecutive_errors >= max_errors:
                        svc._on_alert("TICK_LOOP_FAILURE",
                                       f"连续 {consecutive_errors} 轮 tick 异常，优雅停机等待守护进程重启")
                        svc.stop()
                        return False
                else:
                    consecutive_errors = 0
                    # RECOVERY 卡死升级：tick 正常返回 RECOVERY 但自愈分支永远
                    # 跑不到（如 stale 分支每 tick 提前 return）时，进程会无限
                    # 等待。超过阈值就停机交 systemd 重启（重新预检+对账）。
                    elapsed = svc.recovery_elapsed_ms()
                    if elapsed is not None and elapsed >= escalation_ms:
                        svc._on_alert(
                            "RECOVERY_STUCK",
                            f"RECOVERY 持续 {elapsed / 1000:.0f}s 超过阈值 "
                            f"{escalation_ms / 1000:.0f}s，自愈失效，优雅停机等待守护进程重启。"
                            f"原因: {svc._recovery_reason}",
                        )
                        svc.stop()
                        return False
                time.sleep(tick_seconds)
        finally:
            watchdog.stop()
        return True
