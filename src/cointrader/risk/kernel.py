"""风险内核：同步审批，带规则证据的 ``RiskDecision``（实施计划书 3.0 T3）。

硬约束（AC-04/05）：

- 纯计算：不调用 adapter、不访问网络、不写账本；全部输入注入。
  数值规则（限额/日亏/对冲/基差）通过注入的 ``RiskRulesPort`` 提供，
  本包只依赖 config + domain（execution 侧实现位于 ``risk.adapter``）。
- 新开风险必须同时满足：``SafetyState=RUNNING``、市场快照 ``FRESH``、
  账户快照完整、数值规则全部通过（证据逐条落在 ``RiskDecision.rules``）。
- 降级决定：数据/账户/安全状态异常 → ``CLOSE_ONLY``；安全状态 ``HALTED``
  → ``HALT``；名义额非法 → ``REJECT``。
- 限额超限时若更小名义额可行 → ``RESIZE``（批准额 = 可用额度），
  否则 ``CLOSE_ONLY``。风险只收紧不放宽。
- 平仓/减仓意图：除 ``STOPPED``（优雅停机，一切拒绝）外放行（reduce-only
  语义由 execution 层继续强制）。
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from typing import Protocol

from ..config import Config
from ..domain.account import AccountSnapshot
from ..domain.control import SafetyState, SafetyStateKind
from ..domain.market import DataQuality, MarketSnapshot
from ..domain.portfolio import IntentAction, PortfolioIntent
from ..domain.risk import RiskDecision, RiskDecisionKind, RuleEvidence

__all__ = ["RiskExposure", "RiskRulesPort", "RiskKernel"]


@dataclass(frozen=True, slots=True)
class RiskExposure:
    """内核所需的最小敞口事实（由 execution 侧 RiskState 投影而来）。

    金额边界一律 ``Decimal``（实施计划书 4.0 T1）：禁止 float 进入内核，
    也禁止 float→Decimal 往返。
    """

    total_exposure: Decimal
    symbol_exposure: dict[str, Decimal]


class RiskRulesPort(Protocol):
    """数值风控规则端口（限额/日亏/对冲/基差）。

    实现见 ``risk.adapter.RiskRulesAdapter``（包裹
    ``execution.risk.RiskManager``）。返回 ``(规则名, 是否通过, 原因)``。
    金额边界一律 ``Decimal``。
    """

    def preflight_all(self, state: object) -> tuple[tuple[str, bool, str], ...]: ...

    def check_order(self, symbol: str, notional: Decimal, state: object) -> tuple[bool, str]: ...

    def exposure(self, state: object) -> RiskExposure: ...


def _decision_id(intent: PortfolioIntent, now_ms: int) -> str:
    """确定性审批 id：意图指纹 + 决定时点（同输入同刻重复审批 id 相同）。"""
    payload = f"{intent.fingerprint()}|{now_ms}"
    return "rd-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class RiskKernel:
    """同步风险审批内核。

    Args:
        config: 顶层配置（限额唯一来源 = risk 段）。
        rules: 数值风控规则端口（execution.risk 的适配实现）。
        now_fn: 可注入时钟（秒）；决定时点以 ``approve(now_ms=...)`` 为准。
        validity_ms: 审批有效期（毫秒）。
    """

    def __init__(
        self,
        config: Config,
        *,
        rules: RiskRulesPort,
        now_fn: Callable[[], float] = time.time,  # noqa: ARG002
        validity_ms: int = 30_000,
    ) -> None:
        self.config = config
        self._rules = rules
        self._validity_ms = validity_ms

    def approve(
        self,
        intent: PortfolioIntent,
        *,
        now_ms: int,
        safety: SafetyState,
        market: MarketSnapshot | None = None,
        account: AccountSnapshot | None = None,
        risk_state: object | None = None,
    ) -> RiskDecision:
        """审批单个组合意图。返回带规则证据的 ``RiskDecision``。

        ``risk_state`` 为 execution 侧 ``RiskState``（不直接依赖该类型）。
        """
        closing = intent.action is IntentAction.CLOSE
        requested_spot = intent.target_spot_notional
        requested_perp = intent.target_perp_notional
        requested_total = requested_spot + requested_perp

        rules: list[RuleEvidence] = []
        decision_kind = RiskDecisionKind.ALLOW
        approved_spot, approved_perp = requested_spot, requested_perp
        reason = ""

        # 1) 名义额合法性（平仓 = 两腿 0）
        if not closing and requested_total <= 0:
            rules.append(
                RuleEvidence(
                    "notional_valid", False,
                    f"新开风险名义额必须为正（当前 {format(requested_total, 'f')}）",
                )
            )
            return self._decision(
                intent, now_ms, RiskDecisionKind.REJECT, requested_spot, requested_perp,
                Decimal("0"), Decimal("0"), rules, "名义额非法",
            )
        if requested_total < 0 or requested_spot < 0 or requested_perp < 0:
            rules.append(RuleEvidence("notional_valid", False, "名义额不得为负"))
            return self._decision(
                intent, now_ms, RiskDecisionKind.REJECT, requested_spot, requested_perp,
                Decimal("0"), Decimal("0"), rules, "名义额非法",
            )
        rules.append(RuleEvidence("notional_valid", True, f"名义额 {format(requested_total, 'f')}"))

        # 2) 安全状态
        if closing:
            if safety.state is SafetyStateKind.STOPPED:
                rules.append(
                    RuleEvidence("safety_state", False, f"STOPPED（{safety.reason}）：优雅停机，一切拒绝")
                )
                return self._decision(
                    intent, now_ms, RiskDecisionKind.REJECT,
                    requested_spot, requested_perp, Decimal("0"), Decimal("0"),
                    rules, "优雅停机，拒绝平仓请求",
                )
            rules.append(RuleEvidence("safety_state", True, f"{safety.state.value}: 减仓放行"))
            return self._decision(
                intent, now_ms, RiskDecisionKind.ALLOW,
                requested_spot, requested_perp, Decimal("0"), Decimal("0"),
                rules, "平仓意图，reduce-only 放行（execution 层继续强制）",
            )
        if not safety.allows_new_risk:
            kind = (
                RiskDecisionKind.HALT
                if safety.state is SafetyStateKind.HALTED
                else RiskDecisionKind.CLOSE_ONLY
            )
            rules.append(
                RuleEvidence("safety_state", False, f"{safety.state.value}（{safety.reason}）：禁止新增风险")
            )
            return self._decision(
                intent, now_ms, kind, requested_spot, requested_perp,
                Decimal("0"), Decimal("0"), rules,
                f"安全状态 {safety.state.value}，禁止新增风险",
            )
        rules.append(RuleEvidence("safety_state", True, "RUNNING"))

        # 3) 市场快照新鲜度
        if market is None:
            rules.append(RuleEvidence("market_fresh", False, "市场快照缺失"))
            return self._decision(
                intent, now_ms, RiskDecisionKind.CLOSE_ONLY,
                requested_spot, requested_perp, Decimal("0"), Decimal("0"),
                rules, "市场快照缺失，禁止新增风险",
            )
        if market.quality is not DataQuality.FRESH:
            rules.append(
                RuleEvidence("market_fresh", False, f"市场快照质量 {market.quality.value}")
            )
            return self._decision(
                intent, now_ms, RiskDecisionKind.CLOSE_ONLY,
                requested_spot, requested_perp, Decimal("0"), Decimal("0"),
                rules, f"市场快照非 FRESH（{market.quality.value}）",
            )
        if market.generated_at_ms > now_ms:
            rules.append(
                RuleEvidence("market_fresh", False, "市场快照时间晚于当前（无前瞻被破坏）")
            )
            return self._decision(
                intent, now_ms, RiskDecisionKind.REJECT,
                requested_spot, requested_perp, Decimal("0"), Decimal("0"),
                rules, "市场快照时间异常",
            )
        rules.append(RuleEvidence("market_fresh", True, f"snapshot={market.snapshot_id}"))

        # 4) 账户完整性
        if account is None or not account.complete:
            detail = "账户快照缺失" if account is None else "账户快照不完整（complete=False）"
            rules.append(RuleEvidence("account_complete", False, detail))
            return self._decision(
                intent, now_ms, RiskDecisionKind.CLOSE_ONLY,
                requested_spot, requested_perp, Decimal("0"), Decimal("0"),
                rules, detail + "，禁止新增风险",
            )
        if account.capture_end_ms > now_ms:
            rules.append(RuleEvidence("account_complete", False, "账户 capture 结束时间晚于当前"))
            return self._decision(
                intent, now_ms, RiskDecisionKind.REJECT,
                requested_spot, requested_perp, Decimal("0"), Decimal("0"),
                rules, "账户快照时间异常",
            )
        rules.append(RuleEvidence("account_complete", True, f"snapshot={account.snapshot_id}"))
        snap_ids = (intent.snapshot_id, market.snapshot_id, account.snapshot_id)

        # 5) 数值风控规则（经 RiskRulesPort；限额/日亏/对冲/基差）
        if risk_state is None:
            rules.append(
                RuleEvidence("risk_limits", False, "RiskState 缺失：无账户事实时禁止新增风险")
            )
            return self._decision(
                intent, now_ms, RiskDecisionKind.CLOSE_ONLY,
                requested_spot, requested_perp, Decimal("0"), Decimal("0"),
                rules, "RiskState 缺失，禁止新增风险",
            )

        preflight = self._rules.preflight_all(risk_state)
        for rule_name, passed, why in preflight:
            rules.append(RuleEvidence(rule_name, passed, why))
        if not all(passed for _, passed, _ in preflight):
            failed = next(why for _, passed, why in preflight if not passed)
            return self._decision(
                intent, now_ms, RiskDecisionKind.REJECT,
                requested_spot, requested_perp, Decimal("0"), Decimal("0"),
                rules, failed, snapshot_ids=snap_ids,
            )

        ok, why = self._rules.check_order(intent.symbol, requested_total, risk_state)
        if ok:
            rules.append(RuleEvidence("risk_limits", True, why))
        else:
            # 超限：尝试 RESIZE（批准额 = 可用额度；ROUND_FLOOR 只收紧方向，
            # 杜绝舍入后批准额超过请求）
            affordable = self._affordable(intent.symbol, risk_state)
            if affordable > 0:
                scale = affordable / requested_total
                new_spot = (requested_spot * scale).quantize(
                    Decimal("0.01"), rounding=ROUND_FLOOR
                )
                new_perp = (affordable - new_spot) if requested_perp > 0 else Decimal("0")
                if new_spot < 0 or new_perp < 0:
                    new_spot, new_perp = Decimal("0"), Decimal("0")
                rules.append(
                    RuleEvidence(
                        "risk_limits", True,
                        f"{why} → RESIZE {format(affordable, 'f')}",
                    )
                )
                return self._decision(
                    intent, now_ms, RiskDecisionKind.RESIZE,
                    requested_spot, requested_perp, new_spot, new_perp,
                    rules, f"限额超限，收紧至 {format(affordable, 'f')}",
                    snapshot_ids=snap_ids,
                )
            rules.append(RuleEvidence("risk_limits", False, why))
            return self._decision(
                intent, now_ms, RiskDecisionKind.CLOSE_ONLY,
                requested_spot, requested_perp, Decimal("0"), Decimal("0"),
                rules, why, snapshot_ids=snap_ids,
            )

        reason = reason or "全部规则通过"
        return self._decision(
            intent, now_ms, decision_kind, requested_spot, requested_perp,
            approved_spot, approved_perp, rules, reason, snapshot_ids=snap_ids,
        )

    def _affordable(self, symbol: str, risk_state: object) -> Decimal:
        """单币种/总敞口/单笔上限内可放行的最大名义额（全 Decimal）。"""
        cfg = self.config.risk
        exp = self._rules.exposure(risk_state)
        return min(
            Decimal(str(cfg.max_notional_per_order)),
            max(
                Decimal(str(cfg.max_exposure_per_symbol))
                - exp.symbol_exposure.get(symbol, Decimal("0")),
                Decimal("0"),
            ),
            max(
                Decimal(str(cfg.max_total_exposure)) - exp.total_exposure,
                Decimal("0"),
            ),
        )

    def _decision(
        self,
        intent: PortfolioIntent,
        now_ms: int,
        kind: RiskDecisionKind,
        requested_spot: Decimal,
        requested_perp: Decimal,
        approved_spot: Decimal,
        approved_perp: Decimal,
        rules: list[RuleEvidence],
        reason: str,
        snapshot_ids: tuple[str, ...] = (),
    ) -> RiskDecision:
        return RiskDecision(
            decision_id=_decision_id(intent, now_ms),
            decision=kind,
            symbol=intent.symbol,
            requested_spot_notional=requested_spot,
            requested_perp_notional=requested_perp,
            approved_spot_notional=approved_spot,
            approved_perp_notional=approved_perp,
            snapshot_ids=tuple(snapshot_ids) if snapshot_ids else (intent.snapshot_id,),
            rules=tuple(rules),
            decided_at_ms=now_ms,
            valid_until_ms=now_ms + self._validity_ms,
            reason=reason,
        )
