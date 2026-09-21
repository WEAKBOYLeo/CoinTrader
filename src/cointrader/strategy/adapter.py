"""legacy 策略兼容 adapter：评估结果 → 领域 StrategyProposal（实施计划书 3.0 T2）。

- ``StrategyPort``：目标策略端口（实施计划书 §5 外部契约 4 的结构协议）。
- ``to_strategy_proposal``：把 ``FundingCarryEvaluator`` 的中性评估结果汇总为
  领域 ``StrategyProposal``（目标名义额 + 证据），供 T3/T4 的账本落盘、
  风险审批与可观测性消费。legacy ``LiveStrategy`` 决策路径（StrategyDecision）
  保持不变，本 adapter 是新增的领域出口，不替换旧路径。
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Protocol

from ..domain.portfolio import PortfolioView
from ..domain.strategy import (
    PositionReason,
    StrategyAction,
    StrategyProposal,
    TargetPortfolio,
    TargetPosition,
)
from .funding_carry import CarryEvaluation, EvalKind

__all__ = ["LegacyStrategyDecision", "StrategyPort", "decisions_to_proposal", "to_strategy_proposal"]


class StrategyPort(Protocol):
    """策略端口：评估结果 + 当前组合 → 可序列化 StrategyProposal（纯计算）。"""

    def evaluate(
        self,
        evaluations: Sequence[CarryEvaluation],
        view: PortfolioView,
        *,
        proposal_id: str,
        snapshot_id: str,
        decision_cutoff_ms: int,
        strategy_version: str,
        config_hash: str,
        valid_until_ms: int,
    ) -> StrategyProposal:
        """相同输入输出相同结果；不含订单/网络/账本能力。"""
        ...


def _action_of(evaluation: CarryEvaluation) -> StrategyAction:
    if evaluation.kind == EvalKind.OPEN:
        return StrategyAction.OPEN
    if evaluation.kind in (EvalKind.EXIT, EvalKind.REPLACE):
        return StrategyAction.EXIT if evaluation.kind == EvalKind.EXIT else StrategyAction.REPLACE
    if evaluation.kind == EvalKind.HOLD:
        return StrategyAction.HOLD
    return StrategyAction.SKIP


def to_strategy_proposal(
    evaluations: Sequence[CarryEvaluation],
    view: PortfolioView,
    *,
    proposal_id: str,
    snapshot_id: str,
    decision_cutoff_ms: int,
    strategy_version: str,
    config_hash: str,
    valid_until_ms: int,
) -> StrategyProposal:
    """目标组合 = 当前组合 − 退出/换仓 symbol + 新开 symbol（目标名义额）。

    证据：每个 symbol 一条 ``PositionReason``（action + reason_code/text）。
    """
    targets: dict[str, tuple[Decimal, Decimal]] = {
        e.symbol: (e.spot_notional, e.perp_notional) for e in view.entries
    }
    reasons: list[PositionReason] = []
    for evaluation in sorted(evaluations, key=lambda e: e.symbol):
        action = _action_of(evaluation)
        if action in (StrategyAction.EXIT, StrategyAction.REPLACE):
            targets.pop(evaluation.symbol, None)
        elif action is StrategyAction.OPEN and (
            evaluation.requested_notional is not None and evaluation.requested_notional > 0
        ):
            notional = evaluation.requested_notional
            targets[evaluation.symbol] = (notional, notional)
        evidence = (evaluation.reason_code,)
        reasons.append(
            PositionReason(
                symbol=evaluation.symbol,
                action=action,
                reason=evaluation.reason_text,
                evidence=evidence,
            )
        )

    target_portfolio = TargetPortfolio(
        entries=tuple(
            TargetPosition(symbol=symbol, spot_notional=spot, perp_notional=perp)
            for symbol, (spot, perp) in sorted(targets.items())
            if spot + perp > 0
        )
    )
    return StrategyProposal(
        proposal_id=proposal_id,
        snapshot_id=snapshot_id,
        decision_cutoff_ms=decision_cutoff_ms,
        strategy_version=strategy_version,
        config_hash=config_hash,
        valid_until_ms=valid_until_ms,
        target=target_portfolio,
        reasons=tuple(reasons),
    )


class LegacyStrategyDecision(Protocol):
    """legacy ``live.decisions.StrategyDecision`` 的结构契约（只读）。

    本包不 import ``cointrader.live``（依赖方向由 application 层组装），
    与 ``portfolio.adapter.LegacySignal`` 同风格。
    """

    @property
    def symbol(self) -> str: ...

    @property
    def decision_kind(self) -> str: ...  # OPEN/HOLD/EXIT/REPLACE/SKIP/PENDING_QUOTE

    @property
    def allowed(self) -> bool: ...

    @property
    def requested_notional(self) -> Decimal | None: ...

    @property
    def reason_code(self) -> str: ...

    @property
    def reason_text(self) -> str: ...


def decisions_to_proposal(
    decisions: Sequence[LegacyStrategyDecision],
    view: PortfolioView,
    *,
    proposal_id: str,
    snapshot_id: str,
    decision_cutoff_ms: int,
    strategy_version: str,
    config_hash: str,
    valid_until_ms: int,
) -> StrategyProposal:
    """legacy 策略决策序列 → 领域 ``StrategyProposal``（实施计划书 4.0 T2）。

    数值语义与 legacy signal 路径一致：

    - 目标组合 = 当前组合 − EXIT/REPLACE symbol + OPEN（allowed）symbol，
      两腿目标名义额 = ``requested_notional``（与 ``Signal.to_intent`` 一致：
      spot = perp = 请求名义额，拆半由执行层负责）；
    - 拒绝/跳过（SKIP/未放行 OPEN）不改变目标组合，但证据逐条落
      ``reasons``（拒绝也留痕）；
    - 纯计算：不含订单/网络/账本能力。
    """
    targets: dict[str, tuple[Decimal, Decimal]] = {
        e.symbol: (e.spot_notional, e.perp_notional) for e in view.entries
    }
    reasons: list[PositionReason] = []
    for decision in sorted(decisions, key=lambda d: d.symbol):
        kind = decision.decision_kind
        if kind in ("EXIT", "REPLACE"):
            action = StrategyAction.EXIT if kind == "EXIT" else StrategyAction.REPLACE
            targets.pop(decision.symbol, None)
        elif kind == "OPEN" and decision.allowed:
            action = StrategyAction.OPEN
            notional = decision.requested_notional
            if notional is not None and notional > 0:
                targets[decision.symbol] = (notional, notional)
        elif kind == "HOLD":
            action = StrategyAction.HOLD
        else:  # SKIP / 未放行 OPEN / 中间态
            action = StrategyAction.SKIP
        reasons.append(
            PositionReason(
                symbol=decision.symbol,
                action=action,
                reason=decision.reason_text or decision.reason_code,
                evidence=(decision.reason_code,),
            )
        )
    target_portfolio = TargetPortfolio(
        entries=tuple(
            TargetPosition(symbol=s, spot_notional=sp, perp_notional=pe)
            for s, (sp, pe) in sorted(targets.items())
            if sp + pe > 0
        )
    )
    return StrategyProposal(
        proposal_id=proposal_id,
        snapshot_id=snapshot_id,
        decision_cutoff_ms=decision_cutoff_ms,
        strategy_version=strategy_version,
        config_hash=config_hash,
        valid_until_ms=valid_until_ms,
        target=target_portfolio,
        reasons=tuple(reasons),
    )
