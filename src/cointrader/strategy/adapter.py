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

__all__ = ["StrategyPort", "to_strategy_proposal"]


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
