"""应用层生产 pipeline 组装（实施计划书 4.0 T2，AC-03/04）。

``Pipeline`` 是领域生产流水线中 **策略侧** 的唯一装配点：

    MarketSnapshot → StrategyProposal → PortfolioIntent        （T2，本模块）
    → RiskDecision → ApprovedIntent → ExecutionPlan → submit   （T3 接通）

边界（AC-01/04）：

- 本模块**不含任何 broker/executor 引用**，不解析 Binance JSON，
  不做重试决策，不改变安全状态 —— 只做「落账 + diff + 幂等」编排。
- 策略结果（含拒绝/跳过）与每个 intent 都**先写账本**（幂等键 =
  记录 id/fingerprint）再进入后续审批阶段。
- 相同输入产生相同 intent 序列；重复 fingerprint 不产生第二个 intent
  （``append_portfolio_intent`` 幂等命中 → 该 intent 不再输出）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.portfolio import PortfolioIntent, PortfolioView
from ..domain.strategy import StrategyProposal
from ..ledger.ports import PipelineLedgerPort
from ..portfolio.planner import PortfolioPlannerPort

__all__ = ["Pipeline"]


@dataclass(frozen=True)
class Pipeline:
    """T2 流水线：proposal 落账 → planner diff → intent 逐条幂等落账。

    Args:
        planner: 组合规划端口（生产 = ``PortfolioPlanner``，纯计算）。
        ledger: pipeline 持久化端口（生产 = ``StateStore``，实现见
            ``execution/store.py`` 的 ``PipelineLedgerPort`` 协议方法）。
    """

    planner: PortfolioPlannerPort
    ledger: PipelineLedgerPort

    def run_round(
        self,
        *,
        proposal: StrategyProposal,
        view: PortfolioView,
        now_ms: int,
        run_id: str = "",
    ) -> tuple[StrategyProposal, tuple[PortfolioIntent, ...]]:
        """执行一轮策略侧流水线。

        1. 策略结果（含拒绝/跳过证据）先写账本（幂等键 = proposal_id）；
        2. ``planner.diff(target, view)`` → 稳定排序、指纹幂等的 intent 序列
           （CLOSE 先于 REPLACE/OPEN，再 RESIZE）；
        3. 每个 intent 先写账本（幂等键 = intent_id == fingerprint）；
           幂等命中（重复 fingerprint）的 intent 不输出，不产生第二份意图。

        Returns:
            ``(proposal, 本轮新落账的 intents)``。

        Raises:
            StoreError 等账本写失败原样上抛 —— 调用方必须 fail closed，
            不得吞掉后继续审批/下单。
        """
        proposal_payload = dict(proposal.to_dict())
        proposal_payload["run_id"] = run_id
        self.ledger.append_strategy_proposal(proposal_payload)

        intents = self.planner.diff(
            proposal.target, view, created_at_ms=now_ms, correlation_id=proposal.proposal_id
        )
        new_intents: list[PortfolioIntent] = []
        for intent in intents:
            payload = dict(intent.to_dict())
            payload["run_id"] = run_id
            if self.ledger.append_portfolio_intent(payload):
                new_intents.append(intent)
            # else：幂等命中（重复 fingerprint）→ 不产生第二个 intent
        return proposal, tuple(new_intents)
