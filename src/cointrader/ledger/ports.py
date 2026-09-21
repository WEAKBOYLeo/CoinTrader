"""LedgerPort：账本读模型协议面（T4）。

结构化协议：``execution.store.StateStore`` 天然满足（方法签名兼容）。
本模块不 import execution（保持 ledger 包对实现零依赖，实现由注入）。

``PipelineLedgerPort``（实施计划书 4.0 T1）：pipeline 持久化写侧协议
（proposal/intent/risk decision/execution plan 的 append + 幂等），
实现随 T2/T3 落在 ``execution.store``（必要时 additive schema v3）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Protocol


class LedgerPort(Protocol):
    """账本/读模型最小能力面（current projection + 历史 read model）。"""

    def current_positions(self, *, include_tombstones: bool = False) -> list[dict[str, Any]]: ...

    def open_pairs(self) -> list[dict[str, Any]]: ...

    def orders_in_states(self, states: tuple[str, ...] | list[str]) -> list[dict[str, Any]]: ...

    def signal_decisions(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        symbol: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def run_sessions(self, limit: int = 10) -> list[dict[str, Any]]: ...

    def latest_run_session(self) -> dict[str, Any] | None: ...

    def run_session(self, run_id: str) -> dict[str, Any] | None: ...

    def online_stats(self, now_ms: int, *, heartbeat_fresh_ms: int = 120_000) -> dict[str, Any]: ...

    def current_account(self) -> dict[str, Any] | None: ...

    def runtime_state(self) -> dict[str, dict[str, Any]]: ...

    def expected_positions(self) -> dict[str, dict[str, Decimal]]: ...

    def fills_for_orders(self, client_order_ids: Sequence[str]) -> list[dict[str, Any]]: ...

    def get_pair(self, pair_execution_id: str) -> dict[str, Any] | None: ...

    # -- T4.4：WebUI/CLI/reporting 查询面（全部只读；StateStore 天然满足） --

    def orders(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        symbol: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]: ...

    def fills(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        symbol: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]: ...

    def funding_cashflows(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        symbol: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]: ...

    def position_snapshots(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        symbol: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]: ...

    def account_snapshots(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]: ...

    def reconciliation_runs(
        self, *, since_ms: int | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]: ...

    def exchange_events(
        self,
        *,
        since_ms: int | None = None,
        market: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]: ...

    def lease_holders(self) -> list[dict[str, Any]]: ...

    def position_opened_ms(self, symbol: str) -> int | None: ...

    def scan_epochs(self, *, limit: int = 50) -> list[dict[str, Any]]: ...

    def sync_cursors(self, *, limit: int = 1000) -> list[dict[str, Any]]: ...

    def schema_version(self) -> int: ...

    def orders_for_pair(self, pair_execution_id: str) -> list[dict[str, Any]]: ...

    def pair_executions(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        symbol: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]: ...


class PipelineLedgerPort(Protocol):
    """pipeline 持久化写侧：append + 幂等（实现落 ``execution.store``）。

    每个 ``append_*`` 均为幂等写：

    - 新插入返回 ``True``；
    - 幂等命中（相同 intent_id/fingerprint/decision_id/plan_id）返回 ``False``；
    - 写失败（数据库错误）原样抛出，调用方按账本失败 fail closed 处理，
      不得吞掉后当成功。

    ``record`` 参数为领域契约的序列化输出（``to_dict``，金额已为字符串
    Decimal），实现不得依赖领域对象本身。
    """

    def append_strategy_proposal(self, proposal: Mapping[str, object]) -> bool:
        """策略结果落账（含拒绝/跳过）；幂等键 = proposal_id。"""
        ...

    def append_portfolio_intent(self, intent: Mapping[str, object]) -> bool:
        """组合意图落账（风险审批前）；幂等键 = intent_id/fingerprint。"""
        ...

    def append_risk_decision(self, decision: Mapping[str, object]) -> bool:
        """风险审批结果落账（含 CLOSE_ONLY/REJECT/HALT 理由）；幂等键 = decision_id。"""
        ...

    def append_execution_plan(self, plan: Mapping[str, object]) -> bool:
        """执行计划落账（下单前）；幂等键 = plan_id。"""
        ...

    def has_portfolio_intent_fingerprint(self, fingerprint: str) -> bool:
        """意图指纹是否已落账（重复评估去重，不产生第二个 intent）。"""
        ...
