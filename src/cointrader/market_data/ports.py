"""市场数据端口（实施计划书 3.0 T2）。

- ``MarketDataPort``：目标市场事实端口（§5 外部契约 1）。只读；
  失败返回 ``STALE``/``INCOMPLETE`` 快照或领域数据错误，不伪造默认价格。
- ``ScanEpochSource``：结构协议，``live.market_sync.MarketDataSynchronizer``
  天然满足；本包不 import ``cointrader.live``（依赖方向由 application 层组装）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Protocol

from ..domain.market import MarketSnapshot

__all__ = [
    "CandidateSnapshotView",
    "MarketDataPort",
    "ScanEpochView",
    "ScanEpochSource",
]


class MarketDataPort(Protocol):
    """公开市场事实端口：发布 ``MarketSnapshot`` 与质量状态。

    契约（实施计划书 4.0 T1 统一）：

    - ``snapshot`` 是**唯一**事实入口；quality/cutoff/freshness 不另设独立
      端口方法，全部由 ``MarketSnapshot`` 字段承载：
      质量 = ``quality``（FRESH 才可新增风险）；
      cutoff = ``decision_cutoff_ms``（不得越过决定时点，无前瞻）；
      新鲜度 = 调用方以 ``generated_at_ms``/报价 ``quote_time_ms`` 对当前
      时钟比较得出，不得由策略自行推断。
    - 失败/无数据不得伪造默认价格：返回带明确非 FRESH 质量的快照
      （空 quotes = INCOMPLETE）或领域数据错误。
    """

    def snapshot(self, symbols: Sequence[str] | None = None) -> MarketSnapshot:
        """当前市场事实横截面。``symbols`` 给定则只含这些 symbol。"""
        ...


class CandidateSnapshotView(Protocol):
    """单候选 epoch 快照的结构视图（匹配 ``live.market_sync.CandidateSnapshot``，只读）。"""

    @property
    def symbol(self) -> str: ...

    @property
    def rates(self) -> tuple[Decimal, ...]: ...

    @property
    def mark_prices(self) -> tuple[Decimal, ...]: ...

    @property
    def timestamps(self) -> tuple[int, ...]: ...

    @property
    def interval_hours(self) -> int: ...

    @property
    def quote_volume_3d_avg(self) -> Decimal: ...

    @property
    def fetched_ms(self) -> int: ...

    @property
    def error(self) -> str: ...


class ScanEpochView(Protocol):
    """scan epoch 的结构视图（匹配 ``live.market_sync.ScanEpoch``，只读）。"""

    @property
    def epoch_id(self) -> str: ...

    @property
    def decision_cutoff_ms(self) -> int: ...

    @property
    def status(self) -> object: ...  # ScanEpochStatus；READY 才可作为 FRESH 事实源

    @property
    def excluded(self) -> Mapping[str, str]: ...


class ScanEpochSource(Protocol):
    """epoch 源：``MarketDataSynchronizer`` 的结构契约。"""

    def latest_ready(self) -> ScanEpochView | None:
        ...

    def latest(self) -> ScanEpochView | None:
        ...

    def snapshots_for(self, epoch_id: str) -> Mapping[str, CandidateSnapshotView]:
        ...

    def expected_symbols(self) -> tuple[str, ...]:
        ...
