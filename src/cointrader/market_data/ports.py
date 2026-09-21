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
    """公开市场事实端口：发布 ``MarketSnapshot`` 与质量状态。"""

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
