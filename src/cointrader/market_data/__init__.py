"""市场数据包（market_data）：公开市场事实边界。

不发起网络副作用；生产数据源（``MarketDataSynchronizer`` + 限流协调器）
经 ``ScanEpochSource`` 结构协议注入，由 application 层组装。
"""

from __future__ import annotations

from .ports import CandidateSnapshotView, MarketDataPort, ScanEpochSource, ScanEpochView
from .service import MarketDataService

__all__ = [
    "CandidateSnapshotView",
    "MarketDataService",
    "MarketDataPort",
    "ScanEpochSource",
    "ScanEpochView",
]
