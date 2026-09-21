"""组合规划包（portfolio）：目标组合边界。

纯计算：无网络、无执行依赖。暴露 ``PortfolioPlanner``（目标差异/排序/去重/
换仓规划）与 legacy 信号适配器。
"""

from __future__ import annotations

from .adapter import LegacyQuote, LegacySignal, quote_to_instrument_quotes, signal_to_intent
from .planner import PortfolioPlanner, PortfolioPlannerPort

__all__ = [
    "LegacyQuote",
    "LegacySignal",
    "PortfolioPlanner",
    "PortfolioPlannerPort",
    "quote_to_instrument_quotes",
    "signal_to_intent",
]
