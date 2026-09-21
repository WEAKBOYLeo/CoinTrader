"""策略包（strategy）：纯策略 evaluator 与兼容 adapter。

边界：不 import 网络库、Binance client、``execution`` 或 ``cointrader.live``
（``tests/test_architecture_boundaries.py`` 静态强制）。所有 IO 经
协议/注入外置；相同输入与同一时钟输出确定。
"""

from __future__ import annotations

from .adapter import StrategyPort, to_strategy_proposal
from .funding_carry import (
    CandidateInput,
    CarryEvaluation,
    DedupProbe,
    EvalContext,
    EvalKind,
    FundingCarryEvaluator,
    HeldInput,
    QuoteInput,
    Reason,
)

__all__ = [
    "CandidateInput",
    "CarryEvaluation",
    "DedupProbe",
    "EvalContext",
    "EvalKind",
    "FundingCarryEvaluator",
    "HeldInput",
    "QuoteInput",
    "Reason",
    "StrategyPort",
    "to_strategy_proposal",
]
