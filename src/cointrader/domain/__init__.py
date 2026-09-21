"""领域契约内核（domain）。

跨模块不可变契约：值对象、状态枚举、事件 envelope。

硬边界（由 ``tests/test_architecture_boundaries.py`` 静态强制）：

- 本包**无 IO**：不 import 网络、数据库、Binance、``execution`` 或 WebUI。
- 金额/数量/费率一律 ``Decimal``；对象 frozen dataclass；时间为 UTC 毫秒。
- 非法值与未知 schema version 显式抛领域异常，不猜值。
"""

from __future__ import annotations

from .account import AccountSnapshot, AssetBalance, PositionSnapshot
from .common import (
    DomainError,
    ExpiredDomainObject,
    InvalidDomainValue,
    RiskNotApproved,
    UnknownSchemaVersion,
)
from .control import CommandKind, ControlCommand, SafetyState, SafetyStateKind
from .events import (
    SUPPORTED_SCHEMA_VERSIONS,
    EventEnvelope,
    HealthEvent,
    HealthKind,
    OrderEvent,
    Payload,
)
from .execution import (
    PAIR_TERMINAL_STATES,
    ExecutionPlan,
    PairStatus,
    PlanOrder,
    PlanOrderType,
    PlanSide,
)
from .market import DataQuality, InstrumentQuote, MarketKind, MarketSnapshot
from .portfolio import CurrentPosition, IntentAction, PortfolioIntent, PortfolioView
from .risk import ApprovedIntent, RiskDecision, RiskDecisionKind, RuleEvidence
from .strategy import (
    PositionReason,
    StrategyAction,
    StrategyProposal,
    TargetPortfolio,
    TargetPosition,
)

__all__ = [
    "AccountSnapshot",
    "ApprovedIntent",
    "AssetBalance",
    "CommandKind",
    "ControlCommand",
    "CurrentPosition",
    "DataQuality",
    "DomainError",
    "EventEnvelope",
    "ExpiredDomainObject",
    "ExecutionPlan",
    "HealthEvent",
    "HealthKind",
    "InstrumentQuote",
    "InvalidDomainValue",
    "IntentAction",
    "MarketKind",
    "MarketSnapshot",
    "OrderEvent",
    "PAIR_TERMINAL_STATES",
    "PairStatus",
    "Payload",
    "PlanOrder",
    "PlanOrderType",
    "PlanSide",
    "PositionReason",
    "PositionSnapshot",
    "PortfolioIntent",
    "PortfolioView",
    "RiskDecision",
    "RiskDecisionKind",
    "RiskNotApproved",
    "RuleEvidence",
    "SafetyState",
    "SafetyStateKind",
    "SUPPORTED_SCHEMA_VERSIONS",
    "StrategyAction",
    "StrategyProposal",
    "TargetPortfolio",
    "TargetPosition",
    "UnknownSchemaVersion",
]
