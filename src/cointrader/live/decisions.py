"""实时策略决策领域对象（开发文档 §6.3）。

每个候选 symbol 每轮评估都必须产出一条可持久化的决策（含拒绝），
落盘到 ``signal_decisions`` 表。决策对象是纯数据，不含 IO。

决策类型（``decision_kind``）：

- ``OPEN``   允许开仓（``allowed=True``，service 据此生成 intent 并下单）
- ``HOLD``   继续持有（已持仓 symbol 的退出评估结果）
- ``EXIT``   策略退出（负资金费 / 最长持仓）
- ``REPLACE`` 换仓（新候选相对优势超过 premium）
- ``SKIP``   拒绝开仓（``allowed=False``，必须带 reason_code）
- ``PENDING_QUOTE`` 临时中间态：前置门槛全部通过但本轮上下文无新鲜报价。
  仅动态候选池模式使用；LiveService 按需获取报价后经
  ``LiveStrategy.complete_open`` 产出最终决策，本中间态不落账本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from ..execution.models import new_id

__all__ = [
    "DecisionKind",
    "ReasonCode",
    "StrategyDecision",
]


class DecisionKind:
    """决策类型常量。"""

    OPEN = "OPEN"
    HOLD = "HOLD"
    EXIT = "EXIT"
    REPLACE = "REPLACE"
    SKIP = "SKIP"
    PENDING_QUOTE = "PENDING_QUOTE"


class ReasonCode:
    """拒绝/退出原因码（开发文档 §6.3 固定集合）。"""

    # 拒绝开仓
    SERVICE_NOT_RUNNING = "SERVICE_NOT_RUNNING"
    STREAM_NOT_FRESH = "STREAM_NOT_FRESH"
    RECONCILIATION_BLOCKED = "RECONCILIATION_BLOCKED"
    ACCOUNT_STATE_UNKNOWN = "ACCOUNT_STATE_UNKNOWN"
    EXCLUDED_ASSET = "EXCLUDED_ASSET"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    STALE_DATA = "STALE_DATA"
    TRAILING_RATE_BELOW_THRESHOLD = "TRAILING_RATE_BELOW_THRESHOLD"
    CONSECUTIVE_POSITIVE_TOO_SHORT = "CONSECUTIVE_POSITIVE_TOO_SHORT"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    ALREADY_HELD = "ALREADY_HELD"
    ACTIVE_INTENT = "ACTIVE_INTENT"
    ACTIVE_ORDER = "ACTIVE_ORDER"
    MAX_POSITIONS = "MAX_POSITIONS"
    STALE_QUOTE = "STALE_QUOTE"
    BASIS_DISCOUNT = "BASIS_DISCOUNT"
    INVALID_SYMBOL = "INVALID_SYMBOL"
    NOTIONAL_TOO_SMALL = "NOTIONAL_TOO_SMALL"
    ALREADY_SUBMITTED = "ALREADY_SUBMITTED"
    # 中间态：前置门槛通过，等待 service 获取新鲜报价后由 complete_open 定案
    PENDING_QUOTE = "PENDING_QUOTE"
    # 退出
    NEGATIVE_EXIT_AVG = "NEGATIVE_EXIT_AVG"
    MAX_HOLDING = "MAX_HOLDING"
    REPLACEMENT = "REPLACEMENT"
    # 正常
    ENTRY_OK = "ENTRY_OK"
    HOLD_OK = "HOLD_OK"


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """一条可持久化的策略决策（§6.3 字段全集）。

    金额/数量/价格/费率一律 ``Decimal``；时间为 UTC 毫秒。
    """

    symbol: str
    run_id: str
    ts_ms: int
    decision_kind: str
    allowed: bool
    reason_code: str
    reason_text: str = ""
    strategy_version: str = "funding_carry-1.0"
    config_hash: str = ""
    funding_interval_hours: int | None = None
    trailing_annualized: Decimal | None = None
    exit_average_annualized: Decimal | None = None
    consecutive_positive_periods: int | None = None
    quote_volume_3d_avg: Decimal | None = None
    entry_threshold: Decimal | None = None
    exit_threshold: Decimal | None = None
    position_age_periods: int | None = None
    spot_price: Decimal | None = None
    perp_price: Decimal | None = None
    quote_ts_ms: int | None = None
    requested_notional: Decimal | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    decision_id: str = field(default_factory=lambda: new_id("dec"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "run_id": self.run_id,
            "ts_ms": self.ts_ms,
            "symbol": self.symbol,
            "decision_kind": self.decision_kind,
            "allowed": self.allowed,
            "reason_code": self.reason_code,
            "reason_text": self.reason_text,
            "strategy_version": self.strategy_version,
            "config_hash": self.config_hash,
            "funding_interval_hours": self.funding_interval_hours,
            "trailing_annualized": str(self.trailing_annualized) if self.trailing_annualized is not None else None,
            "exit_average_annualized": str(self.exit_average_annualized) if self.exit_average_annualized is not None else None,
            "consecutive_positive_periods": self.consecutive_positive_periods,
            "quote_volume_3d_avg": str(self.quote_volume_3d_avg) if self.quote_volume_3d_avg is not None else None,
            "entry_threshold": str(self.entry_threshold) if self.entry_threshold is not None else None,
            "exit_threshold": str(self.exit_threshold) if self.exit_threshold is not None else None,
            "position_age_periods": self.position_age_periods,
            "spot_price": str(self.spot_price) if self.spot_price is not None else None,
            "perp_price": str(self.perp_price) if self.perp_price is not None else None,
            "quote_ts_ms": self.quote_ts_ms,
            "requested_notional": str(self.requested_notional) if self.requested_notional is not None else None,
            "metrics": dict(self.metrics),
        }
