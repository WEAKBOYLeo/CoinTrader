"""市场领域契约：快照、质量与新鲜度。

硬约束（实施计划书 §4/§5）：

- Spot/Perp 时间偏差、symbol、价格、funding cutoff 必须合法；
  非法快照拒绝新开仓（由 market data 服务降级为 ``STALE``/``INCOMPLETE``）。
- 无前瞻：``quote_time_ms`` 不得晚于 ``generated_at_ms``；
  funding 结算时点不得晚于决策截止 ``decision_cutoff_ms``。
- 本模块无 IO、无网络、无执行层依赖。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from .common import (
    InvalidDomainValue,
    parse_decimal,
    parse_enum,
    parse_int_ms,
    parse_optional,
    parse_str,
)

__all__ = [
    "DataQuality",
    "InstrumentQuote",
    "MarketKind",
    "MarketSnapshot",
]


class MarketKind(str, Enum):
    """市场腿：Spot 多头腿 / Futures（USDⓈ-M 永续）空头腿。"""

    SPOT = "SPOT"
    FUTURES = "FUTURES"


class DataQuality(str, Enum):
    """市场快照质量。``FRESH`` 之外一律禁止用于新增风险。"""

    FRESH = "FRESH"
    STALE = "STALE"
    INCOMPLETE = "INCOMPLETE"
    INVALID = "INVALID"
    DEGRADED = "DEGRADED"


@dataclass(frozen=True)
class InstrumentQuote:
    """单市场单标的报价。

    - ``price`` / ``funding_rate_8h`` / ``volume_3d_avg`` 一律 ``Decimal``。
    - ``funding_rate_8h`` 为 8h 结算桶口径（与回测 ``normalize_funding_to_8h`` 一致）。
    - ``funding_cutoff_ms`` 是该报价所用 funding 数据的结算截止（无前瞻边界）。
    """

    symbol: str
    market: MarketKind
    price: Decimal
    funding_rate_8h: Decimal | None
    funding_cutoff_ms: int | None
    quote_time_ms: int
    volume_3d_avg: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise InvalidDomainValue("InstrumentQuote.symbol 不能为空")
        if self.price <= 0:
            raise InvalidDomainValue(f"价格必须为正: {self.symbol} {self.price}")
        if self.funding_rate_8h is not None and self.funding_cutoff_ms is None:
            raise InvalidDomainValue(
                f"有 funding 费率必须有结算截止时点: {self.symbol}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "market": self.market.value,
            "price": str(self.price),
            "funding_rate_8h": None if self.funding_rate_8h is None else str(self.funding_rate_8h),
            "funding_cutoff_ms": self.funding_cutoff_ms,
            "quote_time_ms": self.quote_time_ms,
            "volume_3d_avg": None
            if self.volume_3d_avg is None
            else str(self.volume_3d_avg),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> InstrumentQuote:
        return cls(
            symbol=parse_str(data.get("symbol"), "InstrumentQuote.symbol"),
            market=parse_enum(MarketKind, data.get("market"), "InstrumentQuote.market"),
            price=parse_decimal(data.get("price"), "InstrumentQuote.price"),
            funding_rate_8h=parse_optional(
                data.get("funding_rate_8h"),
                "InstrumentQuote.funding_rate_8h",
                parse_decimal,
            ),
            funding_cutoff_ms=parse_optional(
                data.get("funding_cutoff_ms"),
                "InstrumentQuote.funding_cutoff_ms",
                parse_int_ms,
            ),
            quote_time_ms=parse_int_ms(
                data.get("quote_time_ms"), "InstrumentQuote.quote_time_ms"
            ),
            volume_3d_avg=parse_optional(
                data.get("volume_3d_avg"), "InstrumentQuote.volume_3d_avg", parse_decimal
            ),
        )


@dataclass(frozen=True)
class MarketSnapshot:
    """一次 scan epoch 发布的不可变市场事实。

    - 下一次 snapshot 整体替换，不允许逐字段合并。
    - ``quality != FRESH`` 的快照只可用于降低风险（由上层保证），不可用于新开仓。
    """

    snapshot_id: str
    generated_at_ms: int
    decision_cutoff_ms: int
    quality: DataQuality
    quotes: tuple[InstrumentQuote, ...] = ()

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise InvalidDomainValue("MarketSnapshot.snapshot_id 不能为空")
        if self.decision_cutoff_ms > self.generated_at_ms:
            raise InvalidDomainValue(
                "decision_cutoff_ms 不得晚于 generated_at_ms"
                f"（{self.decision_cutoff_ms} > {self.generated_at_ms}）"
            )
        for quote in self.quotes:
            if quote.quote_time_ms > self.generated_at_ms:
                raise InvalidDomainValue(
                    f"报价时间晚于快照生成时间（无前瞻被破坏）: {quote.symbol} "
                    f"{quote.quote_time_ms} > {self.generated_at_ms}"
                )
            if (
                quote.funding_cutoff_ms is not None
                and quote.funding_cutoff_ms > self.decision_cutoff_ms
            ):
                raise InvalidDomainValue(
                    f"funding 结算时点晚于决策截止（无前瞻被破坏）: {quote.symbol} "
                    f"{quote.funding_cutoff_ms} > {self.decision_cutoff_ms}"
                )

    def quote(self, symbol: str, market: MarketKind) -> InstrumentQuote | None:
        for item in self.quotes:
            if item.symbol == symbol and item.market == market:
                return item
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "generated_at_ms": self.generated_at_ms,
            "decision_cutoff_ms": self.decision_cutoff_ms,
            "quality": self.quality.value,
            "quotes": [q.to_dict() for q in self.quotes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MarketSnapshot:
        quotes_raw = data.get("quotes")
        if not isinstance(quotes_raw, list):
            raise InvalidDomainValue("MarketSnapshot.quotes 必须是列表")
        return cls(
            snapshot_id=parse_str(data.get("snapshot_id"), "MarketSnapshot.snapshot_id"),
            generated_at_ms=parse_int_ms(
                data.get("generated_at_ms"), "MarketSnapshot.generated_at_ms"
            ),
            decision_cutoff_ms=parse_int_ms(
                data.get("decision_cutoff_ms"), "MarketSnapshot.decision_cutoff_ms"
            ),
            quality=parse_enum(DataQuality, data.get("quality"), "MarketSnapshot.quality"),
            quotes=tuple(InstrumentQuote.from_dict(item) for item in quotes_raw),
        )
