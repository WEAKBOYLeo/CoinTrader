"""账户领域契约：账户/持仓快照与 capture 元数据。

硬约束（实施计划书 §5）：

- ``AccountSnapshot.complete=False`` 不更新 current projection，不可放行新风险
  （由 ledger/projector 层强制，这里只承载事实）。
- 交易所是当前账户事实源；``PositionSnapshot`` 允许零量 tombstone，
  候选池外的实际持仓也必须可恢复。
- 本模块无 IO、无网络、无执行层依赖。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from .common import (
    InvalidDomainValue,
    parse_bool,
    parse_decimal,
    parse_enum,
    parse_int_ms,
    parse_str,
)
from .market import MarketKind

__all__ = [
    "AccountSnapshot",
    "AssetBalance",
    "PositionSnapshot",
]


@dataclass(frozen=True)
class AssetBalance:
    """单一资产的余额事实（交易所返回的正数口径）。"""

    asset: str
    total: Decimal
    available: Decimal

    def __post_init__(self) -> None:
        if not self.asset:
            raise InvalidDomainValue("AssetBalance.asset 不能为空")
        if self.total < 0 or self.available < 0:
            raise InvalidDomainValue(f"余额不得为负: {self.asset}")

    def to_dict(self) -> dict[str, object]:
        return {
            "asset": self.asset,
            "total": str(self.total),
            "available": str(self.available),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AssetBalance:
        return cls(
            asset=parse_str(data.get("asset"), "AssetBalance.asset"),
            total=parse_decimal(data.get("total"), "AssetBalance.total"),
            available=parse_decimal(data.get("available"), "AssetBalance.available"),
        )


@dataclass(frozen=True)
class AccountSnapshot:
    """一次 REST capture 的账户事实。

    ``capture_start_ms``/``capture_end_ms`` 描述 capture 时间窗；
    ``complete=True`` 表示窗口内所有必需账户数据（权益、可用资金、各市场余额）
    全部成功取回。不完整 bundle 不得用于更新 current projection。
    """

    snapshot_id: str
    capture_start_ms: int
    capture_end_ms: int
    complete: bool
    equity: Decimal
    available: Decimal
    balances: tuple[AssetBalance, ...] = ()

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise InvalidDomainValue("AccountSnapshot.snapshot_id 不能为空")
        if self.capture_end_ms < self.capture_start_ms:
            raise InvalidDomainValue(
                "capture_end_ms 不得早于 capture_start_ms"
                f"（{self.capture_end_ms} < {self.capture_start_ms}）"
            )
        if self.equity < 0 or self.available < 0:
            raise InvalidDomainValue("权益/可用资金不得为负")

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "capture_start_ms": self.capture_start_ms,
            "capture_end_ms": self.capture_end_ms,
            "complete": self.complete,
            "equity": str(self.equity),
            "available": str(self.available),
            "balances": [b.to_dict() for b in self.balances],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AccountSnapshot:
        balances_raw = data.get("balances")
        if not isinstance(balances_raw, list):
            raise InvalidDomainValue("AccountSnapshot.balances 必须是列表")
        return cls(
            snapshot_id=parse_str(data.get("snapshot_id"), "AccountSnapshot.snapshot_id"),
            capture_start_ms=parse_int_ms(
                data.get("capture_start_ms"), "AccountSnapshot.capture_start_ms"
            ),
            capture_end_ms=parse_int_ms(
                data.get("capture_end_ms"), "AccountSnapshot.capture_end_ms"
            ),
            complete=parse_bool(data.get("complete"), "AccountSnapshot.complete"),
            equity=parse_decimal(data.get("equity"), "AccountSnapshot.equity"),
            available=parse_decimal(data.get("available"), "AccountSnapshot.available"),
            balances=tuple(AssetBalance.from_dict(item) for item in balances_raw),
        )


@dataclass(frozen=True)
class PositionSnapshot:
    """单一市场单标的持仓事实。

    交易所事实优先。``qty=0`` 是合法的零量 tombstone（用于表达
    "该仓位已确认关闭"），查询层不得把历史非零快照当作当前持仓。
    """

    symbol: str
    market: MarketKind
    qty: Decimal
    notional: Decimal
    updated_at_ms: int

    def __post_init__(self) -> None:
        if not self.symbol:
            raise InvalidDomainValue("PositionSnapshot.symbol 不能为空")
        if self.qty < 0 or self.notional < 0:
            raise InvalidDomainValue(f"持仓数量/名义额不得为负: {self.symbol}")

    @property
    def is_tombstone(self) -> bool:
        return self.qty == 0 and self.notional == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "market": self.market.value,
            "qty": str(self.qty),
            "notional": str(self.notional),
            "updated_at_ms": self.updated_at_ms,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PositionSnapshot:
        return cls(
            symbol=parse_str(data.get("symbol"), "PositionSnapshot.symbol"),
            market=parse_enum(MarketKind, data.get("market"), "PositionSnapshot.market"),
            qty=parse_decimal(data.get("qty"), "PositionSnapshot.qty"),
            notional=parse_decimal(data.get("notional"), "PositionSnapshot.notional"),
            updated_at_ms=parse_int_ms(
                data.get("updated_at_ms"), "PositionSnapshot.updated_at_ms"
            ),
        )
