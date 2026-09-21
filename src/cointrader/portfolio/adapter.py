"""legacy 信号/报价 → 领域组合契约的兼容适配器（实施计划书 3.0 T2）。

边界：

- 本模块**不 import** ``cointrader.live``（避免 live → portfolio → live 循环）；
  legacy ``Signal``/``Quote`` 以结构协议（Protocol）匹配。
- ``Signal/build_signal`` 在迁移期保留于 ``live/portfolio.py``；本适配器只负责
  把 legacy 产物转换为领域 ``PortfolioIntent``（策略只输出目标名义额，
  不直接生成订单参数）。
"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from ..domain.market import InstrumentQuote, MarketKind
from ..domain.portfolio import IntentAction, PortfolioIntent, intent_fingerprint

__all__ = [
    "LegacyQuote",
    "LegacySignal",
    "quote_to_instrument_quotes",
    "signal_to_intent",
]


class LegacySignal(Protocol):
    """legacy ``live.portfolio.Signal`` 的结构契约（只读成员）。"""

    @property
    def symbol(self) -> str: ...

    @property
    def target_notional(self) -> Decimal: ...

    @property
    def spot_price(self) -> Decimal: ...

    @property
    def perp_price(self) -> Decimal: ...

    @property
    def quote_ts_ms(self) -> int: ...

    @property
    def reason(self) -> str: ...

    @property
    def strategy_version(self) -> str: ...


class LegacyQuote(Protocol):
    """legacy ``live.strategy.Quote`` 的结构契约（只读成员）。"""

    @property
    def spot_price(self) -> Decimal: ...

    @property
    def perp_price(self) -> Decimal: ...

    @property
    def ts_ms(self) -> int: ...

    @property
    def spot_ts_ms(self) -> int: ...

    @property
    def perp_ts_ms(self) -> int: ...


def signal_to_intent(
    signal: LegacySignal,
    *,
    snapshot_id: str,
    decision_cutoff_ms: int,
    created_at_ms: int,
    correlation_id: str | None = None,
    causation_id: str | None = None,
) -> PortfolioIntent:
    """legacy 开仓信号 → 领域 OPEN 意图（两腿目标名义额 = signal.target_notional）。

    ``intent_id = fingerprint``：同一信号在同一快照/截止下重复转换不产生新意图。
    """
    action = IntentAction.OPEN
    intent_id = intent_fingerprint(
        action,
        signal.symbol,
        signal.target_notional,
        signal.target_notional,
        snapshot_id,
        decision_cutoff_ms,
    )
    return PortfolioIntent(
        intent_id=intent_id,
        action=action,
        symbol=signal.symbol,
        target_spot_notional=signal.target_notional,
        target_perp_notional=signal.target_notional,
        reason=signal.reason or "funding_carry",
        snapshot_id=snapshot_id,
        decision_cutoff_ms=decision_cutoff_ms,
        created_at_ms=created_at_ms,
        correlation_id=correlation_id,
        causation_id=causation_id,
    )


def quote_to_instrument_quotes(
    quote: LegacyQuote,
    symbol: str,
    *,
    generated_at_ms: int,
    decision_cutoff_ms: int,
) -> tuple[InstrumentQuote, InstrumentQuote]:
    """legacy 双市场报价 → 两条领域报价（Spot 多腿 / Futures 空腿）。

    两腿 funding 字段为 None（8h 费率与结算截止由 scan epoch 快照提供，
    不在此处伪造）。
    """
    spot_ts = quote.spot_ts_ms or quote.ts_ms
    perp_ts = quote.perp_ts_ms or quote.ts_ms
    if spot_ts > generated_at_ms or perp_ts > generated_at_ms:
        raise ValueError("报价时间晚于快照生成时间（无前瞻被破坏）")
    return (
        InstrumentQuote(
            symbol=symbol,
            market=MarketKind.SPOT,
            price=quote.spot_price,
            funding_rate_8h=None,
            funding_cutoff_ms=None,
            quote_time_ms=spot_ts,
        ),
        InstrumentQuote(
            symbol=symbol,
            market=MarketKind.FUTURES,
            price=quote.perp_price,
            funding_rate_8h=None,
            funding_cutoff_ms=None,
            quote_time_ms=perp_ts,
        ),
    )
