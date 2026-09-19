"""实盘上层：信号到交易意图的转换（开发设计文档 §4.1 Portfolio Coordinator）。

边界：

1. 研究层（``research/``）产出候选币种与资金费率，本层只负责把候选
   转换成带价格保护的**交易意图**（Signal），不做任何下单决策细节。
2. 意图必须携带新鲜报价（spot_price / perp_price / quote_ts_ms）——
   执行层（``execution/``）会用 ``max_market_data_age_seconds`` 拒绝过期报价，
   本层先做一次同样的本地拦截，避免把过期数据推进风控。
3. 目标名义额受 ``canary_notional`` 上限约束（文档 §7.2：canary 期极小名义额，
   禁止自动放量）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal

from ..config import Config
from ..errors import LiveGateBlocked

__all__ = ["Signal", "build_signal"]


@dataclass(frozen=True, slots=True)
class Signal:
    """一条开仓意图。执行层据此生成 OrderIntent 并走双腿状态机。"""

    symbol: str
    target_notional: Decimal
    spot_price: Decimal
    perp_price: Decimal
    quote_ts_ms: int
    reason: str
    strategy_version: str = "funding_carry-1.0"

    @property
    def basis_pct(self) -> Decimal:
        """基差 = (perp - spot) / spot。正 = 永续溢价（对我们有利，空头赚资金费）。"""
        if self.spot_price <= 0:
            return Decimal("0")
        return (self.perp_price - self.spot_price) / self.spot_price


def _dec(value: float | int | str | Decimal) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def build_signal(
    symbol: str,
    *,
    spot_price: float | int | str | Decimal,
    perp_price: float | int | str | Decimal,
    quote_ts_ms: int,
    requested_notional: float | int | str | Decimal,
    config: Config,
    reason: str = "",
    strategy_version: str = "funding_carry-1.0",
    now_ms: int | None = None,
) -> Signal | None:
    """把一条候选转成 Signal；任何一项不满足返回 None（本地拒绝，不发请求）。

    拒绝条件：

    - 价格为非正或不是永续对（symbol 必须以 USDT 结尾）。
    - 报价过期（超过 ``max_market_data_age_seconds``）。
    - 目标名义额 <= 0，或超过 ``canary_notional`` 上限（超出部分**截断到上限**，
      不是拒绝 —— canary 配置就是本进程的名义额硬顶）。
    - 基差为负且绝对值超过 ``hedge_tolerance_pct``（深度贴水，开空头不利，
      等贴水收敛；文档 §7.2 基差检查）。
    """
    exc = config.execution

    if not symbol.endswith("USDT"):
        return None

    spot = _dec(spot_price)
    perp = _dec(perp_price)
    if spot <= 0 or perp <= 0:
        return None

    now = now_ms if now_ms is not None else int(time.time() * 1000)
    age_ms = now - quote_ts_ms
    if age_ms < 0 or age_ms > int(exc.max_market_data_age_seconds * 1000):
        raise LiveGateBlocked(
            f"报价过期：{symbol} 报价年龄 {age_ms}ms > {exc.max_market_data_age_seconds * 1000:.0f}ms。"
            "禁止用过期行情生成交易意图。"
        )

    notional = _dec(requested_notional)
    if notional <= 0:
        return None
    cap = _dec(str(exc.canary_notional))
    if notional > cap:
        notional = cap

    basis = (perp - spot) / spot
    if basis < 0 and abs(basis) > _dec(str(exc.hedge_tolerance_pct)):
        return None

    return Signal(
        symbol=symbol,
        target_notional=notional,
        spot_price=spot,
        perp_price=perp,
        quote_ts_ms=quote_ts_ms,
        reason=reason or "funding_carry",
        strategy_version=strategy_version,
    )
