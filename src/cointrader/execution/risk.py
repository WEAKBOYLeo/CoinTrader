"""风控限额检查 —— 与安全闸门**独立**的第二道防线。

为什么要有两道：``guard.py`` 关心的是「有没有被授权下单」，
本模块关心的是「这一单下下去会不会造成不可接受的敞口」。

两者的失效模式不同：guard 可能被误配置绕过（比如环境变量被继承），
风控则是纯粹的数字比较，更难弄错。任何一关拒绝，下单就不发生。

**本模块所有方法都是纯函数式的**：给定当前状态和意图，返回
``(是否允许, 原因)``。它不修改任何状态，也不发起任何 IO。
这样它能被完整单测覆盖，而不需要真的连交易所。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..config import RiskConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Position:
    """单个币种的当前持仓状态。"""

    symbol: str
    spot_qty: float = 0.0       # 现货多头数量
    perp_qty: float = 0.0       # 永续空头数量（正数表示空头）
    spot_price: float = 0.0     # 现货最新价
    perp_price: float = 0.0     # 永续最新价
    spot_entry_price: float = 0.0

    @property
    def spot_notional(self) -> float:
        return abs(self.spot_qty) * self.spot_price

    @property
    def perp_notional(self) -> float:
        return abs(self.perp_qty) * self.perp_price

    @property
    def notional(self) -> float:
        """该币种的总名义敞口（两腿之和，保守口径）。"""
        return self.spot_notional + self.perp_notional

    @property
    def basis(self) -> float:
        """基差 = (永续价 - 现货价) / 现货价。现货价为 0 时返回 0。"""
        if self.spot_price <= 0:
            return 0.0
        return (self.perp_price - self.spot_price) / self.spot_price

    @property
    def hedge_ratio(self) -> float:
        """对冲比例。1.0 表示完全对冲，偏离 1 意味着裸露方向性风险。

        两腿名义额应当相等。偏离超过 1% 就说明有一腿没成交或部分成交，
        此时组合变成有方向敞口 —— 这是最危险的中间态。
        """
        if self.spot_notional <= 0:
            return 0.0
        return self.perp_notional / self.spot_notional

    @property
    def is_hedged(self, tolerance: float = 0.01) -> bool:
        return abs(self.hedge_ratio - 1.0) <= tolerance


@dataclass(slots=True)
class RiskState:
    """风控所需的全部状态快照。"""

    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl_today: float = 0.0
    total_capital: float = 0.0
    day_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    #: 最近一次出现「单腿裸露」的时间戳（秒）。None 表示当前无裸露。
    unhedged_since: float | None = None
    #: 交易所快照时间（UTC ms）。0 = 未知（默认值不得用于放行开仓）。
    snapshot_ts_ms: int = 0
    #: 快照来源（periodic/startup/reconcile）。
    snapshot_source: str = ""
    available_balance: float = 0.0
    futures_wallet_balance: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def total_exposure(self) -> float:
        return sum(p.notional for p in self.positions.values())

    def exposure_for(self, symbol: str) -> float:
        return self.positions.get(symbol, Position(symbol)).notional


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    """风控判定结果。"""

    allowed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls, reason: str = "通过", **details: Any) -> RiskVerdict:
        return cls(allowed=True, reason=reason, details=details)

    @classmethod
    def deny(cls, reason: str, **details: Any) -> RiskVerdict:
        return cls(allowed=False, reason=reason, details=details)


class RiskManager:
    """限额检查器。

    Args:
        config: 风控配置。
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    # -- 单笔订单检查 -------------------------------------------------------

    def check_order(
        self,
        symbol: str,
        notional: float,
        state: RiskState,
        *,
        is_closing: bool = False,
    ) -> RiskVerdict:
        """检查一笔订单是否放行。

        Args:
            symbol: 交易对。
            notional: 本笔订单的名义额（USDT）。
            state: 当前风控状态。
            is_closing: 是否为**平仓**订单。平仓应该比开仓更宽松 ——
                如果风控把平仓也拦了，你会在最需要离场的时候被困住。

        Returns:
            RiskVerdict。
        """
        if notional <= 0:
            return RiskVerdict.deny("订单名义额必须为正", notional=notional)

        # 平仓单跳过大部分限额 —— 让减仓永远可行
        if is_closing:
            return RiskVerdict.allow("平仓单，放行", notional=notional)

        if notional > self.config.max_notional_per_order:
            return RiskVerdict.deny(
                f"单笔名义额 {notional:.2f} 超过上限 {self.config.max_notional_per_order:.2f}",
                notional=notional,
                limit=self.config.max_notional_per_order,
            )

        projected_symbol = state.exposure_for(symbol) + notional
        if projected_symbol > self.config.max_exposure_per_symbol:
            return RiskVerdict.deny(
                f"{symbol} 敞口将达到 {projected_symbol:.2f}，"
                f"超过单币种上限 {self.config.max_exposure_per_symbol:.2f}",
                projected=projected_symbol,
                limit=self.config.max_exposure_per_symbol,
            )

        projected_total = state.total_exposure + notional
        if projected_total > self.config.max_total_exposure:
            return RiskVerdict.deny(
                f"总敞口将达到 {projected_total:.2f}，"
                f"超过总上限 {self.config.max_total_exposure:.2f}",
                projected=projected_total,
                limit=self.config.max_total_exposure,
            )

        return RiskVerdict.allow(
            "通过全部限额检查",
            notional=notional,
            projected_total=projected_total,
        )

    # -- 全局状态检查 -------------------------------------------------------

    def check_daily_loss(self, state: RiskState) -> RiskVerdict:
        """检查 24h 亏损是否触及停机线。

        ``realized_pnl_today`` 应为负数表示亏损。
        """
        if state.total_capital <= 0:
            return RiskVerdict.deny("总资金未知或为零，无法评估亏损限额")

        loss_pct = -state.realized_pnl_today / state.total_capital
        if loss_pct >= self.config.max_daily_loss_pct:
            return RiskVerdict.deny(
                f"24h 亏损 {loss_pct:.2%} 已达停机线 {self.config.max_daily_loss_pct:.2%}",
                loss_pct=loss_pct,
            )
        return RiskVerdict.allow(f"24h 亏损 {loss_pct:.2%} 在限额内", loss_pct=loss_pct)

    def check_hedge_integrity(self, state: RiskState) -> list[RiskVerdict]:
        """检查所有持仓的对冲完整性。

        返回每个**未完全对冲**持仓的告警。空列表表示全部健康。

        为什么这个检查重要：现货腿成交了但永续腿没成交（或反之），
        组合就从 delta 中性变成了裸多头/裸空头。这是资金费套利唯一
        会真正亏大钱的方式 —— 一次没对冲上的暴跌能抹掉几个月的收益。
        """
        warnings: list[RiskVerdict] = []
        for symbol, position in state.positions.items():
            if position.spot_notional <= 0 and position.perp_notional <= 0:
                continue
            if position.is_hedged:
                continue
            warnings.append(
                RiskVerdict.deny(
                    f"{symbol} 未完全对冲：现货 {position.spot_notional:.2f} / "
                    f"永续 {position.perp_notional:.2f}（比例 {position.hedge_ratio:.4f}）",
                    symbol=symbol,
                    hedge_ratio=position.hedge_ratio,
                    spot_notional=position.spot_notional,
                    perp_notional=position.perp_notional,
                )
            )
        return warnings

    def check_unhedged_duration(self, state: RiskState, now: float) -> RiskVerdict:
        """检查单腿裸露是否超时。

        超过 ``max_unhedged_seconds`` 说明补腿失败，需要人工介入。
        """
        if state.unhedged_since is None:
            return RiskVerdict.allow("无裸露头寸")

        elapsed = now - state.unhedged_since
        if elapsed > self.config.max_unhedged_seconds:
            return RiskVerdict.deny(
                f"单腿裸露已持续 {elapsed:.1f}s，超过上限 "
                f"{self.config.max_unhedged_seconds}s，需要人工介入",
                elapsed_seconds=elapsed,
            )
        return RiskVerdict.allow(f"裸露 {elapsed:.1f}s，未超时", elapsed_seconds=elapsed)

    def check_basis(self, position: Position) -> RiskVerdict:
        """检查基差不利变动是否触及减仓线。

        注意方向：我们是**空永续 + 多现货**，所以
        **永续溢价扩大（basis 变大）对我们不利**（空头亏损）。
        """
        basis = position.basis
        if basis >= self.config.max_basis_adverse_pct:
            return RiskVerdict.deny(
                f"{position.symbol} 基差 {basis:.4%} 已达减仓线 "
                f"{self.config.max_basis_adverse_pct:.4%}（空永续方不利）",
                symbol=position.symbol,
                basis=basis,
            )
        return RiskVerdict.allow(f"基差 {basis:.4%} 在正常范围", basis=basis)

    # -- 综合检查 -----------------------------------------------------------

    def preflight(self, state: RiskState, now: float | None = None) -> RiskVerdict:
        """下单前的全面自检。

        在每次真实下单前调用。任一项不通过则整体拒绝。

        检查顺序按「后果严重程度」排列，先报最严重的。
        """
        import time as _time

        now = now if now is not None else _time.time()

        daily = self.check_daily_loss(state)
        if not daily.allowed:
            return daily

        unhedged = self.check_unhedged_duration(state, now)
        if not unhedged.allowed:
            return unhedged

        hedge_warnings = self.check_hedge_integrity(state)
        if hedge_warnings:
            # 未对冲的持仓存在时，不允许**新增**仓位 ——
            # 先把手里的烂摊子收拾干净
            return RiskVerdict.deny(
                f"存在 {len(hedge_warnings)} 个未完全对冲的持仓，禁止开新仓",
                warnings=[w.reason for w in hedge_warnings],
            )

        for position in state.positions.values():
            basis_check = self.check_basis(position)
            if not basis_check.allowed:
                return basis_check

        return RiskVerdict.allow(
            "全部风控检查通过",
            n_positions=len(state.positions),
            total_exposure=round(state.total_exposure, 2),
        )


__all__ = ["Position", "RiskManager", "RiskState", "RiskVerdict"]
