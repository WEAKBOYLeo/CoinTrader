"""成本模型 —— 全项目唯一的手续费真相源。

**为什么这个模块值得单独存在并配一套专门的测试：**

策略的年化收益通常在 10%~40% 之间，而往返交易成本是 0.29%。
看似只占 1%，但策略平均持仓约 10~30 天，一年要调仓 12~36 次：:

    年成本 = 0.29% × 24 次 = 6.96%

这相当于**吃掉毛收益的 20%~70%**。费率算错 0.02%（比如忘了 BNB 抵扣）
一年就是 0.5% 的收益差；把 maker 当成 taker 算错 0.05%，一年是 1.2%。

所以本模块的每个数字都必须能被手工验算，且都有测试覆盖。

一次完整往返 = **4 条腿**：

===  ====  ==========================================
腿    方向  说明
===  ====  ==========================================
1     买入  现货建仓（多头）
2     卖出  永续建仓（空头）
...   持有  期间收资金费（不计入成本）
3     卖出  现货平仓
4     买入  永续平仓
===  ====  ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..config import CostsConfig


class LiquidityTier(str, Enum):
    """流动性分档 —— 决定滑点假设。

    滑点是回测里最容易被低估的一项。BTC 的滑点可以忽略，
    但日成交额 1000 万 USDT 的小币，单边 5bp 都是乐观的。
    """

    MAJOR = "major"
    MID = "mid"
    SMALL = "small"


#: 按 24h 成交额自动分档的阈值（USDT）
TIER_THRESHOLDS: tuple[tuple[float, LiquidityTier], ...] = (
    (500_000_000.0, LiquidityTier.MAJOR),   # > 5 亿  → major
    (50_000_000.0, LiquidityTier.MID),      # > 5000 万 → mid
    (0.0, LiquidityTier.SMALL),             # 其余      → small
)


def classify_liquidity(quote_volume_24h: float) -> LiquidityTier:
    """按 24h 成交额把币种分档。"""
    for threshold, tier in TIER_THRESHOLDS:
        if quote_volume_24h >= threshold:
            return tier
    return LiquidityTier.SMALL


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """一次往返成本的完整分解。所有字段都是**占名义额的比例**。

    滑点按**每条腿**计（4 条腿共 4 次），而不是每边一次。理由：
    每条腿都是真实的一次市场成交，都要承担价差与冲击成本。
    按边计会低估一半滑点 —— 而滑点恰是最容易被低估的一项。

    ``total_entry`` / ``total_exit`` / ``total`` 直接取自 ``CostModel``
    的对应方法，而不是把下面的明细字段相加。这样做是为了让"分解展示"
    和"实际用于计算的数字"**永远是同一个来源**，不会因为改了计费规则
    而两处漂移（这正是本类第一版踩过的坑）。
    """

    spot_entry: float
    perp_entry: float
    slippage_per_leg: float
    spot_exit: float
    perp_exit: float
    total_entry: float
    total_exit: float

    @property
    def total(self) -> float:
        """完整往返成本。"""
        return self.total_entry + self.total_exit

    @property
    def legs_per_side(self) -> int:
        return 2

    def as_dict(self) -> dict[str, float]:
        return {
            "spot_entry": self.spot_entry,
            "perp_entry": self.perp_entry,
            "slippage_per_leg": self.slippage_per_leg,
            "spot_exit": self.spot_exit,
            "perp_exit": self.perp_exit,
            "total_entry": self.total_entry,
            "total_exit": self.total_exit,
            "total_round_trip": self.total,
        }


class CostModel:
    """交易成本模型。

    默认配置（VIP0 + BNB 抵扣 + 全 taker + major 流动性）下的往返成本为
    **0.29%**，这个数字在 docs/ARCHITECTURE.md §2.4 有推导过程。

    Args:
        costs: 成本配置。
    """

    def __init__(self, costs: CostsConfig) -> None:
        self._costs = costs

    # -- 单腿费率 -----------------------------------------------------------

    def spot_rate(self, *, maker: bool = False) -> float:
        """现货单边费率（已含 BNB 抵扣）。

        Note:
            币安的 BNB 抵扣只适用于现货手续费，合约**不适用**。
            这个不对称性很容易被搞混。
        """
        base = self._costs.spot.maker if maker else self._costs.spot.taker
        return base * self._costs.spot_bnb_discount

    def perp_rate(self, *, maker: bool = False) -> float:
        """永续单边费率，应用合约 BNB 支付折扣。"""
        base = self._costs.perp.maker if maker else self._costs.perp.taker
        return base * self._costs.perp_bnb_discount

    def slippage_rate(
        self,
        tier: LiquidityTier | str,
        override_per_leg: float | None = None,
    ) -> float:
        """给定流动性档位的单腿滑点，可由回测显式覆盖。"""
        if override_per_leg is not None:
            if not 0.0 <= override_per_leg <= 1.0:
                raise ValueError(f"override_per_leg 必须在 [0,1] 内，当前 {override_per_leg}")
            return override_per_leg
        tier = LiquidityTier(tier)
        slippage = self._costs.slippage
        return {
            LiquidityTier.MAJOR: slippage.major,
            LiquidityTier.MID: slippage.mid,
            LiquidityTier.SMALL: slippage.small,
        }[tier]

    # -- 组合成本 -----------------------------------------------------------

    def entry_cost(
        self,
        tier: LiquidityTier | str = LiquidityTier.MAJOR,
        *,
        use_maker: bool | None = None,
        slippage_per_leg: float | None = None,
    ) -> float:
        """建仓成本比例（腿 1 + 腿 2 + 两条腿的滑点）。"""
        maker = (not self._costs.assume_all_taker) if use_maker is None else use_maker
        return (
            self.spot_rate(maker=maker)
            + self.perp_rate(maker=maker)
            + 2 * self.slippage_rate(tier, slippage_per_leg)
        )

    def exit_cost(
        self,
        tier: LiquidityTier | str = LiquidityTier.MAJOR,
        *,
        use_maker: bool | None = None,
        slippage_per_leg: float | None = None,
    ) -> float:
        """平仓成本比例（腿 3 + 腿 4 + 两条腿的滑点）。"""
        maker = (not self._costs.assume_all_taker) if use_maker is None else use_maker
        return (
            self.spot_rate(maker=maker)
            + self.perp_rate(maker=maker)
            + 2 * self.slippage_rate(tier, slippage_per_leg)
        )

    def round_trip(
        self,
        tier: LiquidityTier | str = LiquidityTier.MAJOR,
        *,
        use_maker: bool | None = None,
        slippage_per_leg: float | None = None,
    ) -> float:
        """完整往返成本比例。"""
        return self.entry_cost(
            tier,
            use_maker=use_maker,
            slippage_per_leg=slippage_per_leg,
        ) + self.exit_cost(
            tier,
            use_maker=use_maker,
            slippage_per_leg=slippage_per_leg,
        )

    def breakdown(
        self,
        tier: LiquidityTier | str = LiquidityTier.MAJOR,
        *,
        use_maker: bool | None = None,
        slippage_per_leg: float | None = None,
    ) -> CostBreakdown:
        """完整成本分解，用于回测报告展示。"""
        maker = (not self._costs.assume_all_taker) if use_maker is None else use_maker
        spot = self.spot_rate(maker=maker)
        perp = self.perp_rate(maker=maker)
        slip = self.slippage_rate(tier, slippage_per_leg)
        return CostBreakdown(
            spot_entry=spot,
            perp_entry=perp,
            slippage_per_leg=slip,
            spot_exit=spot,
            perp_exit=perp,
            total_entry=self.entry_cost(
                tier,
                use_maker=use_maker,
                slippage_per_leg=slippage_per_leg,
            ),
            total_exit=self.exit_cost(
                tier,
                use_maker=use_maker,
                slippage_per_leg=slippage_per_leg,
            ),
        )

    # -- 绝对金额 -----------------------------------------------------------

    def round_trip_usdt(
        self,
        notional: float,
        tier: LiquidityTier | str = LiquidityTier.MAJOR,
        *,
        use_maker: bool | None = None,
        slippage_per_leg: float | None = None,
    ) -> float:
        """给定名义额，算出往返成本的绝对金额（USDT）。"""
        return notional * self.round_trip(
            tier,
            use_maker=use_maker,
            slippage_per_leg=slippage_per_leg,
        )

    # -- 盈亏平衡 -----------------------------------------------------------

    def breakeven_periods(
        self,
        mean_rate_per_period: float,
        interval_hours: int,
        tier: LiquidityTier | str = LiquidityTier.MAJOR,
        *,
        use_maker: bool | None = None,
    ) -> float:
        """在给定平均单期费率下，需要持有多少期才回本。

        这是判断一笔交易"值不值得做"的第一道关卡。

        Args:
            mean_rate_per_period: 平均单期资金费率（小数）。
            interval_hours: 结算周期。
            tier: 流动性档位。

        Returns:
            需要的结算期数。费率为负或为 0 时返回 ``inf``。
        """
        if mean_rate_per_period <= 0:
            return float("inf")
        return self.round_trip(tier, use_maker=use_maker) / mean_rate_per_period

    def breakeven_days(
        self,
        mean_rate_per_period: float,
        interval_hours: int,
        tier: LiquidityTier | str = LiquidityTier.MAJOR,
        *,
        use_maker: bool | None = None,
    ) -> float:
        """盈亏平衡所需的天数。"""
        periods = self.breakeven_periods(
            mean_rate_per_period, interval_hours, tier, use_maker=use_maker
        )
        if periods == float("inf"):
            return float("inf")
        return periods * interval_hours / 24.0


def pessimistic_config(costs: CostsConfig) -> CostsConfig:
    """把成本配置调成**最悲观**的版本，用于压力测试。

    规则：
    - 所有手续费按 taker 计（即使配置允许 maker）
    - BNB 抵扣取消（设为 1.0）—— 万一 BNB 持仓不足或规则变更
    - 滑点翻倍

    稳健的策略应该在悲观成本下依然为正。如果只有乐观成本下才赚钱，
    那赚的是假设的钱，不是市场的钱。
    """
    from ..config import PerpFeeConfig, SlippageConfig, SpotFeeConfig

    return CostsConfig(
        fee_tier=f"{costs.fee_tier}_pessimistic",
        spot=SpotFeeConfig(maker=costs.spot.taker, taker=costs.spot.taker),
        perp=PerpFeeConfig(maker=costs.perp.taker, taker=costs.perp.taker),
        spot_bnb_discount=1.0,
        perp_bnb_discount=1.0,
        bnb_discount=1.0,
        slippage=SlippageConfig(
            major=costs.slippage.major * 2,
            mid=costs.slippage.mid * 2,
            small=costs.slippage.small * 2,
        ),
        assume_all_taker=True,
    )


__all__ = [
    "TIER_THRESHOLDS",
    "CostBreakdown",
    "CostModel",
    "LiquidityTier",
    "classify_liquidity",
    "pessimistic_config",
]
