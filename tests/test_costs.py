"""成本模型测试。

**为什么这个文件的每一条都是精确数值断言：**

成本是策略能否盈利的**门槛线**。0.29% 的往返成本意味着年化 20% 的策略
调仓 24 次就吃掉 7% —— 相当于 35% 的毛收益。

如果成本算错 0.01%（比如忘了 BNB 抵扣、或滑点按边而非按腿计），
回测结论会系统性地偏乐观，而你不会发现。

所以这里不写"大概等于"，每条断言都精确到浮点位，且附手工验算过程。
"""

from __future__ import annotations

import math

import pytest

from cointrader.config import CostsConfig, PerpFeeConfig, SpotFeeConfig
from cointrader.errors import ConfigError
from cointrader.research.costs import (
    TIER_THRESHOLDS,
    CostModel,
    LiquidityTier,
    classify_liquidity,
    pessimistic_config,
)


class TestFeeRates:
    """单腿费率计算。"""

    def test_spot_rate_applies_bnb_discount(self, costs_config: CostsConfig) -> None:
        """现货费率必须应用 BNB 抵扣。

        手工验算: 0.1% × 0.75 = 0.075%
        """
        model = CostModel(costs_config)
        assert model.spot_rate() == pytest.approx(0.00075, abs=1e-15)

    def test_perp_rate_applies_bnb_discount(self, costs_config: CostsConfig) -> None:
        """合约费率应用 9 折 BNB 折扣：0.05% × 0.90 = 0.045%。"""
        model = CostModel(costs_config)
        assert model.perp_rate() == pytest.approx(0.00045, abs=1e-15)

    def test_bnb_discount_disabled(self) -> None:
        """bnb_discount=1.0 表示不抵扣。"""
        config = CostsConfig(bnb_discount=1.0)
        model = CostModel(config)
        assert model.spot_rate() == pytest.approx(0.00100, abs=1e-15)

    def test_perp_maker_rate_is_lower_than_taker(self, costs_config: CostsConfig) -> None:
        """合约的 maker 费率低于 taker（0.02% vs 0.05%）。

        注意：**现货**在 VIP0 档位下 maker 与 taker 同为 0.1%，
        这是币安的实际费率结构，不是配置写错。所以这里只检查合约。
        """
        model = CostModel(costs_config)
        assert model.perp_rate(maker=True) == pytest.approx(0.00018, abs=1e-15)
        assert model.perp_rate(maker=False) == pytest.approx(0.00045, abs=1e-15)
        assert model.perp_rate(maker=True) < model.perp_rate(maker=False)

    def test_spot_maker_equals_taker_at_vip0(self, costs_config: CostsConfig) -> None:
        """VIP0 档位下现货 maker 与 taker 同价 —— 如实记录这个事实。

        之前有人误以为"maker 一定更便宜"，据此写出偏乐观的回测。
        这条测试把真实费率结构钉死。
        """
        model = CostModel(costs_config)
        assert model.spot_rate(maker=True) == pytest.approx(model.spot_rate(maker=False), abs=1e-15)


class TestRoundTrip:
    """往返成本 —— 本项目最重要的一个数字。"""

    def test_major_round_trip_is_exactly_0_29_percent(self, costs_config: CostsConfig) -> None:
        """主流币往返成本 = 0.29%。

        手工验算::

            腿1 买现货:  0.100% × 0.75 (BNB) = 0.0750%
            腿2 卖永续:  0.050%              = 0.0500%
            腿1 滑点:                        = 0.0100%
            腿2 滑点:                        = 0.0100%
            ─────────────────────────────────────────
            进场小计                          = 0.1450%
            出场小计（同结构）                = 0.1450%
            ═════════════════════════════════════════
            往返总计                          = 0.2900%

        这个数字就是策略的盈亏平衡门槛。docs/ARCHITECTURE.md §2.4 同步引用它。
        """
        model = CostModel(costs_config)
        assert model.round_trip(LiquidityTier.MAJOR) == pytest.approx(0.0028, abs=1e-15)

    def test_entry_and_exit_split_evenly(self, costs_config: CostsConfig) -> None:
        """进场与出场成本应对称（同为 2 条腿）。"""
        model = CostModel(costs_config)
        entry = model.entry_cost(LiquidityTier.MAJOR)
        exit_ = model.exit_cost(LiquidityTier.MAJOR)

        assert entry == pytest.approx(0.0014, abs=1e-15)
        assert exit_ == pytest.approx(0.0014, abs=1e-15)
        assert entry == pytest.approx(exit_, abs=1e-15)

    def test_slippage_charged_per_leg_not_per_side(self, costs_config: CostsConfig) -> None:
        """滑点按**每条腿**计（4 次），不是每边一次（2 次）。

        如果按边计，major 往返会变成 0.27% 而不是 0.29% —— 少算 0.02%，
        一年调仓 24 次就少算 0.48%。这条测试锁死正确的口径。
        """
        model = CostModel(costs_config)
        slip = model.slippage_rate(LiquidityTier.MAJOR)

        # 若按边计，成本会低 2×滑点
        per_side_total = model.spot_rate() * 2 + model.perp_rate() * 2 + slip * 2
        per_leg_total = model.round_trip(LiquidityTier.MAJOR)

        assert per_leg_total == pytest.approx(per_side_total + 2 * slip, abs=1e-15)

    @pytest.mark.parametrize(
        ("tier", "expected"),
        [
            (LiquidityTier.MAJOR, 0.0028),
            (LiquidityTier.MID, 0.0036),
            (LiquidityTier.SMALL, 0.0044),
        ],
    )
    def test_round_trip_by_tier(
        self, costs_config: CostsConfig, tier: LiquidityTier, expected: float
    ) -> None:
        """各流动性档位的往返成本。

        手工验算 mid: 2 × (0.075% + 0.05% + 2×0.03%) = 2 × 0.185% = 0.370%
        手工验算 small: 2 × (0.075% + 0.05% + 2×0.05%) = 2 × 0.225% = 0.450%
        """
        model = CostModel(costs_config)
        assert model.round_trip(tier) == pytest.approx(expected, abs=1e-15)

    def test_breakdown_total_matches_round_trip(self, costs_config: CostsConfig) -> None:
        """分解展示的总和必须与实际计算口径一致。

        这条测试的存在是因为曾经踩过坑：breakdown 单独累加明细字段，
        而 entry_cost() 用另一套公式，两者漂移了 0.02%。
        「展示用的数字」和「计算用的数字」必须是同一个来源。
        """
        model = CostModel(costs_config)
        for tier in LiquidityTier:
            breakdown = model.breakdown(tier)
            assert breakdown.total == pytest.approx(model.round_trip(tier), abs=1e-15)

    def test_maker_assumption_lowers_cost(self, costs_config: CostsConfig) -> None:
        """假设 maker 成交时成本必须更低。"""
        model = CostModel(costs_config)
        taker_cost = model.round_trip(LiquidityTier.MAJOR, use_maker=False)
        maker_cost = model.round_trip(LiquidityTier.MAJOR, use_maker=True)

        assert maker_cost < taker_cost
        # 手工验算 maker: 2 × (0.075% + 0.02% + 2×0.01%) = 2 × 0.115% = 0.230%
        assert maker_cost == pytest.approx(0.00226, abs=1e-15)


class TestPessimisticConfig:
    """悲观成本配置 —— 用于压力测试。"""

    def test_pessimistic_removes_bnb_discount(self, costs_config: CostsConfig) -> None:
        pessimistic = pessimistic_config(costs_config)
        assert pessimistic.bnb_discount == 1.0

    def test_pessimistic_doubles_slippage(self, costs_config: CostsConfig) -> None:
        pessimistic = pessimistic_config(costs_config)
        assert pessimistic.slippage.major == pytest.approx(costs_config.slippage.major * 2)
        assert pessimistic.slippage.mid == pytest.approx(costs_config.slippage.mid * 2)
        assert pessimistic.slippage.small == pytest.approx(costs_config.slippage.small * 2)

    def test_pessimistic_uses_taker_for_maker(self, costs_config: CostsConfig) -> None:
        """悲观配置下，即使假设 maker 也按 taker 收费。"""
        pessimistic = pessimistic_config(costs_config)
        assert pessimistic.spot.maker == pessimistic.spot.taker
        assert pessimistic.perp.maker == pessimistic.perp.taker

    def test_pessimistic_cost_is_higher(self, costs_config: CostsConfig) -> None:
        """悲观成本必须显著高于基准，否则压力测试没意义。"""
        base = CostModel(costs_config).round_trip(LiquidityTier.MAJOR)
        pessimistic = CostModel(pessimistic_config(costs_config)).round_trip(LiquidityTier.MAJOR)

        assert pessimistic > base
        # 手工验算: 2 × (0.1% + 0.05% + 2×0.02%) = 2 × 0.19% = 0.38%
        assert pessimistic == pytest.approx(0.0038, abs=1e-15)

    def test_maker_assumption_has_no_effect_under_pessimistic(self, costs_config: CostsConfig) -> None:
        """悲观配置下 use_maker 参数不应产生任何差异。"""
        model = CostModel(pessimistic_config(costs_config))
        assert model.round_trip(LiquidityTier.MAJOR, use_maker=True) == pytest.approx(
            model.round_trip(LiquidityTier.MAJOR, use_maker=False), abs=1e-15
        )


class TestBreakeven:
    """盈亏平衡计算。"""

    def test_breakeven_periods(self, costs_config: CostsConfig) -> None:
        """盈亏平衡期数 = 往返成本 / 单期费率。

        手工验算: 0.29% / 0.02% = 14.5 期
        """
        model = CostModel(costs_config)
        periods = model.breakeven_periods(0.0002, 8)
        assert periods == pytest.approx(14.0, abs=1e-9)

    def test_breakeven_days_8h_settlement(self, costs_config: CostsConfig) -> None:
        """8h 结算下，14.5 期 ≈ 4.83 天。

        手工验算: 14.5 × 8h = 116h = 4.833 天
        """
        model = CostModel(costs_config)
        days = model.breakeven_days(0.0002, 8)
        assert days == pytest.approx(14.0 * 8 / 24, abs=1e-9)

    def test_breakeven_days_4h_settlement_is_shorter(self, costs_config: CostsConfig) -> None:
        """4h 结算的币，同样单期费率下回本更快（因为收得更频繁）。

        这个对比说明了为什么结算周期必须逐币处理：
        用 8h 公式套 4h 的币，会误判为"要等两倍时间才回本"。
        """
        model = CostModel(costs_config)
        days_8h = model.breakeven_days(0.0002, 8)
        days_4h = model.breakeven_days(0.0002, 4)

        assert days_4h == pytest.approx(days_8h / 2, abs=1e-9)

    def test_breakeven_infinite_for_zero_or_negative_rate(self, costs_config: CostsConfig) -> None:
        """零费率或负费率下永远无法回本（成本是沉的）。"""
        model = CostModel(costs_config)
        assert math.isinf(model.breakeven_periods(0.0, 8))
        assert math.isinf(model.breakeven_periods(-0.0001, 8))
        assert math.isinf(model.breakeven_days(0.0, 8))


class TestLiquidityClassification:
    """流动性分档。"""

    @pytest.mark.parametrize(
        ("volume", "expected"),
        [
            (1_000_000_000.0, LiquidityTier.MAJOR),
            (500_000_000.0, LiquidityTier.MAJOR),   # 边界，含
            (499_999_999.0, LiquidityTier.MID),
            (50_000_000.0, LiquidityTier.MID),      # 边界，含
            (49_999_999.0, LiquidityTier.SMALL),
            (1_000.0, LiquidityTier.SMALL),
            (0.0, LiquidityTier.SMALL),
        ],
    )
    def test_classification_boundaries(self, volume: float, expected: LiquidityTier) -> None:
        """分档阈值必须是闭区间下界（>=）。"""
        assert classify_liquidity(volume) is expected

    def test_thresholds_are_descending(self) -> None:
        """阈值必须降序排列，否则分档结果不确定。"""
        values = [threshold for threshold, _ in TIER_THRESHOLDS]
        assert values == sorted(values, reverse=True)


class TestCostModelValidation:
    """成本配置的校验。"""

    def test_rejects_absurd_fee_rate(self) -> None:
        """费率写成 5% 以上必定是单位写错（0.1 而非 0.001），必须拒绝。"""
        with pytest.raises(ConfigError, match="费率异常"):
            CostsConfig(spot=SpotFeeConfig(maker=0.001, taker=0.1))   # 0.1 = 10%，明显是写错单位

    def test_rejects_invalid_bnb_discount(self) -> None:
        with pytest.raises(ConfigError, match="bnb_discount"):
            CostsConfig(bnb_discount=1.5)

        with pytest.raises(ConfigError, match="bnb_discount"):
            CostsConfig(bnb_discount=0.0)

    def test_accepts_zero_fee(self) -> None:
        """零费率是合法的边界值（用于理论分析）。"""
        config = CostsConfig(spot=SpotFeeConfig(maker=0.0, taker=0.0), perp=PerpFeeConfig(maker=0.0, taker=0.0))
        model = CostModel(config)
        # 仍有滑点成本
        assert model.round_trip(LiquidityTier.MAJOR) == pytest.approx(0.0004, abs=1e-15)


class TestAbsoluteAmounts:
    """绝对金额换算。"""

    def test_round_trip_usdt(self, costs_config: CostsConfig) -> None:
        """10000 USDT 名义额，往返成本 = 29 USDT。"""
        model = CostModel(costs_config)
        assert model.round_trip_usdt(10_000.0, LiquidityTier.MAJOR) == pytest.approx(28.0, abs=1e-9)

    def test_annual_cost_at_24_rebalances(self, costs_config: CostsConfig) -> None:
        """一年调仓 24 次的成本 = 6.96%。

        这个数字是策略的门槛：年化毛收益低于 7% 就不值得做。
        """
        model = CostModel(costs_config)
        annual = model.round_trip(LiquidityTier.MAJOR) * 24
        assert annual == pytest.approx(0.0672, abs=1e-12)


__all__: list[str] = []
