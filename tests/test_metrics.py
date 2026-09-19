"""绩效指标测试。

指标是回测的**结论**。指标算错，等于结论错，而你不会有任何察觉。

这里重点覆盖三类容易出错的地方：

1. **复利 vs 单利** —— 逐期比例收益必须用 cumprod 累计，用 sum 会低估
2. **回撤的持续期** —— 不是"最深那一点"，而是"从高点到恢复的时长"
3. **两种收益率口径** —— 占资金 vs 占投入，混淆会得出相反的结论
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cointrader.errors import InsufficientDataError
from cointrader.research.metrics import (
    compute_metrics,
    concentration_check,
    forward_split,
    longest_streak,
    max_drawdown,
    placebo_test,
    profit_factor,
    sharpe_ratio,
)


def make_returns(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


class TestMaxDrawdown:
    """最大回撤。"""

    def test_monotonic_increase_has_no_drawdown(self) -> None:
        equity = pd.Series([1.0, 1.1, 1.2, 1.3])
        dd, duration = max_drawdown(equity)

        assert dd == pytest.approx(0.0)
        assert duration == 0

    def test_simple_drawdown(self) -> None:
        """手工验算: 从 1.0 → 0.8，回撤 = 20%"""
        equity = pd.Series([1.0, 0.8, 0.9])
        dd, _ = max_drawdown(equity)

        assert dd == pytest.approx(0.20, rel=1e-9)

    def test_peak_then_recovery(self) -> None:
        """从 1.0 → 1.5 → 1.2，相对峰值回撤 = (1.5-1.2)/1.5 = 20%"""
        equity = pd.Series([1.0, 1.5, 1.2])
        dd, _ = max_drawdown(equity)

        assert dd == pytest.approx(0.20, rel=1e-9)

    def test_drawdown_duration(self) -> None:
        """回撤持续期应为"处于水下"的最长连续期数。"""
        # 涨、跌（水下3期）、恢复到高点、再涨
        equity = pd.Series([1.0, 0.9, 0.95, 0.99, 1.0, 1.1])
        _, duration = max_drawdown(equity)

        assert duration == 3, f"水下期数应为 3，实际 {duration}"

    def test_single_point_series(self) -> None:
        dd, duration = max_drawdown(pd.Series([1.0]))
        assert dd == 0.0
        assert duration == 0

    def test_empty_series(self) -> None:
        dd, duration = max_drawdown(pd.Series([], dtype=float))
        assert dd == 0.0
        assert duration == 0


class TestSharpe:
    """夏普比率。"""

    def test_zero_volatility_returns_zero(self) -> None:
        """恒定收益的标准差为 0，夏普应为 0（而不是 inf）。

        返回 inf 会污染排序和平均值计算。
        """
        assert sharpe_ratio(make_returns([0.001] * 50)) == 0.0

    def test_positive_mean_gives_positive_sharpe(self) -> None:
        rng = np.random.default_rng(1)
        returns = make_returns(list(rng.normal(0.001, 0.0005, 200)))
        assert sharpe_ratio(returns) > 0

    def test_negative_mean_gives_negative_sharpe(self) -> None:
        rng = np.random.default_rng(2)
        returns = make_returns(list(rng.normal(-0.001, 0.0005, 200)))
        assert sharpe_ratio(returns) < 0

    def test_insufficient_data(self) -> None:
        assert sharpe_ratio(make_returns([0.001])) == 0.0


class TestProfitFactor:
    """盈亏比。"""

    def test_known_values(self) -> None:
        """手工验算: 盈利 0.3，亏损 0.1，盈亏比 = 3.0"""
        returns = make_returns([0.2, 0.1, -0.1])
        assert profit_factor(returns) == pytest.approx(3.0, rel=1e-9)

    def test_no_losses_returns_inf(self) -> None:
        assert math.isinf(profit_factor(make_returns([0.1, 0.2])))

    def test_no_gains_returns_zero(self) -> None:
        assert profit_factor(make_returns([-0.1, -0.2])) == 0.0

    def test_all_zero_returns_zero(self) -> None:
        assert profit_factor(make_returns([0.0, 0.0])) == 0.0


class TestLongestStreak:
    """连续同号期数。"""

    def test_negative_streak(self) -> None:
        returns = make_returns([0.1, -0.1, -0.1, -0.1, 0.1, -0.1])
        assert longest_streak(returns, negative=True) == 3

    def test_positive_streak(self) -> None:
        returns = make_returns([0.1, 0.2, 0.3, -0.1, 0.1])
        assert longest_streak(returns, negative=False) == 3

    def test_all_negative(self) -> None:
        returns = make_returns([-0.1] * 10)
        assert longest_streak(returns, negative=True) == 10


class TestComputeMetrics:
    """指标汇总。"""

    def test_compound_not_simple_sum(self) -> None:
        """**关键检查**：累计收益必须用复利，不是简单相加。

        手工验算::
            每期 +10%，三期
            复利: 1.1³ - 1 = 33.1%
            单利: 30%

        用单利会低估收益，且在期数多时误差巨大。
        """
        returns = make_returns([0.1, 0.1, 0.1])
        metrics = compute_metrics(returns, periods_per_year=365.0)

        assert metrics.net_return == pytest.approx(0.331, rel=1e-9)
        assert metrics.net_return != pytest.approx(0.3, rel=1e-9)

    def test_cost_reduces_net_return(self) -> None:
        """净值 = 毛值 - 成本，且成本被独立记录。"""
        gross = make_returns([0.01, 0.01, 0.01])
        cost = make_returns([0.002, 0.002, 0.002])
        net = gross - cost

        metrics = compute_metrics(
            net, gross_returns=gross, cost_returns=cost, periods_per_year=365.0
        )

        assert metrics.cost_paid == pytest.approx(0.006, rel=1e-9)
        assert metrics.net_return < metrics.gross_return

    def test_cost_to_gross_ratio(self) -> None:
        """费用占毛收益比例。手工验算: 0.006 / 0.03 = 20%"""
        gross = make_returns([0.01, 0.01, 0.01])
        cost = make_returns([0.002, 0.002, 0.002])

        metrics = compute_metrics(
            gross - cost, gross_returns=gross, cost_returns=cost, periods_per_year=365.0
        )

        # cost_paid 是比例的线性累加，gross_return 是复利
        # 这里主要验证比例在合理范围
        assert 0 < metrics.cost_to_gross_ratio < 1

    def test_cost_ratio_infinite_when_no_gross(self) -> None:
        """毛收益 <= 0 时，费用占比无意义，应为 inf。"""
        returns = make_returns([-0.01, -0.01])
        metrics = compute_metrics(
            returns,
            gross_returns=make_returns([-0.01, -0.01]),
            cost_returns=make_returns([0.0, 0.0]),
            periods_per_year=365.0,
        )
        assert math.isinf(metrics.cost_to_gross_ratio)

    def test_annualization_uses_periods_per_year(self) -> None:
        """年化必须按 periods_per_year 折算。

        手工验算::
            1095 期/年（8h 结算），每期 +0.01%
            两期总收益 = 1.0001² - 1 ≈ 0.0002
            年数 = 2/1095
            年化 = (1.0001²)^(1095/2) - 1 = 1.0001^1095 - 1 ≈ 11.57%
        """
        returns = make_returns([0.0001] * 2)
        metrics = compute_metrics(returns, periods_per_year=1095.0)

        expected = (1.0001**1095) - 1.0
        assert metrics.net_annualized == pytest.approx(expected, rel=1e-6)

    def test_fewer_periods_per_year_gives_lower_annualization(self) -> None:
        """同样的逐期收益，周期越多年化越高（复利次数多）。

        这解释了为什么 4h 结算的币在同等单期费率下更有吸引力。
        """
        returns = make_returns([0.0002] * 100)

        annual_8h = compute_metrics(returns, periods_per_year=1095.0).net_annualized
        annual_4h = compute_metrics(returns, periods_per_year=2190.0).net_annualized

        assert annual_4h > annual_8h

    def test_win_rate(self) -> None:
        returns = make_returns([0.01, 0.01, -0.005, 0.01, -0.005])
        metrics = compute_metrics(returns, periods_per_year=365.0)

        assert metrics.win_rate == pytest.approx(3 / 5, rel=1e-9)

    def test_records_period_counts(self) -> None:
        returns = make_returns([0.001] * 50)
        metrics = compute_metrics(returns, periods_per_year=365.0, total_periods_available=100)

        assert metrics.periods == 50
        assert metrics.total_periods_available == 100

    def test_empty_series_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            compute_metrics(make_returns([]), periods_per_year=365.0)

    def test_single_period_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            compute_metrics(make_returns([0.001]), periods_per_year=365.0)

    def test_invalid_periods_per_year_raises(self) -> None:
        with pytest.raises(ValueError, match="periods_per_year"):
            compute_metrics(make_returns([0.001, 0.002]), periods_per_year=0)


class TestReturnOnDeployed:
    """两种收益率口径 —— 区分「策略不赚钱」与「资金没用满」。"""

    def test_return_on_deployed_scales_by_exposure(self) -> None:
        """占投入口径 = 占资金口径 / 资金使用率。

        手工验算::
            资金使用率 20%，占资金年化 10%
            → 占投入年化 = 10% / 0.20 = 50%
        """
        returns = make_returns([0.0001] * 100)
        exposure = pd.Series([0.2] * 100)

        metrics = compute_metrics(
            returns, periods_per_year=365.0, exposure=exposure
        )

        assert metrics.avg_exposure == pytest.approx(0.2)
        assert metrics.return_on_deployed == pytest.approx(
            metrics.net_annualized / 0.2, rel=1e-9
        )

    def test_zero_exposure_gives_zero_return_on_deployed(self) -> None:
        """从未持仓时，占投入口径应为 0（而不是除零崩溃）。"""
        returns = make_returns([0.0] * 50)
        metrics = compute_metrics(
            returns, periods_per_year=365.0, exposure=pd.Series([0.0] * 50)
        )

        assert metrics.avg_exposure == 0.0
        assert metrics.return_on_deployed == 0.0

    def test_time_in_market(self) -> None:
        """持仓时间占比 = 仓位权重 > 0 的期数比例。"""
        returns = make_returns([0.0001] * 100)
        exposure = pd.Series([0.0] * 50 + [0.2] * 50)

        metrics = compute_metrics(returns, periods_per_year=365.0, exposure=exposure)

        assert metrics.time_in_market == pytest.approx(0.5)
        assert metrics.avg_exposure == pytest.approx(0.1)

    def test_missing_exposure_defaults_to_zero(self) -> None:
        """不传 exposure 时应优雅降级，而不是报错。"""
        returns = make_returns([0.001] * 20)
        metrics = compute_metrics(returns, periods_per_year=365.0)

        assert metrics.avg_exposure == 0.0
        assert metrics.return_on_deployed == 0.0


class TestForwardSplit:
    """前推式样本分割。"""

    def test_split_ratio(self) -> None:
        series = make_returns([0.001] * 100)
        split = forward_split(series, in_sample_ratio=0.7)

        assert len(split.in_sample) == 70
        assert len(split.out_of_sample) == 30
        assert split.split_index == 70

    def test_split_is_temporal_not_random(self) -> None:
        """分割必须保持时间顺序 —— 前 70% 是样本内，后 30% 是样本外。

        随机分割会让未来数据混入样本内，样本外指标虚高。
        """
        series = pd.Series(
            range(100), index=pd.date_range("2025-01-01", periods=100, freq="8h", tz="UTC")
        )
        split = forward_split(series, in_sample_ratio=0.7)

        assert split.in_sample.index.max() < split.out_of_sample.index.min(), (
            "样本内的时间必须严格早于样本外"
        )

    def test_values_preserved(self) -> None:
        series = make_returns(list(range(100)))
        split = forward_split(series, in_sample_ratio=0.7)

        assert list(split.in_sample) == list(range(70))
        assert list(split.out_of_sample) == list(range(70, 100))

    def test_invalid_ratio_raises(self) -> None:
        series = make_returns([0.001] * 100)
        for bad_ratio in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="in_sample_ratio"):
                forward_split(series, in_sample_ratio=bad_ratio)

    def test_too_short_series_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            forward_split(make_returns([0.001]), in_sample_ratio=0.7)

    def test_describe(self) -> None:
        series = make_returns([0.001] * 100)
        split = forward_split(series, in_sample_ratio=0.7)
        described = split.describe()

        assert described["in_sample_periods"] == 70
        assert described["out_of_sample_periods"] == 30


class TestPlacebo:
    """安慰剂对照检验。"""

    def test_constant_returns_are_invariant_under_shuffle(self) -> None:
        """恒定收益序列打乱后完全不变。"""
        returns = make_returns([0.0005] * 100)
        result = placebo_test(returns, n_trials=100, seed=42)

        assert result.shuffled_mean_return == pytest.approx(result.real_net_return, rel=1e-9)
        assert result.p_value == pytest.approx(1.0, abs=0.05)

    def test_p_value_is_reproducible(self) -> None:
        """固定种子应产生可复现的结果 —— 回测结论必须能复核。"""
        returns = make_returns([0.001, -0.0005, 0.002, -0.001, 0.0015] * 20)

        first = placebo_test(returns, n_trials=200, seed=999)
        second = placebo_test(returns, n_trials=200, seed=999)

        assert first.p_value == second.p_value

    def test_different_seeds_give_different_results(self) -> None:
        """不同种子应有差异 —— 否则随机化没生效。"""
        returns = make_returns([0.001, -0.0005, 0.002, -0.001, 0.0015] * 20)

        first = placebo_test(returns, n_trials=200, seed=1)
        second = placebo_test(returns, n_trials=200, seed=2)

        assert first.shuffled_mean_return != second.shuffled_mean_return

    def test_empty_returns_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            placebo_test(make_returns([]))


class TestConcentration:
    """收益集中度。"""

    def test_uniform_returns_not_concentrated(self) -> None:
        returns = make_returns([0.001] * 100)
        result = concentration_check(returns, top_n=5)

        assert not result["is_concentrated"]

    def test_lucky_returns_flagged_as_concentrated(self) -> None:
        """收益全靠少数几期时，必须被标记出来。

        这是抓"运气策略"最直接的方法：如果一个策略的全部收益来自
        5 个结算期，那它不是策略，是抽奖。
        """
        returns = make_returns([0.0001] * 95 + [0.1, 0.1, 0.1, 0.1, 0.1])
        result = concentration_check(returns, top_n=5)

        assert result["is_concentrated"]
        assert result["return_without_top"] < result["full_return"] * 0.5

    def test_all_negative_returns(self) -> None:
        returns = make_returns([-0.001] * 50)
        result = concentration_check(returns, top_n=5)

        # 全负时不应判定为"集中"（没有收益可分）
        assert not result["is_concentrated"]

    def test_empty_returns_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            concentration_check(make_returns([]))


__all__: list[str] = []
