"""破产情景分析测试。

这组测试的意义：**仓位规模由这里决定，而不是由收益率决定。**

一个只按收益率定仓位的策略，会在资金费最高的时候（通常也是波动最大的时候）
加仓到最大。而这里的原则是：单所敞口不应超过你能承受全损的金额。

数学必须精确 —— 这直接对应真实的资金损失。
"""

from __future__ import annotations

import math

import pytest

from cointrader.backtest.scenarios import (
    Scenario,
    analyze_scenarios,
    position_sizing_advice,
    standard_scenarios,
    survival_analysis,
)
from cointrader.config import RiskConfig


class TestStandardScenarios:
    """标准情景集。"""

    def test_returns_all_four_scenarios(self) -> None:
        scenarios = standard_scenarios()

        names = {s.name for s in scenarios}
        assert names == {
            "basis_spike",
            "stablecoin_depeg",
            "withdrawal_suspended",
            "exchange_failure",
        }

    def test_scenarios_ordered_by_severity(self) -> None:
        """情景应按损失递增排列 —— 便于阅读和理解最坏情况。"""
        scenarios = standard_scenarios()
        losses = [s.loss_pct for s in scenarios]

        assert losses == sorted(losses), f"情景未按严重程度排序: {losses}"

    def test_exchange_failure_is_total_loss_at_full_exposure(self) -> None:
        """100% 单所敞口下，交易所暴雷 = 全部损失。"""
        scenarios = {s.name: s for s in standard_scenarios(exchange_exposure_pct=1.0)}

        assert scenarios["exchange_failure"].loss_pct == pytest.approx(1.0)

    def test_diversification_halves_exchange_failure_loss(self) -> None:
        """**分散的价值**：敞口从 100% 降到 50%，暴雷损失直接减半。

        这是本模块最重要的一个数字 —— 它把"分散投资"从口号
        变成了可量化的收益。
        """
        full = {s.name: s for s in standard_scenarios(exchange_exposure_pct=1.0)}
        half = {s.name: s for s in standard_scenarios(exchange_exposure_pct=0.5)}

        assert half["exchange_failure"].loss_pct == pytest.approx(0.5)
        assert half["exchange_failure"].loss_pct == pytest.approx(
            full["exchange_failure"].loss_pct / 2
        )

    def test_higher_leverage_worsens_liquidation_scenario(self) -> None:
        """杠杆越高，基差小幅扩张就可能爆仓。

        手工推算 3 倍杠杆: 强平线 = 1/3 ≈ 33%，基差扩张 20% 时
        损失 = (20%/33%) × 0.5 ≈ 30%
        """
        low_lev = {s.name: s for s in standard_scenarios(leverage=2.0)}
        high_lev = {s.name: s for s in standard_scenarios(leverage=10.0)}

        assert high_lev["basis_spike"].loss_pct > low_lev["basis_spike"].loss_pct

    def test_every_scenario_has_mitigations(self) -> None:
        """每个情景都必须给出可执行的缓解措施。

        只报告风险不给对策是没用的 —— 用户需要知道**能做什么**。
        """
        for scenario in standard_scenarios():
            assert scenario.mitigations, f"{scenario.name} 缺少缓解措施"
            assert scenario.probability_note, f"{scenario.name} 缺少概率说明"
            assert scenario.description, f"{scenario.name} 缺少描述"


class TestAnalyzeScenarios:
    """情景分析计算。"""

    def test_loss_amounts(self) -> None:
        """手工验算: 10000 USDT × 100% = 10000 损失"""
        results = analyze_scenarios(10_000.0)
        by_name = {r.scenario.name: r for r in results}

        failure = by_name["exchange_failure"]
        assert failure.loss_usdt == pytest.approx(10_000.0)
        assert failure.remaining_usdt == pytest.approx(0.0)

    def test_partial_loss_leaves_remainder(self) -> None:
        """部分损失后剩余资金计算正确，且损失 + 剩余 = 本金。"""
        results = analyze_scenarios(10_000.0, exchange_exposure_pct=0.5)

        for result in results:
            assert result.loss_usdt + result.remaining_usdt == pytest.approx(
                result.capital_at_risk
            )

    def test_rejects_nonpositive_capital(self) -> None:
        with pytest.raises(ValueError, match="capital"):
            analyze_scenarios(0.0)

        with pytest.raises(ValueError, match="capital"):
            analyze_scenarios(-100.0)

    def test_custom_scenarios(self) -> None:
        custom = [
            Scenario(
                name="custom",
                description="测试情景",
                loss_pct=0.25,
                probability_note="仅用于测试",
            )
        ]
        results = analyze_scenarios(1000.0, scenarios=custom)

        assert len(results) == 1
        assert results[0].loss_usdt == pytest.approx(250.0)
        assert results[0].remaining_usdt == pytest.approx(750.0)

    def test_as_dict_roundtrip(self) -> None:
        results = analyze_scenarios(10_000.0)
        payload = results[0].as_dict()

        assert "scenario" in payload
        assert "loss_usdt" in payload
        assert "mitigations" in payload


class TestSurvivalAnalysis:
    """恢复时间分析。"""

    def test_total_loss_is_unrecoverable(self) -> None:
        """**本金归零后，任何收益率都无法恢复。**

        这不是"需要很久"，而是"永远"。这个区别在面对
        "交易所暴雷"这类情景时至关重要。
        """
        result = survival_analysis(
            10_000.0,
            expected_annual_return=0.20,
            exchange_exposure_pct=1.0,
        )

        assert result["recovery_years"]["exchange_failure"] == float("inf")
        assert "无法恢复" in result["verdict"] or "本金归零" in result["verdict"]

    def test_half_loss_recovery_time(self) -> None:
        """手工验算: 亏 50%，年化 20%
        需要 ln(2)/ln(1.2) ≈ 3.80 年
        """
        result = survival_analysis(
            10_000.0,
            expected_annual_return=0.20,
            exchange_exposure_pct=0.5,
        )

        expected = math.log(2) / math.log(1.2)
        assert result["recovery_years"]["exchange_failure"] == pytest.approx(expected, rel=1e-6)

    def test_higher_return_shortens_recovery(self) -> None:
        """收益率越高，恢复越快（这对复利是显然的）。"""
        low = survival_analysis(10_000.0, expected_annual_return=0.10, exchange_exposure_pct=0.5)
        high = survival_analysis(10_000.0, expected_annual_return=0.40, exchange_exposure_pct=0.5)

        assert (
            high["recovery_years"]["exchange_failure"]
            < low["recovery_years"]["exchange_failure"]
        )

    def test_negative_return_never_recovers(self) -> None:
        """负收益率下任何损失都无法通过策略自身恢复。"""
        result = survival_analysis(10_000.0, expected_annual_return=-0.05)

        assert "预期收益为负" in result["verdict"]
        assert all(y is None for y in result["recovery_years"].values())

    def test_worst_recoverable_years(self) -> None:
        result = survival_analysis(
            10_000.0,
            expected_annual_return=0.20,
            exchange_exposure_pct=0.5,
        )

        # 最坏的可恢复情景就是 50% 损失
        assert result["worst_recoverable_years"] == pytest.approx(
            math.log(2) / math.log(1.2), rel=1e-3
        )

    def test_long_recovery_triggers_warning(self) -> None:
        """恢复时间过长时应给出明确警告。

        年化 5% 下亏 50% 需要 ln(2)/ln(1.05) ≈ 14.2 年 —— 不可接受。
        """
        result = survival_analysis(
            10_000.0,
            expected_annual_return=0.05,
            exchange_exposure_pct=0.5,
        )

        assert "过长" in result["verdict"]
        assert "降低" in result["verdict"] or "分散" in result["verdict"]


class TestPositionSizingAdvice:
    """仓位建议 —— 按「全损可承受」原则。"""

    def test_recommended_exposure_from_loss_tolerance(self) -> None:
        """手工验算: 总资金 10000，可承受 20% 回撤 → 最大可承受损失 2000
        单所暴雷损失 100% → 建议单所资金 <= 2000
        """
        advice = position_sizing_advice(
            10_000.0, RiskConfig(), max_loss_tolerance_pct=0.20
        )

        assert advice["max_acceptable_loss"] == pytest.approx(2000.0)
        assert advice["recommended_max_per_exchange"] == pytest.approx(2000.0)

    def test_tighter_tolerance_reduces_recommendation(self) -> None:
        """可承受回撤越小，建议仓位越小。"""
        loose = position_sizing_advice(10_000.0, RiskConfig(), max_loss_tolerance_pct=0.30)
        tight = position_sizing_advice(10_000.0, RiskConfig(), max_loss_tolerance_pct=0.10)

        assert (
            tight["recommended_max_per_exchange"]
            < loose["recommended_max_per_exchange"]
        )

    def test_warns_when_configured_limit_exceeds_recommendation(self) -> None:
        """配置的敞口上限超过建议值时，必须给出警告。

        这是把"配置"和"风险承受能力"对齐的关键检查 ——
        默认配置可能远大于个人能承受的损失。
        """
        # 默认 RiskConfig 总敞口 2000，容忍度 5% → 建议 500
        advice = position_sizing_advice(
            10_000.0, RiskConfig(max_total_exposure=2000.0), max_loss_tolerance_pct=0.05
        )

        assert advice["warnings"], "配置上限远超建议值时应给出警告"
        assert any("超过" in w for w in advice["warnings"])

    def test_warns_when_total_exposure_exceeds_capital(self) -> None:
        """总敞口上限超过总资金时，存在隐性杠杆，必须警告。

        现货腿 + 合约保证金同时占用资金，敞口名义额可以超过本金 ——
        这时"总敞口"这个词的语义已经变了。
        """
        advice = position_sizing_advice(
            500.0, RiskConfig(max_total_exposure=2000.0), max_loss_tolerance_pct=0.2
        )

        assert any("总资金" in w for w in advice["warnings"])

    def test_suggests_diversification(self) -> None:
        """建议值明显低于总资金时，应提示分散。

        「分散」是 notes（提示），不是 warnings（严重警告）——
        这两者必须区分，否则用户会对所有警告脱敏。
        """
        advice = position_sizing_advice(
            10_000.0, RiskConfig(), max_loss_tolerance_pct=0.20
        )

        assert any("分散" in n for n in advice["notes"])

    def test_no_diversification_hint_when_fully_deployed(self) -> None:
        """建议值接近总资金时，不应提示分散（剩余资金太少，没意义）。"""
        advice = position_sizing_advice(
            10_000.0, RiskConfig(), max_loss_tolerance_pct=1.0
        )

        # 容忍度 100% → 建议单所 10000 = 全部资金，没有剩余可分
        assert advice["recommended_max_per_exchange"] == pytest.approx(10_000.0)
        assert not any("分散" in n for n in advice["notes"])

    def test_diversification_is_note_not_warning(self) -> None:
        """分散提示必须归入 notes 而非 warnings。

        这条测试锁死语义：warnings 只放"配置与风险承受能力冲突"，
        notes 放"建议性做法"。混淆会让 warnings 失去信号价值。
        """
        advice = position_sizing_advice(
            10_000.0, RiskConfig(), max_loss_tolerance_pct=0.20
        )

        assert not any("分散" in w for w in advice["warnings"]), (
            "分散提示不应出现在 warnings 中"
        )
        assert any("分散" in n for n in advice["notes"])

    def test_no_warnings_when_conservatively_configured(self) -> None:
        """配置保守时不应产生**严重**警告。"""
        conservative = RiskConfig(
            max_notional_per_order=100.0,
            max_exposure_per_symbol=300.0,
            max_total_exposure=1000.0,
        )
        advice = position_sizing_advice(
            10_000.0, conservative, max_loss_tolerance_pct=0.50
        )

        assert advice["recommended_max_per_exchange"] == pytest.approx(5000.0)
        assert not advice["warnings"], f"保守配置不应有严重警告: {advice['warnings']}"


__all__: list[str] = []
