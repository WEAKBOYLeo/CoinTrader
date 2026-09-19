"""破产情景分析。

**为什么这个模块不可或缺：**

资金费套利的日常波动很小（已对冲），所以回测看起来会是一条平稳上升的曲线，
夏普很高，最大回撤很小。**这会给人虚假的安全感。**

这类策略真正的杀手不是行情波动，而是**交易所层面的尾部事件**：

- 交易所暴雷（FTX 型）：资金全锁，损失 100%
- 永续腿被强平：极端插针下空头腿被强制平仓，对冲瞬间失效
- 稳定币脱锚：USDT/USDC 抵押品贬值
- 提币暂停：赚了钱但取不出来，等于没赚

这些事件在历史回测数据里**根本不会出现**（因为样本期内没发生过，
或者发生的那次被当作异常值剔除了）。所以不能用回测来评估它们，
必须用**独立的情景分析**。

**核心结论的性质**：本模块输出的不是"预期损失"，而是
"如果发生 X，我会亏多少"。它回答的问题是**仓位该多大**，
而不是**收益该多高**。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..config import RiskConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Scenario:
    """一个破产情景的定义。"""

    name: str
    description: str
    loss_pct: float           # 该情景下损失的**总资金**比例
    probability_note: str     # 概率的定性描述（诚实：无法量化）
    mitigations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """情景分析结果。"""

    scenario: Scenario
    capital_at_risk: float
    loss_usdt: float
    remaining_usdt: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.name,
            "description": self.scenario.description,
            "loss_pct_of_capital": round(self.scenario.loss_pct, 4),
            "capital_at_risk": round(self.capital_at_risk, 2),
            "loss_usdt": round(self.loss_usdt, 2),
            "remaining_usdt": round(self.remaining_usdt, 2),
            "probability_note": self.scenario.probability_note,
            "mitigations": list(self.scenario.mitigations),
        }


# ---------------------------------------------------------------------------
# 标准情景
# ---------------------------------------------------------------------------


def standard_scenarios(
    *,
    exchange_exposure_pct: float = 1.0,
    leverage: float = 3.0,
    basis_spike_pct: float = 0.20,
    stablecoin_depeg_pct: float = 0.05,
) -> list[Scenario]:
    """构造标准破产情景集。

    Args:
        exchange_exposure_pct: 放在**单一交易所**的资金占比。这是最重要的参数：
            分散到两家交易所，交易所暴雷的损失立刻从 100% 降到 50%。
        leverage: 永续腿的杠杆倍数。
        basis_spike_pct: 极端行情下基差的不利扩张幅度。
        stablecoin_depeg_pct: 抵押稳定币的脱锚幅度。

    Returns:
        Scenario 列表，按损失比例**升序**排列（最轻的排最前）。

    Note:
        损失比例依赖传入的参数（杠杆、敞口），所以顺序不是写死的常量，
        必须在构造后排序。此前把顺序写死在字面量里，导致
        "稳定币脱锚 5%" 被排在 "基差插针 30%" 之后，报告读起来
        像是严重程度忽高忽低。
    """
    # 杠杆越高，永续腿离强平越近。3 倍杠杆下，基差反向扩张 33% 就会爆仓。
    liquidation_threshold = 1.0 / leverage
    # 基差扩张到爆仓线时，现货腿虽然赚钱但永续腿已被强平，
    # 剩余头寸变成裸多头，在后续下跌中继续亏。保守估计净损失。
    forced_liquidation_loss = min(1.0, basis_spike_pct / max(liquidation_threshold, 1e-9)) * 0.5

    scenarios = [
        Scenario(
            name="basis_spike",
            description=f"极端行情下基差不利扩张 {basis_spike_pct:.0%}，永续腿逼近强平线",
            loss_pct=forced_liquidation_loss,
            probability_note="每年都会发生数次（插针行情），是**最可能**触发的情景",
            mitigations=(
                "降低永续腿杠杆（提高维持保证金缓冲）",
                "设置基差监控告警，触及阈值自动减仓",
                "避免在资金费极端高企时满仓（那通常伴随高波动）",
            ),
        ),
        Scenario(
            name="stablecoin_depeg",
            description=f"抵押稳定币脱锚 {stablecoin_depeg_pct:.0%}（如 USDC 跌破面值）",
            loss_pct=stablecoin_depeg_pct,
            probability_note="低概率但非零（2023-03 USDC 曾跌至 0.87）",
            mitigations=(
                "优先使用 USDT 结算的合约",
                "避免把全部抵押品放在同一种稳定币",
                "脱锚时不要恐慌平仓（会实现损失）",
            ),
        ),
        Scenario(
            name="withdrawal_suspended",
            description="交易所暂停提币，资金被锁定（可能数周至永久）",
            loss_pct=exchange_exposure_pct * 0.30,
            probability_note="中低概率。币安历史上有过短暂的提币拥堵",
            mitigations=(
                "定期提走利润，不在交易所囤积超过必要资金",
                "分散到至少两家交易所",
                "关注交易所偿付能力的公开信号",
            ),
        ),
        Scenario(
            name="exchange_failure",
            description="交易所暴雷/破产，资金全部损失（FTX 型）",
            loss_pct=exchange_exposure_pct * 1.0,
            probability_note="极低概率，但**后果是毁灭性的**，必须按此定仓位",
            mitigations=(
                "单所资金上限 = 你能承受全损的金额",
                "分散到多家交易所（直接减半风险）",
                "自托管现货腿（用去中心化永续对冲）",
            ),
        ),
    ]

    # 按损失比例升序。顺序依赖参数，所以必须在这里排。
    scenarios.sort(key=lambda s: s.loss_pct)
    return scenarios


def analyze_scenarios(
    capital: float,
    scenarios: list[Scenario] | None = None,
    **scenario_kwargs: Any,
) -> list[ScenarioResult]:
    """对给定资金规模跑情景分析。

    Args:
        capital: 配置到该策略的总资金（USDT）。
        scenarios: 自定义情景集。为 None 时用 ``standard_scenarios()``。
        **scenario_kwargs: 透传给 ``standard_scenarios()``。

    Returns:
        ScenarioResult 列表。
    """
    if capital <= 0:
        raise ValueError(f"capital 必须为正，当前 {capital}")

    scenarios = scenarios if scenarios is not None else standard_scenarios(**scenario_kwargs)

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        loss = capital * scenario.loss_pct
        results.append(
            ScenarioResult(
                scenario=scenario,
                capital_at_risk=capital,
                loss_usdt=loss,
                remaining_usdt=capital - loss,
            )
        )
    return results


def survival_analysis(
    capital: float,
    *,
    expected_annual_return: float,
    scenarios: list[Scenario] | None = None,
    **scenario_kwargs: Any,
) -> dict[str, Any]:
    """回答「一个毁灭性事件要多久才能恢复」。

    这是把收益和风险放在同一把尺子上的关键计算。

    若预期年化 20%，而单次交易所暴雷会亏掉 100%，那么需要**永远**才能恢复
    （因为本金归零，复利归零）。若分散到两家交易所，损失 50%，
    则需 ``ln(2)/ln(1.2) ≈ 3.8`` 年才能回本 —— 这个数字才让人真正理解
    "分散"值多少钱。

    Returns:
        含每个情景恢复年数的字典，以及一句结论性判断。
    """
    if expected_annual_return <= 0:
        recovery_years: dict[str, Any] = {s.name: None for s in (scenarios or standard_scenarios())}
        verdict = "预期收益为负，任何损失都无法通过策略自身恢复"
        return {"recovery_years": recovery_years, "verdict": verdict}

    import math

    results = analyze_scenarios(capital, scenarios, **scenario_kwargs)
    recovery: dict[str, Any] = {}

    for result in results:
        if result.scenario.loss_pct >= 1.0:
            recovery[result.scenario.name] = float("inf")  # 本金归零，无法恢复
        elif result.scenario.loss_pct <= 0.0:
            recovery[result.scenario.name] = 0.0
        else:
            remaining = result.remaining_usdt
            # 从剩余资金恢复到原始资金所需的年数
            growth_needed = capital / remaining
            recovery[result.scenario.name] = math.log(growth_needed) / math.log(
                1.0 + expected_annual_return
            )

    worst_recoverable = max(
        (v for v in recovery.values() if v != float("inf")), default=float("inf")
    )

    if any(v == float("inf") for v in recovery.values()):
        verdict = (
            "存在无法恢复的情景（本金归零）。"
            "必须降低单所集中度，或把仓位规模缩小到「全损也能承受」的水平。"
        )
    elif worst_recoverable > 3.0:
        verdict = (
            f"最坏可恢复情景需要 {worst_recoverable:.1f} 年回本，过长。"
            "建议降低单所敞口或提高分散度。"
        )
    else:
        verdict = f"最坏可恢复情景需要 {worst_recoverable:.1f} 年回本，处于可接受范围。"

    return {
        "recovery_years": recovery,
        "worst_recoverable_years": (
            None if worst_recoverable == float("inf") else round(worst_recoverable, 2)
        ),
        "verdict": verdict,
    }


def position_sizing_advice(
    total_capital: float,
    risk_config: RiskConfig,
    *,
    max_loss_tolerance_pct: float = 0.20,
    exchange_failure_loss_pct: float = 1.0,
) -> dict[str, Any]:
    """按「全损可承受」原则给出仓位建议。

    原则：**单所敞口不应超过你能承受全损的金额。**

    这和按收益率定仓位是两种完全不同的思路。按收益率定仓位，你会
    在资金费高的时候加仓到最大 —— 而那恰恰是波动最大、最容易出事的时刻。

    Args:
        total_capital: 你打算投入这个策略的总资金。
        risk_config: 风控配置。
        max_loss_tolerance_pct: 你能承受的最大回撤（占总资金）。
        exchange_failure_loss_pct: 单所暴雷的损失比例。

    Returns:
        含建议的字典。``warnings`` 是**严重**警告（配置与风险承受能力冲突），
        ``notes`` 是**提示性**建议（例如"剩余资金应分散"）。

        区分两者很重要：如果提示性建议也放进 warnings，
        用户会对所有警告脱敏，真正危险的那条反而被忽略。
    """
    max_acceptable_loss = total_capital * max_loss_tolerance_pct
    # 单所最多能放多少钱：使得全损时不超过可承受损失
    max_per_exchange = max_acceptable_loss / exchange_failure_loss_pct

    configured_single_limit = risk_config.max_exposure_per_symbol
    configured_total_limit = risk_config.max_total_exposure

    warnings: list[str] = []
    notes: list[str] = []

    if configured_total_limit > max_per_exchange:
        warnings.append(
            f"配置的总敞口上限 {configured_total_limit:.0f} USDT 超过按全损可承受算出的 "
            f"{max_per_exchange:.0f} USDT。建议下调。"
        )
    if configured_total_limit > total_capital:
        warnings.append(
            f"总敞口上限 {configured_total_limit:.0f} USDT 超过总资金 {total_capital:.0f} USDT，"
            "存在隐性杠杆（合约保证金+现货同时占用）。"
        )

    # 剩余资金占比足够大时才提示分散 —— 否则是噪音
    remaining = total_capital - max_per_exchange
    if remaining > total_capital * 0.2:
        notes.append(
            f"建议单所资金不超过 {max_per_exchange:.0f} USDT，"
            f"剩余 {remaining:.0f} USDT 应分散到其他交易所或留作现金。"
        )

    return {
        "total_capital": total_capital,
        "max_acceptable_loss": max_acceptable_loss,
        "recommended_max_per_exchange": round(max_per_exchange, 2),
        "configured_total_exposure": configured_total_limit,
        "configured_per_symbol_limit": configured_single_limit,
        "warnings": warnings,
        "notes": notes,
    }


__all__ = [
    "Scenario",
    "ScenarioResult",
    "analyze_scenarios",
    "position_sizing_advice",
    "standard_scenarios",
    "survival_analysis",
]
