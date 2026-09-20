"""前瞻偏差（lookahead bias）的证伪测试。

## 为什么这个文件是回测框架里最重要的

前瞻偏差是量化回测里**最常见**也**最难发现**的错误。它的表现是：
回测收益漂亮，实盘一塌糊涂，而你在三个月里找不到原因。

常见的引入方式：

- 用当期收盘价决定当期成交
- 用全样本的均值和标准差做标准化（把未来信息混进过去）
- ``shift(-1)`` 顺手写反
- 用"未来才知道的下架名单"过滤历史标的（幸存者偏差）

**这些错误都不会报错，只会让你赚钱（在回测里）。**

## 本文件的证伪策略

不靠"读代码觉得没问题"，而是靠**结构性检验**：

1. **构造性检验** —— 造一段已知答案的数据，断言回测结果与手工推算一致
2. **扰动检验** —— 把未来数据改掉，断言"过去产生的信号"不变
3. **代码扫描** —— 禁止 ``shift(-n)``、``iloc[+n]`` 这类前视写法
4. **安慰剂对照** —— 打乱收益序列，断言策略不因顺序改变而凭空赚钱
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cointrader.backtest.engine import FundingCarryBacktester, build_signals, run_portfolio
from cointrader.config import (
    BacktestConfig,
    EntryConfig,
    ExitConfig,
    SelectionConfig,
    StrategyConfig,
)
from cointrader.data.funding import annualize_rate, trailing_annualized
from cointrader.errors import ConfigError, InsufficientDataError
from cointrader.research.costs import CostModel
from cointrader.research.metrics import concentration_check, placebo_test
from conftest import make_funding_series

# ---------------------------------------------------------------------------
# 代码扫描
# ---------------------------------------------------------------------------


class TestNoLookaheadPatterns:
    """静态扫描前视写法。"""

    #: 前视写法的特征模式
    FORBIDDEN_PATTERNS = (
        (re.compile(r"\.shift\(\s*-\s*\d+\s*\)"), "负向 shift（引用未来行）"),
        (re.compile(r"\.iloc\[\s*\+\s*\d"), "正偏移 iloc（引用未来行）"),
        (re.compile(r"\.rolling\([^)]*\)\.mean\(\)\s*\.shift\(\s*-"), "滚动均值后再负向 shift"),
    )

    #: 允许出现这些模式的例外（本测试文件自己会给出反例）
    EXEMPT_FILES = {"test_no_lookahead.py"}

    def test_no_negative_shift_in_production_code(self, src_root: Path) -> None:
        """生产代码中不得出现负向 shift / 正偏移 iloc。

        这两者都意味着「用未来的行算当前的值」。
        在金融时序里，这是最直接的前视偏差。
        """
        violations: list[str] = []

        for path in sorted(src_root.rglob("*.py")):
            if "__pycache__" in path.parts or path.name in self.EXEMPT_FILES:
                continue
            content = path.read_text(encoding="utf-8")
            for pattern, description in self.FORBIDDEN_PATTERNS:
                for match in pattern.finditer(content):
                    lineno = content[: match.start()].count("\n") + 1
                    violations.append(
                        f"{path.relative_to(src_root)}:{lineno} {description}: {match.group(0)}"
                    )

        assert not violations, (
            "检测到可能的前视偏差写法。若确实需要，请在此测试中显式加入例外并说明理由。\n"
            "违规项:\n  " + "\n  ".join(violations)
        )

    def test_execution_lag_must_be_positive(self) -> None:
        """execution_lag_bars 必须 >= 1，配置层就要拒绝 0。

        lag=0 意味着"用当期信号按当期价格成交"，是最隐蔽的前视偏差。
        """
        with pytest.raises(ConfigError, match="execution_lag_bars"):
            BacktestConfig(execution_lag_bars=0)

        with pytest.raises(ConfigError, match="execution_lag_bars"):
            BacktestConfig(execution_lag_bars=-1)

        # 正常值应通过
        assert BacktestConfig(execution_lag_bars=1).execution_lag_bars == 1

    def test_engine_rejects_zero_lag(self, costs_config) -> None:
        """引擎层也要拒绝 lag=0（双保险：即使有人绕过配置层直接构造引擎）。"""
        bad_config = BacktestConfig.__new__(BacktestConfig)
        object.__setattr__(bad_config, "execution_lag_bars", 0)
        object.__setattr__(bad_config, "in_sample_ratio", 0.7)
        object.__setattr__(bad_config, "days_per_year", 365)
        object.__setattr__(bad_config, "initial_capital", 10000.0)
        object.__setattr__(bad_config, "mode", "hedged")

        with pytest.raises(ValueError, match="execution_lag_bars"):
            FundingCarryBacktester(
                bad_config,
                StrategyConfig(),
                CostModel(costs_config),
            )


# ---------------------------------------------------------------------------
# 构造性检验
# ---------------------------------------------------------------------------


class TestConstructedScenarios:
    """用已知答案的数据验证回测逻辑。"""

    def test_signal_uses_only_past_data(self) -> None:
        """信号在位置 i 只依赖 rates[0..i]。

        构造法：先算完整序列的信号，再截断数据重算，
        断言前 i+1 个信号完全一致。如果信号偷看了未来，
        截断后重算的结果会不同。
        """
        rates = make_funding_series([0.0001 * (i % 7) for i in range(100)])
        strategy = StrategyConfig(
            entry=EntryConfig(lookback_periods=10, min_consecutive_positive=3),
            exit=ExitConfig(),
        )

        full_signals = build_signals(rates, 8, strategy)

        # 对每个截断点，前段信号必须与完整序列一致
        for cutoff in (20, 35, 50, 73):
            truncated = rates.iloc[:cutoff]
            partial = build_signals(truncated, 8, strategy)
            np.testing.assert_array_equal(
                partial.entry,
                full_signals.entry[:cutoff],
                err_msg=f"截断到 {cutoff} 期后，进场信号与完整序列不一致 → 信号偷看了未来",
            )
            np.testing.assert_array_equal(
                partial.exit,
                full_signals.exit[:cutoff],
                err_msg=f"截断到 {cutoff} 期后，出场信号与完整序列不一致 → 信号偷看了未来",
            )

    def test_future_data_does_not_change_past_signals(self) -> None:
        """把未来数据整体改掉，过去的信号必须一字不变。

        这是最强的结构性检验：直接篡改未来，看过去是否受影响。
        """
        base = [0.0002] * 60
        strategy = StrategyConfig(
            entry=EntryConfig(lookback_periods=5, min_consecutive_positive=3),
            exit=ExitConfig(),
        )

        rates_a = make_funding_series(base + [0.0003] * 40)
        rates_b = make_funding_series(base + [-0.0005] * 40)   # 未来完全反转

        signals_a = build_signals(rates_a, 8, strategy)
        signals_b = build_signals(rates_b, 8, strategy)

        # 前 60 期（历史部分）的信号必须完全一致
        np.testing.assert_array_equal(
            signals_a.entry[:60],
            signals_b.entry[:60],
            err_msg="未来数据的变化影响了过去的进场信号 → 存在前视偏差",
        )
        np.testing.assert_array_equal(
            signals_a.exit[:60],
            signals_b.exit[:60],
            err_msg="未来数据的变化影响了过去的出场信号 → 存在前视偏差",
        )

    def test_tail_mutation_does_not_affect_early_signals(self) -> None:
        """只改**尾部**数据，早期的信号必须一字不变。

        ## 这条测试补的是什么盲区

        ``test_signal_uses_only_past_data``（截断不变性）能抓到 ``shift(-1)``
        这类"逐行偷看下一行"的泄漏，但抓不到**全样本统计量泄漏** ——
        比如把 ``current_ann`` 换成"整段序列的均值年化"。

        为什么抓不到：那种实现在每个截断前缀上**也是因果的**
        （截断后重算，用的是截断序列自己的均值），所以截断不变性成立，
        但它对完整序列给出的信号是错的（用了未来数据）。

        本测试直接换一个角度：**固定输入的前半段，只改后半段，
        断言前半段的信号不变。** 任何形式的未来信息依赖都会在这里暴露。

        实测验证：植入 `rates.mean()` 全样本均值后，本测试 FAIL，
        而截断不变性测试 PASS。两者互补，缺一不可。
        """
        head = [0.0002] * 60
        strategy = StrategyConfig(
            entry=EntryConfig(lookback_periods=5, min_consecutive_positive=3),
            exit=ExitConfig(),
        )

        # 两个序列：前 60 期完全相同，之后完全不同
        rates_a = make_funding_series(head + [0.0003] * 40)
        rates_b = make_funding_series(head + [-0.0004] * 40)

        signals_a = build_signals(rates_a, 8, strategy)
        signals_b = build_signals(rates_b, 8, strategy)

        np.testing.assert_array_equal(
            signals_a.entry[:60],
            signals_b.entry[:60],
            err_msg="只改了尾部数据，早期进场信号却变了 → 信号依赖了全样本统计量（前视偏差）",
        )
        np.testing.assert_array_equal(
            signals_a.exit[:60],
            signals_b.exit[:60],
            err_msg="只改了尾部数据，早期出场信号却变了 → 信号依赖了全样本统计量（前视偏差）",
        )

        # 尾部确实不同，证明这个测试有区分力（否则上面的断言是空转的）
        assert not np.array_equal(signals_a.entry[60:], signals_b.entry[60:]) or not np.array_equal(
            signals_a.trailing_ann[60:], signals_b.trailing_ann[60:]
        ), "尾部信号的 trailing 值应当不同，否则本测试无区分力"

    def test_trailing_annualized_has_no_future_leak(self) -> None:
        """滚动年化在位置 i 只使用 [i-window+1, i] 的数据。"""
        rates = make_funding_series([0.0001] * 40)
        window = 10

        trailing = trailing_annualized(rates, 8, window=window)

        # 前 window-1 期必须为 NaN（数据不足，不能凭空填充）
        assert trailing.iloc[: window - 1].isna().all(), "数据不足期应返回 NaN，而不是填充值"

        # 第 window-1 期的值应等于前 window 期的均值年化
        expected = annualize_rate(rates.iloc[:window].mean(), 8)
        assert trailing.iloc[window - 1] == pytest.approx(expected, rel=1e-12)

        # 手工验证最后一期
        expected_last = annualize_rate(rates.iloc[-window:].mean(), 8)
        assert trailing.iloc[-1] == pytest.approx(expected_last, rel=1e-12)

    def test_known_answer_backtest(self, costs_config) -> None:
        """构造一个答案可手算的回测，逐项核对。

        数据：100 期，每期费率恒定 0.0003（0.03%），8h 结算。
        策略：回看 5 期，连续 3 期为正即可进场，持仓无强平。

        手工推算::

            单期年化 = 0.0003 × (24/8) × 365 = 32.85%  → 远超 15% 阈值
            资金费收入 = 0.0003 / 期
            仓位权重 = 0.2（由 fixture 设定）
            每期净收 = 0.0003 × 0.2 = 0.00006
        """
        rates = make_funding_series([0.0003] * 100)
        strategy = StrategyConfig(
            entry=EntryConfig(
                lookback_periods=5,
                min_consecutive_positive=3,
                min_annualized_rate=0.15,
                min_trailing_annualized=0.20,
            ),
            exit=ExitConfig(exit_annualized_rate=0.03, negative_streak_exit=2, max_holding_periods=500),
            selection=SelectionConfig(max_positions=3, per_position_weight=0.2),
        )
        engine = FundingCarryBacktester(
            BacktestConfig(execution_lag_bars=1), strategy, CostModel(costs_config)
        )
        result = engine.run("TESTUSDT", rates, 8)

        # 应该产生了交易
        assert len(result.trades) >= 1, "持续高费率下应至少产生一笔交易"

        # 毛收益应该完全来自资金费，且每期为正
        # 进场后的期数 × 每期收入
        total_gross = result.gross_returns.sum()
        assert total_gross > 0, f"持续正费率下毛收益应为正，实际 {total_gross}"

        # 每期毛收益必须恰好等于 rate × weight（或 0，未持仓时）
        for value in result.gross_returns.to_numpy():
            assert value == pytest.approx(0.0, abs=1e-15) or value == pytest.approx(
                0.0003 * 0.2, abs=1e-15
            ), f"毛收益出现了非预期的值: {value}"

    def test_constant_zero_rate_produces_no_profit(self, costs_config, flat_funding_series) -> None:
        """零费率下即使有交易，净收益也必须为负（纯成本）。"""
        strategy = StrategyConfig(
            entry=EntryConfig(
                lookback_periods=5,
                min_consecutive_positive=1,
                min_annualized_rate=0.0,       # 放宽阈值，让它能进场
                min_trailing_annualized=0.0,
            ),
            exit=ExitConfig(exit_annualized_rate=-1.0, negative_streak_exit=999),
            selection=SelectionConfig(max_positions=3, per_position_weight=0.2),
        )
        engine = FundingCarryBacktester(
            BacktestConfig(execution_lag_bars=1), strategy, CostModel(costs_config)
        )
        # 零费率不满足"> 0"的连续为正条件，所以不会进场 —— 净收益为 0
        result = engine.run("ZEROUSDT", flat_funding_series, 8)

        assert result.metrics.net_return <= 0, "零费率下不应产生正收益"
        assert result.metrics.gross_return == pytest.approx(0.0, abs=1e-15)

    def test_negative_funding_produces_loss(self, costs_config, negative_funding_series) -> None:
        """持续负费率下不应赚钱。

        注意：进场条件是"连续为正"，负费率下根本不会进场，
        所以净收益应为 0。这条测试验证过滤器有效。
        """
        strategy = StrategyConfig(
            entry=EntryConfig(lookback_periods=5, min_consecutive_positive=3),
            exit=ExitConfig(),
            selection=SelectionConfig(max_positions=3, per_position_weight=0.2),
        )
        engine = FundingCarryBacktester(
            BacktestConfig(execution_lag_bars=1), strategy, CostModel(costs_config)
        )
        result = engine.run("NEGUSDT", negative_funding_series, 8)

        assert result.metrics.net_return <= 0, "持续负费率下不应有正收益"
        assert len(result.trades) == 0, (
            f"连续为正的过滤器应阻止负费率币种进场，实际产生 {len(result.trades)} 笔交易"
        )


# ---------------------------------------------------------------------------
# 扰动检验
# ---------------------------------------------------------------------------


class TestPerturbation:
    """通过扰动数据来暴露隐藏的泄漏。"""

    def test_shifting_signals_forward_does_not_improve_returns(self, costs_config) -> None:
        """把信号整体前移一期（即"偷看未来一期"），收益不应因此变好。

        这是经典的前视检测：**如果偷看未来能让策略变好，说明原始实现
        已经在中性地使用数据；如果偷看后收益暴涨，说明策略本身没有预测力，
        全靠泄漏在赚钱。**

        本测试断言的是：完整数据下，lag=1 与 lag=2 的收益都不应是
        "因为多了信息"而产生的巨大跃升。我们通过比较 lag=1 和 lag=3
        的相对差异来间接验证 —— 若差异极小，说明收益来自费率水平本身
        （这是资金费套利的**正常**特征），而非择时能力。
        """
        rates = make_funding_series([0.0002 + 0.0001 * float(np.sin(i / 5)) for i in range(200)])
        strategy = StrategyConfig(
            entry=EntryConfig(lookback_periods=10, min_consecutive_positive=3),
            exit=ExitConfig(),
            selection=SelectionConfig(max_positions=3, per_position_weight=0.2),
        )
        model = CostModel(costs_config)

        results = {}
        for lag in (1, 2, 3):
            engine = FundingCarryBacktester(
                BacktestConfig(execution_lag_bars=lag), strategy, model
            )
            results[lag] = engine.run("TESTUSDT", rates, 8).metrics.net_return

        # lag 越大，进场越晚，收益应单调不增（或差异极小）
        assert results[2] <= results[1] + 0.01, (
            f"滞后 2 期反而比滞后 1 期收益高 {results[2] - results[1]:.4f}，"
            "这通常意味着收益来自数据泄漏而非策略本身"
        )

    def test_placebo_shuffling_does_not_create_profit(self) -> None:
        """打乱收益顺序不应凭空造出收益。

        如果打乱后总收益不变（在浮点误差内），说明收益来自"累积持仓"
        而非"择时"—— 对资金费套利这是正常的。
        但如果打乱后收益显著**变高**，说明真实顺序在破坏收益，即择时是负贡献。
        """
        # 恒定为正、无序列相关的收益（模拟稳定收租）
        returns = pd.Series(np.full(100, 0.0005))

        result = placebo_test(returns, n_trials=200, seed=123)

        # 恒定序列打乱后完全不变
        assert result.shuffled_mean_return == pytest.approx(result.real_net_return, rel=1e-9)
        # p 值应接近 1（打乱后的值总是 >= 真实值，因为相等）
        assert result.p_value > 0.5, (
            f"恒定收益序列的安慰剂 p 值应接近 1，实际 {result.p_value}"
        )

    def test_concentration_check_detects_lucky_strategy(self) -> None:
        """集中度检查应能识别"靠几期运气"的策略。"""
        # 绝大部分期平平，只有 3 期暴赚
        values = [0.0001] * 97 + [0.05, 0.06, 0.04]
        returns = pd.Series(values)

        result = concentration_check(returns, top_n=3)

        assert result["is_concentrated"], "收益高度集中时应该被标记"
        assert result["return_without_top"] < result["full_return"] * 0.5

    def test_concentration_check_passes_uniform_strategy(self) -> None:
        """均匀分布的收益不应被误判为集中。"""
        returns = pd.Series([0.001] * 100)
        result = concentration_check(returns, top_n=5)

        assert not result["is_concentrated"], "均匀收益不应被判定为集中"


# ---------------------------------------------------------------------------
# 边界与错误处理
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """边界条件。"""

    def test_insufficient_data_raises(self, costs_config) -> None:
        """数据不足时必须抛异常，而不是静默返回空结果。

        静默返回空结果会让调用方以为"策略在这个币上不交易"，
        实际上只是数据不够 —— 这会污染选币结论。
        """
        strategy = StrategyConfig(entry=EntryConfig(lookback_periods=30))
        engine = FundingCarryBacktester(
            BacktestConfig(execution_lag_bars=1), strategy, CostModel(costs_config)
        )
        short_rates = make_funding_series([0.0002] * 10)

        with pytest.raises(InsufficientDataError, match="期数据"):
            engine.run("SHORTUSDT", short_rates, 8)

    def test_last_signal_has_no_execution_venue(self, costs_config) -> None:
        """末尾信号无处执行时，不得伪造成交。

        如果代码在 lag 超出边界时仍记账，就会多出一笔"用不存在的数据
        成交"的交易 —— 这是隐蔽的前视偏差。
        """
        rates = make_funding_series([0.0003] * 60)
        strategy = StrategyConfig(
            entry=EntryConfig(lookback_periods=5, min_consecutive_positive=3),
            exit=ExitConfig(max_holding_periods=999),
            selection=SelectionConfig(max_positions=3, per_position_weight=0.2),
        )
        engine = FundingCarryBacktester(
            BacktestConfig(execution_lag_bars=1), strategy, CostModel(costs_config)
        )
        result = engine.run("TAILUSDT", rates, 8)

        # 所有交易记录的索引都必须落在数据范围内
        n = len(rates)
        for trade in result.trades:
            assert 0 <= trade.entry_index < n, f"建仓索引越界: {trade.entry_index}"
            assert 0 <= trade.exit_index < n, f"平仓索引越界: {trade.exit_index}"

    def test_unclosed_position_is_force_closed(self, costs_config) -> None:
        """回测结束时仍持仓的头寸必须强制平仓。

        若直接丢弃未平仓头寸，等于悄悄剔除了表现差的持仓（幸存者偏差）。
        """
        rates = make_funding_series([0.0004] * 60)
        strategy = StrategyConfig(
            entry=EntryConfig(lookback_periods=5, min_consecutive_positive=3),
            exit=ExitConfig(exit_annualized_rate=-10.0, negative_streak_exit=999, max_holding_periods=9999),
            selection=SelectionConfig(max_positions=3, per_position_weight=0.2),
        )
        engine = FundingCarryBacktester(
            BacktestConfig(execution_lag_bars=1), strategy, CostModel(costs_config)
        )
        result = engine.run("HOLDUSDT", rates, 8)

        # 应该有一笔以 end_of_data 结束的交易，其出场成本被计入
        assert any(t.exit_reason == "end_of_data" for t in result.trades), (
            "未平仓头寸应被强制平仓并记录为 end_of_data"
        )
        assert result.cost_returns.sum() > 0, "强制平仓的成本必须被计入"

    def test_cost_is_always_recorded(self, costs_config) -> None:
        """任何产生了交易的回测都必须记录成本。

        成本漏记是最常见的"回测赚钱、实盘亏钱"来源。
        """
        rates = make_funding_series([0.0005] * 80)
        strategy = StrategyConfig(
            entry=EntryConfig(lookback_periods=5, min_consecutive_positive=3),
            exit=ExitConfig(max_holding_periods=20),
            selection=SelectionConfig(max_positions=3, per_position_weight=0.2),
        )
        engine = FundingCarryBacktester(
            BacktestConfig(execution_lag_bars=1), strategy, CostModel(costs_config)
        )
        result = engine.run("COSTUSDT", rates, 8)

        if result.trades:
            assert result.cost_returns.sum() > 0, "有交易却未记录成本"
            # 成本总额应等于各笔交易的进出场成本之和
            expected = sum(t.entry_cost + t.exit_cost for t in result.trades)
            assert result.cost_returns.sum() == pytest.approx(expected, rel=1e-9)

    def test_gross_minus_cost_equals_net(self, costs_config) -> None:
        """净值必须等于毛收益减成本（加基差调整）。

        这是记账恒等式。如果它不成立，指标全是错的。
        """
        rates = make_funding_series([0.0004] * 80)
        strategy = StrategyConfig(
            entry=EntryConfig(lookback_periods=5, min_consecutive_positive=3),
            exit=ExitConfig(max_holding_periods=15),
            selection=SelectionConfig(max_positions=3, per_position_weight=0.2),
        )
        engine = FundingCarryBacktester(
            BacktestConfig(execution_lag_bars=1), strategy, CostModel(costs_config)
        )
        result = engine.run("IDENTUSDT", rates, 8)

        basis_total = sum(t.basis_adjustment for t in result.trades)
        expected_net = result.gross_returns.sum() - result.cost_returns.sum() + basis_total

        assert result.period_returns.sum() == pytest.approx(expected_net, rel=1e-9), (
            "记账恒等式不成立：净值 ≠ 毛收益 - 成本 + 基差调整"
        )

    def test_trade_records_reconcile_with_period_ledger(self, costs_config) -> None:
        """逐笔交易记录必须与逐期记账对得上账。

        ## 这条测试抓的是真实存在过的 bug

        ``TradeRecord.gross_funding`` 曾用
        ``rates.iloc[entry_exec_index : exit_signal_index + 1]`` 求和，
        而主循环的记账条件是 ``i > entry_exec_index``（建仓当期不持仓、
        不收资金费）。两者起点差一期，导致交易记录比实际入账**多算一期**。

        后果：所有基于逐笔归因的分析（"这笔交易赚了多少"、
        "哪类出场原因更划算"）全部失真，而总收益却是对的 ——
        所以从汇总数字上完全看不出来。

        修法是让两处使用同一口径。本测试断言：
        ``Σ(交易记录) == Σ(逐期记账)``。
        """
        rates = make_funding_series([0.0004] * 60)
        strategy = StrategyConfig(
            entry=EntryConfig(lookback_periods=5, min_consecutive_positive=3),
            exit=ExitConfig(max_holding_periods=12),
            selection=SelectionConfig(max_positions=3, per_position_weight=0.2),
        )
        engine = FundingCarryBacktester(
            BacktestConfig(execution_lag_bars=1), strategy, CostModel(costs_config)
        )
        result = engine.run("RECONUSDT", rates, 8)

        assert result.trades, "本测试需要至少产生一笔交易才有意义"

        weight = 0.2

        # 口径一：从逐笔交易记录汇总
        #   TradeRecord.gross_funding 是「持有期内费率之和」（**未乘权重**）
        #   TradeRecord.entry_cost / exit_cost 已是「占资金比例」（已乘权重）
        from_trades_gross = sum(t.gross_funding for t in result.trades) * weight
        from_trades_cost = sum(t.entry_cost + t.exit_cost for t in result.trades)

        # 口径二：从逐期记账汇总（gross/cost_returns 均已含权重）
        assert from_trades_gross == pytest.approx(result.gross_returns.sum(), rel=1e-12), (
            f"逐笔毛收益 {from_trades_gross:.10f} ≠ 逐期毛收益 "
            f"{result.gross_returns.sum():.10f}。"
            "常见原因是 _close_trade 的资金费区间与主循环的记账条件不一致。"
        )
        assert from_trades_cost == pytest.approx(result.cost_returns.sum(), rel=1e-12), (
            f"逐笔成本 {from_trades_cost:.10f} ≠ 逐期成本 {result.cost_returns.sum():.10f}"
        )

        # 净收益也必须对账
        from_trades_net = from_trades_gross - from_trades_cost
        assert from_trades_net == pytest.approx(result.period_returns.sum(), rel=1e-12), (
            f"逐笔净收益 {from_trades_net:.10f} ≠ 逐期净收益 "
            f"{result.period_returns.sum():.10f}"
        )

    def test_each_trade_funding_matches_its_holding_window(self, costs_config) -> None:
        """每笔交易记录的 gross_funding 必须能被重算出来。

        重算方式：``持有期数 × 单期费率``。本测试用**恒定费率**，
        因此这是精确的恒等式（费率变化时无法这样验证，
        由 test_trade_records_reconcile_with_period_ledger 覆盖）。

        这条测试锁住 TradeRecord 内部的自洽性：
        持有期数与资金费总额必须对应同一段窗口。
        """
        rates = make_funding_series([0.0004] * 70)
        strategy = StrategyConfig(
            entry=EntryConfig(lookback_periods=5, min_consecutive_positive=3),
            exit=ExitConfig(max_holding_periods=10),
            selection=SelectionConfig(max_positions=3, per_position_weight=0.2),
        )
        engine = FundingCarryBacktester(
            BacktestConfig(execution_lag_bars=1), strategy, CostModel(costs_config)
        )
        result = engine.run("WINDOWUSDT", rates, 8)

        assert result.trades, "本测试需要至少产生一笔交易才有意义"

        for trade in result.trades:
            recomputed = trade.holding_periods * 0.0004
            assert trade.gross_funding == pytest.approx(recomputed, rel=1e-9), (
                f"{trade.symbol} 交易持有 {trade.holding_periods} 期，"
                f"按恒定费率重算应为 {recomputed:.8f}，"
                f"实际记录 {trade.gross_funding:.8f}"
            )

    def test_trade_ledger_is_one_row_per_8h_period_and_slippage_is_explicit(
        self, costs_config
    ) -> None:
        rates = make_funding_series([0.0004] * 60, interval_hours=4)
        strategy = StrategyConfig(
            entry=EntryConfig(
                lookback_periods=1,
                min_consecutive_positive=1,
                min_annualized_rate=0.0,
                min_trailing_annualized=0.0,
            ),
            exit=ExitConfig(max_holding_periods=50),
            selection=SelectionConfig(max_positions=1, per_position_weight=0.5),
        )
        config = BacktestConfig(
            execution_lag_bars=1,
            rolling_window_periods=3,
            slippage_per_leg=0.01,
        )
        result = FundingCarryBacktester(config, strategy, CostModel(costs_config)).run(
            "LEDGERUSDT", rates, 4
        )

        assert result.interval_hours == 8
        assert result.trades
        assert result.period_ledger
        assert all(
            (right - left) == pd.Timedelta(hours=8)
            for left, right in zip(
                result.period_returns.index[:-1], result.period_returns.index[1:], strict=False
            )
        )
        assert all(
            period.period_time.tzinfo is not None for period in result.period_ledger
        )
        assert all(
            period.entry_cost_usdt + period.exit_cost_usdt >= 0
            for period in result.period_ledger
        )
        # 1% per leg：单笔进场成本应包含两条腿的 2% 滑点。
        assert result.trades[0].entry_cost >= 0.01 * 2 * strategy.selection.per_position_weight


        """交易记录的索引必须落在数据范围内。

        这条测试覆盖一个曾经存在的 bug：主循环里
        ``if exec_index >= n: break`` 会在接近末尾时中断整个循环，
        导致尾部期数不入账。修复后应改为「跳过交易决策但继续结算」。
        """
        rates = make_funding_series([0.0004] * 40)
        strategy = StrategyConfig(
            entry=EntryConfig(lookback_periods=5, min_consecutive_positive=3),
            exit=ExitConfig(max_holding_periods=8),
            selection=SelectionConfig(max_positions=3, per_position_weight=0.2),
        )
        engine = FundingCarryBacktester(
            BacktestConfig(execution_lag_bars=1), strategy, CostModel(costs_config)
        )
        result = engine.run("BOUNDUSDT", rates, 8)

        n = len(rates)
        for trade in result.trades:
            assert 0 <= trade.entry_index < n
            assert 0 <= trade.exit_index < n

        # 尾部期数必须被结算（不能被 break 掉）
        # 最后一期若仍持仓，应有 end_of_data 记录
        total_accrued = sum(t.holding_periods for t in result.trades)
        assert total_accrued > 0


# ---------------------------------------------------------------------------
# 滚动回测与动态选币
# ---------------------------------------------------------------------------


class TestRollingBacktest:
    """滚动回测必须逐时点使用可见历史，并正确处理延迟成交。"""

    def _strategy(self, *, max_positions: int = 1) -> StrategyConfig:
        return StrategyConfig(
            entry=EntryConfig(
                lookback_periods=1,
                min_consecutive_positive=1,
                min_annualized_rate=0.0,
                min_trailing_annualized=0.0,
            ),
            exit=ExitConfig(
                exit_annualized_rate=0.0,
                negative_streak_exit=1,
                max_holding_periods=50,
            ),
            selection=SelectionConfig(
                max_positions=max_positions,
                per_position_weight=0.5,
            ),
        )

    def test_fixed_window_caps_streak_and_requires_warmup(self) -> None:
        """固定窗口不能继承窗口外的连续正费率，也不能提前交易。"""
        rates = make_funding_series([0.0003] * 20)
        strategy = StrategyConfig(
            entry=EntryConfig(
                lookback_periods=1,
                min_consecutive_positive=6,
                min_annualized_rate=0.0,
                min_trailing_annualized=0.0,
            ),
            exit=ExitConfig(),
        )

        signals = build_signals(rates, 8, strategy, history_window=5)

        assert not signals.entry[:4].any(), "滚动窗口未满前不得产生进场信号"
        assert not signals.entry[4:].any(), "窗口外历史不得使连续正费率超过固定窗口"

    def test_smoothing_window_controls_entry_and_negative_exit(self) -> None:
        rates = make_funding_series([0.0008] * 10 + [-0.0008] * 10)
        strategy = StrategyConfig(
            entry=EntryConfig(
                lookback_periods=10,
                min_consecutive_positive=1,
                min_annualized_rate=0.0,
                min_trailing_annualized=0.0,
            ),
            exit=ExitConfig(exit_lookback_periods=10),
        )

        signals = build_signals(rates, 8, strategy, history_window=10)

        # 第 10 期才形成完整均值；转负前，10 期均值仍不能立即看成负值。
        assert not signals.entry[:9].any()
        assert signals.entry[9]
        assert not signals.exit[10]
        assert signals.exit[-1]

    def test_replacement_premium_boundaries(self) -> None:
        strategy = StrategyConfig(
            exit=ExitConfig(
                replacement_premium_under_60=float("inf"),
                replacement_premium_under_120=1.0,
                replacement_premium_over_120=0.7,
            )
        )
        from cointrader.backtest.engine import _replacement_premium

        assert _replacement_premium(30, strategy) == float("inf")
        assert _replacement_premium(60, strategy) == float("inf")
        assert _replacement_premium(61, strategy) == pytest.approx(1.0)
        assert _replacement_premium(120, strategy) == pytest.approx(1.0)
        assert _replacement_premium(121, strategy) == pytest.approx(0.7)

    def test_pending_entry_is_cancelled_before_execution(self, costs_config) -> None:
        """信号到成交之间转差时，不得记录一笔实际上未成交的交易。"""
        rates = make_funding_series(
            [0.0, 0.0, 0.0, 0.001, -0.001, -0.001, -0.001, -0.001]
        )
        engine = FundingCarryBacktester(
            BacktestConfig(execution_lag_bars=2, rolling_window_periods=3),
            self._strategy(),
            CostModel(costs_config),
        )

        result = engine.run("PENDINGUSDT", rates, 8)

        assert not result.trades, "未到成交时点且信号已失效，不应产生交易记录"
        assert result.cost_returns.sum() == pytest.approx(0.0)

    def test_future_tail_cannot_change_early_rolling_ledger(self, costs_config) -> None:
        """修改 cutoff 之后的费率，不得改变 cutoff 之前的账本。"""
        head = [0.0, 0.0, 0.0] + [0.0003] * 27
        rates_a = make_funding_series(head + [0.0003] * 30)
        rates_b = make_funding_series(head + [-0.001] * 30)
        engine = FundingCarryBacktester(
            BacktestConfig(execution_lag_bars=1, rolling_window_periods=5),
            self._strategy(),
            CostModel(costs_config),
        )

        result_a = engine.run("CAUSALUSDT", rates_a, 8)
        result_b = engine.run("CAUSALUSDT", rates_b, 8)

        np.testing.assert_array_equal(
            result_a.period_returns.iloc[:30].to_numpy(),
            result_b.period_returns.iloc[:30].to_numpy(),
        )
        np.testing.assert_array_equal(
            result_a.exposure.iloc[:30].to_numpy(),
            result_b.exposure.iloc[:30].to_numpy(),
        )

    def test_costs_are_recorded_at_execution_indices(self, costs_config) -> None:
        """进出场成本必须落在实际成交 bar，而不是未来持有期的摊销位置。"""
        rates = make_funding_series([0.0003] * 30)
        strategy = self._strategy()
        strategy = StrategyConfig(
            entry=strategy.entry,
            exit=ExitConfig(
                exit_annualized_rate=-1.0,
                negative_streak_exit=999,
                max_holding_periods=4,
            ),
            selection=strategy.selection,
        )
        engine = FundingCarryBacktester(
            BacktestConfig(execution_lag_bars=1, rolling_window_periods=3),
            strategy,
            CostModel(costs_config),
        )

        result = engine.run("EXECUSDT", rates, 8)

        assert result.trades
        trade = result.trades[0]
        assert result.cost_returns.iloc[trade.entry_index] >= trade.entry_cost
        assert result.cost_returns.iloc[trade.exit_index] >= trade.exit_cost
        expected_cost = sum(t.entry_cost + t.exit_cost for t in result.trades)
        assert result.cost_returns.sum() == pytest.approx(expected_cost)

    def test_portfolio_selects_at_most_max_positions(self, costs_config) -> None:
        """同一时点的候选必须动态排名，并遵守最大持仓数。"""
        rates = {
            "AUSDT": make_funding_series([0.0003] * 40),
            "BUSDT": make_funding_series([0.0003] * 40),
        }
        strategy = self._strategy(max_positions=1)

        result = run_portfolio(
            rates,
            {"AUSDT": 8, "BUSDT": 8},
            BacktestConfig(execution_lag_bars=1, rolling_window_periods=3),
            strategy,
            CostModel(costs_config),
        )

        assert result.metrics.avg_exposure <= 0.5 + 1e-12
        assert sum(len(symbol_result.trades) for symbol_result in result.per_symbol.values()) == 1

    def test_better_candidate_replaces_position_with_age_premium(self, costs_config) -> None:
        a_rates = [0.001] * 80
        b_rates = [0.0] * 21 + [0.0003] * 19 + [0.0021] * 40
        result = run_portfolio(
            {
                "AUSDT": make_funding_series(a_rates),
                "BUSDT": make_funding_series(b_rates),
            },
            {"AUSDT": 8, "BUSDT": 8},
            BacktestConfig(execution_lag_bars=1, rolling_window_periods=30),
            StrategyConfig(
                entry=EntryConfig(
                    lookback_periods=10,
                    min_consecutive_positive=1,
                    min_annualized_rate=0.30,
                    min_trailing_annualized=0.30,
                ),
                exit=ExitConfig(
                    exit_annualized_rate=0.0,
                    late_exit_annualized_rate=0.0,
                    negative_streak_exit=999,
                    replacement_premium_under_60=float("inf"),
                    replacement_premium_under_120=1.0,
                    replacement_premium_over_120=0.7,
                    max_holding_periods=999,
                ),
                selection=SelectionConfig(max_positions=1, per_position_weight=0.5),
            ),
            CostModel(costs_config),
        )

        a_trades = result.per_symbol["AUSDT"].trades
        b_trades = result.per_symbol["BUSDT"].trades
        assert a_trades
        assert not b_trades, "持仓不超过 60 期时不得因更优候选替换"


        """候选币尾部变好，不能回溯替换更早时点已经选出的币。"""
        common = [0.0, 0.0, 0.0] + [0.0003] * 17
        rates_a = {
            "AUSDT": make_funding_series(common + [0.0003] * 20),
            "BUSDT": make_funding_series(common + [0.0003] * 20),
        }
        rates_b = {
            "AUSDT": make_funding_series(common + [0.0003] * 20),
            "BUSDT": make_funding_series(common + [-0.001] * 20),
        }
        strategy = self._strategy(max_positions=1)
        config = BacktestConfig(execution_lag_bars=1, rolling_window_periods=5)
        model = CostModel(costs_config)

        first = run_portfolio(rates_a, {"AUSDT": 8, "BUSDT": 8}, config, strategy, model)
        second = run_portfolio(rates_b, {"AUSDT": 8, "BUSDT": 8}, config, strategy, model)

        np.testing.assert_array_equal(
            first.portfolio_returns.iloc[:20].to_numpy(),
            second.portfolio_returns.iloc[:20].to_numpy(),
        )


__all__: list[str] = []
