"""配置加载与校验测试。

配置校验的价值：**在启动时崩溃，而不是在半夜下单时崩溃。**

对交易系统，"启动失败"是特性不是缺陷。一个非法的配置
（比如 execution_lag_bars=0）如果被容忍，会一路带着它跑完全程，
产出一份漂亮的、错误的回测报告。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cointrader.config import (
    BacktestConfig,
    Config,
    CostsConfig,
    PerpFeeConfig,
    RateLimitConfig,
    RiskConfig,
    SelectionConfig,
    SpotFeeConfig,
    load_config,
)
from cointrader.errors import ConfigError


class TestLoadConfig:
    """配置文件加载。"""

    def test_loads_real_project_config(self, project_root: Path) -> None:
        """项目自带的 config.yaml 必须能成功加载 —— 这是最基本的健全性检查。"""
        config = load_config(project_root / "config" / "config.yaml")

        assert isinstance(config, Config)
        assert config.data.rate_limit.futures_weight_per_min == 2400
        assert config.costs.bnb_discount == 0.75
        assert config.backtest.execution_lag_bars >= 1
        assert config.backtest.rolling_window_periods == 30
        assert config.backtest.history_days == 365
        assert config.backtest.slippage_per_leg == pytest.approx(0.0015)
        assert config.strategy.entry.min_annualized_rate == pytest.approx(0.30)
        assert config.strategy.entry.lookback_periods == 10
        assert config.strategy.exit.exit_annualized_rate == pytest.approx(0.03)
        assert config.strategy.exit.late_exit_annualized_rate == pytest.approx(0.10)
        assert config.strategy.exit.replacement_premium_under_30 == pytest.approx(1.0)
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="不存在"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")

        with pytest.raises(ConfigError, match="为空"):
            load_config(path)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("key: [unclosed\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="YAML 解析失败"):
            load_config(path)

    def test_non_mapping_toplevel_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- item1\n- item2\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="顶层必须是映射"):
            load_config(path)

    def test_env_var_overrides_path(self, monkeypatch: pytest.MonkeyPatch, write_config) -> None:
        """COINTRADER_CONFIG 环境变量应能指定配置文件。"""

        path = write_config()
        monkeypatch.setenv("COINTRADER_CONFIG", str(path))

        config = load_config()
        assert config.source_path == path.resolve()

    def test_yaml_safe_load_used(self, tmp_path: Path) -> None:
        """必须用 yaml.safe_load 而非 yaml.load。

        yaml.load 可以构造任意 Python 对象（``!!python/object/apply``），
        如果配置文件来自不可信来源就是远程代码执行漏洞。
        """
        malicious = tmp_path / "malicious.yaml"
        malicious.write_text(
            "data: !!python/object/apply:os.system ['echo pwned']\n",
            encoding="utf-8",
        )

        # safe_load 会拒绝这种构造
        with pytest.raises(ConfigError, match="YAML 解析失败"):
            load_config(malicious)


class TestBacktestConfigValidation:
    """回测配置校验 —— 前瞻偏差的第一道防线。"""

    @pytest.mark.parametrize("lag", [0, -1, -100])
    def test_rejects_nonpositive_execution_lag(self, lag: int) -> None:
        """execution_lag_bars 必须 >= 1。

        lag=0 意味着"用当期信号按当期价格成交"，是最隐蔽的前瞻偏差。
        配置层必须在启动时就拒绝它。
        """
        with pytest.raises(ConfigError, match="execution_lag_bars"):
            BacktestConfig(execution_lag_bars=lag)

    @pytest.mark.parametrize("days", [0, -1, -365])
    def test_rejects_nonpositive_history_days(self, days: int) -> None:
        with pytest.raises(ConfigError, match="history_days"):
            BacktestConfig(history_days=days)

    @pytest.mark.parametrize("slippage", [-0.01, 1.01])
    def test_rejects_invalid_slippage(self, slippage: float) -> None:
        with pytest.raises(ConfigError, match="slippage_per_leg"):
            BacktestConfig(slippage_per_leg=slippage)
    @pytest.mark.parametrize("window", [0, -1, -30])
    def test_rejects_nonpositive_rolling_window(self, window: int) -> None:
        with pytest.raises(ConfigError, match="rolling_window_periods"):
            BacktestConfig(rolling_window_periods=window)


    def test_accepts_rolling_window(self) -> None:
        assert BacktestConfig(rolling_window_periods=60).rolling_window_periods == 60

    @pytest.mark.parametrize("ratio", [0.0, 1.0, -0.5, 1.5])
    def test_rejects_invalid_in_sample_ratio(self, ratio: float) -> None:
        with pytest.raises(ConfigError, match="in_sample_ratio"):
            BacktestConfig(in_sample_ratio=ratio)

    def test_rejects_nonpositive_capital(self) -> None:
        with pytest.raises(ConfigError, match="initial_capital"):
            BacktestConfig(initial_capital=0.0)


class TestCostsConfigValidation:
    """成本配置校验 —— 防止单位写错。"""

    @pytest.mark.parametrize("bad_rate", [0.1, 0.05, 1.0, 10.0])
    def test_rejects_absurd_fee_rates(self, bad_rate: float) -> None:
        """费率超过 5% 必定是把 0.1% 写成了 0.1，必须拒绝。

        0.1 与 0.001 差 100 倍。如果被容忍，回测会把一个
        高成本策略误判为盈利策略。
        """
        with pytest.raises(ConfigError, match="费率异常"):
            CostsConfig(spot=SpotFeeConfig(maker=0.001, taker=bad_rate))

    @pytest.mark.parametrize("bad_discount", [0.0, -0.1, 1.5])
    def test_rejects_invalid_bnb_discount(self, bad_discount: float) -> None:
        with pytest.raises(ConfigError, match="bnb_discount"):
            CostsConfig(bnb_discount=bad_discount)

    def test_accepts_zero_fee(self) -> None:
        """零费率合法（理论分析用）。"""
        config = CostsConfig(
            spot=SpotFeeConfig(maker=0.0, taker=0.0),
            perp=PerpFeeConfig(maker=0.0, taker=0.0),
        )
        assert config.spot.taker == 0.0


class TestRiskConfigValidation:
    """风控配置校验 —— 限额层级必须自洽。"""

    def test_rejects_nonpositive_limits(self) -> None:
        with pytest.raises(ConfigError, match="必须为正"):
            RiskConfig(max_notional_per_order=0.0)

        with pytest.raises(ConfigError, match="必须为正"):
            RiskConfig(max_total_exposure=-100.0)

    def test_rejects_inconsistent_limit_hierarchy(self) -> None:
        """限额必须满足: 单笔 <= 单币种 <= 总敞口。

        层级倒挂会让某个限额永远不生效，等于风控少了一道。
        """
        with pytest.raises(ConfigError, match="单币种敞口上限不能大于总敞口"):
            RiskConfig(max_exposure_per_symbol=5000.0, max_total_exposure=1000.0)

        with pytest.raises(ConfigError, match="单笔上限不能大于单币种"):
            RiskConfig(max_notional_per_order=1000.0, max_exposure_per_symbol=500.0)

    def test_accepts_valid_hierarchy(self) -> None:
        config = RiskConfig(
            max_notional_per_order=200.0,
            max_exposure_per_symbol=500.0,
            max_total_exposure=2000.0,
        )
        assert config.max_notional_per_order < config.max_exposure_per_symbol
        assert config.max_exposure_per_symbol < config.max_total_exposure


class TestApiConfigValidation:
    """API 端点校验 —— 必须使用 HTTPS。"""

    @pytest.mark.parametrize(
        "field",
        ["spot_base", "futures_base", "spot_testnet_base", "futures_testnet_base"],
    )
    def test_rejects_http(self, field: str) -> None:
        """明文 HTTP 会暴露请求内容（含签名），必须拒绝。"""
        from cointrader.config import ApiConfig

        with pytest.raises(ConfigError, match="必须使用 https"):
            ApiConfig(**{field: "http://example.com"})


class TestRateLimitConfigValidation:
    """限流配置校验。"""

    @pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
    def test_rejects_invalid_soft_limit_ratio(self, ratio: float) -> None:
        with pytest.raises(ConfigError, match="soft_limit_ratio"):
            RateLimitConfig(soft_limit_ratio=ratio)

    def test_rejects_nonpositive_weight(self) -> None:
        with pytest.raises(ConfigError, match="必须为正"):
            RateLimitConfig(futures_weight_per_min=0)


class TestSelectionConfigValidation:
    """选币配置校验。"""

    def test_rejects_nonpositive_max_positions(self) -> None:
        with pytest.raises(ConfigError, match="max_positions"):
            SelectionConfig(max_positions=0)

    @pytest.mark.parametrize("weight", [0.0, -0.1, 1.5])
    def test_rejects_invalid_position_weight(self, weight: float) -> None:
        with pytest.raises(ConfigError, match="per_position_weight"):
            SelectionConfig(per_position_weight=weight)

    def test_rejects_overallocated_portfolio(self) -> None:
        with pytest.raises(ConfigError, match="不能超过"):
            SelectionConfig(max_positions=3, per_position_weight=0.34)

    def test_default_portfolio_is_three_positions_at_thirty_three_percent(self) -> None:
        selection = SelectionConfig()

        assert selection.max_positions == 3
        assert selection.per_position_weight == pytest.approx(0.33)


    def test_config_uses_three_position_defaults(self, project_root: Path) -> None:
        config = load_config(project_root / "config" / "config.yaml")

        assert config.strategy.selection.max_positions == 3
        assert config.strategy.selection.per_position_weight == pytest.approx(0.33)



    def test_relative_path_resolved_from_project_root(self, project_root: Path) -> None:
        config = load_config(project_root / "config" / "config.yaml")
        resolved = config.resolved_path(Path("data/cache"))

        assert resolved.is_absolute()
        assert str(resolved).startswith(str(project_root))

    def test_absolute_path_unchanged(self, project_root: Path, tmp_path: Path) -> None:
        config = load_config(project_root / "config" / "config.yaml")
        absolute = tmp_path / "somewhere"

        assert config.resolved_path(absolute) == absolute.resolve()


class TestConfigImmutability:
    """配置必须不可变。"""

    def test_cannot_mutate_at_runtime(self, project_root: Path) -> None:
        """运行期改配置是危险的（回测中途改参数会让结果不可复现）。

        frozen dataclass 会抛 FrozenInstanceError。
        """
        config = load_config(project_root / "config" / "config.yaml")

        with pytest.raises(Exception):  # FrozenInstanceError
            config.backtest.execution_lag_bars = 0  # type: ignore[misc]

    def test_nested_config_is_frozen(self) -> None:
        risk = RiskConfig()
        with pytest.raises(Exception):
            risk.max_notional_per_order = 999999.0  # type: ignore[misc]


class TestOverrideMerging:
    """配置文件的部分覆盖。"""

    def test_partial_costs_section_uses_defaults(self, write_config) -> None:
        """只覆盖部分字段时，其余应使用默认值。"""
        path = write_config({"costs": {"bnb_discount": 1.0}})
        config = load_config(path)

        assert config.costs.bnb_discount == 1.0
        assert config.costs.spot.taker == 0.001   # 默认值
        assert config.costs.perp.taker == 0.0005  # 默认值

    def test_missing_sections_use_defaults(self, tmp_path: Path) -> None:
        """完全最小化的配置也应能加载（所有段用默认值）。"""
        path = tmp_path / "minimal.yaml"
        path.write_text("backtest:\n  execution_lag_bars: 1\n", encoding="utf-8")

        config = load_config(path)
        assert config.costs.spot.taker == 0.001
        assert config.risk.max_total_exposure == 2000.0
        assert config.data.rate_limit.futures_weight_per_min == 2400

    def test_invalid_section_type_raises(self, tmp_path: Path) -> None:
        """配置段类型错误时应给出清晰的报错。"""
        path = tmp_path / "bad_section.yaml"
        path.write_text("costs: not_a_mapping\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="必须是映射"):
            load_config(path)


__all__: list[str] = []
