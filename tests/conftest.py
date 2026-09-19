"""共享测试夹具。

设计原则：**测试不依赖网络、不依赖密钥、不依赖真实交易所。**

任何需要联网的测试都标 ``@pytest.mark.network``，默认被 pyproject 里的
``-m 'not network'`` 跳过。这样核心逻辑的验证在任何环境下都能跑。
"""

from __future__ import annotations

import math
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from cointrader.config import (
    ApiConfig,
    BacktestConfig,
    CostsConfig,
    DataConfig,
    EntryConfig,
    ExitConfig,
    PerpFeeConfig,
    RiskConfig,
    SelectionConfig,
    SlippageConfig,
    SpotFeeConfig,
    StrategyConfig,
)

UTC = "UTC"


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root() -> Path:
    """项目根目录。"""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def src_root(project_root: Path) -> Path:
    return project_root / "src" / "cointrader"


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


# ---------------------------------------------------------------------------
# 配置夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def api_config() -> ApiConfig:
    return ApiConfig()


@pytest.fixture
def data_config(tmp_cache_dir: Path) -> DataConfig:
    return DataConfig(cache_dir=tmp_cache_dir, raw_dir=tmp_cache_dir.parent / "raw")


@pytest.fixture
def costs_config() -> CostsConfig:
    """默认成本配置：VIP0 + BNB 抵扣 + 全 taker。往返应为 0.29%。"""
    return CostsConfig()


@pytest.fixture
def pessimistic_costs_config() -> CostsConfig:
    return CostsConfig(
        fee_tier="pessimistic",
        spot=SpotFeeConfig(maker=0.001, taker=0.001),
        perp=PerpFeeConfig(maker=0.0005, taker=0.0005),
        bnb_discount=1.0,
        slippage=SlippageConfig(major=0.0002, mid=0.0006, small=0.001),
        assume_all_taker=True,
    )


@pytest.fixture
def backtest_config() -> BacktestConfig:
    return BacktestConfig(execution_lag_bars=1, in_sample_ratio=0.70)


@pytest.fixture
def strategy_config() -> StrategyConfig:
    return StrategyConfig(
        entry=EntryConfig(
            min_annualized_rate=0.15,
            min_consecutive_positive=3,
            lookback_periods=5,
            min_trailing_annualized=0.20,
        ),
        exit=ExitConfig(
            exit_annualized_rate=0.03,
            negative_streak_exit=2,
            max_holding_periods=50,
        ),
        selection=SelectionConfig(max_positions=3, per_position_weight=0.2),
    )


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        max_notional_per_order=200.0,
        max_total_exposure=2000.0,
        max_exposure_per_symbol=500.0,
    )


# ---------------------------------------------------------------------------
# 数据夹具
# ---------------------------------------------------------------------------


def make_funding_series(
    rates: list[float],
    *,
    start: str = "2025-01-01",
    interval_hours: int = 8,
) -> pd.Series:
    """由费率列表构造带时间索引的 Series。

    时间索引按真实的结算周期生成，这样下游的周期换算逻辑能被真实检验。
    """
    timestamps = pd.date_range(start=start, periods=len(rates), freq=f"{interval_hours}h", tz=UTC)
    return pd.Series(rates, index=timestamps, name="funding_rate", dtype=float)


@pytest.fixture
def positive_funding_series() -> pd.Series:
    """持续正资金费 —— 理想情况，策略应该持仓并盈利。"""
    rng = np.random.default_rng(42)
    # 平均 0.02% / 8h ≈ 年化 21.9%，足够覆盖 0.29% 的往返成本
    rates = 0.0002 + rng.normal(0, 0.00005, size=200)
    return make_funding_series(rates.tolist())


@pytest.fixture
def negative_funding_series() -> pd.Series:
    """持续负资金费 —— 策略不应该持仓，或持仓后应快速止损离场。"""
    rng = np.random.default_rng(43)
    rates = -0.0002 + rng.normal(0, 0.00005, size=200)
    return make_funding_series(rates.tolist())


@pytest.fixture
def mixed_funding_series() -> pd.Series:
    """先正后负 —— 用于测试出场逻辑。"""
    positive = [0.0003] * 60
    negative = [-0.0003] * 40
    return make_funding_series(positive + negative)


@pytest.fixture
def flat_funding_series() -> pd.Series:
    """恒定零费率 —— 应触发"不应交易"的判定（收益全被成本吃掉）。"""
    return make_funding_series([0.0] * 150)


# ---------------------------------------------------------------------------
# 交易所响应夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_exchange_info_spot() -> dict[str, Any]:
    """现货 exchangeInfo 的简化响应（字段结构与真实响应一致）。"""
    return {
        "timezone": "UTC",
        "serverTime": 1789560670000,
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "quotePrecision": 8,
                "filters": [
                    {
                        "filterType": "PRICE_FILTER",
                        "minPrice": "0.01000000",
                        "maxPrice": "1000000.00000000",
                        "tickSize": "0.01000000",
                    },
                    {
                        "filterType": "LOT_SIZE",
                        "minQty": "0.00001000",
                        "maxQty": "9000.00000000",
                        "stepSize": "0.00001000",
                    },
                    {
                        "filterType": "NOTIONAL",
                        "minNotional": "5.00000000",
                        "applyMinToMarket": True,
                    },
                ],
            },
            {
                "symbol": "SHIBUSDT",
                "status": "TRADING",
                "baseAsset": "SHIB",
                "quoteAsset": "USDT",
                "filters": [
                    {
                        "filterType": "PRICE_FILTER",
                        "minPrice": "0.00001000",
                        "tickSize": "0.00000001",
                    },
                    {
                        "filterType": "LOT_SIZE",
                        "minQty": "1.00000000",
                        "maxQty": "1000000000.00000000",
                        "stepSize": "1.00000000",
                    },
                    {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
                ],
            },
        ],
    }


@pytest.fixture
def sample_exchange_info_futures() -> dict[str, Any]:
    """永续 exchangeInfo 的简化响应。"""
    return {
        "timezone": "UTC",
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "contractType": "PERPETUAL",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "quantityPrecision": 3,
                "pricePrecision": 2,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                    {
                        "filterType": "LOT_SIZE",
                        "minQty": "0.001",
                        "maxQty": "1000",
                        "stepSize": "0.001",
                    },
                    {"filterType": "NOTIONAL", "minNotional": "5.00"},
                ],
            },
            {
                "symbol": "BTCUSDT_250926",   # 季度交割合约，应被过滤掉
                "status": "TRADING",
                "contractType": "CURRENT_QUARTER",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "filters": [],
            },
            {
                "symbol": "ETHUSDT",
                "status": "TRADING",
                "contractType": "PERPETUAL",
                "baseAsset": "ETH",
                "quoteAsset": "USDT",
                "quantityPrecision": 3,
                "pricePrecision": 2,
                "filters": [
                    {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "10000", "stepSize": "0.001"},
                    {"filterType": "NOTIONAL", "minNotional": "5.00"},
                ],
            },
            {
                "symbol": "USDCUSDT",          # 稳定币对，应按配置被排除
                "status": "TRADING",
                "contractType": "PERPETUAL",
                "baseAsset": "USDC",
                "quoteAsset": "USDT",
                "filters": [],
            },
            {
                "symbol": "DELISTEDUSDT",      # 非 TRADING 状态，应被过滤
                "status": "SETTLING",
                "contractType": "PERPETUAL",
                "baseAsset": "DELISTED",
                "quoteAsset": "USDT",
                "filters": [],
            },
        ],
    }


@pytest.fixture
def sample_funding_info() -> list[dict[str, Any]]:
    """fundingInfo 响应 —— **只包含非默认周期的合约**。

    这是币安的实际行为，很容易被误解为"全部合约列表"。
    """
    return [
        {
            "symbol": "LPTUSDT",
            "adjustedFundingRateCap": "0.02000000",
            "fundingIntervalHours": 4,
            "disclaimer": False,
        },
        {
            "symbol": "ARKUSDT",
            "adjustedFundingRateCap": "0.02000000",
            "fundingIntervalHours": 1,
            "disclaimer": False,
        },
    ]


@pytest.fixture
def sample_funding_history() -> list[dict[str, Any]]:
    """资金费历史 —— fundingTime 带毫秒级抖动，用于验证判重逻辑。"""
    base = 1789488000000
    return [
        {"symbol": "BTCUSDT", "fundingTime": base + 1, "fundingRate": "0.00009812", "markPrice": "76473.70"},
        {"symbol": "BTCUSDT", "fundingTime": base + 28800000, "fundingRate": "0.00003951", "markPrice": "75600.00"},
        {"symbol": "BTCUSDT", "fundingTime": base + 57600000, "fundingRate": "-0.00002788", "markPrice": "75726.00"},
    ]


@pytest.fixture
def sample_klines() -> list[list[Any]]:
    """K 线响应 —— 价格/数量都是字符串，这是解析最容易出错的地方。"""
    base = 1789516800000
    step = 28800000
    return [
        [
            base,
            "75599.90", "76096.40", "75405.50", "75726.00", "26780.638",
            base + step - 1,
            "2030309231.93350", 780094,
            "13308.914", "1009037157.69900", "0",
        ],
        [
            base + step,
            "75726.00", "76200.00", "75600.00", "76100.00", "21000.000",
            base + 2 * step - 1,
            "1600000000.00000", 700000,
            "10000.000", "760000000.00000", "0",
        ],
    ]


# ---------------------------------------------------------------------------
# 实时策略数据源 fake（开发文档 §7.1-7.4 测试基础设施）
# ---------------------------------------------------------------------------


class FakeStrategyData:
    """可控资金费率的策略数据源（StrategyDataProvider）。

    rates[symbol] = [(funding_ts_ms, rate, mark_rate), ...] 按时间升序，
    默认间隔 8h（24 条/天 → 1095 条/年）。与生产一致：只返回已结算条目。
    """

    def __init__(self, rates: dict[str, list[tuple[int, Decimal, Decimal]]],
                 interval_hours: dict[str, int] | None = None,
                 volumes: dict[str, Decimal] | None = None) -> None:
        self._rates = {k: list(v) for k, v in rates.items()}
        self._interval = interval_hours or {}
        self._volume = volumes or {}

    def rates(self, symbol: str) -> list[tuple[int, Decimal, Decimal]]:
        return list(self._rates.get(symbol, []))

    def set_rates(self, symbol: str, rates: list[tuple[int, Decimal, Decimal]]) -> None:
        self._rates[symbol] = list(rates)

    def funding_rates(self, symbol: str, periods: int) -> list[tuple[int, Decimal, Decimal]]:
        raw = self._rates.get(symbol, [])
        now_ms = int(time.time() * 1000)
        settled = [(ts, r, m) for (ts, r, m) in raw if ts <= now_ms]
        return settled[-periods:]

    def funding_interval_hours(self, symbol: str) -> int:
        return int(self._interval.get(symbol, 8))

    def quote_volume_3d_avg(self, symbol: str) -> Decimal:
        return self._volume.get(symbol, Decimal("10000000"))


def make_rate_series(n: int, rate: str, start_ms: int = 1_700_000_000_000,
                     interval_ms: int = 8 * 3600 * 1000,
                     end_ago_ms: int = 0) -> list[tuple[int, Decimal, Decimal]]:
    """生成 n 条恒定费率序列（funding_ts_ms, rate, mark_rate），全部在过去。"""
    value = Decimal(rate)
    base = int(time.time() * 1000) - (n * interval_ms + max(end_ago_ms, 0))
    return [(base + i * interval_ms, value, value) for i in range(n)]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """清空所有 COINTRADER_* 与 BINANCE_* 环境变量。

    安全测试必须从"全新环境"出发，否则开发者本地的环境变量
    会让测试结果不可复现（甚至误判为"交易已启用"）。
    """
    import os

    for key in list(os.environ):
        if key.startswith("COINTRADER_") or key.startswith("BINANCE_"):
            monkeypatch.delenv(key, raising=False)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    """带容差的近似相等，处理浮点误差。"""
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


@pytest.fixture
def write_config(tmp_path: Path):
    """生成一个最小可用的配置文件，用于测试配置加载。"""

    def _write(overrides: dict[str, Any] | None = None) -> Path:
        config: dict[str, Any] = {
            "data": {"cache_dir": str(tmp_path / "cache")},
            "costs": {
                "spot": {"maker": 0.001, "taker": 0.001},
                "perp": {"maker": 0.0002, "taker": 0.0005},
                "bnb_discount": 0.75,
            },
            "backtest": {"execution_lag_bars": 1},
            "strategy": {"name": "funding_carry"},
            "risk": {},
            "api": {},
            "logging": {"log_dir": str(tmp_path / "logs")},
        }
        if overrides:
            for section, values in overrides.items():
                config.setdefault(section, {}).update(values)

        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    return _write
