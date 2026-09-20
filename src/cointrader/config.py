"""配置加载与校验。

设计要点：

1. **单一真相源** —— 所有可调参数都在 ``config/config.yaml``，代码里不散落魔数。
2. **密钥隔离** —— 本模块**只读非敏感配置**。密钥路径见 ``secrets.py``，
   两者物理分开，避免「顺手从 config 读密钥」的习惯。
3. **加载即校验** —— 非法配置在启动时崩溃，而不是在半夜下单时崩溃。
   对交易系统，启动时崩溃是特性，不是缺陷。
4. **不可变** —— 配置对象用 frozen dataclass，运行期改不了。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/config.yaml")

# 项目根目录：本文件位于 <root>/src/cointrader/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 配置段
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    futures_weight_per_min: int = 2400
    spot_weight_per_min: int = 6000
    soft_limit_ratio: float = 0.80
    # P0(下单)/P1(恢复对账) 保留预算比例；P2-P4 不得占用（实施计划书 v2.0 §5）
    critical_reserve_ratio: float = 0.30
    # 429 无/短 Retry-After 时的最小保护冻结（秒）
    rate_limit_freeze_seconds: float = 120.0
    # 418 无有效 Retry-After 时的 IP 封禁（秒）
    ip_ban_seconds: float = 3600.0
    max_retries: int = 5
    base_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not 0.0 < self.soft_limit_ratio <= 1.0:
            raise ConfigError(f"soft_limit_ratio 必须在 (0,1] 内，当前 {self.soft_limit_ratio}")
        if not 0.0 < self.critical_reserve_ratio < self.soft_limit_ratio:
            raise ConfigError(
                f"critical_reserve_ratio 必须在 (0, soft_limit_ratio) 内，"
                f"当前 {self.critical_reserve_ratio}"
            )
        if self.futures_weight_per_min <= 0:
            raise ConfigError("futures_weight_per_min 必须为正")
        if self.spot_weight_per_min <= 0:
            raise ConfigError("spot_weight_per_min 必须为正")
        if self.rate_limit_freeze_seconds < 1:
            raise ConfigError(
                f"rate_limit_freeze_seconds 必须 >= 1，当前 {self.rate_limit_freeze_seconds}"
            )
        if self.ip_ban_seconds < 60:
            raise ConfigError(f"ip_ban_seconds 必须 >= 60，当前 {self.ip_ban_seconds}")
        if self.max_retries < 0:
            raise ConfigError("max_retries 不能为负")


@dataclass(frozen=True, slots=True)
class CacheTTLConfig:
    funding_history: int = 604800
    klines_closed: int = 604800
    premium_index: int = 5
    exchange_info: int = 3600
    funding_info: int = 86400


@dataclass(frozen=True, slots=True)
class DataConfig:
    cache_dir: Path = Path("data/cache")
    raw_dir: Path = Path("data/raw")
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    cache_ttl: CacheTTLConfig = field(default_factory=CacheTTLConfig)


@dataclass(frozen=True, slots=True)
class SpotFeeConfig:
    maker: float = 0.00100
    taker: float = 0.00100


@dataclass(frozen=True, slots=True)
class PerpFeeConfig:
    maker: float = 0.00020
    taker: float = 0.00050


@dataclass(frozen=True, slots=True)
class SlippageConfig:
    major: float = 0.00010
    mid: float = 0.00030
    small: float = 0.00050


@dataclass(frozen=True, slots=True)
class CostsConfig:
    fee_tier: str = "vip0_bnb"
    spot: SpotFeeConfig = field(default_factory=SpotFeeConfig)
    perp: PerpFeeConfig = field(default_factory=PerpFeeConfig)
    spot_bnb_discount: float = 0.75
    perp_bnb_discount: float = 0.90
    # 兼容旧配置名；新配置使用 spot_bnb_discount/perp_bnb_discount。
    bnb_discount: float = 0.75
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    assume_all_taker: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("spot.maker", self.spot.maker),
            ("spot.taker", self.spot.taker),
            ("perp.maker", self.perp.maker),
            ("perp.taker", self.perp.taker),
        ):
            if not 0.0 <= value < 0.05:
                # 5% 以上单边费率极可能是单位写错（把 0.1% 写成 0.1）
                raise ConfigError(f"{name} 费率异常: {value}（应为小数，如 0.001 = 0.1%）")
        if not 0.0 < self.bnb_discount <= 1.0:
            raise ConfigError(f"bnb_discount 必须在 (0,1] 内，当前 {self.bnb_discount}")
        # 旧配置显式修改 bnb_discount 时，保留旧的单一折扣语义。
        if self.bnb_discount != 0.75:
            object.__setattr__(self, "spot_bnb_discount", self.bnb_discount)
            object.__setattr__(self, "perp_bnb_discount", 1.0)
        for name, value in (
            ("spot_bnb_discount", self.spot_bnb_discount),
            ("perp_bnb_discount", self.perp_bnb_discount),
        ):
            if not 0.0 < value <= 1.0:
                raise ConfigError(f"{name} 必须在 (0,1] 内，当前 {value}")


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    execution_lag_bars: int = 1
    # 固定滚动窗口：每个时点的信号最多使用最近 N 期已完成/已观测数据。
    # 该值是回测的历史预热长度，不是把未来数据切进训练集。
    rolling_window_periods: int = 30
    # 回测数据范围：从当前时间向前取最近 N 天。
    history_days: int = 365
    # 回测专用滑点：每条现货/永续成交腿的比例。
    slippage_per_leg: float = 0.0015
    in_sample_ratio: float = 0.70
    days_per_year: int = 365
    initial_capital: float = 10000.0
    mode: str = "hedged"

    def __post_init__(self) -> None:
        # 这是整个回测最容易自欺欺人的地方，用配置校验硬卡。
        if self.execution_lag_bars < 1:
            raise ConfigError(
                "execution_lag_bars 必须 >= 1。"
                "设为 0 意味着用当期信号按当期价格成交，即前瞻偏差。"
            )
        if self.rolling_window_periods <= 0:
            raise ConfigError(
                f"rolling_window_periods 必须为正，当前 {self.rolling_window_periods}"
            )
        if self.history_days <= 0:
            raise ConfigError(f"history_days 必须为正，当前 {self.history_days}")
        if not 0.0 <= self.slippage_per_leg <= 1.0:
            raise ConfigError(
                f"slippage_per_leg 必须在 [0,1] 内，当前 {self.slippage_per_leg}"
            )
        if not 0.0 < self.in_sample_ratio < 1.0:
            raise ConfigError(f"in_sample_ratio 必须在 (0,1) 内，当前 {self.in_sample_ratio}")
        if self.initial_capital <= 0:
            raise ConfigError("initial_capital 必须为正")


@dataclass(frozen=True, slots=True)
class EntryConfig:
    min_annualized_rate: float = 0.30
    min_consecutive_positive: int = 6
    lookback_periods: int = 10
    min_trailing_annualized: float = 0.30


@dataclass(frozen=True, slots=True)
class ExitConfig:
    # 出场判断使用最近 30 期资金费均值；窗口未满时不用于出场。
    exit_lookback_periods: int = 30
    replacement_premium_under_60: float = 1.00
    replacement_premium_under_120: float = 1.00
    replacement_premium_over_120: float = 0.70
    max_holding_periods: int = 270
    # 旧字段保留用于兼容外部测试/调用；新策略不读取这些字段。
    exit_annualized_rate: float = 0.03
    negative_streak_exit: int = 3
    late_exit_annualized_rate: float = 0.10
    replacement_premium_under_30: float = 1.00
    replacement_premium_under_50: float = 0.70
    replacement_premium_under_100: float = 0.50
    replacement_premium_over_100: float = 0.30

    def __post_init__(self) -> None:
        if self.exit_lookback_periods <= 0:
            raise ConfigError("exit_lookback_periods 必须为正")
        if self.max_holding_periods <= 0:
            raise ConfigError("max_holding_periods 必须为正")
        for name in (
            "replacement_premium_under_60",
            "replacement_premium_under_120",
            "replacement_premium_over_120",
        ):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} 不能为负")


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    max_positions: int = 3
    per_position_weight: float = 0.33
    min_quote_volume_3d_avg: float = 1_000_000.0
    # 兼容旧调用方；新准入逻辑不读取该字段。
    min_quote_volume_24h: float | None = None
    exclude_bases: tuple[str, ...] = ("USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP")

    def __post_init__(self) -> None:
        if self.max_positions <= 0:
            raise ConfigError("max_positions 必须为正")
        if not 0.0 < self.per_position_weight <= 1.0:
            raise ConfigError("per_position_weight 必须在 (0,1] 内")
        if self.max_positions * self.per_position_weight > 1.0 + 1e-9:
            raise ConfigError(
                "max_positions × per_position_weight 不能超过 1.0，避免组合超配"
            )


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    name: str = "funding_carry"
    entry: EntryConfig = field(default_factory=EntryConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)


@dataclass(frozen=True, slots=True)
class RiskConfig:
    max_notional_per_order: float = 200.0
    max_total_exposure: float = 2000.0
    max_exposure_per_symbol: float = 500.0
    max_daily_loss_pct: float = 0.02
    max_unhedged_seconds: int = 60
    max_basis_adverse_pct: float = 0.0015

    def __post_init__(self) -> None:
        for name, value in (
            ("max_notional_per_order", self.max_notional_per_order),
            ("max_total_exposure", self.max_total_exposure),
            ("max_exposure_per_symbol", self.max_exposure_per_symbol),
        ):
            if value <= 0:
                raise ConfigError(f"{name} 必须为正，当前 {value}")
        if self.max_exposure_per_symbol > self.max_total_exposure:
            raise ConfigError("单币种敞口上限不能大于总敞口上限")
        if self.max_notional_per_order > self.max_exposure_per_symbol:
            raise ConfigError("单笔上限不能大于单币种敞口上限")


@dataclass(frozen=True, slots=True)
class ApiConfig:
    spot_base: str = "https://api.binance.com"
    futures_base: str = "https://fapi.binance.com"
    spot_testnet_base: str = "https://testnet.binance.vision"
    futures_testnet_base: str = "https://testnet.binancefuture.com"
    # 测试网 user data stream 端点（demo trading 与经典 testnet 域名不同，可配置）
    spot_testnet_ws_base: str = "wss://stream.binance.com:9443"
    futures_testnet_ws_base: str = "wss://stream.binancefuture.com"
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        for name, url in (
            ("spot_base", self.spot_base),
            ("futures_base", self.futures_base),
            ("spot_testnet_base", self.spot_testnet_base),
            ("futures_testnet_base", self.futures_testnet_base),
        ):
            if not url.startswith("https://"):
                # 明文 HTTP 会泄露请求内容（含签名），必须拒绝
                raise ConfigError(f"{name} 必须使用 https，当前: {url}")
        for name, url in (
            ("spot_testnet_ws_base", self.spot_testnet_ws_base),
            ("futures_testnet_ws_base", self.futures_testnet_ws_base),
        ):
            if not url.startswith("wss://"):
                # 明文 WS 会泄露 listen key 与成交事件，必须拒绝
                raise ConfigError(f"{name} 必须使用 wss，当前: {url}")
        if self.timeout_seconds <= 0:
            raise ConfigError("timeout_seconds 必须为正")


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str = "INFO"
    log_dir: Path = Path("logs")
    structured: bool = True


#: 运行模式（开发设计文档 §4.2）：PAPER 本地模拟 / TESTNET 测试网真实订单 /
#: SHADOW 读真实行情但只生成意图 / LIVE 主网真实订单（需双重开关 + 人工确认）。
EXECUTION_MODES = ("paper", "testnet", "shadow", "live")


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """实盘执行层配置（开发设计文档 §4.2）。凭证不在此处，只从环境变量读。"""

    mode: str = "paper"
    recv_window_ms: int = 5000
    request_timeout_seconds: float = 5.0
    order_ack_timeout_seconds: float = 3.0
    state_db: Path = Path("data/live/trading.sqlite3")
    audit_path: Path = Path("logs/audit.log")
    user_stream_keepalive_seconds: int = 1800
    # 用户流模式：stream=WebSocket 用户流（经典 testnet/主网）；
    # poll=REST 轮询（demo trading 用户流不可用：现货 listenKey 410、合约 WS 20 秒被断）
    user_stream_mode: str = "stream"
    # poll 模式专用：轮询间隔与新鲜度窗口（秒）
    user_stream_poll_seconds: float = 5.0
    user_stream_fresh_seconds: float = 15.0
    # 对账忽略的非策略资产（如 demo 平台发放的 USDC）；只跳过无本地期望持仓的 symbol
    reconcile_ignore_assets: list[str] = field(default_factory=list)
    reconciliation_interval_seconds: int = 30
    max_market_data_age_seconds: float = 5.0
    max_leg_slippage_pct: float = 0.001
    hedge_tolerance_pct: float = 0.005
    canary_notional: float = 10.0  # 单 pair 目标名义额（canary 期极小）
    leverage: int = 1
    margin_type: str = "isolated"
    one_way_position_mode: bool = True
    server_time_offset_limit_ms: int = 3000
    # server time 周期性重校准（秒）：代理链路 RTT 漂移会让启动时一次性偏移过期，
    # 导致签名请求 -1021。长跑必须定期重校准（§3.5 防线）。
    time_resync_seconds: float = 300.0
    # 主循环连续 tick 异常阈值：达到后优雅停机并退出码 1，交给 systemd Restart 拉起。
    max_consecutive_tick_errors: int = 10
    # 主循环看门狗心跳超时（秒）：tick 心跳超时未更新 = 主线程挂死，
    # 看门狗以非 0 码终止进程，systemd Restart=always 拉起后从账本恢复。
    # 必须大于正常单轮最坏耗时（对账/候选刷新/权重休眠），默认 300s。
    watchdog_timeout_seconds: float = 300.0
    # WebUI 实时仪表盘（live run 进程内独立守护线程；只读，故障与主循环隔离）
    webui_enabled: bool = True
    webui_host: str = "0.0.0.0"  # noqa: S104 —— 需局域网/tailscale 访问，见 config.yaml 注释
    webui_port: int = 8888
    # 实时策略候选池（§7.1）：
    #   空 = 动态候选池（回测同口径：可交易 USDT 永续 → 成交额过滤 → top N）
    #   非空 = 固定候选列表（单 pair 闭环 / 手工选币模式）
    live_symbols: tuple[str, ...] = ()
    # 动态候选池最大 symbol 数（0 = 不限；越大 Binance API 权重开销越大）
    candidate_pool_max_symbols: int = 100
    # 动态候选池刷新周期（秒）；池内指标仍按各币资金费结算周期刷新
    universe_refresh_seconds: float = 1800.0
    # ⚠️ 已废弃（实施计划书 v2.0）：固定 symbol/分钟 预算被共享 weight 调度取代，
    # 本字段不再控制任何行为，保留仅为加载兼容；将在 scan epoch 任务中彻底移除。
    candidate_refetch_per_minute: int = 6
    # 候选后台刷新有界并发（1..16）
    candidate_refresh_concurrency: int = 4
    # scan epoch 构建截止（秒）：超时标 DEGRADED/告警，绝不放行交易
    scan_epoch_deadline_seconds: float = 600.0
    # READY 排名后取执行报价的最大候选数（top K）
    candidate_quote_top_k: int = 10
    # Spot/Futures 报价接收时间最大偏差（毫秒）
    max_quote_skew_ms: int = 500
    # 无同步游标时最大初始补账窗口（天，1..365）
    recovery_backfill_days: int = 30
    # 同一轮账户/对账/poll 的 exchange snapshot single-flight 复用窗口（秒）
    exchange_snapshot_reuse_seconds: float = 2.0
    # 候选最小刷新间隔（秒）；实际每币间隔 = max(本值, 该币资金费结算周期)
    candidate_refresh_seconds: float = 300.0
    # 候选指标缓存允许的最大年龄（秒）；超过则拒绝开仓
    max_candidate_data_age_seconds: float = 1800.0
    # 账户/持仓快照采样周期（秒，§8.5 至少 30s）
    snapshot_interval_seconds: int = 30
    # 报告导出根目录
    report_dir: Path = Path("reports/live")

    def __post_init__(self) -> None:
        if self.mode not in EXECUTION_MODES:
            raise ConfigError(
                f"execution.mode 必须是 {EXECUTION_MODES}，当前: {self.mode!r}"
            )
        if self.user_stream_mode not in ("stream", "poll"):
            raise ConfigError(
                f"execution.user_stream_mode 必须是 ('stream', 'poll')，当前: {self.user_stream_mode!r}"
            )
        if self.max_consecutive_tick_errors < 1:
            raise ConfigError(
                f"execution.max_consecutive_tick_errors 必须 >= 1，当前 {self.max_consecutive_tick_errors}"
            )
        if not 1_000 <= self.recv_window_ms <= 60_000:
            raise ConfigError(f"recv_window_ms 必须在 [1000, 60000]，当前 {self.recv_window_ms}")
        for name, value in (
            ("request_timeout_seconds", self.request_timeout_seconds),
            ("order_ack_timeout_seconds", self.order_ack_timeout_seconds),
            ("user_stream_keepalive_seconds", self.user_stream_keepalive_seconds),
            ("reconciliation_interval_seconds", self.reconciliation_interval_seconds),
            ("user_stream_poll_seconds", self.user_stream_poll_seconds),
            ("user_stream_fresh_seconds", self.user_stream_fresh_seconds),
            ("max_market_data_age_seconds", self.max_market_data_age_seconds),
            ("max_leg_slippage_pct", self.max_leg_slippage_pct),
            ("hedge_tolerance_pct", self.hedge_tolerance_pct),
            ("canary_notional", self.canary_notional),
            ("candidate_refresh_seconds", self.candidate_refresh_seconds),
            ("max_candidate_data_age_seconds", self.max_candidate_data_age_seconds),
            ("snapshot_interval_seconds", self.snapshot_interval_seconds),
            ("time_resync_seconds", self.time_resync_seconds),
            ("watchdog_timeout_seconds", self.watchdog_timeout_seconds),
            ("universe_refresh_seconds", self.universe_refresh_seconds),
        ):
            if value <= 0:
                raise ConfigError(f"execution.{name} 必须为正，当前 {value}")
        if self.candidate_pool_max_symbols < 0:
            raise ConfigError(
                f"execution.candidate_pool_max_symbols 必须 >= 0，"
                f"当前 {self.candidate_pool_max_symbols}"
            )
        if self.candidate_refetch_per_minute < 1:
            raise ConfigError(
                f"execution.candidate_refetch_per_minute 必须 >= 1，"
                f"当前 {self.candidate_refetch_per_minute}"
            )
        if not 1 <= self.candidate_refresh_concurrency <= 16:
            raise ConfigError(
                f"execution.candidate_refresh_concurrency 必须在 [1,16]，"
                f"当前 {self.candidate_refresh_concurrency}"
            )
        if self.scan_epoch_deadline_seconds <= 0:
            raise ConfigError(
                f"execution.scan_epoch_deadline_seconds 必须 > 0，当前 {self.scan_epoch_deadline_seconds}"
            )
        if self.candidate_quote_top_k < 1:
            raise ConfigError(
                f"execution.candidate_quote_top_k 必须 >= 1，当前 {self.candidate_quote_top_k}"
            )
        if self.max_quote_skew_ms < 1:
            raise ConfigError(f"execution.max_quote_skew_ms 必须 >= 1，当前 {self.max_quote_skew_ms}")
        if not 1 <= self.recovery_backfill_days <= 365:
            raise ConfigError(
                f"execution.recovery_backfill_days 必须在 [1,365]，当前 {self.recovery_backfill_days}"
            )
        if self.exchange_snapshot_reuse_seconds <= 0:
            raise ConfigError(
                f"execution.exchange_snapshot_reuse_seconds 必须 > 0，"
                f"当前 {self.exchange_snapshot_reuse_seconds}"
            )
        if self.exchange_snapshot_reuse_seconds > self.user_stream_fresh_seconds:
            raise ConfigError(
                f"execution.exchange_snapshot_reuse_seconds 必须 <= user_stream_fresh_seconds"
                f"（{self.user_stream_fresh_seconds}），当前 {self.exchange_snapshot_reuse_seconds}"
            )
        for symbol in self.live_symbols:
            if not symbol.upper().endswith("USDT"):
                raise ConfigError(f"live_symbols 只允许 USDT 永续对，当前: {symbol}")
        if self.leverage not in (1, 2):
            # 文档 §7.2：主网初始 1x，最多人工审批 2x，禁止动态加杠杆
            raise ConfigError(f"leverage 只允许 1 或 2，当前 {self.leverage}")
        if self.margin_type not in ("isolated", "cross"):
            raise ConfigError(f"margin_type 只允许 isolated/cross，当前 {self.margin_type!r}")


@dataclass(frozen=True, slots=True)
class Config:
    """顶层配置。不可变。"""

    data: DataConfig
    costs: CostsConfig
    backtest: BacktestConfig
    strategy: StrategyConfig
    risk: RiskConfig
    api: ApiConfig
    logging: LoggingConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    source_path: Path | None = None

    def resolved_path(self, path: Path) -> Path:
        """把配置里的相对路径解析为「相对项目根」的绝对路径。

        避免「在哪个目录执行命令就写到哪个目录」的经典陷阱。
        """
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"配置段 '{key}' 必须是映射，实际是 {type(value).__name__}")
    return value


def _build_data(raw: dict[str, Any]) -> DataConfig:
    sec = _section(raw, "data")
    rl = _section(sec, "rate_limit")
    ttl = _section(sec, "cache_ttl")
    return DataConfig(
        cache_dir=Path(sec.get("cache_dir", "data/cache")),
        raw_dir=Path(sec.get("raw_dir", "data/raw")),
        rate_limit=RateLimitConfig(**{k: v for k, v in rl.items() if k in RateLimitConfig.__slots__}),
        cache_ttl=CacheTTLConfig(**{k: v for k, v in ttl.items() if k in CacheTTLConfig.__slots__}),
    )


def _build_costs(raw: dict[str, Any]) -> CostsConfig:
    sec = _section(raw, "costs")
    return CostsConfig(
        fee_tier=sec.get("fee_tier", "vip0_bnb"),
        spot=SpotFeeConfig(**_section(sec, "spot")),
        perp=PerpFeeConfig(**_section(sec, "perp")),
        spot_bnb_discount=float(sec.get("spot_bnb_discount", sec.get("bnb_discount", 0.75))),
        perp_bnb_discount=float(
            sec.get("perp_bnb_discount", 1.0 if "bnb_discount" in sec else 0.90)
        ),
        bnb_discount=float(sec.get("bnb_discount", 0.75)),
        slippage=SlippageConfig(**_section(sec, "slippage")),
        assume_all_taker=bool(sec.get("assume_all_taker", True)),
    )


def _build_backtest(raw: dict[str, Any]) -> BacktestConfig:
    sec = _section(raw, "backtest")
    return BacktestConfig(
        execution_lag_bars=int(sec.get("execution_lag_bars", 1)),
        rolling_window_periods=int(sec.get("rolling_window_periods", 30)),
        history_days=int(sec.get("history_days", 365)),
        slippage_per_leg=float(sec.get("slippage_per_leg", 0.0015)),
        in_sample_ratio=float(sec.get("in_sample_ratio", 0.70)),
        days_per_year=int(sec.get("days_per_year", 365)),
        initial_capital=float(sec.get("initial_capital", 10000.0)),
        mode=str(sec.get("mode", "hedged")),
    )


def _build_strategy(raw: dict[str, Any]) -> StrategyConfig:
    sec = _section(raw, "strategy")
    sel = _section(sec, "selection")
    default_exclude = ("USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP")
    exclude = tuple(sel.get("exclude_bases") or default_exclude)
    return StrategyConfig(
        name=str(sec.get("name", "funding_carry")),
        entry=EntryConfig(**_section(sec, "entry")),
        exit=ExitConfig(**_section(sec, "exit")),
        selection=SelectionConfig(
            max_positions=int(sel.get("max_positions", 5)),
            per_position_weight=float(sel.get("per_position_weight", 0.15)),
            min_quote_volume_3d_avg=float(
                sel.get("min_quote_volume_3d_avg", sel.get("min_quote_volume_24h", 1_000_000.0))
            ),
            min_quote_volume_24h=(
                float(sel["min_quote_volume_24h"]) if "min_quote_volume_24h" in sel else None
            ),
            exclude_bases=exclude,
        ),
    )


def _build_risk(raw: dict[str, Any]) -> RiskConfig:
    return RiskConfig(**_section(raw, "risk"))


def _build_api(raw: dict[str, Any]) -> ApiConfig:
    return ApiConfig(**_section(raw, "api"))


def _build_execution(raw: dict[str, Any]) -> ExecutionConfig:
    sec = _section(raw, "execution")
    if "candidate_refetch_per_minute" in sec:
        # 旧字段不再控制行为（实施计划书 v2.0）：候选刷新由共享 weight 调度驱动
        logger.warning(
            "execution.candidate_refetch_per_minute 已废弃，不再控制候选刷新节奏；"
            "请改用 data.rate_limit（共享 weight 调度）"
        )
    return ExecutionConfig(
        mode=str(sec.get("mode", "paper")).lower(),
        recv_window_ms=int(sec.get("recv_window_ms", 5000)),
        request_timeout_seconds=float(sec.get("request_timeout_seconds", 5.0)),
        order_ack_timeout_seconds=float(sec.get("order_ack_timeout_seconds", 3.0)),
        state_db=Path(sec.get("state_db", "data/live/trading.sqlite3")),
        audit_path=Path(sec.get("audit_path", "logs/audit.log")),
        user_stream_keepalive_seconds=int(sec.get("user_stream_keepalive_seconds", 1800)),
        user_stream_mode=str(sec.get("user_stream_mode", "stream")).lower(),
        user_stream_poll_seconds=float(sec.get("user_stream_poll_seconds", 5.0)),
        user_stream_fresh_seconds=float(sec.get("user_stream_fresh_seconds", 15.0)),
        reconcile_ignore_assets=[str(a).upper() for a in sec.get("reconcile_ignore_assets", [])],
        reconciliation_interval_seconds=int(sec.get("reconciliation_interval_seconds", 30)),
        max_market_data_age_seconds=float(sec.get("max_market_data_age_seconds", 5.0)),
        max_leg_slippage_pct=float(sec.get("max_leg_slippage_pct", 0.001)),
        hedge_tolerance_pct=float(sec.get("hedge_tolerance_pct", 0.005)),
        canary_notional=float(sec.get("canary_notional", 10.0)),
        leverage=int(sec.get("leverage", 1)),
        margin_type=str(sec.get("margin_type", "isolated")),
        one_way_position_mode=bool(sec.get("one_way_position_mode", True)),
        server_time_offset_limit_ms=int(sec.get("server_time_offset_limit_ms", 3000)),
        live_symbols=tuple(str(s).upper() for s in sec.get("live_symbols", [])),
        candidate_pool_max_symbols=int(sec.get("candidate_pool_max_symbols", 100)),
        universe_refresh_seconds=float(sec.get("universe_refresh_seconds", 1800.0)),
        candidate_refetch_per_minute=int(sec.get("candidate_refetch_per_minute", 6)),
        candidate_refresh_concurrency=int(sec.get("candidate_refresh_concurrency", 4)),
        scan_epoch_deadline_seconds=float(sec.get("scan_epoch_deadline_seconds", 600.0)),
        candidate_quote_top_k=int(sec.get("candidate_quote_top_k", 10)),
        max_quote_skew_ms=int(sec.get("max_quote_skew_ms", 500)),
        recovery_backfill_days=int(sec.get("recovery_backfill_days", 30)),
        exchange_snapshot_reuse_seconds=float(sec.get("exchange_snapshot_reuse_seconds", 2.0)),
        candidate_refresh_seconds=float(sec.get("candidate_refresh_seconds", 300.0)),
        max_candidate_data_age_seconds=float(sec.get("max_candidate_data_age_seconds", 1800.0)),
        snapshot_interval_seconds=int(sec.get("snapshot_interval_seconds", 30)),
        max_consecutive_tick_errors=int(sec.get("max_consecutive_tick_errors", 10)),
        watchdog_timeout_seconds=float(sec.get("watchdog_timeout_seconds", 300.0)),
        webui_enabled=bool(sec.get("webui_enabled", True)),
        webui_host=str(sec.get("webui_host", "0.0.0.0")),  # noqa: S104
        webui_port=int(sec.get("webui_port", 8888)),
        report_dir=Path(sec.get("report_dir", "reports/live")),
    )


def _build_logging(raw: dict[str, Any]) -> LoggingConfig:
    sec = _section(raw, "logging")
    return LoggingConfig(
        level=str(sec.get("level", "INFO")).upper(),
        log_dir=Path(sec.get("log_dir", "logs")),
        structured=bool(sec.get("structured", True)),
    )


def load_config(path: str | Path | None = None) -> Config:
    """加载并校验配置。

    Args:
        path: 配置文件路径。为 None 时按以下顺序查找：
              环境变量 COINTRADER_CONFIG → ./config/config.yaml → <root>/config/config.yaml

    Returns:
        校验通过的不可变 Config 对象。

    Raises:
        ConfigError: 文件不存在、YAML 语法错误、或任何字段校验失败。
    """
    if path is None:
        env_path = os.environ.get("COINTRADER_CONFIG")
        candidates = [Path(env_path)] if env_path else []
        candidates += [DEFAULT_CONFIG_PATH, PROJECT_ROOT / DEFAULT_CONFIG_PATH]
        resolved = next((p for p in candidates if p.is_file()), None)
        if resolved is None:
            raise ConfigError(
                f"未找到配置文件。查找过: {[str(c) for c in candidates]}。"
                "可用 COINTRADER_CONFIG 环境变量指定。"
            )
        path = resolved

    path = Path(path).expanduser()
    if not path.is_file():
        raise ConfigError(f"配置文件不存在: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件 {path}: {exc}") from exc

    try:
        # yaml.safe_load 而非 yaml.load —— 后者可执行任意 Python 对象构造，
        # 如果配置文件来自不可信来源就是远程代码执行漏洞。
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件 YAML 解析失败 {path}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"配置文件为空: {path}")
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件顶层必须是映射: {path}")

    return Config(
        data=_build_data(raw),
        costs=_build_costs(raw),
        backtest=_build_backtest(raw),
        strategy=_build_strategy(raw),
        risk=_build_risk(raw),
        api=_build_api(raw),
        logging=_build_logging(raw),
        execution=_build_execution(raw),
        source_path=path.resolve(),
    )


__all__ = [
    "ApiConfig",
    "BacktestConfig",
    "CacheTTLConfig",
    "Config",
    "CostsConfig",
    "DataConfig",
    "EntryConfig",
    "ExecutionConfig",
    "ExitConfig",
    "LoggingConfig",
    "RateLimitConfig",
    "RiskConfig",
    "SelectionConfig",
    "StrategyConfig",
    "load_config",
]
