"""MarketDataSynchronizer —— 同一截止时点（scan epoch）的候选数据同步。

实施计划书 v2.0 T2（AC-03/04/05）：

- 每次全量刷新建立一个不可变 **scan epoch**：一次 universe 快照 +
  统一 ``decision_cutoff_ms``（交易所校准时间）+ expected/excluded/failed 集合。
- 只有当 ``completed + excluded == expected`` 且 ``failed == 0`` 且所有
  cutoff 不变量成立时，epoch 才原子封存为 ``READY``；超时标 ``DEGRADED``
  （deadline 只决定状态/告警，**绝不**放行交易）。
- cutoff 后任一候选应有的新 funding 结算或 4h K 线闭合发生 → 上一 READY
  epoch 标为 ``EXPIRED``：禁止新开仓/换仓，已有仓位的风险降低退出不受影响。
- 网络错误、限流、解析错误、结算滞后一律计 **failed**（不得当作 excluded
  或「不够优质」静默排除）。
- 后台单 worker 线程 + 有界并发（``candidate_refresh_concurrency``）拉取
  候选；交易主循环只读不可变 READY 快照，不因限流阻塞持仓管理。
- 固定等待时间（如「等 13 分钟」）在数据完整性上没有任何地位 ——
  完整性只能由「同一 cutoff 的全候选横截面 + 不变量校验」证明。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import Enum
from inspect import signature
from typing import Any, Protocol

from ..config import Config
from ..data.funding import normalize_funding_records_to_8h
from ..errors import CoinTraderError

logger = logging.getLogger(__name__)

__all__ = [
    "CandidateSnapshot",
    "DataReadiness",
    "EpochBuildError",
    "MarketDataSynchronizer",
    "ScanEpoch",
    "ScanEpochStatus",
]

_KLINE_INTERVAL_MS = 4 * 3600 * 1000
#: 3 日平均成交额需要的闭合 4h K 线根数
_VOLUME_KLINES = 18
#: 内存保留的最近 epoch 数（READY/BUILDING 之外的诊断 epoch）
_MAX_KEPT_EPOCHS = 8


class EpochBuildError(CoinTraderError):
    """universe 快照失败（候选池本身无法构建）。"""


class ScanEpochStatus(str, Enum):
    """epoch 状态。只有 READY 可开仓/换仓；DEGRADED 不因超时自动 READY。"""

    BUILDING = "BUILDING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    """单个候选在某个 epoch 内的不可变数据快照（同一 cutoff）。"""

    epoch_id: str
    symbol: str
    interval_hours: int
    rates: tuple[Decimal, ...]  # 单期费率，升序
    mark_prices: tuple[Decimal, ...]  # 结算标记价（与 rates 对齐）
    timestamps: tuple[int, ...]  # 结算时间 ms
    expected_last_funding_ms: int  # cutoff 前最后一个应有的结算时间
    volume_window_end_ms: int  # 成交额窗口最后一根闭合 4h K 线的闭合时间
    quote_volume_3d_avg: Decimal
    fetched_ms: int
    error: str = ""


@dataclass(frozen=True, slots=True)
class ScanEpoch:
    """一次全量候选横截面。READY 后完全不可变。

    不变量：expected symbol 必须恰有 candidate snapshot 或确定性 excluded；
    failed 非空不可 READY。
    """

    epoch_id: str
    universe_snapshot_ts_ms: int
    decision_cutoff_ms: int
    expected_symbols: tuple[str, ...]
    excluded: Mapping[str, str]
    failed: Mapping[str, str]
    funding_last_ts_ms: Mapping[str, int]  # symbol -> 最后一次结算 ms（EXPIRED 判定）
    funding_interval_hours: Mapping[str, int]  # symbol -> 结算周期
    status: ScanEpochStatus
    created_ms: int
    completed_ms: int | None
    expires_ms: int
    error: str = ""


@dataclass(frozen=True, slots=True)
class DataReadiness:
    """epoch 就绪度（service/web 查询用）。can_rank = 且仅是 READY。"""

    epoch_id: str
    status: ScanEpochStatus
    expected: int
    completed: int
    excluded_count: int
    failed_count: int
    missing: tuple[str, ...]
    failed: Mapping[str, str]
    age_ms: int
    reason: str

    @property
    def can_rank(self) -> bool:
        return self.status is ScanEpochStatus.READY


@dataclass
class _EpochBuilder:
    """BUILDING 期间的可变状态；封存为 ScanEpoch 后丢弃。"""

    epoch_id: str
    universe_snapshot_ts_ms: int
    decision_cutoff_ms: int
    expected_symbols: tuple[str, ...]
    excluded: dict[str, str]
    failed: dict[str, str]
    created_ms: int
    expires_ms: int
    snapshots: dict[str, CandidateSnapshot] = field(default_factory=dict)
    error: str = ""


class EpochDataProvider(Protocol):
    """epoch 构建所需的数据源（live 层可依赖 data 层；单测用 fake）。

    与 ``StrategyDataProvider`` 的区别：``funding_rates`` /
    ``quote_volume_3d_avg`` 支持 ``end_ms``，把窗口固定到 cutoff 前。
    """

    def funding_rates(
        self, symbol: str, periods: int, *, end_ms: int | None = None
    ) -> list[tuple[int, Decimal, Decimal]]:
        """升序 (结算时间戳 ms, 单期费率, 结算标记价)；``end_ms`` 给定只返回 <= end_ms。"""

    def funding_interval_hours(self, symbol: str) -> int:
        """该合约的资金费结算周期（小时），必须来自 fundingInfo。"""

    def quote_volume_3d_avg(self, symbol: str, *, end_ms: int | None = None) -> Decimal:
        """3 天平均日成交额（USDT）；``end_ms`` 给定只使用闭合于其前的 4h K 线。"""

    def tradable_universe(self) -> tuple[str, ...]:
        """全部可交易 USDT 永续 symbol（已排除 exclude_bases）。"""

    def quote_volume_24h(self) -> dict[str, float]:
        """全市场各 symbol 的 24h 成交额（USDT）。"""


def _accepts_end_ms(fn: Any) -> bool:
    """provider 方法是否支持 end_ms 关键字（旧 fake 兼容判断）。"""
    try:
        params = signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "end_ms" in params or any(
        p.kind is p.VAR_KEYWORD for p in params.values()
    )


class MarketDataSynchronizer:
    """后台单 worker + 有界并发的 epoch 构建器。

    公开生命周期接口只有 ``start() / stop() / trigger_refresh() /
    latest_ready() / readiness()``。交易主循环只读 ``latest_ready()``
    返回的不可变 ScanEpoch。

    Args:
        config: 顶层配置（universe 门槛、并发、deadline、top K）。
        data: 数据源（注入）。
        server_time_fn: 交易所校准时间（ms）；决定 decision_cutoff_ms。
            缺省用本地 now（测试/离线）。
        now_fn: 本地时钟（秒）。
        excluded_symbols: 确定性 excluded（无共同规则/非 TRADING 等，带 reason）；
            由 service 在装配时传入，universe 内命中即 excluded 而非 failed。
        exclusion_fn: 动态排除判定（symbol → reason 或 None）；每次 epoch 构建时
            对每个候选调用，命中即 excluded。用于可开仓对预筛（无现货/合约
            交易对、非 TRADING、最小名义额超 canary）——未接入时这些币会进入
            后续筛选并永久占用开仓槽位（demo 受限对实测）。
    """

    def __init__(
        self,
        *,
        config: Config,
        data: EpochDataProvider,
        server_time_fn: Callable[[], int] | None = None,
        now_fn: Callable[[], float] = time.time,
        excluded_symbols: Mapping[str, str] | None = None,
        exclusion_fn: Callable[[str], str | None] | None = None,
    ) -> None:
        self._config = config
        self._data = data
        self._now = now_fn
        self._server_time = server_time_fn
        self._excluded_symbols: dict[str, str] = dict(excluded_symbols or {})
        self._exclusion_fn = exclusion_fn

        self._lock = threading.Lock()
        self._building: _EpochBuilder | None = None
        self._latest: ScanEpoch | None = None  # 任意状态的最新 epoch（含 EXPIRED 标记）
        self._last_universe_ts_ms = 0
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._epoch_seq = 0
        self._epoch_snapshots: dict[str, dict[str, CandidateSnapshot]] = {}
        # 旧版 fake provider 可能不支持 end_ms 关键字；不支持时由 synchronizer
        # 自行按 cutoff 过滤（语义等价）。
        self._funding_supports_end_ms = _accepts_end_ms(data.funding_rates)
        self._volume_supports_end_ms = _accepts_end_ms(data.quote_volume_3d_avg)

    # -- 生命周期 -------------------------------------------------------------

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop, name="market-data-sync", daemon=True
        )
        self._worker.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=timeout)
            if worker.is_alive():
                logger.warning("market-data-sync worker 未在 %.0fs 内退出（继续安全停机）", timeout)
        self._worker = None

    # -- 查询 -----------------------------------------------------------------

    def latest_ready(self) -> ScanEpoch | None:
        """最近 READY epoch（不可变）。惰性标记 EXPIRED 后返回 None。"""
        self._expire_ready_if_needed()
        with self._lock:
            epoch = self._latest
        if epoch is None or epoch.status is not ScanEpochStatus.READY:
            return None
        return epoch

    def latest(self) -> ScanEpoch | None:
        """任意状态的最新 epoch（展示/诊断用）。"""
        with self._lock:
            return self._latest

    def readiness(self, now_ms: int | None = None) -> DataReadiness | None:
        """当前 epoch 就绪度。无 epoch 时 None。"""
        now = now_ms if now_ms is not None else int(self._now() * 1000)
        with self._lock:
            epoch = self._latest
            building = self._building
        if epoch is None and building is None:
            return None
        if building is not None:
            expected = len(building.expected_symbols)
            completed = len(building.snapshots) - len(building.failed)
            return DataReadiness(
                epoch_id=building.epoch_id,
                status=ScanEpochStatus.BUILDING,
                expected=expected,
                completed=completed,
                excluded_count=len(building.excluded),
                failed_count=len(building.failed),
                missing=tuple(
                    s
                    for s in building.expected_symbols
                    if s not in building.snapshots and s not in building.excluded
                ),
                failed=dict(building.failed),
                age_ms=max(0, now - building.created_ms),
                reason="构建中（同一 cutoff 全候选横截面未完成）",
            )
        assert epoch is not None
        completed = len(epoch.expected_symbols) - len(epoch.excluded) - len(epoch.failed)
        return DataReadiness(
            epoch_id=epoch.epoch_id,
            status=epoch.status,
            expected=len(epoch.expected_symbols),
            completed=completed,
            excluded_count=len(epoch.excluded),
            failed_count=len(epoch.failed),
            missing=(),
            failed=dict(epoch.failed),
            age_ms=max(0, now - (epoch.completed_ms or epoch.created_ms)),
            reason={
                ScanEpochStatus.READY: "完整横截面，允许横向排名",
                ScanEpochStatus.DEGRADED: f"截止 {epoch.expires_ms} 未完整：{epoch.error}",
                ScanEpochStatus.EXPIRED: "cutoff 后出现新结算/K 线闭合，禁止新增风险",
                ScanEpochStatus.BUILDING: "构建中",
            }[epoch.status],
        )

    # -- 触发 -----------------------------------------------------------------

    def trigger_refresh(self) -> str | None:
        """请求一次全量构建（由 service tick 调用，内部按 universe 周期节流）。

        返回本次启动/已存在的 epoch_id；universe 未到期时返回现有 epoch_id。
        """
        now_ms = int(self._now() * 1000)
        interval_ms = int(self._config.execution.universe_refresh_seconds * 1000)
        with self._lock:
            if self._building is not None:
                return self._building.epoch_id
            epoch = self._latest
            if epoch is None:
                builder = self._start_building_locked(now_ms)
            elif epoch.status is ScanEpochStatus.READY:
                if self._is_expired(epoch, now_ms):
                    self._mark_expired_now_locked(epoch)
                    builder = self._start_building_locked(now_ms)
                elif now_ms - epoch.universe_snapshot_ts_ms < interval_ms:
                    return epoch.epoch_id
                else:
                    builder = self._start_building_locked(now_ms)
            else:
                # DEGRADED/EXPIRED：等满 universe 周期再重建（避免限流空转）
                if now_ms - epoch.universe_snapshot_ts_ms < interval_ms:
                    return epoch.epoch_id
                builder = self._start_building_locked(now_ms)
            return builder.epoch_id

    def build_once(self) -> str | None:
        """同步执行一次完整 epoch 构建（worker 与单测共用入口）。

        无活跃构建时启动并执行；返回 epoch_id。universe 失败抛
        ``EpochBuildError``（保留上一 READY 展示，交易闸门按未就绪处理）。
        """
        now_ms = int(self._now() * 1000)
        with self._lock:
            if self._building is None:
                self._start_building_locked(now_ms)
            builder = self._building
        assert builder is not None
        self._fill_builder(builder)
        return builder.epoch_id

    # -- 内部 -----------------------------------------------------------------

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.trigger_refresh()
                self._expire_ready_if_needed()
                with self._lock:
                    builder = self._building
                if builder is not None:
                    self._fill_builder(builder)
                self._mark_degraded_if_due()
            except EpochBuildError:
                pass  # 保留上一 READY；下一轮 universe 再试
            except Exception as exc:  # noqa: BLE001 —— worker 不得死掉
                logger.warning("market-data-sync worker 异常（下一轮重试）: %s", exc)
            self._stop_event.wait(1.0)

    def _mark_expired_now_locked(self, epoch: ScanEpoch) -> None:
        """调用方已持有 lock 的 EXPIRED 替换。"""
        expired = replace(epoch, status=ScanEpochStatus.EXPIRED)
        if self._latest is epoch:
            self._latest = expired
        logger.warning("scan epoch %s 已 EXPIRED（cutoff 后出现新 K 线闭合/结算），禁止新增风险", epoch.epoch_id)

    def _start_building_locked(self, now_ms: int) -> _EpochBuilder:
        """调用方持有 lock。universe 快照 + cutoff + expected/excluded 划分。"""
        try:
            universe = tuple(self._data.tradable_universe())
            volumes = {str(k): float(v) for k, v in self._data.quote_volume_24h().items()}
        except Exception as exc:
            raise EpochBuildError(f"universe 快照失败: {type(exc).__name__}: {exc}") from exc

        selection = self._config.strategy.selection
        min_volume = Decimal(str(selection.min_quote_volume_3d_avg))
        pool = [s for s in universe if Decimal(str(volumes.get(s, 0.0))) >= min_volume]
        pool.sort(key=lambda s: volumes.get(s, 0.0), reverse=True)
        max_n = int(self._config.execution.candidate_pool_max_symbols)
        if max_n > 0:
            pool = pool[:max_n]

        cutoff_ms = self._server_time() if self._server_time is not None else now_ms
        self._epoch_seq += 1
        epoch_id = f"ep-{now_ms}-{self._epoch_seq}"
        deadline_ms = int(self._config.execution.scan_epoch_deadline_seconds * 1000)

        excluded: dict[str, str] = {}
        expected: list[str] = []
        for symbol in pool:
            reason = self._excluded_symbols.get(symbol)
            if reason is None and self._exclusion_fn is not None:
                try:
                    reason = self._exclusion_fn(symbol)
                except Exception:  # noqa: BLE001 —— fn 异常 ≠ 确定性排除，交给后续闸门
                    logger.warning("可开仓预筛 %s 异常（不剔除）", symbol, exc_info=True)
                    reason = None
            if reason is not None:
                excluded[symbol] = reason
            else:
                expected.append(symbol)

        builder = _EpochBuilder(
            epoch_id=epoch_id,
            universe_snapshot_ts_ms=now_ms,
            decision_cutoff_ms=cutoff_ms,
            expected_symbols=tuple(sorted(expected)),
            excluded=excluded,
            failed={},
            created_ms=now_ms,
            expires_ms=now_ms + deadline_ms,
        )
        self._building = builder
        self._last_universe_ts_ms = now_ms
        logger.info(
            "scan epoch %s 开始构建: expected=%d excluded=%d cutoff=%d",
            epoch_id, len(expected), len(excluded), cutoff_ms,
        )
        return builder

    def _fill_builder(self, builder: _EpochBuilder) -> None:
        """按有界并发补齐 expected 候选；全部到齐则封存 READY。

        每轮 fill 是一个独立的完整横截面尝试：重置上轮的 snapshots/failed
        后重新拉取（失败 symbol 在 deadline 内自然重试）。
        """
        builder.snapshots = {}
        builder.failed = {}
        if not builder.expected_symbols:
            self._seal(builder, now_ms=builder.created_ms)
            return
        concurrency = int(self._config.execution.candidate_refresh_concurrency)
        entry = self._config.strategy.entry
        exit_cfg = self._config.strategy.exit
        periods = max(
            entry.lookback_periods + entry.min_consecutive_positive,
            exit_cfg.exit_lookback_periods,
        ) + 5

        def fetch(symbol: str) -> CandidateSnapshot:
            return self._fetch_candidate(builder, symbol, periods)

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            results = list(pool.map(fetch, builder.expected_symbols))

        for snapshot in results:
            if snapshot.error:
                builder.failed[snapshot.symbol] = snapshot.error
            else:
                builder.snapshots[snapshot.symbol] = snapshot
        # failed 与 completed 互斥：失败的不进 snapshots
        for symbol in list(builder.failed):
            builder.snapshots.pop(symbol, None)
        # 上一轮失败、本轮成功的：清掉失败标记
        for symbol in list(builder.failed):
            if symbol in builder.snapshots:
                builder.failed.pop(symbol, None)
        newly_ready = not builder.failed

        with self._lock:
            still_active = self._building is builder
        if not still_active:
            return  # 已被取消/替换（不应发生，单 worker 串行）
        now_ms = int(self._now() * 1000)
        if newly_ready:
            self._seal(builder, now_ms=now_ms)
        elif now_ms >= builder.expires_ms:
            builder.error = "截止时仍未完整（见 failed 明细）"
            self._seal(builder, now_ms=now_ms, status=ScanEpochStatus.DEGRADED)

    def _fetch_candidate(
        self, builder: _EpochBuilder, symbol: str, periods: int
    ) -> CandidateSnapshot:
        """拉取单个候选并校验 cutoff 不变量；任何失败带 error 返回（failed 而非 excluded）。

        费率统一聚合到 8h 结算桶（与回测同口径）：4h 币每桶 2 期、1h 币
        每桶 8 期，桶费率 = 桶内求和。候选的「期」一律 = 8h 日历天，
        避免不同周期币的决策窗口/持有期/交易频率语义漂移。
        """
        cutoff = builder.decision_cutoff_ms
        try:
            # 原始事件数 = 目标 8h 桶数 × 每桶最多 8 期（1h 币），保证桶数充足
            raw_periods = periods * 8
            if self._funding_supports_end_ms:
                raw = self._data.funding_rates(symbol, raw_periods, end_ms=cutoff)
            else:
                raw = self._data.funding_rates(symbol, raw_periods)
            # 不变量 1：只用 cutoff 前已发生的结算（synchronizer 兜底过滤）
            raw = [(ts, r, m) for ts, r, m in raw if ts <= cutoff]
            if not raw:
                raise ValueError("无资金费历史")
            # 不变量 2：统一 8h 桶；结束点 > cutoff 的桶未可见（无前瞻）
            timestamps, bucket_rates, marks = normalize_funding_records_to_8h(
                raw, cutoff_ms=cutoff
            )
            if not timestamps:
                raise ValueError("无已可见的 8h 结算桶")
            if timestamps[-1] > cutoff:  # 防御：上面已过滤，正常不可达
                raise ValueError(f"funding 桶 {timestamps[-1]} 晚于 cutoff {cutoff}")
            # 不变量 3：cutoff 前应可见的最后一个 8h 桶必须已到账（结算滞后 = failed）
            expected_last = timestamps[-1]
            next_settle = timestamps[-1] + 8 * 3600 * 1000
            if next_settle <= cutoff:
                raise ValueError(
                    f"应有 8h 桶 {next_settle} 未到账（cutoff={cutoff}），结算滞后"
                )
            # 取「闭合时间 <= cutoff 的最后一根 4h K 线」的闭合时间
            volume_end_ms = (cutoff // _KLINE_INTERVAL_MS + 1) * _KLINE_INTERVAL_MS
            if volume_end_ms > cutoff:
                volume_end_ms -= _KLINE_INTERVAL_MS
            if self._volume_supports_end_ms:
                volume = self._data.quote_volume_3d_avg(symbol, end_ms=volume_end_ms)
            else:
                volume = self._data.quote_volume_3d_avg(symbol)
        except Exception as exc:  # noqa: BLE001 —— failed 语义：网络/限流/解析/结算滞后
            return CandidateSnapshot(
                epoch_id=builder.epoch_id,
                symbol=symbol,
                interval_hours=8,
                rates=(),
                mark_prices=(),
                timestamps=(),
                expected_last_funding_ms=0,
                volume_window_end_ms=0,
                quote_volume_3d_avg=Decimal("0"),
                fetched_ms=int(self._now() * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
        return CandidateSnapshot(
            epoch_id=builder.epoch_id,
            symbol=symbol,
            interval_hours=8,
            rates=tuple(bucket_rates),
            mark_prices=tuple(marks),
            timestamps=tuple(timestamps),
            expected_last_funding_ms=expected_last,
            volume_window_end_ms=volume_end_ms,
            quote_volume_3d_avg=Decimal(str(volume)),
            fetched_ms=int(self._now() * 1000),
            error="",
        )

    def _seal(
        self,
        builder: _EpochBuilder,
        *,
        now_ms: int,
        status: ScanEpochStatus | None = None,
    ) -> None:
        """原子封存：BUILDING 不可交易对象 → 不可变 ScanEpoch。"""
        if status is None:
            status = ScanEpochStatus.READY
        last_ts: dict[str, int] = {}
        intervals: dict[str, int] = {}
        for symbol, snap in builder.snapshots.items():
            if snap.timestamps:
                last_ts[symbol] = int(snap.timestamps[-1])
                intervals[symbol] = int(snap.interval_hours)
        epoch = ScanEpoch(
            epoch_id=builder.epoch_id,
            universe_snapshot_ts_ms=builder.universe_snapshot_ts_ms,
            decision_cutoff_ms=builder.decision_cutoff_ms,
            expected_symbols=builder.expected_symbols,
            excluded=dict(builder.excluded),
            failed=dict(builder.failed),
            funding_last_ts_ms=last_ts,
            funding_interval_hours=intervals,
            status=status,
            created_ms=builder.created_ms,
            completed_ms=now_ms if status is ScanEpochStatus.READY else None,
            expires_ms=builder.expires_ms,
            error=builder.error,
        )
        with self._lock:
            self._building = None
            self._latest = epoch
            if status is ScanEpochStatus.READY:
                self._epoch_snapshots[epoch.epoch_id] = dict(builder.snapshots)
            else:
                self._epoch_snapshots.pop(epoch.epoch_id, None)
            self._trim_locked()
        logger.info(
            "scan epoch %s 封存为 %s（expected=%d excluded=%d failed=%d）",
            epoch.epoch_id, status.value,
            len(epoch.expected_symbols), len(epoch.excluded), len(builder.failed),
        )

    def _trim_locked(self) -> None:
        """内存保留策略：最多保留最近 _MAX_KEPT_EPOCHS 个 epoch 的快照（有界）。"""
        kept = list(self._epoch_snapshots)
        for epoch_id in kept[:-_MAX_KEPT_EPOCHS]:
            self._epoch_snapshots.pop(epoch_id, None)

    def _mark_degraded_if_due(self) -> None:
        """BUILDING 超过 deadline → DEGRADED（不放行交易）。"""
        with self._lock:
            builder = self._building
            if builder is None:
                return
            if int(self._now() * 1000) < builder.expires_ms:
                return
        builder.error = "截止时仍未完整（见 failed 明细）"
        self._seal(builder, now_ms=int(self._now() * 1000), status=ScanEpochStatus.DEGRADED)

    # -- 过期判定 --------------------------------------------------------------

    def _is_expired(self, epoch: ScanEpoch, now_ms: int) -> bool:
        """cutoff 后是否已出现任一候选应有的新 4h K 线闭合或新 funding 结算。

        4h K 线：now 跨过 cutoff 之后任何一个 4h 边界即过期（成交额横截面已旧）。
        funding：对每个候选用其 (最后一行时间, 周期) 推算下一次结算时间，
        若 <= now 则过期（收益率横截面已旧）。
        """
        cutoff = epoch.decision_cutoff_ms
        if now_ms <= cutoff:
            return False
        # 4h K 线边界：cutoff 之后第一个闭合点
        next_kline_close = (cutoff // _KLINE_INTERVAL_MS + 1) * _KLINE_INTERVAL_MS
        if now_ms >= next_kline_close:
            return True
        # funding：任一候选在 cutoff 之后已到期下一次结算 → 收益率横截面已旧
        for symbol, last_ts in epoch.funding_last_ts_ms.items():
            next_settle = (
                last_ts + epoch.funding_interval_hours.get(symbol, 8) * 3600 * 1000
            )
            if next_settle > cutoff and next_settle <= now_ms:
                return True
        return False

    def _expire_ready_if_needed(self) -> None:
        with self._lock:
            epoch = self._latest
        if epoch is None or epoch.status is not ScanEpochStatus.READY:
            return
        now_ms = int(self._now() * 1000)
        if self._is_expired(epoch, now_ms):
            self._mark_expired(epoch)

    def _mark_expired(self, epoch: ScanEpoch) -> None:
        """将 epoch 标记 EXPIRED（原子替换 _latest；调用方不得已持 lock）。"""
        expired = replace(epoch, status=ScanEpochStatus.EXPIRED)
        with self._lock:
            if self._latest is epoch:
                self._latest = expired
        logger.warning("scan epoch %s 已 EXPIRED（cutoff 后出现新 K 线闭合/结算），禁止新增风险", epoch.epoch_id)

    # -- epoch 快照读取 ----------------------------------------------------------

    def expected_symbols(self) -> tuple[str, ...]:
        """当前构建中/最新 epoch 的 expected 候选（每个 symbol 恰好一条决策用）。"""
        with self._lock:
            if self._building is not None:
                return self._building.expected_symbols
            if self._latest is not None:
                return self._latest.expected_symbols
        return ()

    def snapshots_for(self, epoch_id: str) -> Mapping[str, CandidateSnapshot]:
        """某 epoch 的候选快照（READY 封存后的不可变视图）。"""
        with self._lock:
            return dict(self._epoch_snapshots.get(epoch_id, ()))
