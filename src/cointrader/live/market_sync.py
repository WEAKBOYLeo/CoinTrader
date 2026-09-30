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
from ..strategy.funding_carry import CandidateInput, FundingCarryEvaluator

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


_KNOWN_SETTLE_INTERVALS_MS = (3600 * 1000, 4 * 3600 * 1000, 8 * 3600 * 1000)


def _infer_settle_interval_ms(raw: list[tuple[int, Decimal, Decimal]]) -> int:
    """从连续结算记录推断结算间隔（1h/4h/8h），未知时回退 8h。

    仅在 fundingInfo 无声明周期时的兜底：动态周期币历史会混合 1h/4h/8h
    间隔，最小间隔推断会把健康币误判为滞后，优先用声明值。
    """
    gaps = [
        b[0] - a[0]
        for a, b in zip(raw, raw[1:], strict=False)
        if b[0] > a[0]
    ]
    if not gaps:
        return 8 * 3600 * 1000
    return min(_KNOWN_SETTLE_INTERVALS_MS, key=lambda k: abs(k - min(gaps)))


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
    # Funding 历史统一到 8h 桶后保留原始结算周期与最近已结算事件，
    # 供监控展示，不参与新的策略判定。
    settle_interval_hours: int = 8
    latest_settled_rate: Decimal | None = None
    latest_settled_ms: int | None = None
    # ``NOT_REQUESTED`` is meaningful: funding pre筛未通过时没有请求 3 日 K 线，
    # 不能把默认 0 渲染成真实成交额。
    volume_status: str = "NOT_REQUESTED"


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


def _declared_interval_hours(data: Any, symbol: str) -> int | None:
    """provider 声明的结算周期（小时）；不支持/异常时 None。"""
    fn = getattr(data, "funding_interval_hours", None)
    if fn is None:
        return None
    try:
        hours = int(fn(symbol))
    except Exception:  # noqa: BLE001 —— 兜底推断，不得阻断候选
        return None
    return hours if hours > 0 else None


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
        long_history_symbols_fn: Callable[[], set[str]] | None = None,
        epoch_persist_fn: Callable[[ScanEpoch, Mapping[str, CandidateSnapshot]], None] | None = None,
    ) -> None:
        self._config = config
        self._data = data
        self._now = now_fn
        self._server_time = server_time_fn
        self._excluded_symbols: dict[str, str] = dict(excluded_symbols or {})
        self._exclusion_fn = exclusion_fn
        self._long_history_symbols_fn = long_history_symbols_fn
        self._epoch_persist_fn = epoch_persist_fn

        self._lock = threading.Lock()
        self._building: _EpochBuilder | None = None
        self._latest: ScanEpoch | None = None  # 任意状态的最新 epoch（含 EXPIRED 标记）
        self._last_universe_ts_ms = 0
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._epoch_seq = 0
        self._epoch_snapshots: dict[str, dict[str, CandidateSnapshot]] = {}
        self._premium_index: dict[str, dict[str, Any]] = {}
        self._last_premium_refresh_ms = 0
        self._stage_lock = threading.Lock()
        self._stage_stats: dict[str, dict[str, Any]] = {}
        self._stage_perf_started: dict[str, float] = {}
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
            completed = max(0, len(building.snapshots) - len(building.failed))
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

    def _reset_stage_stats(self, now_ms: int) -> None:
        stages = (
            ("universe", "合约池"),
            ("volume_24h", "24h成交额"),
            ("pair_rules", "交易规则"),
            ("funding_data", "资金费数据"),
            ("funding_signal", "资金费信号"),
            ("volume_3d", "3日成交额"),
            ("ready", "READY横截面"),
        )
        with self._stage_lock:
            self._stage_stats = {
                stage_id: {
                    "id": stage_id,
                    "label": label,
                    "status": "WAITING",
                    "input_count": None,
                    "output_count": None,
                    "duration_ms": None,
                    "started_at_ms": None,
                    "completed_at_ms": None,
                    "updated_at_ms": now_ms,
                    "excluded_count": None,
                    "failed_count": None,
                    "error": None,
                }
                for stage_id, label in stages
            }
            self._stage_perf_started = {}

    def _stage_start(self, stage_id: str, *, input_count: int | None = None) -> None:
        now_ms = int(self._now() * 1000)
        with self._stage_lock:
            stage = self._stage_stats.get(stage_id)
            if stage is None:
                return
            stage.update({
                "status": "RUNNING",
                "input_count": input_count,
                "started_at_ms": now_ms,
                "updated_at_ms": now_ms,
                "error": None,
            })
            self._stage_perf_started[stage_id] = time.perf_counter()

    def _stage_done(
        self,
        stage_id: str,
        *,
        output_count: int | None = None,
        excluded_count: int | None = None,
        failed_count: int | None = None,
        error: str | None = None,
    ) -> None:
        now_ms = int(self._now() * 1000)
        with self._stage_lock:
            stage = self._stage_stats.get(stage_id)
            if stage is None:
                return
            started = self._stage_perf_started.pop(stage_id, None)
            stage.update({
                "status": "FAILED" if error else "DONE",
                "output_count": output_count,
                "excluded_count": excluded_count,
                "failed_count": failed_count,
                "completed_at_ms": now_ms,
                "updated_at_ms": now_ms,
                "duration_ms": (
                    round((time.perf_counter() - started) * 1000, 1)
                    if started is not None else None
                ),
                "error": error,
            })

    def _refresh_premium_if_due(self) -> None:
        """按短 TTL 批量刷新最终候选的当前费率，不逐币请求。"""
        now_ms = int(self._now() * 1000)
        if now_ms - self._last_premium_refresh_ms < 5_000:
            return
        with self._lock:
            epoch = self._latest
            snapshots = dict(self._epoch_snapshots.get(epoch.epoch_id, ())) if epoch and epoch.status is ScanEpochStatus.READY else {}
        if not snapshots:
            return
        min_volume = Decimal(str(self._config.strategy.selection.min_quote_volume_3d_avg))
        symbols = tuple(
            sorted(
                symbol for symbol, snapshot in snapshots.items()
                if snapshot.volume_status == "FETCHED"
                and snapshot.quote_volume_3d_avg >= min_volume
                and not snapshot.error
            )
        )
        if symbols:
            self._refresh_premium_index(symbols)
            self._last_premium_refresh_ms = now_ms

    def premium_snapshots(self) -> Mapping[str, dict[str, Any]]:
        """最近一次批量 premiumIndex 快照；WebUI 只读，不触发网络请求。"""
        with self._lock:
            return {symbol: dict(value) for symbol, value in self._premium_index.items()}

    def _refresh_premium_index(self, symbols: tuple[str, ...]) -> None:
        """在 epoch 后台构建阶段批量刷新当前费率。

        provider 没有该能力时保持 UNKNOWN；刷新失败不使历史 epoch 从 READY
        变成失败，避免实时展示反过来改变筛选完整性闸门。
        """
        refresh = getattr(self._data, "refresh_premium_index", None)
        if not callable(refresh) or not symbols:
            return
        try:
            payload = refresh(symbols)
            if not isinstance(payload, Mapping):
                return
            with self._lock:
                self._premium_index = {
                    str(symbol): dict(value)
                    for symbol, value in payload.items()
                    if isinstance(value, Mapping)
                }
        except Exception as exc:  # noqa: BLE001 —— 当前费率失败只局部降级
            logger.warning("premiumIndex 批量刷新失败（候选当前费率显示 UNKNOWN）: %s", exc)

    def stage_readiness(self, now_ms: int | None = None) -> dict[str, Any]:
        """阶段级筛选诊断；只读副本，供 WebUI 展示数量/耗时/新鲜度。"""
        now = now_ms if now_ms is not None else int(self._now() * 1000)
        with self._stage_lock:
            stages = [dict(stage) for stage in self._stage_stats.values()]
        for stage in stages:
            updated = stage.get("updated_at_ms")
            stage["age_ms"] = max(0, now - int(updated)) if updated else None
            stage["quality"] = (
                "OK" if stage["status"] == "DONE"
                else "DEGRADED" if stage["status"] == "FAILED"
                else "STALE" if stage["status"] == "RUNNING"
                else "UNKNOWN"
            )
        latest = max(
            (int(stage["updated_at_ms"]) for stage in stages if stage.get("updated_at_ms")),
            default=0,
        )
        freshness_window = max(
            30_000,
            int(self._config.execution.candidate_refresh_seconds * 2_000),
        )
        return {
            "stages": stages,
            "as_of_ms": now,
            "age_ms": max(0, now - latest) if latest else None,
            "freshness": (
                "UNKNOWN" if not latest
                else "FRESH" if now - latest <= freshness_window
                else "STALE"
            ),
        }

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
                self._refresh_premium_if_due()
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
        self._reset_stage_stats(now_ms)
        self._stage_start("universe")
        try:
            universe = tuple(self._data.tradable_universe())
        except Exception as exc:
            self._stage_done("universe", error=f"{type(exc).__name__}: {exc}")
            raise EpochBuildError(f"universe 快照失败: {type(exc).__name__}: {exc}") from exc
        self._stage_done("universe", output_count=len(universe), excluded_count=0, failed_count=0)

        self._stage_start("volume_24h", input_count=len(universe))
        try:
            volumes = {str(k): float(v) for k, v in self._data.quote_volume_24h().items()}
        except Exception as exc:
            self._stage_done("volume_24h", error=f"{type(exc).__name__}: {exc}")
            raise EpochBuildError(f"universe 快照失败: {type(exc).__name__}: {exc}") from exc

        selection = self._config.strategy.selection
        min_volume = Decimal(str(selection.min_quote_volume_3d_avg))
        pool = [s for s in universe if Decimal(str(volumes.get(s, 0.0))) >= min_volume]
        pool.sort(key=lambda s: volumes.get(s, 0.0), reverse=True)
        max_n = int(self._config.execution.candidate_pool_max_symbols)
        if max_n > 0:
            pool = pool[:max_n]
        self._stage_done(
            "volume_24h",
            output_count=len(pool),
            excluded_count=len(universe) - len(pool),
            failed_count=0,
        )

        self._stage_start("pair_rules", input_count=len(pool))

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
        self._stage_done(
            "pair_rules",
            output_count=len(expected),
            excluded_count=len(excluded),
            failed_count=0,
        )

        cutoff_ms = self._server_time() if self._server_time is not None else now_ms
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
        """先筛 funding，再为通过者补成交额 K 线，减少逐币请求。

        未持仓候选只需入场窗口；持仓候选保留完整退出窗口。成交额是入场
        条件，在 funding 信号筛选后再读取。任何已开始的第二阶段请求失败仍
        计 failed，不能把数据缺失伪装成策略未通过。
        """
        builder.snapshots = {}
        builder.failed = {}
        if not builder.expected_symbols:
            self._stage_done("ready", output_count=0)
            self._seal(builder, now_ms=builder.created_ms)
            return
        self._stage_start("ready", input_count=len(builder.expected_symbols))
        concurrency = int(self._config.execution.candidate_refresh_concurrency)
        entry = self._config.strategy.entry
        exit_cfg = self._config.strategy.exit
        entry_periods = entry.lookback_periods + entry.min_consecutive_positive + 5
        full_periods = max(
            entry.lookback_periods + entry.min_consecutive_positive,
            exit_cfg.exit_lookback_periods,
        ) + 5
        try:
            long_symbols = set(self._long_history_symbols_fn() if self._long_history_symbols_fn else ())
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取持仓长历史 symbol 失败，按无持仓处理: %s", exc)
            long_symbols = set()

        evaluator = FundingCarryEvaluator(config=self._config, now_fn=self._now)

        def fetch_funding(symbol: str) -> CandidateSnapshot:
            periods = full_periods if symbol in long_symbols else entry_periods
            return self._fetch_candidate(builder, symbol, periods, fetch_volume=False)

        self._stage_start("funding_data", input_count=len(builder.expected_symbols))
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            initial = list(pool.map(fetch_funding, builder.expected_symbols))

        for snapshot in initial:
            if snapshot.error:
                builder.failed[snapshot.symbol] = snapshot.error
                continue
            builder.snapshots[snapshot.symbol] = snapshot
        self._stage_done(
            "funding_data",
            output_count=len(builder.snapshots),
            excluded_count=0,
            failed_count=len(builder.failed),
            error=(f"{len(builder.failed)} 个 symbol 失败" if builder.failed else None),
        )

        self._stage_start("funding_signal", input_count=len(builder.snapshots))
        signal_targets: list[CandidateSnapshot] = []
        for snapshot in builder.snapshots.values():
            if self._passes_entry_signal(snapshot, evaluator):
                signal_targets.append(snapshot)
        self._stage_done(
            "funding_signal",
            output_count=len(signal_targets),
            excluded_count=len(builder.snapshots) - len(signal_targets),
            failed_count=0,
        )

        volume_targets = list(signal_targets)
        signal_symbols = {s.symbol for s in signal_targets}
        volume_targets.extend(
            snapshot
            for snapshot in builder.snapshots.values()
            if snapshot.symbol in long_symbols and snapshot.symbol not in signal_symbols
        )

        self._stage_start("volume_3d", input_count=len(volume_targets))
        def fetch_volume(snapshot: CandidateSnapshot) -> CandidateSnapshot:
            return self._fetch_volume(snapshot)

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            completed = list(pool.map(fetch_volume, volume_targets))
        for snapshot in completed:
            if snapshot.error:
                builder.failed[snapshot.symbol] = snapshot.error
            else:
                builder.snapshots[snapshot.symbol] = snapshot
        min_volume = Decimal(str(self._config.strategy.selection.min_quote_volume_3d_avg))
        volume_pass_count = sum(
            1
            for snapshot in completed
            if not snapshot.error and snapshot.quote_volume_3d_avg >= min_volume
        )
        self._stage_done(
            "volume_3d",
            output_count=volume_pass_count,
            excluded_count=sum(
                1
                for snapshot in completed
                if not snapshot.error and snapshot.quote_volume_3d_avg < min_volume
            ),
            failed_count=len(builder.failed),
            error=(f"{len(builder.failed)} 个 symbol 失败" if builder.failed else None),
        )

        # 一次批量请求覆盖最终候选；页面轮询只读该快照，不按币逐个请求。
        final_symbols = tuple(
            sorted(
                snapshot.symbol
                for snapshot in completed
                if not snapshot.error and snapshot.volume_status == "FETCHED"
                and snapshot.quote_volume_3d_avg >= min_volume
            )
        )
        self._refresh_premium_index(final_symbols)

        for symbol in list(builder.failed):
            builder.snapshots.pop(symbol, None)
        newly_ready = not builder.failed

        with self._lock:
            still_active = self._building is builder
        if not still_active:
            return  # 已被取消/替换（不应发生，单 worker 串行）
        now_ms = int(self._now() * 1000)
        if newly_ready:
            self._stage_done(
                "ready",
                output_count=len(builder.snapshots),
                excluded_count=len(builder.excluded),
                failed_count=len(builder.failed),
            )
            self._seal(builder, now_ms=now_ms)
        elif now_ms >= builder.expires_ms:
            builder.error = "截止时仍未完整（见 failed 明细）"
            self._stage_done(
                "ready",
                output_count=len(builder.snapshots),
                excluded_count=len(builder.excluded),
                failed_count=len(builder.failed),
                error=builder.error,
            )
            self._seal(builder, now_ms=now_ms, status=ScanEpochStatus.DEGRADED)

    def _passes_entry_signal(
        self, snapshot: CandidateSnapshot, evaluator: FundingCarryEvaluator
    ) -> bool:
        """只用 funding 指标判断是否值得读取精确 3 日成交额。"""
        candidate = CandidateInput(
            symbol=snapshot.symbol,
            rates=snapshot.rates,
            mark_prices=snapshot.mark_prices,
            timestamps=snapshot.timestamps,
            interval_hours=snapshot.interval_hours,
            volume_3d_avg=Decimal("0"),
            refreshed_ts_ms=snapshot.fetched_ms,
        )
        trailing, streak = evaluator.entry_metrics(candidate)
        threshold = max(
            Decimal(str(self._config.strategy.entry.min_trailing_annualized)),
            Decimal(str(self._config.strategy.entry.min_annualized_rate)),
        )
        return trailing >= threshold and streak >= self._config.strategy.entry.min_consecutive_positive

    def _fetch_volume(self, snapshot: CandidateSnapshot) -> CandidateSnapshot:
        """为已通过 funding 的候选补精确成交额；不重复请求 funding。"""
        try:
            if self._volume_supports_end_ms:
                volume = self._data.quote_volume_3d_avg(
                    snapshot.symbol, end_ms=snapshot.volume_window_end_ms
                )
            else:
                volume = self._data.quote_volume_3d_avg(snapshot.symbol)
            return replace(
                snapshot,
                quote_volume_3d_avg=Decimal(str(volume)),
                fetched_ms=int(self._now() * 1000),
                volume_status="FETCHED",
            )
        except Exception as exc:  # noqa: BLE001 —— 已开始请求，失败必须显式进入 failed
            return replace(
                snapshot,
                error=f"{type(exc).__name__}: {exc}",
                volume_status="FAILED",
            )

    def _fetch_candidate(
        self,
        builder: _EpochBuilder,
        symbol: str,
        periods: int,
        *,
        fetch_volume: bool = True,
    ) -> CandidateSnapshot:
        """拉取 funding，并可选补成交额；任何已请求失败都显式进入 failed。

        funding 与成交额分阶段：未通过入场 funding 条件的候选不请求 K 线。
        """
        cutoff = builder.decision_cutoff_ms
        declared_hours: int | None = None
        raw: list[tuple[int, Decimal, Decimal]] = []
        try:
            # FundingInfo 给出实际结算间隔。8h 桶需要 8/interval 个事件；
            # 周期未知时按最密集的 1h 保守读取，避免漏窗口。
            declared_hours = _declared_interval_hours(self._data, symbol)
            events_per_bucket = (
                8 // declared_hours
                if declared_hours in (1, 2, 4, 8)
                else 8
            )
            raw_periods = periods * events_per_bucket
            if self._funding_supports_end_ms:
                raw = self._data.funding_rates(symbol, raw_periods, end_ms=cutoff)
            else:
                raw = self._data.funding_rates(symbol, raw_periods)
            # 不变量 1：只用 cutoff 前已发生的结算（synchronizer 兜底过滤）
            raw = [(ts, r, m) for ts, r, m in raw if ts <= cutoff]
            if not raw:
                raise ValueError("无资金费历史")
            # 不变量 2：统一 8h 桶；不完整桶（桶内最后结算 > cutoff）已丢弃（无前瞻）
            timestamps, bucket_rates, marks, last_raw_ts = normalize_funding_records_to_8h(
                raw, cutoff_ms=cutoff
            )
            if not timestamps:
                raise ValueError("无已可见的 8h 结算桶")
            if timestamps[-1] > cutoff:  # 防御：上面已过滤，正常不可达
                raise ValueError(f"funding 桶 {timestamps[-1]} 晚于 cutoff {cutoff}")
            # 不变量 3：真实结算滞后 = cutoff 距最后一条到账结算超过 3 个结算
            # 间隔。4h 币最坏情况（最后结算 04:00，下一笔 16:00，cutoff 在
            # (04:00,16:00)）滞后 = 2×间隔+ε，故阈值取 3× 防误判；真滞后
            # （>3 个间隔未出结算）仍能抓住。间隔取 fundingInfo 声明值
            # （动态周期币历史混合间隔，最小间隔推断会误判）；不能用
            # 「cutoff 整点结算未到账」判滞后（刚发生的结算 API 可能未返回）
            settle_ms = (
                declared_hours * 3600 * 1000
                if declared_hours is not None
                else _infer_settle_interval_ms(raw)
            )
            if cutoff - last_raw_ts[-1] > 3 * settle_ms:
                raise ValueError(
                    f"最后结算 {last_raw_ts[-1]} 距 cutoff {cutoff} 超过 3 个结算间隔，结算滞后"
                )
            # 取「闭合时间 <= cutoff 的最后一根 4h K 线」的闭合时间。
            # 第一阶段只记录窗口边界；第二阶段才请求 K 线。
            volume_end_ms = (cutoff // _KLINE_INTERVAL_MS + 1) * _KLINE_INTERVAL_MS
            if volume_end_ms > cutoff:
                volume_end_ms -= _KLINE_INTERVAL_MS
            volume = Decimal("0")
            if fetch_volume:
                if self._volume_supports_end_ms:
                    volume = self._data.quote_volume_3d_avg(
                        symbol, end_ms=volume_end_ms
                    )
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
                settle_interval_hours=declared_hours or 8,
                latest_settled_rate=Decimal(str(raw[-1][1])) if "raw" in locals() and raw else None,
                latest_settled_ms=int(raw[-1][0]) if "raw" in locals() and raw else None,
                volume_status="FAILED" if fetch_volume else "NOT_REQUESTED",
            )
        return CandidateSnapshot(
            epoch_id=builder.epoch_id,
            symbol=symbol,
            interval_hours=8,
            rates=tuple(bucket_rates),
            mark_prices=tuple(marks),
            timestamps=tuple(timestamps),
            expected_last_funding_ms=last_raw_ts[-1],
            volume_window_end_ms=volume_end_ms,
            quote_volume_3d_avg=Decimal(str(volume)),
            fetched_ms=int(self._now() * 1000),
            error="",
            settle_interval_hours=declared_hours or 8,
            latest_settled_rate=Decimal(str(raw[-1][1])),
            latest_settled_ms=int(raw[-1][0]),
            volume_status="FETCHED" if fetch_volume else "NOT_REQUESTED",
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
                intervals[symbol] = int(snap.settle_interval_hours)
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
        if self._epoch_persist_fn is not None:
            try:
                self._epoch_persist_fn(epoch, dict(builder.snapshots))
            except Exception as exc:  # noqa: BLE001 —— 持久化失败不伪造 READY，但不阻断内存闸门
                logger.warning("scan epoch %s 诊断持久化失败: %s", epoch.epoch_id, exc)
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
