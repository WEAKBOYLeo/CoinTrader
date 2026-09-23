"""历史行情区间覆盖索引与缺口复用（实施计划书 v5.0 T1）。

在既有 ``DiskCache`` 之上提供**可验证的完整区间索引**：请求一个历史时间
范围时，只向交易所拉取尚未覆盖的缺口（gap），已覆盖部分直接从本地段
（segment）读取；重复请求相同范围零网络。

硬语义（与计划 §4.2 一致）：

1. **半开区间**：请求范围 ``[start_ms, end_ms)``。Kline 身份 = open time，
   Funding 身份 = ``fundingTime``；只保留 ``start_ms <= 身份 < end_ms``
   的记录（legacy 无界路径仍保持原有含端语义，不受影响）。
2. **闭合边界**：只有已闭合 Kline 才能进入历史覆盖。
   闭合 ⇔ ``open_time + interval_ms <= 交易所当前时间``（``futures_time()``）。
   未闭合 candle 既不返回、也不标记覆盖；Funding 事件要求
   ``fundingTime < 交易所当前时间``（含毫秒抖动保守处理）。
3. **覆盖只前进到已验证的完整段**：段数据先写 ``hist_segments_v1`` 并读回
   重算 checksum，成功后才把段并入索引（``coverage_v1``，原子替换）。
   请求/解析/限流错误绝不推进覆盖（该 gap 留在 missing）；索引/段损坏、
   版本不符一律按 miss 重拉，绝不把磁盘文件存在当作覆盖证据。
4. **single-flight**：同一进程内相同 key 的并发调用共享一把锁，先完成者
   发布覆盖，后来者直接命中，不产生重复拉取或假完整。跨进程语义不做
   承诺（原子文件替换 + 保守合并的最坏结果是重复拉取，不是假完整）。

磁盘布局（均经 ``DiskCache`` 原子写，路径只用 hash，绝不拼原始 symbol）::

    <cache_dir>/coverage_v1/<key_hash>.json          每 key 一个覆盖索引
    <cache_dir>/hist_segments_v1/<segment_key>.json  每段一个不可变记录集
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..domain.market import DataQuality
from ..errors import ParseError
from .cache import DiskCache, make_key

if TYPE_CHECKING:  # 仅类型引用，避免 data/coverage ↔ data/binance 循环 import
    from .binance import BinancePublicClient

logger = logging.getLogger(__name__)

#: 覆盖元数据格式版本。结构变更时递增，让旧索引整体失效（按 miss 重拉）。
COVERAGE_FORMAT_VERSION = 1

#: 覆盖索引 namespace（每 CoverageKey 一个文件，key = key_hash）。
INDEX_NAMESPACE = "coverage_v1"
#: 历史段数据 namespace（每段一个文件，ttl=None 永不过期）。
SEGMENT_NAMESPACE = "hist_segments_v1"

#: 常被使用的 K 线周期 → 毫秒数。回测对齐资金费结算时间、闭合判定都要用。
#: （由 data/klines.py 迁入本模块，klines.py 保留 re-export 不破坏既有 import。）
INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

#: 半开区间 ``(start_ms, end_ms)``，start < end。
Interval = tuple[int, int]

_DATASET_KLINES = "futures_klines"
_DATASET_FUNDING = "funding_history"


# ---------------------------------------------------------------------------
# 区间运算
# ---------------------------------------------------------------------------


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """合并半开区间集为不相交、升序、无冗余的最小集合。

    相邻区间（``[a,b)`` 与 ``[b,c)``）合并为 ``[a,c)``；退化区间
    （start >= end）被忽略。
    """
    valid = [(int(s), int(e)) for s, e in intervals if int(e) > int(s)]
    if not valid:
        return []
    valid.sort()
    merged: list[Interval] = [valid[0]]
    for start, end in valid[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def clip_to_requested(interval: Interval, requested: Interval) -> Interval | None:
    """两区间交集；不相交返回 None。"""
    start = max(interval[0], requested[0])
    end = min(interval[1], requested[1])
    return (start, end) if end > start else None


def subtract_intervals(requested: Interval, covered: Iterable[Interval]) -> list[Interval]:
    """``requested`` 减去 ``union(covered)`` 的缺口集（升序、不相交）。

    covered 中超出 requested 的部分自动裁剪。
    """
    start, end = requested
    if end <= start:
        raise ValueError(f"requested 区间非法: {requested}")
    result: list[Interval] = []
    cursor = start
    for seg_start, seg_end in merge_intervals(covered):
        clipped = clip_to_requested((seg_start, seg_end), requested)
        if clipped is None:
            continue
        if clipped[0] > cursor:
            result.append((cursor, clipped[0]))
        cursor = max(cursor, clipped[1])
        if cursor >= end:
            break
    if cursor < end:
        result.append((cursor, end))
    return result


# ---------------------------------------------------------------------------
# 覆盖键 / 段 / 结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoverageKey:
    """一个可复用历史数据集的规范键。

    所有会影响响应内容的参数都在这里表示：venue / market / dataset /
    symbol / interval / 格式版本。路径安全：文件 key 只用 ``key_hash()``
    （规范化 JSON 的 sha256 前 32 位 hex），绝不拼接原始 symbol。
    """

    venue: str
    market: str
    dataset: str
    symbol: str
    interval: str = ""
    format_version: int = COVERAGE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.venue, str) or not self.venue:
            raise ValueError("CoverageKey.venue 不能为空")
        if not isinstance(self.market, str) or not self.market:
            raise ValueError("CoverageKey.market 不能为空")
        if self.dataset not in (_DATASET_KLINES, _DATASET_FUNDING):
            raise ValueError(f"CoverageKey.dataset 不支持: {self.dataset!r}")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("CoverageKey.symbol 不能为空")
        if self.dataset == _DATASET_KLINES and self.interval not in INTERVAL_MS:
            raise ValueError(f"未知 K 线周期: {self.interval!r}")
        if self.dataset == _DATASET_FUNDING and self.interval:
            raise ValueError("funding_history 数据集不带 interval")

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue": self.venue,
            "market": self.market,
            "dataset": self.dataset,
            "symbol": self.symbol,
            "interval": self.interval,
            "format_version": self.format_version,
        }

    def key_hash(self) -> str:
        """规范化 JSON → sha256 前 32 位 hex。仅作文件 key，无密码学用途。"""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class CoverageSegment:
    """一段**已验证完整**的历史记录（半开区间 ``[start_ms, end_ms)``）。

    只有段数据原子落盘并通过读回 checksum 验证后才会出现在索引里；
    ``complete=False`` 的段绝不写入索引（defensive：读到的也丢弃）。
    """

    start_ms: int
    end_ms: int
    segment_key: str
    record_count: int
    checksum: str
    complete: bool
    format_version: int = COVERAGE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.end_ms <= self.start_ms:
            raise ValueError(f"CoverageSegment 必须是有效半开区间: {self.start_ms}..{self.end_ms}")
        if self.record_count < 0:
            raise ValueError(f"record_count 不能为负: {self.record_count}")
        if not self.segment_key:
            raise ValueError("segment_key 不能为空")
        if not self.checksum:
            raise ValueError("checksum 不能为空")

    @property
    def range(self) -> Interval:
        return (self.start_ms, self.end_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "segment_key": self.segment_key,
            "record_count": self.record_count,
            "checksum": self.checksum,
            "complete": self.complete,
            "format_version": self.format_version,
        }

    @classmethod
    def from_dict(cls, data: Any) -> CoverageSegment:
        """解析索引里的段元数据；任何字段缺失/类型错误抛 ValueError（按 miss 处理）。"""
        if not isinstance(data, dict):
            raise ValueError(f"段元数据必须是对象: {type(data).__name__}")
        return cls(
            start_ms=int(data["start_ms"]),
            end_ms=int(data["end_ms"]),
            segment_key=str(data["segment_key"]),
            record_count=int(data["record_count"]),
            checksum=str(data["checksum"]),
            complete=bool(data["complete"]),
            format_version=int(data["format_version"]),
        )


@dataclass(frozen=True, slots=True)
class HistoricalDatasetResult:
    """一次历史区间请求的完整结果（不可变）。

    - ``records``：按身份时间升序、去重、满足半开边界与闭合条件的记录。
    - ``covered_ranges`` / ``missing_ranges``：相对 ``requested_range`` 的
      覆盖/缺口（半开区间，升序不相交）。
    - ``quality``：FRESH = 全覆盖且无 IO 故障；INCOMPLETE = 仍有缺口
      （未闭合尾段 / 记录数上限截断 / 拉取失败后重试的中间态）；
      DEGRADED = 本次已补齐但发生过段读回校验失败或写失败（可见诊断）。
    - ``cache_hits``：命中的已缓存段数（区别于短 TTL 实时缓存的
      ``ClientStats.cache_hits``）；``network_pages``：本次发出的数据页数。
    """

    records: list[Any]
    requested_range: Interval
    covered_ranges: list[Interval]
    missing_ranges: list[Interval]
    quality: DataQuality
    cache_hits: int
    network_pages: int

    @property
    def is_complete(self) -> bool:
        return not self.missing_ranges


# ---------------------------------------------------------------------------
# 记录身份与校验
# ---------------------------------------------------------------------------


def kline_identity(candle: Any) -> int:
    """Kline 身份 = open time（毫秒）。字段异常抛 ParseError（不静默跳过）。"""
    if not isinstance(candle, (list, tuple)) or len(candle) < 12:
        snippet = candle[:2] if isinstance(candle, (list, tuple)) else candle
        raise ParseError(f"K 线字段数异常（期望 >=12）: {snippet!r}")
    try:
        return int(candle[0])
    except (TypeError, ValueError) as exc:
        raise ParseError(f"K 线 open_time 非法: {candle[0]!r}") from exc


def funding_identity(record: Any) -> int:
    """Funding 身份 = fundingTime（毫秒）。字段缺失/非法抛 ParseError。"""
    if not isinstance(record, dict):
        raise ParseError(f"fundingRate 记录必须是对象: {record!r}")
    if "fundingTime" not in record:
        raise ParseError(f"fundingRate 响应缺少 fundingTime 字段: {record!r}")
    try:
        return int(record["fundingTime"])
    except (TypeError, ValueError) as exc:
        raise ParseError(f"fundingTime 非法: {record.get('fundingTime')!r}") from exc


def canonical_checksum(records: list[Any]) -> str:
    """记录集的规范 checksum（sha256 of canonical JSON）。

    写段与读回验证必须用同一算法；canonical 形式（sort_keys + 紧凑分隔符）
    对 JSON 原生值（list/dict/str/int/float）是幂等的。
    """
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 仓库
# ---------------------------------------------------------------------------


class HistoricalRepository:
    """历史区间仓库：读覆盖索引 → 只拉缺口 → 验证后发布段。

    线程安全：同一 ``CoverageKey`` 的并发调用经 per-key ``threading.Lock``
    single-flight 串行化（锁粒度 = 数据集键，不锁全局）。跨进程并发不在
    保证范围内——最坏结果是多进程重复拉取同一段（覆盖保守，不会假完整）。
    """

    def __init__(self, client: BinancePublicClient, cache: DiskCache) -> None:
        self._client = client
        self._cache = cache
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # -- single-flight -------------------------------------------------------

    def _lock_for(self, key_hash: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(key_hash)
            if lock is None:
                lock = threading.Lock()
                self._locks[key_hash] = lock
            return lock

    # -- 公开入口 -------------------------------------------------------------

    def fetch_futures_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        *,
        limit: int,
    ) -> HistoricalDatasetResult:
        """永续 K 线历史区间 ``[start_ms, end_ms)``（按 open time，仅已闭合）。

        ``limit`` 是单页条数（同 legacy 语义，交易所上限 1500 由客户端钳制）；
        不是返回条数上限——范围会完整取回（缺口可分多次请求补齐）。
        """
        if start_ms >= end_ms:
            raise ValueError(f"start_ms 必须 < end_ms: {start_ms} >= {end_ms}")
        key = CoverageKey(
            venue="binance", market="futures", dataset=_DATASET_KLINES,
            symbol=symbol, interval=interval,
        )
        interval_ms = INTERVAL_MS[interval]
        # 函数内 import 避免与 binance 的循环引用（binance 模块级 import 本模块）
        from .binance import KLINES_MAX_LIMIT

        page_limit = max(1, min(limit, KLINES_MAX_LIMIT))  # 与 _klines_page 的钳制一致

        def horizon_ms() -> int:
            # 闭合边界（exclusive）：open <= now - interval_ms ⇔ open < now - interval_ms + 1
            return self._client.futures_time() - interval_ms + 1

        return self._fetch_range(
            key,
            start_ms=start_ms,
            end_ms=end_ms,
            record_cap=None,
            identity_of=kline_identity,
            page_fetch=lambda cursor, gap_end: self._client._klines_page(
                symbol, interval, start_ms=cursor, end_ms=gap_end - 1, limit=limit
            ),
            page_limit=page_limit,
            horizon_ms_fn=horizon_ms,
        )

    def fetch_funding_history(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        *,
        limit: int,
    ) -> HistoricalDatasetResult:
        """资金费历史区间 ``[start_ms, end_ms)``（按 fundingTime，半开）。

        ``limit`` 是返回总条数上限（legacy 语义：从 start 起取满即停）；
        达到上限后的尾部区间留在 missing，下次请求续拉（断点恢复）。
        """
        if start_ms >= end_ms:
            raise ValueError(f"start_ms 必须 < end_ms: {start_ms} >= {end_ms}")
        if limit < 1:
            raise ValueError(f"limit 必须 >= 1，当前 {limit}")
        key = CoverageKey(
            venue="binance", market="futures", dataset=_DATASET_FUNDING, symbol=symbol,
        )

        def horizon_ms() -> int:
            # 只有严格早于交易所当前时间的结算事件才视为已发生（毫秒抖动保守）。
            return self._client.futures_time()

        from .binance import FUNDING_PAGE_SIZE  # 函数内 import 避免循环引用

        return self._fetch_range(
            key,
            start_ms=start_ms,
            end_ms=end_ms,
            record_cap=limit,
            identity_of=funding_identity,
            page_fetch=lambda cursor, gap_end: self._client._funding_page(
                symbol, start_ms=cursor, end_ms=gap_end - 1
            ),
            page_limit=FUNDING_PAGE_SIZE,
            horizon_ms_fn=horizon_ms,
        )

    # -- 核心 -----------------------------------------------------------------

    def _fetch_range(
        self,
        key: CoverageKey,
        *,
        start_ms: int,
        end_ms: int,
        record_cap: int | None,
        identity_of: Callable[[Any], int],
        page_fetch: Callable[[int, int], list[Any]],
        page_limit: int,
        horizon_ms_fn: Callable[[], int],
    ) -> HistoricalDatasetResult:
        key_hash = key.key_hash()
        with self._lock_for(key_hash):
            return self._fetch_range_locked(
                key,
                key_hash=key_hash,
                start_ms=start_ms,
                end_ms=end_ms,
                record_cap=record_cap,
                identity_of=identity_of,
                page_fetch=page_fetch,
                page_limit=page_limit,
                horizon_ms_fn=horizon_ms_fn,
            )

    def _fetch_range_locked(
        self,
        key: CoverageKey,
        *,
        key_hash: str,
        start_ms: int,
        end_ms: int,
        record_cap: int | None,
        identity_of: Callable[[Any], int],
        page_fetch: Callable[[int, int], list[Any]],
        page_limit: int,
        horizon_ms_fn: Callable[[], int],
    ) -> HistoricalDatasetResult:
        requested: Interval = (start_ms, end_ms)
        stats = self._client.stats

        # 1) 读索引（损坏/版本不符 → 空 = 全 miss）
        index_segments = self._load_index(key, key_hash)

        # 2) 读与请求相交的段；读回校验失败 = 该段按 miss 重拉
        records_by_id: dict[int, Any] = {}
        good_segments: list[CoverageSegment] = []
        cache_hits = 0
        read_failures = 0
        for seg in index_segments:
            if clip_to_requested(seg.range, requested) is None:
                continue
            entry = self._cache.get(SEGMENT_NAMESPACE, seg.segment_key)
            records = entry.data if entry is not None else None
            if records is not None and self._verify_records(records, seg, identity_of):
                cache_hits += 1
                stats.historical_cache_hits += 1
                good_segments.append(seg)
                for record in records:
                    records_by_id.setdefault(identity_of(record), record)
            else:
                read_failures += 1
                stats.cache_read_failures += 1
                logger.warning(
                    "历史段读回校验失败，按 miss 重拉: %s %s..%s",
                    key.symbol, seg.start_ms, seg.end_ms,
                )

        covered = merge_intervals([seg.range for seg in good_segments])
        missing = subtract_intervals(requested, covered)

        # 3) 只拉缺口（闭合边界之内）；错误不推进覆盖。
        #    每成功发布一段立即原子重写索引（断点检查点）：后续 gap 失败时，
        #    前面已验证的段不会丢，下次请求只拉剩余缺口。
        #    注意：以全部索引段为基线（含不相交段），避免窄请求覆盖宽索引。
        verified_segments: list[CoverageSegment] = list(index_segments)
        network_pages = 0
        write_failures = 0
        if missing:
            horizon_end = min(end_ms, horizon_ms_fn())
            for gap_start, gap_end in missing:
                if record_cap is not None and len(records_by_id) >= record_cap:
                    break  # 记录数上限已满，尾部留给下次请求
                eff_end = min(gap_end, horizon_end)
                if eff_end <= gap_start:
                    continue  # 整个缺口都在未闭合/未来区，不拉取
                stop_after = None
                if record_cap is not None:
                    stop_after = record_cap - len(records_by_id)  # 缺口内拉满剩余额度即停
                fetched = self._fetch_gap(
                    key,
                    gap_start=gap_start,
                    gap_end=eff_end,
                    identity_of=identity_of,
                    page_fetch=page_fetch,
                    page_limit=page_limit,
                    stop_after=stop_after,
                )
                network_pages += fetched["pages"]
                for record in fetched["records"]:
                    records_by_id.setdefault(identity_of(record), record)
                segment = self._persist_segment(
                    key_hash=key_hash,
                    start_ms=gap_start,
                    end_ms=fetched["fetched_end"],
                    records=fetched["records"],
                )
                if segment is None:
                    write_failures += 1
                    stats.cache_write_failures += 1
                    continue  # 段未落盘：区间留在 missing，下次重拉
                verified_segments = self._merge_segment_entries(verified_segments, [segment])
                if not self._publish_index(key, key_hash, verified_segments):
                    write_failures += 1
                    stats.cache_write_failures += 1

        # 4) 组装结果（升序去重 + 记录数上限截断）
        ordered = [records_by_id[record_id] for record_id in sorted(records_by_id)]
        if record_cap is not None:
            ordered = ordered[:record_cap]

        full_covered = merge_intervals([seg.range for seg in verified_segments])
        covered_in_request = [
            clipped for clipped in (clip_to_requested(iv, requested) for iv in full_covered)
            if clipped is not None
        ]
        missing_ranges = subtract_intervals(requested, full_covered)

        if missing_ranges:
            quality = DataQuality.INCOMPLETE
        elif read_failures or write_failures:
            quality = DataQuality.DEGRADED
        else:
            quality = DataQuality.FRESH

        return HistoricalDatasetResult(
            records=ordered,
            requested_range=requested,
            covered_ranges=covered_in_request,
            missing_ranges=missing_ranges,
            quality=quality,
            cache_hits=cache_hits,
            network_pages=network_pages,
        )

    # -- 段读取/发布 -----------------------------------------------------------

    def _load_index(self, key: CoverageKey, key_hash: str) -> list[CoverageSegment]:
        """读覆盖索引；文件损坏/版本不符/key 不符 → 空列表（整体按 miss）。"""
        entry = self._cache.get(INDEX_NAMESPACE, key_hash)
        if entry is None:
            return []
        payload = entry.data
        if not isinstance(payload, dict):
            return []
        if payload.get("format_version") != COVERAGE_FORMAT_VERSION:
            logger.info("覆盖索引版本不符，整体按 miss 重拉: %s", key.symbol)
            return []
        if payload.get("key") != key.to_dict():
            logger.warning("覆盖索引 key 不符（损坏），整体按 miss 重拉: %s", key.symbol)
            return []
        raw_segments = payload.get("segments")
        if not isinstance(raw_segments, list):
            return []
        segments: list[CoverageSegment] = []
        for item in raw_segments:
            try:
                seg = CoverageSegment.from_dict(item)
            except (ValueError, TypeError, KeyError):
                logger.warning("索引段元数据损坏，丢弃该段: %r", item)
                continue
            if not seg.complete or seg.format_version != COVERAGE_FORMAT_VERSION:
                continue
            segments.append(seg)
        return segments

    @staticmethod
    def _verify_records(records: Any, seg: CoverageSegment, identity_of: Callable[[Any], int]) -> bool:
        """读回段数据：结构、条数、checksum 全对才算验证通过。"""
        if not isinstance(records, list):
            return False
        if len(records) != seg.record_count:
            return False
        if canonical_checksum(records) != seg.checksum:
            return False
        # 身份可解析且确在段区间内（防止元数据与内容错位）
        try:
            for record in records:
                identity = identity_of(record)
                if not seg.start_ms <= identity < seg.end_ms:
                    return False
        except ParseError:
            return False
        return True

    def _fetch_gap(
        self,
        key: CoverageKey,
        *,
        gap_start: int,
        gap_end: int,
        identity_of: Callable[[Any], int],
        page_fetch: Callable[[int, int], list[Any]],
        page_limit: int,
        stop_after: int | None = None,
    ) -> dict[str, Any]:
        """完整拉取一个缺口 ``[gap_start, gap_end)``（分页推进）。

        返回 ``{"records", "pages", "fetched_end"}``：``fetched_end`` 是本次
        **实际完整取回**的子区间右端（正常结束 = gap_end；达到 stop_after
        记录数上限时 = 停页时的 cursor，剩余留给下次请求）。
        传输/限流/解析错误直接抛出（不推进覆盖）；服务端忽略 startTime
        造成无进展时抛 ParseError（绝不发布假完整）。
        """
        collected: list[Any] = []
        seen: set[int] = set()
        pages = 0
        cursor = gap_start
        fetched_end = gap_end  # 默认：子缺口整体完整（空页/越界/取完都如此）
        while cursor < gap_end:
            page = page_fetch(cursor, gap_end)
            pages += 1
            if not isinstance(page, list):
                raise ParseError(f"{key.dataset} 返回非列表: {type(page).__name__}")
            if not page:
                break  # 空页：交易所确认该窗口无记录 → 缺口完整（可为空段）
            page_max = -1
            max_fresh = -1
            fresh_count = 0
            for record in page:
                identity = identity_of(record)  # 字段异常 → ParseError
                page_max = max(page_max, identity)
                if not gap_start <= identity < gap_end:
                    continue
                if identity in seen:
                    continue
                seen.add(identity)
                collected.append(record)
                max_fresh = max(max_fresh, identity)
                fresh_count += 1
            if fresh_count == 0:
                if page_max >= gap_end:
                    break  # 页面已越过缺口右端 → 缺口完整
                raise ParseError(
                    f"{key.dataset} 分页无进展（服务端忽略 startTime?）: "
                    f"cursor={cursor}, page_max={page_max}, gap_end={gap_end}"
                )
            if len(page) < page_limit:
                break  # 短页 = 窗口内无更多记录（同 legacy 终止条件）
            cursor = max_fresh + 1
            if stop_after is not None and len(seen) >= stop_after:
                fetched_end = cursor  # 只发布已完整取回的子区间
                break
        collected.sort(key=identity_of)
        return {"records": collected, "pages": pages, "fetched_end": fetched_end}

    def _publish_index(
        self,
        key: CoverageKey,
        key_hash: str,
        segments: list[CoverageSegment],
    ) -> bool:
        """原子重写索引并读回校验。失败 = 覆盖未推进（下次请求重拉）。"""
        payload = {
            "format_version": COVERAGE_FORMAT_VERSION,
            "key": key.to_dict(),
            "segments": [seg.to_dict() for seg in segments],
        }
        self._cache.put(INDEX_NAMESPACE, key_hash, payload, ttl=None)
        readback = self._cache.get(INDEX_NAMESPACE, key_hash)
        if readback is None or readback.data != payload:
            logger.warning("覆盖索引写回校验失败: %s", key.symbol)
            return False
        return True

    def _persist_segment(
        self,
        *,
        key_hash: str,
        start_ms: int,
        end_ms: int,
        records: list[Any],
    ) -> CoverageSegment | None:
        """段数据原子落盘 + 读回 checksum 验证；失败返回 None（不推进覆盖）。"""
        checksum = canonical_checksum(records)
        segment_key = make_key(key_hash, start_ms, end_ms, checksum)
        self._cache.put(SEGMENT_NAMESPACE, segment_key, records, ttl=None)
        entry = self._cache.get(SEGMENT_NAMESPACE, segment_key)
        if (
            entry is None
            or not isinstance(entry.data, list)
            or len(entry.data) != len(records)
            or canonical_checksum(entry.data) != checksum
        ):
            logger.warning(
                "历史段写回校验失败（不推进覆盖）: %s..%s", start_ms, end_ms
            )
            return None
        return CoverageSegment(
            start_ms=start_ms,
            end_ms=end_ms,
            segment_key=segment_key,
            record_count=len(records),
            checksum=checksum,
            complete=True,
        )

    @staticmethod
    def _merge_segment_entries(
        old: list[CoverageSegment], new: list[CoverageSegment]
    ) -> list[CoverageSegment]:
        """合并索引段：同区间冲突时新段（刚验证的）胜出；被新区间完整包含的
        旧段丢弃。正常路径下新旧段区间不重叠（缺口按覆盖差集计算）。"""
        by_range: dict[Interval, CoverageSegment] = {}
        for seg in old:
            by_range[seg.range] = seg
        for seg in new:
            by_range[seg.range] = seg
        merged: list[CoverageSegment] = []
        for seg in sorted(by_range.values(), key=lambda s: (s.start_ms, s.end_ms)):
            contained = any(
                other is not seg and other.start_ms <= seg.start_ms and seg.end_ms <= other.end_ms
                for other in by_range.values()
            )
            if not contained:
                merged.append(seg)
        return merged


__all__ = [
    "COVERAGE_FORMAT_VERSION",
    "INDEX_NAMESPACE",
    "SEGMENT_NAMESPACE",
    "INTERVAL_MS",
    "CoverageKey",
    "CoverageSegment",
    "HistoricalDatasetResult",
    "HistoricalRepository",
    "clip_to_requested",
    "canonical_checksum",
    "funding_identity",
    "kline_identity",
    "merge_intervals",
    "subtract_intervals",
]
