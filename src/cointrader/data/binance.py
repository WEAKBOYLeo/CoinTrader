"""币安**只读**公开 API 客户端。

⚠️ 本模块的存在意义之一是：**它没有能力下单。**

这个类里不存在 API 凭证参数、不存在签名方法、不存在 POST/DELETE 方法。
要交易必须去 ``execution/`` 用另一个类。这样「数据层误下单」在类型层面
就不可能发生，而不是依赖开发者记得别写。

实测的限流参数（2026-09 探测）::

    fapi  REQUEST_WEIGHT  2400 / 分钟
    fapi  ORDERS          1200 / 分钟, 300 / 10 秒
    spot  REQUEST_WEIGHT  6000 / 分钟 (IP 维度)
    响应头  x-mbx-used-weight-1m  ← 已用权重，直接读，不要自己估

限流处理策略（实施计划书 v2.0 T1）：

- 所有请求经共享 ``RateLimitCoordinator`` 申请 permit（按 scope/priority/估算 weight），
  每次响应都把 ``x-mbx-used-weight-1m`` 反馈给 coordinator（以服务端视角为准）
- 429 → coordinator 冻结该市场（至少 120s），读请求可退避重试
- 418 → coordinator 封禁该市场（至少 3600s），**立即抛 IPBanError 停止一切请求**
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode

import httpx

from ..config import ApiConfig, DataConfig, UniverseConfig
from ..errors import (
    BinanceError,
    DataUnavailableError,
    IPBanError,
    NetworkError,
    ParseError,
    RateLimitError,
)
from ..rate_limit import (
    RateLimitCoordinator,
    RateLimitScope,
    RequestPriority,
    endpoint_weight,
)
from ..redact import redact_url
from .cache import DiskCache, json_or_raise, make_key
from .coverage import INTERVAL_MS, HistoricalRepository
from .venue import Venue

logger = logging.getLogger(__name__)

# API health is grouped by the logical data source used by the pool page.  The
# client records attempts only after a permit is granted, so coordinator waits
# never become request failures.
_ENDPOINT_HEALTH_META: dict[str, tuple[str, int, str]] = {
    "contract_rules": ("合约清单与交易规则", 3_600_000, "futures"),
    "volume_24h": ("24h 成交额", 900_000, "futures"),
    "funding_interval": ("资金费结算周期", 86_400_000, "futures"),
    "funding_history": ("历史资金费", 1_800_000, "futures"),
    "volume_3d": ("3 日成交额", 1_800_000, "futures"),
    "premium_index": ("当前资金费快照", 30_000, "futures"),
    "server_time": ("交易所时间", 300_000, "futures"),
    "other": ("其他公开接口", 900_000, "unknown"),
}


def _health_source_for(path: str) -> str:
    if path.endswith("exchangeInfo"):
        return "contract_rules"
    if path.endswith("ticker/24hr"):
        return "volume_24h"
    if path.endswith("fundingInfo"):
        return "funding_interval"
    if path.endswith("fundingRate"):
        return "funding_history"
    if path.endswith("premiumIndex"):
        return "premium_index"
    if path.endswith("/klines"):
        return "volume_3d"
    if path.endswith("/time"):
        return "server_time"
    return "other"


@dataclass(slots=True)
class _EndpointHealth:
    events: deque[tuple[int, bool, int | None]] = field(default_factory=deque)
    retries: deque[int] = field(default_factory=deque)
    waits: deque[tuple[int, int]] = field(default_factory=deque)
    last_attempt_ms: int | None = None
    last_success_ms: int | None = None
    last_failure_ms: int | None = None
    last_data_event_ms: int | None = None
    last_error: str | None = None
    consecutive_failures: int = 0


@dataclass(slots=True)
class ClientStats:
    """客户端运行统计，用于诊断与告警。"""

    requests: int = 0
    cache_hits: int = 0
    retries: int = 0
    throttled_seconds: float = 0.0
    errors: int = 0
    # 历史区间复用（v5.0 T1）：与短 TTL 实时缓存的 cache_hits 分开计数，
    # 让调用方区分“命中已验证历史段”和“命中短 TTL 快照”。
    historical_cache_hits: int = 0
    cache_write_failures: int = 0
    cache_read_failures: int = 0
    _health: dict[str, _EndpointHealth] = field(default_factory=dict, init=False, repr=False)
    _health_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "throttled_seconds": round(self.throttled_seconds, 2),
            "errors": self.errors,
            "historical_cache_hits": self.historical_cache_hits,
            "cache_write_failures": self.cache_write_failures,
            "cache_read_failures": self.cache_read_failures,
        }

    def record_attempt(
        self,
        source_id: str,
        *,
        success: bool,
        status: int | None = None,
        error: str | None = None,
        data_event: bool = False,
        now_ms: int | None = None,
    ) -> None:
        """记录一次已发出的请求尝试；调用方已持有网络请求边界。"""
        now = int(time.time() * 1000) if now_ms is None else int(now_ms)
        with self._health_lock:
            state = self._health.setdefault(source_id, _EndpointHealth())
            state.events.append((now, not success, status))
            state.last_attempt_ms = now
            if success:
                state.last_success_ms = now
                state.last_data_event_ms = now if data_event else state.last_data_event_ms
                state.consecutive_failures = 0
            else:
                state.last_failure_ms = now
                state.consecutive_failures += 1
                state.last_error = error or (f"HTTP {status}" if status else "request failed")
            self._prune_health_locked(state, now)

    def record_retry(self, source_id: str, *, now_ms: int | None = None) -> None:
        now = int(time.time() * 1000) if now_ms is None else int(now_ms)
        with self._health_lock:
            state = self._health.setdefault(source_id, _EndpointHealth())
            state.retries.append(now)
            self._prune_health_locked(state, now)

    def record_wait(self, source_id: str, wait_ms: int, *, now_ms: int | None = None) -> None:
        if wait_ms <= 0:
            return
        now = int(time.time() * 1000) if now_ms is None else int(now_ms)
        with self._health_lock:
            state = self._health.setdefault(source_id, _EndpointHealth())
            state.waits.append((now, int(wait_ms)))
            self._prune_health_locked(state, now)

    @staticmethod
    def _prune_health_locked(state: _EndpointHealth, now_ms: int) -> None:
        cutoff = now_ms - 3_600_000
        while state.events and state.events[0][0] < cutoff:
            state.events.popleft()
        while state.retries and state.retries[0] < cutoff:
            state.retries.popleft()
        while state.waits and state.waits[0][0] < cutoff:
            state.waits.popleft()

    def endpoint_health(self, *, now_ms: int | None = None) -> list[dict[str, Any]]:
        """导出脱敏 endpoint 类别健康快照（5 分钟窗口 + 1 小时有限事件）。"""
        now = int(time.time() * 1000) if now_ms is None else int(now_ms)
        out: list[dict[str, Any]] = []
        with self._health_lock:
            states = {key: state for key, state in self._health.items()}
            for source_id, (label, expected_ms, scope) in _ENDPOINT_HEALTH_META.items():
                state = states.get(source_id, _EndpointHealth())
                self._prune_health_locked(state, now)
                window_cutoff = now - 300_000
                events = [event for event in state.events if event[0] >= window_cutoff]
                attempts = len(events)
                failures = sum(1 for _, failed, _ in events if failed)
                rate_errors = sum(1 for _, _, status in events if status in (418, 429))
                retries = sum(1 for ts in state.retries if ts >= window_cutoff)
                wait_ms = sum(wait for ts, wait in state.waits if ts >= window_cutoff)
                data_age = (
                    max(0, now - state.last_data_event_ms)
                    if state.last_data_event_ms is not None
                    else None
                )
                if state.last_attempt_ms is None:
                    status = "UNKNOWN"
                    reason = "尚无请求样本"
                elif state.consecutive_failures >= 3:
                    status = "DOWN"
                    reason = f"连续失败 {state.consecutive_failures} 次"
                elif data_age is not None and data_age > expected_ms * 2:
                    status = "STALE"
                    reason = f"数据年龄 {data_age}ms 超过预期"
                elif failures or retries or (
                    state.last_success_ms is not None and now - state.last_success_ms > expected_ms
                ):
                    status = "DEGRADED"
                    reason = state.last_error or "近期存在失败/重试或数据接近过期"
                else:
                    status = "HEALTHY"
                    reason = "最近请求成功"
                out.append({
                    "id": source_id,
                    "label": label,
                    "scope": scope,
                    "status": status,
                    "reason": reason,
                    "attempts_window": attempts,
                    "failures_window": failures,
                    "consecutive_failures": state.consecutive_failures,
                    "retries_window": retries,
                    "rate_limit_errors_window": rate_errors,
                    "wait_ms_window": wait_ms,
                    "last_attempt_ms": state.last_attempt_ms,
                    "last_success_ms": state.last_success_ms,
                    "last_failure_ms": state.last_failure_ms,
                    "last_data_event_ms": state.last_data_event_ms,
                    "last_error": state.last_error,
                    "data_age_ms": data_age,
                    "expected_update_ms": expected_ms,
                    "as_of_ms": now,
                })
        return out


SPOT_KLINES = "/api/v3/klines"
SPOT_EXCHANGE_INFO = "/api/v3/exchangeInfo"
SPOT_TICKER_24H = "/api/v3/ticker/24hr"
SPOT_PRICE = "/api/v3/ticker/price"
SPOT_TIME = "/api/v3/time"

FAPI_PING = "/fapi/v1/ping"
FAPI_TIME = "/fapi/v1/time"
FAPI_EXCHANGE_INFO = "/fapi/v1/exchangeInfo"
FAPI_FUNDING_RATE = "/fapi/v1/fundingRate"
FAPI_FUNDING_INFO = "/fapi/v1/fundingInfo"
FAPI_PREMIUM_INDEX = "/fapi/v1/premiumIndex"
FAPI_KLINES = "/fapi/v1/klines"
FAPI_TICKER_24H = "/fapi/v1/ticker/24hr"
FAPI_OPEN_INTEREST = "/fapi/v1/openInterest"

#: 币安返回的错误码 → 是否可重试（403 HTML = WAF 按速率拦截，退避重试可恢复）
_RETRYABLE_STATUS = frozenset({403, 408, 425, 500, 502, 503, 504})

#: K 线单次请求最大条数（币安硬限制，滑动窗口分页时必须遵守）
KLINES_MAX_LIMIT = 1500

#: 资金费历史分页单页条数的**回退默认值**（正常路径从
#: ``UniverseConfig.funding_page_size_for(venue)`` 读，按 venue 不同）。
#:
#: 2026-09-28 实测（同一出口 IP，交替重试确认稳定）：
#:   主网 /fapi/v1/fundingRate: limit=500 → 200；limit=500/1000 均只返回 500 条
#:   demo /fapi/v1/fundingRate: limit≤200 → 200；limit≥210 → **403**
#: 同一 host 的 klines/exchangeInfo 无此限制 → 是 fundingRate 路由级的 WAF
#: 阈值，不是 IP 封禁。用一个全局值必然踩雷：取 500 则 demo 全量预热 100% 403。
#: 注意 limit=500 的 RTT 与 limit=100 几乎相同（0.51s vs 0.49s），
#: 所以主网用 500 纯粹是省请求数（1 年 8h 结算 12 页 → 3 页）。
#: **跨页总量不受单页上限限制**，由调用方的 limit 参数控制
#: （1 年 4h 结算 ≈ 2190 条、1h 结算 ≈ 8760 条）。
FUNDING_PAGE_SIZE = 500
#: 单次请求 limit 的名义上限（仅文档/参考用；分页总量可以超过它）。
FUNDING_MAX_LIMIT = 1000


class BinancePublicClient:
    """币安公开数据客户端（只读，无需密钥）。

    Args:
        api: API 端点配置。
        data: 数据/限流/缓存配置。
        client: 可注入的 ``httpx.Client``，便于测试时替换为 mock transport。
        sleep_fn: 可注入的 sleep，便于测试时跳过等待。
        rate_limiter: 可注入的共享限流协调器；未注入时自建一个（保持研究
            CLI/单测兼容）。live 装配必须注入与执行层客户端共享的实例。
    """

    def __init__(
        self,
        api: ApiConfig,
        data: DataConfig,
        *,
        universe: UniverseConfig | None = None,
        venue: Venue = Venue.MAINNET,
        client: httpx.Client | None = None,
        sleep_fn: Any = time.sleep,
        rate_limiter: RateLimitCoordinator | None = None,
    ) -> None:
        self.api = api
        self.data = data
        # venue 决定两件事，二者必须一致：
        #   1. exchangeInfo / ticker 等币池相关请求走哪个端点；
        #   2. 缓存键（含历史覆盖 namespace）的隔离前缀。
        # 默认 MAINNET 保持既有调用点与单测行为不变（回测恒走主网，见 data/venue.py）。
        self.venue = venue
        self.universe = universe or UniverseConfig()
        self.cache = DiskCache(data.cache_dir)
        self.stats = ClientStats()
        self._sleep = sleep_fn
        self._historical_repository: HistoricalRepository | None = None

        if rate_limiter is not None:
            self.coordinator = rate_limiter
        else:
            self.coordinator = RateLimitCoordinator(
                {
                    RateLimitScope.SPOT: data.rate_limit.spot_weight_per_min,
                    RateLimitScope.FUTURES: data.rate_limit.futures_weight_per_min,
                },
                soft_limit_ratio=data.rate_limit.soft_limit_ratio,
                critical_reserve_ratio=data.rate_limit.critical_reserve_ratio,
                freeze_seconds=data.rate_limit.rate_limit_freeze_seconds,
                ban_seconds=data.rate_limit.ip_ban_seconds,
                sleep=sleep_fn,
            )

        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(api.timeout_seconds),
            headers={
                # 明确标识自己；币安会据此在必要时联系，也便于区分流量来源
                "User-Agent": "CoinTrader/0.1 (research; read-only)",
                "Accept": "application/json",
            },
            follow_redirects=False,  # 重定向可能是钓鱼/中间人信号，不跟随
            trust_env=True,  # 读取 HTTPS_PROXY/HTTP_PROXY（服务器直连币安受限时必须走代理）
        )

    # -- 生命周期 -----------------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # -- venue 端点与缓存键 ---------------------------------------------------

    @property
    def venue_name(self) -> str:
        """venue 字符串（用于缓存 namespace / 键前缀与 CoverageKey.venue）。"""
        return self.venue.value

    @property
    def futures_base(self) -> str:
        """本 venue 的永续合约端点。"""
        return self.universe.futures_base_for(self.venue_name)

    @property
    def spot_base(self) -> str:
        """本 venue 的现货端点。"""
        return self.universe.spot_base_for(self.venue_name)

    def ckey(self, *parts: Any) -> str:
        """带 venue 前缀的缓存键。

        ⚠️ 所有缓存键**必须**经此方法生成。demo 与主网的合约清单/历史均可
        不同，不带 venue 的键会让两个场地互相污染（先写的一方胜出），
        表现为“换了 venue 却拿到另一边的数据”。
        """
        return make_key(self.venue_name, *parts)

    @property
    def funding_page_size(self) -> int:
        """资金费历史分页单页条数（**按 venue 配置**）。

        主网可到 500，demo 的 fundingRate 路由 WAF 阈值更低（实测 limit≥210
        即 403），故必须按 venue 取。缺配置时回退模块常量。
        """
        return int(
            self.universe.funding_page_size_for(self.venue_name)
            if hasattr(self.universe, "funding_page_size_for")
            else FUNDING_PAGE_SIZE
        )

    def __enter__(self) -> BinancePublicClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- 内部：请求 ---------------------------------------------------------

    def _backoff_seconds(self, attempt: int, retry_after: float | None = None) -> float:
        """指数退避 + 抖动。

        抖动很重要：多个进程同时被限流时，如果退避时间相同，
        它们会在同一时刻一起重试，再次撞限流（惊群）。
        """
        if retry_after is not None:
            return max(0.0, retry_after)
        base = self.data.rate_limit.base_backoff_seconds
        cap = self.data.rate_limit.max_backoff_seconds
        delay = min(cap, base * (2**attempt))
        return float(delay * (0.5 + random.random() * 0.5))  # 50%~100% 抖动

    def _request(
        self,
        base_url: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        scope: RateLimitScope,
        priority: RequestPriority = RequestPriority.P3_CANDIDATE,
        estimated_weight: int | None = None,
    ) -> Any:
        """发起一次 GET 请求，经共享限流协调器，含重试与错误分类。

        每次网络请求恰好 acquire/observe/release 一次：
        acquire 按 scope/priority/估算 weight 排队；响应头反馈给 coordinator；
        429/418 分别冻结/封禁该 scope。

        Returns:
            解析后的 JSON。

        Raises:
            IPBanError: 收到 418，必须立刻停机。
            RateLimitError: 重试耗尽仍是 429。
            NetworkError: 网络层反复失败。
            BinanceError: 其他 4xx（参数错误、币种不存在等），不重试。
        """
        url = f"{base_url}{path}"
        params = params or {}
        weight = (
            estimated_weight
            if estimated_weight is not None
            else endpoint_weight(scope, path, params, on_unknown=self.coordinator.note_unknown_endpoint)
        )
        max_retries = self.data.rate_limit.max_retries
        source_id = _health_source_for(path)

        for attempt in range(max_retries + 1):
            before = self.coordinator.now()
            with self.coordinator.acquire(scope, priority, weight):
                wait_ms = int(max(0.0, self.coordinator.now() - before) * 1000)
                self.stats.record_wait(source_id, wait_ms)
                self.stats.requests += 1

                try:
                    response = self._client.get(url, params=params)
                except httpx.TimeoutException as exc:
                    self.stats.errors += 1
                    self.stats.record_attempt(
                        source_id, success=False, error="请求超时"
                    )
                    if attempt >= max_retries:
                        raise NetworkError(
                            f"请求超时（已重试 {attempt} 次）: {redact_url(str(exc))}"
                        ) from exc
                    self.stats.retries += 1
                    self.stats.record_retry(source_id)
                    delay = self._backoff_seconds(attempt)
                    logger.warning("请求超时，%.1fs 后重试 (%d/%d)", delay, attempt + 1, max_retries)
                    self._sleep(delay)
                    continue
                except httpx.HTTPError as exc:
                    self.stats.errors += 1
                    self.stats.record_attempt(
                        source_id, success=False, error="网络错误"
                    )
                    if attempt >= max_retries:
                        raise NetworkError(
                            f"网络错误（已重试 {attempt} 次）: {redact_url(str(exc))}"
                        ) from exc
                    self.stats.retries += 1
                    self.stats.record_retry(source_id)
                    delay = self._backoff_seconds(attempt)
                    logger.warning("网络错误，%.1fs 后重试 (%d/%d): %s", delay, attempt + 1, max_retries, exc)
                    self._sleep(delay)
                    continue

                # 每次响应都把服务端视角的已用权重与熔断状态反馈给 coordinator
                retry_after = _parse_float_header(response.headers.get("retry-after"))
                self.coordinator.observe(
                    scope,
                    _parse_int_header(response.headers.get("x-mbx-used-weight-1m")),
                    response.status_code,
                    retry_after,
                )
                self.stats.throttled_seconds += max(0.0, self.coordinator.now() - before)

                status = response.status_code

                if status == 200:
                    try:
                        payload = json_or_raise(response.content, f"GET {path}")
                    except Exception as exc:  # noqa: BLE001
                        self.stats.errors += 1
                        self.stats.record_attempt(
                            source_id, success=False, status=status, error="响应解析失败"
                        )
                        raise ParseError(f"响应解析失败 at {path}: {redact_url(str(exc))}") from exc
                    self.stats.record_attempt(
                        source_id, success=True, status=status, data_event=True
                    )
                    return payload

                error_label = f"HTTP {status}"
                self.stats.record_attempt(
                    source_id,
                    success=False,
                    status=status,
                    error=error_label,
                )

                if status == 418:
                    # 不重试。立即上报。
                    self.stats.errors += 1
                    raise IPBanError(
                        f"IP 已被币安临时封禁 (HTTP 418) at {path}。"
                        "必须立即停止所有请求并等待封禁解除。",
                        status=status,
                    )

                if status == 429:
                    self.stats.errors += 1
                    if attempt >= max_retries:
                        raise RateLimitError(
                            f"限流重试耗尽 (HTTP 429) at {path}",
                            status=status,
                        )
                    self.stats.retries += 1
                    self.stats.record_retry(source_id)
                    delay = self._backoff_seconds(attempt, retry_after)
                    logger.warning("被限流 (429)，%.1fs 后重试 (%d/%d)", delay, attempt + 1, max_retries)
                    self.stats.throttled_seconds += delay
                    self._sleep(delay)
                    continue

                if status in _RETRYABLE_STATUS:
                    self.stats.errors += 1
                    if attempt >= max_retries:
                        raise BinanceError(
                            f"服务端错误重试耗尽 (HTTP {status}) at {path}",
                            status=status,
                        )
                    self.stats.retries += 1
                    self.stats.record_retry(source_id)
                    delay = self._backoff_seconds(attempt)
                    logger.warning("服务端 %d，%.1fs 后重试 (%d/%d)", status, delay, attempt + 1, max_retries)
                    self._sleep(delay)
                    continue

                # 4xx（除 429/418）：不可重试。解析币安错误码以给出可读信息。
                self.stats.errors += 1
                raise self._build_client_error(response, path)

        # 理论上不可达（循环内每条路径要么 return 要么 raise）
        raise BinanceError(f"请求失败且未产生明确结果: {path}")

    @staticmethod
    def _build_client_error(response: httpx.Response, path: str) -> BinanceError:
        """把 4xx 响应转换为带语义的异常。"""
        code: int | None = None
        message = response.text[:300]
        try:
            body = response.json()
            if isinstance(body, dict):
                code = body.get("code")
                message = str(body.get("msg", message))
        except ValueError:
            pass  # 非 JSON 响应，保留原始文本片段

        # 币安约定：-1121 无效交易对，-1100 参数异常
        if code in {-1121, -1122}:
            return DataUnavailableError(
                f"交易对/参数无效 at {path}: {message}", code=code, status=response.status_code
            )
        return BinanceError(
            f"客户端错误 at {path}: {message}", code=code, status=response.status_code
        )

    # -- 内部：带缓存的分页拉取 ---------------------------------------------

    @property
    def historical_repository(self) -> HistoricalRepository:
        """历史区间仓库（v5.0 T1）：覆盖索引 + 只拉缺口。

        懒加载；共享本实例的 cache 与 RateLimitCoordinator。
        """
        if self._historical_repository is None:
            self._historical_repository = HistoricalRepository(self, self.cache)
        return self._historical_repository

    def _klines_page(
        self,
        symbol: str,
        interval: str,
        *,
        start_ms: int,
        end_ms: int,
        limit: int,
    ) -> list[Any]:
        """拉取一页永续 K 线（单页 limit ≤ KLINES_MAX_LIMIT，经 _request 限流/重试）。"""
        limit = min(limit, KLINES_MAX_LIMIT)
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_ms,
            "endTime": end_ms,
        }
        page = self._request(self.futures_base, FAPI_KLINES, params, scope=RateLimitScope.FUTURES)
        if not isinstance(page, list):
            raise ParseError(f"klines 返回非列表: {type(page).__name__}")
        return page

    def _funding_page(
        self,
        symbol: str,
        *,
        start_ms: int,
        end_ms: int,
    ) -> list[Any]:
        """拉取一页资金费历史（单页大小由配置的 funding_page_size 决定，经 _request）。"""
        page_size = self.funding_page_size
        params: dict[str, Any] = {
            "symbol": symbol,
            "limit": page_size,
            "startTime": start_ms,
            "endTime": end_ms,
        }
        page = self._request(self.futures_base, FAPI_FUNDING_RATE, params, scope=RateLimitScope.FUTURES)
        if not isinstance(page, list):
            raise ParseError(f"fundingRate 返回非列表: {type(page).__name__}")
        return page

    def _cached_get(
        self,
        namespace: str,
        key: str,
        ttl: int,
        loader: Any,
    ) -> Any:
        """缓存命中则返回缓存，否则调用 loader 并写入缓存。"""
        cached = self.cache.get_fresh(namespace, key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached

        data = loader()
        self.cache.put(namespace, key, data, ttl=ttl)
        return data

    # ======================================================================
    # 公开接口 —— 现货
    # ======================================================================

    def spot_time(self) -> int:
        """现货服务器时间（毫秒）。用于校验本地时钟偏移。

        本地时钟偏差过大会导致签名请求被拒（真实交易时的经典故障）。
        """
        payload = self._request(self.spot_base, SPOT_TIME, scope=RateLimitScope.SPOT)
        return int(payload["serverTime"])

    def spot_exchange_info(self, symbol: str | None = None) -> dict[str, Any]:
        """现货交易规则（含 tickSize / stepSize / minNotional），按 venue 取端点。"""
        params = {"symbol": symbol} if symbol else {}
        key = self.ckey("spot_exchange_info", symbol)
        return self._cached_get(
            "spot_exchange_info",
            key,
            self.data.cache_ttl.exchange_info,
            lambda: self._request(
                self.spot_base,
                SPOT_EXCHANGE_INFO,
                params,
                scope=RateLimitScope.SPOT,
            ),
        )

    def spot_klines(
        self,
        symbol: str,
        interval: str = "8h",
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = KLINES_MAX_LIMIT,
    ) -> list[list[Any]]:
        """现货 K 线。返回原始数组列表（12 字段）。"""
        return self._klines(
            self.spot_base,
            SPOT_KLINES,
            RateLimitScope.SPOT,
            "spot_klines",
            symbol,
            interval,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=limit,
        )

    def spot_price(self, symbol: str, *, priority: RequestPriority | None = None) -> Decimal:
        """单币种现货最新价（实时行情，不缓存）。"""
        payload = self._request(
            self.spot_base,
            "/api/v3/ticker/price",
            {"symbol": symbol},
            scope=RateLimitScope.SPOT,
            priority=priority if priority is not None else RequestPriority.P3_CANDIDATE,
        )
        return Decimal(str(payload["price"]))

    def spot_tickers_24h(self) -> list[dict[str, Any]]:
        """全部现货交易对 24h 行情（权重 80，务必缓存）。"""
        return self._cached_get(
            "spot_ticker_24h",
            self.ckey("spot_ticker_24h"),
            60,  # 1 分钟：24h 成交额不需要更实时
            lambda: self._request(
                self.spot_base,
                SPOT_TICKER_24H,
                scope=RateLimitScope.SPOT,
            ),
        )

    # ======================================================================
    # 公开接口 —— 永续合约
    # ======================================================================

    def futures_time(self) -> int:
        payload = self._request(self.futures_base, FAPI_TIME, scope=RateLimitScope.FUTURES)
        return int(payload["serverTime"])

    def futures_exchange_info(self) -> dict[str, Any]:
        """永续合约交易规则（按 venue 取端点；demo 与主网清单不同，见 data/venue.py）。权重 1。"""
        return self._cached_get(
            "futures_exchange_info",
            self.ckey("futures_exchange_info"),
            self.data.cache_ttl.exchange_info,
            lambda: self._request(
                self.futures_base, FAPI_EXCHANGE_INFO, scope=RateLimitScope.FUTURES
            ),
        )

    def funding_info(self) -> list[dict[str, Any]]:
        """各永续合约的资金费**结算周期**。

        ⚠️ 这是本项目**必须**依赖、却极易被忽略的接口。

        实测（2026-09）：782 个永续合约中，4 小时结算的 466 个、
        8 小时结算的 312 个、1 小时结算的 4 个。

        如果用统一的 8 小时折算年化，对 4 小时结算的币会
        **低估一半**，直接导致策略筛选出错。
        """
        return self._cached_get(
            "funding_info",
            self.ckey("funding_info"),
            self.data.cache_ttl.funding_info,
            lambda: self._request(self.futures_base, FAPI_FUNDING_INFO, scope=RateLimitScope.FUTURES),
        )

    def premium_index(self, symbol: str | None = None, *,
                      priority: RequestPriority | None = None) -> Any:
        """实时标记价 / 指数价 / 当前资金费率 / 下次结算时间。"""
        params = {"symbol": symbol} if symbol else {}
        key = self.ckey("premium_index", symbol)
        return self._cached_get(
            "premium_index",
            key,
            self.data.cache_ttl.premium_index,
            lambda: self._request(
                self.futures_base,
                FAPI_PREMIUM_INDEX,
                params,
                scope=RateLimitScope.FUTURES,
                priority=priority if priority is not None else RequestPriority.P3_CANDIDATE,
            ),
        )

    def funding_history(
        self,
        symbol: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = FUNDING_MAX_LIMIT,
        settled_at_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """单个合约的资金费结算历史。

        ⚠️ 币安**只保留约 1 年的资金费历史**。更早的数据需要自己持续采集
        或从第三方获取。这个限制会影响回测的时间跨度，必须心里有数。

        分页方式：``startTime`` 向前推，每页固定 ``funding_page_size`` 条
        （按 venue 配置：主网 500、demo 200；不能把总量当单页 limit 发）。

        Args:
            symbol: 合约符号，如 ``BTCUSDT``。
            start_ms: 起始时间（毫秒，含）。
            end_ms: 结束时间（毫秒，含）。
            limit: 返回总条数上限。分页跨页取全，**不受单次请求 1000 名义
                上限约束**（钳制到 1000 会静默丢弃长周期回测的近期数据）。

        Returns:
            按时间升序的结算记录列表，每条含 ``fundingTime`` /
            ``fundingRate`` / ``markPrice``。

        Raises:
            ValueError: limit < 1。
        """
        if limit < 1:
            raise ValueError(f"limit 必须 >= 1，当前 {limit}")

        # v5.0 T1：显式历史区间（start+end 都给）走覆盖仓库——只拉缺口、
        # 半开 [start,end) 语义、仅已结算事件；返回形状不变。
        if start_ms is not None and end_ms is not None and start_ms < end_ms:
            result = self.historical_repository.fetch_funding_history(
                symbol,
                start_ms,
                end_ms,
                limit=limit,
                settled_at_ms=settled_at_ms,
            )
            return result.records

        def loader() -> list[dict[str, Any]]:
            all_records: list[dict[str, Any]] = []
            cursor = start_ms
            seen_times: set[int] = set()
            page_size = self.funding_page_size

            while len(all_records) < limit:
                params: dict[str, Any] = {
                    "symbol": symbol,
                    "limit": page_size,
                }
                if cursor is not None:
                    params["startTime"] = cursor
                if end_ms is not None:
                    params["endTime"] = end_ms

                page = self._request(
                    self.futures_base,
                    FAPI_FUNDING_RATE,
                    params,
                    scope=RateLimitScope.FUTURES,
                )
                if not page:
                    break
                if not isinstance(page, list):
                    raise ParseError(f"fundingRate 返回非列表: {type(page).__name__}")

                # 先校验字段完整性再接续分页逻辑。缺少 fundingTime 时
                # 下面的 max()/判重会抛 KeyError —— 那是实现细节泄漏到调用方，
                # 应该在这里给出带上下文的 ParseError。
                missing = [i for i, r in enumerate(page) if "fundingTime" not in r]
                if missing:
                    raise ParseError(
                        f"fundingRate 响应缺少 fundingTime 字段（第 {missing[:3]} 条）: "
                        f"{[page[i] for i in missing[:2]]!r}"
                    )

                fresh = [r for r in page if r.get("fundingTime") not in seen_times]
                if not fresh:
                    # 服务端忽略了 startTime（返回了同一页），停止避免死循环
                    logger.debug("funding_history %s 分页无进展，停止", symbol)
                    break
                for record in fresh:
                    seen_times.add(record["fundingTime"])
                all_records.extend(fresh)

                if len(all_records) >= limit or len(page) < page_size:
                    break  # 够数或最后一页

                cursor = max(r["fundingTime"] for r in fresh) + 1
                if end_ms is not None and cursor > end_ms:
                    break

            all_records.sort(key=lambda r: r["fundingTime"])
            return all_records[:limit]

        key = self.ckey("funding_history", symbol, start_ms, end_ms, limit)
        # 历史资金费不会变（币安偶尔修订，但概率低），长 TTL
        return self._cached_get(
            "funding_history", key, self.data.cache_ttl.funding_history, loader
        )

    def futures_klines(
        self,
        symbol: str,
        interval: str = "8h",
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = KLINES_MAX_LIMIT,
        closed_at_ms: int | None = None,
    ) -> list[list[Any]]:
        """永续合约 K 线。

        v5.0 T1：``start_ms`` 与 ``end_ms`` 均给定（且 interval 已知）时走历史
        区间仓库——覆盖索引命中零请求、只拉缺口、仅返回**已闭合** candle
        （open time 半开 ``[start_ms, end_ms)``）；返回类型不变。
        """
        if (
            start_ms is not None
            and end_ms is not None
            and start_ms < end_ms
            and interval in INTERVAL_MS
        ):
            result = self.historical_repository.fetch_futures_klines(
                symbol,
                interval,
                start_ms,
                end_ms,
                limit=limit,
                closed_at_ms=closed_at_ms,
            )
            return result.records
        return self._klines(
            self.futures_base,
            FAPI_KLINES,
            RateLimitScope.FUTURES,
            "futures_klines",
            symbol,
            interval,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=limit,
        )

    def futures_tickers_24h(self) -> list[dict[str, Any]]:
        return self._cached_get(
            "futures_ticker_24h",
            self.ckey("futures_ticker_24h"),
            60,
            lambda: self._request(
                self.futures_base,
                FAPI_TICKER_24H,
                scope=RateLimitScope.FUTURES,
            ),
        )

    def open_interest(self, symbol: str) -> dict[str, Any]:
        """当前未平仓合约量。用于评估该合约的深度。"""
        return self._request(
            self.futures_base,
            FAPI_OPEN_INTEREST,
            {"symbol": symbol},
            scope=RateLimitScope.FUTURES,
        )

    # ======================================================================
    # 内部：K 线通用实现
    # ======================================================================

    def _klines(
        self,
        base_url: str,
        path: str,
        scope: RateLimitScope,
        namespace: str,
        symbol: str,
        interval: str,
        *,
        start_ms: int | None,
        end_ms: int | None,
        limit: int,
    ) -> list[list[Any]]:
        limit = min(limit, KLINES_MAX_LIMIT)

        def loader() -> list[list[Any]]:
            all_candles: list[list[Any]] = []
            cursor = start_ms
            seen_open_times: set[int] = set()

            while True:
                params: dict[str, Any] = {
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit,
                }
                if cursor is not None:
                    params["startTime"] = cursor
                if end_ms is not None:
                    params["endTime"] = end_ms

                page = self._request(base_url, path, params, scope=scope)
                if not page:
                    break
                if not isinstance(page, list):
                    raise ParseError(f"klines 返回非列表: {type(page).__name__}")

                fresh = [c for c in page if c[0] not in seen_open_times]
                if not fresh:
                    break
                for candle in fresh:
                    seen_open_times.add(candle[0])
                all_candles.extend(fresh)

                if len(page) < limit:
                    break

                cursor = max(c[0] for c in fresh) + 1
                if end_ms is not None and cursor > end_ms:
                    break

            all_candles.sort(key=lambda c: c[0])
            return all_candles

        key = self.ckey(namespace, symbol, interval, start_ms, end_ms, limit)
        return self._cached_get(namespace, key, self.data.cache_ttl.klines_closed, loader)

    # ======================================================================

    def health_check(self) -> dict[str, Any]:
        """连通性 + 时钟偏移自检。

        时钟偏移超过 1 秒时给出警告 —— 真实交易签名依赖时间戳，
        偏移过大会被币安以 -1021 拒绝。
        """
        local_ms = int(time.time() * 1000)
        spot_server = self.spot_time()
        futures_server = self.futures_time()
        return {
            "spot_ok": True,
            "futures_ok": True,
            "clock_offset_ms_spot": local_ms - spot_server,
            "clock_offset_ms_futures": local_ms - futures_server,
            "stats": self.stats.as_dict(),
        }


def _parse_int_header(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float_header(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def encode_query(params: dict[str, Any]) -> str:
    """URL 编码查询串（保留供调试与测试使用）。"""
    return urlencode(params)


__all__ = [
    "FUNDING_MAX_LIMIT",
    "KLINES_MAX_LIMIT",
    "BinancePublicClient",
    "ClientStats",
]
