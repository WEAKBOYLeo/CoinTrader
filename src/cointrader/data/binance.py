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
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode

import httpx

from ..config import ApiConfig, DataConfig
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 端点路径（公开只读）
# ---------------------------------------------------------------------------

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

#: 资金费历史 API 名义上限 1000，但实测大 limit（>100）会被 WAF 间歇
#: 403 拦截（2026-09-21 实测 limit=1000/500 → 403，limit=100 → 200）。
#: 分页固定小页，总量由 limit 参数控制。
FUNDING_PAGE_SIZE = 100
FUNDING_MAX_LIMIT = 1000


@dataclass(slots=True)
class ClientStats:
    """客户端运行统计，用于诊断与告警。"""

    requests: int = 0
    cache_hits: int = 0
    retries: int = 0
    throttled_seconds: float = 0.0
    errors: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "throttled_seconds": round(self.throttled_seconds, 2),
            "errors": self.errors,
        }


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
        client: httpx.Client | None = None,
        sleep_fn: Any = time.sleep,
        rate_limiter: RateLimitCoordinator | None = None,
    ) -> None:
        self.api = api
        self.data = data
        self.cache = DiskCache(data.cache_dir)
        self.stats = ClientStats()
        self._sleep = sleep_fn

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

        for attempt in range(max_retries + 1):
            before = self.coordinator.now()
            with self.coordinator.acquire(scope, priority, weight):
                self.stats.requests += 1

                try:
                    response = self._client.get(url, params=params)
                except httpx.TimeoutException as exc:
                    self.stats.errors += 1
                    if attempt >= max_retries:
                        raise NetworkError(
                            f"请求超时（已重试 {attempt} 次）: {redact_url(str(exc))}"
                        ) from exc
                    self.stats.retries += 1
                    delay = self._backoff_seconds(attempt)
                    logger.warning("请求超时，%.1fs 后重试 (%d/%d)", delay, attempt + 1, max_retries)
                    self._sleep(delay)
                    continue
                except httpx.HTTPError as exc:
                    self.stats.errors += 1
                    if attempt >= max_retries:
                        raise NetworkError(
                            f"网络错误（已重试 {attempt} 次）: {redact_url(str(exc))}"
                        ) from exc
                    self.stats.retries += 1
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
                    return json_or_raise(response.content, f"GET {path}")

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
        payload = self._request(self.api.spot_base, SPOT_TIME, scope=RateLimitScope.SPOT)
        return int(payload["serverTime"])

    def spot_exchange_info(self, symbol: str | None = None) -> dict[str, Any]:
        """现货交易规则（含 tickSize / stepSize / minNotional）。"""
        params = {"symbol": symbol} if symbol else {}
        key = make_key("spot_exchange_info", symbol)
        return self._cached_get(
            "spot_exchange_info",
            key,
            self.data.cache_ttl.exchange_info,
            lambda: self._request(
                self.api.spot_base,
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
            self.api.spot_base,
            SPOT_KLINES,
            RateLimitScope.SPOT,
            "spot_klines",
            symbol,
            interval,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=limit,
        )

    def spot_price(self, symbol: str) -> Decimal:
        """单币种现货最新价（实时行情，不缓存）。"""
        payload = self._request(
            self.api.spot_base,
            "/api/v3/ticker/price",
            {"symbol": symbol},
            scope=RateLimitScope.SPOT,
        )
        return Decimal(str(payload["price"]))

    def spot_tickers_24h(self) -> list[dict[str, Any]]:
        """全部现货交易对 24h 行情（权重 80，务必缓存）。"""
        return self._cached_get(
            "spot_ticker_24h",
            make_key("spot_ticker_24h"),
            60,  # 1 分钟：24h 成交额不需要更实时
            lambda: self._request(
                self.api.spot_base,
                SPOT_TICKER_24H,
                scope=RateLimitScope.SPOT,
            ),
        )

    # ======================================================================
    # 公开接口 —— 永续合约
    # ======================================================================

    def futures_time(self) -> int:
        payload = self._request(self.api.futures_base, FAPI_TIME, scope=RateLimitScope.FUTURES)
        return int(payload["serverTime"])

    def futures_exchange_info(self) -> dict[str, Any]:
        """永续合约交易规则。权重 1。"""
        return self._cached_get(
            "futures_exchange_info",
            make_key("futures_exchange_info"),
            self.data.cache_ttl.exchange_info,
            lambda: self._request(
                self.api.futures_base, FAPI_EXCHANGE_INFO, scope=RateLimitScope.FUTURES
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
            make_key("funding_info"),
            self.data.cache_ttl.funding_info,
            lambda: self._request(self.api.futures_base, FAPI_FUNDING_INFO, scope=RateLimitScope.FUTURES),
        )

    def premium_index(self, symbol: str | None = None) -> Any:
        """实时标记价 / 指数价 / 当前资金费率 / 下次结算时间。"""
        params = {"symbol": symbol} if symbol else {}
        key = make_key("premium_index", symbol)
        return self._cached_get(
            "premium_index",
            key,
            self.data.cache_ttl.premium_index,
            lambda: self._request(
                self.api.futures_base,
                FAPI_PREMIUM_INDEX,
                params,
                scope=RateLimitScope.FUTURES,
            ),
        )

    def funding_history(
        self,
        symbol: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = FUNDING_MAX_LIMIT,
    ) -> list[dict[str, Any]]:
        """单个合约的资金费结算历史。

        ⚠️ 币安**只保留约 1 年的资金费历史**。更早的数据需要自己持续采集
        或从第三方获取。这个限制会影响回测的时间跨度，必须心里有数。

        分页方式：``startTime`` 向前推，每页固定 ``FUNDING_PAGE_SIZE`` 条
        （大 limit 会被 WAF 间歇 403 拦截，不能把总量当单页 limit 发）。

        Args:
            symbol: 合约符号，如 ``BTCUSDT``。
            start_ms: 起始时间（毫秒，含）。
            end_ms: 结束时间（毫秒，含）。
            limit: 返回总条数上限。

        Returns:
            按时间升序的结算记录列表，每条含 ``fundingTime`` /
            ``fundingRate`` / ``markPrice``。
        """
        limit = min(limit, FUNDING_MAX_LIMIT)

        def loader() -> list[dict[str, Any]]:
            all_records: list[dict[str, Any]] = []
            cursor = start_ms
            seen_times: set[int] = set()

            while len(all_records) < limit:
                params: dict[str, Any] = {
                    "symbol": symbol,
                    "limit": FUNDING_PAGE_SIZE,
                }
                if cursor is not None:
                    params["startTime"] = cursor
                if end_ms is not None:
                    params["endTime"] = end_ms

                page = self._request(
                    self.api.futures_base,
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

                if len(all_records) >= limit or len(page) < FUNDING_PAGE_SIZE:
                    break  # 够数或最后一页

                cursor = max(r["fundingTime"] for r in fresh) + 1
                if end_ms is not None and cursor > end_ms:
                    break

            all_records.sort(key=lambda r: r["fundingTime"])
            return all_records[:limit]

        key = make_key("funding_history", symbol, start_ms, end_ms, limit)
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
    ) -> list[list[Any]]:
        """永续合约 K 线。"""
        return self._klines(
            self.api.futures_base,
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
            make_key("futures_ticker_24h"),
            60,
            lambda: self._request(
                self.api.futures_base,
                FAPI_TICKER_24H,
                scope=RateLimitScope.FUTURES,
            ),
        )

    def open_interest(self, symbol: str) -> dict[str, Any]:
        """当前未平仓合约量。用于评估该合约的深度。"""
        return self._request(
            self.api.futures_base,
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

        key = make_key(namespace, symbol, interval, start_ms, end_ms, limit)
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
