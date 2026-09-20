"""签名 HTTP 传输层 —— 限流、超时、错误分类的唯一出口。

安全规则（开发设计文档 §3.1/§3.5）：

1. **全局共享限流**：每个客户端实例一个权重跟踪器（Spot 与 Futures
   各建一个实例），以响应头 ``x-mbx-used-weight-1m`` 为准（服务端视角）。
2. **订单类请求（critical）永不重试**。
   超时 / 408 / 5xx / 网络中断一律抛 ``UnknownSubmission`` ——
   订单可能已成交，唯一合法的下一步是用 clientOrderId 查询。
   429 对 critical 请求意味着请求**未被提交**，同样不自动重试，上抛决策。
3. 错误分类（HTTP 状态 + 币安错误码）：
   - 400 → BinanceError / OrderRejected（订单明确拒单，不重试）
   - 401/403 / -1022 → AuthError（立即停机）
   - 408（读请求）/ 5xx → 读请求可重试；critical → UnknownSubmission
   - 429 → 读请求尊重 Retry-After 重试；critical 直接上抛
   - 418 → IPBanError，停止一切请求
   - -1021 → ClockError（停止交易，校准 NTP）
4. 日志永不包含签名、完整认证 URL 或 Secret（redact 兜底）。
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any
from urllib.parse import quote

import httpx

from ..errors import (
    AuthError,
    BinanceError,
    ClockError,
    IPBanError,
    NetworkError,
    OrderRejected,
    RateLimitError,
    UnknownSubmission,
)
from ..redact import redact_url
from .auth import sign_params
from .models import Market

logger = logging.getLogger(__name__)

#: 币安错误码 → 订单明确被拒（不需要进入 UNKNOWN 恢复流程）
ORDER_REJECTED_CODES = frozenset(
    {
        -2010,  # New order rejected
        -2011,  # Cancel order rejected
        -2013,  # Order does not exist
        -2015,  # TIF invalid
        -2016,  # TIF missing
        -2018,  # Notional below minimum
        -2019,  # Balance insufficient
        -2022,  # Position quantity is not equal to order quantity
        -4131,  # ReduceOnly order is rejected
        -1100,  # Bad request（参数格式）
        -1102,  # Mandatory parameter missing
        -1104,  # Unknown parameter
        -1112,  # Precision too high
    }
)

#: 时钟偏移 / 签名错误
CLOCK_DRIFT_CODE = -1021
SIGNATURE_ERROR_CODE = -1022


@dataclass
class _WeightTracker:
    """按市场跟踪已用权重。以响应头为准（含同 IP 其他进程）。"""

    limit: int
    soft_limit_ratio: float
    used: int = 0
    window_start: float = time.monotonic()
    _lock: Lock = Lock()

    def observe(self, used_weight: int | None) -> None:
        if used_weight is None:
            return
        with self._lock:
            self.used = used_weight
            self._maybe_roll()

    def _maybe_roll(self) -> None:
        if time.monotonic() - self.window_start >= 60.0:
            self.window_start = time.monotonic()
            self.used = 0

    def should_throttle(self) -> bool:
        with self._lock:
            self._maybe_roll()
            return self.used >= self.limit * self.soft_limit_ratio

    def seconds_until_reset(self) -> float:
        with self._lock:
            self._maybe_roll()
            return max(0.0, 60.0 - (time.monotonic() - self.window_start))

    def used_ratio(self) -> float:
        with self._lock:
            self._maybe_roll()
            return self.used / self.limit if self.limit else 0.0


@dataclass
class TransportStats:
    requests: int = 0
    retries: int = 0
    throttled_seconds: float = 0.0
    unknown_submissions: int = 0
    auth_errors: int = 0
    rate_limit_hits: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "retries": self.retries,
            "throttled_seconds": round(self.throttled_seconds, 2),
            "unknown_submissions": self.unknown_submissions,
            "auth_errors": self.auth_errors,
            "rate_limit_hits": self.rate_limit_hits,
        }


class _Retry:
    """内部标记：读请求需要按退避策略重试。"""


class SignedClient:
    """带签名与限流的私有 API 客户端。

    Args:
        base_url: API 根（必须 https，含 https://）。
        api_key: API Key（SecretStr，repr 安全）。
        secret: API Secret（SecretStr）。
        market: 权重跟踪器命名（spot/futures）。
        weight_limit: 每分钟权重上限。
        client: 可注入的 httpx.Client（测试用 mock transport）。
        read_max_retries: 读请求最大重试次数。**critical 请求恒为 0 次。**
        sleep_fn / now_fn: 可注入，便于测试。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Any,
        secret: Any,
        market: Market | str,
        weight_limit: int = 2400,
        soft_limit_ratio: float = 0.80,
        recv_window_ms: int = 5000,
        timeout_seconds: float = 5.0,
        read_max_retries: int = 3,
        client: httpx.Client | None = None,
        sleep_fn: Any = time.sleep,
        now_fn: Any = time.time,
    ) -> None:
        if not base_url.startswith("https://"):
            raise ValueError(f"base_url 必须使用 https，当前: {base_url}")
        self.base_url = base_url.rstrip("/")
        self._secret = secret
        self.market = Market(market) if isinstance(market, Market) else Market(str(market).upper())
        self.recv_window_ms = recv_window_ms
        self.read_max_retries = max(0, int(read_max_retries))
        self.stats = TransportStats()
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"X-MBX-APIKEY": api_key.reveal(), "Accept": "application/json"},
            follow_redirects=False,
            trust_env=True,  # 读取 HTTPS_PROXY/HTTP_PROXY（服务器直连币安受限时必须走代理）
        )
        self._sleep = sleep_fn
        self._now = now_fn
        self._weight = _WeightTracker(weight_limit, soft_limit_ratio)
        #: server time 偏移（毫秒）= server - local。由 adapter 的 calibrate() 设置。
        self.time_offset_ms: int = 0

    # -- 生命周期 -----------------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> SignedClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- 时间戳 -------------------------------------------------------------

    def timestamp_ms(self) -> int:
        """带 server 偏移的请求时间戳（毫秒）。"""
        return int(self._now() * 1000) + self.time_offset_ms

    # -- 请求入口 -----------------------------------------------------------

    def get(self, path: str, params: dict[str, Any] | None = None, *, sign: bool = True) -> Any:
        """读请求：有限重试 + 限流退避。sign=False 用于公开接口（不附加签名参数）。"""
        return self._execute("GET", path, params, critical=False, client_order_id=None, sign=sign)

    def post(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """非 critical 写请求（如创建 listenKey）：按读请求策略重试。"""
        return self._execute("POST", path, params, critical=False, client_order_id=None)

    def delete(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._execute("DELETE", path, params, critical=False, client_order_id=None)

    def put(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._execute("PUT", path, params, critical=False, client_order_id=None)

    def place_order(self, path: str, params: dict[str, Any], *, client_order_id: str) -> Any:
        """下单。永不重试；结果未知时抛 UnknownSubmission。"""
        return self.execute_critical("POST", path, params, client_order_id=client_order_id)

    def cancel_order(
        self, path: str, params: dict[str, Any], *, client_order_id: str | None = None
    ) -> Any:
        """撤单。结果未知同样需要查询恢复（撤单可能已生效）。"""
        return self.execute_critical("DELETE", path, params, client_order_id=client_order_id)

    def execute_critical(
        self, method: str, path: str, params: dict[str, Any] | None = None, *, client_order_id: str | None = None
    ) -> Any:
        """任意 critical 请求（会改变账户状态）。永不重试，结果未知抛 UnknownSubmission。

        用于下单/撤单/countdownCancelAll 等接口。
        """
        return self._execute(method.upper(), path, params, critical=True, client_order_id=client_order_id)

    # -- 内部 ----------------------------------------------------------------

    def _execute(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None,
        *,
        critical: bool,
        client_order_id: str | None,
        sign: bool = True,
    ) -> Any:
        max_retries = 0 if critical else self.read_max_retries
        self._throttle_if_needed()

        for attempt in range(max_retries + 1):
            self.stats.requests += 1
            try:
                response = self._send(method, path, params, sign=sign)
            except httpx.TimeoutException as exc:
                self._on_network_failure(
                    f"请求超时: {redact_url(str(exc))}", path, critical, client_order_id, attempt
                )
                continue
            except httpx.HTTPError as exc:
                self._on_network_failure(
                    f"网络错误: {redact_url(str(exc))}", path, critical, client_order_id, attempt
                )
                continue

            self._weight.observe(_parse_int_header(response.headers.get("x-mbx-used-weight-1m")))
            result = self._handle_response(
                response, path, critical, client_order_id, attempt, max_retries
            )
            if result is _Retry:
                continue
            return result

        # 重试耗尽仍失败
        raise NetworkError(f"请求 {path} 重试耗尽（{max_retries} 次）") from None

    def _send(self, method: str, path: str, params: dict[str, Any] | None, *, sign: bool = True) -> httpx.Response:
        params = dict(params or {})
        if sign:
            signed = sign_params(
                params,
                self._secret,
                timestamp_ms=self.timestamp_ms(),
                recv_window_ms=self.recv_window_ms,
            )
            # 签名走 query string（与币安文档示例一致）。
            query = "&".join(f"{key}={quote(str(value), safe='')}" for key, value in signed.items())
        else:
            # 公开接口：不附加 timestamp/signature/recvWindow（demo 端点会拒绝多余参数）。
            query = "&".join(f"{key}={quote(str(value), safe='')}" for key, value in params.items())
        url = f"{self.base_url}{path}?{query}" if query else f"{self.base_url}{path}"
        return self._client.request(method, url)

    def _on_network_failure(
        self,
        message: str,
        path: str,
        critical: bool,
        client_order_id: str | None,
        attempt: int,
    ) -> None:
        if critical:
            # 订单请求网络失败 = 结果未知。禁止自动重发。
            self.stats.unknown_submissions += 1
            raise UnknownSubmission(
                f"critical 请求网络失败，结果未知: {message} ({path})",
                client_order_id=client_order_id,
                market=self.market.value,
            )
        if attempt >= self.read_max_retries:
            raise NetworkError(f"{message}（重试耗尽）{path}") from None
        self.stats.retries += 1
        delay = self._backoff(attempt)
        logger.warning("%s，%.1fs 后重试 (%d/%d, %s)", message, delay, attempt + 1, self.read_max_retries, path)
        self._sleep(delay)

    def _handle_response(
        self,
        response: httpx.Response,
        path: str,
        critical: bool,
        client_order_id: str | None,
        attempt: int,
        max_retries: int,
    ) -> Any:
        status = response.status_code
        body = _safe_json(response.content)

        if status == 200:
            return body

        if status == 418:
            raise IPBanError(
                f"IP 已被币安临时封禁 (HTTP 418) at {path}。立即停止所有请求。", status=status
            )

        if status in (401, 403):
            self.stats.auth_errors += 1
            raise AuthError(
                f"认证/权限失败 (HTTP {status}) at {path}: {body}。必须立即停机检查 Key 权限。",
                status=status,
            )

        binance_code = _extract_code(body)

        if binance_code == CLOCK_DRIFT_CODE:
            raise ClockError(
                f"时钟偏移 (binance -1021) at {path}。停止交易，重新校准 server time 并检查 NTP。",
                code=binance_code,
                status=status,
            )
        if binance_code == SIGNATURE_ERROR_CODE:
            self.stats.auth_errors += 1
            raise AuthError(
                f"签名错误 (binance -1022) at {path}。停止交易，禁止自动重发。",
                code=binance_code,
                status=status,
            )

        if status == 429:
            self.stats.rate_limit_hits += 1
            if critical:
                # 429 = 请求未被提交。critical 请求不自动重试，上抛决策。
                raise RateLimitError(
                    f"critical 请求被限流 (HTTP 429) at {path}，请求未提交，不自动重试",
                    status=status,
                )
            if attempt >= max_retries:
                raise RateLimitError(f"限流重试耗尽 (HTTP 429) at {path}", status=status)
            retry_after = _parse_float_header(response.headers.get("retry-after"))
            delay = retry_after if retry_after is not None else self._backoff(attempt)
            logger.warning("被限流 (429)，%.1fs 后重试 (%d/%d, %s)", delay, attempt + 1, max_retries, path)
            self.stats.throttled_seconds += delay
            self.stats.retries += 1
            self._sleep(delay)
            return _Retry

        if status in (408, 500, 502, 503, 504):
            if critical:
                self.stats.unknown_submissions += 1
                raise UnknownSubmission(
                    f"critical 请求收到 HTTP {status}，订单结果未知: {body} ({path})",
                    client_order_id=client_order_id,
                    market=self.market.value,
                )
            if attempt >= max_retries:
                raise BinanceError(
                    f"服务端错误重试耗尽 (HTTP {status}) at {path}: {body}",
                    code=binance_code,
                    status=status,
                )
            delay = self._backoff(attempt)
            logger.warning("服务端 %d，%.1fs 后重试 (%d/%d, %s)", status, delay, attempt + 1, max_retries, path)
            self.stats.retries += 1
            self._sleep(delay)
            return _Retry

        # 其他 4xx：明确拒绝，不重试
        if binance_code is not None and binance_code in ORDER_REJECTED_CODES:
            raise OrderRejected(
                f"订单被交易所拒绝 at {path}: {body}", code=binance_code, status=status
            )
        raise BinanceError(
            f"API 错误 (HTTP {status}) at {path}: {body}", code=binance_code, status=status
        )

    def _throttle_if_needed(self) -> None:
        if self._weight.should_throttle():
            wait = self._weight.seconds_until_reset()
            logger.warning(
                "权重接近上限 (%d/%d)，休眠 %.1fs 等待窗口重置 (%s)",
                self._weight.used,
                self._weight.limit,
                wait,
                self.base_url,
            )
            self.stats.throttled_seconds += wait
            if wait > 0:
                self._sleep(wait)

    def _backoff(self, attempt: int) -> float:
        """指数退避 + 50%~100% 抖动（避免惊群）。"""
        base = 0.5
        cap = 30.0
        delay = min(cap, base * (2**attempt))
        return float(delay * (0.5 + random.random() * 0.5))


def _safe_json(content: bytes) -> Any:
    try:
        return json.loads(content)
    except (ValueError, TypeError):
        return {"raw": content.decode("utf-8", errors="replace")[:500]}


def _extract_code(body: Any) -> int | None:
    if isinstance(body, dict):
        code = body.get("code")
        if isinstance(code, int):
            return code
    return None


def _parse_int_header(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_float_header(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


__all__ = ["ORDER_REJECTED_CODES", "SignedClient", "TransportStats"]
