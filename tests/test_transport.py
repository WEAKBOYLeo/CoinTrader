"""transport 传输层单元测试（开发设计文档 §3.5 / §11.1）。

用 httpx.MockTransport 注入假响应，禁止真实网络。
重点：错误分类、critical 请求永不重试、限流退避、敏感信息不落日志/异常。
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from cointrader.errors import (
    AuthError,
    BinanceError,
    ClockError,
    IPBanError,
    NetworkError,
    OrderRejected,
    RateLimitError,
    UnknownSubmission,
)
from cointrader.execution.transport import SignedClient
from cointrader.secrets import SecretStr

SECRET = "test-secret-do-not-log"
KEY = "test-api-key"


def make_client(
    handler: Callable[[httpx.Request], httpx.Response], **overrides: object
) -> tuple[SignedClient, _RecordingSleep]:
    transport = httpx.MockTransport(handler)
    sleeper = _RecordingSleep()
    kwargs: dict[str, object] = {
        "base_url": "https://testnet.binance.example",
        "api_key": SecretStr(KEY, name="API_KEY"),
        "secret": SecretStr(SECRET, name="API_SECRET"),
        "market": "spot",
        "client": httpx.Client(transport=transport),
        "sleep_fn": sleeper,
        "now_fn": lambda: 1_700_000_000.0,
    }
    kwargs.update(overrides)
    return SignedClient(**kwargs), sleeper  # type: ignore[arg-type]


class _RecordingSleep:
    """记录退避调用，避免测试真实休眠。"""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _resp(status: int, body: dict | None = None, headers: dict[str, str] | None = None) -> httpx.Response:
    import json

    return httpx.Response(
        status,
        content=json.dumps(body or {}).encode(),
        headers=headers or {},
    )


class TestErrorClassification:
    def test_400_order_rejected_code_gives_order_rejected(self) -> None:
        c, _ = make_client(lambda req: _resp(400, {"code": -2010, "msg": "New order rejected"}))
        with pytest.raises(OrderRejected) as exc_info:
            c.get("/api/v3/order")
        assert exc_info.value.code == -2010

    def test_400_unknown_code_gives_binance_error(self) -> None:
        c, _ = make_client(lambda req: _resp(400, {"code": -1111, "msg": "weird"}))
        with pytest.raises(BinanceError) as exc_info:
            c.get("/api/v3/order")
        assert not isinstance(exc_info.value, OrderRejected)

    def test_401_gives_auth_error_and_counts_stat(self) -> None:
        c, _ = make_client(lambda req: _resp(401, {"code": -2014, "msg": "Bad API key"}))
        with pytest.raises(AuthError):
            c.get("/api/v3/account")
        assert c.stats.auth_errors == 1

    def test_403_gives_auth_error(self) -> None:
        c, _ = make_client(lambda req: _resp(403, {"code": -2015, "msg": "no perm"}))
        with pytest.raises(AuthError):
            c.get("/api/v3/account")

    def test_minus_1021_gives_clock_error(self) -> None:
        c, _ = make_client(lambda req: _resp(400, {"code": -1021, "msg": "timestamp outside recvWindow"}))
        with pytest.raises(ClockError) as exc_info:
            c.get("/api/v3/account")
        assert exc_info.value.code == -1021

    def test_minus_1022_gives_auth_error(self) -> None:
        c, _ = make_client(lambda req: _resp(400, {"code": -1022, "msg": "Bad signature"}))
        with pytest.raises(AuthError):
            c.get("/api/v3/account")

    def test_418_gives_ip_ban_error(self) -> None:
        c, _ = make_client(lambda req: _resp(418, {"code": -1003, "msg": "IP limited"}))
        with pytest.raises(IPBanError):
            c.get("/api/v3/account")

    def test_200_returns_body(self) -> None:
        c, _ = make_client(lambda req: _resp(200, {"serverTime": 123}))
        assert c.get("/api/v3/time") == {"serverTime": 123}


class TestCriticalNeverRetries:
    def test_5xx_on_place_order_raises_unknown_submission_once(self) -> None:
        calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            return _resp(500, {"code": -1003, "msg": "server down"})

        c, _ = make_client(handler)
        with pytest.raises(UnknownSubmission):
            c.place_order("/api/v3/order", {"symbol": "BTCUSDT"}, client_order_id="ct-test-1")
        assert len(calls) == 1, "critical 请求禁止自动重发（可能已成交）"
        assert c.stats.unknown_submissions == 1

    def test_408_on_critical_raises_unknown_submission(self) -> None:
        c, _ = make_client(lambda req: _resp(408, {}))
        with pytest.raises(UnknownSubmission):
            c.execute_critical("POST", "/fapi/v1/order", {"symbol": "BTCUSDT"}, client_order_id="ct-1")

    def test_timeout_on_critical_raises_unknown_submission_not_network(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("timeout")

        c, _ = make_client(handler)
        with pytest.raises(UnknownSubmission) as exc_info:
            c.place_order("/api/v3/order", {"symbol": "BTCUSDT"}, client_order_id="ct-timeout")
        assert exc_info.value.client_order_id == "ct-timeout"

    def test_429_on_critical_raises_rate_limit_error_without_retry(self) -> None:
        calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            return _resp(429, {"code": -1003, "msg": "too many"}, headers={"Retry-After": "2"})

        c, _ = make_client(handler)
        with pytest.raises(RateLimitError):
            c.place_order("/api/v3/order", {"symbol": "BTCUSDT"}, client_order_id="ct-rl")
        assert len(calls) == 1, "429 = 请求未提交，但 critical 仍不自动重试，上抛决策"

    def test_cancel_order_critical_5xx_is_unknown(self) -> None:
        c, _ = make_client(lambda req: _resp(502, {}))
        with pytest.raises(UnknownSubmission):
            c.cancel_order("/api/v3/order", {"symbol": "BTCUSDT"}, client_order_id="ct-c")


class TestReadRequestRetry:
    def test_5xx_retries_then_succeeds(self) -> None:
        calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            if len(calls) < 3:
                return _resp(503, {})
            return _resp(200, {"ok": True})

        c, _ = make_client(handler, read_max_retries=3)
        assert c.get("/api/v3/time") == {"ok": True}
        assert len(calls) == 3
        assert len(c._sleep.calls) == 2, "每次重试前应退避"  # noqa: SLF001

    def test_5xx_retries_exhausted_raises(self) -> None:
        calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            return _resp(500, {})

        c, _ = make_client(handler, read_max_retries=2)
        with pytest.raises(BinanceError):
            c.get("/api/v3/time")
        assert len(calls) == 3

    def test_429_respects_retry_after_header(self) -> None:
        calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            if len(calls) == 1:
                return _resp(429, {}, headers={"Retry-After": "7"})
            return _resp(200, {"ok": 1})

        c, sleeper = make_client(handler, read_max_retries=2)
        c.get("/api/v3/time")
        assert sleeper.calls == [7.0], "必须尊重 Retry-After 头"
        assert c.stats.rate_limit_hits == 1

    def test_network_timeout_retries_then_network_error(self) -> None:
        calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            raise httpx.ReadTimeout("read timeout")

        c, _ = make_client(handler, read_max_retries=1)
        with pytest.raises(NetworkError):
            c.get("/api/v3/time")
        assert len(calls) == 2


class TestWeightThrottle:
    def test_soft_limit_triggers_backoff_sleep(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return _resp(200, {"ok": 1}, headers={"x-mbx-used-weight-1m": "2000"})

        c, _ = make_client(handler, weight_limit=2400, soft_limit_ratio=0.80)
        c.get("/api/v3/time")  # 第一次：observe 2000 >= 2400*0.8
        c.get("/api/v3/time")  # 第二次：should_throttle → 休眠
        assert c.stats.throttled_seconds > 0, "软阈值前应主动降速"

    def test_low_weight_does_not_throttle(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return _resp(200, {"ok": 1}, headers={"x-mbx-used-weight-1m": "100"})

        c, _ = make_client(handler, weight_limit=2400)
        c.get("/api/v3/time")
        c.get("/api/v3/time")
        assert c.stats.throttled_seconds == 0


class TestSecretRedaction:
    def test_secret_never_in_url_or_exception(self) -> None:
        seen_urls: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen_urls.append(str(req.url))
            return _resp(401, {"code": -2014, "msg": "bad key"})

        c, _ = make_client(handler)
        with pytest.raises(AuthError) as exc_info:
            c.get("/api/v3/account")
        for url in seen_urls:
            assert SECRET not in url, "Secret 不得出现在请求 URL"
            assert KEY not in url, "API Key 走 header，不得进 URL query"
        text = str(exc_info.value)
        assert SECRET not in text and KEY not in text, "异常信息不得含凭证"

    def test_signature_present_in_query_but_not_secret(self) -> None:
        seen: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(str(req.url))
            return _resp(200, {"ok": 1})

        c, _ = make_client(handler)
        c.get("/api/v3/account")
        assert "signature=" in seen[0]
        assert SECRET not in seen[0]
