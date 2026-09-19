"""异常层级。

刻意区分「可重试」与「不可重试」，因为交易系统里把不可重试的错误
当成可重试（或反之）都会造成实际损失：

- 把限流错误当成致命错误 → 错过行情
- 把参数错误当成可重试 → 无限重试，浪费权重配额，最终被封 IP
"""

from __future__ import annotations


class CoinTraderError(Exception):
    """本项目所有异常的基类。"""


# ---------------------------------------------------------------------------
# 网络与 API
# ---------------------------------------------------------------------------


class BinanceError(CoinTraderError):
    """币安 API 返回的错误。"""

    def __init__(self, message: str, *, code: int | None = None, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.code is not None:
            parts.append(f"binance_code={self.code}")
        if self.status is not None:
            parts.append(f"http_status={self.status}")
        return " | ".join(parts)


class RateLimitError(BinanceError):
    """HTTP 429 —— 触及限流。可重试，但必须退避。"""


class IPBanError(BinanceError):
    """HTTP 418 —— IP 已被临时封禁。

    ⚠️ 这不是「再试一次」的场景。继续请求会延长封禁时间。
    收到此错误必须停止所有请求并通知运维。
    """


class NetworkError(CoinTraderError):
    """连接层失败（超时、DNS、TLS）。可重试。"""


class ParseError(CoinTraderError):
    """响应格式与预期不符。不可重试 —— 重试只会拿到同样的坏数据。"""


# ---------------------------------------------------------------------------
# 配置与数据
# ---------------------------------------------------------------------------


class ConfigError(CoinTraderError):
    """配置缺失或非法。不可重试，必须人工修正。"""


class DataUnavailableError(BinanceError):
    """请求的数据不存在（如币种已下架）。不可重试。

    继承 ``BinanceError`` 以便携带 ``code`` / ``status`` ——
    调用方经常需要区分"币种不存在"（-1121）与"参数格式错"（-1100）。
    """


class InsufficientDataError(CoinTraderError):
    """数据不足以完成计算（如回测期数不足）。不可重试。"""


# ---------------------------------------------------------------------------
# 执行与风控
# ---------------------------------------------------------------------------


class GuardRejected(CoinTraderError):
    """下单被安全闸门拒绝。

    这是**预期行为**，不是 bug。默认配置下所有下单都会走到这里。
    """


class RiskLimitExceeded(CoinTraderError):
    """触及风控限额。"""


class KillSwitchEngaged(CoinTraderError):
    """停机开关已触发。"""


class AuthError(BinanceError):
    """认证/权限错误（HTTP 401/403 或 -1022 签名错误）。

    ⚠️ 收到此错误必须**立即停止交易**。不能自动重试同一请求，
    签名错误重复发送只会暴露更多失败并浪费权重。
    """


class ClockError(BinanceError):
    """时钟偏移错误（-1021 timestamp outside recvWindow）。

    停止交易，重新获取 server time 并检查系统 NTP。
    """


class OrderRejected(BinanceError):
    """交易所明确拒绝了订单/撤单（-2010/-2011/-2013 等）。

    与网络故障不同：交易所**明确**说“这单没接”。
    因此不需要进入 UNKNOWN 恢复流程，直接按拒单处理。
    """


class UnknownSubmission(CoinTraderError):
    """下单/撤单请求发出后结果未知（超时、408、5xx、网络中断）。

    ⚠️ 此时订单**可能已经成交**。绝对禁止盲目重发同一订单。
    唯一合法的下一步：用 ``clientOrderId`` 查询实际状态。
    """

    def __init__(self, message: str, *, client_order_id: str | None = None, market: str | None = None):
        super().__init__(message)
        self.client_order_id = client_order_id
        self.market = market


class IllegalStateTransition(CoinTraderError):
    """订单状态机收到非法转移（如 FILLED → NEW）。"""


class RecoveryRequired(CoinTraderError):
    """系统处于 RECOVERY，禁止开新仓。"""


class LiveGateBlocked(CoinTraderError):
    """实盘闸门阻止操作（模式不对、未对账、HALT_NEW_RISK 等）。"""


__all__ = [
    "BinanceError",
    "CoinTraderError",
    "ConfigError",
    "DataUnavailableError",
    "AuthError",
    "ClockError",
    "GuardRejected",
    "IPBanError",
    "IllegalStateTransition",
    "LiveGateBlocked",
    "OrderRejected",
    "RecoveryRequired",
    "UnknownSubmission",
    "InsufficientDataError",
    "KillSwitchEngaged",
    "NetworkError",
    "ParseError",
    "RateLimitError",
    "RiskLimitExceeded",
]
