"""进程内 Binance 权重限流协调器（Spot/Futures 共享调度的唯一拥有者）。

设计（实施计划书 v2.0 T1）：

- 同一进程里公开客户端（``data/binance.py``）与签名客户端
  （``execution/transport.py``）注入**同一个** ``RateLimitCoordinator`` 实例，
  共享按市场分组的 weight 预算 —— 消除两套私有 ``_WeightTracker``
  互相遮蔽的问题。
- 预算真相以响应头 ``x-mbx-used-weight-1m`` 为准（服务端视角，含同出口 IP
  其他进程流量）；缺头时仅用本地 in-flight/已完成估算保守记账。
- 优先级固定（数值越小越高）：P0 下单/撤单、P1 恢复/对账、P2 已有仓位、
  P3 候选刷新、P4 历史回填。P0/P1 可用满限额（保留预算）；P2 受软限；
  P3/P4 受（软限 − critical reserve）约束，候选刷新不能吃掉下单与对账预算。
- 429 → 冻结该市场（至少 ``freeze_seconds``）；418 → 封禁该市场
  （至少 ``ban_seconds``）。冻结/封禁期间禁止对该 scope 的任何探测请求。
- 等待/睡眠不持锁；in-flight 权重在全部异常路径上释放（permit 上下文配对）。
- critical 请求不获得任何自动重试语义 —— 重试策略仍由各客户端自己决定。

本模块只依赖标准库，位于包顶层，``data/`` 与 ``execution/`` 均可引用，
不破坏「下层包不得 import execution」的依赖方向约束。
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

from .errors import RateLimitError

logger = logging.getLogger(__name__)

#: 币安服务端限流窗口长度（分钟 → 秒）。
_WINDOW_SECONDS = 60.0
#: 让步/轮询等待的最小值，避免紧密循环烧 CPU。
_MIN_WAIT_SECONDS = 0.05
#: 未知端点的保守默认 weight（宁高勿低，响应头观测值会覆盖估算）。
DEFAULT_ENDPOINT_WEIGHT = 10


class RateLimitScope(str, Enum):
    """限流 scope：Spot 与 USDS-M 合约是两套独立计数器（不同域名）。"""

    SPOT = "spot"
    FUTURES = "futures"


class RequestPriority(IntEnum):
    """请求优先级。数值越小优先级越高；低优先级不得越过先等待的高优先级。"""

    P0_CRITICAL = 0
    P1_RECOVERY = 1
    P2_POSITION = 2
    P3_CANDIDATE = 3
    P4_BACKFILL = 4


#: 可使用完整限额（含保留预算）的优先级：下单/恢复。
_RESERVE_PRIORITIES = frozenset({RequestPriority.P0_CRITICAL, RequestPriority.P1_RECOVERY})
#: 受软限约束的优先级：已有仓位行情/事实。
_SOFT_PRIORITIES = frozenset({RequestPriority.P2_POSITION})


class RateLimitBannedError(RateLimitError):
    """418 IP 封禁生效中：继续请求会延长封禁，必须停止而非重试。"""


class RateLimitBusyError(RateLimitError):
    """acquire 等待达到 timeout，请求未发出（与「已发送结果未知」不同，可安全重发）。"""


@dataclass(frozen=True)
class RateLimitSnapshot:
    """某 scope 的只读限流状态快照（供 Web/日志诊断）。

    不暴露 URL、签名或密钥。时间均为注入时钟的单调秒值。
    """

    scope: RateLimitScope
    limit: int
    soft_limit: float
    low_limit: float
    observed_used: int | None
    in_flight: int
    local_used: int
    frozen_until: float | None
    ban_until: float | None
    waiting_by_priority: tuple[tuple[RequestPriority, int], ...]
    freezes: int
    bans: int
    diagnostic_errors: int


@dataclass
class _ScopeState:
    """单个 scope 的可变限流状态（仅在 coordinator 锁内访问）。"""

    limit: int
    observed_used: int | None = None
    observed_at: float = 0.0
    local_used: int = 0
    window_start: float = 0.0
    in_flight: int = 0
    frozen_until: float | None = None
    ban_until: float | None = None
    waiting: dict[RequestPriority, int] = field(default_factory=dict)
    freezes: int = 0
    bans: int = 0
    diagnostic_errors: int = 0


class RateLimitCoordinator:
    """进程内共享的按市场 weight 限流协调器。

    调用方契约：

    - 每次 HTTP 请求发出前 ``acquire(scope, priority, estimated_weight)``，
      响应到达后 ``observe(scope, header, status, retry_after)``；
    - permit 是上下文管理器，HTTP 执行不在锁内，异常路径自动释放 in-flight。

    Args:
        limits: 每个 scope 的每分钟 weight 硬限额。可只给部分 scope，
            其余用 ``default_limit``。
        soft_limit_ratio: 软限比例；P2 受软限，P3/P4 受（软限 − reserve）。
        critical_reserve_ratio: P0/P1 保留预算比例，P2-P4 不得占用。
        freeze_seconds: 429 无/短 Retry-After 时的最小保护冻结（秒）。
        ban_seconds: 418 无有效 Retry-After 时的 IP 封禁（秒）。
        clock/sleep: 可注入（测试用假时钟）；sleep 应推进 clock 所在时间轴。
    """

    def __init__(
        self,
        limits: Mapping[RateLimitScope, int],
        *,
        default_limit: int = 2400,
        soft_limit_ratio: float = 0.80,
        critical_reserve_ratio: float = 0.30,
        freeze_seconds: float = 120.0,
        ban_seconds: float = 3600.0,
        clock: Any = time.monotonic,
        sleep: Any = time.sleep,
    ) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit 必须为正")
        self._limits: dict[RateLimitScope, int] = {scope: int(limits.get(scope, default_limit)) for scope in RateLimitScope}
        if any(limit <= 0 for limit in self._limits.values()):
            raise ValueError("scope 限额必须为正")
        self._soft = float(soft_limit_ratio)
        self._reserve = float(critical_reserve_ratio)
        self._freeze_seconds = float(freeze_seconds)
        self._ban_seconds = float(ban_seconds)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._states: dict[RateLimitScope, _ScopeState] = {
            scope: _ScopeState(limit=limit, waiting={}) for scope, limit in self._limits.items()
        }
        self._seq = itertools.count()
        self._warned_paths: set[tuple[RateLimitScope, str]] = set()

    # -- 基础访问 -----------------------------------------------------------

    def now(self) -> float:
        """注入时钟的当前单调秒值（客户端用它统计等待时长）。"""
        return float(self._clock())

    def cap_for(self, scope: RateLimitScope, priority: RequestPriority) -> float:
        """该优先级的可用 weight 上限（不含已用）。"""
        limit = self._states[scope].limit
        if priority in _RESERVE_PRIORITIES:
            return float(limit)
        if priority in _SOFT_PRIORITIES:
            return limit * self._soft
        return limit * max(0.0, self._soft - self._reserve)

    def effective_used(self, scope: RateLimitScope) -> int:
        """当前已用（含 in-flight）的估算值，仅用于诊断。"""
        with self._lock:
            return self._effective_used_locked(self._states[scope])

    @staticmethod
    def _effective_used_locked(st: _ScopeState) -> int:
        return (st.observed_used if st.observed_used is not None else st.local_used) + st.in_flight

    # -- 核心：permit 申请 ---------------------------------------------------

    @contextmanager
    def acquire(
        self,
        scope: RateLimitScope,
        priority: RequestPriority | int,
        estimated_weight: int = 1,
        *,
        critical: bool = False,
        timeout: float | None = None,
    ) -> Iterator[None]:
        """申请一次请求 permit：按优先级/预算等待，发请求前获取、结束后释放。

        Args:
            scope: 限流市场。
            priority: 请求优先级。
            estimated_weight: 估算 weight（>=1）；响应头观测值才是权威。
            critical: 仅为语义标记 —— 本协调器从不重试，重试策略归调用方。
            timeout: 最长等待（注入时钟秒）；None = 等到可行为止。

        Raises:
            RateLimitBannedError: scope 处于 418 封禁期，立即上抛。
            RateLimitBusyError: timeout 到期仍未获准。
            ValueError: estimated_weight < 1。
        """
        if estimated_weight < 1:
            raise ValueError(f"estimated_weight 必须 >= 1，当前 {estimated_weight}")
        priority = RequestPriority(int(priority))
        deadline: float | None = None if timeout is None else self._clock() + timeout
        start = self._clock()
        while True:
            wait_s = _MIN_WAIT_SECONDS
            remaining = 0.0
            with self._lock:
                st = self._states[scope]
                st.waiting[priority] = st.waiting.get(priority, 0) + 1
                try:
                    wait_s, reason = self._wait_needed_locked(st, scope, priority, estimated_weight)
                    if wait_s <= 0.0:
                        st.in_flight += estimated_weight
                        break
                    if reason == "banned":
                        remaining = wait_s
                finally:
                    st.waiting[priority] -= 1
            if reason == "banned":
                raise RateLimitBannedError(
                    f"{scope.value} scope 处于 418 IP 封禁期（约 {remaining:.0f}s），"
                    "继续请求会延长封禁。必须停止该 scope 的所有请求。"
                )
            if reason == "budget" and (self._clock() - start) >= _WINDOW_SECONDS:
                # 已等满一个服务端分钟窗口仍无新 header：放行，由 429 熔断兜底
                # （旧 _WeightTracker 的「休眠到窗口重置后继续」语义）
                with self._lock:
                    self._states[scope].in_flight += estimated_weight
                logger.warning(
                    "%s scope 预算等待已达一个完整窗口（%.0fs），放行并依赖 429 熔断判断",
                    scope.value, _WINDOW_SECONDS,
                )
                break
            if deadline is not None:
                remaining_budget = deadline - self._clock()
                if remaining_budget <= 0:
                    raise RateLimitBusyError(
                        f"{scope.value} scope 限流等待超时（{timeout}s），请求未发出"
                    )
                wait_s = min(wait_s, remaining_budget)
            self._sleep(max(wait_s, 0.0))
        waited = self._clock() - start
        if waited > 0.05:
            logger.info(
                "限流等待 %.1fs 后放行（scope=%s priority=%s weight=%d critical=%s）",
                waited, scope.value, priority.name, estimated_weight, critical,
            )
        else:
            logger.debug(
                "限流放行（scope=%s priority=%s weight=%d waited=%.3fs）",
                scope.value, priority.name, estimated_weight, waited,
            )
        try:
            yield
        finally:
            self._release(scope, estimated_weight)

    def _release(self, scope: RateLimitScope, weight: int) -> None:
        with self._lock:
            st = self._states[scope]
            st.in_flight = max(0, st.in_flight - weight)
            # 没有服务端 header 时本地记账：完成请求计入 local_used
            if st.observed_used is None:
                st.local_used += weight
            st.window_start = self._clock()

    def _wait_needed_locked(
        self, st: _ScopeState, scope: RateLimitScope, priority: RequestPriority, weight: int
    ) -> tuple[float, str]:
        """返回 (需要等待的秒数, 原因)。0 表示可立即放行（调用方随后放行）。"""
        now = self._clock()
        # 封禁：立即上抛（不等待 —— 等待一小时没有意义，语义是停机）
        if st.ban_until is not None:
            if now < st.ban_until:
                return st.ban_until - now, "banned"
            st.ban_until = None
        # 冻结：等到冻结期结束
        if st.frozen_until is not None:
            if now < st.frozen_until:
                return st.frozen_until - now, "frozen"
            st.frozen_until = None
        # 本地估算窗口滚动：60 秒无任何观测/请求 → 本地计数清零
        if st.observed_used is None and st.window_start and now - st.window_start >= _WINDOW_SECONDS:
            st.local_used = 0
            st.window_start = now
        # 预算
        cap = self.cap_for(scope, priority)
        if self._effective_used_locked(st) + weight > cap:
            anchor = st.observed_at if st.observed_used is not None else max(st.window_start, 1.0)
            eta = anchor + _WINDOW_SECONDS
            return max(_MIN_WAIT_SECONDS, eta - now), "budget"
        # 公平性：存在先等待的高优先级请求时让步
        for wait_priority, count in st.waiting.items():
            if count > 0 and wait_priority < priority:
                return _MIN_WAIT_SECONDS, "preempt"
        return 0.0, "ok"

    # -- 核心：响应反馈 -------------------------------------------------------

    def observe(
        self,
        scope: RateLimitScope,
        used_weight: int | None,
        status: int,
        retry_after_s: float | None = None,
    ) -> None:
        """每次收到 HTTP 响应后调用：更新已用权重观测并执行 429/418 熔断。

        - header 下降视为服务端窗口滚动（直接覆盖旧值）；
        - 负值/非法 header 忽略并计诊断错误；
        - 429 → 冻结 ``max(Retry-After, freeze_seconds)``；
        - 418 → 封禁 ``max(Retry-After, ban_seconds)``。
        """
        with self._lock:
            st = self._states[scope]
            now = self._clock()
            if used_weight is not None:
                if used_weight >= 0:
                    st.observed_used = int(used_weight)
                    st.observed_at = now
                    st.local_used = 0  # 服务端 header 接管记账
                    st.window_start = now
                else:
                    st.diagnostic_errors += 1
            if status == 429:
                wait = max(retry_after_s or 0.0, self._freeze_seconds)
                st.frozen_until = max(st.frozen_until or 0.0, now + wait)
                st.freezes += 1
                logger.warning("429 限流：冻结 %s scope %.0fs", scope.value, wait)
            elif status == 418:
                wait = max(retry_after_s or 0.0, self._ban_seconds)
                st.ban_until = max(st.ban_until or 0.0, now + wait)
                st.bans += 1
                logger.error("418 IP 封禁：封禁 %s scope %.0fs", scope.value, wait)

    # -- 诊断 ----------------------------------------------------------------

    def snapshot(self) -> Mapping[RateLimitScope, RateLimitSnapshot]:
        """只读快照（按 scope 排序），供 Web/日志展示；不含 URL/签名/密钥。"""
        result: dict[RateLimitScope, RateLimitSnapshot] = {}
        with self._lock:
            for scope in RateLimitScope:
                st = self._states[scope]
                result[scope] = RateLimitSnapshot(
                    scope=scope,
                    limit=st.limit,
                    soft_limit=st.limit * self._soft,
                    low_limit=st.limit * max(0.0, self._soft - self._reserve),
                    observed_used=st.observed_used,
                    in_flight=st.in_flight,
                    local_used=st.local_used,
                    frozen_until=st.frozen_until,
                    ban_until=st.ban_until,
                    waiting_by_priority=tuple(
                        (p, c) for p, c in sorted(st.waiting.items()) if c > 0
                    ),
                    freezes=st.freezes,
                    bans=st.bans,
                    diagnostic_errors=st.diagnostic_errors,
                )
        return result

    # -- 未知端点告警 ---------------------------------------------------------

    def note_unknown_endpoint(self, scope: RateLimitScope, path: str) -> None:
        """未知端点使用保守默认 weight；每个 (scope,path) 只告警一次。"""
        with self._lock:
            key = (scope, path)
            if key not in self._warned_paths:
                self._warned_paths.add(key)
                logger.warning(
                    "%s 端点 %s 无已核实 weight 映射，使用保守默认 %d（响应头观测仍会覆盖）",
                    scope.value, path, DEFAULT_ENDPOINT_WEIGHT,
                )


# ---------------------------------------------------------------------------
# 端点 weight 估算（集中在本模块；响应头观测值始终是权威）
# ---------------------------------------------------------------------------


def klines_weight(limit: int) -> int:
    """K 线端点按请求 limit 分档（官方口径，2026-09 本地实测吻合）。"""
    if limit <= 100:
        return 1
    if limit <= 500:
        return 2
    if limit <= 1000:
        return 5
    return 10


# 简单端点（不随 symbol/limit 变化）的已核实 weight。
_SPOT_ENDPOINT_WEIGHTS: dict[str, int] = {
    "/api/v3/ping": 1,
    "/api/v3/time": 1,
    "/api/v3/ticker/price": 2,
    "/api/v3/account": 20,
    "/api/v3/openOrders": 5,
    "/api/v3/order": 1,
    "/api/v3/listenKey": 1,
}

_FAPI_ENDPOINT_WEIGHTS: dict[str, int] = {
    "/fapi/v1/ping": 1,
    "/fapi/v1/time": 1,
    "/fapi/v1/exchangeInfo": 1,
    "/fapi/v1/fundingInfo": 30,
    "/fapi/v1/openInterest": 1,
    "/fapi/v2/account": 5,
    "/fapi/v2/balance": 5,
    "/fapi/v2/positionRisk": 5,
    "/fapi/v2/order": 1,
    "/fapi/v2/openOrders": 1,
    "/fapi/v1/userTrades": 10,
    "/fapi/v1/income": 30,
    "/fapi/v1/order": 1,
    "/fapi/v1/countdownCancelAll": 1,
    "/fapi/v1/listenKey": 1,
}


def endpoint_weight(
    scope: RateLimitScope,
    path: str,
    params: Mapping[str, Any] | None = None,
    *,
    on_unknown: Any = None,
) -> int:
    """估算一次请求的 weight。

    集中管理本项目实际使用的端点；未知端点返回保守默认
    ``DEFAULT_ENDPOINT_WEIGHT`` 并（可选）通过 ``on_unknown`` 记录告警。
    响应头 ``x-mbx-used-weight-1m`` 的观测值始终是权威，估算只影响
    in-flight 预算记账。
    """
    params = dict(params or {})
    has_symbol = "symbol" in params
    limit = params.get("limit", 100)
    try:
        limit = int(limit)  # type: ignore[no-untyped-call]
    except (TypeError, ValueError):
        limit = 100

    if path in ("/api/v3/klines", "/fapi/v1/klines"):
        return klines_weight(limit)
    if path == "/fapi/v1/fundingRate":
        return 10 if limit > 1000 else 1
    if path in ("/api/v3/ticker/24hr", "/fapi/v1/ticker/24hr"):
        return 2 if has_symbol else 80
    if path == "/api/v3/exchangeInfo":
        return 2 if has_symbol else 20
    if path == "/fapi/v1/premiumIndex":
        return 1 if has_symbol else 10
    if path == "/api/v3/myTrades":
        return 50 if limit > 100 else 25

    table = _SPOT_ENDPOINT_WEIGHTS if scope is RateLimitScope.SPOT else _FAPI_ENDPOINT_WEIGHTS
    weight = table.get(path)
    if weight is None:
        if on_unknown is not None:
            on_unknown(scope, path)
        return DEFAULT_ENDPOINT_WEIGHT
    return weight


__all__ = [
    "DEFAULT_ENDPOINT_WEIGHT",
    "RateLimitBannedError",
    "RateLimitBusyError",
    "RateLimitCoordinator",
    "RateLimitScope",
    "RateLimitSnapshot",
    "RequestPriority",
    "endpoint_weight",
    "klines_weight",
]
