"""Spot/Futures 用户数据流管理（开发设计文档 §3.4）。

规则：

1. REST 与 WebSocket **不能互相替代**：WS 负责低延迟事件，
   REST 负责提交与校准。
2. 断线、过期、序列不连续或事件解析失败 → 立即 ``on_untrusted``
   （上层进入 RECOVERY，先 REST 对账再继续）。
3. 重连**不能只等新事件**：重连后必须 REST 拉取账户/开放订单/持仓
   再恢复接收 —— 这个动作由上层的 ``on_untrusted`` 回调触发，
   本模块只负责标记与重连。
4. 事件带接收时间、交易所事件时间、连接代次和原始事件指纹；
   重复事件（同指纹）只上报一次。
5. 本模块不解析业务语义（那是 pair_executor 的事），
   只做连接、保活、时间新鲜度和指纹去重。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["PollingUserStream", "StreamEvent", "UserStream"]


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """一次用户流事件（脱敏后）。"""

    market: str
    event_type: str
    exchange_ts_ms: int | None
    recv_ts_ms: int
    generation: int  # 连接代次。代次变化 = 发生过重连
    fingerprint: str  # 原始消息 sha256 前 16 位
    payload: dict[str, Any]


class UserStream:
    """单条用户数据流的连接/保活/新鲜度管理（后台守护线程）。

    Args:
        market: "spot" | "perp"。
        adapter: 具备 create_listen_key/keepalive_listen_key/close_listen_key 的对象。
        ws_base: WebSocket 根 URL（如 wss://stream.binance.com:9443/ws）。
        on_event: 事件回调（调用方负责解析与持久化）。
        on_untrusted: 状态不可信回调（断线/过期/乱序/解析失败）。
        keepalive_seconds: listenKey 保活间隔。
        staleness_seconds: 超过该秒数无任何事件即视为陈旧。
        connect_factory: 可注入的 WS 连接工厂（测试用）。
        max_reconnect_backoff_seconds: 重连退避上限。
    """

    def __init__(
        self,
        *,
        market: str,
        adapter: Any,
        ws_base: str,
        on_event: Callable[[StreamEvent], None],
        on_untrusted: Callable[[str], None] | None = None,
        keepalive_seconds: float = 1800.0,
        staleness_seconds: float = 30.0,
        connect_factory: Callable[[str], Any] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = time.time,
        max_reconnect_backoff_seconds: float = 30.0,
    ) -> None:
        self.market = market
        self.adapter = adapter
        self.ws_base = ws_base
        self.on_event = on_event
        self.on_untrusted = on_untrusted
        self.keepalive_seconds = keepalive_seconds
        self.staleness_seconds = staleness_seconds
        self._connect_factory = connect_factory or _default_connect
        self._sleep = sleep_fn
        self._now = now_fn
        self._max_backoff = max_reconnect_backoff_seconds

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ws: Any | None = None
        self._listen_key: str | None = None
        self.generation = 0
        self.last_event_ts: float | None = None
        self._last_exchange_ts: int | None = None
        self._seen_fps: set[str] = set()
        self.connected = False
        self.untrusted = False

    # -- 状态 ---------------------------------------------------------------

    @property
    def is_fresh(self) -> bool:
        if self.untrusted or self.last_event_ts is None:
            return False
        return (self._now() - self.last_event_ts) <= self.staleness_seconds

    @property
    def age_seconds(self) -> float | None:
        if self.last_event_ts is None:
            return None
        return self._now() - self.last_event_ts

    def mark_untrusted(self, reason: str) -> None:
        """标记状态不可信并通知上层。幂等：连续标记只通知一次，
        直到重连成功并收到新鲜事件后解除。"""
        if self.untrusted:
            return
        self.untrusted = True
        logger.error("【用户流不可信】%s: %s", self.market, reason)
        if self.on_untrusted:
            self.on_untrusted(reason)

    # -- 生命周期 -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"user-stream-{self.market}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        ws, self._ws = self._ws, None
        if ws is not None:
            with suppress(Exception):
                ws.close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._listen_key:
            try:
                self.adapter.close_listen_key(self._listen_key)
            except Exception:  # noqa: BLE001
                logger.warning("关闭 listenKey 失败（孤儿 key 无害，60 分钟自动过期）", exc_info=True)
            self._listen_key = None

    # -- 主循环 -------------------------------------------------------------

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._ensure_listen_key()
                self.generation += 1
                url = f"{self.ws_base.rstrip('/')}/ws/{self._listen_key}"
                self._ws = self._connect_factory(url)
                self.connected = True
                self.untrusted = False
                self._last_exchange_ts = None
                backoff = 1.0
                logger.info("【用户流已连接】%s 代次=%d", self.market, self.generation)
                self._recv_loop()
            except _StaleStream as exc:
                self.mark_untrusted(str(exc))
            except _StopRequested:
                break
            except Exception as exc:  # noqa: BLE001
                self.mark_untrusted(f"连接异常: {exc}")
            finally:
                self.connected = False
                ws, self._ws = self._ws, None
                if ws is not None:
                    with suppress(Exception):
                        ws.close()
            if self._stop.is_set():
                break
            # 断线重连：先退避。重连成功后的 REST 对账由 on_untrusted 触发。
            self._sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff)

    def _ensure_listen_key(self) -> None:
        if self._listen_key is None:
            self._listen_key = self.adapter.create_listen_key()

    def _recv_loop(self) -> None:
        ws = self._ws
        if ws is None:  # pragma: no cover - 防御性分支
            return
        last_keepalive = self._now()
        while not self._stop.is_set():
            if self._now() - last_keepalive >= self.keepalive_seconds:
                self.adapter.keepalive_listen_key(self._listen_key)
                last_keepalive = self._now()

            try:
                message = ws.recv(timeout=5)
            except TimeoutError:
                self._check_freshness()
                continue

            self._handle_message(message)

    def _check_freshness(self) -> None:
        age = self.age_seconds
        if age is not None and age > self.staleness_seconds * 4:
            # 长时间无事件。用户流在无事件时本来就静默，
            # 所以只有**曾经活跃后长时间无事件**才算异常。
            raise _StaleStream(f"用户流 {age:.0f}s 无事件（超过 {self.staleness_seconds * 4:.0f}s 阈值）")

    def _handle_message(self, message: str | bytes) -> None:
        try:
            payload = json.loads(message)
        except (ValueError, TypeError) as exc:
            self.mark_untrusted(f"事件解析失败: {exc}")
            return
        if not isinstance(payload, dict):
            self.mark_untrusted("事件不是 JSON 对象")
            return

        raw = message if isinstance(message, str) else message.decode("utf-8", errors="replace")
        fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        if fingerprint in self._seen_fps:
            return  # 重复事件
        self._seen_fps.add(fingerprint)
        if len(self._seen_fps) > 4096:  # 防内存膨胀
            self._seen_fps = set(list(self._seen_fps)[-1024:])

        event_type = str(payload.get("e") or payload.get("event") or "unknown")

        # listenKey 过期事件 → 不可信，需要重建
        if event_type.lower() == "listenkeyexpired":
            self.mark_untrusted("listenKey 已过期")
            self._listen_key = None
            raise _StaleStream("listenKey expired")

        exchange_ts = payload.get("E")
        if exchange_ts is None:
            exchange_ts = payload.get("T")
        exchange_ts = int(exchange_ts) if exchange_ts is not None else None

        # 乱序检查（容忍 5 秒时钟抖动）
        if (
            exchange_ts is not None
            and self._last_exchange_ts is not None
            and exchange_ts < self._last_exchange_ts - 5000
        ):
            self.mark_untrusted(
                f"乱序事件: {exchange_ts} < {self._last_exchange_ts}（代次 {self.generation}）"
            )
            return
        if exchange_ts is not None:
            self._last_exchange_ts = max(exchange_ts, self._last_exchange_ts or 0)

        self.last_event_ts = self._now()
        self.on_event(
            StreamEvent(
                market=self.market,
                event_type=event_type,
                exchange_ts_ms=exchange_ts,
                recv_ts_ms=int(self._now() * 1000),
                generation=self.generation,
                fingerprint=fingerprint,
                payload=payload,
            )
        )


class _StaleStream(Exception):
    """用户流不可信（超时/过期）。"""


class _StopRequested(Exception):
    pass


def _default_connect(url: str) -> Any:
    """默认 WS 连接：websockets 同步客户端。"""
    import websockets.sync.client

    return websockets.sync.client.connect(url, open_timeout=10, close_timeout=5)


class PollingUserStream:
    """用户流的 REST 轮询替代（demo trading 用）。

    币安 demo trading 的用户数据流不可用：现货 POST /userDataStream 返回
    410 Gone，合约 user stream WS 连接约 20 秒后被服务端强制断开（行情流正常）。
    本类以周期性 REST 账户查询替代「新鲜度」判定：

    - 连续轮询成功且在新鲜窗口内 → ``is_fresh``；
    - 连续失败超过阈值 → 标记不可信并回调 ``on_untrusted``（上层进入 RECOVERY）；
    - 成交/订单状态由执行器自身订单轮询与周期对账发现，本类无事件语义。

    存活冗余（单线程挂起防线）：轮询跑在独立 worker 线程，另有 monitor
    线程监视 worker 心跳。worker 卡死（如代理连接黑洞，account() 永不返回）
    或异常退出 → monitor 弃旧线程、以新代次重建 worker，无需重启进程。
    被弃的旧线程是 daemon，若日后从挂起中醒来只看到 retire 标记即静默退出，
    其迟到的失败不得影响新代次状态（所有代次敏感字段均在锁内替换）。

    接口与 :class:`UserStream` 保持子集兼容（market/generation/is_fresh/
    start/stop），便于上层无缝切换。
    """

    def __init__(
        self,
        *,
        market: str,
        adapter: Any,
        poll_seconds: float = 5.0,
        fresh_seconds: float = 15.0,
        max_consecutive_failures: int = 3,
        hang_threshold_seconds: float = 120.0,
        monitor_check_seconds: float = 5.0,
        on_untrusted: Callable[[str], None] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        if poll_seconds <= 0 or fresh_seconds < poll_seconds:
            raise ValueError("fresh_seconds 必须 >= poll_seconds > 0")
        if hang_threshold_seconds <= 0 or monitor_check_seconds <= 0:
            raise ValueError("hang_threshold_seconds / monitor_check_seconds 必须 > 0")
        self.market = market
        self.adapter = adapter
        self.poll_seconds = poll_seconds
        self.fresh_seconds = fresh_seconds
        self.max_consecutive_failures = max(1, max_consecutive_failures)
        self.hang_threshold_seconds = hang_threshold_seconds
        self.monitor_check_seconds = monitor_check_seconds
        self.on_untrusted = on_untrusted
        self._sleep = sleep_fn
        self._now = now_fn
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._retire: threading.Event | None = None
        self._monitor: threading.Thread | None = None
        self.generation = 0
        self.last_poll_ts: float | None = None
        self._last_attempt_ts: float | None = None
        self.connected = False
        self.untrusted = False

    # -- 状态 ---------------------------------------------------------------

    @property
    def is_fresh(self) -> bool:
        if self.untrusted or self.last_poll_ts is None:
            return False
        return (self._now() - self.last_poll_ts) <= self.fresh_seconds

    @property
    def age_seconds(self) -> float | None:
        if self.last_poll_ts is None:
            return None
        return self._now() - self.last_poll_ts

    def _mark_untrusted(self, reason: str) -> None:
        if self.untrusted:
            return
        self.untrusted = True
        logger.error("【轮询流不可信】%s: %s", self.market, reason)
        if self.on_untrusted:
            self.on_untrusted(reason)

    # -- 生命周期 -----------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop.clear()
            self._start_worker_locked("start")
            if self._monitor is None or not self._monitor.is_alive():
                self._monitor = threading.Thread(
                    target=self._monitor_loop, name=f"poll-monitor-{self.market}", daemon=True
                )
                self._monitor.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            retire, self._retire = self._retire, None
            worker, self._worker = self._worker, None
        if retire is not None:
            retire.set()
        if worker is not None:
            worker.join(timeout=5)
        if self._monitor is not None:
            self._monitor.join(timeout=5)
            self._monitor = None
        self.connected = False

    # -- worker / monitor（挂起自愈） ---------------------------------------

    def _start_worker_locked(self, reason: str) -> None:
        """（持锁）弃旧 worker、以新代次重建。被弃线程无法强杀，靠 retire
        标记使其醒来后静默退出；其迟到写入因 last_poll_ts 代次校验被忽略。"""
        old_retire, self._retire = self._retire, threading.Event()
        if old_retire is not None:
            old_retire.set()
        self.generation += 1
        self._last_attempt_ts = self._now()
        self._worker = threading.Thread(
            target=self._run, name=f"polling-stream-{self.market}-gen{self.generation}", daemon=True
        )
        self._worker.start()
        if reason != "start":
            logger.warning("【轮询流 worker 重建】%s 代次=%d 原因=%s", self.market, self.generation, reason)

    def _monitor_loop(self) -> None:
        """监视 worker 存活：卡死（心跳超阈值）或意外退出 → 重建代次。"""
        while not self._stop.wait(self.monitor_check_seconds):
            with self._lock:
                if self._stop.is_set():
                    return
                worker = self._worker
                if worker is None:
                    self._start_worker_locked("worker 缺失")
                    continue
                if not worker.is_alive():
                    self._start_worker_locked("worker 意外退出")
                    continue
                attempt = self._last_attempt_ts
                if attempt is not None and self._now() - attempt > self.hang_threshold_seconds:
                    self._start_worker_locked(
                        f"worker 挂起（{self._now() - attempt:.0f}s 无轮询心跳，"
                        f"阈值 {self.hang_threshold_seconds:.0f}s）"
                    )

    # -- 主循环 -------------------------------------------------------------

    def _run(self) -> None:
        # 闭包捕获本代次的 retire 事件：旧代次线程被弃后醒来只认自己的标记。
        # 代次内的失败计数也是线程局部，旧线程迟到的失败不得重置新代次。
        retire = self._retire
        my_generation = self.generation
        failures = 0
        while not self._stop.is_set() and (retire is None or not retire.is_set()):
            self._last_attempt_ts = self._now()
            try:
                # 账户快照查询（现货/合约均有 account()）：成功 = 状态可观测
                self.adapter.account()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                if failures >= self.max_consecutive_failures:
                    self.connected = False
                    self._mark_untrusted(f"连续 {failures} 次轮询失败: {exc}")
            else:
                failures = 0
                # 代次校验：被弃旧线程的迟到成功不得污染新代次的新鲜度
                if self.generation == my_generation:
                    self.last_poll_ts = self._now()
                self.connected = True
                if self.untrusted:
                    # 恢复可观测后解除不可信标记（上层 RECOVERY 解除仍按其自身规则）
                    logger.warning("【轮询流恢复】%s", self.market)
                    self.untrusted = False
            if self._stop.is_set() or (retire is not None and retire.is_set()):
                break
            self._sleep(self.poll_seconds)
