"""主循环存活看门狗（长跑兜底防线）。

背景：主循环是单线程。任何一处阻塞调用挂死（代理连接黑洞、锁等待、
第三方库内部无超时调用）都会让整个进程冻结——进程活着（systemd 不会
重启它），但 tick 停转、状态永久卡死。本模块提供最后一道闸：

- 独立守护线程监视 ``beat()`` 心跳；
- 心跳超过 ``timeout_seconds`` 未更新 → 判定主循环挂死 → 调 ``kill_fn``
  （默认 ``os._exit(1)``，非 0 退出码，systemd ``Restart=always`` 拉起，
  从 SQLite 事件账本 + 交易所对账恢复，历史数据零丢失）。

设计取舍：

- 心跳由主循环每轮开头打点（不是轮末）：一轮正常耗时（对账/候选刷新/
  权重休眠）不触发误杀，阈值需覆盖最慢的正常单轮（默认 300s）。
- 用 ``os._exit`` 而非 ``os.kill``：挂死进程无法保证清理逻辑能跑完，
  直接终止最可靠；未落盘状态由账本恢复机制兜底。
- 正常停机（``stop()``）先停看门狗，不会误杀优雅退出路径。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = ["LoopWatchdog"]

#: 看门狗触发的进程退出码（非 0 → systemd Restart=always 拉起）
WATCHDOG_EXIT_CODE = 3


class LoopWatchdog:
    """主循环心跳看门狗。

    Args:
        timeout_seconds: 心跳超时（秒）。必须大于正常单轮 tick 最坏耗时。
        check_seconds: 检查间隔（秒）。
        kill_fn: 挂死处置回调，接收退出码。默认 ``os._exit``（测试可注入）。
        now_fn: 时钟注入（测试用）。
    """

    def __init__(
        self,
        *,
        timeout_seconds: float,
        check_seconds: float = 5.0,
        kill_fn: Callable[[int], None] = os._exit,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        if timeout_seconds <= 0 or check_seconds <= 0:
            raise ValueError("timeout_seconds / check_seconds 必须 > 0")
        self.timeout_seconds = timeout_seconds
        self.check_seconds = check_seconds
        self._kill = kill_fn
        self._now = now_fn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_beat: float | None = None

    # -- 心跳 ---------------------------------------------------------------

    def beat(self) -> None:
        """主循环每轮调用：刷新心跳。"""
        self._last_beat = self._now()

    # -- 生命周期 -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._last_beat = self._now()
        self._thread = threading.Thread(target=self._loop, name="loop-watchdog", daemon=True)
        self._thread.start()
        logger.info(
            "【看门狗已启动】心跳超时 %.0fs（检查间隔 %.0fs），触发即终止进程等待守护重启",
            self.timeout_seconds, self.check_seconds,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    # -- 内部 ---------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(self.check_seconds):
            last = self._last_beat
            if last is None:
                continue
            age = self._now() - last
            if age <= self.timeout_seconds:
                continue
            # 连续确认一次，防时钟跳变误杀（NTP 校时等）
            if self._stop.wait(min(self.check_seconds, 2.0)):
                return
            if self._now() - last <= self.timeout_seconds:
                continue
            logger.critical(
                "【看门狗触发】主循环心跳 %.0fs 未更新（阈值 %.0fs），判定挂死，"
                "终止进程（退出码 %d）等待守护重启",
                age, self.timeout_seconds, WATCHDOG_EXIT_CODE,
            )
            self._kill(WATCHDOG_EXIT_CODE)
            return
