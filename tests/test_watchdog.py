"""LoopWatchdog 测试：心跳超时 → kill；持续心跳 → 不 kill；stop 后不 kill。

全部不联网、不等真实 300s：timeout/check 缩到毫秒级，时钟用真实 time.time。
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from cointrader.live.watchdog import WATCHDOG_EXIT_CODE, LoopWatchdog


def _wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TestLoopWatchdog:
    def test_missed_beats_trigger_kill_with_nonzero_code(self) -> None:
        kills: list[int] = []
        wd = LoopWatchdog(timeout_seconds=0.1, check_seconds=0.02,
                          kill_fn=kills.append)
        wd.start()
        try:
            assert _wait_until(lambda: len(kills) == 1), "心跳停转应触发 kill"
            assert kills[0] == WATCHDOG_EXIT_CODE, "必须非 0 退出码（systemd 才拉起）"
        finally:
            wd.stop()

    def test_continuous_beats_prevent_kill(self) -> None:
        kills: list[int] = []
        wd = LoopWatchdog(timeout_seconds=0.05, check_seconds=0.01,
                          kill_fn=kills.append)
        wd.start()
        try:
            # 持续打点 200ms（4 倍超时窗口）→ 不得触发
            deadline = time.time() + 0.2
            while time.time() < deadline:
                wd.beat()
                time.sleep(0.01)
            assert kills == [], f"持续心跳不得触发 kill: {kills}"
        finally:
            wd.stop()

    def test_stop_prevents_late_kill(self) -> None:
        kills: list[int] = []
        wd = LoopWatchdog(timeout_seconds=0.05, check_seconds=0.01,
                          kill_fn=kills.append)
        wd.start()
        wd.stop()  # 立即停止，不再打点
        time.sleep(0.2)  # 远超超时窗口
        assert kills == [], "stop 后的看门狗不得再 kill"

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError):
            LoopWatchdog(timeout_seconds=0)
        with pytest.raises(ValueError):
            LoopWatchdog(timeout_seconds=1.0, check_seconds=0)

    def test_kill_called_once(self) -> None:
        kills: list[int] = []
        wd = LoopWatchdog(timeout_seconds=0.05, check_seconds=0.01,
                          kill_fn=kills.append)
        wd.start()
        try:
            assert _wait_until(lambda: len(kills) >= 1)
            time.sleep(0.1)
            assert len(kills) == 1, "触发一次即返回，不得重复 kill"
        finally:
            wd.stop()
