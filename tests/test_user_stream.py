"""用户数据流单元测试（开发设计文档 §3.4 / §11.1）。

用 fake WebSocket 工厂注入消息序列，禁止真实网络。
重点：事件上报、指纹去重、乱序 → 不可信、断线重连代次递增、stop 清理。
"""

from __future__ import annotations

import json
import time

import pytest

from cointrader.execution.user_stream import UserStream


class FakeWS:
    """按脚本出牌的假 WebSocket。"""

    def __init__(self, messages: list[str | Exception] | None = None) -> None:
        self.messages = list(messages or [])
        self.closed = False
        self.recv_count = 0

    def recv(self, timeout: float | None = None) -> str:  # noqa: ARG002
        self.recv_count += 1
        if not self.messages:
            raise TimeoutError()
        item = self.messages.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


class FakeAdapter:
    def __init__(self) -> None:
        self.keys_created = 0
        self.keepalives = 0
        self.closed_keys: list[str] = []

    def create_listen_key(self) -> str:
        self.keys_created += 1
        return f"lk{self.keys_created}"

    def keepalive_listen_key(self, key: str) -> None:  # noqa: ARG002
        self.keepalives += 1

    def close_listen_key(self, key: str) -> None:
        self.closed_keys.append(key)


def _msg(event: str, ts: int | None = None, **extra: object) -> str:
    payload: dict[str, object] = {"e": event, **extra}
    if ts is not None:
        payload["E"] = ts
    return json.dumps(payload)


def _wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("等待条件超时")


@pytest.fixture
def adapter() -> FakeAdapter:
    return FakeAdapter()


class TestEventDelivery:
    def test_events_delivered_with_metadata(self, adapter: FakeAdapter) -> None:
        ws = FakeWS([_msg("executionReport", 1000, i=1), _msg("executionReport", 2000, i=2)])
        received: list = []

        def factory(url: str) -> FakeWS:  # noqa: ARG001
            assert url == "wss://example/ws/ws/lk1", f"WS URL 应为 base/ws/listenKey: {url}"
            return ws

        stream = UserStream(
            market="spot",
            adapter=adapter,
            ws_base="wss://example/ws",
            on_event=received.append,
            staleness_seconds=3600,
            connect_factory=factory,
        )
        stream.start()
        try:
            _wait_until(lambda: len(received) == 2)
        finally:
            stream.stop()
        assert received[0].event_type == "executionReport"
        assert received[0].exchange_ts_ms == 1000
        assert received[0].market == "spot"
        assert received[0].generation == 1
        assert len(received[0].fingerprint) == 16
        assert received[1].exchange_ts_ms == 2000
        assert ws.closed, "stop 必须关闭 ws 连接"

    def test_listen_key_created_once_and_url_used(self, adapter: FakeAdapter) -> None:
        ws = FakeWS([_msg("executionReport", 100)])
        urls: list[str] = []
        received: list = []

        def _connect(_url: str):
            urls.append(_url)
            return ws

        stream = UserStream(
            market="perp",
            adapter=adapter,
            ws_base="wss://example",
            on_event=received.append,
            staleness_seconds=3600,
            connect_factory=_connect,
        )
        stream.start()
        try:
            _wait_until(lambda: len(received) == 1)
        finally:
            stream.stop()
        assert urls == ["wss://example/ws/lk1"]
        assert adapter.keys_created == 1


class TestDedup:
    def test_duplicate_event_reported_once(self, adapter: FakeAdapter) -> None:
        same = _msg("executionReport", 1000, i=7)
        ws = FakeWS([same, same, _msg("executionReport", 2000, i=8)])
        received: list = []
        stream = UserStream(
            market="spot",
            adapter=adapter,
            ws_base="wss://example/ws",
            on_event=received.append,
            staleness_seconds=3600,
            connect_factory=lambda url: ws,
        )
        stream.start()
        try:
            _wait_until(lambda: len(received) == 2)
        finally:
            stream.stop()
        assert len(received) == 2, "相同指纹的重复事件必须只上报一次"


class TestUntrusted:
    def test_out_of_order_event_marks_untrusted(self, adapter: FakeAdapter) -> None:
        ws = FakeWS([_msg("executionReport", 10000), _msg("executionReport", 1000)])
        received: list = []
        untrusted: list[str] = []
        stream = UserStream(
            market="spot",
            adapter=adapter,
            ws_base="wss://example/ws",
            on_event=received.append,
            on_untrusted=untrusted.append,
            staleness_seconds=3600,
            connect_factory=lambda url: ws,
        )
        stream.start()
        try:
            _wait_until(lambda: len(untrusted) == 1)
        finally:
            stream.stop()
        assert stream.untrusted is True
        assert "乱序" in untrusted[0]
        assert len(received) == 1, "乱序事件不得作为正常事件上报"

    def test_malformed_json_marks_untrusted_and_not_delivered(self, adapter: FakeAdapter) -> None:
        ws = FakeWS(["not-json{{"])
        received: list = []
        untrusted: list[str] = []
        stream = UserStream(
            market="spot",
            adapter=adapter,
            ws_base="wss://example/ws",
            on_event=received.append,
            on_untrusted=untrusted.append,
            staleness_seconds=3600,
            connect_factory=lambda url: ws,
        )
        stream.start()
        try:
            _wait_until(lambda: len(untrusted) == 1)
        finally:
            stream.stop()
        assert "解析失败" in untrusted[0]
        assert len(received) == 0

    def test_mark_untrusted_is_idempotent(self, adapter: FakeAdapter) -> None:
        untrusted: list[str] = []
        stream = UserStream(
            market="spot",
            adapter=adapter,
            ws_base="wss://example/ws",
            on_event=lambda e: None,
            on_untrusted=untrusted.append,
            connect_factory=lambda url: FakeWS(),
        )
        stream.mark_untrusted("第一次")
        stream.mark_untrusted("第二次")
        assert len(untrusted) == 1, "连续标记只通知一次"


class TestReconnect:
    def test_disconnect_reconnects_with_new_generation(self, adapter: FakeAdapter) -> None:
        ws1 = FakeWS([_msg("executionReport", 100), ConnectionError("dropped")])
        ws2 = FakeWS([_msg("executionReport", 200)])
        wss = [ws1, ws2]
        sleeps: list[float] = []

        def factory(url: str) -> FakeWS:  # noqa: ARG001
            return wss.pop(0)

        received: list = []
        untrusted: list[str] = []
        stream = UserStream(
            market="spot",
            adapter=adapter,
            ws_base="wss://example/ws",
            on_event=received.append,
            on_untrusted=untrusted.append,
            staleness_seconds=3600,
            connect_factory=factory,
            sleep_fn=sleeps.append,
            max_reconnect_backoff_seconds=1.0,
        )
        stream.start()
        try:
            _wait_until(lambda: len(received) == 2 and stream.generation >= 2)
        finally:
            stream.stop()
        assert stream.generation == 2, "重连后代次必须递增"
        assert len(untrusted) >= 1, "断线必须触发不可信回调（上层做 REST 对账）"
        assert sleeps, "断线重连前必须退避"
        assert ws1.closed and ws2.closed

    def test_stop_terminates_thread_and_closes_listen_key(self, adapter: FakeAdapter) -> None:
        # 消息永不断：stop 后线程必须退出、listenKey 被关闭
        messages: list[str | Exception] = [_msg("executionReport", 100 + i) for i in range(200)]
        stream = UserStream(
            market="spot",
            adapter=adapter,
            ws_base="wss://example/ws",
            on_event=lambda e: None,
            staleness_seconds=3600,
            connect_factory=lambda url: FakeWS(messages),
        )
        stream.start()
        time.sleep(0.05)
        stream.stop()
        assert stream._thread is None  # noqa: SLF001
        assert adapter.closed_keys == ["lk1"], "stop 必须关闭 listenKey"

    def test_is_fresh_requires_recent_event(self, adapter: FakeAdapter) -> None:
        stream = UserStream(
            market="spot",
            adapter=adapter,
            ws_base="wss://example/ws",
            on_event=lambda e: None,
            staleness_seconds=30,
            connect_factory=lambda url: FakeWS(),
        )
        assert stream.is_fresh is False, "没有事件前不算新鲜"
        stream.last_event_ts = stream._now()  # noqa: SLF001
        assert stream.is_fresh is True
        stream.mark_untrusted("x")
        assert stream.is_fresh is False, "不可信状态下不算新鲜"


class PollFakeAdapter:
    def __init__(self) -> None:
        self.calls = 0
        self.fail_forever = False
        self.failures_left = 0

    def account(self) -> dict:
        self.calls += 1
        if self.fail_forever or self.failures_left > 0:
            if self.failures_left > 0:
                self.failures_left -= 1
            raise RuntimeError("boom")
        return {"ok": True}


class TestPollingUserStream:
    """demo trading 用户流不可用时的 REST 轮询替代。"""

    def test_poll_success_marks_fresh(self) -> None:
        from cointrader.execution.user_stream import PollingUserStream

        now = [1000.0]
        adapter = PollFakeAdapter()
        stream = PollingUserStream(
            market="spot", adapter=adapter, poll_seconds=0.01, fresh_seconds=0.05,
            sleep_fn=lambda s: None, now_fn=lambda: now[0],
        )
        stream.start()
        try:
            _wait_until(lambda: adapter.calls >= 2, timeout=3.0)
            assert stream.is_fresh is True
            now[0] += 1.0
            assert stream.is_fresh is False, "超出新鲜窗口后必须不新鲜"
        finally:
            stream.stop()

    def test_consecutive_failures_mark_untrusted_once(self) -> None:
        from cointrader.execution.user_stream import PollingUserStream

        adapter = PollFakeAdapter()
        adapter.fail_forever = True
        untrusted: list[str] = []
        stream = PollingUserStream(
            market="perp", adapter=adapter, poll_seconds=0.01, fresh_seconds=0.05,
            max_consecutive_failures=3,
            on_untrusted=untrusted.append,
            sleep_fn=lambda s: None, now_fn=time.time,
        )
        stream.start()
        try:
            _wait_until(lambda: stream.untrusted, timeout=3.0)
            _wait_until(lambda: adapter.calls >= 10, timeout=3.0)
            assert len(untrusted) == 1, "不可信通知只发一次（幂等）"
        finally:
            stream.stop()

    def test_recovery_clears_untrusted(self) -> None:
        from cointrader.execution.user_stream import PollingUserStream

        adapter = PollFakeAdapter()
        adapter.fail_forever = True
        stream = PollingUserStream(
            market="spot", adapter=adapter, poll_seconds=0.01, fresh_seconds=0.05,
            max_consecutive_failures=3, sleep_fn=lambda s: None, now_fn=time.time,
        )
        stream.start()
        try:
            _wait_until(lambda: stream.untrusted, timeout=3.0)
            adapter.fail_forever = False
            _wait_until(lambda: stream.untrusted is False and stream.is_fresh, timeout=3.0)
        finally:
            stream.stop()

    def test_invalid_windows_raise(self) -> None:
        import pytest as _pytest

        from cointrader.execution.user_stream import PollingUserStream

        with _pytest.raises(ValueError):
            PollingUserStream(market="spot", adapter=object(), poll_seconds=5.0, fresh_seconds=1.0)

    def test_hung_worker_replaced_by_monitor(self) -> None:
        """worker 卡死在 account()（代理连接黑洞）→ monitor 弃旧重建，
        解除后新代次轮询成功 → 恢复新鲜，无需重启进程。"""
        import threading

        from cointrader.execution.user_stream import PollingUserStream

        release = threading.Event()

        class HangingAdapter:
            def account(self) -> dict:
                release.wait(timeout=30)  # 模拟永不返回的连接
                return {"ok": True}

        stream = PollingUserStream(
            market="perp", adapter=HangingAdapter(),
            poll_seconds=0.05, fresh_seconds=0.2,
            hang_threshold_seconds=0.3, monitor_check_seconds=0.05,
            sleep_fn=time.sleep, now_fn=time.time,
        )
        stream.start()
        try:
            # monitor 检测到心跳超时 → 代次递增（worker 被替换）
            _wait_until(lambda: stream.generation >= 2, timeout=10.0)
            assert stream.generation >= 2, "挂起 worker 必须被 monitor 替换"
            # 解除挂起 → 当前代次轮询成功 → 新鲜
            release.set()
            _wait_until(lambda: stream.is_fresh, timeout=10.0)
            assert stream.is_fresh
        finally:
            release.set()
            stream.stop()

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_dead_worker_restarted_by_monitor(self) -> None:
        """worker 线程意外死亡（BaseException 泄漏）→ monitor 重建，服务不冻结。"""
        from cointrader.execution.user_stream import PollingUserStream

        class SuicidalAdapter:
            calls = 0

            def account(self) -> dict:
                SuicidalAdapter.calls += 1
                if SuicidalAdapter.calls == 1:
                    raise SystemExit(9)  # except Exception 捕不到 → 线程死亡
                return {"ok": True}

        stream = PollingUserStream(
            market="spot", adapter=SuicidalAdapter(),
            poll_seconds=0.05, fresh_seconds=0.2,
            hang_threshold_seconds=30.0, monitor_check_seconds=0.05,
            sleep_fn=time.sleep, now_fn=time.time,
        )
        stream.start()
        try:
            # 第一次调用杀死 worker → monitor 发现线程不存活 → 新代次
            _wait_until(lambda: stream.generation >= 2 and stream.is_fresh, timeout=10.0)
            assert stream.generation >= 2
            assert stream.is_fresh, "重建后的 worker 必须恢复轮询"
        finally:
            stream.stop()
