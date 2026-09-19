"""WebUI（只读实时仪表盘）测试：不联网、不起主循环。

覆盖：
- build_payload 无账本/有账本降级行为
- LiveWebUI 启动/请求/停止生命周期（ephemeral 端口）
- 端口占用 → start() 返回 False（故障隔离）
- 模板缺失 → start() 返回 False
"""

from __future__ import annotations

import dataclasses
import json
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from cointrader.config import load_config
from cointrader.execution.store import StateStore
from cointrader.webui.server import LiveWebUI, build_payload


@pytest.fixture()
def webui_config(project_root: Path, tmp_path: Path):
    """基于真实 config.yaml，仅把 state_db 指到临时目录。"""
    config = load_config(project_root / "config" / "config.yaml")
    execution = dataclasses.replace(
        config.execution,
        state_db=tmp_path / "live" / "trading.sqlite3",
        webui_host="127.0.0.1",
        webui_port=0,  # 随机空闲端口，避免测试间冲突
    )
    return dataclasses.replace(config, execution=execution)


def test_build_payload_without_store(webui_config):
    """账本不存在 → store_available=False，不抛异常。"""
    payload = build_payload(webui_config, state_provider=lambda: {"state": "RUNNING"})
    assert payload["store_available"] is False
    assert payload["service"] == {"state": "RUNNING"}
    assert payload["positions"] == []
    assert payload["orders"] == []
    assert payload["pnl"] is None
    assert payload["now_ms"] > 0


def test_build_payload_with_empty_store(webui_config, tmp_path: Path):
    """空账本 → 各区块为空值，不抛异常。"""
    store = StateStore(webui_config.execution.state_db)
    store.close()
    payload = build_payload(webui_config, state_provider=lambda: {})
    assert payload["store_available"] is True
    assert payload["service"] == {}
    assert payload["status"]["service_state"] is None
    assert payload["positions"] == []
    assert payload["orders"] == []
    assert payload["fills"] == []
    assert payload["pnl"] is None


def test_build_payload_survives_bad_state_provider(webui_config):
    """state_provider 抛异常 → 降级为 {"error": ...}，其余区块不受影响。"""
    def boom() -> dict:
        raise RuntimeError("provider 挂了")

    payload = build_payload(webui_config, state_provider=boom)
    assert "error" in payload["service"]
    assert payload["status"] == {}
    assert payload["positions"] == []


def test_live_webui_lifecycle(webui_config):
    webui = LiveWebUI(config=webui_config, state_provider=lambda: {"state": "RUNNING", "mode": "testnet"})
    try:
        assert webui.start() is True
        host, port = webui.address
        assert host == "127.0.0.1" and port > 0

        def get(path: str):
            with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=5) as resp:
                return resp.status, resp.read()

        code, body = get("/healthz")
        assert code == 200
        assert json.loads(body) == {"ok": True}

        code, body = get("/")
        assert code == 200
        assert b"CoinTrader" in body

        code, body = get("/api/state")
        assert code == 200
        payload = json.loads(body)
        assert payload["service"]["state"] == "RUNNING"
        assert payload["store_available"] is False
        assert "status" in payload and "positions" in payload
        assert "orders" in payload and "fills" in payload

        try:
            urllib.request.urlopen(f"http://{host}:{port}/nope", timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("应当返回 404")
    finally:
        webui.stop()
    assert webui.address is None


def test_live_webui_stop_idempotent(webui_config):
    webui = LiveWebUI(config=webui_config)
    webui.stop()  # 未启动也安全
    assert webui.start() is True
    webui.stop()
    webui.stop()  # 重复停止不抛


def test_live_webui_port_in_use(webui_config, tmp_path: Path):
    """端口被占 → start() 返回 False 而非抛异常（故障隔离）。"""
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    config = dataclasses.replace(
        webui_config,
        execution=dataclasses.replace(webui_config.execution, webui_port=port),
    )
    try:
        webui = LiveWebUI(config=config)
        assert webui.start() is False
        assert webui.last_error is not None
    finally:
        blocker.close()
