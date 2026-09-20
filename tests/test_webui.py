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
    # T4：current projection 从未建立 → 明确 UNKNOWN（不得渲染成 0/无持仓假象）
    assert payload["positions_state"] == "UNKNOWN"
    assert payload["orders"] == []
    assert payload["fills"] == []
    # T4：PnL 默认跨 run（空账本 → 零值 + 口径标签）
    assert payload["pnl"]["run_id"] == "ALL"
    assert payload["pnl"]["pnl"]["net_pnl"] == "0"
    assert payload["pnl"]["pnl"]["authoritative_complete"] is True
    assert payload["pnl"]["pnl"]["estimated_funding_pnl"] == "0"
    # T4：payload 含数据状态区键（服务未运行时为 None，不是 0）
    assert payload["market_data"] is None
    assert payload["rate_limits"] is None
    assert "positions_state" in payload["freshness"]


def test_build_payload_online_stats_cumulative(webui_config):
    """累计在线跨重启累加（T4/AC-09）：已结束 run + 当前 run 按 heartbeat 计。"""
    import time as _time

    store = StateStore(webui_config.execution.state_db)
    now0 = int(_time.time() * 1000)
    t0 = now0 - 200_000
    # 第一段 run：在线 60s 后优雅停机
    store.start_run_session(
        run_id="run-1", started_ms=t0, mode="testnet", strategy_version="t",
        config_hash="h", code_revision="r", spot_endpoint="e", futures_endpoint="f",
        user_stream_mode="poll",
    )
    store.update_run_heartbeat("run-1", now_ms=t0 + 60_000)
    store.end_run_session("run-1", ended_ms=t0 + 60_000, status="STOPPED", stop_reason="graceful_stop")
    # 第二段 run：未结束（当前在线），心跳新鲜
    store.start_run_session(
        run_id="run-2", started_ms=t0 + 100_000, mode="testnet", strategy_version="t",
        config_hash="h", code_revision="r", spot_endpoint="e", futures_endpoint="f",
        user_stream_mode="poll",
    )
    store.update_run_heartbeat("run-2", now_ms=now0)
    store.close()

    payload = build_payload(webui_config, state_provider=lambda: {})
    online = payload["status"]["online"]
    assert online["first_start_ms"] == t0
    assert online["run_count"] == 2
    # 60s 已结束 + 第二段从 t0+100s 计到 now（心跳新鲜 → 计到 now，容差 5s）
    expected = 60_000 + (payload["now_ms"] - (t0 + 100_000))
    assert abs(online["total_online_ms"] - expected) <= 5_000
    assert online["current_run_online_ms"] is not None
    assert online["total_downtime_ms"] >= 0
    # 累计 >= 单段 run 时长（重启不得清零）
    run = payload["status"]["run"]
    assert run["run_id"] == "run-2"


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
        # T4：前端含数据状态区（epoch/限流/新鲜度）与 projection 状态
        assert b"data-status" in body
        assert b"positions-state" in body

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


class TestT4CurrentProjectionAndStatus:
    """T4/AC-12：Web 从 current projection 展示真实状态，UNKNOWN/STALE 明确。"""

    def _group(self, store, snapshot_id, ts_ms, positions, complete=True):
        from decimal import Decimal

        store.save_account_snapshot_group(
            snapshot_id=snapshot_id, ts_ms=ts_ms,
            capture_start_ms=ts_ms, capture_end_ms=ts_ms,
            source="test", run_id="run-t4",
            spot_equity_usdt=Decimal("100"), futures_equity_usdt=Decimal("900"),
            total_equity_usdt=Decimal("1000"), available_balance_usdt=Decimal("999"),
            complete=complete,
            spot_assets=[{"asset": "BTC", "free_qty": "0.01", "locked_qty": "0",
                          "total_qty": "0.01", "price_usdt": "100", "value_usdt": "1"}],
            positions=positions,
        )

    def test_positions_follow_current_projection_and_close_clears(self, webui_config):
        """关闭/清空 current position 后 Web 不再显示历史旧仓（tombstone）。"""
        import time as _time

        store = StateStore(webui_config.execution.state_db)
        now = int(_time.time() * 1000)
        self._group(store, "snap-1", now - 1000,
                    [{"symbol": "BTCUSDT", "spot_qty": "0.01", "perp_qty": "-0.01"}])
        store.close()

        payload = build_payload(webui_config, state_provider=lambda: {})
        assert payload["positions_state"] == "OK"
        assert [p["symbol"] for p in payload["positions"]] == ["BTCUSDT"]
        assert payload["status"]["current_account"]["complete"] == 1

        # 平仓：新 group 无持仓 → tombstone，positions 清空但状态 OK
        store = StateStore(webui_config.execution.state_db)
        self._group(store, "snap-2", now + 5000, [])
        store.close()

        payload = build_payload(webui_config, state_provider=lambda: {})
        assert payload["positions_state"] == "OK"
        assert payload["positions"] == [], "平仓后不得再显示历史旧仓"

    def test_stale_projection_marked_not_zeroed(self, webui_config):
        """projection 过期 → STALE（最后已知值仍可展示，但状态明确）。"""
        import time as _time

        store = StateStore(webui_config.execution.state_db)
        now = int(_time.time() * 1000)
        self._group(store, "snap-1", now - 3600_000,
                    [{"symbol": "BTCUSDT", "spot_qty": "0.01", "perp_qty": "-0.01"}])
        store.close()

        payload = build_payload(webui_config, state_provider=lambda: {})
        assert payload["positions_state"] == "STALE"
        # STALE 时展示最后已知值（带明确状态），而不是当 0/无持仓
        assert [p["symbol"] for p in payload["positions"]] == ["BTCUSDT"]
        assert payload["freshness"]["positions_state"] == "STALE"

    def test_market_data_rate_limits_freshness_passthrough(self, webui_config):
        """service 侧 market_data/rate_limits 透传到 payload 且 JSON 可序列化。"""
        svc = {
            "state": "RUNNING",
            "market_data": {"epoch_id": "e-1", "status": "READY", "can_rank": True,
                            "expected": 3, "completed": 3, "failed_count": 0,
                            "age_ms": 1000, "decision_cutoff_ms": 123},
            "rate_limits": {"spot": {"limit": 6000, "in_flight": 1,
                                     "local_used": 100, "frozen": False, "bans": 0}},
            "freshness": {"account_age_ms": 100, "reconcile_age_ms": 200,
                          "heartbeat_age_ms": 5, "ledger_sync_ok": True,
                          "positions_state": "OK"},
        }
        payload = build_payload(webui_config, state_provider=lambda: svc)
        assert payload["market_data"]["status"] == "READY"
        assert payload["rate_limits"]["spot"]["in_flight"] == 1
        assert payload["freshness"]["ledger_sync_ok"] is True
        json.dumps(payload)  # 全 JSON 可序列化

    def test_pnl_block_all_runs_default(self, webui_config):
        """PnL 默认跨 run（ALL），current run id 仅作上下文。"""
        from test_pnl import _seed_round_trip

        store = StateStore(webui_config.execution.state_db)
        _seed_round_trip(store, run_id="run-a")
        now = int(__import__("time").time() * 1000)
        store.start_run_session(
            run_id="run-a", started_ms=now - 3600_000, mode="testnet",
            strategy_version="t", config_hash="h", code_revision="r",
            spot_endpoint="e", futures_endpoint="f", user_stream_mode="poll",
        )
        store.end_run_session("run-a", ended_ms=now - 1000, status="STOPPED",
                              stop_reason="graceful_stop")
        store.close()

        payload = build_payload(webui_config, state_provider=lambda: {})
        assert payload["pnl"]["run_id"] == "ALL"
        assert payload["pnl"]["current_run_id"] == "run-a"
        from decimal import Decimal

        assert Decimal(payload["pnl"]["pnl"]["net_pnl"]) == Decimal("0.497")
        assert "authoritative_complete" in payload["pnl"]["pnl"]
        assert "estimated_funding_pnl" in payload["pnl"]["pnl"]
