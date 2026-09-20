"""只读 WebUI HTTP 服务（stdlib，无第三方依赖）。

设计：
- ``LiveWebUI`` 在一个守护线程里跑 ``ThreadingHTTPServer``，与主循环完全隔离：
  handler 内所有异常 → HTTP 500 JSON，端口占用/绑定失败 → start() 返回 False，
  均不影响调用方（``cmd_live_run``）。
- 数据只来自两处：
  1. ``state_provider()``：LiveService 内存快照（tick 级新鲜度，见
     ``LiveService.web_snapshot``），属性读取无锁安全（GIL 原子读）。
  2. 状态账本 SQLite：WebUI 自行打开独立连接（WAL 多读者安全），
  不共享主循环的 StateStore 实例。
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from collections.abc import Callable
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..config import Config

logger = logging.getLogger(__name__)

_INDEX_HTML_PATH = Path(__file__).parent / "index.html"
_ORDER_LIMIT = 50
_FILL_LIMIT = 50


# ---------------------------------------------------------------------------
# 数据聚合（纯函数，可单测）
# ---------------------------------------------------------------------------


def _latest_positions(store: Any) -> list[dict[str, Any]]:
    """每个 symbol 最新一条持仓快照（过滤已平仓的空快照）。"""
    snaps = store.position_snapshots(limit=10000)
    latest: dict[str, dict] = {}
    for row in snaps:
        symbol = row.get("symbol")
        if symbol and symbol not in latest:
            latest[symbol] = row
    positions: list[dict[str, Any]] = []
    now_ms = int(time.time() * 1000)
    for symbol, row in sorted(latest.items()):
        spot_qty = Decimal(str(row.get("spot_qty") or 0))
        perp_qty = Decimal(str(row.get("perp_qty") or 0))
        if spot_qty <= 0 and abs(perp_qty) <= 0:
            continue
        opened_ms = store.position_opened_ms(symbol) or row.get("ts_ms")
        positions.append({
            "symbol": symbol,
            "spot_qty": str(spot_qty),
            "perp_qty": str(perp_qty),
            "spot_price": row.get("spot_price"),
            "perp_price": row.get("perp_price"),
            "basis_pct": row.get("basis_pct"),
            "hedge_ratio": str(abs(perp_qty) / spot_qty) if spot_qty > 0 else None,
            "opened_ms": opened_ms,
            "age_ms": now_ms - int(opened_ms) if opened_ms else None,
            "snapshot_ts_ms": row.get("ts_ms"),
        })
    return positions


def _latest_pnl(store: Any) -> dict[str, Any] | None:
    """最近一个 run 的 PnL 分项（无实时报价 → unrealized 用最近快照口径）。"""
    run = store.latest_run_session()
    if run is None:
        return None
    try:
        from ..reporting.pnl import PnlAggregator

        summary = PnlAggregator(store).for_run(str(run["run_id"]))
        return {"run_id": str(run["run_id"]), "pnl": summary.to_dict()}
    except Exception as exc:  # noqa: BLE001
        logger.debug("PnL 聚合失败（不影响 WebUI 其余区块）: %s", exc)
        return {"run_id": str(run["run_id"]), "pnl": None, "error": str(exc)}


def build_payload(config: Config, state_provider: Callable[[], dict] | None = None) -> dict[str, Any]:
    """聚合 WebUI 单次轮询需要的全部数据。任何子块失败降级，不抛异常。"""
    now_ms = int(time.time() * 1000)
    payload: dict[str, Any] = {
        "now_ms": now_ms,
        "service": {},
        "status": {},
        "positions": [],
        "orders": [],
        "fills": [],
        "pnl": None,
        "store_available": False,
    }

    # 内存快照（主循环最新状态）
    try:
        if state_provider is not None:
            payload["service"] = state_provider() or {}
    except Exception as exc:  # noqa: BLE001
        payload["service"] = {"error": str(exc)}

    db_path = config.resolved_path(config.execution.state_db)
    if not db_path.exists():
        return payload
    payload["store_available"] = True

    from ..execution.store import StateStore

    store = StateStore(db_path)
    try:
        # 状态块
        try:
            runtime = store.runtime_state()
            latest_run = store.latest_run_session()
            recon_rows = store.reconciliation_runs(limit=1)
            recon = recon_rows[0] if recon_rows else None
            acct_rows = store.account_snapshots(limit=2)
            acct = acct_rows[0] if acct_rows else None
            alerts = [
                e for e in store.exchange_events(limit=20)
                if str(e.get("event_type")) == "alert"
            ]
            payload["status"] = {
                "service_state": runtime.get("service_state", {}).get("value"),
                "recovery_reason": runtime.get("recovery_reason", {}).get("value", ""),
                "run_id": runtime.get("run_id", {}).get("value") or (latest_run or {}).get("run_id"),
                "run": latest_run,
                "online": store.online_stats(now_ms),
                "mode": runtime.get("mode", {}).get("value"),
                "can_open": runtime.get("can_open", {}).get("value") == "1",
                "total_capital": runtime.get("total_capital", {}).get("value"),
                "account_snapshot_ts_ms": acct.get("ts_ms") if acct else None,
                "account_snapshot_age_ms": (
                    now_ms - int(acct["ts_ms"]) if acct and acct.get("ts_ms") else None
                ),
                "last_reconciliation": {
                    "ts_ms": recon.get("ts_ms") if recon else None,
                    "consistent": bool(recon.get("consistent")) if recon else None,
                    "mismatches": [
                        m for m in str(recon.get("mismatches", "")).split(",") if m
                    ] if recon else [],
                },
                "open_pairs": [
                    {"pair_execution_id": str(r.get("pair_execution_id")),
                     "symbol": str(r.get("symbol")), "kind": str(r.get("kind")),
                     "run_id": r.get("run_id")}
                    for r in store.open_pairs()
                ],
                "lease_holders": store.lease_holders(),
                "recent_alerts": [
                    {"ts_ms": a.get("recv_ts"), "market": a.get("market"),
                     "payload": a.get("payload")}
                    for a in alerts
                ],
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("WebUI 状态块聚合失败: %s", exc)
            payload["status"] = {"error": str(exc)}

        # 持仓块
        try:
            positions = _latest_positions(store)
            run = store.latest_run_session()
            if run and positions:
                from ..reporting.pnl import PnlAggregator

                summary = PnlAggregator(store).for_run(str(run["run_id"]))
                by_symbol = {item.symbol: item.unrealized_pnl for item in summary.per_pair}
                for pos in positions:
                    if pos["symbol"] in by_symbol:
                        pos["unrealized_pnl"] = str(by_symbol[pos["symbol"]])
            payload["positions"] = positions
        except Exception as exc:  # noqa: BLE001
            logger.debug("WebUI 持仓块聚合失败: %s", exc)
            payload["positions"] = [{"error": str(exc)}]

        # 订单/成交块（最近 N 条，倒序）
        try:
            payload["orders"] = store.orders(limit=_ORDER_LIMIT)
        except Exception as exc:  # noqa: BLE001
            payload["orders"] = [{"error": str(exc)}]
        try:
            payload["fills"] = store.fills(limit=_FILL_LIMIT)
        except Exception as exc:  # noqa: BLE001
            payload["fills"] = [{"error": str(exc)}]

        # PnL 块
        try:
            payload["pnl"] = _latest_pnl(store)
        except Exception as exc:  # noqa: BLE001
            payload["pnl"] = {"error": str(exc)}
    finally:
        store.close()

    return payload


# ---------------------------------------------------------------------------
# HTTP 服务
# ---------------------------------------------------------------------------


def _make_handler(
    config: Config,
    state_provider: Callable[[], dict] | None,
    index_html: bytes,
) -> type[BaseHTTPRequestHandler]:
    """handler 工厂：把闭包变量挂到子类上。

    注意：可调用对象不能直接挂成类属性（实例访问会触发描述符协议
    绑定成 bound method）。用非描述符的普通对象包装，避免该坑。
    """

    class _UiContext:
        __slots__ = ("config", "state_provider", "index_html")

        def __init__(
            self,
            config: Config,
            state_provider: Callable[[], dict] | None,
            index_html: bytes,
        ) -> None:
            self.config = config
            self.state_provider = state_provider
            self.index_html = index_html

    ctx = _UiContext(config, state_provider, index_html)

    class _Handler(BaseHTTPRequestHandler):
        server_version = "CoinTraderWebUI/1.0"
        protocol_version = "HTTP/1.1"

        # 由工厂注入（非描述符，不会发生绑定）
        _ui: _UiContext

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
            logger.debug("webui: " + fmt, *args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, obj: Any) -> None:
            body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
            self._send(code, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            try:
                path = urlsplit(self.path).path
                if path in ("/", "/index.html"):
                    self._send(200, self._ui.index_html, "text/html; charset=utf-8")
                elif path == "/healthz":
                    self._send_json(200, {"ok": True})
                elif path == "/api/state":
                    self._send_json(
                        200, build_payload(self._ui.config, self._ui.state_provider)
                    )
                else:
                    self._send_json(404, {"error": f"not found: {path}"})
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:  # noqa: BLE001 —— 故障隔离底线：绝不外抛
                logger.warning("webui handler 异常 %s: %s", self.path, exc)
                with contextlib.suppress(Exception):  # 响应本身也失败则静默（已告警）
                    self._send_json(500, {"error": str(exc)})

    _Handler._ui = ctx
    return _Handler


class LiveWebUI:
    """进程内只读仪表盘。start() 失败/抛异常均不影响调用方主循环。"""

    def __init__(
        self,
        *,
        config: Config,
        state_provider: Callable[[], dict] | None = None,
    ) -> None:
        self._config = config
        self._state_provider = state_provider
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: str | None = None

    @property
    def address(self) -> tuple[str, int] | None:
        if self._httpd is None:
            return None
        host, port = self._httpd.server_address[:2]
        return str(host), int(port)

    def start(self) -> bool:
        """启动守护线程并等待端口绑定（≤3s）。失败返回 False，不抛异常。"""
        if self._thread is not None:
            return True
        try:
            index_html = _INDEX_HTML_PATH.read_bytes()
        except OSError as exc:
            logger.warning("WebUI 无法加载页面模板 %s: %s", _INDEX_HTML_PATH, exc)
            return False
        handler = _make_handler(self._config, self._state_provider, index_html)
        host = self._config.execution.webui_host
        port = int(self._config.execution.webui_port)

        def _run() -> None:
            try:
                httpd = ThreadingHTTPServer((host, port), handler)
                httpd.daemon_threads = True
            except OSError as exc:
                self._error = f"端口 {host}:{port} 绑定失败: {exc}"
                self._ready.set()
                return
            self._httpd = httpd
            self._ready.set()
            try:
                httpd.serve_forever(poll_interval=0.5)
            except Exception as exc:  # noqa: BLE001
                logger.warning("WebUI serve_forever 异常退出: %s", exc)
            finally:
                with contextlib.suppress(OSError):
                    httpd.server_close()

        thread = threading.Thread(target=_run, name="cointrader-webui", daemon=True)
        self._thread = thread
        thread.start()
        if not self._ready.wait(timeout=3.0):
            self._error = "WebUI 启动超时（3s 内未完成端口绑定）"
            logger.warning("%s", self._error)
            return False
        if self._error is not None:
            logger.warning("%s", self._error)
            return False
        logger.info("WebUI 已启动: http://%s:%d/", host, port)
        return True

    def stop(self) -> None:
        """停止服务（幂等）。任何异常都被吞掉 —— WebUI 停机不能影响主流程。"""
        try:
            if self._httpd is not None:
                self._httpd.shutdown()
                self._httpd.server_close()
                self._httpd = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("WebUI 停止异常（忽略）: %s", exc)
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3.0)

    @property
    def last_error(self) -> str | None:
        return self._error
