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


def _current_positions(
    store: Any, now_ms: int, *, stale_after_ms: int
) -> tuple[list[dict[str, Any]], str]:
    """current positions projection（T4/AC-12）+ 状态标签。

    只读 current projection（对账/恢复写入），不以历史 position_snapshots
    冒充当前持仓：平仓后 tombstone 行 qty=0 被过滤，不再显示旧仓。
    状态：OK / STALE（projection 过期）/ UNKNOWN（从未建立投影）。
    价格/基准率仅作展示从最近快照补充（不影响数量口径）。
    """
    rows = store.current_positions(include_tombstones=False)
    live = [r for r in rows if _is_live_position(r)]
    if not rows:
        state = "UNKNOWN" if store.current_account() is None else "OK"
        return [], state
    latest_observed = max(int(r.get("observed_at_ms") or 0) for r in rows)
    state = ("STALE"
             if not latest_observed or now_ms - latest_observed > stale_after_ms
             else "OK")
    # 展示层补充：最近快照的价格/基准（趋势参考，不是数量来源）
    price_map: dict[str, dict[str, Any]] = {}
    for snap in store.position_snapshots(limit=10000):
        symbol = snap.get("symbol")
        if symbol and symbol not in price_map:
            price_map[symbol] = snap
    positions: list[dict[str, Any]] = []
    for row in sorted(live, key=lambda r: str(r.get("symbol"))):
        symbol = str(row["symbol"])
        spot_qty = Decimal(str(row.get("spot_qty") or 0))
        perp_qty = Decimal(str(row.get("perp_qty") or 0))
        snap = price_map.get(symbol, {})
        opened_ms = store.position_opened_ms(symbol) or row.get("observed_at_ms")
        positions.append({
            "symbol": symbol,
            "spot_qty": str(spot_qty),
            "perp_qty": str(perp_qty),
            "spot_price": row.get("spot_price") or snap.get("spot_price"),
            "perp_price": row.get("perp_price") or snap.get("perp_price"),
            "basis_pct": snap.get("basis_pct"),
            "hedge_ratio": str(abs(perp_qty) / spot_qty) if spot_qty > 0 else None,
            "opened_ms": opened_ms,
            "age_ms": now_ms - int(opened_ms) if opened_ms else None,
            "observed_at_ms": row.get("observed_at_ms"),
        })
    return positions, state


def _is_live_position(row: dict[str, Any]) -> bool:
    spot_qty = Decimal(str(row.get("spot_qty") or 0))
    perp_qty = Decimal(str(row.get("perp_qty") or 0))
    return spot_qty > 0 or abs(perp_qty) > 0


def _pnl_block(store: Any) -> dict[str, Any] | None:
    """PnL 默认跨 run（for_all_runs）；current run id 仅作上下文。

    口径标签（T4/AC-12）：``authoritative_complete`` 与
    ``estimated_funding_pnl`` 单独字段展示，估算不得混入 authoritative。
    """
    run = store.latest_run_session()
    try:
        from ..reporting.pnl import PnlAggregator

        summary = PnlAggregator(store).for_all_runs()
        return {
            "run_id": "ALL",
            "current_run_id": str(run["run_id"]) if run else None,
            "pnl": summary.to_dict(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("PnL 聚合失败（不影响 WebUI 其余区块）: %s", exc)
        return {
            "run_id": "ALL",
            "current_run_id": str(run["run_id"]) if run else None,
            "pnl": None,
            "error": str(exc),
        }


def build_payload(config: Config, state_provider: Callable[[], dict] | None = None) -> dict[str, Any]:
    """聚合 WebUI 单次轮询需要的全部数据。任何子块失败降级，不抛异常。"""
    now_ms = int(time.time() * 1000)
    service: dict[str, Any] = {}
    if state_provider is not None:
        try:
            service = state_provider() or {}
        except Exception as exc:  # noqa: BLE001
            service = {"error": str(exc)}
    payload: dict[str, Any] = {
        "now_ms": now_ms,
        "service": service,
        "status": {},
        "positions": [],
        "positions_state": "UNKNOWN",
        "orders": [],
        "fills": [],
        "pnl": None,
        # T4：来自服务内存快照的数据状态区（服务未运行时为 None，不得渲染成 0）
        "market_data": service.get("market_data"),
        "rate_limits": service.get("rate_limits"),
        "freshness": {},
        "store_available": False,
    }

    db_path = config.resolved_path(config.execution.state_db)
    if not db_path.exists():
        # 无账本时仍暴露服务侧新鲜度（不得把 UNKNOWN 渲染成 0）
        svc_freshness = service.get("freshness")
        if isinstance(svc_freshness, dict):
            payload["freshness"] = svc_freshness
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
                "current_account": store.current_account(),
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
                "reconcile_age_ms": (
                    now_ms - int(recon["ts_ms"]) if recon and recon.get("ts_ms") else None
                ),
                "heartbeat_age_ms": (
                    now_ms - int(latest_run["last_heartbeat_ms"])
                    if latest_run and latest_run.get("last_heartbeat_ms") else None
                ),
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

        # 持仓块（T4：current projection，不是历史快照；UNKNOWN/STALE 明确标记）
        try:
            stale_after_ms = max(
                int(config.execution.snapshot_interval_seconds) * 3000, 60_000
            )
            positions, positions_state = _current_positions(
                store, now_ms, stale_after_ms=stale_after_ms
            )
            if positions:
                from ..reporting.pnl import PnlAggregator

                summary = PnlAggregator(store).for_all_runs()
                by_symbol = {item.symbol: item.unrealized_pnl for item in summary.per_pair}
                for pos in positions:
                    if pos["symbol"] in by_symbol:
                        pos["unrealized_pnl"] = str(by_symbol[pos["symbol"]])
            payload["positions"] = positions
            payload["positions_state"] = positions_state
        except Exception as exc:  # noqa: BLE001
            logger.debug("WebUI 持仓块聚合失败: %s", exc)
            payload["positions"] = [{"error": str(exc)}]
            payload["positions_state"] = "UNKNOWN"

        # 订单/成交块（最近 N 条，倒序）
        try:
            payload["orders"] = store.orders(limit=_ORDER_LIMIT)
        except Exception as exc:  # noqa: BLE001
            payload["orders"] = [{"error": str(exc)}]
        try:
            payload["fills"] = store.fills(limit=_FILL_LIMIT)
        except Exception as exc:  # noqa: BLE001
            payload["fills"] = [{"error": str(exc)}]

        # PnL 块（T4：默认跨 run，口径标签在 pnl 字典内）
        try:
            payload["pnl"] = _pnl_block(store)
        except Exception as exc:  # noqa: BLE001
            payload["pnl"] = {"error": str(exc)}

        # 新鲜度汇总（T4：Web/CLI 统一口径；服务未运行时仅有账本侧数据）
        try:
            status = payload["status"]
            svc_freshness = service.get("freshness") or {}
            payload["freshness"] = {
                "account_ts_ms": svc_freshness.get("account_ts_ms",
                                                  status.get("account_snapshot_ts_ms")),
                "account_age_ms": svc_freshness.get("account_age_ms",
                                                   status.get("account_snapshot_age_ms")),
                "account_complete": svc_freshness.get(
                    "account_complete",
                    (status.get("current_account") or {}).get("complete") == 1
                    if status.get("current_account") else None,
                ),
                "reconcile_age_ms": svc_freshness.get("reconcile_age_ms",
                                                      status.get("reconcile_age_ms")),
                "reconcile_ok": svc_freshness.get("reconcile_ok",
                                                  (status.get("last_reconciliation") or {})
                                                  .get("consistent")),
                "heartbeat_age_ms": status.get("heartbeat_age_ms"),
                "positions_state": payload["positions_state"],
                "ledger_sync_ok": svc_freshness.get("ledger_sync_ok"),
                "ledger_sync_error": svc_freshness.get("ledger_sync_error"),
            }
        except Exception:  # noqa: BLE001
            payload["freshness"] = {"positions_state": payload["positions_state"]}
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
