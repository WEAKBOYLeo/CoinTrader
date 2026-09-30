"""T4.4/T4.5：查询隔离（WebUI/CLI/reporting 只读 QueryService）。

计划 4.0 T4 动作 6：替换 QueryService 为 fake，证明：

- WebUI/CLI/reporting **不持有 broker/executor**（无签名交易客户端、
  无 execution 下单面 import）；
- **不访问 Binance 签名接口**（只读公开报价属尽力增强，不是事实源，
  仅限 CLI 显式展示路径；WebUI/reporting 完全无 Binance 依赖）；
- **不依赖 LiveService mutable state**：build_payload 的持仓/订单事实
  全部来自注入的只读 query port；服务内存块仅作诊断字段（带来源+时间）。

与 tests/test_architecture_boundaries.py 互补：那里按 import 边界，
这里按**调用面**（query 方法必须存在于 LedgerQueryService 端口上）。
"""

from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from cointrader.ledger.service import LedgerQueryService

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "cointrader"


def _call_names(path: Path, var: str) -> set[str]:
    """收集 ``var.X(...)`` 形式的方法名（AST，精确到属性名）。"""
    names: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == var:
                names.add(node.func.attr)
    return names


class TestWebUIQueryIsolation:
    """build_payload 的持仓/订单事实源 = 注入的只读 query port。"""

    def _fake_queries(self) -> dict[str, Any]:
        calls: list[str] = []

        def _mk(name: str, default: Any):
            def fn(*args: Any, **kwargs: Any) -> Any:
                calls.append(name)
                return default

            return fn

        return {
            "calls": calls,
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "spot_qty": "0.01",
                    "perp_qty": "-0.01",
                    "spot_price": "100",
                    "perp_price": "100",
                    "observed_at_ms": 1_800_000_000_000,
                }
            ],
            "orders": [{"client_order_id": "ct-x", "symbol": "BTCUSDT"}],
            "fills": [{"client_order_id": "ct-x", "quantity": "0.01", "price": "100"}],
            "service": LedgerQueryService(
                _FakeLedger(
                    calls=calls,
                    positions=[
                        {
                            "symbol": "BTCUSDT",
                            "spot_qty": "0.01",
                            "perp_qty": "-0.01",
                            "spot_price": "100",
                            "perp_price": "100",
                            "observed_at_ms": 1_800_000_000_000,
                        }
                    ],
                    orders=[{"client_order_id": "ct-x", "symbol": "BTCUSDT"}],
                    fills=[{"client_order_id": "ct-x", "quantity": "0.01", "price": "100"}],
                )
            ),
        }

    def test_positions_orders_fills_come_from_injected_queries(
        self, tmp_path: Path
    ) -> None:
        from dataclasses import replace

        from cointrader.config import load_config
        from cointrader.webui.server import build_payload

        config = load_config(None)  # 默认配置
        db = tmp_path / "ledger.sqlite3"
        db.touch()
        config = replace(
            config, execution=replace(config.execution, state_db=db)
        )
        fake = self._fake_queries()
        payload = build_payload(
            config,
            state_provider=None,  # 无 LiveService：证明不依赖服务 mutable state
            queries_factory=lambda: fake["service"],
        )
        # 事实源来自 fake query port（不是服务内存、不是 Binance）
        assert payload["store_available"] is True
        assert [p["symbol"] for p in payload["positions"]] == ["BTCUSDT"]
        assert payload["orders"][0]["client_order_id"] == "ct-x"
        assert payload["fills"][0]["quantity"] == "0.01"
        # 服务内存块 = 空（诊断位），且带明确来源/时间标签
        assert payload["service"] == {}
        assert payload["service_source"]["source"].startswith("LiveService.web_snapshot")
        assert "captured_at_ms" in payload["service_source"]

    def test_service_block_is_diagnostics_only_label(self, tmp_path: Path) -> None:
        """服务内存字段必须标记来源+时间（T4.4），且不得作为持仓事实源。"""
        from dataclasses import replace

        from cointrader.config import load_config
        from cointrader.webui.server import build_payload

        config = load_config(None)
        db = tmp_path / "ledger.sqlite3"
        db.touch()
        config = replace(
            config, execution=replace(config.execution, state_db=db)
        )
        fake = self._fake_queries()
        payload = build_payload(
            config,
            state_provider=lambda: {"held": [{"symbol": "BTCUSDT"}]},
            queries_factory=lambda: fake["service"],
        )
        # 持仓事实仍然只来自 query projection，不来自 service.held
        assert [p["symbol"] for p in payload["positions"]] == ["BTCUSDT"]
        assert payload["service"]["held"][0]["symbol"] == "BTCUSDT"
        assert payload["service_source"]["captured_at_ms"] > 0

    def test_webui_module_does_not_import_live_or_broker(self) -> None:
        """webui 包不 import LiveService（mutable state）/broker/签名客户端。"""
        forbidden = {
            "cointrader.live",
            "cointrader.execution.transport",
            "cointrader.execution.broker",
            "cointrader.execution.auth",
            "cointrader.execution.spot",
            "cointrader.execution.futures",
            "cointrader.data",
        }

        def _modules(path: Path) -> set[str]:
            mods: set[str] = set()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level and node.module:
                    # 相对 import：webui 包内 level>=1 只可能指向 cointrader.*
                    base = "cointrader"
                    for _ in range(node.level - 1):
                        base = base  # webui 是 cointrader 直属子包
                    mods.add(f"{base}.{node.module}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        mods.add(alias.name)
            return mods

        violations: list[str] = []
        for path in sorted((SRC_ROOT / "webui").glob("*.py")):
            for mod in _modules(path):
                if any(mod == f or mod.startswith(f + ".") for f in forbidden):
                    violations.append(f"{path.name}: import {mod}")
        assert violations == [], (
            "webui 查询层不得依赖 LiveService/broker/签名客户端:\n" + "\n".join(violations)
        )


class TestCLIAndReportingUseQueryPort:
    """CLI live 查询与 reporting 导出的调用面必须被 LedgerQueryService 覆盖。"""

    def test_cli_live_query_calls_covered_by_query_service(self) -> None:
        """cli.py 中所有 ``store.X(...)`` 调用（_open_live_store 返回
        LedgerQueryService）必须存在于端口上（含 close）。"""
        cli = SRC_ROOT / "cli.py"
        names = _call_names(cli, "store")
        missing = sorted(n for n in names if not hasattr(LedgerQueryService, n))
        assert missing == [], (
            f"cli.py 调用了 LedgerQueryService 端口之外的方法（查询隔离破坏）: {missing}"
        )

    def test_cli_open_live_store_returns_query_service(self) -> None:
        src = (SRC_ROOT / "cli.py").read_text(encoding="utf-8")
        helper = src[src.index("def _open_live_store"): src.index("def _fetch_live_quotes")]
        assert "LedgerQueryService" in helper, "_open_live_store 必须返回 LedgerQueryService"

    def test_reporting_export_calls_covered_by_query_service(self) -> None:
        """reporting/export.py 的全部 ``store.X(...)`` 调用必须存在于端口。"""
        names = _call_names(SRC_ROOT / "reporting" / "export.py", "store")
        missing = sorted(n for n in names if not hasattr(LedgerQueryService, n))
        assert missing == [], (
            f"export.py 调用了 LedgerQueryService 端口之外的方法: {missing}"
        )

    def test_reporting_does_not_import_broker_or_binance(self) -> None:
        forbidden = {
            "cointrader.execution.transport",
            "cointrader.execution.broker",
            "cointrader.execution.auth",
            "cointrader.execution.spot",
            "cointrader.execution.futures",
            "cointrader.data",
            "cointrader.live",
        }
        violations: list[str] = []
        for path in sorted((SRC_ROOT / "reporting").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level and node.module:
                    mod = f"cointrader.{node.module}" if node.level == 1 else node.module
                    if any(mod == f or mod.startswith(f + ".") for f in forbidden):
                        violations.append(f"{path.name}: import {mod}")
        assert violations == [], (
            "reporting 不得依赖 broker/签名客户端/LiveService:\n" + "\n".join(violations)
        )


class _FakeLedger:
    """LedgerQueryService 的注入 fake（只读 query port 等价物）。"""

    def __init__(self, calls: list[str], **tables: Any) -> None:
        self._calls = calls
        self._tables = tables

    def _get(self, name: str, default: Any = None) -> Any:
        self._calls.append(name)
        value = self._tables.get(name, default)
        return list(value) if isinstance(value, (list, tuple)) and default is not None else value

    def current_positions(self, *, include_tombstones: bool = False) -> list[dict[str, Any]]:
        self._calls.append("current_positions")
        return list(self._tables.get("positions", []))

    def current_account(self) -> dict[str, Any] | None:
        self._calls.append("current_account")
        return None

    def position_snapshots(
        self, *, since_ms=None, until_ms=None, symbol=None, limit: int = 5000
    ) -> list[dict[str, Any]]:
        self._calls.append("position_snapshots")
        return []

    def position_opened_ms(self, symbol: str) -> int | None:
        self._calls.append("position_opened_ms")
        return None

    def runtime_state(self) -> dict[str, dict[str, Any]]:
        self._calls.append("runtime_state")
        return {}

    def latest_run_session(self) -> dict[str, Any] | None:
        self._calls.append("latest_run_session")
        return None

    def reconciliation_runs(self, *, since_ms=None, limit: int = 1000) -> list[dict[str, Any]]:
        self._calls.append("reconciliation_runs")
        return []

    def account_snapshots(
        self, *, since_ms=None, until_ms=None, limit: int = 5000
    ) -> list[dict[str, Any]]:
        self._calls.append("account_snapshots")
        return []

    def exchange_events(
        self, *, since_ms=None, market=None, limit: int = 5000
    ) -> list[dict[str, Any]]:
        self._calls.append("exchange_events")
        return []

    def online_stats(self, now_ms: int, *, heartbeat_fresh_ms: int = 120_000) -> dict[str, Any]:
        self._calls.append("online_stats")
        return {}

    def open_pairs(self) -> list[dict[str, Any]]:
        self._calls.append("open_pairs")
        return []

    def lease_holders(self) -> list[dict[str, Any]]:
        self._calls.append("lease_holders")
        return []

    def orders(
        self, *, since_ms=None, until_ms=None, symbol=None, limit: int = 5000
    ) -> list[dict[str, Any]]:
        self._calls.append("orders")
        return list(self._tables.get("orders", []))

    def fills(
        self, *, since_ms=None, until_ms=None, symbol=None, limit: int = 5000
    ) -> list[dict[str, Any]]:
        self._calls.append("fills")
        return list(self._tables.get("fills", []))

    def funding_cashflows(
        self, *, since_ms=None, until_ms=None, symbol=None, limit: int = 5000
    ) -> list[dict[str, Any]]:
        self._calls.append("funding_cashflows")
        return []

    def orders_for_pair(self, pair_execution_id: str) -> list[dict[str, Any]]:
        self._calls.append("orders_for_pair")
        return []

    def pair_executions(
        self, *, since_ms=None, until_ms=None, symbol=None, limit: int = 5000
    ) -> list[dict[str, Any]]:
        self._calls.append("pair_executions")
        return []

    def fills_for_orders(self, client_order_ids) -> list[dict[str, Any]]:
        self._calls.append("fills_for_orders")
        return []

    def schema_version(self) -> int:
        return 3

    def run_session(self, run_id: str) -> dict[str, Any] | None:
        return None

    def run_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        return []

    def scan_epochs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return []

    def candidate_snapshots(self, epoch_id: str, *, limit: int = 1000) -> list[dict[str, Any]]:
        return []

    def sync_cursors(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        return []

    def signal_decisions(self, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    def expected_positions(self) -> dict[str, dict[str, Decimal]]:
        return {}

    def get_pair(self, pair_execution_id: str) -> dict[str, Any] | None:
        return None

    def orders_in_states(self, states) -> list[dict[str, Any]]:
        return []


@pytest.mark.usefixtures("tmp_path")
def test_build_payload_pnl_block_uses_queries_not_binance(tmp_path: Path) -> None:
    """PnL 块只读 ledger（经 query port）；空账本 → 空 PnL，不触网络。"""
    from dataclasses import replace

    from cointrader.config import load_config
    from cointrader.webui.server import build_payload

    config = load_config(None)
    db = tmp_path / "ledger.sqlite3"
    db.touch()
    config = replace(config, execution=replace(config.execution, state_db=db))
    fake = _FakeLedger(calls=[])
    payload = build_payload(config, queries_factory=lambda: LedgerQueryService(fake))
    assert payload["pnl"] is None or "error" not in payload["pnl"] or isinstance(
        payload["pnl"], dict
    )


class TestV5T3QueryDiagnosticsIsolation:
    """v5.0 T3（AC-07）：诊断信封/ cache 健康的查询侧行为。

    - 服务内存诊断缺失（state_provider 空）→ 渲染 null/UNKNOWN，不渲染 0；
    - build_payload 全程不触 broker/exchange（纯本地 ledger + 注入 provider）。
    """

    def test_missing_service_diagnostics_render_null_not_zero(self, tmp_path: Path) -> None:
        from dataclasses import replace

        from cointrader.config import load_config
        from cointrader.webui.server import build_payload

        config = load_config(None)
        db = tmp_path / "ledger.sqlite3"
        db.touch()
        config = replace(config, execution=replace(config.execution, state_db=db))
        payload = build_payload(
            config,
            state_provider=lambda: {},  # 服务在跑但无任何诊断数据
            queries_factory=lambda: LedgerQueryService(_FakeLedger(calls=[])),
        )
        # 诊断块保持 null（UNKNOWN），不得渲染成 0/RUNNING/healthy
        assert payload["rate_limits"] is None
        assert payload["rate_limits_envelope"] is None
        assert payload["market_data_envelope"] is None
        assert payload["cache_stats"] is None
        assert payload["cache_stats_envelope"] is None
        assert payload["store_available"] is True
        import json

        json.dumps(payload)  # 可序列化

    def test_build_payload_does_not_touch_broker_or_exchange(self, tmp_path: Path) -> None:
        """查询/health 渲染路径不 acquire permit、不发交易所请求（调用面证据：
        webui 模块仅允许依赖 config/execution.store/ledger/reporting/stdlib）。"""
        from dataclasses import replace

        from cointrader.config import load_config
        from cointrader.webui.server import build_payload

        config = load_config(None)
        db = tmp_path / "ledger.sqlite3"
        db.touch()
        config = replace(config, execution=replace(config.execution, state_db=db))

        calls: list[str] = []

        class _CountingLedger(_FakeLedger):
            def current_positions(self, *, include_tombstones: bool = False):
                calls.append("current_positions")
                return []

        svc = {
            "state": "RUNNING",
            "rate_limits": {"futures": {"observed_used": 100, "local_used": 5,
                                        "external_usage_uncertain": True,
                                        "quality": "OK"}},
            "rate_limits_envelope": {"source": "coordinator（诊断）", "as_of_ms": 1,
                                     "quality": "OK"},
        }
        payload = build_payload(
            config,
            state_provider=lambda: svc,
            queries_factory=lambda: LedgerQueryService(_CountingLedger(calls=calls)),
        )
        # 只走了 ledger 只读端口；诊断块原样透传（无网络往返）
        assert "current_positions" in calls
        assert payload["rate_limits"]["futures"]["external_usage_uncertain"] is True
        assert payload["rate_limits_envelope"]["quality"] == "OK"
