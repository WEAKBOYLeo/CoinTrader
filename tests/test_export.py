"""报告导出测试（开发文档 §8.4/§8.7：数据包、manifest 完整性、脱敏）。"""

from __future__ import annotations

import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from cointrader.execution.store import StateStore
from cointrader.live.decisions import DecisionKind, ReasonCode, StrategyDecision
from cointrader.reporting.export import export_report, scan_for_secrets
from live_helpers import make_live_config
from test_pnl import T0, T1, _seed_round_trip


def _seed_run(store: StateStore, run_id: str = "run-exp") -> None:
    store.start_run_session(
        run_id=run_id, started_ms=T0, mode="demo",
        code_revision="test-rev", strategy_version="t", config_hash="h",
        spot_endpoint="https://demo-spot", futures_endpoint="https://demo-fut",
        user_stream_mode="poll", timezone="UTC",
    )
    store.record_signal_decision(StrategyDecision(
        symbol="BTCUSDT", run_id=run_id, ts_ms=T0,
        decision_kind=DecisionKind.OPEN, allowed=True,
        reason_code=ReasonCode.ENTRY_OK, reason_text="ok",
        trailing_annualized=Decimal("0.55"),
    ))
    _seed_round_trip(store, run_id=run_id)
    store.end_run_session(run_id, ended_ms=T1, status="STOPPED", stop_reason="test")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestExport:
    def test_export_full_package_with_valid_manifest(self, tmp_path):
        store = StateStore(tmp_path / "t.sqlite3")
        _seed_run(store)
        out = tmp_path / "reports" / "run-exp"
        result = export_report(
            store, make_live_config(), run_id="run-exp",
            output_dir=out, code_revision="test-rev", strategy_version="t",
            now_fn=lambda: T1 / 1000,
        )
        assert result.ok, f"导出应通过完整性与脱敏检查: {result.redaction_hits}"
        assert result.manifest_path.is_file()
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["run_id"] == "run-exp"
        # T4：迁移版本 + authority 口径 + cursor/epoch 完整性
        assert manifest["schema_version"] == 2
        authority = manifest["funding_authority"]
        assert isinstance(authority["authoritative_rows"], int)
        assert isinstance(authority["estimated_rows"], int)
        assert isinstance(authority["authoritative_complete"], bool)
        assert isinstance(manifest["sync_cursors"], list)
        assert "count" in manifest["scan_epochs"] and "latest" in manifest["scan_epochs"]

        # 每个声明文件的 SHA-256 与磁盘一致
        listed = {f["file"]: f["sha256"] for f in manifest["files"]}
        for name, digest in listed.items():
            path = out / name
            assert path.is_file(), f"manifest 声明的文件不存在: {name}"
            assert digest and digest == _sha256(path), f"{name} 的 SHA-256 不匹配"

        # 关键文件齐全
        for required in ("signal_decisions.jsonl", "pair_executions.csv", "orders.csv",
                         "fills.csv", "funding_cashflows.csv", "pnl_ledger.csv",
                         "executive_summary.json", "config_snapshot.redacted.yaml",
                         "report_prompt.md", "service.log", "audit.log"):
            assert required in listed, f"缺少导出文件 {required}"

        # CSV 可解析且行数正确
        with (out / "fills.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 4
        with (out / "signal_decisions.jsonl").open(encoding="utf-8") as fh:
            decisions = [json.loads(line) for line in fh if line.strip()]
        assert len(decisions) == 1
        assert decisions[0]["decision_kind"] == "OPEN"

        # 摘要含 PnL 分项与限制声明
        summary = json.loads((out / "executive_summary.json").read_text(encoding="utf-8"))
        assert Decimal(summary["pnl"]["net_pnl"]) == Decimal("0.497")
        assert Decimal(summary["pnl"]["funding_pnl"]) == Decimal("0.5")
        assert Decimal(summary["pnl"]["trading_fee"]) == Decimal("-0.003")
        assert "Demo" in summary["note"]
        assert (out / "report_prompt.md").read_text(encoding="utf-8").startswith("#")

    def test_redaction_flags_secret_in_export(self, tmp_path):
        """导出内容命中密钥模式 → manifest 标记脱敏失败（ok=False）。"""
        assert scan_for_secrets("X-MBX-APIKEY: abcdef1234567890abcdef12")
        assert scan_for_secrets("api_secret=xyz")
        assert scan_for_secrets("GET /api?signature=" + "ab" * 16)
        assert scan_for_secrets("hello world") == []

        store = StateStore(tmp_path / "t.sqlite3")
        _seed_run(store)
        # 注入一条含「签名」痕迹的决策（模拟误写日志进账本）
        store.record_signal_decision(StrategyDecision(
            symbol="BTCUSDT", run_id="run-exp", ts_ms=T0 + 1,
            decision_kind=DecisionKind.SKIP, allowed=False,
            reason_code=ReasonCode.STALE_QUOTE,
            reason_text=f"debug: signature={'cd' * 20}",
        ))
        out = tmp_path / "reports2"
        result = export_report(
            store, make_live_config(), run_id="run-exp",
            output_dir=out, now_fn=lambda: T1 / 1000,
        )
        assert result.ok is False
        assert any("签名" in hit for hit in result.redaction_hits)

    def test_export_without_run_uses_latest(self, tmp_path):
        store = StateStore(tmp_path / "t.sqlite3")
        _seed_run(store, run_id="run-only")
        out = tmp_path / "reports3"
        # 显式传一个不存在的 run_id → 回退到 latest_run_session
        result = export_report(
            store, make_live_config(), run_id="run-missing",
            output_dir=out, now_fn=lambda: T1 / 1000,
        )
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["run_id"] == "run-missing"
        # 数据仍按 latest run 的时间窗导出
        assert any(f["file"] == "fills.csv" for f in manifest["files"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
