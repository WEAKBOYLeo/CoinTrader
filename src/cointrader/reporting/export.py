"""AI 报告数据包导出（开发文档 §8.7）。

导出流程（只读账本 + 只读文件，绝不触发交易 API）::

    停止新增信号（调用方负责）
      -> 最终对账（调用方负责）
      -> 最终账户/持仓快照（调用方负责）
      -> 关闭 run_session（调用方负责）
      -> 导出表和日志
      -> 计算行数与 SHA-256
      -> 执行密钥/签名扫描
      -> 写 manifest

报告只基于账本和快照，不从普通日志猜收益（§8.4 / §12 错误八）。
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config
from ..execution.store import StateStore
from .pnl import PnlAggregator

__all__ = ["ManifestResult", "REPORT_PROMPT", "export_report", "scan_for_secrets"]

#: 敏感信息扫描（§8.7 脱敏检查 + §4.1 安全规则）
_SECRET_PATTERNS = [
    (re.compile(r"api[_-]?secret", re.IGNORECASE), "api_secret 关键字"),
    (re.compile(r"X-MBX-APIKEY[\s:]*\S+", re.IGNORECASE), "API Key 头"),
    (re.compile(r"signature=[0-9a-f]{16,}", re.IGNORECASE), "签名参数"),
    (re.compile(r"listenKey[\"']?\s*[:=]\s*[\"']?[0-9a-f]{20,}", re.IGNORECASE), "listenKey 值"),
    (re.compile(r"(?i)\b(binanc|spot|fut|futures|mbx)[-_]?key\b\s*[:=]\s*[\"']?[A-Za-z0-9]{24,}"), "疑似 API Key 值"),
]


def scan_for_secrets(text: str) -> list[str]:
    """在文本中扫描密钥/签名/完整认证 URL 痕迹。返回命中的描述列表。"""
    hits: list[str] = []
    for pattern, label in _SECRET_PATTERNS:
        if pattern.search(text):
            hits.append(label)
    return hits


@dataclass(frozen=True, slots=True)
class ManifestResult:
    output_dir: Path
    manifest_path: Path
    ok: bool
    redaction_hits: list[str]
    files: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "redaction_hits": list(self.redaction_hits),
            "files": list(self.files),
        }


REPORT_PROMPT = """# CoinTrader 7 天 Demo 试运行 —— AI 报告生成要求

数据来源：本目录下的账本导出文件（唯一事实源）。`service.log` / `audit.log`
只作辅助审计材料，禁止用日志文本推断收益。

## 必须遵守的输出顺序与规则

1. **先报告数据完整性和对账，再报告收益**：
   - `reconciliation_runs.csv`：是否全部一致；列出所有 mismatch 及人工处理结果。
   - `manifest.json`：完整性检查（行数、SHA-256）与脱敏检查结果。
2. **收益必须分项且分开口径**：
   - 已实现（realized）与未实现（unrealized）分开；
   - 资金费（funding）、手续费（fee）、基差（basis）分开；
   - 引用 `pnl_ledger.csv` 与 `pair_executions.csv`，注明 calculation_version；
   - `cash_delta` 仅作交叉验证：账本 PnL 与账户现金变化不一致时**报告差异，
     不得强行修正成盈利**。
3. **逐笔交易**：从 `signal_decisions.jsonl` 开始，沿
   `signal_decision → pair_execution → order → fill` 链路列出每笔开/平仓、
   两腿订单与成交、费用、资金费、对账结果。被拒绝的候选也要列拒绝原因。
4. **异常与人工动作**：列出 `risk_decisions.csv` 中所有拒绝、所有告警
   （`alerts.jsonl`）、停机/恢复事件，各带时间、原因、动作、结果。
5. **策略规则**：说明实际使用的入场（最近 N 期滑动平均年化、连续正费率、
   流动性）、退出（负均值、最长持仓）、换仓（premium）规则及其阈值来源
   （config hash / strategy_version）。
6. **实际 vs 假设**：比较实际成交滑点、手续费与成本模型假设。
7. **限制声明（必须写明）**：
   - 这是 Binance **Demo Trading**，不是主网；
   - 用户流为 **poll 轮询**，成交感知有延迟，不能替代主网 WebSocket 验证；
   - 样本只有 7 天，资金费样本量极小，**禁止把 7 天结果外推成年化策略结论**；
   - 没有交易信号时保持没有交易是策略忠实执行的结果，不是系统故障。
8. 明确区分：策略表现、执行质量、Demo 环境限制、数据缺口、未验证假设。
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_rows(store: StateStore, method: str, **kwargs: Any) -> list[dict[str, Any]]:
    return list(getattr(store, method)(**kwargs))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        path.write_text("", encoding="utf-8")
        return 0
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return len(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return len(rows)


def _copy_file(src: Path, dst: Path) -> None:
    if src.is_file():
        shutil.copyfile(src, dst)
    else:
        dst.write_text("", encoding="utf-8")


def _redacted_config(config: Config, output: Path) -> None:
    """导出配置快照（config.yaml 本身禁止含密钥，仍做脱敏扫描兜底）。"""
    text = ""
    if config.source_path is not None and config.source_path.is_file():
        text = config.source_path.read_text(encoding="utf-8")
    else:
        text = "# config snapshot unavailable\n"
    output.write_text(text, encoding="utf-8")


def _scan_epochs_summary(store: Any) -> dict[str, Any]:
    """epoch 完整性摘要（manifest 用）。"""
    epochs = store.scan_epochs(limit=1000)
    latest = epochs[0] if epochs else None
    return {
        "count": len(epochs),
        "latest": (
            {
                "epoch_id": latest.get("epoch_id"),
                "status": latest.get("status"),
                "decision_cutoff_ms": latest.get("decision_cutoff_ms"),
            }
            if latest else None
        ),
    }


def _funding_authority_summary(store: Any) -> dict[str, Any]:
    """资金费 authority 口径汇总（T4：报告 manifest 可验证性）。

    authoritative_complete：每个 ESTIMATED 结算 (symbol, ts) 都有对应的
    AUTHORITATIVE 行（否则报告需声明估算成分）。
    """
    rows = store.funding_cashflows(limit=100000)
    authoritative: set[tuple[str, int]] = set()
    estimated: set[tuple[str, int]] = set()
    for r in rows:
        key = (str(r.get("symbol")), int(r.get("funding_ts_ms") or 0))
        if str(r.get("authority")) == "AUTHORITATIVE":
            authoritative.add(key)
        else:
            estimated.add(key)
    return {
        "authoritative_rows": len(authoritative),
        "estimated_rows": len(estimated),
        "authoritative_complete": estimated <= authoritative,
    }


def export_report(
    store: StateStore,
    config: Config,
    *,
    run_id: str,
    output_dir: str | Path,
    code_revision: str = "unknown",
    strategy_version: str = "",
    config_hash: str = "",
    service_log: Path | None = None,
    audit_log: Path | None = None,
    quotes: dict[str, tuple] | None = None,
    now_fn=None,
) -> ManifestResult:
    """导出完整数据包并生成 manifest。

    Args:
        store: 只读账本（调用方保证已停止新增信号并完成最终对账/快照）。
        config: 配置（端点/日志路径/时区）。
        run_id: 要导出的 run_session。
        output_dir: 输出目录（默认 reports/live/<run_id>）。
    """
    now_fn = now_fn or time.time
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    run = store.run_session(run_id) or store.latest_run_session() or {}
    started_ms = int(run.get("started_ms") or 0)
    ended_ms = int(run.get("ended_ms") or int(now_fn() * 1000))
    since = started_ms or 0
    tz = str(run.get("timezone") or "UTC")

    files: list[dict[str, Any]] = []
    redaction_hits: list[str] = []

    def emit(name: str, writer) -> None:
        path = output / name
        rows = writer(path)
        files.append({"file": name, "rows": rows, "sha256": _sha256(path)})

    # -- 表导出 ---------------------------------------------------------------
    emit("signal_decisions.jsonl",
         lambda p: _write_jsonl(p, _table_rows(store, "signal_decisions", since_ms=since, until_ms=ended_ms)))
    emit("pair_executions.csv",
         lambda p: _write_csv(p, _table_rows(store, "pair_executions", since_ms=since, until_ms=ended_ms)))
    emit("orders.csv",
         lambda p: _write_csv(p, _table_rows(store, "orders", since_ms=since, until_ms=ended_ms)))
    emit("fills.csv",
         lambda p: _write_csv(p, _table_rows(store, "fills", since_ms=since, until_ms=ended_ms)))
    emit("funding_cashflows.csv",
         lambda p: _write_csv(p, _table_rows(store, "funding_cashflows", since_ms=since, until_ms=ended_ms)))
    emit("pnl_ledger.csv",
         lambda p: _write_csv(p, _table_rows(store, "pnl_ledger", since_ms=since, until_ms=ended_ms)))
    emit("position_snapshots.csv",
         lambda p: _write_csv(p, _table_rows(store, "position_snapshots", since_ms=since, until_ms=ended_ms)))
    emit("account_snapshots.csv",
         lambda p: _write_csv(p, _table_rows(store, "account_snapshots", since_ms=since, until_ms=ended_ms)))
    emit("reconciliation_runs.csv",
         lambda p: _write_csv(p, _table_rows(store, "reconciliation_runs", since_ms=since)))
    emit("risk_decisions.csv",
         lambda p: _write_csv(p, _table_rows(store, "risk_decisions", since_ms=since)))
    emit("exchange_events.jsonl",
         lambda p: _write_jsonl(p, _table_rows(store, "exchange_events", since_ms=since)))
    emit(
        "alerts.jsonl",
        lambda p: _write_jsonl(
            p,
            [e for e in _table_rows(store, "exchange_events", since_ms=since)
             if str(e.get("market")) == "service" and str(e.get("event_type")) == "alert"],
        ),
    )

    # -- 摘要 -----------------------------------------------------------------
    aggregator = PnlAggregator(store, now_fn=now_fn)
    summary = aggregator.for_run(str(run_id), quotes=quotes)
    executive = {
        "run_id": run_id,
        "status": run.get("status"),
        "stop_reason": run.get("stop_reason"),
        "started_ms": started_ms,
        "ended_ms": ended_ms,
        "mode": run.get("mode"),
        "user_stream_mode": run.get("user_stream_mode"),
        "spot_endpoint": run.get("spot_endpoint"),
        "futures_endpoint": run.get("futures_endpoint"),
        "code_revision": code_revision,
        "strategy_version": strategy_version or run.get("strategy_version"),
        "config_hash": config_hash or run.get("config_hash"),
        "timezone": tz,
        "pnl": summary.to_dict(),
        "counts": {
            "signal_decisions": len(_table_rows(store, "signal_decisions", since_ms=since, until_ms=ended_ms)),
            "pair_executions": len(_table_rows(store, "pair_executions", since_ms=since, until_ms=ended_ms)),
            "orders": len(_table_rows(store, "orders", since_ms=since, until_ms=ended_ms)),
            "fills": len(_table_rows(store, "fills", since_ms=since, until_ms=ended_ms)),
            "funding_cashflows": len(_table_rows(store, "funding_cashflows", since_ms=since, until_ms=ended_ms)),
            "reconciliation_runs": len(_table_rows(store, "reconciliation_runs", since_ms=since)),
            "reconciliation_mismatches": sum(
                1 for r in _table_rows(store, "reconciliation_runs", since_ms=since)
                if not int(r.get("consistent") or 0)
            ),
        },
        "note": "本报告基于 Demo Trading + poll 用户流；7 天样本不得外推为年化策略结论。",
    }
    exec_path = output / "executive_summary.json"
    exec_path.write_text(json.dumps(executive, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    files.append({"file": "executive_summary.json", "rows": 1, "sha256": _sha256(exec_path)})

    # -- 配置快照 / 日志 / prompt ----------------------------------------------
    _redacted_config(config, output / "config_snapshot.redacted.yaml")
    files.append({"file": "config_snapshot.redacted.yaml", "rows": None, "sha256": ""})
    _copy_file(service_log or (config.resolved_path(config.logging.log_dir) / "service.log"),
               output / "service.log")
    files.append({"file": "service.log", "rows": None, "sha256": ""})
    _copy_file(audit_log or config.resolved_path(config.execution.audit_path), output / "audit.log")
    files.append({"file": "audit.log", "rows": None, "sha256": ""})
    prompt_path = output / "report_prompt.md"
    prompt_path.write_text(REPORT_PROMPT, encoding="utf-8")
    files.append({"file": "report_prompt.md", "rows": None, "sha256": ""})

    # -- 补算非表格文件 hash ----------------------------------------------------
    for item in files:
        if not item["sha256"]:
            item["sha256"] = _sha256(output / item["file"])

    # -- 脱敏扫描（所有导出文件） ------------------------------------------------
    for item in files:
        path = output / item["file"]
        if path.is_file() and path.stat().st_size < 32 * 1024 * 1024:
            hits = scan_for_secrets(path.read_text(encoding="utf-8", errors="replace"))
            if hits:
                redaction_hits.extend(f"{item['file']}: {h}" for h in hits)

    integrity_ok = all(item["sha256"] for item in files) and not redaction_hits

    manifest = {
        "run_id": run_id,
        "start_time_ms": started_ms,
        "end_time_ms": ended_ms,
        "timezone": tz,
        "mode": run.get("mode"),
        "user_stream_mode": run.get("user_stream_mode"),
        "spot_endpoint": run.get("spot_endpoint"),
        "futures_endpoint": run.get("futures_endpoint"),
        "code_revision": code_revision,
        "strategy_version": strategy_version or run.get("strategy_version"),
        "config_hash": config_hash or run.get("config_hash"),
        # T4：迁移版本 + authority 口径 + cursor/epoch 完整性（报告可验证性）
        "schema_version": store.schema_version(),
        "funding_authority": _funding_authority_summary(store),
        "sync_cursors": store.sync_cursors(),
        "scan_epochs": _scan_epochs_summary(store),
        "files": files,
        "integrity_check": "PASS" if integrity_ok else "FAIL",
        "redaction_check": "PASS" if not redaction_hits else "FAIL",
        "redaction_hits": redaction_hits,
        "generated_ms": int(now_fn() * 1000),
        "note": "Demo Trading + poll 用户流；7 天样本不构成策略长期结论。",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return ManifestResult(
        output_dir=output,
        manifest_path=manifest_path,
        ok=integrity_ok,
        redaction_hits=redaction_hits,
        files=files,
    )
