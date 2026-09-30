"""Recovery 状态诊断：统一服务内存、账本、CLI 与 WebUI 的错误口径。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from typing import Any


def classify_recovery_reason(reason: str) -> str:
    """把旧的自然语言原因映射为稳定错误码。"""
    text = str(reason or "").lower()
    if "用户流" in text or "不新鲜" in text or "untrusted" in text:
        return "USER_STREAM_UNTRUSTED"
    if "对账" in text or "余额不一致" in text or "持仓不一致" in text or "mismatch" in text:
        return "RECONCILIATION_MISMATCH"
    if "capture" in text:
        return "EXCHANGE_CAPTURE_FAILED"
    if "事实同步" in text or "账本同步" in text:
        return "LEDGER_SYNC_FAILED"
    if "server time" in text or "时钟偏移" in text or "-1021" in text:
        return "SERVER_TIME_OFFSET"
    if "账本写入" in text or "signal_decision" in text:
        return "LEDGER_WRITE_FAILED"
    if "平仓" in text:
        return "CLOSE_FAILED"
    if "tick" in text or "主循环" in text:
        return "TICK_EXCEPTION"
    if "闸门" in text or "gate" in text:
        return "RISK_GATE"
    if "市场数据" in text or "epoch" in text:
        return "MARKET_DATA_NOT_READY"
    return "RECOVERY_UNKNOWN"


def decode_recovery_diagnostic(raw: Any) -> dict[str, Any]:
    """解析 runtime_state 中的 JSON；旧版本/坏值安全降级。"""
    if isinstance(raw, Mapping):
        return dict(raw)
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def recovery_error_from_runtime(
    runtime: Mapping[str, Mapping[str, Any]],
    reconciliation: Mapping[str, Any] | None,
    *,
    now_ms: int | None = None,
) -> dict[str, Any] | None:
    """从账本运行时状态 + 最近对账结果构造 UI/CLI 错误块。"""
    reason = str(runtime.get("recovery_reason", {}).get("value") or "")
    raw = runtime.get("recovery_diagnostic", {}).get("value")
    error = decode_recovery_diagnostic(raw)
    if not error and not reason:
        return None
    error.setdefault("code", classify_recovery_reason(reason))
    error.setdefault("message", reason or "服务处于 RECOVERY，但没有记录具体原因")
    if runtime.get("recovery_since_ms", {}).get("value"):
        error.setdefault("entered_at_ms", int(runtime["recovery_since_ms"]["value"]))
    if runtime.get("recovery_last_retry_ms", {}).get("value"):
        error.setdefault("last_retry_at_ms", int(runtime["recovery_last_retry_ms"]["value"]))
    if runtime.get("recovery_retry_count", {}).get("value"):
        error.setdefault("retry_count", int(runtime["recovery_retry_count"]["value"]))
    if reconciliation is not None:
        details = reconciliation.get("details")
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except (TypeError, json.JSONDecodeError):
                details = None
        position_details = details.get("positions", {}) if isinstance(details, dict) else {}
        affected: list[dict[str, Any]] = []
        if isinstance(position_details, dict):
            for symbol, legs in position_details.items():
                if not isinstance(legs, dict):
                    continue
                for market, values in legs.items():
                    if not isinstance(values, dict):
                        continue
                    within = values.get("within_tolerance")
                    if within is True:
                        continue
                    if within is None:
                        try:
                            if Decimal(str(values.get("expected"))) == Decimal(str(values.get("actual"))):
                                continue
                        except (ArithmeticError, TypeError, ValueError):
                            pass
                    expected_value = values.get("expected")
                    actual_value = values.get("actual")
                    difference = values.get("difference")
                    if difference is None:
                        try:
                            difference = format(
                                Decimal(str(actual_value)) - Decimal(str(expected_value)), "f"
                            )
                        except (ArithmeticError, TypeError, ValueError):
                            difference = None
                    affected.append({
                        "symbol": symbol,
                        "market": market,
                        "expected": expected_value,
                        "actual": actual_value,
                        "difference": difference,
                        "tolerance": values.get("tolerance"),
                    })
        if affected:
            error["affected"] = affected
        error["last_reconciliation"] = {
            "ts_ms": reconciliation.get("ts_ms"),
            "consistent": bool(reconciliation.get("consistent")),
            "mismatches": [
                item for item in str(reconciliation.get("mismatches", "")).split(",") if item
            ],
            "details": details if isinstance(details, dict) else {},
        }
    if now_ms is not None and error.get("entered_at_ms") is not None:
        error["elapsed_ms"] = max(0, int(now_ms) - int(error["entered_at_ms"]))
    return error


__all__ = [
    "classify_recovery_reason",
    "decode_recovery_diagnostic",
    "recovery_error_from_runtime",
]
