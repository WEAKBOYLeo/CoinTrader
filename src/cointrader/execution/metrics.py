"""实盘指标（开发设计文档 §10）。

轻量实现：进程内计数器/仪表 + ``snapshot()`` 输出。
不引入 Prometheus 依赖 —— canary 阶段先看数字，出口格式稳定后再接。

关键指标（与文档 §10 对应）：

- order_submit_total{market,side,result}
- order_ack_latency_seconds（最近值）
- order_unknown_total
- order_reconcile_mismatch_total
- pair_hedge_ratio / unhedged_notional / unhedged_duration_seconds
- account_available_balance / futures_margin_ratio
- realized_pnl_24h / unrealized_pnl / funding_pnl / fee_paid
- basis_pct
- user_stream_age_seconds
- rest_rate_limit_used_ratio
- strategy_signal_count / order_rejection_count
"""

from __future__ import annotations

import threading
from typing import Any  # noqa: F401

__all__ = ["Metrics"]


class Metrics:
    """极简指标注册表（线程安全）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}

    # -- 计数 ---------------------------------------------------------------

    def inc(self, name: str, *, labels: dict[str, str] | None = None, by: float = 1.0) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + by

    def set_gauge(self, name: str, value: float, *, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._gauges[key] = float(value)

    # -- 输出 ---------------------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
            }

    def render(self) -> str:
        snap = self.snapshot()
        lines = []
        for section, table in snap.items():
            for key, value in sorted(table.items()):
                lines.append(f"{section} {key} {value:.6g}")
        return "\n".join(lines)

    def _key(self, name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        parts = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{parts}}}"
