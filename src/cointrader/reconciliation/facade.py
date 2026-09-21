"""ReconciliationFacade（T4）。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..execution.models import ExchangeSnapshotBundle, ReconciliationResult
from ..execution.reconcile import Reconciler
from ..execution.sync import ExchangeStateSynchronizer, SyncCaptureError

__all__ = ["GateOutcome", "ReconciliationFacade"]


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """一次 capture + 账本同步 + 对账的 gate 输入。"""

    bundle: ExchangeSnapshotBundle | None
    result: ReconciliationResult
    ledger_sync_ok: bool
    ledger_sync_error: str
    #: 应用 gate 结论：True = 对账一致 + 账本同步通过 + 允许开仓
    ok: bool


class ReconciliationFacade:
    """对账统一入口（组合 exch_sync + reconciler；无自身状态）。

    Args:
        exch_sync: 交易所事实同步器（None = legacy 路径，无事实同步）。
        reconciler: 对账器。
        ledger_symbols_fn: 返回需要 facts 同步的 symbol 列表。
        gate_allows: gate 侧的额外放行条件（如风控闸门 NORMAL）。
    """

    def __init__(
        self,
        *,
        exch_sync: ExchangeStateSynchronizer | None,
        reconciler: Reconciler,
        ledger_symbols_fn: Callable[[], list[str]] | None = None,
        gate_allows: Callable[[], bool] | None = None,
    ) -> None:
        self._exch_sync = exch_sync
        self._reconciler = reconciler
        self._ledger_symbols_fn = ledger_symbols_fn or (lambda: [])
        self._gate_allows = gate_allows or (lambda: True)

    def capture(self) -> ExchangeSnapshotBundle | None:
        """短期 capture bundle（失败 → None，不抛）。"""
        if self._exch_sync is None:
            return None
        try:
            return self._exch_sync.capture()
        except SyncCaptureError:
            return None

    def sync_ledger(self, symbols: list[str] | None = None) -> tuple[bool, str]:
        """fills/funding 增量 facts 同步。返回 (是否通过, 失败原因)。"""
        if self._exch_sync is None:
            return True, ""
        symbols = self._ledger_symbols_fn() if symbols is None else symbols
        fill_results = self._exch_sync.sync_fills(symbols)
        income_results = self._exch_sync.sync_funding_income(symbols)
        bad = [
            f"{r.market}/{r.stream}/{r.symbol}: {r.error or '不完整'}"
            for r in (*fill_results, *income_results)
            if r.error or not r.complete
        ]
        return (not bad, "; ".join(bad[:3]))

    def run(self, *, reason: str = "periodic", snapshot: ExchangeSnapshotBundle | None = None) -> ReconciliationResult:
        return self._reconciler.run(reason=reason, snapshot=snapshot)

    def evaluate(self, *, reason: str = "periodic") -> GateOutcome:
        """capture → facts 同步 → 对账 → gate 结论（一次性组合）。"""
        bundle: ExchangeSnapshotBundle | None = None
        capture_failed = False
        if self._exch_sync is not None:
            try:
                bundle = self._exch_sync.capture()
            except SyncCaptureError:
                capture_failed = True
        ledger_ok, ledger_err = self.sync_ledger()
        if capture_failed:
            ledger_ok = False
            ledger_err = "capture 失败"
        result = self.run(reason=reason, snapshot=bundle)
        ok = result.can_open and ledger_ok and self._gate_allows()
        return GateOutcome(
            bundle=bundle,
            result=result,
            ledger_sync_ok=ledger_ok,
            ledger_sync_error=ledger_err,
            ok=ok,
        )
