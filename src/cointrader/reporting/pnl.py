"""基于账本的 live PnL 聚合器（开发文档 §8.4）。

口径（固定，公式版本 ``pnl-v1``；每个数字带来源与时间范围）::

    funding_pnl   = Σ 已确认资金费（funding_cashflows，正 = 收到）
    trading_fee   = -Σ 两腿实际手续费价值（负数 = 成本）
    basis_pnl     = qty × (basis_entry − basis_exit)
                    basis = perp 价 − spot 价（我们用「spot 多 + perp 空」，
                    开仓时收到的永续溢价 − 平仓时付出的溢价 = 基差收益）
    realized_pnl  = funding_pnl + trading_fee + basis_pnl（已平仓 pair）
    unrealized_pnl= q × (basis_entry − basis_now) + 已入账 funding + fee
                    （未平仓 pair，按最新可信报价估值）
    net_pnl       = Σ realized + Σ unrealized
    cash_delta    = 账户快照总权益变化（仅交叉验证，不替代策略 PnL）

禁止计入策略收益：平台发放资产、无法对应成交/资金费的权益变化、
未实现当已实现、报告生成时间之后的数据。

幂等：重复 fill（唯一约束 (market, exchange_trade_id)）与重复资金费
结算事件（唯一约束 (symbol, funding_ts_ms)）在数据库层就不重复，
聚合器对账本做 SUM，天然不重复计入。
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from ..execution.store import StateStore

__all__ = ["CALC_VERSION", "PairPnl", "PnlSummary", "PnlAggregator"]

CALC_VERSION = "pnl-v1"


def _d(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    return Decimal(str(value))


def _fee_value(fee_asset: str, fee_amount: Decimal, price: Decimal, base: str) -> Decimal:
    """把手续费价值折算成 USDT。"""
    if fee_asset.upper() == "USDT":
        return fee_amount
    if fee_asset.upper() == base.upper() and price > 0:
        return fee_amount * price
    # 其他计费资产（如 BNB）：无法可靠折算 → 按 USDT 面额计并标记差异
    return fee_amount


@dataclass(frozen=True, slots=True)
class PairPnl:
    """单个 round trip（开仓 pair + 平仓 pair，或仅开仓）的 PnL 分解。"""

    pair_execution_id: str
    close_pair_id: str | None
    symbol: str
    kind: str  # open / close
    status: str
    funding_pnl: Decimal
    trading_fee: Decimal
    basis_pnl: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    net_pnl: Decimal
    period_start_ms: int
    period_end_ms: int
    source: str
    reconciled: bool
    calc_version: str
    last_updated_ms: int
    estimated_funding_pnl: Decimal = Decimal("0")
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_execution_id": self.pair_execution_id,
            "close_pair_id": self.close_pair_id,
            "symbol": self.symbol,
            "kind": self.kind,
            "status": self.status,
            "funding_pnl": str(self.funding_pnl),
            "trading_fee": str(self.trading_fee),
            "basis_pnl": str(self.basis_pnl),
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "net_pnl": str(self.net_pnl),
            "period_start_ms": self.period_start_ms,
            "period_end_ms": self.period_end_ms,
            "source": self.source,
            "reconciled": self.reconciled,
            "calculation_version": self.calc_version,
            "last_updated_ms": self.last_updated_ms,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class PnlSummary:
    """一次聚合结果（pair / run 级）。"""

    funding_pnl: Decimal
    estimated_funding_pnl: Decimal
    trading_fee: Decimal
    basis_pnl: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    net_pnl: Decimal
    cash_delta: Decimal | None  # None = 无法从快照计算
    per_pair: list[PairPnl]
    period_start_ms: int
    period_end_ms: int
    calc_version: str
    last_updated_ms: int
    differences: list[str] = field(default_factory=list)
    #: False = 窗口内存在只有 estimated 口径的资金费结算（authoritative 不完整）
    authoritative_complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "funding_pnl": str(self.funding_pnl),
            "estimated_funding_pnl": str(self.estimated_funding_pnl),
            "trading_fee": str(self.trading_fee),
            "basis_pnl": str(self.basis_pnl),
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "net_pnl": str(self.net_pnl),
            "cash_delta": str(self.cash_delta) if self.cash_delta is not None else None,
            "per_pair": [p.to_dict() for p in self.per_pair],
            "period_start_ms": self.period_start_ms,
            "period_end_ms": self.period_end_ms,
            "calculation_version": self.calc_version,
            "last_updated_ms": self.last_updated_ms,
            "differences": list(self.differences),
            "authoritative_complete": self.authoritative_complete,
        }


class PnlAggregator:
    """从账本（fills / funding_cashflows / snapshots / pairs）计算 PnL。"""

    def __init__(self, store: StateStore, now_fn=None) -> None:
        self.store = store
        self._now = now_fn or time.time

    # -- 内部 ---------------------------------------------------------------

    @staticmethod
    def _weighted_avg(fills: list[dict[str, Any]]) -> tuple[Decimal, Decimal]:
        qty = sum((_d(f["quantity"]) for f in fills), Decimal("0"))
        notional = sum((_d(f["quantity"]) * _d(f["price"]) for f in fills), Decimal("0"))
        if qty <= 0:
            return Decimal("0"), Decimal("0")
        return qty, notional / qty

    def _legs(self, pair_rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        order_ids: list[str] = []
        for row in pair_rows:
            for o in self.store.orders_for_pair(str(row["pair_execution_id"])):
                order_ids.append(str(o["client_order_id"]))
        fills = self.store.fills_for_orders(order_ids)
        legs: dict[str, list[dict[str, Any]]] = {
            "spot_entry": [], "spot_exit": [], "perp_entry": [], "perp_exit": [],
        }
        for f in fills:
            market = str(f["market"]).upper()
            side = str(f["side"]).upper()
            if market == "SPOT" and side == "BUY":
                legs["spot_entry"].append(f)
            elif market == "SPOT" and side == "SELL":
                legs["spot_exit"].append(f)
            elif market == "PERP" and side == "SELL":
                legs["perp_entry"].append(f)
            elif market == "PERP" and side == "BUY":
                legs["perp_exit"].append(f)
        return legs

    def _fees(self, fills: list[dict[str, Any]]) -> tuple[Decimal, list[str]]:
        base_note: list[str] = []
        total = Decimal("0")
        base = ""
        for f in fills:
            base = str(f["symbol"]).replace("USDT", "")
            total += _fee_value(
                str(f.get("fee_asset") or ""),
                _d(f.get("fee_amount")),
                _d(f.get("price")),
                base,
            )
        if any(str(f.get("fee_asset") or "").upper() not in ("USDT", base.upper()) for f in fills):
            base_note.append("存在非 USDT/基础币计费资产，按面额折算（需人工核对）")
        return total, base_note

    def _funding(self, symbol: str, start_ms: int, end_ms: int) -> tuple[Decimal, Decimal]:
        """拆分 (authoritative, estimated-only) 资金费（T3，AC-08）。

        同一 (symbol, funding_ts) 存在 authoritative 时只计 authoritative；
        只有 estimated 的结算单独计入返回值第二项，不得混入 authoritative
        net_pnl。
        """
        rows = self.store.funding_cashflows(symbol=symbol, limit=5000)
        auth_by_ts: dict[int, Decimal] = {}
        est_by_ts: dict[int, Decimal] = {}
        for row in rows:
            ts = int(row.get("funding_ts_ms") or 0)
            if start_ms and ts < start_ms:
                continue
            if ts > end_ms:
                continue
            authority = str(row.get("authority") or "ESTIMATED").upper()
            amount = _d(row.get("amount"))
            target = auth_by_ts if authority == "AUTHORITATIVE" else est_by_ts
            target[ts] = target.get(ts, Decimal("0")) + amount
        auth_total = sum(auth_by_ts.values(), Decimal("0"))
        est_total = sum(
            (v for ts, v in est_by_ts.items() if ts not in auth_by_ts),
            Decimal("0"),
        )
        return auth_total, est_total

    def _cash_delta(self, start_ms: int, end_ms: int) -> Decimal | None:
        rows = self.store.account_snapshots(since_ms=start_ms, until_ms=end_ms, limit=100000)
        totals: dict[int, Decimal] = {}
        for row in rows:
            ts = int(row["ts_ms"])
            source = str(row.get("source") or "").upper()
            if source == "SPOT":
                totals[ts] = totals.get(ts, Decimal("0")) + _d(row.get("wallet_balance"))
            elif source == "PERP":
                totals[ts] = totals.get(ts, Decimal("0")) + _d(row.get("equity"))
        if len(totals) < 2:
            return None
        keys = sorted(totals)
        return totals[keys[-1]] - totals[keys[0]]

    def _round_trips(
        self, pairs: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
        """把 pair 行组合成 round trip：(open_row, close_row)。未平仓 → (open, None)。"""
        opens = sorted(
            (p for p in pairs if str(p.get("kind")) == "open"),
            key=lambda r: int(r.get("created_ms") or 0),
        )
        closes = sorted(
            (p for p in pairs if str(p.get("kind")) == "close"),
            key=lambda r: int(r.get("created_ms") or 0),
        )
        trips: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
        used: set[str] = set()
        for close in closes:
            match = None
            for open_row in opens:
                if open_row["pair_execution_id"] in used:
                    continue
                if str(open_row["symbol"]) == str(close["symbol"]):
                    match = open_row
                    break
            if match is not None:
                used.add(match["pair_execution_id"])
            trips.append((match, close))
        for open_row in opens:
            if open_row["pair_execution_id"] not in used:
                trips.append((open_row, None))
        return trips

    # -- 对外 ---------------------------------------------------------------

    def for_trip(
        self,
        open_row: dict[str, Any] | None,
        close_row: dict[str, Any] | None,
        *,
        quotes: dict[str, tuple[Decimal, Decimal]] | None = None,
    ) -> PairPnl:
        """计算一个 round trip 的 PnL（开仓 pair + 平仓 pair 的成交合并）。"""
        now_ms = int(self._now() * 1000)
        rows = [r for r in (open_row, close_row) if r is not None]
        primary = open_row if open_row is not None else close_row
        if primary is None:  # pragma: no cover
            raise ValueError("for_trip 需要至少一个 pair 行")
        symbol = str(primary["symbol"])
        legs = self._legs(rows)
        notes: list[str] = []

        spot_entry_qty, spot_entry_px = self._weighted_avg(legs["spot_entry"])
        perp_entry_qty, perp_entry_px = self._weighted_avg(legs["perp_entry"])
        spot_exit_qty, spot_exit_px = self._weighted_avg(legs["spot_exit"])
        perp_exit_qty, perp_exit_px = self._weighted_avg(legs["perp_exit"])

        entry_qty = min(spot_entry_qty, perp_entry_qty)
        all_fills = legs["spot_entry"] + legs["spot_exit"] + legs["perp_entry"] + legs["perp_exit"]
        fees, fee_notes = self._fees(all_fills)
        notes.extend(fee_notes)

        start_ms = int((open_row or primary).get("created_ms") or now_ms)
        if close_row is not None:
            end_ms = int(
                close_row.get("completed_ts_ms")
                or close_row.get("updated_ms")
                or now_ms
            )
        else:
            # 未平仓：资金费窗口右端是当前时刻（只用已结算事件，不用未来数据）
            end_ms = now_ms
        closed = bool(
            close_row is not None
            and legs["spot_exit"]
            and legs["perp_exit"]
            and spot_exit_qty > 0
            and perp_exit_qty > 0
            and spot_entry_qty > 0
            and perp_entry_qty > 0
        )

        funding_auth, funding_est = self._funding(symbol, start_ms, end_ms)
        if funding_est != 0:
            notes.append(f"存在 {funding_est} USDT 仅 estimated 口径资金费（未计入 authoritative net）")
        basis = Decimal("0")
        if closed:
            qty = min(entry_qty, min(spot_exit_qty, perp_exit_qty))
            basis = qty * ((perp_entry_px - spot_entry_px) - (perp_exit_px - spot_exit_px))
        elif entry_qty > 0:
            quote = (quotes or {}).get(symbol)
            if quote is not None:
                spot_now, perp_now = quote
                basis = entry_qty * ((perp_entry_px - spot_entry_px) - (perp_now - spot_now))
            else:
                notes.append("无最新报价，未实现基差部分未估值")

        if closed:
            realized = funding_auth + (-fees) + basis
            unrealized = Decimal("0")
        else:
            realized = Decimal("0")
            unrealized = funding_auth + (-fees) + basis

        net = realized + unrealized
        return PairPnl(
            pair_execution_id=str(primary["pair_execution_id"]),
            close_pair_id=str(close_row["pair_execution_id"]) if close_row is not None else None,
            symbol=symbol,
            kind="round_trip" if close_row is not None else "open",
            status=str((close_row or primary).get("status") or ""),
            funding_pnl=funding_auth,
            estimated_funding_pnl=funding_est,
            trading_fee=-fees,
            basis_pnl=basis,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            net_pnl=net,
            period_start_ms=start_ms,
            period_end_ms=end_ms,
            source="ledger",
            reconciled=False,
            calc_version=CALC_VERSION,
            last_updated_ms=now_ms,
            notes=notes,
        )

    def for_run(self, run_id: str, *, quotes: dict[str, tuple[Decimal, Decimal]] | None = None) -> PnlSummary:
        now_ms = int(self._now() * 1000)
        rows = self.store.pair_executions(limit=100000)
        pairs = [r for r in rows if str(r.get("run_id") or "") == run_id]
        return self._aggregate(pairs, quotes=quotes, now_ms=now_ms)

    def for_all_runs(self, *, quotes: dict[str, tuple[Decimal, Decimal]] | None = None) -> PnlSummary:
        """跨所有 run 聚合（断点重连后的全局视图）：不限 run_id，
        未平仓 pair 自然跨 run 保留。聚合路径与 ``for_run`` 完全共享。"""
        now_ms = int(self._now() * 1000)
        pairs = self.store.pair_executions(limit=100000)
        return self._aggregate(pairs, quotes=quotes, now_ms=now_ms)

    def _aggregate(
        self,
        pairs: list[dict[str, Any]],
        *,
        quotes: dict[str, tuple[Decimal, Decimal]] | None,
        now_ms: int,
    ) -> PnlSummary:
        per_pair: list[PairPnl] = []
        start_ms, end_ms = now_ms, 0
        for open_row, close_row in self._round_trips(pairs):
            pnl = self.for_trip(open_row, close_row, quotes=quotes)
            per_pair.append(pnl)
            start_ms = min(start_ms, pnl.period_start_ms)
            end_ms = max(end_ms, pnl.period_end_ms)
        if not pairs:
            start_ms = end_ms = now_ms

        def _sum(attr: str) -> Decimal:
            return sum((getattr(p, attr) for p in per_pair), Decimal("0"))

        cash = self._cash_delta(start_ms, end_ms)
        est_total = _sum("estimated_funding_pnl")
        authoritative_complete = est_total == 0
        differences: list[str] = [] if cash is not None else ["账户快照不足，cash_delta 无法交叉验证"]
        if not authoritative_complete:
            differences.append(
                f"authoritative 资金费不完整：{est_total} USDT 仅 estimated 口径，未计入 authoritative net"
            )
        return PnlSummary(
            funding_pnl=_sum("funding_pnl"),
            estimated_funding_pnl=est_total,
            trading_fee=_sum("trading_fee"),
            basis_pnl=_sum("basis_pnl"),
            realized_pnl=_sum("realized_pnl"),
            unrealized_pnl=_sum("unrealized_pnl"),
            net_pnl=_sum("net_pnl"),
            cash_delta=cash,
            per_pair=per_pair,
            period_start_ms=start_ms,
            period_end_ms=end_ms,
            calc_version=CALC_VERSION,
            last_updated_ms=now_ms,
            differences=differences,
            authoritative_complete=authoritative_complete,
        )

    def persist_round_trip(self, open_row: dict[str, Any] | None,
                            close_row: dict[str, Any] | None) -> PairPnl:
        """把一个 round trip 的 PnL 分项写入 pnl_ledger（幂等）。"""
        if close_row is None:
            raise ValueError("只有已平仓 round trip 才写 REALIZED 账本")
        primary = open_row if open_row is not None else close_row
        pnl = self.for_trip(open_row, close_row)
        run_id = str(primary.get("run_id") or "")
        pid = str(primary["pair_execution_id"])

        def ledger_id(kind: str) -> str:
            digest = hashlib.sha256(
                f"{pid}|{str(close_row['pair_execution_id'])}|{kind}|{CALC_VERSION}".encode()
            ).hexdigest()[:20]
            return f"led-{digest}"

        for kind, amount in (
            ("FUNDING", pnl.funding_pnl),
            ("FEE", pnl.trading_fee),
            ("BASIS", pnl.basis_pnl),
            ("REALIZED", pnl.realized_pnl),
        ):
            self.store.record_ledger_entry(
                ledger_id=ledger_id(kind),
                kind=kind,
                amount=amount,
                ts_ms=pnl.last_updated_ms,
                source="pnl_aggregator",
                run_id=run_id,
                pair_execution_id=pid,
                period_start_ms=pnl.period_start_ms,
                period_end_ms=pnl.period_end_ms,
                calculation_version=CALC_VERSION,
            )
        self.store.update_pair_pnl(
            str(close_row["pair_execution_id"]),
            funding_pnl=pnl.funding_pnl,
            fee_pnl=pnl.trading_fee,
            basis_pnl=pnl.basis_pnl,
            realized_pnl=pnl.realized_pnl,
            unrealized_pnl=pnl.unrealized_pnl,
            net_pnl=pnl.net_pnl,
        )
        if open_row is not None:
            self.store.update_pair_pnl(str(open_row["pair_execution_id"]), **{
                "funding_pnl": pnl.funding_pnl,
                "fee_pnl": pnl.trading_fee,
                "basis_pnl": pnl.basis_pnl,
                "realized_pnl": pnl.realized_pnl,
                "unrealized_pnl": pnl.unrealized_pnl,
                "net_pnl": pnl.net_pnl,
            })
        return pnl


__all__ = ["CALC_VERSION", "PairPnl", "PnlSummary", "PnlAggregator"]
