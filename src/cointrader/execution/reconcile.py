"""启动、重连、周期性对账（开发设计文档 §8）。

原则：

1. **交易所状态是当前风险事实**，本地账本只是事件历史。
   对账方向永远是：拉交易所快照 → 逐项比对本地 → 修复本地/报告差异。
2. 对账发现**交易所存在本地未知订单或持仓** → 不一致，禁止开新仓；
   只允许补腿/平仓等减风险动作（pair_executor 的 reduce 路径）。
3. 对账不一致、用户流不新鲜时 ``can_open=False``。
4. 本地订单状态与交易所终态不一致 → 按交易所修复本地（记入 repaired，
   不算 mismatch —— 这是正常的事件丢失补偿）。

对账触发点：启动、进程重启、用户流断线/过期、REST 下单超时恢复后、
固定周期（reconciliation_interval_seconds）。
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from .futures import FuturesAdapter
from .models import ExchangeSnapshotBundle, ReconciliationResult
from .order_state import OrderState, validate_transition
from .spot import SpotAdapter
from .store import StateStore

logger = logging.getLogger(__name__)

__all__ = ["Reconciler"]

#: 本地非终态订单中，需要与交易所开放订单比对的集合
NON_TERMINAL = ("NEW", "PARTIALLY_FILLED", "UNKNOWN")

#: 持仓数量容差（绝对值）。低于该值视为零（灰尘仓位）。
QTY_EPSILON = Decimal("0.000001")


class Reconciler:
    """REST 快照与本地账本对账。"""

    def __init__(
        self,
        store: StateStore,
        spot: SpotAdapter,
        futures: FuturesAdapter,
        *,
        ignore_assets: Sequence[str] = (),
    ) -> None:
        self.store = store
        self.spot = spot
        self.futures = futures
        # 平台发放/非策略资产（如 demo 的 USDC）：无本地期望持仓时不参与对账
        self._ignore_assets = frozenset(a.upper() for a in ignore_assets)

    def run(
        self,
        *,
        reason: str = "periodic",
        snapshot: ExchangeSnapshotBundle | None = None,
    ) -> ReconciliationResult:
        """执行一次完整对账。返回 ReconciliationResult 并写入账本。

        ``snapshot`` 给定（live 路径）：消费同一 capture bundle，不重复拉
        账户/仓位/开放订单 API；bundle 不完整 → ``consistent=False, can_open=False``。
        未给定（非 live 工具薄包装）：自行经 adapter 拉取（兼容旧调用）。
        """
        mismatches: list[str] = []
        repaired: list[str] = []
        details: dict[str, Any] = {}

        if snapshot is not None:
            details["snapshot_id"] = snapshot.snapshot_id
            if not snapshot.complete:
                mismatches.append("capture bundle 不完整：账户/仓位/开放订单字段缺失，状态不可信")
                result = ReconciliationResult(
                    ts_ms=int(time.time() * 1000),
                    consistent=False,
                    mismatches=tuple(mismatches),
                    repaired=(),
                    can_open=False,
                    details=details,
                )
                self.store.record_reconciliation(result, reason=reason)
                logger.error("【对账拒绝】bundle 不完整，can_open=false（reason=%s）", reason)
                return result

        # 1. 订单对账（两市场）
        for market, _adapter, query in (
            ("SPOT", self.spot, self._reconcile_spot_orders),
            ("PERP", self.futures, self._reconcile_futures_orders),
        ):
            try:
                m, r = query(mismatches, repaired, snapshot=snapshot)
            except Exception as exc:  # noqa: BLE001
                # 对账请求本身失败 = 状态不可信
                mismatches.append(f"{market}: 对账查询失败: {exc}")
                continue
            mismatches.extend(m)
            repaired.extend(r)

        # 2. 持仓对账
        try:
            m, r, pos_details = self._reconcile_positions(snapshot=snapshot)
            mismatches.extend(m)
            repaired.extend(r)
            details["positions"] = pos_details
        except Exception as exc:  # noqa: BLE001
            mismatches.append(f"持仓对账查询失败: {exc}")

        consistent = not mismatches
        can_open = consistent
        result = ReconciliationResult(
            ts_ms=int(time.time() * 1000),
            consistent=consistent,
            mismatches=tuple(mismatches),
            repaired=tuple(repaired),
            can_open=can_open,
            details=details,
        )
        self.store.record_reconciliation(result, reason=reason)

        if consistent:
            logger.info("【对账通过】reason=%s repaired=%d", reason, len(repaired))
        else:
            logger.error("【对账不一致】reason=%s mismatches=%s", reason, mismatches)
        return result

    # -- 订单对账 -----------------------------------------------------------

    def _reconcile_spot_orders(
        self,
        mismatches: list[str],
        repaired: list[str],
        *,
        snapshot: ExchangeSnapshotBundle | None = None,
    ) -> tuple[list[str], list[str]]:
        return self._reconcile_orders(self.spot, "SPOT", mismatches, repaired, snapshot=snapshot)

    def _reconcile_futures_orders(
        self,
        mismatches: list[str],
        repaired: list[str],
        *,
        snapshot: ExchangeSnapshotBundle | None = None,
    ) -> tuple[list[str], list[str]]:
        return self._reconcile_orders(self.futures, "PERP", mismatches, repaired, snapshot=snapshot)

    def _reconcile_orders(
        self,
        adapter: Any,
        market: str,
        mismatches: list[str],
        repaired: list[str],
        *,
        snapshot: ExchangeSnapshotBundle | None = None,
    ) -> tuple[list[str], list[str]]:
        if snapshot is not None:
            exchange_open = (
                snapshot.spot_open_orders if market == "SPOT" else snapshot.perp_open_orders
            )
        else:
            exchange_open = adapter.open_orders()
        open_by_client: dict[str, dict[str, Any]] = {}
        for o in exchange_open:
            cid = str(o.get("clientOrderId") or "")
            if cid:
                open_by_client[cid] = o

        local_non_terminal = self.store.orders_in_states(NON_TERMINAL)
        local_by_client = {o["client_order_id"]: o for o in local_non_terminal if o["client_order_id"].startswith("ct-")}

        # 1) 交易所开放订单 vs 本地：本地未知的开放订单 = 严重不一致
        for cid, o in open_by_client.items():
            if cid in local_by_client:
                continue
            if self.store.get_order(cid) is None:
                mismatches.append(
                    f"{market}: 交易所存在本地未知开放订单 {cid} "
                    f"(symbol={o.get('symbol')}, qty={o.get('origQty')}, type={o.get('type')})"
                )

        # 2) 本地非终态订单 vs 交易所：交易所已终态 → 修复本地
        for cid, local in local_by_client.items():
            if cid in open_by_client:
                continue
            order = self._query_order(adapter, local["symbol"], cid)
            if order is None:
                # 交易所无此订单：本地标记 NEW/UNKNOWN → 说明未成交
                if local["state"] in ("NEW", "UNKNOWN"):
                    repaired.append(f"{market}: {cid} 交易所确认未接单，本地标记 CANCELED")
                    self._mark_terminal(cid, "CANCELED")
                else:
                    mismatches.append(f"{market}: {cid} 本地 {local['state']} 但交易所查不到")
            elif order.is_terminal:
                try:
                    validate_transition(local["state"], OrderState(order.state))
                except Exception:  # noqa: BLE001
                    # 本地已是终态而交易所给出不同终态：不一致
                    mismatches.append(f"{market}: {cid} 终态冲突 本地={local['state']} 交易所={order.state}")
                    continue
                repaired.append(f"{market}: {cid} 交易所终态 {order.state}，本地已更新")
                self.store.upsert_order(order)
        return mismatches, repaired

    def _query_order(self, adapter: Any, symbol: str, client_order_id: str) -> Any:
        return adapter.query_by_client_order_id(symbol, client_order_id)

    def _mark_terminal(self, client_order_id: str, state: str) -> None:
        row = self.store.get_order(client_order_id)
        if row is None:
            return
        from .guard import Market, OrderSide, OrderType
        from .models import Order

        try:
            order = Order(
                client_order_id=client_order_id,
                exchange_order_id=row.get("exchange_order_id"),
                symbol=row["symbol"],
                market=Market(row["market"]),
                side=OrderSide(row["side"]),
                order_type=OrderType(row["order_type"]),
                quantity=Decimal(row["quantity"]),
                state=state,
                executed_qty=Decimal(row["executed_qty"] or 0),
                avg_price=Decimal(row["avg_price"]) if row.get("avg_price") else None,
                updated_ms=int(time.time() * 1000),
            )
        except ValueError:
            return
        self.store.upsert_order(order)

    # -- 持仓对账 -----------------------------------------------------------

    def _reconcile_positions(
        self, *, snapshot: ExchangeSnapshotBundle | None = None
    ) -> tuple[list[str], list[str], dict[str, Any]]:
        mismatches: list[str] = []
        repaired: list[str] = []
        expected = self.store.expected_positions()
        if snapshot is not None:
            spot_balances = {
                a["asset"]: Decimal(str(a.get("free") or "0")) + Decimal(str(a.get("locked") or "0"))
                for a in (snapshot.spot_account.get("balances") or [])
                if isinstance(a, dict) and a.get("asset")
            }
            perp_positions = {
                str(p.get("symbol")): Decimal(str(p.get("positionAmt") or "0"))
                for p in snapshot.positions
                if str(p.get("positionSide") or "BOTH") == "BOTH"
            }
        else:
            spot_balances = self.spot.balances()
            perp_positions = {p["symbol"]: Decimal(str(p.get("positionAmt") or "0"))
                              for p in self.futures.positions()
                              if str(p.get("positionSide") or "BOTH") == "BOTH"}

        details: dict[str, Any] = {}
        symbols = set(expected) | {s for s, q in spot_balances.items() if q > QTY_EPSILON}

        for symbol in sorted(symbols):
            base = symbol.replace("USDT", "")
            # 不可交易灰尘容忍：低于一个 step 的余额无法再下最小单，
            # 平仓手续费扣减后必然存在（demo 实测 0.0007 扣费后 0.0006993）。
            # 对账容差 = 基础 epsilon 与该 symbol 最小可交易 step 取大，
            # 否则 step 以下的灰尘余额会被永久判为不一致 → 卡在 RECOVERY。
            tolerance_spot = QTY_EPSILON
            tolerance_perp = QTY_EPSILON
            if hasattr(self.spot, "rule"):
                with contextlib.suppress(KeyError):
                    tolerance_spot = max(tolerance_spot, self.spot.rule(f"{symbol}USDT").step_size)
            if hasattr(self.futures, "rule"):
                with contextlib.suppress(KeyError):
                    tolerance_perp = max(tolerance_perp, self.futures.rule(f"{symbol}USDT").step_size)
            exp_entry = expected.get(symbol)
            if exp_entry is None:
                # 余额派生的独立 symbol（无同名单元）：
                # 已被某 pair 管理（如 USDCUSDT 覆盖 USDC）→ 该 pair 的比较已覆盖此资产；
                # 平台发放/非策略资产（如 demo 的 USDC）→ 跳过；其余 = 未知资产，照常报差异。
                covered_by_pair = any(k != symbol and k.startswith(symbol) for k in expected)
                if covered_by_pair or symbol in self._ignore_assets:
                    continue
            exp_spot = exp_entry.get("SPOT", Decimal("0")) if exp_entry else Decimal("0")
            act_spot = spot_balances.get(base, Decimal("0"))
            diff_spot = act_spot - exp_spot
            if abs(diff_spot) > tolerance_spot:
                mismatches.append(
                    f"{symbol}: Spot 余额不一致 本地期望={exp_spot} 交易所={act_spot} 差={diff_spot}"
                )

            exp_perp = expected.get(symbol, {}).get("PERP", Decimal("0"))
            act_perp = perp_positions.get(symbol, Decimal("0"))
            diff_perp = act_perp - exp_perp
            if abs(diff_perp) > tolerance_perp:
                mismatches.append(
                    f"{symbol}: 永续持仓不一致 本地期望={exp_perp} 交易所={act_perp} 差={diff_perp}"
                )

            details[symbol] = {
                "spot": {"expected": str(exp_spot), "actual": str(act_spot)},
                "perp": {"expected": str(exp_perp), "actual": str(act_perp)},
            }
        return mismatches, repaired, details
