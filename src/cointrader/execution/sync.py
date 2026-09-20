"""交易所事实同步（实施计划书 v2.0 T3，AC-06/07/08）。

职责边界：

1. **capture()** 一次抓取账户/资产/全部持仓/开放订单为不可变
   ``ExchangeSnapshotBundle``；同一 ``exchange_snapshot_reuse_seconds`` 窗口
   内 single-flight 复用，poll/account/reconcile 共享同一 bundle，不重复拉 API。
2. **sync_fills() / sync_funding_income()** 按 (market, stream, symbol) 游标
   分页回补：每页 facts 与游标**同一事务**提交（store 层保证），失败 cursor
   留在上一页可重跑；重复事件由数据库唯一键（(market, exchange_trade_id)、
   (market, exchange_income_id)）挡住。
3. income 正负号直接采用交易所事实，禁止按多空方向重算；字段缺失抛
   ``SyncParseError``，不用估算冒充 authoritative。
4. 本模块只做抓取/解析/幂等入账，不含策略、风控或仓位推导。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol, cast

from ..config import Config
from ..errors import BinanceError, CoinTraderError
from .guard import Market, OrderSide
from .models import ExchangeSnapshotBundle, Fill, FundingAuthority, new_id
from .store import StateStore

logger = logging.getLogger(__name__)

#: 币安「Illegal symbol」：该市场无此交易对/该账户无此交易对（demo 受限交易对）。
_ILLEGAL_SYMBOL_CODE = -1121


def _is_illegal_symbol(exc: Exception) -> bool:
    return isinstance(exc, BinanceError) and exc.code == _ILLEGAL_SYMBOL_CODE


__all__ = [
    "ExchangeStateSynchronizer",
    "SyncCaptureError",
    "SyncParseError",
    "SyncResult",
]


class SyncError(CoinTraderError):
    """事实同步失败（分类基类）。"""


class SyncCaptureError(SyncError):
    """capture 的必要查询失败（bundle 不完整）。"""


class SyncParseError(SyncError):
    """交易所响应字段与契约不符（不得猜测字段）。"""


@dataclass(frozen=True, slots=True)
class SyncResult:
    """单次 (market, stream, symbol) 同步结果。"""

    market: str
    stream: str
    symbol: str
    inserted: int
    pages: int
    cursor_time_ms: int
    complete: bool
    error: str = ""


class _SpotLike(Protocol):
    def account(self) -> dict[str, Any]: ...
    def open_orders(self, symbol: str | None = ...) -> list[dict[str, Any]]: ...
    def my_trades(
        self, symbol: str, *, from_id: int | None = ...,
        start_ms: int | None = ..., limit: int = ...,
    ) -> list[dict[str, Any]]: ...


class _FuturesLike(Protocol):
    def account(self) -> dict[str, Any]: ...
    def positions(self, symbol: str | None = ...) -> list[dict[str, Any]]: ...
    def open_orders(self, symbol: str | None = ...) -> list[dict[str, Any]]: ...
    def user_trades(
        self, symbol: str, *, from_id: int | None = ...,
        start_ms: int | None = ..., limit: int = ...,
    ) -> list[dict[str, Any]]: ...
    def income_history(
        self, *, income_type: str = ..., start_ms: int | None = ...,
        end_ms: int | None = ..., limit: int = ...,
    ) -> list[dict[str, Any]]: ...


class ExchangeStateSynchronizer:
    """交易所事实抓取 + 幂等回补（单进程单写者）。"""

    PAGE_LIMIT = 1000
    #: 单 symbol 单流单次同步的分页上限（防止游标卡死时的无限回补）
    MAX_PAGES = 50

    def __init__(
        self,
        *,
        store: StateStore,
        spot: _SpotLike,
        futures: _FuturesLike,
        config: Config,
        now_fn: Any = time.time,
    ) -> None:
        self._store = store
        self._spot = spot
        self._futures = futures
        self._config = config
        self._now = now_fn
        self._lock = threading.Lock()
        self._last_complete_bundle: ExchangeSnapshotBundle | None = None

    # -- capture -------------------------------------------------------------

    def capture(self) -> ExchangeSnapshotBundle:
        """抓取当前交易所快照束。复用窗口内返回同一 complete bundle。

        任一必需查询失败 → 返回 ``complete=False`` 的 bundle 并抛
        ``SyncCaptureError``（bundle 同时供 Reconciler 判定 can_open=false）。
        """
        now_ms = int(self._now() * 1000)
        reuse_ms = int(self._config.execution.exchange_snapshot_reuse_seconds * 1000)
        with self._lock:
            bundle = self._last_complete_bundle
            if bundle is not None and now_ms - bundle.capture_start_ms < reuse_ms:
                return bundle
        start_ms = now_ms
        errors: list[str] = []
        spot_account: dict[str, Any] = {}
        futures_account: dict[str, Any] = {}
        positions: list[dict[str, Any]] = []
        spot_orders: list[dict[str, Any]] = []
        perp_orders: list[dict[str, Any]] = []
        for label, fn in (
            ("spot.account", lambda: self._spot.account()),
            ("futures.account", lambda: self._futures.account()),
            ("futures.positions", lambda: self._futures.positions()),
            ("spot.open_orders", lambda: self._spot.open_orders()),
            ("futures.open_orders", lambda: self._futures.open_orders()),
        ):
            try:
                result = cast(list[dict[str, Any]] | dict[str, Any], fn())
            except Exception as exc:  # noqa: BLE001 —— 分类为 capture 失败，不用旧值冒充
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
                continue
            if label == "spot.account":
                spot_account = cast(dict[str, Any], result)
            elif label == "futures.account":
                futures_account = cast(dict[str, Any], result)
            elif label == "futures.positions":
                positions = cast(list[dict[str, Any]], result)
            elif label == "spot.open_orders":
                spot_orders = cast(list[dict[str, Any]], result)
            else:
                perp_orders = cast(list[dict[str, Any]], result)
        end_ms = int(self._now() * 1000)
        complete = not errors
        bundle = ExchangeSnapshotBundle(
            snapshot_id=new_id("snap"),
            capture_start_ms=start_ms,
            capture_end_ms=end_ms,
            spot_account=spot_account,
            futures_account=futures_account,
            positions=positions,
            spot_open_orders=spot_orders,
            perp_open_orders=perp_orders,
            source="rest",
            complete=complete,
        )
        if complete:
            with self._lock:
                self._last_complete_bundle = bundle
        if not complete:
            logger.error("【capture 不完整】%s", "; ".join(errors))
            raise SyncCaptureError("; ".join(errors))
        return bundle

    # -- fills 分页回补 --------------------------------------------------------

    def sync_fills(self, symbols: list[str] | tuple[str, ...]) -> list[SyncResult]:
        results: list[SyncResult] = []
        for market, adapter in (("SPOT", self._spot), ("PERP", self._futures)):
            for symbol in symbols:
                try:
                    results.append(self._sync_fills_one(market, adapter, symbol))
                except Exception as exc:  # noqa: BLE001 —— 单 symbol 失败不影响其余
                    if _is_illegal_symbol(exc):
                        # 该市场无此交易对（demo 受限交易对）：确定性错误，
                        # 视为「无数据且完整」而非失败，重试不会改变结果。
                        results.append(SyncResult(market, "fills", symbol, 0, 0, 0, True, ""))
                    else:
                        results.append(SyncResult(market, "fills", symbol, 0, 0, 0, False, str(exc)))
        return results

    def _sync_fills_one(self, market: str, adapter: Any, symbol: str) -> SyncResult:
        scope, stream = market, "fills"
        cursor = self._store.get_sync_cursor(scope, stream, symbol)
        last_time = cursor.last_time_ms if cursor is not None else self._backfill_start_ms()
        last_id = cursor.last_id if cursor is not None else ""
        inserted_total = 0
        pages = 0
        complete = False
        while pages < self.MAX_PAGES:
            kwargs: dict[str, Any] = {"limit": self.PAGE_LIMIT}
            if last_time > 0:
                kwargs["start_ms"] = last_time
            rows = (
                self._spot.my_trades(symbol, **kwargs)
                if market == "SPOT"
                else self._futures.user_trades(symbol, **kwargs)
            )
            if not rows:
                complete = True
                break
            pages += 1
            fills = [f for f in (self._parse_fill(market, symbol, row) for row in rows) if f is not None]
            if not fills:
                complete = True
                break
            new_last_time = max(int(f.ts_ms) for f in fills)
            new_last_id = str(rows[-1].get("id") or rows[-1].get("tradeId") or last_id)
            result = self._store.write_facts_and_advance_cursor(
                scope=scope, stream=stream, symbol_key=symbol, fills=fills,
                new_last_time_ms=max(new_last_time, last_time),
                new_last_id=new_last_id,
            )
            inserted_total += result["inserted_fills"]
            if len(rows) < self.PAGE_LIMIT:
                complete = True
                break
            if new_last_time <= last_time:
                # 游标无法前进（分页边界重复）：facts 已幂等写入，停止防死循环
                complete = False
                break
            last_time = new_last_time
            last_id = new_last_id
        cursor = self._store.get_sync_cursor(scope, stream, symbol)
        return SyncResult(
            market=market, stream=stream, symbol=symbol,
            inserted=inserted_total, pages=pages,
            cursor_time_ms=cursor.last_time_ms if cursor is not None else last_time,
            complete=complete,
        )

    def _parse_fill(self, market: str, symbol: str, row: dict[str, Any]) -> Fill | None:
        """交易所成交行 → Fill。必填字段缺失抛 SyncParseError（不猜测）。"""
        trade_id = row.get("id") or row.get("tradeId")
        if trade_id in (None, ""):
            raise SyncParseError(f"{market} 成交行缺少 id/tradeId: {row!r}")
        price_raw = row.get("price")
        qty_raw = row.get("qty")
        time_raw = row.get("time")
        if price_raw in (None, "") or qty_raw in (None, "") or time_raw in (None, ""):
            raise SyncParseError(f"{market} 成交行字段不完整（price/qty/time）: {row!r}")
        is_buyer = row.get("isBuyer", row.get("buyer"))
        if is_buyer is None:
            raise SyncParseError(f"{market} 成交行缺少 isBuyer/buyer: {row!r}")
        side = OrderSide.BUY if bool(is_buyer) else OrderSide.SELL
        fee_asset = str(row.get("commissionAsset") or "USDT")
        fee_amount = Decimal(str(row.get("commission") or "0"))
        if "isMaker" in row:
            maker_taker: str | None = "MAKER" if row["isMaker"] else "TAKER"
        elif "maker" in row:
            maker_taker = "MAKER" if row["maker"] else "TAKER"
        else:
            maker_taker = None
        quote_raw = row.get("quoteQty")
        client_order_id = str(row.get("clientOrderId") or "") or f"external-{trade_id}"
        ts_ms = int(str(time_raw))
        return Fill(
            # v2 新写入规则：fill_id 固定 "<MARKET>:<exchange_trade_id>"
            fill_id=f"{market}:{trade_id}",
            client_order_id=client_order_id,
            exchange_order_id=(
                str(row["orderId"]) if row.get("orderId") not in (None, "") else None
            ),
            symbol=str(row.get("symbol") or symbol),
            market=Market(market),
            side=side,
            quantity=Decimal(str(qty_raw)),
            price=Decimal(str(price_raw)),
            fee_asset=fee_asset,
            fee_amount=fee_amount,
            ts_ms=ts_ms,
            exchange_trade_id=str(trade_id),
            quote_qty=Decimal(str(quote_raw)) if quote_raw not in (None, "") else None,
            maker_taker=maker_taker,
            exchange_ts_ms=ts_ms,
            received_ts_ms=int(self._now() * 1000),
        )

    # -- funding income 分页回补 ----------------------------------------------

    def sync_funding_income(self, symbols: list[str] | tuple[str, ...]) -> list[SyncResult]:
        results: list[SyncResult] = []
        for symbol in symbols:
            try:
                results.append(self._sync_income_one(symbol))
            except Exception as exc:  # noqa: BLE001
                if _is_illegal_symbol(exc):
                    results.append(SyncResult("PERP", "funding_income", symbol, 0, 0, 0, True, ""))
                else:
                    results.append(
                        SyncResult("PERP", "funding_income", symbol, 0, 0, 0, False, str(exc))
                    )
        return results

    def _sync_income_one(self, symbol: str) -> SyncResult:
        scope, stream = "PERP", "funding_income"
        cursor = self._store.get_sync_cursor(scope, stream, symbol)
        start_ms = cursor.last_time_ms if cursor is not None else self._backfill_start_ms()
        last_time = start_ms
        inserted_total = 0
        pages = 0
        complete = False
        while pages < self.MAX_PAGES:
            rows = self._futures.income_history(
                income_type="FUNDING_FEE", start_ms=start_ms or None, limit=self.PAGE_LIMIT
            )
            rows = [r for r in rows if str(r.get("symbol")) == symbol]
            if not rows:
                complete = True
                break
            pages += 1
            income_rows: list[dict[str, Any]] = []
            for row in rows:
                parsed = self._parse_income(symbol, row)
                if parsed is not None:
                    income_rows.append(parsed)
            if not income_rows:
                complete = True
                break
            new_last_time = max(int(r["funding_ts_ms"]) for r in income_rows)
            result = self._store.write_facts_and_advance_cursor(
                scope=scope, stream=stream, symbol_key=symbol,
                income_rows=income_rows,
                new_last_time_ms=max(new_last_time, last_time),
                new_last_id=str(max(str(r["exchange_income_id"]) for r in income_rows)),
            )
            inserted_total += result["inserted_income"]
            if len(rows) < self.PAGE_LIMIT:
                complete = True
                break
            # 满页：把窗口起点前移到本页最大时间之后（+1 跳过边界，重复由唯一键挡住）
            if new_last_time + 1 <= start_ms:
                complete = False
                break
            start_ms = new_last_time + 1
            last_time = new_last_time
        cursor = self._store.get_sync_cursor(scope, stream, symbol)
        return SyncResult(
            market=scope, stream=stream, symbol=symbol,
            inserted=inserted_total, pages=pages,
            cursor_time_ms=cursor.last_time_ms if cursor is not None else last_time,
            complete=complete,
        )

    def _parse_income(self, symbol: str, row: dict[str, Any]) -> dict[str, Any] | None:
        """income 行 → 入账 dict。字段缺失抛 SyncParseError；正负号原样保留。"""
        if str(row.get("incomeType")) != "FUNDING_FEE":
            return None
        income_id = row.get("id")
        time_raw = row.get("time")
        amount_raw = row.get("income")
        if income_id in (None, "") or time_raw in (None, "") or amount_raw in (None, ""):
            raise SyncParseError(f"funding income 行字段不完整（id/time/income）: {row!r}")
        ts_ms = int(str(time_raw))
        return {
            "cashflow_id": f"inc-{income_id}",
            "market": "PERP",
            "symbol": str(row.get("symbol") or symbol),
            "funding_ts_ms": ts_ms,
            "funding_rate": "0",  # income 接口不含费率；事实金额以 income 为准
            "interval_hours": 8,
            "asset": str(row.get("asset") or "USDT"),
            "amount": Decimal(str(amount_raw)),
            "source": "exchange_income",
            "raw_summary": f'{{"income_type": "FUNDING_FEE", "income_id": "{income_id}"}}',
            "exchange_income_id": str(income_id),
            "authority": FundingAuthority.AUTHORITATIVE.value,
            "observed_ms": int(self._now() * 1000),
        }

    # -- 回补起点 --------------------------------------------------------------

    def _backfill_start_ms(self) -> int:
        """无 cursor 时的初始回补起点：min(当前 - recovery_backfill_days,
        最早未关闭 pair 起点)；交易所保留窗口之外的部分无法证明完整。"""
        now_ms = int(self._now() * 1000)
        window_start = now_ms - int(self._config.execution.recovery_backfill_days * 86_400_000)
        try:
            open_pairs = self._store.open_pairs()
        except Exception:  # noqa: BLE001 —— 查不到未关闭 pair 时退回配置窗口
            return window_start
        earliest = None
        for pair in open_pairs:
            created = pair.get("created_ms")
            if created in (None, ""):
                continue
            value = int(str(created))
            earliest = value if earliest is None else min(earliest, value)
        return min(window_start, earliest) if earliest is not None else window_start
