"""交易所事实同步测试（实施计划书 v2.0 T3，AC-06/08）。

全部离线：fake adapter + 临时 SQLite。覆盖：

- fills 分页（满页推进 / 非满页终止 / 游标推进）；
- 重复事件幂等（(market, exchange_trade_id) 唯一索引）；
- 断页重启：中途失败 → cursor 留在上一页 → 重跑无重复、最终完整；
- income 回补：(market, exchange_income_id) 幂等、正负号保留、AUTHORITATIVE；
- 重复恢复不改变 facts/PnL 数量。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cointrader.config import Config
from cointrader.execution.store import StateStore
from cointrader.execution.sync import ExchangeStateSynchronizer

NOW = 1_800_000_000.0
NOW_MS = int(NOW * 1000)
SYMBOL = "BTCUSDT"


def _cfg() -> Config:
    from live_helpers import make_live_config

    return make_live_config(
        live_symbols=(SYMBOL,),
        exec_overrides={
            "recovery_backfill_days": 30,
            "exchange_snapshot_reuse_seconds": 0.001,  # 测试窗口≈0：不复用 bundle
        },
    )


class _FakeFutures:
    def __init__(self, trades: dict[str, list[dict]], income: list[dict]) -> None:
        self._trades = trades
        self._income = income
        self._fail_from_page: int | None = None

    def account(self) -> dict:
        return {"totalWalletBalance": "10000", "availableBalance": "9999"}

    def positions(self, symbol: str | None = None) -> list[dict]:
        return []

    def open_orders(self, symbol: str | None = None) -> list[dict]:
        return []

    def user_trades(self, symbol: str, *, from_id=None, start_ms=None, limit=100) -> list[dict]:
        if self._fail_from_page is not None:
            if start_ms is None or start_ms >= self._fail_from_page:
                raise RuntimeError("network down (fake)")
        rows = self._trades.get(symbol, [])
        if start_ms is not None:
            rows = [r for r in rows if r["time"] >= start_ms]
        return rows[:limit]

    def income_history(
        self, *, income_type: str = "FUNDING_FEE", start_ms=None, end_ms=None, limit=1000
    ) -> list[dict]:
        rows = self._income
        if start_ms is not None:
            rows = [r for r in rows if r["time"] >= start_ms]
        return rows[:limit]


class _FakeSpot:
    def __init__(self, trades: dict[str, list[dict]]) -> None:
        self._trades = trades

    def account(self) -> dict:
        return {"balances": []}

    def open_orders(self, symbol: str | None = None) -> list[dict]:
        return []

    def my_trades(self, symbol: str, *, from_id=None, start_ms=None, limit=100) -> list[dict]:
        rows = self._trades.get(symbol, [])
        if start_ms is not None:
            rows = [r for r in rows if r["time"] >= start_ms]
        return rows[:limit]


def _trade(i: int, *, market_side: str = "BUY") -> dict:
    return {
        "id": i,
        "clientOrderId": f"ct-x-{i}",
        "orderId": 1000 + i,
        "price": "100",
        "qty": "0.01",
        "quoteQty": "1",
        "commission": "0.001",
        "commissionAsset": "USDT",
        "time": NOW_MS - (10000 - i) * 1000,
        "isBuyer": market_side == "BUY",
        "isMaker": False,
        "symbol": SYMBOL,
    }


def _income(i: int, amount: str) -> dict:
    return {
        "id": 9000 + i,
        "incomeType": "FUNDING_FEE",
        "time": NOW_MS - i * 3_600_000,
        "symbol": SYMBOL,
        "asset": "USDT",
        "income": amount,
    }


@pytest.fixture()
def store(tmp_path) -> StateStore:
    return StateStore(tmp_path / "trading.sqlite3")


@pytest.fixture()
def cfg() -> Config:
    return _cfg()


class TestFillsSync:
    def test_multi_page_pagination_and_cursor(self, store, cfg) -> None:
        """2500 条成交：3 页（1000/1000/500），游标推进到最后一页。"""
        trades = [_trade(i) for i in range(1, 2501)]
        spot = _FakeSpot({SYMBOL: trades})
        fut = _FakeFutures({}, [])
        sync = ExchangeStateSynchronizer(
            store=store, spot=spot, futures=fut, config=cfg, now_fn=lambda: NOW
        )
        results = sync.sync_fills([SYMBOL])
        spot_res = [r for r in results if r.market == "SPOT"][0]
        assert spot_res.complete is True
        assert spot_res.pages == 3
        assert spot_res.inserted == 2500
        cursor = store.get_sync_cursor("SPOT", "fills", SYMBOL)
        assert cursor is not None
        assert cursor.last_time_ms == trades[-1]["time"]

        # 重复恢复：游标续跑，零新增，幂等
        results2 = sync.sync_fills([SYMBOL])
        spot_res2 = [r for r in results2 if r.market == "SPOT"][0]
        assert spot_res2.inserted == 0
        assert len(store.fills(limit=10000)) == 2500

    def test_mid_page_failure_keeps_cursor_on_last_page(self, store, cfg) -> None:
        """第二页失败 → cursor 停在第一页边界；修复后重跑无重复、最终完整。"""
        trades = [_trade(i) for i in range(1, 2501)]
        fut = _FakeFutures({SYMBOL: trades}, [])
        spot = _FakeSpot({SYMBOL: []})
        # 先让 futures 侧第一页成功后在第二页断掉
        fut._fail_from_page = trades[999]["time"]  # 第二页起点

        sync = ExchangeStateSynchronizer(
            store=store, spot=spot, futures=fut, config=cfg, now_fn=lambda: NOW
        )
        results = sync.sync_fills([SYMBOL])
        perp_res = [r for r in results if r.market == "PERP"][0]
        assert perp_res.complete is False
        assert perp_res.error != ""
        cursor = store.get_sync_cursor("PERP", "fills", SYMBOL)
        assert cursor is not None
        # 第一页 1000 条已落盘，cursor 停在第一页最大时间
        assert cursor.last_time_ms == trades[999]["time"]
        assert len(store.fills(limit=10000)) == 1000

        # 网络恢复 → 续跑：不重复、补齐剩余
        fut._fail_from_page = None
        results2 = sync.sync_fills([SYMBOL])
        perp_res2 = [r for r in results2 if r.market == "PERP"][0]
        assert perp_res2.complete is True
        assert perp_res2.inserted == 1500
        rows = store.fills(limit=10000)
        assert len(rows) == 2500
        # v2 写入规则：fill_id = "<MARKET>:<exchange_trade_id>"
        assert {r["fill_id"] for r in rows} == {f"PERP:{i}" for i in range(1, 2501)}

    def test_duplicate_trade_ids_never_double_counted(self, store, cfg) -> None:
        """同一 (market, exchange_trade_id) 重复回补被唯一索引挡住。"""
        spot = _FakeSpot({SYMBOL: [_trade(1), _trade(2)]})
        fut = _FakeFutures({}, [])
        sync = ExchangeStateSynchronizer(
            store=store, spot=spot, futures=fut, config=cfg, now_fn=lambda: NOW
        )
        sync.sync_fills([SYMBOL])
        sync.sync_fills([SYMBOL])
        assert len(store.fills(limit=1000)) == 2

    def test_missing_required_fields_raise_parse_error(self, store, cfg) -> None:
        """字段缺失 → SyncParseError（不猜测字段），结果带 error。"""
        bad = {"id": 1, "price": "100", "time": NOW_MS}  # 缺 qty
        spot = _FakeSpot({SYMBOL: [bad]})
        fut = _FakeFutures({}, [])
        sync = ExchangeStateSynchronizer(
            store=store, spot=spot, futures=fut, config=cfg, now_fn=lambda: NOW
        )
        results = sync.sync_fills([SYMBOL])
        spot_res = [r for r in results if r.market == "SPOT"][0]
        assert spot_res.error != ""
        assert "字段不完整" in spot_res.error
        # 解析失败不得推进 cursor 后丢失数据（无合法事实）
        assert len(store.fills(limit=100)) == 0

    def test_backfill_start_uses_earliest_open_pair(self, store, cfg) -> None:
        """无 cursor 时回补起点 = min(配置窗口, 最早未关闭 pair 起点)。"""
        spot = _FakeSpot({SYMBOL: [_trade(1)]})
        fut = _FakeFutures({}, [])
        sync = ExchangeStateSynchronizer(
            store=store, spot=spot, futures=fut, config=cfg, now_fn=lambda: NOW
        )
        assert sync._backfill_start_ms() == NOW_MS - 30 * 86_400_000  # noqa: SLF001
        from cointrader.execution.models import PairExecution

        old_ms = NOW_MS - 60 * 86_400_000  # 早于 30 天窗口
        store.upsert_pair(PairExecution(
            pair_execution_id="pair-old", symbol=SYMBOL, target_notional=Decimal("1"),
            strategy_version="t", status="SUBMIT_SPOT", kind="open",
            created_ms=old_ms, updated_ms=old_ms,
        ))
        assert sync._backfill_start_ms() == old_ms  # noqa: SLF001


class TestIncomeSync:
    def test_income_idempotent_and_sign_preserved(self, store, cfg) -> None:
        """重复恢复不重复入账；正负号采用交易所事实（空头收正/付负）。"""
        income = [
            _income(1, "0.008"),
            _income(2, "-0.012"),  # 负数 = 我们付出
            _income(3, "0.005"),
        ]
        spot = _FakeSpot({})
        fut = _FakeFutures({}, income)
        sync = ExchangeStateSynchronizer(
            store=store, spot=spot, futures=fut, config=cfg, now_fn=lambda: NOW
        )
        for _ in range(3):  # 连续三次「重启恢复」
            results = sync.sync_funding_income([SYMBOL])
            res = results[0]
            assert res.complete is True
        rows = store.funding_cashflows(limit=100)
        assert len(rows) == 3
        amounts = sorted(str(r["amount"]) for r in rows)
        assert amounts == ["-0.012", "0.005", "0.008"]
        assert all(r["authority"] == "AUTHORITATIVE" for r in rows)
        assert all(r["exchange_income_id"] for r in rows)

    def test_parse_error_on_missing_income_fields(self, store, cfg) -> None:
        bad = [{"id": 1, "incomeType": "FUNDING_FEE", "time": NOW_MS, "symbol": SYMBOL}]  # 缺 income
        spot = _FakeSpot({})
        fut = _FakeFutures({}, bad)
        sync = ExchangeStateSynchronizer(
            store=store, spot=spot, futures=fut, config=cfg, now_fn=lambda: NOW
        )
        results = sync.sync_funding_income([SYMBOL])
        assert results[0].error != ""
        assert "字段不完整" in results[0].error
