"""账户快照 → RiskState 测试（开发文档 §7.5：资金未知 = 拒绝，不得用默认值放行）。"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cointrader.execution.store import StateStore
from cointrader.live.account_state import (
    AccountStateBuilder,
    AccountStateError,
    parse_account,
)
from live_helpers import NOW, make_live_config

NOW_MS = int(NOW * 1000)


def _spot_payload() -> dict:
    return {
        "balances": [
            {"asset": "USDT", "free": "9999", "locked": "1"},
            {"asset": "BTC", "free": "0.01", "locked": "0"},
        ]
    }


def _futures_payload() -> dict:
    return {
        "totalWalletBalance": "10000",
        "availableBalance": "9999",
        "totalUnrealizedProfit": "0",
    }


def _parse(**overrides):
    kw = dict(
        spot_payload=_spot_payload(),
        futures_payload=_futures_payload(),
        perp_positions=[],
        realized_pnl_today=Decimal("0"),
        now_ms=NOW_MS,
        candidate_symbols=("BTCUSDT",),
    )
    kw.update(overrides)
    return parse_account(**kw)


class TestParseAccount:
    def test_valid_payload_builds_risk_state(self):
        result = _parse()
        assert result.total_capital == Decimal("9999") + Decimal("1") + Decimal("10000")
        assert result.state.total_capital == float(result.total_capital)
        assert result.state.available_balance == float(Decimal("9999") + Decimal("9999"))
        assert result.state.futures_wallet_balance == 10000.0
        assert result.state.snapshot_source == "periodic"
        assert result.state.snapshot_ts_ms == NOW_MS

    def test_position_from_exchange_not_ledger(self):
        result = _parse(perp_positions=[
            {"symbol": "BTCUSDT", "positionSide": "BOTH", "positionAmt": "-0.01"},
        ])
        assert "BTCUSDT" in result.state.positions
        pos = result.state.positions["BTCUSDT"]
        assert pos.spot_qty == 0.01
        assert pos.perp_qty == 0.01

    @pytest.mark.parametrize("mutate,ctx_key", [
        (lambda p: p.pop("balances"), "spot.balances"),
        (lambda p: p["balances"].clear(), "spot.balances"),
        (lambda p: p.update(balances=[{"asset": "BTC", "free": "1", "locked": "0"}]),
         "USDT"),
        (lambda p: p.pop("totalWalletBalance"), "futures.totalWalletBalance"),
        (lambda p: p.update(availableBalance=""), "futures.availableBalance"),
    ])
    def test_missing_or_empty_fields_raise(self, mutate, ctx_key):
        spot = _spot_payload()
        fut = _futures_payload()
        if "futures" in ctx_key:
            mutate(fut)
        else:
            mutate(spot)
        with pytest.raises(AccountStateError):
            _parse(spot_payload=spot, futures_payload=fut)

    def test_unparseable_number_raises(self):
        fut = _futures_payload()
        fut["totalWalletBalance"] = "not-a-number"
        with pytest.raises(AccountStateError):
            _parse(futures_payload=fut)

    def test_zero_capital_raises_not_default(self):
        """资金为零 = 资金未知 → 抛错；不得返回 0 资金 RiskState 放行。"""
        spot = _spot_payload()
        spot["balances"] = [{"asset": "USDT", "free": "0", "locked": "0"}]
        fut = _futures_payload()
        fut["totalWalletBalance"] = "0"
        with pytest.raises(AccountStateError, match="零"):
            _parse(spot_payload=spot, futures_payload=fut)


class TestBuilderSnapshot:
    def test_snapshot_writes_ledger_and_returns_state(self, tmp_path):
        cfg = make_live_config()
        store = StateStore(tmp_path / "t.sqlite3")

        class _Spot:
            def account(self):
                return _spot_payload()

        class _Futures:
            def account(self):
                return _futures_payload()

            def positions(self, symbol=None):
                return []

        builder = AccountStateBuilder(
            config=cfg, store=store,
            candidate_symbols=("BTCUSDT",), now_fn=lambda: NOW,
        )
        result = builder.snapshot(spot=_Spot(), futures=_Futures(), source="periodic")
        assert result.total_capital > 0
        snaps = store.account_snapshots(limit=10)
        sources = {s["source"] for s in snaps}
        assert {"SPOT", "PERP"} <= sources

    def test_snapshot_failure_raises(self, tmp_path):
        cfg = make_live_config()
        store = StateStore(tmp_path / "t.sqlite3")

        class _Spot:
            def account(self):
                raise RuntimeError("api down")

        class _Futures:
            def account(self):
                return _futures_payload()

            def positions(self, symbol=None):
                return []

        builder = AccountStateBuilder(
            config=cfg, store=store,
            candidate_symbols=("BTCUSDT",), now_fn=lambda: NOW,
        )
        with pytest.raises(Exception, match="api down"):
            builder.snapshot(spot=_Spot(), futures=_Futures())
