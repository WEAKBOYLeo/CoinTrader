"""市场数据/账户状态 facade 契约测试（实施计划书 3.0 T2 新增测试辅助文件）。

覆盖（AC-05/08 的 T2 部分）：

- ``MarketDataService``：READY → FRESH；非 READY → DEGRADED/STALE/INCOMPLETE；
  无数据 → INCOMPLETE 空 quotes（不伪造价格）；mark 价非正的候选跳过。
- ``AccountProjector``：仅完整 capture 更新 current projection 并写账本；
  不完整不覆盖可信状态、不写账本；新鲜度判定。
- legacy 报价 → 领域报价适配器（无前瞻校验）。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cointrader.account.projector import AccountProjector
from cointrader.domain.account import AccountSnapshot
from cointrader.domain.market import DataQuality
from cointrader.domain.portfolio import PortfolioView
from cointrader.market_data.service import MarketDataService
from cointrader.portfolio.adapter import quote_to_instrument_quotes

NOW_MS = 1_800_000_000_000


class _FakeSnapshot:
    def __init__(self, symbol: str, mark: Decimal, rate: Decimal, ts: int) -> None:
        self.symbol = symbol
        self.rates = (rate,)
        self.mark_prices = (mark,)
        self.timestamps = (ts,)
        self.interval_hours = 8
        self.quote_volume_3d_avg = Decimal("10000000")
        self.fetched_ms = ts
        self.error = ""


class _FakeEpoch:
    def __init__(self, epoch_id: str, status: str, cutoff_ms: int) -> None:
        self.epoch_id = epoch_id
        self.status = status
        self.decision_cutoff_ms = cutoff_ms
        self.excluded: dict[str, str] = {}


class _FakeSource:
    def __init__(self) -> None:
        self.ready: _FakeEpoch | None = None
        self.latest_epoch: _FakeEpoch | None = None
        self.snaps: dict[str, dict[str, _FakeSnapshot]] = {}

    def latest_ready(self) -> _FakeEpoch | None:
        return self.ready

    def latest(self) -> _FakeEpoch | None:
        return self.latest_epoch

    def snapshots_for(self, epoch_id: str) -> dict[str, _FakeSnapshot]:
        return self.snaps.get(epoch_id, {})

    def expected_symbols(self) -> tuple[str, ...]:
        return tuple(self.snaps.get(self.ready.epoch_id if self.ready else "", {}).keys())


def _make_source() -> _FakeSource:
    source = _FakeSource()
    epoch = _FakeEpoch("ep1", "READY", NOW_MS - 1000)
    source.ready = epoch
    source.latest_epoch = epoch
    source.snaps["ep1"] = {
        "BTCUSDT": _FakeSnapshot("BTCUSDT", Decimal("100"), Decimal("0.0001"), NOW_MS - 5000),
        "ETHUSDT": _FakeSnapshot("ETHUSDT", Decimal("0"), Decimal("0.0001"), NOW_MS - 5000),
    }
    return source


class TestMarketDataService:
    def test_ready_epoch_is_fresh_and_drops_bad_prices(self) -> None:
        service = MarketDataService(_make_source(), now_fn=lambda: NOW_MS / 1000)
        snap = service.snapshot()
        assert snap.quality is DataQuality.FRESH
        assert snap.snapshot_id == "ep1"
        symbols = {q.symbol for q in snap.quotes}
        assert symbols == {"BTCUSDT"}  # ETH mark=0 不伪造价格，跳过
        btc_quote = next(q for q in snap.quotes if q.symbol == "BTCUSDT")
        assert btc_quote.price == Decimal("100")

    def test_no_epoch_is_incomplete_with_no_prices(self) -> None:
        source = _FakeSource()
        service = MarketDataService(source, now_fn=lambda: NOW_MS / 1000)
        snap = service.snapshot()
        assert snap.quality is DataQuality.INCOMPLETE
        assert snap.quotes == ()

    def test_degraded_epoch_not_fresh(self) -> None:
        source = _make_source()
        source.ready = None
        source.latest_epoch = _FakeEpoch("ep2", "DEGRADED", NOW_MS - 2000)
        source.snaps["ep2"] = source.snaps["ep1"]
        service = MarketDataService(source, now_fn=lambda: NOW_MS / 1000)
        snap = service.snapshot()
        assert snap.quality is DataQuality.DEGRADED

    def test_symbol_filter(self) -> None:
        source = _make_source()
        service = MarketDataService(source, now_fn=lambda: NOW_MS / 1000)
        snap = service.snapshot(symbols=["BTCUSDT"])
        assert [q.symbol for q in snap.quotes] == ["BTCUSDT"]


class _FakeQuery:
    def __init__(self, snapshots: list[AccountSnapshot]) -> None:
        self._snaps = list(snapshots)
        self.calls = 0
        self._current: AccountSnapshot | None = None

    def capture(self) -> AccountSnapshot:
        self.calls += 1
        snap = self._snaps.pop(0)
        if snap.complete:
            self._current = snap
        return snap

    def current(self) -> AccountSnapshot | None:
        return self._current


def _snap(complete: bool, id: str = "a1") -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=id,
        capture_start_ms=NOW_MS - 100,
        capture_end_ms=NOW_MS,
        complete=complete,
        equity=Decimal("1000"),
        available=Decimal("800"),
    )


class TestAccountProjector:
    def test_incomplete_capture_does_not_update_current_or_ledger(self) -> None:
        saved: list[AccountSnapshot] = []

        class _Writer:
            def save_complete(self, snapshot: AccountSnapshot) -> None:
                saved.append(snapshot)

        query = _FakeQuery([_snap(complete=False)])
        projector = AccountProjector(query, writer=_Writer())
        result = projector.refresh()
        assert result.complete is False
        assert projector.current is None  # 可信状态不被覆盖/初始化
        assert saved == []  # 不写账本

    def test_complete_capture_updates_current_and_writes(self) -> None:
        saved: list[AccountSnapshot] = []

        class _Writer:
            def save_complete(self, snapshot: AccountSnapshot) -> None:
                saved.append(snapshot)

        query = _FakeQuery([_snap(complete=False), _snap(complete=True, id="a2")])
        projector = AccountProjector(query, writer=_Writer())
        projector.refresh()
        assert projector.current is None
        projector.refresh()
        assert projector.current is not None
        assert projector.current.snapshot_id == "a2"
        assert [s.snapshot_id for s in saved] == ["a2"]

    def test_freshness(self) -> None:
        query = _FakeQuery([_snap(complete=True)])
        projector = AccountProjector(query)
        projector.refresh()
        assert projector.freshness_ok(NOW_MS + 1000, max_age_ms=5000) is True
        assert projector.freshness_ok(NOW_MS + 10_000, max_age_ms=5000) is False


class TestQuoteAdapter:
    def test_legacy_quote_to_domain(self) -> None:
        class _Q:
            spot_price = Decimal("100")
            perp_price = Decimal("100.2")
            ts_ms = NOW_MS - 100
            spot_ts_ms = 0
            perp_ts_ms = 0

        spot, perp = quote_to_instrument_quotes(
            _Q(), "BTCUSDT", generated_at_ms=NOW_MS, decision_cutoff_ms=NOW_MS
        )
        assert spot.symbol == perp.symbol == "BTCUSDT"
        assert float(spot.price) == 100.0
        assert float(perp.price) == 100.2
        assert spot.quote_time_ms == NOW_MS - 100

    def test_future_quote_rejected(self) -> None:
        class _Q:
            spot_price = Decimal("100")
            perp_price = Decimal("100")
            ts_ms = NOW_MS + 100  # 未来报价
            spot_ts_ms = 0
            perp_ts_ms = 0

        with pytest.raises(ValueError, match="无前瞻"):
            quote_to_instrument_quotes(
                _Q(), "BTCUSDT", generated_at_ms=NOW_MS, decision_cutoff_ms=NOW_MS
            )


def _view() -> PortfolioView:
    return PortfolioView(snapshot_id="s1", as_of_ms=NOW_MS, entries=())


def test_planner_view_from_entries_stable_order() -> None:
    from cointrader.domain.portfolio import CurrentPosition
    from cointrader.portfolio.planner import PortfolioPlanner

    view = PortfolioPlanner.view_from_entries(
        (
            CurrentPosition("B", Decimal("1"), Decimal("1"), NOW_MS),
            CurrentPosition("A", Decimal("1"), Decimal("1"), NOW_MS),
        ),
        snapshot_id="s1",
        as_of_ms=NOW_MS,
    )
    assert [e.symbol for e in view.entries] == ["A", "B"]
