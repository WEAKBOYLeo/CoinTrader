"""组合 planner 契约测试（实施计划书 3.0 T2，AC-03）。

覆盖：目标差异（open/resize/close/replace）、稳定排序（CLOSE 先于 OPEN/
REPLACE）、确定性幂等（重复 diff 不产生新意图）、换仓不先扩大敞口
（顺序保证）、legacy Signal → 领域意图适配器。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cointrader.domain.common import InvalidDomainValue
from cointrader.domain.portfolio import (
    CurrentPosition,
    IntentAction,
    PortfolioView,
)
from cointrader.domain.strategy import TargetPortfolio, TargetPosition
from cointrader.live.portfolio import build_signal
from cointrader.portfolio.planner import PortfolioPlanner
from live_helpers import NOW_MS, make_live_config

T0 = 1_800_000_000_000


def _view(symbols: dict[str, tuple[str, str]], snapshot_id: str = "view-1", as_of_ms: int = T0) -> PortfolioView:
    return PortfolioView(
        snapshot_id=snapshot_id,
        as_of_ms=as_of_ms,
        entries=tuple(
            CurrentPosition(
                symbol=symbol,
                spot_notional=Decimal(spot),
                perp_notional=Decimal(perp),
                updated_at_ms=as_of_ms,
            )
            for symbol, (spot, perp) in sorted(symbols.items())
        ),
    )


def _target(symbols: dict[str, tuple[str, str]]) -> TargetPortfolio:
    return TargetPortfolio(
        entries=tuple(
            TargetPosition(symbol=symbol, spot_notional=Decimal(spot), perp_notional=Decimal(perp))
            for symbol, (spot, perp) in sorted(symbols.items())
        )
    )


class TestDiff:
    def test_open_resize_close(self) -> None:
        planner = PortfolioPlanner()
        view = _view({"AAAUSDT": ("100", "100"), "BBBUSDT": ("50", "50")})
        target = _target({"AAAUSDT": ("200", "200"), "CCCUSDT": ("80", "80")})
        intents = planner.diff(target, view, created_at_ms=T0)
        by_symbol = {i.symbol: i for i in intents}
        # BBB 目标缺失 → CLOSE；CCC 新增且与 CLOSE 共存 → REPLACE（换仓腿）；AAA 变化 → RESIZE
        assert by_symbol["BBBUSDT"].action is IntentAction.CLOSE
        assert by_symbol["BBBUSDT"].target_spot_notional == 0
        assert by_symbol["CCCUSDT"].action is IntentAction.REPLACE
        assert by_symbol["AAAUSDT"].action is IntentAction.RESIZE
        assert by_symbol["AAAUSDT"].target_spot_notional == Decimal("200")
        # 稳定排序：CLOSE 先于 REPLACE/OPEN，REPLACE/OPEN 先于 RESIZE
        assert [i.action for i in intents] == [
            IntentAction.CLOSE,
            IntentAction.REPLACE,
            IntentAction.RESIZE,
        ]

    def test_no_change_no_intent(self) -> None:
        planner = PortfolioPlanner()
        view = _view({"AAAUSDT": ("100", "100")})
        target = _target({"AAAUSDT": ("100", "100")})
        assert planner.diff(target, view, created_at_ms=T0) == ()

    def test_idempotent_repeated_diff(self) -> None:
        """重复输入不增加新意图：intent_id = 确定性指纹。"""
        planner = PortfolioPlanner()
        view = _view({"AAAUSDT": ("100", "100")})
        target = _target({"BBBUSDT": ("30", "30")})
        first = planner.diff(target, view, created_at_ms=T0)
        second = planner.diff(target, view, created_at_ms=T0 + 999_999)
        assert [i.fingerprint() for i in first] == [i.fingerprint() for i in second]
        assert [i.intent_id for i in first] == [i.intent_id for i in second]
        assert len({i.intent_id for i in first}) == len(first)

    def test_replace_marks_open_leg_and_orders_close_first(self) -> None:
        """换仓：close+open 共存 → OPEN 腿标记 REPLACE；CLOSE 在 OPEN 之前
        （先平旧后开新，不先扩大总敞口）。"""
        planner = PortfolioPlanner()
        view = _view({"AAAUSDT": ("100", "100")})
        target = _target({"BBBUSDT": ("100", "100")})
        intents = planner.diff(target, view, created_at_ms=T0)
        assert len(intents) == 2
        assert intents[0].action is IntentAction.CLOSE
        assert intents[0].symbol == "AAAUSDT"
        assert intents[1].action is IntentAction.REPLACE
        assert intents[1].symbol == "BBBUSDT"
        # 顺序保证：任何时刻敞口 ≤ max(旧, 新)
        assert intents[0].target_spot_notional + intents[0].target_perp_notional == 0

    def test_stable_ordering_by_symbol(self) -> None:
        planner = PortfolioPlanner()
        view = _view({f"{s}USDT": ("10", "10") for s in ("Z", "A", "M")})
        target = _target({})
        intents = planner.diff(target, view, created_at_ms=T0)
        assert [i.symbol for i in intents] == ["AUSDT", "MUSDT", "ZUSDT"]
        assert all(i.action is IntentAction.CLOSE for i in intents)

    def test_intent_fields_traceable(self) -> None:
        planner = PortfolioPlanner()
        view = _view({}, snapshot_id="snap-42", as_of_ms=T0 + 123)
        target = _target({"AAAUSDT": ("7", "7")})
        intent = planner.diff(
            target, view, created_at_ms=T0, correlation_id="corr-1", causation_id="cause-1"
        )[0]
        assert intent.snapshot_id == "snap-42"
        assert intent.decision_cutoff_ms == T0 + 123
        assert intent.correlation_id == "corr-1"
        assert intent.causation_id == "cause-1"
        assert intent.is_closing is False


class TestLegacySignalAdapter:
    def test_build_signal_to_intent_round_trip(self) -> None:
        config = make_live_config()
        signal = build_signal(
            "BTCUSDT",
            spot_price=Decimal("100"),
            perp_price=Decimal("100"),
            quote_ts_ms=NOW_MS,
            requested_notional=Decimal("50"),
            config=config,
            reason="funding_carry",
            strategy_version="v-test",
            now_ms=NOW_MS,
        )
        assert signal is not None
        intent = signal.to_intent(
            snapshot_id="s1", decision_cutoff_ms=NOW_MS, created_at_ms=NOW_MS
        )
        assert intent.action is IntentAction.OPEN
        assert intent.target_spot_notional == signal.target_notional
        assert intent.target_perp_notional == signal.target_notional
        assert intent.reason == "funding_carry"
        # 确定性：同输入重复转换同意图
        again = signal.to_intent(
            snapshot_id="s1", decision_cutoff_ms=NOW_MS, created_at_ms=NOW_MS
        )
        assert again.intent_id == intent.intent_id

    def test_build_signal_rejects_bad_price(self) -> None:
        config = make_live_config()
        signal = build_signal(
            "BTCUSDT",
            spot_price=Decimal("0"),
            perp_price=Decimal("100"),
            quote_ts_ms=NOW_MS,
            requested_notional=Decimal("50"),
            config=config,
            now_ms=NOW_MS,
        )
        assert signal is None


class TestDomainPortfolioEdge:
    def test_close_intent_nonzero_target_rejected(self) -> None:
        from cointrader.domain.portfolio import PortfolioIntent

        with pytest.raises(InvalidDomainValue):
            PortfolioIntent(
                intent_id="i",
                action=IntentAction.CLOSE,
                symbol="A",
                target_spot_notional=Decimal("1"),
                target_perp_notional=Decimal("0"),
                reason="r",
                snapshot_id="s",
                decision_cutoff_ms=0,
                created_at_ms=0,
            )


class TestReducesRiskAndExposureInvariant:
    """T1：减风险可观察字段 + 换仓「任何时刻敞口不先扩大」前缀不变量。"""

    def test_reduces_risk_flags(self) -> None:
        planner = PortfolioPlanner()
        # CLOSE → True；RESIZE-down → True；RESIZE-up → False；OPEN → False
        view = _view(
            {
                "AAAUSDT": ("100", "100"),  # RESIZE up → 200
                "BBBUSDT": ("100", "100"),  # RESIZE down → 50
                "CCCUSDT": ("10", "10"),  # CLOSE
            }
        )
        target = _target(
            {
                "AAAUSDT": ("200", "200"),
                "BBBUSDT": ("50", "50"),
                "DDDUSDT": ("80", "80"),  # OPEN
            }
        )
        intents = planner.diff(target, view, created_at_ms=T0)
        by_symbol = {i.symbol: i for i in intents}
        assert by_symbol["CCCUSDT"].reduces_risk is True
        assert by_symbol["BBBUSDT"].reduces_risk is True
        assert by_symbol["AAAUSDT"].reduces_risk is False
        assert by_symbol["DDDUSDT"].reduces_risk is False

    def test_resize_down_only_when_target_below_held(self) -> None:
        planner = PortfolioPlanner()
        view = _view({"AAAUSDT": ("200", "200")})
        up = planner.diff(
            _target({"AAAUSDT": ("400", "400")}), view, created_at_ms=T0
        )[0]
        down = planner.diff(
            _target({"AAAUSDT": ("100", "100")}), view, created_at_ms=T0
        )[0]
        assert up.action is IntentAction.RESIZE and up.reduces_risk is False
        assert down.action is IntentAction.RESIZE and down.reduces_risk is True

    def test_replace_prefix_never_exceeds_max_exposure(self) -> None:
        """换仓：按 planner 输出顺序逐个应用，任何前缀的该组总敞口
        ≤ max(旧总量, 新总量)（先平旧后开新，不先扩大）。"""
        planner = PortfolioPlanner()
        old_total = Decimal("300")  # AAA (100,100) + BBB (50,50)
        new_total = Decimal("90")  # CCC (45,45)
        view = _view({"AAAUSDT": ("100", "100"), "BBBUSDT": ("50", "50")})
        target = _target({"CCCUSDT": ("45", "45")})
        intents = planner.diff(target, view, created_at_ms=T0)
        # CLOSE 全部在前，REPLACE 在后
        assert all(i.action is IntentAction.CLOSE for i in intents[:2])
        assert intents[2].action is IntentAction.REPLACE

        exposure = {e.symbol: e.spot_notional + e.perp_notional for e in view.entries}
        bound = max(old_total, new_total)
        for intent in intents:
            delta = (
                intent.target_spot_notional + intent.target_perp_notional
            ) - exposure.get(intent.symbol, Decimal("0"))
            exposure[intent.symbol] = exposure.get(intent.symbol, Decimal("0")) + delta
            assert sum(exposure.values()) <= bound, (
                f"前缀 {intent.symbol}/{intent.action} 后敞口超过 max(旧,新)"
            )

    def test_close_intent_serialization_carries_reduces_risk(self) -> None:
        import json

        from cointrader.domain.portfolio import PortfolioIntent

        planner = PortfolioPlanner()
        view = _view({"AAAUSDT": ("100", "100")})
        close = planner.diff(_target({}), view, created_at_ms=T0)[0]
        assert close.reduces_risk is True
        restored = PortfolioIntent.from_dict(json.loads(json.dumps(close.to_dict())))
        assert restored.reduces_risk is True
        assert restored == close
