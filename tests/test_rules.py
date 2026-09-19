"""rules 规则解析与 Decimal 归一化单元测试（开发设计文档 §3.3 / §11.1）。"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cointrader.execution.rules import (
    RuleError,
    SymbolRules,
    check_notional,
    floor_to_step,
    format_decimal,
    normalize_price,
    normalize_qty,
    parse_futures_exchange_info,
    parse_spot_exchange_info,
    round_price,
)


def make_rules(**overrides: object) -> SymbolRules:
    base: dict[str, object] = dict(
        symbol="BTCUSDT",
        market="spot",
        status="TRADING",
        base_asset="BTC",
        quote_asset="USDT",
        tick_size=Decimal("0.01"),
        min_qty=Decimal("0.0001"),
        max_qty=Decimal("1000"),
        step_size=Decimal("0.0001"),
        min_notional=Decimal("10"),
    )
    base.update(overrides)
    return SymbolRules(**base)  # type: ignore[arg-type]


class TestFormatDecimal:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (Decimal("0.0010"), "0.001"),
            (Decimal("100"), "100"),
            (Decimal("1E+2"), "100"),
            (Decimal("0.000"), "0"),
            (Decimal("-0.000"), "0"),
            (Decimal("1.500"), "1.5"),
            ("0.1", "0.1"),
            (2, "2"),
        ],
    )
    def test_no_scientific_notation(self, value: object, expected: str) -> None:
        text = format_decimal(value)
        assert text == expected
        assert "e" not in text.lower(), "交易所参数禁止科学计数法"

    def test_float_is_converted_via_str(self) -> None:
        assert format_decimal(0.1) == "0.1"


class TestFloorToStep:
    def test_floors_to_step_multiple(self) -> None:
        assert floor_to_step(Decimal("1.99"), Decimal("0.01")) == Decimal("1.99")
        assert floor_to_step(Decimal("1.999"), Decimal("0.01")) == Decimal("1.99")

    def test_uneven_step(self) -> None:
        assert floor_to_step(Decimal("0.0015"), Decimal("0.001")) == Decimal("0.001")
        assert floor_to_step(Decimal("0.0014"), Decimal("0.001")) == Decimal("0.001")

    def test_zero_step_rejected(self) -> None:
        with pytest.raises(RuleError, match="step 必须为正"):
            floor_to_step(Decimal("1"), Decimal("0"))

    def test_negative_value_rejected(self) -> None:
        with pytest.raises(RuleError, match="不能为负"):
            floor_to_step(Decimal("-1"), Decimal("0.01"))


class TestRoundPrice:
    def test_down(self) -> None:
        assert round_price(Decimal("100.009"), Decimal("0.01"), direction="down") == Decimal("100.00")
        assert round_price(Decimal("100.004"), Decimal("0.01"), direction="down") == Decimal("100.00")

    def test_up(self) -> None:
        assert round_price(Decimal("100.001"), Decimal("0.01"), direction="up") == Decimal("100.01")

    def test_nearest(self) -> None:
        assert round_price(Decimal("100.005"), Decimal("0.01"), direction="nearest") == Decimal("100.01")
        assert round_price(Decimal("100.004"), Decimal("0.01"), direction="nearest") == Decimal("100.00")

    def test_zero_tick_rejected(self) -> None:
        with pytest.raises(RuleError, match="tick 必须为正"):
            round_price(Decimal("1"), Decimal("0"))


class TestNormalizeQty:
    def test_limits_order_to_step_and_min(self) -> None:
        rules = make_rules()
        assert normalize_qty(Decimal("0.512349"), rules) == Decimal("0.5123")

    def test_below_min_qty_rejected(self) -> None:
        rules = make_rules()
        with pytest.raises(RuleError, match="低于最小值"):
            normalize_qty(Decimal("0.00001"), rules)

    def test_normalizes_to_zero_rejected(self) -> None:
        rules = make_rules(min_qty=Decimal("0.1"), step_size=Decimal("0.1"))
        with pytest.raises(RuleError, match="低于最小值"):
            normalize_qty(Decimal("0.09"), rules)

    def test_above_max_rejected(self) -> None:
        rules = make_rules(max_qty=Decimal("1"))
        with pytest.raises(RuleError, match="超过最大值"):
            normalize_qty(Decimal("1.5"), rules)

    def test_market_order_uses_market_lot_size(self) -> None:
        rules = make_rules(
            step_size=Decimal("0.0001"),
            min_qty=Decimal("0.0001"),
            market_step_size=Decimal("0.01"),
            market_min_qty=Decimal("0.01"),
        )
        # 按 MARKET_LOT_SIZE 归一化，0.015 → 0.01
        assert normalize_qty(Decimal("0.015"), rules, is_market=True) == Decimal("0.01")
        # 市价单最小量不同：0.005 对 LIMIT 合法但对 MARKET 不足
        assert normalize_qty(Decimal("0.005"), rules, is_market=False) == Decimal("0.005")
        with pytest.raises(RuleError):
            normalize_qty(Decimal("0.005"), rules, is_market=True)


class TestCheckNotional:
    def test_below_min_notional_rejected(self) -> None:
        rules = make_rules(min_notional=Decimal("10"))
        with pytest.raises(RuleError, match="低于最小值"):
            check_notional(Decimal("0.05"), Decimal("100"), rules)

    def test_above_min_notional_passes(self) -> None:
        rules = make_rules(min_notional=Decimal("10"))
        assert check_notional(Decimal("0.2"), Decimal("100"), rules) == Decimal("20")


class TestNormalizePrice:
    def test_price_floors_to_tick(self) -> None:
        rules = make_rules(tick_size=Decimal("0.01"))
        assert normalize_price(Decimal("99.999"), rules) == Decimal("99.99")

    def test_price_rounds_to_non_positive_rejected(self) -> None:
        rules = make_rules(tick_size=Decimal("1"))
        with pytest.raises(RuleError, match="非正"):
            normalize_price(Decimal("0.4"), rules)


class TestParseSpotExchangeInfo:
    PAYLOAD = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "quantityPrecision": 8,
                "pricePrecision": 2,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {"filterType": "LOT_SIZE", "minQty": "0.00001", "maxQty": "1000", "stepSize": "0.00001"},
                    {"filterType": "MARKET_LOT_SIZE", "minQty": "0.0001", "maxQty": "500", "stepSize": "0.0001"},
                    {"filterType": "NOTIONAL", "minNotional": "10"},
                ],
            },
            {
                "symbol": "BROKENUSDT",
                "status": "TRADING",
                "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.1"}],  # 缺 LOT_SIZE
            },
        ]
    }

    def test_parses_rules(self) -> None:
        rules = parse_spot_exchange_info(self.PAYLOAD)
        assert "BTCUSDT" in rules
        assert "BROKENUSDT" not in rules, "缺 LOT_SIZE 的 symbol 必须被跳过"
        r = rules["BTCUSDT"]
        assert r.tick_size == Decimal("0.01")
        assert r.step_size == Decimal("0.00001")
        assert r.market_step_size == Decimal("0.0001")
        assert r.min_notional == Decimal("10")
        assert r.market == "spot"

    def test_missing_tick_raises(self) -> None:
        payload = {
            "symbols": [
                {
                    "symbol": "XUSDT",
                    "filters": [
                        {"filterType": "LOT_SIZE", "minQty": "1", "maxQty": "9", "stepSize": "1"},
                        {"tickSize": "0.1"},  # filterType 错误
                    ],
                }
            ]
        }
        assert parse_spot_exchange_info(payload) == {}, "缺 PRICE_FILTER 应跳过而非崩溃"

    def test_min_notional_defaults_to_zero_when_absent(self) -> None:
        payload = {
            "symbols": [
                {
                    "symbol": "YUSDT",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                        {"filterType": "LOT_SIZE", "minQty": "1", "maxQty": "9", "stepSize": "1"},
                    ],
                }
            ]
        }
        assert parse_spot_exchange_info(payload)["YUSDT"].min_notional == Decimal("0")


class TestParseFuturesExchangeInfo:
    PAYLOAD = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "contractType": "PERPETUAL",
                "baseAsset": "BTC",
                "quantityPrecision": 5,
                "pricePrecision": 2,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                    {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "100", "stepSize": "0.001"},
                    {"filterType": "NOTIONAL", "minNotional": "5"},
                ],
            },
            {
                "symbol": "ETHUSDT",
                "status": "TRADING",
                "contractType": "CURRENT_QUARTER",  # 非永续 → 跳过
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                    {"filterType": "LOT_SIZE", "minQty": "0.01", "maxQty": "100", "stepSize": "0.01"},
                ],
            },
            {
                "symbol": "OLDUSDT",
                "status": "CLOSED",
                "contractType": "PERPETUAL",
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                    {"filterType": "LOT_SIZE", "minQty": "0.01", "maxQty": "100", "stepSize": "0.01"},
                ],
            },
        ]
    }

    def test_only_trading_perpetuals(self) -> None:
        rules = parse_futures_exchange_info(self.PAYLOAD)
        assert set(rules) == {"BTCUSDT"}, "只收录 TRADING + PERPETUAL 合约"
        r = rules["BTCUSDT"]
        assert r.market == "perp"
        assert r.contract_type == "PERPETUAL"
        assert r.min_notional == Decimal("5")
