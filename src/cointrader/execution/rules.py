"""交易规则解析与 Decimal 归一化。

规则：

1. 每个 symbol 启动前从 exchangeInfo 读取并缓存规则（本模块）。
2. 数量/价格**只能**用 ``Decimal`` 归一化，禁止 float 直接发单。
3. 数量一律**向下**对齐 step（多买就是加风险，宁少勿多）；
   价格向下对齐 tick（BUY 时买价低于 tick 上界，SELL 时对称处理由调用方决定方向）。
4. 归一化后如果名义额低于最小值、数量变为零，本地直接拒绝。

Spot 与 Futures 的精度规则不同（Spot 有 MARKET_LOT_SIZE，
Futures 有 quantityPrecision/pricePrecision），分开解析。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal
from typing import Any


class RuleError(Exception):
    """规则缺失或归一化后订单非法（数量为零、低于最小名义额等）。"""


@dataclass(frozen=True, slots=True)
class SymbolRules:
    """单个 symbol 的交易规则（全部 Decimal）。"""

    symbol: str
    market: str  # "spot" | "perp"
    status: str
    base_asset: str
    quote_asset: str
    tick_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    step_size: Decimal
    min_notional: Decimal
    # 市价单独立数量规则（Spot MARKET_LOT_SIZE；Futures 无则同 LOT_SIZE）
    market_step_size: Decimal | None = None
    market_min_qty: Decimal | None = None
    market_max_qty: Decimal | None = None
    quantity_precision: int | None = None
    price_precision: int | None = None
    contract_type: str | None = None

    def qty_step(self, *, is_market: bool) -> Decimal:
        if is_market and self.market_step_size is not None:
            return self.market_step_size
        return self.step_size

    def min_qty_for(self, *, is_market: bool) -> Decimal:
        if is_market and self.market_min_qty is not None:
            return self.market_min_qty
        return self.min_qty

    def max_qty_for(self, *, is_market: bool) -> Decimal:
        if is_market and self.market_max_qty is not None:
            return self.market_max_qty
        return self.max_qty


# ---------------------------------------------------------------------------
# Decimal 工具
# ---------------------------------------------------------------------------


def _dec(value: Any, name: str) -> Decimal:
    """把 exchangeInfo 里的字符串数值转成 Decimal（拒绝科学计数法之外的异常）。"""
    if value is None:
        raise RuleError(f"规则字段 {name} 缺失")
    try:
        return Decimal(str(value))
    except Exception as exc:  # noqa: BLE001
        raise RuleError(f"规则字段 {name} 非法: {value!r}") from exc


def _dec_or(value: Any, name: str, default: Decimal = Decimal("0")) -> Decimal:
    return default if value is None else _dec(value, name)


def format_decimal(value: Decimal | int | float | str) -> str:
    """Decimal → 交易所参数字符串。

    规则：十进制、无科学计数法、去掉多余的尾零（"0.0010" → "0.001"，
    "0.000" → "0"）。绝不允许出现 "e"。
    """
    d = value if isinstance(value, Decimal) else Decimal(str(value))
    text = format(d.normalize(), "f")
    # Decimal.normalize() 对 0 会得到 "0"，对 100 会得到 "1E+2" → format "f" 已处理
    if text == "-0":
        text = "0"
    return text


def floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    """向下对齐到 step 的整数倍。"""
    if step <= 0:
        raise RuleError(f"step 必须为正，当前 {step}")
    if value < 0:
        raise RuleError(f"数量不能为负: {value}")
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def round_price(value: Decimal, tick: Decimal, *, direction: str = "down") -> Decimal:
    """价格对齐到 tick。direction: down/up/nearest。"""
    if tick <= 0:
        raise RuleError(f"tick 必须为正，当前 {tick}")
    rounding = {"down": ROUND_DOWN, "up": ROUND_UP, "nearest": ROUND_HALF_UP}[direction]
    return (value / tick).to_integral_value(rounding=rounding) * tick


def normalize_price(value: Decimal, rules: SymbolRules, *, direction: str = "down") -> Decimal:
    price = round_price(value, rules.tick_size, direction=direction)
    if price <= 0:
        raise RuleError(f"价格归一化后非正: {value} (tick={rules.tick_size})")
    return price


def normalize_qty(value: Decimal, rules: SymbolRules, *, is_market: bool = False) -> Decimal:
    """数量按 LOT_SIZE（或 MARKET_LOT_SIZE）向下对齐，并检查最小/最大。"""
    step = rules.qty_step(is_market=is_market)
    qty = floor_to_step(value, step)
    min_qty = rules.min_qty_for(is_market=is_market)
    max_qty = rules.max_qty_for(is_market=is_market)
    if qty < min_qty:
        raise RuleError(
            f"数量 {value} 归一化后 {format_decimal(qty)} 低于最小值 {format_decimal(min_qty)}"
        )
    if qty > max_qty:
        raise RuleError(f"数量 {format_decimal(qty)} 超过最大值 {format_decimal(max_qty)}")
    return qty


def check_notional(qty: Decimal, price: Decimal, rules: SymbolRules) -> Decimal:
    """检查名义额。低于最小名义额直接拒绝（本地拒单，不发请求）。"""
    notional = qty * price
    if notional < rules.min_notional:
        raise RuleError(
            f"名义额 {format_decimal(notional)} 低于最小值 {format_decimal(rules.min_notional)}"
        )
    return notional


# ---------------------------------------------------------------------------
# exchangeInfo 解析
# ---------------------------------------------------------------------------


def _filter_by_type(filters: list[dict[str, Any]], ftype: str) -> dict[str, Any] | None:
    for f in filters:
        if f.get("filterType") == ftype:
            return f
    return None


def parse_spot_exchange_info(payload: dict[str, Any]) -> dict[str, SymbolRules]:
    """解析 Spot /api/v3/exchangeInfo，返回 {symbol: SymbolRules}。"""
    out: dict[str, SymbolRules] = {}
    for sym in payload.get("symbols", []):
        if not isinstance(sym, dict):
            continue
        symbol = sym.get("symbol")
        filters = sym.get("filters", []) or []
        price_f = _filter_by_type(filters, "PRICE_FILTER") or {}
        lot_f = _filter_by_type(filters, "LOT_SIZE") or {}
        market_lot_f = _filter_by_type(filters, "MARKET_LOT_SIZE")
        notional_f = _filter_by_type(filters, "NOTIONAL") or _filter_by_type(filters, "MIN_NOTIONAL") or {}

        if not symbol or not price_f or not lot_f:
            continue

        out[symbol] = SymbolRules(
            symbol=symbol,
            market="spot",
            status=str(sym.get("status", "")),
            base_asset=str(sym.get("baseAsset", "")),
            quote_asset=str(sym.get("quoteAsset", "")),
            tick_size=_dec(price_f.get("tickSize"), f"{symbol}.tickSize"),
            min_qty=_dec(lot_f.get("minQty"), f"{symbol}.minQty"),
            max_qty=_dec(lot_f.get("maxQty"), f"{symbol}.maxQty"),
            step_size=_dec(lot_f.get("stepSize"), f"{symbol}.stepSize"),
            min_notional=_dec_or(
                notional_f.get("minNotional") or notional_f.get("minNotional"),
                f"{symbol}.minNotional",
            ),
            market_step_size=_dec(market_lot_f["stepSize"], f"{symbol}.marketStepSize")
            if market_lot_f is not None and market_lot_f.get("stepSize") is not None
            else None,
            market_min_qty=_dec(market_lot_f.get("minQty"), f"{symbol}.marketMinQty")
            if market_lot_f and market_lot_f.get("minQty") is not None
            else None,
            market_max_qty=_dec(market_lot_f.get("maxQty"), f"{symbol}.marketMaxQty")
            if market_lot_f and market_lot_f.get("maxQty") is not None
            else None,
            quantity_precision=sym.get("quantityPrecision"),
            price_precision=sym.get("pricePrecision"),
        )
    return out


def parse_futures_exchange_info(payload: dict[str, Any]) -> dict[str, SymbolRules]:
    """解析 USDⓈ-M Futures /fapi/v1/exchangeInfo。

    只收录 **TRADING** 状态且为 **PERPETUAL** 的合约（本策略只用永续）。
    """
    out: dict[str, SymbolRules] = {}
    for sym in payload.get("symbols", []):
        if not isinstance(sym, dict):
            continue
        if sym.get("status") != "TRADING":
            continue
        if sym.get("contractType") != "PERPETUAL":
            continue

        symbol = sym.get("symbol")
        filters = sym.get("filters", []) or []
        price_f = _filter_by_type(filters, "PRICE_FILTER") or {}
        lot_f = _filter_by_type(filters, "LOT_SIZE") or {}
        # 期货端最小名义额过滤器是 MIN_NOTIONAL（字段名 notional），
        # 与现货端的 NOTIONAL/minNotional 不同；两者都兼容。
        notional_f = (
            _filter_by_type(filters, "MIN_NOTIONAL")
            or _filter_by_type(filters, "NOTIONAL")
            or {}
        )
        notional_raw = notional_f.get("notional") or notional_f.get("minNotional")

        if not symbol or not price_f or not lot_f:
            continue

        out[symbol] = SymbolRules(
            symbol=symbol,
            market="perp",
            status=str(sym.get("status", "")),
            base_asset=str(sym.get("baseAsset", "")),
            quote_asset=str(sym.get("quoteAsset", "")),
            tick_size=_dec(price_f.get("tickSize"), f"{symbol}.tickSize"),
            min_qty=_dec(lot_f.get("minQty"), f"{symbol}.minQty"),
            max_qty=_dec(lot_f.get("maxQty"), f"{symbol}.maxQty"),
            step_size=_dec(lot_f.get("stepSize"), f"{symbol}.stepSize"),
            min_notional=_dec_or(notional_raw, f"{symbol}.minNotional"),
            quantity_precision=sym.get("quantityPrecision"),
            price_precision=sym.get("pricePrecision"),
            contract_type=sym.get("contractType"),
        )
    return out


__all__ = [
    "RuleError",
    "SymbolRules",
    "check_notional",
    "floor_to_step",
    "format_decimal",
    "normalize_price",
    "normalize_qty",
    "parse_futures_exchange_info",
    "parse_spot_exchange_info",
    "round_price",
]
