"""K 线与交易规则。

包含两件事：

1. **K 线规范化** —— 把币安返回的 12 元素数组转成带正确 dtype 的 DataFrame。
   币安返回的价格/数量都是**字符串**，直接参与计算会得到静默的错误结果
   （Python 里 ``"1.5" * 2 == "1.51.5"``，在 pandas 里则是 dtype=object 的灾难）。

2. **交易规则解析** —— ``stepSize`` / ``tickSize`` / ``minNotional``。
   这些决定了回测里的"理想仓位"能不能真的在交易所下出来。
   忽略它们会让回测跑得通、实盘跑不通 —— 这是最经典的一类失败。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..errors import DataUnavailableError, ParseError
from .binance import BinancePublicClient

logger = logging.getLogger(__name__)

# 币安 K 线数组的字段顺序（现货与合约一致，共 12 个字段）
KLINE_COLUMNS = [
    "open_time",       # 0
    "open",            # 1
    "high",            # 2
    "low",             # 3
    "close",           # 4
    "volume",          # 5
    "close_time",      # 6
    "quote_volume",    # 7
    "trades",          # 8
    "taker_buy_base",  # 9
    "taker_buy_quote", # 10
    "ignore",          # 11
]

_NUMERIC_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_base", "taker_buy_quote",
]

#: 数值列 → 在原始数组中的下标（与 KLINE_COLUMNS 对齐，写死避免运行时查表）
_NUMERIC_INDEX: dict[str, int] = {
    "open": 1,
    "high": 2,
    "low": 3,
    "close": 4,
    "volume": 5,
    "quote_volume": 7,
    "taker_buy_base": 9,
    "taker_buy_quote": 10,
}

#: 常被使用的 K 线周期 → 毫秒数。回测对齐资金费结算时间时要用。
INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def klines_to_frame(raw: list[list[Any]], symbol: str) -> pd.DataFrame:
    """把币安原始 K 线数组转成规范化 DataFrame。

    Args:
        raw: 币安返回的二维数组列表。
        symbol: 合约符号（仅用于报错信息）。

    Returns:
        DataFrame，``open_time`` 为 UTC datetime 索引，
        价格/数量列均为 float64。

    Raises:
        InsufficientDataError: 输入为空。
        ParseError: 元素长度或字段格式异常。
    """
    if not raw:
        raise ParseError(f"{symbol} K 线数据为空")

    rows: list[dict[str, Any]] = []
    for index, candle in enumerate(raw):
        if not isinstance(candle, (list, tuple)) or len(candle) < 12:
            raise ParseError(
                f"{symbol} 第 {index} 根 K 线字段数异常（期望 >=12，实际 "
                f"{len(candle) if hasattr(candle, '__len__') else '?'}）: {candle!r}"
            )
        try:
            row: dict[str, Any] = {
                "open_time": int(candle[0]),
                "close_time": int(candle[6]),
                "trades": int(candle[8]),
            }
            for column, index in _NUMERIC_INDEX.items():
                row[column] = float(candle[index])
            rows.append(row)
        except (TypeError, ValueError) as exc:
            raise ParseError(f"{symbol} 第 {index} 根 K 线数值解析失败: {candle!r} ({exc})") from exc

    frame = pd.DataFrame(rows)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame = frame.drop_duplicates(subset="open_time", keep="last")
    frame = frame.sort_values("open_time").set_index("open_time")

    return frame[
        ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "trades"]
    ]


def fetch_spot_klines(
    client: BinancePublicClient,
    symbol: str,
    interval: str = "8h",
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> pd.DataFrame:
    """拉取现货 K 线并规范化。"""
    raw = client.spot_klines(symbol, interval, start_ms=start_ms, end_ms=end_ms)
    return klines_to_frame(raw, symbol)


def fetch_futures_klines(
    client: BinancePublicClient,
    symbol: str,
    interval: str = "8h",
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> pd.DataFrame:
    """拉取永续 K 线并规范化。"""
    raw = client.futures_klines(symbol, interval, start_ms=start_ms, end_ms=end_ms)
    return klines_to_frame(raw, symbol)


def historical_quote_volume_24h(frame: pd.DataFrame) -> pd.Series:
    """由已收盘的 4h K 线计算历史 24h 成交额。

    返回索引是 K 线结束时刻；因此在事件时点 ``t`` 使用 ``asof(t)``，
    不会把尚未收盘的 K 线成交量带入历史信号。
    """
    if "quote_volume" not in frame:
        raise ValueError("K 线缺少 quote_volume 列")
    if frame.empty:
        return pd.Series(dtype=float, name="quote_volume_24h")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("K 线必须按时间升序")

    volumes = frame["quote_volume"].astype(float).sort_index()
    close_times = volumes.index + pd.Timedelta(hours=4)
    result = volumes.rolling(window=6, min_periods=6).sum()
    result.index = close_times
    result.name = "quote_volume_24h"
    return result.dropna()


def fetch_historical_quote_volume_24h(
    client: BinancePublicClient,
    symbol: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> pd.Series:
    """拉取 4h K 线并返回因果历史 24h 成交额序列。"""
    frame = fetch_futures_klines(
        client,
        symbol,
        interval="4h",
        start_ms=start_ms,
        end_ms=end_ms,
    )
    return historical_quote_volume_24h(frame)



def historical_quote_volume_3d_avg(frame: pd.DataFrame) -> pd.Series:
    """由已收盘的 4h K 线计算最近 3 天平均日成交额。

    3 天窗口包含 18 根 4h K 线；返回窗口总额除以 3，索引为最后一根
    K 线的结束时刻。只使用已收盘 K 线，事件时点通过 ``asof`` 取值。
    """
    if "quote_volume" not in frame:
        raise ValueError("K 线缺少 quote_volume 列")
    if frame.empty:
        return pd.Series(dtype=float, name="quote_volume_3d_avg")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("K 线必须按时间升序")

    volumes = frame["quote_volume"].astype(float).sort_index()
    close_times = volumes.index + pd.Timedelta(hours=4)
    result = volumes.rolling(window=18, min_periods=18).sum() / 3.0
    result.index = close_times
    result.name = "quote_volume_3d_avg"
    return result.dropna()


def fetch_historical_quote_volume_3d_avg(
    client: BinancePublicClient,
    symbol: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> pd.Series:
    """拉取 4h K 线并返回因果的 3 天平均日成交额序列。"""
    frame = fetch_futures_klines(
        client,
        symbol,
        interval="4h",
        start_ms=start_ms,
        end_ms=end_ms,
    )
    return historical_quote_volume_3d_avg(frame)





@dataclass(frozen=True, slots=True)
class SymbolRules:
    """一个交易对的交易规则。回测用它把理想仓位变成可下单的仓位。"""

    symbol: str
    step_size: float
    min_qty: float
    max_qty: float
    tick_size: float
    min_notional: float
    quantity_precision: int
    price_precision: int

    def round_qty(self, qty: float, *, mode: str = "down") -> float:
        """把数量对齐到 ``stepSize`` 的整数倍。

        Args:
            qty: 期望数量。
            mode: ``down`` 向下取整（默认，保守 —— 宁可少买不可多买）、
                  ``up`` 向上取整、``nearest`` 就近。

        Returns:
            对齐后的数量。小于 ``minQty`` 时返回 0.0（表示无法下单）。
        """
        if self.step_size <= 0:
            return 0.0
        steps = qty / self.step_size
        if mode == "down":
            aligned = math.floor(steps)
        elif mode == "up":
            aligned = math.ceil(steps)
        elif mode == "nearest":
            aligned = round(steps)
        else:
            raise ValueError(f"未知的取整模式: {mode}")

        result = aligned * self.step_size
        # 浮点乘除会引入误差（0.1 * 3 = 0.30000000000000004），
        # 按精度截断，否则交易所会以"精度不符"拒单
        result = round(result, self.quantity_precision)

        if result < self.min_qty:
            return 0.0
        if result > self.max_qty:
            result = self.round_qty(self.max_qty, mode="down")
        return result

    def round_price(self, price: float, *, mode: str = "down") -> float:
        """把价格对齐到 ``tickSize``。"""
        if self.tick_size <= 0:
            return price
        steps = price / self.tick_size
        if mode == "down":
            aligned = math.floor(steps)
        elif mode == "up":
            aligned = math.ceil(steps)
        else:
            aligned = round(steps)
        return round(aligned * self.tick_size, self.price_precision)

    def is_tradeable(self, qty: float, price: float) -> tuple[bool, str]:
        """判断某个 (数量, 价格) 组合能否真的下单。

        Returns:
            ``(是否可下单, 原因)``。不可下单时原因是人类可读的说明。
        """
        aligned_qty = self.round_qty(qty)
        if aligned_qty <= 0:
            return False, f"数量 {qty} 对齐 stepSize={self.step_size} 后低于 minQty={self.min_qty}"
        notional = aligned_qty * price
        if notional < self.min_notional:
            return False, (
                f"名义额 {notional:.4f} USDT 低于 minNotional={self.min_notional} USDT"
            )
        return True, "可下单"


def _extract_filter(filters: list[dict[str, Any]], filter_type: str) -> dict[str, Any]:
    for entry in filters:
        if entry.get("filterType") == filter_type:
            return entry
    return {}


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    # 币安用 0 表示"无限制"，对 min/max 语义不同，交由调用方处理
    return result


def _precision_from_step(step: Any) -> int:
    """由 stepSize/tickSize 推断小数位数。

    ``"0.00001000"`` → 5；``"1.00000000"`` → 0。

    ⚠️ **必须传原始字符串，不能传 float。**

    币安返回的是固定宽度的字符串（``"0.00001000"``），直接数小数位即可。
    但如果先 ``float()`` 再 ``str()``，Python 会给出科学计数法
    （``str(0.00001) == '1e-05'``），小数位推断立刻失效并返回 0，
    进而使 ``round(qty, 0)`` 把所有数量归零 —— 每一笔订单都会被判为
    "低于最小数量"而无法下单。

    为防御这个陷阱，本函数同时接受 float，走 ``Decimal`` 路径精确还原。
    """
    text = str(step)

    # 常规路径：十进制字符串，直接数小数位
    if "." in text and "e" not in text.lower():
        return len(text.split(".", 1)[1].rstrip("0"))

    # 防御路径：科学计数法或整数（可能来自 float 转换）
    try:
        from decimal import Decimal

        decimal = Decimal(text)
    except Exception:  # noqa: BLE001 - 无法解析时保守返回 0
        return 0

    exponent = decimal.as_tuple().exponent
    if not isinstance(exponent, int) or exponent >= 0:
        return 0
    return -exponent


def _precision_from_field_or_step(entry: dict[str, Any], field: str, step: Any) -> int:
    """从显式精度字段取值；字段缺失时**由 stepSize 推断**。

    ⚠️ 现货 ``exchangeInfo`` **没有** ``quantityPrecision`` 字段
    （实测确认：返回的字段里只有 ``baseAssetPrecision`` /
    ``quotePrecision`` / ``baseCommissionPrecision`` 等）。
    合约 ``exchangeInfo`` 才有 ``quantityPrecision`` / ``pricePrecision``。

    因此逻辑顺序必须是「先看字段，再回退到 stepSize 推断」，
    而不是 ``int(entry.get(field, 推断值) or 0)`` ——
    后者在字段缺失时会把推断值丢掉，取到 0，
    导致 ``round(qty, 0)`` 把合法数量归零、所有订单被判为不可下单。
    """
    if field in entry and entry[field] is not None:
        try:
            return int(entry[field])
        except (TypeError, ValueError):
            pass
    return _precision_from_step(step)


def parse_symbol_rules(payload: dict[str, Any], symbol: str, *, market: str = "spot") -> SymbolRules:
    """从 ``exchangeInfo`` 响应解析单交易对规则。

    Args:
        payload: ``exchangeInfo`` 的原始 JSON。
        symbol: 目标交易对。
        market: ``spot`` 或 ``futures``（两者的过滤器字段名略有差异）。

    Returns:
        SymbolRules。

    Raises:
        DataUnavailableError: 交易对不存在或其状态不可交易。
    """
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        raise ParseError("exchangeInfo 缺少 symbols 字段")

    entry = next((s for s in symbols if s.get("symbol") == symbol), None)
    if entry is None:
        raise DataUnavailableError(f"{market} 市场不存在交易对 {symbol}")

    status = entry.get("status") or entry.get("contractStatus")
    if status != "TRADING":
        raise DataUnavailableError(f"{symbol} 当前状态为 {status}，不可交易")

    filters = entry.get("filters", [])
    price_filter = _extract_filter(filters, "PRICE_FILTER")
    lot_filter = _extract_filter(filters, "LOT_SIZE")
    notional_filter = _extract_filter(filters, "NOTIONAL")

    tick_size = _safe_float(price_filter.get("tickSize"), 0.0)
    step_size = _safe_float(lot_filter.get("stepSize"), 0.0)
    min_qty = _safe_float(lot_filter.get("minQty"), 0.0)
    max_qty = _safe_float(lot_filter.get("maxQty"), float("inf"))

    # 合约的 NOTIONAL 过滤器用 "minNotional"；现货同样是 "minNotional"。
    # 若缺失，现货按币安默认 5 USDT 兜底；合约默认 5 USDT。
    min_notional = _safe_float(notional_filter.get("minNotional"), 5.0)
    if min_notional <= 0:
        min_notional = 5.0

    return SymbolRules(
        symbol=symbol,
        step_size=step_size,
        min_qty=min_qty,
        max_qty=max_qty,
        tick_size=tick_size,
        min_notional=min_notional,
        quantity_precision=_precision_from_field_or_step(entry, "quantityPrecision", step_size),
        price_precision=_precision_from_field_or_step(entry, "pricePrecision", tick_size),
    )


def tradable_perpetuals(
    futures_info: dict[str, Any],
    *,
    quote_asset: str = "USDT",
    exclude_bases: tuple[str, ...] = (),
) -> list[str]:
    """从 ``exchangeInfo`` 筛出可交易的 USDT 本位永续合约。

    过滤条件：
    - ``contractType == PERPETUAL``（排除季度交割合约 —— 它们会到期，不适合长期持有）
    - ``status == TRADING``
    - 报价币为 ``quote_asset``
    - 基础币不在排除列表（稳定币对无意义）

    Returns:
        合约符号列表，已排序。
    """
    excluded = {base.upper() for base in exclude_bases}
    result: list[str] = []

    for entry in futures_info.get("symbols", []):
        if entry.get("contractType") != "PERPETUAL":
            continue
        if entry.get("status") != "TRADING":
            continue
        if entry.get("quoteAsset") != quote_asset:
            continue
        base = str(entry.get("baseAsset", "")).upper()
        if base in excluded:
            continue
        result.append(str(entry["symbol"]))

    return sorted(result)


__all__ = [
    "INTERVAL_MS",
    "KLINE_COLUMNS",
    "SymbolRules",
    "fetch_futures_klines",
    "fetch_historical_quote_volume_24h",
    "fetch_historical_quote_volume_3d_avg",
    "fetch_spot_klines",
    "historical_quote_volume_24h",
    "historical_quote_volume_3d_avg",
    "klines_to_frame",
    "parse_symbol_rules",
    "tradable_perpetuals",
]
