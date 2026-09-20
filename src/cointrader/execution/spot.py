"""Spot 私有 REST 适配器（开发设计文档 §3.2/§4.1）。

职责边界：

1. 只做**接口映射 + 响应解析**，不含策略/风控/状态机逻辑。
2. 下单走 ``SignedClient.place_order`` —— 永不重试，
   结果未知时抛 ``UnknownSubmission``（恢复由 pair_executor 负责）。
3. 数量/价格参数一律经 ``format_decimal``，绝不直接发 float。
4. 禁止调用提现、转账、借贷等接口 —— 本模块根本不提供这些方法。
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, cast

from .guard import Market, OrderSide, OrderType
from .models import Order, OrderRequest
from .order_state import map_exchange_status
from .rules import SymbolRules, format_decimal
from .transport import SignedClient

logger = logging.getLogger(__name__)

__all__ = ["SpotAdapter"]


def _parse_order(payload: dict[str, Any], req: OrderRequest) -> Order:
    executed = Decimal(str(payload.get("executedQty") or "0"))
    quote = payload.get("cummulativeQuoteQty") or payload.get("cumQuote")
    avg_price: Decimal | None = None
    if quote and executed > 0:
        avg_price = Decimal(str(quote)) / executed
    elif req.price is not None:
        avg_price = req.price
    state = map_exchange_status(payload.get("status"))
    return Order(
        client_order_id=str(payload.get("clientOrderId") or req.client_order_id),
        exchange_order_id=str(payload["orderId"]) if payload.get("orderId") is not None else None,
        symbol=str(payload.get("symbol") or req.symbol),
        market=Market(req.market.value),
        side=req.side,
        order_type=req.order_type,
        quantity=Decimal(str(payload.get("origQty") or format_decimal(req.quantity))),
        executed_qty=executed,
        avg_price=avg_price,
        price=req.price,
        reduce_only=bool(req.reduce_only),
        state=state.value,
        updated_ms=int(payload.get("time") or payload.get("updateTime") or time.time() * 1000),
        raw=payload,
    )


class SpotAdapter:
    """Spot 私有 API。初版只实现策略必需的接口（文档 §3.2 表）。"""

    MARKET = Market.SPOT

    def __init__(self, client: SignedClient, *, now_fn: Any = time.time) -> None:
        self.client = client
        self._now = now_fn
        self.rules: dict[str, SymbolRules] = {}

    # -- 时间与规则 ----------------------------------------------------------

    def server_time_ms(self) -> int:
        payload = self.client.get("/api/v3/time", sign=False)
        return int(payload["serverTime"])

    def calibrate(self, samples: int = 5) -> int:
        """校准 server time 偏移（多点采样，取 RTT 最小样本的中点估计）。

        代理链路 RTT 抖动会让单次采样偏移漂移数百 ms；取 RTT 最小样本
        误差下界最接近真实偏移。返回 offset（毫秒）。
        """
        best_rtt: int | None = None
        best_offset = 0
        for _ in range(max(1, samples)):
            t0 = int(self._now() * 1000)
            server = self.server_time_ms()
            t1 = int(self._now() * 1000)
            rtt = max(t1 - t0, 0)
            offset = server - (t0 + rtt // 2)
            if best_rtt is None or rtt < best_rtt:
                best_rtt, best_offset = rtt, offset
        self.client.time_offset_ms = best_offset
        logger.info("Spot server time 偏移: %d ms（%d 次采样，最佳 RTT %d ms）",
                    best_offset, max(1, samples), best_rtt or 0)
        return best_offset

    def load_rules(self) -> dict[str, SymbolRules]:
        from .rules import parse_spot_exchange_info

        payload = self.client.get("/api/v3/exchangeInfo", sign=False)
        self.rules = parse_spot_exchange_info(payload)
        return self.rules

    def rule(self, symbol: str) -> SymbolRules:
        if symbol not in self.rules:
            raise KeyError(f"未加载 {symbol} 的 Spot 规则（先调用 load_rules）")
        return self.rules[symbol]

    # -- 只读快照 ------------------------------------------------------------

    def account(self) -> dict[str, Any]:
        return cast("dict[str, Any]", self.client.get("/api/v3/account"))

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {"symbol": symbol} if symbol else None
        return self.client.get("/api/v3/openOrders", params) or []

    def get_order(self, symbol: str, *, client_order_id: str | None = None, order_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": symbol}
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        elif order_id:
            params["orderId"] = order_id
        else:
            raise ValueError("client_order_id 或 order_id 必须提供一个")
        return cast("dict[str, Any]", self.client.get("/api/v3/order", params))

    def all_orders(self, symbol: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return self.client.get("/api/v3/allOrders", {"symbol": symbol, "limit": limit}) or []

    def my_trades(
        self,
        symbol: str,
        *,
        from_id: int | None = None,
        start_ms: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """我的成交（分页参数以 Binance 官方文档为准：fromId/startTime/limit）。"""
        params: dict[str, Any] = {"symbol": symbol, "limit": int(limit)}
        if from_id is not None:
            params["fromId"] = int(from_id)
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        return self.client.get("/api/v3/myTrades", params) or []

    def trade_fee(self, symbol: str | None = None) -> Any:
        params = {"symbol": symbol} if symbol else None
        return self.client.get("/sapi/v1/asset/tradeFee", params)

    def balances(self) -> dict[str, Decimal]:
        """{asset: totalQty}。"""
        payload = self.account()
        return {
            a["asset"]: Decimal(str(a["free"])) + Decimal(str(a["locked"]))
            for a in payload.get("balances", [])
        }

    def available(self) -> dict[str, Decimal]:
        """{asset: free}。"""
        payload = self.account()
        return {a["asset"]: Decimal(str(a["free"])) for a in payload.get("balances", [])}

    # -- 交易 ----------------------------------------------------------------

    def place_order(self, req: OrderRequest) -> Order:
        """提交订单。永不重试。结果未知时抛 UnknownSubmission。"""
        params: dict[str, Any] = {
            "symbol": req.symbol,
            "side": req.side.value,
            "type": req.order_type.value,
            "quantity": format_decimal(req.quantity),
            "newClientOrderId": req.client_order_id,
        }
        if req.order_type is OrderType.LIMIT:
            if req.price is None:
                raise ValueError("LIMIT 单必须提供 price")
            params["price"] = format_decimal(req.price)
            params["timeInForce"] = req.time_in_force or "GTC"
        payload = self.client.place_order("/api/v3/order", params, client_order_id=req.client_order_id)
        return _parse_order(payload, req)

    def cancel_order(
        self,
        symbol: str,
        *,
        client_order_id: str | None = None,
        order_id: str | None = None,
    ) -> Any:
        params: dict[str, Any] = {"symbol": symbol}
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        elif order_id:
            params["orderId"] = order_id
        else:
            raise ValueError("client_order_id 或 order_id 必须提供一个")
        return self.client.cancel_order("/api/v3/order", params, client_order_id=client_order_id)

    def parse_order(self, payload: dict[str, Any], req: OrderRequest) -> Order:
        return _parse_order(payload, req)

    # -- 用户数据流（listenKey 管理，WebSocket 连接在 user_stream.py）--------

    def create_listen_key(self) -> str:
        payload = self.client.post("/api/v3/userDataStream")
        return str(payload["listenKey"])

    def keepalive_listen_key(self, listen_key: str) -> Any:
        return self.client.put("/api/v3/userDataStream", {"listenKey": listen_key})

    def close_listen_key(self, listen_key: str) -> Any:
        return self.client.delete("/api/v3/userDataStream", {"listenKey": listen_key})

    # -- 查询恢复 ------------------------------------------------------------

    def query_by_client_order_id(self, symbol: str, client_order_id: str) -> Order | None:
        """UNKNOWN_SUBMISSION 恢复的唯一入口：按 clientOrderId 查实际状态。

        订单不存在时返回 None（交易所明确未接单，可重新规划，不能盲目重下）。
        """
        from ..errors import BinanceError

        try:
            payload = self.get_order(symbol, client_order_id=client_order_id)
        except BinanceError as exc:
            if exc.code in (-2013, -2011):  # 订单不存在 / 撤单对象不存在
                return None
            raise
        return _parse_order(payload, self._rebuild_request(symbol, client_order_id))

    def _rebuild_request(self, symbol: str, client_order_id: str) -> OrderRequest:
        """查询恢复时重建最小 OrderRequest（响应解析用，不用于提交）。"""
        return OrderRequest(
            client_order_id=client_order_id,
            symbol=symbol,
            side=OrderSide.BUY,
            market=self.MARKET,
            order_type=OrderType.MARKET,
            quantity=Decimal("0"),
        )
