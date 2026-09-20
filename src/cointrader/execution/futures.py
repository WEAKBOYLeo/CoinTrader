"""USDⓈ-M Futures 私有 REST 适配器（开发设计文档 §3.2）。

与 SpotAdapter 的差异：

1. 下单固定**单向持仓模式** ``positionSide=BOTH``（文档 §3.2 初版约束）。
   多仓位模式（LONG/SHORT）在代码层面被禁止 —— 本策略用空单开仓 +
   reduceOnly 买入平仓，不需要。
2. 平仓单必须带 ``reduceOnly=true`` 或明确的平仓语义，并校验不会反向开仓。
3. 持仓用 ``GET /fapi/v3/positionRisk``；账户用 ``GET /fapi/v2/account``。
4. 提供 ``countdown_cancel_all`` —— 订单自动过期的保护网。
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, cast

from ..errors import BinanceError
from .guard import Market, OrderSide, OrderType
from .models import Order, OrderRequest
from .rules import SymbolRules, format_decimal
from .spot import _parse_order
from .transport import SignedClient

logger = logging.getLogger(__name__)

__all__ = ["FuturesAdapter"]


class FuturesAdapter:
    """USDⓈ-M Futures 私有 API。"""

    MARKET = Market.PERP

    def __init__(self, client: SignedClient, *, now_fn: Any = time.time) -> None:
        self.client = client
        self._now = now_fn
        self.rules: dict[str, SymbolRules] = {}

    # -- 时间与规则 ----------------------------------------------------------

    def server_time_ms(self) -> int:
        payload = self.client.get("/fapi/v1/time", sign=False)
        return int(payload["serverTime"])

    def calibrate(self, samples: int = 5) -> int:
        """校准 server time 偏移（多点采样，取 RTT 最小样本的中点估计）。返回 offset（毫秒）。"""
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
        logger.info("Futures server time 偏移: %d ms（%d 次采样，最佳 RTT %d ms）",
                    best_offset, max(1, samples), best_rtt or 0)
        return best_offset

    def load_rules(self) -> dict[str, SymbolRules]:
        from .rules import parse_futures_exchange_info

        payload = self.client.get("/fapi/v1/exchangeInfo", sign=False)
        self.rules = parse_futures_exchange_info(payload)
        return self.rules

    def rule(self, symbol: str) -> SymbolRules:
        if symbol not in self.rules:
            raise KeyError(f"未加载 {symbol} 的 Futures 规则（先调用 load_rules）")
        return self.rules[symbol]

    # -- 只读快照 ------------------------------------------------------------

    def account(self) -> dict[str, Any]:
        return cast("dict[str, Any]", self.client.get("/fapi/v2/account"))

    def positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {"symbol": symbol} if symbol else None
        return self.client.get("/fapi/v3/positionRisk", params) or []

    def position_qty(self, symbol: str) -> Decimal:
        """该 symbol 的净持仓数量（负数 = 空头）。无持仓返回 0。"""
        for pos in self.positions(symbol):
            if pos.get("symbol") == symbol and str(pos.get("positionSide") or "BOTH") == "BOTH":
                return Decimal(str(pos.get("positionAmt") or "0"))
        return Decimal("0")

    def leverage_and_margin(self, symbol: str) -> tuple[int, str]:
        """读取（不修改）当前杠杆与保证金模式。启动预检用。"""
        payload = self.client.get("/fapi/v2/leverage", {"symbol": symbol})
        leverage = int(payload.get("leverage", 0)) if isinstance(payload, dict) else 0
        margin = str(payload.get("marginType", "")) if isinstance(payload, dict) else ""
        return leverage, margin

    def ensure_leverage_and_margin(self, symbol: str, leverage: int, margin_type: str) -> None:
        """显式设置杠杆与保证金模式。

        Demo trading 不暴露 GET /fapi/v2/leverage（-5000），无法读回账户设置。
        启动时（无持仓前提下）直接写成配置期望值。仅 testnet/demo 使用。
        """
        self.client.post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})
        # 文档标准值为大写（demo 对小写直接 -1102）；-4046 = 当前已是该模式，幂等成功
        try:
            self.client.post("/fapi/v1/marginType", {"symbol": symbol, "margintype": margin_type.upper()})
        except BinanceError as exc:
            if exc.code != -4046:
                raise
            logger.debug("marginType 无需变更（-4046）: %s", symbol)

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {"symbol": symbol} if symbol else None
        return self.client.get("/fapi/v1/openOrders", params) or []

    def get_order(self, symbol: str, *, client_order_id: str | None = None, order_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": symbol}
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        elif order_id:
            params["orderId"] = order_id
        else:
            raise ValueError("client_order_id 或 order_id 必须提供一个")
        return cast("dict[str, Any]", self.client.get("/fapi/v1/order", params))

    def all_orders(self, symbol: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return self.client.get("/fapi/v1/allOrders", {"symbol": symbol, "limit": limit}) or []

    def user_trades(
        self,
        symbol: str,
        *,
        from_id: int | None = None,
        start_ms: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """合约成交（分页参数以 Binance 官方文档为准：fromId/startTime/limit）。"""
        params: dict[str, Any] = {"symbol": symbol, "limit": int(limit)}
        if from_id is not None:
            params["fromId"] = int(from_id)
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        return self.client.get("/fapi/v1/userTrades", params) or []

    def income_history(
        self,
        *,
        income_type: str = "FUNDING_FEE",
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """实际 income（GET /fapi/v1/income，USER_DATA 只读）。

        字段契约（A-01，官方文档核对）：``id``（同一 incomeType 内用户唯一）、
        ``incomeType``、``time``、``symbol``、``asset``、``income``（正负号为
        交易所事实，禁止自行翻转）。字段缺失时调用方必须报错，不得猜测。
        """
        params: dict[str, Any] = {"limit": int(limit)}
        if income_type:
            params["incomeType"] = income_type
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        return cast("list[dict[str, Any]]", self.client.get("/fapi/v1/income", params) or [])

    # -- 交易 ----------------------------------------------------------------

    def place_order(self, req: OrderRequest) -> Order:
        """提交订单。永不重试。结果未知时抛 UnknownSubmission。"""
        self._validate(req)
        params: dict[str, Any] = {
            "symbol": req.symbol,
            "side": req.side.value,
            "type": req.order_type.value,
            "quantity": format_decimal(req.quantity),
            "positionSide": req.position_side,
            "newClientOrderId": req.client_order_id,
        }
        if req.reduce_only:
            params["reduceOnly"] = "true"
        if req.order_type is OrderType.LIMIT:
            if req.price is None:
                raise ValueError("LIMIT 单必须提供 price")
            params["price"] = format_decimal(req.price)
            params["timeInForce"] = req.time_in_force or "GTC"
        payload = self.client.place_order("/fapi/v1/order", params, client_order_id=req.client_order_id)
        return _parse_order(payload, req)

    def _validate(self, req: OrderRequest) -> None:
        """单向持仓模式约束：positionSide 固定 BOTH。"""
        if req.position_side != "BOTH":
            raise ValueError(
                f"初版只允许单向持仓模式 positionSide=BOTH，收到 {req.position_side!r}"
                "（多仓位模式未纳入设计，禁止自动切换）"
            )

    def reduce_only_close(self, symbol: str, quantity: Decimal, *, price: Decimal | None = None,
                          client_order_id: str) -> Order:
        """平仓专用：买入 reduceOnly（平空单）。

        校验不会反向开仓：reduceOnly 订单交易所层面就不可能反向开仓，
        这里再做一层数量/方向断言。
        """
        if quantity <= 0:
            raise ValueError(f"平仓数量必须为正，收到 {quantity}")
        req = OrderRequest(
            client_order_id=client_order_id,
            symbol=symbol,
            side=OrderSide.BUY,  # 平空单 = 买入
            market=self.MARKET,
            order_type=OrderType.LIMIT if price is not None else OrderType.MARKET,
            quantity=quantity,
            price=price,
            time_in_force="IOC" if price is not None else None,
            reduce_only=True,
        )
        return self.place_order(req)

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
        return self.client.cancel_order("/fapi/v1/order", params, client_order_id=client_order_id)

    def countdown_cancel_all(self, symbol: str, countdown_seconds: int = 1) -> Any:
        """订单自动过期保护：countdown 秒后自动撤掉该 symbol 所有挂单。

        用于 LIMIT 单提交后防止意外长时间挂单（文档 §3.2）。
        """
        params = {"symbol": symbol, "countdownSeconds": int(countdown_seconds)}
        return self.client.execute_critical(
            "POST", "/fapi/v1/countdownCancelAll", params, client_order_id=None
        )

    # -- 用户数据流 ----------------------------------------------------------

    def create_listen_key(self) -> str:
        payload = self.client.post("/fapi/v1/listenKey")
        return str(payload["listenKey"])

    def keepalive_listen_key(self, listen_key: str) -> Any:
        return self.client.put("/fapi/v1/listenKey", {"listenKey": listen_key})

    def close_listen_key(self, listen_key: str) -> Any:
        return self.client.delete("/fapi/v1/listenKey", {"listenKey": listen_key})

    # -- 查询恢复 ------------------------------------------------------------

    def query_by_client_order_id(self, symbol: str, client_order_id: str) -> Order | None:
        """UNKNOWN_SUBMISSION 恢复入口：按 clientOrderId 查实际状态。"""
        from ..errors import BinanceError

        try:
            payload = self.get_order(symbol, client_order_id=client_order_id)
        except BinanceError as exc:
            if exc.code in (-2013, -2011):
                return None
            raise
        return _parse_order(payload, self._rebuild_request(symbol, client_order_id))

    def _rebuild_request(self, symbol: str, client_order_id: str) -> OrderRequest:
        return OrderRequest(
            client_order_id=client_order_id,
            symbol=symbol,
            side=OrderSide.BUY,
            market=self.MARKET,
            order_type=OrderType.MARKET,
            quantity=Decimal("0"),
        )
