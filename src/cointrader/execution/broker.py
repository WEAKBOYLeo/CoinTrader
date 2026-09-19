"""下单接口 —— 默认干跑，且必须先过闸门。

**这个类的每一层都在说"不"：**

1. 构造函数不接收密钥 —— 密钥从环境变量读，且真实盘需要双开关
2. ``exchange`` 默认是模拟实现，不发任何网络请求
3. ``dry_run`` 默认 ``True``
4. 每个 ``place_*`` 方法第一步就是调 ``Guard.authorize()``

## 关于 Exchange 抽象

真实交易需要 HMAC-SHA256 签名，而**本项目在完成回测结论（M5）之前
不实现签名逻辑**。这是刻意的：签名代码一旦存在，就有被误调用的风险，
而目前没有任何回测结论支持投入真金白银。

当前只提供：

- ``SimulatedExchange`` —— 模拟撮合，用于验证下单流程的逻辑正确性

真实交易所实现应在 M5 结论为「值得做」之后再添加，
且必须配套测试网端到端验证。

## 双腿执行的风险

资金费套利要下**两条腿**（现货买 + 永续卖）。如果第一条腿成交、
第二条腿失败，账户就变成**裸头寸** —— 这是该策略唯一会真正亏大钱的方式。

``open_position()`` 采用「先开永续、后开现货」的顺序，因为：
合约腿更容易失败（保证金不足、精度不符），让它先失败可以避免
现货已经买入却无法对冲的局面。

失败时返回明确的 ``LegResult``，调用方**必须**处理单腿失败的情况。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import RiskConfig
from ..errors import CoinTraderError
from ..redact import fingerprint
from ..secrets import ApiCredentials, describe_security_posture, load_credentials
from .guard import Guard, Market, OrderIntent, OrderSide, OrderType
from .risk import RiskState

logger = logging.getLogger(__name__)


class ExchangeError(CoinTraderError):
    """交易所返回错误。"""


@dataclass(slots=True)
class OrderResult:
    """一笔订单的执行结果。"""

    ok: bool
    order_id: str | None
    symbol: str
    side: OrderSide
    market: Market
    filled_qty: float
    avg_price: float
    error: str | None = None

    @property
    def notional(self) -> float:
        return abs(self.filled_qty) * abs(self.avg_price)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "market": self.market.value,
            "filled_qty": self.filled_qty,
            "avg_price": self.avg_price,
            "notional": round(self.notional, 4),
            "error": self.error,
        }


@dataclass(slots=True)
class LegResult:
    """双腿操作的合并结果。"""

    spot: OrderResult | None = None
    perp: OrderResult | None = None
    aborted: bool = False
    abort_reason: str = ""

    @property
    def both_ok(self) -> bool:
        return bool(
            self.spot is not None
            and self.perp is not None
            and self.spot.ok
            and self.perp.ok
        )

    @property
    def is_naked(self) -> bool:
        """是否形成了裸头寸 —— 只有一腿成交。**这是最危险的中间态。**"""
        if self.spot is None or self.perp is None:
            return False
        return self.spot.ok != self.perp.ok

    def describe(self) -> str:
        if self.aborted:
            return f"操作中止（未下单）: {self.abort_reason}"
        if self.both_ok:
            return "两腿均成功"
        if self.is_naked:
            return "⚠️ 裸头寸！只有一腿成交，必须立即人工干预"
        return "两腿均失败"


class Exchange(Protocol):
    """交易所接口。真实实现必须做 HMAC 签名，且需在 M5 之后才添加。"""

    def place_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        market: Market,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
    ) -> OrderResult: ...


class SimulatedExchange:
    """模拟交易所 —— 只做记账，不发网络请求。

    用途：验证下单流程的逻辑（闸门拦截、双腿顺序、失败处理）是否正确。
    它**故意**实现得很简单：没有任何撮合逻辑、没有滑点、没有部分成交。

    参数 ``fail_on`` 可以注入失败，用于测试裸头寸的处理路径。
    """

    def __init__(
        self,
        *,
        fail_on: set[tuple[str, str]] | None = None,
        price_override: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            fail_on: 需要失败的 ``(symbol, side.value)`` 组合集合。
            price_override: 指定 symbol 的成交价。
        """
        self.fail_on = fail_on or set()
        self.price_override = price_override or {}
        self.orders: list[OrderResult] = []
        self._counter = 0

    def place_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        market: Market,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
    ) -> OrderResult:
        self._counter += 1
        order_id = f"sim-{self._counter:06d}"

        if (symbol, side.value) in self.fail_on:
            result = OrderResult(
                ok=False,
                order_id=None,
                symbol=symbol,
                side=side,
                market=market,
                filled_qty=0.0,
                avg_price=0.0,
                error=f"模拟失败: {symbol} {side.value}",
            )
            self.orders.append(result)
            return result

        fill_price = price or self.price_override.get(symbol, 0.0)
        result = OrderResult(
            ok=True,
            order_id=order_id,
            symbol=symbol,
            side=side,
            market=market,
            filled_qty=quantity,
            avg_price=fill_price,
        )
        self.orders.append(result)
        return result


class Broker:
    """下单协调器。

    Args:
        risk_config: 风控配置。
        exchange: 交易所实现。为 None 时用 ``SimulatedExchange``。
        dry_run: 干跑模式。**默认 True。**
        credentials: API 凭证。为 None 时不加载（回测/模拟场景无需凭证）。
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        *,
        exchange: Exchange | None = None,
        dry_run: bool = True,
        credentials: ApiCredentials | None = None,
        audit_path: str = "logs/audit.log",
    ) -> None:
        self.guard = Guard(risk_config, audit_path=audit_path, dry_run=dry_run)
        self.exchange: Exchange = exchange if exchange is not None else SimulatedExchange()
        self.dry_run = dry_run
        self.credentials = credentials

        posture = describe_security_posture()
        logger.info(
            "Broker 初始化：dry_run=%s mode=%s credentials=%s",
            dry_run,
            posture["mode"],
            fingerprint(credentials.key.reveal()) if credentials else "<未加载>",
        )

    @classmethod
    def from_env(
        cls,
        risk_config: RiskConfig,
        *,
        dry_run: bool = True,
        require_credentials: bool = False,
        audit_path: str = "logs/audit.log",
    ) -> Broker:
        """从环境变量构造 Broker。

        Args:
            require_credentials: 为 False（默认）时，缺少凭证也能构造 ——
                这让模拟/测试流程完全不需要密钥。
        """
        credentials = load_credentials(require=require_credentials)
        return cls(
            risk_config,
            dry_run=dry_run,
            credentials=credentials,
            audit_path=audit_path,
        )

    # ------------------------------------------------------------------
    # 双腿开仓
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        quantity: float,
        spot_price: float,
        perp_price: float,
        state: RiskState,
        *,
        reason: str = "",
    ) -> LegResult:
        """开一对对冲头寸：现货买入 + 永续卖出。

        **执行顺序：先开永续，后开现货。**

        理由：永续腿更容易失败（保证金不足、精度不符、合约暂停）。
        让它先失败，则还没买现货，可以直接放弃；
        若反过来，现货已买入而永续开不出来，就变成裸多头。

        Args:
            symbol: 交易对。
            quantity: 基础币数量（两腿相同）。
            spot_price: 现货参考价（用于限额检查与模拟成交）。
            perp_price: 永续参考价。
            state: 当前风控状态。
            reason: 下单理由（进审计日志）。

        Returns:
            LegResult。调用方必须检查 ``is_naked``。
        """
        notional = quantity * perp_price

        # --- 闸门检查（两条腿分别检查）---
        perp_ok = self._authorize(
            symbol, OrderSide.SELL, Market.PERP, quantity, perp_price, state, reason, False
        )
        if not perp_ok:
            return LegResult(
                aborted=True,
                abort_reason=f"永续腿未获授权（{perp_ok}），未下任何单",
            )

        spot_ok = self._authorize(
            symbol, OrderSide.BUY, Market.SPOT, quantity, spot_price, state, reason, False
        )
        if not spot_ok:
            return LegResult(
                aborted=True,
                abort_reason=f"现货腿未获授权（{spot_ok}），未下任何单",
            )

        logger.warning(
            "【开仓】%s qty=%.8f 名义额≈%.2f USDT dry_run=%s", symbol, quantity, notional, self.dry_run
        )

        if self.dry_run:
            logger.info("干跑模式：跳过真实下单，返回模拟结果")
            return LegResult(
                spot=self._simulate(symbol, OrderSide.BUY, Market.SPOT, quantity, spot_price),
                perp=self._simulate(symbol, OrderSide.SELL, Market.PERP, quantity, perp_price),
            )

        # --- 真实下单：先永续 ---
        perp_result = self.exchange.place_order(
            symbol=symbol,
            side=OrderSide.SELL,
            market=Market.PERP,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )

        if not perp_result.ok:
            logger.error("永续腿失败，放弃现货腿以避免裸多头: %s", perp_result.error)
            return LegResult(spot=None, perp=perp_result)

        # --- 后现货 ---
        spot_result = self.exchange.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            market=Market.SPOT,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )

        legs = LegResult(spot=spot_result, perp=perp_result)
        if legs.is_naked:
            logger.critical(
                "⚠️ 裸头寸！%s 永续已成交但现货失败（%s）。"
                "必须立即人工干预：要么补现货，要么平永续。",
                symbol,
                spot_result.error,
            )
        return legs

    # ------------------------------------------------------------------
    # 双腿平仓
    # ------------------------------------------------------------------

    def close_position(
        self,
        symbol: str,
        quantity: float,
        spot_price: float,
        perp_price: float,
        state: RiskState,
        *,
        reason: str = "",
    ) -> LegResult:
        """平掉一对对冲头寸：卖出现货 + 买入永续。

        **平仓顺序与开仓相反：先平现货，后平永续。**

        理由：平仓时我们希望先消除多头风险。卖出现货后即使永续腿失败，
        账户变成裸空头，但在上涨行情中裸空头的风险小于裸多头
        （浮亏不会因价格下跌而放大）。更实际的理由是：
        平仓通常发生在需要离场时，先卖现货能更快拿回现金。

        平仓单在风控中享有豁免（``is_closing=True``）——
        减仓永远应该可行，风控不该把你在最需要离场时困住。
        """
        # 平仓单豁免限额检查，但仍走闸门（交易开关、停机开关仍生效）
        spot_ok = self._authorize(
            symbol, OrderSide.SELL, Market.SPOT, quantity, spot_price, state, reason, True
        )
        perp_ok = self._authorize(
            symbol, OrderSide.BUY, Market.PERP, quantity, perp_price, state, reason, True
        )
        if not spot_ok or not perp_ok:
            return LegResult(
                aborted=True,
                abort_reason=f"平仓未获授权（现货={spot_ok}, 永续={perp_ok}），未下任何单",
            )

        logger.warning(
            "【平仓】%s qty=%.8f dry_run=%s", symbol, quantity, self.dry_run
        )

        if self.dry_run:
            return LegResult(
                spot=self._simulate(symbol, OrderSide.SELL, Market.SPOT, quantity, spot_price),
                perp=self._simulate(symbol, OrderSide.BUY, Market.PERP, quantity, perp_price),
            )

        spot_result = self.exchange.place_order(
            symbol=symbol,
            side=OrderSide.SELL,
            market=Market.SPOT,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )
        perp_result = self.exchange.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            market=Market.PERP,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )

        legs = LegResult(spot=spot_result, perp=perp_result)
        if legs.is_naked:
            logger.critical(
                "⚠️ 裸头寸！%s 平仓时只有一腿成功。必须立即人工干预。", symbol
            )
        return legs

    # ------------------------------------------------------------------

    def _authorize(
        self,
        symbol: str,
        side: OrderSide,
        market: Market,
        quantity: float,
        price: float,
        state: RiskState,
        reason: str,
        is_closing: bool,
    ) -> bool:
        intent = OrderIntent(
            symbol=symbol,
            side=side,
            market=market,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            is_closing=is_closing,
            reason=reason,
        )
        result = self.guard.authorize(intent, state)
        return result.authorized

    def _simulate(
        self,
        symbol: str,
        side: OrderSide,
        market: Market,
        quantity: float,
        price: float,
    ) -> OrderResult:
        """干跑模式下的模拟成交结果（不触碰 exchange）。"""
        exchange = SimulatedExchange(price_override={symbol: price})
        return exchange.place_order(
            symbol=symbol,
            side=side,
            market=market,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )

    def status(self) -> dict[str, Any]:
        return {
            "guard": self.guard.status(),
            "dry_run": self.dry_run,
            "exchange": type(self.exchange).__name__,
            "has_credentials": self.credentials is not None,
        }


__all__ = [
    "Broker",
    "Exchange",
    "ExchangeError",
    "LegResult",
    "OrderResult",
    "SimulatedExchange",
]
