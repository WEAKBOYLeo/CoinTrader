"""实盘执行领域模型。

硬规则：

1. **所有数量、价格、金额字段一律 ``Decimal``**。禁止用二进制浮点数
   直接格式化下单参数（开发设计文档 §2.2）。
2. 模型对象是不可变的（frozen dataclass），状态变化 = 新对象。
3. 本模块不 import 网络库，不发起 IO —— 纯数据结构。
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

from .guard import Market, OrderSide, OrderType

__all__ = [
    "AccountSnapshot",
    "ExchangeSnapshotBundle",
    "Fill",
    "FundingAuthority",
    "FundingCashflow",
    "HedgeCheck",
    "Order",
    "OrderIntent",
    "OrderRequest",
    "PairExecution",
    "PairStatus",
    "PositionSnapshot",
    "ReconciliationResult",
    "SyncCursor",
    "new_client_order_id",
    "new_id",
]


#: 资金费事实的权威口径（实施计划书 v2.0 T3，AC-08）：
# AUTHORITATIVE = 交易所 income 接口实际入账；ESTIMATED = 费率×名义额估算，
# 两者绝不混入同一个 PnL 口径，同 (symbol, funding_ts) 有 AUTHORITATIVE 时只计它。
class FundingAuthority(str, Enum):
    AUTHORITATIVE = "AUTHORITATIVE"
    ESTIMATED = "ESTIMATED"


@dataclass(frozen=True, slots=True)
class FundingCashflow:
    """一条资金费事实（exchange income 或估算）。amount 正负号采用交易所事实。"""

    cashflow_id: str
    market: Market
    symbol: str
    funding_ts_ms: int
    funding_rate: Decimal
    interval_hours: int
    amount: Decimal
    authority: FundingAuthority
    asset: str = "USDT"
    exchange_income_id: str | None = None
    observed_ms: int = 0
    run_id: str = ""
    pair_execution_id: str | None = None
    source: str = ""
    raw_summary: str = "{}"


@dataclass(frozen=True, slots=True)
class SyncCursor:
    """同步游标：只前进不后退；单写者（store 事务保护）。"""

    scope: str  # 市场：SPOT / PERP
    stream: str  # 数据流：fills / funding_income
    symbol_key: str  # symbol
    last_time_ms: int = 0
    last_id: str = ""
    updated_ms: int = 0


@dataclass(frozen=True, slots=True)
class ExchangeSnapshotBundle:
    """同一次 capture 的交易所当前快照束（账户/资产/全部持仓/开放订单）。

    字段完整才 ``complete=True``；不完整不得更新可信 current projection /
    can_open，Reconciler 消费同一 bundle 不重复拉 API。
    """

    snapshot_id: str
    capture_start_ms: int
    capture_end_ms: int
    spot_account: dict[str, Any]
    futures_account: dict[str, Any]
    positions: list[dict[str, Any]]
    spot_open_orders: list[dict[str, Any]]
    perp_open_orders: list[dict[str, Any]]
    source: str = "rest"
    complete: bool = True


def new_id(prefix: str) -> str:
    """生成短唯一 id：prefix-时间戳-随机后缀。"""
    ts = int(time.time() * 1000)
    rand = os.urandom(4).hex()
    return f"{prefix}-{ts}-{rand}"


def new_client_order_id(strategy_version: str, pair_execution_id: str, leg: str) -> str:
    """生成交易所 clientOrderId。

    格式：``ct-{strategy}-{pair_id}-{leg}-{rand6}``。
    币安限制：最多 32 个字符，字符集 [\\.A-Z\\^a-z\\_]。
    超长时截断 pair 部分，保留策略版本/腿/随机尾（随机尾是去重关键）。
    """
    allowed = lambda ch: ch.isalnum() or ch in "._^"  # noqa: E731
    strategy = "".join(c for c in strategy_version if allowed(c))[:8] or "s"
    pair = "".join(c for c in pair_execution_id if allowed(c))
    leg = "".join(c for c in leg if allowed(c))[:4]
    rand = os.urandom(3).hex()
    head = f"ct-{strategy}-{pair}-{leg}-"
    budget = 32 - len(head) - len(rand)
    if budget < 1:
        head = f"ct-{strategy}-{leg}-"
        budget = 32 - len(head) - len(rand)
    pair = pair[: max(0, budget)]
    return f"ct-{strategy}-{pair}-{leg}-{rand}"


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """策略交易意图（开发设计文档 §5.1）。

    描述「策略想做什么」，尚未经过规则归一化与风控。
    """

    symbol: str
    side: OrderSide
    market: Market
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None = None  # MARKET 单的参考价（风控必需）
    reduce_only: bool = False
    is_closing: bool = False
    reason: str = ""
    strategy_version: str = "funding_carry-1.0"
    signal_time_ms: int = 0
    intent_id: str = field(default_factory=lambda: new_id("int"))
    pair_execution_id: str | None = None
    time_in_force: str | None = None  # IOC / GTC，LIMIT 单用

    @property
    def notional(self) -> Decimal:
        price = self.price if self.price is not None else Decimal(0)
        return abs(self.quantity) * abs(price)


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """经规则与风控归一化后的交易所请求。"""

    client_order_id: str
    symbol: str
    side: OrderSide
    market: Market
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None = None
    time_in_force: str | None = None
    reduce_only: bool = False
    position_side: str = "BOTH"  # 单向持仓模式固定 BOTH


@dataclass(slots=True)
class Order:
    """交易所订单状态（可变快照，由状态机受控更新）。"""

    client_order_id: str
    symbol: str
    market: Market
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    state: str  # OrderState 值，见 order_state.py
    exchange_order_id: str | None = None
    executed_qty: Decimal = Decimal("0")
    avg_price: Decimal | None = None
    price: Decimal | None = None
    reduce_only: bool = False
    updated_ms: int = 0
    # -- 关联字段（§6.1 统一关联链）--
    run_id: str = ""
    pair_execution_id: str | None = None
    intent_id: str | None = None
    submit_ts_ms: int | None = None
    ack_ts_ms: int | None = None
    terminal_ts_ms: int | None = None
    failure_class: str | None = None
    raw_status_summary: str | None = None
    #: 从交易所响应的累计成交量/均价刷新。调用方必须先通过状态机校验。
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def is_terminal(self) -> bool:
        from .order_state import TERMINAL_STATES

        return self.state in TERMINAL_STATES

    @property
    def executed_notional(self) -> Decimal:
        if self.avg_price is None:
            return Decimal(0)
        return self.executed_qty * self.avg_price


@dataclass(frozen=True, slots=True)
class Fill:
    """一次实际成交。"""

    fill_id: str
    client_order_id: str
    symbol: str
    market: Market
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee_asset: str
    fee_amount: Decimal
    ts_ms: int
    exchange_order_id: str | None = None
    # -- 关联字段（§6.1 / §8.1）--
    run_id: str = ""
    pair_execution_id: str | None = None
    intent_id: str | None = None
    #: 交易所 fill/trade id；与 market 组成唯一键（不同市场 id 可能相同）
    exchange_trade_id: str | None = None
    quote_qty: Decimal | None = None
    maker_taker: str | None = None
    exchange_ts_ms: int | None = None
    received_ts_ms: int | None = None


@dataclass(slots=True)
class HedgeCheck:
    """两腿名义额对冲检查结果。"""

    spot_notional: Decimal
    perp_notional: Decimal
    diff: Decimal
    diff_pct: Decimal  # diff / 较大一侧
    within_tolerance: bool
    residual_qty: Decimal  # 永续腿需要 reduce 的残量


class PairStatus(str, Enum):
    """PairExecution 状态机状态（开发设计文档 §6.1）。"""

    # 开仓
    NEW_INTENT = "NEW_INTENT"
    PRECHECK = "PRECHECK"
    SUBMIT_PERP = "SUBMIT_PERP"
    PERP_ACKNOWLEDGED = "PERP_ACKNOWLEDGED"
    PERP_FILLED = "PERP_FILLED"
    SUBMIT_SPOT = "SUBMIT_SPOT"
    SPOT_FILLED = "SPOT_FILLED"
    HEDGE_VERIFIED = "HEDGE_VERIFIED"
    COMPLETE = "COMPLETE"
    # 异常/补偿
    UNKNOWN_SUBMISSION = "UNKNOWN_SUBMISSION"
    COMPENSATING = "COMPENSATING"
    COMPENSATED = "COMPENSATED"
    FLATTENED = "FLATTENED"
    # 终态
    FAILED = "FAILED"
    HALTED = "HALTED"
    # 平仓
    CLOSE_INTENT = "CLOSE_INTENT"
    PRECHECK_CLOSE = "PRECHECK_CLOSE"
    FILL_TRACKING = "FILL_TRACKING"
    RESIDUAL_CHECK = "RESIDUAL_CHECK"
    COMPENSATE_RESIDUAL = "COMPENSATE_RESIDUAL"


PAIR_TERMINAL_STATES = frozenset(
    {
        PairStatus.COMPLETE.value,
        PairStatus.COMPENSATED.value,
        PairStatus.FLATTENED.value,
        PairStatus.FAILED.value,
        PairStatus.HALTED.value,
    }
)


@dataclass(slots=True)
class PairExecution:
    """一组 Spot/Futures 对冲操作。"""

    pair_execution_id: str
    symbol: str
    target_notional: Decimal
    strategy_version: str
    status: str = PairStatus.NEW_INTENT.value
    intent_id: str | None = None
    kind: str = "open"  # open / close
    perp: Order | None = None
    spot: Order | None = None
    perp_request: OrderRequest | None = None
    spot_request: OrderRequest | None = None
    hedge_ratio: Decimal | None = None
    residual_qty: Decimal = Decimal("0")
    error: str = ""
    created_ms: int = 0
    updated_ms: int = 0
    # -- 关联与审计字段（§6.1 / §8.1）--
    run_id: str = ""
    open_or_close_reason: str = ""
    signal_decision_id: str = ""
    signal_ts_ms: int = 0
    decision_ts_ms: int = 0
    submit_ts_ms: int = 0
    exchange_confirmed_ts_ms: int = 0
    completed_ts_ms: int = 0
    actual_spot_notional: Decimal | None = None
    actual_perp_notional: Decimal | None = None
    # -- PnL（由报告层计算后回填，执行层不填）--
    funding_pnl: Decimal | None = None
    fee_pnl: Decimal | None = None
    basis_pnl: Decimal | None = None
    realized_pnl: Decimal | None = None
    unrealized_pnl: Decimal | None = None
    net_pnl: Decimal | None = None

    def touches(self, status: str, error: str = "") -> None:
        self.status = status
        if error:
            self.error = error
        self.updated_ms = int(time.time() * 1000)


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """某时刻某 symbol 的仓位快照。"""

    ts_ms: int
    symbol: str
    spot_qty: Decimal
    perp_qty: Decimal  # 负数 = 空头
    spot_price: Decimal | None = None
    perp_price: Decimal | None = None
    basis_pct: Decimal | None = None
    source: str = "reconcile"
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """账户快照（原始 payload + 关键字段）。"""

    ts_ms: int
    source: str  # "SPOT" | "PERP"
    available_balance: Decimal | None = None
    wallet_balance: Decimal | None = None
    equity: Decimal | None = None
    unrealized_pnl: Decimal | None = None
    margin_ratio: Decimal | None = None
    run_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """对账结果（开发设计文档 §5.1）。"""

    ts_ms: int
    consistent: bool
    mismatches: tuple[str, ...]
    repaired: tuple[str, ...]
    can_open: bool
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.consistent and self.mismatches:
            raise ValueError("consistent=True 但存在 mismatches，逻辑矛盾")

    def fingerprint(self) -> str:
        text = f"{self.consistent}|{','.join(self.mismatches)}|{','.join(self.repaired)}"
        return hashlib.sha256(text.encode()).hexdigest()[:16]
