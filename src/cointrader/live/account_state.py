"""交易所快照 → 真实 RiskState 转换（开发文档 §7.5）。

硬规则：

1. 账户字段缺失、查询失败、余额为 0 或无法解析 → 抛 ``AccountStateError``。
   调用方必须停止放行开仓（资金未知 = 拒绝，不得用默认值/initial_capital 冒充）。
2. ``total_capital=0`` 只能表示资金未知；本模块在资金为 0 时直接抛错，
   不返回「0 资金的 RiskState」。
3. 所有数量/金额先用 ``Decimal`` 计算；仅在最后一步（风控接口边界）
   转 float 进入 ``RiskState``，原始 Decimal 保留在返回的快照字段里。
4. PnL、手续费、资金费不通过余额差额推导；``realized_pnl_today`` 来自
   PnL 账本（调用方传入），账户余额变化只做交叉验证。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ..config import Config
from ..execution.risk import Position, RiskState
from ..execution.store import StateStore

logger = logging.getLogger(__name__)

__all__ = ["AccountStateError", "AccountStateResult", "AccountStateBuilder", "parse_account"]


class AccountStateError(Exception):
    """账户快照失败/字段缺失/资金为零。资金未知 → 禁止开仓。"""


@dataclass(frozen=True, slots=True)
class AccountStateResult:
    """解析结果：RiskState + 原始 Decimal 快照（审计用）。"""

    state: RiskState
    #: {asset: totalQty} 现货总余额
    spot_balances: dict[str, Decimal]
    #: 现货可用（USDT free）
    spot_available: Decimal
    #: 合约钱包余额
    futures_wallet: Decimal
    #: 合约可用余额
    futures_available: Decimal
    #: 合约未实现盈亏
    futures_unrealized: Decimal
    #: 总资金（Decimal 口径）
    total_capital: Decimal
    ts_ms: int
    source: str


def _require(payload: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in payload or payload[key] is None or payload[key] == "":
        raise AccountStateError(f"账户字段缺失 {ctx}.{key}，资金未知，禁止放行开仓")
    return payload[key]


def _dec(value: Any, ctx: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception as exc:  # noqa: BLE001
        raise AccountStateError(f"账户字段无法解析为数字: {ctx}: {value!r}") from exc


def parse_account(
    *,
    spot_payload: dict[str, Any],
    futures_payload: dict[str, Any],
    perp_positions: list[dict[str, Any]],
    realized_pnl_today: Decimal,
    now_ms: int,
    candidate_symbols: Sequence[str],
    source: str = "periodic",
) -> AccountStateResult:
    """从原始账户 payload 构建 RiskState。纯函数（无 IO），失败抛错。"""
    # -- Spot ----------------------------------------------------------------
    balances_raw = _require(spot_payload, "balances", "spot")
    if not isinstance(balances_raw, list) or not balances_raw:
        raise AccountStateError("spot.balances 为空，资金未知")
    spot_total: dict[str, Decimal] = {}
    spot_free: dict[str, Decimal] = {}
    for item in balances_raw:
        asset = str(item.get("asset") or "")
        if not asset:
            continue
        try:
            spot_total[asset] = Decimal(str(item.get("free") or "0")) + Decimal(str(item.get("locked") or "0"))
            spot_free[asset] = Decimal(str(item.get("free") or "0"))
        except Exception as exc:  # noqa: BLE001
            raise AccountStateError(f"spot 余额解析失败 {asset}: {item!r}") from exc
    if "USDT" not in spot_total:
        raise AccountStateError("spot 账户无 USDT 余额记录，资金未知")

    # -- Futures ---------------------------------------------------------------
    fut_wallet = _dec(_require(futures_payload, "totalWalletBalance", "futures"), "futures.totalWalletBalance")
    fut_available_raw = futures_payload.get("availableBalance")
    if fut_available_raw is None or fut_available_raw == "":
        raise AccountStateError("账户字段缺失 futures.availableBalance，资金未知")
    fut_available = _dec(fut_available_raw, "futures.availableBalance")
    upl_raw = futures_payload.get("totalUnrealizedProfit")
    fut_upl = _dec(upl_raw, "futures.totalUnrealizedProfit") if upl_raw not in (None, "") else Decimal("0")

    total_capital = spot_total["USDT"] + fut_wallet
    if total_capital <= 0:
        raise AccountStateError(
            f"总资金 {total_capital} 为零或负数：资金未知，禁止放行开仓"
            "（不能用默认值或 initial_capital 冒充账户资金）"
        )

    # -- 持仓（只取候选 symbol；以交易所持仓为准，不是本地账本） ---------------
    positions: dict[str, Position] = {}
    for symbol in candidate_symbols:
        base = symbol.replace("USDT", "")
        spot_qty = spot_total.get(base, Decimal("0"))
        perp_qty = Decimal("0")
        for pos in perp_positions:
            if pos.get("symbol") == symbol and str(pos.get("positionSide") or "BOTH") == "BOTH":
                perp_qty = _dec(pos.get("positionAmt") or "0", f"futures.positionAmt[{symbol}]")
                break
        if spot_qty <= 0 and abs(perp_qty) <= 0:
            continue
        positions[symbol] = Position(
            symbol=symbol,
            # float 转换集中在风控边界（§7.5）
            spot_qty=float(spot_qty),
            perp_qty=float(-perp_qty) if perp_qty < 0 else 0.0,
        )

    state = RiskState(
        positions=positions,
        realized_pnl_today=float(realized_pnl_today),
        total_capital=float(total_capital),
        snapshot_ts_ms=now_ms,
        snapshot_source=source,
        available_balance=float(spot_free.get("USDT", Decimal("0")) + fut_available),
        futures_wallet_balance=float(fut_wallet),
        unrealized_pnl=float(fut_upl),
    )
    return AccountStateResult(
        state=state,
        spot_balances=spot_total,
        spot_available=spot_free.get("USDT", Decimal("0")),
        futures_wallet=fut_wallet,
        futures_available=fut_available,
        futures_unrealized=fut_upl,
        total_capital=total_capital,
        ts_ms=now_ms,
        source=source,
    )


class AccountStateBuilder:
    """周期采样：拉账户快照 → 校验 → 写账本 → 返回 RiskState。"""

    def __init__(
        self,
        *,
        config: Config,
        store: StateStore,
        candidate_symbols: Sequence[str],
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.store = store
        self.candidate_symbols = tuple(candidate_symbols)
        self._now = now_fn

    def snapshot(
        self,
        *,
        spot: Any,
        futures: Any,
        source: str = "periodic",
        run_id: str = "",
    ) -> AccountStateResult:
        """拉取并校验账户快照。失败抛 ``AccountStateError``（调用方处理）。"""
        now_ms = int(self._now() * 1000)
        spot_payload = spot.account()
        futures_payload = futures.account()
        positions = futures.positions()

        realized_today = self.store.realized_pnl_since(now_ms - 24 * 3600 * 1000)

        result = parse_account(
            spot_payload=spot_payload,
            futures_payload=futures_payload,
            perp_positions=positions,
            realized_pnl_today=realized_today,
            now_ms=now_ms,
            candidate_symbols=self.candidate_symbols,
            source=source,
        )

        # 周期采样（§8.5）：至少每 30s 写账户快照
        from ..execution.models import AccountSnapshot

        try:
            self.store.record_account_snapshot(
                AccountSnapshot(
                    ts_ms=now_ms,
                    source="SPOT",
                    available_balance=result.spot_available,
                    wallet_balance=result.spot_balances.get("USDT"),
                    run_id=run_id,
                )
            )
            self.store.record_account_snapshot(
                AccountSnapshot(
                    ts_ms=now_ms,
                    source="PERP",
                    available_balance=result.futures_available,
                    wallet_balance=result.futures_wallet,
                    equity=result.futures_wallet + result.futures_unrealized,
                    unrealized_pnl=result.futures_unrealized,
                    run_id=run_id,
                )
            )
        except Exception as exc:  # noqa: BLE001
            # 数据库写失败 = 状态不可信（§8.5）
            raise AccountStateError(f"账户快照入账本失败: {exc}") from exc
        return result
