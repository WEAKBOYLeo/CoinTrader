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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any

from ..config import Config
from ..execution.models import new_id
from ..execution.risk import Position, RiskState
from ..execution.store import StateStore

logger = logging.getLogger(__name__)

__all__ = ["AccountStateError", "AccountStateResult", "AccountStateBuilder", "parse_account"]


class AccountStateError(Exception):
    """账户快照失败/字段缺失/资金为零。资金未知 → 禁止开仓。"""


@dataclass(frozen=True, slots=True)
class AccountStateResult:
    """解析结果：RiskState + 原始 Decimal 快照（审计用）。

    v2（T3）：equity 与 available 分离；``total_capital`` 兼容属性返回
    ``total_equity``（内部新代码统一用 total_equity）。
    """

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
    #: 现货权益（USDT + 受管理现货资产按同一快照价格估值）
    spot_equity: Decimal
    #: 合约权益（优先 totalMarginBalance，缺失时 wallet + unrealized 显式回退）
    futures_equity: Decimal
    #: 总权益 = spot_equity + futures_equity
    total_equity: Decimal
    ts_ms: int
    source: str
    #: 完整快照组 id（current projection 关联）
    snapshot_id: str = ""
    #: False = 必需估值缺失/字段不完整 → 禁止开仓（不得当完整快照放行）
    complete: bool = True

    @property
    def total_capital(self) -> Decimal:
        """兼容属性：旧调用方继续用 total_capital，口径 = total_equity。"""
        return self.total_equity

    @property
    def available_balance(self) -> Decimal:
        return self.spot_available + self.futures_available


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
    asset_prices: Mapping[str, Decimal] | None = None,
) -> AccountStateResult:
    """从原始账户 payload 构建 RiskState。纯函数（无 IO），失败抛错。

    ``asset_prices``：受管理现货资产的估值价格（{base: USDT 价}）；缺失时
    该资产无法估值 → ``complete=False``（禁止开仓，不用错误权益冒充）。
    """
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

    # futures 权益：优先 totalMarginBalance；缺失时 wallet + unrealized 显式回退
    margin_balance_raw = futures_payload.get("totalMarginBalance")
    if margin_balance_raw not in (None, ""):
        fut_equity = _dec(margin_balance_raw, "futures.totalMarginBalance")
    else:
        fut_equity = fut_wallet + fut_upl

    # spot 权益：USDT + 受管理现货资产按同一快照价格估值
    asset_prices = dict(asset_prices or {})
    spot_equity = spot_total["USDT"]
    complete = True
    for symbol in candidate_symbols:
        base = symbol.replace("USDT", "")
        if base == "USDT":
            continue
        qty = spot_total.get(base, Decimal("0"))
        if qty <= 0:
            continue
        price = asset_prices.get(base)
        if price is None or price <= 0:
            complete = False  # 受管理资产无新鲜估值 → 快照不完整，禁止开仓
            continue
        spot_equity += qty * price

    total_equity = spot_equity + fut_equity
    if total_equity <= 0:
        raise AccountStateError(
            f"总权益 {total_equity} 为零或负数：资金未知，禁止放行开仓"
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
        total_capital=float(total_equity),
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
        spot_equity=spot_equity,
        futures_equity=fut_equity,
        total_equity=total_equity,
        ts_ms=now_ms,
        source=source,
        complete=complete,
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
        bundle: Any | None = None,
        asset_prices: Mapping[str, Decimal] | None = None,
    ) -> AccountStateResult:
        """拉取并校验账户快照。失败抛 ``AccountStateError``（调用方处理）。

        ``bundle`` 给定（T3 live 路径）：消费同一 capture bundle，不重复拉
        账户/仓位 API；未给定则经 adapter 拉取（非 live 工具兼容）。
        完成后按 snapshot group 单事务写账本；complete 时才更新
        current projection，消失持仓写 qty=0 tombstone。
        """
        now_ms = int(self._now() * 1000)
        if bundle is not None:
            if bundle.spot_account is None or bundle.futures_account is None:
                raise AccountStateError(
                    "capture bundle 不完整：部分市场账户数据缺失，不更新 current projection"
                )
            spot_payload = dict(bundle.spot_account)
            futures_payload = dict(bundle.futures_account)
            positions = list(bundle.positions)
            capture_start = int(bundle.capture_start_ms)
            capture_end = int(bundle.capture_end_ms)
        else:
            try:
                spot_payload = spot.account()
                futures_payload = futures.account()
                positions = futures.positions()
            except Exception as exc:  # noqa: BLE001
                # 账户接口失败 → 未知状态（AccountStateError），不得用旧值/默认值
                raise AccountStateError(f"账户接口获取失败: {exc!r}") from exc
            capture_start = now_ms
            capture_end = now_ms

        realized_today = self.store.realized_pnl_since(now_ms - 24 * 3600 * 1000)

        result = parse_account(
            spot_payload=spot_payload,
            futures_payload=futures_payload,
            perp_positions=positions,
            realized_pnl_today=realized_today,
            now_ms=now_ms,
            candidate_symbols=self.candidate_symbols,
            source=source,
            asset_prices=asset_prices,
        )
        snapshot_id = str(bundle.snapshot_id) if bundle is not None else new_id("snap")
        result = replace(result, snapshot_id=snapshot_id)

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
            # v2：同 snapshot group 写资产/持仓 + current projection（单事务）
            asset_prices = dict(asset_prices or {})
            spot_assets: list[dict[str, Any]] = []
            for asset, qty in sorted(result.spot_balances.items()):
                if qty <= 0:
                    continue
                base = asset
                if base == "USDT":
                    price: Decimal | None = Decimal("1")
                else:
                    price = asset_prices.get(base)
                value = qty * price if price is not None else None
                spot_assets.append({
                    "asset": asset,
                    "free_qty": qty,
                    "locked_qty": Decimal("0"),
                    "total_qty": qty,
                    "price_usdt": price,
                    "value_usdt": value,
                })
            positions_rows: list[dict[str, Any]] = []
            perp_by_symbol = {
                str(p.get("symbol")): Decimal(str(p.get("positionAmt") or "0"))
                for p in positions
                if str(p.get("positionSide") or "BOTH") == "BOTH"
            }
            seen_bases: set[str] = set()
            for base, qty in sorted(result.spot_balances.items()):
                if base == "USDT" or qty <= 0:
                    continue
                seen_bases.add(base)
                positions_rows.append({
                    "symbol": f"{base}USDT",
                    "spot_qty": qty,
                    "perp_qty": perp_by_symbol.get(f"{base}USDT", Decimal("0")),
                })
            for symbol, qty in sorted(perp_by_symbol.items()):
                if qty == 0:
                    continue
                base = symbol.replace("USDT", "")
                if base in seen_bases or base == "USDT":
                    continue
                seen_bases.add(base)
                positions_rows.append({
                    "symbol": symbol,
                    "spot_qty": Decimal("0"),
                    "perp_qty": qty,
                })
            self.store.save_account_snapshot_group(
                snapshot_id=snapshot_id,
                ts_ms=now_ms,
                capture_start_ms=capture_start,
                capture_end_ms=capture_end,
                source=source,
                run_id=run_id,
                spot_equity_usdt=result.spot_equity,
                futures_equity_usdt=result.futures_equity,
                total_equity_usdt=result.total_equity,
                available_balance_usdt=result.available_balance,
                complete=result.complete,
                spot_assets=spot_assets,
                positions=positions_rows,
            )
        except Exception as exc:  # noqa: BLE001
            # 数据库写失败 = 状态不可信（§8.5）
            raise AccountStateError(f"账户快照入账本失败: {exc}") from exc
        return result
