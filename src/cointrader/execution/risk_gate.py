"""实盘风控闸门 —— 开仓权限与减仓权限分离（开发设计文档 §7.3）。

停机语义拆分（替换原单一 KILL_SWITCH 语义）：

- ``NORMAL``：正常交易。
- ``HALT_NEW_RISK``：**拒绝**开仓/换仓/加仓；**允许**经过校验的
  撤单与 reduce-only 平仓。这是事故处理的核心状态。
- ``EMERGENCY_FLATTEN``：显式人工操作触发，只尝试撤单+平仓，
  其他一切请求拒绝。

关键规则：

1. 停机文件（KILL_SWITCH）存在 = 强制 HALT_NEW_RISK（只读检查，
   不改内部状态）。
2. **删除停机文件不能自动恢复交易**。恢复必须：
   重新完成启动预检 + 对账 + 显式调用 ``recover()``。
3. 任何对账不一致、未知订单、账户事件过期、保证金率异常
   都应通过 ``halt()`` 进入 HALT_NEW_RISK。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..errors import LiveGateBlocked
from .risk import RiskManager, RiskState

logger = logging.getLogger(__name__)

__all__ = ["HaltState", "RiskGate"]


class HaltState(str, Enum):
    NORMAL = "NORMAL"
    HALT_NEW_RISK = "HALT_NEW_RISK"
    EMERGENCY_FLATTEN = "EMERGENCY_FLATTEN"


@dataclass(frozen=True, slots=True)
class GateDecision:
    allowed: bool
    reason: str
    state: HaltState
    details: dict[str, Any]


class RiskGate:
    """开仓/平仓/减仓权限闸门。

    Args:
        risk_manager: 现有限额检查器（数字比较层）。
        kill_check: 停机文件检查函数（默认 secrets.is_kill_switch_engaged）。
        audit: 审计日志（可选）。所有 halt/recover 都会记录。
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        *,
        kill_check: Callable[[], bool] | None = None,
        audit: Any | None = None,
    ) -> None:
        self.risk = risk_manager
        self._kill_check = kill_check
        self._audit = audit
        self._state = HaltState.NORMAL
        self._halt_reason = ""
        self._recovered_since_halt = False

    # -- 状态 ---------------------------------------------------------------

    @property
    def state(self) -> HaltState:
        """当前有效状态。停机文件存在时至少是 HALT_NEW_RISK。"""
        if self._kill_check and self._kill_check():
            if self._state == HaltState.EMERGENCY_FLATTEN:
                return self._state
            return HaltState.HALT_NEW_RISK
        return self._state

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def halt(self, reason: str, *, emergency: bool = False) -> None:
        """进入停机状态。只能**升级**，不能降级（降级必须走 recover）。"""
        new = HaltState.EMERGENCY_FLATTEN if emergency else HaltState.HALT_NEW_RISK
        if self._state == HaltState.EMERGENCY_FLATTEN:
            return
        self._state = new
        self._halt_reason = reason
        self._recovered_since_halt = False
        logger.critical("【风控停机】%s: %s", new.value, reason)
        self._record("HALT", allowed=False, reason=reason)

    def recover(self, *, reconciliation_ok: bool, preflight_ok: bool) -> None:
        """从停机恢复。**必须**附带对账通过与预检通过证明。

        删除停机文件本身不能恢复 —— 文件存在时调用 recover 会失败。
        """
        if self._kill_check and self._kill_check():
            raise LiveGateBlocked(
                "停机文件仍存在，禁止恢复。删除文件后仍需重新预检+对账并再次调用 recover()。"
            )
        if not (reconciliation_ok and preflight_ok):
            raise LiveGateBlocked(
                f"恢复条件不满足: reconciliation_ok={reconciliation_ok}, preflight_ok={preflight_ok}"
            )
        self._state = HaltState.NORMAL
        self._halt_reason = ""
        self._recovered_since_halt = True
        logger.warning("【风控恢复】停机解除（预检+对账均通过）")
        self._record("RECOVER", allowed=True, reason="preflight+reconciliation passed")

    def is_recovered(self) -> bool:
        return self._recovered_since_halt

    # -- 权限判定 -------------------------------------------------------------

    def allow_open(
        self,
        symbol: str,
        notional: float,
        state: RiskState,
        *,
        is_closing: bool = False,
        now: float | None = None,
    ) -> GateDecision:
        """开仓/加仓权限。HALT_NEW_RISK / EMERGENCY_FLATTEN 下拒绝。"""
        now = now if now is not None else time.time()
        cur = self.state
        if cur in (HaltState.HALT_NEW_RISK, HaltState.EMERGENCY_FLATTEN):
            reason = f"{cur.value}: 禁止开新风险（{self._halt_reason or '未知原因'}）"
            self._record("ALLOW_OPEN", allowed=False, reason=reason)
            return GateDecision(False, reason, cur, {"symbol": symbol, "notional": notional})

        preflight = self.risk.preflight(state, now)
        if not preflight.allowed:
            self._record("ALLOW_OPEN", allowed=False, reason=preflight.reason)
            return GateDecision(False, preflight.reason, cur, {"symbol": symbol})

        order_check = self.risk.check_order(symbol, notional, state, is_closing=is_closing)
        if not order_check.allowed:
            self._record("ALLOW_OPEN", allowed=False, reason=order_check.reason)
            return GateDecision(False, order_check.reason, cur, {"symbol": symbol})

        self._record("ALLOW_OPEN", allowed=True, reason="通过")
        return GateDecision(True, "开仓权限通过", cur, {"symbol": symbol, "notional": notional})

    def allow_reduce(self, *, now: float | None = None) -> GateDecision:
        """撤单 / reduce-only 平仓权限。

        HALT_NEW_RISK 下**放行**（事故处理必须能减仓）；
        EMERGENCY_FLATTEN 下放行（这就是它的存在目的）；
        NORMAL 下放行。

        注意：这里放行的是「权限」，具体订单仍要过签名、账户一致性、
        symbol 规则和只减仓语义（pair_executor / adapter 层保证）。
        """
        cur = self.state
        if now is not None:
            _ = now
        self._record("ALLOW_REDUCE", allowed=True, reason=f"{cur.value}: 减仓放行")
        return GateDecision(True, f"减仓权限通过（状态 {cur.value}）", cur, {})

    def _record(self, kind: str, allowed: bool, reason: str) -> None:
        if self._audit is not None:
            try:
                self._audit.record(
                    "RISK_GATE",
                    {
                        "kind": kind,
                        "allowed": allowed,
                        "reason": reason,
                        "state": self.state.value,
                        "halt_reason": self._halt_reason,
                        "ts": time.time(),
                    },
                )
            except Exception:  # noqa: BLE001
                logger.warning("risk gate 审计写入失败", exc_info=True)
