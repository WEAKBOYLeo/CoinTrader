"""安全闸门 —— 真实下单的**唯一**入口。

设计哲学：**默认拒绝，且拒绝的摩擦要大于放行的摩擦。**

绝大多数交易系统的安全事故不是因为攻击者太强，而是因为
「默认允许 + 一个配置项写错」。本模块把默认值放在安全侧：

======================  ==========================  ==================
闸门                     通过条件                     默认值
======================  ==========================  ==================
① 交易总开关            环境变量**精确**等于         未设置 → 拒绝
                        ``YES_I_AM_SURE``
② 测试网                默认走测试网                  测试网 → 放行
③ 停机开关              ``KILL_SWITCH`` 文件不存在     不存在 → 放行
④ 风控限额              独立检查                      见 risk.py
⑤ 干跑模式              显式关闭                     开启 → 不真下单
======================  ==========================  ==================

注意闸门 ① 的严格性：必须是那个特定字符串。写成 ``1`` / ``true`` / ``yes``
都不生效。这不是「不方便」，这是**故意的**：它强迫你在下真单前
明确地、有意识地打出一句话，而不是顺手设个环境变量。

## 审计

每一次下单尝试（**包括被拒绝的**）都会写入 ``logs/audit.log``，
JSON Lines 格式，含时间、参数、判定结果与拒绝原因。
日志中永不含密钥，只有指纹。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..config import RiskConfig
from ..errors import GuardRejected
from ..redact import redact
from ..secrets import (
    TRADING_ENABLED_ENV,
    TRADING_ENABLED_MAGIC,
    describe_security_posture,
    is_kill_switch_engaged,
    is_trading_enabled,
    kill_switch_path,
    use_testnet,
)
from .risk import RiskManager, RiskState

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class Market(str, Enum):
    SPOT = "SPOT"
    PERP = "PERP"


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """一笔**意图**订单 —— 尚未经过任何检查。

    这个对象的名字是刻意的：它描述"你想做什么"，不是"已经做了什么"。
    只有穿过 ``Guard.authorize()`` 之后，它才可能变成真实的订单。
    """

    symbol: str
    side: OrderSide
    market: Market
    order_type: OrderType
    quantity: float
    price: float | None = None       # MARKET 单为 None
    is_closing: bool = False
    reason: str = ""                 # 人类可读的下单理由，进审计日志

    @property
    def notional(self) -> float:
        """名义额。MARKET 单必须显式提供参考价才能算。"""
        if self.price is None:
            return 0.0
        return abs(self.quantity) * abs(self.price)

    def as_audit_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "market": self.market.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "notional": round(self.notional, 4),
            "is_closing": self.is_closing,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class AuthorizationResult:
    """闸门判定结果。"""

    authorized: bool
    dry_run: bool
    reason: str
    checks: dict[str, str] = field(default_factory=dict)

    def as_audit_dict(self) -> dict[str, Any]:
        return {
            "authorized": self.authorized,
            "dry_run": self.dry_run,
            "reason": self.reason,
            "checks": self.checks,
        }


class AuditLog:
    """审计日志。每次下单尝试都记一条。"""

    def __init__(self, path: Path | str, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        if self.enabled:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("无法创建审计日志目录 %s: %s（审计已禁用）", self.path.parent, exc)
                self.enabled = False

    def record(self, event: str, payload: dict[str, Any]) -> None:
        """写入一条 JSON Lines 审计记录。"""
        if not self.enabled:
            return

        entry = {
            "ts": time.time(),
            "event": event,
            # 脱敏兜底：即使调用方不小心传了敏感字段，也不会落盘
            **redact(payload),
        }
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            # 审计写失败必须让人知道 —— 但不应阻断交易流程，
            # 否则磁盘满会导致无法平仓。这是刻意的取舍。
            logger.error("审计日志写入失败 %s: %s", self.path, exc)


class Guard:
    """安全闸门。

    Args:
        risk_config: 风控配置。
        audit_path: 审计日志路径。
        dry_run: 是否为干跑模式。**默认 True。**
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        *,
        audit_path: Path | str = "logs/audit.log",
        dry_run: bool = True,
    ) -> None:
        self.risk = RiskManager(risk_config)
        self.audit = AuditLog(audit_path)
        self.dry_run = dry_run

    # -- 闸门 ①-③ ----------------------------------------------------------

    def _check_switches(self) -> tuple[bool, str, dict[str, str]]:
        """检查环境级开关（交易总开关、停机开关、测试网状态）。

        Returns:
            ``(是否全部通过, 原因, 各项检查明细)``。
        """
        checks: dict[str, str] = {}

        # 闸门 ③ 优先：停机开关是最后一道人工闸门，必须先检查
        if is_kill_switch_engaged():
            path = kill_switch_path()
            checks["kill_switch"] = f"ENGAGED ({path})"
            return (
                False,
                f"停机开关已触发（文件存在: {path}）。删除该文件才能恢复。",
                checks,
            )
        checks["kill_switch"] = "clear"

        # 闸门 ① 交易总开关
        if not is_trading_enabled():
            current = os.environ.get(TRADING_ENABLED_ENV, "<未设置>")
            checks["trading_enabled"] = f"rejected (当前值: {current!r})"
            return (
                False,
                f"交易未启用。需要环境变量 {TRADING_ENABLED_ENV} 精确等于 "
                f"'{TRADING_ENABLED_MAGIC}'（当前 {current!r}）。"
                "注意：'1'/'true'/'yes' 都不生效，这是刻意的。",
                checks,
            )
        checks["trading_enabled"] = "enabled"

        # 闸门 ② 测试网（信息性，不阻断 —— 但记录下来让人知道现在连的是哪）
        checks["network"] = "testnet" if use_testnet() else "MAINNET"

        return True, "环境开关全部通过", checks

    # -- 主入口 -------------------------------------------------------------

    def authorize(
        self,
        intent: OrderIntent,
        state: RiskState,
        *,
        now: float | None = None,
    ) -> AuthorizationResult:
        """判定一笔订单是否放行。

        **这个方法不发起任何网络请求。** 它只做判定与记录。
        真正的下单在 ``broker.py``，且必须先拿到这里返回的
        ``authorized=True``。

        Args:
            intent: 订单意图。
            state: 当前风控状态。
            now: 当前时间戳（秒），便于测试注入。

        Returns:
            AuthorizationResult。默认配置下**几乎总是** ``authorized=False``。
        """
        now = now if now is not None else time.time()
        checks: dict[str, str] = {}

        # 环境开关
        switches_ok, switches_reason, switch_checks = self._check_switches()
        checks.update(switch_checks)
        if not switches_ok:
            result = AuthorizationResult(
                authorized=False,
                dry_run=self.dry_run,
                reason=switches_reason,
                checks=checks,
            )
            self._audit(intent, result, state)
            return result

        # 风控全面自检
        preflight = self.risk.preflight(state, now)
        checks["preflight"] = "pass" if preflight.allowed else f"reject: {preflight.reason}"
        if not preflight.allowed:
            result = AuthorizationResult(
                authorized=False,
                dry_run=self.dry_run,
                reason=preflight.reason,
                checks=checks,
            )
            self._audit(intent, result, state)
            return result

        # 单笔限额（需要名义额；MARKET 单缺价格时无法检查，此时拒绝）
        if intent.order_type is OrderType.MARKET and intent.price is None:
            reason = (
                "MARKET 单必须提供参考价（intent.price）才能做限额检查。"
                "缺少价格时无法评估名义额，出于安全默认拒绝。"
            )
            checks["order_limit"] = "reject: missing reference price"
            result = AuthorizationResult(
                authorized=False, dry_run=self.dry_run, reason=reason, checks=checks
            )
            self._audit(intent, result, state)
            return result

        order_check = self.risk.check_order(
            intent.symbol, intent.notional, state, is_closing=intent.is_closing
        )
        checks["order_limit"] = "pass" if order_check.allowed else f"reject: {order_check.reason}"
        if not order_check.allowed:
            result = AuthorizationResult(
                authorized=False,
                dry_run=self.dry_run,
                reason=order_check.reason,
                checks=checks,
            )
            self._audit(intent, result, state)
            return result

        # 全部通过
        result = AuthorizationResult(
            authorized=True,
            dry_run=self.dry_run,
            reason="全部闸门通过" + ("（干跑模式：不会真实下单）" if self.dry_run else ""),
            checks=checks,
        )
        self._audit(intent, result, state)
        return result

    def authorize_or_raise(self, intent: OrderIntent, state: RiskState, **kwargs: Any) -> AuthorizationResult:
        """同 ``authorize()``，但拒绝时抛 ``GuardRejected``。

        用于调用点希望"拒绝即中断"的场景（比如一条腿失败后必须停止另一条腿）。
        """
        result = self.authorize(intent, state, **kwargs)
        if not result.authorized:
            raise GuardRejected(result.reason)
        return result

    # -- 审计 ---------------------------------------------------------------

    def _audit(self, intent: OrderIntent, result: AuthorizationResult, state: RiskState) -> None:
        """记录审计条目，并输出醒目日志。"""
        posture = describe_security_posture()
        payload = {
            "intent": intent.as_audit_dict(),
            "result": result.as_audit_dict(),
            "state": {
                "total_exposure": round(state.total_exposure, 2),
                "n_positions": len(state.positions),
                "realized_pnl_today": round(state.realized_pnl_today, 4),
                "total_capital": state.total_capital,
            },
            "posture": posture,
        }
        self.audit.record("ORDER_ATTEMPT", payload)

        if result.authorized:
            logger.warning(
                "【下单已授权】%s %s %s qty=%.8f notional=%.2f dry_run=%s",
                intent.market.value,
                intent.symbol,
                intent.side.value,
                intent.quantity,
                intent.notional,
                result.dry_run,
            )
        else:
            logger.info(
                "【下单被拒】%s %s %s qty=%.8f → %s",
                intent.market.value,
                intent.symbol,
                intent.side.value,
                intent.quantity,
                result.reason,
            )

    # -- 诊断 ---------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """当前闸门状态摘要。供启动横幅与 ``cli doctor`` 使用。"""
        posture = describe_security_posture()
        return {
            "dry_run": self.dry_run,
            "audit_log": str(self.audit.path),
            "audit_enabled": self.audit.enabled,
            **posture,
        }

    def require_confirmation(self, text: str) -> bool:
        """交互式确认 —— 真实下单前请人类明确点头。

        Returns:
            用户是否确认。非交互环境（stdin 非 tty）**一律返回 False**。
        """
        import sys

        if not sys.stdin.isatty():
            logger.error("非交互环境，无法获得确认，拒绝执行: %s", text)
            return False

        logger.warning("需要人工确认: %s", text)
        try:
            answer = input(f"{text}\n输入 'CONFIRM' 继续: ").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer == "CONFIRM"


__all__ = [
    "AuditLog",
    "AuthorizationResult",
    "Guard",
    "Market",
    "OrderIntent",
    "OrderSide",
    "OrderType",
]
