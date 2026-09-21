"""控制命令发布（T3，AC-05）。

``ControlPublisher`` 把服务侧事件（对账不一致、限流、用户数据流断开、
DB 故障、人工操作）转成领域 ``ControlCommand`` 推给订阅者
（``SafetyStateMachine`` / 审计 / WebUI）。**不直接调用 BrokerPort** ——
撤单/平仓是执行意图，必须走 策略→风险→执行 审批链。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

from ..domain.control import CommandKind, ControlCommand

__all__ = ["ControlPublisher"]


def _now_ms() -> int:
    return int(time.time() * 1000)


class ControlPublisher:
    """控制命令发布器（只发布，不执行）。

    Args:
        subscribers: 命令订阅者（状态机/审计）。
    """

    def __init__(self, subscribers: Sequence[Callable[[ControlCommand], None]] = ()) -> None:
        self._subscribers = list(subscribers)
        self._recent: list[ControlCommand] = []

    def publish(
        self,
        command: CommandKind | str,
        *,
        reason: str,
        source: str,
        correlation_id: str = "",
        now_ms: int | None = None,
    ) -> ControlCommand:
        kind = command if isinstance(command, CommandKind) else CommandKind(command)
        now_ms = _now_ms() if now_ms is None else now_ms
        cmd = ControlCommand(
            command=kind,
            reason=reason,
            source=source,
            issued_at_ms=now_ms,
            correlation_id=correlation_id or None,
        )
        self._recent.append(cmd)
        if len(self._recent) > 256:
            del self._recent[: len(self._recent) - 256]
        for subscriber in self._subscribers:
            try:
                subscriber(cmd)
            except Exception:  # noqa: BLE001
                # 订阅者失败不得阻塞其他订阅者；状态机非法迁移由其自身抛异常
                logging.getLogger(__name__).warning("控制命令订阅者异常", exc_info=True)
        return cmd

    def halt_new_risk(self, *, reason: str, source: str, now_ms: int | None = None) -> ControlCommand:
        return self.publish(CommandKind.HALT_NEW_RISK, reason=reason, source=source, now_ms=now_ms)

    def recovery(self, *, reason: str, source: str, now_ms: int | None = None) -> ControlCommand:
        return self.publish(CommandKind.RECOVERY, reason=reason, source=source, now_ms=now_ms)

    def resume_after_checks(self, *, reason: str, source: str, now_ms: int | None = None) -> ControlCommand:
        return self.publish(
            CommandKind.RESUME_AFTER_CHECKS, reason=reason, source=source, now_ms=now_ms
        )

    def emergency_flatten(self, *, reason: str, now_ms: int | None = None) -> ControlCommand:
        """人工紧急平仓。``source`` 固定 manual（领域层强制人工确认）。"""
        return self.publish(
            CommandKind.EMERGENCY_FLATTEN, reason=reason, source="manual", now_ms=now_ms
        )

    @property
    def recent(self) -> tuple[ControlCommand, ...]:
        return tuple(self._recent)
