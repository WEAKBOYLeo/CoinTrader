"""健康/告警事件发布（T3，AC-05）。

``HealthPublisher`` 是**只发布**的出口：把对账差异、订单未知、限流、
数据质量、保证金异常、崩溃/恢复等事件转成领域 ``HealthEvent`` 推给
订阅者（日志/WebUI/审计）。它不产生订单、不调用 BrokerPort、不写账本。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from ..domain.events import HealthEvent, HealthKind

logger = logging.getLogger(__name__)

__all__ = ["HealthPublisher", "Subscriber"]

Subscriber = Callable[[HealthEvent], None]


class HealthPublisher:
    """健康事件发布器。

    Args:
        subscribers: 事件订阅者列表（WebUI/审计等）。
    """

    def __init__(self, subscribers: Sequence[Subscriber] = ()) -> None:
        self._subscribers: list[Subscriber] = list(subscribers)
        self._recent: list[HealthEvent] = []

    def publish(
        self,
        kind: HealthKind,
        *,
        source: str,
        message: str,
        details: tuple[str, ...] = (),
    ) -> HealthEvent:
        """发布一个健康事件；订阅者异常被吞掉（可观测性不得影响主流程）。"""
        event = HealthEvent(source=source, kind=kind, message=message, details=details)
        self._recent.append(event)
        if len(self._recent) > 512:
            del self._recent[: len(self._recent) - 512]
        if kind is HealthKind.ERROR:
            logger.critical("【健康事件】%s: %s", source, message)
        elif kind in (HealthKind.DEGRADED,):
            logger.error("【健康事件】%s: %s", source, message)
        else:
            logger.info("【健康事件】%s: %s", source, message)
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception:  # noqa: BLE001
                logger.warning("健康事件订阅者异常", exc_info=True)
        return event

    @property
    def recent(self) -> tuple[HealthEvent, ...]:
        return tuple(self._recent)
