"""生产服务启动顺序固化（实施计划书 4.0 T4.3）。

唯一装配/启动入口：``run_startup(svc)``。阶段顺序固定且带 trace：

    config→mode → server time → 规则 → lease → capture → facts
    → 对账 → 用户流 → run_session → 账户完整 →（市场 READY/闸门）→ RUNNING

- 硬失败（锁冲突 / 时钟超阈 / 规则不符 / 用户流不新鲜 / 账本硬错误）：
  阶段抛异常 → teardown（释放锁/关流）→ 异常上抛（启动时崩溃是特性）。
- 软闸门（capture / facts 同步 / 启动对账 / 账户完整 / 市场 READY /
  风控闸门）失败：默认 RECOVERY（禁止开仓），主循环闸门全绿后经
  ``RESUME_AFTER_CHECKS`` 自动回 RUNNING。
- 状态迁移全部经 LiveService 的控制面（ControlPublisher → SafetyStateMachine），
  本模块不直接写状态。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..live.service import LiveService, StartupReport

__all__ = ["StartupContext", "STARTUP_STAGES", "run_startup"]

#: 固定启动顺序（T4.3）。修改顺序 = 修改生产启动契约，必须同步测试。
STARTUP_STAGES: tuple[str, ...] = (
    "mode",
    "server_time",
    "rules",
    "lease",
    "capture",
    "facts_sync",
    "reconcile",
    "streams",
    "session",
    "account",
    "final_gates",
)


@dataclass
class StartupContext:
    """启动阶段的共享上下文（report + bundle + 闸门 + trace）。"""

    report: StartupReport
    bundle: Any = None
    reconcile_result: Any = None
    gates: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)


def run_startup(svc: LiveService) -> StartupReport:
    """按 ``STARTUP_STAGES`` 顺序执行启动；trace 写入 ``report.extra``。"""
    ctx = StartupContext(report=_new_report(svc))
    for name in STARTUP_STAGES:
        stage: Callable[..., Any] = getattr(svc, f"_stage_{name}")
        started = time.monotonic()
        try:
            if name in ("mode", "lease", "streams", "session"):
                detail: Any = stage()
            elif name in ("capture", "facts_sync", "reconcile", "account", "final_gates"):
                detail = stage(ctx)
            else:  # server_time / rules
                detail = stage(ctx.report)
        except Exception as exc:
            ctx.trace.append(
                {
                    "stage": name,
                    "ok": False,
                    "detail": str(exc),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                }
            )
            # 启动失败：释放锁、关流，不留半成品（启动时崩溃是特性）
            svc._teardown()  # noqa: SLF001 - lifecycle 与 service 同属生产装配面
            raise
        ctx.trace.append(
            {
                "stage": name,
                "ok": True,
                "detail": str(detail or "ok"),
                "duration_ms": int((time.monotonic() - started) * 1000),
            }
        )
    ctx.report.extra["startup_trace"] = list(ctx.trace)
    return ctx.report


def _new_report(svc: LiveService) -> StartupReport:
    from ..live.service import StartupReport

    return StartupReport(mode=svc.mode)
