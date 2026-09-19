"""执行层 —— 下单能力（默认全部禁用）。

⚠️ **本包的设计前提是：它随时可能被写错，所以外部必须有多道闸门兜住。**

三层防护：

1. ``guard.py`` —— 唯一的真实下单入口，默认拒绝一切
2. ``risk.py`` —— 独立的限额检查，即使 guard 被绕过也会拦住
3. ``broker.py`` —— 所有方法默认 ``dry_run=True``

**本包永不被 ``data`` / ``research`` / ``backtest`` import。**
这条约束由 ``tests/test_safety.py`` 静态强制。
"""

from __future__ import annotations

__all__: list[str] = []
