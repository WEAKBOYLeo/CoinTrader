"""研究层 —— 纯计算，无网络 IO，无文件写入。

⚠️ 本包的**所有模块**禁止出现 ``httpx`` / ``requests`` / ``urllib`` 的 import，
由 ``tests/test_safety.py`` 静态强制。

这条约束的价值：研究逻辑可以在没有网络、没有密钥、没有交易所账号的情况下
完整跑通和测试。策略能不能赚钱是数学问题，不是连接问题。
"""

from __future__ import annotations

__all__: list[str] = []
