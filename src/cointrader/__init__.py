"""CoinTrader —— 币安资金费率套利研究框架。

分层架构（依赖方向严格单向，见 docs/ARCHITECTURE.md §1）::

    data/      →  只读公开 API，永不含密钥、永不下单
    research/  →  纯计算，无网络 IO
    backtest/  →  确定性模拟，无网络 IO
    execution/ →  下单（默认禁用，被 guard 拦截）

安全约束由 tests/test_safety.py 静态强制，不依赖开发者自觉。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
