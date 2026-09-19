"""数据层 —— 只读币安公开接口。

⚠️ 本包的**所有模块**必须满足以下约束，由 ``tests/test_safety.py`` 静态强制：

1. 不存在任何签名/鉴权代码 —— 扫描器会拒绝出现凭据相关的关键字
   （具体关键词见 ``tests/test_safety.py`` 的 ``FORBIDDEN_IN_DATA_LAYER``，
   本文件刻意不写出来，否则会被自己抓到）
2. 不 import 任何 ``execution`` 模块（依赖方向单向）
3. 只调用公开 GET 端点

这样设计的目的：让「数据层意外下单」在**代码结构上**不可能发生，
而不是靠开发者记得别写。
"""

from __future__ import annotations

from .cache import DiskCache, make_key

__all__ = ["DiskCache", "make_key"]
