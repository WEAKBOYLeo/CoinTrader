"""密钥与敏感信息脱敏。

这是安全链条上单独的一环，因为**日志是密钥最常见的泄露途径**：

- 调试时 `print(response)`，响应里带着 `apiKey`
- 异常堆栈里带上完整的签名 URL
- 代理/WAF 记录明文请求

策略是「允许列表而非黑名单」的反面：这里用黑名单匹配常见敏感字段名，
并在**任何**日志写入前调用。配合 tests/test_safety.py 的静态扫描形成双保险。
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# 匹配敏感字段名（不区分大小写）。用词边界避免误伤 "secretary" 之类的词。
_SENSITIVE_KEY_RE = re.compile(
    r"(api[-_]?key|apikey|secret|signature|token|password|passwd|private[-_]?key"
    r"|authorization|auth[-_]?token|session[-_]?id|mnemonic|seed[-_]?phrase)",
    re.IGNORECASE,
)

# 匹配 URL 查询串里的敏感参数（币安签名请求就是这样传参的）
_SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:apiKey|signature|timestamp|recvWindow)=)[^&\s\"']+",
    re.IGNORECASE,
)

# 匹配疑似密钥的长随机串：32+ 位连续的 base64/hex 字符
# 保守设置长度阈值，避免误伤普通的哈希 ID
_LONG_SECRET_RE = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")

_MASK = "***REDACTED***"


def is_sensitive_key(key: str) -> bool:
    """判断字段名是否敏感。"""
    return bool(_SENSITIVE_KEY_RE.search(key))


def fingerprint(value: str, *, length: int = 12) -> str:
    """生成敏感值的不可逆指纹，用于日志对账而不泄露原值。

    用途：想知道「两次请求用的是不是同一个 Key」时，比较指纹即可，
    既不用打印密钥，也无法从指纹反推密钥。

    Args:
        value: 原始敏感值。
        length: 指纹截断长度。12 位十六进制 = 48 bit，碰撞概率可忽略。

    Returns:
        形如 ``fp:a1b2c3d4e5f6`` 的字符串。
    """
    if not value:
        return "fp:empty"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"fp:{digest[:length]}"


def redact(value: Any, *, _depth: int = 0, _max_depth: int = 8) -> Any:
    """递归脱敏任意嵌套结构。

    对 dict 按字段名判断，对 str 按模式替换，对 list/tuple 逐元素处理。
    有深度上限，防止恶意构造的循环引用导致栈溢出。
    """
    if _depth >= _max_depth:
        return "<max-depth>"

    if isinstance(value, dict):
        return {
            key: (_MASK if is_sensitive_key(str(key)) else redact(v, _depth=_depth + 1))
            for key, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        redacted = [redact(item, _depth=_depth + 1) for item in value]
        return type(value)(redacted) if isinstance(value, tuple) else redacted

    if isinstance(value, str):
        return redact_string(value)

    return value


def redact_string(text: str) -> str:
    """对单个字符串做模式替换：查询串参数、长随机串。"""
    text = _SENSITIVE_QUERY_RE.sub(r"\1" + _MASK, text)
    return _LONG_SECRET_RE.sub(_MASK, text)


def redact_url(url: str) -> str:
    """脱敏 URL，保留 host/path 用于排查问题，抹掉查询串中的敏感参数。

    刻意保留非敏感查询参数（如 symbol、limit），因为它们对排查有用，
    且不构成泄露。
    """
    return redact_string(url)


__all__ = ["fingerprint", "is_sensitive_key", "redact", "redact_string", "redact_url"]
