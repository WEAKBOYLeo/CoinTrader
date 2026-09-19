"""HMAC-SHA256 签名 —— 独立模块，便于用官方示例做 golden 测试。

安全规则（开发设计文档 §3.1）：

1. 签名输入的参数顺序 = 调用方传入的 dict 插入顺序（Python dict 保序）。
   本模块**不**对参数排序，因为排序规则必须与发送时完全一致；
   把"顺序"这个不变量集中在这里，用 golden 测试钉死。
2. 参数值必须**先**用 ``format_decimal`` 转成十进制字符串，
   禁止把 float 直接塞进签名 payload。
3. 签名 = HMAC-SHA256(secret, urlencode(params))，小写十六进制。
4. 本模块不发起任何网络请求，不持有凭证，不做日志 ——
   唯一的输入是显式传入的 secret 明文（调用点负责不外泄）。
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from urllib.parse import urlencode

from ..secrets import SecretStr

__all__ = ["add_signature", "canonical_payload", "hmac_sha256_hex", "sign_params"]


def hmac_sha256_hex(secret: str, payload: str) -> str:
    """对 payload 做 HMAC-SHA256，返回小写十六进制。"""
    digest = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


def canonical_payload(params: Mapping[str, str]) -> str:
    """把参数序列化成用于签名（与发送）的 query string。

    顺序 = dict 插入顺序。值必须已是字符串（数字必须先格式化）。
    """
    return urlencode({key: str(value) for key, value in params.items()})


def sign_params(
    params: Mapping[str, object],
    secret: SecretStr | str,
    *,
    timestamp_ms: int,
    recv_window_ms: int,
) -> dict[str, str]:
    """构造已签名的请求参数。

    固定顺序：调用方原有参数（原顺序）→ timestamp → recvWindow → signature。

    Args:
        params: 业务参数。值必须是 str/int/Decimal 这类可 ``str()`` 的标量。
            数量/价格必须已经用 ``rules.format_decimal`` 归一化。
        secret: API Secret（SecretStr 或明文，明文调用点负责不外泄）。
        timestamp_ms: 请求时间戳（毫秒，应已加上 server time 偏移）。
        recv_window_ms: recvWindow。必须 1000~60000，初始建议 5000。

    Returns:
        可直接作为 httpx 请求 params 的 dict（含 signature）。
    """
    if not 1_000 <= recv_window_ms <= 60_000:
        raise ValueError(f"recv_window_ms 必须在 [1000, 60000] 内，当前 {recv_window_ms}")

    secret_plain = secret.reveal() if isinstance(secret, SecretStr) else secret

    signed: dict[str, str] = {key: str(value) for key, value in params.items()}
    signed["timestamp"] = str(int(timestamp_ms))
    signed["recvWindow"] = str(int(recv_window_ms))

    payload = canonical_payload(signed)
    signed["signature"] = hmac_sha256_hex(secret_plain, payload)
    return signed

# 别名：语义更明确的入口
add_signature = sign_params
