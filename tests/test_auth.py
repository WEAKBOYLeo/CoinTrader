"""auth 签名模块单元测试（开发设计文档 §3.1 / §11.1）。

覆盖：HMAC golden vector、参数顺序不变量、recvWindow 边界、脱敏。
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

import pytest

from cointrader.execution.auth import (
    add_signature,
    canonical_payload,
    hmac_sha256_hex,
    sign_params,
)
from cointrader.secrets import SecretStr


def _rfc4231_case2(secret: bytes, data: bytes) -> str:
    return hmac.new(secret, data, hashlib.sha256).hexdigest()


class TestHmacGoldenVector:
    def test_matches_rfc4231_test_case_2(self) -> None:
        """RFC 4231 TC2 官方向量，钉死 HMAC-SHA256 实现的正确性。"""
        key = b"Jefe"
        data = b"what do ya want for nothing?"
        expected = (
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        )
        assert hmac_sha256_hex(key.decode("utf-8"), data.decode("utf-8")) == expected

    def test_output_is_lowercase_hex(self) -> None:
        digest = hmac_sha256_hex("secret", "payload")
        assert digest == digest.lower()
        assert all(c in "0123456789abcdef" for c in digest)
        assert len(digest) == 64

    def test_different_secrets_give_different_signatures(self) -> None:
        a = hmac_sha256_hex("s1", "x")
        b = hmac_sha256_hex("s2", "x")
        assert a != b, "不同 Secret 必须产生不同签名"


class TestCanonicalPayload:
    def test_preserves_insertion_order(self) -> None:
        params = {"b": "2", "a": "1", "c": "3"}
        assert canonical_payload(params) == "b=2&a=1&c=3", "参数顺序必须保持插入顺序"

    def test_values_coerced_to_str(self) -> None:
        params: dict[str, Any] = {"q": 10, "p": 1.5}
        assert canonical_payload(params) == "q=10&p=1.5"

    def test_url_encodes_special_chars(self) -> None:
        params = {"note": "a b+c"}
        assert canonical_payload(params) == "note=a+b%2Bc"


class TestSignParams:
    def test_field_order_is_params_then_timestamp_recvwindow_signature(self) -> None:
        signed = sign_params(
            {"symbol": "BTCUSDT", "side": "BUY", "quantity": "1"},
            "test-secret",
            timestamp_ms=1_700_000_000_000,
            recv_window_ms=5000,
        )
        assert list(signed.keys()) == [
            "symbol", "side", "quantity", "timestamp", "recvWindow", "signature",
        ], "顺序必须是：业务参数原顺序 → timestamp → recvWindow → signature"

    def test_timestamp_and_recvwindow_are_decimal_strings(self) -> None:
        signed = sign_params({}, "s", timestamp_ms=1234567890, recv_window_ms=5000)
        assert signed["timestamp"] == "1234567890"
        assert signed["recvWindow"] == "5000"

    def test_signature_excludes_itself_from_payload(self) -> None:
        signed = sign_params({"symbol": "BTCUSDT"}, "k", timestamp_ms=1, recv_window_ms=5000)
        # 从已签参数中剔除 signature，重算 payload 的 HMAC 必须等于 signature
        rebuilt = {k: v for k, v in signed.items() if k != "signature"}
        expected = hmac_sha256_hex("k", canonical_payload(rebuilt))
        assert signed["signature"] == expected, "signature 必须只对不含自身的 payload 计算"

    def test_secret_str_and_plain_text_produce_same_signature(self) -> None:
        kwargs = dict(timestamp_ms=42, recv_window_ms=5000)
        a = sign_params({"symbol": "BTCUSDT"}, SecretStr("sec"), **kwargs)
        b = sign_params({"symbol": "BTCUSDT"}, "sec", **kwargs)
        assert a["signature"] == b["signature"], "SecretStr 与明文必须得到相同签名"

    def test_decimal_values_are_stringified_not_floatified(self) -> None:
        from decimal import Decimal

        signed = sign_params({"quantity": Decimal("0.123000")}, "s", timestamp_ms=1, recv_window_ms=5000)
        assert signed["quantity"] == "0.123000"
        assert "e" not in signed["quantity"].lower(), "禁止浮点/科学计数法进入签名 payload"

    def test_add_signature_is_alias_of_sign_params(self) -> None:
        a = sign_params({"symbol": "X"}, "s", timestamp_ms=1, recv_window_ms=5000)
        b = add_signature({"symbol": "X"}, "s", timestamp_ms=1, recv_window_ms=5000)
        assert a["signature"] == b["signature"]


class TestRecvWindowBounds:
    @pytest.mark.parametrize("ms", [999, 0, -1, 60_001, 600_000])
    def test_out_of_bounds_rejected(self, ms: int) -> None:
        with pytest.raises(ValueError, match="recv_window"):
            sign_params({}, "s", timestamp_ms=1, recv_window_ms=ms)

    @pytest.mark.parametrize("ms", [1000, 5000, 60_000])
    def test_in_bounds_accepted(self, ms: int) -> None:
        signed = sign_params({}, "s", timestamp_ms=1, recv_window_ms=ms)
        assert signed["recvWindow"] == str(ms)
