"""领域内核公共部分：领域异常与序列化校验工具。

本模块属于 domain 包（跨模块不可变契约）：

- 无 IO：不 import 网络、数据库、Binance、执行层或 WebUI。
- 时间一律 UTC 毫秒（``int``），不使用裸 datetime（DTZ 规则）。
- 金额/数量/费率一律 ``Decimal``；反序列化拒绝二进制浮点数，
  防止浮点精度泄漏进交易参数。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import TypeVar

E = TypeVar("E", bound=Enum)
T = TypeVar("T")


class DomainError(Exception):
    """所有领域契约异常的基类。"""


class InvalidDomainValue(DomainError):
    """领域对象收到非法字段值。

    覆盖：缺字段、非法类型、非法枚举、负数名义额、时间逆序等。
    校验失败必须显式抛错，不允许猜值或回填默认值。
    """


class UnknownSchemaVersion(DomainError):
    """事件/manifest 的 schema_version 未知。

    未知版本拒绝解析，不静默猜字段（实施计划书 §5 ``schema_version`` 不变量）。
    """


class ExpiredDomainObject(DomainError):
    """领域对象已过期，不可再提交或审批。"""


class RiskNotApproved(DomainError):
    """风险审批未通过（无 ALLOW/RESIZE 决定），不得进入执行层。"""


def parse_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise InvalidDomainValue(f"{field} 必须是 str，实际是 {type(value).__name__}")
    return value


def parse_int_ms(value: object, field: str) -> int:
    """UTC 毫秒时间戳：必须是 int（拒绝 bool/float）且非负。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidDomainValue(f"{field} 必须是 int 毫秒时间戳，实际是 {type(value).__name__}")
    if value < 0:
        raise InvalidDomainValue(f"{field} 不得为负: {value}")
    return value


def parse_decimal(value: object, field: str) -> Decimal:
    """金额/数量/费率：只接受 ``Decimal`` 或数字字符串，拒绝二进制浮点数。"""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, str):
        try:
            return Decimal(value)
        except InvalidOperation as exc:
            raise InvalidDomainValue(f"{field} 不是合法十进制数: {value!r}") from exc
    raise InvalidDomainValue(f"{field} 必须是 Decimal 或数字字符串，实际是 {type(value).__name__}")


def parse_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise InvalidDomainValue(f"{field} 必须是 bool，实际是 {type(value).__name__}")
    return value


def parse_enum(enum_cls: type[E], value: object, field: str) -> E:
    try:
        return enum_cls(value)
    except (ValueError, TypeError) as exc:
        allowed = ", ".join(m.value for m in enum_cls)
        raise InvalidDomainValue(f"{field} 是非法枚举 {value!r}，允许值: {allowed}") from exc


def parse_optional(value: object, field: str, parse: Callable[[object, str], T]) -> T | None:
    """可选字段：None 原样通过，否则交给对应 parse_*（签名均为 (value, field)）。"""
    if value is None:
        return None
    return parse(value, field)


def reject_unknown_keys(data: Mapping[str, object], allowed: frozenset[str], field: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise InvalidDomainValue(f"{field} 含未知字段: {', '.join(unknown)}")
