"""日志配置。

核心要求：**任何日志记录都不能泄露密钥。**

实现方式是给 root logger 挂一个 ``RedactionFilter``，而不是依赖每个调用点
自己注意。这样即使有人写 ``logger.info("resp=%s", response.text)``，
其中的 ``apiKey=...`` 也会在写入前被替换掉。

结构化模式输出 JSON Lines，每条一行，便于 ``jq`` 或日志系统解析；
非结构化模式输出人类可读格式，便于本地开发。
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .redact import redact, redact_string

#: 标准 LogRecord 自带字段，结构化输出时需要排除
_RESERVED_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)


class RedactionFilter(logging.Filter):
    """在日志写出前，对消息与结构化字段做脱敏。

    两个位置都要处理：
    1. ``record.msg`` —— 消息模板本身可能带 URL
    2. ``record.args`` —— ``logger.info("%s", resp)`` 的参数可能是整个响应体
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_string(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = redact(record.args)
            else:
                record.args = tuple(redact(a) for a in record.args)

        # 异常文本同样可能包含完整的签名 URL
        if record.exc_text:
            record.exc_text = redact_string(record.exc_text)

        return True


class JsonLinesFormatter(logging.Formatter):
    """把日志记录格式化为单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        if record.exc_info:
            payload["exc"] = redact_string(self.formatException(record.exc_info))

        # 附加的自定义字段（通过 logger.info(..., extra={...}) 传入）
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = redact(value)

        return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))


class HumanFormatter(logging.Formatter):
    """人类可读的本地开发格式。"""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
            datefmt="%H:%M:%S",
        )


def setup_logging(
    *,
    level: str = "INFO",
    log_dir: Path | None = None,
    structured: bool = True,
    console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """配置 root logger。

    Args:
        level: 日志级别名。
        log_dir: 日志目录。为 None 则只输出到控制台。
        structured: True 输出 JSON Lines，False 输出人类可读。
        console: 是否同时输出到 stderr。
        max_bytes: 单文件轮转大小。
        backup_count: 保留的轮转文件数。

    Returns:
        配置好的 root logger。
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 幂等：重复调用不叠加 handler（否则日志会重复 N 遍）
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter: logging.Formatter = JsonLinesFormatter() if structured else HumanFormatter()
    redactor = RedactionFilter()

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        stream.addFilter(redactor)
        root.addHandler(stream)

    if log_dir is not None:
        log_path = Path(log_dir)
        try:
            log_path.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_path / "cointrader.log",
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(redactor)
            root.addHandler(file_handler)
        except OSError as exc:
            # 日志目录不可写不应导致程序崩溃，但必须让人知道
            root.warning("无法创建日志文件 %s: %s（仅输出到控制台）", log_path, exc)

    # 第三方库的噪音抑制
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


__all__ = ["HumanFormatter", "JsonLinesFormatter", "RedactionFilter", "setup_logging"]
