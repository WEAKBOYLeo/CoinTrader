"""磁盘缓存 —— 带 TTL 与原子写。

为什么缓存要做得这么讲究：

1. **限流是有配额的。** 币安权重 2400/分钟，全市场扫描一次就是几百个请求。
   没有缓存意味着每次调试都重新烧配额，很容易把自己送进 418 封禁。

2. **数据可复现性。** 回测结论必须能在三个月后复现。如果每次拉的数据
   都不同（币安会修订历史资金费），结论就无法复核。缓存固化了数据集。

3. **原子写不是洁癖。** 如果进程在写 800KB JSON 的中途被 Ctrl-C，
   留下的是半个文件。下次读它会 JSONDecodeError，或者更糟 —— 静默读出
   残缺数据参与回测。写临时文件再 ``os.replace`` 在 POSIX 上是原子的。

缓存布局::

    <cache_dir>/<namespace>/<key>.json          数据与元信息
    <cache_dir>/<namespace>/<key>.lock          (仅调试用, 不参与逻辑)

key 由参数哈希生成，保证不同参数天然隔离。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ParseError

logger = logging.getLogger(__name__)

#: 缓存格式版本。数据结构变更时递增，让旧缓存自动失效。
CACHE_FORMAT_VERSION = 1


def make_key(*parts: Any) -> str:
    """由任意可序列化参数生成稳定的缓存键。

    使用 sort_keys + 紧凑分隔符，保证 ``{"a":1,"b":2}`` 与
    ``{"b":2,"a":1}`` 生成同一个键。
    """
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """一条缓存记录。"""

    data: Any
    created_at: float
    ttl: int | None

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def is_fresh(self) -> bool:
        """TTL 为 None 表示永不过期（历史数据）。"""
        if self.ttl is None:
            return True
        return self.age_seconds < self.ttl


class DiskCache:
    """基于文件系统的 JSON 缓存。

    Args:
        root: 缓存根目录。
        enabled: 设为 False 则所有读写退化为空操作（调试"从零拉取"时有用）。
    """

    def __init__(self, root: Path | str, *, enabled: bool = True) -> None:
        self.root = Path(root).expanduser()
        self.enabled = enabled
        if self.enabled:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("无法创建缓存目录 %s: %s（缓存已禁用）", self.root, exc)
                self.enabled = False

    # -- 路径 ---------------------------------------------------------------

    def _path(self, namespace: str, key: str) -> Path:
        # namespace 直接来自调用点，做一次白名单字符校验防止路径穿越
        if not namespace or "/" in namespace or "\\" in namespace or ".." in namespace:
            raise ValueError(f"非法缓存 namespace: {namespace!r}")
        if not key or "/" in key or "\\" in key:
            raise ValueError(f"非法缓存 key: {key!r}")
        return self.root / namespace / f"{key}.json"

    # -- 读 -----------------------------------------------------------------

    def get(self, namespace: str, key: str) -> CacheEntry | None:
        """读取缓存。不存在、损坏、或版本不符时返回 None。"""
        if not self.enabled:
            return None

        path = self._path(namespace, key)
        if not path.is_file():
            return None

        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            # 损坏的缓存直接当作未命中，并删掉避免反复失败
            logger.warning("缓存损坏，将删除 %s: %s", path, exc)
            self._unlink_quietly(path)
            return None

        if not isinstance(payload, dict) or payload.get("v") != CACHE_FORMAT_VERSION:
            logger.debug("缓存版本不符，忽略 %s", path)
            return None

        try:
            return CacheEntry(
                data=payload["data"],
                created_at=float(payload["created_at"]),
                ttl=payload.get("ttl"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("缓存结构异常，将删除 %s: %s", path, exc)
            self._unlink_quietly(path)
            return None

    def get_fresh(self, namespace: str, key: str) -> Any | None:
        """读取缓存，仅在未过期时返回数据，否则返回 None。"""
        entry = self.get(namespace, key)
        if entry is None or not entry.is_fresh():
            return None
        return entry.data

    # -- 写 -----------------------------------------------------------------

    def put(self, namespace: str, key: str, data: Any, *, ttl: int | None = None) -> None:
        """原子写入缓存。

        Args:
            namespace: 缓存命名空间（子目录）。
            key: 缓存键。
            data: 必须可 JSON 序列化。DataFrame 请先 ``.to_dict()``。
            ttl: 生存期（秒）。None 表示永不过期。
        """
        if not self.enabled:
            return

        path = self._path(namespace, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("无法创建缓存子目录 %s: %s", path.parent, exc)
            return

        payload = {
            "v": CACHE_FORMAT_VERSION,
            "created_at": time.time(),
            "ttl": ttl,
            "data": data,
        }

        try:
            self._atomic_write_json(path, payload)
        except (OSError, TypeError, ValueError) as exc:
            # 缓存写失败不应中断业务逻辑 —— 数据本身是能从网络重新拿的
            logger.warning("缓存写入失败 %s: %s", path, exc)

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        """写临时文件 → fsync → 原子替换。

        fsync 这一步不能省：没有它，断电时可能留下一个长度为 0 或
        内容撕裂的文件（文件系统元数据已更新但数据块未落盘）。
        """
        directory = path.parent
        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"), default=str)
                fh.flush()
                os.fsync(fh.fileno())
            tmp_path.replace(path)  # POSIX 原子替换，等价于 os.replace
        except BaseException:
            # 任何失败（含 KeyboardInterrupt）都要清理临时文件
            tmp_path.unlink(missing_ok=True)
            raise

    # -- 维护 ---------------------------------------------------------------

    @staticmethod
    def _unlink_quietly(path: Path) -> None:
        try:  # noqa: SIM105
            path.unlink(missing_ok=True)
        except OSError:
            # 缓存清理失败不应中断任何业务流程：
            # 删除失败最坏的结果是磁盘上多留一个文件，而抛异常会中断回测。
            # 这里刻意不用 contextlib.suppress —— 显式写明"为什么吞掉"。
            pass

    def invalidate(self, namespace: str, key: str) -> bool:
        """删除单条缓存。返回是否实际删除。"""
        if not self.enabled:
            return False
        path = self._path(namespace, key)
        existed = path.is_file()
        self._unlink_quietly(path)
        return existed

    def purge_namespace(self, namespace: str) -> int:
        """删除整个命名空间。返回删除的文件数。"""
        if not self.enabled:
            return 0
        directory = self.root / namespace
        if not directory.is_dir():
            return 0
        count = 0
        for path in directory.glob("*.json"):
            self._unlink_quietly(path)
            count += 1
        return count

    def stats(self) -> dict[str, int]:
        """统计各命名空间的缓存文件数与总字节数。"""
        result: dict[str, int] = {}
        if not self.enabled or not self.root.is_dir():
            return result
        for namespace_dir in sorted(self.root.iterdir()):
            if not namespace_dir.is_dir():
                continue
            files = list(namespace_dir.glob("*.json"))
            result[namespace_dir.name] = len(files)
        return result

    def total_bytes(self) -> int:
        """缓存占用的总字节数。"""
        if not self.enabled or not self.root.is_dir():
            return 0
        return sum(p.stat().st_size for p in self.root.rglob("*.json") if p.is_file())


def json_or_raise(payload: Any, context: str) -> Any:
    """把 bytes/str 解析为 JSON，失败时抛出带上下文的 ParseError。"""
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", errors="replace")
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ParseError(f"{context}: JSON 解析失败 ({exc}); 前 200 字符: {payload[:200]!r}") from exc
    return payload


__all__ = ["CACHE_FORMAT_VERSION", "CacheEntry", "DiskCache", "json_or_raise", "make_key"]
