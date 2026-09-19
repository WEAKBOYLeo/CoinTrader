"""密钥加载 —— 与 config.py 物理分离的唯一入口。

设计原则：**把密钥处理隔离在一个文件里，这样审计范围就只有这一个文件。**

硬规则（docs/ARCHITECTURE.md §4.2）：
1. 只从环境变量读取
2. 支持从 ``.env`` 文件加载到 ``os.environ``，但**不要求** python-dotenv 依赖
3. 密钥值在 repr / str / 日志中**永不**出现（见 SecretStr.safe_repr）
4. 真实盘密钥默认**不加载**，只有显式开启交易时才读取

对交易系统，密钥进内存的时机越晚越好：如果程序只是跑回测，
BINANCE_API_KEY 根本不会被读进进程。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError
from .redact import fingerprint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 安全闸门常量
# ---------------------------------------------------------------------------

#: 开启真实交易的**精确**魔法字符串。
#: 刻意不写成 "1"/"true"/"yes" —— 那些值可能在复制粘贴或环境继承中意外为真。
#: 必须是人手动敲出来的、有语义的短语。
TRADING_ENABLED_MAGIC = "YES_I_AM_SURE"

#: 检查交易是否开启的环境变量名
TRADING_ENABLED_ENV = "COINTRADER_TRADING_ENABLED"

#: 是否使用测试网（默认 true）
USE_TESTNET_ENV = "COINTRADER_USE_TESTNET"

#: 停机开关文件路径
KILL_SWITCH_ENV = "COINTRADER_KILL_SWITCH_FILE"
DEFAULT_KILL_SWITCH_FILE = "KILL_SWITCH"

#: 密钥的环境变量名。
#:
#: ⚠️ 这些常量是**变量名**，不是密钥值本身。
#: flake8-bandit 的 S105 会因为变量名里含 "SECRET" 而误报，
#: 这里显式豁免并说明理由 —— 用 noqa 而非全局关掉该规则，
#: 这样新代码里真出现硬编码密钥时仍然会被抓到。
REAL_KEY_ENV = "BINANCE_API_KEY"
REAL_SECRET_ENV = "BINANCE_API_SECRET"  # noqa: S105
TESTNET_KEY_ENV = "BINANCE_TESTNET_API_KEY"
TESTNET_SECRET_ENV = "BINANCE_TESTNET_API_SECRET"  # noqa: S105


@dataclass(frozen=True, slots=True, repr=False)
class SecretStr:
    """一个不会在日志/repr/异常里泄露明文的密钥包装。

    ``repr()`` 只输出指纹，所以 ``logger.info("key=%s", secret)`` 是安全的。
    取明文必须显式调用 ``.reveal()``，这样任何明文使用点在代码审查时一眼可见。
    """

    _value: str
    name: str = "secret"

    def reveal(self) -> str:
        """返回明文。**调用点必须确保不外泄。**"""
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __repr__(self) -> str:
        return f"SecretStr(name={self.name!r}, {fingerprint(self._value)})"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:  # noqa: ARG002
        # 防止 f"{secret}" 意外泄露
        return repr(self)


@dataclass(frozen=True, slots=True)
class ApiCredentials:
    """一对 API 凭证，附带来源标记用于审计。"""

    key: SecretStr
    secret: SecretStr
    is_testnet: bool

    def __repr__(self) -> str:
        env = "testnet" if self.is_testnet else "MAINNET"
        return f"ApiCredentials({env}, key={self.key!r}, secret={self.secret!r})"


# ---------------------------------------------------------------------------
# .env 加载（极简实现，避免引入 python-dotenv 依赖）
# ---------------------------------------------------------------------------


def load_dotenv(path: Path | str = ".env", *, override: bool = False) -> int:
    """把 .env 文件里的键值对加载到 ``os.environ``。

    实现刻意保守：
    - 只支持 ``KEY=VALUE`` 和 ``export KEY=VALUE``
    - 支持 ``#`` 注释与成对引号
    - **已存在的环境变量默认不覆盖**（环境变量优先级高于文件，
      这样 ``COINTRADER_TRADING_ENABLED=... python -m ...`` 可以临时覆盖）

    Returns:
        实际写入的变量个数。
    """
    env_path = Path(path)
    if not env_path.is_file():
        return 0

    count = 0
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        # 去掉成对引号
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            count += 1

    return count


# ---------------------------------------------------------------------------
# 闸门状态
# ---------------------------------------------------------------------------


def is_trading_enabled() -> bool:
    """真实/测试网交易是否被显式开启。

    ⚠️ 必须**精确**等于魔法字符串。这个严格性是有意的：
    它让「不小心开启」变得不可能，同时让「有意开启」在 shell 历史里
    留下一眼可见的记录。
    """
    return os.environ.get(TRADING_ENABLED_ENV, "") == TRADING_ENABLED_MAGIC


def use_testnet() -> bool:
    """是否使用测试网。默认 ``True`` —— 缺失该变量时走测试网。

    只有显式写成 "false"/"0"/"no" 才切到真实盘。
    """
    raw = os.environ.get(USE_TESTNET_ENV, "").strip().lower()
    if raw == "":
        return True  # 默认安全
    return raw not in {"false", "0", "no", "off"}


def kill_switch_path() -> Path:
    """停机开关文件路径。"""
    return Path(os.environ.get(KILL_SWITCH_ENV, DEFAULT_KILL_SWITCH_FILE)).expanduser()


def is_kill_switch_engaged() -> bool:
    """停机开关是否已触发（即该文件是否存在）。

    文件系统级检查，不需要改代码或重启进程就能立即停机。
    """
    return kill_switch_path().exists()


# ---------------------------------------------------------------------------
# 凭证读取
# ---------------------------------------------------------------------------


def load_credentials(*, require: bool = True) -> ApiCredentials | None:
    """从环境变量加载 API 凭证。

    Args:
        require: 为 True 时，缺失凭证抛 ConfigError；为 False 时返回 None。

    Returns:
        ApiCredentials，或 None（当 require=False 且凭证缺失时）。

    Raises:
        ConfigError: 启用真实盘但凭证不全，或 require=True 但凭证缺失。

    Notes:
        当既未开启交易、又未提供任何凭证时，返回 None —— 这让回测流程
        完全不需要任何密钥就能跑通。
    """
    on_testnet = use_testnet()
    trading = is_trading_enabled()

    if on_testnet:
        key_env, secret_env = TESTNET_KEY_ENV, TESTNET_SECRET_ENV
    else:
        key_env, secret_env = REAL_KEY_ENV, REAL_SECRET_ENV

        # 真实盘的双重确认：不仅要 COINTRADER_USE_TESTNET=false，
        # 还必须有交易开关。少任何一个都拒绝加载真实密钥。
        if not trading:
            raise ConfigError(
                f"检测到 {USE_TESTNET_ENV}=false（真实盘），但 "
                f"{TRADING_ENABLED_ENV} 未设置为 '{TRADING_ENABLED_MAGIC}'。"
                "加载真实盘密钥需要两道开关同时打开，这是有意的保护。"
            )

    raw_key = os.environ.get(key_env, "").strip()
    raw_secret = os.environ.get(secret_env, "").strip()

    if not raw_key and not raw_secret:
        if require:
            raise ConfigError(
                f"未找到 API 凭证。请设置环境变量 {key_env} 与 {secret_env}。"
                f"（当前模式: {'测试网' if on_testnet else '真实盘'}）"
            )
        return None

    if not raw_key or not raw_secret:
        missing = key_env if not raw_key else secret_env
        raise ConfigError(f"API 凭证不完整，缺少 {missing}")

    creds = ApiCredentials(
        key=SecretStr(raw_key, name=key_env),
        secret=SecretStr(raw_secret, name=secret_env),
        is_testnet=on_testnet,
    )

    # 日志只记录指纹与模式，永不记录明文。
    logger.info("已加载 API 凭证 %s (mode=%s)", creds, "testnet" if on_testnet else "MAINNET")
    return creds


def describe_security_posture() -> dict[str, object]:
    """汇总当前安全状态，供启动横幅与审计日志使用。

    这是一个「一眼看懂现在处于什么模式」的函数，输出中**不含**任何明文密钥。
    """
    on_testnet = use_testnet()
    trading = is_trading_enabled()
    ks_path = kill_switch_path()

    key_env = TESTNET_KEY_ENV if on_testnet else REAL_KEY_ENV
    has_key = bool(os.environ.get(key_env, "").strip())

    # 真实盘是否可达：需要关掉测试网 + 打开交易开关 + 密钥存在
    mainnet_reachable = (not on_testnet) and trading and has_key

    return {
        "use_testnet": on_testnet,
        "trading_enabled": trading,
        "kill_switch_file": str(ks_path),
        "kill_switch_engaged": ks_path.exists(),
        "has_api_key": has_key,
        "mainnet_reachable": mainnet_reachable,
        "mode": "MAINNET" if mainnet_reachable else ("TESTNET" if on_testnet else "LOCKED"),
    }


__all__ = [
    "ApiCredentials",
    "SecretStr",
    "describe_security_posture",
    "is_kill_switch_engaged",
    "is_trading_enabled",
    "kill_switch_path",
    "load_credentials",
    "load_dotenv",
    "use_testnet",
]
