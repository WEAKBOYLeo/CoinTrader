"""安全约束的静态强制。

**这些测试失败 = 构建失败。** 安全不能靠开发者自觉，要靠测试卡住。

本文件检查两大类约束：

**A. 静态结构约束**（源码字符串/import 分析）
   A1. 数据层不含任何签名/鉴权代码
   A2. 研究层与回测层不引入网络库
   A3. 底层不 import 执行层（依赖方向单向）
   A4. 全仓库无硬编码密钥
   A5. 配置文件中无密钥字段

**B. 运行时行为约束**（实际调用闸门）
   B1. 默认拒绝一切下单
   B2. 环境变量的写法必须**精确**匹配
   B3. 停机开关立即生效
   B4. 审计日志记录被拒绝的尝试
   B5. 密钥不在日志/repr 中泄露

之所以静态和动态都要查：静态检查能抓住"代码里写死了密钥"这类问题，
但抓不住"闸门逻辑写反了"；动态检查反之。两者互补。
"""

from __future__ import annotations

import ast
import logging
import os
import re
from pathlib import Path

import pytest

from cointrader.config import RiskConfig
from cointrader.execution.broker import Broker, SimulatedExchange
from cointrader.execution.guard import (
    Guard,
    Market,
    OrderIntent,
    OrderSide,
    OrderType,
)
from cointrader.execution.risk import Position, RiskState
from cointrader.redact import fingerprint, redact, redact_url
from cointrader.secrets import (
    TRADING_ENABLED_ENV,
    TRADING_ENABLED_MAGIC,
    SecretStr,
    describe_security_posture,
    is_kill_switch_engaged,
    is_trading_enabled,
    load_credentials,
    load_dotenv,
)

# ---------------------------------------------------------------------------
# 静态约束 A
# ---------------------------------------------------------------------------

#: 数据层禁止出现的敏感关键词。数据层不应有任何签名能力。
FORBIDDEN_IN_DATA_LAYER = (
    "apikey",
    "api_key",
    "secret",
    "signature",
    "signed",
    "hmac",
    "X-MBX-APIKEY",
    "recvwindow",
)

#: 研究层与回测层禁止 import 的网络库
FORBIDDEN_NETWORK_IMPORTS = frozenset(
    {"httpx", "requests", "urllib", "urllib3", "aiohttp", "http", "socket"}
)

#: 疑似硬编码密钥的模式：40+ 位连续随机字符
_SECRET_LIKE_RE = re.compile(r"""["']([A-Za-z0-9+/]{40,}={0,2})["']""")

#: 允许出现的长随机串（测试夹具里的假数据、哈希常量等）
_ALLOWED_SECRET_LIKE = frozenset({
    # RFC 4231 HMAC-SHA256 官方测试向量（test_auth.py golden vector，非凭证）
    "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
})


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _iter_imports(tree: ast.AST) -> list[str]:
    """收集一个模块所有 import 的顶层包名。"""
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # 跳过相对 import（level > 0 表示 from . import x）
            if node.level == 0 and node.module:
                modules.append(node.module.split(".")[0])
    return modules


class TestStaticConstraints:
    """A. 静态结构约束。"""

    def test_data_layer_has_no_signing_code(self, src_root: Path) -> None:
        """A1. 数据层不得包含任何签名/鉴权相关代码。

        这是「数据层不可能下单」这个保证的根基。如果这条测试红了，
        说明有人往数据层加了需要密钥的能力 —— 必须改成放在 execution 层。
        """
        data_dir = src_root / "data"
        assert data_dir.is_dir(), f"数据层目录不存在: {data_dir}"

        violations: list[str] = []
        for path in _python_files(data_dir):
            content = path.read_text(encoding="utf-8")
            lowered = content.lower()
            for keyword in FORBIDDEN_IN_DATA_LAYER:
                if keyword.lower() in lowered:
                    # 找出具体行号，方便定位
                    for lineno, line in enumerate(content.splitlines(), start=1):
                        if keyword.lower() in line.lower():
                            violations.append(
                                f"{path.relative_to(src_root)}:{lineno} 含 '{keyword}': {line.strip()[:90]}"
                            )

        assert not violations, (
            "数据层出现了签名/鉴权相关代码。数据层必须保持只读、无凭证。\n"
            "如需下单能力，请放到 execution/ 层。\n"
            "违规项:\n  " + "\n  ".join(violations)
        )

    def test_research_layer_has_no_network_imports(self, src_root: Path) -> None:
        """A2a. 研究层不得引入网络库（研究逻辑必须是纯计算）。"""
        self._assert_no_network_imports(src_root / "research", "研究层")

    def test_backtest_layer_has_no_network_imports(self, src_root: Path) -> None:
        """A2b. 回测层不得引入网络库（回测必须确定性、可离线重跑）。"""
        self._assert_no_network_imports(src_root / "backtest", "回测层")

    def _assert_no_network_imports(self, directory: Path, label: str) -> None:
        assert directory.is_dir(), f"{label}目录不存在: {directory}"
        violations: list[str] = []

        for path in _python_files(directory):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for module in _iter_imports(tree):
                if module in FORBIDDEN_NETWORK_IMPORTS:
                    violations.append(f"{path.relative_to(directory.parent)} 引入了 '{module}'")

        assert not violations, (
            f"{label}不得进行网络 IO。这保证了该层可以完整离线测试。\n"
            "违规项:\n  " + "\n  ".join(violations)
        )

    def test_lower_layers_do_not_import_execution(self, src_root: Path) -> None:
        """A3. 依赖方向单向：data/research/backtest 不得 import execution。

        若这条被破坏，一次误 import 就可能让数据层获得下单能力。
        """
        violations: list[str] = []

        for layer in ("data", "research", "backtest"):
            directory = src_root / layer
            if not directory.is_dir():
                continue
            for path in _python_files(directory):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if "execution" in node.module:
                            violations.append(
                                f"{path.relative_to(src_root)} import 了 '{node.module}'"
                            )
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if "execution" in alias.name:
                                violations.append(
                                    f"{path.relative_to(src_root)} import 了 '{alias.name}'"
                                )

        assert not violations, (
            "依赖方向被破坏：下层不得 import execution 层。\n"
            "违规项:\n  " + "\n  ".join(violations)
        )

    def test_no_hardcoded_secrets(self, src_root: Path, project_root: Path) -> None:
        """A4. 全仓库不得出现硬编码的密钥字样。

        币安 API Key 是 64 位字母数字串，Secret 也是。用长度 + 字符集
        作为启发式检测。误报可以通过在代码里拼接字符串来规避。
        """
        violations: list[str] = []

        for path in _python_files(src_root) + _python_files(project_root / "tests"):
            content = path.read_text(encoding="utf-8")
            for match in _SECRET_LIKE_RE.finditer(content):
                candidate = match.group(1)
                if candidate in _ALLOWED_SECRET_LIKE:
                    continue
                # 跳过明显的非密钥（全数字的长 ID、纯重复字符）
                if candidate.isdigit():
                    continue
                if len(set(candidate)) < 6:
                    continue
                lineno = content[: match.start()].count("\n") + 1
                violations.append(f"{path.name}:{lineno} 疑似硬编码密钥: {candidate[:16]}...")

        assert not violations, (
            "检测到疑似硬编码密钥。密钥必须来自环境变量，见 docs/ARCHITECTURE.md §4.2。\n"
            "违规项:\n  " + "\n  ".join(violations)
        )

    def test_config_file_has_no_secret_fields(self, project_root: Path) -> None:
        """A5. config.yaml 中不得出现密钥字段。"""
        import yaml

        config_path = project_root / "config" / "config.yaml"
        assert config_path.is_file(), f"配置文件不存在: {config_path}"

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        def walk(node: object, path: str = "") -> list[str]:
            found: list[str] = []
            if isinstance(node, dict):
                for key, value in node.items():
                    key_text = str(key).lower()
                    if any(token in key_text for token in ("apikey", "api_key", "secret", "password", "token")):
                        found.append(f"{path}.{key}")
                    found.extend(walk(value, f"{path}.{key}"))
            elif isinstance(node, list):
                for index, item in enumerate(node):
                    found.extend(walk(item, f"{path}[{index}]"))
            return found

        hits = walk(raw)
        assert not hits, f"配置文件中出现密钥字段: {hits}。密钥只能来自环境变量。"

    def test_gitignore_covers_secrets(self, project_root: Path) -> None:
        """A6. .gitignore 必须忽略 .env 与 KILL_SWITCH 文件。"""
        gitignore = (project_root / ".gitignore").read_text(encoding="utf-8")
        for pattern in (".env", "KILL_SWITCH"):
            assert pattern in gitignore, f".gitignore 未忽略 {pattern}"
        # 必须为 .env.example 开白名单，否则模板提交不上去
        assert "!.env.example" in gitignore, ".gitignore 未为 .env.example 开白名单"


# ---------------------------------------------------------------------------
# 运行时约束 B
# ---------------------------------------------------------------------------


class TestGuardDefaults:
    """B1-B2. 默认拒绝与环境变量精确匹配。"""

    def test_guard_denies_by_default(self, clean_env, risk_config: RiskConfig) -> None:
        """B1a. 全新环境下，任何下单都应被拒绝。

        这是本项目最重要的一条测试。它红了意味着默认安全假设不成立。
        """
        guard = Guard(risk_config, audit_path="/dev/null")
        intent = OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            market=Market.SPOT,
            order_type=OrderType.MARKET,
            quantity=0.01,
            price=70000.0,
        )
        result = guard.authorize(intent, RiskState(total_capital=10000.0))

        assert not result.authorized, "全新环境下下单不应被授权"
        assert "交易未启用" in result.reason

    def test_broker_defaults_to_dry_run(self, clean_env, risk_config: RiskConfig) -> None:
        """B1b. Broker 默认 dry_run=True，且默认不加载凭证。"""
        broker = Broker(risk_config, audit_path="/dev/null")
        assert broker.dry_run is True
        assert broker.credentials is None
        status = broker.status()
        assert status["dry_run"] is True
        assert status["has_credentials"] is False

    @pytest.mark.parametrize(
        "wrong_value",
        ["1", "true", "True", "TRUE", "yes", "YES", "on", "enabled", "YES_I_AM_SURE ", " YES_I_AM_SURE", ""],
    )
    def test_trading_enabled_requires_exact_match(
        self, clean_env, monkeypatch: pytest.MonkeyPatch, wrong_value: str
    ) -> None:
        """B2. 交易开关必须**精确**等于魔法字符串。

        这条测试列举了所有"看起来对"但不应生效的写法。
        容忍任何一个是安全漏洞（环境变量很容易被意外继承或写错）。
        """
        monkeypatch.setenv(TRADING_ENABLED_ENV, wrong_value)
        assert not is_trading_enabled(), f"'{wrong_value}' 不应被当作启用交易的标准写法"

    def test_trading_enabled_accepts_exact_magic(
        self, clean_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B2 反面：精确值必须能生效（否则这道闸门就永远打不开了）。"""
        monkeypatch.setenv(TRADING_ENABLED_ENV, TRADING_ENABLED_MAGIC)
        assert is_trading_enabled()

    def test_use_testnet_defaults_to_true(self, clean_env, monkeypatch: pytest.MonkeyPatch) -> None:
        """B2c. 未设置 USE_TESTNET 时必须默认走测试网。"""
        from cointrader.secrets import use_testnet

        assert use_testnet() is True, "缺失该变量时必须默认走测试网"

        # 只有显式的 false 类值才切到真实盘
        monkeypatch.setenv("COINTRADER_USE_TESTNET", "false")
        assert use_testnet() is False

    def test_mainnet_credentials_require_double_switch(
        self, clean_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B2d. 加载真实盘密钥需要两道开关同时打开。"""
        monkeypatch.setenv("COINTRADER_USE_TESTNET", "false")
        # 有真实密钥但没开交易开关 → 必须拒绝
        monkeypatch.setenv("BINANCE_API_KEY", "x" * 64)
        monkeypatch.setenv("BINANCE_API_SECRET", "y" * 64)

        from cointrader.errors import ConfigError

        with pytest.raises(ConfigError, match="真实盘"):
            load_credentials(require=True)


class TestKillSwitch:
    """B3. 停机开关。"""

    def test_kill_switch_blocks_orders(
        self, clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, risk_config: RiskConfig
    ) -> None:
        """B3a. 停机开关文件存在时，即使交易已启用也必须拒绝。"""
        monkeypatch.setenv(TRADING_ENABLED_ENV, TRADING_ENABLED_MAGIC)
        switch = tmp_path / "KILL_SWITCH"
        switch.write_text("stop")
        monkeypatch.setenv("COINTRADER_KILL_SWITCH_FILE", str(switch))

        assert is_kill_switch_engaged()

        guard = Guard(risk_config, audit_path="/dev/null")
        intent = OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            market=Market.SPOT,
            order_type=OrderType.MARKET,
            quantity=0.01,
            price=70000.0,
        )
        result = guard.authorize(intent, RiskState(total_capital=10000.0))

        assert not result.authorized
        assert "停机开关" in result.reason

    def test_kill_switch_file_absent_by_default(self, clean_env) -> None:
        """B3b. 默认环境下停机开关不应是触发状态。"""
        assert not is_kill_switch_engaged()


class TestRiskLimits:
    """B4. 风控限额独立生效。"""

    def test_order_exceeding_max_notional_rejected(self, clean_env, monkeypatch, risk_config):
        """单笔超过上限必须被拒。"""
        monkeypatch.setenv(TRADING_ENABLED_ENV, TRADING_ENABLED_MAGIC)
        monkeypatch.setenv("COINTRADER_KILL_SWITCH_FILE", "/nonexistent/KILL_SWITCH")

        guard = Guard(risk_config, audit_path="/dev/null")
        # 上限 200，这里下 1000 USDT（0.014 BTC @ 70000）
        intent = OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            market=Market.SPOT,
            order_type=OrderType.MARKET,
            quantity=0.014,
            price=70000.0,
        )
        result = guard.authorize(intent, RiskState(total_capital=10000.0))

        assert not result.authorized
        assert "超过上限" in result.reason

    def test_market_order_without_price_rejected(self, clean_env, monkeypatch, risk_config):
        """MARKET 单缺少参考价时必须拒绝。

        没有价格就无法评估名义额，也就无法执行任何限额检查。
        此时放行等于绕过所有风控。
        """
        monkeypatch.setenv(TRADING_ENABLED_ENV, TRADING_ENABLED_MAGIC)
        monkeypatch.setenv("COINTRADER_KILL_SWITCH_FILE", "/nonexistent/KILL_SWITCH")

        guard = Guard(risk_config, audit_path="/dev/null")
        intent = OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            market=Market.SPOT,
            order_type=OrderType.MARKET,
            quantity=0.001,
            price=None,
        )
        result = guard.authorize(intent, RiskState(total_capital=10000.0))

        assert not result.authorized
        assert "参考价" in result.reason

    def test_closing_order_bypasses_limits(self, clean_env, monkeypatch, risk_config):
        """平仓单应豁免限额 —— 减仓必须永远可行。

        如果把平仓也拦住，你会在最需要离场的时候被困住。
        """
        monkeypatch.setenv(TRADING_ENABLED_ENV, TRADING_ENABLED_MAGIC)
        monkeypatch.setenv("COINTRADER_KILL_SWITCH_FILE", "/nonexistent/KILL_SWITCH")

        guard = Guard(risk_config, audit_path="/dev/null")
        intent = OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            market=Market.SPOT,
            order_type=OrderType.MARKET,
            quantity=0.014,     # 名义额 980 USDT，超过单笔上限 200
            price=70000.0,
            is_closing=True,
        )
        result = guard.authorize(intent, RiskState(total_capital=10000.0))

        assert result.authorized, "平仓单不应被限额拦住"

    def test_naked_position_blocks_new_orders(self, clean_env, monkeypatch, risk_config):
        """存在未对冲持仓时，禁止开新仓。

        裸头寸是资金费套利唯一会真正亏大钱的方式，
        必须先把手里的烂摊子收拾干净。
        """
        monkeypatch.setenv(TRADING_ENABLED_ENV, TRADING_ENABLED_MAGIC)
        monkeypatch.setenv("COINTRADER_KILL_SWITCH_FILE", "/nonexistent/KILL_SWITCH")

        state = RiskState(
            total_capital=10000.0,
            positions={
                "BTCUSDT": Position(
                    symbol="BTCUSDT",
                    spot_qty=0.01,
                    perp_qty=0.0,      # 永续腿缺失 → 裸多头
                    spot_price=70000.0,
                    perp_price=70000.0,
                )
            },
        )

        guard = Guard(risk_config, audit_path="/dev/null")
        intent = OrderIntent(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            market=Market.SPOT,
            order_type=OrderType.MARKET,
            quantity=0.001,
            price=3000.0,
        )
        result = guard.authorize(intent, state)

        assert not result.authorized
        assert "未完全对冲" in result.reason


class TestDoubleLegExecution:
    """B5. 双腿执行的失败处理。"""

    def test_open_position_aborts_when_unauthorized(self, clean_env, tmp_path, risk_config):
        """未授权时两腿都不应下单。"""
        exchange = SimulatedExchange()
        broker = Broker(risk_config, exchange=exchange, dry_run=False, audit_path=str(tmp_path / "audit.log"))

        legs = broker.open_position(
            symbol="BTCUSDT",
            quantity=0.001,
            spot_price=70000.0,
            perp_price=70000.0,
            state=RiskState(total_capital=10000.0),
        )

        assert legs.aborted
        assert legs.spot is None and legs.perp is None
        assert exchange.orders == [], "未授权时不应向交易所发送任何订单"

    def test_perp_leg_fails_first_leaves_no_naked_spot(
        self, clean_env, monkeypatch, tmp_path, risk_config
    ) -> None:
        """永续腿失败时，现货腿（尚未下单）不应被下单。

        顺序设计：先开永续。让它先失败，就避免了"现货已买、永续没开"
        的裸多头局面。
        """
        monkeypatch.setenv(TRADING_ENABLED_ENV, TRADING_ENABLED_MAGIC)
        monkeypatch.setenv("COINTRADER_KILL_SWITCH_FILE", "/nonexistent/KILL_SWITCH")

        exchange = SimulatedExchange(fail_on={("BTCUSDT", "SELL")})
        broker = Broker(
            risk_config,
            exchange=exchange,
            dry_run=False,
            audit_path=str(tmp_path / "audit.log"),
        )

        legs = broker.open_position(
            symbol="BTCUSDT",
            quantity=0.001,
            spot_price=70000.0,
            perp_price=70000.0,
            state=RiskState(total_capital=10000.0),
        )

        assert legs.perp is not None and not legs.perp.ok
        assert legs.spot is None, "永续腿失败后不应再下现货腿"
        assert not legs.is_naked
        assert len(exchange.orders) == 1, f"应只尝试了永续腿，实际: {exchange.orders}"

    def test_spot_leg_failure_detected_as_naked(
        self, clean_env, monkeypatch, tmp_path, risk_config
    ) -> None:
        """现货腿失败时必须被识别为裸头寸（用于告警）。"""
        monkeypatch.setenv(TRADING_ENABLED_ENV, TRADING_ENABLED_MAGIC)
        monkeypatch.setenv("COINTRADER_KILL_SWITCH_FILE", "/nonexistent/KILL_SWITCH")

        exchange = SimulatedExchange(fail_on={("BTCUSDT", "BUY")})
        broker = Broker(
            risk_config,
            exchange=exchange,
            dry_run=False,
            audit_path=str(tmp_path / "audit.log"),
        )

        legs = broker.open_position(
            symbol="BTCUSDT",
            quantity=0.001,
            spot_price=70000.0,
            perp_price=70000.0,
            state=RiskState(total_capital=10000.0),
        )

        assert legs.perp is not None and legs.spot is not None and legs.perp.ok and not legs.spot.ok
        assert legs.is_naked, "只有一腿成交时必须被识别为裸头寸"
        assert "裸头寸" in legs.describe()

    def test_dry_run_does_not_touch_exchange(self, clean_env, monkeypatch, tmp_path, risk_config):
        """干跑模式下不应向交易所发送任何真实订单。"""
        monkeypatch.setenv(TRADING_ENABLED_ENV, TRADING_ENABLED_MAGIC)
        monkeypatch.setenv("COINTRADER_KILL_SWITCH_FILE", "/nonexistent/KILL_SWITCH")

        exchange = SimulatedExchange()
        broker = Broker(
            risk_config, exchange=exchange, dry_run=True, audit_path=str(tmp_path / "audit.log")
        )

        legs = broker.open_position(
            symbol="BTCUSDT",
            quantity=0.001,
            spot_price=70000.0,
            perp_price=70000.0,
            state=RiskState(total_capital=10000.0),
        )

        assert legs.both_ok, "干跑应返回模拟成功结果"
        assert exchange.orders == [], "干跑模式绝不能触碰真实 exchange"


class TestAuditLog:
    """B6. 审计日志。"""

    def test_rejected_attempts_are_audited(self, clean_env, tmp_path, risk_config):
        """被拒绝的下单尝试也必须进审计日志。"""
        import json

        audit_path = tmp_path / "audit.log"
        guard = Guard(risk_config, audit_path=audit_path)

        intent = OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            market=Market.SPOT,
            order_type=OrderType.MARKET,
            quantity=0.001,
            price=70000.0,
            reason="测试用下单意图",
        )
        guard.authorize(intent, RiskState(total_capital=10000.0))

        assert audit_path.exists(), "审计日志未创建"
        lines = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        assert len(lines) == 1

        entry = lines[0]
        assert entry["event"] == "ORDER_ATTEMPT"
        assert entry["result"]["authorized"] is False
        assert entry["intent"]["symbol"] == "BTCUSDT"
        assert entry["intent"]["reason"] == "测试用下单意图"
        # 安全状态快照也应记录，便于事后复盘"当时处于什么模式"
        assert "posture" in entry

    def test_audit_log_never_contains_secrets(self, tmp_path, risk_config) -> None:
        """审计日志不得包含密钥明文。"""
        audit_path = tmp_path / "audit.log"
        guard = Guard(risk_config, audit_path=audit_path)

        secret_value = "SUPERSECRETVALUE" + "x" * 40
        intent = OrderIntent(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            market=Market.SPOT,
            order_type=OrderType.MARKET,
            quantity=0.001,
            price=70000.0,
            reason=f"测试 apiKey={secret_value}",
        )
        guard.authorize(intent, RiskState(total_capital=10000.0))

        content = audit_path.read_text()
        assert secret_value not in content, "审计日志泄露了密钥明文"


class TestRedaction:
    """B7. 脱敏工具。"""

    def test_redact_masks_sensitive_keys(self) -> None:
        payload = {
            "apiKey": "abc123",
            "api_key": "def456",
            "secret": "ghi789",
            "signature": "jkl012",
            "symbol": "BTCUSDT",        # 非敏感，应保留
            "limit": 100,
        }
        result = redact(payload)

        for key in ("apiKey", "api_key", "secret", "signature"):
            assert result[key] == "***REDACTED***", f"{key} 未被脱敏"
        assert result["symbol"] == "BTCUSDT", "非敏感字段不应被改动"
        assert result["limit"] == 100

    def test_redact_handles_nested_structures(self) -> None:
        payload = {
            "outer": {"inner": {"token": "secret-token", "value": 1}},
            "list": [{"password": "pw"}, "plain"],
        }
        result = redact(payload)
        assert result["outer"]["inner"]["token"] == "***REDACTED***"
        assert result["outer"]["inner"]["value"] == 1
        assert result["list"][0]["password"] == "***REDACTED***"
        assert result["list"][1] == "plain"

    def test_redact_url_masks_query_params(self) -> None:
        url = "https://api.binance.com/api/v3/order?symbol=BTCUSDT&apiKey=AK123456&signature=abc"
        result = redact_url(url)
        assert "AK123456" not in result
        assert "abc" not in result.split("signature=")[-1][:5]
        assert "symbol=BTCUSDT" in result, "非敏感参数应保留以便排查"

    def test_fingerprint_is_deterministic_and_irreversible(self) -> None:
        secret = "my-api-key-value"
        fp1 = fingerprint(secret)
        fp2 = fingerprint(secret)

        assert fp1 == fp2, "同一输入应产生相同指纹"
        assert secret not in fp1, "指纹不得包含原值"
        assert len(fp1) < len(secret) + 10

        # 不同输入应有不同指纹
        assert fingerprint("other-key") != fp1

    def test_secret_str_never_leaks_in_repr(self) -> None:
        """SecretStr 的 repr/str/f-string 都不得泄露明文。"""
        plaintext = "MY-SECRET-API-KEY-VALUE-1234567890"
        secret = SecretStr(plaintext, name="TEST_KEY")

        assert plaintext not in repr(secret)
        assert plaintext not in str(secret)
        assert plaintext not in f"{secret}"
        assert plaintext not in f"{secret!r}"
        assert plaintext not in f"{secret:>20}"
        # 显式 reveal 才拿得到
        assert secret.reveal() == plaintext

    def test_logging_filter_redacts_messages(self, caplog: pytest.LogCaptureFixture) -> None:
        """日志过滤器必须在写入前脱敏。"""
        from cointrader.logging_setup import RedactionFilter

        logger = logging.getLogger("test.redaction")
        logger.addFilter(RedactionFilter())

        with caplog.at_level(logging.INFO, logger="test.redaction"):
            logger.info("请求 URL: https://api.binance.com/x?apiKey=LEAKME123&symbol=BTC")

        assert "LEAKME123" not in caplog.text, f"日志泄露了密钥: {caplog.text}"
        assert "symbol=BTC" in caplog.text, "非敏感参数应保留"


class TestDotenvLoading:
    """B8. .env 加载。"""

    def test_dotenv_does_not_override_existing_env(
        self, clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """环境变量优先级必须高于 .env 文件。

        否则 ``COINTRADER_TRADING_ENABLED=... python -m ...`` 这种
        临时覆盖会被文件悄悄改写，安全闸门形同虚设。
        """
        env_file = tmp_path / ".env"
        env_file.write_text("MY_TEST_VAR=from_file\n", encoding="utf-8")

        monkeypatch.setenv("MY_TEST_VAR", "from_shell")
        load_dotenv(env_file)

        assert os.environ["MY_TEST_VAR"] == "from_shell"

    def test_dotenv_parses_quotes_and_comments(
        self, clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# 注释行\n"
            "PLAIN=value1\n"
            'QUOTED="value2"\n'
            "SINGLE='value3'\n"
            "export EXPORTED=value4\n"
            "\n"
            "MALFORMED_LINE_NO_EQUALS\n",
            encoding="utf-8",
        )
        count = load_dotenv(env_file)

        assert os.environ["PLAIN"] == "value1"
        assert os.environ["QUOTED"] == "value2"
        assert os.environ["SINGLE"] == "value3"
        assert os.environ["EXPORTED"] == "value4"
        assert count == 4, f"应写入 4 个变量，实际 {count}"


class TestSecurityPosture:
    """B9. 安全状态自述。"""

    def test_posture_defaults_to_safe(self, clean_env) -> None:
        posture = describe_security_posture()

        assert posture["use_testnet"] is True
        assert posture["trading_enabled"] is False
        assert posture["mainnet_reachable"] is False
        assert posture["mode"] == "TESTNET"

    def test_posture_content_has_no_secrets(self, clean_env, monkeypatch) -> None:
        """安全状态描述本身不得包含密钥明文。"""
        monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "SECRETKEYVALUE123")
        posture = describe_security_posture()

        assert "SECRETKEYVALUE123" not in str(posture)
        assert posture["has_api_key"] is True


__all__: list[str] = []
