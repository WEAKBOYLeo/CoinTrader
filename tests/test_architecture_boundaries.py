"""模块依赖边界静态强制（AC-01）。

**这些测试失败 = 构建失败。** 依赖方向是模块边界的安全根基：

- ``domain/`` 不导入网络、数据库、Binance、``execution`` 或 WebUI ——
  领域契约必须可以独立导入，不携带任何 IO 或执行能力。
- ``strategy/``/``portfolio/``/``risk/`` 不导入 Binance client
  （``data/`` 公开客户端与 ``execution/`` 签名/下单 adapter）。
- ``webui/`` 不导入 broker/executor（查询层与执行解耦，
  查询失败不得触碰下单路径）。

与 ``tests/test_safety.py`` 的 A 组互补：本文件按**目标包**划分边界，
safety 按**能力**（签名/网络/密钥）划分。

违规失败信息逐条给出 ``文件:行号: import 符号``，便于定位。

T1 阶段 ``strategy/``/``portfolio/``/``risk/`` 等目标包尚未创建，
对应检查在目录不存在时视为自然满足（vacuously true），
由 T2/T3 创建后自动生效。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: 网络/数据库/IO 相关顶层模块：domain 内核一律禁止。
FORBIDDEN_IO_MODULES = frozenset(
    {
        "httpx",
        "requests",
        "urllib",
        "urllib3",
        "aiohttp",
        "http",
        "socket",
        "ssl",
        "websockets",
        "sqlite3",
        "pandas",
        "numpy",
        "asyncio",
        "subprocess",
        "os",
    }
)

#: 带 Binance client 的 cointrader 子模块（strategy/portfolio/risk 禁止导入）。
BINANCE_CLIENT_MODULES = frozenset(
    {
        "cointrader.data",
        "cointrader.execution.transport",
        "cointrader.execution.spot",
        "cointrader.execution.futures",
        "cointrader.execution.broker",
        "cointrader.execution.auth",
    }
)

#: webui 禁止导入的 broker/executor 子模块。
WEBUI_FORBIDDEN_EXECUTION_MODULES = frozenset(
    {
        "cointrader.execution.broker",
        "cointrader.execution.guard",
        "cointrader.execution.pair_executor",
        "cointrader.execution.transport",
        "cointrader.execution.spot",
        "cointrader.execution.futures",
        "cointrader.execution.auth",
        "cointrader.execution.sync",
        "cointrader.execution.reconcile",
    }
)

#: 纯策略/组合/风控包：禁止导入任何 cointrader.execution 网络 adapter
#: 与 data 层 Binance 客户端（允许导入 execution 的规则/闸门模块，如 risk_gate）。
STRATEGY_SIDE_FORBIDDEN = BINANCE_CLIENT_MODULES | FORBIDDEN_IO_MODULES


def _python_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _resolved_module(package_parts: tuple[str, ...], node: ast.Import | ast.ImportFrom) -> list[str]:
    """返回该 import 语句涉及的全部（可能完整）cointrader 模块路径或顶层模块名。

    绝对 import：直接取 module 名（含包名）。
    相对 import：按文件包层级解析为 cointrader.* 绝对路径。
    """
    modules: list[str] = []
    if isinstance(node, ast.Import):
        modules.extend(alias.name for alias in node.names)
        return modules
    if node.level > 0:
        if node.module:
            base = ".".join(package_parts[: len(package_parts) - (node.level - 1)])
            modules.append(f"{base}.{node.module}")
        else:
            base = ".".join(package_parts[: len(package_parts) - (node.level - 1)])
            modules.append(base)
    elif node.module:
        modules.append(node.module)
    return modules


def _package_parts(path: Path, src_root: Path) -> tuple[str, ...]:
    """文件所属包的模块路径（相对 src/，如 ('cointrader','domain')）。"""
    rel = path.relative_to(src_root.parent)
    parts = rel.parts
    if parts[-1] == "__init__.py":
        return parts[:-1]
    return parts[:-1]


def _imports_in_file(path: Path, src_root: Path) -> list[tuple[int, str]]:
    """返回 [(行号, 模块路径)]；模块路径为完整绝对模块名。"""
    package_parts = _package_parts(path, src_root)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for module in _resolved_module(package_parts, node):
                result.append((node.lineno, module))
    return result


def _module_prefix_match(module: str, forbidden: frozenset[str]) -> str | None:
    """module 命中 forbidden 中任一项（精确或前缀，如 cointrader.execution.* 匹配子模块）。"""
    for item in forbidden:
        if module == item or module.startswith(item + "."):
            return item
    return None


def _check(
    root: Path,
    src_root: Path,
    forbidden: frozenset[str],
    label: str,
) -> list[str]:
    """收集 root 下所有违规 import，返回可读违规列表（空 = 通过）。"""
    violations: list[str] = []
    for path in _python_files(root):
        rel = path.relative_to(src_root.parent)
        for lineno, module in _imports_in_file(path, src_root):
            hit = _module_prefix_match(module, forbidden)
            if hit is not None:
                violations.append(f"{rel}:{lineno}: import {module}（禁止: {label}）")
    return violations


class TestDomainIsIoFree:
    """domain/ 无网络、无数据库、无 Binance、无 execution、无 WebUI。"""

    def test_domain_package_exists(self, src_root: Path) -> None:
        assert (src_root / "domain").is_dir(), "domain 包尚未创建（T1 必须完成）"

    def test_domain_has_no_io_or_execution_or_webui_imports(self, src_root: Path) -> None:
        forbidden = (
            FORBIDDEN_IO_MODULES
            | {"cointrader.execution", "cointrader.webui", "cointrader.data"}
            | {"http.server", "socketserver"}
        )
        violations = _check(src_root / "domain", src_root, forbidden, "domain 无 IO/无执行层")
        assert violations == [], "domain 边界违规:\n" + "\n".join(violations)

    def test_domain_only_imports_itself_and_stdlib(self, src_root: Path) -> None:
        """domain 内所有 cointrader 内部 import 必须仍在 cointrader.domain.* 内。"""
        violations: list[str] = []
        for path in _python_files(src_root / "domain"):
            rel = path.relative_to(src_root.parent)
            for lineno, module in _imports_in_file(path, src_root):
                top = module.split(".")[0]
                if top == "cointrader" and not (
                    module == "cointrader.domain"
                    or module.startswith("cointrader.domain.")
                ):
                    violations.append(f"{rel}:{lineno}: import {module}（只允许 cointrader.domain.*）")
        assert violations == [], "\n".join(violations)


class TestStrategySideHasNoExchangeClient:
    """strategy/portfolio/risk 不导入 Binance client（T2/T3 生效后自动检查）。"""

    @pytest.mark.parametrize("pkg", ["strategy", "portfolio", "risk"])
    def test_no_binance_client_imports(self, src_root: Path, pkg: str) -> None:
        root = src_root / pkg
        if not root.is_dir():
            pytest.skip(f"{pkg} 包尚未创建（T1 阶段自然满足，T2/T3 后自动生效）")
        violations = _check(root, src_root, STRATEGY_SIDE_FORBIDDEN, f"{pkg} 无 Binance client")
        assert violations == [], f"{pkg} 边界违规:\n" + "\n".join(violations)


class TestRiskPackageDependencyScope:
    """risk 包只依赖 config/domain/errors 与 execution 的 rule/gate 模块（T3）。"""

    ALLOWED_COINTRADER = frozenset(
        {
            "cointrader.config",
            "cointrader.domain",
            "cointrader.errors",
            "cointrader.execution.rules",
            "cointrader.execution.risk",
            "cointrader.execution.risk_gate",
        }
    )

    def test_risk_imports_within_scope(self, src_root: Path) -> None:
        root = src_root / "risk"
        if not root.is_dir():
            pytest.skip("risk 包尚未创建")
        violations: list[str] = []
        for path in _python_files(root):
            rel = path.relative_to(src_root.parent)
            for lineno, module in _imports_in_file(path, src_root):
                top = module.split(".")[0]
                if top != "cointrader":
                    continue
                if module == "cointrader.risk" or module.startswith("cointrader.risk."):
                    continue
                if not any(
                    module == allowed or module.startswith(allowed + ".")
                    for allowed in self.ALLOWED_COINTRADER
                ):
                    violations.append(f"{rel}:{lineno}: import {module}（超出 risk 包允许范围）")
        assert violations == [], "risk 依赖范围违规:\n" + "\n".join(violations)


class TestWebuiHasNoBrokerOrExecutor:
    """webui/ 不导入 broker/executor（查询层与执行解耦）。"""

    def test_no_broker_executor_imports(self, src_root: Path) -> None:
        root = src_root / "webui"
        assert root.is_dir(), "webui 目录不存在"
        # webui 本身是只读 HTTP 服务，stdlib http.server/urllib.parse 允许；
        # 只禁止第三方网络库与 broker/executor 执行能力。
        third_party_network = frozenset({"httpx", "requests", "aiohttp", "websockets", "urllib3", "socket", "ssl"})
        forbidden = WEBUI_FORBIDDEN_EXECUTION_MODULES | third_party_network
        violations = _check(root, src_root, forbidden, "webui 无 broker/executor")
        assert violations == [], "webui 边界违规:\n" + "\n".join(violations)

    def test_webui_imports_nothing_from_execution_broker_surface(self, src_root: Path) -> None:
        """双保险：webui 不得以符号名方式引用下单能力（broker/transport/guard 直连）。"""
        import re

        pattern = re.compile(
            r"^(from\s+\S*(broker|transport|guard|pair_executor)\S*\s+import|import\s+\S*(broker|transport|guard|pair_executor)\S*\s*)",
            re.MULTILINE,
        )
        violations: list[str] = []
        for path in _python_files(src_root / "webui"):
            content = path.read_text(encoding="utf-8")
            rel = path.relative_to(src_root.parent)
            if pattern.search(content):
                violations.append(f"{rel}: 源码文本出现 broker/transport/guard import")
        assert violations == [], "\n".join(violations)


class TestRiskKernelHasNoFloatArithmetic:
    """T1：风险内核金额边界全 Decimal（kernel.py 内禁止 float 金额转换）。

    legacy ``RiskManager`` 的 float API 只允许在 ``risk/adapter.py`` 做
    一次性单向边界转换（Decimal→float，返回仅 bool/str），不得进入内核。
    """

    def test_kernel_source_has_no_float_calls(self, src_root: Path) -> None:
        import re

        kernel = src_root / "risk" / "kernel.py"
        assert kernel.is_file(), "risk/kernel.py 不存在"
        hits = [
            f"line {i}: {line.strip()}"
            for i, line in enumerate(kernel.read_text(encoding="utf-8").splitlines(), 1)
            if re.search(r"\bfloat\s*\(", line) and not line.strip().startswith("#")
        ]
        assert hits == [], (
            "risk/kernel.py 出现 float 金额转换（全链路应 Decimal）:\n" + "\n".join(hits)
        )

    def test_risk_exposure_port_is_decimal_typed(self, src_root: Path) -> None:
        import re

        kernel = (src_root / "risk" / "kernel.py").read_text(encoding="utf-8")
        assert re.search(
            r"total_exposure:\s*Decimal", kernel
        ), "RiskExposure.total_exposure 必须声明为 Decimal"
        assert re.search(
            r"symbol_exposure:\s*dict\[str,\s*Decimal\]", kernel
        ), "RiskExposure.symbol_exposure 必须声明为 dict[str, Decimal]"
        assert re.search(
            r"def check_order\(self,\s*symbol: str,\s*notional: Decimal", kernel
        ), "RiskRulesPort.check_order 金额参数必须为 Decimal"


# ---------------------------------------------------------------------------
# T3：生产 raw 开/平仓入口禁用（AC-06）
# ---------------------------------------------------------------------------

RAW_ENTRY_METHODS = frozenset({"open_pair", "close_pair"})


def _raw_call_violations(files: list[Path], src_root: Path) -> list[str]:
    """收集 production 文件里对 raw ``open_pair``/``close_pair`` 的调用点。

    计划化唯一入口 ``open_pair_from_plan`` 不命中（属性名精确匹配）。
    定义（FunctionDef）与测试 helper 不在扫描范围。
    """
    violations: list[str] = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as exc:
            violations.append(f"{path.relative_to(src_root)}: 无法解析: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                name = node.func.attr
                if name in RAW_ENTRY_METHODS:
                    rel = path.relative_to(src_root.parent)
                    violations.append(
                        f"{rel}:{node.lineno}: 调用 raw {name}(...) —— 生产路径必须走 "
                        f"open_pair_from_plan(ExecutionPlan)"
                    )
    return violations


class TestNoRawPairEntryInProduction:
    """T3.6：application/、live/、CLI 不得调用 raw ``open_pair``/``close_pair``。

    计划化唯一执行入口：``PairExecutor.open_pair_from_plan(plan)``（内部
    adapter 委托保留 partial fill / UNKNOWN / 补偿逻辑）。raw 方法只允许
    在标记为 legacy 的测试 helper（tests/live_helpers.py 的 FakeExecutor）
    与 PairExecutor 自身 adapter 内使用。
    """

    def test_no_raw_open_close_pair_calls(self, src_root: Path) -> None:
        files: list[Path] = []
        for sub in ("application", "live"):
            pkg = src_root / sub
            if pkg.is_dir():
                files.extend(_python_files(pkg))
        cli = src_root / "cli.py"
        if cli.is_file():
            files.append(cli)
        assert files, "扫描目标缺失（application/、live/、cli.py 均应存在）"
        violations = _raw_call_violations(files, src_root)
        assert violations == [], (
            "生产调用图存在 raw 开/平仓直调（T3 禁用）:\n" + "\n".join(violations)
        )
