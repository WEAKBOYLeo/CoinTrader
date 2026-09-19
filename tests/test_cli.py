"""CLI 测试。

重点验证两件事：

1. **解析器结构** —— 特别是「不存在 trade 子命令」这条安全约束
2. **命令的离线行为** —— doctor/costs 不需要网络和密钥就能跑通

不测需要联网的命令（scan/backtest/ping）—— 那些通过手动运行验证，
避免让测试套件依赖币安可用性。
"""

from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path

import pytest

from cointrader.cli import build_parser, cmd_costs, cmd_doctor, main
from cointrader.config import load_config
from cointrader.errors import ConfigError


class TestParserStructure:
    """参数解析器结构。"""

    def test_has_expected_commands(self) -> None:
        parser = build_parser()

        # 各子命令的最小合法参数
        minimal: dict[str, list[str]] = {
            "doctor": ["doctor"],
            "ping": ["ping"],
            "costs": ["costs"],
            "scan": ["scan"],
            "backtest": ["backtest"],
            "portfolio": ["portfolio", "BTCUSDT", "ETHUSDT"],
            "scenarios": ["scenarios", "--capital", "1000"],
        }

        for command, argv in minimal.items():
            args = parser.parse_args(argv)
            assert args.command == command, f"{command} 未正确注册"

    def test_has_no_trade_command(self) -> None:
        """**安全约束**：不得存在一键下单的子命令。

        在回测结论（M5）出来之前，不应该有任何方便的下单入口。
        执行层的代码存在，但触发它必须显式写 Python 并打开三道开关，
        而不是敲一条命令。
        """
        parser = build_parser()

        for forbidden in ("trade", "order", "buy", "sell", "execute", "live"):
            with pytest.raises(SystemExit):
                parser.parse_args([forbidden])

    def test_version_flag(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_requires_subcommand(self) -> None:
        """不带子命令应报错，而不是静默无操作。"""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_backtest_defaults_to_automatic_portfolio(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["backtest"])

        assert not hasattr(args, "symbol")
        assert args.max_symbols is None
        assert args.min_volume is None

    def test_scenarios_requires_capital(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["scenarios"])

    def test_scenarios_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["scenarios", "--capital", "1000"])

        assert args.capital == 1000.0
        assert args.expected_return == 0.15
        assert args.exchange_exposure == 1.0
        assert args.leverage == 3.0

    def test_scan_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["scan"])

        assert args.top == 20
        assert args.max_symbols is None
        assert args.min_volume is None

    def test_rolling_window_override_is_parsed(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["backtest", "--rolling-window", "60"])

        assert args.rolling_window == 60

    def test_portfolio_defaults_to_automatic_universe(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["portfolio"])

        assert args.symbols == []
        assert args.max_symbols is None
        assert args.min_volume is None

    def test_portfolio_parses_multiple_symbols(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["portfolio", "BTCUSDT", "ETHUSDT"])

        assert args.command == "portfolio"
        assert args.symbols == ["BTCUSDT", "ETHUSDT"]



def run_command(func, **kwargs) -> tuple[int, str]:
    """运行命令并捕获 stdout。"""
    import argparse

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = func(argparse.Namespace(**kwargs))
    return code, buffer.getvalue()


class TestDoctorCommand:
    """doctor 自检 —— 不需要网络和密钥。"""

    def test_passes_with_project_config(self, clean_env, project_root: Path) -> None:
        code, output = run_command(
            cmd_doctor,
            config=project_root / "config" / "config.yaml",
        )

        assert code == 0, f"自检未通过:\n{output}"
        assert "自检通过" in output

    def test_shows_security_posture(self, clean_env, project_root: Path) -> None:
        """必须显示安全状态横幅 —— 让人永远清楚当前处于什么模式。"""
        _, output = run_command(
            cmd_doctor, config=project_root / "config" / "config.yaml"
        )

        assert "安全状态" in output
        assert "测试网" in output or "锁定" in output
        assert "停机开关" in output

    def test_reports_cost_model(self, clean_env, project_root: Path) -> None:
        _, output = run_command(
            cmd_doctor, config=project_root / "config" / "config.yaml"
        )

        assert "0.2800%" in output, f"应报告往返成本 0.28%:\n{output}"

    def test_fails_on_bad_config(self, clean_env, tmp_path: Path) -> None:
        code, output = run_command(cmd_doctor, config=tmp_path / "nonexistent.yaml")
        assert code == 1

    def test_detects_inconsistent_risk_limits(self, clean_env, tmp_path: Path) -> None:
        """风控限额倒挂时自检必须失败。

        限额层级倒挂意味着某个限额永远不生效 —— 风控少了一道。
        """
        config_path = tmp_path / "bad_risk.yaml"
        config_path.write_text(
            """
risk:
  max_notional_per_order: 5000.0
  max_exposure_per_symbol: 500.0
  max_total_exposure: 2000.0
""",
            encoding="utf-8",
        )

        code, output = run_command(cmd_doctor, config=config_path)
        # ConfigError 会在构造 RiskConfig 时抛出，doctor 捕获为失败
        assert code == 1


class TestCostsCommand:
    """costs 命令 —— 人工复核费率假设的入口。"""

    def test_prints_all_tiers(self, clean_env, project_root: Path) -> None:
        code, output = run_command(
            cmd_costs, config=project_root / "config" / "config.yaml"
        )

        assert code == 0
        for tier in ("major", "mid", "small"):
            assert tier in output

    def test_shows_round_trip_figures(self, clean_env, project_root: Path) -> None:
        _, output = run_command(
            cmd_costs, config=project_root / "config" / "config.yaml"
        )

        assert "0.2800%" in output    # major 往返
        assert "0.4400%" in output    # small 往返

    def test_shows_pessimistic_scenario(self, clean_env, project_root: Path) -> None:
        """必须展示悲观情景，让人知道费率假设恶化时的影响。"""
        _, output = run_command(
            cmd_costs, config=project_root / "config" / "config.yaml"
        )

        assert "悲观" in output
        assert "0.3800%" in output

    def test_warns_about_fee_verification(self, clean_env, project_root: Path) -> None:
        """必须提醒用户核对真实费率 —— 假设与实际的差异直接改变结论。"""
        _, output = run_command(
            cmd_costs, config=project_root / "config" / "config.yaml"
        )

        assert "核对" in output


class TestLivePnlAllRuns:
    """live pnl --all-runs（断点重连后的跨 run 视图，计划 1.0 T2 / AC-04）。"""

    def test_all_runs_and_run_id_mutually_exclusive(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["live", "pnl", "--all-runs", "--run-id", "run-x"])
        assert exc_info.value.code == 2, "argparse 互斥组应退出码 2"

    def test_all_runs_flag_parsed(self) -> None:
        args = build_parser().parse_args(["live", "pnl", "--all-runs"])
        assert args.all_runs is True
        assert args.run_id is None

    def test_all_runs_output_aggregates_across_runs(self, clean_env, tmp_path: Path) -> None:
        import json as _json

        from cointrader.cli import cmd_live_pnl
        from cointrader.execution.store import StateStore
        from test_pnl import _seed_closed_trip_b, _seed_round_trip

        db = tmp_path / "t.sqlite3"
        store = StateStore(db)
        _seed_round_trip(store, run_id="run-a")   # +0.497
        _seed_closed_trip_b(store)                # run-b −0.003
        now = int(time.time() * 1000)
        for run_id, started in (("run-a", now - 2 * 3600_000), ("run-b", now - 3600_000)):
            store.start_run_session(
                run_id=run_id, started_ms=started, mode="testnet",
                strategy_version="t", config_hash="h", code_revision="r",
                spot_endpoint="e", futures_endpoint="f", user_stream_mode="poll",
            )
            store.end_run_session(run_id, ended_ms=started + 1000,
                                  status="STOPPED", stop_reason="graceful_stop")
        store.close()

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(f"execution:\n  state_db: {db}\n", encoding="utf-8")

        code, output = run_command(
            cmd_live_pnl,
            config=config_path,
            run_id=None,
            all_runs=True,
            since="24h",
            until=None,
            symbol=None,
            json=True,
        )
        assert code == 0
        payload = _json.loads(output)
        assert payload["run_id"] == "ALL"
        assert len(payload["pnl"]["per_pair"]) == 2
        from decimal import Decimal

        assert Decimal(payload["pnl"]["net_pnl"]) == Decimal("0.494")


class TestLiveStatusContinuity:
    """live status 运行连续性块（计划 1.0 T2 / AC-04）。"""

    def test_status_shows_continuity_block(self, clean_env, tmp_path: Path) -> None:
        import time as _time

        from cointrader.cli import cmd_live_status
        from cointrader.execution.store import StateStore

        db = tmp_path / "t.sqlite3"
        store = StateStore(db)
        now = int(_time.time() * 1000)
        store.start_run_session(
            run_id="run-1", started_ms=now - 7200_000, mode="testnet",
            strategy_version="t", config_hash="h", code_revision="r",
            spot_endpoint="e", futures_endpoint="f", user_stream_mode="poll",
        )
        store.end_run_session("run-1", ended_ms=now - 3600_000,
                              status="INTERRUPTED",
                              stop_reason="process_exited_without_graceful_stop")
        store.start_run_session(
            run_id="run-2", started_ms=now - 3600_000, mode="testnet",
            strategy_version="t", config_hash="h", code_revision="r",
            spot_endpoint="e", futures_endpoint="f", user_stream_mode="poll",
        )
        store.acquire_lease("live-executor", holder="pid-4242")
        store.close()

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(f"execution:\n  state_db: {db}\n", encoding="utf-8")

        code, output = run_command(cmd_live_status, config=config_path, json=False)
        assert code == 0, output
        assert "运行连续性" in output
        assert "run-2" in output and "RUNNING" in output
        assert "run-1" in output and "INTERRUPTED" in output
        assert "process_exited_without_graceful_stop" in output
        assert "pid-4242" in output

    def test_status_without_sessions_shows_hint(self, clean_env, tmp_path: Path) -> None:

        from cointrader.cli import cmd_live_status
        from cointrader.execution.store import StateStore

        db = tmp_path / "t.sqlite3"
        StateStore(db).close()  # 建库（无 run_session）

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(f"execution:\n  state_db: {db}\n", encoding="utf-8")

        code, output = run_command(cmd_live_status, config=config_path, json=False)
        assert code == 0, output
        assert "运行连续性" in output
        assert "无 run_session" in output


class TestExecModeEnv:
    """COINTRADER_EXEC_MODE 模式覆盖（计划 1.0 T4 / AC-06）。"""

    def _config(self, project_root: Path):
        return load_config(project_root / "config" / "config.yaml")

    def test_no_env_uses_config_source(self, clean_env, project_root: Path) -> None:
        from cointrader.cli import _resolve_live_mode

        mode, source = _resolve_live_mode(self._config(project_root))
        assert (mode, source) == ("testnet", "config")

    @pytest.mark.parametrize("env_mode", ["shadow", "paper", "testnet"])
    def test_env_overrides_config(self, clean_env, monkeypatch, project_root: Path, env_mode: str) -> None:
        from cointrader.cli import _resolve_live_mode

        monkeypatch.setenv("COINTRADER_EXEC_MODE", env_mode)
        mode, source = _resolve_live_mode(self._config(project_root))
        assert mode == env_mode
        assert source == "env:COINTRADER_EXEC_MODE"

    @pytest.mark.parametrize("bad", ["LIVE", "prod", "Testnet", " live"])
    def test_invalid_env_rejected_with_legal_values(self, clean_env, monkeypatch, project_root: Path, bad: str) -> None:
        from cointrader.cli import _resolve_live_mode

        monkeypatch.setenv("COINTRADER_EXEC_MODE", bad)
        with pytest.raises(ConfigError, match="paper/testnet/shadow/live"):
            _resolve_live_mode(self._config(project_root))

    def test_live_via_env_still_requires_magic_string(self, clean_env, monkeypatch, project_root: Path) -> None:
        """闸门不削弱：env=live 且未设魔法串 → 仍拒绝。"""
        from cointrader.cli import _resolve_live_mode

        monkeypatch.setenv("COINTRADER_EXEC_MODE", "live")
        with pytest.raises(ConfigError, match="双重确认"):
            _resolve_live_mode(self._config(project_root))

    def test_doctor_rejects_bogus_env_nonzero(self, clean_env, monkeypatch, project_root: Path) -> None:
        from cointrader.cli import cmd_live_doctor

        monkeypatch.setenv("COINTRADER_EXEC_MODE", "bogus")
        code, output = run_command(cmd_live_doctor, config=project_root / "config" / "config.yaml")
        assert code == 1
        assert "COINTRADER_EXEC_MODE 非法" in output
        assert "paper/testnet/shadow/live" in output

    def test_doctor_shows_mode_source_and_endpoint(self, clean_env, project_root: Path) -> None:
        from cointrader.cli import cmd_live_doctor

        code, output = run_command(cmd_live_doctor, config=project_root / "config" / "config.yaml")
        assert "（来源: config）" in output
        assert "端点: " in output
        assert "spot=" in output and "perp=" in output

    def test_status_shows_mode_source_from_env(self, clean_env, monkeypatch, tmp_path: Path) -> None:
        from cointrader.cli import cmd_live_status
        from cointrader.execution.store import StateStore

        db = tmp_path / "t.sqlite3"
        StateStore(db).close()
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(f"execution:\n  state_db: {db}\n", encoding="utf-8")
        monkeypatch.setenv("COINTRADER_EXEC_MODE", "shadow")

        code, output = run_command(cmd_live_status, config=config_path, json=False)
        assert code == 0, output
        assert "（来源: env:COINTRADER_EXEC_MODE，当前解析: shadow）" in output
        assert "端点选择" in output


class TestSigTermHandler:
    """SIGTERM → 优雅停机路径（计划 1.0 T4 / AC-07）。"""

    def test_handler_registered_and_raises_keyboard_interrupt(self, monkeypatch) -> None:
        import signal as _signal

        from cointrader import cli as cli_mod

        installed: dict[int, object] = {}
        monkeypatch.setattr(cli_mod.signal, "signal", lambda sig, handler: installed.setdefault(sig, handler))
        cli_mod._install_sigterm_handler()
        assert _signal.SIGTERM in installed, "必须在 startup 前注册 SIGTERM handler"
        handler = installed[_signal.SIGTERM]
        with pytest.raises(KeyboardInterrupt):
            handler(_signal.SIGTERM, None)


class TestMainEntry:
    """main() 入口。"""

    def test_doctor_via_main(self, clean_env, project_root: Path) -> None:
        code = main(
            [
                "--config",
                str(project_root / "config" / "config.yaml"),
                "doctor",
            ]
        )
        assert code == 0

    def test_invalid_command_exits(self, clean_env) -> None:
        with pytest.raises(SystemExit):
            main(["nonexistent_command"])

    def test_missing_config_does_not_crash_logging_setup(self, clean_env, tmp_path: Path) -> None:
        """配置缺失时，日志初始化不应崩溃 —— doctor 正是用来诊断这个的。"""
        code = main(["--config", str(tmp_path / "missing.yaml"), "doctor"])
        # doctor 会报告配置加载失败并返回 1，但不应该抛异常
        assert code == 1


__all__: list[str] = []
