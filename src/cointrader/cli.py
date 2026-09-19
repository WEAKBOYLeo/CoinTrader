"""命令行入口。

用法::

    cointrader doctor                  # 安全自检（不需要密钥/网络）
    cointrader ping                    # 币安连通性与时钟偏移
    cointrader costs                   # 打印成本模型明细
    cointrader scan [--top N]          # 扫描资金费候选
    cointrader backtest                  # 自动选币组合滚动回测
    cointrader portfolio                  # 同上：自动选币组合回测
    cointrader portfolio BTCUSDT ETHUSDT # 指定候选池的组合回测
    cointrader scenarios --capital N   # 破产情景分析

**没有 ``trade`` 子命令。** 这是刻意的：在回测结论出来之前，
不应该有任何一键下单的入口。执行层的代码存在，
但触发它需要显式写 Python 代码并打开三道开关。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import __version__
from .config import EXECUTION_MODES, Config, load_config
from .errors import ConfigError
from .logging_setup import setup_logging
from .research.costs import CostModel, LiquidityTier, pessimistic_config
from .research.reports import write_portfolio_reports
from .secrets import describe_security_posture, load_credentials, load_dotenv

# ---------------------------------------------------------------------------
# 输出辅助（不引入 rich 依赖，纯文本更利于管道处理）
# ---------------------------------------------------------------------------

_BANNER = r"""
  ____      _        _____              _
 / ___|___ (_)_ __  |_   _| __ __ _  __| | ___ _ __
| |   / _ \| | '_ \   | || '__/ _` |/ _` |/ _ \ '__|
| |__| (_) | | | | |  | || | | (_| | (_| |  __/ |
 \____\___/|_|_| |_|  |_||_|  \__,_|\__,_|\___|_|
"""


def _print_security_banner() -> None:
    """打印醒目的安全状态横幅。

    每次运行都显示，让人永远清楚当前处于什么模式。
    """
    posture = describe_security_posture()
    mode = posture["mode"]

    if mode == "MAINNET":
        marker = "!!! 真实盘 —— 会动真钱 !!!"
    elif mode == "TESTNET":
        marker = "测试网 / 只读模式"
    else:
        marker = "已锁定（交易未启用）"

    print(f"\n  安全状态: {marker}")
    print(f"  测试网: {posture['use_testnet']}   交易开关: {posture['trading_enabled']}")
    print(f"  停机开关: {posture['kill_switch_file']} "
          f"({'已触发' if posture['kill_switch_engaged'] else '未触发'})")
    print(f"  凭证已加载: {posture['has_api_key']}")
    print()


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    """安全与环境自检。不需要网络和密钥。"""
    _print_security_banner()

    failures: list[str] = []
    warnings: list[str] = []

    # 1. 配置加载
    try:
        config = load_config(args.config)
        print(f"  [OK] 配置加载成功: {config.source_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] 配置加载失败: {exc}")
        return 1

    # 2. 成本模型
    try:
        model = CostModel(config.costs)
        base = model.round_trip(LiquidityTier.MAJOR)
        print(f"  [OK] 成本模型: 主流币往返 {base:.4%}")

        # 合理性检查：往返成本应在 0.05% ~ 1% 之间。
        # 超出这个范围几乎总是配置写错（比如把 0.1% 写成 0.1）
        if not 0.0005 <= base <= 0.01:
            failures.append(
                f"往返成本 {base:.4%} 不在合理区间 [0.05%, 1%]，请检查 config.yaml 的 costs 段"
            )
    except Exception as exc:  # noqa: BLE001
        failures.append(f"成本模型构造失败: {exc}")

    # 3. 回测配置的关键约束
    if config.backtest.execution_lag_bars < 1:
        failures.append("execution_lag_bars < 1 会引入前瞻偏差")
    else:
        print(f"  [OK] 执行滞后 {config.backtest.execution_lag_bars} 期（无前瞻）")
        print(
            f"  [OK] 固定滚动窗口 {config.backtest.rolling_window_periods} 期"
            "（窗口未满不交易）"
        )

    # 4. 风控限额一致性
    risk = config.risk
    if risk.max_notional_per_order > risk.max_exposure_per_symbol:
        failures.append("单笔限额大于单币种限额，配置不一致")
    if risk.max_exposure_per_symbol > risk.max_total_exposure:
        failures.append("单币种限额大于总限额，配置不一致")
    if not failures:
        print(
            f"  [OK] 风控限额: 单笔 {risk.max_notional_per_order:.0f} / "
            f"单币 {risk.max_exposure_per_symbol:.0f} / 总 {risk.max_total_exposure:.0f} USDT"
        )

    # 5. 目录可写性
    for name, path in (
        ("缓存", config.data.cache_dir),
        ("日志", config.logging.log_dir),
    ):
        resolved = config.resolved_path(path)
        try:
            resolved.mkdir(parents=True, exist_ok=True)
            probe = resolved / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            print(f"  [OK] {name}目录可写: {resolved}")
        except OSError as exc:
            failures.append(f"{name}目录不可写 {resolved}: {exc}")

    # 6. 停机开关状态（信息性）
    if describe_security_posture()["kill_switch_engaged"]:
        warnings.append("停机开关当前处于触发状态，所有下单会被拒绝")
    else:
        print("  [OK] 停机开关未触发")

    # 7. 密钥（不要求存在）
    try:
        from .secrets import load_credentials

        creds = load_credentials(require=False)
        print(f"  [OK] 凭证: {'已加载 ' + repr(creds) if creds else '未提供（回测/扫描不需要）'}")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"凭证加载异常: {exc}")

    # 汇总
    print()
    if warnings:
        print("  警告:")
        for warning in warnings:
            print(f"    - {warning}")
        print()

    if failures:
        print("  检查未通过:")
        for failure in failures:
            print(f"    ✗ {failure}")
        return 1

    print("  自检通过。可以运行 'cointrader ping' 验证网络连通性。")
    return 0


def cmd_ping(args: argparse.Namespace) -> int:
    """币安连通性与时钟偏移自检（需要联网）。"""
    from .data.binance import BinancePublicClient

    config = load_config(args.config)
    _print_security_banner()

    with BinancePublicClient(config.api, config.data) as client:
        try:
            health = client.health_check()
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] 无法连接币安: {exc}")
            return 1

        print("  [OK] 现货接口可达")
        print("  [OK] 合约接口可达")

        offset_spot = health["clock_offset_ms_spot"]
        offset_futures = health["clock_offset_ms_futures"]
        print(f"  本地时钟偏移: 现货 {offset_spot:+d} ms, 合约 {offset_futures:+d} ms")

        # 签名请求依赖时间戳，偏移过大会被币安以 -1021 拒绝
        max_offset = max(abs(offset_spot), abs(offset_futures))
        if max_offset > 1000:
            print(f"  [警告] 时钟偏移 {max_offset}ms 超过 1 秒，真实交易会被拒绝。请同步系统时间。")
        else:
            print("  [OK] 时钟偏移在可接受范围")

        print(f"\n  本次请求统计: {json.dumps(health['stats'], ensure_ascii=False)}")

    return 0


def cmd_costs(args: argparse.Namespace) -> int:
    """打印成本模型明细 —— 用于人工复核费率假设。"""
    config = load_config(args.config)
    model = CostModel(config.costs)

    print(f"\n  费率档位: {config.costs.fee_tier}")
    print(f"  BNB 抵扣: 现货 {config.costs.spot_bnb_discount:.0%}，合约 {config.costs.perp_bnb_discount:.0%}")
    print(f"  全部按 taker 计价: {config.costs.assume_all_taker}")
    print()

    print("  ── 往返成本（一次完整进场+出场，4 条腿）──")
    header = f"  {'档位':<8} {'进场':>10} {'出场':>10} {'往返':>10} {'1000U名义额':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for tier in LiquidityTier:
        breakdown = model.breakdown(tier)
        print(
            f"  {tier.value:<8} "
            f"{breakdown.total_entry:>9.4%} "
            f"{breakdown.total_exit:>9.4%} "
            f"{breakdown.total:>9.4%} "
            f"{model.round_trip_usdt(1000.0, tier):>13.2f}U"
        )

    print()
    print("  ── 盈亏平衡（需要持有多少天才能覆盖成本）──")
    for tier in LiquidityTier:
        costs = []
        for annual in (0.10, 0.20, 0.50):
            per_period = annual / ((24 / 8) * 365)   # 按 8h 结算基准
            days = model.breakeven_days(per_period, 8, tier)
            costs.append(f"年化{annual:.0%}→{days:.1f}天")
        print(f"  {tier.value:<8} " + "  ".join(costs))

    print()
    print("  ── 悲观情景（压力测试用）──")
    pessimistic = CostModel(pessimistic_config(config.costs))
    base = model.round_trip(LiquidityTier.MAJOR)
    pess = pessimistic.round_trip(LiquidityTier.MAJOR)
    print(f"  major 往返: {base:.4%} → 悲观 {pess:.4%}（+{pess - base:.4%}）")

    print()
    print("  ⚠️ 费率假设必须与你的账户实际费率一致。")
    print("     上线前请用账户的费率页面核对，并在 config.yaml 中更新。")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """扫描全市场资金费候选。"""
    from .research.scanner import FundingScanner, format_scan_table

    config = load_config(args.config)
    _print_security_banner()

    with FundingScanner(config) as scanner:
        result = scanner.scan(
            max_symbols=args.max_symbols,
            lookback_days=args.lookback_days,
            min_quote_volume=args.min_volume,
        )

    print(format_scan_table(result, limit=args.top))

    if args.json:
        output = {
            "summary": result.summary(),
            "top": [c.as_dict() for c in result.top(args.top)],
        }
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  结果已写入 {out_path}")

    return 0


def cmd_scenarios(args: argparse.Namespace) -> int:
    """破产情景分析 —— 回答"仓位该多大"。"""
    from .backtest.scenarios import (
        analyze_scenarios,
        position_sizing_advice,
        survival_analysis,
    )

    config = load_config(args.config)

    print(f"\n  资金规模: {args.capital:,.0f} USDT")
    print(f"  预期年化: {args.expected_return:.1%}")
    print(f"  单所敞口: {args.exchange_exposure:.0%}")
    print()

    results = analyze_scenarios(
        args.capital,
        exchange_exposure_pct=args.exchange_exposure,
        leverage=args.leverage,
    )

    print("  ── 破产情景 ──")
    header = f"  {'情景':<24} {'损失比例':>10} {'损失金额':>14} {'剩余':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for result in results:
        print(
            f"  {result.scenario.name:<24} "
            f"{result.scenario.loss_pct:>9.2%} "
            f"{result.loss_usdt:>13,.0f}U "
            f"{result.remaining_usdt:>13,.0f}U"
        )

    print()
    print("  ── 恢复时间（用策略自身收益弥补损失需要多久）──")
    survival = survival_analysis(
        args.capital,
        expected_annual_return=args.expected_return,
        exchange_exposure_pct=args.exchange_exposure,
        leverage=args.leverage,
    )
    for name, years in survival["recovery_years"].items():
        if years is None:
            label = "无法恢复"
        elif years == float("inf"):
            label = "永不（本金归零）"
        else:
            label = f"{years:.1f} 年"
        print(f"  {name:<24} {label}")

    print()
    print(f"  结论: {survival['verdict']}")

    print()
    print("  ── 仓位建议（按全损可承受原则）──")
    advice = position_sizing_advice(
        args.capital, config.risk, max_loss_tolerance_pct=args.max_loss_tolerance
    )
    print(f"  可承受最大损失: {advice['max_acceptable_loss']:,.0f} USDT "
          f"({args.max_loss_tolerance:.0%} of {args.capital:,.0f})")
    print(f"  建议单所资金上限: {advice['recommended_max_per_exchange']:,.0f} USDT")
    print(f"  当前配置总敞口上限: {advice['configured_total_exposure']:,.0f} USDT")

    if advice["warnings"]:
        print()
        print("  ⚠️ 严重警告（配置与风险承受能力冲突）:")
        for warning in advice["warnings"]:
            print(f"     {warning}")

    if advice["notes"]:
        print()
        print("  提示:")
        for note in advice["notes"]:
            print(f"     · {note}")

    print()
    print("  详细说明见 docs/ARCHITECTURE.md 与 backtest/scenarios.py 的模块文档。")
    return 0


def _apply_rolling_override(config, args):
    """应用命令行滚动窗口覆盖，不修改原配置对象。"""
    rolling_window = getattr(args, "rolling_window", None)
    if rolling_window is None:
        return config
    return replace(
        config,
        backtest=replace(config.backtest, rolling_window_periods=rolling_window),
    )


def cmd_backtest(args: argparse.Namespace) -> int:
    """策略主回测：自动发现市场候选并运行动态组合。"""
    args.symbols = []
    return cmd_portfolio(args)


def cmd_portfolio(args: argparse.Namespace) -> int:
    """全市场自动选币组合回测，或在显式候选池内动态选币。"""
    from .backtest.engine import run_portfolio
    from .data.binance import BinancePublicClient
    from .data.funding import fetch_funding_history, fetch_funding_intervals
    from .data.klines import fetch_historical_quote_volume_3d_avg, tradable_perpetuals

    config = _apply_rolling_override(load_config(args.config), args)
    min_volume_override = getattr(args, "min_volume", None)
    if min_volume_override is not None:
        config = replace(
            config,
            strategy=replace(
                config.strategy,
                selection=replace(
                    config.strategy.selection,
                    min_quote_volume_3d_avg=float(min_volume_override),
                ),
            ),
        )
    _print_security_banner()

    symbols = sorted({symbol.upper() for symbol in args.symbols})
    automatic_selection = not symbols
    rates: dict[str, object] = {}
    intervals: dict[str, int] = {}
    historical_volumes: dict[str, object] = {}
    start_ms = int((time.time() - config.backtest.history_days * 86_400) * 1000)

    with BinancePublicClient(config.api, config.data) as client:
        funding_intervals = fetch_funding_intervals(client)
        if automatic_selection:
            symbols = tradable_perpetuals(
                client.futures_exchange_info(),
                exclude_bases=config.strategy.selection.exclude_bases,
            )
            max_symbols = getattr(args, "max_symbols", None)
            if max_symbols is not None:
                if max_symbols <= 0:
                    print(f"  [FAIL] max_symbols 必须为正，当前 {max_symbols}")
                    return 1
                symbols = symbols[:max_symbols]
            print(f"  自动候选池: {len(symbols)} 个当前可交易 USDT 永续（历史成交量动态过滤）")

        for symbol in symbols:
            try:
                frame = fetch_funding_history(client, symbol, start_ms=start_ms)
                volume = fetch_historical_quote_volume_3d_avg(
                    client,
                    symbol,
                    start_ms=start_ms - 86_400_000,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [提示] 跳过 {symbol}：历史资金费或成交量获取失败（{exc}）")
                continue
            rates[symbol] = frame["funding_rate"]
            intervals[symbol] = funding_intervals.get(symbol)
            historical_volumes[symbol] = volume

    if not rates:
        print("  [FAIL] 没有可用于组合回测的资金费历史")
        return 1

    # 延迟到这里才导入 pandas 类型，避免 CLI 模块启动时增加额外初始化。
    import pandas as pd

    typed_rates = {symbol: series for symbol, series in rates.items() if isinstance(series, pd.Series)}
    tiers: dict[str, LiquidityTier] = {}
    model = CostModel(config.costs)

    try:
        result = run_portfolio(
            typed_rates,
            intervals,
            config.backtest,
            config.strategy,
            model,
            tiers=tiers,
            quote_volumes_24h=historical_volumes,
            basis_adverse_pct=config.risk.max_basis_adverse_pct,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] 组合回测失败: {exc}")
        return 1

    metrics = result.metrics
    print(f"  候选池: {len(typed_rates)} 个币种")
    print(f"  选币方式: {'自动全市场动态排名' if automatic_selection else '指定候选池动态排名'}")
    print(f"  回测区间: 最近 {config.backtest.history_days} 天")
    print(f"  每腿滑点: {config.backtest.slippage_per_leg:.2%}（对冲往返 4 腿）")
    print(f"  历史预热窗口: 最近 {config.backtest.rolling_window_periods} 期（窗口未满不交易）")
    print(
        f"  资金费均值: 最近 {config.strategy.entry.lookback_periods} 期，"
        f"入场 ≥ {config.strategy.entry.min_trailing_annualized:.0%}"
    )
    print(
        "  替换门槛: 持仓 <=60 期不替换 / 61-120 期高 "
        f"{config.strategy.exit.replacement_premium_under_120:.0%} / >120 期高 "
        f"{config.strategy.exit.replacement_premium_over_120:.0%}；"
        f"出场均值最近 {config.strategy.exit.exit_lookback_periods} 期"
    )
    print(
        f"  动态仓位: 最多 {config.strategy.selection.max_positions} 笔，"
        f"每笔 {config.strategy.selection.per_position_weight:.0%}"
    )
    print()
    print("  ── 滚动组合绩效 ──")
    print(f"  毛收益        : {metrics.gross_return:>10.4%}")
    print(f"  成本          : {metrics.cost_paid:>10.4%}")
    print(f"  净收益        : {metrics.net_return:>10.4%}")
    if metrics.cost_to_gross_ratio != float("inf"):
        print(f"  费用/毛收益   : {metrics.cost_to_gross_ratio:>10.2%}")
    else:
        print("  费用/毛收益   :          n/a（毛收益 <= 0）")
    print(f"  占总资金年化  : {metrics.net_annualized:>10.4%}")
    print(f"  平均组合敞口  : {metrics.avg_exposure:>10.1%}")
    print(f"  最大回撤      : {metrics.max_drawdown:>10.4%}")
    print(f"  完成交易数    : {metrics.n_round_trips:>10d}")
    print()
    print("  ── 实际入选币种 ──")
    active_results = {
        symbol: symbol_result
        for symbol, symbol_result in result.per_symbol.items()
        if symbol_result.trades
    }
    selected_symbols = sorted(active_results)
    print(f"  {', '.join(selected_symbols) if selected_symbols else '无（没有币种满足历史窗口与策略阈值）'}")
    print()
    print("  ── 动态选币结果 ──")
    for symbol, symbol_result in active_results.items():
        print(
            f"  {symbol:<16} 交易 {len(symbol_result.trades):>4d} 笔，"
            f"持仓时间 {symbol_result.metrics.time_in_market:>6.1%}，"
            f"敞口 {symbol_result.metrics.avg_exposure:>6.1%}"
        )

    summary = result.summary()
    if args.report_dir:
        report_paths = write_portfolio_reports(result, args.report_dir)
        print(f"\n  报告已写入: {args.report_dir}")
        print(f"  组合曲线: {report_paths['portfolio_svg']}")
        print(f"  逐期账本: {report_paths['trade_csv']}")

    if args.json:
        summary["n_symbols"] = len(active_results)
        summary["per_symbol"] = {
            symbol: symbol_result.summary()
            for symbol, symbol_result in active_results.items()
        }
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "symbols": selected_symbols,
                    "candidate_count": len(typed_rates),
                    "selection_mode": "automatic" if automatic_selection else "explicit",
                    "history_days": config.backtest.history_days,
                    "rolling_window_periods": config.backtest.rolling_window_periods,
                    "reports": report_paths if args.report_dir else {},
                    **summary,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  详细结果已写入 {out_path}")

    return 0


# ---------------------------------------------------------------------------
# 参数解析


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cointrader",
        description="币安资金费率套利研究与回测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "安全提示：默认状态下本程序无法进行真实交易。\n"
            "开启真实交易需要同时设置三个环境变量，见 docs/ARCHITECTURE.md §4。"
        ),
    )
    parser.add_argument("--version", action="version", version=f"cointrader {__version__}")
    parser.add_argument("--config", type=Path, help="配置文件路径（默认 config/config.yaml）")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="环境变量文件")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出调试日志")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # doctor
    sub = subparsers.add_parser("doctor", help="安全与环境自检（无需网络/密钥）")
    sub.set_defaults(func=cmd_doctor)

    # ping
    sub = subparsers.add_parser("ping", help="币安连通性与时钟偏移自检")
    sub.set_defaults(func=cmd_ping)

    # costs
    sub = subparsers.add_parser("costs", help="打印成本模型明细")
    sub.set_defaults(func=cmd_costs)

    # scan
    sub = subparsers.add_parser("scan", help="扫描全市场资金费候选")
    sub.add_argument("--top", type=int, default=20, help="展示前 N 名（默认 20）")
    sub.add_argument("--max-symbols", type=int, default=None, help="最多扫描多少个币")
    sub.add_argument("--lookback-days", type=int, default=None, help="回看天数")
    sub.add_argument("--min-volume", type=float, default=None, help="最小 3 天平均日成交额（USDT）")
    sub.add_argument("--json", type=str, default=None, help="把结果写入指定 JSON 文件")
    sub.set_defaults(func=cmd_scan)

    # backtest 默认自动发现全市场候选并运行动态组合。
    sub = subparsers.add_parser("backtest", help="自动选币组合回测")
    sub.add_argument("--rolling-window", type=int, default=None, help="覆盖固定滚动窗口期数")
    sub.add_argument("--max-symbols", type=int, default=None, help="自动选币时最多加载多少个候选币")
    sub.add_argument("--min-volume", type=float, default=None, help="自动选币时最低 3 天平均日成交额（USDT）")
    sub.add_argument("--report-dir", type=Path, default=Path("reports/backtest"), help="CSV/SVG 报告目录")
    sub.add_argument("--json", type=str, default=None, help="把结果写入指定 JSON 文件")
    sub.set_defaults(func=cmd_backtest)

    # portfolio 未指定币对时自动发现全市场 USDT 本位永续，并按事件时间动态选币。
    sub = subparsers.add_parser("portfolio", help="全市场自动选币组合滚动回测")
    sub.add_argument("symbols", nargs="*", help="可选：仅用于限定候选池的合约符号")
    sub.add_argument("--rolling-window", type=int, default=None, help="覆盖固定滚动窗口期数")
    sub.add_argument("--max-symbols", type=int, default=None, help="自动选币时最多加载多少个候选币")
    sub.add_argument("--min-volume", type=float, default=None, help="自动选币时最低 3 天平均日成交额（USDT）")
    sub.add_argument("--report-dir", type=Path, default=Path("reports/backtest"), help="CSV/SVG 报告目录")
    sub.add_argument("--json", type=str, default=None, help="把结果写入指定 JSON 文件")
    sub.set_defaults(func=cmd_portfolio)

    # scenarios
    sub = subparsers.add_parser("scenarios", help="破产情景分析与仓位建议")
    sub.add_argument("--capital", type=float, required=True, help="投入资金（USDT）")
    sub.add_argument("--expected-return", type=float, default=0.15, help="预期年化（默认 0.15）")
    sub.add_argument("--exchange-exposure", type=float, default=1.0, help="单一交易所资金占比（默认 1.0）")
    sub.add_argument("--leverage", type=float, default=3.0, help="永续腿杠杆（默认 3）")
    sub.add_argument("--max-loss-tolerance", type=float, default=0.20, help="可承受最大回撤（默认 0.20）")
    sub.set_defaults(func=cmd_scenarios)

    # live：实盘执行（testnet/shadow/live；paper 走回测/broker 路径）
    sub = subparsers.add_parser("live", help="实盘执行服务（testnet/shadow/live）")
    live_sub = sub.add_subparsers(dest="live_command", required=True)
    live_sub.add_parser("doctor", help="实盘启动预检（时钟/凭证/停机开关/模式）").set_defaults(func=cmd_live_doctor)
    live_run = live_sub.add_parser("run", help="启动实盘主循环（Ctrl-C 优雅停机）")
    live_run.add_argument("--confirm-live", action="store_true",
                          help="LIVE 模式必须显式确认（主网启动清单 §13）")
    live_run.set_defaults(func=cmd_live_run)

    # 只读查询命令（§8.6：不触发下单/撤单/杠杆/配置写入）
    live_status = live_sub.add_parser("status", help="只读：服务/账户/对账/流新鲜度/风险状态")
    live_status.add_argument("--json", action="store_true", help="JSON 输出")
    live_status.set_defaults(func=cmd_live_status)

    live_positions = live_sub.add_parser("positions", help="只读：当前策略持仓与未实现 PnL")
    live_positions.add_argument("--json", action="store_true", help="JSON 输出")
    live_positions.set_defaults(func=cmd_live_positions)

    live_orders = live_sub.add_parser("orders", help="只读：订单历史与状态")
    live_orders.add_argument("--since", default="24h", help="时间范围（如 24h/7d/120m）")
    live_orders.add_argument("--until", default=None, help="截止时间（同 --since 格式）")
    live_orders.add_argument("--symbol", default=None, help="按 symbol 过滤")
    live_orders.add_argument("--json", action="store_true", help="JSON 输出")
    live_orders.set_defaults(func=cmd_live_orders)

    live_trades = live_sub.add_parser("trades", help="只读：pair 交易/两腿成交与手续费")
    live_trades.add_argument("--since", default="24h", help="时间范围（如 24h/7d/120m）")
    live_trades.add_argument("--until", default=None, help="截止时间（同 --since 格式）")
    live_trades.add_argument("--symbol", default=None, help="按 symbol 过滤")
    live_trades.add_argument("--json", action="store_true", help="JSON 输出")
    live_trades.set_defaults(func=cmd_live_trades)

    live_pnl = live_sub.add_parser("pnl", help="只读：PnL 分项（funding/fee/basis/realized/unrealized）")
    live_pnl.add_argument("--since", default="24h",
                          help="时间范围（如 24h/7d），或 run_start 从 run 开始")
    live_pnl.add_argument("--until", default=None, help="截止时间（同 --since 格式）")
    live_pnl.add_argument("--symbol", default=None, help="按 symbol 过滤")
    run_group = live_pnl.add_mutually_exclusive_group()
    run_group.add_argument("--run-id", default=None, help="指定 run（默认最近一个）")
    run_group.add_argument("--all-runs", action="store_true",
                           help="跨所有 run 聚合（含跨 run 未平仓 pair）")
    live_pnl.add_argument("--json", action="store_true", help="JSON 输出")
    live_pnl.set_defaults(func=cmd_live_pnl)

    live_report = live_sub.add_parser("report", help="只读：导出完整 AI 报告数据包")
    live_report.add_argument("--run-id", default=None, help="指定 run（默认最近一个）")
    live_report.add_argument("--output", default=None, help="输出目录（默认 reports/live/<run_id>）")
    live_report.set_defaults(func=cmd_live_report)

    return parser


# ---------------------------------------------------------------------------
# live：只读查询（开发文档 §8.6）
#
# 硬约束：这些命令只读 SQLite 和必要的只读缓存/公开行情，
# 不调用下单、撤单、杠杆变更或配置写入接口（§12 错误九）。
# ---------------------------------------------------------------------------


def _parse_since_ms(value: str | None, *, default_hours: float = 24.0) -> int:
    """解析 '24h' / '7d' / '120m' / '90s' 为时长毫秒；空 → 默认 24h。"""
    if not value:
        value = "24h"
    value = value.strip().lower()
    if not value:
        value = "24h"
    units = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}
    suffix = value[-1]
    if suffix in units:
        try:
            return int(float(value[:-1]) * units[suffix])
        except ValueError as exc:
            raise ConfigError(f"无法解析时间范围: {value!r}") from exc
    try:
        return int(value)  # 直接毫秒数
    except ValueError as exc:
        raise ConfigError(f"无法解析时间范围: {value!r}（支持 s/m/h/d 或毫秒数）") from exc


def _open_live_store(config: Config):
    """只读打开账本（只调用 SELECT 方法）。"""
    from .execution.store import StateStore

    db_path = config.resolved_path(config.execution.state_db)
    if not db_path.exists():
        raise ConfigError(f"状态账本不存在: {db_path}（先运行 live run）")
    return StateStore(db_path)


def _fetch_live_quotes(config: Config, symbols: list[str]) -> dict:
    """尽力获取新鲜公开报价（失败 → 空，不阻断查询）。"""
    if not symbols:
        return {}
    try:
        from decimal import Decimal

        from .data.binance import BinancePublicClient

        with BinancePublicClient(config.api, config.data) as client:
            out = {}
            for symbol in symbols:
                try:
                    spot = Decimal(str(client.spot_price(symbol)))
                    perp = Decimal(str(client.premium_index(symbol)["lastPrice"]))
                    out[symbol] = (spot, perp)
                except Exception as exc:  # noqa: BLE001
                    logging.getLogger(__name__).debug("获取 %s 报价失败: %s", symbol, exc)
                    continue
            return out
    except Exception:  # noqa: BLE001
        return {}


def _print_json(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_live_status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    try:
        mode, mode_source = _mode_source_and_value(config)
    except ConfigError as exc:
        print(f"  模式解析失败: {exc}", file=sys.stderr)
        return 1
    endpoint_label, spot_base, perp_base = _describe_endpoints(config)
    store = _open_live_store(config)
    try:
        runtime = store.runtime_state()
        latest_run = store.latest_run_session()
        recon_rows = store.reconciliation_runs(limit=1)
        recon = recon_rows[0] if recon_rows else None
        acct_rows = store.account_snapshots(limit=2)
        acct = acct_rows[0] if acct_rows else None
        alerts = [e for e in store.exchange_events(limit=20) if str(e.get("event_type")) == "alert"]
        now_ms = int(time.time() * 1000)
        payload = {
            "now_ms": now_ms,
            "service_state": runtime.get("service_state", {}).get("value"),
            "recovery_reason": runtime.get("recovery_reason", {}).get("value", ""),
            "run_id": runtime.get("run_id", {}).get("value") or (latest_run or {}).get("run_id"),
            "run": latest_run,
            "mode": runtime.get("mode", {}).get("value"),
            "resolved_mode": mode,
            "mode_source": mode_source,
            "endpoints": {"label": endpoint_label, "spot": spot_base, "futures": perp_base},
            "can_open": runtime.get("can_open", {}).get("value") == "1",
            "account_snapshot": {
                "ts_ms": acct.get("ts_ms") if acct else None,
                "age_ms": (now_ms - int(acct["ts_ms"])) if acct and acct.get("ts_ms") else None,
                "total_capital": runtime.get("total_capital", {}).get("value"),
            },
            "last_reconciliation": {
                "ts_ms": recon.get("ts_ms") if recon else None,
                "consistent": bool(recon.get("consistent")) if recon else None,
                "mismatches": [m for m in str(recon.get("mismatches", "")).split(",") if m] if recon else [],
            },
            "last_reconcile_age_ms": (
                now_ms - int(runtime["last_reconcile_ms"]["value"])
                if runtime.get("last_reconcile_ms") and runtime["last_reconcile_ms"]["value"]
                else None
            ),
            "recent_alerts": [{"ts_ms": a.get("recv_ts"), "market": a.get("market"), "type": a.get("event_type"), "payload": a.get("payload")} for a in alerts],
            "run_continuity": _run_continuity(store, now_ms),
            "schema_version": store.schema_version(),
        }
    finally:
        store.close()
    if args.json:
        return _print_json(payload)
    print(f"  服务状态     : {payload['service_state'] or '未知'}"
          + (f"（{payload['recovery_reason']}）" if payload.get("recovery_reason") else ""))
    print(f"  run_id       : {payload['run_id'] or '无'}")
    print(f"  模式         : {payload['mode'] or '未知'}（来源: {payload['mode_source']}，当前解析: {payload['resolved_mode']}）")
    print(f"  端点选择     : {payload['endpoints']['label']}（spot={payload['endpoints']['spot']} perp={payload['endpoints']['futures']}）")
    print(f"  允许开仓     : {'是' if payload['can_open'] else '否'}")
    acct = payload["account_snapshot"]
    print(f"  账户快照年龄 : {acct['age_ms'] if acct['age_ms'] is not None else '无'} ms"
          f"   总资金: {acct['total_capital'] or '未知'}")
    recon = payload["last_reconciliation"]
    print(f"  最后对账     : {'一致' if recon['consistent'] else '不一致'}"
          f"（{recon['ts_ms']}）差异: {recon['mismatches'] or '无'}")
    print(f"  对账年龄     : {payload['last_reconcile_age_ms']} ms")
    cont = payload["run_continuity"]
    print("  运行连续性   :")
    if not cont["has_any_session"]:
        print("    无 run_session（尚未运行过 live run）")
    else:
        cur = cont["current_run"]
        cur_line = f"    当前 run : {cur['run_id']} [{cur['status']}] 时长 {cur['duration_ms']} ms"
        if cur["stop_reason"]:
            cur_line += f"  停止原因: {cur['stop_reason']}"
        print(cur_line)
        prev = cont["previous_run"]
        if prev["run_id"] is not None:
            print(f"    历史 run : {prev['run_id']} [{prev['status']}] 时长 {prev['duration_ms']} ms"
                  f"  停止原因: {prev['stop_reason'] or '无'}")
        else:
            print("    历史 run : 无")
        if cont["open_pairs"]:
            symbols = ", ".join(sorted({p["symbol"] for p in cont["open_pairs"]}))
            print(f"    未完结 pair: {len(cont['open_pairs'])} 个（{symbols}）")
        else:
            print("    未完结 pair: 无")
        if cont["lease_holders"]:
            for h in cont["lease_holders"]:
                print(f"    锁持有者   : {h['holder']} (pid={h['pid']}, expires_ms={h['expires_ms']})")
        else:
            print("    锁持有者   : 无（当前无实例持有锁）")
    if payload["recent_alerts"]:
        print("  最近告警     :")
        for alert in payload["recent_alerts"][:5]:
            print(f"    - [{alert['ts_ms']}] {alert['payload']}")
    return 0


def cmd_live_positions(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    store = _open_live_store(config)
    try:
        snaps = store.position_snapshots(limit=10000)
        latest: dict[str, dict] = {}
        for row in snaps:
            symbol = row.get("symbol")
            if symbol and symbol not in latest:
                latest[symbol] = row
        positions = []
        for symbol, row in sorted(latest.items()):
            spot_qty = Decimal(str(row.get("spot_qty") or 0))
            perp_qty = Decimal(str(row.get("perp_qty") or 0))
            if spot_qty <= 0 and abs(perp_qty) <= 0:
                continue
            opened_ms = store.position_opened_ms(symbol) or row.get("ts_ms")
            positions.append({
                "symbol": symbol,
                "spot_qty": str(spot_qty),
                "perp_qty": str(perp_qty),
                "spot_price": row.get("spot_price"),
                "perp_price": row.get("perp_price"),
                "basis_pct": row.get("basis_pct"),
                "hedge_ratio": (
                    str(abs(perp_qty) / spot_qty) if spot_qty > 0 else None
                ),
                "opened_ms": opened_ms,
                "age_ms": int(time.time() * 1000) - int(opened_ms) if opened_ms else None,
                "snapshot_ts_ms": row.get("ts_ms"),
            })
        quotes = _fetch_live_quotes(config, [str(p["symbol"]) for p in positions])
        runs = store.latest_run_session()
        if runs and positions:
            try:
                from .reporting.pnl import PnlAggregator

                aggregator = PnlAggregator(store)
                summary = aggregator.for_run(str(runs["run_id"]), quotes=quotes)
                for item in summary.per_pair:
                    for pos in positions:
                        if pos["symbol"] == item.symbol:
                            pos["unrealized_pnl"] = str(item.unrealized_pnl)
                            pos["pair_execution_id"] = item.pair_execution_id
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).debug("PnL 聚合失败（不影响持仓展示）: %s", exc)
        payload = {"positions": positions, "run_id": (runs or {}).get("run_id")}
    finally:
        store.close()
    if args.json:
        return _print_json(payload)
    if not positions:
        print("  当前无策略持仓。")
        return 0
    for pos in positions:
        print(f"  {pos['symbol']}  spot={pos['spot_qty']} perp={pos['perp_qty']} "
              f"hedge={pos['hedge_ratio']} basis={pos['basis_pct']} "
              f"age={pos['age_ms']}ms unrealized={pos.get('unrealized_pnl', 'n/a')}")
    return 0


def _table_rows_for_query(args, method: str) -> list[dict]:
    config = load_config(args.config)
    store = _open_live_store(config)
    try:
        since_ms = int(time.time() * 1000) - _parse_since_ms(args.since)
        until_ms = None
        if args.until:
            until_ms = int(time.time() * 1000) - _parse_since_ms(args.until)
        return list(getattr(store, method)(
            since_ms=since_ms, until_ms=until_ms, symbol=args.symbol or None
        ))
    finally:
        store.close()


def _print_rows(rows: list[dict], columns: list[str]) -> None:
    if not rows:
        print("  无数据。")
        return
    for row in rows:
        parts = [f"{col}={row.get(col)}" for col in columns if row.get(col) not in (None, "")]
        print("  " + "  ".join(parts))


def cmd_live_orders(args: argparse.Namespace) -> int:
    rows = _table_rows_for_query(args, "orders")
    if args.json:
        return _print_json({"orders": rows})
    _print_rows(rows, ["client_order_id", "exchange_order_id", "symbol", "market", "side",
                       "order_type", "quantity", "executed_qty", "avg_price", "state",
                       "pair_execution_id", "intent_id", "failure_class"])
    return 0


def cmd_live_trades(args: argparse.Namespace) -> int:
    rows = _table_rows_for_query(args, "fills")
    if args.json:
        return _print_json({"fills": rows})
    _print_rows(rows, ["fill_id", "symbol", "market", "side", "quantity", "price",
                       "fee_asset", "fee_amount", "ts_ms", "pair_execution_id",
                       "maker_taker", "exchange_trade_id"])
    return 0


def _describe_run_session(session: dict[str, Any] | None, now_ms: int) -> dict[str, Any]:
    """run_session → 连续性块字段（未结束的会话用当前时刻估算时长）。"""
    if session is None:
        return {"run_id": None}
    started = int(session.get("started_ms") or 0)
    ended = session.get("ended_ms")
    end_ms = int(ended) if ended is not None else now_ms
    return {
        "run_id": session.get("run_id"),
        "status": session.get("status"),
        "stop_reason": session.get("stop_reason") or "",
        "started_ms": started or None,
        "ended_ms": ended,
        "duration_ms": end_ms - started if started else None,
    }


def _run_continuity(store: Any, now_ms: int) -> dict[str, Any]:
    """运行连续性块（断点重连诊断：跨重启看当前/历史 run、未完结 pair、锁持有者）。"""
    sessions = store.run_sessions(limit=2)
    latest = sessions[0] if sessions else None
    previous = sessions[1] if len(sessions) > 1 else None
    return {
        "has_any_session": latest is not None,
        "current_run": _describe_run_session(latest, now_ms),
        "previous_run": _describe_run_session(previous, now_ms),
        "open_pairs": [
            {"pair_execution_id": str(r.get("pair_execution_id")),
             "symbol": str(r.get("symbol")), "kind": str(r.get("kind")),
             "run_id": r.get("run_id")}
            for r in store.open_pairs()
        ],
        "lease_holders": store.lease_holders(),
    }


def cmd_live_pnl(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    store = _open_live_store(config)
    try:
        all_runs = bool(getattr(args, "all_runs", False))
        run = store.latest_run_session()
        if run is None:
            raise ConfigError("没有找到 run_session（先运行 live run）")
        if not all_runs and args.run_id:
            run = store.run_session(args.run_id)
            if run is None:
                raise ConfigError(f"没有找到 run_session: {args.run_id}")
        since_arg = (args.since or "24h").strip().lower()
        if since_arg in ("run_start", "runstart") and not all_runs:
            since_ms = int(run.get("started_ms") or 0)
        elif since_arg in ("run_start", "runstart"):
            sessions = store.run_sessions(limit=100000)
            since_ms = min(int(s.get("started_ms") or 0) for s in sessions) if sessions else 0
        else:
            since_ms = int(time.time() * 1000) - _parse_since_ms(args.since)
        if args.until:
            until_ms = int(time.time() * 1000) - _parse_since_ms(args.until)
            since_ms = max(since_ms, 0)
        else:
            until_ms = None
        held_symbols = [str(r.get("symbol")) for r in store.open_pairs()
                        if str(r.get("kind")) == "open"]
        quotes = _fetch_live_quotes(config, held_symbols)
        from .reporting.pnl import PnlAggregator

        aggregator = PnlAggregator(store)
        if all_runs:
            summary = aggregator.for_all_runs(quotes=quotes)
            run_id = "ALL"
        else:
            summary = aggregator.for_run(str(run["run_id"]), quotes=quotes)
            run_id = str(run["run_id"])
        payload: dict[str, Any] = {
            "run_id": run_id,
            "since_ms": since_ms,
            "until_ms": until_ms,
            "pnl": summary.to_dict(),
        }
        if args.symbol:
            payload["pnl"]["per_pair"] = [
                p for p in payload["pnl"]["per_pair"] if p["symbol"] == args.symbol
            ]
    finally:
        store.close()
    if args.json:
        return _print_json(payload)
    pnl = payload["pnl"]
    print(f"  run_id       : {payload['run_id']}")
    print(f"  funding_pnl  : {pnl['funding_pnl']}")
    print(f"  trading_fee  : {pnl['trading_fee']}")
    print(f"  basis_pnl    : {pnl['basis_pnl']}")
    print(f"  realized     : {pnl['realized_pnl']}")
    print(f"  unrealized   : {pnl['unrealized_pnl']}")
    print(f"  net_pnl      : {pnl['net_pnl']}（口径 {pnl['calculation_version']}）")
    print(f"  cash_delta   : {pnl['cash_delta']}（仅交叉验证，不替代策略 PnL）")
    if pnl["differences"]:
        print("  差异         :")
        for difference in pnl["differences"]:
            print(f"    - {difference}")
    return 0


def cmd_live_report(args: argparse.Namespace) -> int:
    from .reporting.export import export_report

    config = load_config(args.config)
    store = _open_live_store(config)
    try:
        run = store.run_session(args.run_id) if args.run_id else store.latest_run_session()
        if run is None:
            raise ConfigError("没有找到 run_session（先运行 live run）")
        run_id = str(run["run_id"])
        if str(run.get("status")) not in ("STOPPED",):
            print(f"  ⚠️ run {run_id} 状态 {run.get('status')}（未结束）："
                  "请先停止新增信号并完成最终对账/快照。")
        output = (
            Path(args.output)
            if args.output
            else config.resolved_path(config.execution.report_dir) / run_id
        )
        held_symbols = [str(r.get("symbol")) for r in store.open_pairs()
                        if str(r.get("kind")) == "open"]
        quotes = _fetch_live_quotes(config, held_symbols)
        result = export_report(
            store,
            config,
            run_id=run_id,
            output_dir=output,
            strategy_version=str(run.get("strategy_version") or ""),
            config_hash=str(run.get("config_hash") or ""),
            quotes=quotes,
        )
    finally:
        store.close()
    print(f"  数据包导出: {result.output_dir}")
    print(f"  manifest  : {result.manifest_path}")
    print(f"  完整性    : {'PASS' if result.ok else 'FAIL'}"
          f"   脱敏检查: {'PASS' if not result.redaction_hits else 'FAIL'}")
    if result.redaction_hits:
        for hit in result.redaction_hits:
            print(f"    - {hit}")
    return 0 if result.ok else 1


# ---------------------------------------------------------------------------
# live：实盘执行（文档 §4.2 / §8 / §13）
# ---------------------------------------------------------------------------


EXEC_MODE_ENV = "COINTRADER_EXEC_MODE"


def _mode_source_and_value(config: Config) -> tuple[str, str]:
    """(mode, source)：诊断用解析，不含 live 闸门检查。

    ``COINTRADER_EXEC_MODE`` 精确覆盖 ``config.execution.mode``（白名单精确匹配，
    小写；不做大小写折叠，``LIVE`` 不是合法值）。非法 → ConfigError 并列合法值。
    """
    mode = config.execution.mode
    source = "config"
    raw_env = os.environ.get(EXEC_MODE_ENV, "")
    if raw_env:
        if raw_env not in EXECUTION_MODES:
            raise ConfigError(
                f"COINTRADER_EXEC_MODE 非法: {raw_env!r}，合法值: paper/testnet/shadow/live"
            )
        mode = raw_env
        source = f"env:{EXEC_MODE_ENV}"
    return mode, source


def _resolve_live_mode(config: Config) -> tuple[str, str]:
    """解析并校验实盘执行模式，返回 (mode, source)。

    env 覆盖不得绕过任何闸门：``live`` 仍要求 secrets 层双重开关
    （魔法串 + USE_TESTNET=false）与 --confirm-live（后者在调用方检查）。
    """
    from .secrets import is_trading_enabled, use_testnet

    mode, source = _mode_source_and_value(config)
    if mode == "live" and not (is_trading_enabled() and not use_testnet()):
        raise ConfigError(
            "execution.mode=live 要求 COINTRADER_TRADING_ENABLED 魔法字符串 "
            "且 COINTRADER_USE_TESTNET=false 同时生效。这是有意的双重确认（§4.2）。"
        )
    return mode, source


def _describe_endpoints(config: Config) -> tuple[str, str, str]:
    """端点选择（与 build_live_context 一致）：use_testnet() 为真 → testnet base（demo/经典测试网），否则 mainnet。"""
    from .secrets import use_testnet

    api = config.api
    if use_testnet():
        return "demo/testnet", api.spot_testnet_base, api.futures_testnet_base
    return "mainnet", api.spot_base, api.futures_base


def cmd_live_doctor(args: argparse.Namespace) -> int:
    """实盘启动预检：模式、时钟偏移、凭证、停机开关。不发任何私有请求。"""
    config = load_config(args.config)
    failures: list[str] = []

    print("\n  【live doctor】实盘启动预检")
    try:
        mode, mode_source = _resolve_live_mode(config)
    except ConfigError as exc:
        print(f"  模式解析失败: {exc}")
        return 1
    exc_cfg = config.execution
    print(f"  模式: {mode.upper()}（来源: {mode_source}）")
    endpoint_label, spot_base, perp_base = _describe_endpoints(config)
    print(f"  端点: {endpoint_label}（spot={spot_base} perp={perp_base}）")
    print(f"  状态账本: {config.resolved_path(exc_cfg.state_db)}")
    print(f"  canary 名义额: {exc_cfg.canary_notional} USDT   杠杆: {exc_cfg.leverage}x   保证金: {exc_cfg.margin_type}")

    posture = describe_security_posture()
    print(f"  停机开关: {posture['kill_switch_file']} "
          f"({'⚠️ 已触发' if posture['kill_switch_engaged'] else '未触发'})")
    if posture["kill_switch_engaged"]:
        failures.append("停机开关已触发：先处理事故或移除停机文件（恢复仍需重新预检+对账）")

    if mode in ("testnet", "live"):
        creds = load_credentials(require=False)
        if creds is None:
            failures.append("缺少 API 凭证（环境变量）")
        else:
            print(f"  凭证: {creds!r}")

    # 时钟偏移（公共接口，无私有请求）
    from .data.binance import BinancePublicClient

    try:
        with BinancePublicClient(config.api, config.data) as client:  # type: ignore[arg-type]
            spot_offset = client.spot_time() - int(time.time() * 1000)
            perp_offset = client.futures_time() - int(time.time() * 1000)
        limit = exc_cfg.server_time_offset_limit_ms
        print(f"  时钟偏移: spot {spot_offset:+d}ms / perp {perp_offset:+d}ms（阈值 ±{limit}ms）")
        if abs(spot_offset) > limit or abs(perp_offset) > limit:
            failures.append("时钟偏移超阈：同步系统 NTP 后重试")
    except Exception as exc2:  # noqa: BLE001
        failures.append(f"时钟自检失败（网络/代理？）：{exc2}")

    if failures:
        print("\n  ❌ 预检未通过:")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  ✅ 预检通过")
    return 0


def _install_sigterm_handler() -> None:
    """SIGTERM → KeyboardInterrupt，复用既有 Ctrl-C 优雅停机路径（systemd stop 用）。"""

    def _handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handler)


def cmd_live_run(args: argparse.Namespace) -> int:
    """启动实盘主循环。Ctrl-C / SIGTERM → 优雅停机（关流 + 释放锁 + 关闭 run_session）。"""
    config = load_config(args.config)
    try:
        mode, mode_source = _resolve_live_mode(config)
    except ConfigError as exc:
        print(f"  ❌ 模式解析失败: {exc}", file=sys.stderr)
        return 1

    if mode == "paper":
        print("  PAPER 模式不走实盘执行服务（本地模拟请用 backtest/broker 路径）。")
        print("  如需真实订单：设 execution.mode=testnet 并配置测试网密钥。")
        return 2
    if mode == "live" and not getattr(args, "confirm_live", False):
        print("  LIVE 模式拒绝启动：缺少 --confirm-live 显式确认。")
        print("  先完成 docs 主网启动清单（§13）：key 权限/IP 白名单/对账/canary 配置/告警通道。")
        return 2

    from .live.service import build_live_context
    from .secrets import load_credentials

    creds = load_credentials(require=True)
    if creds is None:
        print("  缺少 API 凭证（环境变量），无法启动。", file=sys.stderr)
        return 1
    # Spot 与 Futures 可用同一对 Key；也可用独立环境变量覆盖（不同 Key 场景）。
    spot_key = os.environ.get("COINTRADER_SPOT_API_KEY", "") or creds.key
    spot_secret = os.environ.get("COINTRADER_SPOT_API_SECRET", "") or creds.secret
    fut_key = os.environ.get("COINTRADER_FUTURES_API_KEY", "") or creds.key
    fut_secret = os.environ.get("COINTRADER_FUTURES_API_SECRET", "") or creds.secret

    service = build_live_context(
        config=config,
        spot_key=spot_key.reveal() if hasattr(spot_key, "reveal") else spot_key,
        spot_secret=spot_secret.reveal() if hasattr(spot_secret, "reveal") else spot_secret,
        futures_key=fut_key.reveal() if hasattr(fut_key, "reveal") else fut_key,
        futures_secret=fut_secret.reveal() if hasattr(fut_secret, "reveal") else fut_secret,
        is_testnet=creds.is_testnet,
    )

    # SIGTERM 处理必须在 startup() 之前注册（systemd stop → 优雅停机，不留 ended_ms=NULL）
    _install_sigterm_handler()

    print(f"\n  【live run】mode={mode.upper()}（来源: {mode_source}）启动中…（Ctrl-C/SIGTERM 优雅停机）")
    try:
        report = service.startup()
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 启动失败：{exc}", file=sys.stderr)
        return 1
    print(f"  启动完成：symbols={list(report.symbols)} 状态={service.state.value}")
    if service.state.value != "RUNNING":
        print(f"  ⚠️ 处于 {service.state.value}：{service._recovery_reason}（只允许减仓，禁止开仓）")  # noqa: SLF001

    ok = True
    webui = _start_live_webui(config, service)
    try:
        ok = service.run_forever(tick_seconds=min(5.0, config.execution.reconciliation_interval_seconds))
    except KeyboardInterrupt:
        print("\n  收到停止信号（Ctrl-C/SIGTERM），优雅停机…")
    finally:
        _stop_live_webui(webui)
        service.stop()
        service.store.close()
    if not ok:
        print("  ❌ 连续 tick 失败超限，退出码 1（守护进程将重启并重新预检/对账）", file=sys.stderr)
        return 1
    return 0


def _start_live_webui(config: Config, service) -> object | None:
    """随 live run 启动只读 WebUI 仪表盘（独立守护线程）。

    任何失败（端口占用/模板缺失/意外异常）只告警，绝不影响主循环。
    """
    if not config.execution.webui_enabled:
        print("  WebUI      : 已禁用（execution.webui_enabled）")
        return None
    try:
        from .webui.server import LiveWebUI

        webui = LiveWebUI(config=config, state_provider=service.web_snapshot)
        if webui.start():
            host, port = webui.address or ("?", 0)
            shown = "0.0.0.0" if host == "0.0.0.0" else host  # noqa: S104
            print(f"  WebUI      : http://{shown}:{port}/（只读实时仪表盘，故障不影响主循环）")
            return webui
        print(f"  ⚠️ WebUI 未启动: {webui.last_error}（不影响主循环）", file=sys.stderr)
        return None
    except Exception as exc:  # noqa: BLE001 —— 故障隔离底线
        print(f"  ⚠️ WebUI 启动异常（不影响主循环）: {exc}", file=sys.stderr)
        return None


def _stop_live_webui(webui: object | None) -> None:
    if webui is None:
        return
    try:
        webui.stop()  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ WebUI 停止异常（忽略）: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # .env 加载必须在任何配置读取之前。
    # 注意 override=False：环境变量优先级高于文件。
    if args.env_file and Path(args.env_file).is_file():
        loaded = load_dotenv(args.env_file)
        if loaded:
            import logging

            logging.getLogger(__name__).debug("从 %s 加载了 %d 个环境变量", args.env_file, loaded)

    # 日志
    level = "DEBUG" if args.verbose else "INFO"
    log_dir: Path | None = None
    try:
        config = load_config(args.config)
        level = "DEBUG" if args.verbose else config.logging.level
        log_dir = config.resolved_path(config.logging.log_dir)
    except Exception as exc:  # noqa: BLE001
        # 配置有问题时不阻塞 doctor —— doctor 正是用来诊断配置的。
        # 但要记录原因，否则用户看不到真实错误。
        print(f"  [提示] 配置加载失败，使用默认日志设置: {exc}", file=sys.stderr)

    setup_logging(level=level, log_dir=log_dir, structured=False, console=True)

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n  已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
