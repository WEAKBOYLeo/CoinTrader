"""回测报告输出：从引擎 ledger 派生 CSV 与 SVG，不重复计算收益。"""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

from ..backtest.engine import PortfolioResult


def _scale(values: list[float], height: float, padding: float = 24.0) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    span = high - low or 1.0
    return [height - padding - (value - low) / span * (height - 2 * padding) for value in values]


def _svg_chart(
    title: str,
    series: dict[str, list[float]],
    *,
    width: int = 1100,
    height: int = 420,
) -> str:
    """绘制轻量 SVG 折线图，避免报告依赖 matplotlib。"""
    margin = 64
    plot_width = width - margin * 2
    all_values = [value for values in series.values() for value in values]
    if not all_values:
        all_values = [0.0]
    y_values = _scale(all_values, height - margin, padding=margin)
    del y_values

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{margin}" y="30" font-family="sans-serif" font-size="18" fill="#222">'
        f'{html.escape(title)}</text>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#999"/>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" '
        f'y2="{height - margin}" stroke="#999"/>',
    ]
    palette = ["#1565c0", "#c62828", "#2e7d32", "#6a1b9a"]
    for index, (name, values) in enumerate(series.items()):
        if not values:
            continue
        ys = _scale(values, height, padding=margin)
        count = max(1, len(values) - 1)
        points = " ".join(
            f"{margin + position / count * plot_width:.2f},{y:.2f}"
            for position, y in enumerate(ys)
        )
        color = palette[index % len(palette)]
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        legend_x = margin + index * 180
        lines.append(
            f'<line x1="{legend_x}" y1="48" x2="{legend_x + 24}" y2="48" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{legend_x + 30}" y="53" font-family="sans-serif" font-size="13" '
            f'fill="#333">{html.escape(name)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def _scale_to_band(
    values: list[float], top: float, bottom: float, padding: float = 12.0
) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    span = high - low or 1.0
    return [
        bottom - padding - (value - low) / span * (bottom - top - 2 * padding)
        for value in values
    ]


def _svg_two_panel_chart(
    title: str,
    top: tuple[str, list[float]],
    bottom: tuple[str, list[float]],
    *,
    width: int = 1100,
    height: int = 620,
) -> str:
    """两个共享横轴的面板：交易累计收益与资金费率。"""
    margin = 64
    panel_height = (height - margin * 2 - 48) / 2
    plot_width = width - margin * 2
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{margin}" y="30" font-family="sans-serif" font-size="18" fill="#222">'
        f'{html.escape(title)}</text>',
    ]
    for panel, (label, values), color in zip(
        range(2), (top, bottom), ("#1565c0", "#c62828"), strict=True
    ):
        y0 = margin + panel * (panel_height + 48)
        ys = _scale_to_band(values or [0.0], y0, y0 + panel_height)
        count = max(1, len(values) - 1)
        points = " ".join(
            f"{margin + position / count * plot_width:.2f},{y:.2f}"
            for position, y in enumerate(ys)
        )
        lines.extend(
            [
                f'<text x="{margin}" y="{y0 - 8:.0f}" font-family="sans-serif" '
                f'font-size="14" fill="{color}">{html.escape(label)}</text>',
                f'<line x1="{margin}" y1="{y0}" x2="{margin}" '
                f'y2="{y0 + panel_height}" stroke="#999"/>',
                f'<line x1="{margin}" y1="{y0 + panel_height}" x2="{width - margin}" '
                f'y2="{y0 + panel_height}" stroke="#999"/>',
                f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _period_rows(result: PortfolioResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol_result in result.per_symbol.values():
        rows.extend(period.as_dict() for period in symbol_result.period_ledger)
    return rows


def write_portfolio_reports(result: PortfolioResult, output_dir: str | Path) -> dict[str, str]:
    """写组合曲线、组合逐期账本、逐笔交易逐期账本和单笔 SVG。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    portfolio_index = [str(timestamp) for timestamp in result.portfolio_returns.index]
    portfolio_rows = [
        {
            "period_time": timestamp,
            "portfolio_return": float(result.portfolio_returns.iloc[index]),
            "gross_return": float(result.portfolio_gross_returns.iloc[index]),
            "cost_return": float(result.portfolio_cost_returns.iloc[index]),
            "exposure": float(result.portfolio_exposure.iloc[index]),
            "equity": float((1.0 + result.portfolio_returns).cumprod().iloc[index]),
        }
        for index, timestamp in enumerate(portfolio_index)
    ]
    _write_csv(directory / "portfolio_ledger.csv", portfolio_rows)

    equity = (1.0 + result.portfolio_returns).cumprod().tolist()
    portfolio_svg = directory / "portfolio_equity.svg"
    portfolio_svg.write_text(
        _svg_chart("Portfolio equity, 8h periods", {"equity": equity}),
        encoding="utf-8",
    )

    trade_rows = _period_rows(result)
    _write_csv(directory / "trade_ledger.csv", trade_rows)

    written: dict[str, str] = {
        "portfolio_csv": str(directory / "portfolio_ledger.csv"),
        "portfolio_svg": str(portfolio_svg),
        "trade_csv": str(directory / "trade_ledger.csv"),
    }
    for symbol_result in result.per_symbol.values():
        for trade in symbol_result.trades:
            safe_symbol = "".join(char if char.isalnum() else "_" for char in trade.symbol)
            stem = f"trade_{trade.trade_id:03d}_{safe_symbol}"
            rows = [period.as_dict() for period in trade.periods]
            _write_csv(directory / f"{stem}.csv", rows)
            trade_svg = directory / f"{stem}.svg"
            trade_svg.write_text(
                _svg_two_panel_chart(
                    f"{trade.symbol} trade {trade.trade_id}, 8h periods",
                    (
                        "cumulative net PnL (USDT)",
                        [period.cumulative_net_pnl_usdt for period in trade.periods],
                    ),
                    (
                        "8h funding rate",
                        [period.funding_rate_8h for period in trade.periods],
                    ),
                ),
                encoding="utf-8",
            )
            written[f"trade_{trade.trade_id}_{trade.symbol}"] = str(trade_svg)
    return written


__all__ = ["write_portfolio_reports"]
