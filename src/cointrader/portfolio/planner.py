"""组合规划：目标组合与当前组合的差异（实施计划书 3.0 T2）。

硬约束（AC-03）：

- 纯计算：无网络、无执行依赖、无时钟副作用（``created_at_ms`` 为输入参数）。
- 幂等：相同 ``(target, view)`` 输入产生完全相同的 intent 序列
  （``intent_id = fingerprint``，重复输入不产生新意图，可去重）。
- 稳定排序：CLOSE 先于 REPLACE/OPEN，再 RESIZE；同组内按 symbol 升序。
  换仓（close+open 共存）先平旧后开新，**不先扩大总敞口**（任何时刻
  敞口不超过 max(旧, 新)）。
- 换仓语义：同一 diff 内 CLOSE 与 OPEN 共存时，OPEN 腿标记为 ``REPLACE``
  （因果由调用方经 correlation/causation id 串联）。
"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from ..domain.portfolio import (
    CurrentPosition,
    IntentAction,
    PortfolioIntent,
    PortfolioView,
    intent_fingerprint,
)
from ..domain.strategy import TargetPortfolio

__all__ = ["PortfolioPlanner", "PortfolioPlannerPort"]


class PortfolioPlannerPort(Protocol):
    """组合 planner 端口（结构协议）。"""

    def diff(
        self,
        target: TargetPortfolio,
        view: PortfolioView,
        *,
        created_at_ms: int,
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> tuple[PortfolioIntent, ...]:
        """目标与当前的差异 → 稳定排序、幂等的意图序列。"""
        ...


class PortfolioPlanner:
    """目标组合 → 意图列表。确定性、幂等、稳定排序。"""

    def diff(
        self,
        target: TargetPortfolio,
        view: PortfolioView,
        *,
        created_at_ms: int,
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> tuple[PortfolioIntent, ...]:
        current = {e.symbol: e for e in view.entries}
        closes: list[PortfolioIntent] = []
        opens: list[PortfolioIntent] = []
        resizes: list[PortfolioIntent] = []

        def _intent(
            action: IntentAction,
            symbol: str,
            spot: Decimal,
            perp: Decimal,
            reason: str,
        ) -> PortfolioIntent:
            intent_id = intent_fingerprint(
                action, symbol, spot, perp, view.snapshot_id, view.as_of_ms
            )
            return PortfolioIntent(
                intent_id=intent_id,
                action=action,
                symbol=symbol,
                target_spot_notional=spot,
                target_perp_notional=perp,
                reason=reason,
                snapshot_id=view.snapshot_id,
                decision_cutoff_ms=view.as_of_ms,
                created_at_ms=created_at_ms,
                correlation_id=correlation_id,
                causation_id=causation_id,
            )

        # 1) 当前持仓：目标清零/缺失 → CLOSE；目标变化 → RESIZE；一致 → 无意图
        for symbol in sorted(current):
            pos = current[symbol]
            held = pos.spot_notional + pos.perp_notional
            tgt = target.get(symbol)
            target_notional = (
                tgt.spot_notional + tgt.perp_notional if tgt is not None else Decimal("0")
            )
            if held > 0 and target_notional == 0:
                closes.append(
                    _intent(
                        IntentAction.CLOSE,
                        symbol,
                        Decimal("0"),
                        Decimal("0"),
                        f"目标组合已移除 {symbol}（当前名义额 {format(held, 'f')}）",
                    )
                )
            elif held == 0 and target_notional > 0 and tgt is not None:
                opens.append(
                    _intent(
                        IntentAction.OPEN,
                        symbol,
                        tgt.spot_notional,
                        tgt.perp_notional,
                        f"重新开仓 {symbol}（目标 spot={format(tgt.spot_notional, 'f')} "
                        f"perp={format(tgt.perp_notional, 'f')}）",
                    )
                )
            elif held > 0 and tgt is not None and (
                tgt.spot_notional != pos.spot_notional or tgt.perp_notional != pos.perp_notional
            ):
                resizes.append(
                    _intent(
                        IntentAction.RESIZE,
                        symbol,
                        tgt.spot_notional,
                        tgt.perp_notional,
                        f"调整 {symbol}（spot {format(pos.spot_notional, 'f')} → "
                        f"{format(tgt.spot_notional, 'f')}，perp {format(pos.perp_notional, 'f')} → "
                        f"{format(tgt.perp_notional, 'f')}）",
                    )
                )
            # 一致（含双方均为 0）→ 无意图：重复评估不产生重复 intent

        # 2) 目标新增：OPEN（与 CLOSE 共存 = 换仓腿）
        for symbol in sorted({e.symbol for e in target.entries} - set(current)):
            tgt = target.get(symbol)
            if tgt is None or tgt.spot_notional + tgt.perp_notional <= 0:
                continue
            opens.append(
                _intent(
                    IntentAction.OPEN,
                    symbol,
                    tgt.spot_notional,
                    tgt.perp_notional,
                    f"新开 {symbol}（目标 spot={format(tgt.spot_notional, 'f')} "
                    f"perp={format(tgt.perp_notional, 'f')}）",
                )
            )

        # 3) 换仓语义：close 与 open 共存时 OPEN 腿标记 REPLACE（顺序仍 close 在前）
        if closes and opens:
            opens = [
                PortfolioIntent(
                    intent_id=intent_fingerprint(
                        IntentAction.REPLACE,
                        i.symbol,
                        i.target_spot_notional,
                        i.target_perp_notional,
                        i.snapshot_id,
                        i.decision_cutoff_ms,
                    ),
                    action=IntentAction.REPLACE,
                    symbol=i.symbol,
                    target_spot_notional=i.target_spot_notional,
                    target_perp_notional=i.target_perp_notional,
                    reason=f"换仓开新 {i.symbol}（先平旧后开新）",
                    snapshot_id=i.snapshot_id,
                    decision_cutoff_ms=i.decision_cutoff_ms,
                    created_at_ms=i.created_at_ms,
                    correlation_id=i.correlation_id,
                    causation_id=i.causation_id,
                )
                for i in opens
            ]

        # 4) 稳定排序：CLOSE → REPLACE/OPEN → RESIZE；同组 symbol 升序
        return tuple(closes + opens + resizes)

    # -- 视图构造辅助 -----------------------------------------------------------

    @staticmethod
    def view_from_entries(
        entries: tuple[CurrentPosition, ...] | list[CurrentPosition],
        snapshot_id: str,
        as_of_ms: int,
    ) -> PortfolioView:
        """从当前持仓条目构造只读视图（按 symbol 稳定排序）。"""
        return PortfolioView(
            snapshot_id=snapshot_id,
            as_of_ms=as_of_ms,
            entries=tuple(sorted(entries, key=lambda e: e.symbol)),
        )
