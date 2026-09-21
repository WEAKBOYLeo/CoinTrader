"""市场数据服务：scan epoch → 领域 ``MarketSnapshot``（实施计划书 3.0 T2）。

硬约束：

- 只读、不发起网络 IO（数据来自注入的 ``ScanEpochSource``，生产实现为
  ``MarketDataSynchronizer``，限流/缓存/重试全部复用现有实现）。
- READY epoch → ``FRESH``；非 READY → ``DEGRADED``/``STALE``/``INCOMPLETE``，
  调用方不得据此新开仓。
- 不伪造价格：mark 价非正的候选跳过；无数据返回空 quotes + ``INCOMPLETE``。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from ..domain.common import InvalidDomainValue
from ..domain.market import DataQuality, InstrumentQuote, MarketKind, MarketSnapshot
from .ports import MarketDataPort, ScanEpochSource

__all__ = ["MarketDataService"]


def _status_name(status: object) -> str:
    value = getattr(status, "value", status)
    return str(value)


class MarketDataService(MarketDataPort):
    """以最近 READY epoch 为事实源的市场数据 facade。

    Args:
        source: scan epoch 源（结构协议；``MarketDataSynchronizer`` 满足）。
        now_fn: 可注入时钟（秒），便于固定时间测试。
    """

    def __init__(self, source: ScanEpochSource, *, now_fn: Callable[[], float] = time.time) -> None:
        self._source = source
        self._now = now_fn

    def snapshot(self, symbols: Sequence[str] | None = None) -> MarketSnapshot:
        generated_at_ms = int(self._now() * 1000)
        allowed = set(symbols) if symbols is not None else None

        epoch = self._source.latest_ready()
        if epoch is not None:
            if _status_name(epoch.status) != "READY":
                raise InvalidDomainValue(
                    f"latest_ready 返回非 READY epoch（{epoch.epoch_id}），实现契约被破坏"
                )
            quality = DataQuality.FRESH
            cutoff_ms = epoch.decision_cutoff_ms
            snapshots = self._source.snapshots_for(epoch.epoch_id)
        else:
            latest = self._source.latest()
            if latest is None:
                return MarketSnapshot(
                    snapshot_id="none",
                    generated_at_ms=generated_at_ms,
                    decision_cutoff_ms=0,
                    quality=DataQuality.INCOMPLETE,
                    quotes=(),
                )
            status = _status_name(latest.status)
            if status == "DEGRADED":
                quality = DataQuality.DEGRADED
            elif status == "EXPIRED":
                quality = DataQuality.STALE
            else:
                quality = DataQuality.INCOMPLETE
            cutoff_ms = latest.decision_cutoff_ms
            snapshots = self._source.snapshots_for(latest.epoch_id)

        quotes: list[InstrumentQuote] = []
        for symbol in sorted(snapshots):
            if allowed is not None and symbol not in allowed:
                continue
            snap = snapshots[symbol]
            if snap.error:
                continue
            if not snap.mark_prices:
                continue
            price = snap.mark_prices[-1]
            if price <= 0:
                continue  # 不伪造默认价格
            quote_time_ms = snap.timestamps[-1] if snap.timestamps else snap.fetched_ms
            quotes.append(
                InstrumentQuote(
                    symbol=symbol,
                    market=MarketKind.FUTURES,
                    price=price,
                    funding_rate_8h=None,
                    funding_cutoff_ms=None,
                    quote_time_ms=quote_time_ms,
                    volume_3d_avg=snap.quote_volume_3d_avg,
                )
            )

        return MarketSnapshot(
            snapshot_id=epoch.epoch_id if epoch is not None else "no-epoch",
            generated_at_ms=generated_at_ms,
            decision_cutoff_ms=cutoff_ms,
            quality=quality,
            quotes=tuple(quotes),
        )
