"""8h 结算桶归一（live 与回测同口径）测试。

目的：入场/退出/持有的「期」统一按 8h 日历天计——4h 币每桶 2 期求和、
1h 币每桶 8 期求和、8h 币每桶 1 期；结束点 > cutoff 的桶未可见（无前瞻）。
"""

from __future__ import annotations

from decimal import Decimal

from cointrader.data.funding import normalize_funding_records_to_8h

H = 3600 * 1000
#: 对齐到 8h 整点（绝对 UTC 桶边界，非相对锚点）
BASE = 1_700_000_000_000 - (1_700_000_000_000 % (8 * H))


def _recs(pairs: list[tuple[int, str]]) -> list[tuple[int, Decimal, Decimal]]:
    return [(BASE + h * H, Decimal(r), Decimal("1")) for h, r in pairs]


class TestBucketing:
    def test_8h_passthrough(self) -> None:
        recs = _recs([(0, "0.001"), (8, "0.002"), (16, "0.003")])
        ends, rates, _marks = normalize_funding_records_to_8h(recs)
        assert [e - BASE for e in ends] == [0 * H, 8 * H, 16 * H]
        assert rates == [Decimal("0.001"), Decimal("0.002"), Decimal("0.003")]

    def test_4h_pairs_summed_into_8h_buckets(self) -> None:
        # 00:00 结算 → 00:00 桶（右端=自身）；04:00+08:00 → 08:00 桶；12:00+16:00 → 16:00 桶
        recs = _recs([(0, "0.001"), (4, "0.002"), (8, "0.003"), (12, "0.004"), (16, "0.005")])
        ends, rates, _marks = normalize_funding_records_to_8h(recs)
        assert [e - BASE for e in ends] == [0 * H, 8 * H, 16 * H]
        assert rates == [Decimal("0.001"), Decimal("0.005"), Decimal("0.009")]

    def test_1h_octuplets_summed(self) -> None:
        recs = _recs([(h, "0.001") for h in range(16)])
        ends, rates, _marks = normalize_funding_records_to_8h(recs)
        assert [e - BASE for e in ends] == [0 * H, 8 * H, 16 * H]
        # 右端=可见时点（pd.ceil 语义，同回测）：整点结算归以该点结束的桶 →
        # 0 桶仅 h=0；8 桶 = h=1..8；16 桶 = h=9..15（h=16 未生成）
        assert rates == [Decimal("0.001"), Decimal("0.008"), Decimal("0.007")]

    def test_incomplete_trailing_bucket_dropped_at_cutoff(self) -> None:
        # cutoff = 10:00：12:00 结算落在 16:00 桶（未可见）→ 丢弃
        recs = _recs([(0, "0.001"), (4, "0.002"), (8, "0.003"), (12, "0.004")])
        ends, rates, _marks = normalize_funding_records_to_8h(recs, cutoff_ms=BASE + 10 * H)
        assert [e - BASE for e in ends] == [0 * H, 8 * H]
        assert rates == [Decimal("0.001"), Decimal("0.005")]

    def test_no_cutoff_keeps_all_buckets(self) -> None:
        recs = _recs([(4, "0.002"), (8, "0.003")])
        ends, rates, _marks = normalize_funding_records_to_8h(recs)
        assert [e - BASE for e in ends] == [8 * H]
        assert rates == [Decimal("0.005")]

    def test_marks_take_last_settlement_in_bucket(self) -> None:
        recs = [
            (BASE + 4 * H, Decimal("0.001"), Decimal("101")),
            (BASE + 8 * H, Decimal("0.002"), Decimal("102")),
        ]
        _ends, _rates, marks = normalize_funding_records_to_8h(recs)
        assert marks == [Decimal("102")]
