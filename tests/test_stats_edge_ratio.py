from __future__ import annotations

import pytest
import pandas as pd

from mae_analyze.stats import _build_bin_edges, _compute_edge_ratio_records


def _make_trades_df(holding_days: list[float], mae_abs: list[float], mfe: list[float]) -> pd.DataFrame:
    n = len(holding_days)
    return pd.DataFrame(
        {
            "status": ["closed"] * n,
            "holding_days": holding_days,
            "mae_abs_pct": mae_abs,
            "mfe_pct": mfe,
        }
    )


def test_edge_ratio_uses_non_cumulative_bins() -> None:
    df = _make_trades_df(
        holding_days=[0.10, 0.20, 0.60, 1.10],
        mae_abs=[1.0, 2.0, 3.0, 4.0],
        mfe=[2.0, 4.0, 6.0, 8.0],
    )

    records = _compute_edge_ratio_records(df, time_scales=[0.25, 0.5, 1.0])
    assert records, "應產生至少一筆 edge ratio 統計"

    sample_sum = sum(int(r["sample_size"]) for r in records)
    assert sample_sum == 4, "非累積分桶下，樣本總數應與原始交易數一致"
    assert all("-" in str(r["time_scale"]) for r in records), "時間標籤應為區間格式（A-B）"
    assert all("<=" not in str(r["time_scale"]) for r in records), "不應再使用累積格式（<=）"

    # 本組資料每筆 mfe=2*mae，分桶平均後 edge ratio 仍應接近 2
    for rec in records:
        assert rec["edge_ratio"] == pytest.approx(2.0, rel=1e-9, abs=1e-12)


def test_edge_ratio_bin_boundaries_are_monotonic() -> None:
    df = _make_trades_df(
        holding_days=[0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3],
        mae_abs=[1, 1, 1, 1, 1, 1, 1],
        mfe=[1, 1, 1, 1, 1, 1, 1],
    )
    records = _compute_edge_ratio_records(df, time_scales=[0.25, 0.5, 1.0])
    assert records

    starts = [float(r["bucket_start_days"]) for r in records]
    ends = [float(r["bucket_end_days"]) for r in records]
    assert starts == sorted(starts), "桶起點應遞增"
    assert ends == sorted(ends), "桶終點應遞增"
    assert all(e > s for s, e in zip(starts, ends)), "每個桶的終點必須大於起點"


def test_build_bin_edges_uses_finer_bins_near_mean_and_coarser_tail() -> None:
    # 模擬長尾持有時間：平均約 2D，但最大值 120D
    edges = _build_bin_edges(
        mean_holding_days=2.0,
        max_holding_days=120.0,
        step_candidates=[1.0 / 24.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 21.0, 30.0],
    )
    assert len(edges) >= 3
    widths = [b - a for a, b in zip(edges[:-1], edges[1:])]
    assert widths[0] <= 1.0, "前段桶寬應保持細緻"
    assert widths[-1] >= widths[0], "尾段桶寬應不小於前段，才有前細後粗效果"
    assert len(widths) <= 24, "總桶數應受上限保護，避免圖表過度擁擠"
