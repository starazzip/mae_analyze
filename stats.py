"""MAE/MFE 統計計算（精簡版，供 JSON 載入流程使用）。"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

EDGE_RATIO_TIME_BUCKETS_DAYS: Tuple[float, ...] = (
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    14.0,
    21.0,
    30.0,
    45.0,
    60.0,
    90.0,
    120.0,
    180.0,
)


def _format_time_scale_label(days: float) -> str:
    """將時間粒度（天）轉為標籤。"""

    if not math.isfinite(days):
        return "N/A"
    if days < 1.0:
        hours = max(1, int(round(days * 24)))
        return f"<= {hours}H"
    if float(days).is_integer():
        return f"<= {int(days)}D"
    return f"<= {days:.1f}D"


def _compute_edge_ratio_records(trades_df: pd.DataFrame, time_scales: Sequence[float]) -> List[Dict[str, Any]]:
    """計算 Edge Ratio 指標（GMFE/MAE）。"""

    if trades_df.empty or not time_scales:
        return []

    if "status" in trades_df:
        closed_df = trades_df[trades_df["status"] == "closed"]
        if closed_df.empty:
            closed_df = trades_df
    else:
        closed_df = trades_df

    working_df = closed_df.copy()
    holding_days_series = working_df.get("holding_days")
    holding_seconds_series = working_df.get("holding_seconds")
    if holding_days_series is None:
        if holding_seconds_series is not None:
            holding_days_series = holding_seconds_series.astype(float) / 86400.0
        else:
            holding_days_series = pd.Series(np.nan, index=working_df.index, dtype=float)
    else:
        holding_days_series = holding_days_series.astype(float)
        if holding_seconds_series is not None:
            holding_days_series = holding_days_series.fillna(holding_seconds_series.astype(float) / 86400.0)
    working_df["holding_days"] = holding_days_series
    if "mae_abs_pct" in working_df:
        working_df["mae_abs_pct"] = working_df["mae_abs_pct"].astype(float)
    else:
        working_df["mae_abs_pct"] = np.nan
    if "mfe_pct" in working_df:
        working_df["mfe_pct"] = working_df["mfe_pct"].astype(float)
    else:
        working_df["mfe_pct"] = np.nan
    working_df = working_df.replace([np.inf, -np.inf], np.nan)
    working_df = working_df.dropna(subset=["holding_days"])

    records: List[Dict[str, Any]] = []
    for days in time_scales:
        subset = working_df[working_df["holding_days"] <= days]
        if subset.empty:
            continue
        mae_series = subset["mae_abs_pct"].dropna()
        gmfe_series = subset["mfe_pct"].dropna()
        if mae_series.empty and gmfe_series.empty:
            continue
        mean_mae = float(mae_series.mean()) if not mae_series.empty else None
        mean_gmfe = float(gmfe_series.mean()) if not gmfe_series.empty else None
        edge_ratio = None
        if mean_mae not in (None, 0.0) and mean_gmfe is not None:
            edge_ratio = float(mean_gmfe / mean_mae)
        records.append(
            {
                "time_scale": _format_time_scale_label(days),
                "mean_mae_abs_pct": mean_mae,
                "mean_gmfe_pct": mean_gmfe,
                "edge_ratio": edge_ratio,
                "sample_size": int(subset.shape[0]),
            }
        )
    return records


__all__ = ["EDGE_RATIO_TIME_BUCKETS_DAYS", "_compute_edge_ratio_records", "_format_time_scale_label"]
