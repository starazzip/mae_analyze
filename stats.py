"""MAE/MFE 統計計算（精簡版，供 JSON 載入流程使用）。"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# 候選桶寬（天）。會依平均持有時間自動挑選，並生成「非累積」區間桶。
EDGE_RATIO_STEP_CANDIDATES_DAYS: Tuple[float, ...] = (
    1.0 / 24.0,  # 1H
    2.0 / 24.0,  # 2H
    3.0 / 24.0,  # 3H
    4.0 / 24.0,  # 4H
    6.0 / 24.0,  # 6H
    8.0 / 24.0,  # 8H
    12.0 / 24.0,  # 12H
    1.0,
    1.5,
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

# 向後相容：舊名稱保留（runner 仍沿用）。
EDGE_RATIO_TIME_BUCKETS_DAYS = EDGE_RATIO_STEP_CANDIDATES_DAYS


def _format_time_scale_label(start_days: float, end_days: float) -> str:
    """將非累積時間區間（天）轉為標籤。"""

    if not (math.isfinite(start_days) and math.isfinite(end_days)):
        return "N/A"
    if end_days <= start_days:
        return "N/A"

    def _fmt(days: float) -> str:
        if days < 1.0:
            hours = days * 24.0
            if abs(hours - round(hours)) < 1e-9:
                return f"{int(round(hours))}H"
            return f"{hours:.1f}H"
        if abs(days - round(days)) < 1e-9:
            return f"{int(round(days))}D"
        return f"{days:.1f}D"

    left = _fmt(start_days)
    right = _fmt(end_days)
    return f"{left}-{right}"


def _normalize_step_candidates(step_candidates: Sequence[float]) -> List[float]:
    values = [float(x) for x in step_candidates if isinstance(x, (int, float)) and math.isfinite(float(x))]
    values = sorted({x for x in values if x > 0})
    if values:
        return values
    return list(EDGE_RATIO_STEP_CANDIDATES_DAYS)


def _select_step_for_range(
    range_days: float,
    target_bins: int,
    step_candidates: Sequence[float],
    *,
    min_step: float = 0.0,
) -> float:
    candidates = [x for x in _normalize_step_candidates(step_candidates) if x >= max(0.0, float(min_step))]
    if not candidates:
        candidates = _normalize_step_candidates(step_candidates)
    if not candidates:
        return 1.0 / 24.0
    if not math.isfinite(range_days) or range_days <= 0:
        return candidates[0]
    bins_target = max(1, int(target_bins))
    raw_target = range_days / float(bins_target)

    def _score(step: float) -> Tuple[float, float]:
        bins = max(1, int(math.ceil(range_days / step)))
        return abs(bins - bins_target), abs(math.log(step) - math.log(max(raw_target, 1e-9)))

    return min(candidates, key=_score)


def _build_uniform_bin_edges(max_holding_days: float, step_days: float) -> List[float]:
    if not math.isfinite(max_holding_days) or max_holding_days <= 0:
        return [0.0, max(step_days, 1.0 / 24.0)]
    upper = step_days * math.ceil(max_holding_days / step_days)
    count = max(1, int(round(upper / step_days)))
    edges = [i * step_days for i in range(count + 1)]
    if len(edges) < 2:
        edges = [0.0, step_days]
    if edges[-1] > max_holding_days:
        edges[-1] = float(max_holding_days)
    if edges[-1] < max_holding_days:
        edges.append(float(max_holding_days))
    return edges


def _select_bucket_step(
    mean_holding_days: float,
    max_holding_days: float,
    step_candidates: Sequence[float],
    *,
    min_bins: int = 10,
    max_bins: int = 24,
    mean_bins_target: float = 4.0,
    max_bins_target: float = 18.0,
) -> float:
    candidates = _normalize_step_candidates(step_candidates)
    if not math.isfinite(mean_holding_days) or mean_holding_days <= 0:
        return candidates[min(6, len(candidates) - 1)]
    if not math.isfinite(max_holding_days) or max_holding_days <= 0:
        return candidates[0]

    # 以平均持有時間控制「前段解析度」，以最大持有時間限制「總桶數」。
    # 1) 平均持有時間附近希望至少可分成 ~4 個桶。
    target_from_mean = mean_holding_days / max(mean_bins_target, 1.0)
    # 2) 全域最多約 ~18 個桶，避免尾端過碎。
    target_from_max = max_holding_days / max(max_bins_target, 1.0)
    # 取較大者：避免過多小樣本桶。
    target = max(target_from_mean, target_from_max, candidates[0])

    idx = min(range(len(candidates)), key=lambda i: abs(math.log(candidates[i]) - math.log(target)))
    step = candidates[idx]

    def _bin_count(value: float) -> int:
        return max(1, int(math.ceil(max_holding_days / value)))

    bins = _bin_count(step)
    while bins > max_bins and idx < len(candidates) - 1:
        idx += 1
        step = candidates[idx]
        bins = _bin_count(step)
    while bins < min_bins and idx > 0:
        idx -= 1
        step = candidates[idx]
        bins = _bin_count(step)
    return step


def _build_bin_edges(
    mean_holding_days: float,
    max_holding_days: float,
    step_candidates: Sequence[float],
    *,
    min_bins: int = 10,
    max_bins: int = 24,
) -> List[float]:
    """以平均/最大持有時間建立前段細、後段粗的時間分桶。"""

    candidates = _normalize_step_candidates(step_candidates)
    if not candidates:
        candidates = list(EDGE_RATIO_STEP_CANDIDATES_DAYS)
    if not math.isfinite(max_holding_days) or max_holding_days <= 0:
        return [0.0, candidates[0]]
    if not math.isfinite(mean_holding_days) or mean_holding_days <= 0:
        step_days = _select_step_for_range(max_holding_days, max_bins, candidates)
        return _build_uniform_bin_edges(max_holding_days, step_days)

    spread = max_holding_days / max(mean_holding_days, candidates[0])
    desired_bins = int(round(np.clip(8.0 + math.log2(spread + 1.0) * 4.0, min_bins, max_bins)))

    # 短持有區（平均持有時間附近）優先保留解析度。
    early_end_target = min(max_holding_days, max(mean_holding_days * 3.0, candidates[0] * 4.0))
    if early_end_target >= max_holding_days * 0.95:
        step_days = _select_step_for_range(max_holding_days, desired_bins, candidates)
        return _build_uniform_bin_edges(max_holding_days, step_days)

    early_bins_target = max(4, min(desired_bins - 3, int(round(desired_bins * 0.60))))
    early_step = _select_step_for_range(early_end_target, early_bins_target, candidates)
    edges = _build_uniform_bin_edges(early_end_target, early_step)
    early_end = float(edges[-1])
    if early_end >= max_holding_days:
        edges[-1] = float(max_holding_days)
        return edges

    # 長尾區用較粗桶寬，避免單桶樣本過少與圖表過擁擠。
    remaining = max_holding_days - early_end
    used_bins = len(edges) - 1
    remaining_bins_budget = max(3, desired_bins - used_bins)
    tail_step = _select_step_for_range(remaining, remaining_bins_budget, candidates, min_step=early_step)

    current = early_end
    while current + tail_step < max_holding_days - 1e-12 and (len(edges) - 1) < max_bins:
        current += tail_step
        edges.append(float(round(current, 10)))
    if edges[-1] < max_holding_days:
        if (len(edges) - 1) >= max_bins:
            edges[-1] = float(max_holding_days)
        else:
            edges.append(float(max_holding_days))

    # 保留前段細緻度：若桶數超標，優先合併尾段分界。
    while (len(edges) - 1) > max_bins and len(edges) > 2:
        edges.pop(-2)
    return edges


def _compute_edge_ratio_records(trades_df: pd.DataFrame, time_scales: Sequence[float]) -> List[Dict[str, Any]]:
    """計算 Edge Ratio 指標（GMFE/MAE，非累積區間桶）。"""

    if trades_df.empty:
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
    if working_df.empty:
        return []
    working_df = working_df[working_df["holding_days"] >= 0]
    if working_df.empty:
        return []

    mean_holding_days = float(working_df["holding_days"].mean())
    max_holding_days = float(working_df["holding_days"].max())
    edges = _build_bin_edges(mean_holding_days, max_holding_days, time_scales)
    min_samples_per_bin = max(1, int(math.ceil(len(working_df) * 0.01)))

    records: List[Dict[str, Any]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if left <= 0:
            subset = working_df[(working_df["holding_days"] >= left) & (working_df["holding_days"] <= right)]
        else:
            subset = working_df[(working_df["holding_days"] > left) & (working_df["holding_days"] <= right)]
        if subset.empty or len(subset) < min_samples_per_bin:
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
                "time_scale": _format_time_scale_label(left, right),
                "bucket_start_days": float(left),
                "bucket_end_days": float(right),
                "bucket_mid_days": float((left + right) / 2.0),
                "mean_mae_abs_pct": mean_mae,
                "mean_gmfe_pct": mean_gmfe,
                "edge_ratio": edge_ratio,
                "sample_size": int(subset.shape[0]),
            }
        )
    return records


__all__ = [
    "EDGE_RATIO_STEP_CANDIDATES_DAYS",
    "EDGE_RATIO_TIME_BUCKETS_DAYS",
    "_compute_edge_ratio_records",
    "_format_time_scale_label",
]
