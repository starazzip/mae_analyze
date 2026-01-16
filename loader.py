"""從 JSON 交易紀錄轉換為 MAE/MFE 所需欄位。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REQUIRED_TRADE_COLUMNS: Sequence[str] = (
    "symbol",
    "trade_id",
    "direction",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "return_pct",
    "mfe_pct",
    "mae_pct",
    "mae_abs_pct",
    "mfe_pct_with_costs",
    "mae_pct_with_costs",
    "max_favorable_price",
    "max_adverse_price",
    "bmfe_pct",
    "mdd_pct",
    "pdays",
    "pdays_ratio",
    "holding_bars",
    "holding_seconds",
    "holding_days",
    "status",
    "exit_reason",
    "duration",
)


def _ensure_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案: {path}")
    return path


def _safe_float(value: object) -> Optional[float]:
    try:
        num = float(value)
    except Exception:
        return None
    if not np.isfinite(num):
        return None
    return float(num)


def _price_from_orders(trade: Dict[str, object], *, is_entry: bool) -> Optional[float]:
    """從 orders 取出入口/出口的 safe_price 並加權平均，避免 open_rate/close_rate 為 0。"""

    orders = trade.get("orders")
    if not isinstance(orders, list) or not orders:
        return None

    total_cost = 0.0
    total_amount = 0.0
    for order in orders:
        if bool(order.get("ft_is_entry")) != is_entry:
            continue
        price = _safe_float(order.get("safe_price"))
        amount = _safe_float(order.get("amount"))
        if price is None or amount is None:
            continue
        total_cost += price * amount
        total_amount += amount

    if total_amount <= 0:
        return None
    return total_cost / total_amount


def _compute_pct(numerator: Optional[float], denominator: Optional[float], inverse: bool = False) -> Optional[float]:
    """計算百分比，inverse=True 代表使用 (denominator - numerator)/denominator。"""

    if numerator is None or denominator in (None, 0):
        return None
    if inverse:
        return (denominator - numerator) / denominator * 100.0
    return (numerator - denominator) / denominator * 100.0


def load_trades_json(
    json_path: str | Path,
    strategy_name: Optional[str] = None,
    *,
    warnings: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    """讀取回測 JSON，回傳 trades DataFrame 與實際使用的策略名稱。"""

    warn_list = warnings if warnings is not None else []
    path = _ensure_path(json_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    strategies = payload.get("strategy")
    if not isinstance(strategies, dict) or not strategies:
        raise ValueError("JSON 中缺少 'strategy' 物件或為空。")

    strategy_keys = list(strategies.keys())
    chosen = strategy_name or strategy_keys[0]
    if chosen not in strategies:
        raise ValueError(f"策略 {chosen} 不存在，請使用以下其一：{', '.join(strategy_keys)}")
    if strategy_name is None and len(strategy_keys) > 1:
        warn_list.append(f"檔案含多個策略 {strategy_keys}，未指定將使用第一個：{chosen}")

    trades_raw = strategies[chosen].get("trades")
    if not isinstance(trades_raw, list) or not trades_raw:
        raise ValueError(f"策略 {chosen} 未找到 trades 陣列。")

    rows: List[Dict[str, object]] = []
    missing_mae_fields = False
    missing_mfe_fields = False
    approximated_mdd = False
    fee_missing = False

    for idx, trade in enumerate(trades_raw):
        is_short = bool(trade.get("is_short"))
        entry_price = _price_from_orders(trade, is_entry=True) or _safe_float(trade.get("open_rate"))
        exit_price = _price_from_orders(trade, is_entry=False) or _safe_float(trade.get("close_rate"))
        min_rate = _safe_float(trade.get("min_rate"))
        max_rate = _safe_float(trade.get("max_rate"))
        fee_open = _safe_float(trade.get("fee_open")) or 0.0
        fee_close = _safe_float(trade.get("fee_close")) or 0.0
        if fee_open == 0.0 and fee_close == 0.0:
            fee_missing = True

        status = "open" if trade.get("is_open") else "closed"
        entry_time = pd.to_datetime(trade.get("open_date"), errors="coerce")
        exit_time = pd.to_datetime(trade.get("close_date"), errors="coerce") if status == "closed" else None

        duration_minutes = _safe_float(trade.get("trade_duration"))
        holding_seconds = duration_minutes * 60 if duration_minutes is not None else None
        if holding_seconds is None and entry_time is not None and exit_time is not None:
            holding_seconds = (exit_time - entry_time).total_seconds()
        holding_days = holding_seconds / 86400.0 if holding_seconds is not None else None

        profit_ratio = _safe_float(trade.get("profit_ratio"))
        if profit_ratio is not None:
            return_pct = profit_ratio * 100.0
        elif entry_price is not None and exit_price is not None:
            if is_short:
                entry_proceeds = entry_price * (1.0 - fee_open)
                exit_cost = exit_price * (1.0 + fee_close)
                return_pct = (entry_proceeds - exit_cost) / entry_proceeds * 100.0
            else:
                entry_cost = entry_price * (1.0 + fee_open)
                exit_value = exit_price * (1.0 - fee_close)
                return_pct = (exit_value - entry_cost) / entry_cost * 100.0
        else:
            return_pct = None

        if entry_price is None:
            missing_mae_fields = missing_mfe_fields = True
            mae_pct = mfe_pct = None
        else:
            if is_short:
                mae_pct = _compute_pct(max_rate, entry_price, inverse=True)
                mfe_pct = _compute_pct(min_rate, entry_price, inverse=True)
            else:
                mae_pct = _compute_pct(min_rate, entry_price, inverse=False)
                mfe_pct = _compute_pct(max_rate, entry_price, inverse=False)
            if mae_pct is None:
                missing_mae_fields = True
            if mfe_pct is None:
                missing_mfe_fields = True
        mae_abs_pct = abs(mae_pct) if mae_pct is not None else None

        max_fav_price = min_rate if is_short else max_rate
        max_adv_price = max_rate if is_short else min_rate

        # 手續費版 MAE/MFE
        mae_pct_with_costs = None
        mfe_pct_with_costs = None
        if entry_price is not None:
            if is_short:
                entry_proceeds = entry_price * (1.0 - fee_open)
                if max_adv_price is not None:
                    mae_pct_with_costs = (entry_proceeds - max_adv_price * (1.0 + fee_close)) / entry_proceeds * 100.0
                if max_fav_price is not None:
                    mfe_pct_with_costs = (entry_proceeds - max_fav_price * (1.0 + fee_close)) / entry_proceeds * 100.0
            else:
                entry_cost = entry_price * (1.0 + fee_open)
                if max_adv_price is not None:
                    mae_pct_with_costs = (max_adv_price * (1.0 - fee_close) - entry_cost) / entry_cost * 100.0
                if max_fav_price is not None:
                    mfe_pct_with_costs = (max_fav_price * (1.0 - fee_close) - entry_cost) / entry_cost * 100.0

        # BMFE/MDD 近似
        bmfe_pct = mfe_pct
        mdd_pct = mae_pct
        if mdd_pct is None and mae_pct is not None:
            mdd_pct = mae_pct
            approximated_mdd = True
        if mdd_pct is None and mae_pct is None and (max_rate is None or min_rate is None):
            approximated_mdd = True

        rows.append(
            {
                "symbol": trade.get("pair"),
                "trade_id": trade.get("trade_id") or trade.get("id") or idx,
                "direction": 1 if is_short else 0,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price if status == "closed" else None,
                "return_pct": return_pct,
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
                "mae_abs_pct": mae_abs_pct,
                "mfe_pct_with_costs": mfe_pct_with_costs,
                "mae_pct_with_costs": mae_pct_with_costs,
                "max_favorable_price": max_fav_price,
                "max_adverse_price": max_adv_price,
                "bmfe_pct": bmfe_pct,
                "mdd_pct": mdd_pct,
                "pdays": None,
                "pdays_ratio": None,
                "holding_bars": None,
                "holding_seconds": holding_seconds,
                "holding_days": holding_days,
                "status": status,
                "exit_reason": (trade.get("exit_reason") or "").strip(),
                "duration": duration_minutes,
                "mae_time": None,
                "gmfe_time": None,
                "bmfe_time": None,
            }
        )

    df = pd.DataFrame(rows)
    for col in REQUIRED_TRADE_COLUMNS:
        if col not in df:
            df[col] = None
    df = df.replace([pd.NA, pd.NaT], None)

    # 警示整理
    chart_impacts: Dict[str, str] = {}
    if missing_mae_fields:
        chart_impacts["mae"] = "缺少 min_rate 或 safe_price/open_rate，可能無法產生圖 3/4/5/6/7/10"
    if missing_mfe_fields:
        chart_impacts["mfe"] = "缺少 max_rate 或 safe_price/open_rate，可能無法產生圖 4/5/6/8/9/10"
    if approximated_mdd:
        warn_list.append("mdd_pct 以 MAE 近似，序列時序未提供。")
    if fee_missing:
        warn_list.append("未提供手續費或為 0，已以未含成本計算 MAE/MFE（with_costs 欄位可能為 None）。")
    warn_list.extend(chart_impacts.values())

    return df, chosen


__all__ = ["load_trades_json", "REQUIRED_TRADE_COLUMNS"]
