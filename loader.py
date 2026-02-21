"""從 JSON 交易紀錄轉換為 MAE/MFE 所需欄位。"""

from __future__ import annotations

import json
import re
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
    "entry_price_source",
    "exit_price_source",
    "mae_source",
    "mfe_source",
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
        amount = _safe_float(order.get("filled"))
        if amount is None or amount <= 0:
            amount = _safe_float(order.get("amount"))
        if price is None or amount is None:
            continue
        if amount <= 0:
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


def _normalize_excursion(mae_pct: Optional[float], mfe_pct: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """將 MAE/MFE 正規化到語義範圍：MAE <= 0、MFE >= 0。"""

    norm_mae = None if mae_pct is None else min(float(mae_pct), 0.0)
    norm_mfe = None if mfe_pct is None else max(float(mfe_pct), 0.0)
    return norm_mae, norm_mfe


def _normalize_bmfe(bmfe_pct: Optional[float]) -> Optional[float]:
    """BMFE 為「有利幅度」，應不小於 0。"""

    if bmfe_pct is None:
        return None
    return max(float(bmfe_pct), 0.0)


def _normalize_mdd(mdd_pct: Optional[float]) -> Optional[float]:
    """MDD 為回撤，應不大於 0。"""

    if mdd_pct is None:
        return None
    return min(float(mdd_pct), 0.0)


def _to_utc(value: pd.Timestamp) -> pd.Timestamp:
    """將 Timestamp 正規化為 UTC 時區。"""

    if value.tzinfo is None:
        return value.tz_localize("UTC")
    return value.tz_convert("UTC")


def _compute_mdd_from_ohlcv(window_df: pd.DataFrame, entry_price: float, is_short: bool) -> Optional[float]:
    """由 OHLCV 路徑計算單筆交易 MDD（百分比，負值）。"""

    if window_df.empty or entry_price <= 0:
        return None

    if is_short:
        running_trough = float(entry_price)
        worst_dd = 0.0
        for _, row in window_df.sort_index().iterrows():
            low = _safe_float(row.get("low"))
            high = _safe_float(row.get("high"))
            # 保守估計：同一根 K 內先更新有利極值，再檢查不利極值。
            if low is not None and low > 0:
                running_trough = min(running_trough, float(low))
            if high is None or high <= 0 or running_trough <= 0:
                continue
            dd = (running_trough - float(high)) / running_trough * 100.0
            worst_dd = min(worst_dd, dd)
        return _normalize_mdd(worst_dd)

    running_peak = float(entry_price)
    worst_dd = 0.0
    for _, row in window_df.sort_index().iterrows():
        high = _safe_float(row.get("high"))
        low = _safe_float(row.get("low"))
        # 保守估計：同一根 K 內先更新有利極值，再檢查不利極值。
        if high is not None and high > 0:
            running_peak = max(running_peak, float(high))
        if low is None or low <= 0 or running_peak <= 0:
            continue
        dd = (float(low) - running_peak) / running_peak * 100.0
        worst_dd = min(worst_dd, dd)
    return _normalize_mdd(worst_dd)


def _valid_rate(value: object) -> Optional[float]:
    """僅接受大於 0 的價格，將 0 視為缺值（部分回測輸出會以 0 代表缺漏）。"""

    num = _safe_float(value)
    if num is None or num <= 0:
        return None
    return num


def _to_utc_timestamp(value: object) -> Optional[pd.Timestamp]:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if ts is None or pd.isna(ts):
        return None
    return ts


def _default_data_dir(trading_mode: object) -> Path:
    mode = str(trading_mode or "").strip().lower()
    if mode == "futures":
        return Path("user_data/data/futures")
    return Path("user_data/data/spot")


def _pair_token(pair: object) -> str:
    token = str(pair or "").strip()
    token = token.replace("/", "_").replace(":", "_").replace("-", "_")
    token = token.replace(" ", "")
    return token


def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    raw = str(timeframe or "").strip().lower()
    m = re.fullmatch(r"(\d+)([mhdw])", raw)
    if not m:
        return pd.Timedelta(minutes=5)
    value = int(m.group(1))
    unit = m.group(2)
    if value <= 0:
        return pd.Timedelta(minutes=5)
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    return pd.Timedelta(weeks=value)


class _OHLCVLookup:
    """在 min/max 缺值時，從本地 OHLCV 重建交易區間極值。"""

    def __init__(self, data_dir: Path, timeframe: str, file_format: str = "feather") -> None:
        self.data_dir = data_dir
        self.timeframe = str(timeframe or "5m")
        self.file_format = str(file_format or "feather").strip().lstrip(".")
        self._window_pad = _timeframe_to_timedelta(self.timeframe)
        self._path_cache: Dict[str, Optional[Path]] = {}
        self._df_cache: Dict[str, Optional[pd.DataFrame]] = {}
        self._missing_pairs: set[str] = set()

    @property
    def missing_pairs(self) -> Sequence[str]:
        return sorted(self._missing_pairs)

    def _find_path(self, pair: str) -> Optional[Path]:
        token = _pair_token(pair)
        if not token:
            return None
        if token in self._path_cache:
            return self._path_cache[token]
        if not self.data_dir.exists():
            self._path_cache[token] = None
            self._missing_pairs.add(str(pair))
            return None

        pattern = f"{token}-{self.timeframe}*.{self.file_format}"
        candidates = list(self.data_dir.rglob(pattern))
        if not candidates:
            self._path_cache[token] = None
            self._missing_pairs.add(str(pair))
            return None
        candidates.sort(key=lambda p: (len(p.parts), len(str(p))))
        self._path_cache[token] = candidates[0]
        return candidates[0]

    def _load_pair_df(self, pair: str) -> Optional[pd.DataFrame]:
        token = _pair_token(pair)
        if token in self._df_cache:
            return self._df_cache[token]
        path = self._find_path(pair)
        if path is None:
            self._df_cache[token] = None
            return None
        try:
            suffix = path.suffix.lower()
            if suffix == ".feather":
                raw_df = pd.read_feather(path)
            elif suffix == ".parquet":
                raw_df = pd.read_parquet(path)
            elif suffix == ".csv":
                raw_df = pd.read_csv(path)
            else:
                self._df_cache[token] = None
                self._missing_pairs.add(str(pair))
                return None
        except Exception:
            self._df_cache[token] = None
            self._missing_pairs.add(str(pair))
            return None

        if raw_df.empty or "date" not in raw_df or "low" not in raw_df or "high" not in raw_df:
            self._df_cache[token] = None
            self._missing_pairs.add(str(pair))
            return None
        keep_cols = ["date", "low", "high"]
        if "close" in raw_df.columns:
            keep_cols.append("close")
        df = raw_df[keep_cols].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        if "close" in df.columns:
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        if df.empty:
            self._df_cache[token] = None
            self._missing_pairs.add(str(pair))
            return None
        df = df.set_index("date")
        self._df_cache[token] = df
        return df

    def window_frame(
        self,
        pair: str,
        start_ts: Optional[pd.Timestamp],
        end_ts: Optional[pd.Timestamp],
    ) -> Optional[pd.DataFrame]:
        """取得交易區間對應的 K 線視窗（以 UTC 索引）。"""

        if start_ts is None:
            return None
        df = self._load_pair_df(pair)
        if df is None or df.empty:
            return None

        start = _to_utc(start_ts)
        if end_ts is None:
            end = start
        else:
            end = _to_utc(end_ts)
        if end < start:
            start, end = end, start

        window = df.loc[(df.index >= start) & (df.index <= end)]
        if window.empty:
            window = df.loc[(df.index >= start - self._window_pad) & (df.index <= end + self._window_pad)]
        if window.empty:
            return None

        return window

    def window_extremes(
        self,
        pair: str,
        start_ts: Optional[pd.Timestamp],
        end_ts: Optional[pd.Timestamp],
    ) -> Tuple[Optional[float], Optional[float]]:
        window = self.window_frame(pair=pair, start_ts=start_ts, end_ts=end_ts)
        if window is None or window.empty:
            return None, None

        low_series = pd.to_numeric(window["low"], errors="coerce")
        high_series = pd.to_numeric(window["high"], errors="coerce")
        low_series = low_series[(~low_series.isna()) & (low_series > 0)]
        high_series = high_series[(~high_series.isna()) & (high_series > 0)]
        low_val = float(low_series.min()) if not low_series.empty else None
        high_val = float(high_series.max()) if not high_series.empty else None
        return low_val, high_val


def load_trades_json(
    json_path: str | Path,
    strategy_name: Optional[str] = None,
    *,
    warnings: Optional[List[str]] = None,
    data_dir: str | Path | None = None,
    ohlcv_format: str = "feather",
    rebuild_excursions_from_ohlcv: bool = True,
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

    strategy_payload = strategies[chosen]
    trades_raw = strategy_payload.get("trades")
    if not isinstance(trades_raw, list) or not trades_raw:
        raise ValueError(f"策略 {chosen} 未找到 trades 陣列。")

    ohlcv_lookup: Optional[_OHLCVLookup] = None
    if rebuild_excursions_from_ohlcv:
        trading_mode = strategy_payload.get("trading_mode")
        timeframe = str(strategy_payload.get("timeframe") or "5m")
        base_dir = Path(data_dir) if data_dir else _default_data_dir(trading_mode)
        if base_dir.exists():
            ohlcv_lookup = _OHLCVLookup(base_dir, timeframe=timeframe, file_format=ohlcv_format)
        else:
            warn_list.append(f"OHLCV 資料目錄不存在：{base_dir}，將略過 min/max 重建。")

    rows: List[Dict[str, object]] = []
    missing_mae_fields = False
    missing_mfe_fields = False
    approximated_mdd = False
    fee_missing = False
    rebuilt_min_count = 0
    rebuilt_max_count = 0
    missing_min_count = 0
    missing_max_count = 0

    for idx, trade in enumerate(trades_raw):
        pair = str(trade.get("pair") or "")
        is_short = bool(trade.get("is_short"))
        open_rate = _valid_rate(trade.get("open_rate"))
        close_rate = _valid_rate(trade.get("close_rate"))
        entry_order_price = _price_from_orders(trade, is_entry=True)
        exit_order_price = _price_from_orders(trade, is_entry=False)
        entry_price = entry_order_price if entry_order_price is not None else open_rate
        exit_price = exit_order_price if exit_order_price is not None else close_rate
        entry_price_source = (
            "orders_safe_price"
            if entry_order_price is not None
            else ("trade_open_rate" if open_rate is not None else "missing")
        )
        exit_price_source = (
            "orders_safe_price"
            if exit_order_price is not None
            else ("trade_close_rate" if close_rate is not None else "missing")
        )
        min_rate_json = _valid_rate(trade.get("min_rate"))
        max_rate_json = _valid_rate(trade.get("max_rate"))
        min_rate = min_rate_json
        max_rate = max_rate_json
        min_source = "json" if min_rate_json is not None else "missing"
        max_source = "json" if max_rate_json is not None else "missing"
        fee_open = _safe_float(trade.get("fee_open")) or 0.0
        fee_close = _safe_float(trade.get("fee_close")) or 0.0
        if fee_open == 0.0 and fee_close == 0.0:
            fee_missing = True

        status = "open" if trade.get("is_open") else "closed"
        entry_time = _to_utc_timestamp(trade.get("open_date"))
        exit_time = _to_utc_timestamp(trade.get("close_date")) if status == "closed" else None

        window_df: Optional[pd.DataFrame] = None
        if ohlcv_lookup is not None and entry_time is not None:
            window_df = ohlcv_lookup.window_frame(
                pair=pair,
                start_ts=entry_time,
                end_ts=exit_time if exit_time is not None else entry_time,
            )

        if (min_rate is None or max_rate is None) and window_df is not None and not window_df.empty:
            low_series = pd.to_numeric(window_df["low"], errors="coerce")
            high_series = pd.to_numeric(window_df["high"], errors="coerce")
            low_series = low_series[(~low_series.isna()) & (low_series > 0)]
            high_series = high_series[(~high_series.isna()) & (high_series > 0)]
            low_val = float(low_series.min()) if not low_series.empty else None
            high_val = float(high_series.max()) if not high_series.empty else None
            if min_rate is None and low_val is not None:
                min_rate = low_val
                min_source = "ohlcv_rebuild"
                rebuilt_min_count += 1
            if max_rate is None and high_val is not None:
                max_rate = high_val
                max_source = "ohlcv_rebuild"
                rebuilt_max_count += 1
        if min_rate is None:
            missing_min_count += 1
        if max_rate is None:
            missing_max_count += 1

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

        mae_time = None
        gmfe_time = None
        bmfe_time = None
        mae_source = "missing"
        mfe_source = "missing"
        mae_pct = None
        mfe_pct = None
        bmfe_pct = None
        mdd_pct = None

        if entry_price is None:
            missing_mae_fields = missing_mfe_fields = True
        else:
            path_mae_pct = None
            path_mfe_pct = None
            # 優先使用交易區間 K 線計算：可取得 MAE/GMFE/BMFE 時點與真實 MDD。
            if window_df is not None and not window_df.empty:
                low_series = pd.to_numeric(window_df["low"], errors="coerce")
                high_series = pd.to_numeric(window_df["high"], errors="coerce")
                low_series = low_series[(~low_series.isna()) & (low_series > 0)]
                high_series = high_series[(~high_series.isna()) & (high_series > 0)]

                if is_short:
                    mae_price = float(high_series.max()) if not high_series.empty else None
                    gmfe_price = float(low_series.min()) if not low_series.empty else None
                    mae_time = high_series.idxmax() if not high_series.empty else None
                    gmfe_time = low_series.idxmin() if not low_series.empty else None
                    path_mae = _compute_pct(mae_price, entry_price, inverse=True)
                    path_gmfe = _compute_pct(gmfe_price, entry_price, inverse=True)
                else:
                    mae_price = float(low_series.min()) if not low_series.empty else None
                    gmfe_price = float(high_series.max()) if not high_series.empty else None
                    mae_time = low_series.idxmin() if not low_series.empty else None
                    gmfe_time = high_series.idxmax() if not high_series.empty else None
                    path_mae = _compute_pct(mae_price, entry_price, inverse=False)
                    path_gmfe = _compute_pct(gmfe_price, entry_price, inverse=False)

                path_mae_pct, path_mfe_pct = _normalize_excursion(path_mae, path_gmfe)

                # BMFE：MAE 發生前的最大有利幅度（before MAE）。
                if mae_time is not None:
                    pre_window = window_df.loc[window_df.index < _to_utc(mae_time)]
                    if pre_window.empty:
                        bmfe_pct = 0.0
                        bmfe_time = entry_time
                    else:
                        if is_short:
                            pre_fav = pd.to_numeric(pre_window["low"], errors="coerce")
                            pre_fav = pre_fav[(~pre_fav.isna()) & (pre_fav > 0)]
                            if not pre_fav.empty:
                                bmfe_price = float(pre_fav.min())
                                bmfe_time = pre_fav.idxmin()
                                bmfe_pct = _compute_pct(bmfe_price, entry_price, inverse=True)
                        else:
                            pre_fav = pd.to_numeric(pre_window["high"], errors="coerce")
                            pre_fav = pre_fav[(~pre_fav.isna()) & (pre_fav > 0)]
                            if not pre_fav.empty:
                                bmfe_price = float(pre_fav.max())
                                bmfe_time = pre_fav.idxmax()
                                bmfe_pct = _compute_pct(bmfe_price, entry_price, inverse=False)

                # BMFE 可能與 GMFE 相同：當 GMFE 不晚於 MAE 時允許兩者相等。
                if (
                    bmfe_pct is None
                    and mfe_pct is not None
                    and gmfe_time is not None
                    and mae_time is not None
                    and _to_utc(gmfe_time) <= _to_utc(mae_time)
                ):
                    bmfe_pct = mfe_pct
                    bmfe_time = gmfe_time
                bmfe_pct = _normalize_bmfe(bmfe_pct)

                mdd_pct = _compute_mdd_from_ohlcv(window_df, entry_price=entry_price, is_short=is_short)

            # 若無法由 K 線完整取得，回退到交易 JSON 極值欄位。
            if mae_pct is None:
                if is_short:
                    mae_pct = _compute_pct(max_rate, entry_price, inverse=True)
                    mae_source = max_source
                else:
                    mae_pct = _compute_pct(min_rate, entry_price, inverse=False)
                    mae_source = min_source
            if mfe_pct is None:
                if is_short:
                    mfe_pct = _compute_pct(min_rate, entry_price, inverse=True)
                    mfe_source = min_source
                else:
                    mfe_pct = _compute_pct(max_rate, entry_price, inverse=False)
                    mfe_source = max_source

            if mae_pct is None and path_mae_pct is not None:
                mae_pct = path_mae_pct
                mae_source = "ohlcv_window"
            if mfe_pct is None and path_mfe_pct is not None:
                mfe_pct = path_mfe_pct
                mfe_source = "ohlcv_window"

            mae_pct, mfe_pct = _normalize_excursion(mae_pct, mfe_pct)
            bmfe_pct = _normalize_bmfe(bmfe_pct)
            if bmfe_pct is None and mfe_pct is not None:
                # fallback：缺時序資料時，至少不再把 BMFE 視為負值。
                bmfe_pct = _normalize_bmfe(mfe_pct)

            if mdd_pct is None and mae_pct is not None:
                # fallback：沒有完整路徑時，才以 MAE 近似。
                mdd_pct = _normalize_mdd(mae_pct)
                approximated_mdd = True

            if bmfe_pct is not None and mfe_pct is not None:
                bmfe_pct = min(float(bmfe_pct), float(mfe_pct))

            if mae_pct is None:
                missing_mae_fields = True
                mae_source = "missing"
            if mfe_pct is None:
                missing_mfe_fields = True
                mfe_source = "missing"
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

        if mdd_pct is None and mae_pct is None and (max_rate is None or min_rate is None):
            approximated_mdd = True

        rows.append(
            {
                "symbol": pair,
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
                "entry_price_source": entry_price_source,
                "exit_price_source": exit_price_source,
                "mae_source": mae_source if entry_price is not None else "missing",
                "mfe_source": mfe_source if entry_price is not None else "missing",
                "mae_time": mae_time,
                "gmfe_time": gmfe_time,
                "bmfe_time": bmfe_time,
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
        warn_list.append("mdd_pct 無法由 OHLCV 路徑計算，已以 MAE 近似。")
    if fee_missing:
        warn_list.append("未提供手續費或為 0，已以未含成本計算 MAE/MFE（with_costs 欄位可能為 None）。")
    if rebuilt_min_count or rebuilt_max_count:
        warn_list.append(
            f"已由 OHLCV 重建極值：min_rate={rebuilt_min_count}、max_rate={rebuilt_max_count}。"
        )
    if missing_min_count or missing_max_count:
        warn_list.append(
            f"仍有極值缺漏：min_rate={missing_min_count}、max_rate={missing_max_count}。"
        )
    if ohlcv_lookup is not None and ohlcv_lookup.missing_pairs:
        preview = ", ".join(ohlcv_lookup.missing_pairs[:8])
        remain = len(ohlcv_lookup.missing_pairs) - 8
        tail = f" 等 {len(ohlcv_lookup.missing_pairs)} 個" if remain > 0 else ""
        warn_list.append(f"部分交易對找不到 OHLCV：{preview}{tail}")
    warn_list.extend(chart_impacts.values())

    return df, chosen


__all__ = ["load_trades_json", "REQUIRED_TRADE_COLUMNS"]
