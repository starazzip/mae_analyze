from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mae_analyze.loader import load_trades_json

STRATEGY = "ZzHkrsifFreqBaseQuantileRSI_Standalone"


def _write_backtest_json(
    root: Path,
    trades: list[dict],
    *,
    timeframe: str = "5m",
    trading_mode: str = "spot",
) -> Path:
    payload = {
        "strategy": {
            STRATEGY: {
                "trades": trades,
                "timeframe": timeframe,
                "trading_mode": trading_mode,
            }
        }
    }
    path = root / "backtest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_ohlcv_feather(data_dir: Path, pair: str, timeframe: str, rows: list[dict]) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    token = pair.replace("/", "_").replace(":", "_").replace("-", "_").replace(" ", "")
    path = data_dir / f"{token}-{timeframe}.feather"
    pd.DataFrame(rows).to_feather(path)
    return path


def test_load_trades_json_prefers_orders_safe_price_weighted_by_filled(tmp_path: Path) -> None:
    trades = [
        {
            "pair": "BTC/USDT",
            "trade_id": 1,
            "is_short": False,
            "is_open": False,
            "open_date": "2023-01-01 00:00:00+00:00",
            "close_date": "2023-01-01 01:00:00+00:00",
            "open_rate": 0.0,
            "close_rate": 0.0,
            "min_rate": 95.0,
            "max_rate": 140.0,
            "fee_open": 0.0,
            "fee_close": 0.0,
            "orders": [
                {"ft_is_entry": True, "safe_price": 100.0, "filled": 1.0, "amount": 10.0},
                {"ft_is_entry": True, "safe_price": 110.0, "filled": 3.0, "amount": 1.0},
                {"ft_is_entry": False, "safe_price": 120.0, "filled": 1.0, "amount": 5.0},
                {"ft_is_entry": False, "safe_price": 130.0, "filled": 1.0, "amount": 5.0},
            ],
            "trade_duration": 60,
            "exit_reason": "exit_signal",
        }
    ]
    json_path = _write_backtest_json(tmp_path, trades)
    warnings: list[str] = []
    df, strategy = load_trades_json(json_path, STRATEGY, warnings=warnings)

    assert strategy == STRATEGY
    assert len(df) == 1
    row = df.iloc[0]

    # entry/exit 應優先採用 orders.safe_price，且以 filled 權重
    assert row["entry_price"] == 107.5
    assert row["exit_price"] == 125.0
    assert row["entry_price_source"] == "orders_safe_price"
    assert row["exit_price_source"] == "orders_safe_price"

    assert row["mae_source"] == "json"
    assert row["mfe_source"] == "json"
    assert row["mae_pct"] == pytest_approx((95.0 - 107.5) / 107.5 * 100.0)
    assert row["mfe_pct"] == pytest_approx((140.0 - 107.5) / 107.5 * 100.0)


def test_load_trades_json_rebuilds_min_rate_from_ohlcv(tmp_path: Path) -> None:
    data_dir = tmp_path / "spot_data"
    _write_ohlcv_feather(
        data_dir,
        "TEST/USDT",
        "5m",
        [
            {"date": "2023-01-01 00:00:00+00:00", "open": 10.0, "high": 10.8, "low": 9.7, "close": 10.2},
            {"date": "2023-01-01 00:05:00+00:00", "open": 10.2, "high": 11.3, "low": 8.0, "close": 10.9},
            {"date": "2023-01-01 00:10:00+00:00", "open": 10.9, "high": 11.5, "low": 10.0, "close": 11.1},
        ],
    )

    trades = [
        {
            "pair": "TEST/USDT",
            "trade_id": 2,
            "is_short": False,
            "is_open": False,
            "open_date": "2023-01-01 00:00:00+00:00",
            "close_date": "2023-01-01 00:10:00+00:00",
            "open_rate": 0.0,
            "close_rate": 11.0,
            "min_rate": 0.0,  # 這裡故意留 0，應由 OHLCV 補齊
            "max_rate": 12.0,
            "fee_open": 0.0,
            "fee_close": 0.0,
            "orders": [
                {"ft_is_entry": True, "safe_price": 10.0, "filled": 1.0},
                {"ft_is_entry": False, "safe_price": 11.0, "filled": 1.0},
            ],
            "trade_duration": 10,
            "exit_reason": "exit_signal",
        }
    ]
    json_path = _write_backtest_json(tmp_path, trades)
    warnings: list[str] = []
    df, _ = load_trades_json(
        json_path,
        STRATEGY,
        warnings=warnings,
        data_dir=data_dir,
        ohlcv_format="feather",
        rebuild_excursions_from_ohlcv=True,
    )

    row = df.iloc[0]
    assert row["mae_source"] == "ohlcv_rebuild"
    assert row["mfe_source"] == "json"
    assert row["mae_pct"] == pytest_approx(-20.0)
    assert any("重建極值" in msg for msg in warnings)


def test_load_trades_json_marks_missing_when_no_ohlcv(tmp_path: Path) -> None:
    trades = [
        {
            "pair": "MISS/USDT",
            "trade_id": 3,
            "is_short": False,
            "is_open": False,
            "open_date": "2023-01-01 00:00:00+00:00",
            "close_date": "2023-01-01 00:10:00+00:00",
            "open_rate": 0.0,
            "close_rate": 11.0,
            "min_rate": 0.0,
            "max_rate": 12.0,
            "fee_open": 0.0,
            "fee_close": 0.0,
            "orders": [
                {"ft_is_entry": True, "safe_price": 10.0, "filled": 1.0},
                {"ft_is_entry": False, "safe_price": 11.0, "filled": 1.0},
            ],
            "trade_duration": 10,
            "exit_reason": "exit_signal",
        }
    ]
    json_path = _write_backtest_json(tmp_path, trades)
    warnings: list[str] = []
    df, _ = load_trades_json(
        json_path,
        STRATEGY,
        warnings=warnings,
        data_dir=tmp_path / "no_data",
        ohlcv_format="feather",
        rebuild_excursions_from_ohlcv=True,
    )

    row = df.iloc[0]
    assert row["mae_source"] == "missing"
    assert row["mae_pct"] is None
    assert any("缺少 min_rate" in msg for msg in warnings)


def test_load_trades_json_bmfe_is_before_mae_not_gmfe(tmp_path: Path) -> None:
    data_dir = tmp_path / "spot_data"
    _write_ohlcv_feather(
        data_dir,
        "BMFE/USDT",
        "5m",
        [
            {"date": "2023-01-01 00:00:00+00:00", "open": 100.0, "high": 120.0, "low": 100.0, "close": 120.0},
            {"date": "2023-01-01 00:05:00+00:00", "open": 120.0, "high": 115.0, "low": 90.0, "close": 90.0},
            {"date": "2023-01-01 00:10:00+00:00", "open": 90.0, "high": 130.0, "low": 95.0, "close": 125.0},
        ],
    )
    trades = [
        {
            "pair": "BMFE/USDT",
            "trade_id": 4,
            "is_short": False,
            "is_open": False,
            "open_date": "2023-01-01 00:00:00+00:00",
            "close_date": "2023-01-01 00:10:00+00:00",
            "open_rate": 100.0,
            "close_rate": 125.0,
            "min_rate": 90.0,
            "max_rate": 130.0,
            "fee_open": 0.0,
            "fee_close": 0.0,
            "orders": [],
            "trade_duration": 10,
            "exit_reason": "exit_signal",
        }
    ]
    json_path = _write_backtest_json(tmp_path, trades)
    df, _ = load_trades_json(
        json_path,
        STRATEGY,
        data_dir=data_dir,
        ohlcv_format="feather",
        rebuild_excursions_from_ohlcv=True,
    )

    row = df.iloc[0]
    assert row["mfe_pct"] == pytest_approx(30.0)
    assert row["bmfe_pct"] == pytest_approx(20.0)
    assert row["bmfe_pct"] < row["mfe_pct"]


def test_load_trades_json_bmfe_can_equal_gmfe(tmp_path: Path) -> None:
    data_dir = tmp_path / "spot_data"
    _write_ohlcv_feather(
        data_dir,
        "BMFE_EQ/USDT",
        "5m",
        [
            {"date": "2023-01-01 00:00:00+00:00", "open": 100.0, "high": 130.0, "low": 110.0, "close": 125.0},
            {"date": "2023-01-01 00:05:00+00:00", "open": 125.0, "high": 120.0, "low": 90.0, "close": 95.0},
        ],
    )
    trades = [
        {
            "pair": "BMFE_EQ/USDT",
            "trade_id": 41,
            "is_short": False,
            "is_open": False,
            "open_date": "2023-01-01 00:00:00+00:00",
            "close_date": "2023-01-01 00:05:00+00:00",
            "open_rate": 100.0,
            "close_rate": 95.0,
            "min_rate": 90.0,
            "max_rate": 130.0,
            "fee_open": 0.0,
            "fee_close": 0.0,
            "orders": [],
            "trade_duration": 5,
            "exit_reason": "stop_loss",
        }
    ]
    json_path = _write_backtest_json(tmp_path, trades)
    df, _ = load_trades_json(
        json_path,
        STRATEGY,
        data_dir=data_dir,
        ohlcv_format="feather",
        rebuild_excursions_from_ohlcv=True,
    )

    row = df.iloc[0]
    assert row["mfe_pct"] == pytest_approx(30.0)
    assert row["bmfe_pct"] == pytest_approx(30.0)
    assert row["bmfe_pct"] == row["mfe_pct"]


def test_load_trades_json_mdd_uses_path_drawdown_not_mae(tmp_path: Path) -> None:
    data_dir = tmp_path / "spot_data"
    _write_ohlcv_feather(
        data_dir,
        "MDD/USDT",
        "5m",
        [
            {"date": "2023-01-01 00:00:00+00:00", "open": 100.0, "high": 120.0, "low": 100.0, "close": 120.0},
            {"date": "2023-01-01 00:05:00+00:00", "open": 120.0, "high": 115.0, "low": 90.0, "close": 90.0},
            {"date": "2023-01-01 00:10:00+00:00", "open": 90.0, "high": 130.0, "low": 95.0, "close": 125.0},
        ],
    )
    trades = [
        {
            "pair": "MDD/USDT",
            "trade_id": 5,
            "is_short": False,
            "is_open": False,
            "open_date": "2023-01-01 00:00:00+00:00",
            "close_date": "2023-01-01 00:10:00+00:00",
            "open_rate": 100.0,
            "close_rate": 125.0,
            "min_rate": 90.0,
            "max_rate": 130.0,
            "fee_open": 0.0,
            "fee_close": 0.0,
            "orders": [],
            "trade_duration": 10,
            "exit_reason": "exit_signal",
        }
    ]
    json_path = _write_backtest_json(tmp_path, trades)
    df, _ = load_trades_json(
        json_path,
        STRATEGY,
        data_dir=data_dir,
        ohlcv_format="feather",
        rebuild_excursions_from_ohlcv=True,
    )

    row = df.iloc[0]
    assert row["mae_pct"] == pytest_approx(-10.0)
    assert row["mdd_pct"] == pytest_approx(-26.9230769231)
    assert row["mdd_pct"] < row["mae_pct"]


def pytest_approx(value: float):
    # 避免在測試內直接依賴全域 pytest import，降低誤報錯訊。
    import pytest

    return pytest.approx(value, rel=1e-9, abs=1e-12)
