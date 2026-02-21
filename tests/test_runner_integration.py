from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mae_analyze.models import AnalysisOptions
from mae_analyze.runner import run_mae_mfe_from_json

STRATEGY = "ZzHkrsifFreqBaseQuantileRSI_Standalone"


def _write_backtest_json(root: Path, trades: list[dict]) -> Path:
    payload = {
        "strategy": {
            STRATEGY: {
                "trades": trades,
                "timeframe": "5m",
                "trading_mode": "spot",
            }
        }
    }
    path = root / "backtest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_ohlcv(root: Path, pair: str, rows: list[dict]) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    token = pair.replace("/", "_")
    path = data_dir / f"{token}-5m.feather"
    pd.DataFrame(rows).to_feather(path)
    return data_dir


def test_run_mae_mfe_from_json_end_to_end_with_rebuild(tmp_path: Path) -> None:
    pair = "TEST/USDT"
    data_dir = _write_ohlcv(
        tmp_path,
        pair,
        [
            {"date": "2023-02-01 00:00:00+00:00", "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5},
            {"date": "2023-02-01 00:05:00+00:00", "open": 10.5, "high": 12.0, "low": 8.7, "close": 11.2},
            {"date": "2023-02-01 00:10:00+00:00", "open": 11.2, "high": 11.8, "low": 10.8, "close": 11.0},
        ],
    )
    trades = [
        {
            "pair": pair,
            "trade_id": 1,
            "is_short": False,
            "is_open": False,
            "open_date": "2023-02-01 00:00:00+00:00",
            "close_date": "2023-02-01 00:10:00+00:00",
            "open_rate": 0.0,
            "close_rate": 11.0,
            "min_rate": 0.0,
            "max_rate": 12.0,
            "profit_ratio": 0.1,
            "orders": [
                {"ft_is_entry": True, "safe_price": 10.0, "filled": 1.0},
                {"ft_is_entry": False, "safe_price": 11.0, "filled": 1.0},
            ],
            "trade_duration": 10,
            "exit_reason": "exit_signal",
        },
        {
            "pair": pair,
            "trade_id": 2,
            "is_short": False,
            "is_open": False,
            "open_date": "2023-02-02 00:00:00+00:00",
            "close_date": "2023-02-02 00:10:00+00:00",
            "open_rate": 12.0,
            "close_rate": 11.5,
            "min_rate": 11.0,
            "max_rate": 12.5,
            "profit_ratio": -0.02,
            "orders": [],
            "trade_duration": 10,
            "exit_reason": "stop_loss",
        },
    ]
    json_path = _write_backtest_json(tmp_path, trades)
    output_dir = tmp_path / "out"
    options = AnalysisOptions(
        charts=(1, 3),
        export_html=True,
        export_report=True,
        strategy_name=STRATEGY,
        data_dir=str(data_dir),
        ohlcv_format="feather",
        rebuild_excursions_from_ohlcv=True,
        report_title="test report",
    )

    result = run_mae_mfe_from_json(json_path, output_dir, options=options)

    assert result.report_path is not None
    assert result.report_path.exists()
    assert (output_dir / "chart01_return_distribution.html").exists()
    assert (output_dir / "chart03_mae_vs_return.html").exists()
    assert result.summary["total_trades"] == 2
    assert result.summary["closed_trades"] == 2
    assert any("重建極值" in msg for msg in result.warnings)
