"""命令列入口，將回測 JSON 轉為 MAE/MFE 報告。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

# 允許以腳本方式直接執行：若未以套件啟動，將父層加入 sys.path 並使用絕對匯入
import sys

if __package__ is None or __package__ == "":
    pkg_root = Path(__file__).resolve().parent
    sys.path.append(str(pkg_root.parent))
    __package__ = pkg_root.name

from mae_tool.models import AnalysisOptions  # type: ignore
from mae_tool.runner import run_mae_mfe_from_json  # type: ignore


def _parse_charts(raw: str | None) -> Tuple[int, ...]:
    if not raw:
        return tuple(range(1, 11))
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    charts = []
    for part in parts:
        try:
            val = int(part)
        except Exception:
            continue
        if 1 <= val <= 10:
            charts.append(val)
    return tuple(charts) if charts else tuple(range(1, 11))


def _build_output_dir(user_path: str | None) -> Path:
    if user_path:
        return Path(user_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("mae_tool_output") / timestamp


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="將回測 JSON 轉為 MAE/MFE 圖表與報告")
    parser.add_argument("--input", "-i", required=True, help="回測 JSON 路徑")
    parser.add_argument("--strategy", "-s", help="策略名稱；未提供則使用 JSON 第一個策略")
    parser.add_argument("--output", "-o", help="輸出資料夾，預設為 mae_tool_output/<timestamp>")
    parser.add_argument("--charts", "-c", help="欲產生的圖表編號，例如 1,3,4；預設 1-10")
    parser.add_argument("--report-title", default="MAE & MFE JSON Report", help="報告標題")
    parser.add_argument("--no-report", action="store_true", help="僅輸出圖表，不產生總表 HTML")
    parser.add_argument("--no-html", action="store_true", help="不輸出互動 HTML（預設輸出）")

    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = _build_output_dir(args.output)
    charts = _parse_charts(args.charts)
    options = AnalysisOptions(
        charts=charts,
        export_png=False,
        export_html=not args.no_html,
        export_report=not args.no_report,
        report_title=args.report_title,
        strategy_name=args.strategy,
    )

    result = run_mae_mfe_from_json(args.input, output_dir, options=options)
    print(f"策略：{result.strategy_name}")
    print(f"輸出路徑：{output_dir}")
    if result.report_path:
        print(f"報告：{result.report_path}")
    if result.warnings:
        print("警示：")
        for msg in result.warnings:
            print(f" - {msg}")
    else:
        print("警示：無")


if __name__ == "__main__":
    main()
