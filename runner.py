"""MAE/MFE JSON 分析主流程。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from .charts import CHART_BUILDERS, ChartContext
from .loader import load_trades_json
from .models import AnalysisOptions, AnalysisResult, PlotArtifact, ProgressCallback
from .stats import EDGE_RATIO_TIME_BUCKETS_DAYS, _compute_edge_ratio_records

CHART_REQUIRED_COLUMNS: Dict[int, Sequence[str]] = {
    1: ("return_pct",),
    2: ("mae_abs_pct", "mfe_pct", "holding_days"),
    3: ("return_pct", "mae_abs_pct"),
    4: ("return_pct", "mae_abs_pct", "mfe_pct"),
    5: ("return_pct", "mae_abs_pct", "bmfe_pct"),
    6: ("return_pct", "mdd_pct", "mfe_pct"),
    7: ("return_pct", "mae_abs_pct"),
    8: ("return_pct", "bmfe_pct"),
    9: ("return_pct", "mfe_pct"),
    10: ("return_pct",),
}


@dataclass
class _AnalyzerContext:
    trades_df: pd.DataFrame
    edge_df: pd.DataFrame
    output_dir: Path
    options: AnalysisOptions
    strategy_name: str
    warnings: List[str] = field(default_factory=list)


class MAEMFEJsonAnalyzer:
    """封裝 MAE/MFE JSON 分析生命週期，含資料載入、圖表輸出與報告生成。"""

    def __init__(
        self,
        json_path: str | Path,
        output_dir: str | Path,
        options: AnalysisOptions | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.json_path = Path(json_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.options = options or AnalysisOptions()
        self._progress_callback = progress_callback

    def run(self) -> AnalysisResult:
        self._emit_progress("載入 JSON", 0.05)
        warnings: List[str] = []
        trades_df, strategy_used = load_trades_json(
            self.json_path, strategy_name=self.options.strategy_name, warnings=warnings
        )
        edge_records = _compute_edge_ratio_records(trades_df, EDGE_RATIO_TIME_BUCKETS_DAYS)
        edge_df = pd.DataFrame(edge_records)

        ctx = _AnalyzerContext(
            trades_df=trades_df,
            edge_df=edge_df,
            output_dir=self.output_dir,
            options=self.options,
            strategy_name=strategy_used,
            warnings=warnings,
        )
        self._append_chart_requirement_warnings(ctx)

        self._emit_progress("繪製圖表", 0.35)
        plots = self._build_charts(ctx)
        summary_payload = self._summarize(ctx)

        report_path = None
        if self.options.export_report and plots:
            self._emit_progress("輸出報告", 0.15)
            report_path = self._write_report(ctx, plots, summary_payload)

        self._emit_progress("完成", 1.0)
        return AnalysisResult(
            plots=plots,
            report_path=report_path,
            trades_path=self.json_path,
            edge_path=None,
            summary=summary_payload,
            warnings=ctx.warnings,
            strategy_name=strategy_used,
        )

    def _append_chart_requirement_warnings(self, ctx: _AnalyzerContext) -> None:
        """依欄位缺失預先提示哪些圖表可能無法生成。"""

        available_cols: Set[str] = set(ctx.trades_df.columns)
        for chart_index in ctx.options.charts:
            required_cols = CHART_REQUIRED_COLUMNS.get(chart_index, ())
            missing = [col for col in required_cols if col not in available_cols]
            if missing:
                ctx.warnings.append(f"Chart {chart_index} 缺少欄位 {missing}，可能無法產生。")

    def _build_charts(self, ctx: _AnalyzerContext) -> List[PlotArtifact]:
        chart_ctx = ChartContext(
            trades_df=ctx.trades_df,
            edge_df=ctx.edge_df,
            output_dir=ctx.output_dir,
            options=ctx.options,
        )
        plots: List[PlotArtifact] = []
        for chart_index in ctx.options.charts:
            builder = CHART_BUILDERS.get(chart_index)
            if builder is None:
                ctx.warnings.append(f"Chart {chart_index} 尚未實作，已略過。")
                continue
            artifact = builder(chart_ctx)
            if artifact is not None:
                plots.append(artifact)
            else:
                ctx.warnings.append(f"Chart {chart_index} 因資料不足未生成。")
        return plots

    def _write_report(
        self,
        ctx: _AnalyzerContext,
        plots: List[PlotArtifact],
        summary: Dict[str, object],
    ) -> Path:
        """輸出 HTML 報告，包含摘要資訊與圖表展示。"""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = ctx.output_dir / "mae_mfe_report.html"
        summary_rows = [
            ("回測 JSON", self.json_path.name),
            ("策略名稱", ctx.strategy_name),
            ("總交易筆數", summary.get("total_trades")),
            ("已平倉筆數", summary.get("closed_trades")),
            ("勝率 (%)", summary.get("win_rate_pct")),
            ("平均報酬 (%)", summary.get("avg_return_pct")),
            ("平均 MAE (%)", summary.get("avg_mae_abs_pct")),
            ("平均 MFE (%)", summary.get("avg_mfe_pct")),
            ("平均持有天數", summary.get("avg_holding_days")),
            ("標的列表", summary.get("symbols")),
        ]
        warning_html = "<p>暫無警示。</p>"
        if ctx.warnings:
            warning_html = "<ul>" + "".join(f"<li>{msg}</li>" for msg in ctx.warnings) + "</ul>"

        sections = [
            "<!DOCTYPE html>",
            "<html lang='zh-Hant'>",
            "<head>",
            "<meta charset='utf-8' />",
            "<title>MAE/MFE Report</title>",
            "<style>body{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:24px;background:#f4f5f7;}",
            "h1,h2{color:#222;} .summary-table{width:100%;border-collapse:collapse;margin-bottom:24px;}",
            ".summary-table th,.summary-table td{border:1px solid #ddd;padding:8px;text-align:left;}",
            ".gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;}",
            ".card{background:#fff;padding:16px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,0.08);}",
            ".card img{max-width:100%;border-radius:4px;}",
            ".meta{color:#666;font-size:0.9rem;margin-bottom:16px;}",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{ctx.options.report_title}</h1>",
            f"<p class='meta'>產出時間：{timestamp}｜樣本數：{len(ctx.trades_df)}</p>",
            "<h2>摘要</h2>",
            "<table class='summary-table'><tbody>",
        ]
        sections.extend(
            f"<tr><th>{label}</th><td>{self._format_value(value)}</td></tr>" for label, value in summary_rows
        )
        sections.extend(
            [
                "</tbody></table>",
                "<h2>警示與備註</h2>",
                warning_html,
                "<h2>圖表</h2>",
                "<div class='gallery'>",
            ]
        )

        for artifact in plots:
            sections.append("<div class='card'>")
            sections.append(f"<h3>{artifact.title}</h3>")
            if artifact.html_path:
                html_rel = artifact.html_path.name
                sections.append(f"<p><a href='{html_rel}' target='_blank'>開啟互動版</a></p>")
            elif artifact.png_path:
                # 預設不輸出 PNG，但若有仍顯示
                rel_path = artifact.png_path.name
                sections.append(f"<img src='{rel_path}' alt='{artifact.title}' />")
            else:
                sections.append("<p>尚未生成可視化輸出。</p>")
            sections.append("</div>")
        sections.extend(["</div>", "</body></html>"])
        report_path.write_text("\n".join(sections), encoding="utf-8")
        return report_path

    def _summarize(self, ctx: _AnalyzerContext) -> Dict[str, object]:
        trades_df = ctx.trades_df.copy()
        total = int(len(trades_df))
        status_series = trades_df.get("status")
        if status_series is not None:
            closed_mask = status_series.str.lower() == "closed"
            closed_df = trades_df[closed_mask].copy()
        else:
            closed_df = trades_df.copy()
        closed_count = int(len(closed_df))

        def _numeric_mean(column: str) -> Optional[float]:
            if column not in closed_df:
                return None
            values = pd.to_numeric(closed_df[column], errors="coerce").dropna()
            if values.empty:
                return None
            mean_value = values.mean()
            return float(mean_value) if math.isfinite(mean_value) else None

        win_rate = None
        if "return_pct" in closed_df and closed_count > 0:
            returns = pd.to_numeric(closed_df["return_pct"], errors="coerce").dropna()
            if not returns.empty:
                win_rate = float((returns > 0).mean() * 100.0)

        symbols = trades_df.get("symbol")
        symbol_list = sorted({str(sym) for sym in symbols.dropna().unique()}) if symbols is not None else []
        return {
            "total_trades": total,
            "closed_trades": closed_count,
            "win_rate_pct": win_rate,
            "avg_return_pct": _numeric_mean("return_pct"),
            "avg_mae_abs_pct": _numeric_mean("mae_abs_pct"),
            "avg_mfe_pct": _numeric_mean("mfe_pct"),
            "avg_holding_days": _numeric_mean("holding_days"),
            "symbols": ", ".join(symbol_list) if symbol_list else "-",
        }

    @staticmethod
    def _format_value(value: object) -> str:
        if value is None:
            return "-"
        if isinstance(value, str):
            return value or "-"
        if isinstance(value, (int, np.integer)):
            return f"{int(value)}"
        if isinstance(value, (float, np.floating)):
            if math.isnan(value):
                return "-"
            return f"{value:.2f}"
        if isinstance(value, list):
            return ", ".join(map(str, value)) or "-"
        return str(value)

    def _emit_progress(self, message: str, progress: float) -> None:
        if self._progress_callback:
            try:
                self._progress_callback(message, float(progress))
            except Exception:
                # callback 失敗不應影響主要流程
                pass


def run_mae_mfe_from_json(
    json_path: str | Path,
    output_dir: str | Path,
    options: AnalysisOptions | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisResult:
    analyzer = MAEMFEJsonAnalyzer(
        json_path=json_path,
        output_dir=output_dir,
        options=options,
        progress_callback=progress_callback,
    )
    return analyzer.run()


__all__ = ["MAEMFEJsonAnalyzer", "run_mae_mfe_from_json"]
