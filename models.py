"""MAE/MFE 分析模型定義（JSON 版）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ProgressCallback = Callable[[str, float], None]


@dataclass(slots=True)
class AnalysisOptions:
    """控制 MAE/MFE Analyzer 執行行為。"""

    charts: Tuple[int, ...] = tuple(range(1, 11))
    export_png: bool = False  # 需求：不輸出 PNG
    export_html: bool = True
    export_report: bool = True
    chart_kwargs: Dict[str, object] = field(default_factory=dict)
    report_title: str = "MAE & MFE JSON Report"
    strategy_name: Optional[str] = None

    def is_chart_enabled(self, chart_index: int) -> bool:
        return chart_index in self.charts


@dataclass(slots=True)
class PlotArtifact:
    """單張圖的生成結果。"""

    key: str
    title: str
    png_path: Optional[Path] = None
    html_path: Optional[Path] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class AnalysisResult:
    """Analyzer 的輸出結果。"""

    plots: List[PlotArtifact] = field(default_factory=list)
    report_path: Optional[Path] = None
    trades_path: Optional[Path] = None
    edge_path: Optional[Path] = None
    summary: Dict[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    strategy_name: Optional[str] = None


__all__ = [
    "AnalysisOptions",
    "AnalysisResult",
    "PlotArtifact",
    "ProgressCallback",
]
