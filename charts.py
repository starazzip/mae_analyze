"""MAE/MFE 圖表建構器。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .models import AnalysisOptions, PlotArtifact


@dataclass(slots=True)
class ChartContext:
    trades_df: pd.DataFrame
    edge_df: pd.DataFrame
    output_dir: Path
    options: AnalysisOptions


def _export_plotly_figure(
    fig: go.Figure,
    key: str,
    title: str,
    ctx: ChartContext,
    *,
    include_table: bool = False,
    post_script: Optional[str] = None,
) -> PlotArtifact:
    png_path: Optional[Path] = None
    html_path: Optional[Path] = None

    if ctx.options.export_html:
        html_path = ctx.output_dir / f"{key}.html"
        fig.write_html(
            str(html_path),
            include_plotlyjs="cdn",
            full_html=True,
            div_id=key,
            post_script=post_script or "",
        )

    if ctx.options.export_png:
        png_path = ctx.output_dir / f"{key}.png"
        try:
            fig.write_image(str(png_path), scale=2)
        except Exception:
            png_path = None

    return PlotArtifact(
        key=key,
        title=title,
        png_path=png_path,
        html_path=html_path,
        metadata={"has_table": include_table},
    )


def build_chart_1_return_distribution(ctx: ChartContext) -> Optional[PlotArtifact]:
    df = ctx.trades_df
    if df.empty or "return_pct" not in df:
        return None
    closed_df = df[df["status"] == "closed"] if "status" in df else df
    closed_df = closed_df.copy()
    closed_df["return_pct"] = pd.to_numeric(closed_df["return_pct"], errors="coerce")
    closed_df = closed_df.dropna(subset=["return_pct"])
    if closed_df.empty:
        return None

    win_mask = closed_df["return_pct"] > 0
    win_rate = win_mask.mean() * 100
    avg_return = closed_df["return_pct"].mean()
    bins = min(max(int(len(closed_df) / 5), 20), 120)

    fig = go.Figure()
    fig.add_histogram(
        x=closed_df.loc[win_mask, "return_pct"],
        name="Winners",
        nbinsx=bins,
        marker_color="#2ca02c",
        opacity=0.65,
    )
    fig.add_histogram(
        x=closed_df.loc[~win_mask, "return_pct"],
        name="Losers",
        nbinsx=bins,
        marker_color="#d62728",
        opacity=0.55,
    )
    fig.add_vline(x=float(avg_return), line_width=2, line_dash="dash", line_color="#1f77b4")
    fig.update_layout(
        title=f"Return Distribution | Win Rate {win_rate:.1f}% | Avg Return {avg_return:.2f}%",
        xaxis_title="Return (%)",
        yaxis_title="Count",
        barmode="overlay",
        legend=dict(orientation="h"),
    )
    fig.update_traces(hovertemplate="Return: %{x:.2f}%<br>Count: %{y}")
    return _export_plotly_figure(fig, "chart01_return_distribution", "Return Distribution", ctx)


def build_chart_2_edge_ratio(ctx: ChartContext) -> Optional[PlotArtifact]:
    df = ctx.edge_df
    if df.empty:
        return None
    working = df.copy()
    if "time_scale" not in working or "edge_ratio" not in working:
        return None
    working["edge_ratio"] = pd.to_numeric(working["edge_ratio"], errors="coerce")
    working = working.dropna(subset=["edge_ratio"])
    if working.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=working["time_scale"],
            y=working["edge_ratio"],
            mode="lines+markers",
            line=dict(color="#9467bd"),
            marker=dict(size=8),
            name="Edge Ratio",
        )
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="#999999", annotation_text="Edge = 1")
    fig.update_layout(
        title="Edge Ratio vs Time",
        xaxis_title="Time Scale",
        yaxis_title="Edge Ratio (GMFE / MAE)",
    )
    return _export_plotly_figure(fig, "chart02_edge_ratio", "Edge Ratio vs Time", ctx)


CLOSED_STATUS = "closed"
WIN_COLOR = "#2ca02c"
LOSS_COLOR = "#d62728"
ACCENT_COLOR = "#1f77b4"
DETAIL_FIELD_LABELS = {
    "symbol": "Symbol",
    "direction": "方向",
    "entry_time": "進場時間",
    "exit_time": "出場時間",
    "mae_time": "MAE 時間",
    "gmfe_time": "GMFE 時間",
    "bmfe_time": "BMFE 時間",
    "entry_price": "進場價",
    "exit_price": "出場價",
    "return_pct": "報酬 (%)",
    "mae_abs_pct": "MAE (%)",
    "mfe_pct": "GMFE (%)",
    "bmfe_pct": "BMFE (%)",
    "mdd_pct": "MDD (%)",
    "pdays": "pdays",
    "pdays_ratio": "pdays_ratio",
    "holding_days": "持有天數",
    "holding_bars": "持有 bars",
    "exit_reason": "出場原因",
    "status": "狀態",
}


def _build_detail_post_script(div_id: str, fields: Iterable[str]) -> str:  # type: ignore[override]
    """產生 Plotly 匯出 HTML 的互動腳本，點擊散點顯示交易細節與近似走勢。"""

    fields = [field for field in fields]
    if not fields:
        return ""

    labels = [DETAIL_FIELD_LABELS.get(field, field) for field in fields]
    panel_html = (
        "<div style=\"display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;\">"
        "<div style=\"flex:1 1 360px;min-width:320px;border:1px solid #111;"
        "border-radius:8px;padding:12px;background:#ffffff;box-shadow:0 8px 20px rgba(0,0,0,0.08);\">"
        "<h4 style=\"margin:0 0 8px;color:#111;\">交易細節</h4>"
        "<div data-role=\"trade-detail-content\">點擊散點以檢視交易資訊。</div>"
        "</div>"
        "<div style=\"flex:1 1 360px;min-width:320px;border:1px solid #111;"
        "border-radius:8px;padding:12px;background:#ffffff;box-shadow:0 8px 20px rgba(0,0,0,0.08);\">"
        "<h4 style=\"margin:0 0 8px;color:#111;\">近似走勢</h4>"
        f"<div id=\"{div_id}-spark\" data-role=\"trade-spark\" style=\"height:240px;\"></div>"
        "</div>"
        "</div>"
    )
    panel_json = json.dumps(panel_html, ensure_ascii=True)
    fields_json = json.dumps(fields, ensure_ascii=True)
    labels_json = json.dumps(labels, ensure_ascii=True)
    time_label = json.dumps("時間", ensure_ascii=True)
    price_label = json.dumps("相對價格 (%)", ensure_ascii=True)
    node_label = json.dumps("節點", ensure_ascii=True)
    value_label = json.dumps("值", ensure_ascii=True)

    script_lines = [
        f"const plot=document.getElementById('{div_id}');",
        "if(!plot||!window.Plotly)return;",
        f"const panelId='{div_id}-detail';",
        "let panel=document.getElementById(panelId);",
        "if(!panel){panel=document.createElement('div');panel.id=panelId;panel.style.marginTop='12px';",
        f"panel.innerHTML={panel_json};",
        "const parent=plot.parentNode;if(parent){parent.insertBefore(panel,plot.nextSibling);}",
        "}",
        "const content=panel.querySelector('[data-role=\\\"trade-detail-content\\\"]');",
        "const spark=panel.querySelector('[data-role=\\\"trade-spark\\\"]');",
        "if(!content)return;",
        f"const fields={fields_json};",
        f"const labels={labels_json};",
        "function formatVal(val){if(val===null||val===undefined||val==='')return '-';",
        "  const num=Number(val);if(Number.isFinite(num)){",
        "    if(Math.abs(num)>=100)return num.toFixed(2);",
        "    return num.toFixed(4);",
        "  }",
        "  return String(val);",
        "}",
        "function fieldIndex(name){return fields.indexOf(name);}",
        "function buildSpark(data){",
        "  if(!spark||!window.Plotly)return;",
        "  const mae=Math.max(0,Number(data[fieldIndex('mae_abs_pct')])||0)/100;",
        "  const gmfe=Math.max(0,Number(data[fieldIndex('mfe_pct')])||0)/100;",
        "  const bmfe=Math.max(0,Number(data[fieldIndex('bmfe_pct')])||0)/100;",
        "  const retPct=(Number(data[fieldIndex('return_pct')])||0)/100;",
        "  const parseTime=(name)=>{const idx=fieldIndex(name);if(idx<0)return NaN;const raw=data[idx];const v=raw instanceof Date?raw.valueOf():Date.parse(raw);return Number.isFinite(v)?v:NaN;};",
        "  const tEntryMs=parseTime('entry_time');",
        "  const tExitMs=parseTime('exit_time');",
        "  const tMaeMs=parseTime('mae_time');",
        "  const tGmfeMs=parseTime('gmfe_time');",
        "  const tBmfeMs=parseTime('bmfe_time');",
        "  const hasRange=Number.isFinite(tEntryMs)&&Number.isFinite(tExitMs)&&tExitMs>tEntryMs;",
        "  const duration=hasRange?(tExitMs-tEntryMs):null;",
        "  const pct=100;",
        "  const basePts=[",
        "    {label:'Entry',value:100,timeMs:tEntryMs,scale:0,order:0},",
        "    {label:'BMFE',value:100+(bmfe*pct),timeMs:tBmfeMs,scale:0.25,order:1},",
        "    {label:'MAE',value:100-(mae*pct),timeMs:tMaeMs,scale:0.45,order:2},",
        "    {label:'GMFE',value:100+(gmfe*pct),timeMs:tGmfeMs,scale:0.7,order:3},",
        "    {label:'Exit',value:(1+retPct)*pct,timeMs:tExitMs,scale:1,order:4},",
        "  ];",
        "  const mergedByCoord=[];",
        "  basePts.forEach((pt)=>{",
        "    const curTime=Number.isFinite(pt.timeMs)?pt.timeMs:null;",
        "    const found=mergedByCoord.find((item)=>{",
        "      const otherTime=Number.isFinite(item.timeMs)?item.timeMs:null;",
        "      const sameTime=(curTime===null&&otherTime===null)||(curTime!==null&&otherTime!==null&&Math.abs(curTime-otherTime)<=1);",
        "      const sameValue=Math.abs((pt.value||0)-(item.value||0))<=1e-6;",
        "      return sameTime&&sameValue;",
        "    });",
        "    if(found){",
        "      found.label=found.label.includes(pt.label)?found.label:`${found.label}/${pt.label}`;",
        "    }else{",
        "      mergedByCoord.push({...pt});",
        "    }",
        "  });",
        "  const resolveX=(pt)=>{",
        "    if(Number.isFinite(pt.timeMs))return new Date(pt.timeMs);",
        "    if(hasRange&&duration!==null)return new Date(tEntryMs+duration*pt.scale);",
        "    return pt.order;",
        "  };",
        "  const pts=mergedByCoord.map(pt=>({...pt,x:resolveX(pt),y:pt.value}));",
        "  pts.sort((a,b)=>{",
        "    const ax=a.x instanceof Date?a.x.valueOf():a.x;const bx=b.x instanceof Date?b.x.valueOf():b.x;",
        "    if(ax===bx)return (a.order??0)-(b.order??0);",
        "    return ax-bx;",
        "  });",
        "  const merged=[];",
        "  for(const pt of pts){",
        "    const xv=pt.x instanceof Date?pt.x.valueOf():pt.x;",
        "    const key=`${xv}:${pt.y}`;",
        "    const found=merged.find(p=>p.key===key);",
        "    if(found){found.labels.push(pt.label);continue;}",
        "    merged.push({...pt,key,labels:[pt.label]});",
        "  }",
        "  const expandPoints=(items)=>{",
        "    const expanded=[];",
        "    items.forEach((pt)=>{",
        "      const labels=pt.labels&&pt.labels.length?pt.labels:[pt.label];",
        "      const count=labels.length||1;",
        "      const half=(count-1)/2;",
        "      labels.forEach((label,idx)=>{",
        "        let xVal=pt.x;",
        "        let yVal=pt.y??100;",
        "        if(count>1){",
        "          const shift=idx-half;",
        "          if(pt.x instanceof Date){",
        "            const base=pt.x.valueOf();",
        "            const baseDuration=duration||24*3600*1000;",
        "            const spacing=Math.max(baseDuration*0.01,3600*1000);",
        "            xVal=new Date(base+shift*spacing);",
        "          }else{",
        "            const base=typeof pt.x==='number'?pt.x:Number(pt.x)||0;",
        "            xVal=base+shift*0.1;",
        "          }",
        "          yVal+=shift*4;",
        "        }",
        "        expanded.push({...pt,x:xVal,y:yVal,labels:[label],text:label});",
        "      });",
        "    });",
        "    return expanded;",
        "  };",
        "  const adjustedPoints=expandPoints(merged);",
        "  const texts=adjustedPoints.map(p=>p.labels.join('/'));",
        "  const useDateAxis=adjustedPoints.every(p=>p.x instanceof Date);",
        "  const tickvals=useDateAxis?undefined:adjustedPoints.map(p=>p.x);",
        "  const ticktext=useDateAxis?undefined:texts;",
        "  const ys=adjustedPoints.map(p=>p.y);",
        "  const offsets=ys.map(v=>v-100);",
        "  const maxAbs=Math.max(...offsets.map(v=>Math.abs(v)));",
        "  const span=Math.max(5,maxAbs||5);",
        "  const yMin=100-span;",
        "  const yMax=100+span;",
        "  const layout={margin:{t:24,r:12,b:28,l:52},height:240,xaxis:{title:"+time_label+",type:useDateAxis?'date':'linear',",
        "    tickvals:tickvals,ticktext:ticktext},",
        "    yaxis:{title:"+price_label+",range:[yMin,yMax],tickformat:',.0f'},paper_bgcolor:'#fff',plot_bgcolor:'#fff'};",
        "  const trace={type:'scatter',mode:'lines+markers+text',x:adjustedPoints.map(p=>p.x),y:adjustedPoints.map(p=>p.y),text:texts,",
        "    textposition:'top center',line:{color:'#2563eb',width:2},marker:{size:8,color:'#2563eb',line:{width:1,color:'#111'}},",
        "    hovertemplate:`"+node_label+": %{text}<br>"+value_label+": %{y:.1f}%<extra></extra>`};",
        "  window.Plotly.react(spark,[trace],layout,{displayModeBar:false});",
        "}",
        "plot.on('plotly_click',function(ev){",
        "  const pt=ev&&ev.points&&ev.points[0];if(!pt||!pt.customdata)return;",
        "  const data=pt.customdata;let html='<table style=\\\"width:100%;border-collapse:collapse;\\\">';",
        "  for(let i=0;i<fields.length;i++){const lbl=labels[i]||fields[i];",
        "    html+=`<tr><th style=\\\"text-align:left;padding:4px 8px;border-bottom:1px solid #eee;\\\">${lbl}</th>`+",
        "    `<td style=\\\"padding:4px 8px;border-bottom:1px solid #eee;\\\">${formatVal(data[i])}</td></tr>`;",
        "  }",
        "  html+='</table>';content.innerHTML=html;buildSpark(data);",
        "});",
    ]
    script_body = "".join(script_lines)
    return "(function(){try{" + script_body + "}catch(e){console.warn('detail script error',e);}})();"


def _get_closed_trades(df: pd.DataFrame) -> pd.DataFrame:
    """取出已平倉交易，若無狀態欄位則直接回傳副本。"""
    if df.empty:
        return df
    if "status" in df:
        closed = df[df["status"].str.lower() == CLOSED_STATUS]
        if not closed.empty:
            return closed.copy()
    return df.copy()


def _compute_density(points: pd.Series, samples: int = 200) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """以 KDE 為主、退而求其次採用平滑直方估計，產生密度曲線資料。"""
    arr = pd.to_numeric(points, errors="coerce").to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    xs = np.linspace(arr.min(), arr.max(), samples)
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(arr)
        ys = kde(xs)
        return xs, ys
    except Exception:
        bins = min(40, max(10, arr.size // 2))
        hist, bin_edges = np.histogram(arr, bins=bins, density=True)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        xs = np.linspace(centers.min(), centers.max(), samples)
        ys = np.interp(xs, centers, hist)
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        ys = np.convolve(ys, kernel / kernel.sum(), mode="same")
        return xs, ys


def build_chart_3_mae_vs_return(ctx: ChartContext) -> Optional[PlotArtifact]:
    df_raw = _get_closed_trades(ctx.trades_df)
    cols = ["return_pct", "mae_abs_pct"]
    detail_fields = [
        "symbol",
        "direction",
        "entry_time",
        "exit_time",
        "mae_time",
        "gmfe_time",
        "bmfe_time",
        "entry_price",
        "exit_price",
        "return_pct",
        "mae_abs_pct",
        "mfe_pct",
        "bmfe_pct",
        "mdd_pct",
        "pdays",
        "pdays_ratio",
        "holding_days",
        "exit_reason",
    ]
    if not set(cols).issubset(df_raw.columns):
        return None
    available_details = [field for field in detail_fields if field in df_raw.columns]
    select_cols = list(dict.fromkeys(cols + available_details))
    df = df_raw[select_cols].copy()
    numeric_cols = [
        col
        for col in select_cols
        if col.endswith("_pct") or col in {"holding_days", "entry_price", "exit_price", "pdays", "pdays_ratio"}
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=cols)
    if df.empty:
        return None
    df["win"] = df["return_pct"] > 0
    df["abs_return"] = df["return_pct"].abs()
    max_abs_return = float(df["abs_return"].max()) if not df.empty else 0.0
    q3_win = df.loc[df["win"], "mae_abs_pct"].quantile(0.75) if df["win"].any() else None
    q3_loss = df.loc[~df["win"], "mae_abs_pct"].quantile(0.75) if (~df["win"]).any() else None
    fig = go.Figure()
    available_details = [field for field in detail_fields if field in df.columns]
    for label, color, mask in [
        ("Winners", WIN_COLOR, df["win"]),
        ("Losers", LOSS_COLOR, ~df["win"]),
    ]:
        subset = df[mask]
        if subset.empty:
            continue
        customdata = subset[available_details].to_numpy() if available_details else None
        sizes = []
        if max_abs_return > 0:
            for ret in subset["abs_return"]:
                sizes.append(10 + (ret / max_abs_return) * 30)
        else:
            sizes = [10] * len(subset)
        sizeref = (2.0 * max(sizes)) / (40.0**2) if sizes else 1
        fig.add_trace(
            go.Scatter(
                x=subset["return_pct"],
                y=subset["mae_abs_pct"],
                mode="markers",
                marker=dict(
                    color=color,
                    size=sizes,
                    sizemode="area",
                    sizeref=sizeref,
                    opacity=0.75,
                    line=dict(width=1, color="#111"),
                ),
                name=label,
                customdata=customdata,
                hovertemplate="Return: %{x:.2f}%<br>MAE: %{y:.2f}%<extra></extra>",
            )
        )
    if q3_win is not None:
        fig.add_hline(y=q3_win, line_dash="dash", line_color=WIN_COLOR, annotation_text="Q3 (Win)")
    if q3_loss is not None:
        fig.add_hline(y=q3_loss, line_dash="dash", line_color=LOSS_COLOR, annotation_text="Q3 (Loss)")
    fig.update_layout(
        title="Chart 3 - MAE vs Return",
        xaxis_title="Return (%)",
        xaxis=dict(tickformat=".2f"),
        yaxis_title="MAE Abs (%)",
        legend=dict(orientation="h"),
    )
    post_script = _build_detail_post_script("chart03_mae_vs_return", available_details)
    return _export_plotly_figure(fig, "chart03_mae_vs_return", "MAE vs Return", ctx, post_script=post_script)


def build_chart_4_mae_vs_mfe(ctx: ChartContext) -> Optional[PlotArtifact]:
    df_raw = _get_closed_trades(ctx.trades_df)
    required = ["mae_abs_pct", "mfe_pct", "return_pct"]
    detail_fields = [
        "symbol",
        "direction",
        "entry_time",
        "exit_time",
        "mae_time",
        "gmfe_time",
        "bmfe_time",
        "entry_price",
        "exit_price",
        "return_pct",
        "mae_abs_pct",
        "mfe_pct",
        "bmfe_pct",
        "mdd_pct",
        "pdays",
        "pdays_ratio",
        "holding_days",
        "exit_reason",
    ]
    if not set(required).issubset(df_raw.columns):
        return None
    available_details = [field for field in detail_fields if field in df_raw.columns]
    select_cols = list(dict.fromkeys(required + available_details))
    df = df_raw[select_cols].copy()
    numeric_cols = [
        col
        for col in select_cols
        if col.endswith("_pct") or col in {"holding_days", "entry_price", "exit_price", "pdays", "pdays_ratio"}
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=required)
    if df.empty:
        return None
    df["win"] = df["return_pct"] > 0
    df["abs_return"] = df["return_pct"].abs()
    max_abs_return = float(df["abs_return"].max()) if not df.empty else 0.0
    available_details = [field for field in detail_fields if field in df.columns]
    fig = go.Figure()
    for label, color, mask in [
        ("Winners", WIN_COLOR, df["win"]),
        ("Losers", LOSS_COLOR, ~df["win"]),
    ]:
        subset = df[mask]
        if subset.empty:
            continue
        customdata = subset[available_details].to_numpy() if available_details else None
        sizes = []
        if max_abs_return > 0:
            for ret in subset["abs_return"]:
                sizes.append(10 + (ret / max_abs_return) * 30)
        else:
            sizes = [10] * len(subset)
        sizeref = (2.0 * max(sizes)) / (40.0**2) if sizes else 1
        fig.add_trace(
            go.Scatter(
                x=subset["mae_abs_pct"],
                y=subset["mfe_pct"],
                mode="markers",
                marker=dict(
                    color=color,
                    size=sizes,
                    sizemode="area",
                    sizeref=sizeref,
                    opacity=0.75,
                    line=dict(width=1, color="#111"),
                ),
                name=label,
                customdata=customdata,
                hovertemplate="MAE: %{x:.2f}%<br>GMFE: %{y:.2f}%<extra></extra>",
            )
        )
    max_range = float(max(df["mae_abs_pct"].max(), df["mfe_pct"].max()))
    fig.add_trace(
        go.Scatter(
            x=[0, max_range],
            y=[0, max_range],
            mode="lines",
            line=dict(color="#888", dash="dot"),
            name="MFE = MAE",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Chart 4 - MAE vs GMFE",
        xaxis_title="MAE Abs (%)",
        yaxis_title="GMFE (%)",
        legend=dict(orientation="h"),
    )
    post_script = _build_detail_post_script("chart04_mae_vs_gmfe", available_details)
    return _export_plotly_figure(fig, "chart04_mae_vs_gmfe", "MAE vs GMFE", ctx, post_script=post_script)


def build_chart_5_mae_vs_bmfe(ctx: ChartContext) -> Optional[PlotArtifact]:
    df_raw = _get_closed_trades(ctx.trades_df)
    required = ["mae_abs_pct", "bmfe_pct", "return_pct"]
    detail_fields = [
        "symbol",
        "direction",
        "entry_time",
        "exit_time",
        "mae_time",
        "gmfe_time",
        "bmfe_time",
        "entry_price",
        "exit_price",
        "return_pct",
        "mae_abs_pct",
        "mfe_pct",
        "bmfe_pct",
        "mdd_pct",
        "pdays",
        "pdays_ratio",
        "holding_days",
        "exit_reason",
    ]
    if not set(required).issubset(df_raw.columns):
        return None
    available_details = [field for field in detail_fields if field in df_raw.columns]
    select_cols = list(dict.fromkeys(required + available_details))
    df = df_raw[select_cols].copy()
    numeric_cols = [
        col
        for col in select_cols
        if col.endswith("_pct") or col in {"holding_days", "entry_price", "exit_price", "pdays", "pdays_ratio"}
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=required)
    if df.empty:
        return None
    df["win"] = df["return_pct"] > 0
    df["abs_return"] = df["return_pct"].abs()
    max_abs_return = float(df["abs_return"].max()) if not df.empty else 0.0
    available_details = [field for field in detail_fields if field in df.columns]
    fig = go.Figure()
    for label, color, mask in [
        ("Winners", WIN_COLOR, df["win"]),
        ("Losers", LOSS_COLOR, ~df["win"]),
    ]:
        subset = df[mask]
        if subset.empty:
            continue
        customdata = subset[available_details].to_numpy() if available_details else None
        sizes = []
        if max_abs_return > 0:
            for ret in subset["abs_return"]:
                sizes.append(10 + (ret / max_abs_return) * 30)
        else:
            sizes = [10] * len(subset)
        sizeref = (2.0 * max(sizes)) / (40.0**2) if sizes else 1
        fig.add_trace(
            go.Scatter(
                x=subset["mae_abs_pct"],
                y=subset["bmfe_pct"],
                mode="markers",
                marker=dict(
                    color=color,
                    size=sizes,
                    sizemode="area",
                    sizeref=sizeref,
                    opacity=0.75,
                    line=dict(width=1, color="#111"),
                ),
                name=label,
                customdata=customdata,
                hovertemplate="MAE: %{x:.2f}%<br>BMFE: %{y:.2f}%<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, df["mae_abs_pct"].max()],
            y=[0, df["bmfe_pct"].max()],
            mode="lines",
            line=dict(color="#999", dash="dot"),
            name="Reference",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Chart 5 - MAE vs BMFE",
        xaxis_title="MAE Abs (%)",
        yaxis_title="BMFE (%)",
        legend=dict(orientation="h"),
    )
    post_script = _build_detail_post_script("chart05_mae_vs_bmfe", available_details)
    return _export_plotly_figure(fig, "chart05_mae_vs_bmfe", "MAE vs BMFE", ctx, post_script=post_script)


def build_chart_6_mdd_vs_gmfe(ctx: ChartContext) -> Optional[PlotArtifact]:
    df_raw = _get_closed_trades(ctx.trades_df)
    required = ["mdd_pct", "mfe_pct", "return_pct"]
    detail_fields = [
        "symbol",
        "direction",
        "entry_time",
        "exit_time",
        "mae_time",
        "gmfe_time",
        "bmfe_time",
        "entry_price",
        "exit_price",
        "return_pct",
        "mae_abs_pct",
        "mfe_pct",
        "bmfe_pct",
        "mdd_pct",
        "pdays",
        "pdays_ratio",
        "holding_days",
        "exit_reason",
    ]
    if not set(required).issubset(df_raw.columns):
        return None
    available_details = [field for field in detail_fields if field in df_raw.columns]
    select_cols = list(dict.fromkeys(required + available_details))
    df = df_raw[select_cols].copy()
    numeric_cols = [
        col
        for col in select_cols
        if col.endswith("_pct") or col in {"holding_days", "entry_price", "exit_price", "pdays", "pdays_ratio"}
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=required)
    if df.empty:
        return None
    df["mdd_abs"] = df["mdd_pct"].abs()
    df["gmfe_abs"] = df["mfe_pct"].abs()
    df["win"] = df["return_pct"] > 0
    df["abs_return"] = df["return_pct"].abs()
    max_abs_return = float(df["abs_return"].max()) if not df.empty else 0.0
    available_details = [field for field in detail_fields if field in df.columns]
    fig = go.Figure()
    for label, color, mask in [
        ("Winners", WIN_COLOR, df["win"]),
        ("Losers", LOSS_COLOR, ~df["win"]),
    ]:
        subset = df[mask]
        if subset.empty:
            continue
        customdata = subset[available_details].to_numpy() if available_details else None
        sizes = []
        if max_abs_return > 0:
            for ret in subset["abs_return"]:
                sizes.append(10 + (ret / max_abs_return) * 30)
        else:
            sizes = [10] * len(subset)
        sizeref = (2.0 * max(sizes)) / (40.0**2) if sizes else 1
        fig.add_trace(
            go.Scatter(
                x=subset["gmfe_abs"],
                y=subset["mdd_abs"],
                mode="markers",
                marker=dict(
                    color=color,
                    size=sizes,
                    sizemode="area",
                    sizeref=sizeref,
                    opacity=0.75,
                    line=dict(width=1, color="#111"),
                ),
                name=label,
                customdata=customdata,
                hovertemplate="GMFE: %{x:.2f}%<br>MDD: %{y:.2f}%<extra></extra>",
            )
        )
    max_range = float(max(df["gmfe_abs"].max(), df["mdd_abs"].max()))
    fig.add_trace(
        go.Scatter(
            x=[0, max_range],
            y=[0, max_range],
            mode="lines",
            line=dict(color="#999", dash="dot"),
            name="GMFE = MDD",
            showlegend=False,
        )
    )
    winners = df[df["win"]]
    if not winners.empty:
        missed = (winners["mdd_abs"] > winners["gmfe_abs"]).mean() * 100
        safe = (winners["mdd_abs"] <= winners["gmfe_abs"]).mean() * 100
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            text=f"Missed wins: {missed:.1f}%<br>Breakeven-safe: {safe:.1f}%",
            align="left",
            bordercolor="#ccc",
            borderwidth=1,
            borderpad=4,
            bgcolor="#f8f8f8",
        )
    fig.update_layout(
        title="Chart 6 - MDD vs GMFE",
        xaxis_title="GMFE Abs (%)",
        yaxis_title="MDD Abs (%)",
        legend=dict(orientation="h"),
    )
    post_script = _build_detail_post_script("chart06_mdd_vs_gmfe", available_details)
    return _export_plotly_figure(fig, "chart06_mdd_vs_gmfe", "MDD vs GMFE", ctx, post_script=post_script)


def build_chart_7_mae_distribution(ctx: ChartContext) -> Optional[PlotArtifact]:
    """MAE 分布直方圖，勝綠敗紅並附帶密度線與分位標記。"""

    df_raw = _get_closed_trades(ctx.trades_df)
    required = ["mae_abs_pct", "return_pct"]
    if not set(required).issubset(df_raw.columns):
        return None
    df = df_raw[required].copy()
    df[required] = df[required].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["mae_abs_pct"])
    if df.empty:
        return None
    df["win"] = df["return_pct"] > 0
    fig = go.Figure()
    for label, mask, color in [("Winners", df["win"], WIN_COLOR), ("Losers", ~df["win"], LOSS_COLOR)]:
        subset = df.loc[mask, "mae_abs_pct"].dropna()
        if subset.empty:
            continue
        fig.add_histogram(
            x=subset,
            nbinsx=40,
            histnorm="probability density",
            marker=dict(color=color, line=dict(color="#111", width=0.5)),
            opacity=0.45 if label == "Losers" else 0.6,
            name=label,
        )
        density = _compute_density(subset)
        if density:
            xs, ys = density
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    name=f"{label} KDE",
                    showlegend=False,
                )
            )
        q3 = subset.quantile(0.75)
        fig.add_vline(
            x=float(q3),
            line_dash="dash",
            line_color=color,
            annotation_text=f"Q3:{q3:.2f}%",
            annotation_position="top",
        )
    fig.update_layout(
        barmode="overlay",
        title="Chart 7 - MAE distributions",
        xaxis_title="mae(%)",
        yaxis_title="密度",
        bargap=0.05,
        legend=dict(orientation="h"),
    )
    return _export_plotly_figure(fig, "chart07_mae_distribution", "MAE distributions", ctx)


def build_chart_8_mfe_distribution(ctx: ChartContext) -> Optional[PlotArtifact]:
    """BMFE 分布直方圖，勝綠敗紅並附帶密度線與分位標記。"""

    df_raw = _get_closed_trades(ctx.trades_df)
    required = ["bmfe_pct", "return_pct"]
    if not set(required).issubset(df_raw.columns):
        return None
    df = df_raw[required].copy()
    df[required] = df[required].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["bmfe_pct"])
    if df.empty:
        return None
    df["win"] = df["return_pct"] > 0
    fig = go.Figure()
    for label, mask, color in [("Winners", df["win"], WIN_COLOR), ("Losers", ~df["win"], LOSS_COLOR)]:
        subset = df.loc[mask, "bmfe_pct"].dropna()
        if subset.empty:
            continue
        fig.add_histogram(
            x=subset,
            nbinsx=40,
            histnorm="probability density",
            marker=dict(color=color, line=dict(color="#111", width=0.5)),
            opacity=0.45 if label == "Losers" else 0.6,
            name=label,
        )
        density = _compute_density(subset)
        if density:
            xs, ys = density
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    name=f"{label} KDE",
                    showlegend=False,
                )
            )
        q3 = subset.quantile(0.75)
        fig.add_vline(
            x=float(q3),
            line_dash="dash",
            line_color=color,
            annotation_text=f"Q3:{q3:.2f}%",
            annotation_position="top",
        )
    fig.update_layout(
        title="Chart 8 - BMFE distributions",
        xaxis_title="bmfe(%)",
        yaxis_title="密度",
        barmode="overlay",
        bargap=0.05,
        legend=dict(orientation="h"),
    )
    return _export_plotly_figure(fig, "chart08_mfe_distribution", "BMFE distributions", ctx)


def build_chart_9_mfe_distribution(ctx: ChartContext) -> Optional[PlotArtifact]:
    """GMFE 分布直方圖，勝綠敗紅並附帶密度線與分位標記。"""

    df_raw = _get_closed_trades(ctx.trades_df)
    required = ["mfe_pct", "return_pct"]
    if not set(required).issubset(df_raw.columns):
        return None
    df = df_raw[required].copy()
    df[required] = df[required].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["mfe_pct"])
    if df.empty:
        return None
    df["win"] = df["return_pct"] > 0
    fig = go.Figure()
    for label, mask, color in [("Winners", df["win"], WIN_COLOR), ("Losers", ~df["win"], LOSS_COLOR)]:
        subset = df.loc[mask, "mfe_pct"].dropna()
        if subset.empty:
            continue
        fig.add_histogram(
            x=subset,
            nbinsx=40,
            histnorm="probability density",
            marker=dict(color=color, line=dict(color="#111", width=0.5)),
            opacity=0.45 if label == "Losers" else 0.6,
            name=label,
        )
        density = _compute_density(subset)
        if density:
            xs, ys = density
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    name=f"{label} KDE",
                    showlegend=False,
                )
            )
        q3 = subset.quantile(0.75)
        fig.add_vline(
            x=float(q3),
            line_dash="dash",
            line_color=color,
            annotation_text=f"Q3:{q3:.2f}%",
            annotation_position="top",
        )
    fig.update_layout(
        barmode="overlay",
        title="Chart 9 - GMFE distributions",
        xaxis_title="gmfe(%)",
        yaxis_title="密度",
        bargap=0.05,
        legend=dict(orientation="h"),
    )
    return _export_plotly_figure(fig, "chart09_mfe_distribution", "GMFE distributions", ctx)


def build_chart_10_indices_violin(ctx: ChartContext) -> Optional[PlotArtifact]:
    """核心指標分布（勝綠敗紅）小提琴圖，使用 overlay 模式。"""

    df_raw = _get_closed_trades(ctx.trades_df)
    candidates = [
        "return_pct",
        "mae_abs_pct",
        "mfe_pct",
        "bmfe_pct",
        "mdd_pct",
        "pdays_ratio",
        "holding_days",
    ]
    available = [col for col in candidates if col in df_raw.columns]
    if not available:
        return None
    df = df_raw[available].copy()
    df[available] = df[available].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    if df.empty:
        return None
    if "return_pct" not in df:
        return None
    win_mask = df["return_pct"] > 0
    metrics = [col for col in available if col != "return_pct"]
    if not metrics:
        return None
    fig = go.Figure()
    for idx, col in enumerate(metrics):
        label = DETAIL_FIELD_LABELS.get(col, col)
        for trace_name, mask, color in [
            ("Profit", win_mask, WIN_COLOR),
            ("Loss", ~win_mask, LOSS_COLOR),
        ]:
            series = df.loc[mask, col].dropna()
            if series.empty:
                continue
            fig.add_trace(
                go.Violin(
                    y=series,
                    x=[label] * len(series),
                    name=trace_name if idx == 0 else trace_name,
                    legendgroup=trace_name,
                    showlegend=idx == 0,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=color,
                    line=dict(color=color, width=1),
                    opacity=0.45 if trace_name == "Loss" else 0.55,
                    points="all",
                    pointpos=0,
                    jitter=0.15,
                    spanmode="soft",
                )
            )
    fig.update_layout(
        title="Chart 10 - Indices Stats",
        yaxis_title="值",
        violingap=0.1,
        violinmode="overlay",
        legend=dict(orientation="h"),
    )
    return _export_plotly_figure(fig, "chart10_indices_violin", "Indices Stats", ctx)


CHART_BUILDERS: Dict[int, callable] = {
    1: build_chart_1_return_distribution,
    2: build_chart_2_edge_ratio,
    3: build_chart_3_mae_vs_return,
    4: build_chart_4_mae_vs_mfe,
    5: build_chart_5_mae_vs_bmfe,
    6: build_chart_6_mdd_vs_gmfe,
    7: build_chart_7_mae_distribution,
    8: build_chart_8_mfe_distribution,
    9: build_chart_9_mfe_distribution,
    10: build_chart_10_indices_violin,
}

__all__ = [
    "CHART_BUILDERS",
    "ChartContext",
    "build_chart_1_return_distribution",
    "build_chart_2_edge_ratio",
    "build_chart_3_mae_vs_return",
    "build_chart_4_mae_vs_mfe",
    "build_chart_5_mae_vs_bmfe",
    "build_chart_6_mdd_vs_gmfe",
    "build_chart_7_mae_distribution",
    "build_chart_8_mfe_distribution",
    "build_chart_9_mfe_distribution",
    "build_chart_10_indices_violin",
]
