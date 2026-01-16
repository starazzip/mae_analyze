# MAE/MFE JSON Analyzer (mae_tool / mae_analyze)

## What this project does
- CLI helper that converts a Freqtrade-like backtest JSON into MAE/MFE visuals and an optional HTML summary report.
- Supports picking a strategy when the JSON contains multiple strategies, and warns when required fields for certain charts are missing.
- Produces up to 10 Plotly charts: return distribution, edge ratio by holding time, MAE vs return, MAE vs MFE, MAE vs BMFE, MDD vs GMFE, MAE distribution, BMFE distribution, GMFE distribution, and key-metric violin plots.

## Quick start
1) Python 3.10+. Install dependencies:
   - From the project root (e.g., `mae_analyze/` or `mae_tool/`): `pip install -r requirements.txt`
   - Or from the parent directory: `pip install -r mae_analyze/requirements.txt` (adjust the folder name if yours is `mae_tool/`)
2) Convert a backtest JSON:
   ```bash
   # Run as a module from the parent directory, matching your folder name
   # If the folder is mae_analyze/:
   python -m mae_analyze.main --input path/to/backtest.json --strategy MyStrategy \
     --output mae_tool_output/demo_run --charts 1,2,3,4,5,6,7,8,9,10 \
     --report-title "MAE & MFE JSON Report"
   # If the folder is mae_tool/:
   python -m mae_tool.main --input path/to/backtest.json --strategy MyStrategy \
     --output mae_tool_output/demo_run --charts 1,2,3,4,5,6,7,8,9,10 \
     --report-title "MAE & MFE JSON Report"
   # Or, after cd into the folder, run the script directly (name-agnostic)
   python main.py --input path/to/backtest.json --strategy MyStrategy \
     --output mae_tool_output/demo_run --charts 1,2,3,4,5,6,7,8,9,10 \
     --report-title "MAE & MFE JSON Report"
   ```
3) Outputs land in `mae_tool_output/<timestamp>/` (or the `--output` path) as Plotly HTML files plus `mae_mfe_report.html` if reporting is enabled.

## CLI options
- `--input` / `-i` (required): Path to the backtest JSON that contains `strategy.{name}.trades`.
- `--strategy` / `-s`: Strategy key to analyze; defaults to the first strategy in the JSON.
- `--output` / `-o`: Target folder; default `mae_tool_output/<timestamp>`.
- `--charts` / `-c`: Comma list of chart indices to render (1-10).
- `--report-title`: Title shown on the HTML report.
- `--no-report`: Skip the summary HTML (still writes charts).
- `--no-html`: Skip interactive HTML (useful when you only need raw data or PNGs).

## JSON expectations
- The JSON should contain a `strategy` object with one or more strategy keys, each holding a `trades` array with MAE/MFE-related fields (`open_rate`, `close_rate`, `max_rate`, `min_rate`, `profit_ratio`, etc.). Missing fields are tolerated but may reduce which charts can be generated.

---

## 專案說明（中文）
- 這是將 Freqtrade 風格的回測 JSON 轉成 MAE/MFE 圖表與摘要報告的命令列工具。
- 若 JSON 內有多個策略可透過 `--strategy` 指定；缺欄位時會提示哪些圖表可能無法生成。
- 預設可產出 10 張 Plotly 圖：報酬分佈、持有時間對應 Edge Ratio、MAE vs Return、MAE vs MFE、MAE vs BMFE、MDD vs GMFE、MAE 分佈、BMFE 分佈、GMFE 分佈、核心指標小提琴圖。

## 使用步驟
1) 安裝環境：
   - 專案根目錄（如 `mae_analyze/` 或 `mae_tool/`）：`pip install -r requirements.txt`
   - 在父層目錄也可：`pip install -r mae_analyze/requirements.txt`（若資料夾名是 `mae_tool/` 請相應調整）
2) 執行轉換：
   ```bash
   # 從父層目錄以模組方式執行，資料夾名稱請配合實際情況
   # 若資料夾名為 mae_analyze/：
   python -m mae_analyze.main --input path/to/backtest.json --strategy MyStrategy \
     --output mae_tool_output/demo_run --charts 1,2,3,4,5,6,7,8,9,10 \
     --report-title "MAE & MFE JSON Report"
   # 若資料夾名為 mae_tool/：
   python -m mae_tool.main --input path/to/backtest.json --strategy MyStrategy \
     --output mae_tool_output/demo_run --charts 1,2,3,4,5,6,7,8,9,10 \
     --report-title "MAE & MFE JSON Report"
   # 若已 cd 到該資料夾，可直接跑腳本（不依賴資料夾名稱）
   python main.py --input path/to/backtest.json --strategy MyStrategy \
     --output mae_tool_output/demo_run --charts 1,2,3,4,5,6,7,8,9,10 \
     --report-title "MAE & MFE JSON Report"
   ```
3) 產出會放在 `mae_tool_output/<timestamp>/`（或自訂的 `--output` 路徑），包含各圖表的 HTML 與整體報告 `mae_mfe_report.html`（若未停用報告）。

## 參數說明
- `--input` / `-i`：回測 JSON 路徑（必填），需有 `strategy.{name}.trades`。
- `--strategy` / `-s`：指定策略名稱，未填則使用 JSON 第一個策略。
- `--output` / `-o`：輸出資料夾，預設 `mae_tool_output/<timestamp>`。
- `--charts` / `-c`：欲產生的圖表索引（1-10）。
- `--report-title`：報告標題。
- `--no-report`：只輸出圖表，不寫報告 HTML。
- `--no-html`：不輸出互動式 HTML（僅產生必要的檔案或 PNG）。

## JSON 格式期待
- JSON 需包含 `strategy` 物件，底下每個策略含 `trades` 陣列，並帶有 MAE/MFE 所需欄位（如 `open_rate`、`close_rate`、`max_rate`、`min_rate`、`profit_ratio` 等）；若缺部分欄位，可能導致對應圖表被跳過並列出警示。
