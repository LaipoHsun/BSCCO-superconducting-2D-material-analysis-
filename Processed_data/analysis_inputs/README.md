# Analysis Inputs (Processed_data)

這個資料夾提供 `analysis_code/` 直接讀取的分析輸入，避免分析流程碰到 `data/`（raw 區）。

## 檔案內容

- `all_cleaned_iv.csv`
  - 供 `analysis_code/Big_plot.ipynb` 使用

- `Pure_RT/RT_*_NTHU.xlsx`
  - 供 `analysis_code/RT_Curve_analysis.ipynb` 使用

- `from_by_temperature/*.xlsx`
  - 從 `raw_data_NTHU` 切出的分溫資料（搬移備份）
  - 保留於 processed 區，避免污染 `data/` raw-only 原則

## 使用規則

- `analysis_code/` 僅讀取本資料夾（以及 `Processed_data` 其他 processed 檔）。
- 不要在 `analysis_code/` 直接讀 `data/`。
