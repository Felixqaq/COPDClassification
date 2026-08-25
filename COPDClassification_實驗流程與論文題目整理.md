# COPDClassification 實驗流程與論文題目整理

整理日期：2026-05-07

這份文件是把 `COPDClassification` 目前做過的實驗、量化流程、模型訓練結果、論文可用素材與題目發想整理在一起。整體研究的核心不是複雜神經網路，而是「從胸部 CT 影像抽出具臨床意義的量化特徵，再用輕量分類器判斷 Normal / Abnormal」。

---

## 1. 研究定位

### 核心想法

本研究比較適合定位為：

> 基於胸部 CT 影像定量生物標記的 COPD 輔助分類與自動化分析系統。

系統把 CT 影像轉成可解釋的數值特徵，例如肺氣腫比例、血管密度、氣道壁比例、肺動脈直徑，再用一個簡單的全連接神經網路做二元分類。神經網路本身不是主要貢獻，真正的重點是：

- 自動化處理流程完整：CT -> 分割 -> 量化參數 -> 分類 -> 報告。
- 特徵有臨床意義：每個輸入特徵都可以回到 COPD 病理特徵解釋。
- 適合小資料集：54 筆資料用 5-fold cross-validation 評估。
- 可解釋性高：模型不是直接吃原始影像，而是吃 12 個量化特徵。

### 建議避免的說法

- 不建議主打「深度學習模型創新」，因為模型是簡單 FCNN。
- 不建議直接說「GOLD 分級分類」，除非後續取得肺功能資料，例如 FEV1/FVC、FEV1 predicted。
- 不建議只說「COPD severity classification」，因為目前標籤是 Normal / Abnormal，不是 GOLD I-IV。

---

## 2. 專案與資料位置

### 主要程式

| 檔案 | 用途 |
|---|---|
| `unified_pipeline.py` | 完整 pipeline：分割、量化、分類、報告 |
| `copd_segmentation.py` | 呼叫 3D Slicer inference server 做肺部分割 |
| `VesselAirwayParamTransfer/copd_analyzer.py` | 核心量化參數計算 |
| `VesselAirwayParamTransfer/adaptive_emphysema.py` | 多種肺氣腫閾值與自適應方法實驗 |
| `copd_classifier.py` | 特徵擷取、FCNN、5-fold 訓練與評估 |
| `train_copd_model.py` | 訓練入口 |
| `predict_copd.py` | 單筆或批次預測 |
| `generate_viz.py` | 量化特徵視覺化 |

### 資料夾

| 資料夾 | 內容 |
|---|---|
| `NormalDataset/data` | 正常 CT NIfTI |
| `NormalDataset/inference_output` | 正常資料的肺部分割結果 |
| `NormalDataset/param` | 正常資料的量化 JSON |
| `NormalDataset/report` | 正常資料的 Markdown 報告 |
| `AbnormalDataset/data` | 異常 CT NIfTI |
| `AbnormalDataset/inference_output` | 異常資料的肺部分割結果 |
| `AbnormalDataset/param` | 異常資料的量化 JSON |
| `AbnormalDataset/report` | 異常資料的 Markdown 報告 |
| `models/training_*` | 各次 5-fold 訓練輸出 |
| `NTUT-Thesis-Template` | 已寫好的論文章節草稿 |

---

## 3. 目前資料集狀態

### 目前實際用於訓練的資料

從 `NormalDataset/param` 與 `AbnormalDataset/param` 計算，目前模型使用：

| 類別 | 筆數 | 佔比 |
|---|---:|---:|
| Normal | 21 | 38.89% |
| Abnormal | 33 | 61.11% |
| 合計 | 54 | 100% |

### 影像命名與掃描協議粗略分布

依檔名粗略分類：

| 類別 | Thorax Lung | LW/LUNG AXI | Other Thorax/Chest/Aorta |
|---|---:|---:|---:|
| Normal | 5 | 12 | 4 |
| Abnormal | 12 | 15 | 6 |

這代表目前資料不是單一掃描 protocol，論文需要討論 protocol variation 可能造成的 domain shift。

### 舊簡報與目前資料的差異

`Picture/Quantitative Analysis & Classification of COPD Severity via Chest CT.pdf` 裡面記錄過一版實驗：

- Abnormal：振興醫院 33 個病人的胸部 DICOM。
- Normal：MosMedData Chest CT Scans with COVID-19 的 CT-0，共 100 張，取 33 張使用。
- 當時簡報提到分類準確率達 100%，但也註記「氣道、肺葉、血管切割效果其實蠻爛」。

目前論文草稿與訓練結果改成 54 筆，Normal 21、Abnormal 33。這裡要注意：如果最後論文要寫，必須確認最終資料版本到底是哪一版，不能混用舊簡報與目前模型結果。

---

## 4. 完整實驗流程

### Stage 0：資料格式準備

目前 pipeline 期待輸入是 NIfTI：

- `.nii`
- `.nii.gz`

若原始資料是 DICOM，需要先在其他流程轉成 NIfTI，例如專案外層的 `DicomToNii`。

### Stage 1：肺部自動分割

程式：`copd_segmentation.py`

流程：

1. 讀入 NIfTI CT。
2. 用 SimpleITK 將 NIfTI 轉成 NRRD。
3. 透過 HTTP POST 呼叫 3D Slicer inference server。
4. 使用模型 `lungs-v2.0.1`。
5. 取得 NRRD 分割結果。
6. 再轉回 NIfTI。

預設伺服器（實際位址設定於 server_config.py，見 server_config.example.py）：

```text
http://YOUR_SEG_SERVER:8891
```

分割標籤定義：

| Label | 解剖結構 | 用途 |
|---:|---|---|
| 1 | 左肺上葉 | 肺體積、肺氣腫 |
| 2 | 左肺下葉 | 肺體積、肺氣腫 |
| 3 | 右肺上葉 | 肺體積、肺氣腫 |
| 4 | 右肺中葉 | 肺體積、肺氣腫 |
| 5 | 右肺下葉 | 肺體積、肺氣腫 |
| 6 | 血管 | SVV% |
| 7 | 氣管 | WA% 備用來源 |
| 8 | 肺靜脈系統 | 血管密度 |
| 9 | 肺動脈 | PA diameter |

### Stage 2：氣道分割

pipeline 支援 AeroPath 氣道分割：

- 先搜尋 `airway/` 中是否有對應的 `_airway_seg.nii.gz`。
- 如果找不到，`unified_pipeline.py` 有函式可呼叫 AeroPath Gradio API。
- 但目前 54 筆既有 `param` JSON 中，WA% 全部都是使用 `Trachea Label`，也就是沒有用到專用 airway segmentation。

這點很重要：如果論文要強調 AeroPath 對 WA% 的貢獻，最好重新跑一版有專用 airway mask 的參數，或把目前 WA% 說成基於 Label 7 的近似估算。

### Stage 3：COPD 量化參數計算

程式：`VesselAirwayParamTransfer/copd_analyzer.py`

基本體積：

```text
Voxel volume = spacing_x * spacing_y * spacing_z
Structure volume (mm3) = voxel_count * voxel_volume
Structure volume (mL) = volume_mm3 / 1000
```

量化指標分成四類：

- 肺氣腫：低 HU 區域比例。
- 氣道：WA%。
- 血管：SVV%、Vessel Density%。
- 肺動脈與肺容積：PA diameter、Total Lung Volume。

### Stage 4：12 個分類特徵擷取

程式：`copd_classifier.py` 的 `extract_features_from_json`

| # | 特徵名稱 | 說明 | 單位 |
|---:|---|---|---|
| 1 | Total_Emphysema_Percent | 全肺肺氣腫比例 | % |
| 2 | Left_Superior_Lobe_Emphysema | 左上葉肺氣腫比例 | % |
| 3 | Left_Inferior_Lobe_Emphysema | 左下葉肺氣腫比例 | % |
| 4 | Right_Superior_Lobe_Emphysema | 右上葉肺氣腫比例 | % |
| 5 | Right_Middle_Lobe_Emphysema | 右中葉肺氣腫比例 | % |
| 6 | Right_Inferior_Lobe_Emphysema | 右下葉肺氣腫比例 | % |
| 7 | SVV_Percent | 小血管體積佔總肺體積比例 | % |
| 8 | WA_Percent | 氣道壁體積佔總氣道體積比例 | % |
| 9 | Vessel_Density_Percent | 總血管體積佔總肺體積比例 | % |
| 10 | Airway_Lung_Ratio_Percent | 氣道體積佔總肺體積比例 | % |
| 11 | Total_Lung_Volume_ml | 五個肺葉總體積 | mL |
| 12 | PA_Diameter_mm | 肺動脈等效直徑 | mm |

---

## 5. 各量化指標細節

### 5.1 肺氣腫百分比

目前既有 54 筆 JSON 使用的是舊格式，儲存：

- 全肺 `total_emphysema_percent`
- 五個肺葉各自的 `emphysema_percent`
- threshold 是 `HU < -950`

公式：

```text
Emphysema% = emphysema_voxels / total_lung_voxels * 100
emphysema_voxels = count(CT voxel HU < -950 and voxel inside lung/lobe mask)
```

現行 `copd_analyzer.py` 已經支援新版多閾值分級：

| 分級 | 閾值 |
|---|---|
| Mild | HU < -950 |
| Moderate | HU < -960 |
| Severe | HU < -970 |

但目前 54 筆 JSON：

```text
with_grading = 0
without_grading = 54
```

所以如果要在論文中寫「多閾值分級」，建議重跑參數，讓 JSON 真正包含 `grading` 欄位；否則目前分類模型實際用的是 `HU < -950` 的肺氣腫特徵。

### 5.2 WA% 氣道壁面積百分比

公式：

```text
WA% = wall_volume / total_airway_volume * 100
wall_mask = airway_mask - lumen_mask
lumen_mask = binary_erosion(airway_mask, iterations=2)
```

目前 WA% 來源：

```text
NormalDataset: 21 / 21 使用 Trachea Label
AbnormalDataset: 33 / 33 使用 Trachea Label
```

臨床解釋在程式報告中使用：

| 條件 | 解釋 |
|---|---|
| WA% > 70% | 氣道重塑風險 |
| WA% > 67% | 略高，建議追蹤 |
| WA% <= 67% | 正常範圍 |

限制：

- 這是用形態學侵蝕估算 lumen，不是真正的管腔標註。
- 目前使用 Label 7 trachea 當 airway，對小氣道或支氣管壁厚的代表性有限。

### 5.3 SVV% 小血管體積百分比

公式：

```text
SVV% = small_vessel_volume / total_lung_volume * 100
small_vessel = vessel voxels with estimated radius < 2.5 mm
diameter < 5 mm -> radius < 2.5 mm
```

方法：

- 使用 Label 6 當血管遮罩。
- 使用 `scipy.ndimage.distance_transform_edt` 估算每個血管體素到邊界距離。
- 小血管定義為距離圖小於 2.5 mm 的血管區域。

程式報告中的臨床解釋：

| 條件 | 解釋 |
|---|---|
| SVV% < 5% | 小血管密度減少、血管修剪現象 |
| SVV% >= 5% | 正常範圍 |

限制：

- 依賴血管分割品質。
- distance transform 是半徑近似，遇到破碎或黏連血管會偏差。

### 5.4 Vessel Density%

公式：

```text
Vessel_Density% = total_vessel_volume / total_lung_volume * 100
total_vessel_volume = Label 6 + Label 8 + Label 9
```

用途：

- 表示肺部血管結構整體密度。
- 可與 SVV% 搭配討論小血管與總血管變化。

### 5.5 Airway Lung Ratio%

公式：

```text
Airway_Lung_Ratio% = airway_volume / total_lung_volume * 100
```

目前 airway volume 來源：

- 優先使用 WA% 中的 `total_volume_mm3`。
- 若無 WA%，使用 Label 7 trachea volume。

限制：

- 目前全資料都是 Trachea Label，較像大氣道/氣管體積比，不是完整 airway-tree ratio。

### 5.6 Total Lung Volume

公式：

```text
Total_Lung_Volume_ml = sum(Label 1-5 volume_mm3) / 1000
```

觀察：

- 目前 Abnormal 平均總肺體積比 Normal 大，可能反映 COPD hyperinflation，但也可能混入掃描 protocol、吸氣程度或資料來源差異。

### 5.7 PA Diameter

公式：

```text
radius = cubic_root(3 * PA_volume / (4 * pi))
PA_Diameter_mm = 2 * radius
```

目前限制：

- 沒有主動脈分割，所以 `PA_A_Ratio = null`。
- 肺動脈直徑是等效球體估計，非真正血管截面直徑。
- 如果要寫肺動脈高壓或急性惡化風險，最好補主動脈 segmentation 或手動量 PA:A ratio。

---

## 6. 分類模型

### 模型架構

模型是全連接神經網路：

```text
Input: 12 features
Dense(12 -> 64) + BatchNorm + ReLU + Dropout(0.3)
Dense(64 -> 32) + BatchNorm + ReLU + Dropout(0.3)
Dense(32 -> 16) + BatchNorm + ReLU + Dropout(0.21)
Dense(16 -> 2)
Softmax for probability
Output: Normal / Abnormal
```

### 訓練設定

| 項目 | 設定 |
|---|---|
| 標籤 | 0 = Normal, 1 = Abnormal |
| 樣本數 | 54 |
| 驗證方法 | Stratified 5-Fold Cross-Validation |
| 每 fold 訓練集 | 43 或 44 |
| 每 fold 驗證集 | 10 或 11 |
| 特徵標準化 | StandardScaler，每個 fold 只用訓練集 fit |
| Loss | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 1e-5 |
| Epochs | 100 |
| Batch size | 16 |
| LR scheduler | ReduceLROnPlateau，factor=0.5，patience=10 |
| Early stopping | patience=15 |
| Seed | 42 |

### 預測輸出

`predict_copd.py` 可以輸出：

- predicted_class
- predicted_label
- probability_normal
- probability_abnormal

批次預測會輸出 JSON。

---

## 7. 已完成訓練結果

### 訓練輸出版本

目前 `models` 中有三次主要訓練：

| 資料夾 | 時間 | 說明 |
|---|---|---|
| `training_20260128_140828` | 2026-01-28 | 較早版本，整體 accuracy 0.8889 |
| `training_20260130_182130` | 2026-01-30 | 最終結果之一 |
| `training_20260206_132335` | 2026-02-06 | 與 2026-01-30 結果一致，較新的輸出 |

### 最新結果：`training_20260206_132335`

各 fold 結果：

| Fold | Train | Val | Accuracy | Precision | Recall | F1 | AUC | FN | FP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 43 | 11 | 0.9091 | 1.0000 | 0.8333 | 0.9091 | 0.9000 | 1 | 0 |
| 2 | 43 | 11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 3 | 43 | 11 | 0.8182 | 1.0000 | 0.7143 | 0.8333 | 0.9286 | 2 | 0 |
| 4 | 43 | 11 | 0.9091 | 1.0000 | 0.8571 | 0.9231 | 1.0000 | 1 | 0 |
| 5 | 44 | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |

平均與整體結果：

| 指標 | 平均 +- SD | 整體值 |
|---|---:|---:|
| Accuracy | 0.9273 +- 0.0680 | 0.9259 |
| Precision | 1.0000 +- 0.0000 | 1.0000 |
| Recall | 0.8810 +- 0.1086 | 0.8788 |
| F1-score | 0.9331 +- 0.0626 | 0.9355 |
| AUC | 0.9657 +- 0.0430 | 0.9784 |

整體混淆矩陣：

| Actual / Predicted | Normal | Abnormal |
|---|---:|---:|
| Normal | 21 | 0 |
| Abnormal | 4 | 29 |

解讀：

- FP = 0，模型沒有把正常樣本誤判為異常。
- FN = 4，模型漏掉 4 筆異常樣本。
- Precision 很漂亮，但 Recall 還有改善空間。
- 對臨床輔助而言，漏診比誤診更需要謹慎討論。

### 假陰性樣本

| Fold | Sample | True | Predicted | Abnormal probability |
|---:|---|---|---|---:|
| 1 | `4796667_Thorax Lung Br60 S2 3.00` | Abnormal | Normal | 0.131 |
| 3 | `P02_Aorta C+  5.0  B30f` | Abnormal | Normal | 0.461 |
| 3 | `C543831_Thorax Lung Br60 S2 3.00` | Abnormal | Normal | 0.298 |
| 4 | `P11_Chest C-  5.0  B31f` | Abnormal | Normal | 0.489 |

可討論點：

- `P02_Aorta C+` 是 Aorta contrast protocol，和一般 lung protocol 不同，可能影響 CT HU 與 segmentation。
- `P11` 與 `P02` 的 abnormal probability 接近 0.5，屬於決策邊界附近。
- 這些可能是輕度異常、影像 protocol 造成偏差，或量化特徵不夠敏感。

---

## 8. 12 個特徵的量化統計

以下是從目前 54 個 `*_metrics.json` 重新計算的特徵統計。AUC 是單一特徵區分 Normal / Abnormal 的 pairwise AUC，方向已校正，所以越接近 1 代表單特徵越能分開兩類。

| 特徵 | Normal mean +- SD | Abnormal mean +- SD | Abnormal - Normal | Cohen d | 單特徵 AUC |
|---|---:|---:|---:|---:|---:|
| Total_Emphysema_Percent | 8.18 +- 4.62 | 25.99 +- 12.97 | 17.81 | 1.68 | 0.887 |
| Left_Superior_Lobe_Emphysema | 8.07 +- 5.21 | 26.91 +- 14.58 | 18.84 | 1.59 | 0.887 |
| Left_Inferior_Lobe_Emphysema | 10.80 +- 7.36 | 26.63 +- 13.59 | 15.83 | 1.37 | 0.841 |
| Right_Superior_Lobe_Emphysema | 7.46 +- 4.62 | 27.96 +- 15.31 | 20.50 | 1.66 | 0.903 |
| Right_Middle_Lobe_Emphysema | 9.76 +- 5.72 | 24.29 +- 10.73 | 14.53 | 1.59 | 0.869 |
| Right_Inferior_Lobe_Emphysema | 8.52 +- 4.73 | 24.41 +- 12.96 | 15.89 | 1.50 | 0.848 |
| SVV_Percent | 5.77 +- 1.31 | 5.95 +- 1.52 | 0.19 | 0.13 | 0.522 |
| WA_Percent | 63.69 +- 7.33 | 67.26 +- 8.33 | 3.57 | 0.45 | 0.625 |
| Vessel_Density_Percent | 6.53 +- 1.16 | 6.93 +- 1.94 | 0.40 | 0.24 | 0.517 |
| Airway_Lung_Ratio_Percent | 0.65 +- 0.19 | 0.60 +- 0.22 | -0.05 | -0.24 | 0.561 |
| Total_Lung_Volume_ml | 3854.53 +- 1038.06 | 5251.51 +- 909.89 | 1396.99 | 1.45 | 0.841 |
| PA_Diameter_mm | 30.37 +- 13.25 | 36.88 +- 13.51 | 6.51 | 0.49 | 0.667 |

### 重要觀察

目前最有分類能力的單一特徵：

| 排名 | 特徵 | AUC | 方向 |
|---:|---|---:|---|
| 1 | Right_Superior_Lobe_Emphysema | 0.903 | Abnormal 較高 |
| 2 | Total_Emphysema_Percent | 0.887 | Abnormal 較高 |
| 3 | Left_Superior_Lobe_Emphysema | 0.887 | Abnormal 較高 |
| 4 | Right_Middle_Lobe_Emphysema | 0.869 | Abnormal 較高 |
| 5 | Right_Inferior_Lobe_Emphysema | 0.848 | Abnormal 較高 |
| 6 | Total_Lung_Volume_ml | 0.841 | Abnormal 較高 |

比較弱的特徵：

- SVV%：AUC 0.522。
- Vessel Density%：AUC 0.517。
- Airway Lung Ratio%：AUC 0.561。
- WA%：AUC 0.625。

這代表目前模型很可能主要靠肺氣腫比例與總肺體積在判斷，氣道與血管特徵的貢獻不明顯。這不一定是壞事，但論文要誠實寫：

- 量化特徵中，肺氣腫與肺容積對分類最有貢獻。
- 氣道與血管特徵可能受限於 segmentation 品質、Label 7 近似 airway、以及資料量不足。

---

## 9. 已產出的論文素材

### 已有 thesis 草稿

`NTUT-Thesis-Template` 內已經有完整初稿：

| 章節 | 狀態 |
|---|---|
| Chapter 1 緒論 | 已寫 COPD 背景、動機、研究目的 |
| Chapter 2 相關研究 | 已整理 COPD、CT 量化、分割、FCNN、cross-validation |
| Chapter 3 方法 | 已寫三階段 pipeline、公式、模型架構 |
| Chapter 4 實驗 | 已寫 54 筆資料、5-fold 結果、FN 分析 |
| Chapter 5 結論 | 已寫貢獻、限制、未來方向 |

### 已有圖表

訓練圖表：

- `models/training_20260206_132335/kfold_metrics_comparison.png`
- `models/training_20260206_132335/kfold_confusion_matrix.png`
- `models/training_20260206_132335/kfold_roc_curve.png`
- `models/training_20260206_132335/kfold_metrics_boxplot.png`

錯誤分類分析圖：

- `models/fn_samples_comparison.png`
- `models/fn_samples_radar.png`

資料視覺化腳本可生成：

- `heatmap_all.png`
- `heatmap_emphysema.png`
- `feature_distributions.png`
- `correlation_heatmap.png`
- `comparison_bar.png`

### 參考文獻主題

`NTUT-Thesis-Template/reference.bib` 已包含：

- GOLD COPD 指引。
- WHO COPD fact sheet。
- 肺氣腫 CT threshold：Gevenois、Madani。
- CT 量化 COPD：Matsuoka。
- 小血管 CT 指標：Matsuoka。
- 氣道壁與肺功能：Nakano。
- PA enlargement：Wells。
- 深度學習與醫學影像：LeCun、Litjens。
- U-Net、TotalSegmentator、3D Slicer、AeroPath。
- PyTorch、scikit-learn、Adam、K-Fold、Dropout、BatchNorm。

可以補強：

- GOLD 2026 報告，尤其是你根目錄已有 `GOLD分級報告.pdf`。
- 更近年的 imaging biomarker / radiomics COPD 文獻。
- 可解釋 AI、small tabular medical data、feature attribution 的文獻。

---

## 10. GOLD 與肺功能資料的關係

GOLD 診斷 COPD 主要需要肺功能：

```text
post-bronchodilator FEV1/FVC < 0.7
```

GOLD airflow limitation severity 常用：

| GOLD grade | FEV1 predicted |
|---|---|
| GOLD 1 | FEV1 >= 80% |
| GOLD 2 | 50% <= FEV1 < 80% |
| GOLD 3 | 30% <= FEV1 < 50% |
| GOLD 4 | FEV1 < 30% |

另外臨床治療分類還會考慮：

- 症狀量表，例如 CAT、mMRC。
- 急性惡化次數與住院紀錄。
- 現行 GOLD 2026 對 A/B/E 分類與 exacerbation risk 有更新。

目前專案沒有 PFT 欄位，所以現在只能做：

- CT-based Normal / Abnormal classification。
- CT quantitative COPD-related abnormality detection。

若要升級成 GOLD 分期題目，需要新增：

- FEV1/FVC。
- FEV1 predicted。
- 支氣管擴張劑後肺功能。
- 症狀量表與 exacerbation history。

---

## 11. 可寫成論文的主題方向

### 題目方向 A：可解釋 CT 定量特徵分類

建議題目：

> 基於胸部 CT 定量生物標記之 COPD 正常與異常自動分類系統

英文：

> COPD Normal-Abnormal Classification Using Quantitative Chest CT Biomarkers

優點：

- 最貼近目前已完成實驗。
- 不誇大神經網路。
- 可以主打可解釋性、量化、生物標記、自動化。

適合程度：最高。

### 題目方向 B：自動化 CT 量化分析 Pipeline

建議題目：

> 整合肺部分割與定量特徵擷取之 COPD 胸部 CT 自動化分析流程

英文：

> An Automated Chest CT Quantitative Analysis Pipeline for COPD Assessment

優點：

- 可以把系統工程完整性當貢獻。
- 分類只是其中一個應用。
- 即使模型不複雜也合理。

適合程度：很高。

### 題目方向 C：肺氣腫與肺容積特徵主導的分類分析

建議題目：

> 胸部 CT 肺氣腫與肺容積量化特徵於 COPD 異常辨識之研究

優點：

- 目前統計結果顯示肺氣腫與肺容積最有用。
- 可以深入討論 feature importance，不需依賴複雜模型。

限制：

- 會弱化 WA%、SVV%、PA 等其他特徵。

適合程度：高，但題目較窄。

### 題目方向 D：量化特徵模型 vs 3D CNN

建議題目：

> 基於 CT 定量生物標記與三維影像模型之 COPD 分類比較研究

優點：

- 呼應舊簡報中老師提到「下一步想用 3D 神經網路模型直接分類病人」。
- 可以做一個有趣對照：可解釋特徵模型 vs end-to-end 3D CNN。

限制：

- 需要額外做 3D model。
- 54 筆資料對 3D CNN 很小，可能需要 transfer learning 或 patch-based 方法。

適合程度：中高，工作量較大。

### 題目方向 E：GOLD 分級輔助預測

建議題目：

> 結合胸部 CT 定量特徵與肺功能資料之 COPD GOLD 分級預測

優點：

- 臨床價值更直接。
- 題目比較像醫學應用論文。

限制：

- 目前缺 PFT / GOLD label。
- 如果拿不到 FEV1/FVC 與 FEV1 predicted，不建議選這題。

適合程度：取決於是否能拿到肺功能資料。

---

## 12. 最推薦的論文題目

如果以現在已經做好的東西為主，推薦：

> 基於胸部 CT 定量生物標記之 COPD 自動化分析與正常/異常分類系統

英文：

> Automated COPD Analysis and Normal-Abnormal Classification Using Quantitative Chest CT Biomarkers

這個題目涵蓋：

- 自動化 pipeline。
- CT 定量。
- COPD 生物標記。
- Normal / Abnormal 分類。
- 不會假裝自己已經做 GOLD stage。
- 不會過度強調神經網路模型創新。

備用較短版本：

> 基於胸部 CT 量化特徵之 COPD 異常辨識研究

---

## 13. 論文可以主打的貢獻

### 貢獻 1：完整自動化流程

從 NIfTI CT 輸入到 Markdown 報告輸出：

```text
CT -> lung segmentation -> quantitative biomarkers -> FCNN classification -> report
```

### 貢獻 2：可解釋的 12 維量化特徵

不是 end-to-end black box，而是使用臨床可理解的 CT biomarker。

### 貢獻 3：小資料集下的穩定評估

使用 stratified 5-fold，並保留：

- 每 fold 模型。
- 每 fold scaler。
- 每 fold metrics。
- FN/FP 錯誤樣本。

### 貢獻 4：錯誤分類樣本分析

有列出 4 筆 FN，可討論 protocol、輕度病灶、特徵不足與 decision boundary。

### 貢獻 5：量化特徵重要性觀察

目前統計上肺氣腫與肺容積最能分開兩類，這可以成為論文討論的核心結果之一。

---

## 14. 論文風險與限制

### 14.1 資料量小

54 筆資料雖然可以做初步研究，但不能說明強泛化能力。需要明確寫：

- pilot study。
- retrospective small-scale study。
- 需要外部驗證。

### 14.2 Normal / Abnormal 標籤定義需要釐清

論文必須說清楚：

- Normal 是由誰標註？
- Abnormal 是 COPD 嗎？還是包含其他肺部異常？
- 有沒有醫師診斷、病歷、肺功能？

如果標籤只是資料夾分類，論文會比較薄弱。

### 14.3 不能等同 GOLD 分期

沒有肺功能資料就不能說 GOLD I-IV。

可以說：

- CT-based abnormality classification。
- COPD-related quantitative abnormality detection。

### 14.4 分割品質會影響所有特徵

舊簡報已經提到 airway、lobe、vessel segmentation 效果不理想。這會影響：

- WA%
- SVV%
- Vessel Density%
- Lobe emphysema%
- PA diameter

### 14.5 WA% 目前不是完整氣道樹

既有資料全部用 Trachea Label 估算 WA%，所以不宜過度宣稱能精確評估小氣道重塑。

### 14.6 模型可能受資料來源或 protocol bias 影響

如果 Normal 與 Abnormal 來自不同來源或 protocol，模型可能學到：

- 掃描 protocol 差異。
- reconstruction kernel 差異。
- 吸氣程度差異。
- 資料來源差異。

而不是 COPD 本身。

---

## 15. 建議補做的實驗

### 最值得補的實驗

1. 特徵消融實驗。
   - 只用肺氣腫 6 特徵。
   - 只用氣道/血管特徵。
   - 去掉總肺體積。
   - 全部 12 特徵。

2. 傳統機器學習 baseline。
   - Logistic Regression。
   - SVM。
   - Random Forest。
   - XGBoost 或 LightGBM。

3. 特徵重要性。
   - Permutation importance。
   - SHAP。
   - 單特徵 AUC。

4. 閾值式分類 baseline。
   - Total Emphysema% alone。
   - Right upper lobe emphysema alone。
   - Lung volume alone。
   - 比較 rule-based vs FCNN。

5. 重新跑多閾值肺氣腫。
   - 讓 JSON 實際包含 HU < -950/-960/-970。
   - 比較哪個 threshold 對分類最有幫助。

6. 重新跑專用 airway segmentation。
   - 用 AeroPath 產生 airway mask。
   - 比較 Trachea Label WA% vs AeroPath WA%。

7. FN case study。
   - 把 4 筆 FN 的 12 維特徵與 Normal / Abnormal 平均值比較。
   - 做 radar chart、bar chart。
   - 看是不是偏輕微異常或 protocol 問題。

### 如果時間更多

8. 外部驗證。
   - 新醫院資料。
   - 不同 scanner。
   - 不同 reconstruction kernel。

9. 加 PFT。
   - FEV1/FVC。
   - FEV1 predicted。
   - GOLD grade。

10. 影像模型比較。
   - 3D CNN / 3D ResNet / nnMamba。
   - 但要小心資料量太少。

---

## 16. 建議論文表格與圖

### 表格

| 表格 | 內容 |
|---|---|
| Table 1 | 資料集組成：Normal 21、Abnormal 33 |
| Table 2 | 分割標籤定義 Label 1-9 |
| Table 3 | 12 個量化特徵與公式 |
| Table 4 | FCNN 架構 |
| Table 5 | 訓練超參數 |
| Table 6 | 各 fold 結果 |
| Table 7 | 總體 confusion matrix |
| Table 8 | FN 樣本列表 |
| Table 9 | 特徵統計與單特徵 AUC |

### 圖

| 圖 | 內容 |
|---|---|
| Figure 1 | Pipeline overview |
| Figure 2 | 分割標籤示意圖 |
| Figure 3 | 量化特徵計算流程 |
| Figure 4 | FCNN 架構 |
| Figure 5 | 5-fold metrics comparison |
| Figure 6 | Confusion matrix |
| Figure 7 | ROC curve |
| Figure 8 | Feature distribution boxplot |
| Figure 9 | Feature correlation heatmap |
| Figure 10 | FN samples radar / comparison |

---

## 17. 一句話摘要

可以放在論文或簡報開頭：

> 本研究提出一套胸部 CT COPD 自動化定量分析流程，透過肺部結構分割擷取 12 項具臨床意義之影像生物標記，並以輕量全連接神經網路完成 Normal / Abnormal 分類；在 54 筆 CT 資料上以五折交叉驗證取得 92.59% 整體準確率與 0.978 AUC，顯示 CT 定量特徵可作為 COPD 輔助辨識之有效依據。

---

## 18. 下一步決策清單

在正式定題前，建議先回答這幾個問題：

- 最終 Normal 資料到底是 21 筆臨床 CT，還是 MosMed CT-0 取樣？
- Abnormal 是否都是 COPD？有沒有其他肺病混入？
- 有沒有肺功能資料可以取得？
- 要不要重跑多閾值肺氣腫，讓現有 JSON 與論文方法一致？
- 要不要重跑 AeroPath airway，讓 WA% 更合理？
- 論文要主打「自動化系統」還是「量化特徵分類」？

我的建議是：先以「CT 定量生物標記 + 可解釋分類」定題，再用補充實驗把 feature importance、baseline、ablation 做起來。這樣題目會穩，而且不需要假裝模型很深。
