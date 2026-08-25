# COPD 分類與分析系統

基於深度學習的 COPD（慢性阻塞性肺病）自動化分析系統，整合肺部分割、參數計算與神經網路分類功能。

## 📋 功能特色

- 🔬 **自動肺部分割** - 呼叫 3D Slicer 推論伺服器進行肺葉分割
- 📊 **COPD 參數計算** - 計算12項關鍵 COPD 生物標記
- 🧠 **神經網路分類** - 使用全連接神經網路進行正常/異常分類
- 📈 **5-Fold 交叉驗證** - 更可靠的模型評估方法
- 📄 **自動報告生成** - 產生 Markdown 格式分析報告
- 🌬️ **AeroPath 氣道分割** - 自動生成氣道標記檔案

---

## 🚀 快速開始

### 1. 安裝依賴

```bash
conda create -n copd python=3.10
conda activate copd
pip install -r requirements.txt
```

### 2. 訓練模型（5-Fold 交叉驗證）

```bash
python train_copd_model.py
```

### 3. 預測新樣本

```bash
# 單一檔案預測
python predict_copd.py --input-json path/to/sample_metrics.json

# 批量預測
python predict_copd.py --input-dir path/to/metrics_folder
```

### 4. 完整 Pipeline（分割 + 分析 + 分類）

```bash
# 單一 CT 影像分析
python unified_pipeline.py -i patient_ct.nii.gz -o output/

# 批次處理資料夾
python unified_pipeline.py -i ./data/ -o ./results/ --batch

# 只生成參數，不進行預測
python unified_pipeline.py -i ./data/ -o ./results/ --batch --no-predict
```

### 5. 生成視覺化圖表

```bash
python generate_viz.py -i NormalDataset -o NormalDataset/visualizations
python generate_viz.py -i AbnormalDataset -o AbnormalDataset/visualizations
```

---

## 📁 專案結構

```
COPDClassification/
├── 📄 unified_pipeline.py         # 統一 Pipeline（分割→分析→分類）
├── 📄 copd_classifier.py          # 核心模型：神經網路 + 5-Fold 訓練
├── 📄 copd_segmentation.py        # 肺部分割模組
├── 📄 train_copd_model.py         # 訓練腳本
├── 📄 predict_copd.py             # 預測腳本
├── 📄 generate_viz.py             # 視覺化圖表生成
├── 📄 requirements.txt            # Python 依賴
├── 📄 README.md                   # 本說明文件
│
├── 📂 VesselAirwayParamTransfer/  # COPD 參數計算模組
│   ├── copd_analyzer.py           # COPDAnalyzer 類別
│   └── adaptive_emphysema.py      # 自適應肺氣腫計算

├── 📂 NormalDataset/              # 正常資料集
│   ├── data/                      # 原始 CT 影像
│   ├── inference_output/          # 肺部分割結果
│   ├── param/                     # 參數檔案 (*_metrics.json)
│   ├── report/                    # 分析報告
│   └── airway/                    # 氣道分割結果 (可選)
│
├── 📂 AbnormalDataset/            # 異常資料集
│   └── (同上結構)
│
└── 📂 models/                     # 訓練輸出
    ├── fold_1_model.pth ~ fold_5_model.pth
    ├── fold_1_scaler.pkl ~ fold_5_scaler.pkl
    └── kfold_training_results.json
```

---

## 🔧 各腳本使用說明

### `unified_pipeline.py` - 完整分析 Pipeline

整合肺部分割、參數計算、分類預測的完整流程。

```bash
python unified_pipeline.py -i ct.nii.gz -o output/
python unified_pipeline.py -i ./data/ -o ./results/ --batch
python unified_pipeline.py -i ct.nii.gz --seg existing_seg.nii.gz -o output/
```

| 參數           | 說明                             |
| -------------- | -------------------------------- |
| `-i, --input`  | 輸入 CT 影像或資料夾路徑         |
| `-o, --output` | 輸出目錄                         |
| `--seg`        | 現有分割結果路徑（跳過分割步驟） |
| `--airway`     | 氣道分割結果路徑                 |
| `--batch`      | 批次處理模式                     |
| `--no-predict` | 只生成參數，跳過神經網路預測     |
| `--viz-only`   | 只生成視覺化圖表                 |

---

### `train_copd_model.py` - 模型訓練

```bash
python train_copd_model.py
python train_copd_model.py --epochs 150 --n-folds 5
```

| 參數              | 預設值     | 說明                  |
| ----------------- | ---------- | --------------------- |
| `--data-dir`      | `.`        | 資料集根目錄          |
| `--output-dir`    | `models`   | 輸出目錄              |
| `--n-folds`       | `5`        | 交叉驗證折數          |
| `--epochs`        | `100`      | 每個 Fold 訓練輪數    |
| `--batch-size`    | `16`       | Batch 大小            |
| `--learning-rate` | `0.001`    | 學習率                |
| `--hidden1/2/3`   | `64/32/16` | 隱藏層神經元數        |

---

### `predict_copd.py` - 預測

```bash
python predict_copd.py --input-json sample_metrics.json
python predict_copd.py --input-dir NormalDataset/param --output results.json
```

---

### `generate_viz.py` - 視覺化

```bash
python generate_viz.py -i NormalDataset -o NormalDataset/visualizations
```

生成的圖表：
- `heatmap_all.png` - 全參數熱力圖
- `heatmap_emphysema.png` - 肺氣腫專用熱力圖
- `feature_distributions.png` - 特徵分布箱型圖
- `correlation_heatmap.png` - 特徵相關性熱力圖
- `comparison_bar.png` - 核心指標比較圖

---

## 🧠 模型架構

```
Input (12 features)
      ↓
Dense(64) + BatchNorm + ReLU + Dropout(0.3)
      ↓
Dense(32) + BatchNorm + ReLU + Dropout(0.3)
      ↓
Dense(16) + BatchNorm + ReLU + Dropout(0.21)
      ↓
Dense(2) → Softmax
      ↓
Output (Normal / Abnormal)
```

### 輸入特徵（12個）

| #   | 特徵名稱                      | 說明             |
| --- | ----------------------------- | ---------------- |
| 1   | Total_Emphysema_Percent       | 總肺氣腫百分比   |
| 2   | Left_Superior_Lobe_Emphysema  | 左上葉肺氣腫     |
| 3   | Left_Inferior_Lobe_Emphysema  | 左下葉肺氣腫     |
| 4   | Right_Superior_Lobe_Emphysema | 右上葉肺氣腫     |
| 5   | Right_Middle_Lobe_Emphysema   | 右中葉肺氣腫     |
| 6   | Right_Inferior_Lobe_Emphysema | 右下葉肺氣腫     |
| 7   | SVV_Percent                   | 小血管體積百分比 |
| 8   | WA_Percent                    | 氣道壁面積百分比 |
| 9   | Vessel_Density_Percent        | 血管密度百分比   |
| 10  | Airway_Lung_Ratio_Percent     | 氣道肺比率       |
| 11  | Total_Lung_Volume_ml          | 總肺容積 (mL)    |
| 12  | PA_Diameter_mm                | 肺動脈直徑 (mm)  |

---

## ⚠️ 注意事項

1. **小資料集優化** - 系統針對小資料集設計，使用 Dropout、BatchNorm、Early Stopping 等防止過擬合
2. **伺服器依賴** - `unified_pipeline.py` 需要 3D Slicer 推論伺服器進行分割
3. **特徵完整性** - 確保所有 12 個特徵都存在於 JSON 檔案中
4. **GPU 支援** - 自動偵測 CUDA，可使用 `--no-cuda` 強制使用 CPU

---

## 📝 License

此專案用於 COPD 研究與醫學分析。

## 👨‍💻 作者

Felix - COPD Classification Research Project
