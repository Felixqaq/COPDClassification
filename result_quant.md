# RQ 1/2/3 — 量化方法 (12 維特徵 + FCNN) 結果

本表把「這個專案的量化方法」(12 維 COPD 定量生物標記 → FCNN) 套用到 RQ1/2/3,
與 [result.md](result.md) 的兩個深度影像模型 (Mamba+Attention / TAP-CT fusion) 形成對照。

由 `train_rq_quant.py` 產生 (2026-07-15 18:03).

## 方法

- **特徵**：12 維影像量化特徵 (肺氣腫% ×6、SVV%、WA%、血管密度%、氣道/肺比%、肺體積、PA 直徑),與原專案分類器完全相同。
- **模型**：原專案 FCNN (`COPDClassifier`, 12→64→32→16→out)。分類 out=類別數;迴歸 out=1。
- **標籤**：RQ1 臨床資料夾;RQ2 PFT 塌陷角 (`patient_angles_simple.json`);RQ3 `oi_processed.json` (oi≥3)。
- **協定 (對齊 result.md)**：seed=42、5-fold stratified、100 epochs、
  **early stopping 關閉**(固定預算、最後 epoch 評估,避免用 test fold 選停點)、分類用 class-weighted CE、
  per-fold StandardScaler、迴歸 target zscore、決策 argmax 不調閾值。
- **兩種 WA% 特徵來源**：`param` = Trachea Label WA%;`param_aeropath` = AeroPath 氣道樹 WA%。

## 世代 (量化方法可用標籤)

| RQ | 任務 | 型態 | n | 說明 |
|----|------|------|---|------|
| RQ1 | normal_v_abnormal | 二分類 | 66 | 臨床 Normal(21+12 AllExtra)/Abnormal(33),33/33 |
| RQ2a | angle_3class | 三分類 | 66 | 塌陷角 ≤131/132–151/≥152 |
| RQ2b | angle_binary_extreme | 二分類 | 61 | 排除 132–151 灰區 (14 低角 / 47 高角) |
| RQ2c | angle | 迴歸 | 66 | 塌陷角角度值 (105–179°) |
| RQ3 | oi_emphysema | 二分類 | 66 | oi≥3 → 氣腫 (32/34) |

> 註:66 位病人**全部**都有 12 維特徵、臨床標籤、OI 與 PFT 塌陷角量測
> (塌陷角分兩批:原始 54 + 20260421 後補 12)。故 RQ1/RQ2a/RQ2c/RQ3 皆為 66、
> RQ2b 排除 132–151 灰區後 61 —— **與上方 result.md 深度模型的世代完全一致,可逐格直接比較**。

## 結果 — 特徵來源:`param` (Trachea WA%)

### 分類任務 (Acc / Sensitivity / Specificity 為 mean±std over folds;多分類 Sens/Spec 為 macro)

| RQ | 任務 | Accuracy | Sensitivity | Specificity | Macro-F1 | AUC |
|----|------|----------|-------------|-------------|----------|-----|
| RQ1 | normal_v_abnormal | 0.847±0.120 | 0.829±0.210 | 0.876±0.063 | 0.845 | 0.893 |
| RQ2a | angle_3class | 0.549±0.141 | 0.392±0.081 | 0.736±0.044 | 0.365 | — |
| RQ2b | angle_binary_extreme | 0.788±0.078 | 0.767±0.200 | 0.789±0.060 | 0.734 | 0.871 |
| RQ3 | oi_emphysema | 0.771±0.071 | 0.729±0.183 | 0.810±0.189 | 0.762 | 0.802 |

### 迴歸任務 (RQ2c 塌陷角角度值)

| RQ | 任務 | MAE (°) | RMSE (°) | R² | Pearson r |
|----|------|---------|----------|-----|-----------|
| RQ2c | angle | 16.10±1.84 | 20.22 | -0.036 | 0.503 |

## 結果 — 特徵來源:`param_aeropath` (AeroPath WA%)

### 分類任務 (Acc / Sensitivity / Specificity 為 mean±std over folds;多分類 Sens/Spec 為 macro)

| RQ | 任務 | Accuracy | Sensitivity | Specificity | Macro-F1 | AUC |
|----|------|----------|-------------|-------------|----------|-----|
| RQ1 | normal_v_abnormal | 0.848±0.084 | 0.800±0.214 | 0.910±0.074 | 0.844 | 0.893 |
| RQ2a | angle_3class | 0.577±0.119 | 0.458±0.063 | 0.768±0.060 | 0.412 | — |
| RQ2b | angle_binary_extreme | 0.755±0.087 | 0.767±0.200 | 0.747±0.080 | 0.705 | 0.789 |
| RQ3 | oi_emphysema | 0.804±0.054 | 0.733±0.065 | 0.876±0.110 | 0.802 | 0.825 |

### 迴歸任務 (RQ2c 塌陷角角度值)

| RQ | 任務 | MAE (°) | RMSE (°) | R² | Pearson r |
|----|------|---------|----------|-----|-----------|
| RQ2c | angle | 16.56±1.96 | 20.63 | 0.054 | 0.467 |

## 與 result.md (深度影像模型) 對照速覽

量化方法與深度模型現在用**完全相同的世代**(RQ1/RQ2a/RQ2c/RQ3=66、RQ2b=61),
因此可逐格直接比較「12 維量化特徵 + FCNN」vs「深度影像模型」。詳見各自表格。
