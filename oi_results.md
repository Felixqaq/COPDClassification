# COPD 二分類實驗結果 — oi≥3 標籤（全 66 筆，AeroPath WA%）

> 以 `oi_processed.json` 的 oi 值二分：**oi ≥ 3 → Abnormal(1)、oi < 3 → Normal(0)**。
> 資料中無 oi=3.0，故與「oi ≤ 3」切法等價。WA% 採用 AeroPath 完整氣道樹 NIfTI 遮罩。

## 實驗設定

- 樣本數：**66**（Normal 32 / Abnormal 34）
- 特徵：12 維影像特徵（全肺+5 葉 emphysema%、SVV%、WA%、Vessel Density%、Airway/Lung%、Lung Volume、PA Diameter）
- 模型：FCNN 12→64→32→16→2（BatchNorm + ReLU + Dropout 0.3）
- 驗證：Stratified 5-Fold ｜ Epochs 100 ｜ Batch 16 ｜ LR 0.001 ｜ weight decay 1e-05 ｜ early stop 15 ｜ seed 42
- WA% 來源：**AeroPath**（完整氣道樹 NIfTI 遮罩，source = Dedicated File）

## 彙總結果

| 指標 | 平均 ± 標準差 | 合併總體 |
|---|---:|---:|
| Accuracy | 0.8484 ± 0.0488 | 0.8485 |
| Precision | 0.9095 ± 0.0744 | 0.9000 |
| Recall | 0.7905 ± 0.0830 | 0.7941 |
| F1-score | 0.8413 ± 0.0531 | 0.8438 |
| AUC | 0.8762 ± 0.0844 | 0.8713 |

## 總體混淆矩陣

| 實際＼預測 | Normal | Abnormal |
|---|---:|---:|
| Normal | 29 | 3 |
| Abnormal | 7 | 27 |

## 各 Fold 結果

| Fold | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| 1 | 0.8571 | 0.8571 | 0.8571 | 0.8571 | 0.8571 |
| 2 | 0.8462 | 1.0000 | 0.6667 | 0.8000 | 0.8571 |
| 3 | 0.9231 | 1.0000 | 0.8571 | 0.9231 | 0.9762 |
| 4 | 0.8462 | 0.8571 | 0.8571 | 0.8571 | 0.9524 |
| 5 | 0.7692 | 0.8333 | 0.7143 | 0.7692 | 0.7381 |

> 結果資料夾：`training_oi_aeropath_20260626_150657` ｜ seed 42，可重現。
