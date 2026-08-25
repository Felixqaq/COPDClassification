# RQ 1/2/3 — 三方法(量化 / image-only / fusion)× 66 病人 對齊比較

本文件比較論文三個研究問題(RQ)下的**三個模型**,全部跑在**同一份 66 病人世代**、**完全對齊的訓練協定**上。
每個 RQ 一張比對表,欄位為 **Accuracy / Sensitivity / Specificity**(mean±std,5-fold 交叉驗證)。

## 研究問題

- **RQ1｜正常 vs 異常**(`normal_v_abnormal`):CT 能否區分臨床標記的正常與異常肺?
- **RQ2｜塌陷角 collapse angle**:RQ2a 三分類(≤131° / 132–151° / ≥152°)、RQ2b 極端二分類(排除中間灰區)、RQ2c 角度迴歸。
- **RQ3｜OI 氣腫**(`oi_emphysema`):OI 阻塞指數的氣腫二分類(門檻 OI=3)。

## 三個模型

| 方法 | 模型 | 輸入 | 說明 |
|------|------|------|------|
| **量化** | FCNN(`COPDClassifier`) | 12 維定量特徵 | 肺氣腫%×6、SVV%、WA%、血管密度%、氣道/肺比%、肺體積、PA 直徑 |
| **image-only** | Mamba + Attention | CT volume | CT 影像編碼器,無外部 embedding(`hybrid_mamba_attention`) |
| **fusion** | TAP-CT Late Fusion | CT volume + TAP-CT embedding | 同一 CT 編碼器 + 凍結 TAP-CT-S-3D(1152 維)patient embedding 後段串接(`hybrid_mamba_tapct_fusion`) |

## 世代與訓練協定

- **世代(三方法完全相同)**:66 病人。RQ1=66(Normal 33 / Abnormal 33)、RQ2a=66、RQ2b=61(排除 132–151° 灰區)、
  RQ2c=66、RQ3=66(oi<3 為 32 / oi≥3 為 34)。塌陷角分兩批量測(原始 54 + 20260421 後補 12,皆臨床正常、高角),合計全 66 位都有角度。
- **協定**:seed=42、5-fold patient-level stratified、100 epochs、**early stopping 關閉**(固定預算、最終 epoch 評估,
  避免用 test fold 選停點的樂觀偏差)、分類做類別平衡、決策 argmax 不調閾值。**5× augmentation 為影像增強,只用於深度模型**;
  量化方法吃 12 維數值特徵、無影像增強,類別不平衡改以 class-weighted CE 處理,另每折做 StandardScaler。
- **Sensitivity / Specificity 的正類**:RQ1 = Abnormal、RQ2b = 低角(塌陷/異常)、RQ3 = 氣腫(oi≥3);
  RQ2a 為三分類,Sens/Spec 以 one-vs-rest **macro** 平均。**粗體 = 該表 Accuracy 最佳者。**

---

## RQ1｜正常 vs 異常(n=66,33/33)

| 模型 | Accuracy | Sensitivity | Specificity |
|------|----------|-------------|-------------|
| 量化(12 特徵+FCNN) | 0.848±0.084 | 0.800±0.214 | 0.910±0.074 |
| image-only(Mamba) | **0.879±0.078** | 0.848±0.091 | 0.914±0.114 |
| fusion(TAP-CT) | 0.866±0.107 | 0.848±0.091 | 0.886±0.140 |

**重點**:三方法非常接近(Acc 0.85–0.88),image-only 最佳,但量化只差約 3 個百分點 ——
正異常單靠 12 維定量參數即已很接近深度影像模型。

---

## RQ2｜塌陷角 collapse angle

### RQ2a 三分類(n=66;Sens/Spec 為 macro)

| 模型 | Accuracy | Sensitivity | Specificity |
|------|----------|-------------|-------------|
| 量化(12 特徵+FCNN) | 0.577±0.119 | 0.458±0.063 | 0.768±0.060 |
| image-only(Mamba) | 0.518±0.253 | 0.510±0.137 | 0.741±0.121 |
| fusion(TAP-CT) | **0.774±0.042** | 0.550±0.079 | 0.830±0.056 |

### RQ2b 極端二分類(n=61,排除 132–151° 灰區)

| 模型 | Accuracy | Sensitivity | Specificity |
|------|----------|-------------|-------------|
| 量化(12 特徵+FCNN) | 0.755±0.087 | 0.767±0.200 | 0.747±0.080 |
| image-only(Mamba) | **0.885±0.086** | 0.867±0.163 | 0.891±0.122 |
| fusion(TAP-CT) | 0.871±0.080 | 0.867±0.163 | 0.871±0.106 |

**重點**:塌陷角是「CT 影像看得出、12 維定量參數看不太出」的訊號。RQ2a 只有 **fusion** 撐得住(0.774),
量化與 image-only 皆弱(中間灰區僅 5 例是共同瓶頸);RQ2b **image-only 最佳**(0.885),量化最弱。

### RQ2c 角度迴歸(n=66)

角度為連續值,不適用 Accuracy/Sensitivity/Specificity,改用迴歸指標。**粗體 = 該欄最佳**(MAE/RMSE 越低越好、R²/r 越高越好)。

| 模型 | MAE (°) | RMSE (°) | R² | Pearson r |
|------|---------|----------|-----|-----------|
| 量化(12 特徵+FCNN) | 16.56±1.96 | **20.63** | 0.054 | 0.467 |
| image-only(Mamba) | 15.48±1.09 | 23.43 | −0.003 | 0.422 |
| fusion(TAP-CT) | **14.61±2.17** | 20.72 | **0.174** | **0.525** |

**重點**:fusion 最能解釋角度變異(R² 0.174、r 0.525、MAE 最低);image-only 幾乎無效(R²≈0);量化居中。

---

## RQ3｜OI 氣腫(n=66,32/34)

| 模型 | Accuracy | Sensitivity | Specificity |
|------|----------|-------------|-------------|
| 量化(12 特徵+FCNN) | 0.804±0.054 | 0.733±0.065 | 0.876±0.110 |
| image-only(Mamba) | 0.803±0.036 | 0.767±0.065 | 0.843±0.106 |
| fusion(TAP-CT) | **0.834±0.088** | 0.848±0.106 | 0.819±0.149 |

**重點**:三方法接近(Acc 0.80–0.83),fusion 最佳,量化與 image-only 幾乎相同 ——
肺氣腫本就與量化特徵高度相關,故量化在此任務即已足夠。

---

## 綜合觀察

- **量化在 RQ1、RQ3 已逼近深度模型**(差 ≤ 3%):正異常與 OI 氣腫靠 12 維定量參數即可分得很好。
- **深度模型只在 RQ2 塌陷角明顯較強**:fusion 贏 RQ2a / RQ2c、image-only 贏 RQ2b。
- **TAP-CT fusion 對角度/OI 有幫助**:RQ2a(0.518→0.774)、RQ2c(R² −0.003→0.174)、RQ3(0.803→0.834)。
- **RQ2b 反而 image-only 最強**、fusion 略降,與 RQ1 一致 —— 這兩個任務 CT 影像訊號已足夠,frozen embedding 反而稀釋。

## RQ1 困難個案(錯誤分析)

RQ1 準確率的上限,主要由少數「臨床標籤與 CT 影像互相矛盾」的病人決定 —— 這些病人在交叉驗證中反覆被誤判:

| 病人 | 臨床標籤 | 肺氣腫% | PA 直徑 | OI | 為何被分錯 |
|------|----------|---------|---------|-----|-----------|
| `P01` | Normal | 34.3% | 33.5mm | 2.28 | 正常人卻有 34% 肺氣腫 → 判為異常 |
| `P02` | Abnormal | 0.0% | 65.5mm | 8.45 | 異常人肺氣腫為 0(但 PA 明顯擴大)→ 判為正常 |
| `P03` | Abnormal | 6.8% | 34.1mm | 3.62 | 肺氣腫很輕、整體像正常 → 判為正常 |
| `P04` | Abnormal | 13.1% | 43.3mm | 3.48 | 邊緣個案 |
| `P05` | Normal | 18.8% | 37.9mm | 2.28 | 正常人肺氣腫偏高 → 判為異常 |
| `P06` | Abnormal | 24.9% | 40.3mm | 13.67 | 邊緣個案 |

- 模型主要依賴**肺氣腫%**,死角剛好是「正常人卻肺氣腫多」或「異常人卻幾乎無肺氣腫」。
- `P03`、`P02`、`P04` 正是先前(`研究整理.md`)已記錄的 false negative,前後一致,屬穩定的臨床邊緣病人而非流程缺陷。
- **含意**:準確率天花板受**標籤品質**與**特徵鑑別力**限制,而非訓練不足;要再提升應覆核矛盾個案的臨床標籤或補入更具鑑別力的特徵。

---

## 附註

- **補充指標(量化方法)**:二分類 AUC — RQ1 0.893、RQ2b 0.789、RQ3 0.825;Macro-F1 — RQ1 0.844、RQ2a 0.412、RQ2b 0.705、RQ3 0.802。
- **產出檔案**:量化方法 `train_rq_quant.py` → `models/rq_quant/<source>__<task>.json`、彙總 `models/rq_quant/all_results.json`、對照表 `result_quant.md`。
  深度模型 per-fold 明細 `figures/<task>/<uuid>/results.json`、UUID 對照 `train_log/run_all_rq/summary.json`;
  執行:`cd regression && conda run -n nnMamba python scripts/run_all_rq.py`。
