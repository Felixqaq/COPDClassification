"""產生 oi>=3 (66 筆) 二分類實驗結果的 markdown — 只輸出 AeroPath WA% 版。"""
import json
import glob
import os


def latest(tag):
    return sorted(glob.glob(f"models/training_oi_{tag}_*"))[-1]


da = latest("aeropath")
ra = json.load(open(os.path.join(da, "kfold_training_results.json"), encoding="utf-8"))
cfg = ra["configuration"]
info = cfg["data_summary"]
tp = cfg["training_params"]
ma = cfg["model_architecture"]
sa = ra["kfold_results"]["summary"]
fa = ra["kfold_results"]["fold_metrics"]

L = []
A = L.append
A("# COPD 二分類實驗結果 — oi≥3 標籤（全 66 筆，AeroPath WA%）")
A("")
A("> 以 `oi_processed.json` 的 oi 值二分：**oi ≥ 3 → Abnormal(1)、oi < 3 → Normal(0)**。")
A("> 資料中無 oi=3.0，故與「oi ≤ 3」切法等價。WA% 採用 AeroPath 完整氣道樹 NIfTI 遮罩。")
A("")
A("## 實驗設定")
A("")
A(f"- 樣本數：**{info['n_used']}**（Normal {info['n_normal']} / Abnormal {info['n_abnormal']}）")
A("- 特徵：12 維影像特徵（全肺+5 葉 emphysema%、SVV%、WA%、Vessel Density%、Airway/Lung%、Lung Volume、PA Diameter）")
A(f"- 模型：FCNN {ma['input_size']}→{'→'.join(map(str, ma['hidden_sizes']))}→{ma['output_size']}（BatchNorm + ReLU + Dropout {ma['dropout_rate']}）")
A(f"- 驗證：Stratified {tp['n_folds']}-Fold ｜ Epochs {tp['epochs']} ｜ Batch {tp['batch_size']} ｜ LR {tp['learning_rate']} ｜ weight decay {tp['weight_decay']} ｜ early stop {tp['early_stopping_patience']} ｜ seed {tp['random_seed']}")
A("- WA% 來源：**AeroPath**（完整氣道樹 NIfTI 遮罩，source = Dedicated File）")
A("")
A("## 彙總結果")
A("")
A("| 指標 | 平均 ± 標準差 | 合併總體 |")
A("|---|---:|---:|")
A(f"| Accuracy | {sa['mean_accuracy']:.4f} ± {sa['std_accuracy']:.4f} | {sa['overall_accuracy']:.4f} |")
A(f"| Precision | {sa['mean_precision']:.4f} ± {sa['std_precision']:.4f} | {sa['overall_precision']:.4f} |")
A(f"| Recall | {sa['mean_recall']:.4f} ± {sa['std_recall']:.4f} | {sa['overall_recall']:.4f} |")
A(f"| F1-score | {sa['mean_f1_score']:.4f} ± {sa['std_f1_score']:.4f} | {sa['overall_f1_score']:.4f} |")
A(f"| AUC | {sa['mean_auc']:.4f} ± {sa['std_auc']:.4f} | {sa['overall_auc']:.4f} |")
A("")
mp = sa["overall_confusion_matrix"]
A("## 總體混淆矩陣")
A("")
A("| 實際＼預測 | Normal | Abnormal |")
A("|---|---:|---:|")
A(f"| Normal | {mp[0][0]} | {mp[0][1]} |")
A(f"| Abnormal | {mp[1][0]} | {mp[1][1]} |")
A("")
A("## 各 Fold 結果")
A("")
A("| Fold | Accuracy | Precision | Recall | F1 | AUC |")
A("|---|---:|---:|---:|---:|---:|")
for b in fa:
    auc = b["auc"] if b["auc"] is not None else float("nan")
    A(f"| {b['fold']} | {b['accuracy']:.4f} | {b['precision']:.4f} | {b['recall']:.4f} | {b['f1_score']:.4f} | {auc:.4f} |")
A("")
A(f"> 結果資料夾：`{os.path.basename(da)}` ｜ seed {tp['random_seed']}，可重現。")
A("")

out = "oi_results.md"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("written:", out)
