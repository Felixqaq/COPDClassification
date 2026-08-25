"""
RQ 1/2/3 — 「量化方法」(12 維影像量化特徵 + FCNN) 版本
================================================================

本腳本把「這個專案的量化方法」(12 維 COPD 定量生物標記 -> 全連接神經網路)
套用到論文的三個研究問題 (RQ1/RQ2/RQ3),與 result.md 中的兩個深度影像模型
(Mamba+Attention / TAP-CT fusion) 形成對照。

任務 (與 result.md 對齊):
- RQ1  normal_v_abnormal    : 二分類 (臨床 Normal/Abnormal 資料夾標籤)
- RQ2a angle_3class         : 三分類 (塌陷角 <=131 / 132-151 / >=152)
- RQ2b angle_binary_extreme : 極端二分類 (排除 132-151 灰區; 1=低角/塌陷, 0=高角/正常)
- RQ2c angle                : 迴歸 (預測塌陷角角度值)
- RQ3  oi_emphysema         : 二分類 (oi >= 3 -> 氣腫/1)

方法與資料:
- 特徵: 與原專案完全相同的 12 維影像量化特徵 (extract_features_from_json)。
- 模型: 與原專案相同的 FCNN (COPDClassifier),分類時 output_size=類別數,
        迴歸時 output_size=1。
- 標籤: RQ1 來自資料夾;RQ2 來自 PFT 塌陷角 (patient_angles_simple.json);
        RQ3 來自 oi_processed.json。

對齊訓練協定 (與 result.md 相同精神):
- seed=42, 5-fold (patient-level stratified), 100 epochs。
- Early stopping 關閉:固定 100 epoch,以「最後一個 epoch」的模型在 test fold 評估,
  避免用 test fold 選停點造成的樂觀偏差 (與 result.md 一致)。
- 分類:class-weighted CrossEntropy (量化版的 balanced_sampling),per-fold StandardScaler。
- 迴歸:target zscore 正規化 (per-fold),MSE loss。
- 決策:argmax,不調閾值。

指標 (與 result.md 對齊):
- 分類:Accuracy / Sensitivity / Specificity / Macro-F1 (mean±std over 5 folds);
        二分類另附 AUC。多分類 Sens/Spec 為 macro (one-vs-rest)。
- 迴歸:MAE / RMSE / R² / Pearson r。
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import pearsonr

from copd_classifier import (
    load_json_metrics,
    extract_features_from_json,
    COPDClassifier,
)

# --------------------------------------------------------------------------- #
# 設定
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
# 塌陷角量測分兩批:原始 54 位 + 20260421 後補的 12 位(醫院 NormalDataset),合計 66。
ANGLE_JSONS = [
    Path("d:/Felix/Research/DicomToNii/pft_jpg_output/patient_angles_simple.json"),
    Path("d:/Felix/Research/DicomToNii/pft_jpg_output/醫院NormalDataset20260421/20260421patient_angles_simple.json"),
]

FEATURE_NAMES = [
    "Total_Emphysema_Percent",
    "Left_Superior_Lobe_Emphysema",
    "Left_Inferior_Lobe_Emphysema",
    "Right_Superior_Lobe_Emphysema",
    "Right_Middle_Lobe_Emphysema",
    "Right_Inferior_Lobe_Emphysema",
    "SVV_Percent",
    "WA_Percent",
    "Vessel_Density_Percent",
    "Airway_Lung_Ratio_Percent",
    "Total_Lung_Volume_ml",
    "PA_Diameter_mm",
]

DATASET_FOLDERS = ["NormalDataset", "AbnormalDataset", "TestDataset", "AllExtraDataset"]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# 特徵載入 (pid -> 12 維特徵)
# --------------------------------------------------------------------------- #
def load_features(param_subdir: str):
    """回傳 {pid: (feat(12,), folder, sample_name)},每位病人只取一次。"""
    feats = {}
    for folder in DATASET_FOLDERS:
        d = ROOT / folder / param_subdir
        if not d.exists():
            continue
        for jf in sorted(d.glob("*_metrics.json")):
            pid = jf.stem.split("_")[0]
            if pid in feats:
                continue
            try:
                data = load_json_metrics(str(jf))
                vec = np.asarray(extract_features_from_json(data), dtype=np.float32)
            except Exception as e:
                print(f"  [warn] 特徵擷取失敗 {jf.name}: {e}")
                continue
            feats[pid] = (vec, folder, jf.stem.replace("_metrics", ""))
    return feats


def load_angle_map():
    angle = {}
    for p in ANGLE_JSONS:
        if not p.exists():
            print(f"  [warn] 找不到塌陷角檔案 {p}")
            continue
        with open(p, encoding="utf-8") as f:
            for k, v in json.load(f).items():
                angle[str(k)] = float(v)
    return angle


def load_oi_map():
    with open(ROOT / "oi_processed.json", encoding="utf-8") as f:
        return {d["patient_id"]: float(d["oi"]) for d in json.load(f)}


# --------------------------------------------------------------------------- #
# 任務建構: 回傳 X, y, sample_names, meta
# --------------------------------------------------------------------------- #
def build_task(task: str, feats: dict):
    angle = load_angle_map()
    oi = load_oi_map()

    X, y, names = [], [], []

    def add(pid, label):
        vec, _folder, sname = feats[pid]
        X.append(vec)
        y.append(label)
        names.append(sname)

    if task == "rq1_normal_v_abnormal":
        # NormalDataset(21) + AllExtraDataset(12, 皆為臨床正常, 後補病人) = Normal(33);
        # AbnormalDataset(33) = Abnormal。合計 66 (33/33),與 result.md 深度模型的 RQ1 世代對齊。
        for pid, (_v, folder, _s) in feats.items():
            if folder in ("NormalDataset", "AllExtraDataset"):
                add(pid, 0)
            elif folder == "AbnormalDataset":
                add(pid, 1)
        meta = dict(kind="classification", n_classes=2,
                    class_names=["Normal", "Abnormal"], pos_label=1)

    elif task == "rq2a_angle_3class":
        for pid in feats:
            if pid not in angle:
                continue
            a = angle[pid]
            cls = 0 if a <= 131 else (1 if a <= 151 else 2)
            add(pid, cls)
        meta = dict(kind="classification", n_classes=3,
                    class_names=["<=131", "132-151", ">=152"], pos_label=None)

    elif task == "rq2b_angle_binary_extreme":
        for pid in feats:
            if pid not in angle:
                continue
            a = angle[pid]
            if a <= 131:
                add(pid, 1)          # 低角 = 塌陷 = 異常 = 正類
            elif a >= 152:
                add(pid, 0)          # 高角 = 正常
            # 132-151 灰區排除
        meta = dict(kind="classification", n_classes=2,
                    class_names=["High(>=152)", "Low(<=131)"], pos_label=1)

    elif task == "rq2c_angle_reg":
        for pid in feats:
            if pid not in angle:
                continue
            add(pid, angle[pid])
        meta = dict(kind="regression", unit="deg")

    elif task == "rq3_oi_emphysema":
        for pid in feats:
            if pid not in oi:
                continue
            add(pid, 1 if oi[pid] >= 3.0 else 0)
        meta = dict(kind="classification", n_classes=2,
                    class_names=["Normal(oi<3)", "Emphysema(oi>=3)"], pos_label=1)
    else:
        raise ValueError(task)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=(np.float32 if meta["kind"] == "regression" else np.int64))
    return X, y, names, meta


# --------------------------------------------------------------------------- #
# 指標
# --------------------------------------------------------------------------- #
def macro_sensitivity_specificity(y_true, y_pred, n_classes):
    """多分類 one-vs-rest 的 macro sensitivity / specificity。"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    total = cm.sum()
    sens, spec = [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(sens)), float(np.mean(spec))


def classification_fold_metrics(y_true, y_pred, y_score, meta):
    n_classes = meta["n_classes"]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    if n_classes == 2:
        pos = meta["pos_label"]
        neg = 1 - pos
        sens = recall_score(y_true, y_pred, pos_label=pos, zero_division=0)
        spec = recall_score(y_true, y_pred, pos_label=neg, zero_division=0)
        try:
            auc = roc_auc_score((np.asarray(y_true) == pos).astype(int), y_score)
        except Exception:
            auc = None
    else:
        sens, spec = macro_sensitivity_specificity(y_true, y_pred, n_classes)
        auc = None
    return dict(accuracy=acc, sensitivity=sens, specificity=spec,
                macro_f1=macro_f1, auc=auc)


# --------------------------------------------------------------------------- #
# 訓練 (單一 fold,固定 100 epoch,無 early stopping)
# --------------------------------------------------------------------------- #
def train_one_fold_classification(Xtr, ytr, Xte, meta, args, device):
    n_classes = meta["n_classes"]
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)

    # class-weighted loss (量化版 balanced_sampling)
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = (len(ytr) / (n_classes * counts)).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))

    ds = TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(ytr))
    drop_last = (len(ytr) % args.batch_size) == 1  # 避免 BatchNorm 遇到 size-1 batch
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=drop_last)

    model = COPDClassifier(input_size=Xtr.shape[1],
                           hidden_sizes=[args.hidden1, args.hidden2, args.hidden3],
                           output_size=n_classes, dropout_rate=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xte_s).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
    pos_score = probs[:, meta["pos_label"]] if n_classes == 2 else None
    return preds, pos_score


def train_one_fold_regression(Xtr, ytr, Xte, args, device):
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)

    y_mean, y_std = float(ytr.mean()), float(ytr.std() + 1e-8)
    ytr_n = ((ytr - y_mean) / y_std).astype(np.float32)

    ds = TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(ytr_n))
    drop_last = (len(ytr) % args.batch_size) == 1
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=drop_last)

    model = COPDClassifier(input_size=Xtr.shape[1],
                           hidden_sizes=[args.hidden1, args.hidden2, args.hidden3],
                           output_size=1, dropout_rate=args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb).squeeze(1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(Xte_s).to(device)).squeeze(1).cpu().numpy()
    preds = out * y_std + y_mean  # 反正規化回角度
    return preds


# --------------------------------------------------------------------------- #
# 一個任務的 5-fold
# --------------------------------------------------------------------------- #
def run_task(task, feats, args, device):
    X, y, names, meta = build_task(task, feats)
    n = len(y)
    print(f"\n{'=' * 70}\n任務 {task}  |  kind={meta['kind']}  |  n={n}")
    if meta["kind"] == "classification":
        dist = np.bincount(y, minlength=meta["n_classes"]).tolist()
        print(f"  類別分布 {meta['class_names']} = {dist}")
    else:
        print(f"  目標範圍 = [{y.min():.0f}, {y.max():.0f}] {meta.get('unit','')}")

    result = dict(task=task, meta={k: v for k, v in meta.items()},
                  n=n, folds=[])

    if meta["kind"] == "classification":
        splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                                   random_state=args.seed)
        split_iter = splitter.split(X, y)
        oof_true, oof_pred, oof_score = [], [], []
        for k, (tr, te) in enumerate(split_iter):
            preds, score = train_one_fold_classification(
                X[tr], y[tr], X[te], meta, args, device)
            m = classification_fold_metrics(y[te], preds, score, meta)
            m["fold"] = k + 1
            m["n_test"] = int(len(te))
            m["confusion_matrix"] = confusion_matrix(
                y[te], preds, labels=list(range(meta["n_classes"]))).tolist()
            result["folds"].append(m)
            oof_true.extend(y[te].tolist())
            oof_pred.extend(preds.tolist())
            if score is not None:
                oof_score.extend(score.tolist())
            print(f"  fold {k+1}: acc={m['accuracy']:.3f} sens={m['sensitivity']:.3f} "
                  f"spec={m['specificity']:.3f} mF1={m['macro_f1']:.3f}"
                  + (f" auc={m['auc']:.3f}" if m['auc'] is not None else ""))

        keys = ["accuracy", "sensitivity", "specificity", "macro_f1"]
        summary = {f"{key}_mean": float(np.mean([f[key] for f in result["folds"]]))
                   for key in keys}
        summary.update({f"{key}_std": float(np.std([f[key] for f in result["folds"]]))
                        for key in keys})
        aucs = [f["auc"] for f in result["folds"] if f["auc"] is not None]
        summary["auc_mean"] = float(np.mean(aucs)) if aucs else None
        summary["auc_std"] = float(np.std(aucs)) if aucs else None
        summary["overall_confusion_matrix"] = confusion_matrix(
            oof_true, oof_pred, labels=list(range(meta["n_classes"]))).tolist()
        result["summary"] = summary
        print(f"  >>> Acc {summary['accuracy_mean']:.3f}±{summary['accuracy_std']:.3f} "
              f"| Macro-F1 {summary['macro_f1_mean']:.3f}"
              + (f" | AUC {summary['auc_mean']:.3f}" if summary['auc_mean'] else ""))

    else:  # regression
        splitter = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_mae, fold_rmse, fold_r2, fold_r = [], [], [], []
        oof_true, oof_pred = [], []
        for k, (tr, te) in enumerate(splitter.split(X)):
            preds = train_one_fold_regression(X[tr], y[tr], X[te], args, device)
            mae = mean_absolute_error(y[te], preds)
            rmse = float(np.sqrt(mean_squared_error(y[te], preds)))
            r2 = r2_score(y[te], preds) if len(te) > 1 else float("nan")
            try:
                r = pearsonr(y[te], preds)[0]
            except Exception:
                r = float("nan")
            result["folds"].append(dict(fold=k + 1, n_test=int(len(te)),
                                        mae=float(mae), rmse=rmse,
                                        r2=float(r2), pearson=float(r)))
            fold_mae.append(mae); fold_rmse.append(rmse); fold_r2.append(r2); fold_r.append(r)
            oof_true.extend(y[te].tolist()); oof_pred.extend(preds.tolist())
            print(f"  fold {k+1}: MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f} r={r:.3f}")

        oof_true, oof_pred = np.array(oof_true), np.array(oof_pred)
        result["summary"] = dict(
            mae_mean=float(np.mean(fold_mae)), mae_std=float(np.std(fold_mae)),
            rmse_mean=float(np.mean(fold_rmse)), rmse_std=float(np.std(fold_rmse)),
            r2_mean=float(np.nanmean(fold_r2)), r2_std=float(np.nanstd(fold_r2)),
            pearson_mean=float(np.nanmean(fold_r)), pearson_std=float(np.nanstd(fold_r)),
            overall_mae=float(mean_absolute_error(oof_true, oof_pred)),
            overall_rmse=float(np.sqrt(mean_squared_error(oof_true, oof_pred))),
            overall_r2=float(r2_score(oof_true, oof_pred)),
            overall_pearson=float(pearsonr(oof_true, oof_pred)[0]),
        )
        s = result["summary"]
        print(f"  >>> MAE {s['mae_mean']:.2f}±{s['mae_std']:.2f} | "
              f"R2 {s['r2_mean']:.3f} | r {s['pearson_mean']:.3f}")

    return result


# --------------------------------------------------------------------------- #
# Markdown 產出
# --------------------------------------------------------------------------- #
def fmt(m, s):
    return f"{m:.3f}±{s:.3f}"


def write_markdown(all_results: dict, out_path: Path, args):
    """all_results: {param_subdir: {task: result}}"""
    lines = []
    lines.append("# RQ 1/2/3 — 量化方法 (12 維特徵 + FCNN) 結果\n")
    lines.append("本表把「這個專案的量化方法」(12 維 COPD 定量生物標記 → FCNN) 套用到 RQ1/2/3,")
    lines.append("與 [result.md](result.md) 的兩個深度影像模型 (Mamba+Attention / TAP-CT fusion) 形成對照。\n")
    lines.append(f"由 `train_rq_quant.py` 產生 ({datetime.now():%Y-%m-%d %H:%M}).\n")

    lines.append("## 方法\n")
    lines.append("- **特徵**：12 維影像量化特徵 (肺氣腫% ×6、SVV%、WA%、血管密度%、氣道/肺比%、肺體積、PA 直徑),與原專案分類器完全相同。")
    lines.append("- **模型**：原專案 FCNN (`COPDClassifier`, 12→64→32→16→out)。分類 out=類別數;迴歸 out=1。")
    lines.append("- **標籤**：RQ1 臨床資料夾;RQ2 PFT 塌陷角 (`patient_angles_simple.json`);RQ3 `oi_processed.json` (oi≥3)。")
    lines.append(f"- **協定 (對齊 result.md)**：seed={args.seed}、{args.n_folds}-fold stratified、{args.epochs} epochs、")
    lines.append("  **early stopping 關閉**(固定預算、最後 epoch 評估,避免用 test fold 選停點)、分類用 class-weighted CE、")
    lines.append("  per-fold StandardScaler、迴歸 target zscore、決策 argmax 不調閾值。")
    lines.append("- **兩種 WA% 特徵來源**：`param` = Trachea Label WA%;`param_aeropath` = AeroPath 氣道樹 WA%。\n")

    lines.append("## 世代 (量化方法可用標籤)\n")
    lines.append("| RQ | 任務 | 型態 | n | 說明 |")
    lines.append("|----|------|------|---|------|")
    cohort_rows = [
        ("RQ1", "normal_v_abnormal", "二分類", "66", "臨床 Normal(21+12 AllExtra)/Abnormal(33),33/33"),
        ("RQ2a", "angle_3class", "三分類", "66", "塌陷角 ≤131/132–151/≥152"),
        ("RQ2b", "angle_binary_extreme", "二分類", "61", "排除 132–151 灰區 (14 低角 / 47 高角)"),
        ("RQ2c", "angle", "迴歸", "66", "塌陷角角度值 (105–179°)"),
        ("RQ3", "oi_emphysema", "二分類", "66", "oi≥3 → 氣腫 (32/34)"),
    ]
    for r in cohort_rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")
    lines.append("> 註:66 位病人**全部**都有 12 維特徵、臨床標籤、OI 與 PFT 塌陷角量測")
    lines.append("> (塌陷角分兩批:原始 54 + 20260421 後補 12)。故 RQ1/RQ2a/RQ2c/RQ3 皆為 66、")
    lines.append("> RQ2b 排除 132–151 灰區後 61 —— **與上方 result.md 深度模型的世代完全一致,可逐格直接比較**。\n")

    task_label = {
        "rq1_normal_v_abnormal": ("RQ1", "normal_v_abnormal"),
        "rq2a_angle_3class": ("RQ2a", "angle_3class"),
        "rq2b_angle_binary_extreme": ("RQ2b", "angle_binary_extreme"),
        "rq3_oi_emphysema": ("RQ3", "oi_emphysema"),
    }
    cls_tasks = ["rq1_normal_v_abnormal", "rq2a_angle_3class",
                 "rq2b_angle_binary_extreme", "rq3_oi_emphysema"]

    for sub in all_results:
        src = "AeroPath WA%" if sub == "param_aeropath" else "Trachea WA%"
        lines.append(f"## 結果 — 特徵來源:`{sub}` ({src})\n")
        lines.append("### 分類任務 (Acc / Sensitivity / Specificity 為 mean±std over folds;多分類 Sens/Spec 為 macro)\n")
        lines.append("| RQ | 任務 | Accuracy | Sensitivity | Specificity | Macro-F1 | AUC |")
        lines.append("|----|------|----------|-------------|-------------|----------|-----|")
        for t in cls_tasks:
            r = all_results[sub].get(t)
            if not r:
                continue
            s = r["summary"]
            rq, name = task_label[t]
            auc = f"{s['auc_mean']:.3f}" if s.get("auc_mean") else "—"
            lines.append(
                f"| {rq} | {name} | {fmt(s['accuracy_mean'], s['accuracy_std'])} | "
                f"{fmt(s['sensitivity_mean'], s['sensitivity_std'])} | "
                f"{fmt(s['specificity_mean'], s['specificity_std'])} | "
                f"{s['macro_f1_mean']:.3f} | {auc} |")
        lines.append("")
        # regression
        rr = all_results[sub].get("rq2c_angle_reg")
        if rr:
            s = rr["summary"]
            lines.append("### 迴歸任務 (RQ2c 塌陷角角度值)\n")
            lines.append("| RQ | 任務 | MAE (°) | RMSE (°) | R² | Pearson r |")
            lines.append("|----|------|---------|----------|-----|-----------|")
            lines.append(
                f"| RQ2c | angle | {s['mae_mean']:.2f}±{s['mae_std']:.2f} | "
                f"{s['rmse_mean']:.2f} | {s['r2_mean']:.3f} | {s['pearson_mean']:.3f} |")
            lines.append("")

    lines.append("## 與 result.md (深度影像模型) 對照速覽\n")
    lines.append("量化方法與深度模型現在用**完全相同的世代**(RQ1/RQ2a/RQ2c/RQ3=66、RQ2b=61),")
    lines.append("因此可逐格直接比較「12 維量化特徵 + FCNN」vs「深度影像模型」。詳見各自表格。\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown 已寫入: {out_path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="RQ1/2/3 量化方法 (12 維特徵 + FCNN)")
    ap.add_argument("--param-subdirs", nargs="+", default=["param", "param_aeropath"],
                    help="特徵來源子目錄 (可多個)")
    ap.add_argument("--tasks", nargs="+", default=[
        "rq1_normal_v_abnormal", "rq2a_angle_3class", "rq2b_angle_binary_extreme",
        "rq2c_angle_reg", "rq3_oi_emphysema"])
    ap.add_argument("--out-md", type=str, default="result_quant.md")
    ap.add_argument("--out-dir", type=str, default="models/rq_quant")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--hidden1", type=int, default=64)
    ap.add_argument("--hidden2", type=int, default=32)
    ap.add_argument("--hidden3", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用設備: {device}")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for sub in args.param_subdirs:
        print(f"\n{'#' * 70}\n# 特徵來源: {sub}\n{'#' * 70}")
        feats = load_features(sub)
        print(f"載入 {len(feats)} 位病人特徵 (來源 {sub})")
        all_results[sub] = {}
        for task in args.tasks:
            set_seed(args.seed)  # 每個任務前重設,確保可重現
            res = run_task(task, feats, args, device)
            all_results[sub][task] = res
            with open(out_dir / f"{sub}__{task}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)

    with open(out_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    write_markdown(all_results, ROOT / args.out_md, args)
    print("\n完成。")


if __name__ == "__main__":
    main()
