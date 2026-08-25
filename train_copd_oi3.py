"""
COPD 三分類訓練腳本 — 依 oi 阻塞嚴重度分組。

標籤（來自 oi_processed.json 的 oi 值）：
  class 0 = 輕度/早期 (oi <= t_low,   預設 3)
  class 1 = 中度      (t_low < oi < t_high)
  class 2 = 重度      (oi >= t_high,  預設 7)

特徵與模型同 binary 版（12 維影像特徵、FCNN），但 output_size=3、評估改用多分類指標。
WA% 來源用 --param-subdir 切換 (param=Trachea, param_aeropath=AeroPath)。
"""
import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from copd_classifier import COPDClassifier, COPDDataset, load_json_metrics, extract_features_from_json

CLASS_NAMES = ["Mild(oi<=3)", "Moderate(3-7)", "Severe(oi>=7)"]
FEATURE_NAMES = [
    "Total_Emphysema_Percent", "Left_Superior_Lobe_Emphysema", "Left_Inferior_Lobe_Emphysema",
    "Right_Superior_Lobe_Emphysema", "Right_Middle_Lobe_Emphysema", "Right_Inferior_Lobe_Emphysema",
    "SVV_Percent", "WA_Percent", "Vessel_Density_Percent", "Airway_Lung_Ratio_Percent",
    "Total_Lung_Volume_ml", "PA_Diameter_mm",
]


def label_of(oi, t_low, t_high):
    if oi <= t_low:
        return 0
    if oi >= t_high:
        return 2
    return 1


def load_data(metrics_dirs, oi_json_path, t_low, t_high):
    oi_map = {d["patient_id"]: d["oi"] for d in json.load(open(oi_json_path, encoding="utf-8"))}
    feats, labels, names, used = [], [], [], set()
    for d in metrics_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for jf in sorted(p.glob("*_metrics.json")):
            pid = jf.stem.split("_")[0]
            if pid in used or pid not in oi_map:
                continue
            try:
                f = extract_features_from_json(load_json_metrics(str(jf)))
            except Exception as e:
                print(f"略過 {pid}: {e}")
                continue
            feats.append(f)
            labels.append(label_of(oi_map[pid], t_low, t_high))
            names.append(jf.stem.replace("_metrics", ""))
            used.add(pid)
    X = np.array(feats, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    counts = {CLASS_NAMES[i]: int((y == i).sum()) for i in range(3)}
    print(f"\n載入 {len(y)} 筆｜分組: {counts}")
    return X, y, names, counts


def train_fold(X_tr, y_tr, X_va, y_va, args, device):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    tl = DataLoader(COPDDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(COPDDataset(X_va, y_va), batch_size=args.batch_size, shuffle=False)
    model = COPDClassifier(input_size=12, hidden_sizes=[args.hidden1, args.hidden2, args.hidden3],
                           output_size=3, dropout_rate=args.dropout).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    best_loss, best_state, patience = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        model.eval(); vloss = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(device), yb.to(device)
                vloss += crit(model(xb), yb).item()
        vloss /= max(len(vl), 1)
        sched.step(vloss)
        if vloss < best_loss:
            best_loss, best_state, patience = vloss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    model.load_state_dict(best_state)
    # 預測驗證集
    model.eval(); preds, probs = [], []
    with torch.no_grad():
        for xb, yb in vl:
            out = model(xb.to(device))
            p = torch.softmax(out, dim=1).cpu().numpy()
            probs.append(p); preds.append(p.argmax(1))
    return np.concatenate(preds), np.concatenate(probs)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    tag = "aeropath" if args.param_subdir != "param" else "trachea"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"training_oi3_{tag}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"使用設備: {device}｜輸出: {out_dir}")

    sub = args.param_subdir
    metrics_dirs = [os.path.join(args.data_dir, ds, sub) for ds in
                    ("NormalDataset", "AbnormalDataset", "TestDataset", "AllExtraDataset")]
    X, y, names, counts = load_data(metrics_dirs, os.path.join(args.data_dir, args.oi_json),
                                    args.threshold_low, args.threshold_high)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    all_true, all_pred, all_prob = [], [], []
    fold_metrics = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        preds, probs = train_fold(X[tr], y[tr], X[va], y[va], args, device)
        yt = y[va]
        acc = accuracy_score(yt, preds)
        fold_metrics.append({
            "fold": fold,
            "accuracy": float(acc),
            "macro_precision": float(precision_score(yt, preds, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(yt, preds, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(yt, preds, average="macro", zero_division=0)),
            "val_samples": int(len(va)),
        })
        print(f"Fold {fold}: acc={acc:.4f}  macroF1={fold_metrics[-1]['macro_f1']:.4f}")
        all_true.extend(yt); all_pred.extend(preds); all_prob.extend(probs)

    all_true = np.array(all_true); all_pred = np.array(all_pred); all_prob = np.array(all_prob)
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2])
    try:
        macro_auc = float(roc_auc_score(all_true, all_prob, multi_class="ovr", average="macro"))
    except Exception:
        macro_auc = None

    summary = {
        "overall_accuracy": float(accuracy_score(all_true, all_pred)),
        "macro_precision": float(precision_score(all_true, all_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(all_true, all_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(all_true, all_pred, average="weighted", zero_division=0)),
        "macro_auc_ovr": macro_auc,
        "mean_fold_accuracy": float(np.mean([f["accuracy"] for f in fold_metrics])),
        "std_fold_accuracy": float(np.std([f["accuracy"] for f in fold_metrics])),
        "mean_fold_macro_f1": float(np.mean([f["macro_f1"] for f in fold_metrics])),
        "confusion_matrix": cm.tolist(),
        "per_class": {},
    }
    pc_p = precision_score(all_true, all_pred, average=None, labels=[0, 1, 2], zero_division=0)
    pc_r = recall_score(all_true, all_pred, average=None, labels=[0, 1, 2], zero_division=0)
    pc_f = f1_score(all_true, all_pred, average=None, labels=[0, 1, 2], zero_division=0)
    for i, cn in enumerate(CLASS_NAMES):
        summary["per_class"][cn] = {"precision": float(pc_p[i]), "recall": float(pc_r[i]),
                                    "f1": float(pc_f[i]), "support": int((all_true == i).sum())}

    print("\n===== 三分類總結 =====")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.4f}")
    print(f"Macro F1: {summary['macro_f1']:.4f}  Weighted F1: {summary['weighted_f1']:.4f}")
    if macro_auc is not None:
        print(f"Macro AUC (OvR): {macro_auc:.4f}")
    print("混淆矩陣 (列=實際, 欄=預測) order=[Mild,Moderate,Severe]:")
    for row in cm:
        print("  ", row.tolist())

    # 混淆矩陣圖
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"oi 3-class Confusion Matrix ({tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close()

    result = {
        "configuration": {
            "labeling": {"rule": f"oi<= {args.threshold_low} Mild / {args.threshold_low}<oi<{args.threshold_high} Moderate / oi>= {args.threshold_high} Severe",
                         "t_low": args.threshold_low, "t_high": args.threshold_high,
                         "wa_source": "AeroPath" if tag == "aeropath" else "Trachea",
                         "param_subdir": sub, "class_counts": counts},
            "model_architecture": {"input_size": 12, "hidden_sizes": [args.hidden1, args.hidden2, args.hidden3],
                                   "output_size": 3, "dropout_rate": args.dropout},
            "training_params": {"epochs": args.epochs, "batch_size": args.batch_size,
                                "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                                "early_stopping_patience": args.patience, "n_folds": args.n_folds,
                                "random_seed": args.seed},
            "feature_names": FEATURE_NAMES,
        },
        "fold_metrics": fold_metrics,
        "summary": summary,
    }
    with open(os.path.join(out_dir, "kfold_training_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n結果已存: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="oi 阻塞嚴重度三分類 (5-Fold)")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--output-dir", default="models")
    ap.add_argument("--oi-json", default="oi_processed.json")
    ap.add_argument("--param-subdir", default="param", help="param=Trachea WA%, param_aeropath=AeroPath WA%")
    ap.add_argument("--threshold-low", type=float, default=3.0)
    ap.add_argument("--threshold-high", type=float, default=7.0)
    ap.add_argument("--hidden1", type=int, default=64)
    ap.add_argument("--hidden2", type=int, default=32)
    ap.add_argument("--hidden3", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
