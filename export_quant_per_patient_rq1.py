"""
匯出「量化方法 (12 維特徵 + FCNN)」在 RQ1 的 per-patient out-of-fold 預測。

用途:`per_patient_vote_rq1.json` / `.md` 原本只有兩個深度模型 (image / fusion),
      本腳本補上第三個模型 (quant),讓 hard majority vote 三模型齊全。

協定與 `train_rq_quant.py` 完全相同(同樣 import 其函式):
seed=42、5-fold patient-level stratified、100 epochs、early stopping 關閉
(固定預算、最後一個 epoch 評估)、class-weighted CE、per-fold StandardScaler、argmax。
特徵來源預設 `param_aeropath` —— 即 result.md 中「量化(12 特徵+FCNN)」那一列
(RQ1 Acc 0.848±0.084 / Sens 0.800 / Spec 0.910)。

輸出:models/rq_quant/<param_subdir>__rq1_per_patient.json
"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from train_rq_quant import (
    ROOT,
    set_seed,
    load_features,
    build_task,
    train_one_fold_classification,
    classification_fold_metrics,
)

TASK = "rq1_normal_v_abnormal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param-subdir", default="param_aeropath")
    ap.add_argument("--out-dir", default="models/rq_quant")
    # 以下預設值必須與 train_rq_quant.py 的 argparse 預設值一致
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
    # 預設 CPU:原始 result_quant.md 的數字是在 CPU 上跑出來的,CPU 可逐 fold 完全重現
    # (Acc 0.848±0.084 / Sens 0.800 / Spec 0.910,含每折 AUC)。改用 CUDA 會因浮點
    # 累加順序不同,在 fold 5 翻掉一位病人 (Acc 變 0.833)。
    ap.add_argument("--cuda", action="store_true", help="改用 GPU(結果會與 result.md 略有出入)")
    args = ap.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}  |  特徵來源: {args.param_subdir}")

    feats = load_features(args.param_subdir)
    print(f"載入 {len(feats)} 位病人特徵")

    set_seed(args.seed)  # 與 train_rq_quant.py 相同:每個任務前重設
    X, y, names, meta = build_task(TASK, feats)
    pids = [n.split("_")[0] for n in names]
    assert len(set(pids)) == len(pids), "patient_id 有重複"
    print(f"n={len(y)}  類別分布 {meta['class_names']} = {np.bincount(y).tolist()}")

    splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    per_patient, folds = {}, []
    for k, (tr, te) in enumerate(splitter.split(X, y)):
        preds, score = train_one_fold_classification(X[tr], y[tr], X[te], meta, args, device)
        m = classification_fold_metrics(y[te], preds, score, meta)
        m["fold"] = k + 1
        m["n_test"] = int(len(te))
        folds.append(m)
        print(f"  fold {k+1}: acc={m['accuracy']:.3f} sens={m['sensitivity']:.3f} "
              f"spec={m['specificity']:.3f} auc={m['auc']:.3f}")
        for i, idx in enumerate(te):
            per_patient[pids[idx]] = dict(
                patient_id=pids[idx],
                sample_name=names[idx],
                quant_fold=k + 1,
                true_label=meta["class_names"][int(y[idx])],
                pred_label=meta["class_names"][int(preds[i])],
                prob_abnormal=round(float(score[i]), 5),
                correct=bool(int(preds[i]) == int(y[idx])),
            )

    keys = ["accuracy", "sensitivity", "specificity", "macro_f1"]
    summary = {f"{k_}_mean": float(np.mean([f[k_] for f in folds])) for k_ in keys}
    summary.update({f"{k_}_std": float(np.std([f[k_] for f in folds])) for k_ in keys})
    summary["auc_mean"] = float(np.mean([f["auc"] for f in folds]))
    print(f"  >>> Acc {summary['accuracy_mean']:.3f}±{summary['accuracy_std']:.3f} "
          f"| Sens {summary['sensitivity_mean']:.3f} | Spec {summary['specificity_mean']:.3f}")

    n_correct = sum(p["correct"] for p in per_patient.values())
    print(f"  >>> pooled out-of-fold accuracy = {n_correct}/{len(per_patient)} "
          f"= {n_correct / len(per_patient):.4f}")

    out = ROOT / args.out_dir / f"{args.param_subdir}__rq1_per_patient.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(
            task=TASK,
            param_subdir=args.param_subdir,
            protocol=dict(seed=args.seed, n_folds=args.n_folds, epochs=args.epochs,
                          early_stopping=False, eval_epoch="last",
                          batch_size=args.batch_size, learning_rate=args.learning_rate,
                          weight_decay=args.weight_decay, dropout=args.dropout,
                          hidden_sizes=[args.hidden1, args.hidden2, args.hidden3]),
            meta=meta, n=len(per_patient),
            fold_summary=summary, folds=folds,
            patients=[per_patient[p] for p in sorted(per_patient)],
        ), f, indent=2, ensure_ascii=False)
    print(f"\n已寫入: {out}")


if __name__ == "__main__":
    main()
