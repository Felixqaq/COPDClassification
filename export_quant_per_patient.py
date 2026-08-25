"""
匯出「量化方法 (12 維特徵 + FCNN)」的 per-patient out-of-fold 預測(RQ1 / RQ2b / RQ3)。

用途:`per_patient_vote_<rq>.json` / `.md` 原本只有兩個深度模型 (image / fusion),
      本腳本補上第三個模型 (quant),讓 hard majority vote 三模型齊全。

協定與 `train_rq_quant.py` 完全相同(同樣 import 其函式),即 result.md 中
「量化(12 特徵+FCNN)」那幾列的設定:
seed=42、5-fold patient-level stratified、100 epochs、early stopping 關閉
(固定預算、最後一個 epoch 評估)、class-weighted CE、per-fold StandardScaler、argmax、
特徵來源 `param_aeropath`。

跑完會自動比對 `models/rq_quant/<param_subdir>__<task>.json` 的每折指標,
確認與 result.md 的數字逐折一致(不一致會警告)。

輸出:models/rq_quant/<param_subdir>__<rq>_per_patient.json

用法:
    python export_quant_per_patient.py --rq rq2b
    python export_quant_per_patient.py --rq rq3
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

# rq 代號 -> (train_rq_quant 的 task 名稱, 正類 index 對應的 class name index)
TASKS = {
    "rq1": "rq1_normal_v_abnormal",
    "rq2b": "rq2b_angle_binary_extreme",
    "rq3": "rq3_oi_emphysema",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq", default="rq2b", choices=sorted(TASKS))
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
    # 預設 CPU:result.md / result_quant.md 的數字是在 CPU 上跑出來的,CPU 可逐 fold 完全重現。
    # 改用 CUDA 會因浮點累加順序不同而有些微出入。
    ap.add_argument("--cuda", action="store_true", help="改用 GPU(結果會與 result.md 略有出入)")
    args = ap.parse_args()

    task = TASKS[args.rq]
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"任務: {args.rq} ({task})  |  設備: {device}  |  特徵來源: {args.param_subdir}")

    feats = load_features(args.param_subdir)
    print(f"載入 {len(feats)} 位病人特徵")

    set_seed(args.seed)  # 與 train_rq_quant.py 相同:每個任務前重設
    X, y, names, meta = build_task(task, feats)
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
                prob_positive=round(float(score[i]), 5),
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

    check_against_result(args, task, folds)

    out = ROOT / args.out_dir / f"{args.param_subdir}__{args.rq}_per_patient.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(
            task=task,
            rq=args.rq,
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


def check_against_result(args, task, folds):
    """與 train_rq_quant.py 既有產出逐折比對,確認重現 result.md 的數字。"""
    ref_path = ROOT / args.out_dir / f"{args.param_subdir}__{task}.json"
    if not ref_path.exists():
        print(f"  [note] 找不到 {ref_path.name},略過重現性比對")
        return
    ref = json.loads(ref_path.read_text(encoding="utf-8"))["folds"]
    if len(ref) != len(folds):
        print(f"  [warn] 折數不符: ref={len(ref)} vs now={len(folds)}")
        return
    bad = [(r["fold"], k, r[k], f[k]) for r, f in zip(ref, folds)
           for k in ("accuracy", "sensitivity", "specificity", "auc")
           if abs(r[k] - f[k]) > 1e-9]
    if bad:
        print(f"  [warn] 與 {ref_path.name} 不一致({len(bad)} 項):")
        for fold, k, a, b in bad[:8]:
            print(f"         fold {fold} {k}: result={a:.4f} vs now={b:.4f}")
    else:
        print(f"  [ok] 每折指標與 {ref_path.name}(result.md 的數字)完全一致")


if __name__ == "__main__":
    main()
