"""
COPD 二元分類訓練腳本 (以 oi 門檻重新標記)

與 train_copd_model.py 使用「相同的流程」(同樣的 12 維影像特徵、FCNN、Stratified 5-Fold)，
差別只在「標籤來源」與「資料集組成」：

- 資料集：以 all/ 內所有病人為母體 (66 筆)，實際進模型者為其中具備 *_metrics.json
          的病人 (目前 54 筆，來自 NormalDataset/param + AbnormalDataset/param)。
          其餘 12 筆只有原始影像、無 metrics，需先跑分割 pipeline 才能納入，會被略過並列出。
- 標籤：來自 oi_processed.json 的 oi 值。oi >= 門檻 (預設 3) -> Abnormal(1)，否則 Normal(0)。
        資料中沒有 oi 恰等於 3 的樣本，因此 ">=3" 與原始 "<=3" 的切法等價。
- 特徵：與原流程完全相同的 12 維影像特徵 (不加入 oi/a/fvc 量化特徵，避免標籤洩漏)。
"""

import torch
import argparse
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from copd_classifier import (
    train_kfold,
    plot_kfold_results,
    load_json_metrics,
    extract_features_from_json,
)

# 與 copd_classifier.load_copd_data 相同的 12 維特徵名稱
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


def load_data_by_oi(
    metrics_dirs,
    oi_json_path,
    all_dir=None,
    threshold=3.0,
):
    """
    從多個 param 目錄彙整 metrics，並以 oi_processed.json 的 oi 值做二元標記。

    Args:
        metrics_dirs: 含 *_metrics.json 的目錄清單
        oi_json_path: oi_processed.json 路徑
        all_dir: all/ 影像目錄 (僅用於統計母體與列出缺 metrics 的病人)
        threshold: oi 門檻；oi >= threshold -> Abnormal(1)，否則 Normal(0)

    Returns:
        features (N,12), labels (N,), sample_names, info(dict)
    """
    with open(oi_json_path, "r", encoding="utf-8") as f:
        oi_records = json.load(f)
    oi_map = {d["patient_id"]: d["oi"] for d in oi_records}

    all_features = []
    all_labels = []
    sample_names = []
    used_pids = set()
    skipped_no_oi = []
    skipped_error = []

    for d in metrics_dirs:
        p = Path(d)
        if not p.exists():
            print(f"警告：找不到目錄 {p}，略過")
            continue
        for json_file in sorted(p.glob("*_metrics.json")):
            pid = json_file.stem.split("_")[0]
            if pid in used_pids:
                continue  # 同一病人只取一次
            if pid not in oi_map:
                skipped_no_oi.append(pid)
                continue
            try:
                data = load_json_metrics(str(json_file))
                feats = extract_features_from_json(data)
            except Exception as e:
                skipped_error.append((pid, str(e)))
                continue

            oi_val = oi_map[pid]
            label = 1 if oi_val >= threshold else 0
            all_features.append(feats)
            all_labels.append(label)
            sample_names.append(json_file.stem.replace("_metrics", ""))
            used_pids.add(pid)

    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # 統計母體 (all/) 與缺 metrics 的病人
    missing_metrics = []
    all_pids = []
    if all_dir and Path(all_dir).exists():
        all_pids = sorted({f.name.split("_")[0] for f in Path(all_dir).glob("*.nii.gz")})
        missing_metrics = sorted(set(all_pids) - used_pids)

    n_normal = int((labels == 0).sum())
    n_abnormal = int((labels == 1).sum())

    print("\n" + "=" * 70)
    print("資料載入摘要 (以 oi 門檻標記)")
    print("=" * 70)
    print(f"oi 門檻: oi >= {threshold} -> Abnormal(1)，oi < {threshold} -> Normal(0)")
    if all_pids:
        print(f"all/ 影像母體: {len(all_pids)} 筆")
    print(f"納入模型 (有 metrics + oi): {len(labels)} 筆")
    print(f"  Normal (oi <  {threshold}): {n_normal}")
    print(f"  Abnormal (oi >= {threshold}): {n_abnormal}")
    if missing_metrics:
        print(
            f"\n略過 {len(missing_metrics)} 筆 (只有影像、無 metrics，需先跑分割 pipeline):"
        )
        print("  " + ", ".join(missing_metrics))
    if skipped_no_oi:
        print(f"\n略過 {len(skipped_no_oi)} 筆 (metrics 存在但 oi_processed.json 無對應 oi):")
        print("  " + ", ".join(skipped_no_oi))
    if skipped_error:
        print(f"\n略過 {len(skipped_error)} 筆 (特徵擷取錯誤):")
        for pid, err in skipped_error:
            print(f"  {pid}: {err}")
    print("=" * 70)

    info = {
        "threshold": threshold,
        "n_total_images": len(all_pids),
        "n_used": int(len(labels)),
        "n_normal": n_normal,
        "n_abnormal": n_abnormal,
        "missing_metrics": missing_metrics,
        "skipped_no_oi": skipped_no_oi,
        "skipped_error": [pid for pid, _ in skipped_error],
        "label_map": {
            name: int(lab) for name, lab in zip(sample_names, labels.tolist())
        },
    }

    return features, labels, sample_names, info


def main(args):
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"使用設備: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "aeropath" if args.param_subdir != "param" else "trachea"
    output_dir = os.path.join(args.output_dir, f"training_oi_{tag}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"輸出目錄: {output_dir}")

    sub = args.param_subdir  # "param" (原始 Trachea WA%) 或 "param_aeropath" (AeroPath WA%)
    metrics_dirs = [
        os.path.join(args.data_dir, "NormalDataset", sub),
        os.path.join(args.data_dir, "AbnormalDataset", sub),
        os.path.join(args.data_dir, "TestDataset", sub),
        # generate_missing_metrics.py 補跑 all/ 缺 metrics 病人後的輸出
        os.path.join(args.data_dir, "AllExtraDataset", sub),
    ]
    oi_json_path = os.path.join(args.data_dir, args.oi_json)
    all_dir = os.path.join(args.data_dir, "all")

    print("\n" + "=" * 70)
    print("COPD 二元分類訓練 (oi 門檻標記, 5-Fold Cross-Validation)")
    print("=" * 70)

    print("\n步驟 1/3: 載入資料 (以 oi 門檻標記)...")
    features, labels, sample_names, info = load_data_by_oi(
        metrics_dirs, oi_json_path, all_dir=all_dir, threshold=args.threshold
    )

    if len(labels) == 0:
        raise SystemExit("沒有可用樣本，請確認 metrics 目錄與 oi_processed.json")

    print("\n步驟 2/3: 執行 5-Fold 交叉驗證訓練...")
    fold_results = train_kfold(
        features=features,
        labels=labels,
        n_folds=args.n_folds,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_sizes=[args.hidden1, args.hidden2, args.hidden3],
        dropout_rate=args.dropout,
        early_stopping_patience=args.patience,
        device=str(device),
        save_dir=output_dir,
        random_state=args.seed,
        sample_names=sample_names,
    )

    print("\n步驟 3/3: 生成視覺化圖表...")
    plot_kfold_results(fold_results, save_dir=output_dir)

    config_and_results = {
        "configuration": {
            "labeling": {
                "source": args.oi_json,
                "rule": f"oi >= {args.threshold} -> Abnormal(1), else Normal(0)",
                "threshold": args.threshold,
            },
            "data_summary": info,
            "model_architecture": {
                "input_size": 12,
                "hidden_sizes": [args.hidden1, args.hidden2, args.hidden3],
                "output_size": 2,
                "dropout_rate": args.dropout,
            },
            "training_params": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "early_stopping_patience": args.patience,
                "n_folds": args.n_folds,
                "random_seed": args.seed,
            },
            "feature_names": FEATURE_NAMES,
        },
        "kfold_results": {
            "fold_metrics": fold_results["fold_metrics"],
            "summary": fold_results["summary"],
            "best_fold": fold_results["best_fold"],
            "best_fold_accuracy": fold_results["best_fold_accuracy"],
        },
    }

    results_path = os.path.join(output_dir, "kfold_training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(config_and_results, f, indent=2, ensure_ascii=False)

    misclassified_path = os.path.join(output_dir, "misclassified_samples.json")
    misclassified_data = {
        "summary": {
            "total_fn": sum(f["fn_count"] for f in fold_results["fold_metrics"]),
            "total_fp": sum(f["fp_count"] for f in fold_results["fold_metrics"]),
        },
        "per_fold": fold_results["misclassified_samples"],
    }
    with open(misclassified_path, "w", encoding="utf-8") as f:
        json.dump(misclassified_data, f, indent=2, ensure_ascii=False)

    print(f"\n訓練結果已保存至: {results_path}")
    print(f"錯誤分類樣本 (FN/FP) 已保存至: {misclassified_path}")
    print("\n" + "=" * 70)
    print("訓練完成！")
    print("=" * 70)
    print(f"\n所有檔案已保存至: {output_dir}")
    print(f"推薦使用最佳 Fold (Fold {fold_results['best_fold']}) 的模型進行預測")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 oi 門檻標記 + 5-Fold 交叉驗證訓練 COPD 二元分類神經網路"
    )

    parser.add_argument(
        "--data-dir", type=str, default=".", help="資料集根目錄 (預設: 當前目錄)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models", help="輸出目錄 (預設: models)"
    )
    parser.add_argument(
        "--oi-json",
        type=str,
        default="oi_processed.json",
        help="oi 標籤來源 JSON (預設: oi_processed.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="oi 門檻；oi >= threshold 視為 Abnormal (預設: 3.0)",
    )
    parser.add_argument(
        "--param-subdir",
        type=str,
        default="param",
        help="每個資料集底下的 metrics 子目錄 (param=Trachea WA%%, param_aeropath=AeroPath WA%%)",
    )

    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--hidden3", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
