"""
COPD分類模型訓練腳本
使用5-Fold交叉驗證訓練COPD分類神經網路
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
from pathlib import Path
from datetime import datetime
from copd_classifier import (
    COPDClassifier,
    COPDDataset,
    load_copd_data,
    train_kfold,
    plot_kfold_results,
)


def main(args):
    # 設置隨機種子以保證可重複性
    torch.manual_seed(args.seed)

    # 設定設備
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"使用設備: {device}")

    # 創建基於日期的輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "aeropath" if args.param_subdir != "param" else "trachea"
    output_dir = os.path.join(args.output_dir, f"training_normabn_{tag}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"輸出目錄: {output_dir}")

    # 資料路徑 (使用 unified_pipeline.py 生成的標準目錄結構)
    normal_param_dir = os.path.join(args.data_dir, "NormalDataset", args.param_subdir)
    abnormal_param_dir = os.path.join(args.data_dir, "AbnormalDataset", args.param_subdir)

    print("\n" + "=" * 70)
    print("COPD分類模型訓練 (5-Fold Cross-Validation)")
    print("=" * 70)

    # 載入資料
    print("\n步驟 1/3: 載入資料...")
    features, labels, feature_names, sample_names = load_copd_data(
        normal_param_dir, abnormal_param_dir
    )

    # 使用K-Fold交叉驗證訓練
    print("\n步驟 2/3: 執行5-Fold交叉驗證訓練...")
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

    # 生成視覺化圖表
    print("\n步驟 3/3: 生成視覺化圖表...")
    plot_kfold_results(fold_results, save_dir=output_dir)

    # 保存訓練配置和結果
    config_and_results = {
        "configuration": {
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
            "feature_names": feature_names,
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

    # 保存錯誤分類樣本詳細資訊 (FN/FP)
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
    print(f"\n📁 所有檔案已保存至: {output_dir}")
    print(f"\n保存的檔案:")
    for fold in range(1, args.n_folds + 1):
        print(f"  - Fold {fold} 模型: {output_dir}/fold_{fold}_model.pth")
        print(f"  - Fold {fold} Scaler: {output_dir}/fold_{fold}_scaler.pkl")
    print(f"  - 訓練結果: {results_path}")
    print(f"  - 圖表:")
    print(f"         {output_dir}/kfold_metrics_comparison.png")
    print(f"         {output_dir}/kfold_confusion_matrix.png")
    print(f"         {output_dir}/kfold_roc_curve.png")
    print(f"         {output_dir}/kfold_metrics_boxplot.png")
    print(f"         {args.output_dir}/kfold_metrics_comparison.png")
    print(f"         {args.output_dir}/kfold_confusion_matrix.png")
    print(f"         {args.output_dir}/kfold_roc_curve.png")
    print(f"         {args.output_dir}/kfold_metrics_boxplot.png")
    print(f"\n推薦使用最佳Fold (Fold {fold_results['best_fold']}) 的模型進行預測")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用5-Fold交叉驗證訓練COPD分類神經網路"
    )

    # 資料路徑
    parser.add_argument(
        "--data-dir", type=str, default=".", help="資料集根目錄 (預設: 當前目錄)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models", help="輸出目錄 (預設: models)"
    )
    parser.add_argument(
        "--param-subdir",
        type=str,
        default="param",
        help="metrics 子目錄 (param=Trachea WA%%, param_aeropath=AeroPath WA%%)",
    )

    # 模型架構
    parser.add_argument(
        "--hidden1", type=int, default=64, help="第一層隱藏層神經元數 (預設: 64)"
    )
    parser.add_argument(
        "--hidden2", type=int, default=32, help="第二層隱藏層神經元數 (預設: 32)"
    )
    parser.add_argument(
        "--hidden3", type=int, default=16, help="第三層隱藏層神經元數 (預設: 16)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout機率 (預設: 0.3)"
    )

    # 訓練參數
    parser.add_argument(
        "--epochs", type=int, default=100, help="每個Fold的訓練輪數 (預設: 100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch大小 (預設: 16)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="學習率 (預設: 0.001)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="權重衰減 (預設: 1e-5)"
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping耐心值 (預設: 15)"
    )

    # K-Fold參數
    parser.add_argument("--n-folds", type=int, default=5, help="交叉驗證折數 (預設: 5)")

    # 其他
    parser.add_argument("--seed", type=int, default=42, help="隨機種子 (預設: 42)")
    parser.add_argument("--no-cuda", action="store_true", help="不使用CUDA即使可用")

    args = parser.parse_args()

    # 建立輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
