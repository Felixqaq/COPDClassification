"""
COPD Classification Neural Network
使用全連接神經網路對COPD患者進行正常/異常分類
Input: 12個特徵參數
Output: 2個類別 (0: Normal, 1: Abnormal)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings("ignore")


class COPDDataset(Dataset):
    """COPD資料集類別"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class COPDClassifier(nn.Module):
    """
    COPD分類神經網路模型
    Architecture: Input(12) -> Dense(64) -> Dense(32) -> Dense(16) -> Output(2)
    """

    def __init__(
        self, input_size=12, hidden_sizes=[64, 32, 16], output_size=2, dropout_rate=0.3
    ):
        super(COPDClassifier, self).__init__()

        # Input layer to first hidden layer
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second hidden layer
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)

        # Third hidden layer
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.dropout3 = nn.Dropout(dropout_rate * 0.7)  # Lower dropout for last layer

        # Output layer
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)

        # Activation function
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        # Output layer
        x = self.fc4(x)

        return x

    def predict_proba(self, x):
        """返回預測機率"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = self.softmax(logits)
        return probs


def load_json_metrics(json_path: str) -> Dict:
    """載入JSON格式的metrics檔案"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_features_from_json(json_data: Dict) -> List[float]:
    """
    從JSON資料中提取12個特徵
    特徵順序：
    1. Total_Emphysema_Percent
    2-6. 5個肺葉的emphysema百分比
    7. SVV_Percent
    8. WA_Percent
    9. Vessel_Density_Percent
    10. Airway_Lung_Ratio_Percent
    11. Total_Lung_Volume_ml
    12. PA_Diameter_mm
    """
    features = []

    # 1. Total emphysema percent
    features.append(json_data["emphysema"]["total_emphysema_percent"])

    # 2-6. Lobe emphysema percentages (按照label順序: 1,2,3,4,5)
    lobes = json_data["emphysema"]["lobes"]
    for lobe_id in [1, 2, 3, 4, 5]:
        # 支援整數或字串 key
        if lobe_id in lobes:
            features.append(lobes[lobe_id]["emphysema_percent"])
        elif str(lobe_id) in lobes:
            features.append(lobes[str(lobe_id)]["emphysema_percent"])
        else:
            features.append(0.0)  # 如果找不到該肺葉，使用預設值

    # 7. SVV%
    features.append(json_data["SVV%"]["value"])

    # 8. WA%
    features.append(json_data["WA%"]["value"])

    # 9. Vessel Density%
    features.append(json_data["Vessel_Density%"])

    # 10. Airway Lung Ratio%
    features.append(json_data["Airway_Lung_Ratio%"])

    # 11. Total Lung Volume (ml)
    total_volume_ml = json_data["total_lung_volume_mm3"] / 1000.0
    features.append(total_volume_ml)

    # 12. PA Diameter (mm)
    features.append(json_data["PA_Diameter_mm"])

    return features


def load_copd_data(
    normal_dir: str, abnormal_dir: str
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    載入COPD資料集

    Args:
        normal_dir: 正常資料集的param目錄路徑
        abnormal_dir: 異常資料集的param目錄路徑

    Returns:
        features: (N, 12) numpy array
        labels: (N,) numpy array, 0=Normal, 1=Abnormal
        feature_names: 特徵名稱列表
        sample_names: 樣本名稱列表
    """
    feature_names = [
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

    all_features = []
    all_labels = []
    all_sample_names = []

    # 載入正常資料 (label=0)
    normal_param_dir = Path(normal_dir)
    print(f"載入正常資料集從: {normal_param_dir}")
    normal_files = sorted(normal_param_dir.glob("*_metrics.json"))

    for json_file in normal_files:
        try:
            json_data = load_json_metrics(str(json_file))
            features = extract_features_from_json(json_data)
            all_features.append(features)
            all_labels.append(0)  # Normal
            sample_name = json_file.stem.replace("_metrics", "")
            all_sample_names.append(sample_name)
        except Exception as e:
            print(f"警告：無法載入 {json_file.name}: {e}")

    print(f"正常樣本數: {len([l for l in all_labels if l == 0])}")

    # 載入異常資料 (label=1)
    abnormal_param_dir = Path(abnormal_dir)
    print(f"載入異常資料集從: {abnormal_param_dir}")
    abnormal_files = sorted(abnormal_param_dir.glob("*_metrics.json"))

    for json_file in abnormal_files:
        try:
            json_data = load_json_metrics(str(json_file))
            features = extract_features_from_json(json_data)
            all_features.append(features)
            all_labels.append(1)  # Abnormal
            sample_name = json_file.stem.replace("_metrics", "")
            all_sample_names.append(sample_name)
        except Exception as e:
            print(f"警告：無法載入 {json_file.name}: {e}")

    print(f"異常樣本數: {len([l for l in all_labels if l == 1])}")

    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    print(f"\n總樣本數: {len(labels_array)}")
    print(f"特徵維度: {features_array.shape}")

    return features_array, labels_array, feature_names, all_sample_names


def preprocess_data(
    features: np.ndarray,
    labels: np.ndarray,
    test_size=0.2,
    val_size=0.2,
    random_state=42,
):
    """
    資料預處理和分割

    Args:
        features: 特徵數據
        labels: 標籤
        test_size: 測試集比例
        val_size: 驗證集比例（從訓練集中分割）
        random_state: 隨機種子

    Returns:
        訓練集、驗證集、測試集的features和labels，以及scaler
    """
    # 先分割出測試集
    X_temp, X_test, y_temp, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # 從剩餘資料中分割出驗證集
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp,
    )

    # 標準化特徵（只用訓練集的統計量）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n資料分割:")
    print(f"訓練集: {len(y_train)} 樣本")
    print(f"驗證集: {len(y_val)} 樣本")
    print(f"測試集: {len(y_test)} 樣本")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=100,
    device="cpu",
    early_stopping_patience=15,
    save_path="models/best_model.pth",
):
    """
    訓練模型

    Args:
        model: 神經網路模型
        train_loader: 訓練資料loader
        val_loader: 驗證資料loader
        criterion: 損失函數
        optimizer: 優化器
        scheduler: 學習率調度器
        num_epochs: 訓練輪數
        device: 設備 (cpu/cuda)
        early_stopping_patience: Early stopping的耐心值
        save_path: 模型保存路徑

    Returns:
        訓練歷史記錄
    """
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float("inf")
    patience_counter = 0

    # 建立模型保存目錄
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\n開始訓練... (設備: {device})")
    print("=" * 70)

    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # 驗證階段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        # 記錄歷史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # 調整學習率
        scheduler.step(val_loss)

        # 打印進度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                save_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

    print("=" * 70)
    print(f"訓練完成！最佳模型已保存至: {save_path}")
    print(f"最佳驗證損失: {best_val_loss:.4f}")

    return history


def evaluate_model(
    model, test_loader, device="cpu", class_names=["Normal", "Abnormal"]
):
    """
    評估模型性能

    Args:
        model: 訓練好的模型
        test_loader: 測試資料loader
        device: 設備
        class_names: 類別名稱

    Returns:
        評估指標字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(
                probs[:, 1].cpu().numpy()
            )  # Probability of class 1 (Abnormal)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 計算指標
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = None

    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "true_labels": all_labels,
        "probabilities": all_probs,
    }

    print("\n" + "=" * 70)
    print("測試集評估結果:")
    print("=" * 70)
    print(f"準確率 (Accuracy):  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"精確率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分數 (F1-Score):  {f1:.4f}")
    if auc is not None:
        print(f"AUC分數:            {auc:.4f}")
    print("\n混淆矩陣:")
    print(f"                預測: {class_names[0]:>10}  {class_names[1]:>10}")
    for i, row in enumerate(cm):
        print(f"實際: {class_names[i]:>10}  {row[0]:>10}  {row[1]:>10}")
    print("=" * 70)

    return metrics


def plot_training_history(history, save_path="results/training_history.png"):
    """繪製訓練歷史曲線"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss曲線
    ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy曲線
    ax2.plot(history["train_acc"], label="Train Accuracy", linewidth=2)
    ax2.plot(history["val_acc"], label="Validation Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n訓練歷史圖表已保存至: {save_path}")
    plt.close()


def plot_confusion_matrix(
    cm, class_names=["Normal", "Abnormal"], save_path="results/confusion_matrix.png"
):
    """繪製混淆矩陣"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
        annot_kws={"size": 14},
    )
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"混淆矩陣已保存至: {save_path}")
    plt.close()


def plot_roc_curve(true_labels, probabilities, save_path="results/roc_curve.png"):
    """繪製ROC曲線"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
    auc_score = roc_auc_score(true_labels, probabilities)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    plt.title("ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"ROC曲線已保存至: {save_path}")
    plt.close()


def train_kfold(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    hidden_sizes: List[int] = [64, 32, 16],
    dropout_rate: float = 0.3,
    early_stopping_patience: int = 15,
    device: str = "cpu",
    save_dir: str = "models",
    random_state: int = 42,
    sample_names: List[str] | None = None,
) -> Dict:
    """
    使用K-Fold交叉驗證訓練模型

    Args:
        features: 特徵數據 (N, 12)
        labels: 標籤 (N,)
        n_folds: 折數 (預設: 5)
        num_epochs: 每個fold的訓練輪數
        batch_size: Batch大小
        learning_rate: 學習率
        weight_decay: 權重衰減
        hidden_sizes: 隱藏層大小
        dropout_rate: Dropout機率
        early_stopping_patience: Early stopping耐心值
        device: 設備 (cpu/cuda)
        save_dir: 模型保存目錄
        random_state: 隨機種子
        sample_names: 樣本名稱列表 (用於記錄錯誤分類樣本)

    Returns:
        包含所有fold結果的字典
    """
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_results = {
        "fold_metrics": [],
        "all_predictions": [],
        "all_true_labels": [],
        "all_probabilities": [],
        "best_fold": None,
        "best_fold_accuracy": 0.0,
        "misclassified_samples": [],  # FN/FP記錄
    }

    # 用於計算總體指標
    all_predictions = []
    all_true_labels = []
    all_probabilities = []

    print(f"\n{'=' * 70}")
    print(f"開始 {n_folds}-Fold 交叉驗證訓練")
    print(f"{'=' * 70}")

    # 建立保存目錄
    os.makedirs(save_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(features, labels)):
        print(f"\n{'=' * 70}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'=' * 70}")
        print(f"訓練集大小: {len(train_idx)}, 驗證集大小: {len(val_idx)}")

        # 分割資料
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # 標準化特徵（只用訓練集的統計量）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 建立資料集和DataLoader
        train_dataset = COPDDataset(X_train_scaled, y_train)
        val_dataset = COPDDataset(X_val_scaled, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 建立模型
        model = COPDClassifier(
            input_size=12,
            hidden_sizes=hidden_sizes,
            output_size=2,
            dropout_rate=dropout_rate,
        ).to(device)

        # 定義損失函數和優化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        # 訓練模型
        model_save_path = os.path.join(save_dir, f"fold_{fold + 1}_model.pth")
        history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs=num_epochs,
            device=device,
            early_stopping_patience=early_stopping_patience,
            save_path=model_save_path,
        )

        # 載入最佳模型進行評估
        checkpoint = torch.load(
            model_save_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        # 評估該fold
        metrics = evaluate_model(model, val_loader, device=device)

        # 保存scaler
        import joblib

        scaler_path = os.path.join(save_dir, f"fold_{fold + 1}_scaler.pkl")
        joblib.dump(scaler, scaler_path)

        # 記錄錯誤分類樣本 (FN/FP)
        predictions = metrics["predictions"]
        true_labels = metrics["true_labels"]
        fold_misclassified = {
            "fold": fold + 1,
            "false_negatives": [],
            "false_positives": [],
        }

        for i, (pred, true_label) in enumerate(zip(predictions, true_labels)):
            if pred != true_label:
                sample_idx = val_idx[i]
                sample_name = (
                    sample_names[sample_idx] if sample_names else f"sample_{sample_idx}"
                )
                true_class = "Abnormal" if true_label == 1 else "Normal"
                pred_class = "Abnormal" if pred == 1 else "Normal"
                prob = metrics["probabilities"][i]

                error_info = {
                    "sample_name": sample_name,
                    "sample_idx": int(sample_idx),
                    "true_label": true_class,
                    "predicted": pred_class,
                    "probability": float(prob),
                }

                if true_label == 1 and pred == 0:  # FN: 實際是Abnormal但預測Normal
                    fold_misclassified["false_negatives"].append(error_info)
                elif true_label == 0 and pred == 1:  # FP: 實際是Normal但預測Abnormal
                    fold_misclassified["false_positives"].append(error_info)

        fold_results["misclassified_samples"].append(fold_misclassified)

        # 記錄結果
        fold_result = {
            "fold": fold + 1,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "auc": metrics["auc"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "fn_count": len(fold_misclassified["false_negatives"]),
            "fp_count": len(fold_misclassified["false_positives"]),
        }
        fold_results["fold_metrics"].append(fold_result)

        # 累積預測結果
        all_predictions.extend(metrics["predictions"])
        all_true_labels.extend(metrics["true_labels"])
        all_probabilities.extend(metrics["probabilities"])

        # 追蹤最佳fold
        if metrics["accuracy"] > fold_results["best_fold_accuracy"]:
            fold_results["best_fold_accuracy"] = metrics["accuracy"]
            fold_results["best_fold"] = fold + 1

    # 計算總體指標
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    all_probabilities = np.array(all_probabilities)

    fold_results["all_predictions"] = all_predictions.tolist()
    fold_results["all_true_labels"] = all_true_labels.tolist()
    fold_results["all_probabilities"] = all_probabilities.tolist()

    # 計算彙總統計
    accuracies = [f["accuracy"] for f in fold_results["fold_metrics"]]
    precisions = [f["precision"] for f in fold_results["fold_metrics"]]
    recalls = [f["recall"] for f in fold_results["fold_metrics"]]
    f1_scores = [f["f1_score"] for f in fold_results["fold_metrics"]]
    aucs = [f["auc"] for f in fold_results["fold_metrics"] if f["auc"] is not None]

    fold_results["summary"] = {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_precision": float(np.mean(precisions)),
        "std_precision": float(np.std(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "std_recall": float(np.std(recalls)),
        "mean_f1_score": float(np.mean(f1_scores)),
        "std_f1_score": float(np.std(f1_scores)),
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "std_auc": float(np.std(aucs)) if aucs else None,
        "overall_accuracy": float(accuracy_score(all_true_labels, all_predictions)),
        "overall_precision": float(
            precision_score(all_true_labels, all_predictions, average="binary")
        ),
        "overall_recall": float(
            recall_score(all_true_labels, all_predictions, average="binary")
        ),
        "overall_f1_score": float(
            f1_score(all_true_labels, all_predictions, average="binary")
        ),
        "overall_confusion_matrix": confusion_matrix(
            all_true_labels, all_predictions
        ).tolist(),
    }

    try:
        fold_results["summary"]["overall_auc"] = float(
            roc_auc_score(all_true_labels, all_probabilities)
        )
    except:
        fold_results["summary"]["overall_auc"] = None

    # 打印總結
    print(f"\n{'=' * 70}")
    print(f"{n_folds}-Fold 交叉驗證結果總結")
    print(f"{'=' * 70}")
    print(f"各Fold準確率: {[f'{acc:.4f}' for acc in accuracies]}")
    print(f"\n平均指標 (± 標準差):")
    print(
        f"  準確率 (Accuracy):  {fold_results['summary']['mean_accuracy']:.4f} ± {fold_results['summary']['std_accuracy']:.4f}"
    )
    print(
        f"  精確率 (Precision): {fold_results['summary']['mean_precision']:.4f} ± {fold_results['summary']['std_precision']:.4f}"
    )
    print(
        f"  召回率 (Recall):    {fold_results['summary']['mean_recall']:.4f} ± {fold_results['summary']['std_recall']:.4f}"
    )
    print(
        f"  F1分數 (F1-Score):  {fold_results['summary']['mean_f1_score']:.4f} ± {fold_results['summary']['std_f1_score']:.4f}"
    )
    if fold_results["summary"]["mean_auc"] is not None:
        print(
            f"  AUC分數:            {fold_results['summary']['mean_auc']:.4f} ± {fold_results['summary']['std_auc']:.4f}"
        )
    print(f"\n總體指標 (所有Fold合併):")
    print(f"  準確率: {fold_results['summary']['overall_accuracy']:.4f}")
    print(f"  精確率: {fold_results['summary']['overall_precision']:.4f}")
    print(f"  召回率: {fold_results['summary']['overall_recall']:.4f}")
    print(f"  F1分數: {fold_results['summary']['overall_f1_score']:.4f}")
    if fold_results["summary"]["overall_auc"] is not None:
        print(f"  AUC分數: {fold_results['summary']['overall_auc']:.4f}")
    print(
        f"\n最佳Fold: Fold {fold_results['best_fold']} (準確率: {fold_results['best_fold_accuracy']:.4f})"
    )
    print(f"{'=' * 70}")

    return fold_results


def plot_kfold_results(fold_results: Dict, save_dir: str = "results"):
    """
    繪製K-Fold交叉驗證結果圖表

    Args:
        fold_results: train_kfold返回的結果字典
        save_dir: 圖表保存目錄
    """
    os.makedirs(save_dir, exist_ok=True)

    n_folds = len(fold_results["fold_metrics"])

    # 1. 各Fold指標比較圖
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    folds = [f["fold"] for f in fold_results["fold_metrics"]]
    accuracies = [f["accuracy"] for f in fold_results["fold_metrics"]]
    precisions = [f["precision"] for f in fold_results["fold_metrics"]]
    recalls = [f["recall"] for f in fold_results["fold_metrics"]]
    f1_scores = [f["f1_score"] for f in fold_results["fold_metrics"]]

    metrics_data = [
        (accuracies, "Accuracy", "steelblue"),
        (precisions, "Precision", "forestgreen"),
        (recalls, "Recall", "darkorange"),
        (f1_scores, "F1-Score", "crimson"),
    ]

    for ax, (values, name, color) in zip(axes.flatten(), metrics_data):
        bars = ax.bar(folds, values, color=color, alpha=0.7, edgecolor="black")
        ax.axhline(
            y=np.mean(values),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(values):.4f}",
        )
        ax.set_xlabel("Fold", fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(f"{name} per Fold", fontsize=12, fontweight="bold")
        ax.set_xticks(folds)
        ax.set_ylim([0, 1])
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)

        # 在柱狀圖上顯示數值
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle(
        f"{n_folds}-Fold Cross-Validation Results", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "kfold_metrics_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"各Fold指標比較圖已保存至: {save_dir}/kfold_metrics_comparison.png")
    plt.close()

    # 2. 總體混淆矩陣
    cm = np.array(fold_results["summary"]["overall_confusion_matrix"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Abnormal"],
        yticklabels=["Normal", "Abnormal"],
        cbar_kws={"label": "Count"},
        annot_kws={"size": 16},
    )
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.title(
        f"{n_folds}-Fold Cross-Validation Overall Confusion Matrix",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "kfold_confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"總體混淆矩陣已保存至: {save_dir}/kfold_confusion_matrix.png")
    plt.close()

    # 3. 總體ROC曲線
    all_true_labels = np.array(fold_results["all_true_labels"])
    all_probabilities = np.array(fold_results["all_probabilities"])

    if fold_results["summary"]["overall_auc"] is not None:
        fpr, tpr, _ = roc_curve(all_true_labels, all_probabilities)
        auc_score = fold_results["summary"]["overall_auc"]

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            color="steelblue",
            label=f"Overall ROC Curve (AUC = {auc_score:.3f})",
        )
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
        plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        plt.ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        plt.title(
            f"{n_folds}-Fold Cross-Validation Overall ROC Curve",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "kfold_roc_curve.png"), dpi=300, bbox_inches="tight"
        )
        print(f"總體ROC曲線已保存至: {save_dir}/kfold_roc_curve.png")
        plt.close()

    # 4. 指標彙總箱形圖
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_for_boxplot = {
        "Accuracy": accuracies,
        "Precision": precisions,
        "Recall": recalls,
        "F1-Score": f1_scores,
    }

    positions = list(range(1, len(metrics_for_boxplot) + 1))
    box_data = list(metrics_for_boxplot.values())

    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
    colors = ["steelblue", "forestgreen", "darkorange", "crimson"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(metrics_for_boxplot.keys(), fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"{n_folds}-Fold Cross-Validation Metrics Distribution",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "kfold_metrics_boxplot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"指標分佈箱形圖已保存至: {save_dir}/kfold_metrics_boxplot.png")
    plt.close()
