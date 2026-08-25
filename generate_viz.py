"""
COPD 視覺化圖表生成腳本
從已有的參數檔案（param/）生成統計視覺化圖表

使用方式:
    python generate_viz.py -i NormalDataset -o NormalDataset/visualizations
    python generate_viz.py -i AbnormalDataset -o AbnormalDataset/visualizations
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Matplotlib settings
plt.rcParams["axes.unicode_minus"] = False

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def load_metrics_to_dataframe(param_dir: Path) -> pd.DataFrame:
    """從 param 目錄載入所有 metrics JSON 檔案為 DataFrame"""
    data = []

    for json_file in sorted(param_dir.glob("*_metrics.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        row = {"Filename": json_file.stem.replace("_metrics", "")}

        # Emphysema
        emp = metrics.get("emphysema", {})
        grading = emp.get("grading", {})
        if grading:
            row["Total_Emphysema%"] = grading.get("mild", {}).get("percent", 0)
            row["Emphysema_Moderate%"] = grading.get("moderate", {}).get("percent", 0)
            row["Emphysema_Severe%"] = grading.get("severe", {}).get("percent", 0)
        else:
            row["Total_Emphysema%"] = emp.get("total_emphysema_percent", 0)

        # 各肺葉
        lobes = emp.get("lobes", {})
        lobe_map = {1: "LUL", 2: "LLL", 3: "RUL", 4: "RML", 5: "RLL"}
        for lobe_id, lobe_abbr in lobe_map.items():
            key = str(lobe_id) if str(lobe_id) in lobes else lobe_id
            if key in lobes:
                row[f"{lobe_abbr}_Emphysema%"] = lobes[key].get("emphysema_percent", 0)

        # 其他指標
        row["WA%"] = metrics.get("WA%", {}).get("value", 0) if metrics.get("WA%") else 0
        row["SVV%"] = (
            metrics.get("SVV%", {}).get("value", 0) if metrics.get("SVV%") else 0
        )
        row["Vessel_Density%"] = metrics.get("Vessel_Density%", 0)
        row["Airway_Lung_Ratio%"] = metrics.get("Airway_Lung_Ratio%", 0)
        row["Total_Lung_Vol_mL"] = metrics.get("total_lung_volume_mm3", 0) / 1000
        row["PA_Diameter_mm"] = metrics.get("PA_Diameter_mm", 0) or 0

        data.append(row)

    return pd.DataFrame(data)


def plot_heatmap(df: pd.DataFrame, output_path: Path):
    """生成參數熱力圖"""
    df_heatmap = df.set_index("Filename")
    df_normalized = (df_heatmap - df_heatmap.min()) / (
        df_heatmap.max() - df_heatmap.min() + 1e-8
    )

    plt.figure(figsize=(14, max(8, len(df) * 0.5)))
    sns.heatmap(
        df_normalized,
        annot=df_heatmap.round(2),
        fmt="g",
        cmap="RdYlGn_r",
        linewidths=0.5,
        cbar_kws={"label": "Normalized Value"},
        annot_kws={"size": 8},
    )
    plt.title("COPD Parameter Heatmap", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Patient", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path / "heatmap_all.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Heatmap: {output_path / 'heatmap_all.png'}")


def plot_emphysema_heatmap(df: pd.DataFrame, output_path: Path):
    """生成肺氣腫專用熱力圖"""
    emph_cols = [c for c in df.columns if "Emphysema%" in c]
    if not emph_cols:
        return

    df_emph = df[["Filename"] + emph_cols].set_index("Filename")
    df_emph.columns = [
        c.replace("_Emphysema%", "").replace("Emphysema%", "Total")
        for c in df_emph.columns
    ]

    plt.figure(figsize=(10, max(6, len(df) * 0.5)))
    sns.heatmap(
        df_emph,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        linewidths=0.5,
        cbar_kws={"label": "Emphysema %"},
        annot_kws={"size": 9},
    )
    plt.title("Emphysema Percentage Heatmap (by Lobe)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Lung Lobe", fontsize=11)
    plt.ylabel("Patient", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path / "heatmap_emphysema.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Emphysema Heatmap: {output_path / 'heatmap_emphysema.png'}")


def plot_feature_distributions(df: pd.DataFrame, output_path: Path):
    """生成特徵分布箱型圖"""
    features = [
        "Total_Emphysema%",
        "WA%",
        "SVV%",
        "Vessel_Density%",
        "PA_Diameter_mm",
        "Total_Lung_Vol_mL",
    ]
    available = [f for f in features if f in df.columns and df[f].sum() > 0]

    if len(available) < 2:
        return

    n_cols = 3
    n_rows = (len(available) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for idx, feature in enumerate(available):
        ax = axes[idx]
        data = df[feature].dropna()
        bp = ax.boxplot([data], patch_artist=True, showmeans=True, meanline=True)
        bp["boxes"][0].set_facecolor("#3498db")
        bp["boxes"][0].set_alpha(0.6)
        ax.set_ylabel(feature, fontsize=10)
        ax.set_title(f"{feature} Distribution", fontsize=11, fontweight="bold", pad=10)
        ax.grid(alpha=0.3, axis="y")
        ax.set_xticklabels([""])

    for idx in range(len(available), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path / "feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Feature Distributions: {output_path / 'feature_distributions.png'}")


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path):
    """生成特徵相關性熱力圖"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return

    df_numeric = df[numeric_cols].dropna()
    if df_numeric.empty:
        return

    corr_matrix = df_numeric.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Correlation Heatmap: {output_path / 'correlation_heatmap.png'}")


def plot_comparison_bar(df: pd.DataFrame, output_path: Path):
    """生成分類比較條形圖"""
    key_metrics = ["Total_Emphysema%", "WA%", "SVV%"]
    available = [m for m in key_metrics if m in df.columns]

    if not available:
        return

    fig, axes = plt.subplots(
        1, len(available), figsize=(5 * len(available), max(6, len(df) * 0.35))
    )
    if len(available) == 1:
        axes = [axes]

    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for idx, (metric, ax) in enumerate(zip(available, axes)):
        y_pos = np.arange(len(df))
        bars = ax.barh(y_pos, df[metric], color=colors[idx % len(colors)], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["Filename"], fontsize=8)
        ax.set_xlabel(metric)
        ax.set_title(metric, fontweight="bold")
        ax.invert_yaxis()

        for bar, value in zip(bars, df[metric]):
            ax.text(
                value + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                fontsize=7,
            )

    plt.suptitle("COPD Key Metrics Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / "comparison_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Comparison Bar Chart: {output_path / 'comparison_bar.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="COPD 視覺化圖表生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python generate_viz.py -i NormalDataset -o NormalDataset/visualizations
  python generate_viz.py -i AbnormalDataset/param -o results/viz
        """,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="資料目錄（包含 param/ 子目錄）或直接指向 param 目錄",
    )
    parser.add_argument("-o", "--output", required=True, help="輸出目錄")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 尋找 param 目錄
    param_dir = input_path / "param" if (input_path / "param").exists() else input_path

    if not param_dir.exists():
        print(f"❌ 找不到目錄: {param_dir}")
        return 1

    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📊 COPD 視覺化圖表生成")
    print("=" * 60)
    print(f"輸入目錄: {param_dir}")
    print(f"輸出目錄: {output_path}")

    # 載入資料
    df = load_metrics_to_dataframe(param_dir)

    if df.empty:
        print(f"❌ 在 {param_dir} 中找不到 *_metrics.json 檔案")
        return 1

    print(f"\n🔍 找到 {len(df)} 個參數檔案")
    print("\n📈 生成圖表...")

    # 生成各種圖表
    plot_heatmap(df, output_path)
    plot_emphysema_heatmap(df, output_path)
    plot_feature_distributions(df, output_path)
    plot_correlation_heatmap(df, output_path)
    plot_comparison_bar(df, output_path)

    print("\n" + "=" * 60)
    print("✅ 所有圖表生成完成！")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
