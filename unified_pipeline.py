"""
COPD 統一分析 Pipeline
整合肺部分割、COPD 參數計算、神經網路分類

使用方式:
    # 處理單一檔案
    python unified_pipeline.py -i patient_ct.nii.gz -o output/

    # 批次處理資料夾
    python unified_pipeline.py -i ./data/ -o ./results/ --batch

    # 跳過分割（已有分割結果）
    python unified_pipeline.py -i patient_ct.nii.gz --seg existing_seg.nii.gz -o output/
"""

import sys
import json
import argparse
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

from config import SEG_SERVER_URL, AEROPATH_URL

# 設定 matplotlib 中文支援
try:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
except:
    pass
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# 將專案路徑加入 Python 路徑
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "VesselAirwayParamTransfer"))

from copd_segmentation import COPDSegmenter
from copd_classifier import COPDClassifier, extract_features_from_json


class COPDPipeline:
    """
    COPD 統一分析 Pipeline

    功能:
    1. 肺部分割 (3D Slicer Server)
    2. COPD 參數計算 (COPDAnalyzer)
    3. 神經網路分類 (COPDClassifier)
    4. 報告生成
    """

    CLASS_NAMES = ["Normal", "Abnormal"]

    def __init__(
        self,
        model_path: str = None,
        scaler_path: str = None,
        server_url: str = SEG_SERVER_URL,
        device: str = None,
        skip_model_loading: bool = False,
    ):
        """
        初始化 Pipeline

        Args:
            model_path: 模型檔案路徑
            scaler_path: Scaler 檔案路徑
            server_url: 分割伺服器 URL
            device: 運算設備 ('cuda' / 'cpu')
            skip_model_loading: 是否跳過模型載入（用於 --no-predict 模式）
        """
        # 設定預設路徑
        if model_path is None:
            model_path = SCRIPT_DIR / "models" / "best_copd_model.pth"
        if scaler_path is None:
            scaler_path = SCRIPT_DIR / "models" / "scaler.pkl"

        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.server_url = server_url

        # 設定設備
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 載入模型和 scaler（可跳過）
        self.model = None
        self.scaler = None
        if not skip_model_loading:
            self._load_model()
            self._load_scaler()
        else:
            print("📦 跳過模型載入（--no-predict 模式）")

        # 初始化分割器
        self.segmenter = COPDSegmenter(server_url=server_url)

        # 動態載入 COPDAnalyzer
        self._load_analyzer()

    def _load_model(self):
        """載入訓練好的模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到模型檔案: {self.model_path}")

        print(f"📦 載入模型: {self.model_path}")

        # 初始化模型架構
        self.model = COPDClassifier(
            input_size=12, hidden_sizes=[64, 32, 16], output_size=2, dropout_rate=0.3
        ).to(self.device)

        # 載入權重
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"   ↳ 設備: {self.device}")

    def _load_scaler(self):
        """載入特徵標準化器"""
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"找不到 Scaler 檔案: {self.scaler_path}")

        print(f"📦 載入 Scaler: {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)

    def _load_analyzer(self):
        """動態載入 COPDAnalyzer"""
        analyzer_path = SCRIPT_DIR / "VesselAirwayParamTransfer" / "copd_analyzer.py"
        if not analyzer_path.exists():
            raise FileNotFoundError(f"找不到分析器: {analyzer_path}")

        import importlib.util

        spec = importlib.util.spec_from_file_location("copd_analyzer", analyzer_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.COPDAnalyzer = module.COPDAnalyzer

    def analyze(
        self,
        ct_path: str,
        seg_path: Optional[str] = None,
        airway_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        skip_segmentation: bool = False,
        skip_prediction: bool = False,
        verbose: bool = True,
    ) -> Dict:
        """
        執行完整分析流程

        Args:
            ct_path: CT 影像路徑 (.nii.gz)
            seg_path: 分割結果路徑（若提供則跳過分割步驟）
            airway_path: 氣道分割結果路徑（若提供則傳遞給分析器）
            output_dir: 輸出目錄
            skip_segmentation: 是否跳過分割
            skip_prediction: 是否跳過神經網路預測（只生成參數）
            verbose: 是否輸出詳細訊息

        Returns:
            分析結果字典
        """
        ct_path = Path(ct_path)

        if not ct_path.exists():
            raise FileNotFoundError(f"找不到 CT 檔案: {ct_path}")

        # 設定輸出目錄
        if output_dir is None:
            output_dir = ct_path.parent
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 建立子目錄結構
        seg_output_dir = output_dir / "inference_output"
        param_output_dir = output_dir / "param"
        report_output_dir = output_dir / "report"
        seg_output_dir.mkdir(parents=True, exist_ok=True)
        param_output_dir.mkdir(parents=True, exist_ok=True)
        report_output_dir.mkdir(parents=True, exist_ok=True)

        base_name = ct_path.stem.replace(".nii", "")

        if verbose:
            print("\n" + "=" * 70)
            print(f"🔬 COPD 分析 Pipeline")
            print(f"   輸入: {ct_path.name}")
            print("=" * 70)

        results = {
            "input_file": str(ct_path),
            "timestamp": datetime.now().isoformat(),
            "stages": {},
        }

        # ===== 階段 1: 肺部分割 =====
        if seg_path is None and not skip_segmentation:
            if verbose:
                print("\n📌 階段 1/3: 肺部分割")

            seg_output = seg_output_dir / f"seg_{ct_path.name}"
            try:
                seg_path = self.segmenter.segment(ct_path, seg_output, verbose=verbose)
                results["stages"]["segmentation"] = {
                    "status": "success",
                    "output": seg_path,
                }
            except Exception as e:
                if verbose:
                    print(f"❌ 分割失敗: {e}")
                results["stages"]["segmentation"] = {
                    "status": "failed",
                    "error": str(e),
                }
                # 如果分割失敗，嘗試繼續使用 CT 檔案（部分功能可能受限）
                seg_path = None
        else:
            if seg_path:
                if verbose:
                    print(f"\n📌 階段 1/3: 使用現有分割結果: {Path(seg_path).name}")
                results["stages"]["segmentation"] = {
                    "status": "skipped",
                    "existing": seg_path,
                }
            else:
                if verbose:
                    print("\n📌 階段 1/3: 跳過分割")
                results["stages"]["segmentation"] = {"status": "skipped"}

        # ===== 階段 2: COPD 參數計算 =====
        if verbose:
            print("\n📌 階段 2/3: COPD 參數計算")

        # 自動尋找 airway 檔案（如果未提供）
        if airway_path is None:
            airway_path = self._find_airway_file(ct_path)

        if airway_path and verbose:
            print(f"   ↳ 使用氣道分割: {Path(airway_path).name}")

        try:
            # 使用分割結果 (如果有的話)
            if seg_path and Path(seg_path).exists():
                analyzer = self.COPDAnalyzer(
                    segmentation_path=str(seg_path),
                    ct_path=str(ct_path),
                    airway_path=str(airway_path) if airway_path else None,
                )
            else:
                # 沒有分割結果，只能用 CT 做有限分析
                analyzer = self.COPDAnalyzer(
                    segmentation_path=str(ct_path),
                    ct_path=str(ct_path),
                    airway_path=str(airway_path) if airway_path else None,
                )

            metrics = analyzer.calculate_metrics()

            # 儲存 JSON 參數到 param/ 子目錄
            param_path = param_output_dir / f"{base_name}_metrics.json"
            analyzer.save_params(str(param_path))

            results["stages"]["analysis"] = {
                "status": "success",
                "metrics": metrics,
                "param_file": str(param_path),
            }

            if verbose:
                print(f"   ↳ 參數已儲存: {param_path.name}")

        except Exception as e:
            if verbose:
                print(f"❌ 參數計算失敗: {e}")
            results["stages"]["analysis"] = {"status": "failed", "error": str(e)}
            metrics = None

        # ===== 階段 3: 神經網路分類 =====
        if skip_prediction:
            if verbose:
                print("\n📌 階段 3/3: 跳過神經網路分類（--no-predict 模式）")
            results["stages"]["classification"] = {
                "status": "skipped",
                "reason": "no-predict mode",
            }
        elif verbose:
            print("\n📌 階段 3/3: 神經網路分類")

        if metrics and not skip_prediction:
            try:
                # 提取特徵
                features = extract_features_from_json(metrics)

                # 標準化
                features_scaled = self.scaler.transform([features])

                # 預測
                features_tensor = torch.FloatTensor(features_scaled).to(self.device)

                with torch.no_grad():
                    outputs = self.model(features_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)

                predicted_class = predicted.item()
                class_name = self.CLASS_NAMES[predicted_class]
                confidence = probs[0][predicted_class].item()

                results["stages"]["classification"] = {
                    "status": "success",
                    "predicted_class": predicted_class,
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "probabilities": {
                        "Normal": float(probs[0][0]),
                        "Abnormal": float(probs[0][1]),
                    },
                }

                if verbose:
                    print(f"\n   🎯 分類結果: {class_name}")
                    print(f"   📊 信心度: {confidence:.2%}")
                    print(
                        f"   📈 機率: Normal={probs[0][0]:.2%}, Abnormal={probs[0][1]:.2%}"
                    )

            except Exception as e:
                if verbose:
                    print(f"❌ 分類失敗: {e}")
                results["stages"]["classification"] = {
                    "status": "failed",
                    "error": str(e),
                }
        else:
            results["stages"]["classification"] = {
                "status": "skipped",
                "reason": "no metrics",
            }

        # ===== 生成報告到 report/ 子目錄 =====
        report_path = report_output_dir / f"{base_name}_report.md"
        self._generate_report(results, report_path, metrics)
        results["report_file"] = str(report_path)

        if verbose:
            print("\n" + "=" * 70)
            print(f"✅ 分析完成！")
            print(f"   📄 報告: {report_path}")
            print("=" * 70 + "\n")

        return results

    def _generate_airway_via_aeropath(
        self,
        ct_path: Path,
        output_dir: Path = None,
        aeropath_url: str = AEROPATH_URL,
        verbose: bool = True,
    ) -> Optional[str]:
        """
        使用 AeroPath Gradio API 生成氣道分割檔案

        Args:
            ct_path: CT 影像路徑 (.nii.gz)
            output_dir: 輸出目錄（預設為 ct_path 的 ../airway/）
            aeropath_url: AeroPath Gradio 伺服器 URL
            verbose: 是否輸出詳細訊息

        Returns:
            生成的 airway 檔案路徑，失敗則返回 None
        """
        ct_path = Path(ct_path)
        base_name = ct_path.stem.replace(".nii", "")

        # 設定輸出目錄
        if output_dir is None:
            output_dir = ct_path.parent.parent / "airway"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"   🌬️ 呼叫 AeroPath API 生成氣道分割...")
            print(f"      Server URL: {aeropath_url}")
            print(f"      輸入檔案: {ct_path.name}")

        # 嘗試使用 Gradio Client
        try:
            from gradio_client import Client, handle_file
            import shutil

            if verbose:
                print(f"      📡 連接 Gradio 伺服器...")

            client = Client(aeropath_url)

            # 呼叫 AeroPath 的 /process API
            # 根據 AeroPath 客戶端，參數名稱是 mesh_file_name
            result = client.predict(
                mesh_file_name=handle_file(str(ct_path)), api_name="/process"
            )

            if verbose:
                print(f"      ✅ AeroPath 處理完成")

            # 處理返回結果 - AeroPath 返回 (obj_path, slider_info)
            if result:
                obj_path = result[0] if isinstance(result, (tuple, list)) else result

                if verbose:
                    print(f"      � 返回 OBJ: {obj_path}")

                # AeroPath 返回的是 OBJ 檔案，不是 NIfTI
                # 需要嘗試下載 NIfTI 結果或使用 /process_batch
                if obj_path and Path(obj_path).exists():
                    # 複製 OBJ 檔案到輸出目錄
                    output_obj = output_dir / f"{base_name}_airway.obj"
                    shutil.copy(obj_path, output_obj)
                    if verbose:
                        print(f"      📁 OBJ 已儲存至: {output_obj}")

                    # 嘗試尋找對應的 NIfTI 分割結果
                    # AeroPath 可能會生成 _airway_seg.nii.gz
                    possible_nii = (
                        Path(obj_path).parent / f"{base_name}_airway_seg.nii.gz"
                    )
                    if possible_nii.exists():
                        output_nii = output_dir / f"{base_name}_airway_seg.nii.gz"
                        shutil.copy(possible_nii, output_nii)
                        if verbose:
                            print(f"      📁 NIfTI 已儲存至: {output_nii}")
                        return str(output_nii)

                    # 如果只有 OBJ，返回 OBJ 路徑（可能需要後續處理）
                    return str(output_obj)

        except ImportError:
            if verbose:
                print(f"      ⚠️ 未安裝 gradio_client")
                print(f"      � 請執行: pip install gradio_client")

        except Exception as e:
            if verbose:
                print(f"      ❌ AeroPath 處理失敗: {e}")

        return None

    def _find_airway_file(
        self, ct_path: Path, generate_if_missing: bool = True, verbose: bool = True
    ) -> Optional[str]:
        """
        自動尋找對應的 airway 分割檔案，若找不到則可透過 AeroPath 生成

        搜尋策略:
        1. 同目錄下的 ../airway/{base_name}_airway_seg.nii.gz
        2. 同名檔案但加上 _airway_seg 後綴
        3. 若 generate_if_missing=True，呼叫 AeroPath API 生成

        Args:
            ct_path: CT 影像路徑
            generate_if_missing: 若找不到是否透過 AeroPath 生成（預設 True）
            verbose: 是否輸出詳細訊息

        Returns:
            找到的 airway 檔案路徑，找不到則返回 None
        """
        ct_path = Path(ct_path)
        base_name = ct_path.stem.replace(".nii", "")

        # 策略 1: 在 ../airway/ 資料夾中尋找
        airway_dir = ct_path.parent.parent / "airway"
        if airway_dir.exists():
            # 嘗試不同的命名模式
            patterns = [
                f"{base_name}_airway_seg.nii.gz",
                f"{base_name}_airway_seg.nii",
                f"{base_name}_airway.nii.gz",
                f"{base_name}_airway.nii",
            ]

            for pattern in patterns:
                airway_file = airway_dir / pattern
                if airway_file.exists():
                    return str(airway_file)

        # 策略 2: 在同目錄下尋找
        for pattern in [
            f"{base_name}_airway_seg.nii.gz",
            f"{base_name}_airway_seg.nii",
            f"{base_name}_airway.nii.gz",
            f"{base_name}_airway.nii",
        ]:
            airway_file = ct_path.parent / pattern
            if airway_file.exists():
                return str(airway_file)

        # 策略 3: 透過 AeroPath API 生成
        if generate_if_missing:
            return self._generate_airway_via_aeropath(ct_path, verbose=verbose)

        return None

    def _generate_report(self, results: Dict, output_path: Path, metrics: Dict):
        """生成 Markdown 報告"""
        report = []

        # 標題
        report.append("# COPD 自動化分析報告\n")
        report.append(f"**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**輸入檔案**: `{Path(results['input_file']).name}`\n")
        report.append("\n---\n")

        # 分類結果
        classification = results["stages"].get("classification", {})
        if classification.get("status") == "success":
            class_name = classification["class_name"]
            confidence = classification["confidence"]
            probs = classification["probabilities"]

            report.append("## 🎯 分類結果\n")

            # 根據分類結果顯示不同樣式
            if class_name == "Normal":
                report.append(
                    f"> [!NOTE]\n> **預測結果: {class_name}** (信心度: {confidence:.1%})\n\n"
                )
            else:
                report.append(
                    f"> [!WARNING]\n> **預測結果: {class_name}** (信心度: {confidence:.1%})\n\n"
                )

            report.append("### 機率分佈\n")
            report.append(f"| 類別 | 機率 |\n")
            report.append(f"|------|------|\n")
            report.append(f"| Normal | {probs['Normal']:.2%} |\n")
            report.append(f"| Abnormal | {probs['Abnormal']:.2%} |\n\n")

        # COPD 指標
        if metrics:
            report.append("## 📊 COPD 核心指標\n")

            # Emphysema - 支援多閾值分級
            emp = metrics.get("emphysema")
            if emp:
                report.append(f"### 肺氣腫分析 (多閾值分級)\n")

                grading = emp.get("grading", {})
                if grading:
                    # 新版：顯示三級分級
                    report.append("| 嚴重程度 | 診斷閾值 | 百分比 |\n")
                    report.append("|---------|---------|--------|\n")
                    report.append(
                        f"| 💚 輕度 | HU < -950 | {grading['mild']['percent']:.2f}% |\n"
                    )
                    report.append(
                        f"| 💛 中度 | HU < -960 | {grading['moderate']['percent']:.2f}% |\n"
                    )
                    report.append(
                        f"| 🔴 重度 | HU < -970 | {grading['severe']['percent']:.2f}% |\n\n"
                    )
                else:
                    # 向後兼容舊格式
                    report.append(
                        f"- **全肺肺氣腫百分比**: {emp['total_emphysema_percent']:.2f}%\n"
                    )
                    threshold = emp.get("threshold_hu", -950)
                    report.append(f"- 診斷閾值: HU < {threshold}\n\n")

                if emp.get("lobes"):
                    report.append("**各肺葉詳細資訊**:\n\n")
                    report.append("| 肺葉 | 輕度% | 中度% | 重度% |\n")
                    report.append("|------|-------|-------|-------|\n")
                    for lobe_id, lobe_data in sorted(emp["lobes"].items()):
                        lobe_grading = lobe_data.get("grading", {})
                        if lobe_grading:
                            report.append(
                                f"| {lobe_data['name']} | "
                                f"{lobe_grading['mild']['percent']:.2f}% | "
                                f"{lobe_grading['moderate']['percent']:.2f}% | "
                                f"{lobe_grading['severe']['percent']:.2f}% |\n"
                            )
                        else:
                            report.append(
                                f"| {lobe_data['name']} | {lobe_data['emphysema_percent']:.2f}% | - | - |\n"
                            )
                    report.append("\n")

            # WA%
            wa = metrics.get("WA%")
            if wa:
                report.append(f"### 氣道分析\n")
                report.append(f"- **WA%**: {wa['value']:.2f}%\n")
                report.append(f"- 資料來源: {wa['source']}\n\n")

            # SVV%
            svv = metrics.get("SVV%")
            if svv:
                report.append(f"### 小血管分析\n")
                report.append(f"- **SVV%**: {svv['value']:.2f}%\n\n")

            # 其他指標
            report.append("### 其他指標\n")
            if metrics.get("total_lung_volume_mm3"):
                report.append(
                    f"- **總肺體積**: {metrics['total_lung_volume_mm3'] / 1000:.2f} mL\n"
                )
            if metrics.get("Vessel_Density%"):
                report.append(f"- **血管密度**: {metrics['Vessel_Density%']:.2f}%\n")
            if metrics.get("PA_Diameter_mm"):
                report.append(f"- **肺動脈直徑**: {metrics['PA_Diameter_mm']:.2f} mm\n")
            report.append("\n")

        # 處理流程狀態
        report.append("## 📋 處理流程\n")
        report.append("| 階段 | 狀態 |\n")
        report.append("|------|------|\n")

        for stage_name, stage_data in results["stages"].items():
            status = stage_data.get("status", "unknown")
            status_icon = {"success": "✅", "failed": "❌", "skipped": "⏭️"}.get(
                status, "❓"
            )
            report.append(f"| {stage_name} | {status_icon} {status} |\n")

        report.append("\n---\n")
        report.append("*Generated by COPD Unified Pipeline*\n")

        # 寫入檔案
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(report)

    def analyze_batch(
        self,
        input_dir: str,
        output_dir: str,
        skip_prediction: bool = False,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        批次分析資料夾中的所有 CT 檔案

        Args:
            input_dir: 輸入資料夾
            output_dir: 輸出資料夾
            skip_prediction: 是否跳過神經網路預測
            verbose: 是否輸出詳細訊息

        Returns:
            所有分析結果列表
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 取得所有 CT 檔案 (排除已分割的)
        ct_files = list(input_dir.glob("*.nii.gz")) + list(input_dir.glob("*.nii"))
        ct_files = [f for f in ct_files if not f.name.startswith("seg_")]

        if not ct_files:
            print(f"⚠️ 在 {input_dir} 中找不到 CT 檔案")
            return []

        print(f"\n🔍 找到 {len(ct_files)} 個檔案待處理")
        print("=" * 70)

        results = []
        for idx, ct_file in enumerate(ct_files, 1):
            print(f"\n[{idx}/{len(ct_files)}] 處理: {ct_file.name}")
            print("-" * 50)

            try:
                result = self.analyze(
                    ct_file,
                    output_dir=output_dir,
                    skip_prediction=skip_prediction,
                    verbose=verbose,
                )
                results.append(result)
            except Exception as e:
                print(f"❌ 處理失敗: {e}")
                results.append({"input_file": str(ct_file), "error": str(e)})

        # 生成彙總報告
        self._generate_summary_report(results, output_dir / "summary_report.md")

        # 生成視覺化圖表
        if verbose:
            print("\n📊 生成統計圖表...")
        self._generate_visualizations(results, output_dir, verbose=verbose)

        print("\n" + "=" * 70)
        print(f"🎉 批次處理完成！")
        print(f"   成功: {sum(1 for r in results if 'error' not in r)}/{len(ct_files)}")
        print(f"   彙總報告: {output_dir / 'summary_report.md'}")
        print(f"   統計圖表: {output_dir / 'visualizations/'}")
        print("=" * 70)

        return results

    def _generate_summary_report(self, results: List[Dict], output_path: Path):
        """生成批次處理彙總報告"""
        report = []
        report.append("# COPD 批次分析彙總報告\n")
        report.append(f"**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**處理總數**: {len(results)}\n\n")

        report.append("## 分類結果彙總\n")
        report.append("| 檔案 | 分類 | 信心度 | Emphysema% |\n")
        report.append("|------|------|--------|------------|\n")

        for result in results:
            filename = Path(result.get("input_file", "Unknown")).name

            if "error" in result:
                report.append(f"| {filename} | ❌ 錯誤 | - | - |\n")
                continue

            classification = result.get("stages", {}).get("classification", {})
            if classification.get("status") == "success":
                class_name = classification["class_name"]
                confidence = classification["confidence"]

                metrics = (
                    result.get("stages", {}).get("analysis", {}).get("metrics", {})
                )
                emp = metrics.get("emphysema", {}).get("total_emphysema_percent", "-")
                if isinstance(emp, float):
                    emp = f"{emp:.2f}%"

                icon = "🟢" if class_name == "Normal" else "🔴"
                report.append(
                    f"| {filename} | {icon} {class_name} | {confidence:.1%} | {emp} |\n"
                )
            else:
                report.append(f"| {filename} | ⚠️ 未知 | - | - |\n")

        report.append("\n---\n")
        report.append("*Generated by COPD Unified Pipeline*\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(report)

    def _generate_visualizations(
        self, results: List[Dict], output_dir: Path, verbose: bool = True
    ):
        """
        生成學術研究常用的統計視覺化圖表

        Args:
            results: 分析結果列表
            output_dir: 輸出目錄
            verbose: 是否輸出詳細訊息
        """
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # 1. 提取數據到 DataFrame
        df = self._extract_data_for_visualization(results)

        if df.empty or len(df) < 2:
            if verbose:
                print("   ⚠️ 數據不足，跳過視覺化生成")
            return

        # 2. 生成各種圖表
        try:
            self._plot_classification_summary(df, viz_dir)
            self._plot_feature_distributions(df, viz_dir)
            self._plot_correlation_heatmap(df, viz_dir)
            self._plot_confidence_analysis(df, viz_dir)
            self._plot_emphysema_analysis(df, viz_dir)

            if verbose:
                print(f"   ✅ 已生成 5 組統計圖表於 {viz_dir}")
        except Exception as e:
            if verbose:
                print(f"   ⚠️ 部分圖表生成失敗: {e}")

    def _extract_data_for_visualization(self, results: List[Dict]) -> pd.DataFrame:
        """從結果中提取數據用於視覺化"""
        data = []

        for result in results:
            if "error" in result:
                continue

            row = {"Filename": Path(result["input_file"]).stem}

            # 分類結果
            classification = result.get("stages", {}).get("classification", {})
            if classification.get("status") == "success":
                row["Predicted_Class"] = classification["class_name"]
                row["Confidence"] = classification["confidence"]
                row["Prob_Normal"] = classification["probabilities"]["Normal"]
                row["Prob_Abnormal"] = classification["probabilities"]["Abnormal"]
            else:
                continue

            # COPD 指標
            metrics = result.get("stages", {}).get("analysis", {}).get("metrics", {})
            if not metrics:
                continue

            # Emphysema - 支援多閾值分級
            emp = metrics.get("emphysema", {})
            grading = emp.get("grading", {})

            # 新版：使用三級分級
            if grading:
                row["Total_Emphysema%"] = grading.get("mild", {}).get("percent", 0)
                row["Total_Emphysema_Moderate%"] = grading.get("moderate", {}).get(
                    "percent", 0
                )
                row["Total_Emphysema_Severe%"] = grading.get("severe", {}).get(
                    "percent", 0
                )
            else:
                # 向後兼容舊格式
                row["Total_Emphysema%"] = emp.get("total_emphysema_percent", 0)
                row["Total_Emphysema_Moderate%"] = 0
                row["Total_Emphysema_Severe%"] = 0

            # 各肺葉
            lobes = emp.get("lobes", {})
            lobe_map = {1: "LUL", 2: "LLL", 3: "RUL", 4: "RML", 5: "RLL"}
            for lobe_id, lobe_abbr in lobe_map.items():
                key = str(lobe_id) if str(lobe_id) in lobes else lobe_id
                if key in lobes:
                    row[f"{lobe_abbr}_Emphysema%"] = lobes[key]["emphysema_percent"]

            # 其他指標
            row["WA%"] = metrics.get("WA%", {}).get("value", 0)
            row["SVV%"] = metrics.get("SVV%", {}).get("value", 0)
            row["Vessel_Density%"] = metrics.get("Vessel_Density%", 0)
            row["Airway_Lung_Ratio%"] = metrics.get("Airway_Lung_Ratio%", 0)
            row["Total_Lung_Vol_mL"] = metrics.get("total_lung_volume_mm3", 0) / 1000
            row["PA_Diameter_mm"] = metrics.get("PA_Diameter_mm", 0)

            data.append(row)

        return pd.DataFrame(data)

    def _plot_classification_summary(self, df: pd.DataFrame, output_dir: Path):
        """繪製分類結果總覽"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. 圓餅圖 - 分類比例
        class_counts = df["Predicted_Class"].value_counts()
        colors = ["#2ecc71", "#e74c3c"]
        axes[0].pie(
            class_counts,
            labels=class_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        axes[0].set_title(
            "Classification Distribution", fontsize=14, fontweight="bold", pad=15
        )

        # 2. 條形圖 - 每個樣本的分類
        df_sorted = df.sort_values("Confidence", ascending=False)
        colors_bar = [
            "#2ecc71" if c == "Normal" else "#e74c3c"
            for c in df_sorted["Predicted_Class"]
        ]
        axes[1].barh(
            range(len(df_sorted)), df_sorted["Confidence"], color=colors_bar, alpha=0.7
        )
        axes[1].set_yticks(range(len(df_sorted)))
        axes[1].set_yticklabels(df_sorted["Filename"], fontsize=8)
        axes[1].set_xlabel("Confidence", fontsize=11)
        axes[1].set_title(
            "Classification Confidence by Sample",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        axes[1].set_xlim([0, 1])
        axes[1].invert_yaxis()

        # 3. 信心度分布直方圖
        for class_name, color in [("Normal", "#2ecc71"), ("Abnormal", "#e74c3c")]:
            class_data = df[df["Predicted_Class"] == class_name]["Confidence"]
            if len(class_data) > 0:
                axes[2].hist(
                    class_data, bins=10, alpha=0.6, label=class_name, color=color
                )
        axes[2].set_xlabel("Confidence", fontsize=11)
        axes[2].set_ylabel("Frequency", fontsize=11)
        axes[2].set_title(
            "Confidence Distribution", fontsize=14, fontweight="bold", pad=15
        )
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "1_classification_summary.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_feature_distributions(self, df: pd.DataFrame, output_dir: Path):
        """繪製特徵分布箱型圖"""
        key_features = [
            "Total_Emphysema%",
            "WA%",
            "SVV%",
            "Vessel_Density%",
            "PA_Diameter_mm",
            "Total_Lung_Vol_mL",
        ]

        # 過濾存在的特徵
        available_features = [
            f for f in key_features if f in df.columns and df[f].sum() > 0
        ]

        if len(available_features) < 2:
            return

        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for idx, feature in enumerate(available_features):
            ax = axes[idx]

            # 箱型圖 + 小提琴圖組合
            data_to_plot = [
                df[df["Predicted_Class"] == c][feature].dropna()
                for c in ["Normal", "Abnormal"]
                if c in df["Predicted_Class"].values
            ]
            labels = [
                c for c in ["Normal", "Abnormal"] if c in df["Predicted_Class"].values
            ]

            if len(data_to_plot) > 0:
                bp = ax.boxplot(
                    data_to_plot,
                    labels=labels,
                    patch_artist=True,
                    showmeans=True,
                    meanline=True,
                )

                # 設定顏色
                colors = ["#2ecc71", "#e74c3c"]
                for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)

                ax.set_ylabel(feature, fontsize=10)
                ax.set_title(
                    f"{feature} Distribution", fontsize=11, fontweight="bold", pad=10
                )
                ax.grid(alpha=0.3, axis="y")

        # 隱藏多餘的子圖
        for idx in range(len(available_features), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(
            output_dir / "2_feature_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_correlation_heatmap(self, df: pd.DataFrame, output_dir: Path):
        """繪製特徵相關性熱力圖"""
        # 選擇數值特徵
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [
            c
            for c in numeric_cols
            if c not in ["Confidence", "Prob_Normal", "Prob_Abnormal"]
        ]

        if len(numeric_cols) < 2:
            return

        df_numeric = df[numeric_cols].dropna()

        if df_numeric.empty:
            return

        # 計算相關性矩陣
        corr_matrix = df_numeric.corr()

        # 繪製熱力圖
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
        ax.set_title(
            "Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20
        )

        plt.tight_layout()
        plt.savefig(
            output_dir / "3_correlation_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_confidence_analysis(self, df: pd.DataFrame, output_dir: Path):
        """繪製信心度與特徵關係分析"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 信心度 vs 總肺氣腫
        for class_name, color, marker in [
            ("Normal", "#2ecc71", "o"),
            ("Abnormal", "#e74c3c", "s"),
        ]:
            mask = df["Predicted_Class"] == class_name
            if mask.sum() > 0:
                axes[0, 0].scatter(
                    df[mask]["Total_Emphysema%"],
                    df[mask]["Confidence"],
                    c=color,
                    label=class_name,
                    alpha=0.6,
                    s=100,
                    marker=marker,
                )
        axes[0, 0].set_xlabel("Total Emphysema %", fontsize=11)
        axes[0, 0].set_ylabel("Confidence", fontsize=11)
        axes[0, 0].set_title(
            "Confidence vs Total Emphysema", fontsize=12, fontweight="bold", pad=10
        )
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # 2. WA% vs SVV%
        if "WA%" in df.columns and "SVV%" in df.columns:
            for class_name, color, marker in [
                ("Normal", "#2ecc71", "o"),
                ("Abnormal", "#e74c3c", "s"),
            ]:
                mask = df["Predicted_Class"] == class_name
                if mask.sum() > 0:
                    axes[0, 1].scatter(
                        df[mask]["WA%"],
                        df[mask]["SVV%"],
                        c=color,
                        label=class_name,
                        alpha=0.6,
                        s=100,
                        marker=marker,
                    )
            axes[0, 1].set_xlabel("WA%", fontsize=11)
            axes[0, 1].set_ylabel("SVV%", fontsize=11)
            axes[0, 1].set_title(
                "Airway Wall Area vs Small Vessel Volume",
                fontsize=12,
                fontweight="bold",
                pad=10,
            )
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)

        # 3. 機率分布堆疊圖
        df_sorted = df.sort_values("Prob_Abnormal")
        x = range(len(df_sorted))
        axes[1, 0].bar(
            x, df_sorted["Prob_Normal"], label="Normal", color="#2ecc71", alpha=0.7
        )
        axes[1, 0].bar(
            x,
            df_sorted["Prob_Abnormal"],
            bottom=df_sorted["Prob_Normal"],
            label="Abnormal",
            color="#e74c3c",
            alpha=0.7,
        )
        axes[1, 0].axhline(y=0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
        axes[1, 0].set_xlabel("Samples (sorted by Abnormal probability)", fontsize=11)
        axes[1, 0].set_ylabel("Probability", fontsize=11)
        axes[1, 0].set_title(
            "Classification Probability Stack", fontsize=12, fontweight="bold", pad=10
        )
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 1])

        # 4. 統計摘要表格
        axes[1, 1].axis("off")
        stats_data = []
        stats_data.append(["總樣本數", len(df)])
        stats_data.append(["Normal 樣本數", (df["Predicted_Class"] == "Normal").sum()])
        stats_data.append(
            ["Abnormal 樣本數", (df["Predicted_Class"] == "Abnormal").sum()]
        )
        stats_data.append(["平均信心度", f"{df['Confidence'].mean():.2%}"])
        stats_data.append(["", ""])
        stats_data.append(["平均肺氣腫%", f"{df['Total_Emphysema%'].mean():.2f}%"])
        if "WA%" in df.columns:
            stats_data.append(["平均 WA%", f"{df['WA%'].mean():.2f}%"])
        if "SVV%" in df.columns:
            stats_data.append(["平均 SVV%", f"{df['SVV%'].mean():.2f}%"])

        table = axes[1, 1].table(
            cellText=stats_data, cellLoc="left", colWidths=[0.6, 0.4], loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        axes[1, 1].set_title(
            "Statistical Summary", fontsize=12, fontweight="bold", pad=20
        )

        plt.tight_layout()
        plt.savefig(
            output_dir / "4_confidence_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_emphysema_analysis(self, df: pd.DataFrame, output_dir: Path):
        """繪製肺氣腫分析圖"""
        lobe_cols = [c for c in df.columns if c.endswith("_Emphysema%")]

        if not lobe_cols:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 各肺葉肺氣腫熱力圖
        df_lobes = df[["Filename"] + lobe_cols].set_index("Filename")
        sns.heatmap(
            df_lobes,
            annot=True,
            fmt=".1f",
            cmap="Reds",
            linewidths=0.5,
            cbar_kws={"label": "Emphysema %"},
            ax=axes[0, 0],
        )
        axes[0, 0].set_title(
            "Emphysema Distribution by Lobe", fontsize=12, fontweight="bold", pad=10
        )
        axes[0, 0].set_xlabel("")
        axes[0, 0].set_ylabel("Sample", fontsize=10)

        # 2. 各肺葉平均肺氣腫比較
        lobe_means = df[lobe_cols].mean().sort_values(ascending=False)
        colors_lobe = plt.cm.Reds(np.linspace(0.4, 0.9, len(lobe_means)))
        axes[0, 1].barh(range(len(lobe_means)), lobe_means.values, color=colors_lobe)
        axes[0, 1].set_yticks(range(len(lobe_means)))
        axes[0, 1].set_yticklabels(
            [l.replace("_Emphysema%", "") for l in lobe_means.index]
        )
        axes[0, 1].set_xlabel("Mean Emphysema %", fontsize=11)
        axes[0, 1].set_title(
            "Mean Emphysema by Lobe", fontsize=12, fontweight="bold", pad=10
        )
        axes[0, 1].invert_yaxis()
        for i, v in enumerate(lobe_means.values):
            axes[0, 1].text(v + 0.1, i, f"{v:.2f}%", va="center")

        # 3. 總肺氣腫分組比較
        normal_emp = df[df["Predicted_Class"] == "Normal"]["Total_Emphysema%"].dropna()
        abnormal_emp = df[df["Predicted_Class"] == "Abnormal"][
            "Total_Emphysema%"
        ].dropna()

        bp = axes[1, 0].boxplot(
            [normal_emp, abnormal_emp] if len(abnormal_emp) > 0 else [normal_emp],
            labels=["Normal", "Abnormal"] if len(abnormal_emp) > 0 else ["Normal"],
            patch_artist=True,
            showmeans=True,
        )
        colors = ["#2ecc71", "#e74c3c"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[1, 0].set_ylabel("Total Emphysema %", fontsize=11)
        axes[1, 0].set_title(
            "Total Emphysema: Normal vs Abnormal",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        axes[1, 0].grid(alpha=0.3, axis="y")

        # 4. 肺氣腫與分類關係
        axes[1, 1].scatter(
            df["Total_Emphysema%"],
            df["Confidence"],
            c=df["Predicted_Class"].map({"Normal": "#2ecc71", "Abnormal": "#e74c3c"}),
            s=150,
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )
        axes[1, 1].set_xlabel("Total Emphysema %", fontsize=11)
        axes[1, 1].set_ylabel("Confidence", fontsize=11)
        axes[1, 1].set_title(
            "Emphysema vs Classification Confidence",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        axes[1, 1].grid(alpha=0.3)

        # 添加圖例
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#2ecc71", label="Normal"),
            Patch(facecolor="#e74c3c", label="Abnormal"),
        ]
        axes[1, 1].legend(handles=legend_elements, loc="best")

        plt.tight_layout()
        plt.savefig(
            output_dir / "5_emphysema_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def generate_visualizations_from_reports(
        self, data_dir: str, output_dir: str, verbose: bool = True
    ):
        """
        從已有的參數檔案（param/）和報告（report/）生成視覺化圖表

        Args:
            data_dir: 資料目錄（包含 param/ 和 report/ 子目錄）
            output_dir: 輸出目錄
            verbose: 是否輸出詳細訊息
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)

        # 尋找 param 目錄
        param_dir = data_dir / "param" if (data_dir / "param").exists() else data_dir

        if not param_dir.exists():
            print(f"❌ 找不到參數目錄: {param_dir}")
            return

        if verbose:
            print("\n" + "=" * 70)
            print("📊 生成視覺化圖表模式")
            print(f"   參數目錄: {param_dir}")
            print("=" * 70)

        # 載入所有參數檔案
        param_files = list(param_dir.glob("*_metrics.json"))

        if not param_files:
            print(f"❌ 在 {param_dir} 中找不到參數檔案 (*_metrics.json)")
            return

        if verbose:
            print(f"\n🔍 找到 {len(param_files)} 個參數檔案")

        # 構建假的 results 結構來匹配視覺化函數的期望格式
        results = []

        for param_file in param_files:
            try:
                with open(param_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)

                # 檢查是否有對應的報告檔案來獲取分類結果
                report_dir = (
                    data_dir / "report"
                    if (data_dir / "report").exists()
                    else data_dir.parent / "report"
                )
                report_file = report_dir / param_file.name.replace(
                    "_metrics.json", "_report.md"
                )

                # 構建結果結構
                result = {
                    "input_file": str(param_file),
                    "stages": {
                        "analysis": {"status": "success", "metrics": metrics},
                        "classification": {
                            "status": "success",
                            "class_name": "Normal",  # 預設值
                            "confidence": 0.5,
                            "probabilities": {"Normal": 0.5, "Abnormal": 0.5},
                        },
                    },
                }

                # 嘗試從報告檔案讀取分類結果
                if report_file.exists():
                    with open(report_file, "r", encoding="utf-8") as f:
                        report_content = f.read()

                        # 簡單解析報告內容
                        if (
                            "預測結果: Normal" in report_content
                            or "**預測結果: Normal**" in report_content
                        ):
                            result["stages"]["classification"]["class_name"] = "Normal"
                        elif (
                            "預測結果: Abnormal" in report_content
                            or "**預測結果: Abnormal**" in report_content
                        ):
                            result["stages"]["classification"]["class_name"] = (
                                "Abnormal"
                            )

                        # 嘗試提取信心度
                        import re

                        conf_match = re.search(
                            r"信心度[：:] ([0-9.]+)%", report_content
                        )
                        if conf_match:
                            result["stages"]["classification"]["confidence"] = (
                                float(conf_match.group(1)) / 100
                            )

                results.append(result)

            except Exception as e:
                if verbose:
                    print(f"   ⚠️ 無法載入 {param_file.name}: {e}")

        if not results:
            print("❌ 沒有成功載入任何參數檔案")
            return

        if verbose:
            print(f"   ✅ 成功載入 {len(results)} 個參數檔案")
            print(f"\n📊 開始生成圖表...")

        # 生成視覺化圖表
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._generate_visualizations(results, output_dir, verbose=verbose)

        if verbose:
            print("\n" + "=" * 70)
            print("✅ 視覺化圖表生成完成！")
            print(f"   輸出目錄: {output_dir / 'visualizations'}")
            print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="COPD 統一分析 Pipeline - 自動化分割、分析、分類",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 處理單一檔案
  python unified_pipeline.py -i patient_ct.nii.gz -o output/
  
  # 批次處理資料夾
  python unified_pipeline.py -i ./data/ -o ./results/ --batch
  
  # 跳過分割（已有分割結果）
  python unified_pipeline.py -i patient_ct.nii.gz --seg existing_seg.nii.gz -o output/
  
  # 指定模型路徑
  python unified_pipeline.py -i ct.nii.gz --model models/best_copd_model.pth
  
  # 只生成參數，不進行預測（用於生成訓練資料）
  python unified_pipeline.py -i NormalDataset/data -o NormalDataset --batch --no-predict
        """,
    )

    # 必要參數
    parser.add_argument("-i", "--input", required=True, help="輸入 CT 影像或資料夾路徑")

    # 選用參數
    parser.add_argument(
        "-o", "--output", default=None, help="輸出目錄（預設：與輸入同目錄）"
    )
    parser.add_argument("--seg", default=None, help="現有分割結果路徑（跳過分割步驟）")
    parser.add_argument(
        "--airway",
        default=None,
        help="氣道分割結果路徑（自動搜尋：../airway/{base_name}_airway_seg.nii.gz）",
    )
    parser.add_argument("--batch", action="store_true", help="批次處理模式")
    parser.add_argument("--skip-seg", action="store_true", help="跳過分割步驟")
    parser.add_argument(
        "--no-predict",
        action="store_true",
        help="只生成參數，跳過神經網路預測（用於生成訓練資料）",
    )
    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="只生成視覺化圖表（需要已有 param/ 和 report/ 資料夾）",
    )

    # 模型設定
    parser.add_argument("--model", default=None, help="模型檔案路徑")
    parser.add_argument("--scaler", default=None, help="Scaler 檔案路徑")
    parser.add_argument(
        "--server", default=SEG_SERVER_URL, help="分割伺服器 URL"
    )
    parser.add_argument("--device", default=None, help="運算設備 (cuda/cpu)")

    # 其他
    parser.add_argument("-q", "--quiet", action="store_true", help="靜音模式")

    args = parser.parse_args()

    # 初始化 Pipeline
    try:
        pipeline = COPDPipeline(
            model_path=args.model,
            scaler_path=args.scaler,
            server_url=args.server,
            device=args.device,
            skip_model_loading=args.no_predict,
        )
    except Exception as e:
        print(f"❌ Pipeline 初始化失敗: {e}")
        return 1

    input_path = Path(args.input)
    verbose = not args.quiet

    # 執行分析
    try:
        if args.viz_only:
            # 只生成視覺化圖表模式
            output_dir = args.output or str(input_path / "results")
            pipeline.generate_visualizations_from_reports(
                input_path, output_dir, verbose=verbose
            )
        elif args.batch or input_path.is_dir():
            output_dir = args.output or str(input_path / "results")
            pipeline.analyze_batch(
                input_path, output_dir, skip_prediction=args.no_predict, verbose=verbose
            )
        else:
            pipeline.analyze(
                ct_path=input_path,
                seg_path=args.seg,
                airway_path=args.airway,
                output_dir=args.output,
                skip_segmentation=args.skip_seg,
                skip_prediction=args.no_predict,
                verbose=verbose,
            )
    except Exception as e:
        print(f"❌ 分析失敗: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
