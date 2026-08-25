"""
為 all/ 中「尚未有 metrics」的病人補跑 分割 + COPDAnalyzer，產生 *_metrics.json。

用途：train_copd_oi.py 需要每位病人的 12 維影像特徵 (來自 *_metrics.json)。
      all/ 內有部分病人只有原始 CT、沒有 metrics，需先跑此程式補上。

前置需求（重要）：
  - 3D Slicer 推論伺服器必須可連線（位址見 server_config.py / config.py，
    或用 --server 指定），用來產生 lung/lobe/vessel/trachea labels。伺服器離線時本程式會直接中止。
  - (選用) AeroPath 氣道伺服器（預設本機 http://127.0.0.1:7860）；沒有時 WA% 會退回用 Trachea label。

輸出：<output>/param/<base>_metrics.json (預設 output=AllExtraDataset)
      train_copd_oi.py 已將 AllExtraDataset/param 納入掃描，產生後直接重跑即可用上全部病人。

範例：
  python generate_missing_metrics.py
  python generate_missing_metrics.py --server http://YOUR_SEG_SERVER:8891
  python generate_missing_metrics.py --only P09 P10
"""

import argparse
import sys
from pathlib import Path

# unified_pipeline 內含 emoji 輸出；Windows 預設 cp950 無法編碼，重導到檔案時會崩潰。
# 強制 stdout/stderr 用 UTF-8，確保不論前景或背景執行都正常。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from copd_segmentation import COPDSegmenter
from config import SEG_SERVER_URL


DEFAULT_METRIC_DIRS = [
    "NormalDataset/param",
    "AbnormalDataset/param",
    "TestDataset/param",
    "AllExtraDataset/param",
]


def existing_metric_pids(dirs):
    pids = set()
    for d in dirs:
        p = Path(d)
        if p.exists():
            for f in p.glob("*_metrics.json"):
                pids.add(f.stem.split("_")[0])
    return pids


def main():
    ap = argparse.ArgumentParser(
        description="為 all/ 中缺 metrics 的病人補跑分割 + COPDAnalyzer"
    )
    ap.add_argument("--all-dir", default="all", help="原始 CT 來源資料夾 (預設: all)")
    ap.add_argument(
        "--output", default="AllExtraDataset", help="輸出根目錄 (預設: AllExtraDataset)"
    )
    ap.add_argument(
        "--server",
        default=SEG_SERVER_URL,
        help="3D Slicer 推論伺服器 URL",
    )
    ap.add_argument(
        "--metric-dirs",
        nargs="*",
        default=DEFAULT_METRIC_DIRS,
        help="已有 metrics 的目錄 (用來判斷哪些病人需要補跑)",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="只處理指定的 patient_id (預設: 自動找出所有缺 metrics 的病人)",
    )
    args = ap.parse_args()

    have = existing_metric_pids(args.metric_dirs)
    all_cts = sorted(Path(args.all_dir).glob("*.nii.gz"))
    if not all_cts:
        print(f"[ERROR] 在 {args.all_dir} 找不到任何 .nii.gz")
        return 1

    todo = [f for f in all_cts if f.name.split("_")[0] not in have]
    if args.only:
        only = set(args.only)
        todo = [f for f in todo if f.name.split("_")[0] in only]

    print(
        f"all/ CT: {len(all_cts)} 筆，已有 metrics: {len(have)} 筆，"
        f"待處理: {len(todo)} 筆"
    )
    for f in todo:
        print("  -", f.name)
    if not todo:
        print("沒有需要處理的病人。")
        return 0

    # 先確認伺服器可連線，否則直接中止（避免逐筆等待逾時）
    # 註：MONAIAuto3DSeg server 的根路徑 / 會回 404，所以不能用 COPDSegmenter.check_server()
    #     (它打 / 判斷 200)。改打 /models 端點判斷服務是否就緒。
    print(f"\n檢查分割伺服器: {args.server} ...")
    import requests

    try:
        r = requests.get(f"{args.server}/models", timeout=8)
        ok = r.status_code == 200
        model_ids = [m["id"] for m in r.json()] if ok else []
    except Exception as e:
        print(f"[ERROR] 分割伺服器無法連線: {args.server} ({e})")
        print("請先啟動 / 連上 3D Slicer MONAIAuto3DSeg server 後再執行本程式。")
        return 2
    if not ok:
        print(f"[ERROR] 分割伺服器回應異常 (status {r.status_code})")
        return 2
    print(f"伺服器可連線，已載入 {len(model_ids)} 個模型。")

    # 延後 import：COPDPipeline 會初始化分析器
    from unified_pipeline import COPDPipeline

    pipeline = COPDPipeline(server_url=args.server, skip_model_loading=True)

    ok = 0
    fail = []
    for i, f in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {f.name}")
        try:
            res = pipeline.analyze(
                ct_path=f,
                output_dir=args.output,
                skip_prediction=True,
                verbose=True,
            )
            status = res["stages"].get("analysis", {}).get("status")
            if status == "success":
                ok += 1
            else:
                fail.append(
                    (f.name, res["stages"].get("analysis", {}).get("error", "unknown"))
                )
        except Exception as e:
            fail.append((f.name, str(e)))

    print(f"\n完成：{ok}/{len(todo)} 成功")
    for n, e in fail:
        print("  FAIL", n, "->", e)
    print(f"\nmetrics 已寫入：{args.output}/param/")
    print("接著直接重跑：python train_copd_oi.py --no-cuda  即可納入全部病人。")
    return 0 if not fail else 3


if __name__ == "__main__":
    raise SystemExit(main())
