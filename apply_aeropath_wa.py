"""
把 AeroPath 產生的 NIfTI airway 遮罩接進 COPDAnalyzer，重算 WA%（source 由
"Trachea Label" 變成 "Dedicated File"），輸出到平行目錄 <Dataset>/param_aeropath/，
不覆蓋原本的 param/。

只處理「同一張掃描」就有 AeroPath 遮罩的病人（base name 完全相同），避免拿到不同
序列的遮罩造成 airway 體積對不上主分割格點。其餘病人維持原狀並列出。

注意（方法學）：目前現成的 AeroPath 遮罩幾乎只有 Abnormal class，若只對這些病人換成
AeroPath WA%、Normal 仍用 Trachea Label，會在類別間造成「萃取方法」的系統性差異（confound），
不應直接拿去重訓練。要公平比較必須對所有病人一致地跑 AeroPath。
"""

import sys
import json
import importlib.util
import contextlib
import io
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).parent.absolute()

# 載入 COPDAnalyzer
_spec = importlib.util.spec_from_file_location(
    "copd_analyzer", SCRIPT_DIR / "VesselAirwayParamTransfer" / "copd_analyzer.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
COPDAnalyzer = _mod.COPDAnalyzer

sys.path.insert(0, str(SCRIPT_DIR))
import copd_classifier as cc  # noqa: E402

AEROPATH_ROOTS = [
    # 本專案新跑的同序列遮罩 (run_aeropath_batch.py 輸出) — 優先採用
    str(SCRIPT_DIR / "airway_aeropath"),
    "D:/Felix/Research/AeroPath/output/best_per_patient",
    "D:/Felix/Research/AeroPath/output/batch_results_20251214_135626_autosave",
    "D:/Felix/Research/AeroPath/output/batch_results_20251215_061508_autosave",
]

# 每個 param 目錄對應的 主分割(seg) 與 CT 來源
DATASETS = [
    {
        "param": "NormalDataset/param",
        "seg_dir": "NormalDataset/inference_output",
        "ct_dir": "NormalDataset/data",
    },
    {
        "param": "AbnormalDataset/param",
        "seg_dir": "AbnormalDataset/inference_output",
        "ct_dir": "AbnormalDataset/data",
    },
    {
        "param": "AllExtraDataset/param",
        "seg_dir": "AllExtraDataset/inference_output",
        "ct_dir": "all",
    },
]


def build_aeropath_index():
    idx = {}
    for r in AEROPATH_ROOTS:
        for f in Path(r).glob("*_airway_seg.nii.gz"):
            base = f.name[: -len("_airway_seg.nii.gz")]
            idx.setdefault(base, str(f))  # 第一個找到的優先（best_per_patient 先）
    return idx


def main():
    ap_idx = build_aeropath_index()
    print(f"AeroPath NIfTI 遮罩可用: {len(ap_idx)} 個 (依 base name)")

    impact = []
    applied = 0
    skipped_no_mask = []
    skipped_missing_io = []

    for ds in DATASETS:
        pdir = Path(ds["param"])
        if not pdir.exists():
            continue
        out_dir = pdir.parent / "param_aeropath"
        out_dir.mkdir(exist_ok=True)
        for mf in sorted(pdir.glob("*_metrics.json")):
            base = mf.stem[: -len("_metrics")]
            old = json.load(open(mf, encoding="utf-8"))
            old_wa = old.get("WA%", {}).get("value") if old.get("WA%") else None
            old_alr = old.get("Airway_Lung_Ratio%")

            airway = ap_idx.get(base)
            if not airway:
                skipped_no_mask.append(base)
                continue

            seg = Path(ds["seg_dir"]) / f"seg_{base}.nii.gz"
            ct = Path(ds["ct_dir"]) / f"{base}.nii.gz"
            if not seg.exists() or not ct.exists():
                skipped_missing_io.append((base, str(seg), str(ct)))
                continue

            # 重算（安靜模式）
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                analyzer = COPDAnalyzer(
                    segmentation_path=str(seg), ct_path=str(ct), airway_path=airway
                )
                metrics = analyzer.calculate_metrics()
                analyzer.save_params(str(out_dir / mf.name))

            new_wa = metrics.get("WA%", {}).get("value") if metrics.get("WA%") else None
            new_src = metrics.get("WA%", {}).get("source") if metrics.get("WA%") else None
            new_alr = metrics.get("Airway_Lung_Ratio%")
            # 驗證可擷取 12 維
            feats = cc.extract_features_from_json(metrics)
            ok = len(feats) == 12 and all(x == x for x in feats)

            impact.append(
                {
                    "base": base,
                    "wa_old": old_wa,
                    "wa_new": new_wa,
                    "wa_source": new_src,
                    "alr_old": old_alr,
                    "alr_new": new_alr,
                    "features_ok": ok,
                }
            )
            applied += 1

    # 報表
    print(f"\n已套用 AeroPath WA% 並輸出到 <Dataset>/param_aeropath/: {applied} 筆")
    print(f"無同序列 AeroPath 遮罩 (維持原狀): {len(skipped_no_mask)} 筆")
    if skipped_missing_io:
        print(f"有遮罩但缺 seg/CT，略過: {len(skipped_missing_io)} 筆")
        for b, s, c in skipped_missing_io:
            print(f"   {b}: seg存在={Path(s).exists()} ct存在={Path(c).exists()}")

    print("\n=== WA% / Airway-Lung-Ratio% 變化 (套用 AeroPath 的病人) ===")
    print(f"{'patient':40} {'WA%old':>8} {'WA%new':>8} {'ALR%old':>9} {'ALR%new':>9} src")
    for r in impact:
        print(
            f"{r['base'][:40]:40} {r['wa_old']:>8.2f} {r['wa_new']:>8.2f} "
            f"{r['alr_old']:>9.3f} {r['alr_new']:>9.3f} {r['wa_source']}"
        )

    if impact:
        import statistics as st

        dwa = [r["wa_new"] - r["wa_old"] for r in impact]
        dalr = [r["alr_new"] - r["alr_old"] for r in impact]
        print(
            f"\nWA% 平均變化: {st.mean(dwa):+.2f} (中位 {st.median(dwa):+.2f})"
        )
        print(
            f"Airway-Lung-Ratio% 平均變化: {st.mean(dalr):+.3f} (中位 {st.median(dalr):+.3f})"
        )
        all_ok = all(r["features_ok"] for r in impact)
        print(f"全部可擷取 12 維特徵: {all_ok}")

    # 存報表
    rep = SCRIPT_DIR / "aeropath_wa_impact.json"
    json.dump(
        {"applied": applied, "no_mask": skipped_no_mask, "impact": impact},
        open(rep, "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )
    print(f"\n影響報表已存: {rep}")


if __name__ == "__main__":
    main()
