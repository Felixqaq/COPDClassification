"""
重算「seg 切片順序(Z)相反」病人的量化指標。

背景：3 位病人 (P07, P02, P08) 的 seg 與 CT 在 Z 軸切片順序相反，
copd_analyzer.py 舊版以直接索引配對 CT/seg，導致肺氣腫(唯一依賴 CT 的指標)算錯。
copd_analyzer.py 已加入 _align_seg_to_ct() 自動校正，本腳本用修正後的分析器重算
這 3 位的 param/ (Trachea WA%) 與 param_aeropath/ (AeroPath WA%) 兩份 metrics.json，
並印出 before/after，確認只有肺氣腫改變、其餘 11 維特徵不變。
"""
import sys, json, io, contextlib, importlib.util, glob, os
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).parent.absolute()
_spec = importlib.util.spec_from_file_location(
    "copd_analyzer", SCRIPT_DIR / "VesselAirwayParamTransfer" / "copd_analyzer.py")
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
COPDAnalyzer = _mod.COPDAnalyzer
sys.path.insert(0, str(SCRIPT_DIR))
import copd_classifier as cc  # noqa: E402

# base name -> AeroPath airway mask
AEROPATH_ROOTS = [
    str(SCRIPT_DIR / "airway_aeropath"),
    "D:/Felix/Research/AeroPath/output/best_per_patient",
    "D:/Felix/Research/AeroPath/output/batch_results_20251214_135626_autosave",
    "D:/Felix/Research/AeroPath/output/batch_results_20251215_061508_autosave",
]
ap_idx = {}
for r in AEROPATH_ROOTS:
    for f in glob.glob(r + "/*_airway_seg.nii.gz"):
        ap_idx.setdefault(os.path.basename(f)[: -len("_airway_seg.nii.gz")], f)

# 3 flagged patients: (dataset dir, base name)
TARGETS = [
    ("NormalDataset",   "2094528_LUNG AX 1_1 LW"),
    ("AbnormalDataset", "5630846_Aorta C+  5.0  B30f"),
    ("NormalDataset",   "8244460_Thorax 1_1 Br40 S3 1.00"),
]
FEAT_NAMES = ["Emph_total", "Lobe1", "Lobe2", "Lobe3", "Lobe4", "Lobe5",
              "SVV%", "WA%", "Vessel%", "Airway/Lung%", "LungVol", "PA_dia"]


def run(seg, ct, airway):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        an = COPDAnalyzer(segmentation_path=str(seg), ct_path=str(ct), airway_path=airway)
        m = an.calculate_metrics()
    return m, buf.getvalue()


def main():
    for ds, base in TARGETS:
        seg = SCRIPT_DIR / ds / "inference_output" / f"seg_{base}.nii.gz"
        ct = SCRIPT_DIR / "all" / f"{base}.nii.gz"
        airway = ap_idx.get(base)
        print("=" * 78)
        print(f"{base}  ({ds})")
        print(f"  seg={seg.name}  ct={ct.name}  aeropath={'yes' if airway else 'no'}")

        for sub, aw in [("param", None), ("param_aeropath", airway)]:
            out = SCRIPT_DIR / ds / sub / f"{base}_metrics.json"
            old = json.load(open(out, encoding="utf-8")) if out.exists() else None
            old_feats = cc.extract_features_from_json(old) if old else None

            metrics, log = run(seg, ct, aw)
            flipped = "[align]" in log
            new_feats = cc.extract_features_from_json(metrics)

            # 存回
            os.makedirs(out.parent, exist_ok=True)
            json.dump(metrics, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

            print(f"  --- {sub} --- (auto-flip: {flipped})")
            if old_feats:
                for nm, o, n in zip(FEAT_NAMES, old_feats, new_feats):
                    mark = "  <== changed" if abs(o - n) > 1e-6 else ""
                    if mark:
                        print(f"      {nm:14s} {o:9.3f} -> {n:9.3f}{mark}")
                unchanged = sum(1 for o, n in zip(old_feats, new_feats) if abs(o - n) <= 1e-6)
                print(f"      ({unchanged}/12 features unchanged)")
            print(f"      emph_total: {old['emphysema']['total_emphysema_percent']:.2f}% -> "
                  f"{metrics['emphysema']['total_emphysema_percent']:.2f}%")
    print("=" * 78)
    print("Done. 6 metrics.json regenerated (3 patients x param/param_aeropath).")


if __name__ == "__main__":
    main()
