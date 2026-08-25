"""
肺部 ROI 自動裁切 (Automated lung ROI cropping)
=================================================

針對 CT 擺位/角度/FOV 每次不同的問題，用「現有的分割結果」自動切出只包住
肺部相關結構的最小長方體 ROI，完全免人工框選 —— 可直接用於部署推論管線。

作法 (對應 Gemini 討論的 CropForegroundd 概念，但用本專案既有的 seg_*.nii.gz)：
  1. 每位病人的 seg_*.nii.gz 已含 9 個標籤：
        1-5 = 五個肺葉, 6 = 血管, 7 = 氣管, 8 = 肺靜脈, 9 = 肺動脈
  2. 取這些標籤的「聯集 (union)」→ 一個 3D 前景遮罩 (預設用全部 1-9)
  3. 算出該遮罩的 3D bounding box，向外加一點 margin
  4. 依 bbox 裁切「原始 CT」(不做 masking，心臟/肋骨等周邊組織原封不動保留，
     提供模型解剖學空間脈絡)，只是把多餘的腹部/頸部/體外空氣切掉
  5. 存成新的 CT (all_roi/)，affine 正確平移，世界座標不跑掉

CT 與 seg 是「同一個 voxel 陣列格線」逐格對應 (copd_analyzer.py 就是這樣直接
以 lobe_mask & (ct_data < thr) 相乘計算肺氣腫的)，所以 bbox 的 index 可直接套
到 CT 陣列上。輸出的 affine 一律以「CT 自己的 affine」為基準平移 (nibabel slicer)，
不使用 seg 的 affine (seg 經 NRRD 來回轉檔，z-origin 有 cosmetic 偏移)。

用法：
  # 全部 66 位，輸出到 all_roi/ (CT) + all_roi_seg/ (對齊的裁切遮罩)
  ./.conda/python.exe make_lung_roi.py

  # 只裁肺葉+氣管 (1-5,7)、margin 15mm、先試跑 3 位
  ./.conda/python.exe make_lung_roi.py --labels 1 2 3 4 5 7 --margin-mm 15 --limit 3
"""

import argparse
import glob
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent


def keep_large_components(mask: np.ndarray, ratio: float):
    """只保留「夠大」的 3D 連通元件，丟掉分割雜訊小島。

    有些掃描(尤其顯影劑胸腹掃描)分割會把腹部組織誤標成肺/肺動脈，形成與主肺塊
    不相連的小島，把 bounding box 撐到腹部。保留 size >= 最大元件 * ratio 的元件
    (真的左右肺分成兩塊也遠大於 ratio、會留下；腹部雜訊島 <1% 會被丟)。

    回傳 (過濾後的遮罩, 元件數, 丟掉的 voxel 數)。ratio<=0 時不過濾。
    """
    if ratio <= 0 or not mask.any():
        return mask, 0, 0
    lbl, ncc = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if ncc <= 1:
        return mask, ncc, 0
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0  # 背景
    thresh = sizes.max() * ratio
    keep_ids = np.where(sizes >= thresh)[0]
    filtered = np.isin(lbl, keep_ids)
    dropped = int(mask.sum() - filtered.sum())
    return filtered, ncc, dropped


def build_seg_map(root: Path) -> dict:
    """掃描所有 inference_output/seg_*.nii.gz，回傳 {CT檔名: seg路徑}。"""
    seg_map = {}
    for p in glob.glob(str(root / "**" / "inference_output" / "seg_*.nii.gz"), recursive=True):
        base = os.path.basename(p)[len("seg_"):]  # 去掉 'seg_' 前綴 = 對應的 CT 檔名
        seg_map.setdefault(base, p)
    return seg_map


def align_seg_to_ct(ct_arr: np.ndarray, seg_arr: np.ndarray):
    """把 seg 對齊到 CT 的 voxel 陣列。

    多數掃描 seg 與 CT 逐格對應 (identity)；少數掃描 seg 的切片順序 (Z) 相反
    (NRRD 來回轉檔造成)，此時肺葉標籤會落在軟組織上 (中位 HU 偏高)。

    判準：肺是空氣，正確對齊時肺葉(1-5)在 CT 的中位 HU 必定 < AIR_HU(-600)。
    肺在 Z 大致對稱，翻轉後仍多半落在肺上、兩方向 HU 差異很小，故**不能**用
    「挑較低者」(會被雜訊誤翻)。規則：只有當「原樣對齊已失敗 (HU ≥ -600)」
    且「Z 翻轉能把它救回空氣 (HU < -600)」時才翻轉；否則一律保留原樣。

    回傳 (對齊後的 seg, flip 標記, 採用方向的肺中位 HU)。
    """
    AIR_HU = -600.0

    def lung_median(seg):
        lobes = np.isin(seg.astype(np.int32), np.array([1, 2, 3, 4, 5], dtype=np.int32))
        return float(np.median(ct_arr[lobes])) if lobes.any() else float("inf")

    hu_id = lung_median(seg_arr)
    if hu_id < AIR_HU:                     # 原樣已把肺對到空氣 → 正確，不動
        return seg_arr, "none", hu_id
    hu_z = lung_median(seg_arr[:, :, ::-1])
    if hu_z < AIR_HU and hu_z < hu_id:     # 原樣失敗、Z 翻轉救回 → 切片順序相反
        return np.ascontiguousarray(seg_arr[:, :, ::-1]), "zflip", hu_z
    return seg_arr, "none", hu_id          # 兩方向都不像空氣 → 保留原樣並在報告標記


def compute_bbox(mask: np.ndarray, margin_vox):
    """回傳前景遮罩的 (start, stop) 索引 (含 margin，已 clamp 到邊界)。

    margin_vox: 每軸的 voxel margin (長度 3)。找不到前景則回傳 None。
    """
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None
    starts, stops = [], []
    for ax in range(3):
        lo = int(coords[ax].min()) - int(margin_vox[ax])
        hi = int(coords[ax].max()) + int(margin_vox[ax]) + 1  # stop 為 exclusive
        lo = max(0, lo)
        hi = min(mask.shape[ax], hi)
        starts.append(lo)
        stops.append(hi)
    return starts, stops


def crop_one(ct_path: Path, seg_path: Path, labels, margin_mm: float, cc_ratio: float,
             mask: bool = False, mask_fill: float = -1000.0, mask_dilate_mm: float = 3.0,
             box_black: bool = False):
    """裁切單一病人，回傳 (cropped_ct_img, cropped_seg_img, info_dict)。"""
    ct_img = nib.load(str(ct_path))
    seg_img = nib.load(str(seg_path))
    ct_arr = np.asanyarray(ct_img.dataobj)          # 原生 dtype，與 seg 逐格對應
    seg_arr = np.asanyarray(seg_img.dataobj).astype(np.int16)

    if ct_img.shape != seg_img.shape:
        raise ValueError(f"CT {ct_img.shape} 與 seg {seg_img.shape} 形狀不符，無法逐格對應")

    # 自動對齊 seg 到 CT (修正少數切片順序相反的掃描)
    seg_arr, flip, _ = align_seg_to_ct(ct_arr, seg_arr)

    # 前景遮罩：指定標籤的聯集 (labels=None → 全部 >0)
    if labels is None:
        fg = seg_arr > 0
    else:
        fg = np.isin(seg_arr, np.asarray(labels, dtype=np.int32))

    # 連通元件過濾：丟掉腹部誤標等不相連小島 (否則 bbox 會被撐大)
    fg, ncc, cc_dropped = keep_large_components(fg, cc_ratio)

    # 每軸 margin(mm) → voxel，用 CT 的 spacing 換算 (z 與 in-plane spacing 差很多)
    zooms = np.asarray(ct_img.header.get_zooms()[:3], dtype=float)
    margin_vox = np.maximum(1, np.round(margin_mm / np.maximum(zooms, 1e-6))).astype(int)

    bbox = compute_bbox(fg, margin_vox)
    if bbox is None:
        raise ValueError("seg 中找不到指定的前景標籤")
    (x0, y0, z0), (x1, y1, z1) = bbox

    ct_crop_arr = ct_arr[x0:x1, y0:y1, z0:z1]           # box 內容 (供 QC / 各模式共用)

    if box_black:
        # 【box 模式】保留原始尺寸；長方體(bbox)內原封不動(心臟/胸壁都留)，
        # 長方體「以外」全部填 mask_fill(-1000) → 變黑。affine/尺寸與原圖相同。
        full = np.full(ct_arr.shape, mask_fill, dtype=ct_arr.dtype)
        full[x0:x1, y0:y1, z0:z1] = ct_crop_arr
        cropped_ct = nib.Nifti1Image(full, ct_img.affine, ct_img.header)
        cropped_seg_arr = seg_arr                        # 全尺寸對齊 seg
        cropped_seg = nib.Nifti1Image(cropped_seg_arr, ct_img.affine, ct_img.header)
    else:
        # 用 nibabel slicer 取得裁切後的正確 affine (依 voxel 偏移平移)
        sliced = ct_img.slicer[x0:x1, y0:y1, z0:z1]
        if mask:
            # 把肺區以外(心臟/胸壁/肋骨/床)填成 mask_fill → 只留 seg 形狀 (hard attention)。
            # 用 CC 過濾後的前景 fg，向外膨脹 mask_dilate_mm(避免削掉肺緣的肺氣腫)。
            fg_crop = fg[x0:x1, y0:y1, z0:z1]
            if mask_dilate_mm > 0:
                dist = ndimage.distance_transform_edt(~fg_crop, sampling=zooms)
                keep = dist <= mask_dilate_mm            # 前景 + mm 外殼
            else:
                keep = fg_crop
            masked = ct_crop_arr.copy()
            masked[~keep] = mask_fill
            cropped_ct = nib.Nifti1Image(masked, sliced.affine, sliced.header)
        else:
            cropped_ct = sliced
        # 裁切後的 seg (已對齊)：直接切陣列，套用「與裁切 CT 完全相同」的 affine
        cropped_seg_arr = seg_arr[x0:x1, y0:y1, z0:z1]
        cropped_seg = nib.Nifti1Image(cropped_seg_arr, cropped_ct.affine, cropped_ct.header)

    orig_vox = int(np.prod(ct_img.shape))
    box_vox = int((x1 - x0) * (y1 - y0) * (z1 - z0))
    crop_vox = int(np.prod(cropped_ct.shape))
    fg_vox = int(fg.sum())

    # 對齊自我檢查：肺葉(1-5)在 box 內的中位 HU 應該很低(空氣，通常 < -700)
    lung_lobes = np.isin(seg_arr[x0:x1, y0:y1, z0:z1], np.array([1, 2, 3, 4, 5], dtype=np.int32))
    lung_median_hu = float(np.median(ct_crop_arr[lung_lobes])) if lung_lobes.any() else float("nan")

    info = {
        "orig_shape": tuple(int(s) for s in ct_img.shape),
        "crop_shape": tuple(int(s) for s in cropped_ct.shape),
        "bbox": [int(x0), int(x1), int(y0), int(y1), int(z0), int(z1)],
        "margin_vox": [int(m) for m in margin_vox],
        "kept_pct": round(100.0 * box_vox / orig_vox, 1),  # ROI 長方體佔原圖比例
        "fg_pct_of_crop": round(100.0 * fg_vox / box_vox, 1) if box_vox else 0.0,
        "lung_median_hu": round(lung_median_hu, 1),
        "flip": flip,
        "n_components": ncc,
        "cc_dropped_vox": cc_dropped,
    }
    return cropped_ct, cropped_seg, info


def main():
    ap = argparse.ArgumentParser(description="自動肺部 ROI 裁切 (用既有 seg_*.nii.gz)")
    ap.add_argument("--all-dir", default=str(SCRIPT_DIR / "all"), help="原始 CT 資料夾")
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR / "all_roi"), help="裁切後 CT 輸出資料夾")
    ap.add_argument("--seg-out-dir", default=str(SCRIPT_DIR / "all_roi_seg"),
                    help="裁切後對齊遮罩輸出資料夾")
    ap.add_argument("--labels", type=int, nargs="*", default=None,
                    help="納入 bbox 的標籤 (預設 None = 全部 1-9)。例：--labels 1 2 3 4 5 7")
    ap.add_argument("--margin-mm", type=float, default=10.0, help="每側外擴 margin (mm)")
    ap.add_argument("--keep-cc-ratio", type=float, default=0.1,
                    help="保留 size >= 最大連通塊 * ratio 的元件，丟腹部誤標小島 (0=關閉)")
    ap.add_argument("--mask", action="store_true",
                    help="把肺區以外(心臟/胸壁/床)填成 -1000 → 只留 seg 形狀 (預設輸出到 all_roi_masked/)")
    ap.add_argument("--box-black", action="store_true",
                    help="保留原始尺寸,長方體(bbox)內原封不動、長方體以外填 -1000 (預設輸出到 all_roi_box/)")
    ap.add_argument("--mask-fill", type=float, default=-1000.0, help="masking 填值 (HU，預設 -1000 空氣)")
    ap.add_argument("--mask-dilate-mm", type=float, default=3.0,
                    help="masking 前把肺遮罩外擴 mm，避免削掉肺緣的肺氣腫 (預設 3)")
    ap.add_argument("--no-save-seg", action="store_true", help="不輸出裁切後的遮罩")
    ap.add_argument("--limit", type=int, default=None, help="只處理前 N 位 (測試用)")
    ap.add_argument("--report", default=None, help="QC 報告路徑 (預設 out-dir/roi_crop_report.md)")
    args = ap.parse_args()

    all_dir = Path(args.all_dir)
    # 若沒指定 out-dir，依模式選預設資料夾，不覆蓋 all_roi/
    if args.out_dir == str(SCRIPT_DIR / "all_roi"):
        if args.box_black:
            args.out_dir = str(SCRIPT_DIR / "all_roi_box")
        elif args.mask:
            args.out_dir = str(SCRIPT_DIR / "all_roi_masked")
    out_dir = Path(args.out_dir)
    seg_out_dir = Path(args.seg_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_seg:
        seg_out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report) if args.report else out_dir / "roi_crop_report.md"

    seg_map = build_seg_map(SCRIPT_DIR)
    cts = sorted(all_dir.glob("*.nii.gz"))
    if args.limit:
        cts = cts[: args.limit]

    label_desc = "全部 1-9" if args.labels is None else str(args.labels)
    if args.box_black:
        mode_desc = f"box塗黑(原尺寸, 長方體外填{args.mask_fill:.0f})"
    elif args.mask:
        mode_desc = f"seg塗黑(填{args.mask_fill:.0f}, 外擴{args.mask_dilate_mm}mm)"
    else:
        mode_desc = "純裁切(保留周邊組織)"
    print(f"CT 數: {len(cts)} | ROI 標籤: {label_desc} | margin: {args.margin_mm}mm"
          f" | 連通元件保留比: {args.keep_cc_ratio} | 模式: {mode_desc}")
    print(f"輸出 CT → {out_dir}")
    print("=" * 70)

    rows, ok, skipped = [], 0, []
    for i, ct_path in enumerate(cts, 1):
        name = ct_path.name
        seg_path = seg_map.get(name)
        if seg_path is None:
            print(f"[{i}/{len(cts)}] ⚠️  找不到 seg，跳過: {name}")
            skipped.append((name, "no seg"))
            continue
        try:
            cropped_ct, cropped_seg, info = crop_one(
                ct_path, Path(seg_path), args.labels, args.margin_mm, args.keep_cc_ratio,
                mask=args.mask, mask_fill=args.mask_fill, mask_dilate_mm=args.mask_dilate_mm,
                box_black=args.box_black)
        except Exception as e:
            print(f"[{i}/{len(cts)}] ❌ {name}: {e}")
            skipped.append((name, str(e)))
            continue

        nib.save(cropped_ct, str(out_dir / name))
        if not args.no_save_seg:
            nib.save(cropped_seg, str(seg_out_dir / name))

        flag = "" if info["lung_median_hu"] < -600 else "  ⚠️HU偏高"
        fliptag = "  [Z翻轉校正]" if info["flip"] == "zflip" else ""
        cctag = f"  [去雜訊島{info['cc_dropped_vox']}vox]" if info["cc_dropped_vox"] > 0 else ""
        print(f"[{i}/{len(cts)}] ✅ {name[:34]:34s} {info['orig_shape']} → "
              f"{info['crop_shape']}  保留{info['kept_pct']:5.1f}%  肺中位HU{info['lung_median_hu']:.0f}{flag}{fliptag}{cctag}")
        rows.append((name, info))
        ok += 1

    # QC 報告
    lines = [
        "# 肺部 ROI 自動裁切報告\n\n",
        f"- ROI 前景標籤: {label_desc}（1-5 肺葉 / 6 血管 / 7 氣管 / 8 肺靜脈 / 9 肺動脈）\n",
        f"- 每側 margin: {args.margin_mm} mm\n",
        f"- Masking: 無（保留心臟/肋骨等周邊組織，只縮小範圍）\n",
        f"- 成功: {ok} / {len(cts)}；跳過: {len(skipped)}\n\n",
        "| 病人 | 原始 shape | 裁切 shape | 體積保留% | ROI占裁切% | 肺中位HU | Z校正 |\n",
        "|------|-----------|-----------|----------|-----------|---------|-------|\n",
    ]
    kept_all, flipped = [], []
    for name, info in rows:
        kept_all.append(info["kept_pct"])
        pid = name.split("_")[0]
        zc = "翻轉" if info["flip"] == "zflip" else ""
        if info["flip"] == "zflip":
            flipped.append(pid)
        hu_warn = " ⚠️" if info["lung_median_hu"] >= -600 else ""
        lines.append(
            f"| {pid} | {info['orig_shape']} | {info['crop_shape']} | "
            f"{info['kept_pct']} | {info['fg_pct_of_crop']} | {info['lung_median_hu']}{hu_warn} | {zc} |\n"
        )
    if kept_all:
        lines.append(
            f"\n**平均體積保留 {np.mean(kept_all):.1f}%**（= 平均切掉 {100 - np.mean(kept_all):.1f}% 的體外/腹部/頸部空間）\n"
        )
    if flipped:
        lines.append(
            f"\n## 自動 Z 翻轉校正（{len(flipped)} 位）\n\n"
            "這些掃描的 seg 切片順序 (Z) 與 CT 相反，已自動翻轉對齊後再裁切。\n"
            "**注意**：本專案 `copd_analyzer.py` 以直接索引配對 CT/seg，未做此校正，"
            "因此這些病人既有的肺氣腫/量化指標是在錯誤對齊下算出的，數值有誤：\n\n"
            + "".join(f"- {p}\n" for p in flipped)
        )
    if skipped:
        lines.append("\n## 跳過清單\n\n")
        for name, why in skipped:
            lines.append(f"- {name}: {why}\n")
    report_path.write_text("".join(lines), encoding="utf-8")

    print("=" * 70)
    if kept_all:
        print(f"完成 {ok}/{len(cts)}。平均體積保留 {np.mean(kept_all):.1f}% "
              f"(切掉約 {100 - np.mean(kept_all):.1f}%)。")
    print(f"CT → {out_dir}")
    if not args.no_save_seg:
        print(f"對齊遮罩 → {seg_out_dir}")
    print(f"報告 → {report_path}")


if __name__ == "__main__":
    main()
