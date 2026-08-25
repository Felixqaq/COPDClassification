"""
並行對「尚無同序列 AeroPath 遮罩」的病人補跑 AeroPath 氣道分割 (CPU)。

作法：用一個已掛載專案目錄的容器 aeropath_batch，透過多個 `docker exec` 並行跑
_aeropath_worker.py，每筆輸出 NIfTI 到 airway_aeropath/<base>_airway_seg.nii.gz。
這些遮罩之後由 apply_aeropath_wa.py 接進 COPDAnalyzer 重算 WA%。

需求：容器 aeropath_batch 已啟動且把專案根目錄掛載在 /work
  docker run -d --name aeropath_batch -v "D:\\Felix\\Research\\COPDClassification:/work" aeropath:latest sleep infinity

用法：
  python run_aeropath_batch.py --dry-run      # 只列出要處理哪些、檢查 CT 是否存在
  python run_aeropath_batch.py --workers 5    # 實際並行補跑
"""
import sys
import argparse
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).parent.absolute()
CONTAINER = "aeropath_batch"
OUT_DIR = SCRIPT_DIR / "airway_aeropath"          # host 端輸出
OUT_DIR_CONTAINER = "/work/airway_aeropath"
ALL_DIR_CONTAINER = "/work/all"

METRIC_DIRS = ["NormalDataset/param", "AbnormalDataset/param", "AllExtraDataset/param"]

# 既有 AeroPath 遮罩 (同序列 base name 完全相同者視為已有，不重跑)
AEROPATH_ROOTS = [
    "D:/Felix/Research/AeroPath/output/best_per_patient",
    "D:/Felix/Research/AeroPath/output/batch_results_20251214_135626_autosave",
    "D:/Felix/Research/AeroPath/output/batch_results_20251215_061508_autosave",
    str(OUT_DIR),
]


def existing_mask_bases():
    bases = set()
    for r in AEROPATH_ROOTS:
        p = Path(r)
        if p.exists():
            for f in p.glob("*_airway_seg.nii.gz"):
                bases.add(f.name[: -len("_airway_seg.nii.gz")])
    return bases


def metric_bases():
    seen = []
    seen_pid = set()
    for d in METRIC_DIRS:
        for mf in sorted(Path(d).glob("*_metrics.json")):
            base = mf.stem[: -len("_metrics")]
            pid = base.split("_")[0]
            if pid in seen_pid:
                continue
            seen_pid.add(pid)
            seen.append(base)
    return seen


def build_todo():
    have = existing_mask_bases()
    todo = []
    missing_ct = []
    for base in metric_bases():
        if base in have:
            continue
        ct = SCRIPT_DIR / "all" / f"{base}.nii.gz"
        if not ct.exists():
            missing_ct.append(base)
            continue
        todo.append(base)
    return todo, missing_ct


def process_one(base):
    in_path = f"{ALL_DIR_CONTAINER}/{base}.nii.gz"
    out_path = f"{OUT_DIR_CONTAINER}/{base}_airway_seg.nii.gz"
    t0 = time.time()
    proc = subprocess.run(
        ["docker", "exec", CONTAINER, "python3", "/work/_aeropath_worker.py",
         in_path, out_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=3600,
    )
    dt = time.time() - t0
    ok = proc.returncode == 0 and (OUT_DIR / f"{base}_airway_seg.nii.gz").exists()
    tail = (proc.stdout or "").strip().splitlines()[-1:] + \
           (proc.stderr or "").strip().splitlines()[-2:]
    return base, ok, dt, " | ".join(tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    todo, missing_ct = build_todo()

    print(f"待補跑 AeroPath: {len(todo)} 筆 (workers={args.workers})")
    for b in todo:
        print("  -", b)
    if missing_ct:
        print(f"\n[警告] 有 metrics 但 all/ 找不到 CT，略過 {len(missing_ct)} 筆:")
        for b in missing_ct:
            print("   ", b)
    if args.dry_run:
        print("\n[dry-run] 不執行。")
        return 0
    if not todo:
        print("沒有需要處理的病人。")
        return 0

    print(f"\n開始並行處理，每筆 CPU 約 8 分鐘...\n")
    done = 0
    fail = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, b): b for b in todo}
        for fut in as_completed(futs):
            base, ok, dt, tail = fut.result()
            done += 1
            status = "OK  " if ok else "FAIL"
            print(f"[{done}/{len(todo)}] {status} ({dt/60:.1f}m) {base}  {tail}",
                  flush=True)
            if not ok:
                fail.append(base)

    print(f"\n完成：{len(todo) - len(fail)}/{len(todo)} 成功，"
          f"總耗時 {(time.time() - t_start)/60:.1f} 分鐘")
    if fail:
        print("失敗清單:")
        for b in fail:
            print("  -", b)
    print(f"\n遮罩輸出於：{OUT_DIR}")
    print("接著重跑：python apply_aeropath_wa.py  (記得把 airway_aeropath 加進其 roots)")
    return 0 if not fail else 3


if __name__ == "__main__":
    raise SystemExit(main())
