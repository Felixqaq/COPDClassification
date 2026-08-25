"""在 aeropath 容器內跑單筆 AeroPath 氣道分割，輸出 NIfTI。
由 run_aeropath_batch.py 透過 `docker exec ... python3 /work/_aeropath_worker.py <in> <out>` 呼叫。
每個 job 用獨立暫存目錄 (避免 run_model 的相對路徑輸出互相覆蓋)，可安全並行。
"""
import sys
import os
import shutil
import tempfile

sys.path.insert(0, "/home/user/app/demo")

inp, outp = sys.argv[1], sys.argv[2]
job = tempfile.mkdtemp(dir="/tmp")
os.chdir(job)

from src.inference import run_model  # noqa: E402

run_model(
    inp,
    model_path="/home/user/app/resources/models/",
    task="CT_Airways",
    name="Airways",
)

if not os.path.exists("prediction.nii.gz"):
    print("ERROR: prediction.nii.gz not produced", flush=True)
    sys.exit(1)

os.makedirs(os.path.dirname(outp), exist_ok=True)
shutil.copy("prediction.nii.gz", outp)
os.chdir("/")
shutil.rmtree(job, ignore_errors=True)
print("OK " + outp, flush=True)
