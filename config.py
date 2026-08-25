"""伺服器位址設定載入器。

真實的內部伺服器位址請放在 `server_config.py`（見 `server_config.example.py`），
該檔已列入 .gitignore，不會被提交、不會公開。也可用環境變數覆蓋：

    COPD_SEG_SERVER      3D Slicer 分割推論伺服器 URL
    COPD_AEROPATH_URL    AeroPath 氣道伺服器 URL

解析順序：環境變數 > server_config.py > 安全的預設 placeholder。
"""
import os

try:
    import server_config as _sc
    _SEG = getattr(_sc, "SEG_SERVER_URL", None)
    _AERO = getattr(_sc, "AEROPATH_URL", None)
except ImportError:
    _SEG = _AERO = None

SEG_SERVER_URL = os.environ.get("COPD_SEG_SERVER") or _SEG or "http://YOUR_SEG_SERVER:8891"
AEROPATH_URL = os.environ.get("COPD_AEROPATH_URL") or _AERO or "http://127.0.0.1:7860"
