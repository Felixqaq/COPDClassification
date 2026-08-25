# 伺服器位址設定範本。
# 複製這個檔為 server_config.py 並填入你實際的內部伺服器位址。
# server_config.py 已列入 .gitignore，不會被提交，可安全存放內部位址。
#
#     cp server_config.example.py server_config.py
#
# 也可改用環境變數 COPD_SEG_SERVER / COPD_AEROPATH_URL 覆蓋（見 config.py）。

SEG_SERVER_URL = "http://YOUR_SEG_SERVER:8891"   # 3D Slicer 分割推論伺服器
AEROPATH_URL = "http://127.0.0.1:7860"           # AeroPath 氣道伺服器（本機）
