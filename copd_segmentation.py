"""
COPD 肺部分割模組
使用 3D Slicer 推論伺服器進行自動化肺部分割

整合自 NiiToLungLabel/3DSlicerSeg.py
"""

import os
import requests
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

from config import SEG_SERVER_URL

try:
    import SimpleITK as sitk
except ImportError:
    raise ImportError("需要安裝 SimpleITK: pip install SimpleITK")


class COPDSegmenter:
    """
    肺部分割器 - 呼叫 3D Slicer 推論伺服器
    
    Usage:
        segmenter = COPDSegmenter()
        output_path = segmenter.segment("patient_ct.nii.gz", "output/seg.nii.gz")
    """
    
    def __init__(self, 
                 server_url: str = SEG_SERVER_URL,
                 model_id: str = "lungs-v2.0.1",
                 timeout: int = 600):
        """
        初始化分割器
        
        Args:
            server_url: 3D Slicer 推論伺服器 URL
            model_id: 模型 ID
            timeout: 請求超時時間（秒）
        """
        self.server_url = server_url
        self.api_url = f"{server_url}/infer"
        self.model_id = model_id
        self.timeout = timeout
        
    def check_server(self) -> bool:
        """檢查伺服器是否可用"""
        try:
            response = requests.get(self.server_url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def segment(self, 
                input_path: Union[str, Path], 
                output_path: Optional[Union[str, Path]] = None,
                verbose: bool = True) -> str:
        """
        執行肺部分割
        
        Args:
            input_path: 輸入 NIfTI 檔案路徑 (.nii 或 .nii.gz)
            output_path: 輸出分割結果路徑（若為 None，則自動生成）
            verbose: 是否輸出詳細訊息
            
        Returns:
            輸出檔案路徑
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"找不到輸入檔案: {input_path}")
        
        # 自動生成輸出路徑
        if output_path is None:
            output_dir = input_path.parent
            output_filename = f"seg_{input_path.name}"
            output_path = output_dir / output_filename
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"📥 輸入: {input_path.name}")
            print(f"📤 輸出: {output_path}")
        
        # 建立暫存檔案用於 NRRD 轉換
        with tempfile.NamedTemporaryFile(suffix='.nrrd', delete=False) as temp_nrrd:
            temp_nrrd_path = temp_nrrd.name
        
        try:
            # 1. 轉換格式: NIfTI -> NRRD
            if verbose:
                print("   ↳ 轉換格式 (NIfTI → NRRD)...")
            img = sitk.ReadImage(str(input_path))
            sitk.WriteImage(img, temp_nrrd_path)
            
            # 2. 上傳到伺服器進行推論
            if verbose:
                print("   ↳ 上傳並推論中...")
            
            with open(temp_nrrd_path, 'rb') as f:
                files = {'image_file': ('input.nrrd', f)}
                params = {'model_name': self.model_id}
                
                response = requests.post(
                    self.api_url, 
                    files=files, 
                    params=params, 
                    stream=True, 
                    timeout=self.timeout
                )
            
            if response.status_code != 200:
                error_msg = f"伺服器錯誤 (Status {response.status_code})"
                try:
                    error_msg += f": {response.json()}"
                except:
                    error_msg += f": {response.text}"
                raise RuntimeError(error_msg)
            
            # 3. 接收結果 (NRRD 格式)
            temp_out_nrrd = temp_nrrd_path + "_out.nrrd"
            with open(temp_out_nrrd, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            
            # 4. 轉回 NIfTI 格式
            if verbose:
                print("   ↳ 轉換格式 (NRRD → NIfTI)...")
            seg_img = sitk.ReadImage(temp_out_nrrd)
            sitk.WriteImage(seg_img, str(output_path))
            
            # 清理暫存輸出檔
            if os.path.exists(temp_out_nrrd):
                os.remove(temp_out_nrrd)
            
            if verbose:
                print(f"✅ 分割完成: {output_path}")
            
            return str(output_path)
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"伺服器請求超時 ({self.timeout}秒)")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"無法連接伺服器: {self.server_url}")
        finally:
            # 清理暫存輸入檔
            if os.path.exists(temp_nrrd_path):
                os.remove(temp_nrrd_path)
    
    def segment_batch(self, 
                      input_dir: Union[str, Path],
                      output_dir: Union[str, Path],
                      verbose: bool = True) -> list:
        """
        批次分割資料夾中的所有 NIfTI 檔案
        
        Args:
            input_dir: 輸入資料夾路徑
            output_dir: 輸出資料夾路徑
            verbose: 是否輸出詳細訊息
            
        Returns:
            成功處理的輸出檔案路徑列表
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 取得所有 NIfTI 檔案
        nii_files = list(input_dir.glob("*.nii.gz")) + list(input_dir.glob("*.nii"))
        nii_files = [f for f in nii_files if not f.name.startswith("seg_")]
        
        if not nii_files:
            print(f"⚠️ 在 {input_dir} 中找不到 NIfTI 檔案")
            return []
        
        if verbose:
            print(f"🔍 找到 {len(nii_files)} 個檔案待處理")
            print("=" * 60)
        
        results = []
        for idx, nii_file in enumerate(nii_files, 1):
            if verbose:
                print(f"\n[{idx}/{len(nii_files)}] 處理: {nii_file.name}")
            
            try:
                output_path = output_dir / f"seg_{nii_file.name}"
                result = self.segment(nii_file, output_path, verbose=verbose)
                results.append(result)
            except Exception as e:
                print(f"❌ 錯誤: {e}")
                continue
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"🎉 批次處理完成！成功: {len(results)}/{len(nii_files)}")
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='COPD 肺部分割工具')
    parser.add_argument('-i', '--input', required=True,
                        help='輸入 NIfTI 檔案或資料夾路徑')
    parser.add_argument('-o', '--output', default=None,
                        help='輸出路徑（預設：與輸入同目錄）')
    parser.add_argument('--server', default=SEG_SERVER_URL,
                        help='推論伺服器 URL')
    parser.add_argument('--model', default="lungs-v2.0.1",
                        help='模型 ID')
    parser.add_argument('--batch', action='store_true',
                        help='批次處理模式')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='靜音模式')
    
    args = parser.parse_args()
    
    segmenter = COPDSegmenter(server_url=args.server, model_id=args.model)
    
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        output_dir = args.output or str(input_path / "seg_output")
        segmenter.segment_batch(input_path, output_dir, verbose=not args.quiet)
    else:
        segmenter.segment(input_path, args.output, verbose=not args.quiet)
