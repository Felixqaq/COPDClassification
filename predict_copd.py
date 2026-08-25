"""
COPD分類預測腳本
使用訓練好的模型對新樣本進行預測
支援 K-Fold 交叉驗證訓練的模型
"""

import torch
import numpy as np
import json
import argparse
import os
import joblib
from pathlib import Path
from copd_classifier import COPDClassifier, extract_features_from_json, load_json_metrics


def load_model_and_scaler(model_path, scaler_path, device='cpu'):
    """載入訓練好的模型和scaler"""
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    
    # 嘗試載入配置檔案（支援新舊格式）
    config = None
    results_dir = model_path.parent
    
    # 新版 K-Fold 格式
    kfold_results_path = results_dir / 'kfold_training_results.json'
    # 舊版格式
    old_results_path = results_dir / 'training_results.json'
    
    if kfold_results_path.exists():
        with open(kfold_results_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    elif old_results_path.exists():
        with open(old_results_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 取得模型配置
    if config:
        model_config = config['configuration']['model_architecture']
    else:
        # 使用預設配置
        model_config = {
            'input_size': 12,
            'hidden_sizes': [64, 32, 16],
            'output_size': 2,
            'dropout_rate': 0.3
        }
    
    # 建立模型
    model = COPDClassifier(
        input_size=model_config['input_size'],
        hidden_sizes=model_config['hidden_sizes'],
        output_size=model_config['output_size'],
        dropout_rate=model_config['dropout_rate']
    ).to(device)
    
    # 載入模型權重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 載入scaler
    scaler = joblib.load(scaler_path)
    
    return model, scaler, config


def predict_single_sample(model, scaler, features, device='cpu'):
    """
    對單一樣本進行預測
    
    Args:
        model: 訓練好的模型
        scaler: 特徵標準化器
        features: 12個特徵值的列表或numpy array
        device: 設備
    
    Returns:
        預測類別、預測機率
    """
    # 標準化特徵
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    
    # 轉換為tensor
    features_tensor = torch.FloatTensor(features_scaled).to(device)
    
    # 預測
    with torch.no_grad():
        outputs = model(features_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    predicted_class = predicted.item()
    probabilities = probs.cpu().numpy()[0]
    
    return predicted_class, probabilities


def predict_from_json(model, scaler, json_path, device='cpu'):
    """從JSON metrics檔案預測"""
    json_data = load_json_metrics(json_path)
    features = extract_features_from_json(json_data)
    return predict_single_sample(model, scaler, features, device)


def find_best_model(model_dir):
    """
    自動尋找最佳模型
    優先使用 K-Fold 結果中的最佳 fold
    """
    model_dir = Path(model_dir)
    
    # 檢查 K-Fold 結果
    kfold_results_path = model_dir / 'kfold_training_results.json'
    if kfold_results_path.exists():
        with open(kfold_results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        best_fold = results.get('kfold_results', {}).get('best_fold', 1)
        model_path = model_dir / f'fold_{best_fold}_model.pth'
        scaler_path = model_dir / f'fold_{best_fold}_scaler.pkl'
        
        if model_path.exists() and scaler_path.exists():
            return str(model_path), str(scaler_path)
    
    # 舊版格式
    old_model_path = model_dir / 'best_copd_model.pth'
    old_scaler_path = model_dir / 'scaler.pkl'
    
    if old_model_path.exists() and old_scaler_path.exists():
        return str(old_model_path), str(old_scaler_path)
    
    # 如果都找不到，返回預設值
    return str(model_dir / 'fold_1_model.pth'), str(model_dir / 'fold_1_scaler.pkl')


def main(args):
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    print("=" * 70)
    print("COPD分類預測")
    print("=" * 70)
    
    # 自動尋找最佳模型（如果未指定）
    model_path = args.model_path
    scaler_path = args.scaler_path
    
    if model_path == 'models/best_copd_model.pth' or model_path == 'models/fold_1_model.pth':
        # 嘗試自動尋找
        auto_model, auto_scaler = find_best_model(Path(model_path).parent)
        if Path(auto_model).exists():
            model_path = auto_model
            scaler_path = auto_scaler
    
    # 載入模型
    print(f"\n載入模型從: {model_path}")
    try:
        model, scaler, config = load_model_and_scaler(
            model_path, 
            scaler_path,
            device=device
        )
        print("模型載入成功！")
    except FileNotFoundError as e:
        print(f"錯誤：找不到模型檔案 - {e}")
        print("\n請先執行訓練腳本: python train_copd_model.py")
        return
    
    # 顯示模型訓練時的性能
    if config:
        if 'kfold_results' in config:
            # K-Fold 結果
            summary = config['kfold_results']['summary']
            print(f"\n模型訓練時的 5-Fold 交叉驗證性能:")
            print(f"  平均準確率: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
            print(f"  平均F1分數: {summary['mean_f1_score']:.4f} ± {summary['std_f1_score']:.4f}")
            if summary.get('mean_auc'):
                print(f"  平均AUC:    {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
        elif 'results' in config:
            # 舊版結果
            test_metrics = config['results']['test_metrics']
            print(f"\n模型訓練時的測試集性能:")
            print(f"  準確率: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
            print(f"  F1分數: {test_metrics['f1_score']:.4f}")
            if test_metrics.get('auc'):
                print(f"  AUC:    {test_metrics['auc']:.4f}")
    
    class_names = ['Normal (正常)', 'Abnormal (異常)']
    
    # 預測
    if args.input_json:
        # 從JSON檔案預測
        print(f"\n預測檔案: {args.input_json}")
        predicted_class, probabilities = predict_from_json(
            model, scaler, args.input_json, device
        )
        
        print("\n預測結果:")
        print("=" * 70)
        print(f"預測類別: {class_names[predicted_class]}")
        print(f"\n類別機率:")
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        print("=" * 70)
        
    elif args.input_dir:
        # 批量預測目錄中的所有JSON檔案
        print(f"\n批量預測目錄: {args.input_dir}")
        json_files = sorted(Path(args.input_dir).glob("*_metrics.json"))
        
        if len(json_files) == 0:
            print("錯誤：目錄中沒有找到 *_metrics.json 檔案")
            return
        
        print(f"找到 {len(json_files)} 個檔案\n")
        
        results = []
        for json_file in json_files:
            try:
                predicted_class, probabilities = predict_from_json(
                    model, scaler, str(json_file), device
                )
                
                result = {
                    'filename': json_file.name,
                    'predicted_class': int(predicted_class),
                    'predicted_label': class_names[predicted_class],
                    'probability_normal': float(probabilities[0]),
                    'probability_abnormal': float(probabilities[1])
                }
                results.append(result)
                
                print(f"✓ {json_file.name:50s} -> {class_names[predicted_class]:20s} "
                      f"(信心度: {probabilities[predicted_class]*100:.1f}%)")
                
            except Exception as e:
                print(f"✗ {json_file.name}: 預測失敗 - {e}")
        
        # 保存批量預測結果
        if args.output:
            output_path = args.output
        else:
            output_path = 'prediction_results.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 統計
        normal_count = sum(1 for r in results if r['predicted_class'] == 0)
        abnormal_count = sum(1 for r in results if r['predicted_class'] == 1)
        
        print("\n" + "=" * 70)
        print("批量預測統計:")
        print(f"  總樣本數: {len(results)}")
        print(f"  預測為正常: {normal_count} ({normal_count/len(results)*100:.1f}%)")
        print(f"  預測為異常: {abnormal_count} ({abnormal_count/len(results)*100:.1f}%)")
        print(f"\n結果已保存至: {output_path}")
        print("=" * 70)
    
    else:
        print("\n錯誤：請提供 --input-json 或 --input-dir 參數")
        print("使用 --help 查看說明")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用訓練好的COPD分類模型進行預測')
    
    parser.add_argument('--model-path', type=str, default='models/fold_1_model.pth',
                        help='模型檔案路徑 (預設: models/fold_1_model.pth，自動選擇最佳 fold)')
    parser.add_argument('--scaler-path', type=str, default='models/fold_1_scaler.pkl',
                        help='Scaler檔案路徑 (預設: models/fold_1_scaler.pkl)')
    
    # 輸入選項（二選一）
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--input-json', type=str,
                            help='單一JSON metrics檔案路徑')
    input_group.add_argument('--input-dir', type=str,
                            help='包含多個JSON metrics檔案的目錄路徑')
    
    parser.add_argument('--output', type=str,
                        help='批量預測結果輸出檔案路徑 (預設: prediction_results.json)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='不使用CUDA即使可用')
    
    args = parser.parse_args()
    
    main(args)
