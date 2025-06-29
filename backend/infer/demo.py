#!/usr/bin/env python3
"""
Script demo để test toàn bộ workflow inference
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description):
    """Chạy command và hiển thị kết quả"""
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"Lệnh: {cmd}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Thành công!")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print("❌ Lỗi!")
            if result.stderr:
                print("Error:")
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False
    
    elapsed = time.time() - start_time
    print(f"⏱️ Thời gian: {elapsed:.2f}s")
    
    return True

def check_requirements():
    """Kiểm tra các dependencies"""
    print("🔍 Kiểm tra dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'timm', 'rasterio', 
        'pillow', 'opencv-python', 'scikit-learn', 'numpy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                import PIL
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - THIẾU")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Cần cài đặt: pip install {' '.join(missing)}")
        return False
    
    return True

def check_models():
    """Kiểm tra các model files"""
    print("\n🔍 Kiểm tra model files...")
    
    models_dir = "/home/nhotin/tinltn/dat301m/as/sen12floods/training/models"
    model_files = [
        'resnet_best.pth',
        'densenet_best.pth', 
        'efficientnet_best.pth',
        'vit_best.pth'
    ]
    
    if not os.path.exists(models_dir):
        print(f"❌ Không tìm thấy thư mục models: {models_dir}")
        return False
    
    missing_models = []
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024*1024)
            print(f"✅ {model_file} ({size_mb:.1f}MB)")
        else:
            print(f"❌ {model_file} - THIẾU")
            missing_models.append(model_file)
    
    if missing_models:
        print(f"\n⚠️ Thiếu models: {missing_models}")
        return False
        
    return True

def check_data():
    """Kiểm tra dataset"""
    print("\n🔍 Kiểm tra dataset...")
    
    data_dir = "/home/nhotin/tinltn/dat301m/as/sen12floods/data/sen12floods1"
    
    if not os.path.exists(data_dir):
        print(f"❌ Không tìm thấy dataset: {data_dir}")
        return False
    
    # Đếm số file S2
    s2_count = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.startswith('S2') and file.endswith('.tif'):
                s2_count += 1
    
    print(f"✅ Dataset: {data_dir}")
    print(f"📊 Tìm thấy {s2_count} file S2")
    
    return s2_count > 0

def main():
    print("🎯 Demo Script - Flood Detection Inference")
    print("="*60)
    
    # Kiểm tra requirements
    if not check_requirements():
        print("\n❌ Vui lòng cài đặt dependencies trước!")
        return
    
    # Kiểm tra models
    if not check_models():
        print("\n❌ Vui lòng đảm bảo có đủ model files!")
        return
    
    # Kiểm tra data
    if not check_data():
        print("\n❌ Vui lòng kiểm tra dataset!")
        return
    
    print("\n🎉 Tất cả kiểm tra đều OK! Bắt đầu demo...")
    
    # Bước 1: Chuẩn bị dữ liệu (copy 20 ảnh để test nhanh)
    success = run_command(
        "python backend/data_prep.py --clear --limit 20",
        "Bước 1: Copy 20 ảnh S2 để test"
    )
    
    if not success:
        print("❌ Lỗi ở bước chuẩn bị dữ liệu!")
        return
    
    # Bước 2: Inference với 2 models đầu tiên
    success = run_command(
        "python backend/inference.py --models resnet densenet --batch_size 4",
        "Bước 2: Inference với ResNet và DenseNet"
    )
    
    if not success:
        print("❌ Lỗi ở bước inference!")
        return
    
    # Bước 3: Inference với ensemble
    success = run_command(
        "python backend/inference.py --models resnet densenet hard_voting --batch_size 4",
        "Bước 3: Inference với Hard Voting Ensemble"
    )
    
    if not success:
        print("❌ Lỗi ở bước ensemble!")
        return
    
    # Bước 4: Kiểm tra kết quả
    print(f"\n{'='*50}")
    print("📋 Kiểm tra kết quả")
    print(f"{'='*50}")
    
    output_dir = "/home/nhotin/tinltn/dat301m/as/sen12floods/backend/output"
    
    if os.path.exists(output_dir):
        for model_dir in os.listdir(output_dir):
            model_path = os.path.join(output_dir, model_dir)
            if os.path.isdir(model_path):
                file_count = len([f for f in os.listdir(model_path) if f.endswith('.png')])
                print(f"📁 {model_dir}: {file_count} ảnh kết quả")
    
    print(f"\n🎉 DEMO HOÀN THÀNH!")
    print(f"📂 Kết quả tại: {output_dir}")
    print(f"💡 Tip: Mở ảnh .png để xem kết quả phân loại")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo bị dừng bởi user")
    except Exception as e:
        print(f"\n\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc() 