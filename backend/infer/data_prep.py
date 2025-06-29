#!/usr/bin/env python3
"""
Script để copy ngẫu nhiên 10 ảnh Sentinel-2 vào thư mục ready cho inference
"""

import os
import shutil
import glob
import json
from pathlib import Path
import argparse
import random

def copy_s2_images(data_dir, ready_dir, limit=10):
    """
    Copy ngẫu nhiên 10 ảnh Sentinel-2 từ dataset vào thư mục ready
    
    Args:
        data_dir: Thư mục chứa dataset
        ready_dir: Thư mục đích để copy ảnh
        limit: Số ảnh ngẫu nhiên cần copy (mặc định là 10)
    """
    # Tạo thư mục ready nếu chưa có
    os.makedirs(ready_dir, exist_ok=True)
    
    # Tìm tất cả file S2 trong dataset
    s2_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.startswith('S2') and file.endswith('.tif'):
                s2_files.append(os.path.join(root, file))
    
    print(f"Tìm thấy {len(s2_files)} file S2")
    
    # Chọn ngẫu nhiên 10 file
    if len(s2_files) < limit:
        print(f"Cảnh báo: Chỉ có {len(s2_files)} file, copy tất cả")
        selected_files = s2_files
    else:
        selected_files = random.sample(s2_files, limit)
        print(f"Đã chọn ngẫu nhiên {limit} file")
    
    # Copy các file
    copied_count = 0
    for i, src_file in enumerate(selected_files):
        try:
            # Tạo tên file mới để tránh trùng lặp
            filename = os.path.basename(src_file)
            folder_name = os.path.basename(os.path.dirname(src_file))
            new_filename = f"{folder_name}_{filename}"
            dst_file = os.path.join(ready_dir, new_filename)
            
            # Copy file
            shutil.copy2(src_file, dst_file)
            copied_count += 1
            print(f"Đã copy file {i+1}/{limit}: {new_filename}")
                
        except Exception as e:
            print(f"Lỗi khi copy {src_file}: {e}")
    
    print(f"Hoàn thành! Đã copy {copied_count} file S2 vào {ready_dir}")
    return copied_count

def clear_ready_dir(ready_dir):
    """Xóa tất cả file trong thư mục ready"""
    if os.path.exists(ready_dir):
        for file in os.listdir(ready_dir):
            file_path = os.path.join(ready_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Lỗi khi xóa {file_path}: {e}")
        print(f"Đã xóa tất cả file trong {ready_dir}")

def main():
    parser = argparse.ArgumentParser(description='Copy ngẫu nhiên 10 ảnh S2 vào thư mục ready')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/nhotin/tinltn/dat301m/as/sen12floods/data/sen12floods1',
                        help='Thư mục chứa dataset')
    parser.add_argument('--ready_dir', type=str,
                        default='/home/nhotin/tinltn/dat301m/as/sen12floods/backend/ready',
                        help='Thư mục đích')
    parser.add_argument('--clear', action='store_true',
                        help='Xóa tất cả file trong thư mục ready trước khi copy')
    
    args = parser.parse_args()
    
    print("=== Script Copy Ngẫu Nhiên 10 Ảnh S2 ===")
    print(f"Dataset: {args.data_dir}")
    print(f"Ready dir: {args.ready_dir}")
    
    # Xóa file cũ nếu được yêu cầu
    if args.clear:
        clear_ready_dir(args.ready_dir)
    
    # Copy ngẫu nhiên 10 ảnh S2
    copy_s2_images(args.data_dir, args.ready_dir, limit=10)
    
    # Thống kê
    total_files = len(os.listdir(args.ready_dir)) if os.path.exists(args.ready_dir) else 0
    print(f"Tổng số file trong ready: {total_files}")

if __name__ == "__main__":
    main()