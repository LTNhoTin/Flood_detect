#!/usr/bin/env python3
"""
Script inference cho các model phân loại lũ sử dụng PyTorch
"""

import os
import glob
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import rasterio
import timm
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Thiết lập device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng device: {device}")

def load_model(model_path, model_name):
    """Load model từ checkpoint với cấu trúc phù hợp"""
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Tạo model dựa trên tên (không dùng pretrained để tránh lỗi HuggingFace)
        if 'densenet' in model_name.lower():
            model = timm.create_model('densenet121', pretrained=False, num_classes=2)
        elif 'resnet' in model_name.lower():
            model = timm.create_model('resnet50', pretrained=False, num_classes=2)
        elif 'efficientnet' in model_name.lower():
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
        elif 'vit' in model_name.lower():
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
        else:
            raise ValueError(f"Không hỗ trợ model: {model_name}")
        
        # Load state dict trực tiếp
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        model.eval()
        
        print(f"✅ Đã load model {model_name}")
        return model
        
    except Exception as e:
        print(f"❌ Lỗi load model {model_name}: {str(e)}")
        return None

class S2Dataset(Dataset):
    """Dataset cho ảnh Sentinel-2"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            with rasterio.open(image_path) as src:
                # Đọc ảnh và chuyển thành RGB
                image = src.read()
                if len(image.shape) == 3:
                    image = np.transpose(image, (1, 2, 0))
                
                # Chuẩn hóa về 0-255 an toàn
                if image.dtype != np.uint8:
                    img_min, img_max = image.min(), image.max()
                    if img_max > img_min:
                        image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        image = np.zeros_like(image, dtype=np.uint8)
                
                # Đảm bảo có 3 channels
                if len(image.shape) == 2:
                    image = np.stack([image, image, image], axis=-1)
                elif image.shape[-1] == 1:
                    image = np.concatenate([image, image, image], axis=-1)
                elif image.shape[-1] > 3:
                    image = image[:, :, :3]
                
                # Resize về 224x224
                image = cv2.resize(image, (224, 224))
                
                # Convert to PIL
                image = Image.fromarray(image)
                
                if self.transform:
                    image = self.transform(image)
                
                return image, image_path
                
        except Exception as e:
            print(f"Lỗi đọc ảnh {image_path}: {str(e)}")
            # Return dummy image
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            dummy_image = Image.fromarray(dummy_image)
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, image_path

def create_rgb_overlay(image_path, prediction, confidence, output_path):
    """Tạo ảnh RGB với overlay dự đoán"""
    try:
        with rasterio.open(image_path) as src:
            image = src.read()
            if len(image.shape) == 3:
                image = np.transpose(image, (1, 2, 0))
            
            # Xử lý dữ liệu single-band Sentinel-2
            if len(image.shape) == 2 or image.shape[-1] == 1:
                # Tạo RGB từ single band với colormap đẹp
                if image.shape[-1] == 1:
                    image = image.squeeze()
                
                # Normalize về 0-1 an toàn
                p2, p98 = np.percentile(image, 2), np.percentile(image, 98)
                if p98 > p2:  # Tránh divide by zero
                    image_norm = (image - p2) / (p98 - p2)
                else:
                    image_norm = np.zeros_like(image, dtype=np.float32)
                image_norm = np.clip(image_norm, 0, 1)
                
                # Tạo colormap tự nhiên dựa trên band type
                filename = os.path.basename(image_path).lower()
                if any(band in filename for band in ['b04', 'b03', 'b02']):  # Visible bands
                    # Sử dụng colormap xám cho visible bands
                    rgb_image = np.stack([image_norm, image_norm, image_norm], axis=-1)
                elif any(band in filename for band in ['b08', 'b8a', 'b11', 'b12']):  # NIR/SWIR bands  
                    # Sử dụng colormap viridis cho NIR/SWIR
                    import matplotlib.pyplot as plt
                    cmap = plt.cm.viridis
                    rgb_image = cmap(image_norm)[:, :, :3]
                else:
                    # Default: grayscale
                    rgb_image = np.stack([image_norm, image_norm, image_norm], axis=-1)
            else:
                # Multi-band: chỉ lấy 3 bands đầu
                if image.shape[-1] > 3:
                    image = image[:, :, :3]
                
                # Normalize từng channel
                rgb_image = np.zeros_like(image, dtype=np.float32)
                for i in range(image.shape[-1]):
                    channel = image[:, :, i].astype(np.float32)
                    p2, p98 = np.percentile(channel, 2), np.percentile(channel, 98)
                    if p98 > p2:
                        channel_norm = (channel - p2) / (p98 - p2)
                        rgb_image[:, :, i] = np.clip(channel_norm, 0, 1)
            
            # Chuyển về 0-255
            rgb_image = (rgb_image * 255).astype(np.uint8)
            
            # Tạo overlay nhẹ nhàng cho prediction
            overlay = rgb_image.copy()
            
            # Tạo mask với góc để hiển thị prediction
            h, w = overlay.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            if prediction == 1:  # Lũ
                # Thêm viền đỏ ở góc trên trái
                overlay[:50, :50, 0] = np.minimum(255, overlay[:50, :50, 0] + 100)  # Tăng Red
                overlay[:50, :50, 1] = np.maximum(0, overlay[:50, :50, 1] - 30)     # Giảm Green
                overlay[:50, :50, 2] = np.maximum(0, overlay[:50, :50, 2] - 30)     # Giảm Blue
                
                # Thêm text "FLOOD"
                import cv2
                cv2.putText(overlay, 'FLOOD', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(overlay, f'{confidence:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:  # Không lũ
                # Thêm viền xanh ở góc trên trái
                overlay[:50, :50, 0] = np.maximum(0, overlay[:50, :50, 0] - 30)     # Giảm Red
                overlay[:50, :50, 1] = np.minimum(255, overlay[:50, :50, 1] + 100) # Tăng Green  
                overlay[:50, :50, 2] = np.maximum(0, overlay[:50, :50, 2] - 30)     # Giảm Blue
                
                # Thêm text "NO FLOOD"
                import cv2
                cv2.putText(overlay, 'NO FLOOD', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(overlay, f'{confidence:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Lưu ảnh
            cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            return True
            
    except Exception as e:
        print(f"Lỗi tạo RGB overlay cho {image_path}: {str(e)}")
        return False

def inference_single_model(model, dataloader, model_name):
    """Inference cho một model"""
    predictions = []
    probabilities = []
    image_paths = []
    
    print(f"🔍 Inference với model {model_name}...")
    
    with torch.no_grad():
        for batch_idx, (images, paths) in enumerate(dataloader):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            image_paths.extend(paths)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Đã xử lý {(batch_idx + 1) * len(images)} ảnh...")
    
    return predictions, probabilities, image_paths

def hard_voting_ensemble(all_predictions):
    """Hard voting ensemble"""
    ensemble_predictions = []
    
    for i in range(len(all_predictions[0])):
        votes = [preds[i] for preds in all_predictions]
        ensemble_pred = Counter(votes).most_common(1)[0][0]
        ensemble_predictions.append(ensemble_pred)
    
    return ensemble_predictions

def soft_voting_ensemble(all_probabilities):
    """Soft voting ensemble"""
    ensemble_predictions = []
    ensemble_probabilities = []
    
    for i in range(len(all_probabilities[0])):
        # Trung bình các xác suất
        avg_probs = np.mean([probs[i] for probs in all_probabilities], axis=0)
        ensemble_pred = np.argmax(avg_probs)
        
        ensemble_predictions.append(ensemble_pred)
        ensemble_probabilities.append(avg_probs)
    
    return ensemble_predictions, ensemble_probabilities

def main():
    parser = argparse.ArgumentParser(description='Inference cho flood detection models')
    parser.add_argument('--models', nargs='+', 
                        choices=['resnet', 'densenet', 'efficientnet', 'vit', 'hard_voting', 'soft_voting'],
                        default=['resnet', 'densenet'],
                        help='Các model để inference')
    parser.add_argument('--ready_dir', type=str,
                        default='/home/nhotin/tinltn/dat301m/as/sen12floods/backend/ready',
                        help='Thư mục chứa ảnh để inference')
    parser.add_argument('--models_dir', type=str,
                        default='/home/nhotin/tinltn/dat301m/as/sen12floods/training/models',
                        help='Thư mục chứa model files')
    parser.add_argument('--output_dir', type=str,
                        default='/home/nhotin/tinltn/dat301m/as/sen12floods/backend/output',
                        help='Thư mục output')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size cho inference')
    
    args = parser.parse_args()
    
    print("=== INFERENCE FLOOD DETECTION MODELS ===")
    print(f"Models: {args.models}")
    print(f"Ready dir: {args.ready_dir}")
    print(f"Output dir: {args.output_dir}")
    
    # Tìm ảnh trong ready dir
    image_extensions = ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.ready_dir, ext)))
    
    if not image_paths:
        print("❌ Không tìm thấy ảnh nào trong ready dir!")
        return
    
    print(f"📁 Tìm thấy {len(image_paths)} ảnh")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset và DataLoader
    dataset = S2Dataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=2)
    
    # Load các models
    models = {}
    model_names = [name for name in args.models if name not in ['hard_voting', 'soft_voting']]
    
    for model_name in model_names:
        model_path = os.path.join(args.models_dir, f"{model_name}_best.pth")
        if os.path.exists(model_path):
            model = load_model(model_path, model_name)
            if model is not None:
                models[model_name] = model
        else:
            print(f"⚠️ Không tìm thấy model file: {model_path}")
    
    if not models:
        print("❌ Không load được model nào!")
        return
    
    print(f"✅ Đã load {len(models)} models: {list(models.keys())}")
    
    # Inference cho từng model
    all_predictions = {}
    all_probabilities = {}
    
    for model_name, model in models.items():
        preds, probs, paths = inference_single_model(model, dataloader, model_name)
        all_predictions[model_name] = preds
        all_probabilities[model_name] = probs
        
        # Tạo output cho model này
        model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Tạo ảnh RGB với overlay
        print(f"🎨 Tạo ảnh RGB overlay cho {model_name}...")
        for i, (pred, prob, img_path) in enumerate(zip(preds, probs, paths)):
            filename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            confidence = max(prob)
            label = "flood" if pred == 1 else "no_flood"
            
            output_filename = f"{name_without_ext}_{label}_conf{confidence:.2f}.jpg"
            output_path = os.path.join(model_output_dir, output_filename)
            
            create_rgb_overlay(img_path, pred, confidence, output_path)
        
        print(f"✅ Hoàn thành {model_name}: {len(preds)} ảnh")
    
    # Ensemble methods
    if len(models) > 1:
        model_predictions = list(all_predictions.values())
        model_probabilities = list(all_probabilities.values())
        
        # Hard Voting
        if 'hard_voting' in args.models:
            print("🗳️ Thực hiện Hard Voting Ensemble...")
            hard_preds = hard_voting_ensemble(model_predictions)
            
            # Tạo output cho hard voting
            ensemble_dir = os.path.join(args.output_dir, 'hard_voting_ensemble')
            os.makedirs(ensemble_dir, exist_ok=True)
            
            for i, (pred, img_path) in enumerate(zip(hard_preds, paths)):
                filename = os.path.basename(img_path)
                name_without_ext = os.path.splitext(filename)[0]
                label = "flood" if pred == 1 else "no_flood"
                
                output_filename = f"{name_without_ext}_{label}_hard_voting.jpg"
                output_path = os.path.join(ensemble_dir, output_filename)
                
                create_rgb_overlay(img_path, pred, 0.8, output_path)
            
            print(f"✅ Hard Voting Ensemble: {len(hard_preds)} ảnh")
        
        # Soft Voting  
        if 'soft_voting' in args.models:
            print("🎯 Thực hiện Soft Voting Ensemble...")
            soft_preds, soft_probs = soft_voting_ensemble(model_probabilities)
            
            # Tạo output cho soft voting
            ensemble_dir = os.path.join(args.output_dir, 'soft_voting_ensemble')
            os.makedirs(ensemble_dir, exist_ok=True)
            
            for i, (pred, prob, img_path) in enumerate(zip(soft_preds, soft_probs, paths)):
                filename = os.path.basename(img_path)
                name_without_ext = os.path.splitext(filename)[0]
                
                confidence = max(prob)
                label = "flood" if pred == 1 else "no_flood"
                
                output_filename = f"{name_without_ext}_{label}_conf{confidence:.2f}_soft_voting.jpg"
                output_path = os.path.join(ensemble_dir, output_filename)
                
                create_rgb_overlay(img_path, pred, confidence, output_path)
            
            print(f"✅ Soft Voting Ensemble: {len(soft_preds)} ảnh")
    
    # Thống kê tổng quan
    print("\n📊 THỐNG KÊ TỔNG QUAN:")
    for model_name, preds in all_predictions.items():
        flood_count = sum(preds)
        no_flood_count = len(preds) - flood_count
        print(f"  {model_name}: {flood_count} lũ, {no_flood_count} không lũ")
    
    print(f"\n🎉 Hoàn thành! Kết quả lưu tại: {args.output_dir}")

if __name__ == "__main__":
    main() 