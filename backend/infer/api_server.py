#!/usr/bin/env python3
"""
Flask API Server cho Flood Detection Models
"""

import os
import uuid
import shutil
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import rasterio
import numpy as np
import cv2
import timm
import matplotlib.pyplot as plt

# Import các functions từ inference.py
import sys
sys.path.append('/home/nhotin/tinltn/dat301m/as/sen12floods/backend')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Configuration
UPLOAD_FOLDER = '/tmp/flood_uploads'
OUTPUT_FOLDER = '/tmp/flood_outputs'
MODELS_DIR = '/home/nhotin/tinltn/dat301m/as/sen12floods/training/models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Thiết lập device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"API Server sử dụng device: {device}")

# Global models storage
loaded_models = {}

def load_model(model_path, model_name):
    """Load model từ checkpoint với cấu trúc phù hợp"""
    try:
        if model_name in loaded_models:
            return loaded_models[model_name]
            
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
        
        # Cache model
        loaded_models[model_name] = model
        print(f"Đã load model {model_name}")
        return model
        
    except Exception as e:
        print(f"Lỗi load model {model_name}: {str(e)}")
        return None

def preprocess_image(image_path):
    """Preprocess ảnh cho inference"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        with rasterio.open(image_path) as src:
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
            
            if transform:
                image = transform(image)
            
            return image.unsqueeze(0)  # Add batch dimension
            
    except Exception as e:
        print(f"Lỗi preprocess ảnh {image_path}: {str(e)}")
        return None

def create_rgb_overlay(image_path, prediction, confidence, output_path):
    """Tạo ảnh RGB với overlay dự đoán (copy từ inference.py)"""
    try:
        with rasterio.open(image_path) as src:
            image = src.read()
            if len(image.shape) == 3:
                image = np.transpose(image, (1, 2, 0))
            
            # Xử lý dữ liệu single-band Sentinel-2
            if len(image.shape) == 2 or image.shape[-1] == 1:
                if image.shape[-1] == 1:
                    image = image.squeeze()
                
                # Normalize về 0-1 an toàn
                p2, p98 = np.percentile(image, 2), np.percentile(image, 98)
                if p98 > p2:
                    image_norm = (image - p2) / (p98 - p2)
                else:
                    image_norm = np.zeros_like(image, dtype=np.float32)
                image_norm = np.clip(image_norm, 0, 1)
                
                # Tạo colormap tự nhiên dựa trên band type
                filename = os.path.basename(image_path).lower()
                if any(band in filename for band in ['b04', 'b03', 'b02']):
                    rgb_image = np.stack([image_norm, image_norm, image_norm], axis=-1)
                elif any(band in filename for band in ['b08', 'b8a', 'b11', 'b12']):
                    cmap = plt.cm.viridis
                    rgb_image = cmap(image_norm)[:, :, :3]
                else:
                    rgb_image = np.stack([image_norm, image_norm, image_norm], axis=-1)
            else:
                if image.shape[-1] > 3:
                    image = image[:, :, :3]
                
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
            
            if prediction == 1:  # Lũ
                overlay[:50, :50, 0] = np.minimum(255, overlay[:50, :50, 0] + 100)
                overlay[:50, :50, 1] = np.maximum(0, overlay[:50, :50, 1] - 30)
                overlay[:50, :50, 2] = np.maximum(0, overlay[:50, :50, 2] - 30)
                
                cv2.putText(overlay, 'FLOOD', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(overlay, f'{confidence:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:  # Không lũ
                overlay[:50, :50, 0] = np.maximum(0, overlay[:50, :50, 0] - 30)
                overlay[:50, :50, 1] = np.minimum(255, overlay[:50, :50, 1] + 100)
                overlay[:50, :50, 2] = np.maximum(0, overlay[:50, :50, 2] - 30)
                
                cv2.putText(overlay, 'NO FLOOD', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(overlay, f'{confidence:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Lưu ảnh
            cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            return True
            
    except Exception as e:
        print(f"Lỗi tạo RGB overlay cho {image_path}: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'models_loaded': list(loaded_models.keys()),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/models', methods=['GET'])
def get_available_models():
    """Lấy danh sách models có sẵn"""
    models = []
    model_files = {
        'resnet': 'resnet_best.pth',
        'densenet': 'densenet_best.pth', 
        'efficientnet': 'efficientnet_best.pth',
        'vit': 'vit_best.pth'
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(model_path):
            models.append({
                'name': model_name,
                'label': model_name.upper(),
                'path': model_path
            })
    
    # Thêm ensemble methods
    if len(models) > 1:
        models.extend([
            {'name': 'hard_voting', 'label': 'Hard Voting Ensemble', 'path': 'ensemble'},
            {'name': 'soft_voting', 'label': 'Soft Voting Ensemble', 'path': 'ensemble'}
        ])
    
    return jsonify({'models': models})

@app.route('/predict/<model_name>', methods=['POST'])
def predict_flood(model_name):
    """Predict flood cho một model cụ thể"""
    try:
        # Kiểm tra file upload
        if 'file' not in request.files:
            return jsonify({'error': 'Không có file được upload'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Không có file được chọn'}), 400
        
        # Lưu file upload
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{filename}")
        file.save(input_path)
        
        # Load model nếu chưa có
        if model_name not in ['hard_voting', 'soft_voting']:
            model_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
            if not os.path.exists(model_path):
                return jsonify({'error': f'Model {model_name} không tồn tại'}), 404
            
            model = load_model(model_path, model_name)
            if model is None:
                return jsonify({'error': f'Không thể load model {model_name}'}), 500
        
        # Preprocess ảnh
        input_tensor = preprocess_image(input_path)
        if input_tensor is None:
            return jsonify({'error': 'Lỗi khi xử lý ảnh'}), 500
        
        # Inference
        with torch.no_grad():
            if model_name in ['hard_voting', 'soft_voting']:
                # Ensemble prediction - cần load tất cả models
                predictions = []
                probabilities = []
                
                for single_model in ['resnet', 'densenet', 'efficientnet', 'vit']:
                    single_model_path = os.path.join(MODELS_DIR, f"{single_model}_best.pth")
                    if os.path.exists(single_model_path):
                        single_model_obj = load_model(single_model_path, single_model)
                        if single_model_obj:
                            outputs = single_model_obj(input_tensor.to(device))
                            probs = F.softmax(outputs, dim=1)
                            pred = torch.argmax(probs, dim=1)
                            
                            predictions.append(pred.cpu().numpy()[0])
                            probabilities.append(probs.cpu().numpy()[0])
                
                if not predictions:
                    return jsonify({'error': 'Không có model nào để ensemble'}), 500
                
                if model_name == 'hard_voting':
                    from collections import Counter
                    final_prediction = Counter(predictions).most_common(1)[0][0]
                    final_confidence = 0.8  # Default confidence cho hard voting
                else:  # soft_voting
                    avg_probs = np.mean(probabilities, axis=0)
                    final_prediction = np.argmax(avg_probs)
                    final_confidence = float(np.max(avg_probs))
            else:
                # Single model prediction
                outputs = model(input_tensor.to(device))
                probs = F.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1)
                
                final_prediction = int(pred.cpu().numpy()[0])
                final_confidence = float(torch.max(probs).cpu().numpy())
        
        # Tạo ảnh output
        output_filename = f"{file_id}_{model_name}_result.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        success = create_rgb_overlay(input_path, final_prediction, final_confidence, output_path)
        if not success:
            return jsonify({'error': 'Lỗi tạo ảnh kết quả'}), 500
        
        # Cleanup input file
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'model': model_name,
            'prediction': 'flood' if final_prediction == 1 else 'no_flood',
            'confidence': final_confidence,
            'result_url': f'/download/{output_filename}',
            'file_id': file_id
        })
        
    except Exception as e:
        print(f"Error in predict_flood: {str(e)}")
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_result(filename):
    """Download file kết quả"""
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=False)
    else:
        return jsonify({'error': 'File không tồn tại'}), 404

@app.route('/cleanup/<file_id>', methods=['DELETE'])
def cleanup_files(file_id):
    """Cleanup files cho một file_id"""
    try:
        # Tìm và xóa tất cả files với file_id
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            for file in os.listdir(folder):
                if file.startswith(file_id):
                    os.remove(os.path.join(folder, file))
        
        return jsonify({'success': True, 'message': 'Files đã được xóa'})
    except Exception as e:
        return jsonify({'error': f'Lỗi cleanup: {str(e)}'}), 500

if __name__ == '__main__':
    print("Khởi động Flood Detection API Server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Models folder: {MODELS_DIR}")
    
    # Print all routes for debugging
    print("\nAvailable routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.methods} {rule.rule}")
    
    app.run(host='0.0.0.0', port=8001, debug=True) 