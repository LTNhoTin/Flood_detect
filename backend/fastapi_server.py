"""
FastAPI server để host 4 mô hình phân loại lũ S2 (.tif) trong thư mục training/models.

Cách chạy:
    poetry install  # hoặc pip install -r requirements.txt
    uvicorn backend.fastapi_server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health                 -> Kiểm tra trạng thái server
    GET  /models                 -> Danh sách mô hình có sẵn
    POST /predict/{model_name}   -> Upload file .tif, trả về dự đoán JSON
    GET  /download/{filename}    -> Tải ảnh kết quả overlay hoặc RGB preview
"""

import os
import uuid
from datetime import datetime
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import rasterio
import cv2
import timm
from torchvision import models
from transformers import ViTForImageClassification

# Thư mục chính
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODELS_DIR = os.path.join(ROOT_DIR, "training", "models")
UPLOAD_FOLDER = os.path.join("/tmp", "flood_uploads")
OUTPUT_FOLDER = os.path.join("/tmp", "flood_outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Thiết lập device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FastAPI app
app = FastAPI(title="Flood Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bộ nhớ cache model
_loaded_models: Dict[str, torch.nn.Module] = {}

# Map tên model sang file checkpoint
MODEL_FILES = {
    "resnet": "resnet_best.pth",
    "densenet": "densenet_best.pth",
    "efficientnet": "efficientnet_best.pth",
    "vit": "vit_best.pth",
}

# Thông tin ensemble
ENSEMBLE_INFO = {
    "hard_voting": "Hard Voting Ensemble",
    "soft_voting": "Soft Voting Ensemble",
}

# Helper functions
def load_model(model_name: str) -> torch.nn.Module:
    """Load model lên GPU/CPU và cache"""
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    checkpoint_path = os.path.join(MODELS_DIR, MODEL_FILES[model_name])
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint cho model {model_name}: {checkpoint_path}")

    if model_name == "resnet":
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    elif model_name == "densenet":
        model = models.densenet121(pretrained=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    elif model_name == "efficientnet":
        model = timm.create_model("efficientnet_b0", pretrained=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    elif model_name == "vit":
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=2, ignore_mismatched_sizes=True
        )
    else:
        raise ValueError(f"Model {model_name} chưa được hỗ trợ")

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    _loaded_models[model_name] = model
    return model

def preprocess_image(paths: List[str]) -> torch.Tensor:
    """Đọc và chuẩn hóa ảnh Sentinel-2 (B04, B03, B02) về tensor 3x224x224"""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Đồng bộ với huấn luyện
    ])

    # Tìm band B04, B03, B02
    path_map = {os.path.basename(p).upper(): p for p in paths}
    b4 = path_map.get('B04.TIF') or next((p for p in paths if 'B04' in p.upper()), None)
    b3 = path_map.get('B03.TIF') or next((p for p in paths if 'B03' in p.upper()), None)
    b2 = path_map.get('B02.TIF') or next((p for p in paths if 'B02' in p.upper()), None)

    if not all([b4, b3, b2]):
        raise HTTPException(status_code=400, detail="Thiếu band cần thiết (B04, B03, B02)")

    imgs = []
    base_shape = None
    for p in [b4, b3, b2]:
        with rasterio.open(p) as src:
            arr = src.read(1).astype(float)
            if base_shape is None:
                base_shape = arr.shape
            else:
                arr = cv2.resize(arr, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_NEAREST)
            p2, p98 = np.percentile(arr, (2, 98))
            arr = np.clip((arr - p2) / (p98 - p2) * 255, 0, 255) if p98 > p2 else np.zeros_like(arr)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            arr = clahe.apply(arr.astype(np.uint8))
            imgs.append(arr)
    
    img = np.stack(imgs, axis=-1).astype(np.uint8)
    img = cv2.resize(img, (224, 224))
    img_pil = Image.fromarray(img)
    return transform(img_pil).unsqueeze(0)

def create_overlay(image_path: str, prediction: int, confidence: float, output_path: str) -> None:
    """Tạo ảnh RGB overlay kết quả và lưu về output_path"""
    with rasterio.open(image_path) as src:
        img = src.read([4, 3, 2]) if src.count >= 4 else src.read([1, 1, 1])
        img = np.transpose(img, (1, 2, 0))
    
    rgb = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        ch = img[:, :, i].astype(np.float32)
        p2, p98 = np.percentile(ch, (2, 98))
        rgb[:, :, i] = np.clip((ch - p2) / (p98 - p2), 0, 1) if p98 > p2 else np.zeros_like(ch)
    
    rgb = (rgb * 255).astype(np.uint8)
    overlay = rgb.copy()
    
    color = (255, 0, 0) if prediction == 1 else (0, 255, 0)
    text = "FLOOD" if prediction == 1 else "NO FLOOD"
    overlay[:50, :50, :] = cv2.addWeighted(overlay[:50, :50, :], 0.4, np.full((50, 50, 3), color, dtype=np.uint8), 0.6, 0)
    cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(overlay, f"{confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def save_rgb_preview(paths: List[str], output_path: str) -> None:
    """Ghép 3 band (B04, B03, B02) thành RGB jpg"""
    path_map = {os.path.basename(p).upper(): p for p in paths}
    b4 = path_map.get('B04.TIF') or next((p for p in paths if 'B04' in p.upper()), None)
    b3 = path_map.get('B03.TIF') or next((p for p in paths if 'B03' in p.upper()), None)
    b2 = path_map.get('B02.TIF') or next((p for p in paths if 'B02' in p.upper()), None)

    if not all([b4, b3, b2]):
        raise HTTPException(status_code=400, detail="Thiếu band cần thiết (B04, B03, B02) cho preview")

    imgs = []
    base_shape = None
    for p in [b4, b3, b2]:
        with rasterio.open(p) as src:
            arr = src.read(1).astype(float)
            if base_shape is None:
                base_shape = arr.shape
            else:
                arr = cv2.resize(arr, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_NEAREST)
            p2, p98 = np.percentile(arr, (2, 98))
            arr = np.clip((arr - p2) / (p98 - p2) * 255, 0, 255) if p98 > p2 else np.zeros_like(arr)
            imgs.append(arr)
    
    img = np.stack(imgs, axis=-1).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# API endpoints
@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "models_loaded": list(_loaded_models.keys()),
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/models")
async def list_models() -> Dict[str, List[Dict[str, str]]]:
    models = []
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models.append({"name": name, "checkpoint": path, "label": name.upper()})
    if len(models) > 1:
        for k, v in ENSEMBLE_INFO.items():
            models.append({"name": k, "label": v, "path": "ensemble"})
    return {"models": models}

@app.post("/predict/{model_name}")
async def predict(model_name: str, files: List[UploadFile] = File(...)):
    model_name = model_name.lower()
    is_ensemble = model_name in ENSEMBLE_INFO
    if not is_ensemble and model_name not in MODEL_FILES:
        raise HTTPException(status_code=404, detail=f"Model {model_name} không được hỗ trợ")

    # Lưu file tạm
    file_id = str(uuid.uuid4())
    saved_paths = []
    for fobj in files:
        fname = os.path.basename(fobj.filename)
        path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{fname}")
        with open(path, "wb") as fw:
            fw.write(await fobj.read())
        saved_paths.append(path)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="Không có file được upload")

    # Xác định B04 cho overlay (nếu có)
    path_map = {os.path.basename(p).upper(): p for p in saved_paths}
    b4_path = path_map.get('B04.TIF') or next((p for p in saved_paths if 'B04' in p.upper()), None)

    # Tiền xử lý
    try:
        tensor = preprocess_image(saved_paths)
    except HTTPException as e:
        for p in saved_paths:
            os.remove(p)
        raise e
    except Exception as err:
        for p in saved_paths:
            os.remove(p)
        raise HTTPException(status_code=500, detail=f"Lỗi tiền xử lý ảnh: {err}")

    # Dự đoán
    if is_ensemble:
        preds = []
        probs_list = []
        for single in MODEL_FILES.keys():
            path = os.path.join(MODELS_DIR, MODEL_FILES[single])
            if not os.path.exists(path):
                continue
            m = load_model(single)
            with torch.no_grad():
                out = m(tensor.to(DEVICE))
                if hasattr(out, "logits"):
                    out = out.logits
                pb = F.softmax(out, dim=1)
                preds.append(torch.argmax(pb, 1).item())
                probs_list.append(pb.cpu().numpy()[0])
        if not preds:
            for p in saved_paths:
                os.remove(p)
            raise HTTPException(status_code=500, detail="Không có model nào cho ensemble")
        if model_name == "hard_voting":
            from collections import Counter
            pred = Counter(preds).most_common(1)[0][0]
            confidence = Counter(preds).most_common(1)[0][1] / len(preds)  # Tỷ lệ phiếu bầu
        else:  # soft_voting
            avg_probs = np.mean(probs_list, axis=0)
            pred = int(np.argmax(avg_probs))
            confidence = float(np.max(avg_probs))
    else:
        model = load_model(model_name)
        with torch.no_grad():
            outputs = model(tensor.to(DEVICE))
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item()

    # Lưu overlay & preview
    overlay_filename = f"{file_id}_{model_name}_result.jpg"
    overlay_path = os.path.join(OUTPUT_FOLDER, overlay_filename)
    preview_filename = f"{file_id}_rgb.jpg"
    preview_path = os.path.join(OUTPUT_FOLDER, preview_filename)

    try:
        # Tạo preview RGB (màu)
        save_rgb_preview(saved_paths, preview_path)

        # Dùng preview RGB để vẽ overlay
        img = cv2.imread(preview_path)
        color = (0,255,0) if pred==0 else (0,0,255)
        label = "NO FLOOD" if pred==0 else "FLOOD"
        cv2.rectangle(img, (0,0), (180,60), color, -1)
        cv2.putText(img, label, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(img, f"{confidence:.2f}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imwrite(overlay_path, img)
    except Exception as err:
        for p in saved_paths:
            os.remove(p)
        raise HTTPException(status_code=500, detail=f"Lỗi tạo overlay/preview: {err}")

    # Xóa file tạm
    for p in saved_paths:
        os.remove(p)

    return JSONResponse({
        "model": model_name,
        "prediction": "flood" if pred == 1 else "no_flood",
        "confidence": confidence,
        "result_url": f"/download/{overlay_filename}",
        "original_url": f"/download/{preview_filename}",
        "file_id": file_id,
        "success": True
    })

@app.get("/download/{filename}")
async def download(filename: str):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File không tồn tại")
    return FileResponse(file_path, media_type="image/jpeg")