# SEN12FLOODS - EDA & Analysis

Thư mục này chứa các files phân tích dữ liệu khám phá (EDA) cho dataset SEN12FLOODS.

## 📁 Files trong thư mục

### 🔧 **Code Files**
- **`EDA.py`** - Code phân tích EDA hoàn chỉnh (10 cells cho Jupyter)
- **`training_example.py`** - ~~Example cách sử dụng flood_labels trong training~~ (đã xóa)

### 📊 **Report Files**  
- **`EDA_Report.md`** - Báo cáo EDA chi tiết bằng tiếng Việt
- **`README.md`** - File này (hướng dẫn sử dụng)

### 📋 **Output Files** (sau khi chạy EDA.py)
- **`flood_labels_mapping.json`** - ⚠️ **CẦN THIẾT CHO TRAINING** - Mapping folder → flood label

## 🚀 Cách sử dụng

### 1. Chạy EDA Analysis

### 2. Xem kết quả

- **Plots**: Tất cả biểu đồ sẽ hiển thị inline (không save file)
- **Variables**: Các variables quan trọng sẽ được tạo trong memory
- **Labels**: File `flood_labels_mapping.json` sẽ được tạo cho training

### 3. Đọc báo cáo

Mở `EDA_Report.md` để xem báo cáo phân tích chi tiết bao gồm:
- Problem Definition & Understanding  
- Data Understanding
- Data Visualization & Insights
- Proposed Pipeline & Models

## 📊 Kết quả EDA chính

```
📈 DATASET STATISTICS:
├── Total files: 36,053 TIFF files
├── Sentinel-1: 9,208 files (25.5%)  
├── Sentinel-2: 26,844 files (74.5%)
├── Flooding: 24,333 files (67.5%)
├── Non-flooding: 11,720 files (32.5%)
├── Unique folders: 336 scenes
└── Average files/folder: 107.3
```

## ⚠️ Quan trọng cho Training

**File bắt buộc:** `flood_labels_mapping.json`
- Được tạo tự động khi chạy EDA.py
- Chứa mapping từ folder_name → flood_label (0/1)
- Cần thiết cho việc load labels trong training
- Tiết kiệm thời gian so với parse S1/S2 JSON mỗi lần

## 📋 Variables được tạo

Sau khi chạy EDA.py, các variables sau sẽ có trong memory:

```python
tif_files          # List[str] - Tất cả TIFF file paths  
flood_labels       # Dict[str, int] - Folder → label mapping
sentinel1_files    # List[str] - Sentinel-1 file paths
sentinel2_files    # List[str] - Sentinel-2 file paths  
df_sensors         # DataFrame - File info với sensor classification
summary_stats      # Dict - Tất cả thống kê quan trọng
```

## 🔧 Customization

### Thay đổi đường dẫn dữ liệu
```python
# Trong CELL 2 của EDA.py
data_dir = "/your/path/to/sen12floods1"  # ← Thay đổi này
```

### Thay đổi số samples hiển thị
```python  
# Trong CELL 8
n_samples = 5  # ← Thay đổi số ảnh sample hiển thị
```

### Thay đổi sample size cho metadata analysis
```python
# Trong CELL 9  
sample_size = min(100, len(tif_files))  # ← Thay đổi số files phân tích
```

## 🐛 Troubleshooting

**Lỗi "Thư mục dữ liệu không tồn tại"**
→ Cập nhật `data_dir` trong CELL 2

**Lỗi thiếu libraries**
→ Install: `pip install rasterio opencv-python pandas matplotlib seaborn`

**Không tạo được flood_labels_mapping.json**  
→ Kiểm tra quyền write trong thư mục data

**Plots không hiển thị trong Jupyter**
→ Thêm `%matplotlib inline` ở đầu notebook

## 📞 Support

Nếu có vấn đề, kiểm tra:
1. Đường dẫn `data_dir` đúng chưa
2. Các thư viện đã install đủ chưa  
3. File S1list.json và S2list.json có tồn tại không
4. Quyền truy cập thư mục có đủ không

---

*Generated from SEN12FLOODS EDA Analysis* 