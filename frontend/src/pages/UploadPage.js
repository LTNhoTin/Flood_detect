import React, { useRef, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:2000";

const FALLBACK_MODELS = [
  { label: "ResNet", value: "resnet", description: "Deep residual network - Nhanh và ổn định" },
  { label: "DenseNet", value: "densenet", description: "Dense connection - Hiệu quả với features" },
  { label: "EfficientNet", value: "efficientnet", description: "Tối ưu kích thước và độ chính xác" },
  { label: "Vision Transformer", value: "vit", description: "Transformer cho computer vision" },
  { label: "Hard Voting Ensemble", value: "hard_voting", description: "Kết hợp voting cứng" },
  { label: "Soft Voting Ensemble", value: "soft_voting", description: "Kết hợp trung bình xác suất" },
];

export default function UploadPage() {
  const [inputFiles, setInputFiles] = useState([]);
  const [inputPreview, setInputPreview] = useState(null); // vẫn giữ để gửi backend nhưng không hiển thị
  const [models, setModels] = useState(FALLBACK_MODELS);
  const [model, setModel] = useState(FALLBACK_MODELS[0].value);
  const inputRef = useRef();
  const navigate = useNavigate();

  // Fetch models từ backend
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/models`);
        if (!res.ok) throw new Error("Không lấy được danh sách models");
        const data = await res.json();
        if (data.models && data.models.length) {
          const mapped = data.models.map((m) => ({
            label: m.label || m.name.toUpperCase(),
            value: m.name,
            description: m.label || m.name,
          }));
          setModels(mapped);
          setModel(mapped[0].value);
        }
      } catch (err) {
        console.log("Model list error", err);
      }
    };
    fetchModels();
  }, []);

  const handleImageChange = (e) => {
    const filesArr = Array.from(e.target.files || []);
    if (filesArr.length) {
      setInputFiles(filesArr);
      setInputPreview(URL.createObjectURL(filesArr[0]));
    }
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    const filesArr = Array.from(e.dataTransfer.files || []);
    if (filesArr.length) {
      setInputFiles(filesArr);
      setInputPreview(URL.createObjectURL(filesArr[0]));
    }
  };
  
  const handleDragOver = (e) => e.preventDefault();
  const handleClick = () => inputRef.current.click();

  // Chuyển sang trang result, truyền file + model + preview
  const handleGoResult = () => {
    if (!inputFiles.length) return;
    navigate("/result", { state: { files: inputFiles, preview: inputPreview, model } });
  };

  const selectedModel = models.find(m => m.value === model);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex flex-col font-sans">
      <header className="flex items-center px-10 py-6 bg-white shadow-sm">
        <img src="/logo.png" alt="FloodAI" className="h-10 mr-4" />
        <span className="text-2xl font-bold text-gray-800 select-none tracking-wide">
          Flood<span className="text-blue-600">Detector.AI</span>
        </span>
      </header>
      
      <main className="flex-1 flex flex-col lg:flex-row items-center justify-center px-6 md:px-16 lg:px-28 py-10 gap-12 w-full">
        {/* Left Section */}
        <div className="flex-1 flex flex-col items-start mt-6 md:mt-16 min-w-[320px]">
          <div className="relative w-64 h-64 mb-7 select-none">
            <div className="absolute -top-8 -left-8 w-72 h-60 bg-blue-200 rounded-full blur-3xl opacity-40 z-0"></div>
            <img
              src="/ava.png"
              alt="Satellite Flood"
              className="w-64 h-64 object-cover rounded-2xl z-10 relative border-4 border-white shadow-xl"
            />
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold leading-tight text-gray-800 mb-6">
            Nhận diện vùng lũ<br />từ ảnh vệ tinh
          </h1>
          <div className="flex items-center text-xl font-semibold mb-2">
            <span>Xử lý tự động bằng </span>
            <span className="bg-gradient-to-r from-blue-400 to-purple-500 text-white rounded-lg px-3 py-1 ml-2 font-bold text-lg shadow-lg">AI</span>
          </div>
          <div className="mt-2 text-gray-600 text-base leading-relaxed max-w-md">
            Chọn model AI phù hợp, tải ảnh vệ tinh Sentinel-2 (.tif), hệ thống sẽ tự động phân tích và khoanh vùng lũ.
          </div>
        </div>

        {/* Right Section - Upload */}
        <div className="flex-1 flex flex-col items-center">
          {/* Model Selection */}
          <div className="mb-6 w-full flex flex-col items-center">
            <label className="text-lg font-semibold mb-4 text-gray-800">Chọn mô hình AI:</label>
            <div className="grid grid-cols-2 gap-3 w-full max-w-lg">
              {models.map((m) => (
                <label 
                  key={m.value} 
                  className={`flex flex-col p-3 border-2 rounded-lg cursor-pointer transition-all ${
                    model === m.value 
                      ? 'border-blue-500 bg-blue-50 shadow-md' 
                      : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-center mb-1">
                    <input
                      type="radio"
                      name="model"
                      value={m.value}
                      checked={model === m.value}
                      onChange={() => setModel(m.value)}
                      className="mr-2 text-blue-600"
                    />
                    <span className="font-semibold text-gray-800">{m.label}</span>
                  </div>
                  <span className="text-xs text-gray-600 leading-tight">{m.description}</span>
                </label>
              ))}
            </div>
            
            {/* Selected Model Info */}
            {selectedModel && (
              <div className="mt-4 p-3 bg-blue-100 rounded-lg border border-blue-200">
                <div className="text-sm font-semibold text-blue-800">Đã chọn: {selectedModel.label}</div>
                <div className="text-xs text-blue-600 mt-1">{selectedModel.description}</div>
              </div>
            )}
          </div>

          {/* Upload Area */}
          <div
            className="bg-white rounded-3xl shadow-2xl p-8 flex flex-col items-center justify-center w-full max-w-md border-2 border-dashed border-blue-200 hover:border-blue-400 transition-all duration-300 mb-4 cursor-pointer hover:shadow-3xl"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={handleClick}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".tif,.tiff,image/*"
              multiple
              onChange={handleImageChange}
              className="hidden"
            />
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 bg-blue-100 rounded-full flex items-center justify-center">
                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <button className="bg-gradient-to-r from-blue-600 to-purple-600 text-white text-lg font-bold rounded-full px-8 py-4 hover:from-blue-700 hover:to-purple-700 mb-3 shadow-lg transform hover:scale-105 transition-all duration-200" type="button">
                Chọn ảnh vệ tinh
              </button>
              <p className="text-gray-500 mb-2">hoặc kéo-thả ảnh vào đây</p>
              <p className="text-xs text-gray-400">Chỉ hỗ trợ: Sentinel-2 *.tif / *.tiff</p>
            </div>
            
            {/* Không hiển thị preview để tránh lỗi browser đối với TIFF lớn */}
          </div>

          {/* Process Button */}
          {inputFiles.length > 0 && (
            <button
              onClick={handleGoResult}
              className="bg-gradient-to-r from-green-600 to-emerald-600 text-white font-bold px-8 py-3 rounded-full mb-4 transition-all duration-300 hover:from-green-700 hover:to-emerald-700 transform hover:scale-105 shadow-lg flex items-center gap-2"
              type="button"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Phát hiện lũ với {selectedModel?.label}
            </button>
          )}
        </div>
      </main>
    </div>
  );
}
