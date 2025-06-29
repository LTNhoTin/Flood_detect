import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import CompareImage from "react-compare-image";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:2000";

export default function ResultPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { files, preview, model } = location.state || {};
  const [resultUrl, setResultUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [fileId, setFileId] = useState(null);
  const [originalUrl, setOriginalUrl] = useState(null);

  useEffect(() => {
    if (!files || !model) {
      navigate("/");
      return;
    }
    
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      
      const formData = new FormData();
      files.forEach(f=>formData.append("files", f));
      
      try {
        const response = await fetch(`${API_BASE_URL}/predict/${model}`, {
          method: "POST",
          body: formData,
        });

        let data = null;
        try {
          data = await response.json();
        } catch (_) {}

        if (!response.ok) {
          const msg = data?.detail || data?.error || `HTTP ${response.status}`;
          throw new Error(msg);
        }
        
        if (data.success) {
          setResultUrl(`${API_BASE_URL}${data.result_url}`);
          const orig = data.original_url ? `${API_BASE_URL}${data.original_url}` : preview;
          setOriginalUrl(orig);
          setPrediction(data.prediction);
          setConfidence(data.confidence);
          setFileId(data.file_id);
        } else {
          throw new Error(data.error || 'Unknown error occurred');
        }
      } catch (err) {
        console.error("API Error:", err);
        setError(`Lỗi xử lý ảnh: ${err.message}`);
        setResultUrl(null);
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [files, model, navigate]);

  // Cleanup function khi unmount
  useEffect(() => {
    return () => {
      if (fileId) {
        fetch(`${API_BASE_URL}/cleanup/${fileId}`, {
          method: 'DELETE'
        }).catch(err => console.log('Cleanup error:', err));
      }
    };
  }, [fileId]);

  const getModelDisplayName = (modelValue) => {
    const modelMap = {
      'resnet': 'ResNet',
      'densenet': 'DenseNet', 
      'efficientnet': 'EfficientNet',
      'vit': 'Vision Transformer',
      'hard_voting': 'Hard Voting Ensemble',
      'soft_voting': 'Soft Voting Ensemble'
    };
    return modelMap[modelValue] || modelValue.toUpperCase();
  };

  const getPredictionDisplay = (pred) => {
    return pred === 'flood' ? 'Có lũ' : 'Không có lũ';
  };

  const getPredictionColor = (pred) => {
    return pred === 'flood' ? 'text-red-600' : 'text-green-600';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 flex flex-col font-sans">
      {/* Header */}
      <header className="flex items-center px-10 py-6 bg-white shadow-sm">
        <img src="/logo.png" alt="FloodAI" className="h-10 mr-4" />
        <span className="text-2xl font-bold text-gray-800 select-none tracking-wide">
          Flood<span className="text-blue-600">Detector.AI</span>
        </span>
        <button
          className="ml-auto rounded-full px-6 py-2 bg-gradient-to-r from-gray-100 to-gray-200 font-medium text-gray-700 hover:from-gray-200 hover:to-gray-300 transition-all duration-200 shadow-sm"
          onClick={() => navigate("/")}
        >
          ← Quay lại trang đầu
        </button>
      </header>

      {/* Main content */}
      <main className="flex flex-col items-center justify-center flex-1 px-4 py-8">
        <div className="w-full max-w-4xl">
          {/* Header Info */}
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Kết quả phân tích lũ lụt
            </h2>
            <div className="flex items-center justify-center gap-4 text-gray-600">
              <span className="px-3 py-1 bg-blue-100 rounded-full text-sm font-medium">
                Model: {getModelDisplayName(model)}
              </span>
              {confidence && (
                <span className="px-3 py-1 bg-green-100 rounded-full text-sm font-medium">
                  Độ tin cậy: {(confidence * 100).toFixed(1)}%
                </span>
              )}
            </div>
          </div>

          {/* Result Display */}
          <div className="bg-white rounded-2xl shadow-xl p-6 mb-6">
            {loading ? (
              <div className="flex flex-col items-center justify-center h-96">
                <div className="relative">
                  <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <svg className="w-6 h-6 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M10 12a2 2 0 100-4 2 2 0 000 4z"/>
                      <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd"/>
                    </svg>
                  </div>
                </div>
                <p className="text-gray-600 text-lg mt-4 animate-pulse">
                  Đang xử lý với {getModelDisplayName(model)}...
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  Vui lòng chờ, quá trình có thể mất vài giây
                </p>
              </div>
            ) : error ? (
              <div className="flex flex-col items-center justify-center h-96 text-center">
                <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mb-4">
                  <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold text-red-600 mb-2">Lỗi xử lý</h3>
                <p className="text-red-500 mb-4">{error}</p>
                <button
                  onClick={() => window.location.reload()}
                  className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                >
                  Thử lại
                </button>
              </div>
            ) : resultUrl ? (
              <>
                {/* Prediction Result */}
                {prediction && (
                  <div className="text-center mb-6">
                    <div className={`inline-flex items-center gap-2 px-6 py-3 rounded-full text-lg font-bold ${
                      prediction === 'flood' 
                        ? 'bg-red-100 text-red-700 border-2 border-red-200' 
                        : 'bg-green-100 text-green-700 border-2 border-green-200'
                    }`}>
                      {prediction === 'flood' ? (
                        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd"/>
                        </svg>
                      ) : (
                        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                        </svg>
                      )}
                      {getPredictionDisplay(prediction)}
                    </div>
                  </div>
                )}

                {/* Hiển thị ảnh kết quả AI */}
                <div className="rounded-xl overflow-hidden shadow-lg max-w-xl mx-auto w-full">
                  <img src={resultUrl} alt="Kết quả AI" className="w-full h-auto" />
                </div>
              </>
            ) : (
              <div className="text-center text-gray-500">Không có kết quả!</div>
            )}
          </div>

          {/* Download Buttons */}
          {!loading && !error && resultUrl && (
            <div className="flex justify-center gap-4 mb-6">
              <a
                href={resultUrl}
                download={`flood_result_${model}.jpg`}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg hover:from-green-700 hover:to-emerald-700 font-medium transition-all duration-200 shadow-lg transform hover:scale-105"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Tải ảnh kết quả
              </a>
              <a
                href={originalUrl || preview}
                download="original_image.jpg"
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-gray-500 to-gray-600 text-white rounded-lg hover:from-gray-600 hover:to-gray-700 font-medium transition-all duration-200 shadow-lg transform hover:scale-105"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                Tải ảnh gốc
              </a>
            </div>
          )}

          {/* Model and Stats Info */}
          {!loading && !error && (
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold text-blue-600">{getModelDisplayName(model)}</div>
                  <div className="text-sm text-gray-600">Model sử dụng</div>
                </div>
                {confidence && (
                  <div>
                    <div className="text-2xl font-bold text-green-600">{(confidence * 100).toFixed(1)}%</div>
                    <div className="text-sm text-gray-600">Độ tin cậy</div>
                  </div>
                )}
                {prediction && (
                  <div>
                    <div className={`text-2xl font-bold ${getPredictionColor(prediction)}`}>
                      {getPredictionDisplay(prediction)}
                    </div>
                    <div className="text-sm text-gray-600">Kết quả dự đoán</div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
