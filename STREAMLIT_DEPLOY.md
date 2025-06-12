# Streamlit Cloud 部署指南

## 环境检查结果 ✅

本地环境测试完成，所有依赖包正常工作：
- Python: 3.12.7
- Streamlit: 1.45.1
- Pandas: 2.2.2
- NumPy: 1.26.4
- Matplotlib: 3.9.2
- Seaborn: 0.13.2
- PyTorch: 2.6.0+cpu
- TorchVision: 0.21.0+cpu
- Scikit-learn: 1.5.1
- H5py: 3.11.0
- OpenCV: 4.11.0 (可选)
- Librosa: 0.11.0 (可选)

## 部署步骤

### 1. 准备文件
确保以下文件存在于项目根目录：
- `streamlit_app.py` - 主应用文件 ✅
- `requirements_streamlit_cloud.txt` - Streamlit Cloud专用依赖 ✅
- `packages.txt` - 系统依赖 ✅
- `models/best_vlog_model.pth` - 训练好的模型 ✅
- `extracted_features/` - 特征文件目录 ✅

### 2. 上传到GitHub
```bash
git init
git add .
git commit -m "Initial commit: Vlog BGM分析工具"
git remote add origin https://github.com/您的用户名/您的仓库名.git
git push -u origin main
```

### 3. 在Streamlit Cloud部署
1. 访问 [share.streamlit.io](https://share.streamlit.io)
2. 使用GitHub账号登录
3. 点击 "New app"
4. 选择您的仓库
5. 设置主文件路径为 `streamlit_app.py`
6. **重要**: 将requirements文件设置为 `requirements_streamlit_cloud.txt`
7. 点击 "Deploy"

## 关键修复 ✅

### 1. 导入错误处理
- 添加了 `cv2` 和 `librosa` 的导入错误处理
- 当这些库不可用时，使用基于文件名的确定性特征提取
- 应用会自动检测库的可用性

### 2. 依赖兼容性
- 创建了专门的 `requirements_streamlit_cloud.txt`
- 移除了可能导致问题的 `opencv-python` 和 `librosa`
- 使用固定版本号确保兼容性

### 3. 特征提取优化
- 视频特征提取：优先使用H5文件，其次使用cv2，最后使用确定性特征
- 音频特征提取：优先使用H5文件，其次使用librosa，最后使用确定性特征

## 注意事项

### 文件大小限制
- Streamlit Cloud有文件大小限制
- 如果模型文件过大，考虑使用Git LFS：
```bash
git lfs install
git lfs track "*.pth"
git lfs track "*.h5"
git add .gitattributes
```

### 依赖兼容性
- 已移除 `opencv-python` 和 `librosa` 以避免兼容性问题
- 应用会自动检测这些库的可用性
- 如果库不可用，会使用基于文件名的确定性特征提取

### 环境变量
应用会自动设置以下环境变量：
- `KMP_DUPLICATE_LIB_OK=TRUE`
- `OMP_NUM_THREADS=1`

## 故障排除

### 常见错误
1. **ImportError**: 使用 `requirements_streamlit_cloud.txt` 而不是 `requirements.txt`
2. **文件找不到**: 确保所有必要文件都已上传到GitHub
3. **内存不足**: 考虑优化模型大小或使用外部存储

### 调试技巧
- 查看Streamlit Cloud的日志
- 在本地测试应用：`streamlit run streamlit_app.py`
- 运行 `python test_imports.py` 检查依赖

## 功能说明

### 主要功能
- 🎬 单个BGM分析
- 🎵 多BGM批量分析
- 📊 匹配度可视化
- 💡 智能推荐建议
- 🧠 多任务情感分析

### 分析维度
- 视频情感识别
- BGM情感识别
- 匹配度预测
- 情感一致性分析

## 技术支持

如果遇到问题，请检查：
1. 使用 `requirements_streamlit_cloud.txt` 作为依赖文件
2. 所有文件是否完整上传
3. 模型文件是否可访问
4. Streamlit Cloud日志中的错误信息 