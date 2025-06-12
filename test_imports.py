#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有导入是否正常工作
"""

import sys
print(f"Python版本: {sys.version}")

# 测试核心依赖
try:
    import streamlit as st
    print(f"✅ Streamlit: {st.__version__}")
except ImportError as e:
    print(f"❌ Streamlit导入失败: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas导入失败: {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy导入失败: {e}")

try:
    import matplotlib.pyplot as plt
    print(f"✅ Matplotlib: {plt.matplotlib.__version__}")
except ImportError as e:
    print(f"❌ Matplotlib导入失败: {e}")

try:
    import seaborn as sns
    print(f"✅ Seaborn: {sns.__version__}")
except ImportError as e:
    print(f"❌ Seaborn导入失败: {e}")

# 测试机器学习库
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch导入失败: {e}")

try:
    import torchvision
    print(f"✅ TorchVision: {torchvision.__version__}")
except ImportError as e:
    print(f"❌ TorchVision导入失败: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn导入失败: {e}")

# 测试数据存储
try:
    import h5py
    print(f"✅ H5py: {h5py.__version__}")
except ImportError as e:
    print(f"❌ H5py导入失败: {e}")

# 测试可选依赖
try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"⚠️ OpenCV不可用: {e}")

try:
    import librosa
    print(f"✅ Librosa: {librosa.__version__}")
except ImportError as e:
    print(f"⚠️ Librosa不可用: {e}")

# 测试其他必要模块
try:
    from pathlib import Path
    print("✅ Pathlib")
except ImportError as e:
    print(f"❌ Pathlib导入失败: {e}")

try:
    import tempfile
    print("✅ Tempfile")
except ImportError as e:
    print(f"❌ Tempfile导入失败: {e}")

try:
    import base64
    print("✅ Base64")
except ImportError as e:
    print(f"❌ Base64导入失败: {e}")

try:
    from typing import List, Dict
    print("✅ Typing")
except ImportError as e:
    print(f"❌ Typing导入失败: {e}")

try:
    import warnings
    print("✅ Warnings")
except ImportError as e:
    print(f"❌ Warnings导入失败: {e}")

try:
    import math
    print("✅ Math")
except ImportError as e:
    print(f"❌ Math导入失败: {e}")

try:
    import hashlib
    print("✅ Hashlib")
except ImportError as e:
    print(f"❌ Hashlib导入失败: {e}")

print("\n=== 导入测试完成 ===") 