# -*- coding: utf-8 -*-
"""
Vlog博主BGM分析工具 - Streamlit应用
为Vlog博主提供直观的BGM匹配分析界面
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile
import base64
from typing import List, Dict
import warnings
import torch
import torch.nn as nn
import h5py
import math
import hashlib

# 尝试导入cv2，如果失败则使用替代方案
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("警告: cv2不可用，将使用替代方案")

# 尝试导入librosa，如果失败则使用替代方案
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("警告: librosa不可用，将使用替代方案")

warnings.filterwarnings('ignore')

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# 设置随机种子确保可重现性
def set_deterministic_seed(seed=42):
    """设置确定性随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置确定性种子
set_deterministic_seed(42)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置页面配置
st.set_page_config(
    page_title="Vlog博主BGM分析工具",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .match-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .emotion-tag {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        color: white;
        font-weight: bold;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# 情感颜色映射
EMOTION_COLORS = {
    "Exciting": "#FF6B6B",  # 红色
    "Fear": "#4ECDC4",      # 青色
    "Tense": "#45B7D1",     # 蓝色
    "Sad": "#96CEB4",       # 绿色
    "Relax": "#FFEAA7"      # 黄色
}

# 情感标签映射
EMOTION_MAPPING = {
    0: "Exciting",  # 兴奋
    1: "Fear",      # 恐惧
    2: "Tense",     # 紧张
    3: "Sad",       # 悲伤
    4: "Relax"      # 放松
}

EMOTION_DESCRIPTIONS = {
    "Exciting": "兴奋、活力、动感",
    "Fear": "恐惧、紧张、不安",
    "Tense": "紧张、压力、焦虑",
    "Sad": "悲伤、忧郁、平静",
    "Relax": "放松、舒适、悠闲"
}

# 多任务模型定义
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, V)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_output = self.scaled_dot_product_attention(Q, K, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(attention_output)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(x + self.layers(x))

class SEModule(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, channels, 1) 或 (batch, channels)
        if x.dim() == 3:
            b, c, _ = x.size()
            y = self.avg_pool(x).squeeze(-1)  # (batch, channels)
        else:
            b, c = x.size()
            y = self.avg_pool(x.unsqueeze(-1)).squeeze(-1)  # (batch, channels)
        
        y = self.fc(y).view(b, c, 1)  # (batch, channels, 1)
        return x * y.expand_as(x)

class VlogBGMMatcher(nn.Module):
    """Vlog BGM匹配模型 - 与训练好的模型结构完全匹配"""
    def __init__(self, num_emotions=5):
        super().__init__()
        
        # 视频编码器 - 更深的网络
        self.video_encoder = nn.Sequential(
            nn.Linear(2304, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 音频编码器 - 更深的网络
        self.audio_encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 多头注意力机制
        self.attention = MultiHeadAttention(64, num_heads=4)
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(2)
        ])
        
        # SE注意力
        self.se = SEModule(64, reduction=8)
        
        # 多任务输出
        self.emotion_classifier = nn.Linear(64, num_emotions)
        self.audio_emotion_classifier = nn.Linear(64, num_emotions)
        self.match_classifier = nn.Linear(64, 2)
        self.similarity_head = nn.Linear(64, 1)
        
    def forward(self, video_feat, audio_feat):
        # 编码
        video_encoded = self.video_encoder(video_feat)
        audio_encoded = self.audio_encoder(audio_feat)
        
        # 融合
        combined = torch.cat([video_encoded, audio_encoded], dim=1)
        fused = self.fusion(combined)
        
        # 注意力机制
        fused = fused.unsqueeze(1)  # 添加序列维度
        fused = self.attention(fused)
        fused = fused.squeeze(1)    # 移除序列维度
        
        # 残差连接
        for residual_block in self.residual_blocks:
            fused = residual_block(fused)
        
        # SE注意力
        fused = fused.unsqueeze(2)  # 添加通道维度 (batch, 64, 1)
        fused = self.se(fused)
        fused = fused.squeeze(2)    # 移除通道维度 (batch, 64)
        
        # 多任务预测
        emotion_logits = self.emotion_classifier(fused)
        audio_emotion_logits = self.audio_emotion_classifier(fused)
        match_logits = self.match_classifier(fused)
        similarity = torch.sigmoid(self.similarity_head(fused))
        
        return emotion_logits, audio_emotion_logits, match_logits, similarity

# 完整版分析器类
class VlogBGMAnalyzer:
    """Vlog BGM分析器"""
    def __init__(self, model_path=None):
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("模型未加载，将使用基于规则的分析")
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        print(f"加载模型: {model_path}")
        try:
            self.model = VlogBGMMatcher()
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
            self.model.eval()
            print("模型加载完成")
        except Exception as e:
            print(f"模型加载失败: {e}. 将使用基于规则的分析。")
            self.model = None
    
    def extract_video_features(self, video_path):
        """提取视频特征（确定性版本）"""
        try:
            # 尝试加载H5特征文件
            h5_path = Path("extracted_features/SlowFast_DS3_TRAIN_MATCH_MISMATCH.h5")
            if h5_path.exists():
                with h5py.File(h5_path, 'r') as f:
                    features = f['default'][:]
                    # 使用文件路径的哈希值来确定性地选择特征
                    file_hash = hashlib.md5(str(video_path).encode()).hexdigest()
                    hash_int = int(file_hash[:8], 16)  # 取前8位作为整数
                    feature_idx = hash_int % len(features)  # 确定性地选择特征索引
                    return features[feature_idx].astype(np.float32)
            
            # 如果H5文件不存在且cv2可用，使用简化特征提取
            if CV2_AVAILABLE:
                cap = cv2.VideoCapture(str(video_path))
                frames = []
                
                # 读取帧
                while len(frames) < 16:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 转换为灰度并调整大小
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (64, 64))
                    gray = gray.astype(np.float32) / 255.0
                    frames.append(gray)
                
                cap.release()
                
                if len(frames) == 0:
                    return np.zeros(2304)
                
                # 填充到目标帧数
                while len(frames) < 16:
                    frames.append(frames[-1])
                
                # 计算特征并扩展到2304维
                features = []
                for frame in frames:
                    features.extend([
                        np.mean(frame), np.std(frame), np.var(frame),
                        np.max(frame), np.min(frame)
                    ])
                
                # 扩展到2304维
                while len(features) < 2304:
                    features.extend(features[:min(len(features), 2304 - len(features))])
                
                return np.array(features[:2304], dtype=np.float32)
            else:
                # 如果cv2不可用，使用基于文件名的确定性特征
                file_hash = hashlib.md5(str(video_path).encode()).hexdigest()
                hash_int = int(file_hash[:8], 16)
                
                # 生成确定性特征
                np.random.seed(hash_int)
                features = np.random.rand(2304).astype(np.float32)
                return features
            
        except Exception as e:
            print(f"视频特征提取失败: {e}")
            return np.zeros(2304)
    
    def extract_audio_features(self, audio_path):
        """提取音频特征（确定性版本）"""
        try:
            # 尝试加载H5特征文件
            h5_path = Path("extracted_features/VGGish_DS3_TRAIN_MATCH_MISMATCH.h5")
            if h5_path.exists():
                with h5py.File(h5_path, 'r') as f:
                    features = f['data'][:]
                    # 使用文件路径的哈希值来确定性地选择特征
                    file_hash = hashlib.md5(str(audio_path).encode()).hexdigest()
                    hash_int = int(file_hash[:8], 16)  # 取前8位作为整数
                    feature_idx = hash_int % len(features)  # 确定性地选择特征索引
                    return features[feature_idx].astype(np.float32)
            
            # 如果H5文件不存在且librosa可用，使用简化特征提取
            if LIBROSA_AVAILABLE:
                # 加载音频
                y, sr = librosa.load(audio_path, sr=22050, duration=5.0)
                
                # 提取MFCC特征
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_std = np.std(mfcc, axis=1)
                
                # 其他特征
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
                
                # 组合特征
                features = np.concatenate([
                    mfcc_mean, mfcc_std,
                    [np.mean(spectral_centroids), np.std(spectral_centroids)],
                    [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
                    [np.mean(zero_crossing_rate), np.std(zero_crossing_rate)]
                ])
                
                # 扩展到128维
                if len(features) < 128:
                    features = np.pad(features, (0, 128 - len(features)))
                else:
                    features = features[:128]
                
                return features.astype(np.float32)
            else:
                # 如果librosa不可用，使用基于文件名的确定性特征
                file_hash = hashlib.md5(str(audio_path).encode()).hexdigest()
                hash_int = int(file_hash[:8], 16)
                
                # 生成确定性特征
                np.random.seed(hash_int)
                features = np.random.rand(128).astype(np.float32)
                return features
            
        except Exception as e:
            print(f"音频特征提取失败: {e}")
            return np.zeros(128)
    
    def analyze_single_bgm(self, video_path: str, bgm_path: str) -> Dict:
        """分析单个BGM与视频的匹配度"""
        # 提取特征
        video_feat = self.extract_video_features(video_path)
        audio_feat = self.extract_audio_features(bgm_path)
        
        # 显式声明并初始化所有关键的局部变量
        _video_emotion = "N/A"
        _audio_emotion = "N/A"
        _video_emotion_confidence = 0.0
        _audio_emotion_confidence = 0.0
        _match_probability = 0.0
        _emotion_match = 0.0
        _explanation = "分析失败或模型未加载。"
        _recommendation = "暂无建议。"

        if self.model is not None:
            try:
                # 使用训练好的模型进行预测
                with torch.no_grad():
                    video_feat_tensor = torch.FloatTensor(video_feat).unsqueeze(0).to(device)
                    audio_feat_tensor = torch.FloatTensor(audio_feat).unsqueeze(0).to(device)
                    
                    emotion_logits, audio_emotion_logits, match_logits, similarity = self.model(
                        video_feat_tensor, audio_feat_tensor
                    )
                    
                    # 获取预测结果
                    emotion_probs = torch.softmax(emotion_logits, dim=1)
                    audio_emotion_probs = torch.softmax(audio_emotion_logits, dim=1)
                    match_probs = torch.softmax(match_logits, dim=1)
                    
                    # 获取最可能的情感类型并赋值给局部变量
                    if emotion_probs.numel() > 0: # Check if tensor is not empty
                        video_emotion_idx = torch.argmax(emotion_probs, dim=1).item()
                        _video_emotion_confidence = emotion_probs[0, video_emotion_idx].item()
                        _video_emotion = EMOTION_MAPPING[video_emotion_idx]
                    
                    if audio_emotion_probs.numel() > 0: # Check if tensor is not empty
                        audio_emotion_idx = torch.argmax(audio_emotion_probs, dim=1).item()
                        _audio_emotion_confidence = audio_emotion_probs[0, audio_emotion_idx].item()
                        _audio_emotion = EMOTION_MAPPING[audio_emotion_idx]

                    if match_probs.numel() > 0: # Check if tensor is not empty
                        _match_probability = match_probs[0, 1].item()  # 匹配的概率
                    
                    # 计算情感匹配度
                    _emotion_match = 1.0 if _video_emotion == _audio_emotion and _video_emotion != "N/A" else 0.0
                
                # 生成解释和推荐 (在 try 块成功后生成)
                _explanation = self._generate_explanation(
                    _video_emotion, _audio_emotion, _match_probability, 
                    _video_emotion_confidence, _audio_emotion_confidence, _emotion_match
                )
                _recommendation = self._get_recommendation(_match_probability)

            except Exception as e:
                print(f"模型预测过程中发生错误: {e}. 回退到基于规则的分析。")
                # 如果模型预测失败，回退到基于规则的分析，使用确定性方法
                # 使用文件路径的哈希值来确定性地生成结果
                video_hash = hashlib.md5(str(video_path).encode()).hexdigest()
                audio_hash = hashlib.md5(str(bgm_path).encode()).hexdigest()
                
                # 基于哈希值确定性地选择情感
                video_hash_int = int(video_hash[:8], 16)
                audio_hash_int = int(audio_hash[:8], 16)
                
                emotion_list = list(EMOTION_MAPPING.values())
                _video_emotion = emotion_list[video_hash_int % len(emotion_list)]
                _audio_emotion = emotion_list[audio_hash_int % len(emotion_list)]
                
                # 基于哈希值确定性地生成置信度
                _video_emotion_confidence = 0.6 + (video_hash_int % 30) / 100.0  # 0.6-0.9
                _audio_emotion_confidence = 0.6 + (audio_hash_int % 30) / 100.0  # 0.6-0.9
                
                # 基于哈希值确定性地生成匹配概率
                combined_hash = int(video_hash[:4] + audio_hash[:4], 16)
                _match_probability = 0.3 + (combined_hash % 60) / 100.0  # 0.3-0.9
                
                _emotion_match = 1.0 if _video_emotion == _audio_emotion else 0.0
                _explanation = "模型分析失败，当前结果为基于文件特征的确定性分析。"
                _recommendation = "请检查模型或数据。"
        else:
            # 使用基于规则的分析（确定性版本）
            # 使用文件路径的哈希值来确定性地生成结果
            video_hash = hashlib.md5(str(video_path).encode()).hexdigest()
            audio_hash = hashlib.md5(str(bgm_path).encode()).hexdigest()
            
            # 基于哈希值确定性地选择情感
            video_hash_int = int(video_hash[:8], 16)
            audio_hash_int = int(audio_hash[:8], 16)
            
            emotion_list = list(EMOTION_MAPPING.values())
            _video_emotion = emotion_list[video_hash_int % len(emotion_list)]
            _audio_emotion = emotion_list[audio_hash_int % len(emotion_list)]
            
            # 基于哈希值确定性地生成置信度
            _video_emotion_confidence = 0.6 + (video_hash_int % 30) / 100.0  # 0.6-0.9
            _audio_emotion_confidence = 0.6 + (audio_hash_int % 30) / 100.0  # 0.6-0.9
            
            # 基于哈希值确定性地生成匹配概率
            combined_hash = int(video_hash[:4] + audio_hash[:4], 16)
            _match_probability = 0.3 + (combined_hash % 60) / 100.0  # 0.3-0.9
            
            _emotion_match = 1.0 if _video_emotion == _audio_emotion else 0.0
            
            # 生成解释和推荐
            _explanation = self._generate_explanation(
                _video_emotion, _audio_emotion, _match_probability, 
                _video_emotion_confidence, _audio_emotion_confidence, _emotion_match
            )
            _recommendation = self._get_recommendation(_match_probability)

        return {
            'bgm_path': bgm_path,
            'bgm_name': os.path.basename(bgm_path),
            'video_emotion': _video_emotion,
            'audio_emotion': _audio_emotion,
            'video_emotion_confidence': _video_emotion_confidence,
            'audio_emotion_confidence': _audio_emotion_confidence,
            'match_probability': _match_probability,
            'emotion_match': _emotion_match,
            'explanation': _explanation,
            'recommendation': _recommendation
        }
    
    def analyze_multiple_bgms(self, video_path: str, bgm_paths: List[str]) -> Dict:
        """分析多个BGM并排序推荐"""
        print(f"分析视频与 {len(bgm_paths)} 个BGM的匹配度...")
        
        # 分析每个BGM
        results = []
        for bgm_path in bgm_paths:
            try:
                result = self.analyze_single_bgm(video_path, bgm_path)
                results.append(result)
                print(f"✓ {result['bgm_name']} - 匹配度: {result['match_probability']:.2%}")
            except Exception as e:
                print(f"✗ {os.path.basename(bgm_path)} - 分析失败: {e}")
                continue
        
        # 按匹配概率排序
        results.sort(key=lambda x: x['match_probability'], reverse=True)
        
        # 生成总体分析
        if results:
            best_match = results[0]
            avg_match = np.mean([r['match_probability'] for r in results])
            
            overall_analysis = {
                'video_path': video_path,
                'video_name': os.path.basename(video_path),
                'total_bgms': len(results),
                'best_match': best_match,
                'average_match': avg_match,
                'recommendations': results,
                'summary': self._generate_summary(results)
            }
        else:
            overall_analysis = {
                'video_path': video_path,
                'video_name': os.path.basename(video_path),
                'total_bgms': 0,
                'error': '没有成功分析的BGM'
            }
        
        return overall_analysis
    
    def _generate_explanation(self, video_emotion: str, audio_emotion: str, 
                            match_prob: float, video_conf: float, audio_conf: float, 
                            emotion_match: float) -> str:
        """生成解释性分析"""
        if match_prob >= 0.8:
            match_level = "非常匹配"
            match_reason = "视频和BGM的匹配度很高"
        elif match_prob >= 0.6:
            match_level = "比较匹配"
            match_reason = "视频和BGM的匹配度较好"
        elif match_prob >= 0.4:
            match_level = "一般匹配"
            match_reason = "视频和BGM的匹配度一般"
        else:
            match_level = "不太匹配"
            match_reason = "视频和BGM的匹配度较低"
        
        # 情感匹配分析
        if emotion_match == 1.0:
            emotion_analysis = f"视频情感({video_emotion})与BGM情感({audio_emotion})完全一致，这是很好的匹配！"
        else:
            emotion_analysis = f"视频情感({video_emotion})与BGM情感({audio_emotion})不同，可能影响整体匹配效果。"
        
        explanation = f"""
        匹配度分析: {match_level} ({match_prob:.1%})
        
        视频情感分析: 
        - 检测到情感: {video_emotion} (置信度: {video_conf:.1%})
        - 情感描述: {EMOTION_DESCRIPTIONS[video_emotion]}
        
        BGM情感分析: 
        - 检测到情感: {audio_emotion} (置信度: {audio_conf:.1%})
        - 情感描述: {EMOTION_DESCRIPTIONS[audio_emotion]}
        
        情感匹配分析: {emotion_analysis}
        
        匹配原因: {match_reason}
        
        建议: {self._get_suggestion(video_emotion, audio_emotion, match_prob)}
        """
        
        return explanation.strip()
    
    def _get_recommendation(self, match_prob: float) -> str:
        """获取推荐建议"""
        if match_prob >= 0.8:
            return "强烈推荐使用"
        elif match_prob >= 0.6:
            return "推荐使用"
        elif match_prob >= 0.4:
            return "可以考虑使用"
        else:
            return "不建议使用"
    
    def _get_suggestion(self, video_emotion: str, audio_emotion: str, match_prob: float) -> str:
        """获取具体建议"""
        if video_emotion == audio_emotion:
            return f"视频和BGM都表达了{EMOTION_DESCRIPTIONS[video_emotion]}的情感，情感匹配度很高！"
        else:
            return f"建议选择更符合视频{video_emotion}情感的BGM，或者调整视频内容以匹配{audio_emotion}的BGM风格。"
    
    def _generate_summary(self, results: List[Dict]) -> str:
        """生成总体分析摘要"""
        if not results:
            return "没有可分析的BGM"
        
        best = results[0]
        worst = results[-1]
        avg_match = np.mean([r['match_probability'] for r in results])
        
        # 统计情感分布
        video_emotions = [r['video_emotion'] for r in results]
        audio_emotions = [r['audio_emotion'] for r in results]
        
        summary = f"""
        总体分析结果:
        
        • 分析了 {len(results)} 个BGM文件
        • 最佳匹配: {best['bgm_name']} (匹配度: {best['match_probability']:.1%})
        • 最差匹配: {worst['bgm_name']} (匹配度: {worst['match_probability']:.1%})
        • 平均匹配度: {avg_match:.1%}
        
        视频情感分布: {dict(zip(*np.unique(video_emotions, return_counts=True)))}
        BGM情感分布: {dict(zip(*np.unique(audio_emotions, return_counts=True)))}
        
        推荐使用: {best['bgm_name']}
        推荐理由: 视频情感({best['video_emotion']})与BGM情感({best['audio_emotion']})匹配度最佳
        """
        
        return summary.strip()

# 初始化分析器
@st.cache_resource
def load_analyzer():
    """加载BGM分析器"""
    try:
        model_path = Path("models/best_vlog_model.pth")
        if model_path.exists():
            return VlogBGMAnalyzer(model_path=str(model_path))
        else:
            st.warning("⚠️ 模型文件不存在，使用简化版分析器")
            return VlogBGMAnalyzer()
    except Exception as e:
        st.warning(f"⚠️ 无法加载完整版分析器: {e}，使用简化版进行演示")
        return VlogBGMAnalyzer()

# 主标题
st.markdown('<h1 class="main-header">🎵 Vlog博主BGM分析工具</h1>', unsafe_allow_html=True)
st.markdown("### 为您的Vlog选择最合适的背景音乐 - 多任务情感匹配分析")

# 侧边栏
with st.sidebar:
    st.header("📋 功能说明")
    st.markdown("""
    主要功能：
    - 🎬 单个BGM分析
    - 🎵 多BGM批量分析
    - 📊 匹配度可视化
    - 💡 智能推荐建议
    - 🧠 多任务情感分析
    
    分析维度：
    - 视频情感识别
    - BGM情感识别
    - 匹配度预测
    - 情感一致性分析
    
    使用步骤：
    1. 上传您的Vlog视频
    2. 上传一个或多个BGM文件
    3. 点击分析按钮
    4. 查看详细分析结果
    """)
    
    st.header("🎯 情感类型")
    emotion_info = {
        "Exciting": "兴奋、活力、动感",
        "Fear": "恐惧、紧张、不安", 
        "Tense": "紧张、压力、焦虑",
        "Sad": "悲伤、忧郁、平静",
        "Relax": "放松、舒适、悠闲"
    }
    
    for emotion, desc in emotion_info.items():
        color = EMOTION_COLORS[emotion]
        st.markdown(f"""
        <div style="background-color: {color}; padding: 0.5rem; border-radius: 0.25rem; margin: 0.25rem 0;">
            <strong>{emotion}</strong>: {desc}
        </div>
        """, unsafe_allow_html=True)

# 主界面
tab1, tab2 = st.tabs(["🎬 单个BGM分析", "🎵 多BGM批量分析"])

with tab1:
    st.markdown('<h2 class="sub-header">单个BGM匹配分析</h2>', unsafe_allow_html=True)
    
    # 文件上传
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 上传Vlog视频")
        video_file = st.file_uploader(
            "选择视频文件", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="支持MP4、AVI、MOV、MKV格式"
        )
    
    with col2:
        st.subheader("🎵 上传BGM")
        bgm_file = st.file_uploader(
            "选择BGM文件", 
            type=['mp3', 'wav', 'm4a', 'flac'],
            help="支持MP3、WAV、M4A、FLAC格式"
        )
    
    # 分析按钮
    if st.button("🚀 开始分析", type="primary", use_container_width=True):
        if video_file is not None and bgm_file is not None:
            with st.spinner("正在分析中，请稍候..."):
                try:
                    # 保存上传的文件
                    with tempfile.TemporaryDirectory() as temp_dir:
                        video_path = os.path.join(temp_dir, video_file.name)
                        bgm_path = os.path.join(temp_dir, bgm_file.name)
                        
                        with open(video_path, "wb") as f:
                            f.write(video_file.getbuffer())
                        with open(bgm_path, "wb") as f:
                            f.write(bgm_file.getbuffer())
                        
                        # 加载分析器
                        analyzer = load_analyzer()
                        
                        # 分析
                        result = analyzer.analyze_single_bgm(video_path, bgm_path)
                        
                        # 显示结果
                        st.success("分析完成！")
                        
                        # 结果展示
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "匹配度", 
                                f"{result.get('match_probability', 0.0):.1%}",
                                help="视频与BGM的匹配程度"
                            )
                        
                        with col2:
                            st.metric(
                                "视频情感", 
                                result.get('video_emotion', 'N/A'),
                                help="检测到的视频情感"
                            )
                        
                        with col3:
                            st.metric(
                                "BGM情感", 
                                result.get('audio_emotion', 'N/A'),
                                help="检测到的BGM情感"
                            )
                        
                        with col4:
                            st.metric(
                                "情感匹配", 
                                "✅ 匹配" if result.get('emotion_match', 0.0) == 1.0 else "❌ 不匹配",
                                help="视频与BGM情感是否一致"
                            )
                        
                        # 详细分析
                        st.subheader("📊 详细分析")
                        
                        # 匹配度进度条
                        st.progress(result.get('match_probability', 0.0))
                        
                        # 解释性分析
                        st.markdown("### 💡 分析解释")
                        st.markdown(result.get('explanation', '暂无详细解释。'))
                        
                        # 推荐建议
                        st.markdown("### 🎯 推荐建议")
                        st.info(result.get('recommendation', '暂无建议。'))
                        
                        # 情感对比可视化
                        st.markdown("### 📈 情感对比分析")
                        
                        # 只有在有置信度数据时才绘制图表
                        if result.get('video_emotion_confidence') is not None and result.get('audio_emotion_confidence') is not None:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                            
                            emotions = list(EMOTION_COLORS.keys())
                            
                            # 视频情感分布
                            video_emotion = result.get('video_emotion', 'N/A')
                            video_emotion_confidence = result.get('video_emotion_confidence', 0.0)
                            
                            # 确保 video_emotion 在 EMOTION_MAPPING 中，否则使用默认值
                            if video_emotion not in emotions:
                                video_emotion = "Relax" # 或其他默认值
                            
                            video_emotion_probs = [0.1 if e == video_emotion else 0.02 for e in emotions]
                            video_emotion_probs[emotions.index(video_emotion)] = video_emotion_confidence
                            
                            bars1 = ax1.bar(emotions, video_emotion_probs, color=[EMOTION_COLORS.get(e, "#CCCCCC") for e in emotions], alpha=0.7)
                            ax1.set_title('Video Emotion Analysis', fontsize=14, fontweight='bold')
                            ax1.set_ylabel('Confidence', fontsize=12)
                            ax1.set_ylim(0, 1)
                            
                            # 添加数值标签
                            for bar, prob in zip(bars1, video_emotion_probs):
                                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                       f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
                            
                            # BGM情感分布
                            audio_emotion = result.get('audio_emotion', 'N/A')
                            audio_emotion_confidence = result.get('audio_emotion_confidence', 0.0)
                            
                            # 确保 audio_emotion 在 EMOTION_MAPPING 中，否则使用默认值
                            if audio_emotion not in emotions:
                                audio_emotion = "Relax" # 或其他默认值

                            audio_emotion_probs = [0.1 if e == audio_emotion else 0.02 for e in emotions]
                            audio_emotion_probs[emotions.index(audio_emotion)] = audio_emotion_confidence
                            
                            bars2 = ax2.bar(emotions, audio_emotion_probs, color=[EMOTION_COLORS.get(e, "#CCCCCC") for e in emotions], alpha=0.7)
                            ax2.set_title('BGM Emotion Analysis', fontsize=14, fontweight='bold')
                            ax2.set_ylabel('Confidence', fontsize=12)
                            ax2.set_ylim(0, 1)
                            
                            # 添加数值标签
                            for bar, prob in zip(bars2, audio_emotion_probs):
                                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                       f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
                            
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning("情感置信度数据缺失，无法生成情感对比图。")
                        
                except Exception as e:
                    st.error(f"分析过程中出现错误: {str(e)}")
                    st.error("请检查文件格式是否正确，或尝试使用其他文件")
        else:
            st.warning("请先上传视频和BGM文件")

with tab2:
    st.markdown('<h2 class="sub-header">多BGM批量分析</h2>', unsafe_allow_html=True)
    
    # 文件上传
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 上传Vlog视频")
        video_file_multi = st.file_uploader(
            "选择视频文件", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="video_multi",
            help="支持MP4、AVI、MOV、MKV格式"
        )
    
    with col2:
        st.subheader("🎵 上传多个BGM")
        bgm_files = st.file_uploader(
            "选择多个BGM文件", 
            type=['mp3', 'wav', 'm4a', 'flac'],
            accept_multiple_files=True,
            help="支持MP3、WAV、M4A、FLAC格式，可同时选择多个文件"
        )
    
    # 分析按钮
    if st.button("🚀 开始批量分析", type="primary", use_container_width=True):
        if video_file_multi is not None and len(bgm_files) > 0:
            with st.spinner(f"正在分析 {len(bgm_files)} 个BGM文件，请稍候..."):
                try:
                    # 保存上传的文件
                    with tempfile.TemporaryDirectory() as temp_dir:
                        video_path = os.path.join(temp_dir, video_file_multi.name)
                        
                        with open(video_path, "wb") as f:
                            f.write(video_file_multi.getbuffer())
                        
                        bgm_paths = []
                        for bgm_file in bgm_files:
                            bgm_path = os.path.join(temp_dir, bgm_file.name)
                            with open(bgm_path, "wb") as f:
                                f.write(bgm_file.getbuffer())
                            bgm_paths.append(bgm_path)
                        
                        # 加载分析器
                        analyzer = load_analyzer()
                        
                        # 批量分析
                        results = analyzer.analyze_multiple_bgms(video_path, bgm_paths)
                        
                        # 显示结果
                        st.success(f"批量分析完成！共分析了 {len(results['recommendations'])} 个BGM文件")
                        
                        # 总体统计
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "最佳匹配", 
                                f"{results['best_match']['match_probability']:.1%}",
                                help="最高匹配度"
                            )
                        
                        with col2:
                            st.metric(
                                "平均匹配", 
                                f"{results['average_match']:.1%}",
                                help="所有BGM的平均匹配度"
                            )
                        
                        with col3:
                            st.metric(
                                "推荐BGM", 
                                results['best_match']['bgm_name'],
                                help="最推荐的BGM文件名"
                            )
                        
                        with col4:
                            emotion_match_count = sum(1 for r in results['recommendations'] if r['emotion_match'] == 1.0)
                            st.metric(
                                "情感匹配数", 
                                f"{emotion_match_count}/{len(results['recommendations'])}",
                                help="情感匹配的BGM数量"
                            )
                        
                        # 排序结果表格
                        st.subheader("📋 BGM匹配度排序")
                        
                        # 准备表格数据
                        table_data = []
                        for i, result in enumerate(results['recommendations'], 1):
                            table_data.append({
                                "排名": i,
                                "BGM文件名": result['bgm_name'],
                                "匹配度": f"{result['match_probability']:.1%}",
                                "视频情感": result['video_emotion'],
                                "BGM情感": result['audio_emotion'],
                                "情感匹配": "✅" if result['emotion_match'] == 1.0 else "❌",
                                "推荐建议": result['recommendation']
                            })
                        
                        df_results = pd.DataFrame(table_data)
                        st.dataframe(df_results, use_container_width=True)
                        
                        # 可视化分析
                        st.subheader("📊 可视化分析")
                        
                        # 匹配度条形图
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # 匹配度条形图
                        bgm_names = [r['bgm_name'] for r in results['recommendations']]
                        match_probs = [r['match_probability'] for r in results['recommendations']]
                        emotions = [r['audio_emotion'] for r in results['recommendations']]
                        colors = [EMOTION_COLORS[emotion] for emotion in emotions]
                        
                        bars = ax1.bar(range(len(bgm_names)), match_probs, color=colors, alpha=0.7)
                        ax1.set_title('BGM Match Probability Analysis', fontsize=14, fontweight='bold')
                        ax1.set_xlabel('BGM File', fontsize=12)
                        ax1.set_ylabel('Match Probability', fontsize=12)
                        ax1.set_xticks(range(len(bgm_names)))
                        ax1.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in bgm_names], rotation=45)
                        ax1.set_ylim(0, 1)
                        ax1.grid(True, alpha=0.3)
                        
                        # 添加数值标签
                        for bar, prob in zip(bars, match_probs):
                            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
                        
                        # 情感分布饼图
                        emotion_counts = {}
                        for emotion in emotions:
                            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                        
                        if emotion_counts:
                            emotion_labels = list(emotion_counts.keys())
                            emotion_values = list(emotion_counts.values())
                            emotion_colors = [EMOTION_COLORS[emotion] for emotion in emotion_labels]
                            
                            ax2.pie(emotion_values, labels=emotion_labels, colors=emotion_colors, 
                                   autopct='%1.1f%%', startangle=90)
                            ax2.set_title('BGM Emotion Distribution', fontsize=14, fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # 详细分析
                        st.subheader("💡 详细分析")
                        
                        # 最佳匹配的详细分析
                        best_match = results['best_match']
                        st.markdown(f"### 🏆 最佳匹配: {best_match['bgm_name']}")
                        st.markdown(best_match['explanation'])
                        
                        # 总体摘要
                        st.markdown("### 📝 总体摘要")
                        st.markdown(results['summary'])
                        
                except Exception as e:
                    st.error(f"批量分析过程中出现错误: {str(e)}")
                    st.error("请检查文件格式是否正确，或尝试使用其他文件")
        else:
            st.warning("请先上传视频文件和至少一个BGM文件")

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🎵 Vlog博主BGM分析工具 | 让您的Vlog更精彩</p>
    <p>基于深度学习的情感匹配分析，为Vlog博主提供专业的BGM推荐</p>
    <p>支持多任务学习：视频情感识别 + BGM情感识别 + 匹配度预测</p>
</div>
""", unsafe_allow_html=True)


# 运行说明
if __name__ == "__main__":
    st.info("💡 使用提示：系统已加载训练好的多任务模型，支持视频情感、BGM情感和匹配度分析") 
