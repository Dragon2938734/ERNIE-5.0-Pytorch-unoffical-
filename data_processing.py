'''
此文件 data_processing.py 的核心任务是将原始的 文本字符串、图像张量、视频帧 转化为模型可输入的三个核心张量：
1.input_ids: 离散化的 Token 序列（包含 Text IDs 和 Visual Codebook IDs）。
2.modality_ids: 标记每个 Token 属于哪个模态（用于 Embedding 路由）。
3.pos_indices: 3D 位置索引 (T, H, W)，这是实现 Unified RoPE 的关键。
'''


import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Union, Tuple
import numpy as np

# 引用之前的配置
from config import ErnieConfig

# ==========================================
# 1. 模拟分词器 (Mock Tokenizers)
# ==========================================
# 在实际项目中，这里会替换为 SentencePiece 和 VQ-VAE 模型

class MockTextTokenizer:
    """ 模拟文本分词器 (BPE) """
    def __init__(self, vocab_size=100000):
        self.vocab_size = vocab_size
        self.bos_id = 1
        self.eos_id = 2
        
    def encode(self, text: str) -> List[int]:
        # 简单模拟：将字符哈希为 ID
        return [hash(word) % (self.vocab_size - 100) + 3 for word in text.split()]

class MockVisualTokenizer:
    """ 模拟视觉离散化编码器 (VQ-VAE / Magvit-v2) """
    def __init__(self, vocab_size=16384, patch_size=16):
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        
        # 特殊标记
        self.img_start_token = vocab_size - 1
        self.img_end_token = vocab_size - 2
        self.vid_start_token = vocab_size - 3
        self.vid_end_token = vocab_size - 4

    def encode_image(self, image_tensor) -> Tuple[List[int], int, int]:
        """
        输入: image_tensor [C, H, W]
        输出: (token_ids, grid_h, grid_w)
        """
        # 模拟：根据 H/W 计算 Patch 数量
        _, h, w = image_tensor.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size
        num_patches = grid_h * grid_w
        
        # 随机生成 Codebook IDs
        tokens = torch.randint(0, self.vocab_size - 10, (num_patches,)).tolist()
        return tokens, grid_h, grid_w

    def encode_video(self, video_tensor) -> Tuple[List[int], int, int, int]:
        """
        输入: video_tensor [T, C, H, W]
        输出: (token_ids, t, grid_h, grid_w)
        """
        t, _, h, w = video_tensor.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size
        num_patches = t * grid_h * grid_w
        
        tokens = torch.randint(0, self.vocab_size - 10, (num_patches,)).tolist()
        return tokens, t, grid_h, grid_w

# ==========================================
# 2. 统一数据处理器 (Unified Processor)
# ==========================================

class UnifiedDataProcessor:
    def __init__(self, config: ErnieConfig):
        self.config = config
        self.text_tokenizer = MockTextTokenizer(config.vocab_size_text)
        self.visual_tokenizer = MockVisualTokenizer(config.vocab_size_image)

    def process_sample(self, sample_items: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        核心函数：将混合模态的数据列表处理为模型输入。
        sample_items 示例:
        [
            {'type': 'text', 'value': 'A picture of a cat:'},
            {'type': 'image', 'value': torch.randn(3, 256, 256)},
            {'type': 'text', 'value': 'and a video of a dog:'},
            {'type': 'video', 'value': torch.randn(16, 3, 256, 256)}
        ]
        """
        
        all_input_ids = []
        all_modality_ids = []
        all_pos_indices = [] # Stores [t, h, w] triplets
        
        # 全局时间步计数器 (Global Time Step)
        # 文本推进 global_t，图像/视频占据当前的 global_t 但有内部空间结构
        current_t = 0
        
        for item in sample_items:
            dtype = item['type']
            content = item['value']
            
            if dtype == 'text':
                # --- 处理文本 ---
                tokens = self.text_tokenizer.encode(content)
                length = len(tokens)
                
                # 1. Input IDs
                all_input_ids.extend(tokens)
                
                # 2. Modality IDs (Text = 0)
                all_modality_ids.extend([0] * length)
                
                # 3. Position Indices (Text uses 1D time, H=0, W=0)
                # 文本随时间线性增加
                for _ in range(length):
                    all_pos_indices.append([current_t, 0, 0])
                    current_t += 1
                    
            elif dtype == 'image':
                # --- 处理图像 ---
                # image tokens, grid height, grid width
                img_tokens, gh, gw = self.visual_tokenizer.encode_image(content)
                
                # 添加 Start/End Token
                full_tokens = [self.visual_tokenizer.img_start_token] + img_tokens + [self.visual_tokenizer.img_end_token]
                
                # 1. Input IDs
                all_input_ids.extend(full_tokens)
                
                # 2. Modality IDs (Image = 1)
                all_modality_ids.extend([1] * len(full_tokens))
                
                # 3. Position Indices (3D RoPE Logic)
                # 关键：图像的所有 Patch 共享同一个 时间步 (Time Step)
                # 但具有不同的 H 和 W 坐标
                
                # Start Token
                all_pos_indices.append([current_t, 0, 0]) 
                
                # Patches
                for h in range(gh):
                    for w in range(gw):
                        # T 不变，H 和 W 变化
                        all_pos_indices.append([current_t, h + 1, w + 1]) # +1 to avoid 0 overlap
                
                # End Token
                all_pos_indices.append([current_t, gh + 1, gw + 1])
                
                # 图像处理完后，全局时间步 + 1 (表示"看完"这张图了)
                current_t += 1
                
            elif dtype == 'video':
                # --- 处理视频 ---
                vid_tokens, vt, gh, gw = self.visual_tokenizer.encode_video(content)
                
                # Start Token
                all_input_ids.append(self.visual_tokenizer.vid_start_token)
                all_modality_ids.append(2) # Video = 2
                all_pos_indices.append([current_t, 0, 0])
                
                # Video Body
                # 视频有 T 维度。这里的 T 是视频内部的相对时间
                # 我们可以让视频的 T 叠加在 global_t 上
                
                token_idx = 0
                for t in range(vt):
                    for h in range(gh):
                        for w in range(gw):
                            # ID
                            all_input_ids.append(vid_tokens[token_idx])
                            token_idx += 1
                            # Modality
                            all_modality_ids.append(2)
                            # Pos: (Global T + Relative T, H, W)
                            all_pos_indices.append([current_t + t, h + 1, w + 1])
                
                # End Token
                all_input_ids.append(self.visual_tokenizer.vid_end_token)
                all_modality_ids.append(2)
                all_pos_indices.append([current_t + vt, 0, 0])
                
                # 视频结束后，全局时间步推进视频的时长
                current_t += vt + 1

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "modality_ids": torch.tensor(all_modality_ids, dtype=torch.long),
            "pos_indices": torch.tensor(all_pos_indices, dtype=torch.long)
        }

    def collate_fn(self, batch_samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        将多个样本打包成 Batch，处理 Padding
        """
        batch_input_ids = []
        batch_modality_ids = []
        batch_pos_indices = []
        
        max_len = 0
        for sample in batch_samples:
            max_len = max(max_len, len(sample["input_ids"]))
            
        for sample in batch_samples:
            length = len(sample["input_ids"])
            pad_len = max_len - length
            
            # Pad Input IDs (0 usually padding)
            padded_input = F.pad(sample["input_ids"], (0, pad_len), value=0)
            batch_input_ids.append(padded_input)
            
            # Pad Modality IDs (Default to Text=0 or a specific Pad ID)
            padded_mod = F.pad(sample["modality_ids"], (0, pad_len), value=0)
            batch_modality_ids.append(padded_mod)
            
            # Pad Pos Indices (Pad with 0,0,0)
            # pos_indices shape: [L, 3] -> pad dim 0
            padded_pos = F.pad(sample["pos_indices"], (0, 0, 0, pad_len), value=0)
            batch_pos_indices.append(padded_pos)
            
        return {
            "input_ids": torch.stack(batch_input_ids),       # [B, L]
            "modality_ids": torch.stack(batch_modality_ids), # [B, L]
            "pos_indices": torch.stack(batch_pos_indices)    # [B, L, 3]
        }

# ==========================================
# 3. 完整流程演示 (Integration Test)
# ==========================================
if __name__ == "__main__":
    config = ErnieConfig()
    processor = UnifiedDataProcessor(config)
    
    # --- 模拟输入数据 ---
    # 1. 纯文本
    sample1 = [
        {'type': 'text', 'value': 'Hello world'}
    ]
    
    # 2. 图文混合 (Interleaved Image-Text)
    sample2 = [
        {'type': 'text', 'value': 'Look at this cat:'},
        {'type': 'image', 'value': torch.randn(3, 32, 32)}, # 32x32 image -> 2x2 grid (patch=16)
        {'type': 'text', 'value': 'It is very cute.'}
    ]
    
    # 3. 视频描述
    sample3 = [
        {'type': 'text', 'value': 'Video analysis:'},
        {'type': 'video', 'value': torch.randn(4, 3, 32, 32)} # 4 frames, 32x32 -> 4 * 2x2 grid
    ]
    
    # --- 处理单个样本 ---
    processed_s2 = processor.process_sample(sample2)
    print("=== Sample 2 (Text + Image) ===")
    print(f"Input IDs Shape: {processed_s2['input_ids'].shape}")
    print(f"Input IDs: {processed_s2['input_ids']}")
    print(f"Modality IDs: {processed_s2['modality_ids']}") # 应该看到 0 变 1 再变 0
    print(f"Pos Indices (First 10): \n{processed_s2['pos_indices'][:10]}")
    # 观察 Pos Indices:
    # 文本部分: [0,0,0], [1,0,0]...
    # 图片部分: [T, 0, 0] (Start), [T, 1, 1], [T, 1, 2]... (Grid)
    
    # --- Batch 打包 ---
    batch = processor.collate_fn([processor.process_sample(s) for s in [sample1, sample2, sample3]])
    print("\n=== Batch Data ===")
    print(f"Batch Input Shape: {batch['input_ids'].shape}")     # [3, MaxLen]
    print(f"Batch Pos Shape: {batch['pos_indices'].shape}")   # [3, MaxLen, 3]
