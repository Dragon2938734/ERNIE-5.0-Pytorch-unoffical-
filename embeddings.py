'''
这是实现“全模态原生统一”的关键。
UnifiedEmbedding: 根据 modality_id 自动选择对应的 Embedding 表，然后投影到统一的 dim。
RotaryEmbedding3D: 将 Head Dimension 切分为 T, H, W 三个子空间，分别编码时间、高度、宽度信息。
'''

import torch
import torch.nn as nn

class UnifiedEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
      # 不同模态的独立 Tokenizer/Embedding 表
        self.text_embed = nn.Embedding(config.vocab_size_text, config.dim)
        self.image_embed = nn.Embedding(config.vocab_size_image, config.dim)
        self.video_embed = nn.Embedding(config.vocab_size_video, config.dim)
        self.audio_embed = nn.Embedding(config.vocab_size_audio, config.dim)

    def forward(self, input_ids, modality_ids):
        """
        input_ids: [Batch, SeqLen]
        modality_ids: [Batch, SeqLen] (0: Text, 1: Image, 2: Video, 3: Audio)
        """
        # 创建一个全零的 Embedding 容器
        embeddings = torch.zeros(input_ids.shape + (self.dim,), device=input_ids.device)

        # 使用掩码填充不同模态的 Embedding
        # 这种实现方式避免了复杂的控制流，利于并行
        mask_text = (modality_ids == 0)
        mask_image = (modality_ids == 1)
        mask_video = (modality_ids == 2)
        mask_audio = (modality_ids == 3)
        
        if mask_text.any(): embeddings[mask_text] = self.text_embed(input_ids[mask_text])
        if mask_image.any(): embeddings[mask_image] = self.image_embed(input_ids[mask_image])
        if mask_video.any(): embeddings[mask_video] = self.video_embed(input_ids[mask_video])
        if mask_audio.any(): embeddings[mask_audio] = self.audio_embed(input_ids[mask_audio])
        return embeddings

class RotaryEmbedding3D(nn.Module):
    """
    Factorized 3D RoPE: 将 head_dim 分解为 (dim_t, dim_h, dim_w)
    用于同时处理 1D(文本), 2D(图像), 3D(视频)
    """
    def __init__(self, dim, max_pos=4096):
        super().__init__()
      # 简单起见，假设 head_dim 可以被 3 整除，或者我们分配固定比例
        self.dim_t = dim // 3
        self.dim_h = dim // 3
        self.dim_w = dim - self.dim_t - self.dim_h
        self.register_buffer("inv_freq_t", self._get_inv_freq(self.dim_t))
        self.register_buffer("inv_freq_h", self._get_inv_freq(self.dim_h))
        self.register_buffer("inv_freq_w", self._get_inv_freq(self.dim_w))

    def _get_inv_freq(self, dim):
        return 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, x, pos_indices):        
        """
        x: [Batch, SeqLen, NumHeads, HeadDim]
        pos_indices: [Batch, SeqLen, 3] -> (t_idx, h_idx, w_idx)
        """
        t, h, w = pos_indices[..., 0], pos_indices[..., 1], pos_indices[..., 2]
      # 计算不同维度的频率
        freqs_t = self._apply_freqs(t, self.inv_freq_t)
        freqs_h = self._apply_freqs(h, self.inv_freq_h)
        freqs_w = self._apply_freqs(w, self.inv_freq_w)
      # 拼接频率: [Batch, SeqLen, HeadDim]
        freqs = torch.cat([freqs_t, freqs_h, freqs_w], dim=-1)
        return self._rotate_half(x, freqs)

    def _apply_freqs(self, idx, inv_freq):
        # idx: [Batch, SeqLen], inv_freq: [Dim/2]
        freqs = torch.einsum("bi,j->bij", idx.float(), inv_freq)
        return torch.cat((freqs, freqs), dim=-1) # sin/cos pairs

    def _rotate_half(self, x, freqs):
        # 模拟旋转 (简化版)
        return x * freqs.unsqueeze(2).cos() # 仅演示接口
