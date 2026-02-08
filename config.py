from dataclasses import dataclass

@dataclass
class ErnieConfig:
    # 基础维度
    dim: int = 512           # 演示用较小维度
    n_layers: int = 4        # 演示用层数
    n_heads: int = 8
    head_dim: int = 64       # dim // n_heads
    
    # 词表大小 (Unified Space)
    vocab_size_text: int = 20000
    vocab_size_image: int = 4096  # VQ-VAE Codebook
    vocab_size_video: int = 4096
    vocab_size_audio: int = 2048
    
    # MoE 配置 (Ultra-Sparse)
    num_experts: int = 16    # 专家总数
    num_shared_experts: int = 2  # 共享专家数量
    top_k: int = 2           # 每次激活专家数
    
    # 序列配置
    max_seq_len: int = 2048
    dropout: float = 0.1
