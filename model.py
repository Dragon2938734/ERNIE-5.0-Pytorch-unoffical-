import torch.nn as nn
from embeddings import UnifiedEmbedding, RotaryEmbedding3D
from moe_layer import UltraSparseMoE

class ErnieBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = nn.MultiheadAttention(config.dim, config.n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(config.dim)
        self.moe = UltraSparseMoE(config)
        self.rope = RotaryEmbedding3D(config.dim // config.n_heads)

    def forward(self, x, pos_indices):
        # Attention (简化版，未完全集成 RoPE 到 Attention 内部计算，仅作接口示意)
        res = x
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = res + attn_out
        
        # MoE
        res = x
        moe_out, router_logits = self.moe(self.ln2(x))
        return res + moe_out, router_logits

class Ernie5Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = UnifiedEmbedding(config)
        self.layers = nn.ModuleList([ErnieBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size_text, bias=False)

    def forward(self, input_ids, modality_ids, pos_indices):
        x = self.embed(input_ids, modality_ids)
        logits_list = []
        for layer in self.layers:
            x, r_logits = layer(x, pos_indices)
            logits_list.append(r_logits)
        return self.lm_head(self.norm(x)), logits_list


# import torch
# import torch.nn as nn
# from config import ErnieConfig
# from embeddings import UnifiedEmbedding, RotaryEmbedding3D
# from moe_layer import UltraSparseMoE

# class ErnieBlock(nn.Module):
#     def __init__(self, config: ErnieConfig):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(config.dim)
#         # 简化的 Attention，重点展示接口
#         self.attn = nn.MultiheadAttention(config.dim, config.n_heads, batch_first=True)
#         self.ln2 = nn.LayerNorm(config.dim)
#         self.moe = UltraSparseMoE(config)
#         self.rope = RotaryEmbedding3D(config.dim // config.n_heads)

#     def forward(self, x, pos_indices):
#         # Attention Part
#         residual = x
#         x_norm = self.ln1(x)
        
#         # 这里省略了复杂的 Attention QKV 拆分和 RoPE 应用过程
#         # 假设 attn_output 已经处理好 RoPE
#         # 实际代码需要将 self.rope 传入 Attention 层内部处理 Q 和 K
#         attn_out, _ = self.attn(x_norm, x_norm, x_norm) 
#         x = residual + attn_out
        
#         # MoE Part
#         residual = x
#         x_norm = self.ln2(x)
#         moe_out, router_logits = self.moe(x_norm)
#         x = residual + moe_out
        
#         return x, router_logits

# class Ernie5Model(nn.Module):
#     def __init__(self, config: ErnieConfig):
#         super().__init__()
#         self.config = config
        
#         # 1. 统一 Embedding
#         self.unified_embed = UnifiedEmbedding(config)
        
#         # 2. 堆叠 MoE Blocks
#         self.layers = nn.ModuleList([
#             ErnieBlock(config) for _ in range(config.n_layers)
#         ])
        
#         # 3. 输出头 (Next-Group-Prediction Head 略, 这里用标准 Head 演示)
#         self.final_norm = nn.LayerNorm(config.dim)
#         self.lm_head = nn.Linear(config.dim, config.vocab_size_text, bias=False)

#     def forward(self, input_ids, modality_ids, pos_indices):
#         """
#         input_ids: [Batch, Seq]
#         modality_ids: [Batch, Seq] 用于区分图/文/视频
#         pos_indices: [Batch, Seq, 3] 用于 3D-RoPE (t, h, w)
#         """
        
#         # Embedding
#         x = self.unified_embed(input_ids, modality_ids)
        
#         all_router_logits = []
        
#         # Layers
#         for layer in self.layers:
#             x, router_logits = layer(x, pos_indices)
#             all_router_logits.append(router_logits)
            
#         x = self.final_norm(x)
#         logits = self.lm_head(x)
        
#         return logits, all_router_logits

# # --- 测试运行 ---
# if __name__ == "__main__":
#     config = ErnieConfig()
#     model = Ernie5Model(config)
    
#     # 模拟数据
#     batch_size = 2
#     seq_len = 128
    
#     # 随机生成 Token ID
#     input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
#     # 随机生成模态 ID (0=文本, 1=图像)
#     modality_ids = torch.randint(0, 2, (batch_size, seq_len))
    
#     # 随机生成 3D 位置索引 (T, H, W)
#     pos_indices = torch.randint(0, 50, (batch_size, seq_len, 3))
    
#     # 前向传播
#     output_logits, router_logits_list = model(input_ids, modality_ids, pos_indices)
    
#     print(f"Model Forward Success!")
#     print(f"Output Shape: {output_logits.shape}")
#     print(f"Num Layers with MoE: {len(router_logits_list)}")
#     print(f"Example Router Logits Shape: {router_logits_list[0].shape}")
    
#     # 计算 Loss
#     from loss import MoELoadBalancingLoss
#     loss_fn = MoELoadBalancingLoss(config.num_experts, config.top_k)
    
#     # 取第一层的 logits 计算 loss 演示
#     aux_loss = loss_fn(router_logits_list[0])
#     print(f"Aux Loss: {aux_loss.item()}")
