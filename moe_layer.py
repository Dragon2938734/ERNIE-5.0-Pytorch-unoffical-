'''
实现 Top-K 路由、共享专家和路由专家。重点是返回 router_logits 以便后续计算 Aux Loss。
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
      # SwiGLU 结构
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class UltraSparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
      # 1. 路由专家 (Routed Experts)
        self.experts = nn.ModuleList([Expert(config.dim, config.dim * 4) for _ in range(self.num_experts)])
      # 2. 共享专家 (Shared Experts - Always Active)
        self.shared_experts = nn.ModuleList([Expert(config.dim, config.dim * 4) for _ in range(config.num_shared_experts)])
      # 3. Router
        self.router = nn.Linear(config.dim, self.num_experts, bias=False)

    def forward(self, x):
      # x: [Batch, SeqLen, Dim]
        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim)
        
        # 共享专家
        shared_out = sum(e(x_flat) for e in self.shared_experts)
        
        # 路由专家
        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=1)
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        routed_out = torch.zeros_like(x_flat)
      # 简单的循环执行 (实际应用会用 Optimized Kernels, e.g., Triton)
        for k in range(self.top_k):
            idx = top_k_indices[:, k]
            w = top_k_weights[:, k].unsqueeze(1)
            for i in range(self.num_experts):
                mask = (idx == i)
                if mask.any():
                    routed_out[mask] += self.experts[i](x_flat[mask]) * w[mask]
        # 返回 output 和 router_logits (用于计算负载均衡 Loss)
        return (shared_out + routed_out).view(batch, seq, dim), logits
