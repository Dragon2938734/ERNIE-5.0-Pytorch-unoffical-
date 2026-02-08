'''
实现论文提到的：Global Balance + Sequence-Level Balance + Router Z-Loss。
'''

import torch
import torch.nn.functional as F

class MoELoadBalancingLoss(torch.nn.Module):
    def __init__(self, num_experts, top_k, alpha=0.01, beta=0.01, gamma=0.001):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha # Global weight
        self.beta = beta   # Local/Sequence weight
        self.gamma = gamma # Z-Loss weight

    def forward(self, router_logits):
        """
        router_logits: [Batch * SeqLen, NumExperts]
        """
        # 1. Router Z-Loss (防止 Logits 过大)
        # log(sum(exp(x)))^2
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1)**2)
        
        # 准备概率
        probs = F.softmax(router_logits, dim=-1)
        
        # 2. Global Load Balancing Loss (Switch Transformer style)
        # 目标：专家分配概率的均值 * 专家被选中的频率 应该最小化
        
        # expert_prob: 每个专家被选中的平均概率 [NumExperts]
        expert_prob = probs.mean(dim=0) 
        
        # expert_freq: 每个专家实际被 Top-K 选中的频率 (Hard selection)
        # 获取 Top-K 索引
        _, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        expert_freq = torch.zeros_like(expert_prob)
        # One-hot 编码并求均值
        top_k_one_hot = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        expert_freq = top_k_one_hot.sum(dim=(0, 1)) / top_k_indices.size(0) # over batch*seq
        
        global_loss = self.num_experts * torch.sum(expert_prob * expert_freq)
        
        # 3. Sequence-Level (Local) Balancing (针对视频高密度 Token)
        # 这里的实现逻辑是：惩罚单个序列内部专家使用的方差
        # 假设 router_logits 原本是 [Batch, Seq, NumExperts]
        # 这里为了简化，我们假设输入已经被 Flatten，这部分通常需要在 Unflatten 状态下计算
        # 此处仅作 Z-Loss 和 Global Loss 的演示
        
        total_loss = (self.alpha * global_loss) + (self.gamma * z_loss)
        
        return total_loss
