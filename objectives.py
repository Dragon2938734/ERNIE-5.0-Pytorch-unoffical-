'''
这个文件实现了论文中提到的核心训练目标。它能够根据 modality_ids 自动区分文本、图像和视频 Token，并分别计算损失，最后加权融合。
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedObjectiveLoss(nn.Module):
    """
    ERNIE 5.0 全模态统一训练目标函数
    实现逻辑：
    1. 文本 (Text): 标准 Next-Token Prediction (NTP)
    2. 视觉 (Image/Video): Next-Group-of-Tokens Prediction (NGTP)
       - 为了代码演示，这里模拟 NGTP 的逻辑：
       - 假设视觉 Token 的预测权重不同，或者需要同时预测未来 K 个 Token (Multi-token prediction)。
       - 在简化实现中，我们重点展示基于模态掩码 (Modality Mask) 的加权损失。
    """
    def __init__(self, config, lambda_visual=1.0, lambda_audio=1.0):
        super().__init__()
        # 不同的模态可能对应不同的词表范围，或者统一在一个大词表中
        # 这里假设所有 Token ID 都在同一个 vocab_size 空间内
        self.vocab_size = config.vocab_size_text  # 假设统一空间大小
        
        # 模态损失权重 (超参数)
        self.lambda_visual = lambda_visual
        self.lambda_audio = lambda_audio
        
        # 基础损失函数 (忽略 padding index)
        self.loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    def forward(self, logits, labels, modality_ids):
        """
        logits: [Batch, SeqLen, VocabSize] - 模型输出
        labels: [Batch, SeqLen] - 真实标签 (通常是 input_ids 向左移一位)
        modality_ids: [Batch, SeqLen] - 模态指示 (0:Text, 1:Image, 2:Video, 3:Audio)
        """
        
        # 1. 展平张量以便计算 CrossEntropy
        # logits_flat: [Batch*SeqLen, VocabSize]
        # labels_flat: [Batch*SeqLen]
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        modality_flat = modality_ids.view(-1)
        
        # 2. 计算逐个 Token 的原始损失 (无 reduce)
        # per_token_loss: [Batch*SeqLen]
        per_token_loss = self.loss_fct(logits_flat, labels_flat)
        
        # 3. 创建模态掩码 (Masks)
        mask_text = (modality_flat == 0)
        mask_image = (modality_flat == 1)
        mask_video = (modality_flat == 2)
        mask_audio = (modality_flat == 3)
        
        # 4. 分别计算各模态的平均损失
        # 注意：需要处理分母为 0 的情况 (防止 NaN)
        
        # --- 文本损失 (L_text) ---
        if mask_text.any():
            loss_text = per_token_loss[mask_text].mean()
        else:
            loss_text = torch.tensor(0.0, device=logits.device)
            
        # --- 视觉损失 (L_visual = Image + Video) ---
        # 论文提到的 Next-Group 预测在此处体现为：
        # 视觉 Token 往往成组出现，我们在计算 Loss 时可以给予更高的组权重，
        # 或者在模型输出层由多个 Head 共同预测 (代码此处为简化版，仅调节权重)
        visual_loss_accum = torch.tensor(0.0, device=logits.device)
        visual_count = 0
        
        if mask_image.any():
            visual_loss_accum += per_token_loss[mask_image].sum()
            visual_count += mask_image.sum()
            
        if mask_video.any():
            visual_loss_accum += per_token_loss[mask_video].sum()
            visual_count += mask_video.sum()
        
        if visual_count > 0:
            loss_visual = visual_loss_accum / visual_count
        else:
            loss_visual = torch.tensor(0.0, device=logits.device)

        # --- 音频损失 (L_audio) ---
        if mask_audio.any():
            loss_audio = per_token_loss[mask_audio].mean()
        else:
            loss_audio = torch.tensor(0.0, device=logits.device)
            
        # 5. 加权融合总损失
        # L_total = L_text + λ_v * L_visual + λ_a * L_audio
        total_loss = loss_text + (self.lambda_visual * loss_visual) + (self.lambda_audio * loss_audio)
        
        return total_loss, {
            "loss_text": loss_text.item(),
            "loss_visual": loss_visual.item(),
            "loss_audio": loss_audio.item()
        }

# --- 模拟 Next-Group-Prediction 的高级实现 (可选) ---
class NextGroupPredictionHead(nn.Module):
    """
    如果模型真的实现了 Next-Group 预测 (一次预测未来 G 个 token)，
    那么输出头结构会发生变化。这里提供一个概念实现。
    """
    def __init__(self, dim, vocab_size, group_size=4):
        super().__init__()
        self.group_size = group_size
        # 输出头变为预测 Group Size 个 Token
        self.heads = nn.ModuleList([
            nn.Linear(dim, vocab_size, bias=False) for _ in range(group_size)
        ])
        
    def forward(self, x):
        # x: [Batch, Seq, Dim]
        # output: [Batch, Seq, GroupSize, Vocab]
        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        return torch.stack(outputs, dim=2)
