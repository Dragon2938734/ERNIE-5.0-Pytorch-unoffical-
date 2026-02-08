'''
这是把所有零件组装起来的“胶水代码”
'''

import torch
import torch.optim as optim

# 导入所有组件
from config import ErnieConfig
from model import Ernie5Model
from objectives import UnifiedObjectiveLoss
from loss import MoELoadBalancingLoss
from data_processing import UnifiedDataProcessor

def main():
    # ==========================================
    # 1. 初始化配置与设备
    # ==========================================
    print(">>> Initializing Configuration...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    config = ErnieConfig()
    # 为了演示方便，稍微调小参数以便在普通 CPU/GPU 运行
    config.dim = 512           
    config.n_heads = 8
    config.n_layers = 2        
    config.num_experts = 8     
    config.top_k = 2
    
    # ==========================================
    # 2. 准备数据 (模拟多模态输入)
    # ==========================================
    print(">>> Preparing Mock Data...")
    processor = UnifiedDataProcessor(config)
    
    # 构造一个混合了 文本、图像、视频 的复杂 Batch
    sample_data = [
        # 样本 1: 纯文本
        [
            {'type': 'text', 'value': 'ERNIE 5.0 is a unified model.'}
        ],
        # 样本 2: 图文混排
        [
            {'type': 'text', 'value': 'Look at this image:'},
            {'type': 'image', 'value': torch.randn(3, 32, 32)}, # 模拟 32x32 图片
            {'type': 'text', 'value': 'It is amazing.'}
        ],
        # 样本 3: 视频
        [
            {'type': 'text', 'value': 'Video analysis:'},
            {'type': 'video', 'value': torch.randn(4, 3, 32, 32)} # 模拟 4 帧视频
        ]
    ]
    
    # 处理并打包成 Tensor
    batch_list = [processor.process_sample(s) for s in sample_data]
    batch = processor.collate_fn(batch_list)
    
    # 搬运到设备
    input_ids = batch['input_ids'].to(device)
    modality_ids = batch['modality_ids'].to(device)
    pos_indices = batch['pos_indices'].to(device)
    
    print(f"Input Batch Shape: {input_ids.shape}") # [Batch, SeqLen]
    
    # ==========================================
    # 3. 初始化模型与损失函数
    # ==========================================
    print(">>> Initializing Model & Objectives...")
    model = Ernie5Model(config).to(device)
    
    # 主任务 Loss (文本生成 + 视觉生成)
    main_criterion = UnifiedObjectiveLoss(config, lambda_visual=1.2, lambda_audio=1.0)
    
    # 辅助任务 Loss (MoE 负载均衡)
    aux_criterion = MoELoadBalancingLoss(config.num_experts, config.top_k)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    # ==========================================
    # 4. 模拟训练循环 (One Step)
    # ==========================================
    print(">>> Starting Training Step...")
    model.train()
    optimizer.zero_grad()
    
    # --- A. 构建自回归目标 ---
    # 输入是 [0, 1, 2, 3], 目标是 [1, 2, 3, 4]
    inp_ids = input_ids[:, :-1]
    inp_mods = modality_ids[:, :-1]
    inp_pos = pos_indices[:, :-1, :]
    
    target_ids = input_ids[:, 1:]
    target_mods = modality_ids[:, 1:] # Loss 需要知道 Target 是什么模态
    
    # --- B. 前向传播 ---
    # logits: [Batch, Seq-1, Vocab]
    # router_logits_list: 每一层的路由分数
    logits, router_logits_list = model(inp_ids, inp_mods, inp_pos)
    
    # --- C. 计算损失 ---
    # 1. 主任务损失 (根据 Target 的模态自动计算 Text/Image/Video Loss)
    task_loss, loss_dict = main_criterion(logits, target_ids, target_mods)
    
    # 2. 辅助损失 (累加每一层的 MoE Balance Loss)
    aux_loss = 0
    for r_logits in router_logits_list:
        # r_logits shape: [Batch*(Seq-1), NumExperts]
        aux_loss += aux_criterion(r_logits)
    
    # 3. 总损失
    total_loss = task_loss + 0.01 * aux_loss
    
    # --- D. 反向传播与优化 ---
    total_loss.backward()
    optimizer.step()
    
    # ==========================================
    # 5. 输出结果
    # ==========================================
    print("\n" + "="*30)
    print("Training Step Completed Successfully!")
    print("="*30)
    print(f"Total Loss   : {total_loss.item():.4f}")
    print(f"Task Loss    : {task_loss.item():.4f}")
    print(f"  > Text     : {loss_dict['loss_text']:.4f}")
    print(f"  > Visual   : {loss_dict['loss_visual']:.4f}")
    print(f"  > Audio    : {loss_dict['loss_audio']:.4f}")
    print(f"Aux Loss     : {aux_loss.item():.4f}")
    print("="*30)

if __name__ == "__main__":
    main()



# import torch
# import torch.optim as optim
# from config import ErnieConfig
# from model import Ernie5Model
# from objectives import UnifiedObjectiveLoss
# from loss import MoELoadBalancingLoss
# from data_processing import UnifiedDataProcessor

# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Running on {device}")
    
#     config = ErnieConfig()
#     processor = UnifiedDataProcessor(config)
    
#     # 构造模拟数据
#     data = [
#         [{'type': 'text', 'value': 'ERNIE 5.0 demo'}],
#         [{'type': 'text', 'value': 'Caption:'}, {'type': 'image', 'value': 'mock_img_tensor'}],
#         [{'type': 'video', 'value': 'mock_video_tensor'}]
#     ]
    
#     batch = processor.collate_fn([processor.process_sample(s) for s in data])
#     inputs = batch['input_ids'].to(device)
#     mods = batch['modality_ids'].to(device)
#     pos = batch['pos_indices'].to(device)

#     model = Ernie5Model(config).to(device)
#     crit = UnifiedObjectiveLoss(config)
#     aux_crit = MoELoadBalancingLoss(config.num_experts, config.top_k)
#     opt = optim.AdamW(model.parameters(), lr=1e-3)
    
#     # Forward
#     out_logits, router_logits = model(inputs[:, :-1], mods[:, :-1], pos[:, :-1, :])
    
#     # Loss
#     task_loss, loss_logs = crit(out_logits, inputs[:, 1:], mods[:, 1:])
#     aux_loss = sum(aux_crit(l) for l in router_logits)
#     total_loss = task_loss + 0.01 * aux_loss
    
#     # Backward
#     opt.zero_grad()
#     total_loss.backward()
#     opt.step()
    
#     print(f"Success! Total Loss: {total_loss.item():.4f}")
#     print(f"Details: {loss_logs}")

# if __name__ == "__main__":
#     main()
