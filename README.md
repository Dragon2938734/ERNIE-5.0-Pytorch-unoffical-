### 📂 项目目录结构 (Project Structure)

创建名为 `ERNIE-5.0-Pytorch` 的文件夹，并将以下内容保存为对应的文件：


ERNIE-5.0-Pytorch/
├── README.md             # 项目说明文档
├── requirements.txt      # 依赖包列表
├── config.py             # 配置文件 (模型超参数)
├── data_processing.py    # 数据处理 (Mock Tokenizer, 3D Position构建)
├── embeddings.py         # 统一嵌入层 & 3D-RoPE 实现
├── model.py              # 模型骨架 (Transformer Block, LayerNorm)
├── moe_layer.py          # 超稀疏 MoE 层 & 路由机制
├── loss.py               # 辅助 Loss (负载均衡)
├── objectives.py         # 主任务 Loss (统一自回归目标)
└── train.py              # 启动脚本 (训练循环)

---

# ERNIE 5.0: Natively Unified Multimodal LLM (Unofficial PyTorch Implementation)

这是一个基于论文 **[arXiv:2602.04705] ERNIE 5.0 Technical Report** 的概念性 PyTorch 实现。

本项目旨在复现 ERNIE 5.0 的核心架构设计，特别是其**“原生统一（Natively Unified）”**和**“超稀疏 MoE（Ultra-Sparse MoE）”**特性。

## ✨ 核心特性实现

1.  **全模态统一架构 (Natively Unified)**: 
    - 摒弃了独立的 Vision Encoder，所有模态（文本、图像、视频）共享同一个 Transformer 骨干。
    - 实现了 **Unified Embeddings**，将不同模态的 Token 映射到同一语义空间。

2.  **3D 旋转位置编码 (3D-RoPE)**:
    - 实现了时空分解的位置编码。
    - 支持文本的一维序列、图像的二维网格、视频的三维时空流的统一处理。

3.  **超稀疏混合专家 (Ultra-Sparse MoE)**:
    - 实现了细粒度的专家路由（Fine-Grained Experts）。
    - 包含 **Shared Experts**（共享专家）与 **Routed Experts**（路由专家）的混合机制。

4.  **统一训练目标**:
    - 实现了 **Next-Group-of-Tokens Prediction** (针对视觉) 与 **Next-Token Prediction** (针对文本) 的联合训练损失。
    - 包含了改进版的 **Load Balancing Aux Loss**。

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行训练演示
该脚本会生成模拟的多模态数据（文本+图像+视频），并运行一次完整的训练迭代：
```bash
python train.py
```

## 📂 代码模块说明

| 文件名 | 功能描述 |
| :--- | :--- |
| `config.py` | 定义模型超参数（层数、维度、专家数量、词表大小等）。 |
| `embeddings.py` | 实现多模态 Embedding 层的查找与 **3D-RoPE** 位置编码逻辑。 |
| `moe_layer.py` | 核心 MoE 层实现，包含 Top-K Router 和专家网络。 |
| `data_processing.py` | 模拟 Tokenizer，处理原始数据并将图像/视频展平为 Token 序列，构建对应的 `pos_indices`。 |
| `objectives.py` | 实现全模态统一的 Loss 计算（区分文本/视觉的权重）。 |
| `loss.py` | 实现 MoE 路由器的负载均衡辅助损失（Auxiliary Loss）。 |
| `model.py` | 组装 Transformer Block 和整体模型结构。 |
| `train.py` | 整合所有组件，执行 Forward/Backward 流程。 |

## ⚠️ 免责声明
此代码为论文分析性质的复现，旨在解释模型原理。生产环境下的 ERNIE 5.0 需要万亿级参数和大规模分布式训练集群支持，此代码中的 Tokenizer 和数据加载部分为模拟实现。

## 🔗 参考
- Paper: [ERNIE 5.0 Technical Report (arXiv:2602.04705)](https://arxiv.org/abs/2602.04705)
```
