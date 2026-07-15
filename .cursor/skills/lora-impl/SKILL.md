\# LoRA 低秩适配实现助手



\## 描述

专门用于实现 LoRA（Low-Rank Adaptation）微调，确保只训练低秩矩阵 A、B，冻结原始权重。



\## 使用场景

\- 在已有 BERT 的 Attention Q/V 投影层上添加 LoRA

\- 实现 LoRA 层的 forward：h = W\_0 @ x + (B @ A) @ x \* (alpha / r)

\- 对比全量微调与 LoRA 微调的参数量和效果

\- 实现不同 rank 值的实验对比



\## 工作流

1\. 确认原始线性层的输入/输出维度

2\. 创建 LoRALinear 类，包含原始线性层 + A + B

3\. 初始化：A 用高斯初始化，B 用零初始化（保证训练起点 ΔW=0）

4\. 实现 merge/unmerge 方法（可选，用于推理时合并权重）

5\. 提供参数量统计函数



\## 约束

\- 原始权重 W\_0 必须冻结（requires\_grad=False）

\- 只训练 A 和 B（requires\_grad=True）

\- scaling = alpha / r

\- 默认 rank r=8, alpha=16（LoRA 论文常用值）

