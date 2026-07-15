# -*- coding: utf-8 -*-
"""
watch_demo.py —— 临时演示脚本（非项目正式模块）

用途：让你在 Cursor 的终端面板里"实时"看到 GPU 在跑，
每一步打印 loss / 用时，而不是我事后报告一句"跑通了"。
跑完可以删除这个文件，不影响 src/ 下的正式代码。
"""

import os
import time

import torch
import torch.nn.functional as F

from src.bert_model import BertConfig, BertModel, MLMHead

# 设备选择（遵守 .cursor/rules/gpu-first.mdc：GPU 优先，严禁静默回退 CPU）
#   - 有 CUDA          -> 用 GPU
#   - 无 CUDA 且 ALLOW_CPU=1 -> 用户显式允许，打印醒目警告后用 CPU
#   - 无 CUDA 且未显式允许    -> 直接抛错，交由用户检查环境
if torch.cuda.is_available():
    device = torch.device("cuda")
elif os.environ.get("ALLOW_CPU") == "1":
    print("[警告] 未检测到 GPU，按环境变量 ALLOW_CPU=1 使用 CPU 运行。")
    device = torch.device("cpu")
else:
    raise RuntimeError(
        "未检测到可用 GPU，torch.cuda.is_available()=False。"
        "所有实验默认必须用 GPU。请检查 CUDA 版 torch，"
        "或显式设置环境变量 ALLOW_CPU=1 以允许在 CPU 上运行。"
    )
print(f"[demo] 运行设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
print("[demo] 开始模拟 20 步 mini-batch 前向+反向，观察 loss 变化与耗时……\n")

config = BertConfig(
    vocab_size=30522, hidden_size=256, num_layers=4,
    num_heads=4, intermediate_size=512, max_seq_len=512,
)
model = BertModel(config).to(device)
mlm_head = MLMHead(config).to(device)
mlm_head.tie_weights(model.embeddings.word_embeddings)

optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(mlm_head.parameters()), lr=1e-4
)

batch_size, seq_len = 8, 64
num_steps = 20

for step in range(1, num_steps + 1):
    t0 = time.time()

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    # 随机挑选位置作为"被 mask 的标签"，模拟 MLM 训练目标
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    last_hidden_state, _ = model(input_ids, token_type_ids, attention_mask)
    logits = mlm_head(last_hidden_state)

    loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    print(f"[demo] step {step:2d}/{num_steps}  loss={loss.item():.4f}  "
          f"耗时={dt*1000:.1f}ms  device={logits.device}")
    time.sleep(0.3)  # 放慢节奏，方便肉眼观察

print("\n[demo] 完成。这只是演示随机数据上的前向/反向是否跑得动，loss 无实际训练意义。")
