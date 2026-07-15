# -*- coding: utf-8 -*-
"""
lora.py —— 从零手写 LoRA 低秩适配器（Day 5）

论文对应：
    [1] Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language Models"
        (https://arxiv.org/abs/2106.09685)
        - 4.1 节 "Low-Rank-Parametrized Update Matrices"：
            冻结预训练权重 W_0 ∈ R^{d×k}，用低秩分解表示增量：
                W_0 + ΔW = W_0 + B A，  B ∈ R^{d×r}，A ∈ R^{r×k}，r << min(d,k)
            前向：h = W_0 x + ΔW x = W_0 x + (α/r) · B A x
        - 4.1 节初始化：A ~ N(0, σ²) 高斯初始化，B = 0，
            使训练开始时 ΔW = B A = 0（不扰动预训练模型）。
        - 4.1 节缩放：ΔW x 乘以 α/r（scaling），α 为常数，便于调 r 时少调超参。

设计原则（严格遵守项目约束）：
    - 禁止 import peft，LoRA 全部手写。
    - 只训练 A、B（requires_grad=True），冻结原始权重 W_0（requires_grad=False）。
    - 提供 merge()/unmerge()（推理时把 ΔW 合并回 W_0 加速）。
    - 可插拔：LoRALinear 包装一个已存在的 nn.Linear，不改变其输入/输出维度。

维度约定（重要，避免与论文符号混淆）：
    原始 nn.Linear(in_features=k, out_features=d)，其权重 weight 形状为 [d, k]，
    前向为 y = x W_0^T + b（x 形状 [..., k]，y 形状 [..., d]）。
    LoRA:
        lora_A（论文 A）: 形状 [r, k]（降维），高斯初始化 N(0, 0.02)
        lora_B（论文 B）: 形状 [d, r]（升维），零初始化
        ΔW = B A ∈ R^{d×k}，与 W_0 同形；
        增量前向 = x A^T B^T · (α/r) = x (B A)^T · (α/r)，与 y 同形 [..., d]。
"""

import argparse
import math
import os

import torch
import torch.nn as nn


# =============================================================================
# LoRALinear：包装原始 nn.Linear 的低秩适配层
# 论文对应：LoRA 4.1 节，h = W_0 x + (α/r) B A x。
# =============================================================================
class LoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        lora_dropout: float = 0.0,
        init_std: float = 0.02,
    ):
        super().__init__()
        assert isinstance(base_linear, nn.Linear), "LoRALinear 只能包装 nn.Linear"
        assert rank > 0, "rank 必须为正整数"

        # 原始线性层（W_0, b）——保留并冻结
        self.base = base_linear
        self.in_features = base_linear.in_features    # k
        self.out_features = base_linear.out_features  # d
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank                   # α/r
        self.init_std = init_std

        # 冻结原始权重 W_0（以及 bias），不参与梯度更新（论文：预训练权重冻结）
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # LoRA 低秩矩阵（唯一可训练参数）
        #   lora_A: [r, k]，高斯初始化 N(0, init_std²)
        #   lora_B: [d, r]，零初始化 -> 训练起点 ΔW = B A = 0
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.normal_(self.lora_A, mean=0.0, std=init_std)
        # lora_B 已是 0

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        # 是否已把 ΔW 合并进 base.weight（合并后前向不再额外走低秩支路）
        self.merged = False

    # ---------- 前向：h = W_0 x + (α/r) B A x ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始支路：W_0 x + b
        result = self.base(x)
        # 已合并则 ΔW 已在 base.weight 中，无需再加低秩支路
        if self.merged:
            return result
        # 低秩支路：x @ A^T -> [..., r] -> @ B^T -> [..., d]，再乘缩放
        #   等价于 (B A) x，因为 x (B A)^T = x A^T B^T
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return result + lora_out * self.scaling

    # ---------- 计算 ΔW = (α/r) B A ----------
    def delta_weight(self) -> torch.Tensor:
        # [d, r] @ [r, k] -> [d, k]，与 W_0 同形
        return (self.lora_B @ self.lora_A) * self.scaling

    # ---------- merge：把 ΔW 合并回 W_0（推理加速，可选）----------
    def merge(self):
        # 论文 4.1：部署时可令 W = W_0 + BA，前向退化为普通 Linear，无额外延迟
        if self.merged:
            return
        self.base.weight.data += self.delta_weight()
        self.merged = True

    def unmerge(self):
        # 还原 W_0（撤销 merge），便于继续训练或移除 LoRA
        if not self.merged:
            return
        self.base.weight.data -= self.delta_weight()
        self.merged = False

    # ---------- 统计 LoRA 引入的可训练参数量 ----------
    def get_trainable_params(self) -> int:
        # 仅 A、B 两个低秩矩阵：r*k + d*r
        return self.lora_A.numel() + self.lora_B.numel()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
                f"merged={self.merged}")


# =============================================================================
# 便捷统计：某模块内所有 LoRALinear 引入的可训练参数量
# =============================================================================
def count_lora_params(module: nn.Module) -> int:
    total = 0
    for m in module.modules():
        if isinstance(m, LoRALinear):
            total += m.get_trainable_params()
    return total


# =============================================================================
# __main__：LoRALinear 单元自测（对应用户规则“实现清单”）
# 检查项：
#   - import 不报错、前向维度正确
#   - 初始化时 ΔW = 0（LoRA 输出与原始 Linear 一致）
#   - W_0 冻结（requires_grad=False），A/B 可训练
#   - 反向后仅 A/B 有梯度，W_0 无梯度
#   - merge 后前向结果与 merge 前一致（数值等价）
#   - 参数量统计正确 = r*k + d*r
# 运行：python src/lora.py
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRALinear 单元自测")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(42)

    # 设备选择（遵守 .cursor/rules/gpu-first.mdc：GPU 优先，严禁静默回退 CPU）
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
    print(f"运行设备：{device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    in_features, out_features = 32, 64
    batch, seq = 4, 10

    base = nn.Linear(in_features, out_features).to(device)
    lora = LoRALinear(base, rank=args.rank, alpha=args.alpha).to(device)
    print("LoRALinear:", lora)

    x = torch.randn(batch, seq, in_features, device=device)

    # ---- 1) 初始化时 ΔW = 0：LoRA 输出应与原始 Linear 完全一致 ----
    lora.eval()
    with torch.no_grad():
        y_base = base(x)          # 注意：base 已被 lora 冻结，但前向仍可用
        y_lora = lora(x)
    assert torch.allclose(y_base, y_lora, atol=1e-6), "初始化时 ΔW 应为 0"
    print("[OK] 初始化 ΔW=0：LoRA 输出与原始 Linear 一致")

    # ---- 2) 冻结/可训练检查 ----
    assert lora.base.weight.requires_grad is False, "W_0 应被冻结"
    if lora.base.bias is not None:
        assert lora.base.bias.requires_grad is False, "bias 应被冻结"
    assert lora.lora_A.requires_grad is True and lora.lora_B.requires_grad is True
    print("[OK] W_0 冻结，A/B 可训练")

    # ---- 3) 反向：仅 A/B 有梯度，W_0 无梯度 ----
    lora.train()
    y = lora(x).sum()
    y.backward()
    assert lora.base.weight.grad is None, "冻结的 W_0 不应有梯度"
    assert lora.lora_A.grad is not None and lora.lora_B.grad is not None, "A/B 应有梯度"
    print("[OK] 反向后仅 A/B 有梯度，W_0 无梯度")

    # ---- 4) 让 B 非零后，验证 merge 前后数值等价 ----
    with torch.no_grad():
        lora.lora_B += torch.randn_like(lora.lora_B) * 0.1  # 打破 B=0，产生非零 ΔW
    lora.eval()
    with torch.no_grad():
        y_before = lora(x)
        lora.merge()
        y_after = lora(x)
    assert torch.allclose(y_before, y_after, atol=1e-5), "merge 前后前向应数值等价"
    print("[OK] merge 前后前向结果一致（数值等价）")
    lora.unmerge()  # 还原，便于后续使用

    # ---- 5) 参数量统计 ----
    expected = args.rank * in_features + out_features * args.rank
    got = lora.get_trainable_params()
    assert got == expected, f"参数量应为 {expected}，实际 {got}"
    print(f"[OK] LoRA 可训练参数量 = {got}  (= r*k + d*r = "
          f"{args.rank}*{in_features} + {out_features}*{args.rank})")

    print("\n[OK] LoRALinear 所有自测通过。")
