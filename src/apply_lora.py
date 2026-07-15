# -*- coding: utf-8 -*-
"""
apply_lora.py —— 将手写 LoRA 注入 BertModel 的 Attention Q/V 投影层（Day 5）

论文对应：
    [1] Hu et al., 2021, "LoRA" 4.2 节 "Applying LoRA to Transformer"：
        论文只把 LoRA 加在自注意力的投影矩阵上（W_q, W_k, W_v, W_o），
        并在实验中发现"只适配 W_q 和 W_v"即可用很小的参数量取得好效果。
        本脚本据此把 LoRA 注入每个 BertSelfAttention 的 query 与 value 线性层。

设计原则（严格遵守项目约束）：
    - 禁止 import peft；复用自写的 lora.LoRALinear 与 bert_model.BertModel。
    - 可插拔：apply_lora() 就地把目标 nn.Linear 替换为 LoRALinear；
      remove_lora() 还原为原始 nn.Linear，实现"开关 LoRA"。
    - 加载 Day 3 预训练 checkpoint 后再注入 LoRA，验证参数量对比与梯度冻结。

提供接口：
    apply_lora(model, rank=8, alpha=16, targets=("query","value"))
    remove_lora(model)
    mark_only_lora_as_trainable(model)   # 冻结除 LoRA 外的所有参数
    count_parameters(model)              # (total, trainable)
"""

import argparse
import json
import os

import torch
import torch.nn as nn

# 允许以脚本方式或包方式导入自写模块
try:
    from .bert_model import BertConfig, BertModel, BertSelfAttention
    from .lora import LoRALinear, count_lora_params
    from .tokenizer import WordPieceTokenizer
except ImportError:
    from bert_model import BertConfig, BertModel, BertSelfAttention
    from lora import LoRALinear, count_lora_params
    from tokenizer import WordPieceTokenizer


# =============================================================================
# apply_lora：遍历所有 BertSelfAttention，把目标投影层替换为 LoRALinear
# 论文对应：LoRA 4.2 节，默认只适配 query 和 value。
# =============================================================================
def apply_lora(model: nn.Module, rank: int = 8, alpha: int = 16,
               targets=("query", "value")) -> nn.Module:
    replaced = 0
    for module in model.modules():
        if isinstance(module, BertSelfAttention):
            for name in targets:
                layer = getattr(module, name, None)
                # 仅替换尚未包装的原始 nn.Linear（幂等：重复调用不会二次包装）
                if isinstance(layer, nn.Linear):
                    setattr(module, name, LoRALinear(layer, rank=rank, alpha=alpha))
                    replaced += 1
    print(f"[LoRA] 已注入 {replaced} 个 LoRALinear（targets={targets}, rank={rank}, alpha={alpha}）")
    return model


# =============================================================================
# remove_lora：把 LoRALinear 还原为原始 nn.Linear（关闭 LoRA）
# 若之前 merge 过，先 unmerge 还原 W_0；恢复原权重的 requires_grad=True。
# =============================================================================
def remove_lora(model: nn.Module) -> nn.Module:
    restored = 0
    for module in model.modules():
        if isinstance(module, BertSelfAttention):
            for name in ("query", "key", "value"):
                layer = getattr(module, name, None)
                if isinstance(layer, LoRALinear):
                    layer.unmerge()                       # 撤销可能的合并
                    base = layer.base                     # 取回原始 nn.Linear
                    base.weight.requires_grad = True      # 解冻，恢复原始可训练状态
                    if base.bias is not None:
                        base.bias.requires_grad = True
                    setattr(module, name, base)
                    restored += 1
    print(f"[LoRA] 已移除 {restored} 个 LoRALinear，恢复原始 Linear")
    return model


# =============================================================================
# mark_only_lora_as_trainable：冻结除 LoRA (A/B) 外的所有参数
# 论文对应：LoRA 训练时只更新低秩矩阵，其余全部冻结。
# =============================================================================
def mark_only_lora_as_trainable(model: nn.Module) -> nn.Module:
    for name, param in model.named_parameters():
        # LoRALinear 的可训练参数命名以 ".lora_A" / ".lora_B" 结尾
        param.requires_grad = name.endswith("lora_A") or name.endswith("lora_B")
    return model


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# 从预训练 checkpoint 重建配置并载入骨干权重（兼容 Day 3 保存格式）
# =============================================================================
def build_model_from_ckpt(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    mc = ckpt["model_config"]
    config = BertConfig(
        vocab_size=mc["vocab_size"], hidden_size=mc["hidden_size"],
        num_layers=mc["num_layers"], num_heads=mc["num_heads"],
        intermediate_size=mc["intermediate_size"], max_seq_len=mc["max_seq_len"],
        type_vocab_size=mc.get("type_vocab_size", 2),
        pad_token_id=mc.get("pad_token_id", 0),
    )
    model = BertModel(config).to(device)
    missing, unexpected = model.load_state_dict(ckpt["bert_state_dict"], strict=False)
    return model, config, missing, unexpected


def select_device(allow_cpu: bool) -> torch.device:
    # 设备选择（GPU 优先，见 .cursor/rules/gpu-first.mdc）
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif allow_cpu:
        print("[警告] 未检测到可用 GPU，已按 --allow_cpu 使用 CPU 运行。")
        return torch.device("cpu")
    else:
        raise RuntimeError(
            "未检测到可用 GPU，当前环境 torch.cuda.is_available()=False。"
            "所有实验默认必须用 GPU。请检查 CUDA 版 torch（见 requirements.txt），"
            "或显式加 --allow_cpu 才能用 CPU 运行。"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="向 BertModel 注入 LoRA 并验证（Day 5）")
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
    parser.add_argument("--vocab", type=str, default="data/vocab.txt")
    parser.add_argument("--pretrained_ckpt", type=str, default="checkpoints/bert_pretrain.pt")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--allow_cpu", action="store_true")
    args = parser.parse_args()

    # --config 支持分段结构（shared / lora），也兼容旧扁平结构
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if any(k in cfg for k in ("shared", "pretrain", "finetune", "lora")):
            merged = {}
            merged.update(cfg.get("shared", {}))
            merged.update(cfg.get("lora", {}))
        else:
            merged = cfg
        for k, v in merged.items():
            if hasattr(args, k):
                setattr(args, k, v)
    return args


# =============================================================================
# __main__：LoRA 注入验证（对应 Day 5 阶段任务 3）
#   1) 加载 Day 3 预训练 checkpoint
#   2) 应用 LoRA（rank=8, alpha=16）到 Q/V
#   3) 打印"原始总参数量 vs LoRA 可训练参数量"对比
#   4) forward + backward 一次，确认冻结的 W_0 无梯度、仅 A/B 有梯度
# 运行：python src/apply_lora.py
#       python src/apply_lora.py --config config.json
# =============================================================================
if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(42)
    device = select_device(args.allow_cpu)
    print(f"[设备] {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ---- 1) 加载预训练骨干 ----
    if not os.path.exists(args.pretrained_ckpt):
        raise FileNotFoundError(
            f"未找到预训练 checkpoint {args.pretrained_ckpt}，请先运行 Day 3：python src/train_pretrain.py")
    model, config, missing, unexpected = build_model_from_ckpt(args.pretrained_ckpt, device)
    print(f"[加载] 预训练骨干载入完成；missing={len(missing)}, unexpected={len(unexpected)}")
    print(f"[模型] {config}")

    # 全参数量（LoRA 之前）：整体基线
    total_before, trainable_before = count_parameters(model)

    # ---- 2) 应用 LoRA 到 Q/V，并只训练 LoRA 参数 ----
    apply_lora(model, rank=args.rank, alpha=args.alpha, targets=("query", "value"))
    model.to(device)
    mark_only_lora_as_trainable(model)

    total_after, trainable_after = count_parameters(model)
    lora_params = count_lora_params(model)

    # ---- 3) 参数量对比 ----
    print("\n===== 参数量对比（原始 vs LoRA）=====")
    print(f"  原始总参数量               : {total_before:,}")
    print(f"  注入 LoRA 后总参数量       : {total_after:,}")
    print(f"  LoRA 可训练参数量          : {trainable_after:,}  (LoRA A/B = {lora_params:,})")
    print(f"  可训练占比                 : {trainable_after / total_after:.4%}")
    print(f"  相对全量微调的压缩倍数     : {total_before / max(1, trainable_after):.1f}x")

    # ---- 4) forward + backward，验证梯度冻结 ----
    vocab_size = config.vocab_size
    batch, seq = 2, min(16, args.max_seq_len)
    input_ids = torch.randint(0, vocab_size, (batch, seq), device=device)
    attention_mask = torch.ones(batch, seq, dtype=torch.long, device=device)

    model.train()
    sequence_output, pooled_output = model(input_ids, attention_mask=attention_mask)
    loss = sequence_output.sum() + pooled_output.sum()
    loss.backward()

    # 收集第一个 BertSelfAttention 的 query（现为 LoRALinear）做逐项检查
    checked = 0
    frozen_ok = True
    lora_ok = True
    for module in model.modules():
        if isinstance(module, BertSelfAttention):
            q = module.query  # LoRALinear
            if isinstance(q, LoRALinear):
                # 冻结的 W_0 不应有梯度
                if q.base.weight.grad is not None:
                    frozen_ok = False
                # A/B 应有梯度
                if q.lora_A.grad is None or q.lora_B.grad is None:
                    lora_ok = False
                checked += 1

    print("\n===== 梯度冻结验证 =====")
    print(f"  检查的 LoRA-Q 层数量        : {checked}")
    print(f"  冻结的 W_0 无梯度           : {'是 [OK]' if frozen_ok else '否 [FAIL]'}")
    print(f"  LoRA A/B 有梯度            : {'是 [OK]' if lora_ok else '否 [FAIL]'}")

    # 统计全模型中"有梯度的参数"是否只来自 LoRA
    grad_param_names = [n for n, p in model.named_parameters()
                        if p.requires_grad and p.grad is not None]
    non_lora_grad = [n for n in grad_param_names
                     if not (n.endswith("lora_A") or n.endswith("lora_B"))]
    print(f"  有梯度的可训练参数总数      : {len(grad_param_names)}（应全部为 LoRA A/B）")
    print(f"  非 LoRA 却有梯度的参数      : {len(non_lora_grad)}（应为 0）")

    assert frozen_ok, "冻结的 W_0 不应产生梯度"
    assert lora_ok, "LoRA A/B 应产生梯度"
    assert len(non_lora_grad) == 0, "除 LoRA 外不应有其他参数被更新"

    # ---- 5) 可插拔验证：remove_lora 后恢复为普通 Linear ----
    remove_lora(model)
    still_lora = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    assert still_lora == 0, "remove_lora 后不应残留 LoRALinear"
    print(f"\n[OK] remove_lora 成功：残留 LoRALinear = {still_lora}")

    print("\n[OK] LoRA 注入 / 参数量对比 / 梯度冻结 / 可插拔 全部验证通过。")
