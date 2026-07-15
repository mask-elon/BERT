# -*- coding: utf-8 -*-
"""
finetune_lora.py —— 下游任务 LoRA 微调（Day 6）

论文对应：
    [1] Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language Models"
        (https://arxiv.org/abs/2106.09685)
        - 冻结预训练权重 W_0，只训练注入到 Attention Q/V 的低秩矩阵 A、B；
        - 用极少的可训练参数达到接近全量微调的效果（本脚本用于与 Day 4 全量微调对比）。
    [2] Devlin et al., 2018, "BERT" 3.5 节 "Fine-tuning BERT"：下游分类任务范式。

设计原则（严格遵守项目约束）：
    - 禁止使用 transformers / peft。
    - 复用 Day 4 的数据与分类模型（finetune_full.py）、Day 5 的 LoRA（lora.py / apply_lora.py）。
    - 使用同一预训练 checkpoint（Day 3）与同一下游任务（SST-2 情感二分类），保证公平对比。
    - 只训练 LoRA 参数（A、B）与新的分类头 classifier；BERT 骨干原始权重全部冻结。
      说明：分类头是下游任务新引入的层（非"原始 BERT 权重"），必须可训练，
            否则模型无法学习任务；这与"冻结原始 BERT 权重"并不矛盾。

本脚本实现：
    - 加载 Day 3 预训练权重，构建 BertForSequenceClassification（BertModel + 分类头）。
    - 对 BertModel 的每个 BertSelfAttention 的 query/value 注入 LoRA（rank, alpha）。
    - 冻结除 LoRA(A/B) 与 classifier 外的所有参数。
    - 训练循环：AdamW，仅优化可训练参数；记录可训练参数量 / 训练时间 / val_acc。
    - 保存最佳模型（按 val_acc），结果写入 results/lora_finetune_r{rank}.json。

运行：
    python src/finetune_lora.py --config config.json               # rank 取自 config.lora
    python src/finetune_lora.py --config config.json --rank 4      # 覆盖 rank
    python src/finetune_lora.py --config config.json --rank 16
"""

import argparse
import json
import os
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# 复用 Day 4 / Day 5 的组件（允许脚本方式或包方式导入）
try:
    from .finetune_full import (
        BertForSequenceClassification, build_config_from_ckpt, load_pretrained_bert,
        load_sst2, SST2Dataset, make_collate_fn, evaluate, count_params,
        select_device, train_one_epoch,
    )
    from .apply_lora import apply_lora
    from .lora import count_lora_params
    from .tokenizer import WordPieceTokenizer
except ImportError:
    from finetune_full import (
        BertForSequenceClassification, build_config_from_ckpt, load_pretrained_bert,
        load_sst2, SST2Dataset, make_collate_fn, evaluate, count_params,
        select_device, train_one_epoch,
    )
    from apply_lora import apply_lora
    from lora import count_lora_params
    from tokenizer import WordPieceTokenizer


# =============================================================================
# 配置加载（LoRA 版）：合并 shared + finetune + lora 三段
#   - 复用 finetune 段的数据/训练超参（num_train/num_val/num_epochs/batch_size...）
#     以保证与 Day 4 全量微调"同任务同数据"；
#   - lora 段覆盖 LoRA 专属项（rank/alpha/lr/ckpt_path/result_path）。
# 兼容旧的扁平结构（无分段则直接返回整份 config）。
# =============================================================================
def load_lora_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if any(k in cfg for k in ("shared", "pretrain", "finetune", "lora")):
        merged = dict(cfg.get("shared", {}))
        finetune = dict(cfg.get("finetune", {}))  # 继承下游任务数据/训练设置
        # 【重要】不继承全量微调专属的输出路径，否则会覆盖 Day 4 的产物
        #   （full_finetune 的 ckpt_path/result_path 指向 bert_finetune_full.pt /
        #     full_finetune.json）。LoRA 用自己的按 rank 派生路径。
        for drop_key in ("ckpt_path", "result_path"):
            finetune.pop(drop_key, None)
        merged.update(finetune)
        merged.update(cfg.get("lora", {}))        # LoRA 专属项覆盖（含 LoRA 学习率）
        return merged
    return cfg


# =============================================================================
# 只训练 LoRA(A/B) 与分类头 classifier，其余全部冻结
# 论文对应：LoRA 训练阶段只更新低秩矩阵；分类头为下游新层需一并训练。
# =============================================================================
def mark_lora_and_head_trainable(model) -> None:
    for name, param in model.named_parameters():
        is_lora = name.endswith("lora_A") or name.endswith("lora_B")
        is_head = name.startswith("classifier.")
        param.requires_grad = is_lora or is_head


def parse_args():
    parser = argparse.ArgumentParser(description="下游任务 LoRA 微调（Day 6）")
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
    parser.add_argument("--vocab", type=str, default="data/vocab.txt")
    parser.add_argument("--pretrained_ckpt", type=str, default="checkpoints/bert_pretrain.pt")

    # LoRA 超参
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)

    # 数据（默认与 Day 4 全量微调一致，保证公平对比）
    parser.add_argument("--num_train", type=int, default=500)
    parser.add_argument("--num_val", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=2)

    # 优化：LoRA 可训练参数少，通常用比全量微调更大的学习率（这里默认 1e-3）
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # 设备与产出
    parser.add_argument("--allow_cpu", action="store_true", help="显式允许 CPU 运行（默认强制 GPU）")
    parser.add_argument("--allow_fake_data", action="store_true",
                        help="显式允许 SST-2 下载失败时用内置假数据（默认禁止）")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="默认按 rank 命名：checkpoints/bert_finetune_lora_r{rank}.pt")
    parser.add_argument("--result_path", type=str, default=None,
                        help="默认按 rank 命名：results/lora_finetune_r{rank}.json")

    args = parser.parse_args()

    # --config 覆盖（合并 shared + finetune + lora）
    if args.config and os.path.exists(args.config):
        merged = load_lora_config(args.config)
        for k, v in merged.items():
            if hasattr(args, k):
                setattr(args, k, v)
    # 重新解析一次命令行，让显式传入的 --rank 等覆盖 config（命令行优先级最高）
    args = _reapply_cli_overrides(parser, args)

    # 产出路径按 rank 派生（若未显式指定）
    if args.ckpt_path is None:
        args.ckpt_path = f"checkpoints/bert_finetune_lora_r{args.rank}.pt"
    if args.result_path is None:
        args.result_path = f"results/lora_finetune_r{args.rank}.json"
    return args


def _reapply_cli_overrides(parser, args):
    """让命令行显式传入的参数优先于 config 文件。
    做法：再次解析 sys.argv，只把"用户实际在命令行出现过的"选项覆盖回 args。
    """
    import sys
    cli = parser.parse_args()
    # 收集命令行中显式出现的 --xxx 选项名
    passed = set()
    for tok in sys.argv[1:]:
        if tok.startswith("--"):
            passed.add(tok.lstrip("-").split("=")[0])
    for name in passed:
        if hasattr(cli, name):
            setattr(args, name, getattr(cli, name))
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.allow_cpu)
    print(f"[设备] {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"[LoRA] rank={args.rank}, alpha={args.alpha}, lr={args.lr}")

    # ---------- 1) tokenizer ----------
    if not os.path.exists(args.vocab):
        raise FileNotFoundError(f"未找到词表 {args.vocab}，请先运行：python src/tokenizer.py")
    tokenizer = WordPieceTokenizer.load(args.vocab)
    print(f"[词表] 已加载 {args.vocab}, vocab_size = {tokenizer.vocab_size}")

    # ---------- 2) 预训练骨干 + 分类头 ----------
    if not os.path.exists(args.pretrained_ckpt):
        raise FileNotFoundError(
            f"未找到预训练 checkpoint {args.pretrained_ckpt}，请先运行 Day 3：python src/train_pretrain.py")
    config = build_config_from_ckpt(args.pretrained_ckpt, device)
    print(f"[模型] 配置（来自预训练 ckpt）: {config}")

    model = BertForSequenceClassification(config, num_classes=args.num_classes)
    _, missing, unexpected = load_pretrained_bert(model, args.pretrained_ckpt, device)
    print(f"[加载] 已载入预训练骨干权重；missing={len(missing)}, unexpected={len(unexpected)}")

    # ---------- 3) 注入 LoRA 到 Q/V，并只训练 LoRA + 分类头 ----------
    apply_lora(model.bert, rank=args.rank, alpha=args.alpha, targets=("query", "value"))
    model.to(device)
    mark_lora_and_head_trainable(model)

    total_params, trainable_params = count_params(model)
    lora_params = count_lora_params(model)
    head_params = sum(p.numel() for n, p in model.named_parameters()
                      if n.startswith("classifier."))
    print(f"[参数] 总参数量 = {total_params:,} | 可训练 = {trainable_params:,} "
          f"(LoRA={lora_params:,}, 分类头={head_params:,})")
    print(f"[参数] 可训练占比 = {trainable_params / total_params:.4%}")

    # ---------- 4) SST-2 数据（与 Day 4 相同）----------
    train_samples, val_samples = load_sst2(args.num_train, args.num_val,
                                           allow_fake_data=args.allow_fake_data)
    train_labels = {lb for _, lb in train_samples}
    val_labels = {lb for _, lb in val_samples}
    if len(train_labels) < 2 or len(val_labels) < 2:
        print(f"[警告] 数据标签不足两类（train={train_labels}, val={val_labels}），"
              f"val_acc 将不具参考意义。")
    collate = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        SST2Dataset(train_samples, tokenizer, args.max_seq_len),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(
        SST2Dataset(val_samples, tokenizer, args.max_seq_len),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # ---------- 5) 优化器（仅可训练参数）+ 训练循环 ----------
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    epoch_times = []
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        # 复用 Day 4 的 train_one_epoch（对 model.parameters() 求梯度，
        # 但冻结参数 requires_grad=False，不会被优化器更新，等价于只训练 LoRA+头）
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.max_norm)
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        val_acc = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}/{args.num_epochs}] train_loss={train_loss:.4f} | "
              f"val_acc={val_acc:.4f} | time={epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(os.path.abspath(args.ckpt_path)), exist_ok=True)
            torch.save({
                "model_config": {
                    "vocab_size": config.vocab_size, "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers, "num_heads": config.num_heads,
                    "intermediate_size": config.intermediate_size,
                    "max_seq_len": config.max_seq_len,
                    "type_vocab_size": config.type_vocab_size,
                    "pad_token_id": config.pad_token_id,
                },
                "num_classes": args.num_classes,
                "rank": args.rank, "alpha": args.alpha,
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_acc": val_acc,
            }, args.ckpt_path)
            print(f"  -> 新最佳 val_acc={val_acc:.4f}，已保存 checkpoint: {args.ckpt_path}")

    avg_epoch_time = sum(epoch_times) / max(1, len(epoch_times))

    # ---------- 6) 保存结果 JSON（与 full_finetune.json 字段对齐，便于 compare.py）----------
    result = {
        "method": f"lora_r{args.rank}",
        "trainable_params": trainable_params,   # LoRA A/B + 分类头（与全量微调口径一致）
        "total_params": total_params,
        "lora_params": lora_params,             # 仅 LoRA A/B
        "head_params": head_params,             # 仅分类头
        "train_time_per_epoch": avg_epoch_time,
        "final_val_acc": best_val_acc,
        "rank": args.rank,
        "alpha": args.alpha,
        "config": {
            "task": "SST-2", "num_train": args.num_train, "num_val": args.num_val,
            "max_seq_len": args.max_seq_len, "batch_size": args.batch_size,
            "num_classes": args.num_classes, "num_epochs": args.num_epochs,
            "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed,
            "device": str(device), "pretrained_ckpt": args.pretrained_ckpt,
            "hidden_size": config.hidden_size, "num_layers": config.num_layers,
            "num_heads": config.num_heads, "intermediate_size": config.intermediate_size,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.result_path)), exist_ok=True)
    with open(args.result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n===== LoRA 微调结果 (rank={args.rank}) =====")
    print(f"  总参数量         : {total_params:,}")
    print(f"  可训练参数量     : {trainable_params:,}  (LoRA={lora_params:,}, 头={head_params:,})")
    print(f"  每 epoch 训练时间: {avg_epoch_time:.1f}s")
    print(f"  最终 val_acc     : {best_val_acc:.4f}")
    print(f"  结果已保存       : {args.result_path}")
    print(f"  LoRA checkpoint  : {args.ckpt_path}")


# =============================================================================
# __main__：直接运行即执行 LoRA 微调。
# 快速自测（小样本、少轮）：
#   python src/finetune_lora.py --num_train 64 --num_val 32 --num_epochs 1 --rank 8
# =============================================================================
if __name__ == "__main__":
    main()
