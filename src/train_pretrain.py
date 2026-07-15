# -*- coding: utf-8 -*-
"""
train_pretrain.py —— BERT MLM 预训练主循环（Day 3）

论文对应：
    [1] Devlin et al., 2018, "BERT: Pre-training of Deep Bidirectional
        Transformers for Language Understanding" (https://arxiv.org/abs/1810.04805)
        3.1 节 "Task #1: Masked LM"：仅在被 mask 位置计交叉熵损失。
        A.2 节 "Pre-training Procedure"：Adam + L2 权重衰减 + 学习率 warmup
        后线性衰减 + dropout。
    [2] Vaswani et al., 2017, "Attention Is All You Need" 5.3 节 "Optimizer"：
        学习率先线性 warmup 再衰减的调度思想。

设计原则（严格遵守项目约束）：
    - 复用自写的 bert_model.py（BertModel + MLMHead）与 data.py（WikiText-2 管线）。
    - 不使用 transformers / peft / 预训练权重。
    - 保持最小可运行（MVP）：小规模配置，CPU 亦可跑通。

本脚本实现：
    - 加载自写 tokenizer 与 WikiText-2 数据（MLM-only）。
    - 初始化 BertModel + MLMHead（解码器与词嵌入权重共享）。
    - 优化器 AdamW（lr=1e-4, weight_decay=0.01, betas=(0.9,0.999)）。
    - 学习率调度：前 warmup_ratio 比例的 step 线性升温到 peak_lr，之后线性衰减到 0。
    - 梯度裁剪 max_norm=1.0。
    - 训练循环：每 epoch 记录 avg train loss；验证循环计算 val loss / perplexity。
    - 保存最佳 checkpoint（按 val loss），可被后续脚本加载。
    - 记录训练日志（时间戳、epoch、train_loss、val_loss、lr）。
    - matplotlib 绘制 loss 曲线到 figures/loss_pretrain.png。

运行：
    python src/train_pretrain.py
    python src/train_pretrain.py --config config.json
"""

import argparse
import json
import logging
import math
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# 允许以脚本方式（python src/train_pretrain.py）或包方式导入自写模块
try:
    from .bert_model import BertConfig, BertModel, MLMHead
    from .data import build_dataloader
    from .tokenizer import WordPieceTokenizer
except ImportError:
    from bert_model import BertConfig, BertModel, MLMHead
    from data import build_dataloader
    from tokenizer import WordPieceTokenizer


# =============================================================================
# 配置加载：支持分段结构 {shared, pretrain, finetune}，也兼容旧的扁平结构
# 分段模式：返回 {**shared, **<section>}（本节覆盖 shared 同名键）。
# 扁平模式：直接返回整份 config（向后兼容 Day 3 早期写法）。
# =============================================================================
def load_config_section(path: str, section: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if any(k in cfg for k in ("shared", "pretrain", "finetune")):
        merged = dict(cfg.get("shared", {}))
        merged.update(cfg.get(section, {}))
        return merged
    return cfg  # 旧的扁平结构


# =============================================================================
# 学习率调度：warmup + 线性衰减
# 论文对应：BERT A.2 / Transformer 5.3。
# 定义一个 step -> 缩放系数 的函数（乘到 peak_lr 上）：
#     step < warmup_steps       : 线性升温    scale = step / warmup_steps
#     warmup_steps <= step      : 线性衰减到 0 scale = (T - step)/(T - warmup_steps)
# 其中 T = total_steps。返回值恒 >= 0。
# =============================================================================
def build_warmup_linear_scheduler(optimizer, warmup_steps: int, total_steps: int):
    warmup_steps = max(1, warmup_steps)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        # 衰减段：从 1 线性降到 0
        remain = total_steps - current_step
        denom = max(1, total_steps - warmup_steps)
        return max(0.0, float(remain) / float(denom))

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# 日志器：同时输出到控制台与文件（时间戳 + 级别 + 内容）
# =============================================================================
def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    logger = logging.getLogger("train_pretrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# =============================================================================
# 收集去重的可训练参数
# 说明：MLMHead.decoder.weight 与 word_embeddings.weight 权重共享（同一张量），
#       若直接 list(model.parameters()) + list(mlm_head.parameters()) 会重复计入，
#       导致优化器 "duplicate parameters" 警告。这里按 id 去重后保序返回。
# =============================================================================
def collect_unique_params(*modules):
    seen, params = set(), []
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return params


# =============================================================================
# MLM 损失：logits[B,S,V] 与 labels[B,S]（非 mask 位为 -100）
# 论文对应：BERT 3.1，仅在被 mask 位置计交叉熵。
# =============================================================================
def mlm_loss_fn(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100,
    )


# =============================================================================
# 验证循环：在 val loader 上计算平均 MLM loss 与 perplexity
# perplexity = exp(avg_loss)（语言模型常用指标）。
# =============================================================================
@torch.no_grad()
def evaluate(model, mlm_head, loader, device, vocab_size):
    model.eval()
    mlm_head.eval()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        sequence_output, _ = model(input_ids, token_type_ids, attention_mask)
        logits = mlm_head(sequence_output)
        loss = mlm_loss_fn(logits, labels, vocab_size)
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    ppl = math.exp(min(20.0, avg_loss))  # clip 指数防溢出
    return avg_loss, ppl


# =============================================================================
# 训练一个 epoch：前向 -> 反向 -> 梯度裁剪 -> optimizer/scheduler step
# 返回该 epoch 的平均 train loss。
# =============================================================================
def train_one_epoch(model, mlm_head, loader, optimizer, scheduler, device,
                    vocab_size, max_norm, logger, epoch, log_interval=20):
    model.train()
    mlm_head.train()
    params = collect_unique_params(model, mlm_head)

    total_loss, n_batches = 0.0, 0
    t0 = time.time()
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        sequence_output, _ = model(input_ids, token_type_ids, attention_mask)
        logits = mlm_head(sequence_output)
        loss = mlm_loss_fn(logits, labels, vocab_size)

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（论文常用 max_norm=1.0，稳定训练）
        torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if step % log_interval == 0:
            cur_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"epoch {epoch} | step {step}/{len(loader)} | "
                f"loss {loss.item():.4f} | lr {cur_lr:.2e}"
            )

    avg_loss = total_loss / max(1, n_batches)
    epoch_time = time.time() - t0
    return avg_loss, epoch_time


# =============================================================================
# checkpoint 保存：包含配置 + 模型/头权重 + 训练元信息，便于后续加载
# =============================================================================
def save_checkpoint(path, config: BertConfig, model, mlm_head, epoch,
                    train_loss, val_loss, args):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ckpt = {
        "model_config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "intermediate_size": config.intermediate_size,
            "max_seq_len": config.max_seq_len,
            "type_vocab_size": config.type_vocab_size,
            "pad_token_id": config.pad_token_id,
        },
        "bert_state_dict": model.state_dict(),
        "mlm_head_state_dict": mlm_head.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "args": vars(args),
    }
    torch.save(ckpt, path)


# =============================================================================
# loss 曲线绘制：train / val loss 随 epoch 变化，保存 png
# =============================================================================
def plot_loss_curve(history, out_path):
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境（服务器/CI）也能出图
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, "o-", label="train loss")
    plt.plot(epochs, val_losses, "s-", label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("MLM cross-entropy loss")
    plt.title("BERT MLM Pre-training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# =============================================================================
# 参数解析：命令行默认值 + 可选 --config JSON 覆盖
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="BERT MLM 预训练主循环（Day 3）")
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
    parser.add_argument("--vocab", type=str, default="data/vocab.txt", help="词表路径")

    # 数据
    parser.add_argument("--fraction", type=float, default=0.1, help="训练集采样比例（前 10%%）")
    parser.add_argument("--val_fraction", type=float, default=0.5, help="验证集采样比例")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16, help="CPU 建议调到 8")
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    # 模型（小规模配置）
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=512)

    # 优化 / 调度
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4, help="peak_lr")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--val_every", type=int, default=1, help="每 N 个 epoch 验证一次")

    # 产出路径
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/bert_pretrain.pt")
    parser.add_argument("--fig_path", type=str, default="figures/loss_pretrain.png")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42)

    # 设备策略：默认强制 GPU（CUDA）。仅当显式加 --allow_cpu 时才允许 CPU 运行。
    parser.add_argument("--allow_cpu", action="store_true", default=False,
                        help="显式允许在 CPU 上运行（默认关闭，强制使用 GPU）")

    args = parser.parse_args()

    # --config 覆盖：支持分段结构 {shared, pretrain, finetune}，
    # 本脚本只合并 shared + pretrain，避免与 finetune 的 lr/ckpt_path 互相污染。
    # 向后兼容：若 config 仍是旧的扁平结构，则按同名键直接覆盖。
    if args.config and os.path.exists(args.config):
        merged = load_config_section(args.config, "pretrain")
        for k, v in merged.items():
            if hasattr(args, k):
                setattr(args, k, v)
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # 设备选择（GPU 优先，最高优先级约束）：
    #   - 检测到 CUDA -> 用 GPU；
    #   - 未检测到 CUDA 且未加 --allow_cpu -> 直接抛错，禁止静默回退 CPU；
    #   - 未检测到 CUDA 且显式加了 --allow_cpu -> 打印警告后用 CPU。
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        print("[警告] 未检测到可用 GPU，已按 --allow_cpu 使用 CPU 运行（速度会很慢）。")
        device = torch.device("cpu")
    else:
        raise RuntimeError(
            "未检测到可用 GPU，当前环境 torch.cuda.is_available()=False。"
            "所有实验默认必须用 GPU。"
            "请检查是否安装了 CUDA 版 torch（见 requirements.txt），"
            "或显式加 --allow_cpu 才能用 CPU 运行。"
        )

    # 日志（文件名带时间戳）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"train_pretrain_{ts}.log")
    logger = setup_logger(log_path)
    logger.info("=" * 70)
    logger.info("BERT MLM 预训练启动")
    logger.info(f"设备: {device}" +
                (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    logger.info(f"超参数: {vars(args)}")

    # ---------- 1) tokenizer ----------
    if not os.path.exists(args.vocab):
        raise FileNotFoundError(
            f"未找到词表 {args.vocab}，请先运行：python src/tokenizer.py")
    tokenizer = WordPieceTokenizer.load(args.vocab)
    logger.info(f"词表已加载: {args.vocab}, vocab_size = {tokenizer.vocab_size}")

    # ---------- 2) 数据（MLM-only，与 BertModel + MLMHead 对应）----------
    _, train_loader = build_dataloader(
        tokenizer, split="train", fraction=args.fraction,
        max_seq_len=args.max_seq_len, batch_size=args.batch_size,
        mlm_probability=args.mlm_probability, nsp=False, shuffle=True,
    )
    _, val_loader = build_dataloader(
        tokenizer, split="validation", fraction=args.val_fraction,
        max_seq_len=args.max_seq_len, batch_size=args.batch_size,
        mlm_probability=args.mlm_probability, nsp=False, shuffle=False,
    )
    logger.info(f"训练 batch 数: {len(train_loader)} | 验证 batch 数: {len(val_loader)}")

    # ---------- 3) 模型：BertModel + MLMHead（权重共享）----------
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_seq_len=args.max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertModel(config).to(device)
    mlm_head = MLMHead(config).to(device)
    mlm_head.apply(model._init_weights)                       # 头部单独初始化
    mlm_head.tie_weights(model.embeddings.word_embeddings)    # 解码器与词嵌入共享权重
    logger.info(f"模型配置: {config}")
    n_params = sum(p.numel() for p in collect_unique_params(model, mlm_head))
    logger.info(f"可训练参数量(去重): {n_params:,}")

    # ---------- 4) 优化器 + 调度器 ----------
    params = collect_unique_params(model, mlm_head)
    optimizer = AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    total_steps = args.num_epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = build_warmup_linear_scheduler(optimizer, warmup_steps, total_steps)
    logger.info(f"总训练步数: {total_steps} | warmup 步数: {warmup_steps}")

    # ---------- 5) 训练主循环 ----------
    history = []
    best_val_loss = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        train_loss, epoch_time = train_one_epoch(
            model, mlm_head, train_loader, optimizer, scheduler, device,
            config.vocab_size, args.max_norm, logger, epoch,
        )

        # 验证（每 val_every 个 epoch，且最后一个 epoch 必验证）
        do_val = (epoch % args.val_every == 0) or (epoch == args.num_epochs)
        if do_val:
            val_loss, val_ppl = evaluate(
                model, mlm_head, val_loader, device, config.vocab_size)
        else:
            val_loss, val_ppl = float("nan"), float("nan")

        cur_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"[epoch {epoch}/{args.num_epochs}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_ppl={val_ppl:.2f} | lr={cur_lr:.2e} | time={epoch_time:.1f}s"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_ppl": val_ppl, "lr": cur_lr,
        })

        # 保存最佳 checkpoint（按 val loss）
        if do_val and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.ckpt_path, config, model, mlm_head,
                            epoch, train_loss, val_loss, args)
            logger.info(f"  -> 新最佳 val_loss={val_loss:.4f}，已保存 checkpoint: {args.ckpt_path}")

    # 若从未触发保存（异常情况），兜底保存最后一轮
    if not os.path.exists(args.ckpt_path):
        last = history[-1]
        save_checkpoint(args.ckpt_path, config, model, mlm_head,
                        last["epoch"], last["train_loss"], last["val_loss"], args)
        logger.info(f"兜底保存最后一轮 checkpoint: {args.ckpt_path}")

    # ---------- 6) 绘制 loss 曲线 ----------
    plot_loss_curve(history, args.fig_path)
    logger.info(f"loss 曲线已保存: {args.fig_path}")
    logger.info(f"最佳 val_loss = {best_val_loss:.4f}")
    logger.info("训练完成。")


# =============================================================================
# __main__：直接运行即启动预训练；也可 --config 覆盖超参数。
# 快速自测（跑通不训练全量）：
#   python src/train_pretrain.py --num_epochs 1 --fraction 0.02 --val_fraction 0.05
# =============================================================================
if __name__ == "__main__":
    main()
