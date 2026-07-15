# -*- coding: utf-8 -*-
"""
finetune_full.py —— 下游任务全量微调（Day 4，作为 LoRA 对比的基线）

论文对应：
    [1] Devlin et al., 2018, "BERT" 3.5 节 "Fine-tuning BERT"：
        在预训练好的 BERT 之上接一个任务相关的输出层，
        对句子级分类任务，取 [CLS] 对应的聚合表示 C ∈ R^H，
        接 Linear(H, num_labels) 做 softmax 分类，微调时所有参数一起更新。
    [2] Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language Models"：
        本脚本产出的"全量微调"结果将作为 LoRA（Day 5）参数量/效果对比的基线。

设计原则（严格遵守项目约束）：
    - 复用自写的 bert_model.py（BertModel）与 tokenizer.py。
    - 不使用 transformers / peft / 现成分类模型。
    - 加载 Day 3 的预训练 checkpoint（checkpoints/bert_pretrain.pt）。
    - 全量微调：BertModel 所有参数 requires_grad=True。
    - 保持最小可运行（MVP）：SST-2 小规模子集，GPU 可跑通。

本脚本实现：
    - 加载预训练 checkpoint，重建 BertModel 并载入骨干权重。
    - 在 BertModel 之上接分类头 Linear(hidden_size -> num_classes=2)。
    - 数据：datasets 加载 SST-2，训练集前 500 条，验证集前 100 条。
    - 训练循环：AdamW(lr=2e-5)，记录 train_loss / val_acc / 每 epoch 时间。
    - 保存最佳模型（按 val_acc）。
    - 统计总参数量 / 可训练参数量 / 每 epoch 时间 / 最终 val_acc。
    - 结果写入 results/full_finetune.json。

运行：
    python src/finetune_full.py
    python src/finetune_full.py --config config.json
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# 允许以脚本方式（python src/finetune_full.py）或包方式导入自写模块
try:
    from .bert_model import BertConfig, BertModel
    from .tokenizer import WordPieceTokenizer
except ImportError:
    from bert_model import BertConfig, BertModel
    from tokenizer import WordPieceTokenizer


# =============================================================================
# 配置加载：支持分段结构 {shared, pretrain, finetune}，也兼容旧的扁平结构
# 分段模式：返回 {**shared, **<section>}（本节覆盖 shared 同名键）。
# 扁平模式：直接返回整份 config（向后兼容旧写法）。
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
# BertForSequenceClassification：BertModel + 分类头
# 论文对应：BERT 3.5 节 —— 取 [CLS] 聚合表示（pooled_output，经 Pooler 的
#           Linear+Tanh）后接 Dropout + Linear(H, num_classes)。
# 全量微调：骨干与分类头所有参数都参与训练。
# =============================================================================
class BertForSequenceClassification(nn.Module):
    def __init__(self, config: BertConfig, num_classes: int = 2, dropout_prob: float = 0.1):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.bert = BertModel(config)                       # 预训练骨干
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_classes)  # 分类头
        # 分类头单独初始化（骨干权重稍后由 checkpoint 覆盖）
        self.classifier.apply(self.bert._init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # 取 pooled_output（[CLS] 经 Pooler 后的句向量）做分类
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)             # [B, num_classes]

        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels.view(-1))
            out["loss"] = loss
        return out


# =============================================================================
# checkpoint 加载：重建 BertConfig，并把预训练骨干权重载入 BertModel
# 兼容 Day 3 train_pretrain.py 的保存格式：
#   {"model_config": {...}, "bert_state_dict": {...}, ...}
# =============================================================================
def load_pretrained_bert(model: BertForSequenceClassification, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    bert_state = ckpt["bert_state_dict"]
    missing, unexpected = model.bert.load_state_dict(bert_state, strict=False)
    return ckpt, missing, unexpected


def build_config_from_ckpt(ckpt_path: str, device) -> BertConfig:
    ckpt = torch.load(ckpt_path, map_location=device)
    mc = ckpt["model_config"]
    return BertConfig(
        vocab_size=mc["vocab_size"],
        hidden_size=mc["hidden_size"],
        num_layers=mc["num_layers"],
        num_heads=mc["num_heads"],
        intermediate_size=mc["intermediate_size"],
        max_seq_len=mc["max_seq_len"],
        type_vocab_size=mc.get("type_vocab_size", 2),
        pad_token_id=mc.get("pad_token_id", 0),
    )


# =============================================================================
# SST-2 数据集（情感二分类）
# 用 datasets 加载 SST-2（优先 stanfordnlp/sst2 的 parquet 版本，列为
# idx / sentence / label，splits: train / validation / test）。
# 文本用自写 tokenizer 编码为 [CLS] A [SEP]。
#
# 【规则约束】下载失败时严禁静默使用假数据：默认直接抛出清晰错误并给出官方
# 下载地址；仅当显式传入 allow_fake_data=True 时才回退到内置小样例（用于离线
# 冒烟测试）。
# =============================================================================
# 候选数据源：(path, name, sentence_key)。按顺序尝试，命中即用。
_SST2_SOURCES = [
    ("stanfordnlp/sst2", None, "sentence"),   # 官方 parquet 版
    ("nyu-mll/glue", "sst2", "sentence"),      # GLUE 版
    ("glue", "sst2", "sentence"),              # 旧别名（可能已弃用）
    ("SetFit/sst2", None, "text"),             # 备选（列名为 text）
]

_SST2_DOWNLOAD_HELP = (
    "SST-2 数据下载失败。请检查网络或按以下方式手动准备数据：\n"
    "  官方 parquet 版:  https://huggingface.co/datasets/stanfordnlp/sst2\n"
    "  GLUE 版(sst2 配置): https://huggingface.co/datasets/nyu-mll/glue\n"
    "  国内镜像可先设置环境变量后重试:\n"
    "    PowerShell:  $env:HF_ENDPOINT='https://hf-mirror.com'; python src/finetune_full.py --config config.json\n"
    "代码已就绪，数据到位后可直接重跑。\n"
    "（如仅需离线冒烟测试，可显式加 --allow_fake_data 使用内置假数据，但结果无意义。）"
)


def _fake_sst2(num_train: int, num_val: int):
    pos = [("this movie is a wonderful and touching masterpiece .", 1)]
    neg = [("a boring , dull and utterly forgettable film .", 0)]
    data = (pos + neg) * max(1, (num_train // 2 + 1))
    return data[:num_train], data[:num_val]


def load_sst2(num_train: int, num_val: int, allow_fake_data: bool = False):
    from datasets import load_dataset

    last_err = None
    for path, name, key in _SST2_SOURCES:
        try:
            ds = load_dataset(path, name) if name else load_dataset(path)
            if "train" not in ds or "validation" not in ds:
                raise ValueError(f"数据集 {path} 缺少 train/validation split")
            n_tr = min(num_train, len(ds["train"]))
            n_va = min(num_val, len(ds["validation"]))
            train = [(r[key], int(r["label"])) for r in ds["train"].select(range(n_tr))]
            val = [(r[key], int(r["label"])) for r in ds["validation"].select(range(n_va))]
            print(f"[数据] SST-2 加载成功（来源 {path}{'/' + name if name else ''}）："
                  f"train={len(train)}, val={len(val)}")
            return train, val
        except Exception as e:
            last_err = e
            print(f"[数据] 尝试来源 {path}{'/' + name if name else ''} 失败："
                  f"{type(e).__name__}: {str(e)[:120]}")

    # 所有来源都失败
    if allow_fake_data:
        print("[数据][警告] 所有 SST-2 来源下载失败，已按 --allow_fake_data 使用内置假数据"
              "（结果无意义，仅供离线冒烟测试）。")
        return _fake_sst2(num_train, num_val)

    raise RuntimeError(
        f"{_SST2_DOWNLOAD_HELP}\n最后一次错误：{type(last_err).__name__}: {last_err}"
    )


class SST2Dataset(Dataset):
    def __init__(self, samples, tokenizer: WordPieceTokenizer, max_seq_len: int = 128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        # [CLS] A [SEP]，截断到 max_seq_len
        input_ids, token_type_ids = self.tokenizer.encode(
            text, add_special_tokens=True, max_len=self.max_seq_len)
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)

        def pad(seq, value):
            return torch.cat([seq, seq.new_full((max_len - len(seq),), value)])

        input_ids, token_type_ids, attention_mask, labels = [], [], [], []
        for item in batch:
            input_ids.append(pad(item["input_ids"], pad_token_id))
            token_type_ids.append(pad(item["token_type_ids"], 0))
            attention_mask.append(pad(item["attention_mask"], 0))
            labels.append(item["label"])
        return {
            "input_ids": torch.stack(input_ids),
            "token_type_ids": torch.stack(token_type_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

    return collate_fn


# =============================================================================
# 验证：在 val loader 上计算准确率
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, token_type_ids, attention_mask)["logits"]
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(1, total)


# =============================================================================
# 训练一个 epoch：全量微调（所有参数 requires_grad=True）
# =============================================================================
def train_one_epoch(model, loader, optimizer, device, max_norm=1.0):
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids, token_type_ids, attention_mask, labels=labels)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def parse_args():
    parser = argparse.ArgumentParser(description="下游任务全量微调（Day 4）")
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
    parser.add_argument("--vocab", type=str, default="data/vocab.txt")
    parser.add_argument("--pretrained_ckpt", type=str, default="checkpoints/bert_pretrain.pt")

    # 数据
    parser.add_argument("--num_train", type=int, default=500)
    parser.add_argument("--num_val", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=2)

    # 优化
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # 设备与产出
    parser.add_argument("--allow_cpu", action="store_true", help="显式允许 CPU 运行（默认强制 GPU）")
    parser.add_argument("--allow_fake_data", action="store_true",
                        help="显式允许在 SST-2 下载失败时使用内置假数据（默认禁止，仅供离线冒烟测试）")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/bert_finetune_full.pt")
    parser.add_argument("--result_path", type=str, default="results/full_finetune.json")

    args = parser.parse_args()

    # --config 覆盖：支持分段结构 {shared, pretrain, finetune}，
    # 本脚本只合并 shared + finetune，避免 pretrain 的 lr/ckpt_path 污染 finetune。
    # 向后兼容：若 config 仍是旧的扁平结构，则按同名键直接覆盖。
    if args.config and os.path.exists(args.config):
        merged = load_config_section(args.config, "finetune")
        for k, v in merged.items():
            if hasattr(args, k):
                setattr(args, k, v)
    return args


def select_device(allow_cpu: bool) -> torch.device:
    # 设备选择（GPU 优先，最高优先级约束，见 .cursor/rules/gpu-first.mdc）
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif allow_cpu:
        print("[警告] 未检测到可用 GPU，已按 --allow_cpu 使用 CPU 运行（速度会很慢）。")
        return torch.device("cpu")
    else:
        raise RuntimeError(
            "未检测到可用 GPU，当前环境 torch.cuda.is_available()=False。"
            "所有实验默认必须用 GPU。"
            "请检查是否安装了 CUDA 版 torch（见 requirements.txt），"
            "或显式加 --allow_cpu 才能用 CPU 运行。"
        )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.allow_cpu)
    print(f"[设备] {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ---------- 1) tokenizer ----------
    if not os.path.exists(args.vocab):
        raise FileNotFoundError(f"未找到词表 {args.vocab}，请先运行：python src/tokenizer.py")
    tokenizer = WordPieceTokenizer.load(args.vocab)
    print(f"[词表] 已加载 {args.vocab}, vocab_size = {tokenizer.vocab_size}")

    # ---------- 2) 依据预训练 checkpoint 重建配置 + 模型 ----------
    if not os.path.exists(args.pretrained_ckpt):
        raise FileNotFoundError(
            f"未找到预训练 checkpoint {args.pretrained_ckpt}，请先运行 Day 3：python src/train_pretrain.py")
    config = build_config_from_ckpt(args.pretrained_ckpt, device)
    print(f"[模型] 配置（来自预训练 ckpt）: {config}")

    model = BertForSequenceClassification(config, num_classes=args.num_classes).to(device)
    _, missing, unexpected = load_pretrained_bert(model, args.pretrained_ckpt, device)
    print(f"[加载] 已载入预训练骨干权重；missing={len(missing)}, unexpected={len(unexpected)}")

    # 全量微调：确保所有参数参与训练
    for p in model.parameters():
        p.requires_grad = True
    total_params, trainable_params = count_params(model)
    print(f"[参数] 总参数量 = {total_params:,} | 可训练参数量 = {trainable_params:,}")

    # ---------- 3) SST-2 数据 ----------
    train_samples, val_samples = load_sst2(args.num_train, args.num_val,
                                           allow_fake_data=args.allow_fake_data)
    # 合理性校验：标签是否单一（假数据/异常的征兆）
    train_labels = {lb for _, lb in train_samples}
    val_labels = {lb for _, lb in val_samples}
    if len(train_labels) < 2 or len(val_labels) < 2:
        print(f"[警告] 数据标签不足两类（train={train_labels}, val={val_labels}），"
              f"可能是假数据或采样异常，val_acc 将不具参考意义。")
    collate = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        SST2Dataset(train_samples, tokenizer, args.max_seq_len),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(
        SST2Dataset(val_samples, tokenizer, args.max_seq_len),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # ---------- 4) 优化器 + 训练循环 ----------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    epoch_times = []
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
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
                    "vocab_size": config.vocab_size,
                    "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers,
                    "num_heads": config.num_heads,
                    "intermediate_size": config.intermediate_size,
                    "max_seq_len": config.max_seq_len,
                    "type_vocab_size": config.type_vocab_size,
                    "pad_token_id": config.pad_token_id,
                },
                "num_classes": args.num_classes,
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, args.ckpt_path)
            print(f"  -> 新最佳 val_acc={val_acc:.4f}，已保存 checkpoint: {args.ckpt_path}")

    avg_epoch_time = sum(epoch_times) / max(1, len(epoch_times))

    # ---------- 5) 保存结果 JSON ----------
    result = {
        "method": "full_finetune",
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_time_per_epoch": avg_epoch_time,
        "final_val_acc": best_val_acc,
        "config": {
            "task": "SST-2",
            "num_train": args.num_train,
            "num_val": args.num_val,
            "max_seq_len": args.max_seq_len,
            "batch_size": args.batch_size,
            "num_classes": args.num_classes,
            "num_epochs": args.num_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
            "pretrained_ckpt": args.pretrained_ckpt,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "intermediate_size": config.intermediate_size,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.result_path)), exist_ok=True)
    with open(args.result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n===== 全量微调结果 =====")
    print(f"  总参数量         : {total_params:,}")
    print(f"  可训练参数量     : {trainable_params:,}")
    print(f"  每 epoch 训练时间: {avg_epoch_time:.1f}s")
    print(f"  最终 val_acc     : {best_val_acc:.4f}")
    print(f"  结果已保存       : {args.result_path}")
    print(f"  微调 checkpoint  : {args.ckpt_path}")

    # 合理性校验：val_acc 恰好 = 1.0 往往是数据过小/单一或异常的征兆
    if abs(best_val_acc - 1.0) < 1e-9:
        print("[警告] val_acc 恰好等于 1.0，请确认使用的是真实 SST-2 数据、样本量足够，"
              "而非假数据或标签泄露。")


# =============================================================================
# __main__：直接运行即执行全量微调。
# 说明：也提供一个不依赖 checkpoint / 数据下载的最小前向自测思路——
#       但按任务要求，主流程即为完整微调，见 main()。
# 快速自测（小样本、少轮）：
#   python src/finetune_full.py --num_train 64 --num_val 32 --num_epochs 1
# =============================================================================
if __name__ == "__main__":
    main()
