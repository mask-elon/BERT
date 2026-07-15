# -*- coding: utf-8 -*-
"""
data.py —— WikiText-2 数据管线 + MLM 掩码 + NSP 句对（Day 2 / NSP 增补）

论文对应：
    [1] Devlin et al., 2018, "BERT"
        3.1 节 "Task #1: Masked LM"：随机遮盖 15% 的 WordPiece token，其中
            - 80% 替换为 [MASK]，10% 替换为随机 token，10% 保持不变；
            仅在被选中位置计损，其余标签置 -100。
        3.1 节 "Task #2: Next Sentence Prediction (NSP)"：构造句对 (A, B)，
            - 50% 情况 B 是 A 的真实下一段（is_next=0, IsNext）
            - 50% 情况 B 取自随机文档（is_next=1, NotNext）
            用 [CLS] 位置的表示做二分类。

设计原则（严格遵守项目约束）：
    - 允许用 `datasets` 库下载 WikiText-2，但文档切分 / 句子切分 / 句对采样 /
      掩码 / batch 构造全部自写。
    - 分词复用自写的 WordPieceTokenizer（src/tokenizer.py）。

样本字段：
    input_ids, token_type_ids, attention_mask, labels
    （NSP 开启时额外含 next_sentence_label）
      - token_type_ids：段 A（含首个 [SEP]）为 0，段 B（含末个 [SEP]）为 1
      - labels：被 mask 位保留原 token id，其余为 -100
      - next_sentence_label：0=IsNext，1=NotNext
"""

import argparse
import json
import os
import random
import re

import torch
from torch.utils.data import Dataset, DataLoader

# 允许以脚本方式（python src/data.py）或包方式导入 tokenizer
try:
    from .tokenizer import WordPieceTokenizer, SPECIAL_TOKENS
except ImportError:
    from tokenizer import WordPieceTokenizer, SPECIAL_TOKENS


# 句子切分：在 . ! ? 及其后空白处断句（简易规则，够用即可）
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str):
    """把一段文本切成句子列表（手写简易规则）。"""
    text = text.strip()
    if not text:
        return []
    sents = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sents if s.strip()]


# =============================================================================
# WikiText-2 语料加载
# =============================================================================
def _fallback_corpus():
    return [
        "It was the best of times, it was the worst of times. It was the age of wisdom.",
        "The quick brown fox jumps over the lazy dog. The dog did not care at all.",
        "Call me Ishmael. Some years ago never mind how long precisely I sailed.",
        "In the beginning God created the heavens and the earth. The earth was formless.",
        "All happy families are alike. Each unhappy family is unhappy in its own way.",
    ] * 200


def load_wikitext2(split: str = "train", fraction: float = 1.0):
    """返回 WikiText-2 指定 split 的非空正文行列表（用于 MLM-only 路径）。"""
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        n = max(1, int(len(ds) * fraction))
        texts = [t for t in ds[:n]["text"] if t and t.strip()]
        texts = [t for t in texts if not t.strip().startswith("=")]  # 过滤小节标题
        if texts:
            print(f"[语料] WikiText-2 {split} 前 {fraction:.0%} -> {len(texts)} 行正文")
            return texts
    except Exception as e:
        print(f"[语料] 下载 WikiText-2 失败（{type(e).__name__}），改用内置样例语料。")
    return _fallback_corpus()


def load_wikitext2_documents(split: str = "train", fraction: float = 1.0):
    """按“文档”组织 WikiText-2（用于 NSP）。
    WikiText 用形如 " = Title = "（一级）/ " = = Sub = = "（多级）的标题分节，
    这里把一级标题作为文档边界，将其后的正文段落聚合为一个文档字符串。
    返回 list[str]，每个元素是一篇文档的正文（可能含多句/多段）。
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        n = max(1, int(len(ds) * fraction))
        lines = ds[:n]["text"]

        docs, current = [], []

        def is_top_heading(s: str) -> bool:
            # 一级标题形如 " = Title = "（两侧恰好一个 =），多级为 " == .. == "
            s = s.strip()
            return s.startswith("= ") and not s.startswith("= =")

        for raw in lines:
            s = raw.strip()
            if not s:
                continue
            if is_top_heading(s):
                # 遇到新文档边界，先收尾旧文档
                if current:
                    docs.append(" ".join(current))
                    current = []
                continue
            if s.startswith("="):  # 各级子标题直接丢弃（噪声）
                continue
            current.append(s)
        if current:
            docs.append(" ".join(current))

        # 只保留有内容的文档
        docs = [d for d in docs if len(d.split()) >= 5]
        if docs:
            print(f"[语料] WikiText-2 {split} 前 {fraction:.0%} -> {len(docs)} 篇文档（NSP）")
            return docs
    except Exception as e:
        print(f"[语料] 下载 WikiText-2 失败（{type(e).__name__}），改用内置样例语料。")
    return _fallback_corpus()


# =============================================================================
# PretrainDataset：MLM（+ 可选 NSP）预训练数据集
# nsp=False：把 token 流切块，单段 [CLS] A [SEP]，仅 MLM。
# nsp=True ：按 BERT 算法构造句对 [CLS] A [SEP] B [SEP]，MLM + NSP。
# 掩码为动态掩码（每次取样重新采样，BERT/RoBERTa 常见做法）。
# =============================================================================
class PretrainDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer: WordPieceTokenizer,
        max_seq_len: int = 128,
        mlm_probability: float = 0.15,
        nsp: bool = False,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mlm_probability = mlm_probability
        self.nsp = nsp
        self.rng = random.Random(seed)

        # 随机替换候选：排除特殊 token
        self.special_ids = tokenizer.all_special_ids()
        self.normal_ids = [i for i in range(tokenizer.vocab_size) if i not in self.special_ids]

        # 统一存成 examples: list of (tokens_a, tokens_b_or_None, is_next_or_None)
        if nsp:
            self.examples = self._build_nsp_examples(data)   # data: list[str] 文档
        else:
            self.examples = self._build_mlm_blocks(data)     # data: list[str] 正文行

    # ---------- MLM-only：切块 ----------
    def _build_mlm_blocks(self, texts):
        block_len = self.max_seq_len - 2  # 预留 [CLS] / [SEP]
        buffer, examples = [], []
        for line in texts:
            ids, _ = self.tokenizer.encode(line, add_special_tokens=False)
            buffer.extend(ids)
            while len(buffer) >= block_len:
                examples.append((buffer[:block_len], None, None))
                buffer = buffer[block_len:]
        if len(buffer) >= 8:
            examples.append((buffer, None, None))
        print(f"[数据] MLM 切块得到 {len(examples)} 个训练样本（block_len<= {block_len}）")
        return examples

    # ---------- NSP：按 BERT 算法构造句对 ----------
    def _build_nsp_examples(self, documents):
        # 先把每篇文档切句并分词成 token-id 句子列表
        docs = []
        for d in documents:
            sents = []
            for s in split_sentences(d):
                ids, _ = self.tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    sents.append(ids)
            if sents:
                docs.append(sents)
        if not docs:
            return []

        target_len = self.max_seq_len - 3  # 预留 [CLS] A [SEP] B [SEP]
        examples = []

        for di, doc in enumerate(docs):
            i = 0
            current_chunk = []   # 累积的句子（token-id 列表）
            current_len = 0
            while i < len(doc):
                current_chunk.append(doc[i])
                current_len += len(doc[i])
                # 到文档末尾，或累计长度够了，就产出一个实例
                if i == len(doc) - 1 or current_len >= target_len:
                    if current_chunk:
                        # A/B 的切分点：至少留 1 句给 A
                        a_end = 1
                        if len(current_chunk) >= 2:
                            a_end = self.rng.randint(1, len(current_chunk) - 1)
                        tokens_a = [t for s in current_chunk[:a_end] for t in s]

                        tokens_b = []
                        # 50% 取随机文档做 NotNext；单句 chunk 只能 NotNext
                        if len(current_chunk) == 1 or self.rng.random() < 0.5:
                            is_next = 1  # NotNext
                            target_b = max(1, target_len - len(tokens_a))
                            rand_di = di
                            for _ in range(10):
                                rand_di = self.rng.randint(0, len(docs) - 1)
                                if rand_di != di:
                                    break
                            rand_doc = docs[rand_di]
                            r_start = self.rng.randint(0, len(rand_doc) - 1)
                            for j in range(r_start, len(rand_doc)):
                                tokens_b.extend(rand_doc[j])
                                if len(tokens_b) >= target_b:
                                    break
                            # 未用于 A 的句子放回，供下个实例复用（BERT 原逻辑）
                            i -= (len(current_chunk) - a_end)
                        else:
                            is_next = 0  # IsNext
                            tokens_b = [t for s in current_chunk[a_end:] for t in s]

                        self._truncate_pair(tokens_a, tokens_b, target_len)
                        if tokens_a and tokens_b:
                            examples.append((tokens_a, tokens_b, is_next))
                    current_chunk = []
                    current_len = 0
                i += 1

        n_is = sum(1 for e in examples if e[2] == 0)
        print(f"[数据] NSP 句对共 {len(examples)} 条：IsNext={n_is}, "
              f"NotNext={len(examples) - n_is}")
        return examples

    @staticmethod
    def _truncate_pair(tokens_a, tokens_b, max_total):
        # 从较长一段的尾部逐个删，直到 A+B 不超过 max_total（原地修改）
        while len(tokens_a) + len(tokens_b) > max_total:
            if len(tokens_a) >= len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def __len__(self):
        return len(self.examples)

    # ---------- MLM 掩码（对非特殊位施加 80/10/10） ----------
    def _apply_mlm_mask(self, input_ids, protected_positions):
        input_ids = list(input_ids)
        labels = [-100] * len(input_ids)
        for pos in range(len(input_ids)):
            if pos in protected_positions:
                continue
            if self.rng.random() < self.mlm_probability:
                labels[pos] = input_ids[pos]           # 记录原 token 作为预测目标
                r = self.rng.random()
                if r < 0.8:
                    input_ids[pos] = self.tokenizer.mask_token_id        # 80% [MASK]
                elif r < 0.9:
                    input_ids[pos] = self.rng.choice(self.normal_ids)    # 10% 随机
                # 其余 10% 保持不变
        return input_ids, labels

    def __getitem__(self, idx):
        tokens_a, tokens_b, is_next = self.examples[idx]
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        if tokens_b is None:
            # 单段（MLM-only）
            input_ids = [cls_id] + list(tokens_a) + [sep_id]
            token_type_ids = [0] * len(input_ids)
            protected = {0, len(input_ids) - 1}  # [CLS] 与 [SEP] 不参与掩码
        else:
            # 句对（NSP）：[CLS] A [SEP] B [SEP]
            input_ids = [cls_id] + list(tokens_a) + [sep_id] + list(tokens_b) + [sep_id]
            sep_a = 1 + len(tokens_a)            # 第一个 [SEP] 下标
            token_type_ids = [0] * (sep_a + 1) + [1] * (len(tokens_b) + 1)
            protected = {0, sep_a, len(input_ids) - 1}

        input_ids, labels = self._apply_mlm_mask(input_ids, protected)
        attention_mask = [1] * len(input_ids)

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if self.nsp:
            item["next_sentence_label"] = torch.tensor(is_next, dtype=torch.long)
        return item


# 向后兼容别名（Day 2 早前代码引用 MLMDataset）
MLMDataset = PretrainDataset


# =============================================================================
# collate_fn：动态 padding 到 batch 内最长长度
# input_ids←[PAD]，labels←-100，token_type_ids←0，attention_mask←0；
# 若样本含 next_sentence_label（NSP），堆叠为 [B]。
# =============================================================================
def make_collate_fn(pad_token_id: int):
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)

        def pad(seq, value):
            return torch.cat([seq, seq.new_full((max_len - len(seq),), value)])

        out = {
            "input_ids": [], "token_type_ids": [], "attention_mask": [], "labels": [],
        }
        has_nsp = "next_sentence_label" in batch[0]
        nsp_labels = []
        for item in batch:
            out["input_ids"].append(pad(item["input_ids"], pad_token_id))
            out["token_type_ids"].append(pad(item["token_type_ids"], 0))
            out["attention_mask"].append(pad(item["attention_mask"], 0))
            out["labels"].append(pad(item["labels"], -100))
            if has_nsp:
                nsp_labels.append(item["next_sentence_label"])

        result = {k: torch.stack(v) for k, v in out.items()}
        if has_nsp:
            result["next_sentence_label"] = torch.stack(nsp_labels)
        return result

    return collate_fn


def build_dataloader(
    tokenizer: WordPieceTokenizer,
    split: str = "train",
    fraction: float = 0.1,
    max_seq_len: int = 128,
    batch_size: int = 8,
    mlm_probability: float = 0.15,
    nsp: bool = False,
    shuffle: bool = True,
):
    """一站式构建 DataLoader。nsp=True 时按文档构造句对。"""
    if nsp:
        data = load_wikitext2_documents(split=split, fraction=fraction)
    else:
        data = load_wikitext2(split=split, fraction=fraction)
    dataset = PretrainDataset(
        data, tokenizer,
        max_seq_len=max_seq_len,
        mlm_probability=mlm_probability,
        nsp=nsp,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
    )
    return dataset, loader


# =============================================================================
# __main__：构建一个 batch 并验证 MLM 掩码 + NSP 句对
# 检查项（对应用户规则“实现清单”）：
#   - import 不报错；打印一个 batch 的 masked 句对输入
#   - token_type_ids 同时含 0 和 1（句对结构正确）
#   - next_sentence_label 分布接近 50/50
#   - MLM 掩码比例 ≈ 15%，labels 正确
# 运行：python src/data.py                （默认开启 NSP）
#       python src/data.py --no_nsp       （仅 MLM）
#       python src/data.py --config config.json
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WikiText-2 MLM + NSP 数据管线自测")
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
    parser.add_argument("--vocab", type=str, default="data/vocab.txt", help="词表路径")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--no_nsp", action="store_true", help="仅测试 MLM，不构造句对")
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    use_nsp = not args.no_nsp

    if not os.path.exists(args.vocab):
        raise FileNotFoundError(f"未找到词表 {args.vocab}，请先运行：python src/tokenizer.py")

    tokenizer = WordPieceTokenizer.load(args.vocab)
    print(f"[词表] 已加载 {args.vocab}，vocab_size = {tokenizer.vocab_size}")
    print(f"[模式] NSP = {use_nsp}")

    dataset, loader = build_dataloader(
        tokenizer, split="train", fraction=args.fraction,
        max_seq_len=args.max_seq_len, batch_size=args.batch_size,
        mlm_probability=args.mlm_probability, nsp=use_nsp,
    )

    batch = next(iter(loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    token_type_ids = batch["token_type_ids"]

    print("\n=== 一个 batch 的形状 ===")
    for k, v in batch.items():
        print(f"  {k:20s}: {tuple(v.shape)}  dtype={v.dtype}")

    # ---- 样本 0：masked 句对输入 ----
    sample_in = input_ids[0].tolist()
    sample_tt = token_type_ids[0].tolist()
    print("\n=== 样本 0：masked 输入（token 形式）===")
    print(" ".join(tokenizer.convert_ids_to_tokens(sample_in)))
    if use_nsp:
        seg_a = sum(1 for t in sample_tt if t == 0)
        print(f"\n段 A(token_type=0) 长度: {seg_a}，段 B(token_type=1) 长度: {len(sample_tt) - seg_a}")
        print(f"样本 0 的 next_sentence_label = {batch['next_sentence_label'][0].item()} "
              f"(0=IsNext, 1=NotNext)")
        assert set(sample_tt) <= {0, 1} and 1 in sample_tt, "句对样本 token_type 应同时含 0 和 1"

    # ---- MLM 掩码比例统计 ----
    n_special_per_seq = 3 if use_nsp else 2  # 句对含 2 个 [SEP]+1 个 [CLS]
    valid_positions = int((attention_mask == 1).sum().item()) - n_special_per_seq * input_ids.size(0)
    num_masked = int((labels != -100).sum().item())
    ratio = num_masked / max(1, valid_positions)
    print("\n=== MLM 掩码比例（整个 batch）===")
    print(f"  可掩码正文 token 数: {valid_positions}")
    print(f"  实际被选中 token 数: {num_masked}")
    print(f"  实际掩码比例       : {ratio:.2%}  (目标 ≈ {args.mlm_probability:.0%})")

    # ---- 80/10/10 拆分 ----
    mask_id = tokenizer.mask_token_id
    n_mask = n_keep = n_rand = 0
    for inp, lb in zip(input_ids.view(-1).tolist(), labels.view(-1).tolist()):
        if lb == -100:
            continue
        if inp == mask_id:
            n_mask += 1
        elif inp == lb:
            n_keep += 1
        else:
            n_rand += 1
    tot = max(1, num_masked)
    print("\n=== 被选中 token 的 80/10/10 拆分 ===")
    print(f"  -> [MASK]     : {n_mask:4d}  ({n_mask / tot:.1%}, 目标 80%)")
    print(f"  -> 随机替换   : {n_rand:4d}  ({n_rand / tot:.1%}, 目标 10%)")
    print(f"  -> 保持不变   : {n_keep:4d}  ({n_keep / tot:.1%}, 目标 10%)")

    # ---- NSP 标签整体分布（遍历整个数据集，验证接近 50/50）----
    if use_nsp:
        n_is = sum(1 for e in dataset.examples if e[2] == 0)
        n_not = len(dataset.examples) - n_is
        print("\n=== NSP 标签分布（整个数据集）===")
        print(f"  IsNext (0) : {n_is}  ({n_is / max(1, len(dataset)):.1%})")
        print(f"  NotNext(1) : {n_not} ({n_not / max(1, len(dataset)):.1%})")
        assert 0.30 <= n_is / max(1, len(dataset)) <= 0.70, "NSP 标签应大致均衡（~50/50）"

    assert 0.05 <= ratio <= 0.30, f"掩码比例 {ratio:.2%} 偏离 15% 过多"
    print("\n[OK] 数据管线（MLM" + ("+NSP" if use_nsp else "") + "）断言全部通过。")
