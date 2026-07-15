# -*- coding: utf-8 -*-
"""
tokenizer.py —— 从零手写 WordPiece 分词器（Day 2）

论文对应：
    [1] Devlin et al., 2018, "BERT: Pre-training of Deep Bidirectional
        Transformers for Language Understanding" (https://arxiv.org/abs/1810.04805)
        3.2 节 "Input/Output Representations"：使用 WordPiece 词表 + [CLS]/[SEP] 等特殊 token。
    [2] Wu et al., 2016, "Google's Neural Machine Translation System"
        （WordPiece 分词的来源）。

设计原则（严格遵守项目约束）：
    - 不使用 transformers 的 BertTokenizer 等现成分词类。
    - 允许用 `tokenizers` 库“训练”出 WordPiece 词表（只借用它的 BPE/WordPiece
      词表学习算法），但 encode / decode / 子词切分逻辑全部自己手写。
    - 特殊 token：[PAD] [UNK] [CLS] [SEP] [MASK]，固定 id 0~4，[PAD]=0 与
      bert_model.BertConfig.pad_token_id 默认值一致。

encode / decode 手写流程（对照 BERT 官方 tokenization.py 思路，但代码自写）：
    text
      └─ 基础分词 BasicTokenizer：小写化 + 去重音(可选) + 按空白切分 + 标点独立成词
           └─ 子词分词 WordPiece：贪心最长匹配（longest-match-first），
              续接子词加 "##" 前缀，无法匹配则整词记为 [UNK]
                └─ convert_tokens_to_ids -> input_ids
    decode 为其逆过程：id -> token -> 跳过特殊token -> 去掉 "##" 拼接 -> 复原空格。
"""

import argparse
import json
import os
import unicodedata

# 特殊 token 固定顺序：id 与列表下标一致（[PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4）
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]


# =============================================================================
# BasicTokenizer：基础分词（手写）
# 对应 BERT tokenization.py 中的 BasicTokenizer。
# 步骤：Unicode 规范化 -> 去除控制字符 -> 小写(+去重音) -> 空白切分 -> 标点独立。
# =============================================================================
class BasicTokenizer:
    def __init__(self, do_lower_case: bool = True):
        self.do_lower_case = do_lower_case

    @staticmethod
    def _is_whitespace(ch: str) -> bool:
        # 空格 / 制表 / 换行 / 回车，以及 Unicode 空白类
        if ch in (" ", "\t", "\n", "\r"):
            return True
        return unicodedata.category(ch) == "Zs"

    @staticmethod
    def _is_control(ch: str) -> bool:
        if ch in ("\t", "\n", "\r"):
            return False
        return unicodedata.category(ch).startswith("C")

    @staticmethod
    def _is_punctuation(ch: str) -> bool:
        cp = ord(ch)
        # ASCII 中非字母数字的可见字符视为标点（与 BERT 一致）
        if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
            return True
        return unicodedata.category(ch).startswith("P")

    def _clean_text(self, text: str) -> str:
        # 去除无效字符与控制字符，空白统一为普通空格
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or self._is_control(ch):
                continue
            if self._is_whitespace(ch):
                out.append(" ")
            else:
                out.append(ch)
        return "".join(out)

    def _run_strip_accents(self, text: str) -> str:
        # NFD 分解后丢弃组合记号 (Mn) —— 去重音
        text = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    def _split_on_punc(self, token: str) -> list:
        # 把标点切成独立 token，例如 "end." -> ["end", "."]
        chars = list(token)
        out, cur = [], []
        for ch in chars:
            if self._is_punctuation(ch):
                if cur:
                    out.append("".join(cur))
                    cur = []
                out.append(ch)
            else:
                cur.append(ch)
        if cur:
            out.append("".join(cur))
        return out

    def tokenize(self, text: str) -> list:
        text = self._clean_text(text)
        # 先按空白粗切
        raw_tokens = text.strip().split()
        split_tokens = []
        for tok in raw_tokens:
            if self.do_lower_case:
                tok = tok.lower()
                tok = self._run_strip_accents(tok)
            split_tokens.extend(self._split_on_punc(tok))
        return split_tokens


# =============================================================================
# WordpieceTokenizer：子词切分（手写贪心最长匹配）
# 对应 BERT tokenization.py 中的 WordpieceTokenizer。
# 算法：对每个词，从最长前缀开始在词表中查找；命中则该子串成为一个子词
#       （非首子词加 "##" 前缀），从命中末尾继续；若某位置找不到任何子词，
#       整个词记为 [UNK]。
# =============================================================================
class WordpieceTokenizer:
    def __init__(self, vocab: dict, unk_token: str = UNK_TOKEN, max_chars_per_word: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_chars_per_word = max_chars_per_word

    def tokenize(self, word: str) -> list:
        if len(word) > self.max_chars_per_word:
            return [self.unk_token]

        chars = list(word)
        sub_tokens = []
        start = 0
        is_bad = False
        while start < len(chars):
            end = len(chars)
            cur_sub = None
            # 贪心最长匹配：从最长子串向短收缩
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    cur_sub = substr
                    break
                end -= 1
            if cur_sub is None:
                is_bad = True
                break
            sub_tokens.append(cur_sub)
            start = end
        if is_bad:
            return [self.unk_token]
        return sub_tokens


# =============================================================================
# WordPieceTokenizer：对外主类
# 组合 BasicTokenizer + WordpieceTokenizer，提供 train / save / load /
# encode / decode / vocab_size 等接口。
# =============================================================================
class WordPieceTokenizer:
    def __init__(self, vocab: dict = None, do_lower_case: bool = True):
        # vocab: {token(str): id(int)}
        self.do_lower_case = do_lower_case
        self.basic = BasicTokenizer(do_lower_case=do_lower_case)
        self.vocab = vocab if vocab is not None else {}
        self.ids_to_tokens = {i: t for t, i in self.vocab.items()}
        self._wordpiece = WordpieceTokenizer(self.vocab) if self.vocab else None

    # ---------- 词表相关 ----------
    def _rebuild_index(self):
        self.ids_to_tokens = {i: t for t, i in self.vocab.items()}
        self._wordpiece = WordpieceTokenizer(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[UNK_TOKEN])

    def id_to_token(self, idx: int) -> str:
        return self.ids_to_tokens.get(idx, UNK_TOKEN)

    # 常用特殊 token id 便捷属性
    @property
    def pad_token_id(self):
        return self.vocab[PAD_TOKEN]

    @property
    def unk_token_id(self):
        return self.vocab[UNK_TOKEN]

    @property
    def cls_token_id(self):
        return self.vocab[CLS_TOKEN]

    @property
    def sep_token_id(self):
        return self.vocab[SEP_TOKEN]

    @property
    def mask_token_id(self):
        return self.vocab[MASK_TOKEN]

    def all_special_ids(self) -> set:
        return {self.vocab[t] for t in SPECIAL_TOKENS}

    # ---------- 训练（仅借用 tokenizers 库学习词表） ----------
    def train(self, corpus_iterable, vocab_size: int = 8000, min_frequency: int = 2):
        """用 `tokenizers` 库的 WordPieceTrainer 学习词表（只借词表学习算法）。
        corpus_iterable：可迭代的字符串（每条为一行/一段文本）。
        学习完成后只保留 {token: id} 词表，后续 encode/decode 全部走自写逻辑。
        """
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.trainers import WordPieceTrainer
        from tokenizers.pre_tokenizers import BertPreTokenizer
        from tokenizers.normalizers import BertNormalizer

        tk = Tokenizer(WordPiece(unk_token=UNK_TOKEN))
        # 归一化与预分词与我们手写的 BasicTokenizer 对齐（小写、去重音、标点独立）
        tk.normalizer = BertNormalizer(lowercase=self.do_lower_case)
        tk.pre_tokenizer = BertPreTokenizer()

        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,  # 顺序决定 id：[PAD]=0 ... [MASK]=4
            continuing_subword_prefix="##",
        )
        tk.train_from_iterator(corpus_iterable, trainer=trainer)

        learned = tk.get_vocab()  # {token: id}
        # 强制特殊 token 占据 0~4，其余 token 顺延重排，保证 id 稳定可控
        self.vocab = {}
        for i, t in enumerate(SPECIAL_TOKENS):
            self.vocab[t] = i
        next_id = len(SPECIAL_TOKENS)
        for token in sorted(learned, key=lambda x: learned[x]):
            if token in self.vocab:
                continue
            self.vocab[token] = next_id
            next_id += 1
        self._rebuild_index()
        return self

    def save(self, path: str):
        """保存为 vocab.txt（每行一个 token，行号即 id）。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ordered = sorted(self.vocab.items(), key=lambda kv: kv[1])
        with open(path, "w", encoding="utf-8") as f:
            for token, _ in ordered:
                f.write(token + "\n")

    @classmethod
    def load(cls, path: str, do_lower_case: bool = True):
        vocab = {}
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.rstrip("\n")
                vocab[token] = idx
        return cls(vocab=vocab, do_lower_case=do_lower_case)

    # ---------- 分词 / 编解码（全手写） ----------
    def tokenize(self, text: str) -> list:
        assert self._wordpiece is not None, "词表为空，请先 train 或 load。"
        tokens = []
        for word in self.basic.tokenize(text):
            tokens.extend(self._wordpiece.tokenize(word))
        return tokens

    def convert_tokens_to_ids(self, tokens) -> list:
        return [self.token_to_id(t) for t in tokens]

    def convert_ids_to_tokens(self, ids) -> list:
        return [self.id_to_token(int(i)) for i in ids]

    def encode(self, text: str, text_pair: str = None,
               add_special_tokens: bool = True, max_len: int = None):
        """text -> input_ids（可选句子对，构造 token_type_ids）。
        单句：[CLS] A [SEP]
        句对：[CLS] A [SEP] B [SEP]
        返回 (input_ids, token_type_ids)。
        """
        tokens_a = self.tokenize(text)
        tokens_b = self.tokenize(text_pair) if text_pair is not None else None

        if add_special_tokens:
            if tokens_b is None:
                # 预留 [CLS] 和 [SEP] 两个位置
                if max_len is not None:
                    tokens_a = tokens_a[: max_len - 2]
                tokens = [CLS_TOKEN] + tokens_a + [SEP_TOKEN]
                token_type_ids = [0] * len(tokens)
            else:
                # 预留 [CLS] A [SEP] B [SEP] 三个特殊位
                if max_len is not None:
                    self._truncate_pair(tokens_a, tokens_b, max_len - 3)
                tokens = [CLS_TOKEN] + tokens_a + [SEP_TOKEN] + tokens_b + [SEP_TOKEN]
                token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        else:
            tokens = tokens_a + (tokens_b if tokens_b else [])
            if max_len is not None:
                tokens = tokens[:max_len]
            token_type_ids = [0] * len(tokens)

        input_ids = self.convert_tokens_to_ids(tokens)
        return input_ids, token_type_ids

    @staticmethod
    def _truncate_pair(tokens_a: list, tokens_b: list, max_tokens: int):
        # 交替从较长序列尾部截断，直到总长不超过 max_tokens（原地修改）
        while len(tokens_a) + len(tokens_b) > max_tokens:
            if len(tokens_a) >= len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        """input_ids -> text（去 "##" 拼接，可选跳过特殊 token）。"""
        special = set(SPECIAL_TOKENS)
        tokens = self.convert_ids_to_tokens(ids)
        words = []
        for tok in tokens:
            if skip_special_tokens and tok in special:
                continue
            if tok.startswith("##"):
                if words:
                    words[-1] = words[-1] + tok[2:]
                else:
                    words.append(tok[2:])
            else:
                words.append(tok)
        return " ".join(words)


# =============================================================================
# 语料加载（用于 __main__ 自训练；WikiText-2 前 10% 作为小语料）
# 允许通过 datasets 下载；失败则回退到内置英文小说片段，保证脚本可独立运行。
# =============================================================================
def _load_training_corpus(fraction: float = 0.1):
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        n = max(1, int(len(ds) * fraction))
        texts = [t for t in ds[:n]["text"] if t and t.strip()]
        if texts:
            print(f"[语料] WikiText-2 train 前 {fraction:.0%} -> {len(texts)} 行非空文本")
            return texts
    except Exception as e:  # 网络不可用等情况回退
        print(f"[语料] 下载 WikiText-2 失败（{type(e).__name__}），改用内置样例语料。")

    fallback = [
        "It was the best of times, it was the worst of times.",
        "The quick brown fox jumps over the lazy dog.",
        "Call me Ishmael. Some years ago never mind how long precisely.",
        "In the beginning God created the heavens and the earth.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
    ] * 200
    return fallback


# =============================================================================
# __main__：训练一个小型 tokenizer 并做编解码自测
# 检查项（对应用户规则“实现清单”）：
#   - import 不报错
#   - encode(text) -> ids，decode(ids) -> text 往返合理
#   - vocab_size 落在 3000~8000
#   - 特殊 token id 固定为 0~4
# 运行：python src/tokenizer.py            （默认 vocab_size=8000）
#       python src/tokenizer.py --config config.json
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练/测试手写 WordPiece 分词器")
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--fraction", type=float, default=0.1, help="WikiText-2 训练集采样比例")
    parser.add_argument("--out", type=str, default="data/vocab.txt", help="词表保存路径")
    args = parser.parse_args()

    # 支持 --config 覆盖命令行默认值
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    print(f"目标词表大小 vocab_size = {args.vocab_size}")
    corpus = _load_training_corpus(args.fraction)

    tokenizer = WordPieceTokenizer(do_lower_case=True)
    tokenizer.train(corpus, vocab_size=args.vocab_size, min_frequency=2)
    tokenizer.save(args.out)
    print(f"[OK] 词表已保存到 {args.out}，实际 vocab_size = {tokenizer.vocab_size}")

    # 特殊 token id 断言
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.cls_token_id == 2
    assert tokenizer.sep_token_id == 3
    assert tokenizer.mask_token_id == 4
    print("特殊 token id: [PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4  [OK]")

    # 编解码往返测试
    sample = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer.tokenize(sample)
    input_ids, type_ids = tokenizer.encode(sample)
    text_back = tokenizer.decode(input_ids)

    print("\n原文       :", sample)
    print("子词切分   :", tokens)
    print("input_ids  :", input_ids)
    print("type_ids   :", type_ids)
    print("decode 还原:", text_back)

    # 句子对编码测试（NSP 场景会用到）
    ids_pair, types_pair = tokenizer.encode("hello world", "how are you")
    print("\n句对 input_ids   :", ids_pair)
    print("句对 token_type  :", types_pair)
    assert set(types_pair) == {0, 1}, "句对 token_type_ids 应同时含 0 和 1"

    assert 3000 <= tokenizer.vocab_size <= 8000, "建议 vocab_size 在 3000~8000"
    assert len(input_ids) == len(type_ids)
    assert input_ids[0] == tokenizer.cls_token_id and input_ids[-1] == tokenizer.sep_token_id
    print("\n[OK] tokenizer 编解码与断言全部通过。")
