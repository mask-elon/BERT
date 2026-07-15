# -*- coding: utf-8 -*-
"""
bert_model.py —— 从零手写 BERT 骨架（Day 1）

论文对应：
    [1] Devlin et al., 2018, "BERT: Pre-training of Deep Bidirectional
        Transformers for Language Understanding" (https://arxiv.org/abs/1810.04805)
    [2] Vaswani et al., 2017, "Attention Is All You Need"
        (https://arxiv.org/abs/1706.03762)

设计原则（严格遵守项目约束）：
    - 不使用 transformers 的 BertModel / BertConfig / BertForMaskedLM 等现成类
    - 不使用 peft 库
    - 不加载任何预训练权重
    - Embedding / Multi-Head Self-Attention / FFN / LayerNorm+残差 / MLM Head 全部手写
    - 仅使用 torch.nn 的基础层（Linear / Embedding / LayerNorm / Dropout）作为积木，
      这些是通用张量运算组件，不属于"现成的 BERT 模型"。

模块命名沿用 HuggingFace 风格（BertSelfAttention / BertSelfOutput ...），
便于对照学习，但实现完全自写。
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# GELU 激活函数
# 论文对应：BERT 论文 3.3 节 "we use a gelu activation"，
#           GELU 原始定义 Hendrycks & Gimpel, 2016。
# 公式（精确版，基于高斯误差函数 erf）：
#     GELU(x) = x * Φ(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
#     其中 Φ 为标准正态分布的累积分布函数 (CDF)。
# 说明：手写以体现公式，与 F.gelu 数值一致。
# =============================================================================
def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


# =============================================================================
# BertConfig：集中存储所有超参数
# 论文对应：BERT 论文 3.1 节 "Model Architecture"（L / H / A 等符号）。
# 小规模默认配置（CPU 可跑）：4 层 / 256 隐藏维 / 4 头 / 512 FFN / 512 序列 / 30522 词表。
# =============================================================================
class BertConfig:
    def __init__(
        self,
        vocab_size: int = 30522,        # 词表大小（WordPiece，BERT-base 默认）
        hidden_size: int = 256,         # H：隐藏维度 d_model
        num_layers: int = 4,            # L：Transformer encoder 层数
        num_heads: int = 4,             # A：注意力头数
        intermediate_size: int = 512,   # FFN 中间层维度（BERT-base 为 4*H）
        max_seq_len: int = 512,         # 最大序列长度（位置编码上限）
        type_vocab_size: int = 2,       # segment 类型数（句子 A / 句子 B）
        hidden_dropout_prob: float = 0.1,        # embedding / 子层输出 dropout
        attention_dropout_prob: float = 0.1,     # attention 权重 dropout
        layer_norm_eps: float = 1e-12,           # LayerNorm 数值稳定项
        pad_token_id: int = 0,          # padding token id（embedding 用）
    ):
        assert hidden_size % num_heads == 0, (
            f"hidden_size({hidden_size}) 必须能被 num_heads({num_heads}) 整除"
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        # 每个头的维度 d_k = H / A
        self.head_dim = hidden_size // num_heads

    def __repr__(self) -> str:
        return (
            f"BertConfig(vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, num_heads={self.num_heads}, "
            f"intermediate_size={self.intermediate_size}, max_seq_len={self.max_seq_len})"
        )


# =============================================================================
# BertEmbeddings：输入表示
# 论文对应：BERT 论文 3.2 节 "Input/Output Representations" + Figure 2。
# 公式：E = TokenEmbedding + SegmentEmbedding + PositionEmbedding
#       随后接 LayerNorm 和 Dropout。
# 说明：位置编码采用"可学习"方式（BERT 原文使用 learned position embeddings，
#       而非 Transformer 的正弦编码）。
# =============================================================================
class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        # 词嵌入：token id -> H 维向量（padding token 不参与梯度贡献偏移）
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        # 段嵌入：区分句子 A / B（NSP 任务用）
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        # 位置嵌入：可学习，索引 0..max_seq_len-1
        self.position_embeddings = nn.Embedding(
            config.max_seq_len, config.hidden_size
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 预先注册位置 id 缓冲区 [1, max_seq_len]，前向时按 seq_len 切片，随模型移动设备
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_seq_len).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor = None):
        # input_ids: [batch, seq_len]
        batch_size, seq_len = input_ids.shape

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 取前 seq_len 个位置索引 [1, seq_len]，广播到 batch
        position_ids = self.position_ids[:, :seq_len]

        # 三种嵌入相加（BERT Figure 2）
        words = self.word_embeddings(input_ids)                 # [B, S, H]
        segments = self.token_type_embeddings(token_type_ids)   # [B, S, H]
        positions = self.position_embeddings(position_ids)      # [1, S, H] 广播

        embeddings = words + segments + positions
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # [B, S, H]


# =============================================================================
# BertSelfAttention：多头缩放点积自注意力
# 论文对应：Transformer 论文 3.2.1（缩放点积）+ 3.2.2（多头）。
# 公式：
#     Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V      —— 公式(1)
#     MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
# 说明：BERT 为双向编码器，attention 不做因果掩码，只用 padding mask。
# =============================================================================
class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim           # d_k = H / A
        self.all_head_size = config.hidden_size   # A * d_k = H

        # Q / K / V 三个独立线性投影：Linear(H, H)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # attention 权重上的 dropout（Transformer 论文附录常用做法）
        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, S, H] -> [B, A, S, d_k]：拆分多头并把头维提到前面
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        # hidden_states: [B, S, H]
        # 1) 线性投影得到 Q/K/V，再拆头
        q = self._split_heads(self.query(hidden_states))  # [B, A, S, d_k]
        k = self._split_heads(self.key(hidden_states))    # [B, A, S, d_k]
        v = self._split_heads(self.value(hidden_states))  # [B, A, S, d_k]

        # 2) 缩放点积得分：Q K^T / sqrt(d_k) -> [B, A, S, S]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # 3) 应用 attention mask（padding 位置加上 -inf，softmax 后趋近 0）
        if attention_mask is not None:
            scores = scores + attention_mask  # attention_mask 已是加性掩码 [B,1,1,S]

        # 4) softmax 归一化到概率分布，再 dropout
        probs = F.softmax(scores, dim=-1)      # [B, A, S, S]
        probs = self.dropout(probs)

        # 5) 加权求和得到上下文向量：[B, A, S, d_k]
        context = torch.matmul(probs, v)

        # 6) 合并多头：[B, A, S, d_k] -> [B, S, H]
        context = context.permute(0, 2, 1, 3).contiguous()
        batch_size, seq_len, _, _ = context.shape
        context = context.view(batch_size, seq_len, self.all_head_size)
        return context  # [B, S, H]


# =============================================================================
# BertSelfOutput：attention 后的 Add & Norm
# 论文对应：Transformer 论文 3.1 "residual connection ... followed by LayerNorm"，
#           即 LayerNorm(x + Sublayer(x))（BERT 采用 Post-LN）。
# 结构：Linear(H, H) -> Dropout -> 残差(+输入) -> LayerNorm
# =============================================================================
class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 输出投影 W^O
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        # hidden_states: attention 的输出；input_tensor: attention 的输入（残差用）
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)  # Add & Norm
        return hidden_states


# =============================================================================
# BertAttention：SelfAttention + SelfOutput 的封装（一个完整注意力子层）
# 论文对应：Transformer 论文 3.1（编码器子层之一）。
# =============================================================================
class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        self_out = self.self(hidden_states, attention_mask)      # [B, S, H]
        attention_out = self.output(self_out, hidden_states)     # Add & Norm 残差用原输入
        return attention_out


# =============================================================================
# BertIntermediate：FFN 第一层（升维 + GELU）
# 论文对应：Transformer 论文 3.3 "Position-wise Feed-Forward Networks"；
#           BERT 3.3 节将 ReLU 替换为 GELU。
# 公式：FFN 第一步 = GELU(x W1 + b1)，W1: H -> intermediate_size
# =============================================================================
class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense(hidden_states)  # [B, S, intermediate_size]
        hidden_states = gelu(hidden_states)        # GELU 激活
        return hidden_states


# =============================================================================
# BertOutput：FFN 第二层（降维）+ Add & Norm
# 论文对应：Transformer 论文 3.3 的 W2 + 3.1 的残差/LayerNorm。
# 结构：Linear(intermediate_size, H) -> Dropout -> 残差(+FFN 输入) -> LayerNorm
# =============================================================================
class BertOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        # hidden_states: intermediate 的输出；input_tensor: 进入 FFN 前的张量（残差用）
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)  # Add & Norm
        return hidden_states


# =============================================================================
# BertLayer：一个完整的 Transformer encoder block
# 论文对应：Transformer 论文 3.1 "Encoder"（Nx 中的一层）。
# 结构：Attention 子层 -> FFN 子层（Intermediate + Output）
# =============================================================================
class BertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        attention_out = self.attention(hidden_states, attention_mask)  # [B, S, H]
        intermediate_out = self.intermediate(attention_out)            # [B, S, ff]
        layer_out = self.output(intermediate_out, attention_out)       # [B, S, H]
        return layer_out


# =============================================================================
# BertEncoder：堆叠 num_layers 个 BertLayer
# 论文对应：Transformer 论文 3.1 "a stack of N identical layers"。
# =============================================================================
class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_layers)]
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states  # [B, S, H] 最后一层输出即 last_hidden_state


# =============================================================================
# BertPooler：句子级表示
# 论文对应：BERT 论文 3 节，"The final hidden state ... of the [CLS] token ...
#           used as the aggregate sequence representation"。
# 结构：取第 0 位 [CLS] 的 hidden -> Linear(H, H) -> Tanh
# =============================================================================
class BertPooler(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: [B, S, H]，取序列第 0 个位置（[CLS]）
        cls_token = hidden_states[:, 0]        # [B, H]
        pooled = self.dense(cls_token)
        pooled = self.activation(pooled)
        return pooled  # [B, H]


# =============================================================================
# BertModel：Embeddings + Encoder + Pooler 的主干
# 论文对应：BERT 论文 3.1 节整体架构。
# 输出：last_hidden_state [B, S, H]，pooled_output [B, H]
# =============================================================================
class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        # 参数初始化（截断正态，参考 BERT 官方实现）
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        # 论文对应：BERT 官方实现使用均值 0、std=0.02 的正态初始化
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # 把 [B, S]（1=有效, 0=padding）转成加性掩码 [B, 1, 1, S]：
        # 有效位为 0，padding 位为 一个很大的负数，加到 scores 上后 softmax≈0。
        extended = attention_mask[:, None, None, :].to(dtype=torch.float32)
        extended = (1.0 - extended) * torch.finfo(torch.float32).min
        return extended

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        # input_ids: [B, S]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        ext_mask = self.get_extended_attention_mask(attention_mask)  # [B,1,1,S]

        embedding_output = self.embeddings(input_ids, token_type_ids)      # [B, S, H]
        sequence_output = self.encoder(embedding_output, ext_mask)         # [B, S, H]
        pooled_output = self.pooler(sequence_output)                       # [B, H]
        return sequence_output, pooled_output


# =============================================================================
# MLMHead：Masked Language Model 预训练头
# 论文对应：BERT 论文 3.1 节 "Task #1: Masked LM"。
# 结构：Linear(H, H) -> GELU -> LayerNorm -> Linear(H, vocab_size)
# 说明：最后的解码器一般与 word_embeddings 权重共享（weight tying），
#       此处提供可选的权重绑定接口。
# =============================================================================
class MLMHead(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        # 变换层：Linear -> GELU -> LayerNorm
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 解码层：投影到词表
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def tie_weights(self, word_embeddings: nn.Embedding):
        # 权重共享：解码器权重 = 词嵌入权重（BERT 常见做法，减少参数量）
        self.decoder.weight = word_embeddings.weight

    def forward(self, sequence_output: torch.Tensor):
        # sequence_output: [B, S, H]
        hidden = self.dense(sequence_output)
        hidden = gelu(hidden)
        hidden = self.layer_norm(hidden)
        logits = self.decoder(hidden)   # [B, S, vocab_size]
        return logits


# =============================================================================
# BertNSPHead：Next Sentence Prediction 预训练头
# 论文对应：BERT 论文 3.1 节 "Task #2: Next Sentence Prediction (NSP)"。
# 结构：取 pooled_output（[CLS] 经 Pooler 后的句子表示）-> Linear(H, 2)
# 输出：二分类 logits [B, 2]，类别 0=IsNext（B 是 A 的真实下一句），
#       类别 1=NotNext（B 为随机句）。
# =============================================================================
class BertNSPHead(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        # 直接从句向量二分类；权重 W ∈ R^{2×H}（论文 3.1 节 C ∈ R^H -> 2 类）
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output: torch.Tensor):
        # pooled_output: [B, H] -> [B, 2]
        return self.seq_relationship(pooled_output)


# =============================================================================
# BertForPreTraining：BERT 预训练整体（骨干 + MLM 头 + NSP 头）
# 论文对应：BERT 论文 3.1 节，预训练损失 = MLM 损失 + NSP 损失。
# 说明：
#   - MLM 头的解码器与词嵌入权重共享（weight tying）。
#   - forward 传入 labels / next_sentence_label 时，直接返回联合损失，
#     其中 MLM 用 ignore_index=-100 只在被 mask 位置计损，NSP 为二分类。
# =============================================================================
class BertForPreTraining(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)           # 内部已完成参数初始化
        self.mlm_head = MLMHead(config)
        self.nsp_head = BertNSPHead(config)
        # 两个预训练头单独初始化（BertModel 的 apply 不覆盖头部）
        self.mlm_head.apply(self.bert._init_weights)
        self.nsp_head.apply(self.bert._init_weights)
        # MLM 解码器与词嵌入共享权重（放在初始化之后，保持二者指向同一张量）
        self.mlm_head.tie_weights(self.bert.embeddings.word_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,                 # MLM 目标：[B, S]，非 mask 位为 -100
        next_sentence_label: torch.Tensor = None,    # NSP 目标：[B]，0=IsNext,1=NotNext
    ):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        mlm_logits = self.mlm_head(sequence_output)  # [B, S, vocab_size]
        nsp_logits = self.nsp_head(pooled_output)    # [B, 2]

        out = {"mlm_logits": mlm_logits, "nsp_logits": nsp_logits}

        # 若给了标签则计算联合损失（预训练时使用）
        if labels is not None and next_sentence_label is not None:
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,        # 只在被 mask 的位置计损
            )
            nsp_loss = F.cross_entropy(nsp_logits, next_sentence_label.view(-1))
            total_loss = mlm_loss + nsp_loss      # 论文：L = L_MLM + L_NSP
            out.update({"loss": total_loss, "mlm_loss": mlm_loss, "nsp_loss": nsp_loss})

        return out


# =============================================================================
# __main__：前向维度自测
# 检查项（对应用户规则的"实现清单"）：
#   - import 不报错
#   - 随机张量前向传播维度正确
#   - 断言 last_hidden_state / pooled_output / mlm_logits 形状
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)

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
    print(f"运行设备：{device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # 小规模配置（CPU / GPU 均可跑）
    config = BertConfig(
        vocab_size=30522,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        intermediate_size=512,
        max_seq_len=512,
    )
    print("配置：", config)

    batch_size, seq_len = 2, 32

    # 随机构造输入
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    token_type_ids = torch.randint(0, config.type_vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    # 模拟第二个样本尾部有 5 个 padding
    attention_mask[1, -5:] = 0

    # 构建模型并搬到目标设备
    model = BertModel(config).to(device)
    mlm_head = MLMHead(config).to(device)
    mlm_head.tie_weights(model.embeddings.word_embeddings)  # 与词嵌入共享权重

    model.eval()
    mlm_head.eval()
    with torch.no_grad():
        last_hidden_state, pooled_output = model(input_ids, token_type_ids, attention_mask)
        mlm_logits = mlm_head(last_hidden_state)

    # 打印形状与所在设备
    print("last_hidden_state.shape :", tuple(last_hidden_state.shape), "device:", last_hidden_state.device)
    print("pooled_output.shape     :", tuple(pooled_output.shape), "device:", pooled_output.device)
    print("mlm_logits.shape        :", tuple(mlm_logits.shape), "device:", mlm_logits.device)

    # 维度断言
    assert last_hidden_state.shape == (batch_size, seq_len, config.hidden_size), \
        "last_hidden_state 维度错误"
    assert pooled_output.shape == (batch_size, config.hidden_size), \
        "pooled_output 维度错误"
    assert mlm_logits.shape == (batch_size, seq_len, config.vocab_size), \
        "mlm_logits 维度错误"

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nBertModel 参数量：{total_params:,}")
    print("[OK] 所有维度断言通过，前向传播正常。")

    # -------------------------------------------------------------------------
    # BertForPreTraining（MLM + NSP 联合）自测
    # 检查项：mlm_logits / nsp_logits 形状，联合 loss 为标量且可反向传播。
    # -------------------------------------------------------------------------
    print("\n=== BertForPreTraining（MLM + NSP）自测 ===")
    pretrain = BertForPreTraining(config).to(device)
    pretrain.train()

    # 构造带 MLM / NSP 标签的随机 batch
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    labels[:, 3] = torch.randint(0, config.vocab_size, (batch_size,), device=device)  # 假设第3位被mask
    next_sentence_label = torch.randint(0, 2, (batch_size,), device=device)

    out = pretrain(input_ids, token_type_ids, attention_mask,
                   labels=labels, next_sentence_label=next_sentence_label)
    print("mlm_logits.shape :", tuple(out["mlm_logits"].shape))
    print("nsp_logits.shape :", tuple(out["nsp_logits"].shape))
    print(f"loss = {out['loss'].item():.4f}  (mlm={out['mlm_loss'].item():.4f}, nsp={out['nsp_loss'].item():.4f})")

    assert out["nsp_logits"].shape == (batch_size, 2), "nsp_logits 维度错误"
    assert out["mlm_logits"].shape == (batch_size, seq_len, config.vocab_size), "mlm_logits 维度错误"
    out["loss"].backward()  # 验证联合损失可反向传播
    assert out["loss"].dim() == 0, "loss 应为标量"
    print("[OK] BertForPreTraining 联合损失前向/反向通过。")
