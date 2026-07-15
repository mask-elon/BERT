# 代码与论文对照表（paper ↔ code mapping）

本表把手写实现的每个模块对应到原始论文的章节 / 公式，并给出代码文件与行号范围，
便于对照阅读与答辩。

参考论文：
- **BERT**：Devlin et al., 2018, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (arXiv:1810.04805)
- **Transformer**：Vaswani et al., 2017, *Attention Is All You Need* (arXiv:1706.03762)
- **LoRA**：Hu et al., 2021, *LoRA: Low-Rank Adaptation of Large Language Models* (arXiv:2106.09685)
- **WordPiece**：Wu et al., 2016, *Google's Neural Machine Translation System*

---

## 一、BERT / Transformer 主干（`src/bert_model.py`）

| 模块 | 论文章节 / 公式 | 代码文件 | 行号范围 |
|---|---|---|---|
| `gelu` 激活 | BERT 3.3「gelu activation」；GELU: `0.5x(1+erf(x/√2))` | `src/bert_model.py` | 40–41 |
| `BertConfig` 超参 | BERT 3.1「Model Architecture」(L/H/A) | `src/bert_model.py` | 49–85 |
| `BertEmbeddings` | BERT 3.2「Input/Output Representations」+ Fig.2；`E = Token+Segment+Position` | `src/bert_model.py` | 97–140 |
| `BertSelfAttention` | Transformer 3.2.1 公式(1) `softmax(QKᵀ/√dₖ)V`；3.2.2 多头 | `src/bert_model.py` | 152–197 |
| `BertSelfOutput`（Add&Norm） | Transformer 3.1「residual + LayerNorm」(Post-LN) | `src/bert_model.py` | 207–218 |
| `BertAttention`（注意力子层封装） | Transformer 3.1 Encoder 子层 | `src/bert_model.py` | 226–234 |
| `BertIntermediate`（FFN 第1层+GELU） | Transformer 3.3「Position-wise FFN」；BERT 用 GELU | `src/bert_model.py` | 244–251 |
| `BertOutput`（FFN 第2层+Add&Norm） | Transformer 3.3 (W₂) + 3.1 残差/LayerNorm | `src/bert_model.py` | 260–271 |
| `BertLayer`（Encoder block） | Transformer 3.1「Encoder」中的一层 | `src/bert_model.py` | 280–290 |
| `BertEncoder`（堆叠 N 层） | Transformer 3.1「stack of N identical layers」 | `src/bert_model.py` | 298–307 |
| `BertPooler`（[CLS] 句向量） | BERT 3「[CLS] aggregate representation」 | `src/bert_model.py` | 317–327 |
| `BertModel`（主干） | BERT 3.1 整体架构 | `src/bert_model.py` | 336–382 |
| `_init_weights`（截断正态 std=0.02） | BERT 官方实现初始化 | `src/bert_model.py` | 346–358 |
| `get_extended_attention_mask`（padding 加性掩码） | Transformer 3.2.3 masking | `src/bert_model.py` | 361–365 |
| `MLMHead`（预训练头） | BERT 3.1「Task #1: Masked LM」；weight tying | `src/bert_model.py` | 393–411 |
| `BertNSPHead`（NSP 头） | BERT 3.1「Task #2: NSP」 | `src/bert_model.py` | 422–429 |
| `BertForPreTraining`（联合损失） | BERT 3.1 `L = L_MLM + L_NSP` | `src/bert_model.py` | 441–478 |

## 二、分词（`src/tokenizer.py`）

| 模块 | 论文章节 / 公式 | 代码文件 | 行号范围 |
|---|---|---|---|
| `BasicTokenizer`（清洗/小写/去重音/标点切分） | BERT tokenization.py 思路 | `src/tokenizer.py` | 47–116 |
| `WordpieceTokenizer`（贪心最长匹配子词） | WordPiece (Wu et al., 2016)；BERT 3.2 | `src/tokenizer.py` | 126–159 |
| `WordPieceTokenizer.encode`（[CLS]A[SEP]B[SEP]） | BERT 3.2 Input Representation | `src/tokenizer.py` | 285–315 |

## 三、数据管线（`src/data.py`）

| 模块 | 论文章节 / 公式 | 代码文件 | 行号范围 |
|---|---|---|---|
| `PretrainDataset._apply_mlm_mask`（80/10/10 掩码） | BERT 3.1「Task #1」：mask 15%，其中 80% [MASK]/10% 随机/10% 不变 | `src/data.py` | 259–273 |
| `PretrainDataset._build_nsp_examples`（句对构造） | BERT 3.1「Task #2: NSP」50/50 IsNext/NotNext | `src/data.py` | 179–244 |
| `make_collate_fn`（动态 padding） | 工程实现（对齐 batch 内最长） | `src/data.py` | 315–340 |

## 四、预训练（`src/train_pretrain.py`）

| 模块 | 论文章节 / 公式 | 代码文件 | 行号范围 |
|---|---|---|---|
| `build_warmup_linear_scheduler`（warmup+线性衰减） | BERT A.2「Pre-training Procedure」；Transformer 5.3 | `src/train_pretrain.py` | 82–96 |
| `mlm_loss_fn`（仅 mask 位交叉熵 `ignore_index=-100`） | BERT 3.1 MLM 损失 | `src/train_pretrain.py` | 138–143 |
| `collect_unique_params`（权重共享去重） | 处理 MLM decoder 与词嵌入 weight tying | `src/train_pretrain.py` | 124–135 |
| `train_one_epoch`（梯度裁剪 max_norm=1.0） | BERT A.2 训练细节 | `src/train_pretrain.py` | 176–210 |

## 五、LoRA（`src/lora.py` + `src/apply_lora.py`）

| 模块 | 论文章节 / 公式 | 代码文件 | 行号范围 |
|---|---|---|---|
| `LoRALinear`（低秩适配层） | LoRA 4.1「Low-Rank-Parametrized Update」`W₀+BA` | `src/lora.py` | 44–127 |
| `LoRALinear.forward` | LoRA 4.1 `h = W₀x + (α/r)·BAx` | `src/lora.py` | 85–95 |
| 初始化 `A~N(0,0.02)`, `B=0` | LoRA 4.1「A 随机高斯、B 零初始化 → 起点 ΔW=0」 | `src/lora.py` | 74–83 |
| 缩放 `scaling = α/r` | LoRA 4.1 缩放项 | `src/lora.py` | 62 |
| `merge`/`unmerge`（推理合并 W=W₀+BA） | LoRA 4.1「部署时无额外推理延迟」 | `src/lora.py` | 102–115 |
| `get_trainable_params` / `count_lora_params` | LoRA 参数量统计 `r·k + d·r` | `src/lora.py` | 117–119 / 130–135 |
| `apply_lora`（注入 Q/V） | LoRA 4.2「Applying LoRA to Transformer」(只适配 W_q, W_v) | `src/apply_lora.py` | 46–63 |
| `remove_lora`（还原开关） | 工程实现（可插拔） | `src/apply_lora.py` | 65–85 |
| `mark_only_lora_as_trainable`（只训 A/B） | LoRA 训练：冻结 W₀，只更新低秩矩阵 | `src/apply_lora.py` | 87–92 |

## 六、下游微调（`src/finetune_full.py` + `src/finetune_lora.py`）

| 模块 | 论文章节 / 公式 | 代码文件 | 行号范围 |
|---|---|---|---|
| `BertForSequenceClassification`（[CLS]→Linear 分类头） | BERT 3.5「Fine-tuning BERT」 | `src/finetune_full.py` | 75–96 |
| 全量微调（所有参数可训练） | BERT 3.5 微调范式（对比基线） | `src/finetune_full.py` | 346–472 |
| `mark_lora_and_head_trainable`（只训 LoRA+分类头） | LoRA 训练范式（冻结骨干） | `src/finetune_lora.py` | 91–95 |
| LoRA 微调主流程 | LoRA 4.2 应用于下游任务 | `src/finetune_lora.py` | 167–277 |

## 七、对比分析（`src/compare.py`）

| 模块 | 用途 | 代码文件 | 行号范围 |
|---|---|---|---|
| `collect_records` / `print_table` | 聚合结果、打印对比表 | `src/compare.py` | 55–90 |
| `plot_charts`（参数量/准确率/合并图） | 生成 `figures/*.png` | `src/compare.py` | 92–171 |
| `write_analysis_md`（自动结论） | 生成 `results/analysis.md` | `src/compare.py` | 173–241 |
