# Handoff: Hierarchical Gist Compression + Selective Unfolding with OPSD (built on SSA)

Nov 19, 2026 · @Someone（Nov 19 更新：设计决定、spike 计划、代码现状；Nov 20 更新：深层树、decode 时的 gist 与 1M 上下文的计划）

## TL;DR 与使用方式

在 SSA（Simplified Sparse Attention）的代码上，把它的两个训练阶段换成：阶段一用**逐层 MSE** 学多层 gist 压缩，阶段二用 **OPSD**（on-policy self-distillation）训练逐级的 selective unfolding。原模型冻结，只训练 **gist 专属的 LoRA**（只作用在 gist / meta-gist 行）、gist embedding 和每个 summary 的 mass 偏置 β；原始 token 的计算与原模型完全相同。

- **阶段一（compression）**：插入 gist 与 meta-gist，用 gist causal mask 挡住原始 token；在同一次前向里，每层用同一组 q/k/v 计算多个 mask 下的 attention，按「老师链」逐级做相对 MSE。
- **阶段二（indexing）**：学生用自己的 gist 打分做逐级 top-k 展开并生成；老师是原模型、读完整 KV；学生按记录的选择 replay，loss = 生成 token 上的反向 KL + 每一级的路由 KL（标签是真实 attention mass）。
- **起点**：Qwen2.5-0.5B-Instruct 跑通全流程，再迁移到 7B。正式的 milestone 之前，先用四个 spike 验证噪声底、路由上限、树的阶段一和 OPSD 机制（见「Spikes」）。

**给 agent 的使用方式**：先按「Spikes」执行，再按「Milestones」逐项执行；每完成一项先跑「Invariants」里的检查再继续。所有新功能都用开关控制，默认值等于原版 SSA。「Design decisions」里已定的事项不要重新讨论；遇到没有覆盖的未定项，停下来问人，不要自行假设。

## Context and motivation

块级稀疏 attention 的路由需要一个块的代表向量；最常见的 mean pooling 会把每个 token 自己的内容和高频 RoPE 位置信息平均掉，导致包含关键 token 的块被漏选。SSA 用学出来的 gist token 代替平均，我们在它的基础上改训练方式。

- **mean pooling 的问题**：一个 chunk 内 key 的平均只剩「这一段的共同语境」，按序列连起来是一条低维轨迹，原始 key 则是一团；平均也稀释了峰值，而 attention 的块 mass 更接近 log-sum-exp。
- **HISA（COLM 2026）**：用 mean pooling 做块级粗筛、再在块内逐 token 精排，免训练替换 DSA 的 indexer。它的失败案例全部来自第一阶段漏选块，原因被归结为 mean pooling（实体关系被混在一起、短的因果片段被稀释、needle 被淹没），并把「学出来的块表示」列为未来工作。前缀越长，选择与 DSA 的重合度越低（约 97% → 79%）。
- **SSA（Mao、Li、Fox，Stanford）**：每 L 个 token 插一个 gist，用 gist causal mask 挡住原始 token，迫使模型把 chunk 信息写进 gist 的 KV；推理时用 q·k\_gist 打分，展开 top-k 个 chunk 的原始 KV。支持 meta-gist（H-SSA）做逐级展开。原版两阶段都用 next-token loss，全量微调：继续预训练 + 可选的 selective finetuning。
- **我们改什么**：阶段一把 NTP 换成逐层的 attention 输出匹配（信号更密、可冻结原模型）；阶段二把 off-policy 的 selective finetuning 换成 on-policy 的 OPSD，并对每一级的 summary 打分加入以真实 mass 为标签的路由监督，补上「没被选中的 summary 收不到梯度」这个盲点。

## Goal and success criteria

目标：在同样的压缩比和展开预算下，用逐层 MSE + OPSD 训练出的多层 gist 与 selective unfolding，质量优于原版 SSA 的训练方式，并接近读完整 KV 的原模型。

- **质量**：在 LongBench、RAG（NQ 等）、passkey / 多键 needle 上，同压缩比（如 16×，chunk4-chunk4）下优于原版 SSA（CPT + selective FT），与 full attention 的差距明显缩小。
- **路由**：每一级的路由召回（选中的 chunk / 段是否覆盖真实 mass 最大的那些）高于原版 SSA，长上下文靠后位置的下降更小。
- **对齐**：阶段二之后，学生与老师在生成 token 上的平均反向 KL 显著下降；全部展开时 KL 为 0（数值误差内）。
- **成本**：原模型冻结，只加 gist 专属参数；推理速度沿用 SSA 的 kernel，展开预算不因训练而变大。
- **非目标**：超越原模型。两个阶段都以原模型为上限；SSA 在 RAG 上超过 full attention 的去噪收益不在本项目的保证范围内（可作为加 NTP 的 ablation）。

## Design decisions

下列事项已经决定，理由写在每条后面；推翻任何一条需要先有 spike 或实验数据。

**原始 token 保持原来的位置（SSA 已经这样做）。** gist 与 meta-gist 取它前面最后一个原始 token 的位置（`x1 x2 g1` → `0 1 1`）：模型 forward 里 RoPE 位置减去了截至当前的 summary 个数（`gist_qwen2.py`、`gist_llama.py` 中的 `rope_pos = pos - cumsum(gist_mask)`），decode 时每个生成 token 取上一个位置加一。`generation_utils.py` 里按 attention mask 累加的 position id 会被这一步修正，不代表真实的 RoPE 位置。这条规则让「全部展开 = 原模型」严格成立：S1 在 Qwen2.5-0.5B-Instruct 上测得替换式全部展开的平均反向 KL 约 1e-9（8K 上下文，最大 6e-7）；作为对照，若原始 token 按插入后的下标取位置（chunk4-chunk4 下拉长约 31%），平均反向 KL 约 0.19 nats/token。已放出的 checkpoint 按同一规则训练，7B 上的原版 SSA 对照可以直接用。`tests/test_gist_noise_floor.py` 守住这条规则。

**只训练 gist 专属的参数（下称方案 D），全局 LoRA（方案 B）作对照。** 每层的 q/k/v/o 投影在 gist / meta-gist 行上换成 W<sup>g</sup> = W + BA（gist 专属 LoRA，B 初始为 0），原始 token 走原权重 W，MLP 先共用。这样选有三个原因：

1. 局部老师 `o_raw` 是用学生自己的 q/k/v 算的，如果训练会改原始 token 的 q/k，模型可以让 query 不看远处，使 `o_raw ≈ o_g1`，MSE 下降而 gist 什么都没学会；纯 MSE 时没有任何 loss 阻止这个退化解。方案 D 冻结了原始 token 的投影，排除了这个解。
2. 全部展开时原始 token 的计算与原模型逐位相同，配合 SSA 的位置规则，噪声底严格为 0，有偏差就是 bug。
3. 关掉 gist 专属参数就是原模型，OPSD 的老师不用另存一份权重。

方案 D 的风险在读的一侧：原模型的 q 必须能读懂学出来的 gist KV。缓解办法依次是每个 summary 的偏置 β、提高 LoRA 的秩、给 gist 加 MLP 专属参数、只在 query 读 summary 时生效的小 LoRA。

**方案 D 用 LoRA 参数化，与方案 B 同秩（r = 32）。** 全秩的 gist 专属投影在 0.5B 上一套约 44M，在 Qwen2-7B 上一套约 823M（每层 q、o 各 3584 × 3584，k、v 各 3584 × 512，共 29.4M × 28 层，约占模型的 11%），0.5B 上的全秩结论迁移不到 7B。用 LoRA 后两个规模用同一种方法，而且方案 D 与方案 B 只差「LoRA 作用在哪些行」一个因素，S3 的 D 对 B 比较只反映原始 token 的计算有没有被改。0.5B 上另跑一个全秩方案 D 作为容量上限。手写实现，不引入 `peft`。

**每个 summary 有自己的 mass 偏置 β，由内容算出。** β<sub>t</sub> = b<sub>level</sub> + u<sub>level</sub> · RMSNorm(h<sub>t</sub>)，每个 KV head 一个；同时加在 attention logit 与路由分数上，使两者一致。b<sub>level</sub> 初始为 log（这一级覆盖的原始 token 数），即 log 4 与 log 16，u 初始为 0；这个初值来自 log Σ<sub>j</sub> exp(q·k<sub>j</sub>) ≥ log n + q·k̄。原来的「每级一个标量 bias」在同一级的 softmax 里是常数，只能校准 summary 与原始 key 之间的 mass，路由 KL 对它的梯度恒为 0；而 chunk 的 log-sum-exp mass 还取决于块内 key 的分散程度（例如有没有一个突出的 needle），单个 key 向量表达不了，需要每个 summary 自己的 β。β 由 MSE、反向 KL 与路由 KL 共同训练。

**gist 与 meta-gist 默认共用一套 gist LoRA。** 共用时每个参数见到的数据更多，层数加深时参数不增长（适合 O(log T) 层的深树），层级身份由各级不同的 embedding（`<G1>`、`<G2>`）从第一层起区分；mass 尺度的差别由各级自己的 b<sub>level</sub>、u<sub>level</sub> 处理。每级一套的理由是两级的输入分布不同（gist 读原始 token，meta-gist 读 gist），以及分阶段训练可以在训第二级时冻结第一级，使目标 `o_g1` 真正不动。因此 S3 的分阶段配置用每级一套并在第二阶段冻结第一级；若「分阶段 + 每级一套」明显优于「同时训 + 共用」，在 M2 补「同时训 + 每级一套」，把训练顺序与参数分开两个因素拆开。

**路由保持逐级自顶向下。** 每一级只在同级候选里做 top-k，与 SSA、CUDA kernel 和 adaptive top-k 一致，路由 KL 按级定义。跨级的 best-first tree cut 只在路由上限的 spike 里离线比较；若它在相同打分成本下明显更好，再作为后续项目，届时 β 已能让不同级的分数直接比较。

**梯度在层之间连通，不做逐层 detach。** 梯度落点有一个结构性的不对称：gist 被读的一侧（W<sup>g</sup><sub>K</sub>、W<sup>g</sup><sub>V</sub>、β）在每一层直接收到 MSE 与路由 KL 的梯度；写的一侧（W<sup>g</sup><sub>Q</sub>、W<sup>g</sup><sub>O</sub>，决定 gist 从自己的 chunk 读什么、写回什么）只能从更高层的 loss 经若干层反传收到梯度，最后一层完全没有。因此逐层 detach 会让写的一侧学不到东西；最后一层不建 W<sup>g</sup><sub>Q</sub>、W<sup>g</sup><sub>O</sub>；训练中按参数组（W<sup>g</sup><sub>Q/O</sub>、W<sup>g</sup><sub>K/V</sub>、β、embedding）逐层记录梯度范数，监测写的一侧是否跟得上。只给 gist 专属 K、V 的变体（D-kv）作为 ablation，它与完整方案 D 的差距衡量「学会写」的价值。

**阶段一用局部老师。** 第 ℓ 层的 `o_raw` 用这一层（已被前面各层压缩误差影响的）输入计算，不另跑原模型。逐层误差是否累积，用后缀 logit KL（学生对比原模型）监测；如果出现「逐层 MSE 下降、logit KL 不降」，再加全局老师作为 ablation。现有代码里的 `--klloss` 分支（`gist_qwen2.py:1617-1639`，默认关）就是一种全局老师：batch 复制成 raw mask 与 gist mask 两份完整前向，逐层对 hidden state 做 smooth-L1。它可以作为多 mask batch 写法和全局老师 ablation 的参考实现。

**两级默认同时训练。** 两项 MSE 从第 0 步一起优化；先训 g1 再加 hier 的分阶段做法作为对照，因为 `o_g1` 在训练中一直在变，第二级追的是移动目标。

**数据和起点。** Qwen2.5-0.5B-Instruct，gist embedding 新初始化（`gist_utils.add_gist_tokens`）。阶段一用 SSA 放出的 CPT 数据（`data_process/convert_cpt_to_stream.py` 转成 streaming 格式，已支持 Qwen2.5 的 gist id）；阶段二的 rollout prompt 用 SSA 放出的 SFT 数据（ActivationBeacon / KVLink）。LongBench 只用于评测，不进训练。agent trajectory 数据需要时自己找。

## Stage 1: hierarchical gist compression with per-layer MSE

一次学生前向（层级 gist mask），每层用同一组 q/k/v 额外计算老师 mask 下的 attention，在后缀位置逐级做相对 MSE；梯度在层之间连通，只更新 gist 专属 LoRA、gist embedding 和 β。

**序列结构（两层）**：每 L 个原始 token 后插入一个 gist g，每 J 个 gist-chunk 对后插入一个 meta-gist G；序列分为压缩区（前缀）和后缀，loss 只在后缀上算。插入用 `src/data/gist_stream.py` 的 streaming 规则（只看位置、不看词边界，前缀的 gist 形式是任何延长序列的前缀）。

```markdown
x1 x2 g1 x3 x4 g2 G1 | x5 x6 g3 x7 x8 g4 G2 | suffix ...
```

**mask 规则（与 SSA 相同）**：chunk 内 token 看本 chunk 与之前所有的 summary；gist 看自己的 chunk 与之前的 summary；gist 之后的 token 看不到该 chunk 的原始 token；G 之后的 token 看不到 G 管辖范围内的原始 token 和 gist，只看 G。

**每层的三种 attention（同一组 q/k/v）**：

1. `o_raw`：纯 causal mask，能看到全部原始 token，看不到任何 summary；
2. `o_g1`：单层 gist mask，只能通过 gist 看远处，meta-gist 不作为 key；
3. `o_hier`：层级 mask，远处只能通过 meta-gist 看（推理时真实的路径，往下一层传的是它）。

三种 mask 都写成 FlexAttention 的 `mask_mod`（`src/data/gist_flex_mask.py` 已有 gist mask），不构造 dense T × T mask。

**loss（老师链，每一级只学压缩紧挨着的下一级）**：

```latex
\mathcal{L}_1 = \sum_{\ell} \Big[\mathrm{relMSE}\big(o^{(\ell)}_{g1},\, \mathrm{sg}(o^{(\ell)}_{raw})\big) + \mathrm{relMSE}\big(o^{(\ell)}_{hier},\, \mathrm{sg}(o^{(\ell)}_{g1})\big)\Big]_{\text{suffix}} + \lambda_r \mathcal{L}_{route} + \lambda_{ntp} \mathcal{L}_{NTP}
```

- sg 表示 stop-gradient（老师只作为目标）；relMSE = ‖a − b‖² / ‖b‖²，按位置平均。
- **路由 loss**：用学生的 q 和原始 token 的 key 算每个 chunk、每个段真实的 log mass（块内 log-sum-exp），对同一级所有 gist / meta-gist 的打分 z = q·k/√d + β 做 KL。只在每条序列采样的几十个后缀 query 上计算，避免 T × T。
- **NTP**：默认 λ\_ntp = 0（纯 MSE）；λ\_ntp > 0 作为 ablation。
- **可训练参数**：方案 D（gist 专属 LoRA，r = 32，见「Design decisions」）；方案 B 为 q/k/v/o 上同秩的全局 LoRA，作对照。原模型冻结。
- **起点配置**：L = 4、J = 4（16×，与 SSA 已放出的 checkpoint 对齐）；第一层不参与 summary 的打分。

## Stage 2: selective unfolding trained with OPSD

展开决定始终来自学生自己的打分 q·k\_summary；真实 mass 只作为训练时的标签；Gumbel 噪声只用于训练时的探索（可选）。

**一个训练步的数据流**

1. **学生生成（no grad）**：带 gist / meta-gist 的上下文做层级 mask 的 prefill；逐 token 生成时，在每一层（第一层除外）、每个 KV 组：对 meta-gist 打分取 top-k₂ 段，再对这些段里的 gist 打分取 top-k₁ 个 chunk；attention 只看被选中的 meta-gist、gist、chunk 原始 token（SSA 的 SG+SR）以及本地未压缩部分；temperature 1 采样。**记录每一步、每层、每组的选择索引**。必须记录而不能在 replay 时重算：增量 decode 与并行前向的 KV 只是 allclose，接近平局的选择会翻转。
2. **老师打分（no grad）**：原模型（关掉 gist 专属参数）、普通 causal mask、**不插 gist 的原始序列**（上下文 + 学生生成的 y），一次并行前向，得到生成 token 位置的分布 p\_T。原始 token 在两边位置相同，只需按插入时的 source map（`insert_gists_with_map`）对齐下标。
3. **学生 replay（with grad）**：把「上下文 + y」一次性并行前向，每层每组每个位置**严格按第 1 步记录的选择**构造 mask（不重新选择）；得到 p\_S、每层对**所有** summary 的打分，以及用学生 q 与原始 key 算出的真实 chunk / 段 log mass（no grad）。
4. **loss**：

```latex
\mathcal{L}_2 = \underbrace{\frac{1}{|G|}\sum_{t \in G} \mathrm{KL}\big(p_S(\cdot\mid y_{<t}) \,\|\, p_T(\cdot\mid y_{<t})\big)}_{\text{reverse KL on generated tokens}} + \lambda_r \sum_{\text{level}} \mathrm{KL}\big(\mathrm{softmax}(\text{true log mass}) \,\|\, \mathrm{softmax}(q^\top k_{summary}/\sqrt{d} + \beta)\big)
```

5. **反向与更新**：反向 KL 的梯度经 attention 流到被选中的 summary 和 gist 专属参数；没被选中的 summary 的 attention 权重为 0，梯度恰好为 0。路由 KL 的梯度流到**所有** summary 的 key 和 β（含没被选中的），补上离散选择拿不到梯度的盲点。注意选择本身只由路由 KL 训练；反向 KL 的作用是让训练时的状态分布与推理一致。

**关键设计约束**

- **预算固定**：top-k 用 SSA 的 adaptive top-k，不让 k 被 KL 优化（否则模型倾向于多展开）；记录每步实际展开数。
- **GQA**：默认沿用 SSA 的 per-head top-k 再组内取并集；组级（组内分数相加、一次 top-k）作为对照。0.5B 每组 7 个 head（14 Q / 2 KV），并集会比 7B 展开更多。
- **探索（可选）**：rollout 时在各级打分上加 Gumbel 噪声 g = −log(−log U) 后取 top-k；replay 必须用实际采样到的选择；推理时关闭。
- **树的误差累积**：上层 k 可大于下层（上宽下窄）；分别统计每一级的路由召回。
- **可选简化**：相邻层共享同一组选择，减少记录量和每层噪声（需验证质量）。

## Codebase

工作分支 `opsd/spikes`，从 `feature/gist-tree-training`（`e3a9d26`）切出。这个分支已经有新项目需要的训练侧基础设施：FlexAttention gist 训练（不构造 T × T mask，BlockMask 构建也不展开成稠密 mask）、position-only streaming 插入与任意层数的 gist 树、3 层以上的 mask 修复与覆盖测试、分块 LM loss、non-reentrant 梯度检查点、深树路由模拟器 `experiments/deep_routing.py`。CPU 测试在 `tests/`（`python -m pytest tests -q`）。

| 文件 | 已有内容 | 我们要做的 |
| --- | --- | --- |
| `src/data/gist.py`、`src/data/gist_flex_mask.py`、`src/data/gist_stream.py` | 插入 gist / meta-gist，dense 与 flex 的 gist mask | 老师 mask（纯 causal、单层 gist）的 `mask_mod` |
| `gist_qwen2.py`（`eager_attention_forward_prefill` / `_decoding` / `_prefill_chunk`） | 逐级打分与 `top_k_from_logits`、`level_meta`、adaptive top-k | gist 专属投影；训练路径内多 mask 的 attention 与逐层 MSE；路由 KL；选择的记录与强制 replay；可选 Gumbel |
| `GistQwen2ForCausalLM.forward` | 训练的 `selective` 分支；多 mask 的 5D 分支与 `--klloss` 逐层对齐（`masked_smoothl1_vocab`）；`masked_kl_vocab`（未被调用） | 把逐层 loss 汇总进总 loss；接受 replay 的选择 |
| `generation_utils.py`、`gist_caching.py` | 带 gist 的生成与 KV cache；position id 在 `generation_utils.py:104` | rollout 时导出每步每层每组的选择 |
| `gist_trainer_long.py`、`gist_training.py`、`gist_cpt_trainer.py`、`gist_sft_trainer.py` | 原版两阶段训练与共用 setup；`--train_attn flex` | 作为模板；阶段一加 gist 专属参数、冻结原模型、NTP 开关 |
| 新文件 `gist_opsd_trainer.py` | — | rollout → 老师 → replay → 两个 loss → 更新 |
| `kernels/`、`attn_candidate.py` | CUDA 稀疏 decode kernel 与调度；`DUMP_DECODE` 只导出第 1 层一次 | 不改 kernel；`ATTN_IMPL=ref` 作为参考路径做对齐 |

**已知限制**：CUDA decode kernel 固定 `D_HEAD=128`（`kernels/sparse_decode_attn.cu:32`），Qwen2.5-0.5B 的 head\_dim 是 64，所以 0.5B 的评测和 rollout 走 `ATTN_IMPL=ref`；系统测量（TTFT / TPOT）在 7B 上做。flex 路径目前只支持非 selective 的前向（`gist_qwen2.py:1267`），长上下文的 replay 需要把记录的选择编码进 `mask_mod`。

**仓库组织**：remote 里你的 fork 是 `Beihao-Zhou/simplified-sparse-attention`，原作者的仓库是 `yuzhenmao/simplified-sparse-attention`；只往 fork 推送、开 issue。issue 跟踪见 `docs/agents/issue-tracker.md`。spike 共用 `opsd/spikes`；之后的 milestone 每个功能一个分支（`stage1-layerwise-mse`、`routing-kl`、`stage2-record-replay`、`stage2-opsd`、`explore-gumbel`），一个功能一个 PR。

**开关**（默认值 = 原版 SSA）：`trainable` ∈ {`full`（默认）, `gist_lora`（方案 D）, `gist_kv`（D-kv）, `gist_full`（全秩方案 D）, `lora`（方案 B）}、`lora_r`（32）、`gist_lora_per_level`（默认关）、`freeze_lower_levels`、`gist_mlp`、`layerwise_mse`、`lambda_ntp`、`lambda_route`、`summary_bias` ∈ {`none`（默认）, `level`, `content`}、`record_selection`、`replay_selection`、`gumbel_tau`（0 = 关）、`share_selection_across_layers`。

## Spikes

正式训练之前，用四个小规模试验回答「答案为否就要改计划」的问题。spike 只看趋势和量级：每个配置 1 个种子，主指标是后缀 logit KL（held-out 长文本，后缀上学生对比 full attention 原模型的平均 token KL，单位 nats）和逐级路由召回；LongBench 与 RAG 留到 milestone。每个 spike 在 fork 上开一个 issue，结果与 commit 贴在 issue 里，数字写进 `results/spikes.csv`。硬件：L4（24 GB）现在可用，H100 几天内到；阶段一的试验赶上 H100 就放到 H100 上跑。

**全部展开时噪声底是不是 0？**（已完成，通过）Qwen2.5-0.5B-Instruct 插入未训练的 gist、全部展开，与原模型在原始序列上比较。替换式（原始 token 看不到 summary）平均反向 KL 约 1e-9，最大 6e-7，远低于 1e-3 的阈值；SG+SR 式（所有未训练的 summary 都可见）平均 3–5 nats，说明未训练的 summary 一旦可见会严重干扰输出；位置拉长的对照约 0.19 nats。结果在 `results/spikes.csv`。

**现有两层树的路由离 oracle 有多远？** 用 `GSA-FT-Qwen2-7B-Instruct-chunk4-chunk4`（L4 已缓存）在 triviaqa、hotpotqa、narrativeqa、2wikimqa 各约 15 条上记录逐级打分与 oracle mass（每条前 32 步 decode，第 2、8、16、24 层），按上下文 < 8K、8–16K、> 16K 分桶，交给 `experiments/deep_routing.py` 算逐级召回、mass 召回与答案 chunk 命中，并模拟更深的树。需要先写记录脚本，输出 `s0/s1/m0/parent0/k0` 格式。成本：L4 上 1–3 小时。每级 mass 召回在所有桶 ≥ 95% 时，路由 KL 降为次要；长桶 < 85% 时，路由 KL 和长上下文是重点。

**OPSD 的机制是否写对？** 在未训练的 gist 上跑 rollout（ref 路径）→ 原始序列上的老师 → 按记录的选择 replay。成本：几分钟，插在阶段一试验之间跑。replay 与 rollout 的生成 token log 概率最大绝对差 < 1e-3（fp32）、全部展开时 KL < 1e-3 为通过；任一不过都是 bug，修复而不是调参。

**两层树的阶段一能不能学？** 0.5B、chunk4-chunk4、8K 训练、约 500 步（8 × 8K），8K 与 16K 两个长度评测，三个配置：方案 D 两级同时训（两级共用 gist LoRA）、方案 D 分阶段训（每级一套 gist LoRA，第二阶段冻结第一级）、方案 B 两级同时训，LoRA 都取 r = 32，都开路由 KL（小 λ\_r）和内容相关的 β，并记录各参数组的逐层梯度范数。成本：每个配置 L4 上 2–6 小时，H100 上不到 1 小时。通过条件：两级 MSE 都下降，不展开时的后缀 KL 比第 0 步下降 ≥ 30%，方案 D 的后缀 KL 不超过方案 B 的 1.2 倍。方案 D 超过 1.2 倍说明容量不够，依次试更大的 r、`gist_mlp`、读侧 LoRA；方案 B 逐层 MSE 更低而后缀 KL 更高，说明出现了退化解；两级 MSE 都不降，停下来重新考虑阶段一。

**OPSD 的训练信号是否有效？** 接着上一项的方案 D 模型，4K 上下文、生成 128 个 token、batch 8、约 300 步。成本：L4 上 3–12 小时（受 ref decode 的开销限制，H100 只快约 2 倍）。生成 token 的平均反向 KL 下降 ≥ 20%、每步展开数变化 < 5% 为通过；KL 不动说明信号太弱，展开数上升说明预算有泄漏。

执行顺序：噪声底 → 路由上限 → OPSD 机制 → 阶段一 → OPSD 训练信号。落在阈值附近的结论，先补种子再下结论。

## Milestones

spike 全部通过后按顺序执行；每一项的验收标准满足后才进入下一项。先在 Qwen2.5-0.5B-Instruct 上完成全部流程，再迁移到 7B（H100）。spike 的代码在 milestone 中沿用，规模扩大时需要新增的只有三处：长上下文 replay 迁到 flex、7B 用 CUDA 路径 rollout 时的选择记录（选择在 kernel 之前的 Python 里完成，不需要改 kernel）、多卡的 rollout 调度。spike 只跑到 8K–16K，长上下文的误差累积要在 milestone 中重新检查。

- [ ] **M0 基线与流程验证**

  环境：L4 上 `~/ssa_venv`（torch 2.6、transformers 4.57.1），跑 CPU 测试、`check_correctness.sh`、`tests/gpu_validate_flex.py`。用 `gsa-ft-qwen2-chunk4-chunk4`（7B）在 LongBench 小子集与 RAG（NQ）上验证评测流程（与论文误差在 1 分以内），`ATTN_IMPL=ref` 与 CUDA 路径一致。0.5B 上拿到 full attention 的分数，并按原版配置（全量、NTP）训一个 chunk4-chunk4 的原版 SSA 作为主对照。
- [ ] **M1 阶段一：逐层 MSE（单层 gist）**

  实现 `o_raw` 与 `o_g1` 的逐层 relMSE、方案 D（gist 专属 LoRA）、冻结原模型、NTP 开关；D-kv 与全秩方案 D 的对照在这里做。验收：所有开关取默认值时与原版 CPT 的 loss 曲线一致；开启后逐层 MSE 稳定下降；单层 SSA 推理（不展开）质量不低于原版 CPT；NTP 对 MSE 的对照在这里做。
- [ ] **M2 阶段一：多层 gist 与老师链**

  加入 `o_hier` 与第二级 MSE、每个 summary 的 β。验收：两级 MSE 都下降；逐级误差可分别报告；H-SSA 推理质量不低于原版。
- [ ] **M3 路由 KL**

  每级真实 log mass 与全部 summary 打分的 KL。验收：没被选中的 summary 的 key 有非零梯度；每级路由召回高于无路由 loss 的版本。
- [ ] **M4 选择的记录与 replay**

  rollout 导出每步每层每组的选择；训练路径接受强制选择；长上下文下走 flex。验收：replay 算出的 log 概率与 rollout 记录的差异在数值误差内。
- [ ] **M5 OPSD trainer**

  rollout → 原始序列上的老师 → replay → 反向 KL + 路由 KL。验收：噪声底测试（全部展开）KL 在数值误差内为 0；训练中生成 token 的平均反向 KL 下降；展开数不随训练增长。
- [ ] **M6 探索与简化（可选）**

  Gumbel 探索；相邻层共享选择；上宽下窄的逐级 k。各自作为 ablation 报告。
- [ ] **M7 完整评估与系统测量**

  按「Evaluation」跑全部对照与 ablation；在 7B 上用 SSA 的 kernel 测 TTFT / TPOT；整理结果表。

## 下一阶段：深层树、decode 时的 gist 与 1M 上下文

目标是支持 4–5 层的 gist 树、1M 上下文，并让 decode 生成的 token 也被压缩。代码层面的改动不大；决定成败的是三件事：逐级路由的误差随层数累积、32K 以外没有可用的老师、1M 时 summary 本身的 KV 放不进显存。父 issue 是 #11。

**为什么要做深。** 每步 decode 都要对顶层的全部 summary 打分。4 叉树在 1M 上下文时，两层的顶层有 62.5K 个单元，五层只有 977 个；代价是从上到下多做三次选择。层数与分叉数一起决定这个权衡，4 叉五层与 8 或 16 叉三层哪个更好由下面的误差累积 spike 回答。

**第 1 步：代码泛化到 L 层（#12）。** 插入规则、mask、选择循环和路由 KL 已经按 level 循环；写死两层的是 decode 快路径（`attn_candidate.py` 的 `num_levels in (1, 2)`、选择 metadata 缓存、`_select_per_group` 的快分支）、阶段一的老师链（g1:raw、hier:g1）、β 的跨度 (4, 16) 和 gist 词表。泛化后老师链是 g<sub>l</sub>:g<sub>l−1</sub>，每层多一次 attention，可以每步随机抽一级训练来控制成本；β 的跨度取 4<sup>l</sup>。CUDA kernel 读的是选好的 key 集合，预计不需要改。验收：L = 4 时噪声底仍约 1e-9，L = 2 时所有输出与现在逐位相同。

**第 2 步：decode 时产生 gist（#13）。** 现在 decode 只在 mask 末尾补 1，生成的部分从不压缩，长推理输出的尾部线性增长且全部可见。改动有四处：decode 接上 `OnlineGistInserter`，每 4 个原始 token 强制喂入一个 gist（不采样，只写 KV），高层按同一规则级联；块关闭后，后续 query 看不到它的原始 token，只能通过展开读到；选择 metadata 缓存从「静态前缀」改成增量追加；OPSD 的 rollout 记录这些 gist 步，replay 带上它们，反向 KL 只在原始 token 上计算。验收的不变式是「边 decode 边插 gist」与「同一序列一次 prefill」的输出在浮点误差内一致。gist 与下一个原始 token 可以放进同一次前向（q_len = 2），避免多出约 33% 的 decode 步。spike 的 SFT 数据在回答部分已经按流式规则插入了 gist，阶段一训练时见过。

**第 3 步：误差累积的 spike（#14）。** S2 测得仅 meta-gist 一级就只保留 43–57% 的 oracle chunk；若每级都丢一半，五级的召回约 6%。在训练过的 L 层模型上，沿用 S2 的方法测 L = 2…5 的逐级 oracle mass 召回（32K–128K），比较三种缓解：高层更宽的 beam（RecSA 每级 256）、没选中的节点作为粗粒度项留在 softmax 中（RecSA 的做法）、每级的路由 KL。逐级召回明显衰减时，先解决路由，再加层。

**第 4 步：32K 以外的位置与老师（#15，待定）。** 原始 token 保持原位置，1M 位置超出 Qwen2.5-0.5B 训练过的 32K，需要 RoPE 外推（如 YaRN），且原模型自己也要能读这个长度。OPSD 的老师在 1M 上既贵又不可靠。候选：只在 ≤ 128K 做 OPSD，靠长度无关的树外推到 1M（前提是固定 k，现在的 k = ⌊k_len / 448⌋ + 1 在 1M 时约为 2233）；换到原生 1M 的模型（如 Qwen2.5-7B-Instruct-1M）；长程只用阶段一的局部老师 g<sub>l</sub>:g<sub>l−1</sub>。倾向是前一种加局部老师，在 1M 上用 RULER / NIAH 验证；目标模型与老师路线需要人来定。

**第 5 步：多级 offload（扩展 #10）。** L 层树的 summary 总数约为 N/3，1M 时约 33 万个 token 的 KV（7B 约 19 GB）。level-0 的 summary 也放到 host，GPU 只留 level ≥ 1（约 8.3 万个，约 4.8 GB），选中 level-1 单元后再取它的子节点；选择需要 L 级的 rank-select kernel，固定 k 是前提。

执行顺序：第 1、2 步在 L4 上做；第 3 步决定是否继续加层；第 4、5 步等 H100 和人的决定。OPSD 与 off-policy 路由 KL 对照的结果出来后，若 OPSD 不优于对照，第 2 步之后的训练方案要重新考虑。

## Evaluation

主对照是原版 SSA；每个 ablation 只改一个因素，每个配置至少 3 个随机种子（spike 除外），差距小于种子波动时不下结论。

**对照组**

| 对照 | 说明 |
| --- | --- |
| Full attention（原模型） | 质量上限，也是 OPSD 的老师 |
| 原版 SSA / H-SSA | CPT（NTP）+ selective FT（NTP），全量微调；0.5B 自己训，7B 用放出的 checkpoint |
| Mean pooling 块路由（HISA / MoBA 式） | 同预算下用块内 key 平均做代表向量，直接对应我们的 motivation |
| Full-PT | 与 SSA 论文一致的「继续预训练但不压缩」参照 |

**Ablation**

| 轴 | 变体 |
| --- | --- |
| 阶段一目标 | 纯 NTP（原版）/ 纯逐层 MSE / 两者相加 |
| 老师链 | 直接对 `o_raw` 匹配 / 逐级老师链；局部老师 / 全局老师 |
| 两级训练方式 | 同时训 / 分阶段训（第二阶段冻结第一级） |
| 各级的 gist LoRA | 共用（默认）/ 每级一套 / 共用 + 每级小 delta |
| 路由 KL | 无 / 只在阶段一 / 只在阶段二 / 两阶段都有 |
| 阶段二目标 | selective FT（NTP）/ OPSD / OPSD + NTP |
| 可训练参数 | 全量（原版）/ 方案 D（gist 专属 LoRA）/ D-kv（只 K、V）/ 全秩方案 D（0.5B）/ 方案 D + gist MLP / 方案 D + 读侧 LoRA / 方案 B（同秩全局 LoRA q/k/v/o）/ 全局 LoRA 只 v/o；LoRA 秩 r |
| summary 偏置 | 无 / 每级标量 b（初值 log n）/ 内容相关的 β（默认） |
| 路由拓扑 | 逐级自顶向下（默认）/ 跨级 best-first tree cut（离线） |
| 选择 | 确定性 top-k / Gumbel 探索；per-head 并集 / 组级；各层独立 / 相邻层共享；逐级 k 相同 / 上宽下窄 |
| 压缩比 | 8× / 16× / 32×（L、J 组合） |

**指标**

- 任务：LongBench、RAG（NQ 等多文档 QA）、passkey、多键 needle 与变量追踪（已知答案依赖哪些 token）。
- 对齐：后缀 logit KL、生成 token 上的反向 KL、逐 token KL 热力图（定位展开失败的位置）。
- 路由：每一级的召回与 mass 覆盖率，按上下文长度和 query 位置分桶（关注靠后位置的下降）。
- 系统：实际展开数、TTFT、TPOT（SSA 的 sparse prefill 与 flash-decode kernel）。

## Invariants, tests, and engineering rules

每个 PR 合入前都要通过下面的检查；任何一项失败，先修复再继续，不要用调参掩盖。

**必须保持的不变量**

1. **开关取默认值 = 原版 SSA**：loss、logits 与基准 commit `e3a9d26` 一致（同种子、同数据）；1–2 层的 mask 另与原仓库 `c47ca56` 对照一次。
2. **参考路径一致**：新逻辑先在 `ATTN_IMPL=ref` 的 eager 路径实现并测试，再对齐 CUDA / FlexAttention 路径；两者输出在数值误差内。
3. **replay 一致**：用 rollout 记录的选择 replay，得到的生成 token log 概率与 rollout 时记录的一致（fp32 下最大绝对差 < 1e-3）。
4. **噪声底为 0**：方案 D 下，学生展开全部 chunk 时原始 token 的计算与原模型相同，与老师在生成 token 上的 KL 在数值误差内为 0；不为 0 就是位置映射、mask 或精度的 bug。这里的「全部展开」指每个 chunk 都用原始 KV 代替 summary、被展开的 summary 不再可见；SSA 的 SG+SR 让被选中的 summary 与其原始 KV 同时可见，那样的全部展开会多出 summary key，噪声底不为 0，需要单独测量（见「Open questions」）。
5. **梯度覆盖**：开启路由 KL 后，没被选中的 summary 的 key 和 β 梯度非零；老师分支（`o_raw`、`o_g1` 作为目标时、OPSD 的老师）没有梯度；方案 D 下原始 token 的投影权重没有梯度。
6. **预算不漂移**：每步实际展开的 chunk 数由 adaptive top-k 决定，训练前后分布一致。

**数值与实现规则**

- 计算 log 概率与 KL 时关闭 TF32，logits 与 softmax 用 fp32；采样 temperature 1、不用 top-p / top-k。
- relMSE 的分母加 ε；对 attention sink 与 massive activation 维度单独检查，必要时归一化后再比较。
- 老师与学生尽量走同一条计算路径（同一个 kernel），只换 mask，降低数值噪声。
- 长上下文不要构造 dense T × T mask；用 FlexAttention 的 block mask 或 SSA 的元数据位图。

**协作规则**

- 一个功能一个分支、一个 PR；PR 描述写清楚改了哪些文件、加了哪些开关、跑了哪些检查及结果。
- 不修改 `kernels/` 的行为，除非任务明确要求；不改变 upstream 的默认配置。
- 实验结果写入统一的结果表（配置、种子、指标、commit），不只在日志里。

## Open questions and risks

原来的待确认问题都已解决：位置规则已由 SSA 的 RoPE 修正实现（见「Design decisions」）；阶段一的「逐层」就是本文的设计（同一组 q/k/v、多 mask、梯度跨层连通），老师用带 gist 的序列；没有比公开版更新的内部分支；`masked_kl_vocab` 未被调用，`--klloss` 的逐层对齐默认关闭；起步模型 Qwen2.5-0.5B-Instruct，H100 几天内到；训练数据沿用 SSA，agent 数据需要时自己找。新的未定项出现时，停下来问人。

**待定：展开后 summary 是否仍可见。** SSA 的 SG+SR 让被选中的 summary 和它的原始 KV 同时可见；替换式展开则只保留原始 KV。前者是原版推理的行为，后者才让噪声底严格为 0。S1 的数据：未训练时让所有 summary 都可见，平均反向 KL 为 3–5 nats，而替换式约 1e-9；训练后 SG+SR 的代价需要重新测。OPSD 机制的 spike 两种都测，用两者的 KL 差与质量差决定阶段二用哪一种。

**风险**

| 风险 | 早期信号 | 应对 |
| --- | --- | --- |
| 纯 MSE 学不出足够好的 gist | M1 的不展开推理质量低于原版 CPT | 加入 NTP（λ\_ntp > 0）；每块多个 gist；β |
| 冻结的读侧读不懂 gist KV（方案 D） | 方案 D 的后缀 KL 超过方案 B 的 1.2 倍 | β；更大的 r；`gist_mlp`；只在读 summary 时生效的读侧 LoRA |
| gist 写的一侧学得慢 | W<sup>g</sup><sub>Q/O</sub> 的梯度范数比 W<sup>g</sup><sub>K/V</sub> 低几个数量级；逐层 MSE 降到平台后不动 | 写的一侧用更大的学习率；对比 D-kv；更长的训练 |
| β 学成与 key 无关的常数 | u 的范数一直接近 0、各 chunk 的 β 几乎相同 | 检查路由 KL 的权重 λ\_r；看 needle 类 chunk 的 β 是否高于均值 |
| 局部老师的退化解（方案 B、全量） | 逐层 MSE 下降、后缀 KL 不降或上升 | 方案 D；或加 NTP / 全局老师 |
| 局部老师偏离原模型 | 逐层 MSE 下降但下游指标不升 | 依赖阶段二 OPSD 对齐原模型；对比不同层的误差；全局老师 ablation |
| 两级老师链目标移动 | 第二级 MSE 震荡或明显慢于第一级 | 分阶段训练 |
| 树的误差累积 | 靠后位置、长上下文的路由召回明显下降 | 上宽下窄的逐级 k；路由 KL；分级、分长度报告 |
| 训练与推理数值不一致 | 噪声底不为 0；replay 不一致 | 统一计算路径；fp32 logits；关 TF32 |
| OPSD 鼓励多展开 | 实际展开数上升 | 固定 adaptive top-k；监控展开数 |
| 长程 OPSD 不稳定 | KL 震荡或熵塌缩 | 截断 rollout；降低学习率；先短后长的课程 |
| 0.5B 上 rollout 太慢 | ref decode 每 token 远超 100 ms | 短上下文、更大 batch；必要时把 kernel 模板化到 D=64（需要明确同意改 `kernels/`） |
| 深层树的召回逐级相乘 | L 增大时逐级 oracle mass 召回明显下降 | 高层更宽的 beam；未选节点作为粗粒度项留在 softmax；每级路由 KL；更大的分叉、更少的层 |
| decode 时的 gist 让缓存失效或行为不一致 | 边 decode 边插 gist 与一次 prefill 的输出不一致 | 增量追加的选择缓存；以一致性测试为门槛 |
| 32K 以外没有可靠的老师 | 原模型在长桶上的质量本身就低 | ≤ 128K 上做 OPSD 并固定 k；局部老师；原生 1M 的模型 |
| 上限为原模型 | 无法复现 SSA 在 RAG 上超过 full attention 的收益 | 作为 OPSD + NTP 的 ablation，不作为承诺 |

## References

- [Simplified Sparse Attention via Gist Tokens (SSA)](https://arxiv.org/abs/2604.20920) · [代码](https://github.com/yuzhenmao/simplified-sparse-attention)
- [HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention](https://arxiv.org/abs/2603.28458)
- [Neural Garbage Collection](https://arxiv.org/abs/2604.18002)（replay mask、RL 训练删除决定）
- [On-Policy Distillation, Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [LoLCATs](https://arxiv.org/abs/2410.10254)（逐层 attention 输出 MSE、分块训练、之后 LoRA）
- [Zyphra Hierarchical Attention (ZHA)](https://zyphra.com/our-work/zyphra-hierarchical-attention)（每层节点 mass 的 KL）
- [FlexAttention](https://pytorch.org/blog/flexattention/)
- [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)
- 按标题检索：Fast KV Compaction via Attention Matching（mass 匹配与每个 key 的 bias，β 的先例）、On-Policy Self-Distillation（Zhao 等，2026）、DeepSeek-V3.2 / DSA、ActivationBeacon（beacon 专属 attention 投影，方案 D 的先例）、KVLink、MoBA、NSA
