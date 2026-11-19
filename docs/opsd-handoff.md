# Handoff: Hierarchical Gist Compression + Selective Unfolding with OPSD (built on SSA)

Nov 19, 2026 · @Someone

## TL;DR 与使用方式

在 SSA（Simplified Sparse Attention）的代码上，把它的两个训练阶段换成：阶段一用**逐层 MSE** 学多层 gist 压缩，阶段二用 **OPSD**（on-policy self-distillation）训练逐级的 selective unfolding；原模型尽量冻结，只训练 gist embedding、LoRA 和 summary 的打分。

- **阶段一（compression）**：插入 gist 与 meta-gist，用 gist causal mask 挡住原始 token；在同一次前向里，每层用同一组 q/k/v 计算多个 mask 下的 attention，按「老师链」逐级做相对 MSE。
- **阶段二（indexing）**：学生用自己的 gist 打分做逐级 top-k 展开并生成；老师是原模型、读完整 KV；学生按记录的选择 replay，loss = 生成 token 上的反向 KL + 每一级的路由 KL（标签是真实 attention mass）。
- **代码**：fork SSA；树和 top-k 选择已存在，主要新写逐层 MSE、选择的记录与 replay、OPSD trainer、路由 KL、可选 Gumbel 探索。

**给 agent 的使用方式**：按「Milestones」逐项执行，每完成一项先跑「Invariants」里的检查再继续；所有新功能都用开关控制、默认关闭，关闭时行为必须与原版 SSA 完全一致。遇到「Open questions」里的未定项，停下来问人，不要自行假设。

## Context and motivation

块级稀疏 attention 的路由需要一个块的代表向量；最常见的 mean pooling 会把每个 token 自己的内容和高频 RoPE 位置信息平均掉，导致包含关键 token 的块被漏选。SSA 用学出来的 gist token 代替平均，我们在它的基础上改训练方式。

- **mean pooling 的问题**：一个 chunk 内 key 的平均只剩「这一段的共同语境」，按序列连起来是一条低维轨迹，原始 key 则是一团；平均也稀释了峰值，而 attention 的块 mass 更接近 log-sum-exp。
- **HISA（COLM 2026）**：用 mean pooling 做块级粗筛、再在块内逐 token 精排，免训练替换 DSA 的 indexer。它的失败案例全部来自第一阶段漏选块，原因被归结为 mean pooling（实体关系被混在一起、短的因果片段被稀释、needle 被淹没），并把「学出来的块表示」列为未来工作。前缀越长，选择与 DSA 的重合度越低（约 97% → 79%）。
- **SSA（Mao、Li、Fox，Stanford）**：每 L 个 token 插一个 gist，用 gist causal mask 挡住原始 token，迫使模型把 chunk 信息写进 gist 的 KV；推理时用 q·k\_gist 打分，展开 top-k 个 chunk 的原始 KV。支持 meta-gist（H-SSA）做逐级展开。原版两阶段都用 next-token loss：继续预训练 + 可选的 selective finetuning。
- **我们改什么**：阶段一把 NTP 换成逐层的 attention 输出匹配（信号更密、可冻结原模型）；阶段二把 off-policy 的 selective finetuning 换成 on-policy 的 OPSD，并对每一级的 summary 打分加入以真实 mass 为标签的路由监督，补上「没被选中的 summary 收不到梯度」这个盲点。

## Goal and success criteria

目标：在同样的压缩比和展开预算下，用逐层 MSE + OPSD 训练出的多层 gist 与 selective unfolding，质量优于原版 SSA 的训练方式，并接近读完整 KV 的原模型。

- **质量**：在 LongBench、RAG（NQ 等）、passkey / 多键 needle 上，同压缩比（如 16×，chunk4-chunk4）下优于原版 SSA（CPT + selective FT），与 full attention 的差距明显缩小。
- **路由**：每一级的路由召回（选中的 chunk / 段是否覆盖真实 mass 最大的那些）高于原版 SSA，长上下文靠后位置的下降更小。
- **对齐**：阶段二之后，学生与老师在生成 token 上的平均反向 KL 显著下降，且噪声底（全部展开时的 KL）接近 0。
- **成本**：原模型冻结或只加小 rank LoRA；推理速度沿用 SSA 的 kernel，展开预算不因训练而变大。
- **非目标**：超越原模型。两个阶段都以原模型为上限；SSA 在 RAG 上超过 full attention 的去噪收益不在本项目的保证范围内（可作为加 NTP 的 ablation）。

## Stage 1: hierarchical gist compression with per-layer MSE

一次学生前向（层级 gist mask），每层用同一组 q/k/v 额外计算老师 mask 下的 attention，在后缀位置逐级做相对 MSE；梯度在层之间连通，只更新 gist embedding 和 LoRA。

**序列结构（两层）**：每 L 个原始 token 后插入一个 gist g，每 J 个 gist-chunk 对后插入一个 meta-gist G；序列分为压缩区（前缀）和后缀，loss 只在后缀上算。

```markdown
x1 x2 g1 x3 x4 g2 G1 | x5 x6 g3 x7 x8 g4 G2 | suffix ...
```

**mask 规则（与 SSA 相同）**：chunk 内 token 看本 chunk 与之前所有的 summary；gist 看自己的 chunk 与之前的 summary；gist 之后的 token 看不到该 chunk 的原始 token；G 之后的 token 看不到 G 管辖范围内的原始 token 和 gist，只看 G。

**每层的三种 attention（同一组 q/k/v）**：

1. `o_raw`：纯 causal mask，能看到全部原始 token；
2. `o_g1`：单层 gist mask，只能通过 gist 看远处；
3. `o_hier`：层级 mask，远处只能通过 meta-gist 看（推理时真实的路径，往下一层传的是它）。

**loss（老师链，每一级只学压缩紧挨着的下一级）**：

```latex
\mathcal{L}_1 = \sum_{\ell} \Big[\mathrm{relMSE}\big(o^{(\ell)}_{g1},\, \mathrm{sg}(o^{(\ell)}_{raw})\big) + \mathrm{relMSE}\big(o^{(\ell)}_{hier},\, \mathrm{sg}(o^{(\ell)}_{g1})\big)\Big]_{\text{suffix}} + \lambda_r \mathcal{L}_{route} + \lambda_{ntp} \mathcal{L}_{NTP}
```

- sg 表示 stop-gradient（老师只作为目标）；relMSE = ‖a − b‖² / ‖b‖²，按位置平均。
- **路由 loss（可选，推荐）**：用学生的 q 和原始 token 的 key 算每个 chunk、每个段真实的 log mass（块内 log-sum-exp），对所有 gist / meta-gist 的打分 q·k 做 KL。
- **NTP**：默认 λ\_ntp = 0（纯 MSE）；λ\_ntp > 0 作为 ablation，λ\_ntp > 0 且 MSE 关闭即原版 SSA。
- **可训练参数**：gist / meta-gist 的输入 embedding、LoRA（起点 r = 8，作用在 attention 的 q/k/v/o 上，另做只加 v/o 的对照）、每一级 summary logit 的标量 bias（补 mass）。原模型冻结。
- **起点配置**：L = 4、J = 4（16×，与 SSA 已放出的 checkpoint 对齐）；第一层不参与 summary 的打分。

## Stage 2: selective unfolding trained with OPSD

展开决定始终来自学生自己的打分 q·k\_summary；真实 mass 只作为训练时的标签；Gumbel 噪声只用于训练时的探索（可选）。

**一个训练步的数据流**

1. **学生生成（no grad）**：带 gist / meta-gist 的上下文做层级 mask 的 prefill；逐 token 生成时，在每一层（第一层除外）、每个 KV 组：对 meta-gist 打分取 top-k₂ 段，再对这些段里的 gist 打分取 top-k₁ 个 chunk；attention 只看被选中的 meta-gist、gist、chunk 原始 token（SSA 的 SG+SR）以及本地未压缩部分；temperature 1 采样。**记录每一步、每层、每组的选择索引**。
2. **老师打分（no grad）**：原模型、普通 causal mask、**不插 gist 的原始序列**（上下文 + 学生生成的 y），一次并行前向，得到生成 token 位置的分布 p\_T。需要把学生序列里生成 token 的位置映射到老师序列里。
3. **学生 replay（with grad）**：把「上下文 + y」一次性并行前向，每层每组每个位置**严格按第 1 步记录的选择**构造 mask（不重新选择）；得到 p\_S、每层对**所有** summary 的打分，以及用学生 q 与原始 key 算出的真实 chunk / 段 log mass（no grad）。
4. **loss**：

```latex
\mathcal{L}_2 = \underbrace{\frac{1}{|G|}\sum_{t \in G} \mathrm{KL}\big(p_S(\cdot\mid y_{<t}) \,\|\, p_T(\cdot\mid y_{<t})\big)}_{\text{reverse KL on generated tokens}} + \lambda_r \sum_{\text{level}} \mathrm{KL}\big(\mathrm{softmax}(\text{true log mass}) \,\|\, \mathrm{softmax}(q^\top k_{summary})\big)
```

5. **反向与更新**：反向 KL 的梯度经 attention 流到被选中的 summary、原始 KV 和 LoRA；路由 KL 的梯度流到**所有** summary 的 key（含没被选中的），补上离散选择拿不到梯度的盲点。

**关键设计约束**

- **预算固定**：top-k 用 SSA 的 adaptive top-k，不让 k 被 KL 优化（否则模型倾向于多展开）；记录每步实际展开数。
- **GQA**：默认沿用 SSA 的 per-head top-k 再组内取并集；组级（组内分数相加、一次 top-k）作为对照。
- **探索（可选）**：rollout 时在各级打分上加 Gumbel 噪声 g = −log(−log U) 后取 top-k；replay 必须用实际采样到的选择；推理时关闭。
- **树的误差累积**：上层 k 可大于下层（上宽下窄）；分别统计每一级的路由召回。
- **可选简化**：相邻层共享同一组选择，减少记录量和每层噪声（需验证质量）。

## Codebase

fork [SSA](https://github.com/yuzhenmao/simplified-sparse-attention)：逐级 top-k 选择、gist mask、生成循环和 CUDA kernel 已经存在；新代码集中在 attention 的训练路径、选择函数的记录 / replay 钩子和一个新的 OPSD trainer。

| 文件 | 已有内容 | 我们要做的 |
| --- | --- | --- |
| `src/data/gist.py` | 插入 gist / meta-gist，构造 gist mask | 新增老师 mask（纯 causal、单层 gist）的构造函数；对齐 gist 的 position id 规则 |
| `gist_qwen2.py`（`eager_attention_forward_prefill` / `_decoding` / `_prefill_chunk`） | 逐级打分与 `top_k_from_logits`、`level_meta`、adaptive top-k | 训练路径内计算多 mask 的 attention 与逐层 MSE；路由 KL；选择的记录与强制 replay；可选 Gumbel |
| `GistQwen2ForCausalLM.forward` | 训练的 `selective` 分支；多 mask 的 5D 分支；`masked_kl_vocab` | 把逐层 loss 汇总进总 loss；接受 replay 的选择 |
| `generation_utils.py`、`gist_caching.py` | 带 gist 的生成与 KV cache | rollout 时导出每步每层每组的选择 |
| `gist_cpt_trainer.py`、`gist_sft_trainer.py` | 原版两阶段训练 | 作为模板；阶段一加 LoRA、冻结原模型、NTP 开关 |
| 新文件 `gist_opsd_trainer.py` | — | rollout → 老师 → replay → 两个 loss → 更新 |
| `kernels/`、`attn_candidate.py` | CUDA 稀疏 decode kernel 与调度 | 不改；`ATTN_IMPL=ref` 作为参考路径做对齐 |

**仓库组织**：GitHub fork，`upstream` 指向原仓库并固定一个 commit；每个功能一个分支（`stage1-layerwise-mse`、`routing-kl`、`stage2-record-replay`、`stage2-opsd`、`explore-gumbel`），一个功能一个 PR。所有新功能用配置开关控制、默认关闭。

**建议的开关**：`layerwise_mse`、`lambda_ntp`、`lambda_route`、`use_lora` / `lora_r`、`summary_logit_bias`、`record_selection`、`replay_selection`、`gumbel_tau`（0 = 关）、`share_selection_across_layers`。

## Milestones

按顺序执行；每一项的验收标准满足后才进入下一项。先在 Llama-3.2-1B（SSA 已放出 checkpoint）上完成全部流程，再迁移到 Qwen2-7B-Instruct。

- [ ] **M0 复现原版 SSA**

  安装环境，用已放出的 checkpoint 在 LongBench / RAG 小子集上复现论文数字（误差在 1 分以内）；`ATTN_IMPL=ref` 与 CUDA 路径输出一致；读通 `src/data/gist.py`、三个 attention 函数、`forward` 的训练分支、生成循环。
- [ ] **M1 阶段一：逐层 MSE（单层 gist）**

  实现 `o_raw` 与 `o_g1` 的逐层 relMSE、LoRA、冻结原模型、NTP 开关。验收：`layerwise_mse=off, lambda_ntp=1` 时与原版 CPT 的 loss 曲线一致；开启后逐层 MSE 稳定下降；单层 SSA 推理（不展开）质量不低于原版 CPT。
- [ ] **M2 阶段一：多层 gist 与老师链**

  加入 `o_hier` 与第二级 MSE、每级 summary logit bias。验收：两级 MSE 都下降；逐级误差可分别报告；H-SSA 推理质量不低于原版。
- [ ] **M3 路由 KL**

  每级真实 log mass 与全部 summary 打分的 KL。验收：没被选中的 summary 的 key 有非零梯度；每级路由召回高于无路由 loss 的版本。
- [ ] **M4 选择的记录与 replay**

  rollout 导出每步每层每组的选择；训练路径接受强制选择。验收：replay 算出的 log 概率与 rollout 记录的差异在数值误差内。
- [ ] **M5 OPSD trainer**

  rollout → 原始序列上的老师 → replay → 反向 KL + 路由 KL。验收：噪声底测试（全部展开）KL ≈ 0；训练中生成 token 的平均反向 KL 下降；展开数不随训练增长。
- [ ] **M6 探索与简化（可选）**

  Gumbel 探索；相邻层共享选择；上宽下窄的逐级 k。各自作为 ablation 报告。
- [ ] **M7 完整评估与系统测量**

  按「Evaluation」跑全部对照与 ablation；用 SSA 的 kernel 测 TTFT / TPOT；整理结果表。

## Evaluation

主对照是原版 SSA；每个 ablation 只改一个因素，每个配置至少 3 个随机种子，差距小于种子波动时不下结论。

**对照组**

| 对照 | 说明 |
| --- | --- |
| Full attention（原模型） | 质量上限，也是 OPSD 的老师 |
| 原版 SSA / H-SSA | CPT（NTP）+ selective FT（NTP），用其 checkpoint 或按原配置重训 |
| Mean pooling 块路由（HISA / MoBA 式） | 同预算下用块内 key 平均做代表向量，直接对应我们的 motivation |
| Full-PT | 与 SSA 论文一致的「继续预训练但不压缩」参照 |

**Ablation**

| 轴 | 变体 |
| --- | --- |
| 阶段一目标 | 纯 NTP（原版）/ 纯逐层 MSE / 两者相加 |
| 老师链 | 直接对 `o_raw` 匹配 / 逐级老师链 |
| 路由 KL | 无 / 只在阶段一 / 只在阶段二 / 两阶段都有 |
| 阶段二目标 | selective FT（NTP）/ OPSD / OPSD + NTP |
| 可训练参数 | 全量 / LoRA（q/k/v/o）/ LoRA（只 v/o）；有无 summary logit bias |
| 选择 | 确定性 top-k / Gumbel 探索；per-head 并集 / 组级；各层独立 / 相邻层共享；逐级 k 相同 / 上宽下窄 |
| 压缩比 | 8× / 16× / 32×（L、J 组合） |

**指标**

- 任务：LongBench、RAG（NQ 等多文档 QA）、passkey、多键 needle 与变量追踪（已知答案依赖哪些 token）。
- 对齐：生成 token 上的反向 KL、逐 token KL 热力图（定位展开失败的位置）。
- 路由：每一级的召回与 mass 覆盖率，按上下文长度和 query 位置分桶（关注靠后位置的下降）。
- 系统：实际展开数、TTFT、TPOT（SSA 的 sparse prefill 与 flash-decode kernel）。

## Invariants, tests, and engineering rules

每个 PR 合入前都要通过下面的检查；任何一项失败，先修复再继续，不要用调参掩盖。

**必须保持的不变量**

1. **开关全关 = 原版 SSA**：所有新开关关闭时，loss、logits 与 upstream 固定 commit 一致（同种子、同数据）。
2. **参考路径一致**：新逻辑先在 `ATTN_IMPL=ref` 的 eager 路径实现并测试，再对齐 CUDA / FlexAttention 路径；两者输出在数值误差内。
3. **replay 一致**：用 rollout 记录的选择 replay，得到的生成 token log 概率与 rollout 时记录的一致。
4. **噪声底**：学生展开全部 chunk 时，与老师在生成 token 上的反向 KL ≈ 0；不为 0 先查位置映射、mask 与精度。
5. **梯度覆盖**：开启路由 KL 后，没被选中的 summary 的 key 梯度非零；老师分支（`o_raw`、`o_g1` 作为目标时、OPSD 的老师）没有梯度。
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

下列未定项会改变实现，agent 遇到时应停下来问人；风险表给出早期信号与应对。

**待确认（问 PhD）**

- gist / meta-gist 的 position id 规则：插入后原始 token 的位置是否被挤开？阶段一的老师用带 gist 的序列还是原始序列？
- 阶段一的「逐层」是否就是本文的设计（同一组 q/k/v、多 mask、梯度跨层连通），还是另有所指？
- 是否有比公开版更新的内部分支；`masked_kl_vocab` 等现有 KL 代码是否已用于某个实验。
- 目标模型与算力：Llama-3.2-1B 起步后，主实验是 Qwen2-7B-Instruct 还是更新的模型；可用 GPU 数量。
- 训练数据：沿用 SSA 的 RedPajama / FineWeb 与 ActivationBeacon / KVLink 的微调数据，还是加入代码或 agent trajectory。

**风险**

| 风险 | 早期信号 | 应对 |
| --- | --- | --- |
| 纯 MSE 学不出足够好的 gist | M1 的不展开推理质量低于原版 CPT | 加入 NTP（λ\_ntp > 0）；每块多个 gist；加 summary logit bias |
| 局部老师偏离原模型 | 逐层 MSE 下降但下游指标不升 | 依赖阶段二 OPSD 对齐原模型；对比不同层的误差 |
| 树的误差累积 | 靠后位置的路由召回明显下降 | 上宽下窄的逐级 k；路由 KL；分级报告 |
| 训练与推理数值不一致 | 噪声底不为 0；replay 不一致 | 统一计算路径；fp32 logits；关 TF32 |
| OPSD 鼓励多展开 | 实际展开数上升 | 固定 adaptive top-k；监控展开数 |
| 长程 OPSD 不稳定 | KL 震荡或熵塌缩 | 截断 rollout；降低学习率；先短后长的课程 |
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
- 按标题检索：Fast KV Compaction via Attention Matching（mass 匹配与 logit bias）、On-Policy Self-Distillation（Zhao 等，2026）、DeepSeek-V3.2 / DSA、ActivationBeacon、KVLink、MoBA、NSA
