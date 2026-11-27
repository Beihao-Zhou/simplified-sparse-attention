# 0004 选择用的 key 与 attention 用的 key 解耦

状态：已接受（2026-11-26）。相关 issue：#11、#34、#35、#44、#45。

## 决定

**每层加一个 route head，只负责给 beam 选择打分；attention 仍用模型自己的 k_gist 和 β。** 对每个 KV 组，f 是 summary 的 attention 输入 x，拼上它子节点 x 的均值：

    h = gelu(W1 LayerNorm(f))
    route key = k_gist + RoPE(B h)
    route bias = β + v · h + c[level]

B、v、c 初始化为 0，所以没训练的 head 与模型的选择完全相同。实现在 `gist_route.py`；训练时的 flex 和 dense 路径都通过 `selection_inputs` 取选择用的 key，gather 和 attention 仍用 k_gist。

每个节点一个 route key（m = 1）。多个 key 用 logsumexp 合并的实现保留（只走 PyTorch 参考路径），但不作为默认。

## 背景

在 32K 上，差距几乎全部来自 router。gate32h（coarse + hybrid）用 learned router 时 decode KL 是 0.379；换成按真实 attention mass 选择，同一套 summary 降到 0.054，passkey 从 0.13 / 0.33 升到 0.93 / 0.87。

**同一个 key 很难同时服务两个目标。** attention 要的 key，是让 summary 在 softmax 里替代它覆盖的原始 token；选择要的 key，是把子树 mass 大的节点排在前面。level scope 下 routing KL 的权重加到 1，suffix KL 从 0.046 升到 0.120（#34 R4），说明在同一个 key 上推 routing 会伤到 attention。

**离线证据。** gate32h 冻结，每层每个 KV 组单独训练一个 router，目标是真实 mass。在 9–17 层的 heldout context 上：

| router | heldout beam |
|---|---|
| 模型自己的 key | 0.470 |
| x 的线性映射 | 0.498 |
| x 过一层 MLP | 0.546 |
| x 加子节点均值，过一层 MLP（采用） | 0.615 |
| 同上，每个节点 4 个 key | 0.642 |
| 逐 context 直接拟合（上限，不能部署） | 0.853 |

子节点均值带来的增量（+0.07）比换成非线性（+0.05）更大，而且 10 个层与 KV 组的组合里，每一个都是它最高。

**端到端。** 1–23 层都换上离线训练的 head，heldout beam 平均从 0.517 升到 0.668。32K 的 select_gate 结果：

| 设置 | decode KL | all KL |
|---|---|---|
| 模型的 key，k 16 | 0.379 | 0.359 |
| 模型的 key，k (16, 48) | 0.217 | 0.180 |
| route head，k 16 | 0.211 | 0.183 |
| route head，k (16, 48) | 0.124 | 0.084 |
| oracle，k 16 | 0.054 | 0.009 |

route head 在 k 16 时与模型 key 在 k (16, 48) 时相当，每步读的量不变。

## 考虑过的其他做法

**只放大选择预算。** 上层 k 放大几乎不增加读量，而且拿到了大部分收益，所以与 route head 叠加使用。但 32K 上每级都放大到 48，all KL 仍是 0.135，补不上 router 的差距。

**只改 routing loss 的范围或梯度**（free-node、detach q、oracle_children）。最好的 oracle_children 在 λ = 1 时，8K 的 beam 只从 0.72 升到 0.75。

**每个节点多个 key。** m = 4 只比 m = 1 好 0.015，decode 却要多读 4 倍的 route key，也走不了 Triton 的选择 kernel。

**另起一个独立的小网络做 router**，不以 k_gist 为基础。没有试：以 k_gist 为残差基础，未训练时等价于现状，可以逐层替换、逐层验证。

## 后果

- **decode 要多存一份 key。** 每个 summary 节点多存一个 route key（与 k_gist 同形）和一个 bias。summary 约占 N/3 个位置，所以总 KV 增加不到 1/6；decode 读的 key 个数不变。CUDA kernel 要能分开读选择用的 key 和 attention 用的 key（#45）。
- **子节点均值在节点关闭时才能算出。** prefill 时一次算好；decode 时生成的节点要在线累加，节点关闭时再过一次 head（#45）。
- **离线训练已经饱和（0.64 附近），离直接拟合的 0.85 还远。** 剩下的差距在冻结的 x 里缺少 routing 需要的信息，不在 head 的容量，所以下一步在阶段一里联合训练 head（#44）。
- **重开条件。** 如果联合训练后，route head 的 beam 不比模型自己 key 的 routing KL 训练更好，就撤掉 head，回到单一 key。
