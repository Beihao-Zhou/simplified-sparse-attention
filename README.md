# SSA - Simplified Sparse Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/yuzhenmao/simplified-sparse-attention/blob/main/LICENSE)
### [Paper](https://arxiv.org/abs/2604.20920) | [HuggingFace Collection](https://huggingface.co/collections/gist-sparse-attention/ssa)

Learnable context compression and selective unfolding for efficient long-context
LLM inference. The model inserts gist tokens that summarize local chunks during
training; at decode time, the query scores those gist tokens, unfolds the most
relevant chunks, and attends over the selected sparse context without changing
the base model architecture or using an external index.

- Query-adaptive: different prompts unfold different context chunks.
- Trainable: gist tokens are learned end-to-end under the standard next-token loss.
- Hierarchical: meta-gists summarize lower-level gists for long contexts.
- CUDA-optimized: sparse prefill and flash-decode kernels are used during inference.

## Models & Datasets

All checkpoints and training datasets are available on the [HuggingFace collection](https://huggingface.co/collections/gist-sparse-attention/ssa).

(PT = continued pretraining (`gist_cpt_trainer.py`); FT = selective finetuning (`gist_sft_trainer.py`).)


## Repository Structure

```text
simplified-sparse-attention/
├── gist_qwen2.py                 # SSA Qwen2 model
├── gist_llama.py                 # SSA Llama model for RAG experiments
├── attn_candidate.py             # CUDA sparse attention dispatcher
├── generation_utils.py           # gist-aware generation loop
├── gist_caching.py               # gist KV-cache helpers
├── gist_utils.py                 # shared runtime config and selection helpers
├── gist_cpt_trainer.py           # continued pretraining
├── gist_sft_trainer.py           # selective finetuning
├── gist_sft_trainer_link.py      # link-token SFT variant for RAG
├── gist_trainer_long.py          # long-context training driver
├── src/
│   ├── common.py
│   └── data/
│       ├── gist.py               # gist insertion and mask construction
│       └── gist_input_preprocessor.py
├── kernels/
│   └── sparse_decode_attn.cu     # CUDA sparse decode kernel
├── benchmark/
│   ├── Longbench_exp/            # LongBench prediction, evaluation, and included data
│   │   ├── LongBench/            # bundled datasets.load_from_disk task folders
│   │   └── longbench_config/     # prompts, max lengths, model paths
│   ├── RAG_exp/                  # multi-document RAG evaluation
│   │   └── nq-open-10_0.jsonl    # bundled NQ-open RAG data
│   └── Passkey_exp/              # passkey retrieval evaluation
├── data_process/                 # optional data preparation utilities
├── run_real_benchmark.sh         # optimized LongBench latency + accuracy run
├── run_longbench.sh              # quick LongBench run
├── run_rag.sh                    # RAG evaluation entrypoint
├── stage*.json                   # DeepSpeed configs
├── step*.yaml                    # training configs
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
export TORCH_CUDA_ARCH_LIST=9.0  # H100; set this to your GPU architecture
```

The CUDA decode kernel in `kernels/sparse_decode_attn.cu` is JIT-compiled on
first use through `torch.utils.cpp_extension.load`; there is no separate build
step.

## Included Data

LongBench data is bundled under `benchmark/Longbench_exp/LongBench/<task>/`
in `datasets.load_from_disk` format.

The NQ-open RAG data used by `benchmark/RAG_exp/nq_*.py` is bundled at
`benchmark/RAG_exp/nq-open-10_0.jsonl`.

## Usage

Run the optimized LongBench path:

```bash
bash run_real_benchmark.sh
```

Or call the prediction script directly:

```bash
DECODE_SELECT=group PREFILL_KERNEL=flex \
  python benchmark/Longbench_exp/longbench_pred_qwen.py \
    --model qwen2-7b-instruct \
    --datasets triviaqa \
    --ckpt_path <GSA-FT-Qwen2-7B-Instruct-chunk4-chunk4> \
    --name smoke \
    --selective \
    --chunk_size 4 4 \
    --meta_gist 1
```

Useful inference knobs:

| variable | default | meaning |
|---|---:|---|
| `ATTN_IMPL` | `sparse` | `sparse` runs the CUDA sparse path; `ref` forces the eager reference (debug). |
| `PREFILL_KERNEL` | `dense` | `flex` enables block-sparse permuted prefill; `dense` uses the compile-compatible path. |
| `DECODE_SELECT` | `head` | `head` = per-head top-k then union within the GQA group (the paper's grouped unfolding; allclose to the eager reference). `group` = sum the group heads' scores, one top-k per group (a different, cheaper selection; F1-validated, not allclose). Same tpot uncompiled; pair `group` with `DECODE_COMPILE` for a lower tpot floor. |
| `DECODE_COMPILE` | `0` | Enables torch.compile for lower decode launch overhead. |
| `DECODE_GMAX_BUCKET` | `0` | Buckets decode shapes to improve graph reuse when compilation is enabled. |

Training starts with continued pretraining, then selective finetuning:

```bash
python gist_cpt_trainer.py
python gist_sft_trainer.py
```

## Validated Results

LongBench Triviaqa F1 for the CUDA sparse attention path:

| model | task (N=200) | score |
|---|---|---:|
| SSA 16x (chunk16, single-level) | TriviaQA | 88.43 |
| H-SSA 16x (chunk4-chunk4) | TriviaQA | 90.06 |

On H100 with GSA-FT-Qwen2-7B chunk4-chunk4, sparse prefill reduces attention
work substantially at long context, and sparse decode keeps per-token latency
flat as context grows.

## Citation

```bibtex
@misc{mao2026ssa,
      title={Simplified Sparse Attention via Gist Tokens}, 
      author={Yuzhen Mao and Michael Y. Li and Emily B. Fox},
      year={2026},
      eprint={2604.20920},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.20920}, 
}
```
