"""Noise floor of a gisted sequence against the original model (spike S1).

The student is the original model with summary tokens inserted (untrained gist
embeddings) and every chunk unfolded. Its next-token distributions at raw positions are
compared with the original model reading the un-gisted text, as mean reverse KL
KL(p_student || p_teacher) per raw token, in fp32 with TF32 off.

Variants:
  replace   raw queries see every raw key and no summary key (replacement unfolding).
            RoPE positions skip summaries, so this should be ~0.
  sgsr      raw queries also see every summary key (SG+SR-style unfolding with all
            chunks selected); measures what extra summary keys cost when untrained.
  shift     counterfactual: the original model reading raw text at the positions raw
            tokens would get if summaries took positions (no gist tokens involved).

    python experiments/noise_floor.py --longbench_dir ~/longbench_data/narrativeqa \
        --lengths 512 2048 8192 --docs 4 --out results/spikes.csv
"""

import argparse
import csv
import datetime
import os
import subprocess

import datasets
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2ForCausalLM

from gist_qwen2 import GistQwen2ForCausalLM
from gist_utils import Global_data, add_gist_tokens
from src.data.gist_stream import insert_gists_with_map


def reverse_kl(student_h, teacher_h, lm_head, chunk=1024):
    """Per-token KL(p_s || p_t) from final hidden states, without holding [T, vocab] logits."""
    out = []
    for s in range(0, student_h.shape[0], chunk):
        ls = F.log_softmax(lm_head(student_h[s : s + chunk]).float(), -1)
        lt = F.log_softmax(lm_head(teacher_h[s : s + chunk]).float(), -1)
        out.append((ls.exp() * (ls - lt)).sum(-1))
    return torch.cat(out)


def gist_mask(src, summaries_visible):
    T = src.numel()
    causal = torch.ones(T, T, dtype=torch.bool, device=src.device).tril()
    keep = (src >= 0)[None, :] | torch.eye(T, dtype=torch.bool, device=src.device)
    if summaries_visible:
        keep = torch.ones_like(keep)
    return (causal & keep)[None, None]


@torch.no_grad()
def measure(student, teacher, raw_ids, gist_ids, chunk_size, group):
    dev = teacher.device
    toks, src = insert_gists_with_map(raw_ids, gist_ids, chunk_size, group)
    toks = torch.tensor([toks], device=dev)
    src = torch.tensor(src, device=dev)
    raw = torch.tensor([raw_ids], device=dev)
    lm_head = teacher.lm_head

    t_h = teacher.model(input_ids=raw).last_hidden_state[0]
    res = {}
    for name, vis in (("replace", False), ("sgsr", True)):
        s_h = student.model(
            input_ids=toks,
            attention_mask=torch.ones_like(toks),
            attention_mask_gist=gist_mask(src, vis),
        ).last_hidden_state[0]
        res[name] = reverse_kl(s_h[src >= 0], t_h, lm_head)
        del s_h
    shifted = torch.nonzero(src >= 0).flatten()[None]  # raw tokens' indices in the gisted sequence
    sh_h = teacher.model(input_ids=raw, position_ids=shifted).last_hidden_state[0]
    res["shift"] = reverse_kl(sh_h, t_h, lm_head)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--longbench_dir", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--docs", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=4)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--out", default="results/spikes.csv")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    teacher = (
        Qwen2ForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, attn_implementation="sdpa"
        )
        .to(dev)
        .eval()
    )
    student = (
        GistQwen2ForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, attn_implementation="sdpa"
        )
        .to(dev)
        .eval()
    )
    gist_ids = add_gist_tokens(tok, student, num_levels=2)
    Global_data.gist_token_id = gist_ids
    Global_data.chunk_size = [args.chunk_size, args.group]
    Global_data.train_attn = "dense"
    Global_data.pad_token_id = tok.pad_token_id

    docs = datasets.load_from_disk(os.path.expanduser(args.longbench_dir))
    docs = [d["context"] for d in docs.select(range(args.docs))]
    commit = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )
    today = datetime.date.today().isoformat()

    rows = []
    for L in args.lengths:
        acc = {}
        for doc in docs:
            ids = tok(doc, add_special_tokens=False).input_ids[:L]
            if len(ids) < L:
                continue
            for k, v in measure(
                student, teacher, ids, gist_ids, args.chunk_size, args.group
            ).items():
                acc.setdefault(k, []).append(v)
            torch.cuda.empty_cache()
        for k, vs in acc.items():
            v = torch.cat(vs)
            for metric, val in (("mean_rkl", v.mean()), ("max_rkl", v.max())):
                rows.append(
                    {
                        "spike": "S1",
                        "config": f"{k}_L{L}",
                        "seed": 0,
                        "metric": metric,
                        "value": f"{val.item():.3e}",
                        "commit": commit,
                        "date": today,
                    }
                )
                print(f"{k:8s} L={L:5d} docs={len(vs)} {metric}={val.item():.3e}")

    new = not os.path.exists(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
