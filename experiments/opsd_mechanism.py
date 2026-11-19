"""S4 first half (#4): the OPSD data flow on Qwen2.5-0.5B-Instruct with untrained gists.

For each prompt (the context part of a spike SFT row): rollout with recorded selections,
teacher on the raw sequence, replay with the recorded selections. Reports

  replay_max_abs_lp   max |log p_replay(y) - log p_rollout(y)| over generated tokens
  replay_max_abs_all  same over the whole vocabulary
  floor_replace       reverse KL vs teacher, every chunk unfolded, no summary visible
  floor_sgsr          same with every summary also visible
  rkl_sparse          reverse KL vs teacher of the actual (sparse) student, untrained
  keys_per_row        visible keys per generated row (mean over layers >= 1)
  ms_per_token        rollout wall time per generated token (ref path, incl. prefill)

and, on the first prompt, gradient coverage: reverse-KL gradient on summary keys that no
generated row saw (must be 0) and the share of those that routing KL reaches.

    ATTN_IMPL=ref python experiments/opsd_mechanism.py --data ~/spike_data/sft
"""

import argparse
import csv
import datetime
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datasets  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer, Qwen2ForCausalLM  # noqa: E402

import gist_opsd  # noqa: E402
from experiments.noise_floor import gist_mask  # noqa: E402
from gist_qwen2 import GistQwen2ForCausalLM  # noqa: E402
from gist_utils import Global_data, add_gist_tokens  # noqa: E402


@torch.no_grad()
def unfolded_logprobs(model, seq, n, vocab, summaries_visible):
    is_sum = torch.isin(seq, torch.tensor(Global_data.gist_token_id, device=seq.device))
    src = torch.where(is_sum, -1, 0)
    h = model.model(input_ids=seq[None], attention_mask=torch.ones_like(seq)[None],
                    attention_mask_gist=gist_mask(src, summaries_visible)).last_hidden_state[0]
    return gist_opsd._logprobs(model.lm_head(h[-n:]), vocab)


def grad_coverage(model, plain, ctx, y, steps, vocab):
    lp_t = gist_opsd.teacher_logprobs(plain, ctx, y, vocab)
    seq = torch.cat([ctx[0], y[:-1]])
    is_sum = torch.isin(seq, torch.tensor(Global_data.gist_token_id, device=seq.device))
    hpg = model.config.num_attention_heads // model.config.num_key_value_heads

    def unseen(st, layer):
        vis = st["masks"][layer][0]
        return ~vis.view(-1, hpg, *vis.shape[1:]).any(1).any(1) & is_sum   # [num_kv, K]

    lp_s, st = gist_opsd.replay(model, ctx, y, steps, vocab, probe=True)
    gist_opsd.reverse_kl(lp_s, lp_t).mean().backward()
    rkl_max = max(float(k.grad[0].abs().sum(-1)[unseen(st, l)].max()) for l, k in st["probe"].items())
    model.zero_grad()
    _, st = gist_opsd.replay(model, ctx, y, steps, vocab, route=True, probe=True)
    st["route"].backward()
    hit = [float(k.grad[0].abs().sum(-1)[unseen(st, l)].gt(0).float().mean()) for l, k in st["probe"].items()]
    model.zero_grad()
    teacher_grads = any(p.grad is not None for p in plain.parameters())
    return rkl_max, sum(hit) / len(hit), teacher_grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", required=True, help="converted spike SFT data (load_from_disk)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ctx_len", type=int, default=2048)
    ap.add_argument("--n_new", type=int, default=64)
    ap.add_argument("--out", default="results/spikes.csv")
    args = ap.parse_args()
    assert os.environ.get("ATTN_IMPL") == "ref", "0.5B (head_dim 64) needs ATTN_IMPL=ref"

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    vocab = len(tok)
    plain = Qwen2ForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                             attn_implementation="sdpa").to(dev).eval()
    model = GistQwen2ForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                 attn_implementation="sdpa").to(dev).eval()
    Global_data.gist_token_id = add_gist_tokens(tok, model, num_levels=2)
    model.selective = True
    Global_data.chunk_size, Global_data.sink_size, Global_data.num_previous_chunks = [4, 4], 3, 1
    Global_data.pad_token_id, Global_data.train_attn = tok.pad_token_id, "dense"

    rows = datasets.load_from_disk(os.path.expanduser(args.data))
    gen = torch.Generator(device=dev).manual_seed(0)
    res = {k: [] for k in ("replay_max_abs_lp", "replay_max_abs_all", "floor_replace",
                           "floor_sgsr", "rkl_sparse", "keys_per_row", "ms_per_token")}
    grad = None
    for i in range(args.n):
        r = rows[i]
        start = next(j for j, l in enumerate(r["labels"]) if l != -100)
        ctx = torch.tensor([r["input_ids"][:min(start, args.ctx_len)]], device=dev)
        torch.cuda.synchronize(); t0 = time.time()
        y, lp_roll, steps = gist_opsd.rollout(model, ctx, args.n_new, vocab, gen)
        torch.cuda.synchronize(); ms = (time.time() - t0) * 1000 / args.n_new
        with torch.no_grad():
            lp_rep, st = gist_opsd.replay(model, ctx, y, steps, vocab)
            lp_t = gist_opsd.teacher_logprobs(plain, ctx, y, vocab)
        idx = torch.arange(len(y), device=dev)
        seq = torch.cat([ctx[0], y[:-1]])
        res["replay_max_abs_lp"].append(float((lp_rep[idx, y] - lp_roll[idx, y]).abs().max()))
        res["replay_max_abs_all"].append(float((lp_rep - lp_roll).abs().max()))
        res["floor_replace"].append(float(gist_opsd.reverse_kl(unfolded_logprobs(model, seq, len(y), vocab, False), lp_t).mean()))
        res["floor_sgsr"].append(float(gist_opsd.reverse_kl(unfolded_logprobs(model, seq, len(y), vocab, True), lp_t).mean()))
        res["rkl_sparse"].append(float(gist_opsd.reverse_kl(lp_roll, lp_t).mean()))
        res["keys_per_row"].append(float(torch.stack([m.sum(-1).float().mean() for m in st["masks"][1:]]).mean()))
        res["ms_per_token"].append(ms)
        print(f"prompt {i}: ctx {ctx.shape[1]} " + " ".join(f"{k}={v[-1]:.3e}" for k, v in res.items()), flush=True)
        if i == 0:
            grad = grad_coverage(model, plain, ctx, y, steps, vocab)
            print(f"grad: reverse-KL max on unselected summaries={grad[0]:.3e}, "
                  f"routing reaches {grad[1]:.1%} of them, teacher has grads={grad[2]}", flush=True)
        torch.cuda.empty_cache()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    today = datetime.date.today().isoformat()
    out = [{"spike": "S4a", "config": f"ctx{args.ctx_len}_new{args.n_new}_n{args.n}", "seed": 0,
            "metric": k,
            "value": f"{(max(v) if k.startswith('replay') else sum(v) / len(v)):.3e}",
            "commit": commit, "date": today} for k, v in res.items()]
    out += [{"spike": "S4a", "config": "grad_prompt0", "seed": 0, "metric": m, "value": f"{float(v):.3e}",
             "commit": commit, "date": today}
            for m, v in zip(("rkl_grad_unselected_max", "route_grad_unselected_frac", "teacher_has_grad"), grad)]
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        if new:
            w.writeheader()
        w.writerows(out)
    for o in out:
        print(o["config"], o["metric"], o["value"])


if __name__ == "__main__":
    main()
