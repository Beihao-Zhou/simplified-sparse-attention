"""Stage 2: OPSD on a stage-1 checkpoint (spike S4, second half).

Each step, for --batch prompts (context parts of spike SFT rows): rollout with recorded
selections (no grad, ATTN_IMPL=ref) -> base model on the raw sequence -> replay with the
recorded selections (grad) -> mean reverse KL on generated tokens + lambda_route x routing
KL -> update of the gist LoRA, gist embeddings and beta. The unfolding budget comes from
the selector (adaptive top-k) and is not optimized; every step logs keys per generated
row, and every --check_every steps a replay-vs-rollout consistency check.

    ATTN_IMPL=ref python gist_opsd_trainer.py --stage1 runs/s3_d_joint/adapters.pt \
        --data ~/spike_data/sft --steps 300 --out runs/s4b
"""

import argparse
import json
import math
import os
import time

import datasets
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

import gist_opsd
from gist_lora import apply_lora, apply_summary_bias, summary_spans
from gist_qwen2 import GistQwen2ForCausalLM
from gist_utils import Global_data, add_gist_tokens
from src.data.gist_stream import reinsert_gists


def load_stage1(path, model_name, device):
    ck = torch.load(path, map_location="cpu")
    a = ck.get("stage1_args") or ck["args"]
    if "lora" not in a:  # OPSD checkpoint saved before stage1_args was recorded
        a = torch.load(a["stage1"], map_location="cpu")["args"]
    tok = AutoTokenizer.from_pretrained(model_name)
    vocab = len(tok)
    model = GistQwen2ForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    levels = a.get("levels", 2)
    gist_ids = add_gist_tokens(tok, model, num_levels=levels)
    assert gist_ids == ck["gist_ids"], (gist_ids, ck["gist_ids"])
    Global_data.gist_token_id = gist_ids
    Global_data.chunk_size, Global_data.sink_size, Global_data.num_previous_chunks = (
        [4] * levels,
        a["sink"],
        1,
    )
    # levels 0 and 1 get the adaptive k inside the eager path; higher levels read this
    Global_data.top_k = [3] * max(3, levels)
    Global_data.pad_token_id, Global_data.train_attn = tok.pad_token_id, "dense"
    groups = apply_lora(model, a["lora"], r=a["r"], per_level=a.get("per_level", False))
    groups["beta"] = apply_summary_bias(model, spans=summary_spans(levels)) if a.get("beta") else []
    state = {k: v for k, v in ck["state"].items() if k != "gist_embed"}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all("lora_" not in k and "summary_bias" not in k for k in missing), missing
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        emb[gist_ids] = ck["state"]["gist_embed"].to(device=device, dtype=emb.dtype)
    model.selective = True
    return tok, vocab, model, groups


def regist(ids):
    """Spike data has 2 levels; an L-level checkpoint needs its prompts re-gisted."""
    if len(Global_data.gist_token_id) <= 2:
        return ids
    return reinsert_gists(ids, Global_data.gist_token_id, 4, 4, sink=Global_data.sink_size)


def complete(ids):
    """ids gisted to completion (the summaries a truncation cut off are appended), as
    rollout(insert_gists=True) needs to continue the stream."""
    return reinsert_gists(ids, Global_data.gist_token_id, 4, 4, sink=Global_data.sink_size)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--stage1", required=True, help="adapters.pt from gist_stage1_trainer.py")
    ap.add_argument("--data", required=True, help="converted spike SFT data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx_len", type=int, default=4096)
    ap.add_argument("--n_new", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lambda_route", type=float, default=0.1)
    ap.add_argument(
        "--rkl_weight",
        type=float,
        default=1.0,
        help="0 with --lambda_route > 0: routing-only control",
    )
    ap.add_argument(
        "--y_source",
        choices=["student", "data"],
        default="student",
        help="data: off-policy control, generated rows are the row's own continuation",
    )
    ap.add_argument(
        "--decode_gists",
        action="store_true",
        help="insert summaries into the generated text as it grows (#13)",
    )
    ap.add_argument("--check_every", type=int, default=50)
    ap.add_argument(
        "--holdout",
        type=int,
        default=16,
        help="leave the last N eligible prompts out of training (experiments/opsd_eval.py uses them)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    assert os.environ.get("ATTN_IMPL") == "ref", "0.5B (head_dim 64) needs ATTN_IMPL=ref"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(args.seed)

    tok, vocab, model, groups = load_stage1(args.stage1, args.model, dev)
    s1 = torch.load(args.stage1, map_location="cpu")
    stage1_args = s1.get("stage1_args") or s1["args"]
    model.eval()
    teacher = (
        Qwen2ForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, attn_implementation="sdpa"
        )
        .to(dev)
        .eval()
    )
    params = [p for k in ("lora_qo", "lora_kv", "embed", "beta") for p in groups[k]]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / args.warmup)
        * 0.5
        * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)),
    )
    rows = datasets.load_from_disk(os.path.expanduser(args.data))
    prompts, conts = [], []
    gist_set = set(Global_data.gist_token_id)
    for r in rows:
        start = next(j for j, l in enumerate(r["labels"]) if l != -100)
        if start >= args.ctx_len:
            p = regist(r["input_ids"])[: args.ctx_len]
            prompts.append(complete(p) if args.decode_gists else p)
            conts.append([t for t in r["input_ids"][args.ctx_len :] if t not in gist_set])
    if args.holdout:
        prompts, conts = prompts[: -args.holdout], conts[: -args.holdout]
    print(f"{len(prompts)} prompts of {args.ctx_len} tokens ({args.holdout} held out)", flush=True)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    order = torch.randperm(
        len(prompts), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()

    for step in range(1, args.steps + 1):
        t0 = time.time()
        agg = {"rkl": 0.0, "route": 0.0, "keys_per_row": 0.0}
        for i in range(args.batch):
            j = order[((step - 1) * args.batch + i) % len(prompts)]
            ctx = torch.tensor([prompts[j]], device=dev)
            forced = None
            if args.y_source == "data":
                forced = torch.tensor(conts[j][: args.n_new], device=dev)
            n = args.n_new if forced is None else forced.numel()
            y, lp_roll, steps = gist_opsd.rollout(
                model, ctx, n, vocab, gen, forced=forced, insert_gists=args.decode_gists
            )
            lp_t = gist_opsd.teacher_logprobs(teacher, ctx, y, vocab)
            lp_s, st = gist_opsd.replay(model, ctx, y, steps, vocab, route=args.lambda_route > 0)
            rkl = gist_opsd.reverse_kl(lp_s, lp_t).mean()
            loss = args.rkl_weight * rkl + (
                args.lambda_route * st["route"] if args.lambda_route > 0 else 0
            )
            (loss / args.batch).backward()
            agg["rkl"] += float(rkl) / args.batch
            agg["route"] += float(st["route"]) / args.batch if args.lambda_route > 0 else 0.0
            agg["keys_per_row"] += (
                float(torch.stack([m.sum(-1).float().mean() for m in st["masks"][1:]]).mean())
                / args.batch
            )
            if args.check_every and (step - 1) % args.check_every == 0 and i == 0:
                yr = gist_opsd.raw_tokens(y)
                idx = torch.arange(len(yr), device=dev)
                agg["replay_max_abs_lp"] = float(
                    (lp_s.detach()[idx, yr] - lp_roll[idx, yr]).abs().max()
                )
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        rec = {
            "step": step,
            "lr": sched.get_last_lr()[0],
            "sec": round(time.time() - t0, 1),
            "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            **agg,
        }
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
    sd = {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if "lora_" in n or "summary_bias" in n
    }
    sd["gist_embed"] = model.get_input_embeddings().weight.detach()[Global_data.gist_token_id].cpu()
    torch.save(
        {
            "args": vars(args),
            "stage1_args": stage1_args,
            "state": sd,
            "gist_ids": Global_data.gist_token_id,
        },
        os.path.join(args.out, "adapters.pt"),
    )
    print("saved", os.path.join(args.out, "adapters.pt"), flush=True)


if __name__ == "__main__":
    main()
