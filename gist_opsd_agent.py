"""Stage 2 on long agent trajectories: OPSD (#21; ADR 0001 and 0002).

Each step, for --batch prompts (training-split trajectories of experiments/agent_data.py,
cut where an assistant turn starts, --ctx_min..--ctx_max raw tokens): the student appends
the prompt and generates the turn on the DECODE_SELECT=beam path, recording the keys each
layer attended to (gist_opsd_long.rollout); the base model scores the raw text; the
replay recomputes the generated rows with gradient (gist_opsd_long.replay); loss =
rkl_weight x reverse KL on generated raw tokens + lambda_route x routing KL. Updates the
gist LoRA, gist embeddings and beta of a stage-1 checkpoint.

    DECODE_SELECT=beam python gist_opsd_agent.py --stage1 runs/s5_L4/adapters.pt \
        --data /data/agent_data --ctx_max 131072 --out runs/opsd_agent
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

import gist_flex_attention
import gist_opsd_long as ol
import gist_resume
from gist_lora import apply_lora, apply_summary_bias, summary_spans
from gist_qwen2 import GistQwen2ForCausalLM
from gist_session import GistSession
from gist_utils import Global_data, add_gist_tokens


def load_stage1_flex(path, model_name):
    """A stage-1 checkpoint on a flex-attention model (bf16 base, fp32 adapters, CUDA)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("stage1_args") or ck["args"]
    L = a.get("levels", 2)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = GistQwen2ForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation=gist_flex_attention.GIST_FLEX).cuda()
    gist_ids = add_gist_tokens(tok, model, num_levels=L)
    assert gist_ids == ck["gist_ids"], (gist_ids, ck["gist_ids"])
    g = Global_data
    g.gist_token_id, g.chunk_size, g.sink_size, g.num_previous_chunks = gist_ids, [4] * L, a["sink"], 1
    g.pad_token_id, g.stage1, g.top_k = tok.pad_token_id, None, [3] * max(3, L)
    groups = apply_lora(model, a["lora"], r=a["r"], per_level=a.get("per_level", False))
    groups["beta"] = apply_summary_bias(model, spans=summary_spans(L)) if a.get("beta") else []
    model.cuda()
    state = {k: v for k, v in ck["state"].items() if k != "gist_embed"}
    _, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    with torch.no_grad():
        model.get_input_embeddings().weight[gist_ids] = ck["state"]["gist_embed"].cuda()
    model.selective = True
    return tok, model, groups, a


def sample_prompt(rows, pool, rng, head, ctx_min, ctx_max):
    """(row, cut): a random trajectory of pool and a random cut just after
    "<|im_start|>assistant\\n" of one of its assistant turns (an increment of the turn
    id), with ctx_min <= cut <= ctx_max. Only the drawn row is scanned."""
    while True:
        row = rows[pool[rng.randrange(len(pool))]]
        turn = row["turn"]
        cuts = [p + head for p in range(1, len(turn))
                if turn[p] != turn[p - 1] and ctx_min <= p + head <= ctx_max]
        if cuts:
            return row, cuts[rng.randrange(len(cuts))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--data", required=True, help="experiments/agent_data.py output")
    ap.add_argument("--heldout_frac", type=float, default=None,
                    help="issues held out (default agent_data's); match the eval sets' (#37)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx_min", type=int, default=8192)
    ap.add_argument("--ctx_max", type=int, default=131072)
    ap.add_argument("--n_new", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rkl_weight", type=float, default=1.0)
    ap.add_argument("--lambda_route", type=float, default=0.1)
    ap.add_argument("--route_scope", choices=["level", "oracle_children"], default="level")
    ap.add_argument("--route_oracle_k", type=int, default=256)
    ap.add_argument("--beam_k", default="256", help="BEAM_K: free nodes per level, level 0 first")
    ap.add_argument("--beam_merge", choices=["union", "sumexp"], default="union")
    ap.add_argument("--coarse", action="store_true", help="unselected candidates as coarse terms")
    ap.add_argument("--raw_window", type=int, default=8192)
    ap.add_argument("--window_turns", type=int, default=2)
    ap.add_argument("--check_every", type=int, default=10, help="replay-vs-rollout check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save_every", type=int, default=5)
    args = ap.parse_args(argv)
    if args.resume and gist_resume.finished(args.out):
        print("already finished:", args.out, flush=True)
        return
    os.environ.update(DECODE_SELECT="beam", BEAM_K=args.beam_k, BEAM_MERGE=args.beam_merge,
                      BEAM_COARSE="1" if args.coarse else "0")
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    from experiments.agent_data import load_split

    tok, model, groups, s1_args = load_stage1_flex(args.stage1, args.model)
    Global_data.raw_window, Global_data.window_turns = args.raw_window, args.window_turns
    vocab = len(tok)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    teacher = Qwen2ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                               attn_implementation="sdpa").cuda().eval()
    params = [p for k in ("lora_qo", "lora_kv", "embed", "beta") for p in groups[k]]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    rows = load_split(args.data, "train") if args.heldout_frac is None else \
        load_split(args.data, "train", args.heldout_frac)
    pool = [i for i, n in enumerate(rows["length"]) if n >= args.ctx_min]
    head = len(tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids)
    print(f"{len(pool)} trajectories of at least {args.ctx_min} tokens", flush=True)
    gen = torch.Generator().manual_seed(args.seed)
    resumed = gist_resume.load(args.out, model, opt, [gen]) if args.resume else 0
    for _ in range(resumed):
        sched.step()
    preempt = gist_resume.Preemption()
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    route_cfg = {"scope": args.route_scope, "oracle_k": args.route_oracle_k} if args.lambda_route > 0 else None

    for step in range(resumed + 1, args.steps + 1):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        rng = random.Random(args.seed * 100003 + step)  # the data order is a function of the step
        agg = {"rkl": 0.0, "route": 0.0, "keys_per_row": 0.0, "ctx": 0.0, "rollout_s": 0.0}
        for i in range(args.batch):
            row, cut = sample_prompt(rows, pool, rng, head, args.ctx_min, args.ctx_max)
            model.eval()
            t1 = time.time()
            s = GistSession(model)
            s.append(row["input_ids"][:cut], row["pinned"][:cut], row["turn"][:cut])
            ctx = (list(s.ids), list(s.pinned), list(s.turn))
            stream, lp_roll, recs = ol.rollout(s, model.lm_head(s.last_hidden), args.n_new, vocab, gen)
            del s
            agg["rollout_s"] += (time.time() - t1) / args.batch
            lp_t = ol.teacher_logprobs(teacher, ctx[0], stream, vocab)
            model.train()
            lp_s, st = ol.replay(model, *ctx, stream, recs, vocab, route=route_cfg)
            rkl = ol.reverse_kl(lp_s, lp_t).mean()
            loss = args.rkl_weight * rkl + (args.lambda_route * st["route"] if route_cfg else 0)
            (loss / args.batch).backward()
            ol.end_replay(st)
            agg["rkl"] += float(rkl) / args.batch
            agg["route"] += float(st["route"]) / args.batch if route_cfg else 0.0
            keys = [float((rec[l] >= 0).sum(1).float().mean()) for rec in recs for l in rec if l > 0]
            agg["keys_per_row"] += sum(keys) / max(len(keys), 1) / args.batch
            agg["ctx"] += len(ctx[0]) / args.batch
            if args.check_every and (step - 1) % args.check_every == 0 and i == 0:
                # log-probabilities of the sampled tokens (the vocabulary tail amplifies bf16 noise)
                gset = set(Global_data.gist_token_id)
                yr = torch.tensor([t for t in stream if t not in gset], device=lp_s.device)
                d = (lp_s.detach().gather(1, yr[:, None]) - lp_roll.gather(1, yr[:, None])).abs()
                agg["replay_max_abs_lp"] = float(d.max())
                agg["replay_mean_abs_lp"] = float(d.mean())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        rec = {"step": step, "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1),
               "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), **agg}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step < args.steps and (step % args.save_every == 0 or preempt.requested):
            gist_resume.save(args.out, step, model, params, opt, [gen])
            if preempt.requested:
                sys.exit(gist_resume.PREEMPTED)
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n or "summary_bias" in n}
    sd["gist_embed"] = model.get_input_embeddings().weight.detach()[Global_data.gist_token_id].cpu()
    torch.save({"args": vars(args), "stage1_args": s1_args, "state": sd,
                "gist_ids": Global_data.gist_token_id}, os.path.join(args.out, "adapters.pt"))
    print("saved", os.path.join(args.out, "adapters.pt"), flush=True)


if __name__ == "__main__":
    main()
