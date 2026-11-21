"""Stage 1 trainer: per-layer MSE gist compression with a frozen base model (spikes S3).

Trains gist LoRA (option D) or global LoRA (option B) plus the summary-token embeddings
on the converted streaming CPT data with the flex attention path. Every --eval_every
steps it reports the suffix logit KL: on held-out rows, the student without unfolding
(the mask passed downstream in training) against the base model reading the raw text,
as the mean reverse KL per raw token past the sink.

    python gist_stage1_trainer.py --data ~/spike_data/cpt_train --heldout ~/spike_data/cpt_heldout \
        --lora gist --pairs g1:raw hier:g1 --down hier --steps 500 --out runs/s3_d_joint
"""

import argparse
import json
import math
import os
import sys
import time

import datasets
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2ForCausalLM

import gist_flex_attention
import gist_resume
from gist_lora import apply_lora, apply_summary_bias, summary_spans
from gist_qwen2 import GistQwen2ForCausalLM
from gist_utils import Global_data, add_gist_tokens
from src.data.gist_stream import gist_segments, reinsert_gists, stream_labels


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument(
        "--heldout_long",
        default=None,
        help="rows for eval lengths above 8192 (the last --eval_rows rows are used)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", choices=["gist", "gist_kv", "all"], default="gist")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="student:teacher mask pairs (hier, g<m>, raw); default: the teacher chain "
        "g1:raw g2:g1 ... hier:g<L-1>",
    )
    ap.add_argument("--down", default="hier", help="mask whose output goes to the next layer")
    ap.add_argument("--lambda_ntp", type=float, default=0.0)
    ap.add_argument(
        "--beta", action="store_true", help="per-summary mass bias (gist_lora.SummaryBias)"
    )
    ap.add_argument("--lambda_route", type=float, default=0.0)
    ap.add_argument("--route_rows", type=int, default=32, help="sampled query rows per sequence")
    ap.add_argument("--per_level", action="store_true", help="one gist LoRA per summary level")
    ap.add_argument(
        "--phase2_at",
        type=int,
        default=0,
        help="staged training: from this step use --phase2_pairs / --phase2_down",
    )
    ap.add_argument("--phase2_pairs", nargs="+", default=["hier:g1"])
    ap.add_argument("--phase2_down", default="hier")
    ap.add_argument(
        "--freeze_level0", action="store_true", help="freeze level-0 adapters in phase 2"
    )
    ap.add_argument("--seq_len", type=int, default=8192)
    ap.add_argument(
        "--batch", type=int, default=8, help="sequences per optimizer step (accumulated)"
    )
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_rows", type=int, default=16)
    ap.add_argument("--eval_len", type=int, nargs="+", default=[8192])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--init_from",
        default=None,
        help="adapters.pt to start from (e.g. phase 1 of a staged run in a fresh process)",
    )
    ap.add_argument(
        "--start_step",
        type=int,
        default=0,
        help="first step is start_step + 1; the LR schedule continues from there",
    )
    ap.add_argument("--stop_at", type=int, default=0, help="stop (and save) after this step")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt if any")
    ap.add_argument("--save_every", type=int, default=25, help="steps between ckpt.pt saves")
    ap.add_argument(
        "--eval_only",
        default=None,
        help="adapters.pt to evaluate (all --eval_len) instead of training; a fresh "
        "process per run keeps long-eval graphs apart from training graphs",
    )
    ap.add_argument("--sink", type=int, default=3)
    ap.add_argument(
        "--raw_window", type=int, default=0, help="raw keys within this many positions stay visible"
    )
    ap.add_argument(
        "--window_turns",
        type=int,
        default=0,
        help="agent data: the raw window also needs the key within this many turns (ADR 0001: 2)",
    )
    ap.add_argument(
        "--levels",
        type=int,
        default=2,
        help="summary levels L; data gisted with fewer levels is re-gisted on the fly",
    )
    args = ap.parse_args(argv)
    if args.pairs is None:
        args.pairs = teacher_chain(args.levels)
    return args


def teacher_chain(L):
    """["g1:raw", "g2:g1", ..., "hier:g<L-1>"]; L = 2 gives the S3 pairs."""
    names = ["raw"] + [f"g{m}" for m in range(1, L)] + ["hier"]
    return [f"{s}:{t}" for t, s in zip(names, names[1:])]


def setup_globals(tok, gist_ids, args):
    Global_data.gist_token_id = gist_ids
    Global_data.chunk_size = [4] * args.levels
    Global_data.sink_size = args.sink
    Global_data.num_previous_chunks = 1
    Global_data.pad_token_id = tok.pad_token_id
    Global_data.train_attn = "flex"
    Global_data.raw_window = getattr(args, "raw_window", 0)
    Global_data.window_turns = getattr(args, "window_turns", 0)


def stage1_config(args, pairs=None, down=None):
    pairs = [tuple(p.split(":")) for p in (args.pairs if pairs is None else pairs)]
    return {
        "pairs": pairs,
        "down": down or args.down,
        "lambda_ntp": args.lambda_ntp,
        "route_rows": args.route_rows if args.lambda_route > 0 else 0,
        "weights": {"route": args.lambda_route},
    }


def row_ids(r, args):
    """A row's tokens, re-gisted when the run has more levels than the data (2)."""
    if args.levels <= 2:
        return r["input_ids"]
    return reinsert_gists(r["input_ids"], Global_data.gist_token_id, 4, 4, sink=args.sink)


def row_inputs(r, args, seq_len):
    """(ids, labels, segment kwargs) of a row truncated to seq_len. Agent trajectories
    (experiments/agent_data.py) are raw tokens with pinned flags and turn ids, gisted
    here; the segment kwargs then carry them to the gist mask (gist_pinned, gist_turn)."""
    gist = Global_data.gist_token_id
    if "pinned" in r:
        ids, pinned, turn = gist_segments(
            r["input_ids"], r["pinned"], r["turn"], gist, 4, 4, sink=args.sink)
        ids, pinned, turn = ids[:seq_len], pinned[:seq_len], turn[:seq_len]
        seg = {"gist_pinned": torch.tensor([pinned]), "gist_turn": torch.tensor([turn])}
        return ids, stream_labels(ids, gist), seg
    if args.levels <= 2:
        return r["input_ids"][:seq_len], r["labels"][:seq_len], {}
    ids = row_ids(r, args)[:seq_len]
    return ids, stream_labels(ids, gist), {}


def batches(ds, seq_len, seed, args):
    """Endless micro-batches of rows truncated to seq_len (rows shorter are skipped)."""
    g = torch.Generator().manual_seed(seed)
    while True:
        for i in torch.randperm(len(ds), generator=g).tolist():
            r = ds[i]
            if r["length"] < seq_len:
                continue
            ids, labels, seg = row_inputs(r, args, seq_len)
            yield torch.tensor([ids]), torch.tensor([labels]), seg


@torch.no_grad()
def suffix_kl(model, teacher, rows, seq_len, args, down):
    """Mean reverse KL per raw token past the sink, student (no unfolding) vs base on raw text."""
    gist = torch.tensor(Global_data.gist_token_id)
    saved = Global_data.stage1
    Global_data.stage1 = {"pairs": [], "down": down, "lambda_ntp": 0.0}
    model.eval()
    tot, n = 0.0, 0
    try:
        for r in rows:
            if r["length"] < seq_len:
                continue
            ids, _, seg = row_inputs(r, args, seq_len)
            ids = torch.tensor([ids], device="cuda")
            seg = {k: v.cuda() for k, v in seg.items()}
            raw_mask = ~torch.isin(ids[0], gist.to(ids.device))
            raw = ids[0][raw_mask][None]
            hs = model.model(
                input_ids=ids, attention_mask=torch.ones_like(ids), **seg
            ).last_hidden_state[0][raw_mask]
            ht = teacher.model(input_ids=raw).last_hidden_state[0]
            for s in range(args.sink, ht.shape[0], 1024):
                ls = F.log_softmax(teacher.lm_head(hs[s : s + 1024]).float(), -1)
                lt = F.log_softmax(teacher.lm_head(ht[s : s + 1024]).float(), -1)
                tot += float((ls.exp() * (ls - lt)).sum())
                n += ls.shape[0]
    finally:
        Global_data.stage1 = saved
        model.train()
    return tot / max(n, 1)


@torch.no_grad()
def beta_stats(model):
    """Mean over layers of |u| per level and of the spread (std) of beta over the summaries
    of the last forward pass; spread 0 means beta stayed a per-level constant."""
    u, spread = [], []
    for layer in model.model.layers:
        attn = layer.self_attn
        if getattr(attn, "summary_bias", None) is None:  # layer 0 has none
            continue
        u.append(attn.summary_bias.u.norm(dim=(1, 2)))
        x, lv = getattr(attn, "_bias_x", None), Global_data.lora_row_level
        if x is not None and lv is not None and x.shape[:2] == lv.shape:
            beta = attn.summary_bias(x, lv)  # [B, H_kv, T]
            spread.append(
                torch.stack([beta.transpose(1, 2)[lv == l].std() for l in range(u[-1].numel())])
            )
    out = {"u_norm": torch.stack(u).mean(0).tolist()}
    if spread:
        out["beta_std"] = torch.stack(spread).mean(0).tolist()
    return out


def group_grad_norms(model):
    """Per-layer gradient norms of the LoRA groups (q/o vs k/v) and the gist embeddings."""
    out = {}
    for i, layer in enumerate(model.model.layers):
        for kind, names in (("qo", ("q_proj", "o_proj")), ("kv", ("k_proj", "v_proj"))):
            sq = 0.0
            for name in names:
                m = getattr(layer.self_attn, name)
                for p in [*getattr(m, "lora_A", []), *getattr(m, "lora_B", [])]:
                    if p.grad is not None:
                        sq += float(p.grad.pow(2).sum())
            out[f"L{i}_{kind}"] = math.sqrt(sq)
        sb = getattr(layer.self_attn, "summary_bias", None)
        if sb is not None and sb.u.grad is not None:
            out[f"L{i}_beta"] = math.sqrt(float(sb.u.grad.pow(2).sum() + sb.b.grad.pow(2).sum()))
    emb = model.get_input_embeddings().weight
    if emb.grad is not None:
        out["embed"] = float(emb.grad[Global_data.gist_token_id].norm())
    return out


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    if args.resume and gist_resume.finished(args.out):
        print("already finished:", args.out, flush=True)
        return
    tok = AutoTokenizer.from_pretrained(args.model)
    teacher = Qwen2ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    model = GistQwen2ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=gist_flex_attention.GIST_FLEX
    ).cuda()
    gist_ids = add_gist_tokens(tok, model, num_levels=args.levels)
    setup_globals(tok, gist_ids, args)
    groups = apply_lora(model, args.lora, r=args.r, per_level=args.per_level)
    groups["beta"] = (
        apply_summary_bias(model, spans=summary_spans(args.levels)) if args.beta else []
    )
    if args.eval_only or args.init_from:
        ck = torch.load(args.eval_only or args.init_from, map_location="cpu")
        state = {k: v for k, v in ck["state"].items() if k != "gist_embed"}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, unexpected
        with torch.no_grad():
            model.get_input_embeddings().weight[gist_ids] = ck["state"]["gist_embed"].cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    params = [p for k in ("lora_qo", "lora_kv", "embed", "beta") for p in groups[k]]
    opt = torch.optim.AdamW(
        [
            {"params": groups["lora_qo"] + groups["lora_kv"]},
            {"params": groups["embed"] + groups["beta"]},
        ],
        lr=args.lr,
        weight_decay=0.0,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / args.warmup)
        * 0.5
        * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)),
    )
    Global_data.stage1 = stage1_config(args)
    train = datasets.load_from_disk(os.path.expanduser(args.data))
    held = datasets.load_from_disk(os.path.expanduser(args.heldout)).select(range(args.eval_rows))
    held_long = held
    if args.heldout_long:
        hl = datasets.load_from_disk(os.path.expanduser(args.heldout_long))
        held_long = hl.select(range(len(hl) - args.eval_rows, len(hl)))
    it = batches(train, args.seq_len, args.seed, args)
    accum = args.batch  # one sequence per forward
    resumed = gist_resume.load(args.out, model, opt) if args.resume else 0  # trims log.jsonl
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    if resumed:
        args.start_step = resumed
        for _ in range(resumed * accum):  # the data order continues where it stopped
            next(it)
    preempt = gist_resume.Preemption()

    def evaluate(step):
        down = Global_data.stage1["down"]
        rec = {
            "step": step,
            "down": down,
            **{
                f"suffix_kl_{L}": suffix_kl(
                    model, teacher, held if L <= 8192 else held_long, L, args, down
                )
                for L in args.eval_len
            },
        }
        if groups["beta"]:
            rec["beta"] = beta_stats(model)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    if args.eval_only:
        Global_data.stage1 = stage1_config(
            args,
            args.phase2_pairs if args.phase2_at else None,
            args.phase2_down if args.phase2_at else None,
        )
        log = open(os.path.join(args.out, "eval.jsonl"), "a")
        evaluate("final")
        return
    if args.start_step:
        for _ in range(args.start_step):
            sched.step()
        if args.phase2_at and args.start_step >= args.phase2_at:
            Global_data.stage1 = stage1_config(args, args.phase2_pairs, args.phase2_down)
        if args.freeze_level0:
            for p in groups["level0"]:
                p.requires_grad_(False)
    if not resumed:
        evaluate(args.start_step)
    last = args.stop_at or args.steps
    for step in range(args.start_step + 1, last + 1):
        if args.phase2_at and step == args.phase2_at:
            Global_data.stage1 = stage1_config(args, args.phase2_pairs, args.phase2_down)
            if args.freeze_level0:
                for p in groups["level0"]:
                    p.requires_grad_(False)
                    p.grad = None
            print(
                f"phase 2 from step {step}: {Global_data.stage1['pairs']} down={args.phase2_down}",
                flush=True,
            )
        t0 = time.time()
        agg = {}
        for _ in range(accum):
            ids, labels, seg = next(it)
            out = model(
                input_ids=ids.cuda(),
                attention_mask=torch.ones_like(ids).cuda(),
                labels=labels.cuda(),
                **{k: v.cuda() for k, v in seg.items()},
            )
            (out.loss / accum).backward()
            for k, v in Global_data.stage1["last"].items():
                agg[k] = agg.get(k, 0.0) + v / accum
        gn = group_grad_norms(model)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        rec = {
            "step": step,
            "lr": sched.get_last_lr()[0],
            "sec": round(time.time() - t0, 2),
            "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            **agg,
        }
        if step % 10 == 0 or step == 1:
            rec["grad"] = gn
            print(json.dumps({k: v for k, v in rec.items() if k != "grad"}), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % args.eval_every == 0 or step == last:
            evaluate(step)
        if step < last and (step % args.save_every == 0 or preempt.requested):
            gist_resume.save(args.out, step, model, params, opt)
            if preempt.requested:
                sys.exit(gist_resume.PREEMPTED)
    sd = {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if "lora_" in n or "summary_bias" in n
    }
    sd["gist_embed"] = model.get_input_embeddings().weight.detach()[gist_ids].cpu()
    torch.save(
        {"args": vars(args), "state": sd, "gist_ids": gist_ids},
        os.path.join(args.out, "adapters.pt"),
    )
    print("saved", os.path.join(args.out, "adapters.pt"), flush=True)


if __name__ == "__main__":
    main()
