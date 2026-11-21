"""#28 gate (ADR 0002): does selecting at every position during stage 1 pay for a training
kernel? Evaluates stage-1 checkpoints under three inference modes:

    hier    no unfolding anywhere (summaries only)
    decode  the last --tail raw rows select (decode-time beam, gist_beam.py); earlier rows
            read summaries only: today's inference, generated tokens select
    all     every raw row selects: prefill with selection

and reports, per mode,
    kl            mean reverse KL per token over the last --tail raw rows, student vs the
                  base model on the raw text (held-out documents)
    passkey_*     pass key reproduced under teacher forcing (every key token is the argmax
                  given the correct prefix, which is what greedy decoding emits), split by
                  the key's depth (< 0.5 = front half)

Gate: a checkpoint trained with --select_train, in mode "all", against the control in
mode "decode": suffix KL at least 20% lower, or front-half passkey at least 10 points
higher.

    python experiments/select_gate.py --ckpts runs/gate_A/adapters.pt runs/gate_B/adapters.pt \
        --out results/select_gate.jsonl
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datasets  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import Qwen2ForCausalLM  # noqa: E402

import experiments.deep_recall as dr  # noqa: E402
import gist_beam  # noqa: E402
from gist_utils import Global_data  # noqa: E402
from src.data.gist_stream import insert_gists_with_map  # noqa: E402


def student(model, ids, mode, tail_from, cfg):
    """Hidden states of the gisted ids [T] under a mode; rows from tail_from select."""
    down = "hier" if mode == "hier" else "sel"
    rows_from = {"decode": tail_from, "all": 0}.get(mode, 0)
    Global_data.stage1 = {"pairs": [], "down": down, "lambda_ntp": 0.0, "route_rows": 0,
                          "select": {"cfg": cfg, "rows_from": rows_from}}
    try:
        return model.model(input_ids=ids[None], attention_mask=torch.ones_like(ids)[None]).last_hidden_state[0]
    finally:
        Global_data.stage1 = None


def gist(raw):
    out, src = insert_gists_with_map(raw, Global_data.gist_token_id, 4, 4, sink=Global_data.sink_size)
    where = [i for i, s in enumerate(src) if s >= 0]
    return torch.tensor(out, device="cuda"), torch.tensor(where, device="cuda")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--docs", default="~/spike_data/cpt_heldout")
    ap.add_argument("--len", type=int, default=8192)
    ap.add_argument("--tail", type=int, default=1024)
    ap.add_argument("--n_kl", type=int, default=16)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--k", type=int, nargs="+", default=[16])
    ap.add_argument("--merge", default="union")
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--modes", nargs="+", default=["hier", "decode", "all"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = gist_beam.BeamConfig(k=tuple(args.k), merge=args.merge, coarse=args.coarse)
    teacher = None
    done = set()
    if os.path.exists(args.out):  # a rerun after preemption keeps what is there
        done = {(r["ckpt"], r["mode"]) for r in map(json.loads, open(args.out))}
    out = open(args.out, "a")
    for ck in args.ckpts:
        tok, model, L = dr.load(ck, args.model)
        if teacher is None:
            teacher = Qwen2ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
        gid = set(Global_data.gist_token_id)
        docs = [[t for t in r["input_ids"] if t not in gid]
                for r in datasets.load_from_disk(os.path.expanduser(args.docs))]
        long_docs = [d_ for d_ in docs if len(d_) >= args.len][: args.n_kl]
        ctxs = [c for rep in range(args.reps) for c in dr.contexts(tok, docs, args.len, args.depths, seed=rep)]
        modes = args.modes + (["teacher"] if ck == args.ckpts[0] else [])
        for mode in modes:
            if (ck, mode) in done:
                continue
            kls = []
            for raw in long_docs if mode != "teacher" else []:
                raw = raw[: args.len]
                ids, where = gist(raw)
                tail_from = int(where[-args.tail])
                hs = student(model, ids, mode, tail_from, cfg)[where[-args.tail - 1: -1]]
                ht = teacher.model(input_ids=torch.tensor([raw], device="cuda")).last_hidden_state[0]
                ht = ht[-args.tail - 1: -1]
                ls = F.log_softmax(teacher.lm_head(hs).float(), -1)
                lt = F.log_softmax(teacher.lm_head(ht).float(), -1)
                kls.append(float((ls.exp() * (ls - lt)).sum(-1).mean()))
            hits = {"front": [], "back": []}
            for raw, key_pos, depth in ctxs:
                # the question ends with "The pass key is"; append the key as the answer
                key = re.search(r"\d{5}", tok.decode([raw[p] for p in key_pos])).group(0)
                ans = tok(" " + key, add_special_tokens=False).input_ids
                full = raw + ans
                if mode == "teacher":  # the base model on the raw prompt: the ceiling
                    t_ids = torch.tensor([full], device="cuda")
                    hs = teacher.model(input_ids=t_ids).last_hidden_state[0][len(raw) - 1: len(full) - 1]
                else:
                    ids, where = gist(full)
                    a_rows = where[len(raw) - 1: len(full) - 1]   # rows predicting the answer
                    hs = student(model, ids, mode, int(a_rows[0]), cfg)[a_rows]
                pred = teacher.lm_head(hs).argmax(-1).tolist()
                hits["front" if depth < 0.5 else "back"].append(float(pred == ans))
            rec = {"ckpt": ck, "mode": mode, "k": args.k, "merge": args.merge, "coarse": args.coarse,
                   "len": args.len, "tail": args.tail, "kl": sum(kls) / len(kls) if kls else 0.0,
                   "passkey_front": sum(hits["front"]) / len(hits["front"]),
                   "passkey_back": sum(hits["back"]) / len(hits["back"]), "n_kl": len(kls),
                   "n_passkey": len(ctxs)}
            print(json.dumps(rec), flush=True)
            out.write(json.dumps(rec) + "\n")
            out.flush()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
