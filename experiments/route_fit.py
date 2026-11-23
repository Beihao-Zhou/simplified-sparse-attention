"""#31: the routing ceiling of a score parameterization, fitted offline.

The oracle (select_gate.py --oracle) shows how far routing by true attention mass gets; it
does not say whether the gap to the learned scores comes from the parameterization (one
linear key plus a bias per node) or from training. Here every node's score parameters are
fitted directly, per context, layer and KV group, to the true mass of real queries, and the
fit is scored on held-out query rows of the same context. Models:

    learned  q . k_gist + beta as the checkpoint has them (no fit)
    fit1     one key + bias per node, post-RoPE (today's parameterization, at its best)
    fit4     4 keys per node, score logsumexp_m(q . k_m) + bias (convex in q)
    nope1    one key + bias, scored against the query with its RoPE undone
    oracle   log true mass (the reference: flat and beam ratios 1 by definition)

Fit: per level, cross entropy between the true mass and the softmax of the scores over the
row's free nodes (closed, not forced), initialized from the learned keys. Metrics, on test
rows (a random --test_frac of the sampled rows, plus the retrieval rows):

    flat[l]    per (row, head): mass of the top-k free level-l nodes by score / by mass
    beam       gist_beam.select with these scores (k per level, union over the group's
               heads): leaf mass captured / that of the oracle's selection
    hit        the pass key's leaf selected by some retrieval row: the rows of the question
               and those predicting the appended answer (teacher forcing, as select_gate);
               hit_rows the share of retrieval rows that select it. Fitted layers only;
               learned and oracle are also scored at every layer (hit_all, hit_ctx).

flat_fit is flat on the rows fitted: fit far above test means the fit memorizes.

    python experiments/route_fit.py --ckpt runs/gate_B/adapters.pt --out results/route_fit8k.jsonl
"""

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datasets  # noqa: E402
import torch  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import rotate_half  # noqa: E402

import experiments.deep_recall as dr  # noqa: E402
import gist_beam as gb  # noqa: E402
from experiments.select_gate import gist  # noqa: E402
from gist_utils import Global_data  # noqa: E402

NEG = -1e9  # finite: a row with no free node must not turn the loss into nan


def score(q, keys, b, scaling):
    """q [h, n, d], keys [G, m, d], b [G] -> [n, h, G]; m = 1 is q . k + b."""
    return (torch.einsum("hnd,gmd->nhgm", q, keys) * scaling).logsumexp(-1) + b


def fit(q, mass, free, k0, b0, m, scaling, steps=300, lr=0.05, seed=0):
    """Score parameters of one level fitted to mass [n, h, G] over free [n, G] nodes,
    from k0 [G, d], b0 [G]. Returns (keys [G, m, d], b [G])."""
    tgt = mass * free[:, None]
    tot = tgt.sum(-1, keepdim=True)
    ok = tot[..., 0] > 0
    tgt = tgt / tot.clamp_min(1e-30)
    g = torch.Generator(device=k0.device).manual_seed(seed)
    unit = k0.std().clamp_min(1e-6)
    noise = torch.randn(k0.shape[0], m, k0.shape[1], generator=g, device=k0.device)
    dk = (0.3 * noise if m > 1 else torch.zeros_like(noise)).requires_grad_()
    db = torch.zeros_like(b0, requires_grad=True)
    opt = torch.optim.Adam([dk, db], lr=lr)
    for _ in range(steps):
        s = score(q, k0[:, None] + unit * dk, b0 + db, scaling).masked_fill(~free[:, None], NEG)
        loss = -(tgt * s.log_softmax(-1)).sum(-1)[ok].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (k0[:, None] + unit * dk).detach(), (b0 + db).detach()


def flat_ratio(s, mass, free, k):
    """Mean over (row, head) with free mass: mass of the top-k free nodes by s / by mass."""
    s = s.masked_fill(~free[:, None], float("-inf"))
    mm = mass.masked_fill(~free[:, None], float("-inf"))
    kk = min(k, s.shape[-1])
    if kk == 0:
        return None
    got = (mass * free[:, None]).gather(-1, s.topk(kk, -1).indices).sum(-1)
    best = mm.topk(kk, -1).values.clamp_min(0).sum(-1)
    ok = best > 0
    return float((got[ok] / best[ok]).mean()) if ok.any() else None


def unrotate(x, cos, sin):
    return x * cos - rotate_half(x) * sin


@torch.no_grad()
def leaf_mass(sel0, free0, mass0):
    """[n, h]: mass of the selected free leaves."""
    return (mass0 * (sel0 & free0)[:, None]).sum(-1)


MODELS = {"learned": None, "fit1": (1, False), "fit4": (4, False), "nope1": (1, True),
          "oracle": None}


def run_group(tree, q, k, b, rows, test, n_ret, key_leaf, cfg, scaling, cos, sin, rope_pos,
              args, names=tuple(MODELS)):
    """Models `names` for one KV group: q [h, n, d] at rows [n], k [T, d], b [T] or None. The
    last n_ret rows (test rows) are the retrieval rows: the question and the answer."""
    L = tree.num_levels
    lo = gb._forced_leaf_lo(tree, rows, 1, 0, 0)
    free = [gb._closed(tree, rows, l) & ~gb._forced(tree, rows, l, lo, cfg.fanout) for l in range(L)]
    mass = gb.oracle_mass(tree, q, k, rows, scaling, cfg.fanout)
    b = torch.zeros(k.shape[0], device=k.device) if b is None else b
    fitr = ~test
    q_nope = unrotate(q, cos[rope_pos[rows]][None], sin[rope_pos[rows]][None])
    out = {}
    for name in names:
        spec = MODELS[name]
        sc = []
        for l in range(L):
            gp = tree.gpos[l]
            if name == "oracle":
                s = mass[l].clamp(min=1e-30).log()
            elif name == "learned":
                s = score(q, k[gp][:, None], b[gp], scaling)
            else:
                m, nope = spec
                qq = q_nope if nope else q
                k0 = unrotate(k[gp], cos[rope_pos[gp]], sin[rope_pos[gp]]) if nope else k[gp]
                with torch.enable_grad():
                    keys, bb = fit(qq[:, fitr], mass[l][fitr], free[l][fitr], k0, b[gp], m,
                                   scaling, args.steps, args.lr)
                s = score(qq, keys, bb, scaling)
            sc.append(s)
        rec = {"flat": [flat_ratio(sc[l][test], mass[l][test], free[l][test], args.k)
                        for l in range(L)],
               "flat_fit": [flat_ratio(sc[l][fitr], mass[l][fitr], free[l][fitr], args.k)
                            for l in range(L)]}
        s_sel = gb.select(tree, q[:, test], k, None, rows[test], cfg, scaling,
                          scores=[x[test] for x in sc])
        out[name] = (rec, s_sel)
    o_sel = out["oracle"][1]
    best = leaf_mass(o_sel.sel[0], free[0][test], mass[0][test]).clamp_min(1e-30)
    res = {}
    for name, (rec, s_sel) in out.items():
        got = leaf_mass(s_sel.sel[0], free[0][test], mass[0][test])
        rec["beam"] = float((got / best).mean())
        if key_leaf.numel():  # per retrieval row: is a leaf holding the key selected
            r_hit = s_sel.sel[0][-n_ret:][:, key_leaf].any(-1)
            rec["hit"], rec["hit_rows"] = bool(r_hit.any()), float(r_hit.float().mean())
        else:
            rec["hit"] = rec["hit_rows"] = None
        res[name] = rec
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--docs", default="~/spike_data/cpt_heldout")
    ap.add_argument("--len", type=int, default=8192)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 8, 14, 20],
                    help="layers fitted; learned and oracle hits are taken at every layer")
    ap.add_argument("--n_rows", type=int, default=1024, help="sampled raw query rows")
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    tok, model, L = dr.load(args.ckpt, args.model)
    Global_data.raw_window = Global_data.window_turns = 0
    gid = set(Global_data.gist_token_id)
    docs = [[t for t in r["input_ids"] if t not in gid]
            for r in datasets.load_from_disk(os.path.expanduser(args.docs))]
    cfg = gb.BeamConfig(k=(args.k,), merge="union")
    scaling = model.model.layers[0].self_attn.scaling
    recorder = dr.Recorder(range(model.config.num_hidden_layers))
    n_q = len(tok(dr.QUESTION, add_special_tokens=False).input_ids)
    done = set()
    if os.path.exists(args.out):  # resumable per context
        done = {(r["depth"], r["rep"]) for r in map(json.loads, open(args.out))}
    out = open(args.out, "a")
    for rep in range(args.reps):
        for raw, key_pos, depth in dr.contexts(tok, docs, args.len, args.depths, seed=rep):
            if (depth, rep) in done:
                continue
            # the answer appended as select_gate's passkey does (teacher forcing): the rows
            # of the question and those predicting the answer are where retrieval happens
            key = re.search(r"\d{5}", tok.decode([raw[p] for p in key_pos])).group(0)
            full = raw + tok(" " + key, add_special_tokens=False).input_ids
            ids, where = gist(full)
            T = ids.numel()
            ret = where[len(raw) - n_q: len(full) - 1]
            g = torch.Generator().manual_seed(rep * 1000 + int(depth * 100))
            pool = where[: len(raw) - n_q]
            pool = pool[pool >= T // 4]
            n_pick = args.n_rows - ret.numel()
            pick = pool[torch.randperm(pool.numel(), generator=g)[:n_pick].sort().values]
            rows = torch.cat([pick, ret])
            test = torch.rand(rows.numel(), generator=g) < args.test_frac
            test[-ret.numel():] = True
            test = test.to(rows.device)
            is_ret = torch.zeros_like(test)
            is_ret[-ret.numel():] = True
            recorder.rows, recorder.data = rows, {}
            with torch.no_grad():
                model.model(input_ids=ids[None], attention_mask=torch.ones_like(ids)[None])
            tree = gb.build_tree(ids, Global_data.gist_token_id, Global_data.sink_size)
            gmask = torch.isin(ids, torch.tensor(Global_data.gist_token_id, device=ids.device))
            rope_pos = torch.arange(T, device=ids.device) - gmask.long().cumsum(0)
            cos, sin = (x[0].float() for x in model.model.rotary_emb(
                torch.zeros(1, device=ids.device, dtype=torch.float32), rope_pos[None]))
            key_leaf = tree.leaf[where[torch.tensor(key_pos, device=ids.device)]].unique()
            key_leaf = key_leaf[key_leaf < tree.gpos[0].numel()]
            for layer, (Q, K, V, beta) in sorted(recorder.data.items()):
                hpg = Q.shape[0] // K.shape[0]
                fitted = layer in args.layers
                # other layers: learned and oracle on the retrieval rows only
                sub = slice(None) if fitted else is_ret
                for gi in range(K.shape[0]):
                    res = run_group(tree, Q[gi * hpg:(gi + 1) * hpg][:, sub], K[gi],
                                    None if beta is None else beta[gi], rows[sub], test[sub],
                                    ret.numel(), key_leaf, cfg, scaling, cos, sin, rope_pos, args,
                                    tuple(MODELS) if fitted else ("learned", "oracle"))
                    for name, r in res.items():
                        rec = {"ckpt": args.ckpt, "len": args.len, "depth": depth, "rep": rep,
                               "layer": layer, "group": gi, "model": name, "k": args.k,
                               "fitted": fitted, **r}
                        out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"depth {depth} rep {rep} done", flush=True)
    summary(args.out)


def summary(path):
    """Fitted layers: ratios and hits per model. All layers (learned, oracle): hit_all, the
    share of (layer, group) that select the key; hit_ctx, the contexts where any does;
    the oracle's hit per layer, which shows the layers that retrieve."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    ctx = collections.defaultdict(lambda: collections.defaultdict(bool))
    per_layer = collections.defaultdict(list)
    for r in map(json.loads, open(path)):
        if r.get("hit") is not None and r["model"] in ("learned", "oracle"):
            acc[r["model"]]["hit_all"].append(float(r["hit"]))
            c = ctx[r["model"]]
            c[(r["depth"], r["rep"])] = c[(r["depth"], r["rep"])] or r["hit"]
            if r["model"] == "oracle":
                per_layer[r["layer"]].append(float(r["hit"]))
        if not r.get("fitted", True):
            continue
        a = acc[r["model"]]
        for key in ("flat", "flat_fit"):
            for l, v in enumerate(r[key]):
                if v is not None:
                    a[f"{key}{l}"].append(v)
        a["beam"].append(r["beam"])
        if r["hit"] is not None:
            a["hit"].append(float(r["hit"]))
            a["hit_rows"].append(r["hit_rows"])
    for name, c in ctx.items():
        acc[name]["hit_ctx"] = [float(v) for v in c.values()]
    for name, a in acc.items():
        print(name, {k: round(sum(v) / len(v), 3) for k, v in sorted(a.items())})
    print("oracle hit per layer", {l: round(sum(v) / len(v), 2) for l, v in sorted(per_layer.items())})


if __name__ == "__main__":
    main()
