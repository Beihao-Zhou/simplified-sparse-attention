"""#34: can a routing key computed from the summary's hidden state reach route_fit's fit1?

fit1 (route_fit.py) fits every node's key per context, so it is a ceiling, not a model.
A separate routing key must be one projection for all contexts:

    score = q . (k_gist + RoPE(B A x)) * scaling + beta + c[l] + w . x       (linear)
    h = gelu(W1 f), score = q . (k_gist + RoPE(B h)) * scaling + beta + c[l] + v . h (mlp)

f is x, or x and the mean x of the node's children (mlp_kids); all of it is known at prefill,
so any of these keys can be stored per node. x is the summary's attention input, k_gist and beta the checkpoint's (B, w, c start at 0,
so step 0 is the learned router). The model is frozen; per layer and KV group only
(A, B, w, c) are fitted, on contexts from --train_docs, by route_fit's loss (cross entropy
to the true mass over the free nodes of each level). Scored on contexts from --docs, as
route_fit scores: beam (leaf mass selected / the oracle's) and hit (retrieval rows).

    python experiments/route_amortize.py --ckpt runs/oc_L1/adapters.pt --raw_window 2048 \
        --out results/route_amortize_oc_L1.jsonl
"""

import argparse
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
from experiments.route_fit import NEG, leaf_mass, score  # noqa: E402
from experiments.select_gate import gist  # noqa: E402
from gist_utils import Global_data  # noqa: E402


class XRecorder(dr.Recorder):
    """dr.Recorder plus each layer's attention input x (summary_bias's _bias_x)."""

    def __call__(self, module, query, key, value, *args, **kwargs):
        if module.layer_idx in self.layers:
            self.x[module.layer_idx] = module._bias_x[0].float()
        return super().__call__(module, query, key, value, *args, **kwargs)


def collect(model, tok, recorder, docs, args, reps, layers, n_q, scaling):
    """Per context: {layer: [per KV group dict]} plus what scoring needs."""
    cfg_fanout = 4
    out = []
    for rep in reps:
        for raw, key_pos, depth in dr.contexts(tok, docs, args.len, args.depths, seed=rep):
            key = re.search(r"\d{5}", tok.decode([raw[p] for p in key_pos])).group(0)
            full = raw + tok(" " + key, add_special_tokens=False).input_ids
            ids, where = gist(full)
            T = ids.numel()
            ret = where[len(raw) - n_q: len(full) - 1]
            g = torch.Generator().manual_seed(rep * 1000 + int(depth * 100))
            pool = where[: len(raw) - n_q]
            pool = pool[pool >= T // 4]
            pick = pool[torch.randperm(pool.numel(), generator=g)[: args.n_rows - ret.numel()]
                        .sort().values]
            rows = torch.cat([pick, ret])
            recorder.rows, recorder.data, recorder.x = rows, {}, {}
            with torch.no_grad():
                model.model(input_ids=ids[None], attention_mask=torch.ones_like(ids)[None])
            tree = gb.build_tree(ids, Global_data.gist_token_id, Global_data.sink_size)
            gmask = torch.isin(ids, torch.tensor(Global_data.gist_token_id, device=ids.device))
            rope_pos = torch.arange(T, device=ids.device) - gmask.long().cumsum(0)
            cos, sin = (x[0].float() for x in model.model.rotary_emb(
                torch.zeros(1, device=ids.device, dtype=torch.float32), rope_pos[None]))
            key_leaf = tree.leaf[where[torch.tensor(key_pos, device=ids.device)]].unique()
            key_leaf = key_leaf[key_leaf < tree.gpos[0].numel()]
            L = tree.num_levels
            lo = gb._forced_leaf_lo(tree, rows, 1, Global_data.raw_window, 0)
            free = [gb._closed(tree, rows, l) & ~gb._forced(tree, rows, l, lo, cfg_fanout)
                    for l in range(L)]
            per_layer = {}
            for layer in layers:
                Q, K, _, beta = recorder.data[layer]
                hpg = Q.shape[0] // K.shape[0]
                groups = []
                for gi in range(K.shape[0]):
                    q = Q[gi * hpg:(gi + 1) * hpg]
                    mass = gb.oracle_mass(tree, q, K[gi], rows, scaling, cfg_fanout)
                    b = torch.zeros(T, device=ids.device) if beta is None else beta[gi]
                    groups.append({"q": q, "mass": mass, "k": [K[gi][gp] for gp in tree.gpos],
                                   "b": [b[gp] for gp in tree.gpos]})
                per_layer[layer] = groups
            # kids[l][layer]: mean x of each node's children (level l-1 summaries; raw tokens
            # for leaves), also known at prefill, so a key built from it can be stored per node
            comp = tree.compressible & (tree.leaf >= 0) & (tree.leaf < tree.gpos[0].numel())
            kids = []
            for lv in range(L):
                kd = {}
                for layer in layers:
                    xl = recorder.x[layer]
                    if lv == 0:
                        src, par = xl[comp], tree.leaf[comp]
                    else:
                        src = xl[tree.gpos[lv - 1]]
                        par = torch.arange(src.shape[0], device=ids.device) // cfg_fanout
                    G = tree.gpos[lv].numel()
                    keep = par < G
                    s = torch.zeros(G, src.shape[1], device=ids.device).index_add_(
                        0, par[keep], src[keep])
                    n = torch.zeros(G, device=ids.device).index_add_(
                        0, par[keep], torch.ones_like(par[keep], dtype=torch.float))
                    kd[layer] = s / n.clamp_min(1)[:, None]
                kids.append(kd)
            out.append({"tree": tree, "rows": rows, "free": free, "n_ret": ret.numel(),
                        "key_leaf": key_leaf, "layers": per_layer, "depth": depth, "rep": rep,
                        "kids": kids,
                        "x": [{l: recorder.x[l][gp] for l in layers} for gp in tree.gpos],
                        "cos": [cos[rope_pos[gp]] for gp in tree.gpos],
                        "sin": [sin[rope_pos[gp]] for gp in tree.gpos]})
            print(f"collected depth {depth} rep {rep}", flush=True)
    return out


class RouteKey(torch.nn.Module):
    """Per layer and KV group: delta key RoPE(B A x) [d] and delta bias c[l] + w . x."""

    def __init__(self, hidden, d, r, levels):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(r, hidden) / hidden ** 0.5)
        self.B = torch.nn.Parameter(torch.zeros(d, r))
        self.w = torch.nn.Parameter(torch.zeros(hidden))
        self.c = torch.nn.Parameter(torch.zeros(levels))

    def scores(self, ctx, layer, gi, l, scaling):
        grp = ctx["layers"][layer][gi]
        x, cos, sin = ctx["x"][l][layer], ctx["cos"][l], ctx["sin"][l]
        dk = x @ self.A.t() @ self.B.t()                                   # [G, d]
        dk = dk * cos + rotate_half(dk) * sin
        b = grp["b"][l] + self.c[l] + x @ self.w
        return score(grp["q"], (grp["k"][l] + dk)[:, None], b, scaling)  # [n, h, G]


class RouteMLP(torch.nn.Module):
    """Non-linear routing key: h = gelu(W1 f + b1), delta key RoPE(B h), delta bias v . h + c[l].
    f is x ("mlp") or [x, mean x of the node's children] ("mlp_kids"). B, v, c start at 0."""

    def __init__(self, hidden, d, r, levels, kids):
        super().__init__()
        self.kids = kids
        f = hidden * (2 if kids else 1)
        self.norm = torch.nn.LayerNorm(f)
        self.W1 = torch.nn.Linear(f, r)
        self.B = torch.nn.Parameter(torch.zeros(d, r))
        self.v = torch.nn.Parameter(torch.zeros(r))
        self.c = torch.nn.Parameter(torch.zeros(levels))

    def scores(self, ctx, layer, gi, l, scaling):
        grp = ctx["layers"][layer][gi]
        f = ctx["x"][l][layer]
        if self.kids:
            f = torch.cat([f, ctx["kids"][l][layer]], -1)
        h = torch.nn.functional.gelu(self.W1(self.norm(f)))              # [G, r]
        dk = h @ self.B.t()
        dk = dk * ctx["cos"][l] + rotate_half(dk) * ctx["sin"][l]
        b = grp["b"][l] + self.c[l] + h @ self.v
        return score(grp["q"], (grp["k"][l] + dk)[:, None], b, scaling)


def make_router(name, hidden, d, levels):
    """linear_r{r} | mlp_r{r} | mlp_kids_r{r}"""
    kind, r = name.rsplit("_r", 1)
    if kind == "linear":
        return RouteKey(hidden, d, int(r), levels)
    return RouteMLP(hidden, d, int(r), levels, kids=kind == "mlp_kids")


def ce(s, mass, free):
    tgt = mass * free[:, None]
    tot = tgt.sum(-1, keepdim=True)
    ok = tot[..., 0] > 0
    s = s.masked_fill(~free[:, None], NEG)
    return -((tgt / tot.clamp_min(1e-30)) * s.log_softmax(-1)).sum(-1)[ok].mean()


def evaluate(ctxs, layer, gi, model_fn, cfg, scaling):
    """beam and hit per context for scores model_fn(ctx, l) (None: the checkpoint's)."""
    res = []
    for ctx in ctxs:
        grp, tree, rows, free = ctx["layers"][layer][gi], ctx["tree"], ctx["rows"], ctx["free"]
        L = tree.num_levels
        with torch.no_grad():
            sc = [model_fn(ctx, l) if model_fn else
                  score(grp["q"], grp["k"][l][:, None], grp["b"][l], scaling) for l in range(L)]
            orc = [grp["mass"][l].clamp(min=1e-30).log() for l in range(L)]
            kw = dict(raw_window=Global_data.raw_window)
            s_sel = gb.select(tree, grp["q"], None, None, rows, cfg, scaling, scores=sc, **kw)
            o_sel = gb.select(tree, grp["q"], None, None, rows, cfg, scaling, scores=orc, **kw)
            best = leaf_mass(o_sel.sel[0], free[0], grp["mass"][0]).clamp_min(1e-30)
            got = leaf_mass(s_sel.sel[0], free[0], grp["mass"][0])
            hit = None
            if ctx["key_leaf"].numel():
                hit = bool(s_sel.sel[0][-ctx["n_ret"]:][:, ctx["key_leaf"]].any(-1).any())
        res.append({"beam": float((got / best).mean()), "hit": hit})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_docs", default="~/spike_data/cpt_train")
    ap.add_argument("--docs", default="~/spike_data/cpt_heldout")
    ap.add_argument("--len", type=int, default=8192)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--train_reps", type=int, default=8)
    ap.add_argument("--test_reps", type=int, default=2)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 8, 14, 20])
    ap.add_argument("--n_rows", type=int, default=512)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--routers", nargs="+", default=["linear_r8", "linear_r64"],
                    help="linear_r{r} | mlp_r{r} | mlp_kids_r{r}")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--raw_window", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    tok, model, _ = dr.load(args.ckpt, args.model)
    Global_data.raw_window, Global_data.window_turns = args.raw_window, 0
    gid = set(Global_data.gist_token_id)

    def load_docs(p):
        return [[t for t in r["input_ids"] if t not in gid]
                for r in datasets.load_from_disk(os.path.expanduser(p))]

    cfg = gb.BeamConfig(k=(args.k,), merge="union")
    scaling = model.model.layers[0].self_attn.scaling
    recorder = XRecorder(args.layers)
    n_q = len(tok(dr.QUESTION, add_special_tokens=False).input_ids)
    train = collect(model, tok, recorder, load_docs(args.train_docs), args,
                    range(100, 100 + args.train_reps), args.layers, n_q, scaling)
    test = collect(model, tok, recorder, load_docs(args.docs), args, range(args.test_reps),
                   args.layers, n_q, scaling)
    hidden, n_kv = model.config.hidden_size, model.config.num_key_value_heads
    d = hidden // model.config.num_attention_heads
    L = train[0]["tree"].num_levels
    out = open(args.out, "w")

    def write(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(rec["layer"], rec["group"], rec["model"], round(rec["beam"], 3), rec["hit"],
              flush=True)

    for layer in args.layers:
        for gi in range(n_kv):
            base = evaluate(test, layer, gi, None, cfg, scaling)
            write({"ckpt": args.ckpt, "layer": layer, "group": gi, "model": "learned",
                   "beam": sum(r["beam"] for r in base) / len(base),
                   "hit": [r["hit"] for r in base]})
            for name in args.routers:
                rk = make_router(name, hidden, d, L).cuda()
                opt = torch.optim.AdamW(rk.parameters(), lr=args.lr, weight_decay=args.wd)
                g = torch.Generator().manual_seed(layer * 10 + gi)
                for step in range(args.steps):
                    ctx = train[int(torch.randint(len(train), (1,), generator=g))]
                    grp = ctx["layers"][layer][gi]
                    loss = torch.stack([ce(rk.scores(ctx, layer, gi, l, scaling),
                                           grp["mass"][l], ctx["free"][l])
                                        for l in range(L)]).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                res = evaluate(test, layer, gi,
                               lambda c, l: rk.scores(c, layer, gi, l, scaling), cfg, scaling)
                tr = evaluate(train[:5], layer, gi,
                              lambda c, l: rk.scores(c, layer, gi, l, scaling), cfg, scaling)
                write({"ckpt": args.ckpt, "layer": layer, "group": gi, "model": name,
                       "beam": sum(x["beam"] for x in res) / len(res),
                       "beam_train": sum(x["beam"] for x in tr) / len(tr),
                       "hit": [x["hit"] for x in res], "final_loss": float(loss)})


if __name__ == "__main__":
    main()
