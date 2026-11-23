"""#39 prototype: m routing keys per summary node, learned from content, scoring only.

route_fit.py fits free score parameters per context: a ceiling. Here the extra keys are a
function of the summary's own hidden state, shared by every node and context, as a model
would have them: k_m = k + U_m V_m h (h the layer input at the summary, rank --rank, per
layer, KV group and level). score = logsumexp_m(q . k_m) * scaling + beta; attention is
untouched. Trained with the routing loss over free nodes (as --route_free) on --n_train
contexts, scored on --n_test others with route_fit's metrics (flat mass ratio, beam leaf
mass, key hit by a retrieval row). m = 1 is trained the same way: the comparison of
interest is m = 1 against m = 2, 4 at equal training.

    python experiments/route_keys_proto.py --ckpt runs/gate_B/adapters.pt \
        --out results/route_keys8k.jsonl
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

import experiments.deep_recall as dr  # noqa: E402
import gist_beam as gb  # noqa: E402
from experiments.route_fit import NEG, flat_ratio, leaf_mass  # noqa: E402
from experiments.select_gate import gist  # noqa: E402
from gist_utils import Global_data  # noqa: E402


def collect(model, tok, docs, args, seeds):
    """Per context: tree, rows, retrieval-row count, key leaves and per layer the recorded
    (q [H, n, d], k [H_kv, T, d], beta [H_kv, T] or None, h [T, D]), on the CPU in fp16."""
    rec = dr.Recorder(args.layers)
    hid = {}
    hooks = [model.model.layers[l].self_attn.register_forward_pre_hook(
        lambda m, a, kw, l=l: hid.__setitem__(l, (a[0] if a else kw["hidden_states"])[0]),
        with_kwargs=True) for l in args.layers]
    n_q = len(tok(dr.QUESTION, add_special_tokens=False).input_ids)
    out = []
    for seed in seeds:
        for raw, key_pos, depth in dr.contexts(tok, docs, args.len, args.depths, seed=seed):
            key = re.search(r"\d{5}", tok.decode([raw[p] for p in key_pos])).group(0)
            full = raw + tok(" " + key, add_special_tokens=False).input_ids
            ids, where = gist(full)
            T = ids.numel()
            ret = where[len(raw) - n_q: len(full) - 1]
            g = torch.Generator().manual_seed(seed * 1000 + int(depth * 100))
            pool = where[: len(raw) - n_q]
            pool = pool[pool >= T // 4]
            pick = pool[torch.randperm(pool.numel(), generator=g)[: args.n_rows].sort().values]
            rows = torch.cat([pick, ret])
            rec.rows, rec.data = rows, {}
            with torch.no_grad():
                model.model(input_ids=ids[None], attention_mask=torch.ones_like(ids)[None])
            tree = gb.build_tree(ids, Global_data.gist_token_id, Global_data.sink_size)
            kl = tree.leaf[where[torch.tensor(key_pos, device=ids.device)]].unique()
            layers = {l: tuple(None if x is None else x.half().cpu() for x in rec.data[l][:2])
                      + (None if rec.data[l][3] is None else rec.data[l][3].cpu(),
                         hid[l].half().cpu()) for l in args.layers}
            out.append({"tree": tree, "rows": rows, "n_ret": ret.numel(), "depth": depth,
                        "seed": seed, "key_leaf": kl[kl < tree.gpos[0].numel()], "layers": layers})
    for h in hooks:
        h.remove()
    return out


class RouteKeys(torch.nn.Module):
    """Extra routing keys of one layer: per KV group and level, m low-rank maps h -> dk."""

    def __init__(self, D, d, groups, levels, m, rank):
        super().__init__()
        self.m = m
        self.V = torch.nn.Parameter(torch.randn(groups, levels, m, D, rank) * D ** -0.5 * 0.1)
        self.U = torch.nn.Parameter(torch.zeros(groups, levels, m, rank, d))
        self.U.data[:, :, 1:] = torch.randn_like(self.U.data[:, :, 1:]) * rank ** -0.5 * 0.1

    def keys(self, g, l, k, h):
        """k [G, d], h [G, D] -> [G, m, d]."""
        dk = torch.einsum("gD,mDr,mrd->gmd", h, self.V[g, l], self.U[g, l])
        return k[:, None] + dk


def level_terms(ctx, layer, g, hpg, dev):
    """Inputs of one (context, layer, group) on dev: q, k, beta, h, mass, free per level."""
    Q, K, beta, H = ctx["layers"][layer]
    tree, rows = ctx["tree"], ctx["rows"]
    q = Q[g * hpg:(g + 1) * hpg].to(dev).float()
    k = K[g].to(dev).float()
    b = torch.zeros(k.shape[0], device=dev) if beta is None else beta[g].to(dev).float()
    lo = gb._forced_leaf_lo(tree, rows, 1, Global_data.raw_window, 0)
    L = tree.num_levels
    free = [gb._closed(tree, rows, l) & ~gb._forced(tree, rows, l, lo, 4) for l in range(L)]
    return q, k, b, H.to(dev).float(), free


def scores(model_l, g, tree, q, k, b, h, scaling):
    out = []
    for l in range(tree.num_levels):
        gp = tree.gpos[l]
        km = model_l.keys(g, l, k[gp], h[gp]) if model_l is not None else k[gp][:, None]
        out.append((torch.einsum("hnd,gmd->nhgm", q, km) * scaling).logsumexp(-1) + b[gp])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--docs", default="~/spike_data/cpt_heldout")
    ap.add_argument("--len", type=int, default=8192)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--n_train", type=int, default=8, help="seeds (x len(depths) contexts)")
    ap.add_argument("--n_test", type=int, default=2)
    ap.add_argument("--layers", type=int, nargs="+", default=[9, 11, 13, 16, 17])
    ap.add_argument("--n_rows", type=int, default=256)
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--raw_window", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    tok, model, L = dr.load(args.ckpt, args.model)
    Global_data.raw_window, Global_data.window_turns = args.raw_window, 0
    gid = set(Global_data.gist_token_id)
    docs = [[t for t in r["input_ids"] if t not in gid]
            for r in datasets.load_from_disk(os.path.expanduser(args.docs))]
    train = collect(model, tok, docs, args, range(100, 100 + args.n_train))
    test = collect(model, tok, docs, args, range(args.n_test))   # route_fit's contexts
    cfg = gb.BeamConfig(k=(args.k,), merge="union")
    scaling = model.model.layers[0].self_attn.scaling
    c0 = model.config
    D, d = c0.hidden_size, c0.hidden_size // c0.num_attention_heads
    G, hpg = c0.num_key_value_heads, c0.num_attention_heads // c0.num_key_value_heads
    dev = "cuda"
    del model
    torch.cuda.empty_cache()
    out = open(args.out, "a")
    for m in [None] + args.ms:                                  # None: the checkpoint's keys
        name = "learned" if m is None else f"m{m}"
        res = collections.defaultdict(list)
        for layer in args.layers:
            rk = None if m is None else RouteKeys(D, d, G, L, m, args.rank).to(dev)
            if rk is not None:
                opt = torch.optim.Adam(rk.parameters(), lr=args.lr)
                for _ in range(args.epochs):
                    for ctx in train:
                        tree = ctx["tree"]
                        loss = 0.0
                        for g in range(G):
                            q, k, b, h, free = level_terms(ctx, layer, g, hpg, dev)
                            mass = gb.oracle_mass(tree, q, k, ctx["rows"], scaling)
                            for l, s in enumerate(scores(rk, g, tree, q, k, b, h, scaling)):
                                tgt = mass[l] * free[l][:, None]
                                tot = tgt.sum(-1, keepdim=True)
                                ok = tot[..., 0] > 0
                                if not ok.any():
                                    continue
                                ls = s.masked_fill(~free[l][:, None], NEG).log_softmax(-1)
                                loss = loss - ((tgt / tot.clamp_min(1e-30)) * ls).sum(-1)[ok].mean()
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
            with torch.no_grad():
                for ctx in test:
                    tree, rows, n_ret = ctx["tree"], ctx["rows"], ctx["n_ret"]
                    for g in range(G):
                        q, k, b, h, free = level_terms(ctx, layer, g, hpg, dev)
                        mass = gb.oracle_mass(tree, q, k, rows, scaling)
                        sc = scores(rk, g, tree, q, k, b, h, scaling)
                        res["flat0"].append(flat_ratio(sc[0], mass[0], free[0], args.k))
                        sel = gb.select(tree, q, k, None, rows, cfg, scaling,
                                        raw_window=Global_data.raw_window, scores=sc)
                        o = gb.select(tree, q, k, None, rows, cfg, scaling,
                                      raw_window=Global_data.raw_window,
                                      scores=[x.clamp(min=1e-30).log() for x in mass])
                        best = leaf_mass(o.sel[0], free[0], mass[0]).clamp_min(1e-30)
                        res["beam"].append(float((leaf_mass(sel.sel[0], free[0], mass[0]) / best).mean()))
                        if ctx["key_leaf"].numel():
                            hit = bool(sel.sel[0][-n_ret:][:, ctx["key_leaf"].to(dev)].any())
                            res["hit"].append(float(hit))
                            res[f"hit_L{layer}"].append(float(hit))
        summ = {k_: round(sum(v) / len(v), 3) for k_, v in sorted(res.items()) if v}
        line = {"ckpt": args.ckpt, "model": name, "rank": args.rank, "layers": args.layers,
                "n_train": args.n_train * len(args.depths), "n_test": args.n_test * len(args.depths),
                **summ}
        print(json.dumps(line), flush=True)
        out.write(json.dumps(line) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
