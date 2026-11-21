"""S5 (#14): per-level routing recall in trained L-level gist trees, 32K-128K.

Contexts are held-out documents concatenated to --lens raw tokens, with a pass key at a
given depth and the question at the end. One gist prefill (flex, no unfolding) records
q/k/v/beta at --layers; for each of the last --q_rows raw positions and each head, the
top-down selection of the eager decode path (gist_qwen2.eager_attention_forward_decoding:
per level keep the top kk = min(beam, G_l // 4) children of the kept nodes, scored by
q . k without beta) is replayed offline under several beam schedules and compared with
the oracle: the top-k0 level-0 units by the query's softmax mass over raw keys.

Per strategy and level l (L-1 = top ... 0 = leaves):
    recall[l]  oracle units whose level-l ancestor was kept
    mass[l]    compressed-region raw mass under the kept level-l nodes
    hit_head   pass-key unit selected (mean over heads); hit_group: any head of a GQA group
    cost       gist scores computed / number of level-0 units
    err_sel    ||o - o_full|| / ||o_full||, o over the kept keys (selected raw units, selected
               summaries, the raw keys outside the compressed region), o_full over all raw keys
    err_coarse same with the unselected siblings at every level kept as coarse terms (with
               beta), which partitions the compressed region (RecSA-style)

    python experiments/deep_recall.py --ckpts runs/s5_L2/adapters.pt runs/s5_L4/adapters.pt \
        --out results/s5_recall.jsonl
    python experiments/deep_recall.py --summary results/s5_recall.jsonl
"""

import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datasets  # noqa: E402
import torch  # noqa: E402
from transformers import AttentionInterface, AutoTokenizer  # noqa: E402

import gist_flex_attention  # noqa: E402
from gist_lora import apply_lora, apply_summary_bias, summary_spans  # noqa: E402
from gist_qwen2 import GistQwen2ForCausalLM, repeat_kv  # noqa: E402
from gist_utils import Global_data, add_gist_tokens  # noqa: E402
from src.data.gist_stream import insert_gists_with_map  # noqa: E402

STRATEGIES = {
    # nodes kept at level l > 0 out of G_l; level 0 always keeps k0 (the inference budget).
    # Inference keeps min(top_k[l], G_l // 4): top_k[1] = top_k[0] = k0, 3 above. The wider
    # beams drop the G_l // 4 cap, which alone bounds every level to a quarter of its nodes.
    "default": lambda l, k0, G: min(k0 if l <= 1 else 3, max(1, G // 4)),
    "beam_x4": lambda l, k0, G: k0 if l <= 1 else 12,
    "beam_256": lambda l, k0, G: max(k0, 256),
    "flat": None,  # top-k0 level-0 summaries directly, no tree
}
NEEDLE = "The pass key is {key}. Remember it. {key} is the pass key. "
QUESTION = "\n\nWhat is the pass key? The pass key is"


def load(path, model_name):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    L = a.get("levels", 2)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = GistQwen2ForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation=gist_flex_attention.GIST_FLEX
    ).cuda()
    gist_ids = add_gist_tokens(tok, model, num_levels=L)
    assert gist_ids == ck["gist_ids"]
    Global_data.gist_token_id, Global_data.chunk_size = gist_ids, [4] * L
    Global_data.sink_size, Global_data.num_previous_chunks = a["sink"], 1
    Global_data.pad_token_id, Global_data.train_attn, Global_data.stage1 = (
        tok.pad_token_id,
        "flex",
        None,
    )
    apply_lora(model, a["lora"], r=a["r"], per_level=a.get("per_level", False))
    if a.get("beta"):
        apply_summary_bias(model, spans=summary_spans(L))
    state = {k: v for k, v in ck["state"].items() if k != "gist_embed"}
    _, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    with torch.no_grad():
        model.get_input_embeddings().weight[gist_ids] = ck["state"]["gist_embed"].cuda()
    return tok, model.eval(), L


def contexts(tok, docs, n_raw, depths, seed):
    """(raw ids, key token positions) per depth: documents up to n_raw, a key, a question."""
    rng = random.Random(seed)
    q = tok(QUESTION, add_special_tokens=False).input_ids
    out = []
    for d in depths:
        key = str(rng.randint(10000, 99999))
        needle = tok(NEEDLE.format(key=key), add_special_tokens=False).input_ids
        hay = []
        while len(hay) < n_raw - len(q) - len(needle):
            hay += docs[rng.randrange(len(docs))]
        hay = hay[: n_raw - len(q) - len(needle)]
        at = int(d * len(hay))
        ids = hay[:at] + needle + hay[at:] + q
        key_toks = set(tok(" " + key, add_special_tokens=False).input_ids) | set(
            tok(key, add_special_tokens=False).input_ids
        )
        key_pos = [at + i for i, t in enumerate(needle) if t in key_toks]
        out.append((ids, key_pos, d))
    return out


class Recorder:
    """Wraps the registered gist_flex attention to keep q (last rows), k, v, beta."""

    def __init__(self, layers):
        self.layers, self.rows, self.data = set(layers), None, {}
        self.orig = gist_flex_attention.gist_flex_attention_forward
        AttentionInterface.register(gist_flex_attention.GIST_FLEX, self)

    def __call__(self, module, query, key, value, *args, **kwargs):
        if module.layer_idx in self.layers:
            sb = getattr(module, "summary_bias", None)
            beta = None
            if sb is not None and getattr(module, "_bias_x", None) is not None:
                beta = sb(module._bias_x, Global_data.lora_row_level)[0].float()
            self.data[module.layer_idx] = (
                query[0][:, self.rows].float(),
                key[0].float(),
                value[0].float(),
                beta,
            )
        return self.orig(module, query, key, value, *args, **kwargs)


def tree(ids, gist_ids, sink):
    """Per level l: positions of its summaries, and for every position the number of
    level-l summaries before it (its unit at level l)."""
    T = ids.numel()
    lvl = torch.full((T,), -1, dtype=torch.long, device=ids.device)
    for l, g in enumerate(gist_ids):
        lvl[ids == g] = l
    gpos, unit = [], []
    for l in range(len(gist_ids)):
        is_g = lvl == l
        gpos.append(torch.nonzero(is_g).flatten())
        unit.append(torch.cumsum(is_g.long(), 0) - is_g.long())
    raw = lvl < 0
    raw[:sink] = False  # the sink is always visible and in no unit
    return lvl, gpos, unit, raw


def route(z_per_level, parent, G, k0, beam):
    """Top-down selection for one head: z_per_level[l] [G_l] scores, parent[l] [G_l] the
    level-(l+1) node of each level-l node. Returns kept nodes per level, candidates per
    level, and the number of scores computed."""
    L = len(G)
    sel, cands, cost = [None] * L, [None] * L, 0
    cand = torch.arange(G[-1], device=z_per_level[-1].device)
    for l in range(L - 1, -1, -1):
        kk = k0 if l == 0 else beam(l, k0, G[l])
        z = z_per_level[l][cand]
        cost += cand.numel()
        cands[l] = cand
        sel[l] = cand[torch.topk(z, min(kk, cand.numel())).indices]
        if l > 0:
            cand = torch.nonzero(torch.isin(parent[l - 1], sel[l])).flatten()
    return sel, cands, cost


@torch.no_grad()
def analyze(rec, ids, key_pos, gist_ids, sink, scaling, rows):
    lvl, gpos, unit, raw_all = tree(ids, gist_ids, sink)
    L = len(gist_ids)
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for layer, (Q, K, V, beta) in rec.items():
        H, hpg = Q.shape[0], Q.shape[0] // K.shape[0]
        K, V = repeat_kv(K[None], hpg)[0], repeat_kv(V[None], hpg)[0]
        B = None if beta is None else beta.repeat_interleave(hpg, 0)  # [H, T]
        for r, p in enumerate(rows):
            p = int(p)
            G = [int((g <= p).sum()) for g in gpos]
            if min(G) < 8:
                continue
            last_top = int(gpos[-1][G[-1] - 1])
            pos = torch.arange(p + 1, device=ids.device)
            raw = lvl[: p + 1] < 0
            comp = raw_all[: p + 1] & (pos <= last_top)
            leaf = unit[0][: p + 1]
            # ancestors of each level-0 unit, parent of each level-l node
            parent = [unit[l + 1][gpos[l][: G[l]]] for l in range(L - 1)]
            anc = [torch.arange(G[0], device=ids.device)]
            for l in range(L - 1):
                anc.append(parent[l][anc[-1]])
            key_units = leaf[torch.tensor(key_pos, device=ids.device)].unique()
            key_units = key_units[key_units < G[0]]
            k0 = min((p + 1) // 448 + 1, G[0] // 4)
            s = torch.einsum("hd,hkd->hk", Q[:, r], K[:, : p + 1])  # [H, p+1], unscaled
            m = (s * scaling).masked_fill(~raw, float("-inf")).softmax(-1)
            m0 = torch.zeros(H, G[0], device=ids.device).index_add_(1, leaf[comp], m[:, comp])
            M = m0.sum(-1).clamp_min(1e-30)
            o_full = torch.einsum("hk,hkd->hd", m, V[:, : p + 1])
            for name, beam in STRATEGIES.items():
                rc = [[] for _ in range(L)]
                ms = [[] for _ in range(L)]
                hit, cost, e_sel, e_coarse = [], [], [], []
                keep_heads = []
                for h in range(H):
                    oracle = torch.topk(m0[h], k0).indices
                    z = [s[h, gpos[l][: G[l]]].double() for l in range(L)]
                    if beam is None:
                        sel = [None] * L
                        sel[0] = torch.topk(z[0], k0).indices
                        cands = [None] * L
                        c = G[0]
                        levels = [0]
                    else:
                        sel, cands, c = route(z, parent, G, k0, beam)
                        levels = range(L)
                    for l in levels:
                        rc[l].append(float(torch.isin(anc[l][oracle], sel[l]).float().mean()))
                        under = torch.isin(anc[l], sel[l])
                        ms[l].append(float(m0[h][under].sum() / M[h]))
                    cost.append(c / G[0])
                    sel0 = torch.isin(leaf, sel[0]) & comp
                    hit.append(bool(torch.isin(key_units, sel[0]).any()) if key_units.numel() else None)
                    keep_heads.append(sel0)
                    keep = (raw & ~comp) | sel0
                    gist_keep = torch.zeros_like(keep)
                    for l in levels:
                        gist_keep[gpos[l][sel[l]]] = True
                    e_sel.append(_err(s[h], scaling, keep | gist_keep, gist_keep, B, h, V[h], o_full[h], p))
                    if beam is not None:
                        coarse = gist_keep.clone()
                        for l in levels:
                            coarse[gpos[l][cands[l]]] = True  # unselected siblings: coarse terms
                        e_coarse.append(
                            _err(s[h], scaling, keep | coarse, coarse, B, h, V[h], o_full[h], p)
                        )
                rec_ = out[(layer, name)]
                rec_["recall"].append([sum(x) / len(x) if x else None for x in rc])
                rec_["mass"].append([sum(x) / len(x) if x else None for x in ms])
                rec_["cost"].append(sum(cost) / H)
                rec_["err_sel"].append(sum(e_sel) / H)
                if e_coarse:
                    rec_["err_coarse"].append(sum(e_coarse) / H)
                if key_units.numel():
                    rec_["hit_head"].append(sum(hit) / H)
                    grp = torch.stack(keep_heads).view(H // hpg, hpg, -1).any(1)
                    rec_["hit_group"].append(
                        float(any(bool(g[torch.isin(leaf, key_units)].any()) for g in grp))
                    )
    return out


def _err(s, scaling, keep, gists, B, h, V, o_full, p):
    logit = s * scaling
    if B is not None:
        logit = logit + torch.where(gists, B[h, : p + 1], torch.zeros_like(logit))
    w = logit.masked_fill(~keep, float("-inf")).softmax(-1)
    o = w @ V[: p + 1]
    return float((o - o_full).norm() / o_full.norm().clamp_min(1e-30))


def mean_lists(xs):
    cols = list(zip(*xs))
    return [
        None if all(v is None for v in c) else sum(v for v in c if v is not None) / sum(v is not None for v in c)
        for c in cols
    ]


def summary(path):
    rows = [json.loads(l) for l in open(path)]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["ckpt"], r["levels"], r["n_raw"], r["strategy"])].append(r)
    print(f"{'ckpt':28s} L  len   strategy   recall[top..leaf]            mass(leaf) hit_g cost  err_sel err_coarse")
    for (ck, L, n, st), rs in sorted(groups.items()):
        rc = mean_lists([r["recall"] for r in rs])
        ms = mean_lists([r["mass"] for r in rs])
        f = lambda k: sum(r[k] for r in rs if r.get(k) is not None) / max(1, sum(r.get(k) is not None for r in rs))  # noqa: E731
        rcs = " ".join("  -  " if v is None else f"{v:.3f}" for v in rc[::-1])
        print(
            f"{ck[-28:]:28s} {L}  {n // 1024:3d}K {st:10s} {rcs:28s} {ms[0]:.3f}      "
            f"{f('hit_group'):.2f}  {f('cost'):.3f} {f('err_sel'):.3f}   {f('err_coarse'):.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ckpts", nargs="+")
    ap.add_argument("--docs", default="~/spike_data/cpt_heldout")
    ap.add_argument("--lens", type=int, nargs="+", default=[32768, 65536, 131072])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--reps", type=int, default=2, help="contexts per (length, depth)")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 8, 14, 20])
    ap.add_argument("--q_rows", type=int, default=8, help="last raw positions used as queries")
    ap.add_argument("--out")
    ap.add_argument("--summary")
    args = ap.parse_args()
    if args.summary:
        summary(args.summary)
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["ckpt"], r["n_raw"], r["depth"], r["rep"]))
    out = open(args.out, "a")
    for ck in args.ckpts:
        tok, model, L = load(ck, args.model)
        gid = set(Global_data.gist_token_id)
        docs = [
            [t for t in r["input_ids"] if t not in gid]
            for r in datasets.load_from_disk(os.path.expanduser(args.docs))
        ]
        recorder = Recorder(args.layers)
        scaling = model.model.layers[0].self_attn.scaling
        for n_raw in args.lens:
            for rep in range(args.reps):
                for raw_ids, key_pos, depth in contexts(tok, docs, n_raw, args.depths, seed=rep):
                    if (ck, n_raw, depth, rep) in done:
                        continue
                    gisted, src = insert_gists_with_map(
                        raw_ids, Global_data.gist_token_id, 4, 4, sink=Global_data.sink_size
                    )
                    where = {s: i for i, s in enumerate(src) if s >= 0}
                    ids = torch.tensor(gisted, device="cuda")
                    raw_pos = torch.tensor([where[i] for i in range(len(raw_ids))], device="cuda")
                    rows = raw_pos[-args.q_rows :]
                    recorder.rows, recorder.data = rows, {}
                    model.model(input_ids=ids[None], attention_mask=torch.ones_like(ids)[None])
                    res = analyze(
                        recorder.data,
                        ids,
                        [where[i] for i in key_pos],
                        Global_data.gist_token_id,
                        Global_data.sink_size,
                        scaling,
                        rows,
                    )
                    for (layer, st), v in res.items():
                        rec = {
                            "ckpt": ck,
                            "levels": L,
                            "n_raw": n_raw,
                            "depth": depth,
                            "rep": rep,
                            "layer": layer,
                            "strategy": st,
                            "recall": mean_lists(v["recall"]),
                            "mass": mean_lists(v["mass"]),
                        }
                        for k in ("cost", "err_sel", "err_coarse", "hit_head", "hit_group"):
                            if v[k]:
                                rec[k] = sum(v[k]) / len(v[k])
                        out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(ck, n_raw, depth, rep, "ok", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
