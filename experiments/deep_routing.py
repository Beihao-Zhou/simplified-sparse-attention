"""Simulate gist routing in deeper trees from recorded 2-level selector inputs.

The released checkpoints have two gist levels (4 and 16 raw tokens). Deeper levels are
emulated by pooling level-1 gist scores over groups of J consecutive level-1 nodes
(virtual levels 2, 3, ...). Routing strategies are compared against the attention
oracle (top-k level-0 chunks by the query's own softmax mass over raw keys) at their
scoring cost, which tells how per-level routing errors compound with depth.

Pooled scores stand in for gists that were never trained, so absolute numbers are a
proxy; the comparison between strategies at equal depth is the point.

Input: a directory of .npz files, one per selector call (sample, decode step, layer):
    s0       [H, N0]  level-0 gist scores (q . k, any common scale)
    s1       [H, N1]  level-1 gist scores
    parent0  [N0]     level-1 parent of each level-0 chunk, -1 if not yet summarized
    m0       [H, N0]  oracle attention mass per level-0 chunk
    k0       ()       level-0 budget the selector used (chunks to unfold)
    answer0  [A]      level-0 chunks that contain the gold answer (optional, may be empty)
    task, bucket      strings (optional, used for grouping)

    python experiments/deep_routing.py RECORDS_DIR [--depths 2 3 4 5] [--fanout 4]
    python experiments/deep_routing.py --synthetic   # self-check on generated data
"""

import argparse
import collections
import glob
import heapq
import os

import numpy as np


def build_tree(n1, depth, fanout):
    """parents[l][i]: index of the level-(l+1) node above level-l node i, for l >= 1."""
    parents = {}
    n = n1
    for lvl in range(1, depth - 1):
        parents[lvl] = np.arange(n) // fanout
        n = int(parents[lvl].max()) + 1 if n else 0
    return parents


def pooled_scores(s1, parents, depth, pool):
    """Scores per level: real for levels 0-1, pooled from children above."""
    scores = {1: s1}
    for lvl in range(2, depth):
        child, par = scores[lvl - 1], parents[lvl - 1]
        n = int(par.max()) + 1 if len(par) else 0
        if pool == "max":
            out = np.full(n, -np.inf)
            np.maximum.at(out, par, child)
        elif pool == "mean":
            out = np.bincount(par, weights=child, minlength=n) / np.bincount(par, minlength=n)
        else:  # logsumexp
            mx = np.full(n, -np.inf)
            np.maximum.at(mx, par, child)
            out = mx + np.log(np.bincount(par, weights=np.exp(child - mx[par]), minlength=n))
        scores[lvl] = out
    return scores


def children_of(parents, lvl):
    """For level lvl >= 2: list of level-(lvl-1) children per node."""
    par = parents[lvl - 1]
    order = np.argsort(par, kind="stable")
    bounds = np.searchsorted(par[order], np.arange(int(par.max()) + 2 if len(par) else 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def top(idx, score, k):
    idx = np.asarray(idx)
    if len(idx) <= k:
        return idx
    return idx[np.argpartition(-score[idx], k - 1)[:k]]


def route_top_down(s0, scores, kids, kids0, orphans0, depth, k0, beam):
    """Strict top-down: keep `beam * k0` nodes per level >= 1, then k0 level-0 chunks.
    Returns (selected level-0 chunks, number of scores computed)."""
    width = max(1, int(round(beam * k0)))
    cand = np.arange(len(scores[depth - 1]))
    cost = len(cand)
    for lvl in range(depth - 1, 1, -1):
        keep = top(cand, scores[lvl], width)
        cand = np.concatenate([kids[lvl][i] for i in keep]) if len(keep) else np.array([], int)
        cost += len(cand)
    keep1 = top(cand, scores[1], width)
    cand0 = np.concatenate([kids0[i] for i in keep1] + [orphans0])
    cost += len(cand0)
    return top(cand0, s0, k0), cost


def route_best_first(s0, scores, kids, kids0, orphans0, depth, k0, budget):
    """Tree cut by best-first search: expand the highest z-scored node (scores
    standardized per level) until `budget` scores are spent; then take the top-k0
    level-0 chunks among those scored."""
    z = {lvl: (v - v.mean()) / (v.std() + 1e-6) for lvl, v in scores.items()}
    z[0] = (s0 - s0.mean()) / (s0.std() + 1e-6)
    heap = [(-z[depth - 1][i], depth - 1, i) for i in range(len(scores[depth - 1]))]
    heap += [(-z[0][i], 0, i) for i in orphans0]
    heapq.heapify(heap)
    cost, scored0 = len(heap), [i for i in orphans0]
    while heap and cost < budget:
        _, lvl, i = heapq.heappop(heap)
        if lvl == 0:
            continue
        sub = kids0[i] if lvl == 1 else kids[lvl][i]
        nxt = lvl - 1
        for c in sub:
            heapq.heappush(heap, (-z[nxt][c], nxt, c))
            if nxt == 0:
                scored0.append(c)
        cost += len(sub)
    return top(np.unique(np.asarray(scored0, int)), s0, k0), cost


def evaluate(records, depths, fanout, pool, beams):
    rows = collections.defaultdict(list)
    for rec in records:
        s0_all, s1_all, m0_all = rec["s0"].astype(np.float64), rec["s1"].astype(np.float64), rec["m0"]
        parent0, k0 = rec["parent0"], int(rec["k0"])
        answer = set(np.asarray(rec.get("answer0", []), int).tolist())
        group = (str(rec.get("task", "")), str(rec.get("bucket", "")))
        kids0 = [np.flatnonzero(parent0 == p) for p in range(s1_all.shape[1])]
        orphans0 = np.flatnonzero(parent0 < 0)
        n0 = s0_all.shape[1]
        for h in range(s0_all.shape[0]):
            s0, s1, m0 = s0_all[h], s1_all[h], m0_all[h]
            oracle = top(np.arange(n0), m0, k0)
            oracle_mass = m0[oracle].sum()

            def score(sel, cost, name, depth):
                sel = np.asarray(sel, int)
                rows[(group, depth, name)].append((
                    len(np.intersect1d(sel, oracle)) / max(len(oracle), 1),
                    m0[sel].sum() / oracle_mass if oracle_mass > 0 else np.nan,
                    cost / n0,
                    (len(answer & set(sel.tolist())) > 0) if answer else np.nan,
                ))

            score(top(np.arange(n0), s0, k0), n0, "flat", 0)
            score(np.random.default_rng(h).choice(n0, min(k0, n0), replace=False), 0, "random", 0)
            for depth in depths:
                parents = build_tree(s1.shape[0], depth, fanout)
                scores = pooled_scores(s1, parents, depth, pool)
                kids = {lvl: children_of(parents, lvl) for lvl in range(2, depth)}
                for beam in beams:
                    sel, cost = route_top_down(s0, scores, kids, kids0, orphans0, depth, k0, beam)
                    score(sel, cost, f"top_down x{beam:g}", depth)
                    sel, cost_bf = route_best_first(s0, scores, kids, kids0, orphans0, depth, k0, cost)
                    score(sel, cost_bf, f"best_first @cost(x{beam:g})", depth)
    return rows


def report(rows):
    by_group = collections.defaultdict(dict)
    for (group, depth, name), vals in rows.items():
        by_group[group][(depth, name)] = np.nanmean(np.asarray(vals, float), axis=0)
    for group, res in sorted(by_group.items()):
        print(f"\n## task={group[0] or '-'} bucket={group[1] or '-'}")
        print(f"{'depth':>5}  {'strategy':<24} {'recall@k':>8} {'mass':>6} {'cost/N0':>8} {'answer':>6}")
        for (depth, name), (rec, mass, cost, ans) in sorted(res.items()):
            print(f"{depth if depth else '-':>5}  {name:<24} {rec:8.3f} {mass:6.3f} {cost:8.3f} {ans:6.3f}")


def synthetic(n_calls=30, H=4, N1=150, J=4, seed=0):
    """Scores = log-mass + noise, so routing quality is known to be imperfect."""
    rng = np.random.default_rng(seed)
    for _ in range(n_calls):
        N0 = N1 * J + 7
        parent0 = np.concatenate([np.repeat(np.arange(N1), J), -np.ones(7, int)])
        logit = rng.normal(size=(H, N0)) * 2
        m0 = np.exp(logit)
        m0 /= m0.sum(1, keepdims=True)
        s0 = logit + rng.normal(size=(H, N0))
        s1 = np.stack([np.log(np.bincount(parent0[parent0 >= 0], weights=np.exp(s0[h, :N1 * J]),
                                          minlength=N1)) for h in range(H)])
        s1 += rng.normal(size=s1.shape) * 0.5
        yield {"s0": s0, "s1": s1, "parent0": parent0, "m0": m0, "k0": N0 // 112 + 1,
               "answer0": rng.choice(N0, 2, replace=False), "task": "synthetic", "bucket": ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="?")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--fanout", type=int, default=4, help="J for virtual levels")
    ap.add_argument("--pool", choices=["max", "lse", "mean"], default="lse")
    ap.add_argument("--beams", type=float, nargs="+", default=[1, 2, 4])
    ap.add_argument("--limit", type=int, default=None, help="use the first N record files")
    args = ap.parse_args()

    if args.synthetic:
        records = list(synthetic())
    else:
        files = sorted(glob.glob(os.path.join(args.records, "*.npz")))[:args.limit]
        records = [dict(np.load(f, allow_pickle=True)) for f in files]
    report(evaluate(records, args.depths, args.fanout, args.pool, args.beams))


if __name__ == "__main__":
    main()
