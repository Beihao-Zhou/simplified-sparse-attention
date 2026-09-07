"""Cache simulation over per-decode-step selected-KV traces.

One *trace* is the sequence of selected logical KV record ids for a single
(request, layer, KV-group) triple. Simulation is always independent per trace:
a hot buffer in the real system is per layer and per KV group, so mixing them
would both inflate the key space and invent sharing that cannot happen.

Semantics (fixed once, used by every policy):

  At the start of step ``t`` the buffer holds a set ``C``. The step needs
  ``S_t``.  ``hits = |S_t & C|`` -- "already resident before this step" --
  and ``misses = S_t - C`` must be transferred from host. Only then is the
  buffer updated: evict enough non-required entries to fit, admit the misses.
  Entries of ``S_t`` are never evicted during step ``t``; with a capacity of at
  least ``max_t |S_t|`` that is always satisfiable, which is why capacities are
  quoted as multiples of ``k_max`` rather than of the mean.

A miss is a compulsory (cold) miss if that key had never been requested in this
trace before, and a capacity miss otherwise. Reporting them apart matters: on a
short trace the compulsory term dominates and no policy, at any capacity, can
beat ``1 - unique/total``.

Key ids are logical KV positions, so they are dense small non-negative ints;
membership is tracked with flag arrays rather than hashing to keep 10^8-event
traces tractable.
"""

from __future__ import annotations

import numpy as np

POLICIES = ("topk_only", "fifo", "lru", "random", "belady")

_INF = np.int64(np.iinfo(np.int64).max)


def _next_use(steps):
    """For every occurrence, the next step index at which that key recurs.

    ``_INF`` when it never recurs. Vectorised: sort all (key, step) occurrences
    by key then step, and read off the successor within each key run.
    """
    sizes = np.array([len(s) for s in steps], dtype=np.int64)
    if not len(sizes) or sizes.sum() == 0:
        return [np.zeros(0, np.int64) for _ in steps]
    keys = np.concatenate([np.asarray(s, np.int64) for s in steps])
    tags = np.repeat(np.arange(len(steps), dtype=np.int64), sizes)
    order = np.lexsort((tags, keys))
    k_sorted, t_sorted = keys[order], tags[order]
    nxt = np.empty(len(keys), dtype=np.int64)
    nxt[:-1] = np.where(k_sorted[:-1] == k_sorted[1:], t_sorted[1:], _INF)
    nxt[-1] = _INF
    out = np.empty(len(keys), dtype=np.int64)
    out[order] = nxt
    return list(np.split(out, np.cumsum(sizes)[:-1]))


def simulate(steps, capacity, policy="lru", seed=0, local_keys=None,
             record_misses=False):
    """Simulate one trace. Returns per-step arrays plus scalar totals.

    ``steps``      list of np.ndarray, each holding the unique selected key ids
                   of one decode step (order irrelevant, duplicates rejected).
    ``capacity``   buffer size in KV records.
    ``local_keys`` optional list of boolean arrays, aligned with ``steps``,
                   marking keys produced by this decode run (tokens the model
                   just generated). Those are written on the GPU and never need
                   a host transfer, so they are counted separately rather than
                   silently dropped from the miss count.
    ``record_misses`` also return, per step, the exact key ids that would have
                   to be fetched -- what a real hot buffer would put on the wire.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    steps = [np.asarray(s, dtype=np.int64) for s in steps]
    T = len(steps)
    z = lambda: np.zeros(max(T, 0), dtype=np.int64)  # noqa: E731
    res = dict(requests=z(), hits=z(), misses=z(), cold_misses=z(),
               set_size=z(), carryover=z(), union_prev=z(),
               local=z(), local_miss=z())
    miss_sets = [] if record_misses else None
    res["miss_sets"] = miss_sets
    if T == 0:
        return _finalise(res, capacity, policy)

    k_max = max(len(s) for s in steps)
    if capacity < k_max:
        raise ValueError(
            f"capacity {capacity} < max selected-set size {k_max}: the required "
            "set does not fit, which makes the eviction problem ill-posed")

    n_keys = int(max(s.max() for s in steps if len(s)) + 1) if any(len(s) for s in steps) else 1
    resident = np.zeros(n_keys, dtype=bool)   # in buffer
    in_step = np.zeros(n_keys, dtype=bool)    # required this step
    seen = np.zeros(n_keys, dtype=bool)       # ever requested
    rng = np.random.default_rng(seed)
    nxt = _next_use(steps) if policy == "belady" else None

    cache = np.zeros(0, dtype=np.int64)       # resident key ids
    score = np.zeros(0, dtype=np.float64)     # lower score == evicted first
    prev = np.zeros(0, dtype=np.int64)

    for t, s in enumerate(steps):
        if len(np.unique(s)) != len(s):
            raise ValueError(f"step {t} contains duplicate key ids")
        res["set_size"][t] = res["requests"][t] = len(s)
        if t:
            in_step[prev] = True
            inter = int(in_step[s].sum())
            in_step[prev] = False
            res["carryover"][t] = inter
            res["union_prev"][t] = prev.size + s.size - inter

        if policy == "topk_only":
            in_step[prev] = True
            hit_mask = in_step[s].copy()
            in_step[prev] = False
        else:
            hit_mask = resident[s]
        miss = s[~hit_mask]
        res["hits"][t] = int(hit_mask.sum())
        res["misses"][t] = len(miss)
        res["cold_misses"][t] = int((~seen[miss]).sum())
        if miss_sets is not None:
            miss_sets.append(miss.copy())
        seen[s] = True
        if local_keys is not None:
            loc = np.asarray(local_keys[t], dtype=bool)
            res["local"][t] = int(loc.sum())
            res["local_miss"][t] = int(loc[~hit_mask].sum())

        if policy == "topk_only":
            prev = s
            continue

        in_step[s] = True
        # --- evict non-required entries until the misses fit, then admit ---
        overflow = cache.size + miss.size - capacity
        if overflow > 0:
            cand = np.flatnonzero(~in_step[cache])
            if cand.size < overflow:  # unreachable while capacity >= k_max
                raise AssertionError("not enough evictable entries")
            # Random replacement draws a fresh victim each eviction rather than
            # freezing a priority at admission time.
            key = rng.random(cand.size) if policy == "random" else score[cand]
            victim = cand[np.argpartition(key, overflow - 1)[:overflow]]
            keep = np.ones(cache.size, dtype=bool)
            keep[victim] = False
            resident[cache[victim]] = False
            cache, score = cache[keep], score[keep]

        if policy in ("lru", "fifo"):
            new_score = np.full(miss.size, float(t))
        elif policy == "random":
            new_score = rng.random(miss.size)
        else:  # belady: evict furthest next use first -> score = -next_use
            new_score = -_as_float(nxt[t][~hit_mask])
        cache = np.concatenate([cache, miss])
        score = np.concatenate([score, new_score])
        resident[miss] = True

        # LRU and Belady also refresh the priority of the keys that hit.
        if policy == "lru":
            score[np.flatnonzero(in_step[cache])] = float(t)
        elif policy == "belady":
            pos = np.flatnonzero(in_step[cache])
            score[pos[np.argsort(cache[pos])]] = -_as_float(nxt[t][np.argsort(s)])
        in_step[s] = False
        prev = s

    return _finalise(res, capacity, policy)


def _as_float(nxt_slice):
    f = nxt_slice.astype(np.float64)
    f[nxt_slice == _INF] = np.inf
    return f


def _finalise(res, capacity, policy):
    tot = int(res["requests"].sum())
    out = dict(res)
    out.pop("miss_sets", None)
    out["miss_sets"] = res.get("miss_sets")
    out.update(capacity=capacity, policy=policy, steps=len(res["requests"]),
               total_requests=tot,
               total_hits=int(res["hits"].sum()),
               total_misses=int(res["misses"].sum()),
               total_cold=int(res["cold_misses"].sum()),
               total_local=int(res["local"].sum()),
               total_local_miss=int(res["local_miss"].sum()))
    d = tot if tot else float("nan")
    out["hit_rate"] = out["total_hits"] / d
    out["miss_rate"] = out["total_misses"] / d
    out["cold_miss_rate"] = out["total_cold"] / d
    out["capacity_miss_rate"] = (out["total_misses"] - out["total_cold"]) / d
    return out


def window(res, lo, hi):
    """Aggregate a simulation result over decode steps [lo, hi) (0-based).

    The cache state is *not* reset -- the window only selects which steps are
    counted, so a late window reports a warmed buffer.
    """
    sl = slice(lo, min(hi, res["steps"]))
    req = int(res["requests"][sl].sum())
    if req == 0:
        return None
    hit = int(res["hits"][sl].sum())
    cold = int(res["cold_misses"][sl].sum())
    return dict(lo=lo, hi=min(hi, res["steps"]), n_steps=int(res["requests"][sl].size),
                requests=req, hits=hit, hit_rate=hit / req,
                miss_rate=(req - hit) / req, cold_miss_rate=cold / req,
                capacity_miss_rate=(req - hit - cold) / req,
                mean_set_size=float(res["set_size"][sl].mean()))


def ceiling(steps):
    """Compulsory-miss ceiling: the hit rate of an infinite buffer."""
    steps = [np.asarray(s, dtype=np.int64) for s in steps]
    tot = int(sum(len(s) for s in steps))
    if not tot:
        return float("nan")
    n = int(max(s.max() for s in steps if len(s)) + 1)
    seen = np.zeros(n, dtype=bool)
    for s in steps:
        seen[s] = True
    return 1.0 - int(seen.sum()) / tot
