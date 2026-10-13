"""Sanity properties of the deep-routing simulator (experiments/deep_routing.py)."""

import numpy as np

from experiments.deep_routing import (
    build_tree,
    children_of,
    evaluate,
    pooled_scores,
    route_best_first,
    route_top_down,
    synthetic,
)


def _setup(depth, N1=64, J=4, seed=0):
    rng = np.random.default_rng(seed)
    parent0 = np.concatenate([np.repeat(np.arange(N1), J), -np.ones(3, int)])
    s0 = rng.normal(size=len(parent0))
    s1 = np.log(np.bincount(parent0[parent0 >= 0], weights=np.exp(s0[:N1 * J]), minlength=N1))
    parents = build_tree(N1, depth, J)
    scores = pooled_scores(s1, parents, depth, "lse")
    kids = {lvl: children_of(parents, lvl) for lvl in range(2, depth)}
    kids0 = [np.flatnonzero(parent0 == p) for p in range(N1)]
    return s0, scores, kids, kids0, np.flatnonzero(parent0 < 0)


def test_tree_shapes():
    parents = build_tree(70, 4, 4)
    assert parents[1].max() == 17 and parents[2].max() == 4
    kids = children_of(parents, 3)
    assert sorted(np.concatenate(kids).tolist()) == list(range(18))


def test_wide_beam_equals_flat():
    """With a beam wide enough to keep every node, top-down is flat top-k."""
    for depth in (2, 3, 4):
        s0, scores, kids, kids0, orphans = _setup(depth)
        k0 = 10
        sel, _ = route_top_down(s0, scores, kids, kids0, orphans, depth, k0, beam=1000)
        assert set(sel) == set(np.argsort(-s0)[:k0])


def test_best_first_with_full_budget_equals_flat():
    s0, scores, kids, kids0, orphans = _setup(4)
    sel, _ = route_best_first(s0, scores, kids, kids0, orphans, 4, 10, budget=10**9)
    assert set(sel) == set(np.argsort(-s0)[:10])


def test_orphans_always_candidates():
    s0, scores, kids, kids0, orphans = _setup(3)
    s0[orphans] = 100.0
    sel, _ = route_top_down(s0, scores, kids, kids0, orphans, 3, 3, beam=1)
    assert set(orphans) <= set(sel)


def test_evaluate_runs_and_flat_beats_random():
    rows = evaluate(list(synthetic(n_calls=5)), [2, 3], 4, "lse", [1])
    mean = {name: np.nanmean([r[0] for r in v]) for (_, _, name), v in rows.items()}
    assert mean["flat"] > 10 * mean["random"]
