"""Long held-out evaluation trajectories (experiments/agent_longeval.py, #23)."""

import random

import pytest

from experiments.agent_longeval import concat

SYS = [7, 7, 8, 9]


def _row(i, repo, n_turns, rng):
    ids, pinned, turn = list(SYS), [True] * len(SYS), [0] * len(SYS)
    ids += [rng.randint(100, 999) for _ in range(5)]  # the task: turn 0
    pinned += [False] * 5
    turn += [0] * 5
    for t in range(1, n_turns + 1):
        k = rng.randint(3, 30)
        ids += [rng.randint(100, 999) for _ in range(k)]
        pinned += [False] * k
        turn += [t] * k
    return {"input_ids": ids, "pinned": pinned, "turn": turn, "trajectory_id": f"t{i:03d}",
            "instance_id": f"{repo}-{i}"}


def test_samples_keep_one_system_block_and_continue_turns():
    rng = random.Random(0)
    rows = [_row(i, f"org__r{i % 4}", rng.randint(2, 9), rng) for i in range(60)]
    samples = concat(rows, 600, 5)
    assert len(samples) == 5
    used = [s for x in samples for s in x["sources"]]
    assert len(used) == len(set(used)), "a trajectory is used once per length"
    by_id = {r["trajectory_id"]: r for r in rows}
    for s in samples:
        assert s["length"] <= 600 and s["length"] == len(s["input_ids"])
        k = sum(s["pinned"])
        assert k == len(SYS) and all(s["pinned"][:k]) and s["input_ids"][:k] == SYS
        assert all(b >= a for a, b in zip(s["turn"], s["turn"][1:])), "turns never go back"
        body = sum(len(by_id[t]["input_ids"]) - len(SYS) for t in s["sources"])
        assert s["length"] == len(SYS) + body
        # each source starts a new turn: its task is not merged into the previous turn
        assert s["turn"][-1] >= sum(by_id[t]["turn"][-1] for t in s["sources"])
    # neighbours come from the same repository first
    assert len(samples[0]["repos"]) == 1


def test_trajectories_longer_than_the_target_are_skipped():
    rng = random.Random(1)
    rows = [_row(0, "a__a", 40, rng), _row(1, "a__a", 2, rng)]
    s = concat(rows, 200, 3)
    assert [x["sources"] for x in s] == [["t001"]]


def test_a_different_system_block_is_an_error():
    rng = random.Random(2)
    a, b = _row(0, "a__a", 2, rng), _row(1, "a__a", 2, rng)
    b["input_ids"][0] = 6
    with pytest.raises(ValueError):
        concat([a, b], 10**6, 1)


def test_one_attempt_per_issue():
    from experiments.agent_longeval import one_per_issue

    rng = random.Random(3)
    rows = [_row(i, f"org__r{i % 3}", rng.randint(2, 9), rng) for i in range(12)]
    for r in rows:
        r["instance_id"] = f"org__r{int(r['trajectory_id'][1:]) % 3}-1"  # 3 issues, 4 attempts
    kept = one_per_issue(rows)
    assert len(kept) == 3 and len({r["instance_id"] for r in kept}) == 3
    for r in kept:
        same = [x for x in rows if x["instance_id"] == r["instance_id"]]
        assert len(r["input_ids"]) == max(len(x["input_ids"]) for x in same)
