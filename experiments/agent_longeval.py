"""Long held-out agent trajectories for evaluation (ADR 0001, #23): concatenate held-out
trajectories (experiments/agent_data.py, split by issue, so none was trained on) to 256K
and 1M tokens.

Each sample keeps one system block (system prompt + tool definitions, the pinned tokens)
from its first trajectory and drops it from the others, asserting the blocks are the
same; turn ids continue across trajectories. Trajectories are taken in (repo, id) order,
so neighbours come from the same repository where the held-out split has several, and
whole trajectories are added while the sample stays within the target length. Each
issue contributes one trajectory (its longest attempt), so a sample never repeats a
task; no trajectory is used twice within one length; every sample records its sources,
and the same inputs give the same samples.

    python experiments/agent_longeval.py --src /data/agent_data --out /data/agent_longeval \
        --lens 262144 1048576 --n 20
"""

import argparse
import json
import os
from typing import Dict, List


def repo_of(instance_id: str) -> str:
    return instance_id.rsplit("-", 1)[0]


def one_per_issue(rows: List[Dict]) -> List[Dict]:
    """The longest attempt of each issue (ties: the lower trajectory id). The dataset has a
    median of 7 attempts per issue; consecutive attempts at one task repeat what the
    earlier ones read, which would make long-range retrieval look easier than it is."""
    best: Dict[str, Dict] = {}
    for r in rows:
        b = best.get(r["instance_id"])
        key = (-len(r["input_ids"]), r["trajectory_id"])
        if b is None or key < (-len(b["input_ids"]), b["trajectory_id"]):
            best[r["instance_id"]] = r
    return list(best.values())


def concat(rows: List[Dict], target: int, n: int) -> List[Dict]:
    """Pack rows (input_ids, pinned, turn, trajectory_id, instance_id) into up to n samples
    of at most target tokens, in (repo, trajectory_id) order, each row used once."""
    rows = sorted(rows, key=lambda r: (repo_of(r["instance_id"]), r["trajectory_id"]))
    out, i = [], 0
    while len(out) < n and i < len(rows):
        first = rows[i]
        k = sum(first["pinned"])
        assert all(first["pinned"][:k]) and not any(first["pinned"][k:]), "pinned must be a prefix"
        head = first["input_ids"][:k]
        ids, pinned, turn = list(head), [True] * k, [0] * k
        sources, last_turn = [], 0
        while i < len(rows):
            r = rows[i]
            kr = sum(r["pinned"])
            if r["input_ids"][:kr] != head:
                raise ValueError(f"{r['trajectory_id']}: system block differs from the sample's first")
            body, bturn = r["input_ids"][kr:], r["turn"][kr:]
            if sources and len(ids) + len(body) > target:
                break
            i += 1
            if len(ids) + len(body) > target:  # a single trajectory longer than the target
                continue
            ids += body
            pinned += [False] * len(body)
            turn += [last_turn + t for t in bturn]  # a trajectory's turns start at 0 (its task)
            last_turn = turn[-1] + 1
            sources.append(r["trajectory_id"])
        if sources:
            out.append({"input_ids": ids, "pinned": pinned, "turn": turn, "length": len(ids),
                        "sources": sources, "repos": sorted({repo_of(x["instance_id"]) for x in rows
                                                             if x["trajectory_id"] in set(sources)})})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="agent_data output (its heldout split is read)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lens", type=int, nargs="+", default=[262144, 1048576])
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    import datasets

    parts = [os.path.join(os.path.expanduser(args.src), "heldout")]
    rows = list(datasets.load_from_disk(parts[0]))
    n_traj = len(rows)
    rows = one_per_issue(rows)
    print(f"{n_traj} held-out trajectories, {len(rows)} issues (longest attempt each)", flush=True)
    report = {}
    for L in args.lens:
        samples = concat(rows, L, args.n)
        d = os.path.join(os.path.expanduser(args.out), f"len{L}")
        datasets.Dataset.from_list(samples).save_to_disk(d)
        lens = [s["length"] for s in samples]
        report[L] = {"samples": len(samples), "min": min(lens), "max": max(lens),
                     "trajectories_per_sample": sum(len(s["sources"]) for s in samples) / len(samples)}
        print(L, json.dumps(report[L]), flush=True)
        if len(samples) < args.n:
            print(f"WARNING: only {len(samples)} samples of {L}: convert more row groups", flush=True)
    for L in args.lens:  # sources per sample, for reproducing or auditing
        ds = datasets.load_from_disk(os.path.join(os.path.expanduser(args.out), f"len{L}"))
        report[L]["sources"] = ds["sources"]
    json.dump({"src_parts": parts, "lens": report},
              open(os.path.join(os.path.expanduser(args.out), "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
