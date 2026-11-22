"""Long held-out agent trajectories for evaluation (ADR 0001, #23): concatenate held-out
trajectories (experiments/agent_data.py, split by issue, so none was trained on) to 256K
and 1M tokens.

Each sample keeps one system block (system prompt + tool definitions, the pinned tokens)
from its first trajectory and drops it from the others, asserting the blocks are the
same; turn ids continue across trajectories. Trajectories are taken in (repo, id) order,
so neighbours come from the same repository where the held-out split has several, and
whole trajectories are added while the sample stays within the target length. No
trajectory is used twice within one length; every sample records its sources, and the
same inputs give the same samples.

    python experiments/agent_longeval.py --src /data/agent_data --out /data/agent_longeval \
        --lens 262144 1048576 --n 20
"""

import argparse
import glob
import json
import os
from typing import Dict, List


def repo_of(instance_id: str) -> str:
    return instance_id.rsplit("-", 1)[0]


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
    ap.add_argument("--src", required=True, help="agent_data output (its parts/rg*/heldout are read)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lens", type=int, nargs="+", default=[262144, 1048576])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--row_groups", type=int, nargs="+", default=None,
                    help="which converted row groups to draw from (fixes the samples while a "
                    "conversion is still adding row groups)")
    args = ap.parse_args()
    import datasets

    parts = sorted(glob.glob(os.path.join(os.path.expanduser(args.src), "parts", "rg*", "heldout")))
    if args.row_groups is not None:
        want = {f"rg{g:02d}" for g in args.row_groups}
        parts = [p for p in parts if os.path.basename(os.path.dirname(p)) in want]
        missing = want - {os.path.basename(os.path.dirname(p)) for p in parts}
        if missing:
            raise FileNotFoundError(f"row groups not converted yet: {sorted(missing)}")
    rows = [r for p in parts for r in datasets.load_from_disk(p)]
    print(f"{len(rows)} held-out trajectories from {len(parts)} row groups", flush=True)
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
