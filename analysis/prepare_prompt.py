"""Extract one LongBench / LongBench-v2 prompt to JSON.

Kept separate from analysis/record_selection.py on purpose: the recorder runs
against the transformers version this repo pins (4.57.1, which needs
huggingface-hub<1.0), while `datasets` in this environment needs hub>=1.0.
Running the two in separate processes avoids pinning one against the other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LB = REPO / "benchmark/Longbench_exp/LongBench"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["longbench", "longbench_v2"], required=True)
    ap.add_argument("--dataset", default="multi_news")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=0,
                    help="v2 only: pick the shortest sample above this context size")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_from_disk

    if args.source == "longbench":
        data = load_from_disk(str(LB / args.dataset))
        fmt = json.load(open(REPO / "benchmark/Longbench_exp/longbench_config/dataset2prompt.json"))[args.dataset]
        obj = data[args.sample]
        prompt = fmt.format(**obj)
        tail = fmt.split("}", 1)[1].format(**obj)
        info = dict(answers=obj.get("answers"), length=obj.get("length"))
    else:
        data = load_from_disk(str(LB / "v2"))
        lens = [len(c) for c in data["context"]]
        cand = [i for i, n in enumerate(lens) if n >= args.min_chars]
        if not cand:
            cand = [max(range(len(lens)), key=lambda i: lens[i])]
        cand.sort(key=lambda i: lens[i])
        obj = data[cand[min(args.sample, len(cand) - 1)]]
        tail = (f"\n\nQuestion: {obj['question']}\nAnswer the question in detail, "
                f"explaining your reasoning step by step.\nAnswer:")
        prompt = obj["context"] + tail
        info = dict(answer=obj.get("answer"), difficulty=obj.get("difficulty"),
                    domain=obj.get("domain"), context_chars=len(obj["context"]))

    out = dict(source=args.source, dataset=args.dataset, sample=args.sample,
               prompt=prompt, tail=tail, info=info)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}: {len(prompt)} chars, tail {len(tail)} chars, {info}")


if __name__ == "__main__":
    main()
