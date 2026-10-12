"""Convert released CPT/SFT data to the streaming gist format.

Released samples are [gisted prefix, no loss][ungisted continuation, loss], so the
model never predicts text right after a gist, which decode-time insertion needs.
Each sample is rewritten: old gists are dropped, new ones are inserted over the whole
stream (src/data/gist_stream.py), and labels cover all raw tokens (`--loss all`) or
only the original continuation (`--loss continuation`).

    python data_process/convert_cpt_to_stream.py IN.arrow OUT_DIR \\
        --old-gist-ids 151646 151647 --new-gist-ids 151665 151666 \\
        --chunk-size 4 --group 4 --sink 2

New ids come from gist_utils.add_gist_tokens (one per level; Qwen2.5 uses 151646-8
for other tokens). --group takes per-level factors for deeper trees.
"""

import argparse
import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.gist_stream import insert_gists_with_map, stream_labels  # noqa: E402


def convert_example(
    input_ids: Sequence[int],
    labels: Sequence[int],
    old_gist_ids: Sequence[int],
    new_gist_ids: Sequence[int],
    chunk_size: int,
    group,
    sink: int,
    loss: str = "all",
) -> dict:
    old = set(old_gist_ids)
    raw, raw_has_loss = [], []
    for t, y in zip(input_ids, labels):
        if t in old:
            continue
        raw.append(t)
        raw_has_loss.append(y != -100)

    tokens, src = insert_gists_with_map(raw, new_gist_ids, chunk_size, group, sink=sink)
    if loss == "all":
        loss_mask = [s >= sink for s in src]   # summary tokens (-1) and sink tokens excluded
    elif loss == "continuation":
        loss_mask = [s >= 0 and raw_has_loss[s] for s in src]
    else:
        raise ValueError(f"unknown loss mode {loss!r}")
    new_labels = stream_labels(tokens, new_gist_ids, loss_mask)
    return {
        "input_ids": tokens,
        "labels": new_labels,
        "attention_mask": [1] * len(tokens),
        "length": len(tokens),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="an .arrow shard of the released CPT data")
    ap.add_argument("output", help="directory for Dataset.save_to_disk")
    ap.add_argument("--old-gist-ids", type=int, nargs="+", required=True)
    ap.add_argument("--new-gist-ids", type=int, nargs="+", required=True)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--group", type=int, nargs="+", required=True,
                    help="lower-level gists per higher-level gist: one value for all levels, "
                         "or J_1 J_2 ... per level (the last value repeats)")
    ap.add_argument("--sink", type=int, default=0, help="leading raw tokens never chunked")
    ap.add_argument("--loss", choices=["all", "continuation"], default="all")
    ap.add_argument("--num-proc", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N rows")
    args = ap.parse_args()

    from datasets import Dataset

    ds = Dataset.from_file(args.input)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    def fn(ex):
        return convert_example(ex["input_ids"], ex["labels"], args.old_gist_ids, args.new_gist_ids,
                               args.chunk_size, args.group, args.sink, args.loss)

    out = ds.map(fn, remove_columns=[c for c in ds.column_names if c != "index"], num_proc=args.num_proc)
    out.save_to_disk(args.output)
    print(f"wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
