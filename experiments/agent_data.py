"""Agent trajectories -> raw token ids with segment labels (ADR 0001, #17).

Reads nebius/SWE-rebench-openhands-trajectories row group by row group over HTTP range
requests (one 2 GB parquet), renders each trajectory with the model's chat template
(tools included), and labels every token:

    pinned  the system block (system prompt + tool definitions): never compressed (R7)
    turn    0 for the system prompt and the task; +1 at every assistant message, so an
            assistant message and the tool results after it share a turn (R6)

Message boundaries come from rendering message prefixes: message i spans from the
longest common prefix of render(m[:i]) and render(m[:i+1]); no template markers are
parsed. Output is two `datasets` directories (train / heldout, split by issue so one
issue never lands in both) with input_ids (raw, no summaries), pinned, turn, length.

    python experiments/agent_data.py --out ~/agent_data --row_groups 0 1
"""

import argparse
import collections
import hashlib
import json
import os

import numpy as np

REPO = "nebius/SWE-rebench-openhands-trajectories"
CLIPPED = ("Observation truncated", "<response clipped>")


def _strip_none(x):
    """The parquet schema unions every tool's parameters; drop the absent (None) fields."""
    if isinstance(x, dict):
        return {k: _strip_none(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [_strip_none(v) for v in x]
    return x


def messages_of(row):
    out = []
    for m in row["trajectory"]:
        msg = {"role": m["role"], "content": m["content"] or ""}
        if m["role"] == "assistant" and m.get("tool_calls") is not None:
            msg["tool_calls"] = [
                {"type": "function", "function": {"name": c["function"]["name"],
                                                  "arguments": json.loads(c["function"]["arguments"])}}
                for c in m["tool_calls"]]
        if m["role"] == "tool":
            msg["name"] = m.get("name")
        out.append(msg)
    return out


def _lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def label(tok, messages, tools):
    """(ids, pinned, turn) of the rendered trajectory."""
    render = lambda ms: tok.apply_chat_template(ms, tools=tools, tokenize=False) if ms else ""  # noqa: E731
    full = render(messages)
    starts, prev = [], ""
    for i in range(len(messages)):
        cur = render(messages[: i + 1])
        starts.append(_lcp(prev, cur))
        prev = cur
    assert full.startswith(prev[: starts[-1]])
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    msg_turn, t = [], 0
    for m in messages:
        if m["role"] == "assistant":
            t += 1
        msg_turn.append(t)
    bounds = np.array(starts[1:] + [len(full) + 1])
    ids, pinned, turn = enc["input_ids"], [], []
    for a, _ in enc["offset_mapping"]:
        i = int(np.searchsorted(bounds, a, side="right"))  # message whose span holds char a
        pinned.append(messages[i]["role"] == "system")
        turn.append(msg_turn[i])
    return ids, pinned, turn


def heldout(instance_id, frac):
    return int(hashlib.sha1(instance_id.encode()).hexdigest(), 16) % 10_000 < frac * 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="tokenizer and chat template")
    ap.add_argument("--row_groups", type=int, nargs="+", default=[0])
    ap.add_argument("--max_rows", type=int, default=0, help="per row group, 0 = all")
    ap.add_argument("--heldout_frac", type=float, default=0.02)
    ap.add_argument("--cache", default="", help="directory to keep downloaded row groups in")
    args = ap.parse_args()

    import datasets
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    pf = pq.ParquetFile(HfFileSystem().open(f"datasets/{REPO}/trajectories.parquet"))
    rows = {"train": collections.defaultdict(list), "heldout": collections.defaultdict(list)}
    audit = collections.Counter()
    tool_chars = []
    for rg in args.row_groups:
        path = os.path.join(os.path.expanduser(args.cache), f"rg{rg:02d}.parquet") if args.cache else ""
        if path and os.path.exists(path):
            table = pq.read_table(path)
        else:
            table = pf.read_row_group(rg)
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                pq.write_table(table, path)
        n = table.num_rows if not args.max_rows else min(args.max_rows, table.num_rows)
        for r in table.slice(0, n).to_pylist():
            msgs = messages_of(r)
            for m in msgs:
                if m["role"] == "tool":
                    tool_chars.append(len(m["content"]))
                    audit["tool_msgs"] += 1
                    audit["tool_clipped"] += any(c in m["content"] for c in CLIPPED)
            ids, pinned, turn = label(tok, msgs, _strip_none(r["tools"]))
            split = "heldout" if heldout(r["instance_id"], args.heldout_frac) else "train"
            d = rows[split]
            d["input_ids"].append(ids)
            d["pinned"].append(pinned)
            d["turn"].append(turn)
            d["length"].append(len(ids))
            d["n_pinned"].append(int(sum(pinned)))
            d["n_turns"].append(turn[-1])
            d["trajectory_id"].append(r["trajectory_id"])
            d["instance_id"].append(r["instance_id"])
            d["resolved"].append(int(r["resolved"] or 0))
        print(f"row group {rg}: {n} rows", flush=True)
    for split, d in rows.items():
        if not d:
            continue
        datasets.Dataset.from_dict(dict(d)).save_to_disk(os.path.join(os.path.expanduser(args.out), split))
    L = np.array(rows["train"]["length"] + rows["heldout"]["length"])
    P = np.array(rows["train"]["n_pinned"] + rows["heldout"]["n_pinned"])
    tc = np.array(tool_chars)
    report = {
        "trajectories": int(L.size),
        "train": len(rows["train"]["length"]),
        "heldout": len(rows["heldout"]["length"]),
        "tokens_p50_p90_max": [int(np.percentile(L, 50)), int(np.percentile(L, 90)), int(L.max())],
        "frac_over_32K": float((L > 32768).mean()),
        "frac_over_128K": float((L > 131072).mean()),
        "pinned_tokens_p50_max": [int(np.percentile(P, 50)), int(P.max())],
        "tool_msgs": audit["tool_msgs"],
        "tool_msgs_clipped": audit["tool_clipped"],
        "tool_chars_p50_p99_max": [int(np.percentile(tc, 50)), int(np.percentile(tc, 99)), int(tc.max())],
    }
    print(json.dumps(report, indent=1))
    json.dump(report, open(os.path.join(os.path.expanduser(args.out), "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
