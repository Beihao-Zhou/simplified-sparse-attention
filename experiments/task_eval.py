"""Task accuracy of OPSD checkpoints: passkey retrieval and LongBench QA F1.

Each checkpoint decodes greedily through the sparse inference path (gist prefill, then
per-step top-down selection, ATTN_IMPL=ref), on prompts with summaries inserted by the
streaming rule the spike data uses (chunk 4, group 4, the checkpoint's sink). `base` is the original
model reading the raw prompt: the ceiling. Prompts are chat-templated, LongBench contexts
are truncated in the middle to --ctx_len tokens.

    ATTN_IMPL=ref python experiments/task_eval.py --ckpts base runs/s3_d_joint/adapters.pt \
        runs/s4b/adapters.pt --out results/task_eval.jsonl
"""

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark", "Longbench_exp"))

import datasets  # noqa: E402
import torch  # noqa: E402
from longbench_metrics import qa_f1_score  # noqa: E402
from transformers import AutoTokenizer, Qwen2ForCausalLM  # noqa: E402

import gist_opsd  # noqa: E402
from gist_opsd_trainer import load_stage1  # noqa: E402
from gist_utils import Global_data  # noqa: E402
from src.data.gist_stream import insert_gists_with_map  # noqa: E402

CFG = os.path.join(
    os.path.dirname(__file__), "..", "benchmark", "Longbench_exp", "longbench_config"
)
FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
)


def passkey_items(tok, ctx_len, n, seed):
    rng = random.Random(seed)
    unit = len(tok(FILLER, add_special_tokens=False).input_ids)
    reps = (ctx_len - 120) // unit
    out = []
    for _ in range(n):
        key = str(rng.randint(10000, 99999))
        at = rng.randint(0, reps)
        text = (
            "There is an important pass key hidden inside a lot of irrelevant text. "
            "Find it and memorize it.\n\n"
            + FILLER * at
            + f"The pass key is {key}. Remember it. {key} is the pass key. "
            + FILLER * (reps - at)
            + "\n\nWhat is the pass key? Answer with the number only."
        )
        out.append({"prompt": text, "answers": [key], "max_new": 16, "depth": at / max(reps, 1)})
    return out


def longbench_items(tok, task, data_dir, ctx_len, n):
    fmt = json.load(open(os.path.join(CFG, "dataset2prompt.json")))[task]
    max_new = json.load(open(os.path.join(CFG, "dataset2maxlen.json")))[task]
    rows = datasets.load_from_disk(os.path.join(os.path.expanduser(data_dir), task))
    out = []
    for r in rows.select(range(min(n, len(rows)))):
        ids = tok(fmt.format(**r), add_special_tokens=False).input_ids
        budget = ctx_len - 40  # chat template
        if len(ids) > budget:
            half = budget // 2
            ids = ids[:half] + ids[-half:]
        out.append({"prompt": tok.decode(ids), "answers": r["answers"], "max_new": max_new})
    return out


def score(task, pred, answers):
    if task == "passkey":
        m = re.search(r"\d{5}", pred)
        return float(m is not None and m.group(0) in answers)
    if task == "triviaqa":
        pred = pred.lstrip("\n").split("\n")[0]
    return max(qa_f1_score(pred, a) for a in answers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ckpts", nargs="+", required=True, help="'base' or adapters.pt paths")
    ap.add_argument("--longbench_dir", default="~/longbench_data")
    ap.add_argument("--tasks", nargs="+", default=["passkey", "triviaqa", "hotpotqa", "2wikimqa"])
    ap.add_argument("--ctx_len", type=int, default=4096)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert os.environ.get("ATTN_IMPL") == "ref"
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    items = {
        t: (
            passkey_items(tok, args.ctx_len, args.n, seed=0)
            if t == "passkey"
            else longbench_items(tok, t, args.longbench_dir, args.ctx_len, args.n)
        )
        for t in args.tasks
    }
    out = open(args.out, "a")
    for ck in args.ckpts:
        if ck == "base":
            model = (
                Qwen2ForCausalLM.from_pretrained(
                    args.model, dtype=torch.float32, attn_implementation="sdpa"
                )
                .to(dev)
                .eval()
            )
        else:
            _, vocab, model, _ = load_stage1(ck, args.model, dev)
            model.eval()
        for task, its in items.items():
            scores = []
            for i, it in enumerate(its):
                raw = tok.apply_chat_template(
                    [{"role": "user", "content": it["prompt"]}], add_generation_prompt=True
                )
                if ck == "base":
                    with torch.no_grad():
                        g = model.generate(
                            torch.tensor([raw], device=dev),
                            max_new_tokens=it["max_new"],
                            do_sample=False,
                        )
                    y = g[0, len(raw) :].tolist()
                else:
                    ids = insert_gists_with_map(
                        raw, Global_data.gist_token_id, 4, 4, sink=Global_data.sink_size
                    )[0]
                    ctx = torch.tensor([ids], device=dev)
                    y, _, _ = gist_opsd.rollout(model, ctx, it["max_new"], vocab, greedy=True)
                    y = y.tolist()
                if tok.eos_token_id in y:
                    y = y[: y.index(tok.eos_token_id)]
                pred = tok.decode(y, skip_special_tokens=True)
                s = score(task, pred, it["answers"])
                scores.append(s)
                rec = {"ckpt": ck, "task": task, "i": i, "score": s, "pred": pred}
                if "depth" in it:
                    rec["depth"] = it["depth"]
                out.write(json.dumps(rec) + "\n")
                out.flush()
            print(
                json.dumps(
                    {"ckpt": ck, "task": task, "n": len(scores), "mean": sum(scores) / len(scores)}
                ),
                flush=True,
            )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
