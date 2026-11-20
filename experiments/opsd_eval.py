"""Before/after comparison for OPSD (#8) on fixed held-out prompts.

For each checkpoint (stage-1 adapters, then the OPSD adapters), roll out on the same
held-out prompts (the last --n rows of the spike SFT data, unused in training) with a
fixed seed and report the mean reverse KL to the base model on generated tokens and the
visible keys per generated row. Forward KL (teacher mass the student drops) and top-1
agreement with the base model guard against reverse KL falling by mode collapse.

    ATTN_IMPL=ref python experiments/opsd_eval.py --data ~/spike_data/sft \
        --ckpts runs/s3_d_joint/adapters.pt runs/s4b/adapters.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datasets  # noqa: E402
import torch  # noqa: E402
from transformers import Qwen2ForCausalLM  # noqa: E402

import gist_opsd  # noqa: E402
from gist_opsd_trainer import load_stage1, regist  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--ctx_len", type=int, default=2048)
    ap.add_argument("--n_new", type=int, default=64)
    args = ap.parse_args()
    assert os.environ.get("ATTN_IMPL") == "ref"
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    rows = datasets.load_from_disk(os.path.expanduser(args.data))
    prompts = []
    for i in range(len(rows) - 1, -1, -1):
        r = rows[i]
        start = next(j for j, l in enumerate(r["labels"]) if l != -100)
        if start >= args.ctx_len:
            prompts.append(r["input_ids"])
        if len(prompts) == args.n:
            break
    teacher = (
        Qwen2ForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, attn_implementation="sdpa"
        )
        .to(dev)
        .eval()
    )
    for ck in args.ckpts:
        _, vocab, model, _ = load_stage1(ck, args.model, dev)
        model.eval()
        gen = torch.Generator(device=dev).manual_seed(0)
        rkl, fkl, agree, keys = [], [], [], []
        for p in prompts:
            ctx = torch.tensor([regist(p)[: args.ctx_len]], device=dev)
            y, lp, steps = gist_opsd.rollout(model, ctx, args.n_new, vocab, gen)
            lp_t = gist_opsd.teacher_logprobs(teacher, ctx, y, vocab)
            rkl.append(float(gist_opsd.reverse_kl(lp, lp_t).mean()))
            fkl.append(float(gist_opsd.forward_kl(lp, lp_t).mean()))
            agree.append(float((lp.argmax(-1) == lp_t.argmax(-1)).float().mean()))
            keys.append(
                float(
                    torch.stack(
                        [
                            torch.stack([m.sum().float() / m.shape[0] for m in st[1:]]).mean()
                            for st in steps
                        ]
                    ).mean()
                )
            )
        print(
            json.dumps(
                {
                    "ckpt": ck,
                    "n": len(prompts),
                    "rkl_mean": sum(rkl) / len(rkl),
                    "fkl_mean": sum(fkl) / len(fkl),
                    "top1_agree": sum(agree) / len(agree),
                    "keys_per_row": sum(keys) / len(keys),
                    "rkl_per_prompt": [round(x, 4) for x in rkl],
                    "fkl_per_prompt": [round(x, 4) for x in fkl],
                    "agree_per_prompt": [round(x, 4) for x in agree],
                }
            ),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
