"""Resumable training on preemptible GPUs (gist_stage1_trainer, gist_opsd_trainer).

Every --save_every steps, and at the first step boundary after SIGTERM (spot VMs get
60 s of notice), <out>/ckpt.pt is written atomically with the trained parameters, the
optimizer, the step and the RNG states. --resume continues from it (the data order is
a function of the seed and the step, so the run continues exactly) and is a no-op once
<out>/adapters.pt exists, so a supervisor can rerun the same command until it succeeds.
"""

import json
import os
import signal

import torch

CKPT = "ckpt.pt"
PREEMPTED = 143  # exit code after a SIGTERM checkpoint: rerun with --resume


class Preemption:
    """SIGTERM sets .requested; the training loop checkpoints and exits at the next step."""

    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGTERM, self._on_term)

    def _on_term(self, *_):
        print("SIGTERM: checkpointing after this step", flush=True)
        self.requested = True


def save(out, step, model, params, opt, generators=()):
    ids = {id(p) for p in params}
    ck = {
        "step": step,
        "params": {n: p.detach().cpu() for n, p in model.named_parameters() if id(p) in ids},
        "opt": opt.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "generators": [g.get_state() for g in generators],
    }
    path = os.path.join(out, CKPT)
    torch.save(ck, path + ".tmp")
    os.replace(path + ".tmp", path)  # a kill mid-write leaves the previous checkpoint


def load(out, model, opt, generators=()):
    """Restore <out>/ckpt.pt into model/opt/generators; returns its step, or 0 if absent."""
    path = os.path.join(out, CKPT)
    if not os.path.exists(path):
        return 0
    ck = torch.load(path, map_location="cpu", weights_only=False)
    named = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in ck["params"].items():
            named[n].copy_(v)
    opt.load_state_dict(ck["opt"])
    torch.set_rng_state(ck["rng_cpu"])
    if ck["rng_cuda"]:
        torch.cuda.set_rng_state_all(ck["rng_cuda"])
    for g, s in zip(generators, ck["generators"]):
        g.set_state(s)
    trim_log(os.path.join(out, "log.jsonl"), ck["step"])
    print(f"resumed from {path} at step {ck['step']}", flush=True)
    return ck["step"]


def trim_log(path, step):
    """Drop log records past the checkpoint (steps that will be rerun)."""
    if not os.path.exists(path):
        return
    keep = []
    for line in open(path):
        s = json.loads(line).get("step")
        if not isinstance(s, int) or s <= step:
            keep.append(line)
    with open(path + ".tmp", "w") as f:
        f.writelines(keep)
    os.replace(path + ".tmp", path)


def finished(out):
    return os.path.exists(os.path.join(out, "adapters.pt"))
