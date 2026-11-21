import json
import os
import signal

import torch

import gist_resume


def _run(out, steps, resume_at=None):
    """Toy loop shaped like the trainers: global RNG + a generator feed each step."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(1)
    start = gist_resume.load(out, model, opt, [gen])
    log = open(os.path.join(out, "log.jsonl"), "a")
    for step in range(start + 1, steps + 1):
        x = torch.randn(8, 4) + torch.rand(8, 4, generator=gen)
        model(x).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
        log.write(json.dumps({"step": step}) + "\n")
        log.flush()
        if step == resume_at:
            gist_resume.save(out, step, model, list(model.parameters()), opt, [gen])
            log.write(json.dumps({"step": step + 1}) + "\n")  # a step the kill interrupted
            return None
    return model.weight.detach().clone()


def test_resume_matches_an_uninterrupted_run(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    full = _run(str(a), 4)
    assert _run(str(b), 4, resume_at=2) is None
    resumed = _run(str(b), 4)
    assert torch.equal(full, resumed)
    steps = [json.loads(l)["step"] for l in open(b / "log.jsonl")]
    assert steps == [1, 2, 3, 4]


def test_sigterm_sets_the_flag():
    p = gist_resume.Preemption()
    os.kill(os.getpid(), signal.SIGTERM)
    assert p.requested
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
