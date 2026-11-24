"""Stage 1: gist-only LoRA leaves raw tokens untouched; per-layer MSE trains only adapters."""

import random

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import gist_flex_attention
from gist_lora import apply_lora
from gist_qwen2 import GistQwen2ForCausalLM
from gist_utils import Global_data
from src.data.gist_stream import insert_gists_with_map

G = [1000, 1001]


@pytest.fixture
def cfg():
    keys = (
        "gist_token_id",
        "chunk_size",
        "sink_size",
        "num_previous_chunks",
        "pad_token_id",
        "train_attn",
        "raw_window",
        "stage1",
        "lora_row_mask",
    )
    saved = {k: getattr(Global_data, k) for k in keys}
    Global_data.gist_token_id, Global_data.chunk_size = G, [4, 4]
    Global_data.sink_size, Global_data.num_previous_chunks = 3, 1
    Global_data.pad_token_id, Global_data.raw_window = 0, 0
    c = Qwen2Config(
        vocab_size=1100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    yield c
    for k, v in saved.items():
        setattr(Global_data, k, v)


def _ids(n_raw=150, seed=0):
    rng = random.Random(seed)
    return torch.tensor(
        [insert_gists_with_map([rng.randrange(1, 900) for _ in range(n_raw)], G, 4, 4, sink=3)[0]]
    )


def _randomize_lora(model):
    torch.manual_seed(3)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.5)


def test_gist_lora_leaves_raw_rows_unchanged(cfg):
    """Raw queries blind to summaries: option D changes nothing, option B does."""
    cfg._attn_implementation = "sdpa"
    Global_data.train_attn = "dense"
    torch.manual_seed(0)
    base = GistQwen2ForCausalLM(cfg).float().eval()
    ids = _ids()
    T = ids.shape[1]
    raw = ~torch.isin(ids[0], torch.tensor(G))
    allowed = torch.ones(T, T, dtype=torch.bool).tril() & (
        raw[None, :] | torch.eye(T, dtype=torch.bool)
    )
    kw = dict(
        input_ids=ids, attention_mask=torch.ones_like(ids), attention_mask_gist=allowed[None, None]
    )
    with torch.no_grad():
        ref = base(**kw).logits[0, raw]
    for mode, same in (("gist", True), ("all", False)):
        m = GistQwen2ForCausalLM(cfg).float().eval()
        m.load_state_dict(base.state_dict())
        apply_lora(m, mode, r=4)
        _randomize_lora(m)
        with torch.no_grad():
            out = m(**kw).logits[0, raw]
        assert torch.allclose(out, ref, atol=1e-5) == same, mode


@pytest.mark.parametrize("mode", ["gist", "all"])
def test_stage1_step_trains_only_adapters(cfg, mode):
    cfg._attn_implementation = gist_flex_attention.GIST_FLEX
    Global_data.train_attn = "flex"
    Global_data.stage1 = {
        "pairs": [("g1", "raw"), ("hier", "g1")],
        "down": "hier",
        "lambda_ntp": 0.0,
    }
    torch.manual_seed(0)
    model = GistQwen2ForCausalLM(cfg).float().train()
    groups = apply_lora(model, mode, r=4)
    ids = _ids()
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
    assert set(Global_data.stage1["last"]) == {"g1|raw", "hier|g1"}
    assert all(v > 0 for v in Global_data.stage1["last"].values())
    out.loss.backward()
    trainable = {id(p) for g in groups.values() for p in g}
    for n, p in model.named_parameters():
        if id(p) in trainable:
            continue
        assert p.grad is None, f"frozen {n} got a gradient"
    # B starts at 0, so only B gets a gradient on the first step
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in groups["lora_kv"][1::2])
    emb_grad = groups["embed"][0].grad
    assert emb_grad[G].abs().sum() > 0 and emb_grad[: G[0]].abs().sum() == 0
    if mode == "gist":  # last layer keeps K/V adapters only
        last = model.model.layers[-1].self_attn
        assert not hasattr(last.q_proj, "lora_A") and hasattr(last.k_proj, "lora_A")


def test_no_summaries_no_loss(cfg):
    """Without summary tokens every mask is causal, so each relMSE is 0."""
    cfg._attn_implementation = gist_flex_attention.GIST_FLEX
    Global_data.train_attn = "flex"
    Global_data.stage1 = {
        "pairs": [("g1", "raw"), ("hier", "g1")],
        "down": "hier",
        "lambda_ntp": 0.0,
    }
    torch.manual_seed(0)
    model = GistQwen2ForCausalLM(cfg).float().train()
    apply_lora(model, "gist", r=4)
    ids = torch.randint(1, 900, (1, 64))
    model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
    assert all(v < 1e-10 for v in Global_data.stage1["last"].values())


def _flex_model(cfg, layers=3):
    from gist_lora import apply_summary_bias

    cfg._attn_implementation = gist_flex_attention.GIST_FLEX
    cfg.num_hidden_layers = layers
    Global_data.train_attn = "flex"
    torch.manual_seed(0)
    model = GistQwen2ForCausalLM(cfg).float().train()
    return model, apply_summary_bias


def test_beta_and_routing_get_gradients(cfg):
    model, apply_summary_bias = _flex_model(cfg)
    groups = apply_lora(model, "gist", r=4)
    bias = apply_summary_bias(model)
    Global_data.stage1 = {
        "pairs": [("g1", "raw"), ("hier", "g1")],
        "down": "hier",
        "lambda_ntp": 0.0,
        "route_rows": 16,
        "weights": {"route": 1.0},
    }
    ids = _ids(300)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
    assert "route" in Global_data.stage1["last"] and Global_data.stage1["last"]["route"] > 0
    out.loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in bias)
    assert all(p.grad is not None for p in groups["lora_kv"])


def test_level_scalar_bias_gets_no_routing_gradient(cfg):
    """With u = 0 and only the routing loss, the last layer's per-level scalar b gets zero
    gradient (a constant inside each level's softmax); its u does not."""
    model, apply_summary_bias = _flex_model(cfg, layers=2)
    apply_lora(model, "gist", r=4)
    apply_summary_bias(model)
    Global_data.stage1 = {
        "pairs": [("hier", "g1")],
        "down": "hier",
        "lambda_ntp": 0.0,
        "route_rows": 16,
        "weights": {"route": 1.0, "hier|g1": 0.0},
    }
    ids = _ids(300)
    model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone()).loss.backward()
    sb = model.model.layers[-1].self_attn.summary_bias
    assert sb.b.grad.abs().max() < 1e-6 and sb.u.grad.abs().sum() > 0


def test_freezing_level0_adapters(cfg):
    model, _ = _flex_model(cfg)
    groups = apply_lora(model, "gist", r=4, per_level=True)
    for p in groups["level0"]:
        p.requires_grad_(False)
    _randomize_lora(model)
    Global_data.stage1 = {"pairs": [("hier", "g1")], "down": "hier", "lambda_ntp": 0.0}
    ids = _ids(200)
    model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone()).loss.backward()
    assert all(p.grad is None for p in groups["level0"])
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in groups["level1"])


def test_fold_beta_adds_bias_to_logits():
    from gist_flex_attention import fold_beta

    torch.manual_seed(0)
    q, k = torch.randn(1, 4, 10, 16), torch.randn(1, 2, 10, 16)
    beta = torch.randn(1, 2, 10)
    s = 16**-0.5
    q2, k2 = fold_beta(q, k, beta, s)
    assert q2.shape[-1] == 32 and k2.shape[-1] == 32
    kr = k.repeat_interleave(2, 1)
    ref = (q @ kr.transpose(-1, -2)) * s + beta.repeat_interleave(2, 1)[:, :, None, :]
    got = (q2 @ k2.repeat_interleave(2, 1).transpose(-1, -2)) * s
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_teacher_chain_at_four_levels(cfg):
    """L = 4: the chain g1:raw, g2:g1, g3:g2, hier:g3 gives one positive relMSE per pair,
    and beta / routing reach every level's summaries."""
    from gist_lora import apply_summary_bias, summary_spans
    from gist_stage1_trainer import teacher_chain

    G4 = [1000, 1001, 1002, 1003]
    Global_data.gist_token_id, Global_data.chunk_size = G4, [4] * 4
    cfg._attn_implementation = gist_flex_attention.GIST_FLEX
    Global_data.train_attn = "flex"
    assert teacher_chain(2) == ["g1:raw", "hier:g1"]
    pairs = [tuple(p.split(":")) for p in teacher_chain(4)]
    Global_data.stage1 = {
        "pairs": pairs,
        "down": "hier",
        "lambda_ntp": 0.0,
        "route_rows": 16,
        "weights": {"route": 1.0},
    }
    torch.manual_seed(0)
    model = GistQwen2ForCausalLM(cfg).float().train()
    groups = apply_lora(model, "gist", r=4)
    bias = apply_summary_bias(model, spans=summary_spans(4))
    rng = random.Random(0)
    ids = torch.tensor(
        [insert_gists_with_map([rng.randrange(1, 900) for _ in range(600)], G4, 4, 4, sink=3)[0]]
    )
    assert (ids == G4[3]).sum() >= 2
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
    last = Global_data.stage1["last"]
    assert {"g1|raw", "g2|g1", "g3|g2", "hier|g3"} <= set(last)
    assert all(last[k] > 0 for k in ("g1|raw", "g2|g1", "g3|g2", "hier|g3"))
    out.loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in groups["lora_kv"][1::2])
    for sb in bias:
        if sb.dim() == 2:  # b [levels, num_kv]
            assert (sb.grad.abs().sum(-1) > 0).all(), sb.grad


@pytest.mark.parametrize("k,zero", [((10**6,), True), ((1,), False)])
def test_select_mask_full_expansion_is_the_raw_teacher(cfg, k, zero):
    """#28: with every node expanded and coarse terms on, the 'sel' student of raw rows is
    plain causal attention over raw tokens, so sel|raw vanishes; a small k selects."""
    import gist_beam

    cfg._attn_implementation = gist_flex_attention.GIST_FLEX
    Global_data.train_attn = "flex"
    Global_data.stage1 = {
        "pairs": [("sel", "raw"), ("hier", "raw")],
        "down": "sel",
        "lambda_ntp": 0.0,
        "select": {"cfg": gist_beam.BeamConfig(k=k, coarse=True), "rows_from": 0},
    }
    torch.manual_seed(0)
    model = GistQwen2ForCausalLM(cfg).float().eval()
    apply_lora(model, "gist", r=4)
    _randomize_lora(model)
    ids = _ids(n_raw=300)
    with torch.no_grad():
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    last = {n: float(torch.stack(v).max()) for n, v in Global_data.stage1["losses"].items()}
    assert last["hier|raw"] > 1e-4
    if zero:
        assert last["sel|raw"] < 1e-10, last
    else:
        assert last["sel|raw"] > 1e-6, last


@pytest.mark.parametrize("k,zero", [((10**6,), True), ((1,), False)])
def test_base_teacher_is_the_raw_stream(cfg, k, zero):
    """#41: "base" is the raw stream recorded layer by layer. With every node expanded and
    coarse terms on, passing sel down reproduces it exactly at every layer; with k = 1 the
    sel stream drifts from it."""
    import gist_beam
    from gist_stage1_trainer import record_base

    cfg._attn_implementation = gist_flex_attention.GIST_FLEX
    Global_data.train_attn = "flex"
    Global_data.stage1 = {
        "pairs": [("sel", "base")],
        "down": "sel",
        "lambda_ntp": 0.0,
        "select": {"cfg": gist_beam.BeamConfig(k=k, coarse=True), "rows_from": 0},
    }
    torch.manual_seed(0)
    model = GistQwen2ForCausalLM(cfg).float().eval()
    apply_lora(model, "gist", r=4)
    _randomize_lora(model)
    ids = _ids(n_raw=300)
    record_base(model, ids, {})
    assert len(Global_data.stage1["base"]) >= 2
    with torch.no_grad():
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    per_layer = [float(x) for x in Global_data.stage1["losses"]["sel|base"]]
    assert len(per_layer) == len(Global_data.stage1["base"])
    if zero:
        assert max(per_layer) < 1e-10, per_layer
    else:
        assert per_layer[-1] > 1e-6, per_layer


def test_hybrid_down_keeps_raw_rows_raw_and_writes_summaries_with_the_gist_mask(cfg):
    """#41: down=hybrid passes raw rows' raw output (the base stream, at every layer) and
    summary rows' gist-mask output; the latter equals down=hier's at the first layer."""
    cfg._attn_implementation = gist_flex_attention.GIST_FLEX
    Global_data.train_attn = "flex"
    torch.manual_seed(0)
    model = GistQwen2ForCausalLM(cfg).float().eval()
    apply_lora(model, "gist", r=4)
    _randomize_lora(model)
    ids = _ids(n_raw=300)
    summ = torch.isin(ids, torch.tensor(Global_data.gist_token_id))[0]
    rec = {}
    for down in ("raw", "hybrid", "hier"):
        Global_data.stage1 = {"pairs": [], "down": down, "lambda_ntp": 0.0, "record": {}}
        with torch.no_grad():
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        rec[down] = Global_data.stage1["record"]
    assert summ.any() and len(rec["hybrid"]) >= 2
    for l in rec["hybrid"]:
        torch.testing.assert_close(rec["hybrid"][l][0, ~summ], rec["raw"][l][0, ~summ],
                                   rtol=0, atol=0)
    first = min(rec["hybrid"])
    torch.testing.assert_close(rec["hybrid"][first][0, summ], rec["hier"][first][0, summ])
    assert (rec["hybrid"][first][0, summ] - rec["raw"][first][0, summ]).abs().max() > 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex backward needs CUDA")
@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("agent", [False, True])
def test_select_hybrid_equals_the_dense_mask(cfg, coarse, k, agent):
    """#29: flex over S + gather over X, merged by log-sum-exp (gist_select_train.py), gives
    the dense 'sel' mask's student: same loss, same gradients of every trained tensor."""
    import gist_beam
    from gist_lora import apply_summary_bias, summary_spans
    from src.data.gist_stream import gist_segments

    cfg._attn_implementation = gist_flex_attention.GIST_FLEX
    Global_data.train_attn = "flex"
    saved_w = (Global_data.raw_window, Global_data.window_turns)
    kw = {}
    if agent:  # a pinned prefix, turns and a raw window: S's always / window rules
        Global_data.raw_window, Global_data.window_turns = 24, 2
        rng = random.Random(1)
        raw, pin, turn = [], [], []
        for t, n in enumerate([12] + [rng.randint(20, 60) for _ in range(6)]):
            raw += [rng.randrange(1, 900) for _ in range(n)]
            pin += [t == 0] * n
            turn += [t] * n
        g, p, tr = gist_segments(raw, pin, turn, G, 4, 4, sink=3)
        ids = torch.tensor([g])
        kw = {"gist_pinned": torch.tensor([p]), "gist_turn": torch.tensor([tr])}
    else:
        ids = _ids(n_raw=300)
    grads, losses = {}, {}
    import os
    try:
        for impl in ("dense", "hybrid_ref", "hybrid"):
            os.environ["GIST_SEL_REF"] = "1" if impl == "hybrid_ref" else "0"
            Global_data.stage1 = {
                "pairs": [("sel", "raw")], "down": "sel", "lambda_ntp": 0.0,
                "select": {"cfg": gist_beam.BeamConfig(k=(k,), coarse=coarse), "rows_from": 0,
                           "impl": impl.split("_")[0]}}
            torch.manual_seed(0)
            model = GistQwen2ForCausalLM(cfg).float().train()
            apply_lora(model, "gist", r=4)
            apply_summary_bias(model, spans=summary_spans(2))
            _randomize_lora(model)
            for n, p in model.named_parameters():
                if "summary_bias" in n:
                    p.data.normal_(0, 0.5)
            model.cuda()
            x = ids.cuda()
            model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                  **{n: v.cuda() for n, v in kw.items()})
            loss = torch.stack(Global_data.stage1["losses"]["sel|raw"]).sum()
            loss.backward()
            losses[impl] = float(loss)
            grads[impl] = {n: p.grad.clone() for n, p in model.named_parameters()
                           if p.grad is not None}
    finally:
        Global_data.raw_window, Global_data.window_turns = saved_w
        os.environ.pop("GIST_SEL_REF", None)
    assert losses["dense"] > 1e-6
    for impl in ("hybrid_ref", "hybrid"):  # the torch gather, then the Triton kernel
        assert abs(losses["dense"] - losses[impl]) <= 1e-5 * losses["dense"], (impl, losses)
        assert grads["dense"].keys() == grads[impl].keys()
        rel = {n: float((grads[impl][n] - grads["dense"][n]).abs().max()
                        / grads["dense"][n].abs().max().clamp_min(1e-30)) for n in grads["dense"]}
        bad = {n: r for n, r in rel.items() if r > 1e-4}  # fp32 rounding: <= 5e-6 seen
        assert not bad, (impl, max(rel.values()), bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex backward needs CUDA")
def test_checkpoint_rerun_keeps_one_loss_per_layer(cfg):
    """Gradient checkpointing reruns each layer during backward; the rerun must not append
    its losses again (their graphs stayed alive until the next forward: the memory of a
    32K stage-1 step)."""
    from gist_lora import apply_summary_bias, summary_spans

    cfg._attn_implementation = gist_flex_attention.GIST_FLEX
    Global_data.train_attn = "flex"
    Global_data.stage1 = {"pairs": [("hier", "raw"), ("g1", "raw")], "down": "hier",
                          "lambda_ntp": 0.0, "route_rows": 8}
    torch.manual_seed(0)
    model = GistQwen2ForCausalLM(cfg).float().train()
    apply_lora(model, "gist", r=4)
    apply_summary_bias(model, spans=summary_spans(2))
    model.cuda().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ids = _ids(n_raw=300).cuda()
    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    losses = Global_data.stage1["losses"]
    sum(torch.stack(v).sum() for v in losses.values()).backward()
    L = cfg.num_hidden_layers
    assert len(losses["hier|raw"]) == L and len(losses["g1|raw"]) == L
    assert len(losses["route"]) == L - 1  # layer 0 has no routing loss
