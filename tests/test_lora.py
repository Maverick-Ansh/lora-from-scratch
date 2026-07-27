"""Correctness tests for the from-scratch LoRA implementation.

Each test pins one *claim* made by the paper, so a failure here means the
reimplementation has drifted from arXiv:2106.09685 -- not merely that some
refactor broke.
"""

import math

import pytest
import torch
import torch.nn as nn

from lora import (
    LoRAConfig,
    LoRAEmbedding,
    LoRALinear,
    MergedLoRA,
    apply_lora,
    count_parameters,
    load_lora_state_dict,
    lora_state_dict,
    merge_lora,
    unmerge_lora,
)
from lora.variants import DoRALinear, apply_dora

torch.manual_seed(0)


def tiny_model(d=32, out=16):
    return nn.Sequential(
        nn.Linear(d, d),
        nn.ReLU(),
        nn.Sequential(nn.Linear(d, d)),  # nested, to exercise dotted-name lookup
        nn.Linear(d, out),
    )


# --------------------------------------------------------------------------
# Claim 1: "ΔW = BA is zero at the beginning of training."
# --------------------------------------------------------------------------

def test_zero_init_makes_adapter_a_no_op():
    base = nn.Linear(32, 16)
    x = torch.randn(8, 32)
    expected = base(x)

    lora = LoRALinear.from_linear(base, r=4, alpha=8)
    torch.testing.assert_close(lora(x), expected, rtol=0, atol=0)
    assert lora.delta_weight().abs().max() == 0


def test_zero_init_holds_for_every_rank_and_init_scheme():
    x = torch.randn(4, 32)
    for r in (1, 2, 8, 16):
        for init_a in ("kaiming", "gaussian"):
            base = nn.Linear(32, 32)
            lora = LoRALinear.from_linear(base, r=r, alpha=16, init_a=init_a)
            torch.testing.assert_close(lora(x), base(x), rtol=0, atol=0)


# --------------------------------------------------------------------------
# Claim 2: the forward path really computes W0 x + (alpha/r) B A x
# --------------------------------------------------------------------------

def test_forward_equals_explicit_delta_weight():
    lora = LoRALinear(32, 16, r=4, alpha=8)
    nn.init.normal_(lora.lora_B)          # break the zero init
    x = torch.randn(8, 32)

    manual = torch.nn.functional.linear(x, lora.weight + lora.delta_weight(), lora.bias)
    torch.testing.assert_close(lora(x), manual, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("r,alpha", [(1, 1.0), (2, 8.0), (8, 8.0), (16, 32.0)])
def test_scaling_is_alpha_over_r(r, alpha):
    lora = LoRALinear(16, 16, r=r, alpha=alpha)
    assert lora.scaling == pytest.approx(alpha / r)
    nn.init.normal_(lora.lora_B)
    expected = (lora.lora_B @ lora.lora_A) * (alpha / r)
    torch.testing.assert_close(lora.delta_weight(), expected)


def test_rslora_uses_sqrt_r():
    lora = LoRALinear(16, 16, r=16, alpha=8.0, use_rslora=True)
    assert lora.scaling == pytest.approx(8.0 / math.sqrt(16))


# --------------------------------------------------------------------------
# Claim 3: "no additional inference latency" -- merging is exact.
# --------------------------------------------------------------------------

def test_merge_preserves_output_and_unmerge_restores_weight():
    base = nn.Linear(32, 32)
    w0 = base.weight.detach().clone()
    lora = LoRALinear.from_linear(base, r=4, alpha=8)
    nn.init.normal_(lora.lora_B, std=0.1)
    x = torch.randn(8, 32)

    before = lora(x)
    lora.merge()
    torch.testing.assert_close(lora(x), before, rtol=1e-5, atol=1e-6)
    assert lora.merged

    lora.merge()  # idempotent
    torch.testing.assert_close(lora(x), before, rtol=1e-5, atol=1e-6)

    lora.unmerge()
    torch.testing.assert_close(lora.weight, w0, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(lora(x), before, rtol=1e-5, atol=1e-6)


def test_merged_context_manager_round_trips_a_whole_model():
    model = tiny_model()
    apply_lora(model, LoRAConfig(target_modules=["0", "3"], r=4, alpha=8))
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B, std=0.05)

    x = torch.randn(4, 32)
    unmerged = model(x)
    with MergedLoRA(model) as merged_model:
        torch.testing.assert_close(merged_model(x), unmerged, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(model(x), unmerged, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------
# Claim 4: at full rank LoRA is as expressive as full fine-tuning
# (paper Sec. 4.1: "as we increase r ... roughly converges to training the
#  original model").
# --------------------------------------------------------------------------

def test_full_rank_lora_can_represent_an_arbitrary_delta():
    d = 12
    target_delta = torch.randn(d, d)
    lora = LoRALinear(d, d, r=d, alpha=float(d), bias=False)

    # scaling is alpha/r = 1 here, so we need B @ A == target_delta exactly.
    u, s, vh = torch.linalg.svd(target_delta)
    with torch.no_grad():
        lora.lora_B.copy_(u * s)
        lora.lora_A.copy_(vh)
    torch.testing.assert_close(lora.delta_weight(), target_delta, rtol=1e-5, atol=1e-5)


def test_rank_r_delta_is_rank_r():
    lora = LoRALinear(32, 32, r=3, alpha=3.0)
    nn.init.normal_(lora.lora_B)
    assert torch.linalg.matrix_rank(lora.delta_weight()).item() == 3


# --------------------------------------------------------------------------
# Claim 5: only A and B train (Sec. 4.1) -- and the subtle consequence of B=0.
# --------------------------------------------------------------------------

def test_only_lora_params_require_grad():
    model = tiny_model()
    apply_lora(model, LoRAConfig(target_modules=["0", "2.0", "3"], r=4, alpha=8))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)

    counts = count_parameters(model)
    assert counts["trainable"] == counts["lora"]
    assert counts["trainable_pct"] < 30  # tiny model, but still a small fraction


def test_bitfit_mode_trains_biases_only_alongside_lora():
    model = tiny_model()
    apply_lora(model, LoRAConfig(target_modules=["0"], r=2, alpha=2, train_biases="all"))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.endswith("bias") for n in trainable)


def test_A_receives_no_gradient_on_the_very_first_step():
    """A subtle, real consequence of initialising B to zero.

    dL/dA = scaling * B^T @ dL/d(lora_out), and B is exactly zero at step 0,
    so A is frozen for one step no matter what the data says. B moves first;
    only once B is non-zero does A start to learn. This is why LoRA is
    insensitive to how A is initialised but *very* sensitive to B being zero.
    """
    lora = LoRALinear(16, 16, r=4, alpha=8)
    x = torch.randn(8, 16)
    lora(x).pow(2).sum().backward()

    assert lora.lora_A.grad.abs().max() == 0, "A should be gradient-free at step 0"
    assert lora.lora_B.grad.abs().max() > 0, "B must move first"

    # After perturbing B, A does receive gradient.
    lora.zero_grad()
    with torch.no_grad():
        lora.lora_B.normal_(std=0.1)
    lora(x).pow(2).sum().backward()
    assert lora.lora_A.grad.abs().max() > 0


def test_frozen_base_weight_never_updates():
    model = tiny_model()
    apply_lora(model, LoRAConfig(target_modules=["0"], r=4, alpha=8))
    layer = model[0]
    w0 = layer.weight.detach().clone()

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(8, 32)).pow(2).sum().backward()
        opt.step()

    torch.testing.assert_close(layer.weight, w0, rtol=0, atol=0)
    assert layer.lora_B.abs().max() > 0  # but the adapter did move


# --------------------------------------------------------------------------
# Claim 6: the deployable artifact is tiny and swappable (Sec. 1, Sec. 4.2).
# --------------------------------------------------------------------------

def test_state_dict_contains_only_adapter_and_round_trips():
    model = tiny_model()
    apply_lora(model, LoRAConfig(target_modules=["0", "3"], r=4, alpha=8))
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B, std=0.05)

    sd = lora_state_dict(model)
    assert sd and all("lora_" in k for k in sd)
    total = sum(t.numel() for t in sd.values())
    assert total < sum(p.numel() for p in model.parameters()) / 4

    x = torch.randn(4, 32)
    expected = model(x)

    fresh = tiny_model()
    apply_lora(fresh, LoRAConfig(target_modules=["0", "3"], r=4, alpha=8))
    # different base weights -> different output, until we also copy the base
    fresh.load_state_dict({k: v for k, v in model.state_dict().items()}, strict=True)
    load_lora_state_dict(fresh, sd)
    torch.testing.assert_close(fresh(x), expected, rtol=1e-6, atol=1e-6)


def test_two_adapters_can_be_hot_swapped_on_one_frozen_base():
    model = tiny_model()
    apply_lora(model, LoRAConfig(target_modules=["0", "3"], r=4, alpha=8))

    adapters = []
    for std in (0.05, 0.2):
        for _, m in model.named_modules():
            if isinstance(m, LoRALinear):
                nn.init.normal_(m.lora_B, std=std)
        adapters.append(lora_state_dict(model))

    x = torch.randn(4, 32)
    outs = []
    for sd in adapters:
        load_lora_state_dict(model, sd)
        outs.append(model(x).clone())
    assert not torch.allclose(outs[0], outs[1])

    load_lora_state_dict(model, adapters[0])
    torch.testing.assert_close(model(x), outs[0], rtol=0, atol=0)


# --------------------------------------------------------------------------
# Injection mechanics
# --------------------------------------------------------------------------

def test_injection_reuses_the_pretrained_tensor_rather_than_copying():
    model = tiny_model()
    original = model[0].weight.data_ptr()
    apply_lora(model, LoRAConfig(target_modules=["0"], r=4, alpha=8))
    assert model[0].weight.data_ptr() == original


def test_unmatched_target_raises_with_a_useful_message():
    model = tiny_model()
    with pytest.raises(ValueError, match="matched nothing"):
        apply_lora(model, LoRAConfig(target_modules=["q_proj"]))


def test_glob_targets_and_exclusions():
    model = nn.ModuleDict({f"layer{i}": nn.Linear(8, 8) for i in range(4)})
    apply_lora(model, LoRAConfig(target_modules=["layer*"], exclude=["layer3"], r=2, alpha=2))
    assert isinstance(model["layer0"], LoRALinear)
    assert not isinstance(model["layer3"], LoRALinear)


def test_double_injection_does_not_nest():
    model = tiny_model()
    cfg = LoRAConfig(target_modules=["0"], r=4, alpha=8)
    apply_lora(model, cfg)
    apply_lora(model, cfg)
    assert isinstance(model[0], LoRALinear)
    assert not isinstance(model[0].weight, LoRALinear)


def test_fan_in_fan_out_matches_a_transposed_layer():
    """GPT-2's Conv1D stores W as (in, out); getting this wrong trains garbage."""
    in_f, out_f = 12, 20
    w_t = torch.randn(in_f, out_f)          # Conv1D orientation
    lora = LoRALinear(in_f, out_f, r=4, alpha=8, fan_in_fan_out=True, bias=False)
    with torch.no_grad():
        lora.weight.copy_(w_t)
        lora.lora_B.normal_(std=0.1)

    x = torch.randn(5, in_f)
    expected = x @ (w_t + lora.delta_weight())
    torch.testing.assert_close(lora(x), expected, rtol=1e-5, atol=1e-5)
    assert lora.delta_weight().shape == w_t.shape


# --------------------------------------------------------------------------
# Embedding adapter
# --------------------------------------------------------------------------

def test_lora_embedding_is_a_no_op_at_init_and_matches_delta():
    emb = nn.Embedding(50, 16)
    ids = torch.randint(0, 50, (4, 7))
    l = LoRAEmbedding.from_embedding(emb, r=4, alpha=8)
    torch.testing.assert_close(l(ids), emb(ids), rtol=0, atol=0)

    with torch.no_grad():
        l.lora_A.normal_()
    manual = torch.nn.functional.embedding(ids, l.weight + l.delta_weight())
    torch.testing.assert_close(l(ids), manual, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# DoRA
# --------------------------------------------------------------------------

def test_dora_is_a_no_op_at_init():
    base = nn.Linear(32, 16)
    x = torch.randn(8, 32)
    d = DoRALinear.from_linear(base, r=4, alpha=8)
    torch.testing.assert_close(d(x), base(x), rtol=1e-5, atol=1e-6)


def test_dora_merge_matches_forward():
    base = nn.Linear(32, 16)
    d = DoRALinear.from_linear(base, r=4, alpha=8)
    with torch.no_grad():
        d.lora_B.normal_(std=0.1)
        d.lora_magnitude.mul_(1.1)
    x = torch.randn(8, 32)
    before = d(x)
    d.merge()
    torch.testing.assert_close(d(x), before, rtol=1e-4, atol=1e-5)


def test_dora_trains_magnitude():
    model = tiny_model()
    apply_dora(model, LoRAConfig(target_modules=["0"], r=4, alpha=8))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("lora_magnitude" in n for n in trainable)
