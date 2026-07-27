"""The adaptation methods being compared.

Every entry takes a freshly-loaded pre-trained model and returns it with the
right subset of parameters trainable.  These are the rows of the paper's
Table 2, transplanted onto a model small enough to run the whole grid many
times over.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from lora import LoRAConfig, apply_lora
from lora.variants import apply_dora

# Attention projections, in the paper's notation.
ATTN = {"Wq": "q_proj", "Wk": "k_proj", "Wv": "v_proj", "Wo": "o_proj"}
MLP = ["up_proj", "down_proj"]


def freeze_all(model: nn.Module) -> nn.Module:
    for p in model.parameters():
        p.requires_grad = False
    return model


def method_zero_shot(model: nn.Module) -> nn.Module:
    """No adaptation at all -- the pre-trained model's loss on the new domain."""
    return freeze_all(model)


def method_full(model: nn.Module) -> nn.Module:
    """Full fine-tuning: the thing LoRA is trying to match with 0.1% of the params."""
    for p in model.parameters():
        p.requires_grad = True
    return model


def method_bitfit(model: nn.Module) -> nn.Module:
    """Bias-only tuning (Ben Zaken et al.) -- Table 2's strongest tiny baseline."""
    freeze_all(model)
    for name, p in model.named_parameters():
        if name.endswith("bias"):
            p.requires_grad = True
    return model


def method_last_block(model: nn.Module) -> nn.Module:
    """Train only the final transformer block.

    A parameter-matched-ish control for "does LoRA win because it is low rank,
    or merely because it trains few parameters?"  This trains *more* parameters
    than LoRA and is spread over one layer instead of all of them.
    """
    freeze_all(model)
    last = model.blocks[-1]
    for p in last.parameters():
        p.requires_grad = True
    return model


def method_layernorm(model: nn.Module) -> nn.Module:
    """LayerNorm-only tuning: the cheapest non-trivial baseline."""
    freeze_all(model)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = True
    return model


def make_lora(targets: list[str], r: int, alpha: float | None = None,
              dropout: float = 0.0, use_rslora: bool = False,
              init_a: str = "kaiming") -> Callable[[nn.Module], nn.Module]:
    """LoRA on the given leaf names.  ``alpha=None`` follows the paper's alpha = r."""

    def fn(model: nn.Module) -> nn.Module:
        cfg = LoRAConfig(
            target_modules=tuple(targets),
            r=r,
            alpha=float(r if alpha is None else alpha),
            dropout=dropout,
            use_rslora=use_rslora,
            init_a=init_a,
        )
        return apply_lora(model, cfg)

    return fn


def make_dora(targets: list[str], r: int, alpha: float | None = None):
    def fn(model: nn.Module) -> nn.Module:
        cfg = LoRAConfig(target_modules=tuple(targets), r=r,
                         alpha=float(r if alpha is None else alpha))
        return apply_dora(model, cfg)
    return fn


#: The paper's headline configuration: adapt Wq and Wv only.
LORA_QV = ["q_proj", "v_proj"]
LORA_ALL_ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_ALL_LINEAR = LORA_ALL_ATTN + MLP

BASELINES = {
    "zero_shot": method_zero_shot,
    "full_ft": method_full,
    "bitfit": method_bitfit,
    "layernorm": method_layernorm,
    "last_block": method_last_block,
}
