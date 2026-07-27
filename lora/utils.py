"""Parameter accounting, checkpointing, and the numbers you quote in a paper."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .layers import _LoRABase, lora_modules

__all__ = [
    "count_parameters",
    "lora_state_dict",
    "load_lora_state_dict",
    "adapter_size_bytes",
    "summarize",
]


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Trainable vs total, plus the ratio the paper reports in Table 2."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(
        m.lora_A.numel() + m.lora_B.numel() for _, m in lora_modules(model)
    )
    return {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "frozen": total - trainable,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }


def lora_state_dict(model: nn.Module, include_biases: bool = False) -> dict[str, torch.Tensor]:
    """The *only* thing you need to ship: a few MB of A and B matrices.

    This is the deployment story of the paper.  One frozen base model in
    memory, one of these per task, swapped at request time.
    """
    out = {}
    for name, param in model.state_dict().items():
        if "lora_A" in name or "lora_B" in name:
            out[name] = param.detach().cpu().clone()
        elif include_biases and name.endswith("bias"):
            out[name] = param.detach().cpu().clone()
    return out


def load_lora_state_dict(model: nn.Module, sd: dict[str, torch.Tensor], strict: bool = True):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if strict:
        # Everything in `sd` must have landed somewhere.
        stray = [k for k in unexpected]
        if stray:
            raise KeyError(f"adapter keys not present in model: {stray[:5]}")
        model_lora_keys = {k for k in model.state_dict() if "lora_" in k}
        absent = model_lora_keys - set(sd)
        if absent:
            raise KeyError(f"model has LoRA params not in checkpoint: {sorted(absent)[:5]}")
    return model


def adapter_size_bytes(model: nn.Module, dtype_bytes: int = 4) -> int:
    return sum(t.numel() for t in lora_state_dict(model).values()) * dtype_bytes


def summarize(model: nn.Module) -> str:
    c = count_parameters(model)
    n_adapted = sum(1 for _ in lora_modules(model))
    mb = adapter_size_bytes(model) / 1024**2
    return (
        f"total={c['total']:,}  trainable={c['trainable']:,} "
        f"({c['trainable_pct']:.4f}%)  adapted_modules={n_adapted}  "
        f"adapter={mb:.2f} MB (fp32)"
    )
