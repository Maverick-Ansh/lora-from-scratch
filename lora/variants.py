"""What LoRA turned into: the variants that are actually deployed in 2026.

The 2021 paper defines one thing -- ``h = W0 x + (alpha/r) B A x``.  Everything
here is a small, well-motivated perturbation of that expression, and each one
exists because a specific term in it was found wanting:

===============  =======================================  =========================
variant          changes                                  because
===============  =======================================  =========================
rsLoRA           ``alpha/r``  ->  ``alpha/sqrt(r)``        alpha/r over-damps large r
DoRA             separates magnitude from direction        LoRA under-uses magnitude
QLoRA            ``W0`` held in NF4, LoRA in bf16          fit 65B on one 48GB card
===============  =======================================  =========================

rsLoRA is a one-line flag on :class:`~lora.layers.LoRALinear`
(``use_rslora=True``); DoRA is here; QLoRA is a quantisation choice orthogonal
to this file (see ``docs/modern-lora.md``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .inject import LoRAConfig, _set_submodule, mark_only_lora_as_trainable
from .layers import LoRALinear

__all__ = ["DoRALinear", "apply_dora"]


class DoRALinear(LoRALinear):
    """Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024, arXiv:2402.09353).

    Decompose the adapted weight into magnitude and direction::

        W' = m * (W0 + dW) / ||W0 + dW||_row

    where the norm is taken over the input dimension, so ``m`` has one entry
    per output unit and is trained directly.

    The motivation is an empirical observation: when you decompose a *full*
    fine-tune the same way, magnitude and direction change fairly independently,
    whereas plain LoRA's magnitude and direction changes are tightly correlated.
    DoRA gives magnitude its own parameters so the adapter can move in the
    directions full fine-tuning does.

    Cost: an extra ``out_features`` parameters per layer -- negligible -- plus a
    norm over the merged weight on *every* forward, which is not negligible.
    That norm is why DoRA trains meaningfully slower than LoRA despite the
    identical parameter count, and it is the main reason plain LoRA is still
    the default.
    """

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.lora_magnitude = nn.Parameter(self._weight_norm(self.weight_2d).detach().clone())

    @classmethod
    def from_linear(cls, layer: nn.Module, **kw) -> "DoRALinear":
        obj = super().from_linear(layer, **kw)
        # weight was swapped in after __init__, so re-derive the magnitude from
        # the *pre-trained* weight -- initialising it from the empty tensor
        # would start training from noise.
        with torch.no_grad():
            obj.lora_magnitude.data = obj._weight_norm(obj.weight_2d).detach().clone()
        return obj

    @property
    def weight_2d(self) -> torch.Tensor:
        """The frozen weight in ``(out_features, in_features)`` orientation."""
        return self.weight.T if self.fan_in_fan_out else self.weight

    @staticmethod
    def _weight_norm(w: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(w, dim=1)  # (out,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            w = self.weight.T if self.fan_in_fan_out else self.weight
            return F.linear(x, w, self.bias)

        w0 = self.weight_2d
        dw = (self.lora_B @ self.lora_A) * self.scaling
        norm = self._weight_norm(w0 + dw).detach()  # detached: see DoRA Sec. 4.3
        scale = (self.lora_magnitude / norm).to(x.dtype)

        base = F.linear(x, w0)
        lora = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B) * self.scaling
        out = (base + lora) * scale
        return out + self.bias if self.bias is not None else out

    @torch.no_grad()
    def merge(self) -> "DoRALinear":
        if not self.merged:
            w0 = self.weight_2d
            dw = (self.lora_B @ self.lora_A) * self.scaling
            merged = (self.lora_magnitude / self._weight_norm(w0 + dw)).unsqueeze(1) * (w0 + dw)
            new = merged.T if self.fan_in_fan_out else merged
            self.weight.data.copy_(new.to(self.weight.dtype))
            self.merged = True
        return self

    @torch.no_grad()
    def unmerge(self) -> "DoRALinear":
        raise NotImplementedError(
            "DoRA's merge is not a simple addition, so it cannot be undone from "
            "the merged weight alone. Keep a copy of W0 if you need to swap "
            "adapters (which is what real serving stacks do anyway)."
        )


def apply_dora(model: nn.Module, config: LoRAConfig, verbose: bool = False) -> nn.Module:
    """Same selection rules as :func:`~lora.inject.apply_lora`, DoRA layers."""
    candidates = [
        (n, m) for n, m in model.named_modules()
        if config.matches(n) and isinstance(m, nn.Linear)
    ]
    if not candidates:
        raise ValueError(f"target_modules={list(config.target_modules)} matched no nn.Linear")
    for name, module in candidates:
        _set_submodule(
            model, name,
            DoRALinear.from_linear(
                module, r=config.r, alpha=config.alpha,
                dropout=config.dropout, init_a=config.init_a,
                use_rslora=config.use_rslora,
            ),
        )
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_A" in n) or ("lora_B" in n) or ("lora_magnitude" in n)
    if verbose:
        print(f"[dora] adapted {len(candidates)} modules")
    return model
