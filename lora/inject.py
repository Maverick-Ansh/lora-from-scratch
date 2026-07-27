"""Inject LoRA into an arbitrary ``nn.Module`` tree.

This is the part every practical LoRA stack needs and the paper does not
describe: given a pre-trained model you did not write, find the right
submodules, swap them, freeze everything else.

The selection API mirrors what people actually type today
(``target_modules=["q_proj", "v_proj"]`` in PEFT) so that the mental model
transfers directly.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .layers import LoRAEmbedding, LoRALinear, _LoRABase, lora_modules

__all__ = ["LoRAConfig", "apply_lora", "merge_lora", "unmerge_lora", "MergedLoRA"]

# HF's GPT-2-style Conv1D keeps weights as (in, out).  Detect by class name so
# we do not have to import transformers just to inject into a plain nn.Module.
_TRANSPOSED_LAYER_NAMES = {"Conv1D"}


@dataclass
class LoRAConfig:
    """Everything that defines an adapter.

    Attributes
    ----------
    target_modules
        Module names to adapt.  Each entry matches if it is (a) the exact
        dotted name, (b) the last dotted component, or (c) an fnmatch glob
        against the full dotted name.  ``["query", "value"]`` therefore hits
        every attention Q/V projection in a BERT-family model, and
        ``["*.layer.11.*"]`` hits only the last block.
    r, alpha
        Rank and scaling numerator; the update is scaled by ``alpha / r``.
        The paper uses ``r=8, alpha=8`` for RoBERTa-base (Appendix D.1).
    train_biases
        ``"none"`` (paper), ``"all"``, or ``"lora_only"``.  Included because
        BitFit -- bias-only tuning -- is the headline baseline in Table 2, and
        this makes it reproducible with the same code path.
    """

    target_modules: Sequence[str] = ("query", "value")
    r: int = 8
    alpha: float = 8.0
    dropout: float = 0.0
    init_a: str = "kaiming"
    use_rslora: bool = False
    train_biases: str = "none"
    adapt_embeddings: bool = False
    exclude: Sequence[str] = field(default_factory=tuple)

    def matches(self, dotted_name: str) -> bool:
        leaf = dotted_name.rsplit(".", 1)[-1]
        if any(_one_matches(p, dotted_name, leaf) for p in self.exclude):
            return False
        return any(_one_matches(p, dotted_name, leaf) for p in self.target_modules)


def _one_matches(pattern: str, dotted_name: str, leaf: str) -> bool:
    if pattern == dotted_name or pattern == leaf:
        return True
    if any(ch in pattern for ch in "*?["):
        return fnmatch.fnmatch(dotted_name, pattern)
    return False


def _set_submodule(root: nn.Module, dotted_name: str, new: nn.Module) -> None:
    parent_path, _, child = dotted_name.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, child, new)


def apply_lora(model: nn.Module, config: LoRAConfig, verbose: bool = False) -> nn.Module:
    """Replace matching submodules with LoRA equivalents, in place.

    Returns the same model object.  After this call the *only* trainable
    parameters are ``lora_A`` / ``lora_B`` (plus biases if requested) -- which
    is what makes the optimizer state small, and the optimizer state is the
    real memory win: Adam keeps two fp32 moments per trainable parameter, so
    freezing 99.7% of the model cuts optimizer memory by the same factor.
    """
    # Materialise the list first: we mutate the tree while iterating it.
    candidates = [(n, m) for n, m in model.named_modules() if config.matches(n)]
    replaced: list[str] = []
    already = 0

    for name, module in candidates:
        if isinstance(module, _LoRABase):
            already += 1  # already adapted; do not nest adapters inside adapters
            continue
        transposed = type(module).__name__ in _TRANSPOSED_LAYER_NAMES

        if isinstance(module, nn.Linear) or transposed:
            new = LoRALinear.from_linear(
                module,
                r=config.r,
                alpha=config.alpha,
                dropout=config.dropout,
                fan_in_fan_out=transposed,
                init_a=config.init_a,
                use_rslora=config.use_rslora,
            )
        elif isinstance(module, nn.Embedding) and config.adapt_embeddings:
            new = LoRAEmbedding.from_embedding(
                module, r=config.r, alpha=config.alpha, use_rslora=config.use_rslora
            )
        else:
            continue

        _set_submodule(model, name, new)
        replaced.append(name)

    if not replaced and not already:
        available = sorted({n.rsplit(".", 1)[-1] for n, m in model.named_modules()
                            if isinstance(m, (nn.Linear, nn.Embedding))})
        raise ValueError(
            f"target_modules={list(config.target_modules)} matched nothing. "
            f"Adaptable leaf names in this model: {available}"
        )

    mark_only_lora_as_trainable(model, train_biases=config.train_biases)
    if verbose:
        print(f"[lora] adapted {len(replaced)} modules (r={config.r}, alpha={config.alpha})")
        print(f"[lora] e.g. {replaced[:4]}{' ...' if len(replaced) > 4 else ''}")
    return model


def mark_only_lora_as_trainable(model: nn.Module, train_biases: str = "none") -> nn.Module:
    """Freeze everything except LoRA parameters (paper Sec. 4.1)."""
    if train_biases not in {"none", "all", "lora_only"}:
        raise ValueError(f"train_biases must be none|all|lora_only, got {train_biases!r}")

    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name

    if train_biases == "all":
        for name, param in model.named_parameters():
            if name.endswith("bias"):
                param.requires_grad = True
    elif train_biases == "lora_only":
        for _, module in lora_modules(model):
            if getattr(module, "bias", None) is not None:
                module.bias.requires_grad = True
    return model


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every adapter into its frozen weight (zero inference overhead)."""
    for _, module in lora_modules(model):
        module.merge()
    return model


@torch.no_grad()
def unmerge_lora(model: nn.Module) -> nn.Module:
    for _, module in lora_modules(model):
        module.unmerge()
    return model


class MergedLoRA:
    """Context manager: merge on enter, unmerge on exit.

    ::

        with MergedLoRA(model):
            logits = model(x)      # single dense matmul per layer

    Useful for timing and for evaluation, without permanently destroying the
    ability to keep training.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def __enter__(self) -> nn.Module:
        return merge_lora(self.model)

    def __exit__(self, *exc) -> None:
        unmerge_lora(self.model)
