"""Low-rank adaptation layers, implemented from scratch in plain PyTorch.

Reference
---------
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li,
Shean Wang, Lu Wang, Weizhu Chen.
*LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685v2.

The whole idea, in one equation (paper Eq. 3)::

    h = W0 @ x + dW @ x = W0 @ x + (alpha / r) * B @ A @ x

with ``W0`` frozen, ``A`` of shape ``(r, in_features)``, ``B`` of shape
``(out_features, r)`` and ``r << min(in_features, out_features)``.

Three properties fall out of that equation, and every design decision in this
file exists to preserve one of them:

1. **Training starts as a no-op.**  ``B`` is initialised to zero, so
   ``dW = B @ A = 0`` at step 0 and the adapted model is *exactly* the
   pre-trained model.  No warm-up damage, no accuracy cliff.
2. **Inference costs nothing.**  ``dW`` is a plain matrix of the same shape as
   ``W0``, so it can be folded in: ``W <- W0 + (alpha/r) * B @ A``.  After
   :meth:`LoRALinear.merge` the module is arithmetically a single ``nn.Linear``
   again -- unlike adapter layers, which add depth you can never remove.
3. **The update is cheap to store.**  ``r * (d + k)`` parameters instead of
   ``d * k``.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

InitA = Literal["kaiming", "gaussian"]

__all__ = ["LoRALinear", "LoRAEmbedding", "lora_modules"]


class _LoRABase(nn.Module):
    """Shared bookkeeping for every LoRA-adapted module.

    Subclasses own the ``lora_A`` / ``lora_B`` parameters and the forward pass;
    this base owns the scaling rule, the merge state machine, and the frozen
    reference to the pre-trained tensor.
    """

    #: set by subclasses -- the frozen pre-trained parameter we are adapting
    weight: torch.Tensor

    def __init__(
        self,
        r: int,
        alpha: float,
        dropout: float = 0.0,
        use_rslora: bool = False,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got r={r}")
        self.r = int(r)
        self.alpha = float(alpha)
        self.use_rslora = bool(use_rslora)

        # Paper, Sec. 4.1: "We then scale dW @ x by alpha / r, where alpha is a
        # constant in r."  The point of tying the scale to r is that when you
        # re-tune r you do not also have to re-tune the learning rate -- the
        # magnitude of the update stays roughly fixed.
        #
        # rsLoRA (Kalajdzievski 2023) argues the correct variance-preserving
        # exponent is 1/2, not 1: with alpha/r the effective update *shrinks*
        # as r grows, which is why large-r LoRA is often reported as "not
        # helping".  We keep alpha/r as the default because that is what the
        # paper specifies, and expose the alternative as a flag.
        self.scaling = alpha / math.sqrt(self.r) if use_rslora else alpha / self.r

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.merged = False

    # -- the update itself ------------------------------------------------

    def delta_weight(self) -> torch.Tensor:
        """Return ``(alpha/r) * B @ A``, shaped like the frozen weight."""
        raise NotImplementedError

    # -- merge state machine ----------------------------------------------

    @torch.no_grad()
    def merge(self) -> "_LoRABase":
        """Fold ``dW`` into the frozen weight.  Idempotent.

        This is the "no additional inference latency" claim of the paper made
        concrete.  After merging, the LoRA path is skipped entirely and the
        module performs exactly one dense matmul.
        """
        if not self.merged:
            self.weight.data += self.delta_weight().to(self.weight.dtype)
            self.merged = True
        return self

    @torch.no_grad()
    def unmerge(self) -> "_LoRABase":
        """Subtract ``dW`` back out, restoring the pre-trained weight.

        Needed for multi-adapter serving: you cannot swap adapter B in while
        adapter A is still baked into the weights.  Note this round-trip is
        only *exactly* lossless in fp32 -- in fp16 the add/subtract pair loses
        a few ULPs, which is why serving stacks keep a pristine copy of W0
        rather than relying on unmerge.
        """
        if self.merged:
            self.weight.data -= self.delta_weight().to(self.weight.dtype)
            self.merged = False
        return self

    def extra_repr(self) -> str:
        return (
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4g}, "
            f"merged={self.merged}, rslora={self.use_rslora}"
        )


class LoRALinear(_LoRABase):
    """``nn.Linear`` with a trainable rank-``r`` update bolted on.

    Constructed by *wrapping* an existing layer (:meth:`from_linear`) so that
    the pre-trained tensor object is reused, never copied -- injection into a
    600M-parameter model costs no extra memory for the frozen weights.

    Parameters
    ----------
    fan_in_fan_out
        ``True`` when the wrapped layer stores its weight transposed, i.e. as
        ``(in_features, out_features)``.  HuggingFace's ``Conv1D`` (GPT-2,
        GPT-Neo) does exactly this, and getting it wrong is the single most
        common way a hand-rolled LoRA silently trains garbage: the shapes
        happen to work out for square attention projections, so there is no
        error -- only a model that does not learn.
    init_a
        ``"kaiming"`` reproduces the reference implementation
        (``kaiming_uniform_(a=sqrt(5))``, i.e. what ``nn.Linear`` uses for its
        own weights); ``"gaussian"`` reproduces the *text* of the paper
        ("random Gaussian initialization for A").  They differ, and the paper
        never says which was used for the reported numbers.  See
        ``experiments/00_rank_intuition.py`` for the measured difference.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
        bias: bool = True,
        fan_in_fan_out: bool = False,
        init_a: InitA = "kaiming",
        gaussian_std: float = 0.02,
        use_rslora: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(r=r, alpha=alpha, dropout=dropout, use_rslora=use_rslora)
        factory = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.fan_in_fan_out = fan_in_fan_out
        self.init_a = init_a
        self.gaussian_std = gaussian_std

        w_shape = (in_features, out_features) if fan_in_fan_out else (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(w_shape, **factory), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features, **factory)) if bias else None

        # A: (r, in) -- projects the input down into the r-dimensional subspace.
        # B: (out, r) -- projects back up.  Both are *always* stored in this
        # orientation regardless of fan_in_fan_out, so the analysis code in
        # experiments/03_analysis.py never has to special-case GPT-2.
        self.lora_A = nn.Parameter(torch.empty(self.r, in_features, **factory))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.r, **factory))
        self.reset_lora_parameters()

    # -- construction ------------------------------------------------------

    @classmethod
    def from_linear(cls, layer: nn.Module, **kw) -> "LoRALinear":
        """Wrap an existing ``nn.Linear`` (or HF ``Conv1D``), reusing its tensors."""
        fan_in_fan_out = kw.get("fan_in_fan_out", False)
        if isinstance(layer, nn.Linear):
            in_f, out_f = layer.in_features, layer.out_features
        else:  # transformers.pytorch_utils.Conv1D stores nf/nx and a (nx, nf) weight
            weight = layer.weight
            if not fan_in_fan_out:
                raise ValueError(
                    f"{type(layer).__name__} stores its weight transposed; "
                    "pass fan_in_fan_out=True"
                )
            in_f, out_f = weight.shape

        kw.setdefault("bias", getattr(layer, "bias", None) is not None)
        obj = cls(in_f, out_f, device=layer.weight.device, dtype=layer.weight.dtype, **kw)

        # Reuse, do not clone: `obj.weight` must *be* the pre-trained tensor.
        obj.weight = nn.Parameter(layer.weight.data, requires_grad=False)
        if obj.bias is not None:
            obj.bias = nn.Parameter(layer.bias.data, requires_grad=False)
        return obj

    def reset_lora_parameters(self) -> None:
        if self.init_a == "kaiming":
            # Reference impl (microsoft/LoRA, loralib/layers.py).
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        elif self.init_a == "gaussian":
            nn.init.normal_(self.lora_A, mean=0.0, std=self.gaussian_std)
        else:
            raise ValueError(f"unknown init_a={self.init_a!r}")
        # The load-bearing line of the entire method: B = 0  =>  dW = 0  =>
        # the adapted model is bit-identical to the pre-trained one at step 0.
        nn.init.zeros_(self.lora_B)

    # -- math --------------------------------------------------------------

    def delta_weight(self) -> torch.Tensor:
        dw = (self.lora_B @ self.lora_A) * self.scaling  # (out, in)
        return dw.T if self.fan_in_fan_out else dw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.T if self.fan_in_fan_out else self.weight
        base = F.linear(x, w, self.bias)
        if self.merged:
            # dW already lives inside self.weight; adding it again would
            # double-count it.
            return base
        # Two skinny matmuls, never materialising the (out, in) product:
        #   x (.., in) @ A.T (in, r) -> (.., r) @ B.T (r, out) -> (.., out)
        # That ordering is why LoRA training is cheap; forming B@A first would
        # cost as much as the frozen layer itself.
        lora = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base + lora * self.scaling


class LoRAEmbedding(_LoRABase):
    """``nn.Embedding`` with a rank-``r`` update.

    The paper adapts attention projections only, but embeddings are the other
    place a downstream task genuinely needs new capacity (new domain tokens),
    and modern recipes do adapt them.  Roles are swapped relative to
    :class:`LoRALinear`: ``A`` is the one that must be zero, because here it is
    ``A`` that is indexed by the token id.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 8,
        alpha: float = 8.0,
        padding_idx: int | None = None,
        use_rslora: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(r=r, alpha=alpha, dropout=0.0, use_rslora=use_rslora)
        factory = {"device": device, "dtype": dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, **factory), requires_grad=False
        )
        self.lora_A = nn.Parameter(torch.zeros(self.r, num_embeddings, **factory))
        self.lora_B = nn.Parameter(torch.empty(embedding_dim, self.r, **factory))
        self.reset_lora_parameters()

    @classmethod
    def from_embedding(cls, layer: nn.Embedding, **kw) -> "LoRAEmbedding":
        obj = cls(
            layer.num_embeddings,
            layer.embedding_dim,
            padding_idx=layer.padding_idx,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
            **kw,
        )
        obj.weight = nn.Parameter(layer.weight.data, requires_grad=False)
        return obj

    def reset_lora_parameters(self) -> None:
        nn.init.zeros_(self.lora_A)
        nn.init.normal_(self.lora_B)

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A).T * self.scaling  # (num_emb, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        base = F.embedding(ids, self.weight, self.padding_idx)
        if self.merged:
            return base
        # Row-gather from A.T, then project up -- never builds the full
        # (vocab, dim) delta.
        lora = F.embedding(ids, self.lora_A.T, self.padding_idx) @ self.lora_B.T
        return base + lora * self.scaling


def lora_modules(model: nn.Module):
    """Yield ``(name, module)`` for every LoRA-adapted submodule."""
    for name, module in model.named_modules():
        if isinstance(module, _LoRABase):
            yield name, module
