"""LoRA from scratch -- a readable, tested reimplementation of arXiv:2106.09685.

Quick start::

    from lora import LoRAConfig, apply_lora, summarize

    model = ...                                   # any nn.Module
    apply_lora(model, LoRAConfig(target_modules=["query", "value"], r=8, alpha=8))
    print(summarize(model))
    # ... train as usual; only lora_A / lora_B receive gradients ...
    merge_lora(model)                             # zero-overhead inference
"""

from .layers import LoRAEmbedding, LoRALinear, lora_modules
from .inject import (
    LoRAConfig,
    MergedLoRA,
    apply_lora,
    mark_only_lora_as_trainable,
    merge_lora,
    unmerge_lora,
)
from .utils import (
    adapter_size_bytes,
    count_parameters,
    load_lora_state_dict,
    lora_state_dict,
    summarize,
)
from .variants import DoRALinear, apply_dora

__version__ = "0.1.0"

__all__ = [
    "LoRALinear",
    "LoRAEmbedding",
    "DoRALinear",
    "LoRAConfig",
    "MergedLoRA",
    "apply_lora",
    "apply_dora",
    "merge_lora",
    "unmerge_lora",
    "mark_only_lora_as_trainable",
    "lora_modules",
    "count_parameters",
    "lora_state_dict",
    "load_lora_state_dict",
    "adapter_size_bytes",
    "summarize",
    "__version__",
]
