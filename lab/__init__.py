"""Experiment scaffolding: a small GPT, an on-disk corpus, training loops.

Kept separate from ``lora/`` on purpose -- ``lora/`` is the paper, this is the
laboratory bench it gets tested on.
"""

from .model import GPT, GPTConfig
from .train import TrainConfig, train, evaluate, save_json, bits_per_byte, set_seed

__all__ = [
    "GPT", "GPTConfig",
    "TrainConfig", "train", "evaluate", "save_json", "bits_per_byte", "set_seed",
]
