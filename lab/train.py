"""Training / evaluation loop shared by every experiment.

Two properties matter more than speed here, because the whole point is
*comparing* adaptation methods:

* **Identical validation batches across runs.**  ``evaluate`` draws its batches
  from a generator re-seeded to a fixed value every call, so run A and run B
  see byte-for-byte the same validation data.  Without this, differences of
  0.01 bits/byte between ranks are indistinguishable from sampling noise.
* **Identical training batch order across runs.**  Seeded per run, not per
  method, so the only thing that changes between two arms of an ablation is
  the parameterisation.

fp16 (not bf16) is forced on Turing: a T4 reports ``is_bf16_supported() == True``
but has no bf16 tensor cores, so bf16 is emulated and several times slower.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 24
    block_size: int = 256
    lr: float = 3e-4
    min_lr_frac: float = 0.1
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_batches: int = 40
    seed: int = 1337
    log_every: int = 100
    amp: bool = True


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def amp_dtype(device: str) -> torch.dtype:
    """fp16 on Turing (sm_75); bf16 only from Ampere onward."""
    if not device.startswith("cuda"):
        return torch.float32
    major = torch.cuda.get_device_capability(torch.device(device).index or 0)[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def get_batch(data: np.ndarray, batch_size: int, block_size: int,
              gen: np.random.Generator, device: str):
    ix = gen.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64)
    xt = torch.from_numpy(x).to(device, non_blocking=True)
    yt = torch.from_numpy(y).to(device, non_blocking=True)
    return xt, yt


@torch.no_grad()
def evaluate(model: nn.Module, data: np.ndarray, cfg: TrainConfig, device: str,
             eval_seed: int = 4242) -> float:
    """Mean cross-entropy (nats/byte) on a *fixed* set of validation batches."""
    was_training = model.training
    model.eval()
    gen = np.random.default_rng(eval_seed)  # re-seeded every call: same batches always
    dt = amp_dtype(device)
    total = 0.0
    for _ in range(cfg.eval_batches):
        x, y = get_batch(data, cfg.batch_size, cfg.block_size, gen, device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            dtype=dt, enabled=cfg.amp and device.startswith("cuda")):
            _, loss = model(x, y)
        total += loss.float().item()
    if was_training:
        model.train()
    return total / cfg.eval_batches


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    prog = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    prog = min(1.0, max(0.0, prog))
    coeff = 0.5 * (1.0 + math.cos(math.pi * prog))
    min_lr = cfg.lr * cfg.min_lr_frac
    return min_lr + coeff * (cfg.lr - min_lr)


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    """Weight decay on matrices only -- not on biases, LayerNorms, or LoRA_A/B.

    Decaying the adapter would pull ``B`` back toward zero, i.e. toward "do
    nothing", which fights the objective directly.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "lora_" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def train(model: nn.Module, train_data: np.ndarray, val_data: np.ndarray,
          cfg: TrainConfig, device: str = "cuda", tag: str = "run",
          log: bool = True) -> dict:
    set_seed(cfg.seed)
    model.to(device).train()
    opt = build_optimizer(model, cfg)
    dt = amp_dtype(device)
    use_scaler = cfg.amp and dt is torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    gen = np.random.default_rng(cfg.seed)

    hist: dict = {"tag": tag, "step": [], "train_loss": [], "eval_step": [], "val_loss": [],
                  "cfg": asdict(cfg), "trainable": sum(p.numel() for p in model.parameters()
                                                       if p.requires_grad)}
    t0 = time.time()
    running = None

    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)

        x, y = get_batch(train_data, cfg.batch_size, cfg.block_size, gen, device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            dtype=dt, enabled=cfg.amp and device.startswith("cuda")):
            _, loss = model(x, y)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        l = loss.float().item()
        running = l if running is None else 0.9 * running + 0.1 * l
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            hist["step"].append(step)
            hist["train_loss"].append(running)
            if log:
                print(f"[{tag}] step {step:5d}/{cfg.steps} loss {running:.4f} "
                      f"lr {lr_at(step, cfg):.2e} {time.time()-t0:.0f}s", flush=True)
        if (step + 1) % cfg.eval_every == 0 or step == cfg.steps - 1:
            v = evaluate(model, val_data, cfg, device)
            hist["eval_step"].append(step)
            hist["val_loss"].append(v)
            if log:
                print(f"[{tag}] step {step:5d} VAL {v:.4f} ({v/math.log(2):.4f} bits/byte)",
                      flush=True)

    hist["wall_s"] = time.time() - t0
    hist["final_val"] = hist["val_loss"][-1] if hist["val_loss"] else None
    hist["best_val"] = min(hist["val_loss"]) if hist["val_loss"] else None
    return hist


def save_json(obj, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))
    print(f"[saved] {path}", flush=True)


def bits_per_byte(nats: float) -> float:
    return nats / math.log(2)
