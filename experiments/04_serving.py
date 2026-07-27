"""The deployment claims: zero added latency, and one base model for N tasks.

The paper's Section 1 makes an operational promise that Table 1 quantifies:
unlike adapter layers, LoRA "introduces no inference latency", because the
update is a matrix of the same shape as the weight and can be summed into it
before serving.  That is a checkable claim, so we check it -- against a real
bottleneck-adapter baseline rather than in the abstract.

Four measurements:

1. **Latency.**  base / LoRA unmerged / LoRA merged / bottleneck adapter,
   across batch sizes.  Merged LoRA should be indistinguishable from base;
   the bottleneck adapter should not be, because it adds sequential depth
   that no amount of algebra can remove.
2. **Merge exactness.**  Merging is only *exactly* free in fp32.  In fp16 the
   add/subtract round trip loses ULPs -- worth knowing before you build a
   serving stack that unmerges.
3. **Hot-swap.**  Time to switch tasks by swapping adapter tensors on one
   resident base model.
4. **Memory.**  N tasks as N full models vs one base plus N adapters.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab.methods import LORA_QV
from lab.model import GPT, GPTConfig
from lora import (LoRAConfig, apply_lora, lora_state_dict, load_lora_state_dict,
                  merge_lora, unmerge_lora)
from lora.layers import lora_modules


class BottleneckAdapter(nn.Module):
    """A Houlsby-style adapter: the thing LoRA is measured against.

    ``h -> h + W_up(GELU(W_down(h)))`` inserted *in series* after a sublayer.
    Same parameter count as a LoRA of comparable rank, but it is a new
    sequential dependency in the graph -- so it costs latency on every token
    forever, and no algebraic trick removes it.  That contrast is the entire
    operational argument for LoRA.
    """

    def __init__(self, d_model: int, bottleneck: int = 8) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(torch.nn.functional.gelu(self.down(x)))


def add_bottleneck_adapters(model: GPT, bottleneck: int = 8) -> GPT:
    """Wrap every block's attention and MLP output with a serial adapter."""
    d = model.cfg.d_model
    for block in model.blocks:
        attn, mlp = block.attn, block.mlp
        a1 = BottleneckAdapter(d, bottleneck).to(next(model.parameters()).device)
        a2 = BottleneckAdapter(d, bottleneck).to(next(model.parameters()).device)
        block.attn = nn.Sequential(attn, a1)
        block.mlp = nn.Sequential(mlp, a2)
    return model


@torch.no_grad()
def bench(model: nn.Module, batch: int, seq: int, device: str, iters: int = 120,
          warmup: int = 40) -> float:
    """Median-of-iters forward latency in milliseconds.

    Generous warmup and a median (not a mean) because the first pass through a
    fresh module triggers cuDNN autotuning and allocator growth; a 30-iteration
    run with 10 warmups produced a *negative* 14.7% "overhead" at batch 1,
    which is measurement drift, not physics.  The `noise_floor` control below
    quantifies what is left.
    """
    model.eval()
    x = torch.randint(0, 256, (batch, seq), device=device)
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def fresh(ckpt: str, device: str) -> tuple[GPT, GPTConfig]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = GPTConfig(**ck["cfg"])
    m = GPT(cfg)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/kaggle/working/ckpt/base.pt")
    ap.add_argument("--out", default="/kaggle/working/results/04_serving.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--r", type=int, default=8)
    args = ap.parse_args()
    dev = args.device
    out: dict = {"r": args.r, "device": torch.cuda.get_device_name(0)}

    # ---------------------------------------------------------- latency ----
    seq = 256
    rows = []
    for batch in (1, 4, 16, 32):
        base, cfg = fresh(args.ckpt, dev)
        t_base = bench(base, batch, seq, dev)

        lora_m, _ = fresh(args.ckpt, dev)
        apply_lora(lora_m, LoRAConfig(target_modules=tuple(LORA_QV), r=args.r,
                                      alpha=float(args.r)))
        # Give the adapter non-zero content so nothing can be optimised away.
        for _, mod in lora_modules(lora_m):
            nn.init.normal_(mod.lora_B, std=0.01)
        t_unmerged = bench(lora_m, batch, seq, dev)
        merge_lora(lora_m)
        t_merged = bench(lora_m, batch, seq, dev)

        adp, _ = fresh(args.ckpt, dev)
        add_bottleneck_adapters(adp, bottleneck=args.r)
        t_adapter = bench(adp, batch, seq, dev)

        # Noise floor: a second, independently loaded copy of the *unmodified*
        # base model. Its "overhead" is by construction zero, so whatever this
        # measures is the run-to-run error, and no difference smaller than it
        # is real.
        base2, _ = fresh(args.ckpt, dev)
        t_base2 = bench(base2, batch, seq, dev)

        row = {
            "batch": batch, "seq": seq,
            "base_ms": t_base, "base_repeat_ms": t_base2,
            "noise_floor_pct": 100 * (t_base2 - t_base) / t_base,
            "lora_unmerged_ms": t_unmerged,
            "lora_merged_ms": t_merged, "bottleneck_adapter_ms": t_adapter,
            "lora_merged_overhead_pct": 100 * (t_merged - t_base) / t_base,
            "lora_unmerged_overhead_pct": 100 * (t_unmerged - t_base) / t_base,
            "adapter_overhead_pct": 100 * (t_adapter - t_base) / t_base,
        }
        rows.append(row)
        print(f"[latency] bs={batch:3d} base {t_base:7.2f}ms | merged "
              f"{row['lora_merged_overhead_pct']:+5.1f}% | unmerged "
              f"{row['lora_unmerged_overhead_pct']:+5.1f}% | bottleneck "
              f"{row['adapter_overhead_pct']:+5.1f}% | noise floor "
              f"{row['noise_floor_pct']:+5.1f}%", flush=True)
        del base, base2, lora_m, adp
        torch.cuda.empty_cache()
    out["latency"] = rows

    # ---------------------------------------------- merge exactness --------
    exact = {}
    for dtype_name, dtype in (("float32", torch.float32), ("float16", torch.float16)):
        m, cfg = fresh(args.ckpt, dev)
        m = m.to(dtype)
        apply_lora(m, LoRAConfig(target_modules=tuple(LORA_QV), r=args.r, alpha=float(args.r)))
        for _, mod in lora_modules(m):
            nn.init.normal_(mod.lora_B, std=0.01)
        x = torch.randint(0, 256, (4, 128), device=dev)
        with torch.no_grad():
            w0 = {n: mod.weight.detach().float().clone() for n, mod in lora_modules(m)}
            before = m(x)[0].float()
            merge_lora(m)
            after = m(x)[0].float()
            unmerge_lora(m)
            # After merge->unmerge the weight should be back to W0 exactly.
            # In fp16 it is not: that residue is the reason serving stacks keep
            # a pristine copy of W0 instead of trusting unmerge.
            drift = max((mod.weight.detach().float() - w0[n]).abs().max().item()
                        for n, mod in lora_modules(m))
        exact[dtype_name] = {
            "max_abs_logit_diff_merge": float((after - before).abs().max()),
            "rel_logit_diff_merge": float((after - before).abs().max()
                                          / before.abs().max()),
            "max_abs_weight_drift_roundtrip": drift,
        }
        print(f"[exactness] {dtype_name}: merge changes logits by at most "
              f"{exact[dtype_name]['max_abs_logit_diff_merge']:.3e}", flush=True)
        del m
        torch.cuda.empty_cache()
    out["merge_exactness"] = exact

    # --------------------------------------------------------- hot swap ----
    m, cfg = fresh(args.ckpt, dev)
    apply_lora(m, LoRAConfig(target_modules=tuple(LORA_QV), r=args.r, alpha=float(args.r)))
    adapters = []
    for i in range(4):
        for _, mod in lora_modules(m):
            nn.init.normal_(mod.lora_B, std=0.01 * (i + 1))
        adapters.append(lora_state_dict(m))
    adapter_bytes = sum(t.numel() * 4 for t in adapters[0].values())

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 40
    for i in range(N):
        load_lora_state_dict(m, adapters[i % len(adapters)])
    torch.cuda.synchronize()
    swap_ms = (time.perf_counter() - t0) * 1000 / N

    full_model_bytes = sum(p.numel() * 4 for p in fresh(args.ckpt, dev)[0].parameters())
    n_tasks = 100
    out["serving"] = {
        "adapter_bytes": adapter_bytes,
        "full_model_bytes": full_model_bytes,
        "swap_ms": swap_ms,
        "n_tasks": n_tasks,
        "naive_total_bytes": full_model_bytes * n_tasks,
        "lora_total_bytes": full_model_bytes + adapter_bytes * n_tasks,
        "ratio": (full_model_bytes * n_tasks) / (full_model_bytes + adapter_bytes * n_tasks),
    }
    print(f"[serving] adapter {adapter_bytes/1024:.1f} KiB vs model "
          f"{full_model_bytes/1e6:.1f} MB | swap {swap_ms:.2f} ms | "
          f"{n_tasks} tasks: {out['serving']['ratio']:.1f}x smaller", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
