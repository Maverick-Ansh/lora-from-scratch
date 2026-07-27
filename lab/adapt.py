"""One adaptation run: load the frozen base, attach a method, train, measure.

Every experiment in this repo is a list of these, so the comparison logic lives
in exactly one place.  Anything that could differ between two arms of an
ablation other than the thing being ablated -- batch order, validation batches,
step count, schedule -- is fixed here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from lora import count_parameters, lora_state_dict
from lora.layers import lora_modules

from . import methods as M
from .data import DOMAIN_ROOTS, cached_corpus
from .model import GPT, GPTConfig
from .train import TrainConfig, bits_per_byte, evaluate, train

_CORPUS_CACHE: dict = {}


def load_base(ckpt_path: str, device: str = "cuda"):
    """Load the pre-trained checkpoint.  Fresh copy every run -- adaptation is
    destructive and we compare methods against the *same* starting point."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device), cfg


def domain_corpus(domain: str):
    if domain not in _CORPUS_CACHE:
        _CORPUS_CACHE[domain] = cached_corpus(
            domain, DOMAIN_ROOTS[domain], val_frac=0.05, max_bytes=40_000_000
        )
    return _CORPUS_CACHE[domain]


def build_method(spec: dict, model):
    """Turn a plain-dict job spec into a configured, partially-frozen model."""
    kind = spec["kind"]
    if kind == "baseline":
        return M.BASELINES[spec["baseline"]](model)
    if kind == "lora":
        return M.make_lora(
            spec["targets"], spec["r"], spec.get("alpha"),
            dropout=spec.get("dropout", 0.0),
            use_rslora=spec.get("use_rslora", False),
            init_a=spec.get("init_a", "kaiming"),
        )(model)
    if kind == "dora":
        return M.make_dora(spec["targets"], spec["r"], spec.get("alpha"))(model)
    raise ValueError(f"unknown kind {kind!r}")


def run_job(spec: dict, ckpt: str, device: str = "cuda", steps: int = 800,
            batch_size: int = 24, seed: int = 1337, log: bool = False) -> dict:
    """Run one adaptation and return a flat record ready for a results table."""
    t0 = time.time()
    corpus = domain_corpus(spec["domain"])
    model, gcfg = load_base(ckpt, device)
    model = build_method(spec, model)

    counts = count_parameters(model)
    tcfg = TrainConfig(
        steps=steps, batch_size=batch_size, block_size=gcfg.block_size,
        lr=spec["lr"], warmup=max(10, steps // 20),
        eval_every=max(1, steps // 4), eval_batches=40, seed=seed,
        log_every=max(1, steps // 4),
    )

    rec = dict(spec)
    rec.update({
        "trainable": counts["trainable"],
        "total": counts["total"],
        "trainable_pct": counts["trainable_pct"],
        "n_adapted": sum(1 for _ in lora_modules(model)),
        "steps": steps,
        "seed": seed,
    })

    if counts["trainable"] == 0:
        # zero-shot: no training, just the base model's loss on the new domain
        v = evaluate(model, corpus.val, tcfg, device)
        rec.update({"final_val": v, "best_val": v, "val_curve": [v], "wall_s": time.time() - t0})
    else:
        hist = train(model, corpus.train, corpus.val, tcfg, device=device,
                     tag=spec["name"], log=log)
        rec.update({
            "final_val": hist["final_val"],
            "best_val": hist["best_val"],
            "val_curve": hist["val_loss"],
            "train_curve": hist["train_loss"],
            "wall_s": hist["wall_s"],
        })
        if spec["kind"] in ("lora", "dora"):
            rec["adapter_params"] = sum(t.numel() for t in lora_state_dict(model).values())

    rec["final_bpb"] = bits_per_byte(rec["final_val"])
    rec["best_bpb"] = bits_per_byte(rec["best_val"])
    del model
    torch.cuda.empty_cache()
    print(f"[done] {rec['name']:34s} {spec['domain']:11s} lr={spec['lr']:<8.0e} "
          f"trainable={rec['trainable']:>9,} best={rec['best_bpb']:.4f} bpb "
          f"({time.time()-t0:.0f}s)", flush=True)
    return rec


def run_jobs(jobs: list[dict], out_path: str, ckpt: str, device: str = "cuda",
             **kw) -> list[dict]:
    """Run a job list, checkpointing results to disk after *every* job.

    The experiment box drops its connection on long silences and cannot be
    interrupted, so partial results have to survive the process dying.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for i, spec in enumerate(jobs):
        print(f"--- job {i+1}/{len(jobs)}: {spec['name']} [{spec['domain']}] ---", flush=True)
        records.append(run_job(spec, ckpt, device=device, **kw))
        out.write_text(json.dumps(records, indent=2))
    print(f"[saved] {out_path}  ({len(records)} records)", flush=True)
    return records
