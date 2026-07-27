"""Every adaptation grid in the study, in one driver.

    python experiments/02_grids.py --grid lr        # tune LR per method family
    python experiments/02_grids.py --grid methods   # LoRA vs the baselines
    python experiments/02_grids.py --grid rank      # paper Table 6
    python experiments/02_grids.py --grid matrix    # paper Table 5
    python experiments/02_grids.py --grid variants  # rsLoRA / DoRA / init

``--shard i --nshards n`` splits a grid across the box's two T4s; the shards
write separate JSON files that ``experiments/09_report.py`` merges.

A note on fairness, because it is the whole ballgame in this kind of
comparison: LoRA and full fine-tuning do **not** share an optimal learning
rate -- the paper itself uses 4e-4 for LoRA and ~1e-5 for full FT on RoBERTa.
Comparing them at one shared LR would be rigged, in whichever direction the LR
happened to favour.  So the ``lr`` grid sweeps each method family
independently on one domain, and every later grid uses each family's own best.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab.adapt import run_jobs
from lab.methods import LORA_ALL_ATTN, LORA_ALL_LINEAR, LORA_QV

DOMAINS = ["sympy", "torch", "matplotlib"]
TUNE_DOMAIN = "sympy"

# Adapter-style methods tolerate (and need) much larger LRs than full FT.
LR_ADAPTER = [1e-4, 3e-4, 1e-3, 3e-3]
LR_FULL = [3e-5, 1e-4, 3e-4, 1e-3]


def baseline(name, domain, lr, family=None):
    return {"name": name, "kind": "baseline", "baseline": name,
            "family": family or name, "domain": domain, "lr": lr}


def lora(name, targets, r, domain, lr, family=None, **kw):
    spec = {"name": name, "kind": "lora", "targets": list(targets), "r": r,
            "family": family or "lora", "domain": domain, "lr": lr}
    spec.update(kw)
    return spec


# ---------------------------------------------------------------- grids ----

def grid_lr():
    jobs = []
    for lr in LR_FULL:
        jobs.append(baseline("full_ft", TUNE_DOMAIN, lr))
    for lr in LR_ADAPTER:
        for b in ("bitfit", "layernorm", "last_block"):
            jobs.append(baseline(b, TUNE_DOMAIN, lr))
        jobs.append(lora("lora_qv_r8", LORA_QV, 8, TUNE_DOMAIN, lr))
    return jobs


def grid_methods(best):
    jobs = []
    for d in DOMAINS:
        jobs.append(baseline("zero_shot", d, 0.0))
        for b in ("full_ft", "bitfit", "layernorm", "last_block"):
            jobs.append(baseline(b, d, best.get(b, 3e-4)))
        jobs.append(lora("lora_qv_r8", LORA_QV, 8, d, best.get("lora", 1e-3)))
    return jobs


def grid_rank(best):
    """Paper Table 6: does a larger rank buy anything?"""
    lr = best.get("lora", 1e-3)
    jobs = []
    for d in (TUNE_DOMAIN, "torch"):
        for r in (1, 2, 4, 8, 16, 32, 64):
            jobs.append(lora(f"lora_qv_r{r}", LORA_QV, r, d, lr))
    return jobs


def grid_matrix(best):
    """Paper Table 5: given a fixed parameter budget, *which* matrices?

    Rank is halved as the number of adapted matrix types doubles, so every row
    trains the same number of adapter parameters -- that equal-budget
    constraint is what makes the comparison mean anything.
    """
    lr = best.get("lora", 1e-3)
    jobs = []
    for d in (TUNE_DOMAIN, "torch"):
        for tgt, r in [(["q_proj"], 8), (["k_proj"], 8), (["v_proj"], 8), (["o_proj"], 8),
                       (["q_proj", "k_proj"], 4), (["q_proj", "v_proj"], 4),
                       (["v_proj", "o_proj"], 4), (LORA_ALL_ATTN, 2)]:
            label = "+".join(t.split("_")[0] for t in tgt)
            jobs.append(lora(f"matrix_{label}_r{r}", tgt, r, d, lr, family="matrix"))
        # Modern practice adapts the MLP too; not budget-matched, reported as-is.
        jobs.append(lora("matrix_all_linear_r2", LORA_ALL_LINEAR, 2, d, lr, family="matrix"))
    return jobs


def grid_variants(best):
    lr = best.get("lora", 1e-3)
    jobs = []
    d = TUNE_DOMAIN
    for r in (2, 8, 64):
        jobs.append(lora(f"lora_r{r}", LORA_QV, r, d, lr, family="lora"))
        jobs.append(lora(f"rslora_r{r}", LORA_QV, r, d, lr, family="rslora", use_rslora=True))
    jobs.append(lora("lora_r8_gaussian", LORA_QV, 8, d, lr, family="init", init_a="gaussian"))
    jobs.append(lora("lora_r8_kaiming", LORA_QV, 8, d, lr, family="init", init_a="kaiming"))
    jobs.append({"name": "dora_r8", "kind": "dora", "targets": LORA_QV, "r": 8,
                 "family": "dora", "domain": d, "lr": lr})
    # alpha decoupled from r: is alpha/r really just a learning-rate knob?
    for alpha in (2, 8, 32):
        jobs.append(lora(f"lora_r8_alpha{alpha}", LORA_QV, 8, d, lr,
                         family="alpha", alpha=float(alpha)))
    return jobs


GRIDS = {"lr": grid_lr, "methods": grid_methods, "rank": grid_rank,
         "matrix": grid_matrix, "variants": grid_variants}


def load_best_lrs(path: str) -> dict:
    """Pick each family's best LR from the tuning grid."""
    if not Path(path).exists():
        print(f"[warn] {path} missing; falling back to defaults", flush=True)
        return {}
    best: dict = {}
    for r in json.load(open(path)):
        fam = r["family"]
        if fam not in best or r["best_val"] < best[fam]["best_val"]:
            best[fam] = r
    out = {fam: rec["lr"] for fam, rec in best.items()}
    print(f"[lr] chosen per family: {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, choices=sorted(GRIDS))
    ap.add_argument("--ckpt", default="/kaggle/working/ckpt/base.pt")
    ap.add_argument("--results", default="/kaggle/working/results")
    ap.add_argument("--lr-json", default="/kaggle/working/results/02_lr_merged.json")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    fn = GRIDS[args.grid]
    jobs = fn() if args.grid == "lr" else fn(load_best_lrs(args.lr_json))
    mine = jobs[args.shard::args.nshards]
    print(f"grid={args.grid} total={len(jobs)} shard={args.shard}/{args.nshards} "
          f"-> {len(mine)} jobs", flush=True)

    out = os.path.join(args.results, f"02_{args.grid}_shard{args.shard}.json")
    run_jobs(mine, out, args.ckpt, device=args.device,
             steps=args.steps, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
