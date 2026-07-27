"""Sections 7.2 and 7.3: what the learned update actually *is*.

This is the part of the paper that stops being an engineering result and
starts being a claim about neural networks -- that the update needed to
specialise a model is not merely compressible, but concentrated in a handful
of directions that the pre-trained weight already contains and under-uses.

Two measurements:

**7.2 -- Subspace similarity.**  Train the same adaptation twice at different
ranks (r=8 and r=64) and ask how much of the low-rank subspace they share.
With ``U_A^i`` the first ``i`` left-singular directions of adapter ``A``:

    phi(A, B, i, j) = || U_A^i.T @ U_B^j ||_F^2 / min(i, j)   in [0, 1]

The paper's finding is that phi is large for i = j = 1 and decays fast: the
top direction is real and reproducible, the rest is mostly noise.  If that
holds, r=64 is not learning 64 useful directions -- it is learning a few and
padding.  That is *why* r=1 works in Table 6, so this measurement and that
table are the same claim seen from two sides.

**7.3 -- Amplification.**  Project the frozen weight onto the subspace the
adapter chose, and compare magnitudes:

    amplification = ||dW||_F / ||U.T @ W0 @ V||_F

Two controls decide what the number means: a *random* r-dimensional subspace
(is the adapter's subspace special at all?) and W0's own *top-r* singular
subspace (is the adapter simply re-scaling what W0 already emphasises?).
The paper reports ~21.5 for r=4, with dW's subspace neither random nor
top-of-W0 -- i.e. the adapter amplifies features the base model had learned
but left quiet.

Everything is written out as JSON; figures are rendered off-box.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab.adapt import build_method, domain_corpus, load_base
from lab.methods import LORA_QV
from lab.train import TrainConfig, train
from lora.layers import lora_modules


# ------------------------------------------------------------- helpers ----

def train_adapter(ckpt, domain, r, lr, steps, device, seed, targets=LORA_QV):
    """Train one LoRA adapter and hand back its per-layer A/B matrices."""
    spec = {"name": f"analysis_r{r}_s{seed}", "kind": "lora", "targets": list(targets),
            "r": r, "domain": domain, "lr": lr, "family": "lora"}
    model, gcfg = load_base(ckpt, device)
    model = build_method(spec, model)
    corpus = domain_corpus(domain)
    tcfg = TrainConfig(steps=steps, batch_size=24, block_size=gcfg.block_size, lr=lr,
                       warmup=max(10, steps // 20), eval_every=steps, eval_batches=40,
                       seed=seed, log_every=steps)
    hist = train(model, corpus.train, corpus.val, tcfg, device=device,
                 tag=spec["name"], log=False)

    mats = {}
    for name, mod in lora_modules(model):
        mats[name] = {
            "A": mod.lora_A.detach().float().cpu().numpy(),
            "B": mod.lora_B.detach().float().cpu().numpy(),
            "W0": mod.weight.detach().float().cpu().numpy(),
            "scaling": mod.scaling,
        }
    del model
    torch.cuda.empty_cache()
    return mats, hist["final_val"]


def right_singular_dirs(A: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the row space of A, as columns, ordered by energy.

    A is (r, in_features); its row space is the r-dimensional input subspace
    the adapter reads from.  This is the object the paper's phi compares.
    """
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    return vh.T  # (in_features, r)


def subspace_similarity(Ua: np.ndarray, Ub: np.ndarray, max_i: int, max_j: int) -> np.ndarray:
    """phi(i, j) = ||Ua[:, :i].T @ Ub[:, :j]||_F^2 / min(i, j), the paper's Eq. 4."""
    phi = np.zeros((max_i, max_j))
    # Grow the Gram matrix once and take cumulative sums: phi for all (i, j)
    # is just a 2-D prefix sum of the squared cross-projections.
    g = (Ua[:, :max_i].T @ Ub[:, :max_j]) ** 2      # (max_i, max_j)
    csum = g.cumsum(axis=0).cumsum(axis=1)
    for i in range(max_i):
        for j in range(max_j):
            phi[i, j] = csum[i, j] / min(i + 1, j + 1)
    return phi


def amplification(mats: dict, rng: np.random.Generator) -> list[dict]:
    """For every adapted layer, compare ||dW|| against W0 seen through dW's subspace."""
    rows = []
    for name, m in mats.items():
        A, B, W0, s = m["A"], m["B"], m["W0"], m["scaling"]
        dW = (B @ A) * s                                    # (out, in)
        r = A.shape[0]

        U, S, Vh = np.linalg.svd(dW, full_matrices=False)
        Ur, Vr = U[:, :r], Vh[:r].T                          # (out, r), (in, r)

        proj_dw = np.linalg.norm(Ur.T @ W0 @ Vr, "fro")      # W0 inside dW's subspace

        # Control A: a random r-dimensional subspace of the same shape.
        Ru = np.linalg.qr(rng.standard_normal((W0.shape[0], r)))[0]
        Rv = np.linalg.qr(rng.standard_normal((W0.shape[1], r)))[0]
        proj_rand = np.linalg.norm(Ru.T @ W0 @ Rv, "fro")

        # Control B: W0's own top-r singular subspace.
        Uw, Sw, Vwh = np.linalg.svd(W0, full_matrices=False)
        proj_top = np.linalg.norm(Uw[:, :r].T @ W0 @ Vwh[:r].T, "fro")

        rows.append({
            "layer": name,
            "r": int(r),
            "dW_fro": float(np.linalg.norm(dW, "fro")),
            "W0_fro": float(np.linalg.norm(W0, "fro")),
            "proj_W0_on_dW_subspace": float(proj_dw),
            "proj_W0_on_random_subspace": float(proj_rand),
            "proj_W0_on_top_r_subspace": float(proj_top),
            "amplification": float(np.linalg.norm(dW, "fro") / max(proj_dw, 1e-12)),
            "dW_top_singular": [float(x) for x in S[:8]],
        })
    return rows


# ---------------------------------------------------------------- main ----

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/kaggle/working/ckpt/base.pt")
    ap.add_argument("--out", default="/kaggle/working/results/03_analysis.json")
    ap.add_argument("--domain", default="sympy")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--r-small", type=int, default=8)
    ap.add_argument("--r-large", type=int, default=64)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    out: dict = {"domain": args.domain, "lr": args.lr, "steps": args.steps,
                 "r_small": args.r_small, "r_large": args.r_large}

    print(f"training r={args.r_small} (seed 1337) ...", flush=True)
    m8, v8 = train_adapter(args.ckpt, args.domain, args.r_small, args.lr,
                           args.steps, args.device, seed=1337)
    print(f"training r={args.r_large} (seed 1337) ...", flush=True)
    m64, v64 = train_adapter(args.ckpt, args.domain, args.r_large, args.lr,
                             args.steps, args.device, seed=1337)
    print(f"training r={args.r_large} (seed 2024) ...", flush=True)
    m64b, v64b = train_adapter(args.ckpt, args.domain, args.r_large, args.lr,
                               args.steps, args.device, seed=2024)
    out["val_loss"] = {"r_small": v8, "r_large": v64, "r_large_seed2": v64b}

    # ---- 7.2 subspace similarity -----------------------------------------
    layers = sorted(m8)
    probe = [layers[0], layers[len(layers) // 2], layers[-1]]
    sim: dict = {}
    for name in probe:
        Ua = right_singular_dirs(m8[name]["A"])
        Ub = right_singular_dirs(m64[name]["A"])
        Ub2 = right_singular_dirs(m64b[name]["A"])
        # Random control: a Gaussian matrix has no privileged directions, so
        # phi against it is the noise floor for these dimensions.
        Ur = right_singular_dirs(rng.standard_normal(m64[name]["A"].shape))
        sim[name] = {
            "phi_r8_vs_r64": np.round(
                subspace_similarity(Ua, Ub, args.r_small, args.r_large), 5).tolist(),
            "phi_r64_seed1_vs_seed2": np.round(
                subspace_similarity(Ub, Ub2, 16, 16), 5).tolist(),
            "phi_r64_vs_random": np.round(
                subspace_similarity(Ub, Ur, 16, 16), 5).tolist(),
        }
    out["subspace_similarity"] = sim
    out["probe_layers"] = probe

    for name in probe:
        d = sim[name]
        print(f"[7.2] {name}: phi(1,1) r8/r64 = {d['phi_r8_vs_r64'][0][0]:.3f} | "
              f"seed1/seed2 = {d['phi_r64_seed1_vs_seed2'][0][0]:.3f} | "
              f"vs random = {d['phi_r64_vs_random'][0][0]:.3f}", flush=True)

    # ---- 7.3 amplification ------------------------------------------------
    out["amplification_r8"] = amplification(m8, rng)
    out["amplification_r64"] = amplification(m64, rng)
    amps = [r["amplification"] for r in out["amplification_r8"]]
    print(f"[7.3] amplification r={args.r_small}: median {np.median(amps):.2f} "
          f"(min {min(amps):.2f}, max {max(amps):.2f})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
