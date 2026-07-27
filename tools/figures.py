"""Render every figure in REPORT.md from results/*.json.

Figures are built off-box: the experiment machine has no network, so it emits
JSON through the notebook and this script turns that into PNGs locally.  That
split also means a figure can be restyled without spending GPU time.

    python tools/figures.py            # writes figures/*.png

Palette and chart rules follow the project's data-viz conventions: a fixed
categorical slot order (never cycled), one y-axis per chart, a legend whenever
there are two or more series, direct value labels, and recessive chrome.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Categorical slots, in fixed order. Validated: worst all-pairs CVD dE 9.2,
# normal-vision 24.0 on the light surface for the first three.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Sequential blue ramp, light -> dark (steps 100..700 of the blue scale).
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

PRETTY = {
    "zero_shot": "no adaptation", "full_ft": "full fine-tune", "bitfit": "BitFit (biases)",
    "layernorm": "LayerNorm only", "last_block": "last block", "lora_qv_r8": "LoRA r=8 (Wq,Wv)",
}


def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=1.0, axis="y")
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    return ax


def legend(ax, **kw):
    lg = ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, **kw)
    return lg


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[fig] {path.relative_to(ROOT)}")


def load(name):
    p = RESULTS / name
    if not p.exists():
        print(f"[skip] {name} not found")
        return None
    return json.loads(p.read_text())


# ------------------------------------------------------------- figures ----

def fig_methods(data):
    """LoRA vs the baselines, as improvement over doing nothing.

    Absolute bits/byte differs a lot between domains, which would make grouped
    bars unreadable; the *reduction* relative to the unadapted model is the
    comparable quantity and is what the comparison is actually about.
    """
    recs = data.get("methods")
    if not recs:
        return
    domains = sorted({r["domain"] for r in recs})
    zero = {d: next(r["best_bpb"] for r in recs if r["domain"] == d and r["name"] == "zero_shot")
            for d in domains}
    # Ordered by trainable-parameter budget, so the x-axis reads as "spend
    # more" and LoRA's position relative to last-block is visible at a glance.
    methods = [m for m in ["layernorm", "bitfit", "lora_qv_r8", "last_block", "full_ft"]
               if any(r["name"] == m for r in recs)]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    n = len(domains)
    width = 0.8 / n
    for si, d in enumerate(domains):
        vals, labels = [], []
        for m in methods:
            r = next((r for r in recs if r["domain"] == d and r["name"] == m), None)
            vals.append(zero[d] - r["best_bpb"] if r else 0.0)
            labels.append(r["trainable"] if r else 0)
        xs = [i + si * width - 0.4 + width / 2 for i in range(len(methods))]
        ax.bar(xs, vals, width=width * 0.88, color=SERIES[si], label=d, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.004, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK_2)

    pct = {r["name"]: r["trainable_pct"] for r in recs}
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([f"{PRETTY.get(m, m)}\n{pct.get(m, 0):.2f}% of params"
                        for m in methods], fontsize=9, color=INK_2)
    style(ax, "LoRA against the baselines",
          ylabel="bits/byte recovered vs no adaptation  (higher is better)")
    legend(ax, loc="upper left", ncol=3, title=None)
    save(fig, "methods.png")


def fig_rank(data):
    """Paper Table 6: bits/byte as a function of rank."""
    recs = data.get("rank")
    if not recs:
        return
    domains = sorted({r["domain"] for r in recs})
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for si, d in enumerate(domains):
        rows = sorted((r for r in recs if r["domain"] == d), key=lambda r: r["r"])
        xs = [r["r"] for r in rows]
        ys = [r["best_bpb"] for r in rows]
        ax.plot(xs, ys, "-o", color=SERIES[si], linewidth=2, markersize=7,
                label=d, zorder=3, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{ys[0]:.3f}", (xs[0], ys[0]), textcoords="offset points",
                    xytext=(0, 10), fontsize=8, color=INK_2, ha="center")
        ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(0, 10), fontsize=8, color=INK_2, ha="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    # Deliberately NOT titled with the paper's conclusion: here rank keeps
    # paying, which is the opposite of Table 6, and the title should say what
    # the data says.
    style(ax, "Rank keeps paying here — the opposite of the paper's Table 6",
          xlabel="LoRA rank r  (Wq, Wv; adapter size grows 64x left to right)",
          ylabel="bits/byte on held-out domain  (lower is better)")
    legend(ax, loc="best")
    save(fig, "rank_sweep.png")


def fig_matrix(data):
    """Paper Table 5: which matrices, at a matched parameter budget."""
    recs = [r for r in data.get("matrix", []) if r["name"] != "matrix_all_linear_r2"]
    if not recs:
        return
    domains = sorted({r["domain"] for r in recs})
    names = [r["name"] for r in recs if r["domain"] == domains[0]]
    pretty = [n.replace("matrix_", "").replace("_", "  ") for n in names]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    n = len(domains)
    height = 0.8 / n
    for si, d in enumerate(domains):
        vals = [next(r["best_bpb"] for r in recs if r["domain"] == d and r["name"] == nm)
                for nm in names]
        ys = [i + si * height - 0.4 + height / 2 for i in range(len(names))]
        ax.barh(ys, vals, height=height * 0.88, color=SERIES[si], label=d, zorder=3)
        for y, v in zip(ys, vals):
            ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(pretty, fontsize=9, color=INK_2)
    ax.invert_yaxis()
    ax.grid(True, color=GRID, linewidth=1.0, axis="x")
    ax.grid(False, axis="y")
    style(ax, "Which matrices to adapt, at equal parameter budget (paper Table 5)",
          xlabel="bits/byte on held-out domain  (lower is better)")
    ax.set_xlim(left=min(r["best_bpb"] for r in recs) - 0.02)
    legend(ax, loc="lower right")
    save(fig, "matrix_ablation.png")


def fig_subspace(data):
    """Paper 7.2: how much subspace two adapters share."""
    if not data:
        return
    sim = data["subspace_similarity"]
    layer = data["probe_layers"][len(data["probe_layers"]) // 2]
    panels = [
        ("phi_r8_vs_r64", f"r={data['r_small']} vs r={data['r_large']}\n(same seed)"),
        ("phi_r64_seed1_vs_seed2", f"r={data['r_large']}, seed 1337 vs 2024"),
        ("phi_r64_vs_random", f"r={data['r_large']} vs random Gaussian"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for ax, (key, title) in zip(axes, panels):
        m = sim[layer][key]
        im = ax.imshow(m, cmap=BLUE_RAMP, vmin=0, vmax=1, aspect="auto", origin="upper")
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
        ax.set_xlabel("j", color=INK_2, fontsize=9)
        ax.set_ylabel("i", color=INK_2, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("subspace similarity  phi(i, j)", color=INK_2, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    cbar.outline.set_visible(False)
    fig.suptitle(f"Only the top direction is shared  —  {layer}",
                 color=INK, fontsize=12, x=0.125, ha="left", fontweight="bold")
    save(fig, "subspace_similarity.png")


def fig_amplification(data):
    """Paper 7.3: dW's subspace is neither random nor W0's dominant one."""
    if not data:
        return
    rows = data["amplification_r8"]
    xs = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    series = [
        ("proj_W0_on_top_r_subspace", "W0's own top-r subspace"),
        ("proj_W0_on_dW_subspace", "the subspace dW chose"),
        ("proj_W0_on_random_subspace", "a random r-dim subspace"),
    ]
    for si, (key, label) in enumerate(series):
        ax.plot(xs, [r[key] for r in rows], "-o", color=SERIES[si], linewidth=2,
                markersize=6, label=label, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["layer"].replace("blocks.", "L").replace(".attn.", " ")
                        .replace("_proj", "") for r in rows],
                       rotation=45, ha="right", fontsize=7.5, color=INK_2)
    style(ax, "How much of W0 lives in the directions the adapter picked (paper 7.3)",
          ylabel="||U.T W0 V||_F   (log scale)")
    legend(ax, loc="best")
    save(fig, "amplification.png")


def fig_latency(data):
    """Paper Section 1 / Table 1: the zero-latency claim, measured."""
    if not data:
        return
    rows = data["latency"]
    xs = [r["batch"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    series = [
        ("lora_merged_overhead_pct", "LoRA, merged"),
        ("lora_unmerged_overhead_pct", "LoRA, unmerged"),
        ("adapter_overhead_pct", "bottleneck adapter"),
    ]
    for si, (key, label) in enumerate(series):
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, "-o", color=SERIES[si], linewidth=2, markersize=7, label=label,
                zorder=3, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{ys[-1]:+.1f}%", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), fontsize=8, color=INK_2, va="center")
    ax.axhline(0, color=BASELINE, linewidth=1.5, zorder=2)
    ax.text(xs[0], 0.4, "unadapted base model", fontsize=8, color=MUTED, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    style(ax, "Merging really does cost nothing",
          xlabel="batch size (sequence length 256)",
          ylabel="forward latency overhead vs base  (%)")
    legend(ax, loc="upper left")
    save(fig, "latency.png")


def main():
    grids = load("02_grids.json") or {}
    fig_methods(grids)
    fig_rank(grids)
    fig_matrix(grids)
    fig_subspace(load("03_analysis.json"))
    fig_amplification(load("03_analysis.json"))
    fig_latency(load("04_serving.json"))


if __name__ == "__main__":
    main()
