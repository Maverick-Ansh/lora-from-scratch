# LoRA in 2026: what the 2021 paper turned into

The paper is five years old. The equation did not change — `h = W0·x + (α/r)·B·A·x` is still exactly what runs — but almost every *default* around it did, and the reasons are worth knowing, because most of them are corrections to something the original paper got approximately right rather than exactly right.

This document is the "how is it actually used now" companion to the reproduction in [REPORT.md](../REPORT.md). Where this repo measures something directly, it says so and links; where it is reporting the field's consensus, it says that instead.

---

## 1. The single biggest change: adapt everything, not just `Wq` and `Wv`

The paper's Table 5 concludes that adapting `Wq` and `Wv` is the best use of a fixed budget, and for years `target_modules=["q_proj","v_proj"]` was the default everywhere.

Current practice is **all linear layers**, MLP included — `q,k,v,o,gate,up,down`. Two things drove the change:

- Once QLoRA made the base model nearly free to hold in memory, the *budget* constraint that made Table 5 interesting stopped binding. Table 5 answers "given N parameters, where do I spend them?" If N is no longer scarce, the question dissolves.
- The QLoRA paper reported that adapting all linear layers was what let LoRA match full fine-tuning quality; with attention-only adapters, a gap remained.

`peft` still defaults to attention-only for many architectures, but `target_modules="all-linear"` is the common override, and most published recipes use it.

> Measured here: `experiments/02_grids.py --grid matrix` runs both the paper's budget-matched attention ablation and an all-linear arm. See [REPORT.md](../REPORT.md).

## 2. `α/r` is a learning-rate knob, and rank is smaller than people expect

Two folk rules dominate practice:

- **`α = 2r`.** Not from the paper (which uses `α = r` for RoBERTa). It is a heuristic that stuck because it works, and what it really does is hold the *effective* step size roughly constant while letting you change `r`.
- **`r = 8` to `r = 64`.** Higher ranks rarely pay for themselves on instruction-tuning-shaped tasks, which is exactly the paper's own Table 6 finding, still holding up.

**rsLoRA** (Kalajdzievski, 2023) is the principled correction. With `α/r`, the variance of the adapter's output *shrinks* as `r` grows, so large-`r` runs are silently under-trained — which is a large part of why "higher rank doesn't help" was reported so consistently. Scaling by `α/√r` instead keeps the variance stable, and large ranks then behave like you would expect. It is one line (`use_rslora=True` here, same flag name in `peft`).

The implication is subtle and worth stating plainly: *"rank doesn't matter" and "the α/r scaling under-trains high ranks" predict the same observation.* Distinguishing them requires re-running the rank sweep under both scalings, which is what the `variants` grid does.

## 3. QLoRA: the reason any of this is on consumer hardware

[QLoRA](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023) is the single most consequential follow-up. The base model is frozen anyway, so it does not need to be in 16 bits:

- **NF4** — a 4-bit data type that is information-theoretically optimal for normally-distributed weights, which is what trained weights approximately are.
- **Double quantization** — quantizing the quantization constants, saving ~0.4 bits/parameter.
- **Paged optimizers** — NVIDIA unified memory to survive gradient-checkpointing spikes.

Adapters stay in bf16 and the forward de-quantizes on the fly. A 65B model fine-tunes on a single 48 GB card; a 7B fits comfortably on a free Colab T4.

This is *orthogonal* to everything in `lora/` — it changes how `W0` is stored, not what `B·A` means — which is exactly why it composes so cleanly. It is also why the modern default is "adapt all the linear layers": the memory you freed goes back into adapter coverage.

## 4. DoRA, and the magnitude/direction split

[DoRA](https://arxiv.org/abs/2402.09353) (Liu et al., 2024) starts from an observation about *full* fine-tuning: decompose a weight into magnitude and direction, and full FT moves them fairly independently, while LoRA's magnitude and direction changes stay tightly coupled. DoRA gives magnitude its own parameter vector:

```
W' = m · (W0 + ΔW) / ‖W0 + ΔW‖_row
```

It generally beats LoRA at equal rank, especially at *low* rank. The cost is a norm over the merged weight on every forward, which is why it trains noticeably slower for an almost identical parameter count — and why plain LoRA remains the default. Implemented here in [`lora/variants.py`](../lora/variants.py) and measured in the `variants` grid.

## 5. Serving: the part the paper predicted correctly

Section 1's claim — one frozen base, many hot-swappable adapters, no added latency — turned into real infrastructure:

- **S-LoRA / Punica** demonstrated thousands of adapters on one GPU, with custom kernels that batch requests using *different* adapters together.
- **vLLM** ships multi-LoRA serving in production.

The trick these need is the one thing the merge story cannot give you: if you merge, you get zero latency but one task per copy of the weights. To serve many tasks in one batch you must keep adapters **unmerged** and pay a small, batched gather. So the deployment choice is real:

| | merged | unmerged |
|---|---|---|
| latency | identical to base | small overhead |
| tasks per resident model | one | many |
| swap cost | re-merge (or reload `W0`) | swap two small tensors |

> Measured here: `experiments/04_serving.py` times base / merged / unmerged against a bottleneck adapter, and includes the fp16 caveat below.

**The fp16 merge caveat.** `merge` then `unmerge` is exactly lossless only in fp32. In fp16 the add/subtract pair loses low-order bits, so a serving stack that repeatedly merges and unmerges an fp16 model will drift. Real systems keep a pristine `W0` instead of trusting `unmerge` — a detail the paper had no reason to mention and every implementer eventually hits.

## 6. Where LoRA is now the default, and where it is not

**Default:** instruction tuning; preference optimization (DPO/GRPO/PPO adapters, often with a second frozen adapter as the KL reference); domain adaptation; and — overwhelmingly — image generation, where "a LoRA" has become a *noun* for a downloadable style.

**Not the default:** anything that has to install genuinely new knowledge. The most useful corrective here is ["LoRA Learns Less and Forgets Less"](https://arxiv.org/abs/2405.09673) (Biderman et al., 2024): on continued pre-training in a new domain (code, in their case), LoRA measurably *underperforms* full fine-tuning — and, symmetrically, *forgets* less of the base model's original abilities. Both halves follow from the same constraint. A low-rank update cannot move the weights very far, which is a limitation when you want distance and a feature when you do not.

That result is the honest boundary of the method, and it is the reason "just use LoRA" is bad advice for continued pre-training and good advice for task adaptation. It is also the finding this repo's own setup is closest to — adapting a base model to a genuinely held-out domain is a small-scale version of exactly that experiment, so [REPORT.md](../REPORT.md) reports the LoRA-vs-full-FT gap rather than burying it.

## 7. A default recipe

For task adaptation of a pre-trained LLM, in the absence of a reason to do otherwise:

| knob | value | why |
|---|---|---|
| `target_modules` | all linear | §1 |
| `r` | 16 (8 if adapter size matters, 64 if the shift is large) | §2 |
| `α` | `2r` | §2 |
| `use_rslora` | `True` if `r > 16` | §2 |
| `lora_dropout` | 0.0–0.05 | small data → nonzero |
| learning rate | 1e-4 – 3e-4 | ~10× a full-FT LR; tune it, and tune it *separately* from full FT's |
| base precision | NF4 if memory-bound, bf16 otherwise | §3 |
| serving | merge for one task, keep unmerged for many | §5 |

The learning-rate row is the one people get wrong most often, and it is the one that most corrupts LoRA-vs-full-FT comparisons: a shared LR is not a fair comparison, it is a rigged one. This repo tunes each method family's LR independently before comparing anything, for exactly that reason.

---

## References

- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 2021
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), 2023
- Kalajdzievski, [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732), 2023
- Liu et al., [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353), 2024
- Biderman et al., [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673), 2024
- Sheng et al., [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285), 2023
- Houlsby et al., [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751), 2019 — the bottleneck adapters LoRA is measured against
