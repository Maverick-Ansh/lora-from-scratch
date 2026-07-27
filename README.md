# LoRA from scratch

A from-first-principles reimplementation of **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** (Hu et al., 2021) — the library, the experiments that test the paper's claims, and an account of what the method turned into by 2026.

No `peft`, no `loralib`. Every line of the adapter is in [`lora/`](lora/), in plain PyTorch, under 700 lines including comments. `peft` appears exactly once, in a parity test, to prove the from-scratch version is numerically identical to the one the ecosystem actually runs.

> **Status:** library complete and tested (27/27), and **bit-exact against `peft`** — identical logits *and* identical gradients. Experiments in progress; see **[REPORT.md](REPORT.md)** for results as they land.

### Headline result so far

Adapting a 25.5M-parameter base model to three domains it has never seen (`sympy`, `torch`, `matplotlib`), every method at its own tuned learning rate:

![LoRA against the baselines](figures/methods.png)

**LoRA recovers 54–62% of full fine-tuning's gain while training 0.51% of the parameters** — 14× more gain per trainable parameter than the next-best method. It also *loses* to full fine-tuning, and to simply tuning the last block, which trains 24× more parameters. Both facts are reported, because the second one is what the control was built to detect: it separates "low rank helps" from "few parameters helps". See [§3](REPORT.md#3-lora-against-the-baselines).

---

## The idea in one equation

A pre-trained weight `W0` is frozen. The *update* you would have learned by fine-tuning is constrained to be low rank:

```
h = W0 @ x  +  ΔW @ x  =  W0 @ x  +  (α/r) · B @ A @ x
```

with `A ∈ ℝ^(r×k)`, `B ∈ ℝ^(d×r)`, and `r ≪ min(d, k)`.

The hypothesis behind it: fine-tuning a big model does not need a full-rank update, because the *change* required to specialise a general model is intrinsically low-dimensional — even when the model itself is not.

Three consequences, and every design choice in this repo exists to preserve one of them:

| Property | Why it holds | Where it is tested |
|---|---|---|
| Training starts as a perfect no-op | `B = 0` at init ⇒ `ΔW = 0` | `test_zero_init_makes_adapter_a_no_op` |
| Inference costs **nothing** extra | `ΔW` is the same shape as `W0`, so fold it in | `test_merge_preserves_output_...` |
| The artifact you ship is tiny | `r(d+k)` params instead of `d·k` | `test_state_dict_contains_only_adapter...` |

That second one is the part people underrate. Adapter layers add *depth*, which you can never remove and which shows up as latency on every token forever. LoRA adds a *parallel* branch, and a parallel branch is just addition — so it can be summed away before you ever serve it.

## Install

```bash
git clone https://github.com/Maverick-Ansh/lora-from-scratch
cd lora-from-scratch
pip install -r requirements.txt
pytest tests/ -q
```

## Use

```python
import torch.nn as nn
from lora import LoRAConfig, apply_lora, merge_lora, summarize, lora_state_dict

model = my_pretrained_transformer()

apply_lora(model, LoRAConfig(target_modules=["q_proj", "v_proj"], r=8, alpha=8))
print(summarize(model))
# total=10,927,872  trainable=73,728 (0.6747%)  adapted_modules=12  adapter=0.28 MB (fp32)

train(model)                      # only lora_A / lora_B get gradients

torch.save(lora_state_dict(model), "task.adapter")   # ship this, not the model
merge_lora(model)                                    # zero-overhead inference
```

`target_modules` matches an exact dotted name, a leaf name, or an fnmatch glob — so `["query", "value"]` hits every attention Q/V projection in a BERT-family model and `["*.layer.11.*"]` hits only the last block.

## What is in here

```
lora/
  layers.py     LoRALinear, LoRAEmbedding — the maths, merge/unmerge, fan_in_fan_out
  inject.py     LoRAConfig + apply_lora: find modules, swap them, freeze the rest
  utils.py      parameter accounting, adapter-only checkpoints
  variants.py   DoRA; rsLoRA is a flag on LoRALinear
tests/          27 tests, each pinning one claim from the paper
lab/            experiment scaffolding: a small GPT, a corpus, training loops
experiments/    the reproductions (see REPORT.md)
```

## Details that matter, and are easy to get wrong

- **`B = 0` is load-bearing, `A`'s init is not.** Because `∂L/∂A = scaling · Bᵀ · δ`, and `B` is exactly zero at step 0, **`A` receives no gradient on the first step at all.** `B` moves first; `A` only starts learning once `B` is non-zero. This is why the method is robust to how you initialise `A` and would break immediately if you initialised `B` randomly (`ΔW ≠ 0` ⇒ you have corrupted the pre-trained model before seeing a single batch). Pinned by `test_A_receives_no_gradient_on_the_very_first_step`.
- **The paper and the reference implementation disagree about `A`.** The text says "random Gaussian initialization for `A`"; [`microsoft/LoRA`](https://github.com/microsoft/LoRA) ships `kaiming_uniform_(a=√5)`. Both are supported here via `init_a=`, and measured in `experiments/`.
- **`α/r` is a *learning-rate* device, not a modelling one.** Tying the scale to `r` means re-tuning `r` does not force you to re-tune the LR. rsLoRA argues the variance-preserving exponent is ½, not 1 — supported as `use_rslora=True`.
- **`fan_in_fan_out` silently destroys training if you get it wrong.** GPT-2's `Conv1D` stores weights as `(in, out)`. For square attention projections the shapes still line up, so there is no error — just a model that does not learn.
- **Never form `B @ A` in the forward pass.** Two skinny matmuls (`x → r → out`) are what make training cheap; materialising the `(d, k)` product costs as much as the frozen layer.

## Reproductions

See **[REPORT.md](REPORT.md)** for the full write-up. Ablations follow the paper's own structure:

| Paper | Claim under test |
|---|---|
| Table 5 | Which matrices to adapt (`Wq, Wk, Wv, Wo`) at a fixed parameter budget |
| Table 6 | Rank `r ∈ {1,2,4,8,64}` — "a rank as small as one suffices" |
| §7.2 | Subspace similarity between adapters trained at different `r` (Grassmann) |
| §7.3 | `ΔW` amplifies directions already in `W0` but not emphasised by it |
| §1 | Merged inference has no latency penalty |

## License

MIT
