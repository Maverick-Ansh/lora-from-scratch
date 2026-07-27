# Reproducing LoRA end to end

A self-contained reproduction of **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** (Hu et al., 2021), using the from-scratch implementation in [`lora/`](lora/).

> **Status: experiments running.** Implementation parity (§1) is complete. The adaptation grids, subspace analysis and serving benchmarks are in flight; this file is updated as each lands.

---

## 0. What is being reproduced, and under what constraint

The experiment machine is a Kaggle backend behind a Colab frontend: **2× Tesla T4, and no network at all** — no `pip`, no HuggingFace, no DNS. That rules out the paper's literal setup (RoBERTa/DeBERTa/GPT-2 on GLUE, WikiSQL, E2E), because there is no checkpoint and no dataset to download.

So the study is made self-contained instead: **pre-train a base model on-box, then adapt it.** That costs the ability to compare absolute numbers against the paper's tables, and buys three things that matter more for testing the paper's *claims*:

- **A genuinely clean held-out domain.** No ambiguity about what leaked into pre-training, because the pre-training corpus is chosen explicitly.
- **Enough runs to be conclusive.** A 25M-parameter model means the rank sweep, the matrix ablation and the LR tuning grid can all be run properly, rather than once.
- **Every claim tested on the same base model**, so cross-experiment comparisons are exact.

The paper's central claims are architecture-agnostic — they are about the *rank of the update*, not about RoBERTa — so they transfer. The claims that do **not** transfer are flagged in §9.

### The corpus

Real Python source already on disk (`/usr/lib/python3.12` plus `dist-packages`), byte-level, vocab 256 — so there is no tokenizer to download either.

| split | contents | size |
|---|---|---|
| pre-train | stdlib, scipy, pandas, sklearn, IPython, numpy, pyspark | 49.8 MB train / 0.8 MB val |
| held-out domain A | `sympy` — symbolic algebra | |
| held-out domain B | `torch` — tensors and autograd | |
| held-out domain C | `matplotlib` — plotting APIs | |

The three adaptation targets are **entirely absent** from pre-training: no symbolic-algebra library, no deep-learning framework, no plotting library. Adapting to one is a real distribution shift within a shared language, which is the setting LoRA is actually used in.

Splits are **by file**, never by byte offset, so a validation snippet is never the continuation of a training snippet.

### The base model

A 25.5M-parameter byte-level GPT: 8 layers, `d_model` 512, 8 heads, block size 256, learned positions, tied embeddings, pre-LN.

Two shape decisions come straight from the experiments:

- **Separate `q_proj` / `k_proj` / `v_proj` / `o_proj`.** Table 5 ablates `Wq`, `Wk`, `Wv`, `Wo` independently, which is impossible with a fused `c_attn` the way GPT-2 and nanoGPT do it. The names match the modern convention, so `target_modules` strings transfer unchanged to Llama-family models.
- **Biases on every projection.** BitFit — bias-only tuning, the strongest small-budget baseline in the paper's Table 2 — does not exist without them.

Pre-training: 15,000 steps, batch 32 × 256 tokens (123M tokens, ≈2.5 passes), AdamW, cosine schedule with 300 warmup steps, peak LR 6e-4, fp16.

> **fp16, not bf16.** A T4 is Turing (sm_75). `torch.cuda.is_bf16_supported()` returns `True`, but there are no bf16 tensor cores, so bf16 is emulated and several times slower. The guard is capability-based, not the flag: `bf16 only if major >= 8`.

---

## 1. Is this implementation actually LoRA? (parity with `peft`)

Before any result means anything, the implementation has to be the same thing the ecosystem runs. Same model built twice, adapted once with [`lora/`](lora/) and once with HuggingFace `peft` 0.19.1, both adapters forced to hold identical numbers:

| check | result |
|---|---|
| scaling factor (`α/r`, α=16, r=8) | 2.0 vs 2.0 |
| max abs logit difference | **0.0** |
| loss | 5.585034370422363 vs 5.585034370422363 |
| max abs gradient difference, `A` | **0.0** |
| max abs gradient difference, `B` | **0.0** |
| merged output vs unmerged reference | 2.98e-07 (fp32 rounding) |

Bit-exact on the forward pass *and* on both gradients.

The gradient check is the one that carries weight. A forward-only comparison at initialisation proves nothing, because `B = 0` makes every correct implementation — and several incorrect ones — agree on a no-op. The gradients pin the backward path, including whether `α/r` is applied where we think it is. `experiments/05_peft_parity.py`, result in [`results/05_peft_parity.json`](results/05_peft_parity.json).

---

## 2. Experimental protocol

**Every arm of every comparison is identical except the thing being ablated.** Fixed across runs: the base checkpoint (reloaded fresh each run — adaptation is destructive), the training batch order, the validation batches, step count, and schedule shape.

Validation batches are drawn from a generator re-seeded to a constant on *every* call, so run A and run B see byte-for-byte the same data. Without that, the 0.01 bits/byte differences that decide a rank sweep are indistinguishable from sampling noise.

**Learning rate is tuned per method family, and this is not a detail.** LoRA and full fine-tuning do not share an optimal LR — the paper itself uses 4e-4 for LoRA and ~1e-5 for full FT on RoBERTa, a 40× difference. Comparing them at one shared LR is not a neutral choice, it is a rigged one, and it is the single most common way published LoRA-vs-full-FT comparisons go wrong. So the `lr` grid sweeps each family independently on one domain (`sympy`) and every later grid uses each family's own best:

- adapter-style methods (LoRA, BitFit, LayerNorm, last-block): `{1e-4, 3e-4, 1e-3, 3e-3}`
- full fine-tuning: `{3e-5, 1e-4, 3e-4, 1e-3}`

Adaptation runs: 800 steps, batch 24 × 256 tokens. Metric is **bits per byte** on the held-out domain's validation split (cross-entropy in nats ÷ ln 2), lower is better.

### The methods compared

| method | what trains |
|---|---|
| no adaptation | nothing — the pre-trained model's loss on the new domain |
| LayerNorm only | all LayerNorm affine parameters |
| BitFit | all biases |
| last block | every parameter of the final transformer block |
| **LoRA** | `lora_A`, `lora_B` on the targeted projections |
| full fine-tune | everything |

`last block` is deliberately included as a control for a confound the paper does not address: does LoRA win because the update is *low-rank*, or merely because *few parameters* train? Last-block tuning trains **more** parameters than LoRA r=8 and concentrates them in one layer. If parameter count were the whole story, it should win.

---

## 3. LoRA against the baselines

*Pending.*

## 4. How much rank do you need? (paper Table 6)

*Pending.*

## 5. Which matrices to adapt? (paper Table 5)

*Pending.*

## 6. Subspace similarity (paper §7.2)

*Pending.*

## 7. What ΔW amplifies (paper §7.3)

*Pending.*

## 8. Serving: latency, merging, hot-swap (paper §1, Table 1)

*Pending.*

## 9. Variants: rsLoRA, DoRA, initialisation, α

*Pending.*

## 10. Limits of this reproduction

Stated up front, so no result here is read as more than it is:

- **No absolute comparison to the paper's tables.** Different model, different scale, different tasks. What transfers is the *shape* of each finding — whether rank matters, which matrices matter, whether merging is free — not the numbers.
- **25M parameters, not 175B.** The paper's strongest claim is that intrinsic rank *falls* as models grow. A single model size cannot test that, and this study does not.
- **Language modelling, not classification.** The paper's Table 2 is GLUE accuracy; this is bits/byte. Bits/byte is a strictly more sensitive metric, which helps, but it is not the same measurement.
- **One seed per configuration** for the grids, with the exception of the subspace analysis (which needs two seeds by construction). Differences smaller than the seed-to-seed spread reported in §6 should not be treated as real.
- **`experiments/glue/` is not run.** The faithful RoBERTa-base GLUE reproduction of Table 2 needs pre-trained weights and a network connection. The script is written and committed so it runs when one is available; it has never executed, and nothing in this report depends on it.

---

## Reproducing this

```bash
git clone https://github.com/Maverick-Ansh/lora-from-scratch
cd lora-from-scratch
pip install -r requirements.txt
pytest tests/ -q                       # 27 tests, one per claim in the paper

python experiments/01_pretrain.py --steps 15000        # ~30 min on one T4
bash run_all.sh                                        # every grid, both GPUs
python tools/figures.py                                # figures from results/
```

On an air-gapped box, `tools/pack.py` and `tools/filesums.py` move the source in through a notebook and verify every file against its repo md5 before anything runs — so the numbers above come from byte-identical code.
