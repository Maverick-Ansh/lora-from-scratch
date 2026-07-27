"""Stage 0: pre-train the base model that everything else adapts.

There is no downloadable checkpoint on this machine, so we make one.  The
corpus is general-purpose Python (stdlib, scipy, pandas, sklearn, IPython);
the three domains we later adapt to -- sympy, torch, matplotlib -- are held
out entirely, so adaptation is a real distribution shift rather than a
refresher on data already memorised.

    python experiments/01_pretrain.py --steps 15000 --out /kaggle/working/ckpt
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab.data import PRETRAIN_ROOTS, DOMAIN_ROOTS, cached_corpus
from lab.model import GPT, GPTConfig
from lab.train import TrainConfig, evaluate, save_json, train, bits_per_byte


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/kaggle/working/ckpt")
    ap.add_argument("--results", default="/kaggle/working/results")
    args = ap.parse_args()

    print("building corpus ...", flush=True)
    t0 = time.time()
    corpus = cached_corpus("pretrain", PRETRAIN_ROOTS, val_frac=0.02)
    print(f"{corpus}  ({time.time()-t0:.0f}s)", flush=True)

    cfg = GPTConfig(block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, d_model=args.d_model)
    model = GPT(cfg)
    print(f"model: {model.num_params():,} params", flush=True)

    tcfg = TrainConfig(steps=args.steps, batch_size=args.batch_size,
                       block_size=args.block_size, lr=args.lr, warmup=300,
                       eval_every=1000, eval_batches=50, log_every=250)
    hist = train(model, corpus.train, corpus.val, tcfg, device=args.device, tag="pretrain")

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "base.pt")
    torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, ckpt_path)
    print(f"[saved] {ckpt_path}", flush=True)

    # Zero-shot loss on every held-out domain: the baseline every adaptation
    # method has to beat, and the number that says how big the shift really is.
    zero_shot = {}
    for name, roots in DOMAIN_ROOTS.items():
        c = cached_corpus(name, roots, val_frac=0.05, max_bytes=40_000_000)
        v = evaluate(model, c.val, tcfg, args.device)
        zero_shot[name] = {"val_loss": v, "bits_per_byte": bits_per_byte(v)}
        print(f"[zero-shot] {name:12s} {v:.4f} nats  {bits_per_byte(v):.4f} bpb", flush=True)
    hist["zero_shot"] = zero_shot
    hist["n_params"] = model.num_params()

    # A qualitative sanity check that the base model learned Python at all.
    model.eval()
    prompt = torch.tensor([[ord(c) for c in "def compute("]], dtype=torch.long,
                          device=args.device)
    out = model.generate(prompt, max_new_tokens=200, temperature=0.7)
    sample = bytes(out[0].tolist()).decode("ascii", errors="replace")
    hist["sample"] = sample
    print("---- sample ----\n" + sample + "\n----------------", flush=True)

    save_json(hist, os.path.join(args.results, "01_pretrain.json"))


if __name__ == "__main__":
    main()
