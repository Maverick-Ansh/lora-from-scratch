"""RoBERTa-base + LoRA on GLUE -- the paper's actual Table 2.

    STATUS: WRITTEN, NEVER EXECUTED.

This is the faithful reproduction: the paper's model, the paper's tasks, the
paper's hyperparameters (Appendix D.1).  It is not part of REPORT.md's results
and nothing in that report depends on it, because the machine this study ran on
has no network -- no pre-trained RoBERTa to download, no GLUE to download.  The
script is committed so the reproduction is one command away from anyone who has
a connection, not because it produced anything here.

Run it with:

    pip install transformers datasets scikit-learn
    python experiments/glue/run_glue.py --task mrpc --seed 42

What it reproduces (Table 2, RoBERTa-base, LoRA row, 0.3M trainable):

    MNLI 87.5   SST-2 95.1   MRPC 89.7   CoLA 63.4
    QNLI 93.3   QQP   90.8   RTE  86.6   STS-B 91.5

Paper hyperparameters, Appendix D.1: AdamW, linear schedule, warmup ratio 0.06,
LoRA r=8 on Wq and Wv, alpha=8, max sequence length 512, batch 16-32, LR
4e-4-5e-4, 25-80 epochs depending on task size.  The small tasks (MRPC, RTE,
STS-B, CoLA) are the affordable ones; MNLI and QQP are ~400k examples.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lora import LoRAConfig, apply_lora, count_parameters, summarize

# task -> (sentence keys, num_labels, metric)
TASKS = {
    "cola": (("sentence", None), 2, "mcc"),
    "sst2": (("sentence", None), 2, "acc"),
    "mrpc": (("sentence1", "sentence2"), 2, "acc_f1"),
    "stsb": (("sentence1", "sentence2"), 1, "corr"),
    "rte": (("sentence1", "sentence2"), 2, "acc"),
    "qnli": (("question", "sentence"), 2, "acc"),
    "qqp": (("question1", "question2"), 2, "acc_f1"),
    "mnli": (("premise", "hypothesis"), 3, "acc"),
}

# Appendix D.1 -- epochs are task-size dependent; these follow the paper's ranges.
EPOCHS = {"cola": 20, "sst2": 60, "mrpc": 30, "stsb": 40,
          "rte": 80, "qnli": 25, "qqp": 25, "mnli": 30}
BATCH = {"cola": 32, "sst2": 16, "mrpc": 16, "stsb": 16,
         "rte": 32, "qnli": 32, "qqp": 16, "mnli": 16}
LR = {"cola": 4e-4, "sst2": 5e-4, "mrpc": 4e-4, "stsb": 4e-4,
      "rte": 5e-4, "qnli": 4e-4, "qqp": 5e-4, "mnli": 5e-4}


def metric_fn(kind):
    from sklearn.metrics import f1_score, matthews_corrcoef
    from scipy.stats import pearsonr, spearmanr

    def fn(preds, labels):
        if kind == "acc":
            return {"acc": float((preds == labels).mean())}
        if kind == "acc_f1":
            return {"acc": float((preds == labels).mean()),
                    "f1": float(f1_score(labels, preds))}
        if kind == "mcc":
            return {"mcc": float(matthews_corrcoef(labels, preds))}
        if kind == "corr":
            return {"pearson": float(pearsonr(preds, labels)[0]),
                    "spearman": float(spearmanr(preds, labels)[0])}
        raise ValueError(kind)
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mrpc", choices=sorted(TASKS))
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--head-lr", type=float, default=None,
                    help="LR for the freshly-initialised classifier head")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full-ft", action="store_true", help="baseline: tune everything")
    ap.add_argument("--out", default="results/glue")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    keys, n_labels, metric_kind = TASKS[args.task]
    epochs = args.epochs or EPOCHS[args.task]
    batch_size = args.batch_size or BATCH[args.task]
    lr = args.lr or LR[args.task]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    raw = load_dataset("glue", args.task)

    def encode(batch):
        text = (batch[keys[0]],) if keys[1] is None else (batch[keys[0]], batch[keys[1]])
        return tok(*text, truncation=True, max_length=args.max_length, padding="max_length")

    ds = raw.map(encode, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    ds.set_format("torch", columns=cols)
    val_split = "validation_matched" if args.task == "mnli" else "validation"
    train_dl = DataLoader(ds["train"], batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(ds[val_split], batch_size=batch_size * 2)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=n_labels,
        problem_type="regression" if n_labels == 1 else "single_label_classification",
    ).to(device)

    if not args.full_ft:
        # RoBERTa attention projections are named query/key/value; the paper
        # adapts Wq and Wv only.
        apply_lora(model, LoRAConfig(target_modules=("query", "value"),
                                     r=args.r, alpha=args.alpha), verbose=True)
        # The classification head is randomly initialised -- it has to train,
        # or there is nothing to read the frozen features. The paper counts it
        # separately from the 0.3M adapter parameters.
        for n, p in model.named_parameters():
            if "classifier" in n:
                p.requires_grad = True
        print(summarize(model))
    print(f"trainable: {count_parameters(model)['trainable']:,}")

    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() > 1]
    nodecay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() <= 1]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": nodecay, "weight_decay": 0.0}],
        lr=lr)
    total = len(train_dl) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * args.warmup_ratio), total)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best: dict = {}
    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            if n_labels == 1:
                labels = labels.float()
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                out = model(**batch, labels=labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(out.loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

        model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("label")
                with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                    logits = model(**batch).logits
                p = logits.squeeze(-1) if n_labels == 1 else logits.argmax(-1)
                preds.append(p.float().cpu().numpy())
                golds.append(labels.cpu().numpy())
        scores = metric_fn(metric_kind)(np.concatenate(preds), np.concatenate(golds))
        primary = list(scores.values())[0]
        if not best or primary > list(best.values())[0]:
            best = scores
        print(f"epoch {epoch+1}/{epochs}  {scores}  best={best}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.task}_{'full' if args.full_ft else 'lora'}.json")
    json.dump({"task": args.task, "best": best, "epochs": epochs, "lr": lr,
               "batch_size": batch_size, "r": args.r, "alpha": args.alpha,
               "seed": args.seed, "full_ft": args.full_ft,
               "trainable": count_parameters(model)["trainable"]},
              open(path, "w"), indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
