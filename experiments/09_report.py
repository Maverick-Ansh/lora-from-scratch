"""Merge shard results, print the tables, and emit a compact JSON for the repo.

The experiment box has no network, so results leave it as text through the
notebook.  Training curves are the bulk of the bytes and none of the argument,
so ``--compact`` drops them; the full records stay on the box for as long as it
lives.
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

CURVE_KEYS = ("val_curve", "train_curve")


def merge_shards(results_dir: str, grid: str) -> list[dict]:
    recs: list[dict] = []
    for f in sorted(glob.glob(os.path.join(results_dir, f"02_{grid}_shard*.json"))):
        recs.extend(json.load(open(f)))
    if recs:
        out = os.path.join(results_dir, f"02_{grid}_merged.json")
        json.dump(recs, open(out, "w"), indent=2)
        print(f"[merged] {grid}: {len(recs)} records -> {out}", flush=True)
    return recs


def table(recs: list[dict], sort_key="best_bpb") -> None:
    if not recs:
        print("  (no records)")
        return
    by_domain = defaultdict(list)
    for r in recs:
        by_domain[r["domain"]].append(r)
    for domain, rows in by_domain.items():
        rows.sort(key=lambda r: r.get(sort_key, math.inf))
        print(f"\n  domain = {domain}")
        print(f"  {'method':30s} {'lr':>9s} {'trainable':>11s} {'%':>8s} {'bpb':>9s}")
        print("  " + "-" * 72)
        for r in rows:
            print(f"  {r['name']:30s} {r['lr']:>9.0e} {r['trainable']:>11,} "
                  f"{r['trainable_pct']:>7.3f}% {r['best_bpb']:>9.4f}")


def compact(recs: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in CURVE_KEYS} for r in recs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/kaggle/working/results")
    ap.add_argument("--grids", nargs="*",
                    default=["lr", "methods", "rank", "matrix", "variants"])
    ap.add_argument("--dump", action="store_true", help="print compact JSON for copying")
    args = ap.parse_args()

    everything: dict = {}
    for grid in args.grids:
        recs = merge_shards(args.results, grid)
        if recs:
            print(f"\n=== grid: {grid} ===")
            table(recs)
            everything[grid] = compact(recs)

    if args.dump:
        print("\n===BEGIN_COMPACT_JSON===")
        print(json.dumps(everything, separators=(",", ":"), sort_keys=True))
        print("===END_COMPACT_JSON===")


if __name__ == "__main__":
    main()
