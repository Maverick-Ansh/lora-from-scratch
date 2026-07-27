#!/bin/bash
# Full experiment pipeline, both T4s, unattended.
# No `set -e`: one failing grid should not cost us the other four.
cd /kaggle/working/lora-from-scratch
export PYTHONUNBUFFERED=1
L=/kaggle/working/logs

run_grid () {
  g=$1
  echo "=== grid $g start $(date +%T) ==="
  CUDA_VISIBLE_DEVICES=0 python -u experiments/02_grids.py --grid $g --shard 0 --nshards 2 --device cuda:0 > $L/grid_${g}_0.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python -u experiments/02_grids.py --grid $g --shard 1 --nshards 2 --device cuda:0 > $L/grid_${g}_1.log 2>&1 &
  wait
  python -u experiments/09_report.py --grids $g
  echo "=== grid $g done $(date +%T) ==="
}

# The LR grids must finish first: every later grid reads the per-family best LR
# they write, so that no comparison is decided by a learning rate that happened
# to suit one method. `lrx` extends the adapter families upward, because the
# first pass put all four of their optima at the top of the swept range.
run_grid lr
run_grid lrx
run_grid methods
run_grid rank
run_grid matrix
run_grid variants

# The subspace analysis must train its adapters at the SAME tuned LR the rest
# of the study uses, or its two ranks are not comparable to the rank sweep.
LORA_LR=$(python - <<'PY'
import json
recs = []
for f in ("/kaggle/working/results/02_lr_merged.json",
          "/kaggle/working/results/02_lrx_merged.json"):
    try:
        recs += json.load(open(f))
    except Exception:
        pass
c = [r for r in recs if r["family"] == "lora"]
print(min(c, key=lambda r: r["best_val"])["lr"] if c else 1e-3)
PY
)
echo "=== analysis (lora lr=$LORA_LR) + serving $(date +%T) ==="
CUDA_VISIBLE_DEVICES=0 python -u experiments/03_analysis.py --device cuda:0 --lr $LORA_LR > $L/analysis.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u experiments/04_serving.py --device cuda:0 > $L/serving.log 2>&1 &
wait
echo "=== ALL DONE $(date +%T) ==="
