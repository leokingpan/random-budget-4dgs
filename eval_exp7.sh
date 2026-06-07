#!/bin/bash
# Evaluate all Exp 7 HP checkpoints
cd /home/leo/evo-4dgs/base_4dgs
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D
mkdir -p results/mgs

BUDGETS="0.1,0.25,1.0"
FRAMES=20

eval_one() {
    local scene=$1 variant=$2 iters=$3
    local ckpt="output/dnerf/${scene}_mgs_hp_${variant}"
    local out="results/mgs/exp7_${scene}_${variant}.csv"
    echo "[$(date '+%H:%M:%S')] $scene/$variant"
    python -u render_mgs.py --checkpoint "$ckpt" --budgets "$BUDGETS" \
        --frames $FRAMES --ordering ours --iter $iters --output "$out" 2>&1 | grep "budget="
}

# Bouncingballs (30000 iter)
for v in weak strong vlow lessaggr; do
    eval_one bouncingballs $v 30000
done

# Trex (30000 iter)
for v in weak strong vlow lessaggr; do
    eval_one trex $v 30000
done

echo ""
echo "=== RESULTS ==="
for f in results/mgs/exp7_*.csv; do
    echo "--- $(basename $f) ---"
    cat "$f"
done
echo "[$(date '+%H:%M:%S')] EXP 7 EVAL DONE"
