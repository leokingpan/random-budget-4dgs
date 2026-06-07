#!/bin/bash
# Exp 6: Multi-Seed Stability
# 3 scenes × 3 seeds, budgets 10%/25%/100% evaluation
# Uses existing checkpoints with different random seeds in render_mgs.py
set -e
cd /home/leo/evo-4dgs/base_4dgs
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D

FRAMES=20
BUDGETS="0.1,0.25,1.0"

run_seed() {
    local scene=$1
    local ckpt=$2
    local seed=$3
    local iters=${4:-30000}
    local output="results/mgs/exp6_${scene}_seed${seed}.csv"
    echo "[$(date '+%H:%M:%S')] Multi-seed: $scene seed=$seed"
    python -u render_mgs.py \
        --checkpoint "$ckpt" \
        --budgets "$BUDGETS" \
        --frames $FRAMES \
        --ordering ours \
        --iter $iters \
        --output "$output" 2>&1 | grep -E "(AVERAGE|SUMMARY|budget=)"
    echo "[$(date '+%H:%M:%S')] $output done"
}

# Scene 1: bouncingballs (already have seeds 42, 123, 2024 in error_bars/)
for seed in 42 123 2024; do
    run_seed "bouncingballs" "output/dnerf/bouncingballs_mgs" $seed
done

# Scene 2: trex
for seed in 42 123 2024; do
    run_seed "trex" "output/dnerf/trex_mgs" $seed
done

# Scene 3: press
for seed in 42 123 2024; do
    run_seed "press" "output/nerfds/press_mgs" $seed 15000
done

echo "[$(date '+%H:%M:%S')] EXP 6 COMPLETE"
