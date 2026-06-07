#!/bin/bash
# Exp 7: Budget Sampling Hyperparameter Sensitivity
# 2 scenes × 4 new variants (default already exists) × 30K iters
# Variants: weak(0.25), strong(0.75), very_low(0.05), less_aggr(0.25min)
# Estimated: ~2.5 hours total

set -e
cd /home/leo/evo-4dgs/base_4dgs
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D

declare -A CONFIG_MAP
CONFIG_MAP[bouncingballs]="configs/mgs_4dgs.py"
CONFIG_MAP[trex]="configs/mgs_trex.py"

train_variant() {
    local scene=$1 variant=$2 mgs_ratio=$3 budget_min=$4 budget_max=$5
    local config="${CONFIG_MAP[$scene]}"
    local expname="dnerf/${scene}_mgs_hp_${variant}"

    fuser -k 6020/tcp 2>/dev/null || true
    echo "[$(date '+%H:%M:%S')] HP: $scene variant=$variant p=$mgs_ratio ρ_min=$budget_min ρ_max=$budget_max"
    TQDM_DISABLE=1 python -u train.py \
        -s "data/dnerf/${scene}" \
        --port 6020 \
        --expname "$expname" \
        --configs "$config" \
        --iterations 30000 \
        --mgs_enabled \
        --mgs_ratio $mgs_ratio \
        --budget_min $budget_min \
        --budget_max $budget_max \
        2>&1 | grep -E "(Training|iter [0-9]+00/|loss=|PSNR|saved|COMPLETED)" | tail -5
    echo "[$(date '+%H:%M:%S')] Done: $expname"
}

for scene in bouncingballs trex; do
    echo "=== Scene: $scene ==="
    train_variant $scene "weak"     0.25 0.1  1.0
    train_variant $scene "strong"   0.75 0.1  1.0
    train_variant $scene "vlow"     0.5  0.05 1.0
    train_variant $scene "lessaggr" 0.5  0.25 1.0
done

echo "[$(date '+%H:%M:%S')] EXP 7 ALL DONE"
