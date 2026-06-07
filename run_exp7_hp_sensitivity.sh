#!/bin/bash
# Exp 7: Budget Sampling Hyperparameter Sensitivity
# 2 scenes (bouncingballs, trex) × 5 config variants × 30K iters
# Variants: weak(0.25), default(0.5), strong(0.75), very_low(0.05), less_aggr(0.25min)
set -e
cd /home/leo/evo-4dgs/base_4dgs
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D

# Kill any lingering port
fuser -k 6020/tcp 2>/dev/null || true

train_variant() {
    local scene=$1
    local variant=$2
    local mgs_ratio=$3
    local budget_min=$4
    local budget_max=$5
    local config_file="configs/mgs_${scene}.py"
    local expname="dnerf/${scene}_mgs_hp_${variant}"
    
    echo "[$(date '+%H:%M:%S')] Training: $scene variant=$variant mgs_ratio=$mgs_ratio min=$budget_min max=$budget_max"
    
    TQDM_DISABLE=1 python -u train.py \
        -s "data/dnerf/${scene}" \
        --port 6020 \
        --expname "$expname" \
        --configs "$config_file" \
        --iterations 30000 \
        --mgs_enabled \
        --mgs_ratio $mgs_ratio \
        --budget_min $budget_min \
        --budget_max $budget_max \
        2>&1 | tail -5
    
    echo "[$(date '+%H:%M:%S')] Done: $expname"
}

for scene in bouncingballs trex; do
    # Variant 1: Low budget exposure weak (p_mgs=0.25)
    train_variant $scene "weak" 0.25 0.1 1.0
    
    # Variant 2: Default (p_mgs=0.5) - already trained, SKIP unless force
    # train_variant $scene "default" 0.5 0.1 1.0
    
    # Variant 3: Low budget exposure strong (p_mgs=0.75)
    train_variant $scene "strong" 0.75 0.1 1.0
    
    # Variant 4: Very low budget (ρ_min=0.05)
    train_variant $scene "vlow" 0.5 0.05 1.0
    
    # Variant 5: Less aggressive (ρ_min=0.25)
    train_variant $scene "lessaggr" 0.5 0.25 1.0
done

echo "[$(date '+%H:%M:%S')] EXP 7 COMPLETE"
