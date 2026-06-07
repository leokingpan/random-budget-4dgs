#!/bin/bash
# Exp 7: Budget Sampling Hyperparameter Sensitivity
# Uses clean HP configs (no model_path/expname/mgs_ratio override)
# 30K iters per variant. 8 variants total. ~2.5 hours.

set -e
cd /home/leo/evo-4dgs/base_4dgs
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D

train_hp() {
    local scene=$1 variant=$2 config=$3 mgs_ratio=$4 budget_min=$5 budget_max=$6
    local expname="dnerf/${scene}_mgs_hp_${variant}"
    fuser -k 6020/tcp 2>/dev/null || true
    echo ""
    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] HP: $scene/$variant  p=$mgs_ratio min=$budget_min max=$budget_max"
    echo "============================================================"
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
        2>&1 | grep -E "(Training progress|Output folder|iter [0-9]+/30|saving|Training complete)" | tail -3
    echo "[$(date '+%H:%M:%S')] ✓ $expname"
}

echo "EXP 7: Budget Sampling Hyperparameter Sensitivity"
echo "Start: $(date '+%H:%M:%S')"
echo ""

# Bouncingballs variants (config: mgs_4dgs_hp.py)
for variant in weak strong vlow lessaggr; do
    case $variant in
        weak)     train_hp bouncingballs weak     configs/mgs_4dgs_hp.py 0.25 0.1  1.0 ;;
        strong)   train_hp bouncingballs strong   configs/mgs_4dgs_hp.py 0.75 0.1  1.0 ;;
        vlow)     train_hp bouncingballs vlow     configs/mgs_4dgs_hp.py 0.5  0.05 1.0 ;;
        lessaggr) train_hp bouncingballs lessaggr configs/mgs_4dgs_hp.py 0.5  0.25 1.0 ;;
    esac
done

# Trex variants (config: mgs_trex_hp.py)
for variant in weak strong vlow lessaggr; do
    case $variant in
        weak)     train_hp trex weak     configs/mgs_trex_hp.py 0.25 0.1  1.0 ;;
        strong)   train_hp trex strong   configs/mgs_trex_hp.py 0.75 0.1  1.0 ;;
        vlow)     train_hp trex vlow     configs/mgs_trex_hp.py 0.5  0.05 1.0 ;;
        lessaggr) train_hp trex lessaggr configs/mgs_trex_hp.py 0.5  0.25 1.0 ;;
    esac
done

echo ""
echo "[$(date '+%H:%M:%S')] EXP 7 ALL COMPLETE"
