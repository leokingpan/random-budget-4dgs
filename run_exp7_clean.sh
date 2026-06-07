#!/bin/bash
# Exp 7: HP Sensitivity — Clean sequential execution, no grep filter
set -e
cd /home/leo/evo-4dgs/base_4dgs
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D

train_one() {
    local scene=$1 variant=$2 config=$3 ratio=$4 bmin=$5 bmax=$6
    local name="dnerf/${scene}_mgs_hp_${variant}"
    fuser -k 6020/tcp 2>/dev/null || true
    echo ""
    echo "===== $(date '+%H:%M:%S') HP: $scene/$variant p=$ratio min=$bmin max=$bmax ====="
    TQDM_DISABLE=1 python -u train.py \
        -s "data/dnerf/${scene}" --port 6020 --expname "$name" \
        --configs "$config" --iterations 30000 \
        --mgs_enabled --mgs_ratio $ratio --budget_min $bmin --budget_max $bmax
    echo "===== $(date '+%H:%M:%S') DONE: $name ====="
}

LOG="/tmp/exp7_$(date +%H%M).log"
echo "Log: $LOG"

# Bouncingballs: 4 variants (default p=0.5 already exists)
train_one bouncingballs weak     configs/mgs_4dgs_hp.py 0.25 0.1  1.0
train_one bouncingballs strong   configs/mgs_4dgs_hp.py 0.75 0.1  1.0
train_one bouncingballs vlow     configs/mgs_4dgs_hp.py 0.5  0.05 1.0
train_one bouncingballs lessaggr configs/mgs_4dgs_hp.py 0.5  0.25 1.0

# Trex: 4 variants
train_one trex weak     configs/mgs_trex_hp.py 0.25 0.1  1.0
train_one trex strong   configs/mgs_trex_hp.py 0.75 0.1  1.0
train_one trex vlow     configs/mgs_trex_hp.py 0.5  0.05 1.0
train_one trex lessaggr configs/mgs_trex_hp.py 0.5  0.25 1.0

echo ""
echo "===== $(date '+%H:%M:%S') EXP 7 ALL DONE ====="
