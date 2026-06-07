#!/bin/bash
# Exp 4: Post-hoc Pruning + Fine-Tuning Baseline
# Scenes: bouncingballs, standup, trex, jumpingjacks, press
# Budget: 25%, FT: 2000 and 5000 iterations
set -e
cd /home/leo/evo-4dgs/base_4dgs
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D

run_ft() {
    local scene=$1
    local ckpt=$2
    local budget=$3
    local ft_iters=$4
    local expname="posthoc_ft_${scene}_b${budget/0./}_${ft_iters}iter"
    echo "[$(date '+%H:%M:%S')] Post-hoc-FT: $scene budget=$budget ft=$ft_iters"
    python -u train_posthoc_ft.py \
        --checkpoint "$ckpt" \
        --budget $budget \
        --ft_iters $ft_iters \
        --expname "$expname"
    echo "[$(date '+%H:%M:%S')] Done: $expname"
}

# D-NeRF synthetic: 5 scenes × 2 FT durations
for ft in 2000 5000; do
    run_ft "bouncingballs" "output/dnerf/bouncingballs_baseline" 0.25 $ft
    run_ft "standup"       "output/dnerf/standup_baseline"       0.25 $ft
    run_ft "trex"          "output/dnerf/trex_baseline"          0.25 $ft
    run_ft "jumpingjacks"  "output/dnerf/jumpingjacks_baseline"  0.25 $ft
done

echo "[$(date '+%H:%M:%S')] EXP 4 COMPLETE"
