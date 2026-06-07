#!/bin/bash
# Exp 2 recovery: run projected_area + remaining canonical_opacity evaluations
set -e
FRAMES=20
BUDGETS="0.1,0.25"
cd /home/leo/evo-4dgs/base_4dgs
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D

run_ablation() {
    local scene=$1
    local checkpoint=$2
    local ordering=$3
    local output=$4
    local iters=${5:-30000}
    echo "[$(date '+%H:%M:%S')] $scene | $ordering"
    python -u render_mgs.py \
        --checkpoint "$checkpoint" \
        --budgets "$BUDGETS" \
        --frames $FRAMES \
        --ordering "$ordering" \
        --iter $iters \
        --output "$output"
}

# bouncingballs - retry projected_area
run_ablation "bb_projected" "output/dnerf/bouncingballs_mgs" "projected_area" "results/mgs/exp2_bouncingballs_projected_area.csv"

# standup
run_ablation "su_canonical" "output/dnerf/standup_mgs" "canonical_opacity" "results/mgs/exp2_standup_canonical_opacity.csv"
run_ablation "su_projected" "output/dnerf/standup_mgs" "projected_area" "results/mgs/exp2_standup_projected_area.csv"

# trex
run_ablation "tr_canonical" "output/dnerf/trex_mgs" "canonical_opacity" "results/mgs/exp2_trex_canonical_opacity.csv"
run_ablation "tr_projected" "output/dnerf/trex_mgs" "projected_area" "results/mgs/exp2_trex_projected_area.csv"

# jumpingjacks
run_ablation "jj_canonical" "output/dnerf/jumpingjacks_mgs" "canonical_opacity" "results/mgs/exp2_jumpingjacks_canonical_opacity.csv"
run_ablation "jj_opacity" "output/dnerf/jumpingjacks_mgs" "opacity" "results/mgs/exp2_jumpingjacks_opacity.csv"
run_ablation "jj_projected" "output/dnerf/jumpingjacks_mgs" "projected_area" "results/mgs/exp2_jumpingjacks_projected_area.csv"

# press
run_ablation "pr_canonical" "output/nerfds/press_mgs" "canonical_opacity" "results/mgs/exp2_press_canonical_opacity.csv" 15000
run_ablation "pr_projected" "output/nerfds/press_mgs" "projected_area" "results/mgs/exp2_press_projected_area.csv" 15000

echo "[$(date '+%H:%M:%S')] EXP 2 RECOVERY COMPLETE"
