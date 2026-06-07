#!/bin/bash
# Exp 1: Static ordering evaluations + missing our/random evaluations
# Run from base_4dgs/
set -e

FRAMES=20
BUDGETS="0.1,0.25,0.5,0.75,1.0"
cd /home/leo/evo-4dgs/base_4dgs
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D

mkdir -p results/mgs

run_eval() {
    local scene=$1
    local checkpoint=$2
    local ordering=$3
    local static_flag=$4
    local output=$5
    local iters=${6:-30000}
    
    echo "========================================="
    echo "[$(date '+%H:%M:%S')] $scene | $ordering $static_flag"
    echo "========================================="
    
    python -u render_mgs.py \
        --checkpoint "$checkpoint" \
        --budgets "$BUDGETS" \
        --frames $FRAMES \
        --ordering "$ordering" \
        $static_flag \
        --iter $iters \
        --output "$output"
    
    echo "[$(date '+%H:%M:%S')] Done: $output"
}

# ====== Exp 1: Static ordering (6 missing scenes) ======
run_eval "trex_static" "output/dnerf/trex_mgs" "ours" "--static" "results/mgs/trex_static.csv"
run_eval "hellwarrior_static" "output/dnerf/hellwarrior_mgs" "ours" "--static" "results/mgs/hellwarrior_static.csv"
run_eval "hook_static" "output/dnerf/hook_mgs" "ours" "--static" "results/mgs/hook_static.csv"
run_eval "mutant_static" "output/dnerf/mutant_mgs" "ours" "--static" "results/mgs/mutant_static.csv"
run_eval "plate_static" "output/nerfds/plate_mgs" "ours" "--static" "results/mgs/nerfds_plate_static.csv"
run_eval "press_static" "output/nerfds/press_mgs" "ours" "--static" "results/mgs/nerfds_press_static.csv" 15000

# ====== Missing "ours" evaluations ======
run_eval "standup_ours" "output/dnerf/standup_mgs" "ours" "" "results/mgs/standup_ours.csv"
run_eval "mutant_ours" "output/dnerf/mutant_mgs" "ours" "" "results/mgs/mutant_ours.csv"

# ====== Missing "random" evaluations ======
run_eval "jumpingjacks_random" "output/dnerf/jumpingjacks_mgs" "random" "" "results/mgs/jumpingjacks_random.csv"
run_eval "trex_random" "output/dnerf/trex_mgs" "random" "" "results/mgs/trex_random.csv"
run_eval "hellwarrior_random" "output/dnerf/hellwarrior_mgs" "random" "" "results/mgs/hellwarrior_random.csv"
run_eval "hook_random" "output/dnerf/hook_mgs" "random" "" "results/mgs/hook_random.csv"

# ====== Exp 5 prep: FPS data for plate ======
run_eval "plate_ours" "output/nerfds/plate_mgs" "ours" "" "results/mgs/nerfds_plate_ours.csv"
run_eval "plate_posthoc" "output/nerfds/plate_mgs" "posthoc_opacity" "" "results/mgs/nerfds_plate_posthoc.csv"
run_eval "plate_random" "output/nerfds/plate_mgs" "random" "" "results/mgs/nerfds_plate_random.csv"

# ====== Exp 3 prep: press random evaluation ======
run_eval "press_random" "output/nerfds/press_mgs" "random" "" "results/mgs/nerfds_press_random.csv" 15000

echo ""
echo "========================================="
echo "[$(date '+%H:%M:%S')] ALL EVALUATIONS COMPLETE"
echo "========================================="
