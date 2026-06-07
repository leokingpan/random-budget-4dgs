#!/bin/bash
# Exp 2: Importance score ablation — 5 score variants on 5 scenes
# 1. Random (下界)
# 2. canonical_opacity (α_i, static MGS style)
# 3. opacity (α̃_i(t), deformed opacity only)
# 4. opacity_area (α̃·s̃_x·s̃_y, current method)
# 5. projected_area (α̃·projected 2D covariance area)
#
# Scenes: bouncingballs, standup, trex, jumpingjacks, press
# Budgets: 0.1, 0.25 (key low-budget comparison points)
set -e

FRAMES=20
BUDGETS="0.1,0.25"
cd /home/leo/evo-4dgs/base_4dgs
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D

mkdir -p results/mgs

run_ablation() {
    local tag=$1
    local checkpoint=$2
    local ordering=$3
    local output=$4
    local iters=${5:-30000}
    
    echo "========================================="
    echo "[$(date '+%H:%M:%S')] $tag | $ordering"
    echo "========================================="
    
    time python -u render_mgs.py \
        --checkpoint "$checkpoint" \
        --budgets "$BUDGETS" \
        --frames $FRAMES \
        --ordering "$ordering" \
        --iter $iters \
        --output "$output" 2>&1
    
    echo "[$(date '+%H:%M:%S')] Done: $output"
}

# ====== bouncingballs (5 variants) ======
run_ablation "bb_random" "output/dnerf/bouncingballs_mgs" "random" "results/mgs/exp2_bouncingballs_random.csv"
run_ablation "bb_canonical" "output/dnerf/bouncingballs_mgs" "canonical_opacity" "results/mgs/exp2_bouncingballs_canonical_opacity.csv"
run_ablation "bb_opacity" "output/dnerf/bouncingballs_mgs" "opacity" "results/mgs/exp2_bouncingballs_opacity.csv"
run_ablation "bb_opacity_area" "output/dnerf/bouncingballs_mgs" "opacity_area" "results/mgs/exp2_bouncingballs_opacity_area.csv"
run_ablation "bb_projected" "output/dnerf/bouncingballs_mgs" "projected_area" "results/mgs/exp2_bouncingballs_projected_area.csv"

# ====== standup (2 new + 3 existing) ======
run_ablation "su_canonical" "output/dnerf/standup_mgs" "canonical_opacity" "results/mgs/exp2_standup_canonical_opacity.csv"
run_ablation "su_projected" "output/dnerf/standup_mgs" "projected_area" "results/mgs/exp2_standup_projected_area.csv"

# ====== trex (2 new + 3 existing) ======
run_ablation "tr_canonical" "output/dnerf/trex_mgs" "canonical_opacity" "results/mgs/exp2_trex_canonical_opacity.csv"
run_ablation "tr_projected" "output/dnerf/trex_mgs" "projected_area" "results/mgs/exp2_trex_projected_area.csv"

# ====== jumpingjacks (3 new + random has) ======
run_ablation "jj_canonical" "output/dnerf/jumpingjacks_mgs" "canonical_opacity" "results/mgs/exp2_jumpingjacks_canonical_opacity.csv"
run_ablation "jj_opacity" "output/dnerf/jumpingjacks_mgs" "opacity" "results/mgs/exp2_jumpingjacks_opacity.csv"
run_ablation "jj_projected" "output/dnerf/jumpingjacks_mgs" "projected_area" "results/mgs/exp2_jumpingjacks_projected_area.csv"

# ====== press (2 new) ======
run_ablation "pr_canonical" "output/nerfds/press_mgs" "canonical_opacity" "results/mgs/exp2_press_canonical_opacity.csv" 15000
run_ablation "pr_projected" "output/nerfds/press_mgs" "projected_area" "results/mgs/exp2_press_projected_area.csv" 15000

echo ""
echo "========================================="
echo "[$(date '+%H:%M:%S')] EXP 2 COMPLETE"
echo "========================================="
