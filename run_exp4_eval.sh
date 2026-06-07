#!/bin/bash
# Evaluate post-hoc+FT checkpoints at budget=1.0 (= 25% of original)
set -e
cd /home/leo/evo-4dgs/base_4dgs
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D
mkdir -p results/mgs

for scene in bouncingballs standup trex jumpingjacks; do
    ckpt="output/dnerf/posthoc_ft_${scene}_b25_2000iter"
    output="results/mgs/exp4_${scene}_posthoc_ft_2k.csv"
    echo "[$(date '+%H:%M:%S')] Evaluating: $scene post-hoc+FT 2K"
    python -u render_mgs.py \
        --checkpoint "$ckpt" \
        --budgets "1.0" \
        --frames 20 \
        --ordering ours \
        --iter 2000 \
        --output "$output" 2>&1 | grep -E "(PSNR|SUMM|Done)"
done
echo "[$(date '+%H:%M:%S')] EXP 4 EVALUATION COMPLETE"
