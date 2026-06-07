#!/bin/bash
# Quick evaluations: press opacity variant + appearance_aware for press
set -e
FRAMES=20
BUDGETS="0.1,0.25,0.5,0.75,1.0"
cd /home/leo/evo-4dgs/base_4dgs
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D

# Press opacity variant (missing from Exp 2)
echo "[$(date '+%H:%M:%S')] Press opacity"
python -u render_mgs.py \
    --checkpoint output/nerfds/press_mgs \
    --budgets "$BUDGETS" \
    --frames $FRAMES \
    --ordering opacity \
    --iter 15000 \
    --output results/mgs/exp2_press_opacity.csv

# Press appearance_aware (Exp 3 - key experiment)
echo "[$(date '+%H:%M:%S')] Press appearance_aware"
python -u render_mgs.py \
    --checkpoint output/nerfds/press_mgs \
    --budgets "$BUDGETS" \
    --frames $FRAMES \
    --ordering appearance_aware \
    --iter 15000 \
    --output results/mgs/exp3_press_appearance_aware.csv

echo "[$(date '+%H:%M:%S')] DONE"
