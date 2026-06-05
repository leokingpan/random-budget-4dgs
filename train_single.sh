#!/bin/bash
# Train MGS × 4DGS for a single scene
# Usage: bash train_single.sh <scene_name> [iterations]

SCENE=$1
ITERS=${2:-30000}

source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D
cd /home/leo/evo-4dgs/base_4dgs

case $SCENE in
  bouncingballs) PORT=6020 ;;   trex) PORT=6021 ;;
  hellwarrior)   PORT=6022 ;;   mutant) PORT=6023 ;;
  standup)       PORT=6024 ;;   hook) PORT=6025 ;;
  jumpingjacks)  PORT=6027 ;;   lego) PORT=6028 ;;
  *) PORT=6039 ;;
esac

fuser -k ${PORT}/tcp 2>/dev/null || true

echo "=== Training $SCENE (port $PORT, $ITERS iters) ==="
echo "Started at $(date)"

TQDM_DISABLE=1 python -u train.py \
  --configs configs/mgs_${SCENE}.py \
  --iterations $ITERS \
  2>&1 | tee /tmp/train_${SCENE}_v2.log

echo "=== Finished at $(date) ==="
