#!/bin/bash
# Auto-collect metrics when all 8 baseline scenes complete
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D
cd /home/leo/evo-4dgs/base_4dgs

echo "===== AUTO METRICS COLLECTOR: $(date) ====="
echo "Waiting for all 8 scenes to complete..."

SCENES=(bouncingballs hellwarrior hook jumpingjacks lego mutant standup trex)
TOTAL=8
MAX_WAIT=300  # 5 minute max wait between checks
CHECKS=72     # 6 hours total max

for ((i=0; i<$CHECKS; i++)); do
    DONE=0
    for scene in "${SCENES[@]}"; do
        # Check for rendered test outputs (reliable completion signal)
        if [ -d "output/dnerf/${scene}_baseline/test/ours_-1/renders" ]; then
            DONE=$((DONE + 1))
        fi
    done
    
    if [ $DONE -eq $TOTAL ]; then
        echo "All $TOTAL scenes rendered!"
        break
    fi
    
    echo "[$(date +%H:%M:%S)] $DONE/$TOTAL done, waiting ${MAX_WAIT}s..."
    sleep $MAX_WAIT
done

echo "===== RUNNING METRICS: $(date) ====="
python collect_metrics.py 2>&1 | tee .omo/evidence/task-7-baseline-metrics.txt

echo "" >> .omo/evidence/task-7-baseline-metrics.txt
echo "===== CHECKPOINT EVIDENCE: $(date) =====" >> .omo/evidence/task-7-baseline-metrics.txt
for scene in "${SCENES[@]}"; do
    p="output/dnerf/${scene}_baseline"
    echo "--- $scene ---" >> .omo/evidence/task-7-baseline-metrics.txt
    if [ -d "$p" ]; then
        ls -la "$p/" >> .omo/evidence/task-7-baseline-metrics.txt 2>&1
        if [ -f "$p/chkpnt_fine_30000.pth" ]; then
            echo "  [OK] Checkpoint present" >> .omo/evidence/task-7-baseline-metrics.txt
        fi
        if [ -d "$p/test" ]; then
            echo "  [OK] test/ renders present" >> .omo/evidence/task-7-baseline-metrics.txt
        fi
    else
        echo "  MISSING" >> .omo/evidence/task-7-baseline-metrics.txt
    fi
    echo "" >> .omo/evidence/task-7-baseline-metrics.txt
done

echo "===== AUTO COLLECT COMPLETE: $(date) ====="
