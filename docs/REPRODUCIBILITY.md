# Reproducibility

## Hardware & Software

| Item | Specification |
|------|--------------|
| GPU (original) | NVIDIA RTX 4080 SUPER 16GB |
| GPU (revision) | NVIDIA RTX 3090 24GB |
| OS (original) | Windows 10 22H2 |
| OS (revision) | WSL2 Ubuntu 22.04 |
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| CUDA Toolkit | 12.1.105 |

## Training Configuration

| Parameter | D-NeRF | NeRF-DS |
|-----------|--------|---------|
| Iterations | 30,000 | 15,000 |
| Coarse iterations | 3,000 | 2,000 |
| Learning rates | configs/mgs_4dgs.py | configs/mgs_nerfds_press.py |
| mgs_ratio | 0.5 | 0.5 |
| budget_min / budget_max | 0.1 / 1.0 | 0.1 / 1.0 |
| mgs_full_weight | 1.0 | 1.0 |
| ordering_strategy | opacity_area | opacity_area |

## Random Seeds

| Experiment | Seeds | Notes |
|-----------|-------|-------|
| Main comparison tables | 6666 (default) | Original RTX 4080 checkpoints |
| Seed stability (bouncingballs) | 0, 1, 2 | Retrained on RTX 3090 for revision |
| Seed stability (trex) | 0, 1, 2 | Retrained on RTX 3090 for revision |
| Seed stability (press) | 0, 1, 2 | Retrained on RTX 3090 for revision |
| Post-hoc fine-tuning | 42 (default) | train_posthoc_ft.py |

## Which Tables Use Which Checkpoints

| Table | Checkpoint Source | Hardware |
|-------|------------------|----------|
| Main comparison (Ours vs post-hoc) | bouncingballs_mgs, trex_mgs, etc. | RTX 4080 (original) |
| Quality-FPS trade-off | Same as main | RTX 4080 |
| Seed stability | *_mgs_seed{0,1,2} | RTX 3090 (retrained) |
| Training/deployment overhead | Original training logs | Estimated |
| Post-hoc pruning + FT | *_baseline (pruned) + RTX 3090 FT | Mixed |
| Temporal consistency | Original MGS checkpoints | RTX 4080 |
| Appearance-aware ranking | Same as main (inference only) | RTX 4080 |
| Static vs adaptive ordering | Same as main (inference only) | RTX 4080 |
| Runtime budget switching | Same as main (inference only) | RTX 4080 |

## Reproducing Results

1. Install environment: `conda env create -f environment.yml`
2. Download datasets (see [DATASETS.md](DATASETS.md))
3. Train baselines: `python train.py -s data/dnerf/{scene} --configs arguments/dnerf/{scene}.py --expname "dnerf/{scene}_baseline" --iterations 30000`
4. Train MGS: `python train.py -s data/dnerf/{scene} --configs configs/mgs_{scene}.py --iterations 30000`
5. Evaluate: `python render_mgs.py --checkpoint output/dnerf/{scene}_mgs --budgets 0.1,0.25,0.5,0.75,1.0`
6. Run specific experiments using the commands listed in [README.md](../README.md#experiments)
