# Checkpoints

## Availability

Trained checkpoints are not stored in this Git repository due to file size constraints. The code, configuration files, and evaluation scripts in this repository are fully public and sufficient to reproduce all reported experiments from scratch.

## Release Plan

Selected checkpoints will be released through a permanent archival repository (Zenodo or Hugging Face) with a DOI before publication or upon acceptance. The following checkpoints will be included:

- All 9 scene MGS models (original RTX 4080 training)
- All 9 scene baseline models (standard 4DGS)
- Post-hoc fine-tuning checkpoints
- Seed-stability checkpoints (RTX 3090 retrained)

## Using Existing Checkpoints

If you have access to pre-trained checkpoints, place them under `output/` following this structure:

```
output/
├── dnerf/
│   ├── bouncingballs_baseline/point_cloud/iteration_30000/
│   ├── bouncingballs_mgs/point_cloud/iteration_30000/
│   ├── trex_baseline/point_cloud/iteration_30000/
│   ├── trex_mgs/point_cloud/iteration_30000/
│   └── ...
└── nerfds/
    ├── press_mgs/point_cloud/iteration_15000/
    └── plate_mgs/point_cloud/iteration_15000/
```

Each checkpoint directory must contain:
- `cfg_args` -- training configuration (auto-saved during training)
- `point_cloud/iteration_{N}/point_cloud.ply` -- Gaussian parameters
- `point_cloud/iteration_{N}/deformation.pth` -- HexPlane deformation network weights
- `point_cloud/iteration_{N}/deformation_table.pth` -- deformation activation table
