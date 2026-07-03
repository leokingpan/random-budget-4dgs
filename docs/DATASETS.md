# Datasets

## D-NeRF (Synthetic Dynamic Scenes)

- **Source**: [Dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0) (official release)
- **Paper**: Pumarola et al., "D-NeRF: Neural Radiance Fields for Dynamic Scenes", CVPR 2021
- **Scenes**: bouncingballs, hellwarrior, hook, jumpingjacks, lego, mutant, standup, trex
- **Format**: transforms_train.json + transforms_test.json + PNG images (RGBA)

## NeRF-DS (Real Specular Dynamic Scenes)

- **Source**: [GitHub](https://github.com/zju3dv/NeRF-DS) (official repository)
- **Paper**: Yan et al., "NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects", CVPR 2023
- **Scenes**: plate, press
- **Format**: dataset.json + PNG images

## Expected Directory Structure

```
data/
├── dnerf/
│   ├── bouncingballs/
│   │   ├── transforms_train.json
│   │   ├── transforms_test.json
│   │   ├── train/
│   │   │   └── r_*.png
│   │   └── test/
│   │       └── r_*.png
│   ├── hellwarrior/
│   ├── hook/
│   ├── jumpingjacks/
│   ├── lego/
│   ├── mutant/
│   ├── standup/
│   └── trex/
└── nerfds/
    ├── plate_novel_view/
    │   ├── dataset.json
    │   └── images/
    └── press_novel_view/
        ├── dataset.json
        └── images/
```

## Usage Notes

- D-NeRF images are RGBA PNGs; the training pipeline handles alpha blending with white background
- NeRF-DS images use a `dataset.json` format with `camera_angle_x` and per-frame `transform_matrix`
- The `source_path` argument should point to the scene directory (e.g., `data/dnerf/bouncingballs` or `data/nerfds/press_novel_view`)
