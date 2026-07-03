# Random Budget Training for Variable-Budget Rendering in 4D Gaussian Splatting

### [Paper (TVC)]() | [Code](https://github.com/leokingpan/random-budget-4dgs)

[Guangzi Zhang](https://github.com/leokingpan)<sup>1,2,3,5,6</sup>, [Yulong Pan](https://github.com/leokingpan)<sup>1,2</sup>, [Conglin Ma](https://github.com/leokingpan)<sup>1,2</sup>, [Kaijie Qu](https://github.com/leokingpan)<sup>1,2</sup>, [Xingquan Cai](https://github.com/leokingpan)<sup>1,2,4\*</sup>

<sup>1</sup> Beijing Key Laboratory of Cross-Modal Intelligent Immersive Computing, North China University of Technology, Beijing 100144, China  
<sup>2</sup> School of Artificial Intelligence and Computer Science, North China University of Technology, Beijing 100144, China  
<sup>3</sup> Communication University of China, Beijing 100024, China  
<sup>4</sup> Beijing Institute of Technology, Beijing 100081, China  
<sup>5</sup> School of Animation and Digital Arts, Communication University of China, Beijing 100024, China  
<sup>6</sup> The Kyoto College of Graduate Studies for Informatics, Kyoto, Japan  

<sup>\*</sup> Corresponding author: caixingquan@ncut.edu.cn

---

![overview](assets/teaserfig.jpg)

**TL;DR:** We train a single 4D Gaussian Splatting model with randomly sampled primitive budgets so that at inference time you can pick any budget (10%–100%) and get usable rendering quality — no retraining needed.

## Abstract

4D Gaussian Splatting (4DGS) produces high-quality dynamic novel views but uses a fixed number of primitives, making it hard to adapt to different hardware budgets. We introduce **random budget training**: during training, the model repeatedly renders with randomly sampled primitive subsets ordered by a deformation-aware importance score. At inference, a user simply truncates the sorted Gaussian sequence to the desired budget. On 9 dynamic scenes (7 D-NeRF + 2 NeRF-DS), our method improves the average 25% budget PSNR from 28.53 dB to 30.70 dB over post-hoc pruning, with an average gain of **+2.17 dB**. At 25% budget, D-NeRF scenes reach 33.63 dB PSNR and 131.7 FPS on average. Full-budget quality degrades by only 0.57 dB on average. No extra parameters, no extra memory, one model serves all budgets.

## Installation

Based on [4DGaussians](https://github.com/hustvl/4DGaussians) (CVPR 2024) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

```bash
git clone https://github.com/leokingpan/random-budget-4dgs.git --recursive
cd random-budget-4dgs
conda create -n Gaussians4D python=3.10
conda activate Gaussians4D

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

Tested with PyTorch 2.5.1 + CUDA 12.1 on RTX 4080 SUPER.

## Data Preparation

**D-NeRF (synthetic):** Download from [Dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0) and extract to `data/dnerf/`.

**NeRF-DS (real specular):** Download from the [NeRF-DS repository](https://github.com/zju3dv/NeRF-DS).

```
data/
├── dnerf/
│   ├── bouncingballs/
│   ├── standup/
│   ├── mutant/
│   ├── trex/
│   ├── jumpingjacks/
│   ├── hook/
│   └── hellwarrior/
└── nerfds/
    ├── plate/
    └── press/
```

## Training

### Standard 4DGS (Baseline)
```bash
python train.py -s data/dnerf/bouncingballs --port 6020 \
    --expname "dnerf/bouncingballs_baseline" \
    --configs arguments/dnerf/bouncingballs.py --iterations 30000
```

### Random Budget Training (Ours)
```bash
python train.py -s data/dnerf/bouncingballs --port 6020 \
    --expname "dnerf/bouncingballs_mgs" \
    --configs configs/mgs_4dgs.py --iterations 30000
```

Key training parameters (in `configs/mgs_4dgs.py`):
- `mgs_ratio`: probability of entering budget branch (0.5)
- `budget_min` / `budget_max`: budget sampling range (0.1–1.0)
- `mgs_full_weight`: full-set loss weight (default 1.0)

### Per-Budget Baselines
```bash
# Train dedicated models for 10%, 25%, 50% budgets
python train.py -s data/dnerf/bouncingballs --port 6020 \
    --expname "dnerf/bouncingballs_small10_mgs" \
    --configs configs/mgs_small10_bouncingballs.py --iterations 30000
```

## Rendering & Evaluation

```bash
# Render test views
python render_mgs.py --model_path output/dnerf/bouncingballs_mgs --skip_train

# Evaluate metrics (PSNR, SSIM, LPIPS across budgets)
python eval_all.py --method mgs
```

## Results

### Low-Budget Rendering (25% budget, PSNR ↑)

| Scene | Post-hoc Pruning | Ours | Δ |
|-------|:---:|:---:|:---:|
| bouncingballs | 35.14 | **38.82** | +3.7 |
| standup | 33.70 | **36.09** | +2.4 |
| mutant | 31.90 | **35.10** | +3.2 |
| trex | 31.33 | **32.45** | +1.1 |
| jumpingjacks | 32.71 | **33.67** | +1.0 |
| hook | 29.79 | **31.57** | +1.8 |
| hellwarrior | 26.08 | **28.06** | +2.0 |
| plate (real) | 15.82 | **17.10** | +1.3 |
| press (real) | 20.33 | **23.47** | +3.1 |
| **Average** | 28.53 | **30.70** | **+2.17** |

### Rendering Speed (FPS, 7 D-NeRF avg)

| Budget | 10% | 25% | 50% | 75% | 100% |
|--------|:---:|:---:|:---:|:---:|:---:|
| Avg PSNR | 29.48 | **33.63** | 34.75 | 34.95 | 35.02 |
| Avg FPS | 119.2 | **131.7** | 118.3 | 101.5 | 89.7 |

25% budget gives the highest FPS (1.47× over full budget) while maintaining 33.63 dB PSNR.

## Experiments

Reproduce all paper experiments:

```bash
# Random seed stability (Exp 1)
bash run_seed_stability.sh

# Training & inference overhead (Exp 2)
python measure_overhead.py --checkpoint output/dnerf/bouncingballs_mgs

# Post-hoc pruning + fine-tuning (Exp 3)
python train_posthoc_ft.py --checkpoint output/dnerf/bouncingballs_baseline --budget 0.25 --ft_iters 3000 --expname posthoc_ft_b25

# Temporal consistency (Exp 4)
python temporal_consistency.py --checkpoint output/dnerf/jumpingjacks_mgs --budgets 0.1,0.25

# Appearance-aware ranking (Exp 5)
python render_mgs.py --checkpoint output/nerfds/press_mgs --budgets 0.1,0.25,0.5,1.0 --ordering appearance_aware

# Static vs adaptive ordering (Exp 6)
python render_mgs.py --checkpoint output/dnerf/jumpingjacks_mgs --budgets 0.1,0.25 --static

# Runtime budget switching (Exp 7)
python run_budget_switch.py --checkpoint output/dnerf/jumpingjacks_mgs

# Generate paper figures
python gen_paper_figures.py

# Compile all results
python synthesize_all.py
```

> **Note on seed stability**: The seed-stability experiment was retrained on RTX 3090 for revision. The main comparison tables use the original RTX 4080 checkpoints. Therefore, the absolute PSNR values in the seed-stability table should not be directly compared with the main tables; the focus is the standard deviation across seeds.

## Method Overview

```
Training:
  For each iteration (50% prob enter budget branch):
    1. HexPlane deformation → deformed Gaussians
    2. Compute importance: I(G_i,t) = α̃_i(t) · s̃_{i,x}(t) · s̃_{i,y}(t)
    3. Sort by importance, select top-K prefix
    4. Render prefix → L_budget (prefix-only L1 + D-SSIM)
    5. Also render full set → densification statistics only

Inference:
  1. Static ordering at t=0 (Kendall τ = 1.0, verified temporally stable)
  2. Truncate to target budget ratio r
  3. Render ⌊rN⌋ primitives
```

## Citation

```bibtex
@article{zhang2026randombudget,
  title     = {Random Budget Training for Variable-Budget Rendering in 4D Gaussian Splatting},
  author    = {Zhang, Guangzi and Pan, Yulong and Ma, Conglin and Qu, Kaijie and Cai, Xingquan},
  journal   = {The Visual Computer (under review)},
  year      = {2026},
}
```

## Acknowledgement

This project builds on [4DGaussians](https://github.com/hustvl/4DGaussians) (Wu et al., CVPR 2024) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Kerbl et al., SIGGRAPH 2023). We thank the authors for their excellent work and open-source release.

The random budget training idea is inspired by [Matryoshka Gaussian Splatting](https://arxiv.org/abs/2603.19234) (Guo et al., 2026) and [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) (Kusupati et al., NeurIPS 2022).

## License

This project is licensed under the LICENSE.md file. See the original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [4DGaussians](https://github.com/hustvl/4DGaussians) repositories for their respective licenses.
