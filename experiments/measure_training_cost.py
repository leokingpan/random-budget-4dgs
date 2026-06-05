#!/usr/bin/env python3
"""
测量 MGS 训练相对于 Baseline 训练的时间与显存开销。

通过对比 100 次迭代的平均时间来估算全训练（30000 次）的总时间，
同时记录峰值 GPU 显存。

用法:
    python experiments/measure_training_cost.py --config configs/mgs_bouncingballs.py
"""
import os, sys, time, argparse, json
import torch
import numpy as np

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from utils.mgs_utils import compute_importance, get_sorted_indices, sample_budget
from random import randint

WARMUP_ITERS = 10
MEASURE_ITERS = 100


def parse_config(config_path):
    """解析 config.py，返回 ModelParams, OptimizationParams, PipelineParams 命名空间。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    # Build namespace
    model_kwargs = dict(sh_degree=getattr(cfg, 'sh_degree', 3),
                        source_path=cfg.source_path,
                        model_path=cfg.model_path,
                        images=cfg.images,
                        resolution=getattr(cfg, 'resolution', -1),
                        white_background=cfg.white_background,
                        data_device=cfg.data_device,
                        eval=getattr(cfg, 'eval', True))
    lp = ModelParams(**model_kwargs)

    op_kwargs = dict(iterations=cfg.iterations,
                     position_lr_init=cfg.position_lr_init,
                     position_lr_final=cfg.position_lr_final,
                     deformation_lr_init=cfg.deformation_lr_init,
                     deformation_lr_final=cfg.deformation_lr_final,
                     grid_lr_init=cfg.grid_lr_init,
                     grid_lr_final=cfg.grid_lr_final,
                     feature_lr=cfg.feature_lr,
                     opacity_lr=cfg.opacity_lr,
                     scaling_lr=cfg.scaling_lr,
                     rotation_lr=cfg.rotation_lr,
                     percent_dense=cfg.percent_dense,
                     lambda_dssim=cfg.lambda_dssim,
                     densification_interval=cfg.densification_interval,
                     densify_from_iter=cfg.densify_from_iter,
                     densify_until_iter=cfg.densify_until_iter,
                     opacity_reset_interval=cfg.opacity_reset_interval,
                     coarse_iterations=cfg.coarse_iterations)
    op = OptimizationParams(**op_kwargs)

    hp = ModelHiddenParams(**cfg.ModelHiddenParams)
    pp = PipelineParams(convert_SHs_python=False,
                        compute_cov3D_python=False, debug=False)

    mgs_cfg = dict(mgs_enabled=getattr(cfg, 'mgs_enabled', False),
                   mgs_ratio=getattr(cfg, 'mgs_ratio', 0.5),
                   budget_min=getattr(cfg, 'budget_min', 0.1),
                   budget_max=getattr(cfg, 'budget_max', 1.0),
                   ordering_strategy=getattr(cfg, 'ordering_strategy', 'opacity_area'))

    return lp, op, hp, pp, mgs_cfg


def measure_training(config_path, label):
    lp, op, hp, pp, mgs_cfg = parse_config(config_path)

    gaussians = GaussianModel(lp.sh_degree, hp)
    scene = Scene(lp, gaussians, shuffle=False)
    gaussians.training_setup(op)

    bg_color = torch.tensor([1.0, 1.0, 1.0] if lp.white_background else [0.0, 0.0, 0.0],
                            dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoints = viewpoint_stack

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    elapsed_times = []

    for iteration in range(1, WARMUP_ITERS + MEASURE_ITERS + 1):
        viewpoint_cam = viewpoints[randint(0, len(viewpoints) - 1)]

        if mgs_cfg['mgs_enabled'] and np.random.random() < mgs_cfg['mgs_ratio']:
            K = sample_budget(gaussians._xyz.shape[0], mgs_cfg['budget_min'], mgs_cfg['budget_max'])
            importance = compute_importance(gaussians, viewpoint_cam)
            sorted_idx = get_sorted_indices(importance)
            subset_idx = sorted_idx[:K]
            gs_subset = gaussians.subset_view(subset_idx)
            render_pkg = render(viewpoint_cam, gs_subset, pp, bg_color)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pp, bg_color)

        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - op.lambda_dssim) * Ll1 + op.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        if iteration > WARMUP_ITERS:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed_times.append(time.perf_counter() - t0)
        else:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration % 30 == 0:
            print(f"  {label} iter {iteration}/{WARMUP_ITERS + MEASURE_ITERS}", end="\r")

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    avg_step_time = np.mean(elapsed_times) * 1000  # ms
    total_30k_min = (avg_step_time / 1000 * 30000) / 60  # minutes

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"    Avg step time (step only): {avg_step_time:.1f} ms")
    print(f"    Est. 30K iter total:       {total_30k_min:.1f} min")
    print(f"    Peak GPU memory:           {peak_mem:.0f} MiB")
    print(f"{'='*50}\n")

    del gaussians, scene
    torch.cuda.empty_cache()

    return {
        'label': label,
        'avg_step_ms': round(avg_step_time, 1),
        'est_30k_min': round(total_30k_min, 1),
        'peak_mem_mib': round(peak_mem, 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='MGS config path')
    parser.add_argument('--baseline', type=str, help='Baseline config path (auto-detect if not provided)')
    args = parser.parse_args()

    print(f"Warmup: {WARMUP_ITERS} iters, Measure: {MEASURE_ITERS} iters")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []

    # 1. Measure baseline
    baseline_cfg = args.baseline or args.config.replace('configs/mgs_', 'configs/baseline_')
    if not os.path.exists(baseline_cfg):
        # Fallback: baseline config = same config but mgs_enabled=False
        baseline_cfg = args.config  # will be handled in code
        print(f"\nNo baseline config found, using MGS config with mgs_enabled manually disabled")
        # We'll edit args.config to disable MGS
        import tempfile, shutil
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        with open(args.config) as f:
            content = f.read()
        content = content.replace('mgs_enabled = True', 'mgs_enabled = False')
        tmp.write(content)
        tmp.close()
        results.append(measure_training(tmp.name, "Baseline (mgs_enabled=False)"))
        os.unlink(tmp.name)
    else:
        results.append(measure_training(baseline_cfg, "Baseline"))

    # 2. Measure MGS
    results.append(measure_training(args.config, "MGS (ours)"))

    # Summary
    if len(results) >= 2:
        baseline = results[0]
        mgs = results[1]
        overhead_ms = mgs['avg_step_ms'] - baseline['avg_step_ms']
        overhead_pct = (overhead_ms / baseline['avg_step_ms']) * 100

        print(f"\n{'='*60}")
        print(f"  OVERHEAD: MGS vs Baseline")
        print(f"    Step time: +{overhead_ms:.1f} ms ({overhead_pct:+.1f}%)")
        print(f"    30K total: {mgs['est_30k_min']:.1f} vs {baseline['est_30k_min']:.1f} min")
        mem_diff = mgs['peak_mem_mib'] - baseline['peak_mem_mib']
        print(f"    GPU mem:   {mgs['peak_mem_mib']:.0f} vs {baseline['peak_mem_mib']:.0f} MiB ({mem_diff:+.0f} MiB)")
        print(f"{'='*60}")

    print("\nDone.")


if __name__ == "__main__":
    main()
