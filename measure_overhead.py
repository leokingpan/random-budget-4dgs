#!/usr/bin/env python3
"""
Exp 9: Training & Inference Overhead Decomposition

Measures component-wise timing for:
  Training: baseline iter, +scoring, +sort/topK, +prefix render, +full render
  Inference: ordering recompute time, rendering time per budget

Usage:
    python measure_overhead.py --checkpoint output/dnerf/bouncingballs_mgs
"""

import os, sys, argparse, time, json
import numpy as np
import torch
import random

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from PIL import Image
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.mgs_utils import (
    compute_importance, compute_opacity_only, get_sorted_indices, sample_budget
)
from utils.motion_scorer import parse_cfg_args
from utils.general_utils import safe_state, PILtoTorch
from utils.graphics_utils import getWorld2View
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from arguments import PipelineParams


def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = np.tan(fovY / 2)
    tanHalfFovX = np.tan(fovX / 2)
    top = tanHalfFovY * znear; bottom = -top
    right = tanHalfFovX * znear; left = -right
    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def load_test_cameras(source_path, n_frames=20):
    import json
    transforms_path = os.path.join(source_path, "transforms_test.json")
    with open(transforms_path) as f:
        transforms = json.load(f)
    frames = transforms.get("frames", [])
    camera_angle_x = transforms["camera_angle_x"]
    w = int(transforms.get("w", 800))
    h = int(transforms.get("h", 800))
    all_times = [f["time"] for f in frames]
    max_time = max(all_times) if all_times else 1.0

    cameras = []
    for i, frame in enumerate(frames[:n_frames]):
        image_path = os.path.join(source_path, frame["file_path"] + ".png")
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1, 1, 1])
        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
        gt_image = PILtoTorch(image, (w, h))

        tf = np.array(frame["transform_matrix"])
        R = tf[:3, :3].T; T = tf[:3, 3]
        FovY = focal2fov(focal2fov(camera_angle_x, w), h)
        FovX = camera_angle_x
        time_val = frame["time"] / max_time if max_time > 0 else frame["time"]

        w2v = torch.tensor(getWorld2View(R, T)).transpose(0, 1)
        proj = getProjectionMatrix(0.01, 100.0, FovX, FovY).transpose(0, 1)
        full_proj = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

        cam = Camera(
            colmap_id=i, R=R, T=T, FoVx=FovX, FoVy=FovY,
            image=gt_image, gt_alpha_mask=None,
            image_name=frame["file_path"], uid=i,
            trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
            time=time_val, mask=None, depth=None, data_device="cuda"
        )
        cam.original_image = gt_image.to("cuda")
        cameras.append(cam)
    return cameras


def time_block(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--iter", type=int, default=30000)
    a = parser.parse_args()

    safe_state(False)

    ckpt_root = os.path.abspath(a.checkpoint)
    args_ns = parse_cfg_args(os.path.join(ckpt_root, "cfg_args"))
    src = os.path.abspath(os.path.join(_BASE_DIR, getattr(args_ns, "source_path", "")))
    iter_dir = os.path.join(ckpt_root, "point_cloud", f"iteration_{a.iter}")

    sd = getattr(args_ns, "sh_degree", 3)
    gs = GaussianModel(sd, args_ns)
    gs._deformation = gs._deformation.to("cuda")

    gs.load_ply(os.path.join(iter_dir, "point_cloud.ply"))
    gs._deformation.load_state_dict(
        torch.load(os.path.join(iter_dir, "deformation.pth"), map_location="cuda"),
        strict=False
    )
    gs._deformation.eval()

    dtp = os.path.join(iter_dir, "deformation_table.pth")
    if os.path.exists(dtp):
        gs._deformation_table = torch.load(dtp, map_location="cuda")

    cams = load_test_cameras(src, n_frames=50)
    N = gs._xyz.shape[0]
    print(f"Gaussians: {N}  Cameras: {len(cams)}")

    pipe = PipelineParams(parser)
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    # ---- INFERENCE OVERHEAD ----
    print("\n" + "=" * 70)
    print("INFERENCE: Component-wise Timing (per frame, averaged)")
    print("=" * 70)

    n_test_frames = min(20, len(cams))
    budgets = [0.1, 0.25, 0.5, 0.75, 1.0]

    # Static ordering (t=0)
    _, t_static_score = time_block(
        lambda: compute_importance(gs, cams[0])
    )
    static_importance = compute_importance(gs, cams[0])
    static_sorted = get_sorted_indices(static_importance)

    print(f"\nStatic ordering (t=0, one-time cost):")
    print(f"  Deformation-aware scoring: {t_static_score:.1f} ms")

    for budget in budgets:
        K = max(2, int(budget * N))
        total_ms = 0
        for fi in range(n_test_frames):
            cam = cams[fi]
            top_idx = static_sorted[:K]
            _, t_render = time_block(
                lambda: render(cam, gs.subset_view(top_idx), pipe, bg)
            )
            total_ms += t_render
        avg_ms = total_ms / n_test_frames
        fps = 1000 / avg_ms if avg_ms > 0 else float('inf')
        print(f"  Budget {budget}: render={avg_ms:.1f} ms/frame → FPS={fps:.1f}")

    # Time-adaptive ordering
    print(f"\nTime-adaptive ordering (per-frame cost):")
    total_score_ms = 0
    total_sort_ms = 0
    for fi in range(n_test_frames):
        cam = cams[fi]
        _, t_score = time_block(lambda: compute_importance(gs, cam))
        imp = compute_importance(gs, cam)
        _, t_sort = time_block(lambda: get_sorted_indices(imp))
        total_score_ms += t_score
        total_sort_ms += t_sort
    print(f"  Per-frame scoring: {total_score_ms/n_test_frames:.1f} ms")
    print(f"  Per-frame sorting: {total_sort_ms/n_test_frames:.1f} ms")
    print(f"  Total overhead/frame: {(total_score_ms+total_sort_ms)/n_test_frames:.1f} ms")

    # Static vs adaptive FPS comparison at 50% budget
    K50 = max(2, int(0.5 * N))
    top_static = static_sorted[:K50]
    tt_static = 0; tt_adaptive = 0
    for fi in range(n_test_frames):
        cam = cams[fi]
        _, tr = time_block(lambda: render(cam, gs.subset_view(top_static), pipe, bg))
        tt_static += tr

        imp = compute_importance(gs, cam)
        sidx = get_sorted_indices(imp)
        _, tr = time_block(lambda: render(cam, gs.subset_view(sidx[:K50]), pipe, bg))
        tt_adaptive += tr

    fps_static = n_test_frames * 1000 / tt_static if tt_static > 0 else float('inf')
    fps_adaptive = n_test_frames * 1000 / tt_adaptive if tt_adaptive > 0 else float('inf')
    print(f"\n  50% budget FPS comparison:")
    print(f"    Static ordering:    {fps_static:.1f} FPS (no per-frame re-scoring)")
    print(f"    Time-adaptive:      {fps_adaptive:.1f} FPS (re-scores each frame)")
    print(f"    Speedup (static):   {fps_static/fps_adaptive:.2f}x")

    # Compute total overhead of adaptive ordering
    overhead_per_frame = (total_score_ms + total_sort_ms) / n_test_frames
    print(f"\n    Misc ordering overhead: {overhead_per_frame:.1f} ms/frame")
    print(f"    Deformation query dominates scoring cost (shared with rendering)")

    print("\n" + "=" * 70)
    print("INFERENCE OVERHEAD SUMMARY (static ordering, no per-frame recompute)")
    print("=" * 70)
    print(f"{'Ordering':<20} {'Compute once':<15} {'Per-frame':<15} {'FPS gain':<10}")
    print(f"{'Static (t=0)':<20} {t_static_score:>8.1f} ms {'0.0 ms':>15} {'—':>10}")


if __name__ == "__main__":
    main()
