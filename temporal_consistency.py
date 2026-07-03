#!/usr/bin/env python3
"""
Temporal Consistency Evaluation for 4D Gaussian Splatting.

Measures how temporally stable rendered outputs are under different
Gaussian budget constraints.  Computes three metrics:

    Frame-PSNR : average PSNR between rendered frames and ground truth
    T-PSNR     : PSNR between consecutive rendered frames (higher = more stable)
    T-L1       : L1 distance between consecutive rendered frames (lower = more stable)

Usage:
    python temporal_consistency.py \
        --checkpoint output/dnerf/bouncingballs_baseline \
        --budgets 0.1,0.25,0.5,0.75,1.0 \
        --method mgs \
        --frames 50
"""

import os
import sys
import argparse
import math

import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from render_mgs import load_transforms, build_camera, compute_max_time
from utils.motion_scorer import parse_cfg_args
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.mgs_utils import (
    compute_importance,
    compute_opacity_only,
    get_sorted_indices,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR in dB between two tensors in [0, 1] range."""
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def compute_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean absolute (L1) error between two tensors in [0, 1] range."""
    return torch.mean(torch.abs(a - b)).item()


def compute_temporal_psnr(rendered_frames: list[torch.Tensor]) -> float:
    """Average PSNR between consecutive rendered frames.

    Higher T-PSNR → less temporal flickering.
    """
    if len(rendered_frames) < 2:
        return float("inf")

    total = 0.0
    for i in range(len(rendered_frames) - 1):
        total += compute_psnr(rendered_frames[i], rendered_frames[i + 1])

    return total / (len(rendered_frames) - 1)


def compute_temporal_l1(rendered_frames: list[torch.Tensor]) -> float:
    """Average L1 distance between consecutive rendered frames.

    Lower T-L1 → less temporal flickering.
    """
    if len(rendered_frames) < 2:
        return 0.0

    total = 0.0
    for i in range(len(rendered_frames) - 1):
        total += compute_l1(rendered_frames[i], rendered_frames[i + 1])

    return total / (len(rendered_frames) - 1)


def compute_frame_psnr(
    rendered_frames: list[torch.Tensor],
    gt_frames: list[torch.Tensor],
) -> float:
    """Average PSNR between rendered frames and ground truth."""
    if not rendered_frames:
        return 0.0

    total = 0.0
    for rendered, gt in zip(rendered_frames, gt_frames):
        total += compute_psnr(rendered, gt)

    return total / len(rendered_frames)


# ═══════════════════════════════════════════════════════════════════════════════
# Camera loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_cameras_sorted_by_time(
    source_path: str,
    n_frames: int,
) -> list:
    """Load test cameras from transforms_test.json, sorted by frame time."""
    transforms = load_transforms(source_path)
    camera_angle_x = transforms.get("camera_angle_x", None)
    if camera_angle_x is None:
        raise ValueError("'camera_angle_x' missing from transforms JSON")

    frames = transforms.get("frames", [])
    if not frames:
        raise ValueError("No frames found in transforms_test.json")

    # Sort frames by raw time for temporal ordering
    frames = sorted(frames, key=lambda f: f["time"])

    # Limit to requested number
    n_frames = min(n_frames, len(frames))
    frames = frames[:n_frames]

    # Compute max_time for normalisation (handle zero)
    max_time = compute_max_time(source_path)
    if max_time <= 0.0:
        max_time = 1.0

    cameras = []
    for i, frame in enumerate(frames):
        norm_t = frame["time"] / max_time
        cam = build_camera(i, frame, camera_angle_x, source_path, norm_t)
        cameras.append(cam)

    return cameras


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_root: str, iteration: int) -> tuple[GaussianModel, int]:
    """Load GaussianModel from a checkpoint.

    Returns:
        (gaussians, n_gaussians)
    """
    cfg_path = os.path.join(checkpoint_root, "cfg_args")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"cfg_args not found at {cfg_path}")

    args_ns = parse_cfg_args(cfg_path)
    sh_degree = getattr(args_ns, "sh_degree", 3)

    iteration_dir = os.path.join(
        checkpoint_root, "point_cloud", f"iteration_{iteration}"
    )

    # Create and load GaussianModel
    gaussians = GaussianModel(sh_degree, args_ns)
    gaussians._deformation = gaussians._deformation.to("cuda")

    ply_path = os.path.join(iteration_dir, "point_cloud.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY not found at {ply_path}")

    print(f"Loading PLY      : {ply_path}")
    gaussians.load_ply(ply_path)

    n_gaussians = gaussians._xyz.shape[0]
    print(f"  Loaded {n_gaussians} Gaussians")

    # Load deformation if available
    deform_path = os.path.join(iteration_dir, "deformation.pth")
    if os.path.exists(deform_path):
        print(f"Loading deform   : {deform_path}")
        gaussians._deformation.load_state_dict(
            torch.load(deform_path, map_location="cuda"), strict=False
        )
        gaussians._deformation.eval()
    else:
        print("WARNING: deformation.pth not found — using canonical model.")

    # Load optional deformation table
    deform_table_path = os.path.join(iteration_dir, "deformation_table.pth")
    if os.path.exists(deform_table_path):
        gaussians._deformation_table = torch.load(
            deform_table_path, map_location="cuda"
        )
        print(f"  Loaded deformation table: {gaussians._deformation_table.shape}")

    return gaussians, n_gaussians


# ═══════════════════════════════════════════════════════════════════════════════
# Core rendering loop for a single budget
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_budget(
    gaussians,
    cameras: list,
    budget: float,
    n_gaussians: int,
    method: str,
    pipe,
    bg_color: torch.Tensor,
) -> dict:
    """Evaluate one budget across all frames.

    Returns dict with keys: budget, frame_psnr, t_psnr, t_l1
    """
    K = max(2, int(budget * n_gaussians))

    rendered_frames: list[torch.Tensor] = []
    gt_frames: list[torch.Tensor] = []

    for frame_idx, cam in enumerate(cameras):
        # Compute importance scores
        if method == "mgs":
            importance = compute_importance(gaussians, cam)
        else:  # posthoc
            importance = compute_opacity_only(gaussians, cam)

        sorted_indices = get_sorted_indices(importance, descending=True)
        top_indices = sorted_indices[:K]

        # Render with subset
        sub = gaussians.subset_view(top_indices)
        result = render(cam, sub, pipe, bg_color)

        rendered = result["render"]
        gt = cam.original_image[0:3, :, :].cuda()

        rendered_frames.append(rendered)
        gt_frames.append(gt)

        if frame_idx == 0:
            print(f"  budget={budget:.2f}  K={K}  frame 1/{len(cameras)} "
                  f"PSNR={compute_psnr(rendered, gt):.4f}")

        del sorted_indices, top_indices, sub, result

    # Compute metrics
    frame_psnr_val = compute_frame_psnr(rendered_frames, gt_frames)
    t_psnr_val = compute_temporal_psnr(rendered_frames)
    t_l1_val = compute_temporal_l1(rendered_frames)

    return {
        "budget": budget,
        "frame_psnr": round(frame_psnr_val, 4),
        "t_psnr": round(t_psnr_val, 4),
        "t_l1": round(t_l1_val, 6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal consistency evaluation"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint root (e.g., output/dnerf/bouncingballs_baseline)"
    )
    parser.add_argument(
        "--budgets", type=str, default="0.1,0.25",
        help="Comma-separated budget ratios (default: 0.1,0.25)"
    )
    parser.add_argument(
        "--method", type=str, default="mgs",
        choices=["mgs", "posthoc"],
        help="Ordering method: mgs (opacity × area) or posthoc (opacity only)"
    )
    parser.add_argument(
        "--frames", type=int, default=50,
        help="Number of test frames to process (default: 50)"
    )
    parser.add_argument(
        "--iter", type=int, default=None,
        help="Checkpoint iteration to load (default: auto-detect max)"
    )
    args = parser.parse_args()

    # ── Parse budgets ──────────────────────────────────────────────────────
    budgets = [float(b.strip()) for b in args.budgets.split(",")]
    for b in budgets:
        if b <= 0.0 or b > 1.0:
            print(f"ERROR: budget {b} must be in (0, 1]", file=sys.stderr)
            sys.exit(1)

    # ── Resolve checkpoint path ────────────────────────────────────────────
    checkpoint_root = os.path.abspath(args.checkpoint)
    cfg_path = os.path.join(checkpoint_root, "cfg_args")
    if not os.path.exists(cfg_path):
        print(f"ERROR: cfg_args not found at {cfg_path}", file=sys.stderr)
        sys.exit(1)

    # ── Resolve iteration ──────────────────────────────────────────────────
    if args.iter is not None:
        iteration = args.iter
    else:
        # Auto-detect max iteration from point_cloud directory
        pc_dir = os.path.join(checkpoint_root, "point_cloud")
        if os.path.isdir(pc_dir):
            iterations = []
            for entry in os.listdir(pc_dir):
                if entry.startswith("iteration_"):
                    try:
                        iterations.append(int(entry.split("_")[1]))
                    except (IndexError, ValueError):
                        pass
            iteration = max(iterations) if iterations else 30000
        else:
            iteration = 30000

    # ── Parse config for source_path ───────────────────────────────────────
    args_ns = parse_cfg_args(cfg_path)
    source_path = os.path.abspath(
        os.path.join(_BASE_DIR, getattr(args_ns, "source_path", ""))
    )

    print(f"Checkpoint : {checkpoint_root}")
    print(f"Dataset    : {source_path}")
    print(f"Iteration  : {iteration}")
    print(f"Method     : {args.method}")
    print(f"Budgets    : {budgets}")
    print(f"Frames     : {args.frames}")

    # ── Load model ─────────────────────────────────────────────────────────
    gaussians, n_gaussians = load_model(checkpoint_root, iteration)

    # ── Load cameras sorted by time ────────────────────────────────────────
    cameras = load_cameras_sorted_by_time(source_path, args.frames)
    n_frames = len(cameras)
    print(f"\nLoaded {n_frames} cameras (sorted by time)")

    # ── Pipeline params ────────────────────────────────────────────────────
    from argparse import ArgumentParser as AP2
    from render_mgs import PipelineParams
    pipe = PipelineParams(convert_SHs_python=False, compute_cov3D_python=False)

    bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    # ── Evaluate each budget ───────────────────────────────────────────────
    results = []

    for budget in budgets:
        print(f"\n{'=' * 60}")
        print(f"Evaluating budget: {budget}")
        print(f"{'=' * 60}")

        result = evaluate_budget(
            gaussians, cameras, budget, n_gaussians,
            args.method, pipe, bg_color,
        )
        results.append(result)

        print(f"  Frame-PSNR: {result['frame_psnr']:.4f} dB")
        print(f"  T-PSNR    : {result['t_psnr']:.4f} dB  (higher = more stable)")
        print(f"  T-L1      : {result['t_l1']:.6f}    (lower = more stable)")

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"SUMMARY  (method={args.method}, frames={n_frames})")
    print(f"{'=' * 60}")
    print(f"{'Budget':<10} {'Frame-PSNR':<14} {'T-PSNR':<14} {'T-L1':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['budget']:<10.2f} {r['frame_psnr']:<14.4f} "
              f"{r['t_psnr']:<14.4f} {r['t_l1']:<12.6f}")

    print(f"\nDone. {len(budgets)} budgets × {n_frames} frames processed.")


if __name__ == "__main__":
    main()
