#!/usr/bin/env python3
"""
Motion Scorer for HexPlane-based 4D Gaussian Splatting.

Extracts time-plane features from the trained HexPlane deformation field
and computes per-Gaussian motion scores for static/dynamic classification.

Reusable utility module — foundation for Experiments 1, 2, 3.
Read-only access to deformation network (no parameter modification).
"""

import os
import sys
import argparse
import json
from typing import List

# Ensure project root is on path for module imports when run as script
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

import torch
import numpy as np

from scene.hexplane import normalize_aabb, grid_sample_wrapper

# Plane indices from itertools.combinations(range(4), 2):
#   [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
#   = [xy, xz, xt, yz, yt, zt]
# Time planes (involving t=dim 3): index 2 (x,t), index 4 (y,t), index 5 (z,t)
TIME_PLANE_INDICES = [2, 4, 5]
# Corresponding coordinate combos for each time plane
TIME_PLANE_COMBS = [(0, 3), (1, 3), (2, 3)]  # (x,t), (y,t), (z,t)


def extract_time_plane_features(gaussians, frame_time: float) -> List[torch.Tensor]:
    """
    Extract per-Gaussian feature vectors from all time planes across all scales.

    The HexPlane has 6 planes per scale (xy, xz, xt, yz, yt, zt).
    Time planes (xt, yt, zt = indices 2, 4, 5) capture temporal dynamics.
    We extract features from these 3 planes × 4 scales = 12 feature tensors.

    Args:
        gaussians: GaussianModel instance with loaded _deformation
        frame_time: scalar time in [0, 1] (normalized frame time)

    Returns:
        List of 12 Tensors, each [N, feat_dim], one per (scale, time_plane) pair.
        Order: scale 0 planes (2,4,5), scale 1 planes (2,4,5), ..., scale 3 planes (2,4,5).
    """
    with torch.no_grad():
        # Access the HexPlane grid
        grid = gaussians._deformation.deformation_net.grid
        aabb = grid.aabb  # Parameter [2, 3]
        grids = grid.grids  # ModuleList[ParameterList] — one ParameterList per scale

        # Get canonical Gaussian positions [N, 3]
        xyz = gaussians._xyz.detach()

        # Normalize spatial coords to [-1, 1] (exact replica of hexplane.py:163)
        pts_norm = normalize_aabb(xyz, aabb)  # [N, 3]

        # Append temporal coordinate (already in [0, 1])
        t_val = torch.full((xyz.shape[0], 1), float(frame_time),
                           device=xyz.device, dtype=xyz.dtype)
        pts_4d = torch.cat([pts_norm, t_val], dim=-1)  # [N, 4]

        features = []

        for scale_idx, scale_grids in enumerate(grids):
            # scale_grids is a ParameterList of 6 Parameters
            # Each Parameter has shape [1, feat_dim, reso_dim1, reso_dim0]
            for tp_idx, coo_comb in zip(TIME_PLANE_INDICES, TIME_PLANE_COMBS):
                grid_param = scale_grids[tp_idx]  # Parameter [1, feat_dim, H, W]

                # Extract the 2 coordinate columns for this plane
                # e.g., plane 2 is (x,t): pts_4d[..., (0,3)] → [N, 2] = (x_norm, t)
                coords = pts_4d[:, coo_comb]  # [N, 2]

                # Grid-sample to get features for this plane
                # grid_sample_wrapper handles batching and reshaping
                feat = grid_sample_wrapper(
                    grid_param, coords,
                    align_corners=True  # matching hexplane.py:43
                )  # → [N, feat_dim]

                # Ensure correct shape
                if feat.dim() == 1:
                    # Single Gaussian edge case
                    feat = feat.unsqueeze(0)

                features.append(feat)

        return features


def compute_motion_score(time_features: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute per-Gaussian motion score from time-plane features.

    Time planes initialized to 1 → static Gaussians have features close to 1.
    Deviation from 1 indicates temporal dynamics.
    Scores are summed across all 12 feature tensors.

    Formula: sum over all 12 tensors of abs(f - 1.0).sum(dim=-1)

    Args:
        time_features: List of 12 Tensors from extract_time_plane_features(),
                       each [N, feat_dim]

    Returns:
        Tensor[N] — per-Gaussian scalar motion score (higher = more dynamic)
    """
    if not time_features:
        return torch.empty(0, device='cuda')

    motion = torch.zeros_like(time_features[0].sum(dim=-1))

    for feat in time_features:
        # Deviation of time-plane features from 1.0 (the initialization value)
        motion = motion + torch.abs(feat - 1.0).sum(dim=-1)

    return motion


def compute_reference_deformation(gaussians, frame_time: float) -> torch.Tensor:
    """
    Compute per-Gaussian L2 displacement using the full deformation network.

    Calls gaussians._deformation() at the given frame time and returns
    the Euclidean distance between deformed and canonical positions.

    Args:
        gaussians: GaussianModel instance with loaded _deformation
        frame_time: scalar time in [0, 1]

    Returns:
        Tensor[N] — per-Gaussian L2 norm of displacement from canonical position
    """
    with torch.no_grad():
        xyz = gaussians._xyz.detach()              # [N, 3]
        scales = gaussians._scaling.detach()       # [N, 3]
        rotations = gaussians._rotation.detach()   # [N, 4]
        opacity = gaussians._opacity.detach()      # [N, 1]
        # For bouncingballs no_dshs=True, so shs is passed through unchanged;
        # we use _features_dc which contains the raw DC color coefficients [N, 1, 3]
        shs = gaussians._features_dc.detach()      # [N, 1, 3]

        time_emb = torch.full((xyz.shape[0], 1), float(frame_time),
                              device=xyz.device, dtype=xyz.dtype)

        # Full forward pass through deform_network → deformation_net.forward_dynamic
        deformed_xyz, _, _, _, _ = gaussians._deformation(
            xyz, scales, rotations, opacity, shs, time_emb
        )

        # Per-Gaussian L2 norm of displacement
        displacement = torch.norm(deformed_xyz - xyz, p=2, dim=-1)

    return displacement


def get_motion_label(motion_score: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Binary label: True for dynamic Gaussians based on motion score.

    Args:
        motion_score: Tensor[N] from compute_motion_score()
        threshold: scalar threshold value

    Returns:
        Tensor[N] bool — True where motion_score > threshold
    """
    return motion_score > threshold


def get_reference_label(ref_deformation: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Binary label: True for dynamic Gaussians based on reference deformation.

    Args:
        ref_deformation: Tensor[N] from compute_reference_deformation()
        threshold: scalar threshold value

    Returns:
        Tensor[N] bool — True where ref_deformation > threshold
    """
    return ref_deformation > threshold


# ---------------------------------------------------------------------------
# CLI dry-run mode
# ---------------------------------------------------------------------------


def parse_cfg_args(path: str):
    """Parse textual cfg_args file (Namespace string)."""
    from argparse import Namespace
    with open(path) as f:
        text = f.read()
    # Extract the Namespace(...) portion
    start = text.index('Namespace(')
    ns = eval(text[start:], {'__builtins__': {}, 'Namespace': Namespace}, {})
    return ns


def _build_dry_run_argparse():
    parser = argparse.ArgumentParser(
        description="Motion Scorer dry-run — load model and compute scores for one frame"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint root (e.g., output/dnerf/bouncingballs_baseline)')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index from transforms_test.json to use (default: 0)')
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='Run single-frame analysis (default if no other mode)')
    return parser


def main():
    parser = _build_dry_run_argparse()
    args = parser.parse_args()

    # ---- 1. Resolve paths ----
    checkpoint_root = os.path.abspath(args.checkpoint)
    cfg_path = os.path.join(checkpoint_root, "cfg_args")
    if not os.path.exists(cfg_path):
        print(f"ERROR: cfg_args not found at {cfg_path}", file=sys.stderr)
        sys.exit(1)

    # ---- 2. Parse config ----
    args_ns = parse_cfg_args(cfg_path)
    source_path = os.path.abspath(getattr(args_ns, 'source_path', ''))
    iteration_dir = os.path.join(checkpoint_root, "point_cloud", "iteration_30000")

    print(f"Checkpoint: {checkpoint_root}")
    print(f"Dataset:    {source_path}")
    print(f"Config:     multires={args_ns.multires}, "
          f"bounds={args_ns.bounds}, "
          f"time_res={args_ns.kplanes_config['resolution'][3]}")

    # ---- 3. Create and load GaussianModel ----
    from scene.gaussian_model import GaussianModel
    from utils.general_utils import safe_state

    safe_state(False)

    sh_degree = getattr(args_ns, 'sh_degree', 3)
    gaussians = GaussianModel(sh_degree, args_ns)
    gaussians._deformation = gaussians._deformation.to("cuda")

    ply_path = os.path.join(iteration_dir, "point_cloud.ply")
    print(f"Loading PLY: {ply_path}")
    gaussians.load_ply(ply_path)
    print(f"  Loaded {gaussians._xyz.shape[0]} Gaussians")

    deform_path = os.path.join(iteration_dir, "deformation.pth")
    print(f"Loading deformation: {deform_path}")
    gaussians._deformation.load_state_dict(torch.load(deform_path, map_location="cuda"),
                                           strict=False)
    gaussians._deformation.eval()

    print(f"  Grid scales: {len(gaussians._deformation.deformation_net.grid.grids)}")
    print(f"  AABB: {gaussians._deformation.deformation_net.grid.aabb.data}")

    # ---- 4. Load 1 camera from transforms_test.json to get frame time ----
    transforms_path = os.path.join(source_path, "transforms_test.json")
    frame_time = None
    if os.path.exists(transforms_path):
        with open(transforms_path) as f:
            transforms = json.load(f)
        frames = transforms.get("frames", [])
        if frames:
            frame_idx = min(args.frame, len(frames) - 1)
            raw_time = frames[frame_idx]["time"]
            # Time normalization: D-NeRF style — read_timeline returns mapper
            # that divides raw_time by max_time across train+test.
            # For bouncingballs, times are already in [0, ~1.4] range,
            # normalized by max_time to [0, 1].
            # We follow the same normalization as read_timeline + readCamerasFromTransforms
            all_times = [f["time"] for f in frames]
            max_time = max(all_times) if all_times else 1.0
            frame_time = raw_time / max_time
            print(f"Camera: frame[{frame_idx}] raw_time={raw_time:.4f}, "
                  f"normalized={frame_time:.4f}")
    if frame_time is None:
        # Fallback: use frame 0 → time 0.0
        frame_time = 0.0
        print(f"Warning: no transforms_test.json found, using frame_time={frame_time}")

    # ---- 5. Extract features ----
    print(f"\n--- Extracting time-plane features at t={frame_time:.4f} ---")
    time_features = extract_time_plane_features(gaussians, frame_time)

    print(f"  Extracted {len(time_features)} feature tensors "
          f"(4 scales × 3 time planes)")
    for i, feat in enumerate(time_features):
        s = i // 3
        p = TIME_PLANE_INDICES[i % 3]
        print(f"  [{i}] scale={s} plane={p}: shape={list(feat.shape)} "
              f"mean={feat.mean().item():.6f} std={feat.std().item():.6f}")

    # ---- 6. Compute motion scores ----
    motion_scores = compute_motion_score(time_features)
    print(f"\n--- Motion Score Stats ---")
    print(f"  min:     {motion_scores.min().item():.6f}")
    print(f"  max:     {motion_scores.max().item():.6f}")
    print(f"  mean:    {motion_scores.mean().item():.6f}")
    print(f"  median:  {motion_scores.median().item():.6f}")
    print(f"  std:     {motion_scores.std().item():.6f}")

    # ---- 7. Compute reference deformation ----
    ref_deform = compute_reference_deformation(gaussians, frame_time)
    print(f"\n--- Reference Deformation Stats ---")
    print(f"  min:     {ref_deform.min().item():.8f}")
    print(f"  max:     {ref_deform.max().item():.8f}")
    print(f"  mean:    {ref_deform.mean().item():.8f}")
    print(f"  median:  {ref_deform.median().item():.8f}")
    print(f"  std:     {ref_deform.std().item():.8f}")

    # ---- 8. Labeling at a few thresholds ----
    print(f"\n--- Labeling at Example Thresholds ---")
    for thresh_name, thresh_val in [("median", motion_scores.median().item()),
                                     ("mean", motion_scores.mean().item()),
                                     ("mean+std", (motion_scores.mean() + motion_scores.std()).item()),
                                     ("mean-std", max(0, (motion_scores.mean() - motion_scores.std()).item()))]:
        labels = get_motion_label(motion_scores, thresh_val)
        pct = labels.float().mean().item() * 100
        print(f"  motion_score > {thresh_name}({thresh_val:.6f}): "
              f"{labels.sum().item()}/{len(labels)} = {pct:.1f}% dynamic")

    print(f"\nDone. Motion scorer dry-run complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
