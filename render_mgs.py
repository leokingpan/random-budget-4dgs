#!/usr/bin/env python3
"""
MGS Inference & Evaluation Script.

Renders a trained checkpoint at multiple budget ratios using Masked Gaussian
Splatting (MGS) and computes PSNR / SSIM / FPS.  Supports three ordering
strategies to enable baseline comparison.

Usage:
    # Standard evaluation
    python render_mgs.py \
        --checkpoint output/dnerf/bouncingballs_mgs \
        --budgets 0.1,0.25,0.5,0.75,1.0 \
        --frames 5 \
        --output results/mgs/bouncingballs_psnr_vs_budget.csv

    # Random-baseline evaluation
    python render_mgs.py \
        --checkpoint output/dnerf/bouncingballs_mgs \
        --ordering random --budgets 0.1,0.25,0.5,0.75,1.0 --frames 5

    # Post-hoc opacity baseline (no area weighting)
    python render_mgs.py \
        --checkpoint output/dnerf/bouncingballs_mgs \
        --ordering posthoc_opacity --budgets 0.1,0.25,0.5,0.75,1.0 --frames 5

    # Quick sanity check
    python render_mgs.py --checkpoint output/dnerf/bouncingballs_mgs --dry-run
"""

import os
import sys
import argparse
import csv
import json
import math
import time
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from utils.general_utils import PILtoTorch
from utils.loss_utils import ssim, lpips_loss
from utils.motion_scorer import parse_cfg_args
from utils.mgs_utils import (
    compute_importance,
    get_sorted_indices,
    compute_opacity_only,
)

_lpips_model = None


def _get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex").cuda()
    return _lpips_model


# ---------------------------------------------------------------------------
# Pipeline params (matching task spec: SH=False)
# ---------------------------------------------------------------------------
class PipelineParams:
    def __init__(self, convert_SHs_python=False, compute_cov3D_python=False,
                 debug=False):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MGS inference: PSNR/SSIM/FPS vs budget ratio"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint root (e.g., output/dnerf/bouncingballs_mgs)"
    )
    parser.add_argument(
        "--budgets", type=str, default="0.1,0.25,0.5,0.75,1.0",
        help=("Comma-separated budget ratios in (0, 1] (default: "
              "0.1,0.25,0.5,0.75,1.0)")
    )
    parser.add_argument(
        "--frames", type=int, default=5,
        help="Number of test frames to process per budget (default: 5)"
    )
    parser.add_argument(
        "--output", type=str,
        default="results/mgs/bouncingballs_psnr_vs_budget.csv",
        help="Output CSV path (default: results/mgs/bouncingballs_psnr_vs_budget.csv)"
    )
    parser.add_argument(
        "--ordering", type=str, default="ours",
        choices=["ours", "random", "posthoc_opacity"],
        help=("Gaussian ordering strategy (default: ours). "
              "ours: importance = deformed_opacity × projected_area. "
              "random: uniformly random permutation. "
              "posthoc_opacity: deformed opacity only, no area weighting.")
    )
    parser.add_argument(
        "--static", action="store_true",
        help="Use static global ordering (importance computed once at t=0, reused for all frames)"
    )
    parser.add_argument(
        "--iter", type=int, default=30000,
        help="Checkpoint iteration to load (default: 30000)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run single budget on single frame, print stats, no CSV output"
    )
    return parser


# ---------------------------------------------------------------------------
# Camera loading helpers
#   Pattern: same structure as exp1_classification.py build_camera()
#   R/T:     canonical inverse-transform (matching scene/dataset_readers.py)
#   Image:   white-background alpha blending for Blender dataset
# ---------------------------------------------------------------------------
def load_transforms(source_path: str) -> dict:
    """Load transforms_test.json from the dataset directory."""
    transforms_path = os.path.join(source_path, "transforms_test.json")
    if not os.path.exists(transforms_path):
        raise FileNotFoundError(
            f"transforms_test.json not found at {transforms_path}"
        )
    with open(transforms_path) as f:
        return json.load(f)


def compute_max_time(source_path: str) -> float:
    """Collect timestamps from train+test transforms; return max."""
    all_times = []

    test_path = os.path.join(source_path, "transforms_test.json")
    if os.path.exists(test_path):
        with open(test_path) as f:
            test_data = json.load(f)
        all_times.extend(f["time"] for f in test_data.get("frames", []))

    train_path = os.path.join(source_path, "transforms_train.json")
    if os.path.exists(train_path):
        with open(train_path) as f:
            train_data = json.load(f)
        all_times.extend(f["time"] for f in train_data.get("frames", []))

    if not all_times:
        return 1.0
    return max(all_times)


def build_camera(idx: int, frame: dict, camera_angle_x: float,
                 source_path: str, normalized_time: float) -> Camera:
    """
    Build a Camera from a transforms.json frame entry.

    R/T using canonical inverse-transform (matching dataset_readers.py).
    White-background alpha blending for Blender-format RGBA images.

    Args:
        idx:              Camera index (uid / colmap_id)
        frame:            Dict with 'file_path', 'time', 'transform_matrix'
        camera_angle_x:   Horizontal FoV from transforms JSON (radians)
        source_path:      Root of the dataset (for image loading)
        normalized_time:  Frame time normalised to [0, 1]

    Returns:
        Camera nn.Module
    """
    # R/T computation — canonical inverse-transform
    matrix = np.linalg.inv(np.array(frame["transform_matrix"], dtype=np.float64))
    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]

    # Image loading with white-background alpha blending
    img_rel = frame["file_path"]  # e.g., "./test/r_000"
    img_path = os.path.join(source_path, img_rel + ".png")
    with Image.open(img_path) as pil_img:
        im_data = np.array(pil_img.convert("RGBA"), dtype=np.float64)
    bg = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    norm_data = im_data / 255.0
    arr = (norm_data[:, :, :3] * norm_data[:, :, 3:4] +
           bg * (1.0 - norm_data[:, :, 3:4]))
    arr_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    rgb_image = Image.fromarray(arr_uint8, "RGB")
    img_tensor = PILtoTorch(rgb_image, (800, 800))

    # FoV — square 800×800 → FoVy = FoVx (fov2focal + focal2fov roundtrip)
    FoVx = float(camera_angle_x)
    FoVy = float(camera_angle_x)

    img_name = os.path.splitext(os.path.basename(img_rel))[0]
    cam = Camera(
        colmap_id=idx,
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        image=img_tensor,
        gt_alpha_mask=None,
        image_name=img_name,
        uid=idx,
        data_device=torch.device("cuda"),
        time=normalized_time,
        mask=None,
    )
    return cam


def load_nerfies_cameras(source_path):
    """Load cameras from Nerfies/HyperNeRF format (dataset.json + camera/)."""
    from scene.hyper_loader import Load_hyper_data
    from utils.camera_utils import loadCam as load_cam_func

    # Load test split — dataset.json, metadata.json, scene.json, rgb/, camera/
    test_dataset = Load_hyper_data(source_path, 0.5, use_bg_points=False,
                                   split="test")

    class FakeArgs:
        resolution = -1
        data_device = "cuda"

    cameras = []
    for cam_info in test_dataset:
        cam = load_cam_func(FakeArgs(), 0, cam_info, 1.0)
        cameras.append(cam)

    return cameras


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_psnr(rendered: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR in dB.  Both tensors [3, H, W] in [0, 1] range."""
    mse = torch.mean((rendered - gt) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim(rendered: torch.Tensor, gt: torch.Tensor) -> float:
    """SSIM via the 3DGS codebase implementation.
    Both tensors [3, H, W] in [0, 1] range; returns scalar."""
    # ssim expects [B, C, H, W]
    return ssim(rendered.unsqueeze(0), gt.unsqueeze(0)).item()


def compute_lpips(rendered: torch.Tensor, gt: torch.Tensor) -> float:
    model = _get_lpips_model()
    rendered_norm = rendered.unsqueeze(0) * 2.0 - 1.0
    gt_norm = gt.unsqueeze(0) * 2.0 - 1.0
    with torch.no_grad():
        return model(rendered_norm, gt_norm).item()


# ---------------------------------------------------------------------------
# Importance / ordering helpers
# ---------------------------------------------------------------------------
def get_sorted_indices_for_ordering(
    gaussians, cam, ordering: str, N: int
) -> torch.Tensor:
    """
    Return a LongTensor of Gaussian indices sorted by the chosen ordering
    strategy (highest importance first).

    Args:
        gaussians: GaussianModel with loaded _deformation
        cam:       Camera with .time attribute
        ordering:  'ours' | 'random' | 'posthoc_opacity'
        N:         Total number of Gaussians

    Returns:
        LongTensor[N] of sorted indices
    """
    if ordering == "random":
        return torch.randperm(N, device="cuda")

    if ordering == "posthoc_opacity":
        importance = compute_opacity_only(gaussians, cam)
    else:  # "ours"
        importance = compute_importance(gaussians, cam)

    return get_sorted_indices(importance, descending=True)


# ---------------------------------------------------------------------------
# Single-frame render helper
# ---------------------------------------------------------------------------
def render_subset(
    gaussians,
    cam: Camera,
    subset_indices: torch.Tensor,
    pipe,
    bg_color: torch.Tensor,
) -> torch.Tensor:
    """
    Render using only a subset of Gaussians.

    Args:
        gaussians:      Full GaussianModel
        cam:            Camera
        subset_indices: LongTensor[K] of indices to keep
        pipe:           PipelineParams
        bg_color:       [3] tensor on GPU

    Returns:
        rendered_image: [3, H, W] tensor on GPU
    """
    sub = gaussians.subset_view(subset_indices)
    result = render(cam, sub, pipe, bg_color)
    return result["render"]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_argparse()
    args = parser.parse_args()

    # ---- Parse budgets -------------------------------------------------------
    budgets = [float(b.strip()) for b in args.budgets.split(",")]
    for b in budgets:
        if b <= 0 or b > 1:
            print(f"ERROR: budget {b} must be in (0, 1]", file=sys.stderr)
            sys.exit(1)

    # ---- 1. Resolve paths ----------------------------------------------------
    checkpoint_root = os.path.abspath(args.checkpoint)
    cfg_path = os.path.join(checkpoint_root, "cfg_args")
    if not os.path.exists(cfg_path):
        print(f"ERROR: cfg_args not found at {cfg_path}", file=sys.stderr)
        print("Checkpoint may not exist yet. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ---- 2. Parse config -----------------------------------------------------
    args_ns = parse_cfg_args(cfg_path)
    source_path = os.path.abspath(
        os.path.join(_BASE_DIR, getattr(args_ns, "source_path", ""))
    )
    iteration_dir = os.path.join(checkpoint_root, "point_cloud",
                                 f"iteration_{args.iter}")

    print(f"Checkpoint : {checkpoint_root}")
    print(f"Dataset    : {source_path}")
    time_res = args_ns.kplanes_config['resolution'][3]
    print(f"Config     : multires={args_ns.multires}, "
          f"bounds={args_ns.bounds}, time_res={time_res}")

    # ---- 3. Create and load GaussianModel ------------------------------------
    sh_degree = getattr(args_ns, "sh_degree", 3)
    gaussians = GaussianModel(sh_degree, args_ns)

    ply_path = os.path.join(iteration_dir, "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"ERROR: PLY not found at {ply_path}", file=sys.stderr)
        print("Checkpoint may be incomplete. Exiting.", file=sys.stderr)
        sys.exit(1)

    gaussians._deformation = gaussians._deformation.to("cuda")

    print(f"Loading PLY : {ply_path}")
    gaussians.load_ply(ply_path)
    n_gaussians = gaussians._xyz.shape[0]
    print(f"  Loaded {n_gaussians} Gaussians")

    deform_path = os.path.join(iteration_dir, "deformation.pth")
    if not os.path.exists(deform_path):
        print(f"ERROR: deformation.pth not found at {deform_path}",
              file=sys.stderr)
        print("Checkpoint may be incomplete. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading deformation : {deform_path}")
    gaussians._deformation.load_state_dict(
        torch.load(deform_path, map_location="cuda"), strict=False
    )
    gaussians._deformation.eval()

    n_grids = len(gaussians._deformation.deformation_net.grid.grids)
    print(f"  Grid scales : {n_grids}")
    print(f"  AABB        : {gaussians._deformation.deformation_net.grid.aabb.data}")

    deform_table_path = os.path.join(iteration_dir, "deformation_table.pth")
    if os.path.exists(deform_table_path):
        gaussians._deformation_table = torch.load(deform_table_path, map_location="cuda")
        print(f"  Loaded deformation table: {gaussians._deformation_table.shape}")

    # ---- 4. Load cameras (auto-detect format) -----------------------------------
    transforms_test_json = os.path.join(source_path, "transforms_test.json")
    dataset_json = os.path.join(source_path, "dataset.json")

    if os.path.exists(transforms_test_json):
        # D-NeRF format
        transforms = load_transforms(source_path)
        frames = transforms.get("frames", [])
        camera_angle_x = transforms.get("camera_angle_x", None)
        if camera_angle_x is None:
            print("ERROR: 'camera_angle_x' missing from transforms JSON",
                  file=sys.stderr)
            sys.exit(1)
        if not frames:
            print("ERROR: No frames found in transforms_test.json", file=sys.stderr)
            sys.exit(1)

        max_time = compute_max_time(source_path)
        print(f"  Max time across train+test : {max_time:.6f}")
        n_frames = min(args.frames, len(frames))
        print(f"  Processing {n_frames} / {len(frames)} test frames")
        print(f"  Format: D-NeRF")

        cameras = []
        for i, frame in enumerate(frames[:n_frames]):
            norm_t = frame["time"] / max_time if max_time > 0 else 0.0
            cam = build_camera(i, frame, camera_angle_x, source_path, norm_t)
            cameras.append(cam)

    elif os.path.exists(dataset_json):
        # Nerfies/HyperNeRF format
        cameras = load_nerfies_cameras(source_path)
        n_frames = min(args.frames, len(cameras))
        cameras = cameras[:n_frames]
        print(f"  Processing {n_frames} frames")
        print(f"  Format: Nerfies/HyperNeRF")

    else:
        print("ERROR: No recognized camera format found "
              "(neither transforms_test.json nor dataset.json)", file=sys.stderr)
        sys.exit(1)

    # ---- 5. Build pipe & bg --------------------------------------------------
    pipe = PipelineParams(convert_SHs_python=False,
                          compute_cov3D_python=False,
                          debug=False)
    bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32,
                            device="cuda")

    # ---- 6. Dry-run: single budget, single frame, no CSV --------------------
    if args.dry_run:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"DRY-RUN MODE -- ordering={args.ordering}, budget={budgets[0]}, frame=0")
        print(sep)

        budget = budgets[0]
        K = max(2, int(budget * n_gaussians))

        cam = cameras[0]

        # Compute ordering
        sorted_idx = get_sorted_indices_for_ordering(
            gaussians, cam, args.ordering, n_gaussians
        )
        top_idx = sorted_idx[:K]

        t_val = getattr(cam, "time", 0.0)
        print(f"\nFrame 0 (t={t_val:.4f}), budget={budget}:")
        print(f"  Gaussians kept: {K}/{n_gaussians} "
              f"({K / n_gaussians * 100:.1f}%)")

        # Render and time
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        rendered = render_subset(gaussians, cam, top_idx, pipe, bg_color)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start

        gt = cam.original_image[0:3, :, :].cuda()
        psnr_val = compute_psnr(rendered, gt)
        ssim_val = compute_ssim(rendered, gt)
        fps_val = 1.0 / elapsed if elapsed > 0 else float("inf")

        print(f"  PSNR: {psnr_val:.4f} dB")
        print(f"  SSIM: {ssim_val:.6f}")
        print(f"  Render time: {elapsed * 1000:.1f} ms  →  FPS: {fps_val:.2f}")

        # Full-budget reference
        print("\n  Full-budget reference (budget=1.0):")
        all_idx = sorted_idx[:n_gaussians]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        rendered_full = render_subset(gaussians, cam, all_idx, pipe, bg_color)
        torch.cuda.synchronize()
        elapsed_full = time.perf_counter() - t2
        psnr_full = compute_psnr(rendered_full, gt)
        ssim_full = compute_ssim(rendered_full, gt)
        fps_full = 1.0 / elapsed_full if elapsed_full > 0 else float("inf")
        print(f"    PSNR: {psnr_full:.4f} dB")
        print(f"    SSIM: {ssim_full:.6f}")
        print(f"    Render time: {elapsed_full * 1000:.1f} ms  "
              f"→  FPS: {fps_full:.2f}")

        print("\nDry-run complete.")
        sys.exit(0)

    # ---- 7. Multi-budget, multi-frame run ------------------------------------
    print(f"\nOrdering strategy : {args.ordering}")
    print(f"Budgets           : {budgets}")
    print(f"Frames            : {n_frames}")

    rows = []

    static_sorted_idx = None
    if args.static and args.ordering == "ours":
        ref_idx = min(10, len(cameras) - 1)
        ref_cam = cameras[ref_idx]
        print(f"\nComputing static global ordering (t={ref_cam.time:.4f}, cam {ref_idx})...")
        static_importance = compute_importance(gaussians, ref_cam)
        static_sorted_idx = get_sorted_indices(static_importance, descending=True)
        print(f"  Done. Using same ordering for all frames.")

    for budget_idx, budget in enumerate(budgets):
        K = max(2, int(budget * n_gaussians))
        print(f"\n--- Budget {budget_idx + 1}/{len(budgets)}: "
              f"{budget} (K={K}) ---")

        total_psnr = 0.0
        total_ssim_val = 0.0
        total_lpips_val = 0.0
        total_elapsed = 0.0

        for frame_idx in range(n_frames):
            cam = cameras[frame_idx]

            if static_sorted_idx is not None:
                sorted_idx = static_sorted_idx
            else:
                sorted_idx = get_sorted_indices_for_ordering(
                    gaussians, cam, args.ordering, n_gaussians
                )
            top_idx = sorted_idx[:K]

            # Render and time
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            rendered = render_subset(gaussians, cam, top_idx, pipe, bg_color)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_start

            gt = cam.original_image[0:3, :, :].cuda()
            psnr_f = compute_psnr(rendered, gt)
            ssim_f = compute_ssim(rendered, gt)
            lpips_f = compute_lpips(rendered, gt)

            total_psnr += psnr_f
            total_ssim_val += ssim_f
            total_lpips_val += lpips_f
            total_elapsed += elapsed

            print(f"  frame {frame_idx + 1}/{n_frames}  "
                  f"PSNR={psnr_f:.4f} dB  SSIM={ssim_f:.6f}  "
                  f"LPIPS={lpips_f:.6f}  "
                  f"{elapsed * 1000:.1f} ms")

            del sorted_idx, top_idx, rendered, gt, cam

        avg_psnr = total_psnr / n_frames
        avg_ssim = total_ssim_val / n_frames
        avg_lpips = total_lpips_val / n_frames
        avg_fps = n_frames / total_elapsed if total_elapsed > 0 else float("inf")

        rows.append({
            "budget": budget,
            "PSNR": round(avg_psnr, 4),
            "SSIM": round(avg_ssim, 6),
            "LPIPS": round(avg_lpips, 6),
            "FPS": round(avg_fps, 2),
        })

        print(f"  AVERAGE over {n_frames} frames: "
              f"PSNR={avg_psnr:.4f} dB  SSIM={avg_ssim:.6f}  "
              f"LPIPS={avg_lpips:.6f}  "
              f"FPS={avg_fps:.2f}")

    # ---- 8. Write CSV --------------------------------------------------------
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    csv_fieldnames = ["budget", "PSNR", "SSIM", "LPIPS", "FPS"]
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {output_path}")
    print(f"\n{'=' * 60}")
    print("SUMMARY (ordering = {})".format(args.ordering))
    print("=" * 60)
    for row in rows:
        print(f"  budget={row['budget']:<6}  "
              f"PSNR={row['PSNR']:.4f} dB  "
              f"SSIM={row['SSIM']:.6f}  "
              f"LPIPS={row['LPIPS']:.6f}  "
              f"FPS={row['FPS']:.2f}")
    print(f"\nDone. {len(budgets)} budgets × {n_frames} frames processed.")


if __name__ == "__main__":
    main()
