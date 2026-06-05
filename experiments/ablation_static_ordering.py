#!/usr/bin/env python3
"""Static vs Time-Adaptive Ordering Ablation for MGS × 4DGS."""
import os, sys, json, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import ssim as ssim_fn, lpips_loss
from utils.motion_scorer import parse_cfg_args
from utils.mgs_utils import compute_importance, get_sorted_indices


_lpips_model = None
def _get_lpips():
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex").cuda()
    return _lpips_model


class MiniPipe:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_cameras(source_path, n_frames):
    from scene.cameras import Camera
    from utils.general_utils import PILtoTorch
    from PIL import Image as PILImage
    transforms_path = os.path.join(source_path, "transforms_test.json")
    with open(transforms_path) as f:
        transforms = json.load(f)
    all_times = [t["time"] for t in transforms["frames"]]
    max_time = max(all_times) if all_times else 1.0
    cameras = []
    for idx, frame in enumerate(transforms["frames"][:n_frames]):
        image_path = os.path.join(source_path, frame["file_path"] + ".png")
        img = PILImage.open(image_path)
        H, W = 800, 800
        img_tensor = PILtoTorch(img, (W, H))
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        c2w[:3, 1:3] *= -1
        w2c = torch.linalg.inv(c2w)
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
        fov_x = frame.get("camera_angle_x", 0.691111)
        fov_y = 2.0 * math.atan(0.5 * H / (0.5 * W / math.tan(0.5 * fov_x)))
        cam = Camera(colmap_id=idx, R=R.numpy(), T=T.numpy(),
                     FoVx=fov_x, FoVy=fov_y, image=img_tensor, gt_alpha_mask=None,
                     image_name=f"{idx}", uid=idx, data_device=torch.device("cuda"),
                     time=frame["time"] / max_time)
        cameras.append(cam)
    return cameras


def compute_psnr(rendered, gt):
    mse = torch.mean((rendered - gt) ** 2).item()
    return 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")


def compute_ssim(rendered, gt):
    return ssim_fn(rendered.unsqueeze(0), gt.unsqueeze(0)).item()


def compute_lpips(rendered, gt):
    model = _get_lpips()
    rn = rendered.unsqueeze(0) * 2.0 - 1.0
    gn = gt.unsqueeze(0) * 2.0 - 1.0
    with torch.no_grad():
        return model(rn, gn).item()


def render_subset(gaussians, cam, indices, pipe, bg):
    sub = gaussians.subset_view(indices)
    return render(cam, sub, pipe, bg)["render"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--output", default="/dev/stdout")
    args = parser.parse_args()

    ckpt = os.path.abspath(args.checkpoint)
    cfg_path = os.path.join(ckpt, "cfg_args")
    args_ns = parse_cfg_args(cfg_path)
    source_rel = getattr(args_ns, "source_path", "")
    source_path = os.path.join(_BASE_DIR, source_rel)
    iteration_dir = os.path.join(ckpt, "point_cloud", "iteration_30000")
    ply_path = os.path.join(iteration_dir, "point_cloud.ply")

    sh_degree = getattr(args_ns, "sh_degree", 3)
    gaussians = GaussianModel(sh_degree, args_ns)
    gaussians._deformation = gaussians._deformation.to("cuda")
    gaussians.load_ply(ply_path)
    gaussians.load_model(iteration_dir)
    gaussians._deformation.eval()

    N = gaussians._xyz.shape[0]
    cameras = load_cameras(source_path, args.frames)
    n_frames = len(cameras)

    pipe = MiniPipe()
    bg = torch.tensor([1.0, 1.0, 1.0], device="cuda")

    budgets = [0.1, 0.25, 0.5, 0.75, 1.0]

    # --- Time-adaptive (ours): compute per frame ---
    print(f"Scene: {os.path.basename(ckpt)}  Gaussians: {N}  Frames: {n_frames}")
    print("=" * 70)
    print(f"{'Method':<18} {'Budget':>6} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'FPS':>8}")
    print("-" * 70)

    for budget in budgets:
        K = max(2, int(budget * N))
        total_psnr = total_ssim = total_lpips = total_time = 0.0
        for idx in range(n_frames):
            cam = cameras[idx]
            importance = compute_importance(gaussians, cam)
            sorted_idx = get_sorted_indices(importance)
            torch.cuda.synchronize()
            t0 = __import__('time').perf_counter()
            rendered = render_subset(gaussians, cam, sorted_idx[:K], pipe, bg)
            torch.cuda.synchronize()
            elapsed = __import__('time').perf_counter() - t0
            gt = cam.original_image[0:3, :, :].cuda()
            total_psnr += compute_psnr(rendered, gt)
            total_ssim += compute_ssim(rendered, gt)
            total_lpips += compute_lpips(rendered, gt)
            total_time += elapsed
        print(f"{'Time-Adaptive':<18} {budget:>6.2f} {total_psnr/n_frames:>8.2f} "
              f"{total_ssim/n_frames:>8.6f} {total_lpips/n_frames:>8.6f} "
              f"{n_frames/total_time if total_time > 0 else 0:>8.1f}")

    # --- Static global: compute importance at first camera, reuse ---
    first_cam = cameras[0]
    importance_static = compute_importance(gaussians, first_cam)
    sorted_static = get_sorted_indices(importance_static)

    for budget in budgets:
        K = max(2, int(budget * N))
        static_idx = sorted_static[:K]
        total_psnr = total_ssim = total_lpips = total_time = 0.0
        for idx in range(n_frames):
            cam = cameras[idx]
            torch.cuda.synchronize()
            t0 = __import__('time').perf_counter()
            rendered = render_subset(gaussians, cam, static_idx, pipe, bg)
            torch.cuda.synchronize()
            elapsed = __import__('time').perf_counter() - t0
            gt = cam.original_image[0:3, :, :].cuda()
            total_psnr += compute_psnr(rendered, gt)
            total_ssim += compute_ssim(rendered, gt)
            total_lpips += compute_lpips(rendered, gt)
            total_time += elapsed
        print(f"{'Static (t=0):':<18} {budget:>6.2f} {total_psnr/n_frames:>8.2f} "
              f"{total_ssim/n_frames:>8.6f} {total_lpips/n_frames:>8.6f} "
              f"{n_frames/total_time if total_time > 0 else 0:>8.1f}")

    print("=" * 70)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
