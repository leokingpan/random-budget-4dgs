#!/usr/bin/env python3
"""
Post-hoc Pruning + Fine-Tuning Baseline (Exp 4).

Loads a trained baseline checkpoint, prunes to target budget via opacity
sorting, then fine-tunes. Cameras are loaded directly (no Scene object).

Usage:
    python train_posthoc_ft.py \
        --checkpoint output/dnerf/bouncingballs_baseline \
        --budget 0.25 --ft_iters 2000 \
        --expname posthoc_ft_025_2k
"""

import os, sys, argparse, random, json
import numpy as np
import torch

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from PIL import Image
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.mgs_utils import compute_opacity_only, get_sorted_indices
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
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def load_cameras_dnerf(source_path, n_frames=50):
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
        bg = np.array([1, 1, 1]) if im_data.shape[-1] == 4 else np.array([0, 0, 0])
        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
        gt_image = PILtoTorch(image, (w, h))

        tf = np.array(frame["transform_matrix"])
        R = tf[:3, :3].T
        T = tf[:3, 3]
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
    print(f"Loaded {len(cameras)} D-NeRF test cameras from {source_path}")
    return cameras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--ft_iters", type=int, default=2000)
    parser.add_argument("--expname", required=True)
    parser.add_argument("--seed", type=int, default=42)
    a = parser.parse_args()

    safe_state(False)
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    ckpt_root = os.path.abspath(a.checkpoint)
    args_ns = parse_cfg_args(os.path.join(ckpt_root, "cfg_args"))
    src = os.path.abspath(os.path.join(_BASE_DIR, getattr(args_ns, "source_path", "")))
    iter_dir = os.path.join(ckpt_root, "point_cloud", "iteration_30000")

    sd = getattr(args_ns, "sh_degree", 3)
    gs = GaussianModel(sd, args_ns)
    gs._deformation = gs._deformation.to("cuda")

    ply = os.path.join(iter_dir, "point_cloud.ply")
    gs.load_ply(ply)
    n0 = gs._xyz.shape[0]
    print(f"Loaded {n0} Gaussians from baseline checkpoint")

    dp = os.path.join(iter_dir, "deformation.pth")
    gs._deformation.load_state_dict(torch.load(dp, map_location="cuda"), strict=False)
    gs._deformation.train()

    dtp = os.path.join(iter_dir, "deformation_table.pth")
    if os.path.exists(dtp):
        gs._deformation_table = torch.load(dtp, map_location="cuda")

    cams = load_cameras_dnerf(src, n_frames=50)

    imp = compute_opacity_only(gs, cams[0])
    sidx = get_sorted_indices(imp)
    K = max(2, int(a.budget * n0))
    km = torch.zeros(n0, dtype=torch.bool, device="cuda")
    km[sidx[:K].long()] = True

    gs._xyz = torch.nn.Parameter(gs._xyz[km].detach().requires_grad_(True))
    gs._features_dc = torch.nn.Parameter(gs._features_dc[km].detach().requires_grad_(True))
    gs._features_rest = torch.nn.Parameter(gs._features_rest[km].detach().requires_grad_(True))
    gs._opacity = torch.nn.Parameter(gs._opacity[km].detach().requires_grad_(True))
    gs._scaling = torch.nn.Parameter(gs._scaling[km].detach().requires_grad_(True))
    gs._rotation = torch.nn.Parameter(gs._rotation[km].detach().requires_grad_(True))
    if hasattr(gs, '_deformation_table') and gs._deformation_table is not None:
        gs._deformation_table = gs._deformation_table[km].clone()
    gs.xyz_gradient_accum = torch.zeros(K, device="cuda")
    gs.denom = torch.zeros(K, device="cuda")
    gs.max_radii2D = torch.zeros(K, device="cuda")
    nk = gs._xyz.shape[0]
    print(f"Pruned {n0} → {nk} ({a.budget*100:.0f}% budget)")

    pipe = PipelineParams(parser)
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    opt = torch.optim.Adam([
        {'params': [gs._xyz], 'lr': 1.6e-6, "name": "xyz"},
        {'params': [gs._features_dc], 'lr': 0.00025, "name": "f_dc"},
        {'params': [gs._features_rest], 'lr': 0.0000625, "name": "f_rest"},
        {'params': [gs._opacity], 'lr': 0.005, "name": "opacity"},
        {'params': [gs._scaling], 'lr': 0.0005, "name": "scaling"},
        {'params': [gs._rotation], 'lr': 0.0001, "name": "rotation"},
        {'params': list(gs._deformation.parameters()), 'lr': 1.6e-6, "name": "deformation"},
    ], lr=0.0, eps=1e-15)

    print(f"\nFine-tuning for {a.ft_iters} iterations...")
    ci = 0
    for it in range(a.ft_iters):
        cam = cams[ci % len(cams)]
        ci += 1
        gt = cam.original_image.cuda()
        rp = render(cam, gs, pipe, bg)
        img = rp["render"]
        L = (0.8) * l1_loss(img, gt) + 0.2 * (1.0 - ssim(img.unsqueeze(0), gt.unsqueeze(0)))
        L.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if it % 500 == 0:
            p = 10 * np.log10(1.0 / max(torch.mean((img - gt) ** 2).item(), 1e-10))
            print(f"  iter {it}/{a.ft_iters}  loss={L.item():.4f}  PSNR={p:.2f}")

    out = os.path.join(_BASE_DIR, "output", "dnerf", a.expname)
    sd_dir = os.path.join(out, "point_cloud", f"iteration_{a.ft_iters}")
    os.makedirs(sd_dir, exist_ok=True)
    gs.save_ply(os.path.join(sd_dir, "point_cloud.ply"))
    torch.save(gs._deformation.state_dict(), os.path.join(sd_dir, "deformation.pth"))
    if hasattr(gs, '_deformation_table') and gs._deformation_table is not None:
        torch.save(gs._deformation_table, os.path.join(sd_dir, "deformation_table.pth"))
    print(f"\nModel saved to {sd_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
