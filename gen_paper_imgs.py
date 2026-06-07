#!/usr/bin/env python3
"""Generate paper images: GT, 25% prefix, multi-budget comparison grid.
Saves to paper/TVC/new/ directory.
"""
import os, sys, json, argparse, numpy as np, torch
from PIL import Image

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.mgs_utils import compute_importance, get_sorted_indices
from utils.motion_scorer import parse_cfg_args
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getWorld2View
from gaussian_renderer import render
from arguments import PipelineParams

OUT = os.path.join(_BASE, "paper", "TVC", "new")
os.makedirs(OUT, exist_ok=True)

def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))

def load_cams(src, n=5):
    with open(os.path.join(src, "transforms_test.json")) as f:
        tf = json.load(f)
    frames = tf["frames"][:n]
    cam_x = tf["camera_angle_x"]
    w, h = 800, 800
    mt = max(f["time"] for f in frames) or 1.0
    cams = []
    for i, fr in enumerate(frames):
        img = Image.open(os.path.join(src, fr["file_path"] + ".png"))
        im = np.array(img.convert("RGBA")) / 255.0
        arr = im[:,:,:3] * im[:,:,3:4] + (1 - im[:,:,3:4])
        img = Image.fromarray(np.array(arr*255, dtype=np.byte), "RGB")
        gt = PILtoTorch(img, (w, h))
        M = np.array(fr["transform_matrix"])
        R, T = M[:3,:3].T, M[:3,3]
        FovY = focal2fov(focal2fov(cam_x, w), h)
        tv = fr["time"] / mt
        w2v = torch.tensor(getWorld2View(R,T)).transpose(0,1)
        P = torch.zeros(4,4)
        P[0,0] = 1/np.tan(cam_x/2); P[1,1] = 1/np.tan(FovY/2)
        P[2,2] = 100/(100-0.01); P[2,3] = -(100*0.01)/(100-0.01); P[3,2] = 1
        fp = w2v.unsqueeze(0).bmm(P.unsqueeze(0)).squeeze(0)
        cam = Camera(colmap_id=i, R=R, T=T, FoVx=cam_x, FoVy=FovY,
                     image=gt, gt_alpha_mask=None, image_name=fr["file_path"],
                     uid=i, trans=np.zeros(3), scale=1.0,
                     time=tv, mask=None, depth=None, data_device="cuda")
        cam.original_image = gt.to("cuda")
        cams.append(cam)
    return cams

def save_img(tensor, path):
    arr = tensor.detach().clamp(0,1).permute(1,2,0).cpu().numpy()
    Image.fromarray((arr*255).astype(np.uint8)).save(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="bouncingballs")
    parser.add_argument("--frames", type=int, default=5)
    a = parser.parse_args()

    ckpt = f"output/dnerf/{a.scene}_mgs"
    cfg = parse_cfg_args(os.path.join(ckpt, "cfg_args"))
    src = os.path.abspath(os.path.join(_BASE, cfg.source_path))
    itd = os.path.join(ckpt, "point_cloud", "iteration_30000")

    gs = GaussianModel(getattr(cfg, "sh_degree", 3), cfg)
    gs._deformation = gs._deformation.to("cuda")
    gs.load_ply(os.path.join(itd, "point_cloud.ply"))
    gs._deformation.load_state_dict(
        torch.load(os.path.join(itd, "deformation.pth"), map_location="cuda"), strict=False)
    gs._deformation.eval()
    dtp = os.path.join(itd, "deformation_table.pth")
    if os.path.exists(dtp):
        gs._deformation_table = torch.load(dtp, map_location="cuda")
    N = gs._xyz.shape[0]
    print(f"Loaded {a.scene}: {N} Gaussians")

    cams = load_cams(src, a.frames)
    pipe = PipelineParams(parser)
    bg = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")

    # Compute ordering once
    imp = compute_importance(gs, cams[0])
    sidx = get_sorted_indices(imp)

    budgets = [0.1, 0.25, 0.5, 1.0]
    for fi, cam in enumerate(cams):
        # A: GT
        gt_img = cam.original_image[:3]
        save_img(gt_img, os.path.join(OUT, f"{a.scene}_frame{fi}_A_gt.png"))

        # B: 25% prefix render
        K25 = max(2, int(0.25 * N))
        rp = render(cam, gs.subset_view(sidx[:K25]), pipe, bg)
        save_img(rp["render"], os.path.join(OUT, f"{a.scene}_frame{fi}_B_25pct.png"))

        # C: Multi-budget grid (composited side-by-side)
        renders = []
        for b in budgets:
            K = max(2, int(b * N))
            rp = render(cam, gs.subset_view(sidx[:K]), pipe, bg)
            renders.append(rp["render"])
        # Horizontal concat: 10% | 25% | 50% | 100%
        grid = torch.cat(renders, dim=2)
        save_img(grid, os.path.join(OUT, f"{a.scene}_frame{fi}_C_multibudget.png"))

    print(f"Saved {a.frames}×3 frames to {OUT}")

if __name__ == "__main__":
    main()
