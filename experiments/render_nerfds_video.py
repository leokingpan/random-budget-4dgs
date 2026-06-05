#!/usr/bin/env python3
"""NeRF-DS Plate comparison video: GT | 25% | 50% | Full | Post-hoc 25%"""
import os, sys, argparse, numpy as np, torch, imageio
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from scene.gaussian_model import GaussianModel
from scene.hyper_loader import Load_hyper_data
from utils.camera_utils import loadCam
from gaussian_renderer import render
from utils.mgs_utils import compute_importance, get_sorted_indices
from arguments import PipelineParams
from utils.motion_scorer import parse_cfg_args
import torchvision


class DummyArgs:
    resolution = -1
    data_device = "cuda"


def main():
    ckpt = os.path.join(_BASE, 'output/nerfds/plate_mgs')
    src = os.path.join(_BASE, 'data/nerfds/plate_novel_view')
    out = os.path.join(_BASE, 'results/videos/plate')

    cfg = parse_cfg_args(os.path.join(ckpt, "cfg_args"))
    iterdir = os.path.join(ckpt, "point_cloud", "iteration_15000")

    g = GaussianModel(getattr(cfg, 'sh_degree', 3), cfg)
    g._deformation = g._deformation.to("cuda")
    g.load_ply(os.path.join(iterdir, "point_cloud.ply"))
    g._deformation.load_state_dict(torch.load(os.path.join(iterdir, "deformation.pth"), map_location="cuda"), strict=False)
    dt_path = os.path.join(iterdir, "deformation_table.pth")
    if os.path.exists(dt_path):
        g._deformation_table = torch.load(dt_path, map_location="cuda")
    g._deformation.eval()
    N = g._xyz.shape[0]
    print(f"Gaussians: {N}")

    infos = Load_hyper_data(src, 0.5, False, split="test")
    cams = [loadCam(DummyArgs(), 0, ci, 1.0) for ci in infos]
    n_frames = min(30, len(cams))
    cams = cams[:n_frames]
    print(f"Cameras: {len(cams)}")

    pipe = PipelineParams(argparse.ArgumentParser())
    bg = torch.tensor([0.0, 0.0, 0.0], device='cuda')

    # Pre-compute ours ordering for first frame as static reference
    imp = compute_importance(g, cams[0])
    ours_idx = get_sorted_indices(imp)

    os.makedirs(out, exist_ok=True)
    frames = []

    for idx in range(n_frames):
        cam = cams[idx]
        gt = cam.original_image[:3].clamp(0, 1).cuda()

        # Ours at multiple budgets
        K25 = max(2, int(0.25 * N))
        K50 = max(2, int(0.50 * N))

        r_ours25 = render(cam, g.subset_view(ours_idx[:K25]), pipe, bg)["render"]
        r_ours50 = render(cam, g.subset_view(ours_idx[:K50]), pipe, bg)["render"]
        r_full = render(cam, g, pipe, bg)["render"]

        # Post-hoc random 25%
        rand_idx = torch.randperm(N, device='cuda')[:K25]
        r_post25 = render(cam, g.subset_view(rand_idx), pipe, bg)["render"]

        # Composite: GT | Post-hoc 25% | Ours 25% | Ours 50% | Ours 100%
        panels = [gt, r_post25, r_ours25, r_ours50, r_full]
        bar = torch.ones((3, 30, gt.shape[2]), device=gt.device)
        comp = torch.cat([torch.cat([bar, p], dim=1) for p in panels], dim=2).detach()

        np_frame = (comp.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frames.append(np_frame)
        print(f"  Frame {idx+1}/{n_frames}", end='\r')

    print(f"\nSaving video ({len(frames)} frames)...")
    imageio.mimwrite(os.path.join(out, "comparison.mp4"), frames, fps=10, quality=8)
    imageio.imwrite(os.path.join(out, "comparison_frame0.png"), frames[0])
    print(f"Done: {out}")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
