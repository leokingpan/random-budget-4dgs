#!/usr/bin/env python3
"""
NeRF-DS 对比视频渲染（plate + press）
5 列: GT | Post-hoc 25% (opacity-only) | Ours 25% | Ours 50% | Ours 100%
与 D-NeRF 视频 baseline 一致。
"""
import os, sys, numpy as np, torch, imageio, argparse
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from scene.gaussian_model import GaussianModel
from scene.hyper_loader import Load_hyper_data
from utils.camera_utils import loadCam
from gaussian_renderer import render
from utils.mgs_utils import compute_importance, get_sorted_indices, compute_opacity_only
from arguments import PipelineParams, ModelHiddenParams
from utils.motion_scorer import parse_cfg_args


def render_scene(checkpoint, source, output_dir, scene_name, n_frames=30):
    os.makedirs(output_dir, exist_ok=True)

    cfg = parse_cfg_args(os.path.join(checkpoint, "cfg_args"))
    # Try iteration_15000 first, fallback to iteration_30000
    for it in [15000, 30000]:
        iterdir = os.path.join(checkpoint, "point_cloud", f"iteration_{it}")
        if os.path.exists(os.path.join(iterdir, "point_cloud.ply")):
            break

    sh_deg = getattr(cfg, 'sh_degree', 3)
    g = GaussianModel(sh_deg, cfg)
    g._deformation = g._deformation.to("cuda")
    g.load_ply(os.path.join(iterdir, "point_cloud.ply"))
    g._deformation.load_state_dict(
        torch.load(os.path.join(iterdir, "deformation.pth"), map_location="cuda"),
        strict=False
    )
    dt_path = os.path.join(iterdir, "deformation_table.pth")
    if os.path.exists(dt_path):
        g._deformation_table = torch.load(dt_path, map_location="cuda")
    g._deformation.eval()
    N = g._xyz.shape[0]
    print(f"[{scene_name}] Gaussians: {N}")

    # Load cameras
    infos = Load_hyper_data(source, 0.5, False, split="test")
    cams = []
    for ci in infos:
        try:
            cams.append(loadCam(argparse.Namespace(resolution=-1, data_device="cuda"), 0, ci, 1.0))
        except:
            # fallback
            cams.append(loadCam(argparse.Namespace(resolution=-1, data_device="cuda"), 0, ci, 1.0))

    n_frames = min(n_frames, len(cams))
    cams = cams[:n_frames]
    print(f"[{scene_name}] Cameras: {len(cams)}")

    pipe = PipelineParams(argparse.ArgumentParser())
    bg = torch.tensor([0.0, 0.0, 0.0], device='cuda')

    # Pre-compute ordering
    ours_imp = compute_importance(g, cams[0])
    ours_idx = get_sorted_indices(ours_imp)
    posthoc_imp = compute_opacity_only(g, cams[0])
    posthoc_idx = get_sorted_indices(posthoc_imp)

    frames = []
    for idx in range(n_frames):
        cam = cams[idx]
        gt = cam.original_image[:3].clamp(0, 1).cuda()

        K25 = max(2, int(0.25 * N))
        K50 = max(2, int(0.50 * N))

        # Post-hoc 25% (opacity-only, consistent with D-NeRF videos)
        r_post25 = render(cam, g.subset_view(posthoc_idx[:K25]), pipe, bg)["render"]
        # Ours 25%
        r_ours25 = render(cam, g.subset_view(ours_idx[:K25]), pipe, bg)["render"]
        # Ours 50%
        r_ours50 = render(cam, g.subset_view(ours_idx[:K50]), pipe, bg)["render"]
        # Ours 100%
        r_full = render(cam, g, pipe, bg)["render"]

        panels = [gt, r_post25, r_ours25, r_ours50, r_full]
        bar = torch.ones((3, 30, gt.shape[2]), device=gt.device)
        comp = torch.cat([torch.cat([bar, p], dim=1) for p in panels], dim=2).detach()

        np_frame = (comp.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frames.append(np_frame)
        print(f"  Frame {idx+1}/{n_frames}", end='\r')

    video_path = os.path.join(output_dir, "comparison.mp4")
    frame_path = os.path.join(output_dir, "comparison_frame0.png")
    imageio.mimwrite(video_path, frames, fps=10, quality=8)
    imageio.imwrite(frame_path, frames[0])
    print(f"\n[{scene_name}] Done: {video_path}  ({len(frames)} frames)")

    del g
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="all", choices=["plate", "press", "all"])
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/videos")
    args = parser.parse_args()

    scenes = []
    if args.scene in ("plate", "all"):
        scenes.append((
            os.path.join(_BASE, "output/nerfds/plate_mgs"),
            os.path.join(_BASE, "data/nerfds/plate_novel_view"),
            os.path.join(_BASE, args.output, "plate"),
            "plate"
        ))
    if args.scene in ("press", "all"):
        scenes.append((
            os.path.join(_BASE, "output/nerfds/press_mgs"),
            os.path.join(_BASE, "data/nerfds/press_novel_view"),
            os.path.join(_BASE, args.output, "press"),
            "press"
        ))

    for ckpt, src, out, name in scenes:
        render_scene(ckpt, src, out, name, args.frames)

    print("\nAll done.")


if __name__ == "__main__":
    main()
