#!/usr/bin/env python3
"""
MGS × 4DGS 视频对比渲染脚本

为论文生成 5 列并排对比视频：
  GT | Post-hoc 25% | Ours 25% | Ours 50% | Ours 100%

支持 D-NeRF（7 场景）和 NeRF-DS（plate + press）两种数据格式。

用法:
  # 单个 D-NeRF 场景
  python experiments/render_comparison_video.py \
    --checkpoint output/dnerf/bouncingballs_mgs \
    --frames 30 --output results/videos/bouncingballs

  # 单个 NeRF-DS 场景
  python experiments/render_comparison_video.py \
    --checkpoint output/nerfds/plate_mgs \
    --frames 30 --output results/videos/plate --nerfds

  # 批量渲染所有场景
  python experiments/render_comparison_video.py --all --frames 30
"""
import os
import sys
import argparse
import json
import math
import time
import numpy as np
import torch
import imageio

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import ssim as ssim_fn
from utils.motion_scorer import parse_cfg_args
from utils.mgs_utils import compute_importance, get_sorted_indices, compute_opacity_only


class MiniPipeline:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


COMPOSITE_BUDGETS = [0.25, 0.5, 1.0]
COMPOSITE_ORDERINGS = ["ours", "posthoc_opacity"]

COLUMN_LABELS = ["GT", "Post-hoc 25%", "Ours 25%", "Ours 50%", "Ours 100%"]

SCENE_CONFIGS = {
    "bouncingballs": "configs/mgs_4dgs.py",
    "trex": "configs/mgs_trex.py",
    "mutant": "configs/mgs_mutant.py",
    "standup": "configs/mgs_standup.py",
    "hook": "configs/mgs_hook.py",
    "jumpingjacks": "configs/mgs_jumpingjacks.py",
    "hellwarrior": "configs/mgs_hellwarrior.py",
}

NERFDS_SCENES = {
    "plate": "configs/mgs_nerfds_plate.py",
    "press": "configs/mgs_nerfds_press.py",
}


def resolve_checkpoint_and_source(scene_name, nerfds=False):
    if nerfds:
        ckpt = os.path.join(_BASE_DIR, f"output/nerfds/{scene_name}_mgs")
        src = os.path.join(_BASE_DIR, f"data/nerfds/{scene_name}_novel_view")
        config = NERFDS_SCENES.get(scene_name, f"configs/mgs_nerfds_{scene_name}.py")
    else:
        ckpt = os.path.join(_BASE_DIR, f"output/dnerf/{scene_name}_mgs")
        src = os.path.join(_BASE_DIR, f"data/dnerf/{scene_name}")
        config = SCENE_CONFIGS.get(scene_name, f"configs/mgs_{scene_name}.py")
    return ckpt, src, config


def load_dnerf_cameras(source_path, n_frames):
    from scene.cameras import Camera
    from utils.general_utils import PILtoTorch
    from PIL import Image as PILImage

    transforms_path = os.path.join(source_path, "transforms_test.json")
    with open(transforms_path) as f:
        transforms = json.load(f)

    all_times = [t["time"] for t in transforms["frames"]]
    max_time = max(all_times) if all_times else 1.0

    cameras = []
    gt_images = []
    for idx, frame in enumerate(transforms["frames"][:n_frames]):
        image_path = os.path.join(source_path, frame["file_path"] + ".png")
        img = PILImage.open(image_path)
        H, W = 800, 800
        img_tensor = PILtoTorch(img, (W, H))
        gt_images.append(img_tensor.clone())

        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        c2w[:3, 1:3] *= -1
        w2c = torch.linalg.inv(c2w)
        R = w2c[:3, :3].T
        T = w2c[:3, 3]

        fov_x = frame.get("camera_angle_x", 0.691111)
        focal = 0.5 * W / math.tan(0.5 * fov_x)
        fov_y = 2.0 * math.atan(0.5 * H / focal)

        cam = Camera(
            colmap_id=idx, R=R.numpy(), T=T.numpy(),
            FoVx=fov_x, FoVy=fov_y,
            image=img_tensor, gt_alpha_mask=None,
            image_name=f"{idx}", uid=idx,
            data_device=torch.device("cuda"),
            time=frame["time"] / max_time,
        )
        gt_images[-1] = cam.original_image[0:3, :, :].clone()
        cameras.append(cam)

    return cameras, gt_images


def build_camera_loader(source_path, n_frames):
    if os.path.exists(os.path.join(source_path, "transforms_test.json")):
        return load_dnerf_cameras(source_path, n_frames)
    elif os.path.exists(os.path.join(source_path, "dataset.json")):
        raise NotImplementedError("NeRF-DS video rendering requires separate camera loading logic")
    else:
        raise FileNotFoundError(f"Cannot detect dataset format at {source_path}")


def render_frame(gaussians, cam, subset_indices, pipe, bg):
    sub = gaussians.subset_view(subset_indices)
    return render(cam, sub, pipe, bg)["render"]


def render_all_variants(gaussians, cam, gt_image, pipe, bg):
    n = gaussians._xyz.shape[0]

    ours_importance = compute_importance(gaussians, cam)
    ours_sorted = get_sorted_indices(ours_importance)
    posthoc_importance = compute_opacity_only(gaussians, cam)
    posthoc_sorted = get_sorted_indices(posthoc_importance)

    renders = {"gt": gt_image.clone().cuda()}

    for budget in COMPOSITE_BUDGETS:
        if budget < 1.0:
            k = max(2, int(budget * n))
            posthoc_idx = posthoc_sorted[:k]
            ours_idx = ours_sorted[:k]
        else:
            posthoc_idx = posthoc_sorted
            ours_idx = ours_sorted

        posthoc_key = f"posthoc_{int(budget * 100)}"
        ours_key = f"ours_{int(budget * 100)}"

        renders[posthoc_key] = render_frame(gaussians, cam, posthoc_idx, pipe, bg)
        renders[ours_key] = render_frame(gaussians, cam, ours_idx, pipe, bg)

    return renders


def compose_frame(renders, label_height=30):
    columns = [
        renders["gt"],
        renders["posthoc_25"],
        renders["ours_25"],
        renders["ours_50"],
        renders["ours_100"],
    ]

    _, H, W = columns[0].shape
    bar = torch.ones((3, label_height, W), device=columns[0].device)
    labeled_columns = []
    for col in columns:
        labeled_columns.append(torch.cat([bar, col], dim=1))
    composite = torch.cat(labeled_columns, dim=2)
    return composite.detach()


def add_text_labels(frame_np, labels, label_height=30):
    import cv2
    h, w, _ = frame_np.shape
    col_w = w // len(labels)
    for i, label in enumerate(labels):
        x = i * col_w + 5
        y = label_height - 7
        cv2.putText(frame_np, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return frame_np


def process_scene(checkpoint_path, source_path, n_frames, output_dir, config_path=None):
    os.makedirs(output_dir, exist_ok=True)

    cfg_path = os.path.join(checkpoint_path, "cfg_args")
    args_ns = parse_cfg_args(cfg_path)

    iteration_dir = os.path.join(checkpoint_path, "point_cloud", "iteration_30000")
    ply_path = os.path.join(iteration_dir, "point_cloud.ply")
    deform_path = os.path.join(iteration_dir, "deformation.pth")

    sh_degree = getattr(args_ns, "sh_degree", 3)
    gaussians = GaussianModel(sh_degree, args_ns)
    gaussians._deformation = gaussians._deformation.to("cuda")
    gaussians.load_ply(ply_path)
    gaussians.load_model(iteration_dir)
    gaussians._deformation.eval()

    n_gaussians = gaussians._xyz.shape[0]
    print(f"  Loaded {n_gaussians} Gaussians")

    cameras, gt_images = build_camera_loader(source_path, n_frames)
    n_frames = len(cameras)
    print(f"  {n_frames} test frames")

    pipe = MiniPipeline()
    bg = torch.tensor([1.0, 1.0, 1.0] if getattr(args_ns, "white_background", True) else [0.0, 0.0, 0.0],
                      device="cuda")

    composite_frames = []
    for idx in range(n_frames):
        cam = cameras[idx]
        gt = gt_images[idx] if gt_images is not None else cam.original_image[0:3, :, :]
        print(f"  Frame {idx + 1}/{n_frames}", end="\r")
        renders = render_all_variants(gaussians, cam, gt, pipe, bg)
        composite = compose_frame(renders)
        composite_frames.append(composite)

    print(f"\n  Compositing {len(composite_frames)} frames into video...")

    frames_np = []
    for comp in composite_frames:
        np_frame = (comp.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        np_frame = np.ascontiguousarray(np_frame)
        np_frame = add_text_labels(np_frame, COLUMN_LABELS)
        frames_np.append(np_frame)

    video_path = os.path.join(output_dir, "comparison.mp4")
    imageio.mimwrite(video_path, frames_np, fps=15, quality=8)
    print(f"  Video saved: {video_path}")

    first_frame_path = os.path.join(output_dir, "comparison_frame0.png")
    imageio.imwrite(first_frame_path, frames_np[0])
    print(f"  Sample frame: {first_frame_path}")

    del gaussians
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="MGS comparison video renderer")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint root path")
    parser.add_argument("--source", type=str, help="Dataset source path (auto-detected if not provided)")
    parser.add_argument("--frames", type=int, default=30, help="Number of test frames")
    parser.add_argument("--output", type=str, default="results/videos", help="Output directory")
    parser.add_argument("--nerfds", action="store_true", help="NeRF-DS format")
    parser.add_argument("--all", action="store_true", help="Render all available scenes")
    parser.add_argument("--dnerf-only", action="store_true", help="Only D-NeRF scenes")
    parser.add_argument("--scene", type=str, help="Single scene name (with --all)")

    args = parser.parse_args()

    if args.all:
        scenes_to_render = []
        if not args.nerfds:
            for name in SCENE_CONFIGS:
                ckpt, src, cfg = resolve_checkpoint_and_source(name, nerfds=False)
                if os.path.exists(os.path.join(ckpt, "cfg_args")):
                    scenes_to_render.append((name, ckpt, src, cfg, False))
        if not args.dnerf_only:
            for name in NERFDS_SCENES:
                ckpt, src, cfg = resolve_checkpoint_and_source(name, nerfds=True)
                if os.path.exists(os.path.join(ckpt, "cfg_args")):
                    scenes_to_render.append((name, ckpt, src, cfg, True))

        if args.scene:
            scenes_to_render = [(n, c, s, cf, nd) for n, c, s, cf, nd in scenes_to_render if n == args.scene]

        print(f"Rendering {len(scenes_to_render)} scenes:")
        for name, _, _, _, nd in scenes_to_render:
            print(f"  - {name} ({'NeRF-DS' if nd else 'D-NeRF'})")

        for name, ckpt, src, cfg, nd in scenes_to_render:
            print(f"\n{'=' * 60}")
            print(f"Scene: {name}")
            print(f"Checkpoint: {ckpt}")
            out_dir = os.path.join(args.output, name)
            process_scene(ckpt, src, args.frames, out_dir, cfg)

    elif args.checkpoint:
        ckpt = args.checkpoint
        if args.source:
            src = args.source
        else:
            cfg_path = os.path.join(ckpt, "cfg_args")
            ns = parse_cfg_args(cfg_path)
            src = os.path.join(_BASE_DIR, getattr(ns, "source_path", ""))
        process_scene(ckpt, os.path.abspath(src), args.frames, args.output)

    else:
        parser.print_help()

    print("\nDone.")


if __name__ == "__main__":
    main()
