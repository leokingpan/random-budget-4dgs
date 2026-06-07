#!/usr/bin/env python3
"""Generate all 4 paper figures for Evo-4DGS TVC submission."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os, csv

FIGS = "/home/leo/evo-4dgs/base_4dgs/paper/TVC/new/tvc_final_submission/tvc_final_submission/figures"
RESULTS = "/home/leo/evo-4dgs/base_4dgs/results/mgs"
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
})

def read_csv(path):
    rows = {}
    if not os.path.exists(path): return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[float(r['budget'])] = float(r['PSNR'])
    return rows

# ===========================================================================
# FIG 2: Budget-quality curves — 9 scenes, 3×3 grid
# ===========================================================================
def fig2():
    scenes_ours = ['bouncingballs','standup','mutant','trex','jumpingjacks','hook','hellwarrior']
    scenes_nerfds = ['plate','press']
    all_scenes = scenes_ours + scenes_nerfds

    budgets = [0.1, 0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(3, 3, figsize=(6.5, 5.5))
    axes = axes.flatten()

    for idx, scene in enumerate(all_scenes):
        ax = axes[idx]
        pfx = f"nerfds_{scene}" if scene in scenes_nerfds else scene

        ours_data = read_csv(f"{RESULTS}/{pfx}_ours.csv")
        posthoc_data = read_csv(f"{RESULTS}/{pfx}_posthoc.csv")

        ours_psnr = [ours_data.get(b, None) for b in budgets]
        posthoc_psnr = [posthoc_data.get(b, None) for b in budgets]

        ax.plot(budgets, ours_psnr, 'o-', color='#2E86AB', linewidth=1.2, markersize=3, label='Ours')
        ax.plot(budgets, posthoc_psnr, 's--', color='#D64045', linewidth=1.0, markersize=3, label='Post-hoc pruning')

        title = scene
        if scene in scenes_nerfds:
            title = f'{scene} (real)'
        ax.set_title(title, fontsize=8, pad=2)
        ax.set_xlabel('Budget ratio', fontsize=7)
        ax.set_ylabel('PSNR (dB)', fontsize=7)
        ax.set_xlim(0.05, 1.05)
        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax.tick_params(pad=1)

        if idx == 0:
            ax.legend(framealpha=0.8, fontsize=6, loc='lower right')

    # Hide extra subplot if needed
    if len(all_scenes) < 9:
        for i in range(len(all_scenes), 9):
            axes[i].set_visible(False)

    fig.tight_layout(pad=1.0, h_pad=1.5, w_pad=1.5)
    fig.savefig(f"{FIGS}/fig2_budget_quality_curves.pdf")
    fig.savefig(f"{FIGS}/fig2_budget_quality_curves.png")
    plt.close()
    print("Fig 2 saved")

# ===========================================================================
# FIG 3: Visual comparison — bouncingballs + press, GT/Ours/Post-hoc at 25%
# ===========================================================================
def fig3():
    import torch, json as jmod, argparse as ap
    from PIL import Image
    from scene.gaussian_model import GaussianModel
    from scene.cameras import Camera
    from utils.motion_scorer import parse_cfg_args
    from utils.mgs_utils import compute_importance, compute_opacity_only, get_sorted_indices
    from utils.general_utils import PILtoTorch
    from gaussian_renderer import render
    from arguments import PipelineParams

    def build_cam_dnerf(src, frame, cam_angle_x, norm_time):
        matrix = np.linalg.inv(np.array(frame["transform_matrix"], dtype=np.float64))
        R = -np.transpose(matrix[:3, :3]); R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        img_path = os.path.join(src, frame["file_path"] + ".png")
        with Image.open(img_path) as pil_img:
            im_data = np.array(pil_img.convert("RGBA"), dtype=np.float64)
        bg_a = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        nd = im_data / 255.0
        arr = nd[:, :, :3] * nd[:, :, 3:4] + bg_a * (1.0 - nd[:, :, 3:4])
        rgb = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), "RGB")
        gt = PILtoTorch(rgb, (800, 800))
        cam = Camera(colmap_id=0, R=R, T=T, FoVx=float(cam_angle_x), FoVy=float(cam_angle_x),
                     image=gt, gt_alpha_mask=None, image_name="t", uid=0,
                     data_device=torch.device("cuda"), time=norm_time, mask=None)
        cam.original_image = gt.to("cuda")
        return cam

    def load_model(ckpt_path, it):
        cfg = parse_cfg_args(os.path.join(ckpt_path, "cfg_args"))
        gs = GaussianModel(getattr(cfg, "sh_degree", 3), cfg)
        gs._deformation = gs._deformation.to("cuda")
        itd = os.path.join(ckpt_path, "point_cloud", f"iteration_{it}")
        gs.load_ply(os.path.join(itd, "point_cloud.ply"))
        gs._deformation.load_state_dict(torch.load(os.path.join(itd, "deformation.pth"), map_location="cuda"), strict=False)
        gs._deformation.eval()
        dtp = os.path.join(itd, "deformation_table.pth")
        if os.path.exists(dtp): gs._deformation_table = torch.load(dtp, map_location="cuda")
        return gs, cfg

    def to_np(t):
        return t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    pipe = PipelineParams(ap.ArgumentParser())
    bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    BASE = "/home/leo/evo-4dgs/base_4dgs"

    # ---- bouncingballs ----
    gs_bb, cfg_bb = load_model(f"{BASE}/output/dnerf/bouncingballs_mgs", 30000)
    src_bb = os.path.abspath(os.path.join(BASE, cfg_bb.source_path))
    with open(os.path.join(src_bb, "transforms_test.json")) as f:
        tf_bb = jmod.load(f)
    all_times_bb = [fr["time"] for fr in tf_bb["frames"]]
    with open(os.path.join(src_bb, "transforms_train.json")) as f:
        all_times_bb.extend([fr["time"] for fr in jmod.load(f)["frames"]])
    max_t_bb = max(all_times_bb) if all_times_bb else 1.0
    cam_bb = build_cam_dnerf(src_bb, tf_bb["frames"][0], tf_bb["camera_angle_x"], tf_bb["frames"][0]["time"] / max_t_bb)
    N_bb = gs_bb._xyz.shape[0]; K25 = max(2, int(0.25 * N_bb))
    gt_bb = to_np(cam_bb.original_image[:3])
    imp_o = compute_importance(gs_bb, cam_bb); so = get_sorted_indices(imp_o)
    ours_bb = to_np(render(cam_bb, gs_bb.subset_view(so[:K25]), pipe, bg)["render"])
    imp_p = compute_opacity_only(gs_bb, cam_bb); sp = get_sorted_indices(imp_p)
    ph_bb = to_np(render(cam_bb, gs_bb.subset_view(sp[:K25]), pipe, bg)["render"])
    print(f"  bouncingballs: GT={gt_bb.mean():.3f} ours={ours_bb.mean():.3f} ph={ph_bb.mean():.3f}")

    # ---- press (NeRF-DS) ----
    gs_pr, cfg_pr = load_model(f"{BASE}/output/nerfds/press_mgs", 15000)
    src_pr = f"{BASE}/data/nerfds/press_novel_view"
    from scene.hyper_loader import Load_hyper_data
    from utils.camera_utils import loadCam as load_cam_func
    class FakeArgs: resolution=-1; data_device="cuda"
    hd = Load_hyper_data(src_pr, 0.5, use_bg_points=False, split="test")
    cam_info = hd[0]
    cam_pr = load_cam_func(FakeArgs(), 0, cam_info, 1.0)
    cam_pr.original_image = cam_pr.original_image.to("cuda")
    N_pr = gs_pr._xyz.shape[0]; K25p = max(2, int(0.25 * N_pr))
    gt_pr = to_np(cam_pr.original_image[:3])
    imp_o = compute_importance(gs_pr, cam_pr); so = get_sorted_indices(imp_o)
    ours_pr = to_np(render(cam_pr, gs_pr.subset_view(so[:K25p]), pipe, bg)["render"])
    imp_p = compute_opacity_only(gs_pr, cam_pr); sp = get_sorted_indices(imp_p)
    ph_pr = to_np(render(cam_pr, gs_pr.subset_view(sp[:K25p]), pipe, bg)["render"])
    print(f"  press: GT={gt_pr.mean():.3f} ours={ours_pr.mean():.3f} ph={ph_pr.mean():.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.5))
    axes[0, 0].imshow(gt_bb); axes[0, 0].set_title('Ground Truth', fontsize=8); axes[0, 0].axis('off')
    axes[0, 1].imshow(ours_bb); axes[0, 1].set_title('Ours (25%)', fontsize=8); axes[0, 1].axis('off')
    axes[0, 2].imshow(ph_bb); axes[0, 2].set_title('Post-hoc (25%)', fontsize=8); axes[0, 2].axis('off')
    axes[1, 0].imshow(gt_pr); axes[1, 0].set_title('Ground Truth', fontsize=8); axes[1, 0].axis('off')
    axes[1, 1].imshow(ours_pr); axes[1, 1].set_title('Ours (25%)', fontsize=8); axes[1, 1].axis('off')
    axes[1, 2].imshow(ph_pr); axes[1, 2].set_title('Post-hoc (25%)', fontsize=8); axes[1, 2].axis('off')
    axes[0, 0].set_ylabel('Bouncingballs', fontsize=8, fontweight='bold')
    axes[1, 0].set_ylabel('Press', fontsize=8, fontweight='bold')
    fig.tight_layout(pad=0.5, h_pad=0.3, w_pad=0.3)
    fig.savefig(f"{FIGS}/fig3_visual_comparison.pdf")
    fig.savefig(f"{FIGS}/fig3_visual_comparison.png")
    plt.close()
    print("Fig 3 saved")

# ===========================================================================
# FIG 4: Quality-speed trade-off — PSNR vs FPS scatter
# ===========================================================================
def fig4():
    budgets = [0.1, 0.25, 0.5, 0.75, 1.0]
    psnr_vals = [29.48, 33.63, 34.75, 34.95, 35.02]
    fps_vals = [119.2, 131.7, 118.3, 101.5, 89.7]

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    colors = plt.cm.viridis([0.15, 0.35, 0.55, 0.75, 0.92])
    for i, b in enumerate(budgets):
        ax.scatter(fps_vals[i], psnr_vals[i], c=[colors[i]], s=80, zorder=5, edgecolors='black', linewidth=0.5)
        offset_x = 2.5 if i != 1 else -12
        offset_y = 0.15 if i != 1 else -0.3
        ax.annotate(f'{int(b*100)}%', (fps_vals[i], psnr_vals[i]),
                     textcoords="offset points", xytext=(offset_x, offset_y),
                     fontsize=7, fontweight='bold')

    ax.plot(fps_vals, psnr_vals, '--', color='gray', linewidth=0.8, alpha=0.5, zorder=1)

    ax.set_xlabel('FPS', fontsize=9)
    ax.set_ylabel('PSNR (dB)', fontsize=9)

    ax.annotate('25% budget\n(131.7 FPS, 33.63 dB)',
                 xy=(131.7, 33.63), xytext=(105, 32.0),
                 arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                 fontsize=7, color='black',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

    ax.set_xlim(85, 140)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig4_quality_speed_tradeoff.pdf")
    fig.savefig(f"{FIGS}/fig4_quality_speed_tradeoff.png")
    plt.close()
    print("Fig 4 saved")

# ===========================================================================
# FIG 5: Importance score ablation — grouped bar chart
# ===========================================================================
def fig5():
    labels = ['Random', 'Canonical\nopacity', 'Deformed\nopacity',
              'Def. opacity\n× scale-area', 'Def. opacity\n× projected-area']
    psnr_10 = [20.71, 23.17, 24.11, 29.02, 27.83]
    psnr_25 = [26.00, 30.64, 30.95, 32.97, 32.43]

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bars1 = ax.bar(x - width/2, psnr_10, width, label='10% budget', color='#2E86AB', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, psnr_25, width, label='25% budget', color='#D64045', edgecolor='white', linewidth=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=6)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=6)

    ax.set_ylabel('Average PSNR (dB)', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=15, ha='right')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(18, 36)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig5_importance_score_ablation.pdf")
    fig.savefig(f"{FIGS}/fig5_importance_score_ablation.png")
    plt.close()
    print("Fig 5 saved")

# ===========================================================================
if __name__ == "__main__":
    fig2()
    fig4()
    fig5()
    print("Figs 2,4,5 done (matplotlib). Now Fig 3 (visual comparison, needs GPU)...")
    # Fig 3 needs GPU — run inside conda env with pytorch
    try:
        fig3()
        print("Fig 3 done")
    except Exception as e:
        print(f"Fig 3 error (GPU likely unavailable in this env): {e}")
    print(f"\nAll figures saved to {FIGS}/")
