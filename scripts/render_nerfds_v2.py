import os, sys, torch, argparse, numpy as np
import torchvision
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from scene.gaussian_model import GaussianModel
from scene.hyper_loader import Load_hyper_data
from utils.camera_utils import loadCam
from gaussian_renderer import render
from utils.mgs_utils import compute_importance, get_sorted_indices
from arguments import PipelineParams
from utils.motion_scorer import parse_cfg_args

class Args:
    resolution = -1
    data_device = "cuda"

def main():
    ckpt = 'output/nerfds/plate_mgs'
    src = 'data/nerfds/plate_novel_view'
    
    cfg = parse_cfg_args(os.path.join(ckpt, "cfg_args"))
    iterdir = os.path.join(ckpt, "point_cloud", "iteration_15000")
    
    g = GaussianModel(getattr(cfg, 'sh_degree', 3), cfg)
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
    print(f"Gaussians: {N}")
    
    infos = Load_hyper_data(src, 0.5, False, split="test")
    cams = [loadCam(Args(), 0, ci, 1.0) for ci in infos]
    print(f"Cameras: {len(cams)}")
    
    pipe = PipelineParams(argparse.ArgumentParser())
    pipe.convert_SHs_python = False
    pipe.compute_cov3D_python = False
    bg = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    
    cam = cams[5]
    gt = cam.original_image[:3].cpu().clamp(0, 1)
    
    # Save GT
    torchvision.utils.save_image(gt, 'results/figures/nerfds_plate_gt.png')
    print("GT saved")
    
    # Full render
    r_full = render(cam, g, pipe, bg)["render"].cpu()
    torchvision.utils.save_image(r_full, 'results/figures/nerfds_plate_mgs_full.png')
    print("Full saved")
    
    # Compute importance
    imp = compute_importance(g, cam)
    sidx = get_sorted_indices(imp)
    
    # MGS 25%
    K = max(2, int(0.25 * N))
    sub = g.subset_view(sidx[:K])
    r25 = render(cam, sub, pipe, bg)["render"].cpu()
    torchvision.utils.save_image(r25, 'results/figures/nerfds_plate_mgs_25.png')
    print(f"25% saved ({K} Gaussians)")
    
    # MGS 50%
    K50 = max(2, int(0.5 * N))
    sub50 = g.subset_view(sidx[:K50])
    r50 = render(cam, sub50, pipe, bg)["render"].cpu()
    torchvision.utils.save_image(r50, 'results/figures/nerfds_plate_mgs_50.png')
    print(f"50% saved ({K50} Gaussians)")
    
    # Post-hoc random 25%
    sidx_r = torch.randperm(N, device='cuda')
    sub_r = g.subset_view(sidx_r[:K])
    rr = render(cam, sub_r, pipe, bg)["render"].cpu()
    torchvision.utils.save_image(rr, 'results/figures/nerfds_plate_posthoc_25.png')
    print("Post-hoc saved")
    
    # PSNR
    for label, img in [("Full", r_full), ("25%", r25), ("50%", r50), ("Rand25%", rr)]:
        mse = ((img - gt)**2).mean().item()
        psnr = 10 * np.log10(1/mse) if mse > 0 else 99
        print(f"  {label}: PSNR={psnr:.2f} dB")
    
    print("\nDone. Files: results/figures/nerfds_plate_*.png")

if __name__ == '__main__':
    main()
