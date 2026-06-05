import os, sys, torch, json, numpy as np
import argparse
import torchvision
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene.gaussian_model import GaussianModel
from scene.hyper_loader import Load_hyper_data
from utils.camera_utils import loadCam
from gaussian_renderer import render
from utils.mgs_utils import compute_importance, get_sorted_indices
from arguments import PipelineParams
from utils.motion_scorer import parse_cfg_args

class Args: resolution=-1; data_device="cuda"

def main():
    src = 'data/nerfds/plate_novel_view'
    ckpt = 'output/nerfds/plate_mgs'
    
    # Load model
    cfg = parse_cfg_args(os.path.join(ckpt, "cfg_args"))
    iterdir = os.path.join(ckpt, "point_cloud", "iteration_15000")
    if not os.path.exists(iterdir):
        iterdir = os.path.join(ckpt, "point_cloud", "iteration_30000")
    g = GaussianModel(getattr(cfg,'sh_degree',3), cfg)
    g._deformation = g._deformation.to("cuda")
    g.load_ply(os.path.join(iterdir, "point_cloud.ply"))
    dp = os.path.join(iterdir, "deformation.pth")
    if os.path.exists(dp):
        g._deformation.load_state_dict(torch.load(dp, map_location="cuda"), strict=False)
    g._deformation.eval()
    N = g._xyz.shape[0]
    print(f"Loaded {N} Gaussians")
    
    # Load cameras
    infos = Load_hyper_data(src, 0.5, False, split="test")
    cams = [loadCam(Args(), 0, ci, 1.0) for ci in infos]
    print(f"Loaded {len(cams)} test cameras")
    
    pipe = PipelineParams(argparse.ArgumentParser())
    pipe.convert_SHs_python = False; pipe.compute_cov3D_python = False
    bg = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    
    os.makedirs('results/figures', exist_ok=True)
    cam = cams[5]
    
    # GT
    gt = cam.original_image[:3].cpu().clamp(0,1)
    torchvision.utils.save_image(gt, 'results/figures/nerfds_plate_gt.png')
    print("GT saved")
    
    # Full
    r = render(cam, g, pipe, bg)["render"].cpu()
    torchvision.utils.save_image(r, 'results/figures/nerfds_plate_mgs_full.png')
    print("Full saved")
    
    # MGS 25%
    imp = compute_importance(g, cam)
    sidx = get_sorted_indices(imp)
    K = max(2, int(0.25 * N))
    sub = g.subset_view(sidx[:K])
    r25 = render(cam, sub, pipe, bg)["render"].cpu()
    torchvision.utils.save_image(r25, 'results/figures/nerfds_plate_mgs_25.png')
    print("25% saved")
    
    # Post-hoc random 25%
    sidx_r = torch.randperm(N, device='cuda')
    sub_r = g.subset_view(sidx_r[:K])
    rr = render(cam, sub_r, pipe, bg)["render"].cpu()
    torchvision.utils.save_image(rr, 'results/figures/nerfds_plate_posthoc_25.png')
    print("Post-hoc 25% saved")
    
    print(f"\nFiles: results/figures/nerfds_plate_*.png")

if __name__ == '__main__':
    main()
