"""Dynamic Gaussian point selection for the study rasterizer buffer ABI."""
from pathlib import Path
import math
import torch
import numpy as np
from diff_gaussian_rasterization import _C

@torch.no_grad()
def buffers(g,c,bg):
 n=len(g._xyz);t=torch.full((n,1),float(c.time),device='cuda')
 xyz,scale,rot,op,sh=g._deformation(g._xyz,g._scaling,g._rotation,g._opacity,g.get_features,t)
 empty=torch.empty(0,device='cuda',dtype=torch.float32)
 out=_C.rasterize_gaussians(bg,xyz.contiguous(),empty,g.opacity_activation(op).contiguous(),g.scaling_activation(scale).contiguous(),g.rotation_activation(rot).contiguous(),1.,empty,c.world_view_transform.cuda(),c.full_proj_transform.cuda(),math.tan(c.FoVx*.5),math.tan(c.FoVy*.5),int(c.image_height),int(c.image_width),sh.contiguous(),g.active_sh_degree,c.camera_center.cuda(),False,False)
 return out
def check_args(geom,bin,img,bg,n,w,h):
 assert n>0 and w>0 and h>0
 assert bg.shape==(3,) and bg.dtype==torch.float32
 assert all(x.is_cuda and x.is_contiguous() and x.device==bg.device for x in [geom,bin,img,bg])
 assert all(x.dtype==torch.uint8 for x in [geom,bin,img])
 assert all(x.data_ptr()%128==0 for x in [geom,bin,img])
 assert 0<n<10000000 and w*h<100000000

def load_scoring_extension():
    from torch.utils.cpp_extension import load
    return load(name="source_ranking_moments_v3",
                sources=[str(Path(__file__).with_name("moments_v3.cu"))],
                extra_cuda_cflags=["-O3"])


@torch.no_grad()
def mean_scores(model, cameras, background, extension):
    """Mean per-point RGB deletion MSE across the supplied calibration observations."""
    n = len(model._xyz)
    if not cameras:
        raise ValueError("Calibration requires at least one camera.")
    scores = torch.zeros(n, device=background.device, dtype=torch.float32)
    for cam in cameras:
        w, h = int(cam.image_width), int(cam.image_height)
        _, _, _, _, geom, bins, image = buffers(model, cam, background)
        check_args(geom, bins, image, background, n, w, h)
        values, _ = extension.score_mean(geom, bins, image, background, n, w, h, 1.)
        scores += values[0] / len(cameras)
    if not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Invalid scores from rasterizer.")
    return scores.cpu().numpy()


def rank_candidates(scores, opacity_order=None, fraction=1.0):
    """Return eligible point IDs in score order. Ties use ascending canonical ID."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1].")
    if fraction == 1:
        ids = np.arange(len(scores))
    else:
        if opacity_order is None or len(opacity_order) != len(scores):
            raise ValueError("Pass the full, descending opacity order.")
        ids = np.asarray(opacity_order)[:round(len(scores) * fraction)]
    return ids[np.lexsort((ids, -scores[ids]))]


@torch.no_grad()
def four_pass_order(model, cameras, background, extension):
    """Return ranked canonical IDs for the final 20% pool used for both endpoints."""
    n = len(model._xyz)
    ids = np.arange(n)
    for fraction in (0.60, 0.35, 0.20, None):
        subset = model if len(ids) == n else model.subset_view(torch.from_numpy(ids).to(background.device))
        scores = mean_scores(subset, cameras, background, extension)
        ranked = ids[np.lexsort((ids, -scores))]
        if fraction is None:
            return ranked
        ids = np.sort(ranked[:round(fraction * n)])
