import random
import torch
from utils.general_utils import build_scaling_rotation


@torch.no_grad()
def compute_importance(gaussians, viewpoint_cam):
    """Compute per-Gaussian importance scores as deformed_opacity × projected_area.

    I(G_i,t) = α̃_i(t) · s̃_{i,x}(t) · s̃_{i,y}(t)   (scale-area variant)

    Uses deformed opacity and canonical-scale area proxy.

    Args:
        gaussians: GaussianModel instance with _deformation, _xyz, _scaling,
                   _rotation, _opacity, get_features, opacity_activation,
                   scaling_activation.
        viewpoint_cam: Camera with a ``time`` attribute (scalar).

    Returns:
        Tensor[N]: importance score for each Gaussian.
    """
    N = gaussians._xyz.shape[0]
    time_emb = torch.tensor(viewpoint_cam.time, device="cuda").expand(N, 1)

    # Get deformed opacity from the deformation network
    _, _, _, opacity_def, _ = gaussians._deformation(
        gaussians._xyz,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
        gaussians.get_features,
        time_emb,
    )
    opacity_act = gaussians.opacity_activation(opacity_def).squeeze(-1)

    # Approximate projected area from canonical scales
    scales_exp = gaussians.scaling_activation(gaussians._scaling)
    projected_area = scales_exp[:, 0] * scales_exp[:, 1]

    return opacity_act * projected_area


@torch.no_grad()
def compute_opacity_only(gaussians, viewpoint_cam):
    """Compute importance using only deformed opacity (no area weighting).

    I(G_i,t) = α̃_i(t)
    """
    N = gaussians._xyz.shape[0]
    time_emb = torch.tensor(viewpoint_cam.time, device="cuda").expand(N, 1)

    _, _, _, opacity_def, _ = gaussians._deformation(
        gaussians._xyz,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
        gaussians.get_features,
        time_emb,
    )
    return gaussians.opacity_activation(opacity_def).squeeze(-1)


@torch.no_grad()
def compute_canonical_opacity(gaussians, viewpoint_cam):
    """Use canonical (pre-deformation) opacity only. Static MGS style.

    I(G_i) = α_i   (no deformation, no time dependence, no area weighting)

    This is the simplest meaningful ordering: sort by raw opacity
    without querying the deformation network.
    """
    return gaussians.get_opacity.squeeze(-1)


@torch.no_grad()
def compute_importance_projected_area(gaussians, viewpoint_cam):
    """Deformed opacity × projected 2D area using deformed covariance.

    I(G_i,t) = α̃_i(t) · √det(Σ_{2D,i}(t))

    Where Σ_{2D} is the upper-left 2×2 of the camera-space covariance
    computed from deformed scales + rotations, projected via the camera's
    world-to-view rotation.

    More accurate screen-space proxy than the scale-area approximation.
    """
    N = gaussians._xyz.shape[0]
    time_emb = torch.tensor(viewpoint_cam.time, device="cuda").expand(N, 1)

    # Full deformation forward pass
    means_def, scales_def, rots_def, opacity_def, _ = gaussians._deformation(
        gaussians._xyz,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
        gaussians.get_features,
        time_emb,
    )
    opacity_act = gaussians.opacity_activation(opacity_def).squeeze(-1)

    scales_act = gaussians.scaling_activation(scales_def)
    rots_act = gaussians.rotation_activation(rots_def)

    L = build_scaling_rotation(scales_act, rots_act)
    cov3D = L @ L.transpose(-1, -2)

    W2C_rot = viewpoint_cam.world_view_transform[:3, :3].to(cov3D.device)
    cov_cam = W2C_rot.unsqueeze(0) @ cov3D @ W2C_rot.unsqueeze(0).transpose(-1, -2)

    cov2D = cov_cam[:, :2, :2]

    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] * cov2D[:, 1, 0]
    area = torch.sqrt(torch.clamp(det, min=1e-12))

    return opacity_act * area


@torch.no_grad()
def compute_importance_appearance_aware(gaussians, viewpoint_cam, gamma=0.5):
    """Deformed opacity × scale-area × (1 + γ·E_SH) with SH energy term.

    I = α̃ · s̃_x · s̃_y · (1 + γ · E_SH)

    E_SH = sum(sh_coeff²) across all bands and color channels.
    gamma controls the weight of appearance information (0 = no SH, 1 = full).
    """
    N = gaussians._xyz.shape[0]
    time_emb = torch.tensor(viewpoint_cam.time, device="cuda").expand(N, 1)

    _, scales_def, _, opacity_def, shs_def = gaussians._deformation(
        gaussians._xyz,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
        gaussians.get_features,
        time_emb,
    )

    opacity_act = gaussians.opacity_activation(opacity_def).squeeze(-1)

    scales_exp = gaussians.scaling_activation(scales_def)
    projected_area = scales_exp[:, 0] * scales_exp[:, 1]

    sh_energy = (shs_def ** 2).sum(dim=(1, 2))  # [N]

    return opacity_act * projected_area * (1.0 + gamma * sh_energy)


@torch.no_grad()
def get_sorted_indices(importance, descending=True):
    """Return indices sorted by importance score.

    Args:
        importance: Tensor[N] of per-Gaussian importance.
        descending: If True (default), highest importance first.

    Returns:
        LongTensor[N]: sorted indices.
    """
    return torch.argsort(importance, descending=descending)


def sample_budget(N, min_ratio=0.1, max_ratio=1.0):
    """Randomly sample a budget K within [min_ratio*N, max_ratio*N].

    Args:
        N: Total number of Gaussians.
        min_ratio: Minimum fraction of N (default 0.1).
        max_ratio: Maximum fraction of N (default 1.0).

    Returns:
        int: Budget K, clamped to ≥ 2.
    """
    ratio = min_ratio + random.random() * (max_ratio - min_ratio)
    return max(2, int(ratio * N))


# Strategy dispatch table for importance score ablation experiments
STRATEGIES = {
    "opacity_area":       lambda g, c: compute_importance(g, c),
    "opacity":            lambda g, c: compute_opacity_only(g, c),
    "canonical_opacity":  lambda g, c: compute_canonical_opacity(g, c),
    "projected_area":     lambda g, c: compute_importance_projected_area(g, c),
    "appearance_aware":   lambda g, c: compute_importance_appearance_aware(g, c),
}
