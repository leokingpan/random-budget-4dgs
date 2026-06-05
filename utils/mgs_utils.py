import random
import torch


@torch.no_grad()
def compute_importance(gaussians, viewpoint_cam):
    """Compute per-Gaussian importance scores as deformed_opacity × projected_area.

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
    """Compute importance using only deformed opacity (no area weighting)."""
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


# Strategy dispatch table for future ablation experiments
STRATEGIES = {
    "opacity_area": lambda g, c: compute_importance(g, c),  # default: opacity × area
    "opacity": lambda g, c: compute_opacity_only(g, c),
}
