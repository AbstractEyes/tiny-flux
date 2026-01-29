"""
TinyFlux Loss Functions

- Main velocity prediction loss (MSE or Huber)
- Lune distillation loss (cosine/huber/soft/hard)
- Sol distillation loss (stats + spatial)
- Min-SNR weighting
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    """
    Huber loss - L2 for small errors, L1 for large.

    More robust to outliers than pure MSE.
    """
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


def min_snr_weight(t: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    """
    Min-SNR weighting from "Efficient Diffusion Training via Min-SNR Weighting".

    Downweights high-SNR (easy) timesteps, upweights low-SNR (hard) timesteps.
    Clamped by gamma to prevent extreme weights.

    Args:
        t: [B] timesteps in [0, 1] (0=noise, 1=data for rectified flow)
        gamma: max SNR clamp value

    Returns:
        weights: [B] per-sample weights
    """
    # For rectified flow: t=0 is noise, t=1 is data
    # SNR = signal^2 / noise^2 = t^2 / (1-t)^2
    snr = (t / (1 - t).clamp(min=1e-5)).pow(2)
    return torch.clamp(snr, max=gamma) / snr.clamp(min=1e-5)


def compute_main_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        spatial_weights: Optional[torch.Tensor] = None,
        fg_weight: float = 2.0,
        bg_weight: float = 0.5,
        snr_weights: Optional[torch.Tensor] = None,
        use_huber: bool = False,
        huber_delta: float = 0.1,
) -> torch.Tensor:
    """
    Compute main velocity prediction loss.

    Args:
        pred: [B, N, C] predicted velocity
        target: [B, N, C] target velocity
        mask: [B, H, W] optional foreground mask
        spatial_weights: [B, 8, 8] optional Sol spatial importance
        fg_weight: foreground loss weight
        bg_weight: background loss weight
        snr_weights: [B] optional min-SNR weights
        use_huber: use Huber loss instead of MSE
        huber_delta: Huber loss delta

    Returns:
        loss: scalar
    """
    B, N, C = pred.shape

    # Per-element loss
    if use_huber:
        loss_per_elem = huber_loss(pred, target, huber_delta)
    else:
        loss_per_elem = (pred - target) ** 2

    # Apply Sol spatial weights if provided
    if spatial_weights is not None:
        H = W = int(math.sqrt(N))
        # Upsample from 8x8 to HxW
        spatial_up = F.interpolate(
            spatial_weights.unsqueeze(1),  # [B, 1, 8, 8]
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [B, H, W]
        # Normalize so mean = 1
        spatial_up = spatial_up / (spatial_up.mean(dim=[1, 2], keepdim=True) + 1e-6)
        spatial_flat = spatial_up.view(B, N, 1)
        loss_per_elem = loss_per_elem * spatial_flat

    # Apply foreground/background mask if provided
    if mask is not None:
        H = W = int(math.sqrt(N))
        mask_flat = mask.view(B, H * W, 1).to(pred.device)
        weights = mask_flat * fg_weight + (1 - mask_flat) * bg_weight
        loss_per_elem = loss_per_elem * weights

    # Reduce to per-sample
    loss_per_sample = loss_per_elem.mean(dim=[1, 2])  # [B]

    # Apply SNR weighting
    if snr_weights is not None:
        loss_per_sample = loss_per_sample * snr_weights

    return loss_per_sample.mean()


def compute_lune_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mode: str = "cosine",
        huber_delta: float = 0.1,
) -> torch.Tensor:
    """
    Compute Lune distillation loss.

    Args:
        pred: [B, 1280] predicted mid-block features
        target: [B, 1280] teacher mid-block features
        mode: "cosine", "huber", "soft", or "hard"
        huber_delta: delta for Huber loss

    Returns:
        loss: scalar
    """
    if mode == "cosine":
        # Cosine similarity loss: 1 - cos_sim
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        return (1 - (pred_norm * target_norm).sum(dim=-1)).mean()
    elif mode == "huber":
        return huber_loss(pred, target, huber_delta).mean()
    elif mode == "soft":
        # Soft L2 with temperature scaling
        return F.mse_loss(pred / 10.0, target / 10.0)
    else:  # "hard"
        return F.mse_loss(pred, target)


def compute_sol_loss(
        pred_stats: torch.Tensor,
        pred_spatial: torch.Tensor,
        target_stats: torch.Tensor,
        target_spatial: torch.Tensor,
        stats_weight: float = 1.0,
        spatial_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute Sol distillation loss.

    Args:
        pred_stats: [B, 3] predicted attention statistics
        pred_spatial: [B, 8, 8] predicted spatial importance
        target_stats: [B, 3] teacher statistics
        target_spatial: [B, 8, 8] teacher spatial importance
        stats_weight: weight for stats loss
        spatial_weight: weight for spatial loss

    Returns:
        loss: scalar
    """
    stats_loss = F.mse_loss(pred_stats, target_stats)
    spatial_loss = F.mse_loss(pred_spatial, target_spatial)
    return stats_weight * stats_loss + spatial_weight * spatial_loss


# =============================================================================
# Smoke Test
# =============================================================================

def _smoke_test():
    print("Losses Smoke Test")
    print("=" * 50)

    B, N, C = 4, 64, 16
    device = "cpu"

    # Test Huber
    print("\n[1] Huber loss...")
    pred = torch.randn(B, N, C)
    target = torch.randn(B, N, C)
    loss = huber_loss(pred, target, delta=0.1)
    assert loss.shape == pred.shape
    print(f"    shape: {loss.shape} ✓")

    # Test min-SNR
    print("\n[2] Min-SNR weighting...")
    t = torch.tensor([0.1, 0.3, 0.7, 0.9])
    w = min_snr_weight(t, gamma=5.0)
    assert w.shape == (4,)
    print(f"    t={t.tolist()}")
    print(f"    w={w.tolist()}")
    print(f"    (low t = high weight, high t = low weight) ✓")

    # Test main loss
    print("\n[3] Main loss...")
    pred = torch.randn(B, N, C)
    target = torch.randn(B, N, C)
    loss = compute_main_loss(pred, target)
    assert loss.dim() == 0
    print(f"    basic: {loss.item():.4f} ✓")

    # With SNR weights
    snr_w = min_snr_weight(torch.rand(B))
    loss_snr = compute_main_loss(pred, target, snr_weights=snr_w)
    print(f"    with SNR: {loss_snr.item():.4f} ✓")

    # With spatial weights
    spatial = torch.rand(B, 8, 8)
    loss_spatial = compute_main_loss(pred, target, spatial_weights=spatial)
    print(f"    with spatial: {loss_spatial.item():.4f} ✓")

    # Test Lune loss
    print("\n[4] Lune loss...")
    pred_lune = torch.randn(B, 1280)
    target_lune = torch.randn(B, 1280)
    for mode in ["cosine", "huber", "soft", "hard"]:
        loss = compute_lune_loss(pred_lune, target_lune, mode=mode)
        print(f"    {mode}: {loss.item():.4f}")
    print("    ✓")

    # Test Sol loss
    print("\n[5] Sol loss...")
    pred_stats = torch.randn(B, 3)
    pred_spatial = torch.randn(B, 8, 8)
    target_stats = torch.randn(B, 3)
    target_spatial = torch.randn(B, 8, 8)
    loss = compute_sol_loss(pred_stats, pred_spatial, target_stats, target_spatial)
    print(f"    loss: {loss.item():.4f} ✓")

    print("\n" + "=" * 50)
    print("✓ All tests passed")


if __name__ == "__main__":
    _smoke_test()