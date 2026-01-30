"""
TinyFlux Schedules

- Timestep sampling (logit-normal with flux shift)
- Flux shift transformation
- Loss weight warmup schedules
- Learning rate schedules
"""

import math
import torch
from typing import Optional


def flux_shift(t: torch.Tensor, shift: float = 3.0) -> torch.Tensor:
    """
    Flux timestep shift - biases sampling toward higher timesteps.

    From Flux/SD3: shifts the uniform distribution to spend more time
    on the "interesting" middle-to-late timesteps where detail emerges.

    Args:
        t: [B] timesteps in [0, 1]
        shift: shift factor (higher = more bias toward t=1)

    Returns:
        t_shifted: [B] shifted timesteps
    """
    return shift * t / (1 + (shift - 1) * t)


def flux_unshift(t: torch.Tensor, shift: float = 3.0) -> torch.Tensor:
    """Inverse of flux_shift."""
    return t / (shift - (shift - 1) * t)


def sample_timesteps(
    batch_size: int,
    device: str = "cuda",
    shift: float = 3.0,
    min_t: float = 1e-4,
    max_t: float = 1 - 1e-4,
    logit_normal: bool = True,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample timesteps for flow matching training.

    Args:
        batch_size: number of timesteps to sample
        device: torch device
        shift: flux shift factor (higher = more bias toward t=1)
        min_t: minimum timestep
        max_t: maximum timestep
        logit_normal: if True, use logit-normal; if False, uniform
        logit_mean: mean for logit-normal distribution
        logit_std: std for logit-normal distribution
        dtype: output dtype

    Returns:
        t: [B] timesteps in [min_t, max_t]
    """
    if logit_normal:
        # Logit-normal: sigmoid(mean + std * N(0,1))
        t = torch.sigmoid(logit_mean + logit_std * torch.randn(batch_size, device=device))
    else:
        # Uniform
        t = torch.rand(batch_size, device=device)

    # Flux shift
    t = flux_shift(t, shift=shift)

    # Clamp to valid range
    t = t.clamp(min_t, max_t).to(dtype)

    return t


def sample_timesteps_uniform(
    batch_size: int,
    device: str = "cuda",
    min_t: float = 1e-4,
    max_t: float = 1 - 1e-4,
) -> torch.Tensor:
    """Sample timesteps uniformly (no shift)."""
    return torch.rand(batch_size, device=device) * (max_t - min_t) + min_t


def sample_timesteps_stratified(
    batch_size: int,
    device: str = "cuda",
    shift: float = 3.0,
    min_t: float = 1e-4,
    max_t: float = 1 - 1e-4,
) -> torch.Tensor:
    """
    Stratified timestep sampling - ensures coverage across range.

    Divides [0,1] into batch_size bins and samples one from each.
    """
    # Stratified uniform
    bins = torch.arange(batch_size, device=device, dtype=torch.float32) / batch_size
    noise = torch.rand(batch_size, device=device) / batch_size
    t = bins + noise

    # Apply shift
    t = flux_shift(t, shift=shift)

    return t.clamp(min_t, max_t)


# =============================================================================
# Loss Weight Schedules
# =============================================================================

def linear_warmup(step: int, warmup_steps: int, target_weight: float) -> float:
    """
    Linear warmup from 0 to target_weight.

    Args:
        step: current training step
        warmup_steps: number of warmup steps
        target_weight: final weight after warmup

    Returns:
        weight: current weight
    """
    if step < warmup_steps:
        return target_weight * (step / warmup_steps)
    return target_weight


def get_lune_weight(
    step: int,
    warmup_steps: int = 1000,
    target_weight: float = 0.1,
) -> float:
    """Get Lune distillation loss weight with warmup."""
    return linear_warmup(step, warmup_steps, target_weight)


def get_sol_weight(
    step: int,
    warmup_steps: int = 2000,
    target_weight: float = 0.05,
) -> float:
    """Get Sol distillation loss weight with warmup."""
    return linear_warmup(step, warmup_steps, target_weight)


# =============================================================================
# Learning Rate Schedules
# =============================================================================

def cosine_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int = 1000,
    min_lr_ratio: float = 0.0,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Args:
        step: current step
        total_steps: total training steps
        warmup_steps: linear warmup steps
        min_lr_ratio: minimum LR as ratio of peak LR

    Returns:
        lr_multiplier: multiply base LR by this
    """
    if step < warmup_steps:
        return step / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

    return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay


def constant_with_warmup(
    step: int,
    warmup_steps: int = 1000,
) -> float:
    """Constant LR with linear warmup."""
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0


def linear_decay(
    step: int,
    total_steps: int,
    warmup_steps: int = 1000,
    min_lr_ratio: float = 0.0,
) -> float:
    """Linear decay with warmup."""
    if step < warmup_steps:
        return step / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(min_lr_ratio, 1 - progress)


# =============================================================================
# Schedule Functions for Optimizer
# =============================================================================

def make_cosine_schedule(total_steps: int, warmup_steps: int = 1000):
    """Create a cosine schedule function for LambdaLR."""
    def schedule(step):
        return cosine_schedule(step, total_steps, warmup_steps)
    return schedule


def make_constant_schedule(warmup_steps: int = 1000):
    """Create a constant schedule function for LambdaLR."""
    def schedule(step):
        return constant_with_warmup(step, warmup_steps)
    return schedule


# =============================================================================
# Smoke Test
# =============================================================================

def _smoke_test():
    print("Schedules Smoke Test")
    print("=" * 50)

    B = 8
    device = "cpu"

    # Test flux shift
    print("\n[1] Flux shift...")
    t = torch.linspace(0, 1, 5)
    t_shifted = flux_shift(t, shift=3.0)
    t_unshifted = flux_unshift(t_shifted, shift=3.0)
    print(f"    original:  {t.tolist()}")
    print(f"    shifted:   {[f'{x:.3f}' for x in t_shifted.tolist()]}")
    print(f"    unshifted: {[f'{x:.3f}' for x in t_unshifted.tolist()]}")
    assert torch.allclose(t, t_unshifted, atol=1e-5)
    print("    roundtrip ✓")

    # Test timestep sampling
    print("\n[2] Timestep sampling...")
    t = sample_timesteps(B, device)
    assert t.shape == (B,)
    assert (t >= 1e-4).all() and (t <= 1-1e-4).all()
    print(f"    logit-normal (μ=0, σ=1) + shift: {[f'{x:.3f}' for x in t.tolist()]}")

    t_biased = sample_timesteps(B, device, logit_mean=0.5, logit_std=0.5)
    print(f"    logit-normal (μ=0.5, σ=0.5): {[f'{x:.3f}' for x in t_biased.tolist()]}")

    t_uniform = sample_timesteps(B, device, logit_normal=False)
    print(f"    uniform: {[f'{x:.3f}' for x in t_uniform.tolist()]}")

    t_strat = sample_timesteps_stratified(B, device)
    print(f"    stratified: {[f'{x:.3f}' for x in sorted(t_strat.tolist())]}")
    print("    ✓")

    # Test warmup schedules
    print("\n[3] Loss weight warmup...")
    for step in [0, 500, 1000, 2000, 3000]:
        lune_w = get_lune_weight(step, warmup_steps=1000, target_weight=0.1)
        sol_w = get_sol_weight(step, warmup_steps=2000, target_weight=0.05)
        print(f"    step {step:4d}: lune={lune_w:.3f}, sol={sol_w:.4f}")
    print("    ✓")

    # Test LR schedules
    print("\n[4] LR schedules...")
    total = 10000
    warmup = 1000
    for step in [0, 500, 1000, 5000, 10000]:
        cos = cosine_schedule(step, total, warmup)
        const = constant_with_warmup(step, warmup)
        lin = linear_decay(step, total, warmup)
        print(f"    step {step:5d}: cosine={cos:.3f}, const={const:.3f}, linear={lin:.3f}")
    print("    ✓")

    print("\n" + "=" * 50)
    print("✓ All tests passed")


if __name__ == "__main__":
    _smoke_test()