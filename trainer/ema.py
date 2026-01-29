"""
Exponential Moving Average for TinyFlux

Maintains shadow weights that are updated each step:
    shadow = decay * shadow + (1 - decay) * model_weights

Use EMA weights for inference/sampling - they're smoother and higher quality.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any


class EMA:
    """
    Exponential Moving Average of model weights.

    Usage:
        ema = EMA(model, decay=0.9999)

        # During training
        for batch in dataloader:
            loss = train_step(model, batch)
            loss.backward()
            optimizer.step()
            ema.update(model)

        # For evaluation/sampling
        ema.apply_shadow(model)
        samples = model.generate(...)
        ema.restore(model)

        # Or use context manager
        with ema.average_parameters(model):
            samples = model.generate(...)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Initialize EMA with model's current weights.

        Args:
            model: model to track (can be compiled)
            decay: EMA decay factor (higher = slower update)
        """
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}

        # Handle torch.compile
        state = self._get_state_dict(model)
        for k, v in state.items():
            self.shadow[k] = v.clone().detach()

    def _get_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Get state dict, handling compiled models."""
        if hasattr(model, '_orig_mod'):
            return model._orig_mod.state_dict()
        return model.state_dict()

    def _load_state_dict(self, model: nn.Module, state: Dict[str, torch.Tensor]):
        """Load state dict, handling compiled models."""
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(state)
        else:
            model.load_state_dict(state)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update shadow weights with current model weights.

        shadow = decay * shadow + (1 - decay) * model
        """
        state = self._get_state_dict(model)
        for k, v in state.items():
            if k in self.shadow:
                # Use lerp for numerical stability
                self.shadow[k].lerp_(v.to(self.shadow[k].dtype), 1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """
        Replace model weights with EMA shadow weights.
        Saves current weights to backup for later restore.
        """
        self._backup = {k: v.clone() for k, v in self._get_state_dict(model).items()}
        self._load_state_dict(model, self.shadow)

    def restore(self, model: nn.Module):
        """Restore model weights from backup."""
        if self._backup:
            self._load_state_dict(model, self._backup)
            self._backup = {}

    class _EMAContext:
        """Context manager for temporary EMA weight application."""

        def __init__(self, ema: "EMA", model: nn.Module):
            self.ema = ema
            self.model = model

        def __enter__(self):
            self.ema.apply_shadow(self.model)
            return self.model

        def __exit__(self, *args):
            self.ema.restore(self.model)

    def average_parameters(self, model: nn.Module) -> _EMAContext:
        """Context manager for using EMA weights temporarily."""
        return self._EMAContext(self, model)

    def sync_from_model(self, model: nn.Module, pattern: Optional[str] = None):
        """
        Copy model weights to shadow (useful for new layers).

        Args:
            model: source model
            pattern: if provided, only sync keys containing this pattern
        """
        state = self._get_state_dict(model)
        synced = 0
        for k, v in state.items():
            if pattern is None or pattern in k:
                if k in self.shadow:
                    self.shadow[k] = v.clone().to(self.shadow[k].device)
                    synced += 1

        pattern_str = f" matching '{pattern}'" if pattern else ""
        print(f"  ✓ Synced EMA: {synced} weights{pattern_str}")

    def state_dict(self) -> Dict[str, Any]:
        """Get EMA state for checkpointing."""
        return {
            'shadow': self.shadow,
            'decay': self.decay,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load EMA state from checkpoint."""
        self.shadow = {k: v.clone() for k, v in state['shadow'].items()}
        self.decay = state.get('decay', self.decay)

    def load_shadow(
            self,
            shadow_state: Dict[str, torch.Tensor],
            model: Optional[nn.Module] = None,
    ) -> Dict[str, int]:
        """
        Load shadow weights from a checkpoint, handling missing/extra keys.

        Args:
            shadow_state: shadow weights from checkpoint
            model: optional model to initialize missing keys from

        Returns:
            stats: dict with 'loaded', 'skipped', 'initialized' counts
        """
        device = next(iter(self.shadow.values())).device if self.shadow else 'cuda'

        loaded = 0
        skipped = 0
        initialized = 0

        # Load matching keys
        for k, v in shadow_state.items():
            if k in self.shadow:
                self.shadow[k] = v.clone().to(device)
                loaded += 1
            else:
                skipped += 1

        # Initialize missing keys from model
        if model is not None:
            model_state = self._get_state_dict(model)
            for k in self.shadow:
                if k not in shadow_state and k in model_state:
                    self.shadow[k] = model_state[k].clone().to(device)
                    initialized += 1

        return {
            'loaded': loaded,
            'skipped': skipped,
            'initialized': initialized,
        }

    def to(self, device: str):
        """Move shadow weights to device."""
        for k in self.shadow:
            self.shadow[k] = self.shadow[k].to(device)
        return self

    def __repr__(self) -> str:
        n_params = sum(v.numel() for v in self.shadow.values())
        return f"EMA(decay={self.decay}, params={n_params:,})"


# =============================================================================
# Smoke Test
# =============================================================================

def _smoke_test():
    print("EMA Smoke Test")
    print("=" * 50)

    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )

    print("\n[1] Initialize EMA...")
    ema = EMA(model, decay=0.9999)
    print(f"    {ema}")

    # Check shadow matches model initially
    for k, v in model.state_dict().items():
        assert torch.equal(v, ema.shadow[k])
    print("    shadow matches model ✓")

    print("\n[2] Update after training step...")
    # Simulate training step
    x = torch.randn(4, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Manually update weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # Update EMA
    old_shadow = {k: v.clone() for k, v in ema.shadow.items()}
    ema.update(model)

    # Check shadow moved toward model but not equal
    for k in ema.shadow:
        assert not torch.equal(ema.shadow[k], old_shadow[k]), "Shadow should change"
        assert not torch.equal(ema.shadow[k], model.state_dict()[k]), "Shadow != model"
    print("    shadow updated (not equal to model) ✓")

    print("\n[3] Apply/restore shadow...")
    original_weight = model.state_dict()['0.weight'].clone()

    ema.apply_shadow(model)
    shadow_weight = model.state_dict()['0.weight'].clone()
    assert not torch.equal(original_weight, shadow_weight)
    print("    applied shadow weights ✓")

    ema.restore(model)
    restored_weight = model.state_dict()['0.weight']
    assert torch.equal(original_weight, restored_weight)
    print("    restored original weights ✓")

    print("\n[4] Context manager...")
    with ema.average_parameters(model):
        ctx_weight = model.state_dict()['0.weight']
        assert torch.equal(ctx_weight, shadow_weight)
    after_weight = model.state_dict()['0.weight']
    assert torch.equal(after_weight, original_weight)
    print("    context manager works ✓")

    print("\n[5] State dict save/load...")
    state = ema.state_dict()
    ema2 = EMA(model, decay=0.999)  # Different decay
    ema2.load_state_dict(state)
    assert ema2.decay == 0.9999  # Should load decay
    for k in ema.shadow:
        assert torch.equal(ema.shadow[k], ema2.shadow[k])
    print("    save/load ✓")

    print("\n" + "=" * 50)
    print("✓ All tests passed")


if __name__ == "__main__":
    _smoke_test()