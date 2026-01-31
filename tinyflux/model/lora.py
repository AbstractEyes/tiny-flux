"""
TinyFlux LoRA Module

Flexible low-rank adaptation with per-layer configuration.

Usage:
    from tinyflux.model.lora import TinyFluxLoRA
    from tinyflux.model.lora_config import TinyFluxLoRAConfig

    # From preset
    lora = TinyFluxLoRA.from_preset(model, "character")

    # From config
    config = TinyFluxLoRAConfig.load("my_config.json")
    lora = TinyFluxLoRA(model, config)

    # Quick usage
    lora = TinyFluxLoRA(model, rank=16, include_single_attn=True)
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Iterator, Any, Tuple

from .lora_config import (
    TinyFluxLoRAConfig,
    LoRALayerSpec,
    LoRADefaults,
    LoRATarget,
    BlockExtensions,
    resolve_layer,
    set_layer,
    get_tinyflux_layer_paths,
)


# =============================================================================
# Stacked LoRA Linear
# =============================================================================

class StackedLoRALinear(nn.Module):
    """
    LoRA adapter with support for stacking (depth > 1).

    For depth=1: y = Wx + (BA)x * scale  (standard LoRA)
    For depth=N: y = Wx + sum_i(B_i @ A_i)x * scale  (stacked)

    Stacking allows for deeper adaptation of critical layers.
    """

    def __init__(
        self,
        base: nn.Linear,
        spec: LoRALayerSpec,
    ):
        super().__init__()

        self.base = base
        self.rank = spec.rank
        self.alpha = spec.alpha
        self.depth = spec.depth
        self.scale = spec.alpha / spec.rank
        self.lr_scale = spec.lr_scale

        self.dropout = nn.Dropout(spec.dropout) if spec.dropout > 0 else nn.Identity()

        # Freeze base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        device = base.weight.device
        dtype = base.weight.dtype

        # Stacked LoRA matrices
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.zeros(spec.rank, base.in_features, device=device, dtype=dtype))
            for _ in range(spec.depth)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(base.out_features, spec.rank, device=device, dtype=dtype))
            for _ in range(spec.depth)
        ])

        # Initialize: A with kaiming, B with zeros (starts as identity)
        for A in self.lora_A:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))

        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        if self._merged:
            return base_out

        x_lora = self.dropout(x).to(self.lora_A[0].dtype)
        lora_out = torch.zeros_like(base_out)

        for A, B in zip(self.lora_A, self.lora_B):
            lora_out = lora_out + (x_lora @ A.T @ B.T).to(base_out.dtype)

        return base_out + lora_out * self.scale

    def merge(self):
        """Merge LoRA into base weights."""
        if not self._merged:
            for A, B in zip(self.lora_A, self.lora_B):
                delta = (B @ A) * self.scale
                self.base.weight.data += delta.to(self.base.weight.dtype)
            self._merged = True

    def unmerge(self):
        """Remove LoRA from base weights."""
        if self._merged:
            for A, B in zip(self.lora_A, self.lora_B):
                delta = (B @ A) * self.scale
                self.base.weight.data -= delta.to(self.base.weight.dtype)
            self._merged = False

    @property
    def merged(self) -> bool:
        return self._merged

    @property
    def num_params(self) -> int:
        return sum(A.numel() + B.numel() for A, B in zip(self.lora_A, self.lora_B))

    def get_lora_params(self) -> Iterator[nn.Parameter]:
        """Yield all LoRA parameters."""
        for A in self.lora_A:
            yield A
        for B in self.lora_B:
            yield B


# =============================================================================
# Main TinyFluxLoRA
# =============================================================================

class TinyFluxLoRA(nn.Module):
    """
    Flexible LoRA for TinyFlux with per-layer configuration.

    Features:
    - Per-layer rank, alpha, learning rate
    - Stacked LoRA for deeper adaptation
    - Block extensions (new trainable blocks)
    - Config serialization for reproducibility

    Examples:
        # From preset
        lora = TinyFluxLoRA.from_preset(model, "character")

        # Custom config
        config = TinyFluxLoRAConfig(
            defaults=LoRADefaults(rank=16),
            targets=[
                LoRATarget("single_blocks.[20-24].*", rank=32, lr_scale=2.0),
            ],
        )
        lora = TinyFluxLoRA(model, config)

        # Quick usage
        lora = TinyFluxLoRA(model, rank=32, include_single_mlp=False)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TinyFluxLoRAConfig] = None,
        # Quick config options (used if config is None)
        rank: int = 16,
        alpha: Optional[float] = None,
        include_double_attn: bool = False,
        include_single_attn: bool = True,
        include_single_mlp: bool = True,
    ):
        super().__init__()

        # Build config if not provided
        if config is None:
            config = TinyFluxLoRAConfig(
                defaults=LoRADefaults(rank=rank, alpha=alpha or float(rank)),
                include_double_attn=include_double_attn,
                include_single_attn=include_single_attn,
                include_single_mlp=include_single_mlp,
            )

        self.config = config
        self._model = model

        # Get model dimensions
        self._num_double = len(model.double_blocks) if hasattr(model, 'double_blocks') else 0
        self._num_single = len(model.single_blocks) if hasattr(model, 'single_blocks') else 0

        # Storage
        self.adapters: nn.ModuleDict = nn.ModuleDict()
        self._layer_paths: Dict[str, str] = {}  # safe_key -> original_path
        self._layer_specs: Dict[str, LoRALayerSpec] = {}

        # Extension blocks
        self._extra_double_blocks: nn.ModuleList = nn.ModuleList()
        self._extra_single_blocks: nn.ModuleList = nn.ModuleList()

        # Inject adapters
        self._inject_adapters(model)

        # Setup extensions if configured
        if config.extensions:
            self._setup_extensions(model)

        # Freeze base, unfreeze LoRA
        self._setup_requires_grad(model)

        # Print summary
        self._print_summary()

    @classmethod
    def from_preset(
        cls,
        model: nn.Module,
        preset: str,
        **kwargs,
    ) -> 'TinyFluxLoRA':
        """Create LoRA from a preset configuration."""
        config = TinyFluxLoRAConfig.from_preset(preset, **kwargs)
        return cls(model, config)

    @classmethod
    def from_config_file(
        cls,
        model: nn.Module,
        path: str,
    ) -> 'TinyFluxLoRA':
        """Create LoRA from a JSON config file."""
        config = TinyFluxLoRAConfig.load(path)
        return cls(model, config)

    def _inject_adapters(self, model: nn.Module):
        """Inject LoRA adapters into model layers."""
        active = self.config.get_active_layers(self._num_double, self._num_single)

        for path, spec in active.items():
            try:
                # Get the original layer
                layer = resolve_layer(model, path)

                if not isinstance(layer, nn.Linear):
                    print(f"Warning: {path} is not Linear, skipping")
                    continue

                # Create adapter
                adapter = StackedLoRALinear(layer, spec)

                # Register with safe key
                safe_key = path.replace('.', '_')
                self.adapters[safe_key] = adapter
                self._layer_paths[safe_key] = path
                self._layer_specs[path] = spec

                # Inject into model
                set_layer(model, path, adapter)

            except Exception as e:
                print(f"Warning: Failed to inject {path}: {e}")

    def _setup_extensions(self, model: nn.Module):
        """Setup block extensions."""
        ext = self.config.extensions
        if not ext:
            return

        # Import block classes
        from .model import DoubleStreamBlock, SingleStreamBlock, TinyFluxConfig

        model_config = model.config if hasattr(model, 'config') else TinyFluxConfig()

        # Extra double blocks
        for i in range(ext.double_blocks):
            block = DoubleStreamBlock(model_config)
            if ext.double_init == "last" and self._num_double > 0:
                block.load_state_dict(model.double_blocks[-1].state_dict())
            self._extra_double_blocks.append(block)

        # Extra single blocks
        for i in range(ext.single_blocks):
            block = SingleStreamBlock(model_config)
            if ext.single_init == "last" and self._num_single > 0:
                block.load_state_dict(model.single_blocks[-1].state_dict())
            self._extra_single_blocks.append(block)

    def _setup_requires_grad(self, model: nn.Module):
        """Freeze base model, unfreeze LoRA and extensions."""
        # Freeze all
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA
        for adapter in self.adapters.values():
            for param in adapter.get_lora_params():
                param.requires_grad = True

        # Unfreeze extensions
        for block in self._extra_double_blocks:
            for param in block.parameters():
                param.requires_grad = True
        for block in self._extra_single_blocks:
            for param in block.parameters():
                param.requires_grad = True

    def _print_summary(self):
        """Print LoRA summary."""
        lora_params = sum(a.num_params for a in self.adapters.values())
        ext_params = sum(p.numel() for p in self._extra_double_blocks.parameters())
        ext_params += sum(p.numel() for p in self._extra_single_blocks.parameters())

        double_count = sum(1 for p in self._layer_paths.values() if 'double' in p)
        single_count = len(self.adapters) - double_count

        depths = set(s.depth for s in self._layer_specs.values())
        lr_scales = set(s.lr_scale for s in self._layer_specs.values())

        print(f"\n[TinyFluxLoRA] {self.config.name}")
        print(f"  Adapters: {len(self.adapters)} (double: {double_count}, single: {single_count})")

        if len(depths) > 1 or 1 not in depths:
            print(f"  Depths: {sorted(depths)}")
        if len(lr_scales) > 1:
            print(f"  LR scales: {sorted(lr_scales)}")

        print(f"  LoRA params: {lora_params:,}")

        if ext_params > 0:
            print(f"  Extension params: {ext_params:,}")
            print(f"    Extra double: {len(self._extra_double_blocks)}")
            print(f"    Extra single: {len(self._extra_single_blocks)}")

        total = lora_params + ext_params
        print(f"  Total trainable: {total:,} ({total * 4 / 1024 / 1024:.2f} MB)")

    # =========================================================================
    # Parameter Access
    # =========================================================================

    def parameters(self) -> Iterator[nn.Parameter]:
        """Yield all trainable parameters."""
        for adapter in self.adapters.values():
            yield from adapter.get_lora_params()

        for block in self._extra_double_blocks:
            yield from block.parameters()
        for block in self._extra_single_blocks:
            yield from block.parameters()

    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """
        Get optimizer param groups with per-layer learning rates.

        Usage:
            groups = lora.get_param_groups(base_lr=1e-4)
            optimizer = torch.optim.AdamW(groups)
        """
        lr_groups = self.config.get_lr_groups(base_lr, self._num_double, self._num_single)

        param_groups = []
        for effective_lr, paths in lr_groups.items():
            params = []
            for path in paths:
                safe_key = path.replace('.', '_')
                if safe_key in self.adapters:
                    params.extend(self.adapters[safe_key].get_lora_params())

            if params:
                param_groups.append({
                    'params': list(params),
                    'lr': effective_lr,
                    'name': f'lr_{effective_lr:.2e}',
                })

        # Extensions at base LR
        ext_params = list(self._extra_double_blocks.parameters()) + \
                     list(self._extra_single_blocks.parameters())
        if ext_params:
            param_groups.append({
                'params': ext_params,
                'lr': base_lr,
                'name': 'extensions',
            })

        return param_groups

    # =========================================================================
    # State Dict
    # =========================================================================

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get LoRA weights."""
        state = {}

        for key, adapter in self.adapters.items():
            for i, (A, B) in enumerate(zip(adapter.lora_A, adapter.lora_B)):
                if adapter.depth == 1:
                    state[f"{key}.lora_A"] = A.data
                    state[f"{key}.lora_B"] = B.data
                else:
                    state[f"{key}.lora_A.{i}"] = A.data
                    state[f"{key}.lora_B.{i}"] = B.data

        # Extensions
        for i, block in enumerate(self._extra_double_blocks):
            for k, v in block.state_dict().items():
                state[f"ext_double_{i}.{k}"] = v
        for i, block in enumerate(self._extra_single_blocks):
            for k, v in block.state_dict().items():
                state[f"ext_single_{i}.{k}"] = v

        return state

    def load_state_dict(self, state: Dict[str, torch.Tensor], strict: bool = True):
        """Load LoRA weights."""
        for key, adapter in self.adapters.items():
            for i, (A, B) in enumerate(zip(adapter.lora_A, adapter.lora_B)):
                if adapter.depth == 1:
                    a_key, b_key = f"{key}.lora_A", f"{key}.lora_B"
                else:
                    a_key, b_key = f"{key}.lora_A.{i}", f"{key}.lora_B.{i}"

                if a_key in state:
                    A.data.copy_(state[a_key])
                elif strict:
                    raise KeyError(f"Missing: {a_key}")

                if b_key in state:
                    B.data.copy_(state[b_key])
                elif strict:
                    raise KeyError(f"Missing: {b_key}")

        # Extensions
        for i, block in enumerate(self._extra_double_blocks):
            prefix = f"ext_double_{i}."
            block_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if block_state:
                block.load_state_dict(block_state, strict=strict)

        for i, block in enumerate(self._extra_single_blocks):
            prefix = f"ext_single_{i}."
            block_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if block_state:
                block.load_state_dict(block_state, strict=strict)

    # =========================================================================
    # Save / Load
    # =========================================================================

    def save(self, path: str, metadata: Optional[Dict[str, str]] = None):
        """Save LoRA weights and config."""
        state = self.state_dict()

        meta = metadata or {}
        meta['config_json'] = self.config.to_json()
        meta['num_adapters'] = str(len(self.adapters))

        if path.endswith('.safetensors'):
            from safetensors.torch import save_file
            save_file(state, path, metadata=meta)
        else:
            torch.save({
                'state_dict': state,
                'config': self.config.to_dict(),
                'metadata': meta,
            }, path)

        size_kb = sum(v.numel() * v.element_size() for v in state.values()) / 1024
        print(f"Saved: {path} ({size_kb:.1f} KB)")

    def load(self, path: str, strict: bool = True):
        """Load LoRA weights."""
        if path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state = load_file(path)
        else:
            data = torch.load(path, map_location='cpu', weights_only=False)
            state = data.get('state_dict', data)

        self.load_state_dict(state, strict=strict)
        print(f"Loaded: {path}")

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        path: str,
        strict: bool = True,
    ) -> 'TinyFluxLoRA':
        """Load LoRA with config from file."""
        if path.endswith('.safetensors'):
            from safetensors import safe_open
            with safe_open(path, framework='pt') as f:
                meta = f.metadata()
            config = TinyFluxLoRAConfig.from_json(meta.get('config_json', '{}'))
        else:
            data = torch.load(path, map_location='cpu', weights_only=False)
            config = TinyFluxLoRAConfig.from_dict(data.get('config', {}))

        lora = cls(model, config)
        lora.load(path, strict=strict)
        return lora

    # =========================================================================
    # Merge / Unmerge
    # =========================================================================

    def merge(self):
        """Merge LoRA weights into base model."""
        for adapter in self.adapters.values():
            adapter.merge()
        print(f"Merged {len(self.adapters)} adapters")

    def unmerge(self):
        """Unmerge LoRA weights from base model."""
        for adapter in self.adapters.values():
            adapter.unmerge()
        print(f"Unmerged {len(self.adapters)} adapters")

    def set_scale(self, scale: float):
        """Adjust LoRA strength."""
        for adapter in self.adapters.values():
            adapter.scale = scale

    @property
    def is_merged(self) -> bool:
        return any(a.merged for a in self.adapters.values())

    # =========================================================================
    # Extension Block Runners
    # =========================================================================

    def run_extra_double_blocks(
        self,
        txt: torch.Tensor,
        img: torch.Tensor,
        vec: torch.Tensor,
        rope: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run extra double blocks."""
        for block in self._extra_double_blocks:
            txt, img = block(txt, img, vec, rope, **kwargs)
        return txt, img

    def run_extra_single_blocks(
        self,
        txt: torch.Tensor,
        img: torch.Tensor,
        vec: torch.Tensor,
        rope: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run extra single blocks."""
        for block in self._extra_single_blocks:
            txt, img = block(txt, img, vec, rope, **kwargs)
        return txt, img


# =============================================================================
# Legacy Compatibility
# =============================================================================

def DoubleStreamLoRA(model, rank=16, **kwargs):
    """Legacy: Double-stream only."""
    return TinyFluxLoRA(
        model,
        rank=rank,
        include_double_attn=True,
        include_single_attn=False,
        include_single_mlp=False,
        **kwargs,
    )

def SingleStreamLoRA(model, rank=16, **kwargs):
    """Legacy: Single-stream only."""
    return TinyFluxLoRA(
        model,
        rank=rank,
        include_double_attn=False,
        include_single_attn=True,
        include_single_mlp=True,
        **kwargs,
    )

def CombinedLoRA(model, rank=16, **kwargs):
    """Legacy: Combined."""
    return TinyFluxLoRA(
        model,
        rank=rank,
        include_double_attn=True,
        include_single_attn=True,
        include_single_mlp=True,
        **kwargs,
    )