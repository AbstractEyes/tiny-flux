"""
TinyFlux LoRA Configuration System

A flexible, layer-by-layer configuration system for LoRA adapters.

Features:
- Per-layer rank, alpha, learning rate multipliers
- Stacked LoRA (multiple A/B matrices for deeper adaptation)
- Pattern matching for targeting layers
- Block extensions (add new trainable blocks)
- JSON serialization for reproducibility

Example config (JSON):
    {
        "name": "my_character_lora",
        "defaults": {"rank": 16, "alpha": 16.0},
        "targets": [
            {"pattern": "single_blocks.[20-24].attn.*", "rank": 32, "lr_scale": 2.0},
            {"pattern": "*.mlp.*", "enabled": false}
        ],
        "extensions": {"single_blocks": 2}
    }
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


# =============================================================================
# Layer Specification
# =============================================================================

@dataclass
class LoRALayerSpec:
    """
    Specification for a single LoRA adapter.

    Each targeted layer gets one of these, either from defaults
    or from explicit configuration.
    """
    rank: int = 16
    alpha: float = 16.0
    lr_scale: float = 1.0  # Multiplier on base learning rate
    depth: int = 1  # Stacked LoRA layers (1 = standard)
    dropout: float = 0.0
    enabled: bool = True  # Can disable specific layers

    def effective_lr(self, base_lr: float) -> float:
        return base_lr * self.lr_scale

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LoRALayerSpec':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LoRATarget:
    """
    Target specification using pattern matching.

    Patterns support:
    - Wildcards: "single_blocks.*.attn.qkv" matches all blocks
    - Ranges: "single_blocks.[20-24].attn.qkv" matches blocks 20-24
    - Lists: "single_blocks.[0,5,10].attn.qkv" matches specific blocks

    Examples:
        # All single-stream QKV with rank 32
        LoRATarget(pattern="single_blocks.*.attn.qkv", rank=32)

        # Last 5 blocks with higher LR
        LoRATarget(pattern="single_blocks.[20-24].*", lr_scale=2.0)

        # Disable MLP layers
        LoRATarget(pattern="*.mlp.*", enabled=False)
    """
    pattern: str

    # Overrides (None = use defaults)
    rank: Optional[int] = None
    alpha: Optional[float] = None
    lr_scale: Optional[float] = None
    depth: Optional[int] = None
    dropout: Optional[float] = None
    enabled: Optional[bool] = None

    def matches(self, path: str) -> bool:
        """Check if this target matches a layer path."""
        regex = self._pattern_to_regex(self.pattern)
        return bool(re.match(regex, path))

    def _pattern_to_regex(self, pattern: str) -> str:
        """Convert pattern to regex."""

        # Handle range/list patterns BEFORE escaping
        # [20-24] -> __RANGE_20_24__, [0,5,10] -> __LIST_0_5_10__

        def mark_range(match):
            content = match.group(1)
            if '-' in content and ',' not in content:
                return f'__RANGE_{content.replace("-", "_")}__'
            elif ',' in content:
                return f'__LIST_{content.replace(",", "_")}__'
            return match.group(0)

        marked = re.sub(r'\[([^\]]+)\]', mark_range, pattern)

        # Now escape
        regex = re.escape(marked)

        # Replace wildcards
        regex = regex.replace(r'\*', r'[^.]+')

        # Restore ranges
        def expand_range(match):
            content = match.group(1)
            if content.startswith('RANGE_'):
                parts = content[6:].split('_')
                start, end = int(parts[0]), int(parts[1])
                options = '|'.join(str(i) for i in range(start, end + 1))
                return f'({options})'
            elif content.startswith('LIST_'):
                items = content[5:].split('_')
                return f'({"|".join(items)})'
            return match.group(0)

        regex = re.sub(r'__([A-Z]+_[^_]+(?:_[^_]+)*)__', expand_range, regex)

        return f'^{regex}$'

    def apply_to(self, spec: LoRALayerSpec) -> LoRALayerSpec:
        """Apply overrides to a base spec."""
        return LoRALayerSpec(
            rank=self.rank if self.rank is not None else spec.rank,
            alpha=self.alpha if self.alpha is not None else spec.alpha,
            lr_scale=self.lr_scale if self.lr_scale is not None else spec.lr_scale,
            depth=self.depth if self.depth is not None else spec.depth,
            dropout=self.dropout if self.dropout is not None else spec.dropout,
            enabled=self.enabled if self.enabled is not None else spec.enabled,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {'pattern': self.pattern}
        for k in ['rank', 'alpha', 'lr_scale', 'depth', 'dropout', 'enabled']:
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LoRATarget':
        # Filter to valid fields only (ignore _comment etc)
        valid = {'pattern', 'rank', 'alpha', 'lr_scale', 'depth', 'dropout', 'enabled'}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class LoRADefaults:
    """Default values for all LoRA layers."""
    rank: int = 16
    alpha: float = 16.0
    lr_scale: float = 1.0
    depth: int = 1
    dropout: float = 0.0

    def to_spec(self) -> LoRALayerSpec:
        return LoRALayerSpec(
            rank=self.rank,
            alpha=self.alpha,
            lr_scale=self.lr_scale,
            depth=self.depth,
            dropout=self.dropout,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LoRADefaults':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Block Extensions
# =============================================================================

@dataclass
class BlockExtensions:
    """
    Configuration for adding new trainable blocks.

    These blocks are added to TinyFlux when the LoRA is active,
    and are fully trainable (not low-rank).
    """
    double_blocks: int = 0
    single_blocks: int = 0

    # Where to insert: "end", "start", or int index
    double_position: Union[str, int] = "end"
    single_position: Union[str, int] = "end"

    # Initialization: "last", "random", "zeros"
    double_init: str = "last"
    single_init: str = "last"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BlockExtensions':
        valid = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in valid})


# =============================================================================
# Main Config
# =============================================================================

@dataclass
class TinyFluxLoRAConfig:
    """
    Complete LoRA configuration for TinyFlux.

    Supports:
    - Global defaults for all layers
    - Pattern-based targeting with per-layer overrides
    - Explicit layer specifications
    - Block extensions (new trainable blocks)

    Example:
        config = TinyFluxLoRAConfig(
            defaults=LoRADefaults(rank=16),
            targets=[
                LoRATarget("single_blocks.[20-24].attn.*", rank=32, lr_scale=2.0),
                LoRATarget("*.mlp.*", enabled=False),
            ],
        )

        # Or from preset
        config = TinyFluxLoRAConfig.from_preset("character")

        # Or from JSON file
        config = TinyFluxLoRAConfig.load("my_lora_config.json")
    """
    # Metadata
    name: str = "tinyflux_lora"
    version: str = "1.0"

    # Global defaults
    defaults: LoRADefaults = field(default_factory=LoRADefaults)

    # Pattern-based targets (processed in order, later overrides earlier)
    targets: List[LoRATarget] = field(default_factory=list)

    # Explicit layer specs (highest priority)
    layer_specs: Dict[str, LoRALayerSpec] = field(default_factory=dict)

    # Block extensions
    extensions: Optional[BlockExtensions] = None

    # Default layer inclusion
    include_double_attn: bool = False
    include_single_attn: bool = True
    include_single_mlp: bool = True

    def __post_init__(self):
        # Convert dicts to proper types
        if isinstance(self.defaults, dict):
            self.defaults = LoRADefaults.from_dict(self.defaults)

        self.targets = [
            LoRATarget.from_dict(t) if isinstance(t, dict) else t
            for t in self.targets
        ]

        self.layer_specs = {
            k: LoRALayerSpec.from_dict(v) if isinstance(v, dict) else v
            for k, v in self.layer_specs.items()
        }

        if isinstance(self.extensions, dict):
            self.extensions = BlockExtensions.from_dict(self.extensions)

    def get_spec(self, layer_path: str) -> LoRALayerSpec:
        """
        Get the LoRA spec for a specific layer path.

        Priority (highest to lowest):
        1. Explicit layer_specs
        2. Matching targets (later targets override earlier)
        3. Defaults
        """
        spec = self.defaults.to_spec()

        for target in self.targets:
            if target.matches(layer_path):
                spec = target.apply_to(spec)

        if layer_path in self.layer_specs:
            spec = self.layer_specs[layer_path]

        return spec

    def should_adapt(self, layer_path: str) -> bool:
        """Check if a layer should be adapted."""
        # Check type inclusion first
        if 'double_blocks' in layer_path and 'attn' in layer_path:
            if not self.include_double_attn:
                # Check if explicitly enabled by target
                for target in self.targets:
                    if target.matches(layer_path) and target.enabled is True:
                        break
                else:
                    if layer_path not in self.layer_specs:
                        return False

        if 'single_blocks' in layer_path and 'attn' in layer_path:
            if not self.include_single_attn:
                for target in self.targets:
                    if target.matches(layer_path) and target.enabled is True:
                        break
                else:
                    if layer_path not in self.layer_specs:
                        return False

        if 'mlp' in layer_path:
            if not self.include_single_mlp:
                for target in self.targets:
                    if target.matches(layer_path) and target.enabled is True:
                        break
                else:
                    if layer_path not in self.layer_specs:
                        return False

        # Check spec
        spec = self.get_spec(layer_path)
        return spec.enabled

    def get_active_layers(
            self,
            num_double: int = 15,
            num_single: int = 25,
    ) -> Dict[str, LoRALayerSpec]:
        """Get all layers that should be adapted with their specs."""
        all_paths = get_tinyflux_layer_paths(num_double, num_single)
        active = {}

        for category, paths in all_paths.items():
            for path in paths:
                if self.should_adapt(path):
                    active[path] = self.get_spec(path)

        return active

    def get_lr_groups(
            self,
            base_lr: float,
            num_double: int = 15,
            num_single: int = 25,
    ) -> Dict[float, List[str]]:
        """Group layers by effective learning rate."""
        groups: Dict[float, List[str]] = {}

        for path, spec in self.get_active_layers(num_double, num_single).items():
            lr = spec.effective_lr(base_lr)
            if lr not in groups:
                groups[lr] = []
            groups[lr].append(path)

        return groups

    def summary(self, num_double: int = 15, num_single: int = 25) -> str:
        """Get a summary of the configuration."""
        active = self.get_active_layers(num_double, num_single)

        lines = [
            f"TinyFluxLoRAConfig: {self.name}",
            f"  Defaults: rank={self.defaults.rank}, alpha={self.defaults.alpha}",
            f"  Targets: {len(self.targets)}",
            f"  Active layers: {len(active)}",
        ]

        # Count by type
        double_count = sum(1 for p in active if 'double_blocks' in p)
        single_attn = sum(1 for p in active if 'single_blocks' in p and 'attn' in p)
        single_mlp = sum(1 for p in active if 'single_blocks' in p and 'mlp' in p)

        lines.append(f"    Double attn: {double_count}")
        lines.append(f"    Single attn: {single_attn}")
        lines.append(f"    Single MLP: {single_mlp}")

        # LR groups
        lr_groups = self.get_lr_groups(1.0, num_double, num_single)
        if len(lr_groups) > 1:
            lines.append(f"  LR scales: {sorted(lr_groups.keys())}")

        # Depths
        depths = set(s.depth for s in active.values())
        if len(depths) > 1 or 1 not in depths:
            lines.append(f"  Depths: {sorted(depths)}")

        # Extensions
        if self.extensions:
            if self.extensions.double_blocks > 0:
                lines.append(f"  Extra double blocks: {self.extensions.double_blocks}")
            if self.extensions.single_blocks > 0:
                lines.append(f"  Extra single blocks: {self.extensions.single_blocks}")

        # Param estimate
        params = estimate_lora_params(active)
        lines.append(f"  Est. params: {params:,} ({params * 4 / 1024 / 1024:.2f} MB)")

        return "\n".join(lines)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'defaults': self.defaults.to_dict(),
            'targets': [t.to_dict() for t in self.targets],
            'layer_specs': {k: v.to_dict() for k, v in self.layer_specs.items()},
            'extensions': self.extensions.to_dict() if self.extensions else None,
            'include_double_attn': self.include_double_attn,
            'include_single_attn': self.include_single_attn,
            'include_single_mlp': self.include_single_mlp,
        }

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """Convert to JSON string, optionally save to file."""
        json_str = json.dumps(self.to_dict(), indent=indent)
        if path:
            Path(path).write_text(json_str)
        return json_str

    def save(self, path: str):
        """Save to JSON file."""
        self.to_json(path)
        print(f"Saved config: {path}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TinyFluxLoRAConfig':
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'TinyFluxLoRAConfig':
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: str) -> 'TinyFluxLoRAConfig':
        """Load from JSON file."""
        data = json.loads(Path(path).read_text())
        config = cls.from_dict(data)
        print(f"Loaded config: {path}")
        return config

    # =========================================================================
    # Presets
    # =========================================================================

    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> 'TinyFluxLoRAConfig':
        """
        Create config from preset.

        Presets:
        - "minimal": Single-stream attention only, rank 8
        - "standard": Single-stream with MLP, rank 16
        - "character": Higher rank on late blocks for style/character
        - "concept": Double-stream focus for concept binding
        - "full": All layers combined
        - "progressive": Increasing rank towards later layers
        """
        presets = {
            'minimal': cls._preset_minimal,
            'standard': cls._preset_standard,
            'character': cls._preset_character,
            'concept': cls._preset_concept,
            'full': cls._preset_full,
            'progressive': cls._preset_progressive,
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

        config = presets[preset]()

        # Apply overrides
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif hasattr(config.defaults, k):
                setattr(config.defaults, k, v)

        return config

    @classmethod
    def _preset_minimal(cls) -> 'TinyFluxLoRAConfig':
        return cls(
            name="minimal",
            defaults=LoRADefaults(rank=8, alpha=8.0),
            include_single_attn=True,
            include_single_mlp=False,
            include_double_attn=False,
        )

    @classmethod
    def _preset_standard(cls) -> 'TinyFluxLoRAConfig':
        return cls(
            name="standard",
            defaults=LoRADefaults(rank=16, alpha=16.0),
            include_single_attn=True,
            include_single_mlp=True,
            include_double_attn=False,
        )

    @classmethod
    def _preset_character(cls) -> 'TinyFluxLoRAConfig':
        return cls(
            name="character",
            defaults=LoRADefaults(rank=16, alpha=16.0),
            targets=[
                # Pattern must match full path structure: block.idx.module.layer
                LoRATarget("single_blocks.[0-9].*.*", rank=8, lr_scale=0.5),
                LoRATarget("single_blocks.[10-19].*.*", rank=16),
                LoRATarget("single_blocks.[20-24].*.*", rank=32, lr_scale=1.5),
            ],
            include_single_attn=True,
            include_single_mlp=True,
            include_double_attn=False,
        )

    @classmethod
    def _preset_concept(cls) -> 'TinyFluxLoRAConfig':
        return cls(
            name="concept",
            defaults=LoRADefaults(rank=16, alpha=16.0),
            targets=[
                LoRATarget("double_blocks.*.attn.*", rank=32),
            ],
            include_double_attn=True,
            include_single_attn=False,
            include_single_mlp=False,
        )

    @classmethod
    def _preset_full(cls) -> 'TinyFluxLoRAConfig':
        return cls(
            name="full",
            defaults=LoRADefaults(rank=16, alpha=16.0),
            include_double_attn=True,
            include_single_attn=True,
            include_single_mlp=True,
        )

    @classmethod
    def _preset_progressive(cls) -> 'TinyFluxLoRAConfig':
        """Increasing rank towards later layers."""
        targets = []

        # Double blocks: 8 -> 24 (pattern: double_blocks.idx.attn.layer)
        for i in range(15):
            rank = 8 + (24 - 8) * i // 14
            targets.append(LoRATarget(f"double_blocks.{i}.attn.*", rank=rank))

        # Single blocks: 8 -> 32 (pattern: double_blocks.idx.module.layer)
        for i in range(25):
            rank = 8 + (32 - 8) * i // 24
            targets.append(LoRATarget(f"single_blocks.{i}.*.*", rank=rank))

        return cls(
            name="progressive",
            defaults=LoRADefaults(rank=16, alpha=16.0),
            targets=targets,
            include_double_attn=True,
            include_single_attn=True,
            include_single_mlp=True,
        )


# =============================================================================
# Utilities
# =============================================================================

def get_tinyflux_layer_paths(
        num_double: int = 15,
        num_single: int = 25,
) -> Dict[str, List[str]]:
    """Get all adaptable layer paths in TinyFlux."""
    paths = {
        'double_attn': [],
        'single_attn': [],
        'single_mlp': [],
    }

    for i in range(num_double):
        paths['double_attn'].extend([
            f"double_blocks.{i}.attn.txt_qkv",
            f"double_blocks.{i}.attn.img_qkv",
            f"double_blocks.{i}.attn.txt_out",
            f"double_blocks.{i}.attn.img_out",
        ])

    for i in range(num_single):
        paths['single_attn'].extend([
            f"single_blocks.{i}.attn.qkv",
            f"single_blocks.{i}.attn.out_proj",
        ])
        paths['single_mlp'].extend([
            f"single_blocks.{i}.mlp.fc1",
            f"single_blocks.{i}.mlp.fc2",
        ])

    return paths


def resolve_layer(model, path: str):
    """Get a layer from model using dot-separated path."""
    parts = path.split('.')
    current = model

    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)

    return current


def set_layer(model, path: str, value):
    """Set a layer in model using dot-separated path."""
    parts = path.split('.')
    current = model

    for part in parts[:-1]:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)

    final = parts[-1]
    if final.isdigit():
        current[int(final)] = value
    else:
        setattr(current, final, value)


def estimate_lora_params(active_layers: Dict[str, LoRALayerSpec]) -> int:
    """Estimate total LoRA parameters."""
    # TinyFlux dimensions
    hidden = 512
    heads = 4
    head_dim = 128
    mlp_hidden = 2048

    sizes = {
        'qkv': hidden + 3 * heads * head_dim,  # 2048
        'txt_qkv': hidden + 3 * heads * head_dim,  # 2048
        'img_qkv': hidden + 3 * heads * head_dim,  # 2048
        'out_proj': heads * head_dim + hidden,  # 1024
        'txt_out': heads * head_dim + hidden,  # 1024
        'img_out': heads * head_dim + hidden,  # 1024
        'fc1': hidden + mlp_hidden,  # 2560
        'fc2': mlp_hidden + hidden,  # 2560
    }

    total = 0
    for path, spec in active_layers.items():
        layer_name = path.split('.')[-1]
        size = sizes.get(layer_name, 1024)
        total += spec.rank * size * spec.depth

    return total