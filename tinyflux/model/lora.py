"""
TinyFlux LoRA Modules

Low-rank adaptation for TinyFlux model components.

Prototype 1: DoubleStreamLoRA
- Targets JointAttention in double-stream blocks (txt↔img cross-attention)
- Where concept binding happens
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Core LoRA Layer
# =============================================================================

class LoRALinear(nn.Module):
    """
    LoRA adapter wrapping a Linear layer.

    y = Wx + (BA)x * scale

    Initializes B=0 so LoRA starts as identity.
    """

    def __init__(
            self,
            base: nn.Linear,
            rank: int,
            alpha: float,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

        # Init A with kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zero -> LoRA starts as identity

        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        if self._merged:
            return base_out

        # LoRA path: x @ A^T @ B^T * scale
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scale

    def merge(self):
        """Merge LoRA into base weights."""
        if not self._merged:
            delta = (self.lora_B @ self.lora_A) * self.scale
            self.base.weight.data += delta.to(self.base.weight.dtype)
            self._merged = True

    def unmerge(self):
        """Remove LoRA from base weights."""
        if self._merged:
            delta = (self.lora_B @ self.lora_A) * self.scale
            self.base.weight.data -= delta.to(self.base.weight.dtype)
            self._merged = False

    @property
    def merged(self) -> bool:
        return self._merged


# =============================================================================
# Double-Stream LoRA (Prototype 1)
# =============================================================================

@dataclass
class DoubleStreamLoRAConfig:
    """Configuration for double-stream LoRA."""
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0

    # Which projections to adapt
    adapt_txt_qkv: bool = True
    adapt_img_qkv: bool = True
    adapt_txt_out: bool = True
    adapt_img_out: bool = True

    # Which blocks (None = all 15)
    block_indices: Optional[List[int]] = None


class DoubleStreamLoRA(nn.Module):
    """
    LoRA for TinyFlux double-stream blocks.

    Targets JointAttention projections where text↔image binding occurs:
    - txt_qkv: text queries/keys/values
    - img_qkv: image queries/keys/values
    - txt_out: text output projection
    - img_out: image output projection

    Usage:
        lora = DoubleStreamLoRA(model, rank=16)
        optimizer = torch.optim.AdamW(lora.parameters(), lr=1e-4)
        # ... train ...
        lora.save("lora.safetensors")
        lora.merge()  # Zero inference overhead
    """

    def __init__(
            self,
            model: nn.Module,
            config: Optional[DoubleStreamLoRAConfig] = None,
            rank: int = 16,
            alpha: float = 16.0,
            dropout: float = 0.0,
    ):
        super().__init__()

        if config is None:
            config = DoubleStreamLoRAConfig(rank=rank, alpha=alpha, dropout=dropout)
        self.config = config

        if not hasattr(model, 'double_blocks'):
            raise ValueError("Model must have double_blocks attribute")

        self.adapters: nn.ModuleDict = nn.ModuleDict()

        num_blocks = len(model.double_blocks)
        block_indices = config.block_indices or list(range(num_blocks))

        # Inject LoRA into each target
        for idx in block_indices:
            if idx >= num_blocks:
                continue

            block = model.double_blocks[idx]
            attn = block.attn  # JointAttention

            targets = []
            if config.adapt_txt_qkv:
                targets.append(('txt_qkv', attn.txt_qkv))
            if config.adapt_img_qkv:
                targets.append(('img_qkv', attn.img_qkv))
            if config.adapt_txt_out:
                targets.append(('txt_out', attn.txt_out))
            if config.adapt_img_out:
                targets.append(('img_out', attn.img_out))

            for name, layer in targets:
                key = f"block_{idx}_{name}"
                adapter = LoRALinear(
                    layer,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                self.adapters[key] = adapter
                setattr(attn, name, adapter)

        # Freeze entire model
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA params
        for adapter in self.adapters.values():
            adapter.lora_A.requires_grad = True
            adapter.lora_B.requires_grad = True

        self._print_stats(model)

    def _print_stats(self, model: nn.Module):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lora_params = sum(
            a.lora_A.numel() + a.lora_B.numel()
            for a in self.adapters.values()
        )

        print(f"\n[DoubleStreamLoRA]")
        print(f"  Adapters: {len(self.adapters)}")
        print(f"  Rank: {self.config.rank}, Alpha: {self.config.alpha}")
        print(f"  LoRA params: {lora_params:,}")
        print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    def parameters(self):
        """Yield only LoRA parameters."""
        for adapter in self.adapters.values():
            yield adapter.lora_A
            yield adapter.lora_B

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get LoRA weights only."""
        state = {}
        for name, adapter in self.adapters.items():
            state[f"{name}.lora_A"] = adapter.lora_A.data
            state[f"{name}.lora_B"] = adapter.lora_B.data
        return state

    def load_state_dict(self, state: Dict[str, torch.Tensor]):
        """Load LoRA weights."""
        for name, adapter in self.adapters.items():
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in state:
                adapter.lora_A.data.copy_(state[a_key])
            if b_key in state:
                adapter.lora_B.data.copy_(state[b_key])

    def save(self, path: str, metadata: Optional[Dict[str, str]] = None):
        """Save LoRA weights."""
        state = self.state_dict()

        meta = metadata or {}
        meta.update({
            'type': 'double_stream_lora',
            'rank': str(self.config.rank),
            'alpha': str(self.config.alpha),
            'num_adapters': str(len(self.adapters)),
        })

        if path.endswith('.safetensors'):
            from safetensors.torch import save_file
            save_file(state, path, metadata=meta)
        else:
            torch.save({'state_dict': state, 'metadata': meta}, path)

        size_kb = sum(v.numel() * v.element_size() for v in state.values()) / 1024
        print(f"Saved: {path} ({size_kb:.1f} KB, {len(state)} tensors)")

    def load(self, path: str):
        """Load LoRA weights."""
        if path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state = load_file(path)
        else:
            data = torch.load(path, map_location='cpu')
            state = data.get('state_dict', data)

        self.load_state_dict(state)
        print(f"Loaded: {path}")

    def merge(self):
        """Merge all LoRA weights into base model."""
        for adapter in self.adapters.values():
            adapter.merge()
        print(f"Merged {len(self.adapters)} adapters")

    def unmerge(self):
        """Unmerge all LoRA weights from base model."""
        for adapter in self.adapters.values():
            adapter.unmerge()
        print(f"Unmerged {len(self.adapters)} adapters")

    def set_scale(self, scale: float):
        """Adjust LoRA contribution."""
        for adapter in self.adapters.values():
            adapter.scale = scale


# =============================================================================
# Utility
# =============================================================================

def calculate_double_lora_size(
        hidden_size: int = 512,
        num_heads: int = 4,
        head_dim: int = 128,
        num_blocks: int = 15,
        rank: int = 16,
) -> Dict[str, int]:
    """Calculate LoRA parameter count for double-stream."""
    qkv_in = hidden_size
    qkv_out = 3 * num_heads * head_dim  # 1536
    out_in = num_heads * head_dim  # 512
    out_out = hidden_size  # 512

    # LoRA params per layer: rank * (in + out)
    txt_qkv_lora = rank * (qkv_in + qkv_out)
    img_qkv_lora = rank * (qkv_in + qkv_out)
    txt_out_lora = rank * (out_in + out_out)
    img_out_lora = rank * (out_in + out_out)

    per_block = txt_qkv_lora + img_qkv_lora + txt_out_lora + img_out_lora
    total = per_block * num_blocks

    return {
        'per_qkv': txt_qkv_lora,
        'per_out': txt_out_lora,
        'per_block': per_block,
        'total': total,
        'total_kb': total * 4 / 1024,
        'total_mb': total * 4 / 1024 / 1024,
    }