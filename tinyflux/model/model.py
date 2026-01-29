"""
TinyFlux-Deep v4.2

Integrates two complementary expert pathways:
- Lune: Trajectory guidance via vec modulation (global conditioning)
- Sol: Attention prior via temperature/spatial bias (structural guidance)

Architecture:
- Lune ExpertPredictor: (t, clip) → expert_signal → ADD to vec
- Sol AttentionPrior: (t, clip) → temperature, spatial_mod → BIAS attention
- David-inspired gate: 70% geometric (timestep), 30% learned (content)

Based on TinyFlux-Deep: 15 double + 25 single blocks.
"""

__version__ = "4.2.0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TinyFluxConfig:
    """Configuration for TinyFlux-Deep v4.2 model."""

    # Core architecture
    hidden_size: int = 512
    num_attention_heads: int = 4
    attention_head_dim: int = 128
    in_channels: int = 16
    patch_size: int = 1
    joint_attention_dim: int = 768
    pooled_projection_dim: int = 768
    num_double_layers: int = 15
    num_single_layers: int = 25
    mlp_ratio: float = 4.0
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)

    # Lune (trajectory guidance)
    use_lune_expert: bool = True
    lune_expert_dim: int = 1280
    lune_hidden_dim: int = 512
    lune_dropout: float = 0.1
    freeze_lune: bool = False

    # Sol (attention prior)
    use_sol_prior: bool = True
    sol_spatial_size: int = 8
    sol_hidden_dim: int = 256
    sol_geometric_weight: float = 0.7
    freeze_sol: bool = False

    # T5 enhancement
    use_t5_vec: bool = True

    def __post_init__(self):
        """Validate configuration constraints."""
        expected_hidden = self.num_attention_heads * self.attention_head_dim
        if self.hidden_size != expected_hidden:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must equal "
                f"num_attention_heads * attention_head_dim ({expected_hidden})"
            )

        if isinstance(self.axes_dims_rope, list):
            self.axes_dims_rope = tuple(self.axes_dims_rope)

        rope_sum = sum(self.axes_dims_rope)
        if rope_sum != self.attention_head_dim:
            raise ValueError(
                f"sum(axes_dims_rope) ({rope_sum}) must equal "
                f"attention_head_dim ({self.attention_head_dim})"
            )

        if not 0.0 <= self.sol_geometric_weight <= 1.0:
            raise ValueError(f"sol_geometric_weight must be in [0, 1], got {self.sol_geometric_weight}")

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["axes_dims_rope"] = list(d["axes_dims_rope"])
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "TinyFluxConfig":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields and not k.startswith("_")}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TinyFluxConfig":
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    def save_json(self, path: Union[str, Path], metadata: Optional[Dict] = None):
        d = self.to_dict()
        if metadata:
            d["_metadata"] = metadata
        with open(path, "w") as f:
            json.dump(d, f, indent=2)


# =============================================================================
# Normalization
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


# =============================================================================
# RoPE
# =============================================================================

class EmbedND(nn.Module):
    """RoPE with cached frequency buffers."""

    def __init__(self, theta: float = 10000.0, axes_dim: Tuple[int, int, int] = (16, 56, 56)):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

        for i, dim in enumerate(axes_dim):
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer(f'freqs_{i}', freqs, persistent=True)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        n_axes = ids.shape[-1]
        emb_list = []

        for i in range(n_axes):
            freqs = getattr(self, f'freqs_{i}').to(device)
            pos = ids[:, i].float()
            angles = pos.unsqueeze(-1) * freqs.unsqueeze(0)
            cos = angles.cos()
            sin = angles.sin()
            emb = torch.stack([cos, sin], dim=-1).flatten(-2)
            emb_list.append(emb)

        rope = torch.cat(emb_list, dim=-1)
        return rope.unsqueeze(1)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings."""
    freqs = freqs_cis.squeeze(1)
    cos = freqs[:, 0::2].repeat_interleave(2, dim=-1)
    sin = freqs[:, 1::2].repeat_interleave(2, dim=-1)
    cos = cos[None, None, :, :].to(x.device)
    sin = sin[None, None, :, :].to(x.device)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


# =============================================================================
# Embeddings
# =============================================================================

class MLPEmbedder(nn.Module):
    """MLP for embedding scalars (timestep)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = 128
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


# =============================================================================
# Lune Expert Predictor
# =============================================================================

class LuneExpertPredictor(nn.Module):
    """
    Predicts Lune's trajectory features from (timestep_emb, CLIP_pooled).
    Output: expert_signal added to vec (global conditioning).
    """

    def __init__(
        self,
        time_dim: int = 512,
        clip_dim: int = 768,
        expert_dim: int = 1280,
        hidden_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.expert_dim = expert_dim

        self.input_proj = nn.Linear(time_dim + clip_dim, hidden_dim)
        self.predictor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, expert_dim),
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(expert_dim),
            nn.Linear(expert_dim, output_dim),
        )
        self.expert_gate = nn.Parameter(torch.tensor(0.0))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        time_emb: torch.Tensor,
        clip_pooled: torch.Tensor,
        real_expert_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        combined = torch.cat([time_emb, clip_pooled], dim=-1)
        hidden = self.input_proj(combined)
        expert_pred = self.predictor(hidden)

        if real_expert_features is not None:
            expert_features = real_expert_features
            expert_used = 'real'
        else:
            expert_features = expert_pred
            expert_used = 'predicted'

        gate = torch.sigmoid(self.expert_gate)
        expert_signal = gate * self.output_proj(expert_features)

        return {
            'expert_signal': expert_signal,
            'expert_pred': expert_pred,
            'expert_used': expert_used,
        }


# =============================================================================
# Sol Attention Prior
# =============================================================================

class SolAttentionPrior(nn.Module):
    """
    Predicts Sol's attention behavior from (timestep_emb, CLIP_pooled).
    Output: Temperature scaling and spatial importance for attention.
    """

    def __init__(
        self,
        time_dim: int = 512,
        clip_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,
        spatial_size: int = 8,
        geometric_weight: float = 0.7,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.spatial_size = spatial_size
        self.geometric_weight = geometric_weight

        self.stat_predictor = nn.Sequential(
            nn.Linear(time_dim + clip_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        self.spatial_predictor = nn.Sequential(
            nn.Linear(time_dim + clip_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, spatial_size * spatial_size),
        )

        self.stat_to_temperature = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_heads),
            nn.Softplus(),
        )

        self.blend_gate = nn.Parameter(self._to_logit(geometric_weight))
        self._init_weights()

    @staticmethod
    def _to_logit(p: float) -> torch.Tensor:
        p = max(1e-4, min(p, 1 - 1e-4))
        return torch.tensor(math.log(p / (1 - p)))

    def _init_weights(self):
        for m in [self.stat_predictor, self.spatial_predictor, self.stat_to_temperature]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def geometric_temperature(self, t_normalized: torch.Tensor) -> torch.Tensor:
        B = t_normalized.shape[0]
        base_temp = 1.0 + t_normalized
        head_bias = torch.linspace(-0.2, 0.2, self.num_heads, device=t_normalized.device)
        temperatures = base_temp.unsqueeze(-1) + head_bias.unsqueeze(0)
        return temperatures.clamp(min=0.5, max=3.0)

    def geometric_spatial(self, t_normalized: torch.Tensor) -> torch.Tensor:
        B = t_normalized.shape[0]
        H = W = self.spatial_size
        device = t_normalized.device

        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        center_dist = (xx**2 + yy**2).sqrt()
        center_bias = torch.exp(-center_dist * 2)

        uniform = torch.ones(H, W, device=device)
        blend = t_normalized.view(B, 1, 1)
        return blend * uniform + (1 - blend) * center_bias.unsqueeze(0)

    def forward(
        self,
        time_emb: torch.Tensor,
        clip_pooled: torch.Tensor,
        t_normalized: torch.Tensor,
        real_stats: Optional[torch.Tensor] = None,
        real_spatial: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = time_emb.shape[0]
        combined = torch.cat([time_emb, clip_pooled], dim=-1)

        pred_stats = self.stat_predictor(combined)
        pred_spatial = self.spatial_predictor(combined)
        pred_spatial = pred_spatial.view(B, self.spatial_size, self.spatial_size)
        pred_spatial = torch.sigmoid(pred_spatial)

        geo_temperature = self.geometric_temperature(t_normalized)
        geo_spatial = self.geometric_spatial(t_normalized)

        stats = real_stats if real_stats is not None else pred_stats
        spatial = real_spatial if real_spatial is not None else pred_spatial

        learned_temperature = self.stat_to_temperature(stats)
        blend = torch.sigmoid(self.blend_gate)

        temperature = blend * geo_temperature + (1 - blend) * learned_temperature
        blended_spatial = blend * geo_spatial + (1 - blend) * spatial

        return {
            'temperature': temperature,
            'spatial_importance': blended_spatial,
            'pred_stats': pred_stats,
            'pred_spatial': pred_spatial,
        }


# =============================================================================
# AdaLayerNorm
# =============================================================================

class AdaLayerNormZero(nn.Module):
    """AdaLN-Zero for double-stream blocks."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.norm = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        emb_out = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb_out.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    """AdaLN-Zero for single-stream blocks."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.norm = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        emb_out = self.linear(self.silu(emb))
        shift, scale, gate = emb_out.chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x, gate


# =============================================================================
# Attention
# =============================================================================

class Attention(nn.Module):
    """Multi-head attention with optional Sol prior."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        use_sol_prior: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.use_sol_prior = use_sol_prior

        self.qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        if use_sol_prior:
            self.spatial_to_mod = nn.Conv2d(1, num_heads, kernel_size=1, bias=True)
            nn.init.zeros_(self.spatial_to_mod.weight)
            nn.init.zeros_(self.spatial_to_mod.bias)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        sol_temperature: Optional[torch.Tensor] = None,
        sol_spatial: Optional[torch.Tensor] = None,
        spatial_size: Optional[Tuple[int, int]] = None,
        num_txt_tokens: int = 0,
    ) -> torch.Tensor:
        B, N, _ = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        if rope is not None:
            q = apply_rotary_emb(q, rope)
            k = apply_rotary_emb(k, rope)

        if self.use_sol_prior and sol_spatial is not None and spatial_size is not None:
            H, W = spatial_size
            N_img = H * W

            sol_up = F.interpolate(
                sol_spatial.unsqueeze(1),
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            )

            img_mod = self.spatial_to_mod(sol_up)
            img_mod = img_mod.reshape(B, self.num_heads, N_img)
            img_mod = torch.exp(img_mod.clamp(-2, 2))

            if num_txt_tokens > 0:
                txt_mod = torch.ones(B, self.num_heads, num_txt_tokens, device=x.device, dtype=img_mod.dtype)
                mod = torch.cat([txt_mod, img_mod], dim=2)
            else:
                mod = img_mod

            q = q * mod.unsqueeze(-1)
            k = k * mod.unsqueeze(-1)

        if sol_temperature is not None:
            temp = sol_temperature.mean(dim=1, keepdim=True).clamp(min=0.1)
            effective_scale = self.scale / temp.unsqueeze(-1).unsqueeze(-1)
            q = q * effective_scale.sqrt()
            k = k * effective_scale.sqrt()
            out = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)


class JointAttention(nn.Module):
    """Joint attention for double-stream blocks."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        use_sol_prior: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.use_sol_prior = use_sol_prior

        self.txt_qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.img_qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.txt_out = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.img_out = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        if use_sol_prior:
            self.spatial_to_mod = nn.Conv2d(1, num_heads, kernel_size=1, bias=True)
            nn.init.zeros_(self.spatial_to_mod.weight)
            nn.init.zeros_(self.spatial_to_mod.bias)

    def forward(
        self,
        txt: torch.Tensor,
        img: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        sol_temperature: Optional[torch.Tensor] = None,
        sol_spatial: Optional[torch.Tensor] = None,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = txt.shape
        _, N, _ = img.shape

        txt_qkv = self.txt_qkv(txt).reshape(B, L, 3, self.num_heads, self.head_dim)
        img_qkv = self.img_qkv(img).reshape(B, N, 3, self.num_heads, self.head_dim)

        txt_q, txt_k, txt_v = txt_qkv.permute(2, 0, 3, 1, 4)
        img_q, img_k, img_v = img_qkv.permute(2, 0, 3, 1, 4)

        if rope is not None:
            img_q = apply_rotary_emb(img_q, rope)
            img_k = apply_rotary_emb(img_k, rope)

        if self.use_sol_prior and sol_spatial is not None and spatial_size is not None:
            H, W = spatial_size

            sol_up = F.interpolate(
                sol_spatial.unsqueeze(1),
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            )

            mod = self.spatial_to_mod(sol_up)
            mod = mod.reshape(B, self.num_heads, H * W)
            mod = torch.exp(mod.clamp(-2, 2))

            img_q = img_q * mod.unsqueeze(-1)
            img_k = img_k * mod.unsqueeze(-1)

        k = torch.cat([txt_k, img_k], dim=2)
        v = torch.cat([txt_v, img_v], dim=2)

        txt_out = F.scaled_dot_product_attention(txt_q, k, v, scale=self.scale)
        txt_out = txt_out.transpose(1, 2).reshape(B, L, -1)

        if sol_temperature is not None:
            temp = sol_temperature.mean(dim=1, keepdim=True).clamp(min=0.1)
            effective_scale = self.scale / temp.unsqueeze(-1).unsqueeze(-1)
            img_q_scaled = img_q * effective_scale.sqrt()
            k_scaled = k * effective_scale.sqrt()
            img_out = F.scaled_dot_product_attention(img_q_scaled, k_scaled, v, scale=1.0)
        else:
            img_out = F.scaled_dot_product_attention(img_q, k, v, scale=self.scale)
        img_out = img_out.transpose(1, 2).reshape(B, N, -1)

        return self.txt_out(txt_out), self.img_out(img_out)


# =============================================================================
# MLP
# =============================================================================

class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden, bias=True)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(mlp_hidden, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# =============================================================================
# Transformer Blocks
# =============================================================================

class DoubleStreamBlock(nn.Module):
    """Double-stream transformer block."""

    def __init__(self, config: TinyFluxConfig):
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        head_dim = config.attention_head_dim

        self.img_norm1 = AdaLayerNormZero(hidden)
        self.txt_norm1 = AdaLayerNormZero(hidden)
        self.attn = JointAttention(hidden, heads, head_dim, use_sol_prior=config.use_sol_prior)
        self.img_norm2 = RMSNorm(hidden)
        self.txt_norm2 = RMSNorm(hidden)
        self.img_mlp = MLP(hidden, config.mlp_ratio)
        self.txt_mlp = MLP(hidden, config.mlp_ratio)

    def forward(
        self,
        txt: torch.Tensor,
        img: torch.Tensor,
        vec: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        sol_temperature: Optional[torch.Tensor] = None,
        sol_spatial: Optional[torch.Tensor] = None,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_normed, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = self.img_norm1(img, vec)
        txt_normed, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = self.txt_norm1(txt, vec)

        txt_attn_out, img_attn_out = self.attn(
            txt_normed, img_normed, rope,
            sol_temperature=sol_temperature,
            sol_spatial=sol_spatial,
            spatial_size=spatial_size,
        )

        txt = txt + txt_gate_msa.unsqueeze(1) * txt_attn_out
        img = img + img_gate_msa.unsqueeze(1) * img_attn_out

        txt_mlp_in = self.txt_norm2(txt) * (1 + txt_scale_mlp.unsqueeze(1)) + txt_shift_mlp.unsqueeze(1)
        img_mlp_in = self.img_norm2(img) * (1 + img_scale_mlp.unsqueeze(1)) + img_shift_mlp.unsqueeze(1)

        txt = txt + txt_gate_mlp.unsqueeze(1) * self.txt_mlp(txt_mlp_in)
        img = img + img_gate_mlp.unsqueeze(1) * self.img_mlp(img_mlp_in)

        return txt, img


class SingleStreamBlock(nn.Module):
    """Single-stream transformer block."""

    def __init__(self, config: TinyFluxConfig):
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        head_dim = config.attention_head_dim

        self.norm = AdaLayerNormZeroSingle(hidden)
        self.attn = Attention(hidden, heads, head_dim, use_sol_prior=config.use_sol_prior)
        self.mlp = MLP(hidden, config.mlp_ratio)
        self.norm2 = RMSNorm(hidden)

    def forward(
        self,
        txt: torch.Tensor,
        img: torch.Tensor,
        vec: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        sol_temperature: Optional[torch.Tensor] = None,
        sol_spatial: Optional[torch.Tensor] = None,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        L = txt.shape[1]
        x = torch.cat([txt, img], dim=1)
        x_normed, gate = self.norm(x, vec)

        x = x + gate.unsqueeze(1) * self.attn(
            x_normed, rope,
            sol_temperature=sol_temperature,
            sol_spatial=sol_spatial,
            spatial_size=spatial_size,
            num_txt_tokens=L,
        )
        x = x + self.mlp(self.norm2(x))
        txt, img = x.split([L, x.shape[1] - L], dim=1)
        return txt, img


# =============================================================================
# Main Model
# =============================================================================

class TinyFluxDeep(nn.Module):
    """TinyFlux-Deep v4.2 with Lune + Sol expert system."""

    def __init__(self, config: Optional[TinyFluxConfig] = None):
        super().__init__()
        self.config = config or TinyFluxConfig()
        cfg = self.config

        # Input projections
        self.img_in = nn.Linear(cfg.in_channels, cfg.hidden_size, bias=True)
        self.txt_in = nn.Linear(cfg.joint_attention_dim, cfg.hidden_size, bias=True)

        # Conditioning
        self.time_in = MLPEmbedder(cfg.hidden_size)
        self.vector_in = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.pooled_projection_dim, cfg.hidden_size, bias=True)
        )

        # T5 pooling (attention-weighted)
        if cfg.use_t5_vec:
            self.t5_pool = nn.Sequential(
                nn.Linear(cfg.joint_attention_dim, cfg.hidden_size),
                nn.SiLU(),
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
            )
            self.text_balance = nn.Parameter(torch.tensor(0.0))
        else:
            self.t5_pool = None
            self.text_balance = None

        # Lune expert predictor
        if cfg.use_lune_expert:
            self.lune_predictor = LuneExpertPredictor(
                time_dim=cfg.hidden_size,
                clip_dim=cfg.pooled_projection_dim,
                expert_dim=cfg.lune_expert_dim,
                hidden_dim=cfg.lune_hidden_dim,
                output_dim=cfg.hidden_size,
                dropout=cfg.lune_dropout,
            )
            if cfg.freeze_lune:
                for p in self.lune_predictor.parameters():
                    p.requires_grad = False
        else:
            self.lune_predictor = None

        # Sol attention prior
        if cfg.use_sol_prior:
            self.sol_prior = SolAttentionPrior(
                time_dim=cfg.hidden_size,
                clip_dim=cfg.pooled_projection_dim,
                hidden_dim=cfg.sol_hidden_dim,
                num_heads=cfg.num_attention_heads,
                spatial_size=cfg.sol_spatial_size,
                geometric_weight=cfg.sol_geometric_weight,
            )
            if cfg.freeze_sol:
                for p in self.sol_prior.parameters():
                    p.requires_grad = False
        else:
            self.sol_prior = None

        # RoPE
        self.rope = EmbedND(theta=10000.0, axes_dim=cfg.axes_dims_rope)

        # Transformer blocks
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(cfg) for _ in range(cfg.num_double_layers)
        ])
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(cfg) for _ in range(cfg.num_single_layers)
        ])

        # Output
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.final_linear = nn.Linear(cfg.hidden_size, cfg.in_channels, bias=True)

        self._init_weights()

    def _init_weights(self):
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_init)
        nn.init.zeros_(self.final_linear.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: Optional[torch.Tensor] = None,
        lune_features: Optional[torch.Tensor] = None,
        sol_stats: Optional[torch.Tensor] = None,
        sol_spatial: Optional[torch.Tensor] = None,
        return_expert_pred: bool = False,
    ) -> torch.Tensor:
        B = hidden_states.shape[0]
        L = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]

        H = W = int(math.sqrt(N))
        spatial_size = (H, W)

        # Dtype alignment
        model_dtype = self.img_in.weight.dtype
        hidden_states = hidden_states.to(dtype=model_dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=model_dtype)
        pooled_projections = pooled_projections.to(dtype=model_dtype)
        timestep = timestep.to(dtype=model_dtype)

        if lune_features is not None:
            lune_features = lune_features.to(dtype=model_dtype)
        if sol_stats is not None:
            sol_stats = sol_stats.to(dtype=model_dtype)
        if sol_spatial is not None:
            sol_spatial = sol_spatial.to(dtype=model_dtype)

        # Input projections
        img = self.img_in(hidden_states)
        txt = self.txt_in(encoder_hidden_states)

        # Conditioning
        time_emb = self.time_in(timestep)
        clip_vec = self.vector_in(pooled_projections)

        # T5 pooling (attention-weighted)
        if self.t5_pool is not None:
            t5_attn_logits = encoder_hidden_states.mean(dim=-1)
            t5_attn = F.softmax(t5_attn_logits, dim=-1)
            t5_pooled = (encoder_hidden_states * t5_attn.unsqueeze(-1)).sum(dim=1)
            t5_vec = self.t5_pool(t5_pooled)

            balance = torch.sigmoid(self.text_balance)
            text_vec = balance * clip_vec + (1 - balance) * t5_vec
        else:
            text_vec = clip_vec

        vec = time_emb + text_vec

        # Lune
        lune_info = None
        if self.lune_predictor is not None:
            lune_out = self.lune_predictor(
                time_emb=time_emb,
                clip_pooled=pooled_projections,
                real_expert_features=lune_features,
            )
            vec = vec + lune_out['expert_signal']
            lune_info = lune_out

        # Sol
        sol_temperature = None
        sol_spatial_blend = None
        sol_info = None

        if self.sol_prior is not None:
            sol_out = self.sol_prior(
                time_emb=time_emb,
                clip_pooled=pooled_projections,
                t_normalized=timestep,
                real_stats=sol_stats,
                real_spatial=sol_spatial,
            )
            sol_temperature = sol_out['temperature']
            sol_spatial_blend = sol_out['spatial_importance']
            sol_info = sol_out

        # RoPE
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        img_rope = self.rope(img_ids)

        # Double-stream blocks
        for block in self.double_blocks:
            txt, img = block(
                txt, img, vec, img_rope,
                sol_temperature=sol_temperature,
                sol_spatial=sol_spatial_blend,
                spatial_size=spatial_size,
            )

        # Single-stream RoPE
        if txt_ids is None:
            txt_ids = torch.zeros(L, 3, device=img_ids.device, dtype=img_ids.dtype)
        elif txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        all_ids = torch.cat([txt_ids, img_ids], dim=0)
        full_rope = self.rope(all_ids)

        # Single-stream blocks
        for block in self.single_blocks:
            txt, img = block(
                txt, img, vec, full_rope,
                sol_temperature=sol_temperature,
                sol_spatial=sol_spatial_blend,
                spatial_size=spatial_size,
            )

        # Output
        img = self.final_norm(img)
        output = self.final_linear(img)

        if return_expert_pred:
            return output, {'lune': lune_info, 'sol': sol_info}
        return output

    @staticmethod
    def create_img_ids(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create image position IDs for RoPE (vectorized)."""
        y = torch.arange(height, device=device).view(-1, 1).expand(-1, width).reshape(-1)
        x = torch.arange(width, device=device).view(1, -1).expand(height, -1).reshape(-1)
        img_ids = torch.stack([torch.zeros_like(y), y, x], dim=-1)  # [H*W, 3]
        return img_ids.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W, 3]

    @staticmethod
    def create_txt_ids(batch_size: int, text_len: int, device: torch.device) -> torch.Tensor:
        """Create text position IDs."""
        txt_ids = torch.zeros(text_len, 3, device=device)
        txt_ids[:, 0] = torch.arange(text_len, device=device)
        return txt_ids.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, 3]

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts['img_in'] = sum(p.numel() for p in self.img_in.parameters())
        counts['txt_in'] = sum(p.numel() for p in self.txt_in.parameters())
        counts['time_in'] = sum(p.numel() for p in self.time_in.parameters())
        counts['vector_in'] = sum(p.numel() for p in self.vector_in.parameters())

        if self.t5_pool is not None:
            counts['t5_pool'] = sum(p.numel() for p in self.t5_pool.parameters()) + 1
        if self.lune_predictor is not None:
            counts['lune_predictor'] = sum(p.numel() for p in self.lune_predictor.parameters())
        if self.sol_prior is not None:
            counts['sol_prior'] = sum(p.numel() for p in self.sol_prior.parameters())

        counts['double_blocks'] = sum(p.numel() for p in self.double_blocks.parameters())
        counts['single_blocks'] = sum(p.numel() for p in self.single_blocks.parameters())
        counts['final'] = sum(p.numel() for p in self.final_norm.parameters()) + \
                          sum(p.numel() for p in self.final_linear.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts


# =============================================================================
# Smoke Test
# =============================================================================

def _smoke_test():
    """Quick validation that model runs correctly."""
    import gc

    print(f"TinyFlux-Deep v{__version__} Smoke Test")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Smaller dims for testing
    B, H, W, L = 1, 16, 16, 32

    # Test 1: Full config (Lune + Sol + T5)
    print("\n[1] Full config (Lune + Sol + T5)...")
    cfg = TinyFluxConfig(use_lune_expert=True, use_sol_prior=True, use_t5_vec=True)
    model = TinyFluxDeep(cfg).to(device)

    x = torch.randn(B, H * W, cfg.in_channels, device=device)
    t5 = torch.randn(B, L, cfg.joint_attention_dim, device=device)
    clip = torch.randn(B, cfg.pooled_projection_dim, device=device)
    t = torch.rand(B, device=device)
    img_ids = TinyFluxDeep.create_img_ids(B, H, W, device)

    out, info = model(x, t5, clip, t, img_ids, return_expert_pred=True)
    assert out.shape == (B, H * W, cfg.in_channels), f"Bad shape: {out.shape}"
    assert info['lune'] is not None, "Lune missing"
    assert info['sol'] is not None, "Sol missing"
    print(f"    Output: {out.shape}, Lune: {info['lune']['expert_used']}, Sol temp: {info['sol']['temperature'].shape}")

    # Cleanup
    counts_full = model.count_parameters()['total']
    del model, out, info
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Test 2: Minimal config (no experts)
    print("\n[2] Minimal config (no experts)...")
    cfg_min = TinyFluxConfig(use_lune_expert=False, use_sol_prior=False, use_t5_vec=False)
    model_min = TinyFluxDeep(cfg_min).to(device)

    out = model_min(x, t5, clip, t, img_ids)
    assert out.shape == (B, H * W, cfg.in_channels), f"Bad shape: {out.shape}"
    print(f"    Output: {out.shape}, No experts loaded")

    # Test 3: Param count sanity
    print("\n[3] Parameter counts...")
    counts_min = model_min.count_parameters()['total']
    print(f"    Full:    {counts_full:,}")
    print(f"    Minimal: {counts_min:,}")
    assert counts_full > counts_min, "Full should have more params"

    # Cleanup
    del model_min, out
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Test 4: create_img_ids shape
    print("\n[4] create_img_ids batch handling...")
    ids = TinyFluxDeep.create_img_ids(4, 32, 32, device)
    assert ids.shape == (4, 32 * 32, 3), f"Bad ids shape: {ids.shape}"
    print(f"    Shape: {ids.shape} ✓")
    del ids

    # Test 5: Freeze flags
    print("\n[5] Freeze flags...")
    cfg_frozen = TinyFluxConfig(
        use_lune_expert=True, freeze_lune=True,
        use_sol_prior=True, freeze_sol=True,
        use_t5_vec=True
    )
    model_frozen = TinyFluxDeep(cfg_frozen).to(device)

    lune_frozen = all(not p.requires_grad for p in model_frozen.lune_predictor.parameters())
    sol_frozen = all(not p.requires_grad for p in model_frozen.sol_prior.parameters())
    assert lune_frozen, "Lune should be frozen"
    assert sol_frozen, "Sol should be frozen"
    print(f"    Lune frozen: {lune_frozen}, Sol frozen: {sol_frozen} ✓")

    # Test 6: Inference mode
    print("\n[6] Inference mode...")
    model_frozen.eval()
    with torch.no_grad():
        out = model_frozen(x, t5, clip, t, img_ids)
    assert out.shape == (B, H * W, cfg.in_channels)
    print(f"    Output: {out.shape} ✓")

    # Cleanup
    del model_frozen, out, x, t5, clip, t, img_ids
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("✓ All smoke tests passed")

    print("\n" + "=" * 50)
    print("✓ All smoke tests passed")


if __name__ == "__main__":
    _smoke_test()