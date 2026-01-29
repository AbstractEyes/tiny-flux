"""
TinyFlux-Deep v4.1 with Dual Expert System

Integrates two complementary expert pathways:
- Lune: Trajectory guidance via vec modulation (global conditioning)
- Sol: Attention prior via temperature/spatial bias (structural guidance)

Key insight: Sol's geometric knowledge lives in its ATTENTION PATTERNS,
not its features. We extract attention statistics (locality, entropy, clustering)
and spatial importance maps to bias TinyFlux's weak 4-head attention.

This avoids the twin-tail paradox: V-pred (Sol) is fundamentally incompatible
with linear flow-matching (TinyFlux), so we don't inject features directly.
Instead, we translate Sol's structural understanding into attention biases.

Architecture:
- Lune ExpertPredictor: (t, clip) → expert_signal → ADD to vec
- Sol AttentionPrior: (t, clip) → temperature, spatial_mod → BIAS attention
- David-inspired gate: 70% geometric (timestep), 30% learned (content)

Based on TinyFlux-Deep: 15 double + 25 single blocks.
"""

__version__ = "4.1.0"

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
    """
    Configuration for TinyFlux-Deep v4.1 model.

    This config fully defines the model architecture and can be used to:
    1. Initialize a new model
    2. Convert checkpoints between versions
    3. Validate checkpoint compatibility

    All dimension constraints are validated on creation.
    """

    # Core architecture
    hidden_size: int = 512
    num_attention_heads: int = 4
    attention_head_dim: int = 128

    in_channels: int = 16
    patch_size: int = 1

    joint_attention_dim: int = 768  # T5 sequence dim
    pooled_projection_dim: int = 768  # CLIP pooled dim

    num_double_layers: int = 15
    num_single_layers: int = 25

    mlp_ratio: float = 4.0
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)

    # Lune expert predictor config (trajectory guidance)
    use_lune_expert: bool = True
    lune_expert_dim: int = 1280  # SD1.5 mid-block dimension
    lune_hidden_dim: int = 512
    lune_dropout: float = 0.1

    # Sol attention prior config (structural guidance)
    use_sol_prior: bool = True
    sol_spatial_size: int = 8  # Sol's feature map resolution
    sol_hidden_dim: int = 256
    sol_geometric_weight: float = 0.7  # David's 70/30 split

    # T5 enhancement config
    use_t5_vec: bool = True  # Add T5 pooled to vec pathway
    t5_pool_mode: str = "attention"  # "attention", "mean", "cls"

    # Loss config
    lune_distill_mode: str = "cosine"  # "hard", "soft", "cosine", "huber"
    use_huber_loss: bool = True
    huber_delta: float = 0.1

    # Legacy (for backward compat)
    use_expert_predictor: bool = True  # Maps to use_lune_expert
    expert_dim: int = 1280
    expert_hidden_dim: int = 512
    expert_dropout: float = 0.1
    guidance_embeds: bool = False

    def __post_init__(self):
        """Validate configuration constraints."""
        # Validate attention dimensions
        expected_hidden = self.num_attention_heads * self.attention_head_dim
        if self.hidden_size != expected_hidden:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must equal "
                f"num_attention_heads * attention_head_dim ({expected_hidden})"
            )

        # Validate RoPE dimensions
        if isinstance(self.axes_dims_rope, list):
            self.axes_dims_rope = tuple(self.axes_dims_rope)

        rope_sum = sum(self.axes_dims_rope)
        if rope_sum != self.attention_head_dim:
            raise ValueError(
                f"sum(axes_dims_rope) ({rope_sum}) must equal "
                f"attention_head_dim ({self.attention_head_dim})"
            )

        # Validate sol_geometric_weight
        if not 0.0 <= self.sol_geometric_weight <= 1.0:
            raise ValueError(f"sol_geometric_weight must be in [0, 1], got {self.sol_geometric_weight}")

        # Legacy mapping
        if self.use_expert_predictor and not self.use_lune_expert:
            self.use_lune_expert = True
            self.lune_expert_dim = self.expert_dim
            self.lune_hidden_dim = self.expert_hidden_dim
            self.lune_dropout = self.expert_dropout

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        d["axes_dims_rope"] = list(d["axes_dims_rope"])
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "TinyFluxConfig":
        """Create from dict, ignoring unknown keys."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields and not k.startswith("_")}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TinyFluxConfig":
        """Load config from JSON file."""
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    def save_json(self, path: Union[str, Path], metadata: Optional[Dict] = None):
        """Save config to JSON file with optional metadata."""
        d = self.to_dict()
        if metadata:
            d["_metadata"] = metadata
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    def validate_checkpoint(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        """
        Validate that a checkpoint matches this config.

        Returns list of warnings (empty if perfect match).
        """
        warnings = []

        # Check double block count
        max_double = 0
        for key in state_dict:
            if key.startswith("double_blocks."):
                idx = int(key.split(".")[1])
                max_double = max(max_double, idx + 1)
        if max_double != self.num_double_layers:
            warnings.append(f"double_blocks: checkpoint has {max_double}, config expects {self.num_double_layers}")

        # Check single block count
        max_single = 0
        for key in state_dict:
            if key.startswith("single_blocks."):
                idx = int(key.split(".")[1])
                max_single = max(max_single, idx + 1)
        if max_single != self.num_single_layers:
            warnings.append(f"single_blocks: checkpoint has {max_single}, config expects {self.num_single_layers}")

        # Check hidden size from a known weight
        if "img_in.weight" in state_dict:
            w = state_dict["img_in.weight"]
            if w.shape[0] != self.hidden_size:
                warnings.append(f"hidden_size: checkpoint has {w.shape[0]}, config expects {self.hidden_size}")

        # Check for v4.1 components
        has_sol = any(k.startswith("sol_prior.") for k in state_dict)
        has_t5 = any(k.startswith("t5_pool.") for k in state_dict)
        has_lune = any(k.startswith("lune_predictor.") for k in state_dict)

        if self.use_sol_prior and not has_sol:
            warnings.append("config expects sol_prior but checkpoint missing it")
        if self.use_t5_vec and not has_t5:
            warnings.append("config expects t5_pool but checkpoint missing it")
        if self.use_lune_expert and not has_lune:
            warnings.append("config expects lune_predictor but checkpoint missing it")

        return warnings


# Backwards compatibility alias
TinyFluxDeepConfig = TinyFluxConfig


# =============================================================================
# Normalization
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = (x * norm).type_as(x)
        if self.weight is not None:
            out = out * self.weight
        return out


# =============================================================================
# RoPE - Cached frequency buffers
# =============================================================================

class EmbedND(nn.Module):
    """Original TinyFlux RoPE with cached frequency buffers."""

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


def apply_rotary_emb_old(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings (old interleaved format)."""
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
# Lune Expert Predictor (Trajectory Guidance → vec)
# =============================================================================

class LuneExpertPredictor(nn.Module):
    """
    Predicts Lune's trajectory features from (timestep_emb, CLIP_pooled).

    Lune learned rich textures and detail via rectified flow.
    Its mid-block features encode "how the denoising trajectory should flow."

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
        self.dropout = dropout

        # Input fusion
        self.input_proj = nn.Linear(time_dim + clip_dim, hidden_dim)

        # Predictor core
        self.predictor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, expert_dim),
        )

        # Project to vec dimension
        self.output_proj = nn.Sequential(
            nn.LayerNorm(expert_dim),
            nn.Linear(expert_dim, output_dim),
        )

        # Learnable gate - store in logit space so sigmoid gives 0.5 at init
        self.expert_gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

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
        """
        Returns:
            expert_signal: [B, output_dim] - add to vec
            expert_pred: [B, expert_dim] - for distillation loss
        """
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
# Sol Attention Prior (Structural Guidance → Attention Bias)
# =============================================================================

class SolAttentionPrior(nn.Module):
    """
    Predicts Sol's attention behavior from (timestep_emb, CLIP_pooled).

    Sol learned geometric structure via DDPM + David assessment.
    Its value isn't in features, but in ATTENTION PATTERNS:
    - locality: how local vs global is attention?
    - entropy: how focused vs diffuse?
    - clustering: how structured vs uniform?
    - spatial_importance: WHERE does structure exist?

    Output: Temperature scaling and Q/K modulation for TinyFlux attention.

    Follows David's philosophy: 70% geometric routing (timestep-based),
    30% learned routing (content-based).
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

        # Statistics predictor: (t, clip) → [locality, entropy, clustering]
        self.stat_predictor = nn.Sequential(
            nn.Linear(time_dim + clip_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Spatial importance predictor: (t, clip) → [H, W] importance map
        self.spatial_predictor = nn.Sequential(
            nn.Linear(time_dim + clip_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, spatial_size * spatial_size),
        )

        # Convert statistics → per-head temperature
        self.stat_to_temperature = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_heads),
            nn.Softplus(),  # Positive temperatures
        )

        # Convert spatial → Q/K modulation
        # Zero-init: starts as identity (no modulation)
        self.spatial_to_qk_scale = nn.Linear(1, num_heads)
        nn.init.zeros_(self.spatial_to_qk_scale.weight)
        nn.init.ones_(self.spatial_to_qk_scale.bias)

        # Learnable blend between geometric and predicted
        # Store in logit space so sigmoid(x) = geometric_weight at init
        self.blend_gate = nn.Parameter(self._to_logit(geometric_weight))

        self._init_weights()

    @staticmethod
    def _to_logit(p: float) -> torch.Tensor:
        """Convert probability to logit for proper sigmoid init."""
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
        """
        Timestep-based temperature prior.

        Early (high t): Higher temperature → softer, more global attention
        Late (low t): Lower temperature → sharper, more local attention

        This matches how denoising naturally progresses:
        - Early: global structure decisions
        - Late: local detail refinement
        """
        B = t_normalized.shape[0]

        # Base temperature: 1.0 at t=0, 2.0 at t=1
        base_temp = 1.0 + t_normalized  # [B]

        # Per-head variation (some heads more local, some more global)
        head_bias = torch.linspace(-0.2, 0.2, self.num_heads, device=t_normalized.device)

        # [B, num_heads]
        temperatures = base_temp.unsqueeze(-1) + head_bias.unsqueeze(0)
        return temperatures.clamp(min=0.5, max=3.0)

    def geometric_spatial(self, t_normalized: torch.Tensor) -> torch.Tensor:
        """
        Timestep-based spatial prior.

        Early (high t): Uniform importance (everything matters for structure)
        Late (low t): Center-biased (details typically in center)

        Returns: [B, H, W] spatial importance
        """
        B = t_normalized.shape[0]
        H = W = self.spatial_size
        device = t_normalized.device

        # Create center-biased gaussian
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        center_dist = (xx**2 + yy**2).sqrt()
        center_bias = torch.exp(-center_dist * 2)  # Gaussian centered

        # Blend: high t → uniform, low t → center-biased
        uniform = torch.ones(H, W, device=device)

        # t as blend factor: high t (1.0) → uniform, low t (0.0) → center
        blend = t_normalized.view(B, 1, 1)
        spatial = blend * uniform + (1 - blend) * center_bias.unsqueeze(0)

        return spatial

    def forward(
        self,
        time_emb: torch.Tensor,
        clip_pooled: torch.Tensor,
        t_normalized: torch.Tensor,
        real_stats: Optional[torch.Tensor] = None,
        real_spatial: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            time_emb: [B, time_dim]
            clip_pooled: [B, clip_dim]
            t_normalized: [B] timestep in [0, 1]
            real_stats: [B, 3] real Sol statistics (training)
            real_spatial: [B, H, W] real Sol spatial importance (training)

        Returns:
            temperature: [B, num_heads] - attention temperature per head
            spatial_mod: [B, num_heads, N] - Q/K modulation per position
            pred_stats: [B, 3] - for distillation loss
            pred_spatial: [B, H, W] - for distillation loss
        """
        B = time_emb.shape[0]
        device = time_emb.device

        combined = torch.cat([time_emb, clip_pooled], dim=-1)

        # === Predict statistics ===
        pred_stats = self.stat_predictor(combined)  # [B, 3]

        # === Predict spatial importance ===
        pred_spatial = self.spatial_predictor(combined)  # [B, 64]
        pred_spatial = pred_spatial.view(B, self.spatial_size, self.spatial_size)
        pred_spatial = torch.sigmoid(pred_spatial)  # [0, 1] importance

        # === Geometric priors ===
        geo_temperature = self.geometric_temperature(t_normalized)
        geo_spatial = self.geometric_spatial(t_normalized)

        # === Learned components ===
        # Use real values if provided (training), else predicted (inference)
        stats = real_stats if real_stats is not None else pred_stats
        spatial = real_spatial if real_spatial is not None else pred_spatial

        learned_temperature = self.stat_to_temperature(stats)  # [B, num_heads]

        # === Blend geometric and learned (David's 70/30) ===
        blend = torch.sigmoid(self.blend_gate)  # Learnable, initialized to 0.7

        temperature = blend * geo_temperature + (1 - blend) * learned_temperature

        # For spatial: blend then convert to Q/K modulation
        blended_spatial = blend * geo_spatial + (1 - blend) * spatial  # [B, H, W]

        return {
            'temperature': temperature,           # [B, num_heads]
            'spatial_importance': blended_spatial,  # [B, H, W] at sol resolution
            'pred_stats': pred_stats,             # [B, 3] for distillation
            'pred_spatial': pred_spatial,         # [B, H, W] for distillation
        }


# =============================================================================
# AdaLayerNorm
# =============================================================================

class AdaLayerNormZero(nn.Module):
    """AdaLN-Zero for double-stream blocks (6 params)."""

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
    """AdaLN-Zero for single-stream blocks (3 params)."""

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
# Attention with Sol Prior Support
# =============================================================================

class Attention(nn.Module):
    """
    Multi-head attention with optional Sol attention prior.

    Sol prior provides:
    - temperature: per-head attention sharpness
    - spatial_mod: per-position Q/K scaling
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        use_bias: bool = False,
        sol_spatial_size: int = 8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.sol_spatial_size = sol_spatial_size

        self.qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=use_bias)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=use_bias)

        # Sol spatial → per-head Q/K modulation
        # Zero-init weight AND bias so exp(0)=1 at init (true identity)
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
        """
        Args:
            x: [B, N, hidden_size]
            rope: RoPE embeddings
            sol_temperature: [B, num_heads] - attention temperature per head
            sol_spatial: [B, H_sol, W_sol] - spatial importance from Sol
            spatial_size: (H, W) of the image tokens for upsampling sol_spatial
            num_txt_tokens: number of text tokens at start of sequence (for single-stream)
        """
        B, N, _ = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [B, heads, N, head_dim]

        if rope is not None:
            q = apply_rotary_emb_old(q, rope)
            k = apply_rotary_emb_old(k, rope)

        # === Sol Spatial Modulation ===
        if sol_spatial is not None and spatial_size is not None:
            H, W = spatial_size
            N_img = H * W

            # Upsample Sol spatial to match image token resolution
            sol_up = F.interpolate(
                sol_spatial.unsqueeze(1),  # [B, 1, H_sol, W_sol]
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            )  # [B, 1, H, W]

            # Convert to per-head modulation for IMAGE tokens only
            img_mod = self.spatial_to_mod(sol_up)  # [B, heads, H, W]
            img_mod = img_mod.reshape(B, self.num_heads, N_img)  # [B, heads, N_img]

            # exp(0) = 1 at init (true identity), learns to scale up/down
            img_mod = torch.exp(img_mod.clamp(-2, 2))  # Clamp for stability

            # For single-stream: prepend ones for text tokens (no modulation)
            if num_txt_tokens > 0:
                txt_mod = torch.ones(B, self.num_heads, num_txt_tokens, device=x.device, dtype=img_mod.dtype)
                mod = torch.cat([txt_mod, img_mod], dim=2)  # [B, heads, N_txt + N_img]
            else:
                mod = img_mod

            # Modulate Q and K (amplify at important positions)
            q = q * mod.unsqueeze(-1)  # [B, heads, N, head_dim]
            k = k * mod.unsqueeze(-1)

        # === Compute attention with SDPA (Flash Attention) ===
        # Sol temperature is applied via scale modification
        if sol_temperature is not None:
            # Average temperature across heads for SDPA scale
            # temperature: [B, num_heads] → scalar per sample (SDPA limitation)
            temp = sol_temperature.mean(dim=1, keepdim=True).clamp(min=0.1)  # [B, 1]
            effective_scale = self.scale / temp.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
            # Pre-scale Q instead of post-scale scores (mathematically equivalent)
            q = q * (effective_scale.sqrt())
            k = k * (effective_scale.sqrt())
            out = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        out = out.transpose(1, 2).reshape(B, N, -1)

        return self.out_proj(out)


class JointAttention(nn.Module):
    """
    Joint attention for double-stream blocks with Sol prior support.

    Image tokens get Sol modulation, text tokens don't.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        use_bias: bool = False,
        sol_spatial_size: int = 8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.sol_spatial_size = sol_spatial_size

        self.txt_qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=use_bias)
        self.img_qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=use_bias)

        self.txt_out = nn.Linear(num_heads * head_dim, hidden_size, bias=use_bias)
        self.img_out = nn.Linear(num_heads * head_dim, hidden_size, bias=use_bias)

        # Sol spatial modulation for image tokens
        # Zero-init so exp(0)=1 at init (true identity)
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
            img_q = apply_rotary_emb_old(img_q, rope)
            img_k = apply_rotary_emb_old(img_k, rope)

        # === Sol Spatial Modulation (image only) ===
        if sol_spatial is not None and spatial_size is not None:
            H, W = spatial_size

            sol_up = F.interpolate(
                sol_spatial.unsqueeze(1),
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            )

            mod = self.spatial_to_mod(sol_up)
            mod = mod.reshape(B, self.num_heads, H * W)
            mod = torch.exp(mod.clamp(-2, 2))  # exp(0)=1 at init, clamp for stability

            img_q = img_q * mod.unsqueeze(-1)
            img_k = img_k * mod.unsqueeze(-1)

        # Concatenate for joint attention
        k = torch.cat([txt_k, img_k], dim=2)
        v = torch.cat([txt_v, img_v], dim=2)

        # Text attention with SDPA (no Sol modulation)
        txt_out = F.scaled_dot_product_attention(txt_q, k, v, scale=self.scale)
        txt_out = txt_out.transpose(1, 2).reshape(B, L, -1)

        # Image attention with SDPA (Sol temperature via scale modification)
        if sol_temperature is not None:
            temp = sol_temperature.mean(dim=1, keepdim=True).clamp(min=0.1)
            effective_scale = self.scale / temp.unsqueeze(-1).unsqueeze(-1)
            img_q_scaled = img_q * (effective_scale.sqrt())
            k_scaled = k * (effective_scale.sqrt())
            img_out = F.scaled_dot_product_attention(img_q_scaled, k_scaled, v, scale=1.0)
        else:
            img_out = F.scaled_dot_product_attention(img_q, k, v, scale=self.scale)
        img_out = img_out.transpose(1, 2).reshape(B, N, -1)

        return self.txt_out(txt_out), self.img_out(img_out)


# =============================================================================
# MLP
# =============================================================================

class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

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
    """Double-stream transformer block with Sol prior support."""

    def __init__(self, config: TinyFluxConfig):
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        head_dim = config.attention_head_dim

        self.img_norm1 = AdaLayerNormZero(hidden)
        self.txt_norm1 = AdaLayerNormZero(hidden)
        self.attn = JointAttention(
            hidden, heads, head_dim,
            use_bias=False,
            sol_spatial_size=config.sol_spatial_size,
        )
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
    """Single-stream transformer block with Sol prior support."""

    def __init__(self, config: TinyFluxConfig):
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        head_dim = config.attention_head_dim

        self.norm = AdaLayerNormZeroSingle(hidden)
        self.attn = Attention(
            hidden, heads, head_dim,
            use_bias=False,
            sol_spatial_size=config.sol_spatial_size,
        )
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
        L = txt.shape[1]  # Number of text tokens
        x = torch.cat([txt, img], dim=1)
        x_normed, gate = self.norm(x, vec)

        # For single stream: text tokens come first, then image tokens
        # Sol spatial only applies to image portion
        x = x + gate.unsqueeze(1) * self.attn(
            x_normed, rope,
            sol_temperature=sol_temperature,
            sol_spatial=sol_spatial,
            spatial_size=spatial_size,
            num_txt_tokens=L,  # Tell attention how many text tokens to skip
        )
        x = x + self.mlp(self.norm2(x))
        txt, img = x.split([L, x.shape[1] - L], dim=1)
        return txt, img


# =============================================================================
# Main Model
# =============================================================================

class TinyFluxDeep(nn.Module):
    """
    TinyFlux-Deep v4.1 with Dual Expert System.

    Lune: Trajectory guidance → vec modulation (global conditioning)
    Sol: Attention prior → temperature/spatial (structural guidance)
    """

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

        # === T5 Enhancement: Add T5 to vec pathway ===
        if cfg.use_t5_vec:
            self.t5_pool = nn.Sequential(
                nn.Linear(cfg.joint_attention_dim, cfg.hidden_size),
                nn.SiLU(),
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
            )
            # Learnable balance: sigmoid(0) = 0.5 (equal weight at init)
            self.text_balance = nn.Parameter(torch.tensor(0.0))
        else:
            self.t5_pool = None
            self.text_balance = None

        # === Lune Expert Predictor (trajectory → vec) ===
        if cfg.use_lune_expert:
            self.lune_predictor = LuneExpertPredictor(
                time_dim=cfg.hidden_size,
                clip_dim=cfg.pooled_projection_dim,
                expert_dim=cfg.lune_expert_dim,
                hidden_dim=cfg.lune_hidden_dim,
                output_dim=cfg.hidden_size,
                dropout=cfg.lune_dropout,
            )
        else:
            self.lune_predictor = None

        # === Sol Attention Prior (structure → attention bias) ===
        if cfg.use_sol_prior:
            self.sol_prior = SolAttentionPrior(
                time_dim=cfg.hidden_size,
                clip_dim=cfg.pooled_projection_dim,
                hidden_dim=cfg.sol_hidden_dim,
                num_heads=cfg.num_attention_heads,
                spatial_size=cfg.sol_spatial_size,
                geometric_weight=cfg.sol_geometric_weight,
            )
        else:
            self.sol_prior = None

        # Legacy guidance
        if cfg.guidance_embeds:
            self.guidance_in = MLPEmbedder(cfg.hidden_size)
        else:
            self.guidance_in = None

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

    @property
    def expert_predictor(self):
        """Legacy API: alias for lune_predictor."""
        return self.lune_predictor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        # Lune inputs
        lune_features: Optional[torch.Tensor] = None,
        # Sol inputs
        sol_stats: Optional[torch.Tensor] = None,
        sol_spatial: Optional[torch.Tensor] = None,
        # Legacy API
        expert_features: Optional[torch.Tensor] = None,
        return_expert_pred: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [B, N, C] - image latents (flattened)
            encoder_hidden_states: [B, L, D] - T5 text embeddings
            pooled_projections: [B, D] - CLIP pooled features
            timestep: [B] - diffusion timestep in [0, 1]
            img_ids: [N, 3] or [B, N, 3] - image position IDs
            txt_ids: [L, 3] or [B, L, 3] - text position IDs (optional)
            guidance: [B] - legacy guidance scale
            lune_features: [B, 1280] - real Lune features (training)
            sol_stats: [B, 3] - real Sol statistics (training)
            sol_spatial: [B, H, W] - real Sol spatial importance (training)
            expert_features: [B, 1280] - legacy API, maps to lune_features
            return_expert_pred: if True, return (output, expert_info) tuple

        Returns:
            output: [B, N, C] - predicted velocity
            expert_info: dict (if return_expert_pred=True)
        """
        B = hidden_states.shape[0]
        L = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]

        # Infer spatial dimensions
        H = W = int(math.sqrt(N))
        assert H * W == N, f"N={N} is not a perfect square, cannot infer spatial size. Pass explicit spatial_size."
        spatial_size = (H, W)

        # Legacy API mapping
        if expert_features is not None and lune_features is None:
            lune_features = expert_features

        # Ensure consistent dtype (text encoders often output float32)
        model_dtype = self.img_in.weight.dtype
        hidden_states = hidden_states.to(dtype=model_dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=model_dtype)
        pooled_projections = pooled_projections.to(dtype=model_dtype)
        timestep = timestep.to(dtype=model_dtype)

        # Cast optional expert inputs if provided
        if lune_features is not None:
            lune_features = lune_features.to(dtype=model_dtype)
        if sol_stats is not None:
            sol_stats = sol_stats.to(dtype=model_dtype)
        if sol_spatial is not None:
            sol_spatial = sol_spatial.to(dtype=model_dtype)
        if guidance is not None:
            guidance = guidance.to(dtype=model_dtype)

        # Input projections
        img = self.img_in(hidden_states)
        txt = self.txt_in(encoder_hidden_states)

        # Conditioning: time + text
        time_emb = self.time_in(timestep)
        clip_vec = self.vector_in(pooled_projections)

        # === T5 Enhancement: Pool T5 and add to vec ===
        t5_pooled = None
        if self.t5_pool is not None:
            # Attention-weighted pooling of T5 sequence
            t5_attn_logits = encoder_hidden_states.mean(dim=-1)  # [B, L]
            t5_attn = F.softmax(t5_attn_logits, dim=-1)  # [B, L]
            t5_pooled = (encoder_hidden_states * t5_attn.unsqueeze(-1)).sum(dim=1)  # [B, D]
            t5_vec = self.t5_pool(t5_pooled)

            # Balanced combination of CLIP and T5
            balance = torch.sigmoid(self.text_balance)
            text_vec = balance * clip_vec + (1 - balance) * t5_vec
        else:
            text_vec = clip_vec

        vec = time_emb + text_vec

        # === Lune: trajectory guidance → vec ===
        lune_info = None
        if self.lune_predictor is not None:
            lune_out = self.lune_predictor(
                time_emb=time_emb,
                clip_pooled=pooled_projections,
                real_expert_features=lune_features,
            )
            vec = vec + lune_out['expert_signal']
            lune_info = lune_out

        # === Sol: attention prior → temperature, spatial ===
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

        # Legacy guidance (fallback)
        if self.guidance_in is not None and guidance is not None:
            vec = vec + self.guidance_in(guidance)

        # Handle img_ids shape
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

        # Build full sequence RoPE for single-stream
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
            expert_info = {
                'lune': lune_info,
                'sol': sol_info,
                # Legacy API
                'expert_signal': lune_info['expert_signal'] if lune_info else None,
                'expert_pred': lune_info['expert_pred'] if lune_info else None,
                'expert_used': lune_info['expert_used'] if lune_info else None,
            }
            return output, expert_info
        return output

    def compute_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        expert_info: Optional[Dict] = None,
        lune_features: Optional[torch.Tensor] = None,
        sol_stats: Optional[torch.Tensor] = None,
        sol_spatial: Optional[torch.Tensor] = None,
        lune_weight: float = 0.1,
        sol_weight: float = 0.05,
        # New options
        use_huber: bool = True,
        huber_delta: float = 0.1,
        lune_distill_mode: str = "cosine",
        spatial_weighting: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss with Huber and soft distillation.

        Args:
            output: [B, N, C] model prediction
            target: [B, N, C] flow matching target (data - noise)
            expert_info: dict from forward pass
            lune_features: [B, 1280] real Lune features
            sol_stats: [B, 3] real Sol statistics
            sol_spatial: [B, H, W] real Sol spatial importance
            lune_weight: weight for Lune distillation loss
            sol_weight: weight for Sol distillation loss
            use_huber: use Huber loss instead of MSE for main loss
            huber_delta: Huber delta (smaller = tighter MSE behavior)
            lune_distill_mode: "hard" (MSE), "cosine" (directional), "soft" (temp-scaled)
            spatial_weighting: weight main loss by Sol spatial importance

        Returns:
            dict with losses
        """
        device = output.device
        B, N, C = output.shape

        # === Main Flow Matching Loss ===
        if use_huber:
            # Huber loss: MSE for small errors, MAE for large (robust to outliers)
            main_loss_unreduced = F.huber_loss(
                output, target,
                reduction='none',
                delta=huber_delta
            )  # [B, N, C]
        else:
            main_loss_unreduced = (output - target).pow(2)  # [B, N, C]

        # === Sol Spatial Weighting ===
        if spatial_weighting and sol_spatial is not None:
            # Upsample Sol spatial to match token resolution
            H = W = int(math.sqrt(N))
            sol_weight_map = F.interpolate(
                sol_spatial.unsqueeze(1),  # [B, 1, H_sol, W_sol]
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            ).reshape(B, N, 1)  # [B, N, 1]

            # Normalize to mean=1 (doesn't change loss scale, just distribution)
            sol_weight_map = sol_weight_map / (sol_weight_map.mean() + 1e-6)

            # Apply spatial weighting
            main_loss_unreduced = main_loss_unreduced * sol_weight_map

        main_loss = main_loss_unreduced.mean()

        losses = {
            'main': main_loss,
            'lune_distill': torch.tensor(0.0, device=device),
            'sol_stat_distill': torch.tensor(0.0, device=device),
            'sol_spatial_distill': torch.tensor(0.0, device=device),
            'total': main_loss,
        }

        if expert_info is None:
            return losses

        # === Lune Distillation (Soft/Directional) ===
        if expert_info.get('lune') and lune_features is not None:
            lune_pred = expert_info['lune']['expert_pred']

            if lune_distill_mode == "cosine":
                # Directional matching - Lune is a guide, not exact target
                # "Go in the same direction" without forcing exact values
                pred_norm = F.normalize(lune_pred, dim=-1)
                real_norm = F.normalize(lune_features, dim=-1)
                cosine_sim = (pred_norm * real_norm).sum(dim=-1)
                losses['lune_distill'] = (1 - cosine_sim).mean()

            elif lune_distill_mode == "soft":
                # Temperature-scaled MSE (mushier matching)
                temp = 2.0  # Higher = softer
                mse = (lune_pred - lune_features).pow(2).mean(dim=-1)
                losses['lune_distill'] = (mse / temp).mean()

            elif lune_distill_mode == "huber":
                # Huber for distillation too
                losses['lune_distill'] = F.huber_loss(
                    lune_pred, lune_features, delta=1.0
                )

            else:  # "hard" - original MSE
                losses['lune_distill'] = F.mse_loss(lune_pred, lune_features)

        # === Sol Distillation (keeps MSE - small vectors, precision matters) ===
        if expert_info.get('sol'):
            if sol_stats is not None:
                sol_pred_stats = expert_info['sol']['pred_stats']
                losses['sol_stat_distill'] = F.mse_loss(sol_pred_stats, sol_stats)

            if sol_spatial is not None:
                sol_pred_spatial = expert_info['sol']['pred_spatial']
                losses['sol_spatial_distill'] = F.mse_loss(sol_pred_spatial, sol_spatial)

        # === Total ===
        losses['total'] = (
            main_loss +
            lune_weight * losses['lune_distill'] +
            sol_weight * (losses['sol_stat_distill'] + losses['sol_spatial_distill'])
        )

        return losses

    @staticmethod
    def create_img_ids(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create image position IDs for RoPE."""
        img_ids = torch.zeros(height * width, 3, device=device)
        for i in range(height):
            for j in range(width):
                idx = i * width + j
                img_ids[idx, 0] = 0
                img_ids[idx, 1] = i
                img_ids[idx, 2] = j
        return img_ids

    @staticmethod
    def create_txt_ids(text_len: int, device: torch.device) -> torch.Tensor:
        """Create text position IDs."""
        txt_ids = torch.zeros(text_len, 3, device=device)
        txt_ids[:, 0] = torch.arange(text_len, device=device)
        return txt_ids

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts['img_in'] = sum(p.numel() for p in self.img_in.parameters())
        counts['txt_in'] = sum(p.numel() for p in self.txt_in.parameters())
        counts['time_in'] = sum(p.numel() for p in self.time_in.parameters())
        counts['vector_in'] = sum(p.numel() for p in self.vector_in.parameters())

        if self.t5_pool is not None:
            counts['t5_pool'] = sum(p.numel() for p in self.t5_pool.parameters()) + 1  # +1 for balance param
        if self.lune_predictor is not None:
            counts['lune_predictor'] = sum(p.numel() for p in self.lune_predictor.parameters())
        if self.sol_prior is not None:
            counts['sol_prior'] = sum(p.numel() for p in self.sol_prior.parameters())
        if self.guidance_in is not None:
            counts['guidance_in'] = sum(p.numel() for p in self.guidance_in.parameters())

        counts['double_blocks'] = sum(p.numel() for p in self.double_blocks.parameters())
        counts['single_blocks'] = sum(p.numel() for p in self.single_blocks.parameters())
        counts['final'] = sum(p.numel() for p in self.final_norm.parameters()) + \
                          sum(p.numel() for p in self.final_linear.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts


# =============================================================================
# Test
# =============================================================================

def test_model():
    """Test TinyFlux-Deep v4.1 with Dual Expert System."""
    print("=" * 60)
    print(f"TinyFlux-Deep v{__version__} - Dual Expert Test")
    print("=" * 60)

    config = TinyFluxConfig(
        use_lune_expert=True,
        use_sol_prior=True,
        lune_expert_dim=1280,
        sol_spatial_size=8,
        sol_geometric_weight=0.7,
        use_t5_vec=True,
        lune_distill_mode="cosine",
        use_huber_loss=True,
        huber_delta=0.1,
    )
    model = TinyFluxDeep(config)

    counts = model.count_parameters()
    print(f"\nConfig:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_double_layers: {config.num_double_layers}")
    print(f"  num_single_layers: {config.num_single_layers}")
    print(f"  use_lune_expert: {config.use_lune_expert}")
    print(f"  use_sol_prior: {config.use_sol_prior}")
    print(f"  sol_geometric_weight: {config.sol_geometric_weight}")
    print(f"  use_t5_vec: {config.use_t5_vec}")
    print(f"  lune_distill_mode: {config.lune_distill_mode}")
    print(f"  use_huber_loss: {config.use_huber_loss}")

    print(f"\nParameters:")
    for name, count in counts.items():
        print(f"  {name}: {count:,}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    B, H, W = 2, 64, 64
    L = 77

    hidden_states = torch.randn(B, H * W, config.in_channels, device=device)
    encoder_hidden_states = torch.randn(B, L, config.joint_attention_dim, device=device)
    pooled_projections = torch.randn(B, config.pooled_projection_dim, device=device)
    timestep = torch.rand(B, device=device)
    img_ids = TinyFluxDeep.create_img_ids(B, H, W, device)

    # Expert inputs
    lune_features = torch.randn(B, config.lune_expert_dim, device=device)
    sol_stats = torch.randn(B, 3, device=device)
    sol_spatial = torch.rand(B, config.sol_spatial_size, config.sol_spatial_size, device=device)

    print("\n[Test 1: Training mode with dual experts]")
    model.train()
    with torch.no_grad():
        output, expert_info = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            lune_features=lune_features,
            sol_stats=sol_stats,
            sol_spatial=sol_spatial,
            return_expert_pred=True,
        )
    print(f"  Output shape: {output.shape}")
    print(f"  Lune used: {expert_info['lune']['expert_used']}")
    print(f"  Sol temperature shape: {expert_info['sol']['temperature'].shape}")
    print(f"  Sol spatial shape: {expert_info['sol']['spatial_importance'].shape}")

    print("\n[Test 2: Inference mode (no expert inputs)]")
    model.eval()
    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
        )
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    print("\n[Test 3: Loss computation with Huber + Cosine distillation]")
    target = torch.randn_like(output)
    model.train()
    output, expert_info = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        timestep=timestep,
        img_ids=img_ids,
        lune_features=lune_features,
        sol_stats=sol_stats,
        sol_spatial=sol_spatial,
        return_expert_pred=True,
    )
    losses = model.compute_loss(
        output=output,
        target=target,
        expert_info=expert_info,
        lune_features=lune_features,
        sol_stats=sol_stats,
        sol_spatial=sol_spatial,
        lune_weight=0.1,
        sol_weight=0.05,
        use_huber=True,
        huber_delta=0.1,
        lune_distill_mode="cosine",
        spatial_weighting=True,
    )
    print(f"  Main loss (Huber): {losses['main']:.4f}")
    print(f"  Lune distill (cosine): {losses['lune_distill']:.4f}")
    print(f"  Sol stat distill: {losses['sol_stat_distill']:.4f}")
    print(f"  Sol spatial distill: {losses['sol_spatial_distill']:.4f}")
    print(f"  Total loss: {losses['total']:.4f}")

    print("\n[Test 4: Legacy API compatibility]")
    with torch.no_grad():
        output, expert_info = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            expert_features=lune_features,  # Legacy API
            return_expert_pred=True,
        )
    print(f"  Legacy expert_pred shape: {expert_info['expert_pred'].shape}")
    print(f"  Legacy expert_used: {expert_info['expert_used']}")

    print("\n[Test 5: T5 Enhancement check]")
    if model.t5_pool is not None:
        balance = torch.sigmoid(model.text_balance).item()
        print(f"  T5 pool: enabled")
        print(f"  Text balance (CLIP vs T5): {balance:.2f} / {1-balance:.2f}")
    else:
        print(f"  T5 pool: disabled")

    print("\n[Test 6: Config serialization]")
    config_dict = config.to_dict()
    config_restored = TinyFluxConfig.from_dict(config_dict)
    print(f"  Serialized keys: {len(config_dict)}")
    print(f"  Restored hidden_size: {config_restored.hidden_size}")
    print(f"  Round-trip successful: {config.hidden_size == config_restored.hidden_size}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


#if __name__ == "__main__":
#   test_model()