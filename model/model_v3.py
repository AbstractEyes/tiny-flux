"""
TinyFlux-Deep with Expert Predictor

Integrates a distillation pathway for SD1.5-flow timestep expertise.
During training: learns to predict expert features from (timestep, CLIP).
During inference: runs standalone, no expert needed.

Based on TinyFlux-Deep: 15 double + 25 single blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class TinyFluxDeepConfig:
    """Configuration for TinyFlux-Deep model."""
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

    # Expert predictor config
    use_expert_predictor: bool = True
    expert_dim: int = 1280  # SD1.5 mid-block dimension
    expert_hidden_dim: int = 512
    expert_dropout: float = 0.1  # Dropout during training for robustness

    # Legacy guidance (disabled when using expert)
    guidance_embeds: bool = False

    def __post_init__(self):
        assert self.num_attention_heads * self.attention_head_dim == self.hidden_size
        assert sum(self.axes_dims_rope) == self.attention_head_dim


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
# RoPE - Old format with cached frequency buffers
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
# Expert Predictor
# =============================================================================

class ExpertPredictor(nn.Module):
    """
    Predicts SD1.5-flow expert features from (timestep_emb, CLIP_pooled).

    Training: learns to match real expert features via distillation loss.
    Inference: runs standalone, no expert model needed.

    The predictor learns:
    - What the expert "sees" at each timestep
    - How text conditioning modulates that view
    - Trajectory shape priors from the expert's knowledge
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

        # Predictor core - learns expert behavior
        self.predictor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, expert_dim),
        )

        # Project predicted expert features to vec dimension
        self.output_proj = nn.Sequential(
            nn.LayerNorm(expert_dim),
            nn.Linear(expert_dim, output_dim),
        )

        # Learnable gate for expert influence
        self.expert_gate = nn.Parameter(torch.ones(1) * 0.5)

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
            force_predictor: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            time_emb: [B, time_dim] - timestep embedding from time_in
            clip_pooled: [B, clip_dim] - pooled CLIP features
            real_expert_features: [B, expert_dim] - real expert output (training only)
            force_predictor: if True, use predictor even when real features available

        Returns:
            dict with:
                - 'expert_signal': [B, output_dim] - signal to add to vec
                - 'expert_pred': [B, expert_dim] - predicted expert features (for loss)
                - 'expert_used': str - 'real' or 'predicted'
        """
        B = time_emb.shape[0]
        device = time_emb.device

        # Fuse inputs
        combined = torch.cat([time_emb, clip_pooled], dim=-1)
        hidden = self.input_proj(combined)

        # Predict expert features
        expert_pred = self.predictor(hidden)

        # Use real features if provided, otherwise predicted
        # Dropout logic moved to trainer to avoid graph breaks with torch.compile
        if real_expert_features is not None and not force_predictor:
            expert_features = real_expert_features
            expert_used = 'real'
        else:
            expert_features = expert_pred
            expert_used = 'predicted'

        # Project to output dimension with gating
        gate = torch.sigmoid(self.expert_gate)
        expert_signal = gate * self.output_proj(expert_features)

        return {
            'expert_signal': expert_signal,
            'expert_pred': expert_pred,
            'expert_used': expert_used,
        }

    def compute_distillation_loss(
            self,
            expert_pred: torch.Tensor,
            real_expert_features: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss between predicted and real expert features."""
        return F.mse_loss(expert_pred, real_expert_features)


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
# Attention
# =============================================================================

class Attention(nn.Module):
    """Multi-head attention."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, use_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=use_bias)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=use_bias)

    def forward(self, x: torch.Tensor, rope: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        if rope is not None:
            q = apply_rotary_emb_old(q, rope)
            k = apply_rotary_emb_old(k, rope)

        attn = F.scaled_dot_product_attention(q, k, v)
        out = attn.transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)


class JointAttention(nn.Module):
    """Joint attention for double-stream blocks."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, use_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.txt_qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=use_bias)
        self.img_qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=use_bias)

        self.txt_out = nn.Linear(num_heads * head_dim, hidden_size, bias=use_bias)
        self.img_out = nn.Linear(num_heads * head_dim, hidden_size, bias=use_bias)

    def forward(
            self,
            txt: torch.Tensor,
            img: torch.Tensor,
            rope: Optional[torch.Tensor] = None,
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

        k = torch.cat([txt_k, img_k], dim=2)
        v = torch.cat([txt_v, img_v], dim=2)

        txt_out = F.scaled_dot_product_attention(txt_q, k, v)
        txt_out = txt_out.transpose(1, 2).reshape(B, L, -1)

        img_out = F.scaled_dot_product_attention(img_q, k, v)
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
    """Double-stream transformer block."""

    def __init__(self, config: TinyFluxDeepConfig):
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        head_dim = config.attention_head_dim

        self.img_norm1 = AdaLayerNormZero(hidden)
        self.txt_norm1 = AdaLayerNormZero(hidden)
        self.attn = JointAttention(hidden, heads, head_dim, use_bias=False)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_normed, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = self.img_norm1(img, vec)
        txt_normed, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = self.txt_norm1(txt, vec)

        txt_attn_out, img_attn_out = self.attn(txt_normed, img_normed, rope)

        txt = txt + txt_gate_msa.unsqueeze(1) * txt_attn_out
        img = img + img_gate_msa.unsqueeze(1) * img_attn_out

        txt_mlp_in = self.txt_norm2(txt) * (1 + txt_scale_mlp.unsqueeze(1)) + txt_shift_mlp.unsqueeze(1)
        img_mlp_in = self.img_norm2(img) * (1 + img_scale_mlp.unsqueeze(1)) + img_shift_mlp.unsqueeze(1)

        txt = txt + txt_gate_mlp.unsqueeze(1) * self.txt_mlp(txt_mlp_in)
        img = img + img_gate_mlp.unsqueeze(1) * self.img_mlp(img_mlp_in)

        return txt, img


class SingleStreamBlock(nn.Module):
    """Single-stream transformer block."""

    def __init__(self, config: TinyFluxDeepConfig):
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        head_dim = config.attention_head_dim

        self.norm = AdaLayerNormZeroSingle(hidden)
        self.attn = Attention(hidden, heads, head_dim, use_bias=False)
        self.mlp = MLP(hidden, config.mlp_ratio)
        self.norm2 = RMSNorm(hidden)

    def forward(
            self,
            txt: torch.Tensor,
            img: torch.Tensor,
            vec: torch.Tensor,
            rope: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        L = txt.shape[1]
        x = torch.cat([txt, img], dim=1)
        x_normed, gate = self.norm(x, vec)
        x = x + gate.unsqueeze(1) * self.attn(x_normed, rope)
        x = x + self.mlp(self.norm2(x))
        txt, img = x.split([L, x.shape[1] - L], dim=1)
        return txt, img


# =============================================================================
# Main Model
# =============================================================================

class TinyFluxDeep(nn.Module):
    """
    TinyFlux-Deep with Expert Predictor.

    The expert predictor learns to emulate SD1.5-flow's timestep expertise,
    allowing the model to benefit from trajectory priors without requiring
    the expert model at inference time.
    """

    def __init__(self, config: Optional[TinyFluxDeepConfig] = None):
        super().__init__()
        self.config = config or TinyFluxDeepConfig()
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

        # Expert predictor (replaces guidance_in)
        if cfg.use_expert_predictor:
            self.expert_predictor = ExpertPredictor(
                time_dim=cfg.hidden_size,
                clip_dim=cfg.pooled_projection_dim,
                expert_dim=cfg.expert_dim,
                hidden_dim=cfg.expert_hidden_dim,
                output_dim=cfg.hidden_size,
                dropout=cfg.expert_dropout,
            )
        else:
            self.expert_predictor = None

        # Legacy guidance (for backward compat / comparison)
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

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            pooled_projections: torch.Tensor,
            timestep: torch.Tensor,
            img_ids: torch.Tensor,
            txt_ids: Optional[torch.Tensor] = None,
            guidance: Optional[torch.Tensor] = None,
            expert_features: Optional[torch.Tensor] = None,
            return_expert_pred: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [B, N, C] - image latents
            encoder_hidden_states: [B, L, D] - T5 text embeddings
            pooled_projections: [B, D] - CLIP pooled features
            timestep: [B] - diffusion timestep
            img_ids: [N, 3] or [B, N, 3] - image position IDs
            txt_ids: [L, 3] or [B, L, 3] - text position IDs (optional)
            guidance: [B] - legacy guidance scale (if guidance_embeds=True)
            expert_features: [B, 1280] - real expert features (training only)
            return_expert_pred: if True, return (output, expert_info) tuple

        Returns:
            output: [B, N, C] - predicted velocity
            expert_info: dict (if return_expert_pred=True)
        """
        B = hidden_states.shape[0]
        L = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]

        # Input projections
        img = self.img_in(hidden_states)
        txt = self.txt_in(encoder_hidden_states)

        # Conditioning: time + pooled text
        time_emb = self.time_in(timestep)
        vec = time_emb + self.vector_in(pooled_projections)

        # Expert predictor (third stream)
        expert_info = None
        if self.expert_predictor is not None:
            expert_out = self.expert_predictor(
                time_emb=time_emb,
                clip_pooled=pooled_projections,
                real_expert_features=expert_features,
            )
            vec = vec + expert_out['expert_signal']
            expert_info = expert_out

        # Legacy guidance (fallback)
        elif self.guidance_in is not None and guidance is not None:
            vec = vec + self.guidance_in(guidance)

        # Handle img_ids shape
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        img_rope = self.rope(img_ids)

        # Double-stream blocks
        for block in self.double_blocks:
            txt, img = block(txt, img, vec, img_rope)

        # Build full sequence RoPE for single-stream
        if txt_ids is None:
            txt_ids = torch.zeros(L, 3, device=img_ids.device, dtype=img_ids.dtype)
        elif txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        all_ids = torch.cat([txt_ids, img_ids], dim=0)
        full_rope = self.rope(all_ids)

        # Single-stream blocks
        for block in self.single_blocks:
            txt, img = block(txt, img, vec, full_rope)

        # Output
        img = self.final_norm(img)
        output = self.final_linear(img)

        if return_expert_pred:
            return output, expert_info
        return output

    def compute_loss(
            self,
            output: torch.Tensor,
            target: torch.Tensor,
            expert_pred: Optional[torch.Tensor] = None,
            real_expert_features: Optional[torch.Tensor] = None,
            distill_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            output: model prediction
            target: flow matching target (data - noise)
            expert_pred: predicted expert features
            real_expert_features: real expert features
            distill_weight: weight for distillation loss

        Returns:
            dict with 'total', 'main', 'distill' losses
        """
        # Main flow matching loss
        main_loss = F.mse_loss(output, target)

        losses = {
            'main': main_loss,
            'distill': torch.tensor(0.0, device=output.device),
            'total': main_loss,
        }

        # Distillation loss
        if expert_pred is not None and real_expert_features is not None:
            distill_loss = self.expert_predictor.compute_distillation_loss(
                expert_pred, real_expert_features
            )
            losses['distill'] = distill_loss
            losses['total'] = main_loss + distill_weight * distill_loss

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

        if self.expert_predictor is not None:
            counts['expert_predictor'] = sum(p.numel() for p in self.expert_predictor.parameters())
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
    """Test TinyFlux-Deep with Expert Predictor."""
    print("=" * 60)
    print("TinyFlux-Deep + Expert Predictor Test")
    print("=" * 60)

    config = TinyFluxDeepConfig(
        use_expert_predictor=True,
        expert_dim=1280,
        expert_hidden_dim=512,
        guidance_embeds=False,
    )
    model = TinyFluxDeep(config)

    counts = model.count_parameters()
    print(f"\nConfig:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_double_layers: {config.num_double_layers}")
    print(f"  num_single_layers: {config.num_single_layers}")
    print(f"  expert_dim: {config.expert_dim}")
    print(f"  use_expert_predictor: {config.use_expert_predictor}")

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
    txt_ids = TinyFluxDeep.create_txt_ids(L, device)

    # Simulated expert features
    expert_features = torch.randn(B, config.expert_dim, device=device)

    print("\n[Test 1: Training mode with expert features]")
    model.train()
    with torch.no_grad():
        output, expert_info = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            expert_features=expert_features,
            return_expert_pred=True,
        )
    print(f"  Output shape: {output.shape}")
    print(f"  Expert used: {expert_info['expert_used']}")
    print(f"  Expert pred shape: {expert_info['expert_pred'].shape}")

    print("\n[Test 2: Inference mode (no expert)]")
    model.eval()
    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            expert_features=None,  # No expert at inference
        )
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    print("\n[Test 3: Loss computation]")
    target = torch.randn_like(output)
    model.train()
    output, expert_info = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        expert_features=expert_features,
        return_expert_pred=True,
    )
    losses = model.compute_loss(
        output=output,
        target=target,
        expert_pred=expert_info['expert_pred'],
        real_expert_features=expert_features,
        distill_weight=0.1,
    )
    print(f"  Main loss: {losses['main']:.4f}")
    print(f"  Distill loss: {losses['distill']:.4f}")
    print(f"  Total loss: {losses['total']:.4f}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()