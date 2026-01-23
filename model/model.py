"""
TinyFlux: A /12 scaled Flux architecture for experimentation.

Architecture:
  - hidden: 256 (3072/12)
  - num_heads: 2 (24/12)
  - head_dim: 128 (preserved for RoPE compatibility)
  - in_channels: 16 (Flux VAE output channels)
  - double_layers: 3
  - single_layers: 3

Text Encoders (runtime):
  - flan-t5-base (768 dim) → txt_in projects to hidden
  - CLIP-L (768 dim pooled) → vector_in projects to hidden
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TinyFluxConfig:
    """Configuration for TinyFlux model."""
    # Core dimensions
    hidden_size: int = 256
    num_attention_heads: int = 2
    attention_head_dim: int = 128  # Preserved for RoPE

    # Input/output (Flux VAE has 16 channels)
    in_channels: int = 16  # Flux VAE output channels
    patch_size: int = 1  # No 2x2 patchification, raw latent tokens

    # Text encoder interfaces (runtime encoding)
    joint_attention_dim: int = 768  # flan-t5-base output dim
    pooled_projection_dim: int = 768  # CLIP-L pooled dim

    # Layers
    num_double_layers: int = 3
    num_single_layers: int = 3

    # MLP
    mlp_ratio: float = 4.0

    # RoPE (must sum to head_dim)
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)

    # Misc
    guidance_embeds: bool = True

    def __post_init__(self):
        assert self.num_attention_heads * self.attention_head_dim == self.hidden_size, \
            f"heads ({self.num_attention_heads}) * head_dim ({self.attention_head_dim}) != hidden ({self.hidden_size})"
        assert sum(self.axes_dims_rope) == self.attention_head_dim, \
            f"RoPE dims {self.axes_dims_rope} must sum to head_dim {self.attention_head_dim}"


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for 2D + temporal."""

    def __init__(self, dim: int, axes_dims: Tuple[int, int, int], theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.axes_dims = axes_dims  # (temporal, height, width)
        self.theta = theta

    def forward(self, ids: torch.Tensor, dtype: torch.dtype = None) -> torch.Tensor:
        """
        ids: (B, N, 3) - temporal, height, width indices
        dtype: output dtype (defaults to ids.dtype, but use model dtype for bf16)
        Returns: (B, N, dim) rotary embeddings
        """
        B, N, _ = ids.shape
        device = ids.device
        # Compute in float32 for precision, cast at the end
        compute_dtype = torch.float32
        output_dtype = dtype if dtype is not None else ids.dtype

        embeddings = []
        dim_offset = 0

        for axis_idx, axis_dim in enumerate(self.axes_dims):
            # Compute frequencies for this axis
            freqs = 1.0 / (self.theta ** (torch.arange(0, axis_dim, 2, device=device, dtype=compute_dtype) / axis_dim))
            # Get positions for this axis
            positions = ids[:, :, axis_idx].to(compute_dtype)  # (B, N)
            # Outer product: (B, N) x (axis_dim/2) -> (B, N, axis_dim/2)
            angles = positions.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
            # Interleave sin/cos
            emb = torch.stack([angles.cos(), angles.sin()], dim=-1)  # (B, N, axis_dim/2, 2)
            emb = emb.flatten(-2)  # (B, N, axis_dim)
            embeddings.append(emb)
            dim_offset += axis_dim

        result = torch.cat(embeddings, dim=-1)  # (B, N, dim)
        return result.to(output_dtype)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor."""
    # x: (B, heads, N, head_dim)
    # rope: (B, N, head_dim)
    B, H, N, D = x.shape

    # Ensure rope matches x dtype
    rope = rope.to(x.dtype).unsqueeze(1)  # (B, 1, N, D)

    # Split into pairs
    x_pairs = x.reshape(B, H, N, D // 2, 2)
    rope_pairs = rope.reshape(B, 1, N, D // 2, 2)

    cos = rope_pairs[..., 0]
    sin = rope_pairs[..., 1]

    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]

    out0 = x0 * cos - x1 * sin
    out1 = x1 * cos + x0 * sin

    return torch.stack([out0, out1], dim=-1).flatten(-2)


class MLPEmbedder(nn.Module):
    """MLP for embedding scalars (timestep, guidance)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sinusoidal embedding first
        half_dim = 128
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, 256)
        return self.mlp(emb)


class AdaLayerNormZero(nn.Module):
    """
    AdaLN-Zero for double-stream blocks.
    Outputs 6 modulation params: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.norm = RMSNorm(hidden_size)

    def forward(
            self, x: torch.Tensor, emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: hidden states (B, N, D)
            emb: conditioning embedding (B, D)
        Returns:
            (normed_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        emb_out = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb_out.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    """
    AdaLN-Zero for single-stream blocks.
    Outputs 3 modulation params: (shift, scale, gate)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.norm = RMSNorm(hidden_size)

    def forward(
            self, x: torch.Tensor, emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: hidden states (B, N, D)
            emb: conditioning embedding (B, D)
        Returns:
            (normed_x, gate)
        """
        emb_out = self.linear(self.silu(emb))
        shift, scale, gate = emb_out.chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x, gate


class Attention(nn.Module):
    """Multi-head attention with optional RoPE."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
            self,
            x: torch.Tensor,
            rope: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, _ = x.shape
        dtype = x.dtype

        # Ensure RoPE matches input dtype
        if rope is not None:
            rope = rope.to(dtype)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 x (B, heads, N, head_dim)

        if rope is not None:
            q = apply_rope(q, rope)
            k = apply_rope(k, rope)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)


class JointAttention(nn.Module):
    """Joint attention for double-stream blocks (separate Q,K,V for txt and img)."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Separate projections for text and image
        self.txt_qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.img_qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)

        self.txt_out = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.img_out = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
            self,
            txt: torch.Tensor,
            img: torch.Tensor,
            rope: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = txt.shape
        _, N, _ = img.shape

        # Ensure consistent dtype (use img dtype as reference)
        dtype = img.dtype
        txt = txt.to(dtype)
        if rope is not None:
            rope = rope.to(dtype)

        # Compute Q, K, V for both streams
        txt_qkv = self.txt_qkv(txt).reshape(B, L, 3, self.num_heads, self.head_dim)
        img_qkv = self.img_qkv(img).reshape(B, N, 3, self.num_heads, self.head_dim)

        txt_q, txt_k, txt_v = txt_qkv.permute(2, 0, 3, 1, 4)
        img_q, img_k, img_v = img_qkv.permute(2, 0, 3, 1, 4)

        # Apply RoPE to image queries/keys only (text doesn't have positions)
        if rope is not None:
            img_q = apply_rope(img_q, rope)
            img_k = apply_rope(img_k, rope)

        # Concatenate keys and values for joint attention
        k = torch.cat([txt_k, img_k], dim=2)  # (B, heads, L+N, head_dim)
        v = torch.cat([txt_v, img_v], dim=2)

        # Text attends to all
        txt_attn = (txt_q @ k.transpose(-2, -1)) * self.scale
        txt_attn = txt_attn.softmax(dim=-1)
        txt_out = (txt_attn @ v).transpose(1, 2).reshape(B, L, -1)

        # Image attends to all
        img_attn = (img_q @ k.transpose(-2, -1)) * self.scale
        img_attn = img_attn.softmax(dim=-1)
        img_out = (img_attn @ v).transpose(1, 2).reshape(B, N, -1)

        return self.txt_out(txt_out), self.img_out(img_out)


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(mlp_hidden, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DoubleStreamBlock(nn.Module):
    """
    Double-stream transformer block (MMDiT style).
    Text and image have separate weights but attend to each other.
    Uses AdaLN-Zero with 6 modulation params per stream.
    """

    def __init__(self, config: TinyFluxConfig):
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        head_dim = config.attention_head_dim
        mlp_hidden = int(hidden * config.mlp_ratio)

        # AdaLN-Zero for each stream (outputs 6 params each)
        self.img_norm1 = AdaLayerNormZero(hidden)
        self.txt_norm1 = AdaLayerNormZero(hidden)

        # Joint attention (separate QKV projections)
        self.attn = JointAttention(hidden, heads, head_dim)

        # Second norm for MLP (not adaptive, uses params from norm1)
        self.img_norm2 = RMSNorm(hidden)
        self.txt_norm2 = RMSNorm(hidden)

        # MLPs
        self.img_mlp = MLP(hidden, config.mlp_ratio)
        self.txt_mlp = MLP(hidden, config.mlp_ratio)

    def forward(
            self,
            txt: torch.Tensor,
            img: torch.Tensor,
            vec: torch.Tensor,
            rope: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Image stream: norm + modulation, get MLP params for later
        img_normed, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = self.img_norm1(img, vec)

        # Text stream: norm + modulation, get MLP params for later
        txt_normed, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = self.txt_norm1(txt, vec)

        # Joint attention
        txt_attn_out, img_attn_out = self.attn(txt_normed, img_normed, rope)

        # Residual with gate
        txt = txt + txt_gate_msa.unsqueeze(1) * txt_attn_out
        img = img + img_gate_msa.unsqueeze(1) * img_attn_out

        # MLP with modulation (using params from norm1)
        txt_mlp_in = self.txt_norm2(txt) * (1 + txt_scale_mlp.unsqueeze(1)) + txt_shift_mlp.unsqueeze(1)
        img_mlp_in = self.img_norm2(img) * (1 + img_scale_mlp.unsqueeze(1)) + img_shift_mlp.unsqueeze(1)

        txt = txt + txt_gate_mlp.unsqueeze(1) * self.txt_mlp(txt_mlp_in)
        img = img + img_gate_mlp.unsqueeze(1) * self.img_mlp(img_mlp_in)

        return txt, img


class SingleStreamBlock(nn.Module):
    """
    Single-stream transformer block.
    Text and image are concatenated and share weights.
    Uses AdaLN-Zero with 3 modulation params (no separate MLP modulation).
    """

    def __init__(self, config: TinyFluxConfig):
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        head_dim = config.attention_head_dim
        mlp_hidden = int(hidden * config.mlp_ratio)

        # AdaLN-Zero (outputs 3 params: shift, scale, gate)
        self.norm = AdaLayerNormZeroSingle(hidden)

        # Combined QKV + MLP projection (Flux fuses these)
        # Linear attention: QKV projection
        self.attn = Attention(hidden, heads, head_dim)

        # MLP
        self.mlp = MLP(hidden, config.mlp_ratio)

        # Pre-MLP norm (not modulated in single-stream)
        self.norm2 = RMSNorm(hidden)

    def forward(
            self,
            txt: torch.Tensor,
            img: torch.Tensor,
            vec: torch.Tensor,
            txt_rope: Optional[torch.Tensor] = None,
            img_rope: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        L = txt.shape[1]

        # Concatenate txt and img
        x = torch.cat([txt, img], dim=1)

        # Concatenate RoPE (zeros for text positions)
        if img_rope is not None:
            B, N, D = img_rope.shape
            txt_rope_zeros = torch.zeros(B, L, D, device=img_rope.device, dtype=img_rope.dtype)
            rope = torch.cat([txt_rope_zeros, img_rope], dim=1)
        else:
            rope = None

        # Norm + modulation (only 3 params for single stream)
        x_normed, gate = self.norm(x, vec)

        # Attention with gated residual
        x = x + gate.unsqueeze(1) * self.attn(x_normed, rope)

        # MLP (no separate modulation in single-stream Flux)
        x = x + self.mlp(self.norm2(x))

        # Split back
        txt, img = x.split([L, x.shape[1] - L], dim=1)
        return txt, img


class TinyFlux(nn.Module):
    """
    TinyFlux: A scaled-down Flux diffusion transformer.

    Scaling: /12 from original Flux
      - hidden: 3072 → 256
      - heads: 24 → 2
      - head_dim: 128 (preserved)
      - in_channels: 16 (Flux VAE)
    """

    def __init__(self, config: Optional[TinyFluxConfig] = None):
        super().__init__()
        self.config = config or TinyFluxConfig()
        cfg = self.config

        # Input projections
        self.img_in = nn.Linear(cfg.in_channels, cfg.hidden_size)
        self.txt_in = nn.Linear(cfg.joint_attention_dim, cfg.hidden_size)

        # Conditioning projections
        self.time_in = MLPEmbedder(cfg.hidden_size)
        self.vector_in = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.pooled_projection_dim, cfg.hidden_size)
        )
        if cfg.guidance_embeds:
            self.guidance_in = MLPEmbedder(cfg.hidden_size)

        # RoPE
        self.rope = RotaryEmbedding(cfg.attention_head_dim, cfg.axes_dims_rope)

        # Transformer blocks
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(cfg) for _ in range(cfg.num_double_layers)
        ])
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(cfg) for _ in range(cfg.num_single_layers)
        ])

        # Output
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.final_linear = nn.Linear(cfg.hidden_size, cfg.in_channels)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""

        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init)

        # Zero-init output projection for residual
        nn.init.zeros_(self.final_linear.weight)

    def forward(
            self,
            hidden_states: torch.Tensor,  # (B, N, in_channels) - image patches
            encoder_hidden_states: torch.Tensor,  # (B, L, joint_attention_dim) - T5 tokens
            pooled_projections: torch.Tensor,  # (B, pooled_projection_dim) - CLIP pooled
            timestep: torch.Tensor,  # (B,) - diffusion timestep
            img_ids: torch.Tensor,  # (B, N, 3) - image position ids
            guidance: Optional[torch.Tensor] = None,  # (B,) - guidance scale
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            Predicted noise/velocity of shape (B, N, in_channels)
        """
        # Input projections
        img = self.img_in(hidden_states)  # (B, N, hidden)
        txt = self.txt_in(encoder_hidden_states)  # (B, L, hidden)

        # Conditioning vector
        vec = self.time_in(timestep)
        vec = vec + self.vector_in(pooled_projections)
        if self.config.guidance_embeds and guidance is not None:
            vec = vec + self.guidance_in(guidance)

        # RoPE for image positions (match model dtype)
        img_rope = self.rope(img_ids, dtype=img.dtype)

        # Double-stream blocks
        for block in self.double_blocks:
            txt, img = block(txt, img, vec, img_rope)

        # Single-stream blocks
        for block in self.single_blocks:
            txt, img = block(txt, img, vec, img_rope=img_rope)

        # Output (image only)
        img = self.final_norm(img)
        img = self.final_linear(img)

        return img

    @staticmethod
    def create_img_ids(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create image position IDs for RoPE."""
        # height, width are in latent space (image_size / 8)
        img_ids = torch.zeros(batch_size, height * width, 3, device=device)

        for i in range(height):
            for j in range(width):
                idx = i * width + j
                img_ids[:, idx, 0] = 0  # temporal (always 0 for images)
                img_ids[:, idx, 1] = i  # height
                img_ids[:, idx, 2] = j  # width

        return img_ids

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        counts = {}
        counts['img_in'] = sum(p.numel() for p in self.img_in.parameters())
        counts['txt_in'] = sum(p.numel() for p in self.txt_in.parameters())
        counts['time_in'] = sum(p.numel() for p in self.time_in.parameters())
        counts['vector_in'] = sum(p.numel() for p in self.vector_in.parameters())
        if hasattr(self, 'guidance_in'):
            counts['guidance_in'] = sum(p.numel() for p in self.guidance_in.parameters())
        counts['double_blocks'] = sum(p.numel() for p in self.double_blocks.parameters())
        counts['single_blocks'] = sum(p.numel() for p in self.single_blocks.parameters())
        counts['final'] = sum(p.numel() for p in self.final_norm.parameters()) + \
                          sum(p.numel() for p in self.final_linear.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts


def test_tiny_flux():
    """Quick test of the model."""
    print("=" * 60)
    print("TinyFlux Model Test")
    print("=" * 60)

    config = TinyFluxConfig()
    print(f"\nConfig:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_heads: {config.num_attention_heads}")
    print(f"  head_dim: {config.attention_head_dim}")
    print(f"  in_channels: {config.in_channels}")
    print(f"  double_layers: {config.num_double_layers}")
    print(f"  single_layers: {config.num_single_layers}")
    print(f"  joint_attention_dim: {config.joint_attention_dim}")
    print(f"  pooled_projection_dim: {config.pooled_projection_dim}")

    model = TinyFlux(config)

    # Count parameters
    counts = model.count_parameters()
    print(f"\nParameters:")
    for name, count in counts.items():
        print(f"  {name}: {count:,} ({count / 1e6:.2f}M)")

    # Test forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    batch_size = 16
    latent_h, latent_w = 64, 64  # 512x512 image / 8
    num_patches = latent_h * latent_w
    text_len = 77

    # Create dummy inputs
    hidden_states = torch.randn(batch_size, num_patches, config.in_channels, device=device)
    encoder_hidden_states = torch.randn(batch_size, text_len, config.joint_attention_dim, device=device)
    pooled_projections = torch.randn(batch_size, config.pooled_projection_dim, device=device)
    timestep = torch.rand(batch_size, device=device)
    img_ids = TinyFlux.create_img_ids(batch_size, latent_h, latent_w, device)
    guidance = torch.ones(batch_size, device=device) * 3.5

    print(f"\nInput shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
    print(f"  pooled_projections: {pooled_projections.shape}")
    print(f"  img_ids: {img_ids.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            guidance=guidance,
        )

    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("\n✓ Forward pass successful!")
    del model
    del output

# if __name__ == "__main__":
#    test_tiny_flux()