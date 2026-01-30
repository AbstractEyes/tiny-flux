"""
TinyFlux Model Zoo

Simple model manager for loading/unloading models.
No caching, no training logic - just model lifecycle management.

Models:
  - vae: FLUX.1 VAE (encode/decode images)
  - clip: CLIP-ViT-L-14 (text encoder + tokenizer)
  - t5: Flan-T5-Base (text encoder + tokenizer)
  - tinyflux: TinyFlux-Deep (main model)
  - lune: SD1.5 UNet with flow-matching weights (mid-block hook installed)
  - sol: SD1.5 UNet with v-pred weights (or geometric fallback)
"""

import gc
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path


class ModelZoo:
    """
    Manages model loading, compilation, and VRAM.

    Usage:
        zoo = ModelZoo(device="cuda")

        # Load models
        zoo.load_vae()
        zoo.load_clip()
        zoo.load_t5()
        zoo.load_lune()
        zoo.load_sol()

        # Load TinyFlux from various sources
        zoo.load_tinyflux("AbstractPhil/tinyflux-deep")  # HF repo
        zoo.load_tinyflux("/path/to/checkpoint/")        # Directory
        zoo.load_tinyflux("/path/to/model.safetensors")  # File
        zoo.load_tinyflux(None, train_mode=True)         # Random init for training
        zoo.load_tinyflux("repo/model", load_ema=True)   # Load EMA weights

        # Access models
        latent = zoo.vae.encode(image).latent_dist.sample()
        clip_out = zoo.clip(**clip_inputs)
        t5_out = zoo.t5(**t5_inputs)

        # Lune has mid-block hook - after forward, access features
        _ = zoo.lune(latents, timesteps, encoder_hidden_states=clip_hidden)
        mid_features = zoo.lune_mid_features  # [B, 1280]

        # Offload to CPU to free VRAM
        zoo.offload("lune")

        # Bring back to GPU
        zoo.onload("lune")

        # Fully unload
        zoo.unload("lune")
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype

        # Model storage
        self._models: Dict[str, nn.Module] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._offloaded: Dict[str, bool] = {}  # True if on CPU

        # Lune mid-block hook storage
        self._lune_mid_features: Optional[torch.Tensor] = None
        self._lune_hook_handle = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def vae(self) -> Optional[nn.Module]:
        return self._models.get("vae")

    @property
    def clip(self) -> Optional[nn.Module]:
        return self._models.get("clip")

    @property
    def clip_tokenizer(self):
        return self._tokenizers.get("clip")

    @property
    def t5(self) -> Optional[nn.Module]:
        return self._models.get("t5")

    @property
    def t5_tokenizer(self):
        return self._tokenizers.get("t5")

    @property
    def tinyflux(self) -> Optional[nn.Module]:
        return self._models.get("tinyflux")

    @property
    def lune(self) -> Optional[nn.Module]:
        return self._models.get("lune")

    @property
    def lune_mid_features(self) -> Optional[torch.Tensor]:
        """Mid-block features from last Lune forward pass. [B, 1280]"""
        return self._lune_mid_features

    @property
    def sol(self) -> Optional[nn.Module]:
        return self._models.get("sol")

    # =========================================================================
    # VAE
    # =========================================================================

    def load_vae(
        self,
        repo_id: str = "black-forest-labs/FLUX.1-schnell",
        subfolder: str = "vae",
        dtype: Optional[torch.dtype] = None,
        compile_model: bool = False,
    ) -> nn.Module:
        """Load FLUX VAE."""
        from diffusers import AutoencoderKL

        dtype = dtype or self.dtype
        vae = AutoencoderKL.from_pretrained(
            repo_id, subfolder=subfolder, torch_dtype=dtype
        ).to(self.device).eval()

        for p in vae.parameters():
            p.requires_grad = False

        if compile_model:
            vae = torch.compile(vae, mode="reduce-overhead")

        self._models["vae"] = vae
        self._offloaded["vae"] = False
        return vae

    # =========================================================================
    # CLIP
    # =========================================================================

    def load_clip(
        self,
        repo_id: str = "openai/clip-vit-large-patch14",
        dtype: Optional[torch.dtype] = None,
        compile_model: bool = False,
    ) -> nn.Module:
        """Load CLIP text encoder and tokenizer."""
        from transformers import CLIPTextModel, CLIPTokenizer

        dtype = dtype or torch.float16

        tokenizer = CLIPTokenizer.from_pretrained(repo_id)
        model = CLIPTextModel.from_pretrained(
            repo_id, torch_dtype=dtype
        ).to(self.device).eval()

        for p in model.parameters():
            p.requires_grad = False

        if compile_model:
            model = torch.compile(model, mode="reduce-overhead")

        self._models["clip"] = model
        self._tokenizers["clip"] = tokenizer
        self._offloaded["clip"] = False
        return model

    # =========================================================================
    # T5
    # =========================================================================

    def load_t5(
        self,
        repo_id: str = "google/flan-t5-base",
        dtype: Optional[torch.dtype] = None,
        compile_model: bool = False,
    ) -> nn.Module:
        """Load T5 encoder and tokenizer."""
        from transformers import T5EncoderModel, T5TokenizerFast

        dtype = dtype or torch.float16

        tokenizer = T5TokenizerFast.from_pretrained(repo_id)
        model = T5EncoderModel.from_pretrained(
            repo_id, torch_dtype=dtype
        ).to(self.device).eval()

        for p in model.parameters():
            p.requires_grad = False

        if compile_model:
            model = torch.compile(model, mode="reduce-overhead")

        self._models["t5"] = model
        self._tokenizers["t5"] = tokenizer
        self._offloaded["t5"] = False
        return model

    # =========================================================================
    # TinyFlux
    # =========================================================================

    def load_tinyflux(
        self,
        source: Optional[str] = None,
        config: Optional["TinyFluxConfig"] = None,
        dtype: Optional[torch.dtype] = None,
        compile_model: bool = False,
        train_mode: bool = False,
        load_ema: bool = False,
        ema_path: Optional[str] = None,
    ) -> nn.Module:
        """
        Load TinyFlux model.

        Args:
            source: One of:
                - HF repo: "AbstractPhil/tinyflux-deep"
                - Local directory: "/path/to/checkpoint/"
                - Local file: "/path/to/model.safetensors"
                - None for random initialization
            config: Model config (loaded from source or uses defaults if None)
            dtype: Model dtype
            compile_model: Whether to torch.compile
            train_mode: If True, keeps requires_grad=True
            load_ema: If True, load EMA weights instead of model weights
            ema_path: Explicit path to EMA weights
        """
        from .model import TinyFluxConfig, TinyFluxDeep
        from .loader import load_model

        dtype = dtype or self.dtype

        if source:
            # Use unified loader
            model = load_model(
                source,
                config=config,
                device=self.device,
                dtype=dtype,
                load_ema=load_ema,
                ema_path=ema_path,
                compile_model=False,  # Compile after if needed
                strict=False,
            )
        else:
            # Random init
            cfg = config or TinyFluxConfig()
            model = TinyFluxDeep(cfg).to(dtype).to(self.device)

        if not train_mode:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        if compile_model:
            model = torch.compile(model, mode="reduce-overhead")

        self._models["tinyflux"] = model
        self._offloaded["tinyflux"] = False
        return model

    # =========================================================================
    # Lune (SD1.5 UNet with flow-matching weights)
    # =========================================================================

    def load_lune(
        self,
        weights_path: Optional[str] = None,
        repo_id: str = "AbstractPhil/tinyflux-experts",
        filename: str = "sd15-flow-lune-unet.safetensors",
        dtype: Optional[torch.dtype] = None,
        compile_model: bool = False,
    ) -> nn.Module:
        """
        Load Lune (SD1.5 UNet with flow-matching weights).

        Installs a hook on mid_block that captures features after each forward.
        Access via zoo.lune_mid_features after calling lune().

        Args:
            weights_path: Local path to weights (if None, downloads from HF)
            repo_id: HuggingFace repo for weights
            filename: Weights filename in repo
            dtype: Model dtype (default float16 for SD1.5)
            compile_model: Whether to torch.compile (recommended)
        """
        from diffusers import UNet2DConditionModel
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        dtype = dtype or torch.float16

        # Load base SD1.5 UNet
        unet = UNet2DConditionModel.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            subfolder="unet",
            torch_dtype=dtype,
        ).to(self.device).eval()

        # Load flow-matching weights
        if weights_path is None:
            weights_path = hf_hub_download(repo_id=repo_id, filename=filename)

        state_dict = load_file(weights_path)
        unet.load_state_dict(state_dict, strict=False)

        for p in unet.parameters():
            p.requires_grad = False

        # Install mid-block hook
        def hook_fn(module, inp, out):
            # Global average pool: [B, 1280, H, W] -> [B, 1280]
            self._lune_mid_features = out.mean(dim=[2, 3])

        self._lune_hook_handle = unet.mid_block.register_forward_hook(hook_fn)

        if compile_model:
            unet = torch.compile(unet, mode="reduce-overhead")

        self._models["lune"] = unet
        self._offloaded["lune"] = False
        return unet

    # =========================================================================
    # Sol (SD1.5 UNet - extracts attention statistics)
    # =========================================================================

    def load_sol(
        self,
        weights_path: Optional[str] = None,
        repo_id: str = "AbstractPhil/tinyflux-experts",
        filename: str = "sd15-flow-sol-unet.safetensors",
        dtype: Optional[torch.dtype] = None,
        compile_model: bool = False,
    ) -> nn.Module:
        """
        Load Sol (SD1.5 UNet) for attention statistics extraction.

        Installs custom attention processor to capture attention weights.
        Use sol_forward() to run inference and get stats.
        """
        from diffusers import UNet2DConditionModel
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        dtype = dtype or torch.float16

        unet = UNet2DConditionModel.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            subfolder="unet",
            torch_dtype=dtype,
        ).to(self.device).eval()

        if weights_path is None:
            weights_path = hf_hub_download(repo_id=repo_id, filename=filename)

        state_dict = load_file(weights_path)
        unet.load_state_dict(state_dict, strict=False)

        for p in unet.parameters():
            p.requires_grad = False

        # Storage for captured attention weights
        self._sol_attn_weights = []

        # Install custom attention processor that captures weights
        processor = SolAttnProcessor(self._sol_attn_weights)
        unet.set_attn_processor(processor)

        if compile_model:
            unet = torch.compile(unet, mode="reduce-overhead")

        self._models["sol"] = unet
        self._offloaded["sol"] = False
        return unet

    @torch.no_grad()
    def sol_forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        clip_hidden: torch.Tensor,
        spatial_size: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run Sol and extract attention statistics.

        Args:
            latents: [B, 4, H, W] noisy latents
            timesteps: [B] in [0, 1]
            clip_hidden: [B, 77, 768] CLIP hidden states
            spatial_size: output spatial resolution

        Returns:
            stats: [B, 3] - locality, entropy, clustering
            spatial: [B, spatial_size, spatial_size] - importance map
        """
        if self.sol is None:
            raise RuntimeError("Sol not loaded. Call zoo.load_sol() first.")

        B = latents.shape[0]
        self._sol_attn_weights.clear()

        # Forward pass - processor captures attention weights
        t_scaled = timesteps * 1000
        _ = self.sol(latents, t_scaled, encoder_hidden_states=clip_hidden)

        # Compute statistics from captured weights
        stats, spatial = self._compute_attention_statistics(B, spatial_size)

        # Clear storage to free memory
        self._sol_attn_weights.clear()
        torch.cuda.empty_cache()

        return stats, spatial

    def _compute_attention_statistics(
        self,
        batch_size: int,
        spatial_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute stats from captured self-attention weights."""
        device = self.device
        B = batch_size

        all_locality = []
        all_entropy = []
        all_spatial = []

        captured_count = len(self._sol_attn_weights)
        valid_count = 0

        for attn_weights, H, W in self._sol_attn_weights:
            # attn_weights: [B, heads, N, N] where N = H * W for self-attention
            if attn_weights.shape[0] != B:
                continue

            # Average over heads: [B, N, N]
            attn = attn_weights.mean(dim=1)
            N = attn.shape[-1]

            # Free the original attention weights immediately
            del attn_weights

            # Skip cross-attention (non-square or wrong size)
            if H * W != N:
                del attn
                continue

            # === Entropy: how diffuse is each query's attention? ===
            # Higher entropy = more uniform attention
            # Lower entropy = focused on few keys
            entropy_per_query = -(attn * (attn + 1e-8).log()).sum(dim=-1)  # [B, N]
            max_entropy = torch.log(torch.tensor(N, dtype=torch.float32, device=device))
            entropy = entropy_per_query.mean(dim=-1) / max_entropy  # [B], normalized to [0,1]
            all_entropy.append(entropy)

            # === Locality: do queries attend nearby or far? ===
            # Build distance matrix
            y_coords = torch.arange(H, device=device, dtype=torch.float32)
            x_coords = torch.arange(W, device=device, dtype=torch.float32)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            positions = torch.stack([yy.flatten(), xx.flatten()], dim=-1)  # [N, 2]

            # Distance between all pairs: [N, N]
            dist = torch.cdist(positions, positions)  # [N, N]
            max_dist = dist.max()

            # Expected distance per query weighted by attention
            expected_dist = (attn * dist.unsqueeze(0)).sum(dim=-1)  # [B, N]
            locality = 1 - (expected_dist.mean(dim=-1) / max_dist)  # [B], higher = more local
            all_locality.append(locality)

            # === Spatial importance: where does attention aggregate? ===
            # Sum attention received by each key position
            importance = attn.sum(dim=1)  # [B, N] - total attention each position receives
            spatial_map = importance.view(B, H, W)

            # Resize to target
            spatial_resized = torch.nn.functional.interpolate(
                spatial_map.unsqueeze(1),
                size=(spatial_size, spatial_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)  # [B, spatial_size, spatial_size]
            all_spatial.append(spatial_resized)
            valid_count += 1

        # Aggregate across layers
        if all_locality:
            locality = torch.stack(all_locality).mean(dim=0)
            entropy = torch.stack(all_entropy).mean(dim=0)
            spatial = torch.stack(all_spatial).mean(dim=0)
            spatial = spatial / spatial.sum(dim=[-2, -1], keepdim=True)  # normalize
        else:
            # Fallback - this means attention capture failed
            print(f"  [WARNING] Sol attention capture failed: {captured_count} captured, {valid_count} valid")
            locality = torch.full((B,), 0.5, device=device)
            entropy = torch.full((B,), 0.5, device=device)
            spatial = torch.ones(B, spatial_size, spatial_size, device=device) / (spatial_size ** 2)

        # Clustering: inverse entropy (focused = clustered)
        clustering = 1 - entropy

        stats = torch.stack([locality, entropy, clustering], dim=-1)  # [B, 3]
        return stats, spatial

    # =========================================================================
    # Memory Management
    # =========================================================================

    def offload(self, name: str):
        """Move model to CPU to free VRAM."""
        if name not in self._models:
            return
        if self._offloaded.get(name, False):
            return

        model = self._models[name]
        if hasattr(model, 'to'):
            model.to("cpu")
        self._offloaded[name] = True
        torch.cuda.empty_cache()

    def onload(self, name: str):
        """Move model back to GPU."""
        if name not in self._models:
            return
        if not self._offloaded.get(name, False):
            return

        model = self._models[name]
        if hasattr(model, 'to'):
            model.to(self.device)
        self._offloaded[name] = False

    def unload(self, name: str):
        """Completely remove model from memory."""
        if name not in self._models:
            return

        # Cleanup Lune hook
        if name == "lune" and self._lune_hook_handle is not None:
            self._lune_hook_handle.remove()
            self._lune_hook_handle = None
            self._lune_mid_features = None

        # Cleanup Sol state
        if name == "sol" and hasattr(self, '_sol_attn_weights'):
            self._sol_attn_weights = None

        del self._models[name]
        if name in self._tokenizers:
            del self._tokenizers[name]
        if name in self._offloaded:
            del self._offloaded[name]

        gc.collect()
        torch.cuda.empty_cache()

    def unload_all(self):
        """Unload all models."""
        for name in list(self._models.keys()):
            self.unload(name)

    # =========================================================================
    # Status
    # =========================================================================

    def loaded(self) -> List[str]:
        """List of loaded model names."""
        return list(self._models.keys())

    def is_loaded(self, name: str) -> bool:
        return name in self._models

    def is_offloaded(self, name: str) -> bool:
        return self._offloaded.get(name, False)

    def __repr__(self) -> str:
        loaded = self.loaded()
        offloaded = [n for n in loaded if self.is_offloaded(n)]
        on_gpu = [n for n in loaded if not self.is_offloaded(n)]
        return f"ModelZoo(gpu={on_gpu}, cpu={offloaded})"


# =============================================================================
# Sol Attention Processor (captures attention weights)
# =============================================================================

class SolAttnProcessor:
    """
    Custom attention processor that captures attention weights for Sol extraction.

    Stores (attn_weights, H, W) tuples for self-attention layers.
    """

    def __init__(self, storage: List):
        self.storage = storage

    def __call__(
        self,
        attn,  # Attention module
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Process attention and capture weights for self-attention."""
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        channel = None  # Will be set for 4D input
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, sequence_length, _ = hidden_states.shape
            # Infer spatial dims from sequence length (assume square)
            hw = int(sequence_length ** 0.5)
            if hw * hw == sequence_length:
                height = width = hw
            else:
                height = width = None

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Compute attention weights explicitly
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Store self-attention weights (not cross-attention)
        if not is_cross and height is not None:
            # Reshape to [B, heads, N, N]
            B = batch_size
            heads = attn.heads
            N = sequence_length
            attn_weights = attention_probs.view(B, heads, N, N)
            self.storage.append((attn_weights.detach().clone(), height, width))

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4 and channel is not None:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# =============================================================================
# Sol Geometric Fallback (no model needed)
# =============================================================================

class SolGeometric:
    """
    Geometric heuristics for Sol attention statistics.
    No model, no VRAM - just timestep-based priors.
    """

    def __init__(self, spatial_size: int = 8):
        self.spatial_size = spatial_size

    def __call__(
        self,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention statistics from timestep.

        Args:
            timesteps: [B] in [0, 1]

        Returns:
            stats: [B, 4] - locality, entropy, clustering, sparsity
            spatial: [B, H, W] - spatial importance map
        """
        return self.forward(timesteps)

    def forward(
        self,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = timesteps.device
        t = timesteps.float()
        B = t.shape[0]
        H = W = self.spatial_size

        # Stats: [B, 3] - locality, entropy, clustering
        locality = 1 - t
        entropy = t
        clustering = 0.5 - 0.3 * (t - 0.5).abs()
        stats = torch.stack([locality, entropy, clustering], dim=-1)

        # Spatial: [B, H, W]
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        center_dist = torch.sqrt(xx ** 2 + yy ** 2)

        t_weight = (1 - t).view(B, 1, 1)
        center_bias = 1 - center_dist.unsqueeze(0) * t_weight
        spatial = center_bias / center_bias.sum(dim=[-2, -1], keepdim=True)

        return stats, spatial

    def to(self, device):
        """No-op for compatibility."""
        return self


# =============================================================================
# Smoke Test
# =============================================================================

def _smoke_test():
    print("ModelZoo Smoke Test")
    print("=" * 50)

    zoo = ModelZoo(device="cpu", dtype=torch.float32)
    print(f"\n[1] Init: {zoo}")

    # Test SolGeometric fallback (no model needed)
    print("\n[2] SolGeometric (fallback)...")
    sol_geo = SolGeometric()
    t = torch.tensor([0.1, 0.5, 0.9])
    stats, spatial = sol_geo(t)
    assert stats.shape == (3, 3), f"Bad stats: {stats.shape}"  # [B, 3]
    assert spatial.shape == (3, 8, 8), f"Bad spatial: {spatial.shape}"
    print(f"    stats: {stats.shape}, spatial: {spatial.shape} ✓")
    print(f"    t=0.1: locality={stats[0,0]:.2f}, entropy={stats[0,1]:.2f}, clustering={stats[0,2]:.2f}")
    print(f"    t=0.9: locality={stats[2,0]:.2f}, entropy={stats[2,1]:.2f}, clustering={stats[2,2]:.2f}")

    print(f"\n[3] Status: {zoo}")
    print(f"    loaded: {zoo.loaded()}")

    print("\n" + "=" * 50)
    print("✓ Smoke tests passed")
    print("\nFull Sol extraction requires:")
    print("  zoo.load_sol()  # Loads SD1.5 UNet")
    print("  stats, spatial = zoo.sol_forward(latents, t, clip_hidden)")


if __name__ == "__main__":
    _smoke_test()