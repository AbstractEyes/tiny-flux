"""
TinyFlux Cache System

All data is precached before training:
- Latents (VAE-encoded images)
- T5 embeddings
- CLIP pooled features
- Lune mid-block features (with timestep buckets)
- Sol attention statistics (with timestep buckets)

Training loop only moves tensors - no model inference overhead.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Tuple, Any, Union
from tqdm import tqdm
from PIL import Image
import numpy as np


# =============================================================================
# Constants
# =============================================================================

DEFAULT_T_BUCKETS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
EXTRACTION_BASE_SEED = 42

LUNE_DIM = 1280
SOL_STATS_DIM = 3
SOL_SPATIAL_SIZE = 8


def get_t_buckets(device: str = "cpu") -> torch.Tensor:
    return torch.tensor(DEFAULT_T_BUCKETS, device=device)


# =============================================================================
# Encoding Cache (T5 + CLIP + Latents)
# =============================================================================

class EncodingCache:
    """
    Precached encodings for a dataset.

    Stores:
        - latents: [N, C, H, W] VAE-encoded images
        - t5_embeds: [N, L, 768] T5 hidden states
        - clip_pooled: [N, 768] CLIP pooled features
        - prompts: List[str] original prompts (optional, for reference)

    Usage:
        # Build from zoo
        cache = EncodingCache.build(zoo, images, prompts)
        cache.save("encodings.pt")

        # Load and use
        cache = EncodingCache.load("encodings.pt")
        latent, t5, clip = cache[idx]
    """

    def __init__(
        self,
        latents: torch.Tensor,
        t5_embeds: torch.Tensor,
        clip_pooled: torch.Tensor,
        prompts: Optional[List[str]] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.latents = latents.to(dtype)        # [N, C, H, W]
        self.t5_embeds = t5_embeds.to(dtype)    # [N, L, 768]
        self.clip_pooled = clip_pooled.to(dtype) # [N, 768]
        self.prompts = prompts
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.latents[idx], self.t5_embeds[idx], self.clip_pooled[idx]

    def get_batch(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get batch by indices."""
        idx = indices.cpu()
        return self.latents[idx], self.t5_embeds[idx], self.clip_pooled[idx]

    def save(self, path: str):
        torch.save({
            'latents': self.latents,
            't5_embeds': self.t5_embeds,
            'clip_pooled': self.clip_pooled,
            'prompts': self.prompts,
            'dtype': self.dtype,
        }, path)

    @classmethod
    def load(cls, path: str, dtype: Optional[torch.dtype] = None) -> "EncodingCache":
        data = torch.load(path, map_location="cpu")
        dtype = dtype or data.get('dtype', torch.float16)
        return cls(
            latents=data['latents'],
            t5_embeds=data['t5_embeds'],
            clip_pooled=data['clip_pooled'],
            prompts=data.get('prompts'),
            dtype=dtype,
        )

    @classmethod
    def build(
        cls,
        zoo,  # ModelZoo
        images: List[Image.Image],
        prompts: List[str],
        batch_size: int = 32,
        max_t5_length: int = 128,
        dtype: torch.dtype = torch.float16,
        vae_scale: Optional[float] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 500,
    ) -> "EncodingCache":
        """
        Build encoding cache from images and prompts.

        Requires zoo with vae, clip, t5 loaded.

        Args:
            checkpoint_dir: If set, save progress every checkpoint_every samples
            checkpoint_every: Save interval (samples)
        """
        device = zoo.device
        n = len(images)
        assert len(prompts) == n, "images and prompts must have same length"

        if vae_scale is None:
            vae_scale = zoo.vae.config.scaling_factor

        # Check for existing progress
        start_idx = 0
        all_latents = []
        all_t5 = []
        all_clip = []

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            progress_file = os.path.join(checkpoint_dir, "encodings_progress.pt")
            if os.path.exists(progress_file):
                print(f"    Resuming encodings from checkpoint...")
                progress = torch.load(progress_file, weights_only=False)
                start_idx = progress['next_idx']
                all_latents = progress['latents']
                all_t5 = progress['t5']
                all_clip = progress['clip']
                print(f"    Loaded {start_idx}/{n} samples")

        with torch.no_grad():
            for i in tqdm(range(start_idx, n, batch_size), desc="Encoding", initial=start_idx//batch_size, total=(n+batch_size-1)//batch_size):
                batch_imgs = images[i:i+batch_size]
                batch_prompts = prompts[i:i+batch_size]

                # === VAE ===
                img_tensors = []
                for img in batch_imgs:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    if img.size != (512, 512):
                        img = img.resize((512, 512), Image.Resampling.LANCZOS)
                    t = torch.from_numpy(np.array(img)).float() / 255.0
                    t = t.permute(2, 0, 1)  # [3, H, W]
                    t = t * 2.0 - 1.0
                    img_tensors.append(t)

                # Use VAE's dtype for encoding, then cast to storage dtype
                vae_dtype = next(zoo.vae.parameters()).dtype
                img_batch = torch.stack(img_tensors).to(device, dtype=vae_dtype)
                latent = zoo.vae.encode(img_batch).latent_dist.sample()
                latent = latent * vae_scale
                all_latents.append(latent.cpu().to(dtype))

                # === T5 ===
                t5_inputs = zoo.t5_tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_t5_length,
                    truncation=True,
                ).to(device)
                t5_out = zoo.t5(**t5_inputs).last_hidden_state
                all_t5.append(t5_out.cpu().to(dtype))

                # === CLIP ===
                clip_inputs = zoo.clip_tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                ).to(device)
                clip_out = zoo.clip(**clip_inputs).pooler_output
                all_clip.append(clip_out.cpu().to(dtype))

                # Checkpoint periodically
                next_idx = min(i + batch_size, n)
                if checkpoint_dir and (next_idx % checkpoint_every < batch_size or next_idx == n):
                    torch.save({
                        'next_idx': next_idx,
                        'latents': all_latents,
                        't5': all_t5,
                        'clip': all_clip,
                    }, progress_file)

        # Clean up checkpoint after completion
        if checkpoint_dir and os.path.exists(progress_file):
            os.remove(progress_file)

        return cls(
            latents=torch.cat(all_latents, dim=0),
            t5_embeds=torch.cat(all_t5, dim=0),
            clip_pooled=torch.cat(all_clip, dim=0),
            prompts=prompts,
            dtype=dtype,
        )

    def __repr__(self) -> str:
        return f"EncodingCache(n={len(self)}, latents={list(self.latents.shape)}, t5={list(self.t5_embeds.shape)})"


# =============================================================================
# Lune Feature Cache
# =============================================================================

def seed_for_extraction(sample_idx: int, bucket_idx: int, base_seed: int = EXTRACTION_BASE_SEED) -> int:
    """Deterministic seed for reproducible extraction."""
    return base_seed + sample_idx * 1000 + bucket_idx


def seed_batch_extraction(
    start_idx: int,
    batch_size: int,
    bucket_idx: int,
    device: str,
    base_seed: int = EXTRACTION_BASE_SEED,
) -> torch.Generator:
    """Create seeded generator for batch extraction."""
    seed = base_seed + start_idx * 1000 + bucket_idx
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


class LuneFeatureCache:
    """
    Precached Lune mid-block features with timestep interpolation.

    Shape: [N, n_buckets, 1280]

    Features extracted at 10 timestep buckets, interpolated at runtime.
    """

    def __init__(
        self,
        features: torch.Tensor,
        t_buckets: torch.Tensor,
        dtype: torch.dtype = torch.float16,
    ):
        self.features = features.to(dtype)  # [N, n_buckets, 1280]
        self.t_buckets = t_buckets
        self.n_buckets = len(t_buckets)
        self.t_min = t_buckets[0].item()
        self.t_max = t_buckets[-1].item()
        self.t_step = (t_buckets[1] - t_buckets[0]).item()
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.features)

    def get(self, indices: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Get interpolated features for batch.

        Args:
            indices: [B] sample indices
            timesteps: [B] timesteps in [0, 1]

        Returns:
            features: [B, 1280]
        """
        device = timesteps.device

        # Compute interpolation weights
        t_clamped = timesteps.float().clamp(self.t_min, self.t_max)
        t_idx_float = (t_clamped - self.t_min) / self.t_step
        t_idx_low = t_idx_float.long().clamp(0, self.n_buckets - 2)
        t_idx_high = (t_idx_low + 1).clamp(0, self.n_buckets - 1)
        alpha = (t_idx_float - t_idx_low.float()).unsqueeze(-1)  # [B, 1]

        # Gather on CPU
        idx_cpu = indices.cpu()
        t_low_cpu = t_idx_low.cpu()
        t_high_cpu = t_idx_high.cpu()

        f_low = self.features[idx_cpu, t_low_cpu]   # [B, 1280]
        f_high = self.features[idx_cpu, t_high_cpu]

        # Interpolate
        result = (1 - alpha.cpu()) * f_low + alpha.cpu() * f_high

        return result.to(device=device, dtype=self.dtype)

    def save(self, path: str):
        torch.save({
            'features': self.features,
            't_buckets': self.t_buckets,
            'dtype': self.dtype,
        }, path)

    @classmethod
    def load(cls, path: str) -> "LuneFeatureCache":
        data = torch.load(path, map_location="cpu")
        return cls(
            features=data['features'],
            t_buckets=data['t_buckets'],
            dtype=data.get('dtype', torch.float16),
        )

    @classmethod
    def build(
        cls,
        zoo,  # ModelZoo with clip and lune loaded
        prompts: List[str],
        t_buckets: Optional[torch.Tensor] = None,
        batch_size: int = 64,
        dtype: torch.dtype = torch.float16,
        base_seed: int = EXTRACTION_BASE_SEED,
        batch_timesteps: bool = True,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 500,
    ) -> "LuneFeatureCache":
        """
        Extract Lune mid-block features for all prompts at all timestep buckets.

        Args:
            zoo: ModelZoo with clip and lune loaded
            prompts: List of prompts
            t_buckets: Timestep buckets (default: 10 from 0.05 to 0.95)
            batch_size: Prompts per batch
            dtype: Storage dtype
            base_seed: Seed for reproducibility
            batch_timesteps: If True, process all timesteps in one forward (faster, more VRAM)
            checkpoint_dir: If set, save progress every checkpoint_every samples
            checkpoint_every: Save interval (samples)
        """
        device = zoo.device

        if t_buckets is None:
            t_buckets = get_t_buckets(device)

        n_prompts = len(prompts)
        n_buckets = len(t_buckets)

        # Check for existing progress
        start_idx = 0
        all_features = torch.zeros(n_prompts, n_buckets, LUNE_DIM, dtype=dtype)

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            progress_file = os.path.join(checkpoint_dir, "lune_progress.pt")
            if os.path.exists(progress_file):
                print(f"    Resuming Lune from checkpoint...")
                progress = torch.load(progress_file, weights_only=False)
                start_idx = progress['next_idx']
                all_features[:start_idx] = progress['features'][:start_idx]
                print(f"    Loaded {start_idx}/{n_prompts} samples")

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            for start in tqdm(range(start_idx, n_prompts, batch_size), desc="Extracting Lune",
                             initial=start_idx//batch_size, total=(n_prompts+batch_size-1)//batch_size):
                end_idx = min(start + batch_size, n_prompts)
                batch_prompts = prompts[start:end_idx]
                B = len(batch_prompts)

                # Encode CLIP
                clip_inputs = zoo.clip_tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                ).to(device)
                clip_out = zoo.clip(**clip_inputs)
                clip_hidden = clip_out.last_hidden_state  # [B, 77, 768]

                if batch_timesteps:
                    # === Batched: All timesteps in one forward ===
                    # Expand clip_hidden: [B, 77, 768] -> [B*n_buckets, 77, 768]
                    clip_expanded = clip_hidden.unsqueeze(1).expand(-1, n_buckets, -1, -1)
                    clip_expanded = clip_expanded.reshape(B * n_buckets, 77, -1)

                    # Create timesteps: [B*n_buckets]
                    t_all = t_buckets.unsqueeze(0).expand(B, -1).reshape(-1)

                    # Seeded latents for all timesteps
                    gen = seed_batch_extraction(start, B * n_buckets, 0, device, base_seed)
                    latents = torch.randn(
                        B * n_buckets, 4, 64, 64,
                        generator=gen,
                        device=device,
                        dtype=torch.float16,
                    )

                    # Single forward
                    t_scaled = t_all * 1000
                    _ = zoo.lune(latents, t_scaled, encoder_hidden_states=clip_expanded)

                    mid_features = zoo.lune_mid_features
                    if mid_features is not None:
                        if mid_features.dim() == 4:
                            mid_features = mid_features.mean(dim=[2, 3])  # [B*n_buckets, 1280]
                        # Reshape back: [B*n_buckets, 1280] -> [B, n_buckets, 1280]
                        mid_features = mid_features.view(B, n_buckets, -1)
                        all_features[start:end_idx] = mid_features.cpu().to(dtype)
                else:
                    # === Sequential: One timestep at a time ===
                    for t_idx, t_val in enumerate(t_buckets):
                        t = torch.full((B,), t_val.item(), device=device)

                        gen = seed_batch_extraction(start, B, t_idx, device, base_seed)
                        latents = torch.randn(
                            B, 4, 64, 64,
                            generator=gen,
                            device=device,
                            dtype=torch.float16,
                        )

                        t_scaled = t * 1000
                        _ = zoo.lune(latents, t_scaled, encoder_hidden_states=clip_hidden)

                        mid_features = zoo.lune_mid_features
                        if mid_features is not None:
                            if mid_features.dim() == 4:
                                mid_features = mid_features.mean(dim=[2, 3])
                            all_features[start:end_idx, t_idx] = mid_features.cpu().to(dtype)

                # Checkpoint periodically
                next_idx = end_idx
                if checkpoint_dir and (next_idx % checkpoint_every < batch_size or next_idx == n_prompts):
                    torch.save({
                        'next_idx': next_idx,
                        'features': all_features,
                    }, progress_file)

        # Clean up checkpoint after completion
        if checkpoint_dir and os.path.exists(progress_file):
            os.remove(progress_file)

        return cls(all_features, t_buckets.cpu(), dtype)

    def __repr__(self) -> str:
        return f"LuneFeatureCache(n={len(self)}, buckets={self.n_buckets}, dim={LUNE_DIM})"


# =============================================================================
# Sol Feature Cache
# =============================================================================

class SolFeatureCache:
    """
    Precached Sol attention statistics with timestep interpolation.

    Shapes:
        - stats: [N, n_buckets, 3] (locality, entropy, clustering)
        - spatial: [N, n_buckets, 8, 8] (attention importance map)
    """

    def __init__(
        self,
        stats: torch.Tensor,
        spatial: torch.Tensor,
        t_buckets: torch.Tensor,
        dtype: torch.dtype = torch.float16,
    ):
        self.stats = stats.to(dtype)      # [N, n_buckets, 3]
        self.spatial = spatial.to(dtype)  # [N, n_buckets, 8, 8]
        self.t_buckets = t_buckets
        self.n_buckets = len(t_buckets)
        self.t_min = t_buckets[0].item()
        self.t_max = t_buckets[-1].item()
        self.t_step = (t_buckets[1] - t_buckets[0]).item()
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.stats)

    def get(self, indices: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get interpolated stats and spatial for batch.

        Args:
            indices: [B] sample indices
            timesteps: [B] timesteps in [0, 1]

        Returns:
            stats: [B, 3]
            spatial: [B, 8, 8]
        """
        device = timesteps.device

        # Interpolation weights
        t_clamped = timesteps.float().clamp(self.t_min, self.t_max)
        t_idx_float = (t_clamped - self.t_min) / self.t_step
        t_idx_low = t_idx_float.long().clamp(0, self.n_buckets - 2)
        t_idx_high = (t_idx_low + 1).clamp(0, self.n_buckets - 1)

        alpha_stats = (t_idx_float - t_idx_low.float()).unsqueeze(-1)
        alpha_spatial = alpha_stats.unsqueeze(-1)

        idx_cpu = indices.cpu()
        t_low_cpu = t_idx_low.cpu()
        t_high_cpu = t_idx_high.cpu()

        # Stats interpolation
        s_low = self.stats[idx_cpu, t_low_cpu]
        s_high = self.stats[idx_cpu, t_high_cpu]
        stats_result = (1 - alpha_stats.cpu()) * s_low + alpha_stats.cpu() * s_high

        # Spatial interpolation
        sp_low = self.spatial[idx_cpu, t_low_cpu]
        sp_high = self.spatial[idx_cpu, t_high_cpu]
        spatial_result = (1 - alpha_spatial.cpu()) * sp_low + alpha_spatial.cpu() * sp_high

        return (
            stats_result.to(device=device, dtype=self.dtype),
            spatial_result.to(device=device, dtype=self.dtype),
        )

    def save(self, path: str):
        torch.save({
            'stats': self.stats,
            'spatial': self.spatial,
            't_buckets': self.t_buckets,
            'dtype': self.dtype,
        }, path)

    @classmethod
    def load(cls, path: str) -> "SolFeatureCache":
        data = torch.load(path, map_location="cpu")
        return cls(
            stats=data['stats'],
            spatial=data['spatial'],
            t_buckets=data['t_buckets'],
            dtype=data.get('dtype', torch.float16),
        )

    @classmethod
    def build(
        cls,
        zoo,  # ModelZoo with clip and sol loaded
        prompts: List[str],
        t_buckets: Optional[torch.Tensor] = None,
        batch_size: int = 64,
        spatial_size: int = SOL_SPATIAL_SIZE,
        dtype: torch.dtype = torch.float16,
        base_seed: int = EXTRACTION_BASE_SEED,
        batch_timesteps: bool = False,  # Default False - Sol attention is VRAM heavy
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 500,
    ) -> "SolFeatureCache":
        """
        Extract Sol attention statistics for all prompts at all timestep buckets.

        Args:
            zoo: ModelZoo with clip and sol loaded
            prompts: List of prompts
            t_buckets: Timestep buckets
            batch_size: Prompts per batch
            spatial_size: Spatial output size
            dtype: Storage dtype
            base_seed: Seed for reproducibility
            batch_timesteps: If True, process all timesteps in one forward (faster but 10x VRAM)
            checkpoint_dir: If set, save progress every checkpoint_every samples
            checkpoint_every: Save interval (samples)
        """
        device = zoo.device

        if t_buckets is None:
            t_buckets = get_t_buckets(device)

        n_prompts = len(prompts)
        n_buckets = len(t_buckets)

        # Check for existing progress
        start_idx = 0
        all_stats = torch.zeros(n_prompts, n_buckets, SOL_STATS_DIM, dtype=dtype)
        all_spatial = torch.zeros(n_prompts, n_buckets, spatial_size, spatial_size, dtype=dtype)

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            progress_file = os.path.join(checkpoint_dir, "sol_progress.pt")
            if os.path.exists(progress_file):
                print(f"    Resuming Sol from checkpoint...")
                progress = torch.load(progress_file, weights_only=False)
                start_idx = progress['next_idx']
                all_stats[:start_idx] = progress['stats'][:start_idx]
                all_spatial[:start_idx] = progress['spatial'][:start_idx]
                print(f"    Loaded {start_idx}/{n_prompts} samples")

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            for start in tqdm(range(start_idx, n_prompts, batch_size), desc="Extracting Sol",
                             initial=start_idx//batch_size, total=(n_prompts+batch_size-1)//batch_size):
                end_idx = min(start + batch_size, n_prompts)
                batch_prompts = prompts[start:end_idx]
                B = len(batch_prompts)

                # Encode CLIP
                clip_inputs = zoo.clip_tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                ).to(device)
                clip_hidden = zoo.clip(**clip_inputs).last_hidden_state  # [B, 77, 768]

                if batch_timesteps:
                    # === Batched: All timesteps in one forward ===
                    # Expand clip_hidden: [B, 77, 768] -> [B*n_buckets, 77, 768]
                    clip_expanded = clip_hidden.unsqueeze(1).expand(-1, n_buckets, -1, -1)
                    clip_expanded = clip_expanded.reshape(B * n_buckets, 77, -1).half()

                    # Create timesteps: [B*n_buckets]
                    t_all = t_buckets.unsqueeze(0).expand(B, -1).reshape(-1)

                    # Seeded latents
                    gen = seed_batch_extraction(start, B * n_buckets, 0, device, base_seed)
                    latents = torch.randn(
                        B * n_buckets, 4, 64, 64,
                        generator=gen,
                        device=device,
                        dtype=torch.float16,
                    )

                    # Single forward
                    stats, spatial = zoo.sol_forward(latents, t_all, clip_expanded, spatial_size)

                    # Reshape: [B*n_buckets, ...] -> [B, n_buckets, ...]
                    stats = stats.view(B, n_buckets, -1)
                    spatial = spatial.view(B, n_buckets, spatial_size, spatial_size)

                    all_stats[start:end_idx] = stats.cpu().to(dtype)
                    all_spatial[start:end_idx] = spatial.cpu().to(dtype)
                else:
                    # === Sequential: One timestep at a time ===
                    for t_idx, t_val in enumerate(t_buckets):
                        t = torch.full((B,), t_val.item(), device=device)

                        gen = seed_batch_extraction(start, B, t_idx, device, base_seed)
                        latents = torch.randn(
                            B, 4, 64, 64,
                            generator=gen,
                            device=device,
                            dtype=torch.float16,
                        )

                        stats, spatial = zoo.sol_forward(latents, t, clip_hidden.half(), spatial_size)

                        all_stats[start:end_idx, t_idx] = stats.cpu().to(dtype)
                        all_spatial[start:end_idx, t_idx] = spatial.cpu().to(dtype)

                # Checkpoint periodically
                next_idx = end_idx
                if checkpoint_dir and (next_idx % checkpoint_every < batch_size or next_idx == n_prompts):
                    torch.save({
                        'next_idx': next_idx,
                        'stats': all_stats,
                        'spatial': all_spatial,
                    }, progress_file)

        # Clean up checkpoint after completion
        if checkpoint_dir and os.path.exists(progress_file):
            os.remove(progress_file)

        return cls(all_stats, all_spatial, t_buckets.cpu(), dtype)

    def __repr__(self) -> str:
        return f"SolFeatureCache(n={len(self)}, buckets={self.n_buckets}, stats={SOL_STATS_DIM}, spatial={SOL_SPATIAL_SIZE}x{SOL_SPATIAL_SIZE})"


# =============================================================================
# Combined Dataset Cache
# =============================================================================

class DatasetCache:
    """
    Complete precached dataset combining encodings + expert features.

    Everything needed for training, no model overhead.

    Usage:
        # Build all caches
        cache = DatasetCache.build(zoo, images, prompts)
        cache.save("dataset_cache.pt")

        # Training
        cache = DatasetCache.load("dataset_cache.pt")
        for idx in indices:
            latent, t5, clip = cache.get_encodings(idx)
            lune = cache.get_lune(idx, t)
            sol_stats, sol_spatial = cache.get_sol(idx, t)
    """

    def __init__(
        self,
        encodings: EncodingCache,
        lune: Optional[LuneFeatureCache] = None,
        sol: Optional[SolFeatureCache] = None,
        name: str = "dataset",
    ):
        self.encodings = encodings
        self.lune = lune
        self.sol = sol
        self.name = name

    def __len__(self) -> int:
        return len(self.encodings)

    def get_encodings(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get latent, t5, clip for sample."""
        return self.encodings[idx]

    def get_encodings_batch(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get batch of encodings."""
        return self.encodings.get_batch(indices)

    def get_lune(self, indices: torch.Tensor, timesteps: torch.Tensor) -> Optional[torch.Tensor]:
        """Get Lune features for batch."""
        if self.lune is None:
            return None
        return self.lune.get(indices, timesteps)

    def get_sol(self, indices: torch.Tensor, timesteps: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get Sol features for batch."""
        if self.sol is None:
            return None, None
        return self.sol.get(indices, timesteps)

    def save(self, directory: str):
        """Save all caches to directory."""
        os.makedirs(directory, exist_ok=True)
        self.encodings.save(os.path.join(directory, "encodings.pt"))
        if self.lune is not None:
            self.lune.save(os.path.join(directory, "lune.pt"))
        if self.sol is not None:
            self.sol.save(os.path.join(directory, "sol.pt"))
        # Metadata
        torch.save({'name': self.name}, os.path.join(directory, "meta.pt"))

    @classmethod
    def load(cls, directory: str) -> "DatasetCache":
        """Load all caches from directory."""
        meta = torch.load(os.path.join(directory, "meta.pt"))
        encodings = EncodingCache.load(os.path.join(directory, "encodings.pt"))

        lune = None
        lune_path = os.path.join(directory, "lune.pt")
        if os.path.exists(lune_path):
            lune = LuneFeatureCache.load(lune_path)

        sol = None
        sol_path = os.path.join(directory, "sol.pt")
        if os.path.exists(sol_path):
            sol = SolFeatureCache.load(sol_path)

        return cls(encodings, lune, sol, name=meta.get('name', 'dataset'))

    @classmethod
    def build(
        cls,
        zoo,  # ModelZoo
        images: List[Image.Image],
        prompts: List[str],
        name: str = "dataset",
        build_lune: bool = True,
        build_sol: bool = True,
        batch_size: int = 32,
        sol_batch_size: Optional[int] = None,  # None = auto (min of batch_size, 4)
        dtype: torch.dtype = torch.float16,
        checkpoint_dir: Optional[str] = None,  # Save progress for resume
        checkpoint_every: int = 500,  # Save every N samples
    ) -> "DatasetCache":
        """
        Build complete cache from images and prompts.

        Models are auto-loaded as needed:
        - Always loads: VAE, T5, CLIP
        - If build_lune=True: loads Lune
        - If build_sol=True: loads Sol

        If checkpoint_dir is provided, saves progress periodically so builds
        can resume from interruption.

        Args:
            zoo: ModelZoo instance
            images: List of PIL images
            prompts: List of text prompts
            name: Cache name
            build_lune: Whether to extract Lune features
            build_sol: Whether to extract Sol features
            batch_size: Batch size for encoding
            sol_batch_size: Batch size for Sol (None = auto)
            dtype: Storage dtype
            checkpoint_dir: Directory to save progress (None = no checkpointing)
            checkpoint_every: Save progress every N samples
        """
        print(f"Building cache: {name}")

        # Encodings (VAE, T5, CLIP)
        print("  [1/3] Encodings...")
        # Auto-load required models
        if zoo.vae is None:
            print("    Loading VAE...")
            zoo.load_vae()
        if zoo.t5 is None:
            print("    Loading T5...")
            zoo.load_t5()
        if zoo.clip is None:
            print("    Loading CLIP...")
            zoo.load_clip()

        encodings = EncodingCache.build(
            zoo, images, prompts, batch_size, dtype=dtype,
            checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every,
        )

        # Free VAE/T5 memory - keep CLIP for Lune/Sol
        zoo.offload("vae")
        zoo.offload("t5")
        torch.cuda.empty_cache()

        # Lune
        lune = None
        if build_lune:
            print("  [2/3] Lune features...")
            if zoo.lune is None:
                print("    Loading Lune...")
                zoo.load_lune()
            lune = LuneFeatureCache.build(
                zoo, prompts,
                batch_size=batch_size,
                dtype=dtype,
                batch_timesteps=True,  # Lune is lighter, can batch
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=checkpoint_every,
            )
            # Unload Lune before Sol
            zoo.unload("lune")
            torch.cuda.empty_cache()

        # Sol
        sol = None
        if build_sol:
            print("  [3/3] Sol features...")
            if zoo.sol is None:
                print("    Loading Sol...")
                zoo.load_sol()
            # Sol captures full attention matrices - default to small batch to avoid OOM
            sol_bs = sol_batch_size if sol_batch_size is not None else min(batch_size, 4)
            sol = SolFeatureCache.build(
                zoo, prompts,
                batch_size=sol_bs,
                dtype=dtype,
                batch_timesteps=False,  # Sol attention is heavy, don't batch timesteps
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=checkpoint_every,
            )

        print(f"  ✓ Cache complete: {len(encodings)} samples")
        return cls(encodings, lune, sol, name)

    @classmethod
    def build_incremental(
        cls,
        zoo,
        images: List[Image.Image],
        prompts: List[str],
        cache_dir: str,
        name: str = "dataset",
        build_lune: bool = True,
        build_sol: bool = True,
        batch_size: int = 32,
        sol_batch_size: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
        shard_size: int = 1000,
    ) -> "DatasetCache":
        """
        Build cache incrementally with shard saving for resume capability.

        Saves progress after each shard so interrupted builds can resume.

        Directory structure:
            cache_dir/
                state.json          # Progress tracking
                encodings/
                    shard_0000.pt
                    ...
                lune/
                    shard_0000.pt
                    ...
                sol/
                    shard_0000.pt
                    ...
        """
        import json

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "encodings"), exist_ok=True)
        if build_lune:
            os.makedirs(os.path.join(cache_dir, "lune"), exist_ok=True)
        if build_sol:
            os.makedirs(os.path.join(cache_dir, "sol"), exist_ok=True)

        n_samples = len(images)
        n_shards = (n_samples + shard_size - 1) // shard_size

        # Load or create state
        state_path = os.path.join(cache_dir, "state.json")
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            print(f"  Resuming from shard {len(state.get('completed_shards', []))}/{n_shards}")
        else:
            state = {
                'name': name,
                'n_samples': n_samples,
                'shard_size': shard_size,
                'build_lune': build_lune,
                'build_sol': build_sol,
                'completed_shards': [],
            }

        completed = set(state.get('completed_shards', []))

        print(f"Building cache: {name} ({n_samples} samples, {n_shards} shards)")

        # Load models once
        if zoo.vae is None:
            print("    Loading VAE...")
            zoo.load_vae()
        if zoo.t5 is None:
            print("    Loading T5...")
            zoo.load_t5()
        if zoo.clip is None:
            print("    Loading CLIP...")
            zoo.load_clip()

        lune_loaded = False
        sol_loaded = False

        for shard_idx in range(n_shards):
            if shard_idx in completed:
                continue

            start_idx = shard_idx * shard_size
            end_idx = min(start_idx + shard_size, n_samples)
            shard_images = images[start_idx:end_idx]
            shard_prompts = prompts[start_idx:end_idx]

            print(f"  Shard {shard_idx + 1}/{n_shards} (samples {start_idx}-{end_idx-1})...")

            # === Encodings ===
            enc_path = os.path.join(cache_dir, "encodings", f"shard_{shard_idx:04d}.pt")
            if not os.path.exists(enc_path):
                enc = EncodingCache.build(zoo, shard_images, shard_prompts, batch_size, dtype=dtype)
                enc.save(enc_path)
                del enc
                torch.cuda.empty_cache()

            # === Lune ===
            if build_lune:
                lune_path = os.path.join(cache_dir, "lune", f"shard_{shard_idx:04d}.pt")
                if not os.path.exists(lune_path):
                    if not lune_loaded:
                        zoo.offload("vae")
                        zoo.offload("t5")
                        if zoo.lune is None:
                            print("    Loading Lune...")
                            zoo.load_lune()
                        lune_loaded = True

                    lune_cache = LuneFeatureCache.build(
                        zoo, shard_prompts,
                        batch_size=batch_size,
                        dtype=dtype,
                        batch_timesteps=True,
                    )
                    lune_cache.save(lune_path)
                    del lune_cache
                    torch.cuda.empty_cache()

            # === Sol ===
            if build_sol:
                sol_path = os.path.join(cache_dir, "sol", f"shard_{shard_idx:04d}.pt")
                if not os.path.exists(sol_path):
                    if lune_loaded:
                        zoo.unload("lune")
                        lune_loaded = False
                    if not sol_loaded:
                        if zoo.sol is None:
                            print("    Loading Sol...")
                            zoo.load_sol()
                        sol_loaded = True

                    sol_bs = sol_batch_size if sol_batch_size is not None else min(batch_size, 4)
                    sol_cache = SolFeatureCache.build(
                        zoo, shard_prompts,
                        batch_size=sol_bs,
                        dtype=dtype,
                        batch_timesteps=False,
                    )
                    sol_cache.save(sol_path)
                    del sol_cache
                    torch.cuda.empty_cache()

            # Mark shard complete
            completed.add(shard_idx)
            state['completed_shards'] = list(completed)
            with open(state_path, 'w') as f:
                json.dump(state, f)

            print(f"    ✓ Shard {shard_idx + 1} saved")

        # Merge all shards
        print("  Merging shards...")
        return cls.load_shards(cache_dir, name)

    @classmethod
    def load_shards(cls, cache_dir: str, name: str = "dataset") -> "DatasetCache":
        """Load and merge shards from incremental build."""
        import json

        state_path = os.path.join(cache_dir, "state.json")
        with open(state_path, 'r') as f:
            state = json.load(f)

        # Find all shard files
        enc_dir = os.path.join(cache_dir, "encodings")
        enc_files = sorted([f for f in os.listdir(enc_dir) if f.endswith('.pt')])

        # Load and concatenate encodings
        all_latents = []
        all_t5 = []
        all_clip = []

        for enc_file in tqdm(enc_files, desc="Loading encodings"):
            enc = EncodingCache.load(os.path.join(enc_dir, enc_file))
            all_latents.append(enc.latents)
            all_t5.append(enc.t5_embeds)
            all_clip.append(enc.clip_pooled)

        encodings = EncodingCache(
            torch.cat(all_latents, dim=0),
            torch.cat(all_t5, dim=0),
            torch.cat(all_clip, dim=0),
        )

        # Load Lune shards
        lune = None
        lune_dir = os.path.join(cache_dir, "lune")
        if os.path.exists(lune_dir) and state.get('build_lune', False):
            lune_files = sorted([f for f in os.listdir(lune_dir) if f.endswith('.pt')])
            all_lune = []
            for lune_file in tqdm(lune_files, desc="Loading lune"):
                lune_cache = LuneFeatureCache.load(os.path.join(lune_dir, lune_file))
                all_lune.append(lune_cache.features)
            lune = LuneFeatureCache(
                torch.cat(all_lune, dim=0),
                lune_cache.t_buckets,
            )

        # Load Sol shards
        sol = None
        sol_dir = os.path.join(cache_dir, "sol")
        if os.path.exists(sol_dir) and state.get('build_sol', False):
            sol_files = sorted([f for f in os.listdir(sol_dir) if f.endswith('.pt')])
            all_stats = []
            all_spatial = []
            for sol_file in tqdm(sol_files, desc="Loading sol"):
                sol_cache = SolFeatureCache.load(os.path.join(sol_dir, sol_file))
                all_stats.append(sol_cache.stats)
                all_spatial.append(sol_cache.spatial)
            sol = SolFeatureCache(
                torch.cat(all_stats, dim=0),
                torch.cat(all_spatial, dim=0),
                sol_cache.t_buckets,
            )

        print(f"  ✓ Loaded {len(encodings)} samples from {len(enc_files)} shards")
        return cls(encodings, lune, sol, name)

    def __repr__(self) -> str:
        return f"DatasetCache(name={self.name}, n={len(self)}, lune={self.lune is not None}, sol={self.sol is not None})"


# =============================================================================
# Multi-Source Cache (multiple datasets)
# =============================================================================

class MultiSourceCache:
    """
    Combines multiple DatasetCaches for training on mixed data.

    Routes by dataset_id to correct cache.

    Usage:
        multi = MultiSourceCache()
        multi.add(portrait_cache, dataset_id=0)
        multi.add(imagenet_cache, dataset_id=1)

        # In training
        lune = multi.get_lune(local_indices, dataset_ids, timesteps)
        sol_stats, sol_spatial = multi.get_sol(local_indices, dataset_ids, timesteps)
    """

    def __init__(self, dtype: torch.dtype = torch.float16):
        self.caches: Dict[int, DatasetCache] = {}
        self.dtype = dtype

    def add(self, cache: DatasetCache, dataset_id: int):
        """Add a dataset cache."""
        self.caches[dataset_id] = cache

    def get_encodings(
        self,
        local_indices: torch.Tensor,
        dataset_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get encodings for batch."""
        B = local_indices.shape[0]

        # Determine shapes from first cache
        first_cache = next(iter(self.caches.values()))
        C, H, W = first_cache.encodings.latents.shape[1:]
        L = first_cache.encodings.t5_embeds.shape[1]

        latents = torch.zeros(B, C, H, W, dtype=self.dtype)
        t5 = torch.zeros(B, L, 768, dtype=self.dtype)
        clip = torch.zeros(B, 768, dtype=self.dtype)

        for ds_id, cache in self.caches.items():
            mask = dataset_ids == ds_id
            if not mask.any():
                continue

            idx = local_indices[mask].cpu()
            lat, t5_e, clip_e = cache.encodings.get_batch(idx)

            latents[mask] = lat.to(self.dtype)
            t5[mask] = t5_e.to(self.dtype)
            clip[mask] = clip_e.to(self.dtype)

        return latents, t5, clip

    def get_lune(
        self,
        local_indices: torch.Tensor,
        dataset_ids: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Get Lune features for batch."""
        if not any(c.lune is not None for c in self.caches.values()):
            return None

        B = local_indices.shape[0]
        device = timesteps.device
        features = torch.zeros(B, LUNE_DIM, device=device, dtype=self.dtype)

        for ds_id, cache in self.caches.items():
            if cache.lune is None:
                continue

            mask = dataset_ids == ds_id
            if not mask.any():
                continue

            ds_idx = local_indices[mask]
            ds_t = timesteps[mask]
            features[mask] = cache.lune.get(ds_idx, ds_t).to(self.dtype)

        return features

    def get_sol(
        self,
        local_indices: torch.Tensor,
        dataset_ids: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get Sol features for batch."""
        if not any(c.sol is not None for c in self.caches.values()):
            return None, None

        B = local_indices.shape[0]
        device = timesteps.device

        stats = torch.zeros(B, SOL_STATS_DIM, device=device, dtype=self.dtype)
        spatial = torch.zeros(B, SOL_SPATIAL_SIZE, SOL_SPATIAL_SIZE, device=device, dtype=self.dtype)

        for ds_id, cache in self.caches.items():
            if cache.sol is None:
                continue

            mask = dataset_ids == ds_id
            if not mask.any():
                continue

            ds_idx = local_indices[mask]
            ds_t = timesteps[mask]
            ds_stats, ds_spatial = cache.sol.get(ds_idx, ds_t)

            stats[mask] = ds_stats.to(self.dtype)
            spatial[mask] = ds_spatial.to(self.dtype)

        return stats, spatial

    def __repr__(self) -> str:
        names = [f"{k}:{v.name}" for k, v in self.caches.items()]
        return f"MultiSourceCache({', '.join(names)})"