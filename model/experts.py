"""
Expert Feature Caches

Stores precached features from Lune and Sol teachers for efficient training.
Features are extracted at discrete timestep buckets and interpolated at runtime.

IMPORTANT: Use deterministic seeds when extracting Lune features.
    seed_for_extraction(sample_idx, bucket_idx) provides reproducible seeds.

Usage:
    # Build cache (done once per dataset)
    cache = LuneFeatureCache.build(zoo, prompts, cache_path)

    # During training
    features = cache.get(sample_indices, timesteps)
"""

import torch
from typing import Tuple, List, Optional
from pathlib import Path


DEFAULT_T_BUCKETS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
LUNE_DIM = 1280
SOL_STATS_DIM = 3  # locality, entropy, clustering (model expects 3)
SOL_SPATIAL_SIZE = 8

# Base seed for reproducibility
EXTRACTION_BASE_SEED = 42


def get_t_buckets(device: str = "cpu") -> torch.Tensor:
    """Default timestep buckets for precaching."""
    return torch.tensor(DEFAULT_T_BUCKETS, device=device)


def seed_for_extraction(sample_idx: int, bucket_idx: int, base_seed: int = EXTRACTION_BASE_SEED) -> int:
    """
    Deterministic seed for a (sample, timestep_bucket) pair.

    Use this when generating random latents for Lune extraction:
        seed = seed_for_extraction(i, t_idx)
        torch.manual_seed(seed)
        latents = torch.randn(1, 4, 64, 64, ...)
    """
    return base_seed + sample_idx * 1000 + bucket_idx


def seed_batch_extraction(
    start_idx: int,
    batch_size: int,
    n_buckets: int,
    device: str = "cuda",
    base_seed: int = EXTRACTION_BASE_SEED,
) -> torch.Generator:
    """
    Create a seeded generator for batched extraction.

    For batch of B samples × T timesteps:
        gen = seed_batch_extraction(start_idx, B, T)
        latents = torch.randn(B * T, 4, 64, 64, generator=gen, device=device)
    """
    # Seed based on start of batch
    seed = base_seed + start_idx * 1000
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


class LuneFeatureCache:
    """
    Precached Lune mid-block features with timestep interpolation.

    Shape: [N_samples, N_buckets, 1280]
    """

    def __init__(
        self,
        features: torch.Tensor,
        t_buckets: torch.Tensor,
        dtype: torch.dtype = torch.float16,
    ):
        self.features = features.to(dtype)  # [N, n_buckets, 1280]
        self.t_buckets = t_buckets.cpu()
        self.t_min = t_buckets[0].item()
        self.t_max = t_buckets[-1].item()
        self.t_step = (t_buckets[1] - t_buckets[0]).item()
        self.n_buckets = len(t_buckets)
        self.dtype = dtype

    def get(self, indices: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Get interpolated features for given samples and timesteps.

        Args:
            indices: [B] sample indices into cache
            timesteps: [B] timesteps in [0, 1]

        Returns:
            features: [B, 1280]
        """
        device = timesteps.device

        # Find bucket indices for interpolation
        t_clamped = timesteps.float().clamp(self.t_min, self.t_max)
        t_idx_float = (t_clamped - self.t_min) / self.t_step
        t_idx_low = t_idx_float.long().clamp(0, self.n_buckets - 2)
        t_idx_high = (t_idx_low + 1).clamp(0, self.n_buckets - 1)
        alpha = (t_idx_float - t_idx_low.float()).unsqueeze(-1)

        # Gather on CPU (features stored on CPU)
        idx_cpu = indices.cpu()
        t_low_cpu = t_idx_low.cpu()
        t_high_cpu = t_idx_high.cpu()

        f_low = self.features[idx_cpu, t_low_cpu]
        f_high = self.features[idx_cpu, t_high_cpu]

        # Interpolate
        result = (1 - alpha.cpu()) * f_low + alpha.cpu() * f_high
        return result.to(device=device, dtype=self.dtype)

    def save(self, path: str):
        """Save cache to file."""
        torch.save({
            "features": self.features,
            "t_buckets": self.t_buckets,
        }, path)
        print(f"Saved LuneFeatureCache to {path}")

    @classmethod
    def load(cls, path: str, dtype: torch.dtype = torch.float16) -> "LuneFeatureCache":
        """Load cache from file."""
        data = torch.load(path, map_location="cpu")
        return cls(data["features"], data["t_buckets"], dtype)

    @classmethod
    def build(
        cls,
        zoo: "ModelZoo",
        prompts: List[str],
        t_buckets: Optional[torch.Tensor] = None,
        batch_size: int = 64,
        dtype: torch.dtype = torch.float16,
        base_seed: int = EXTRACTION_BASE_SEED,
    ) -> "LuneFeatureCache":
        """
        Extract Lune features for all prompts at all timestep buckets.

        Requires zoo with clip and lune loaded:
            zoo.load_clip()
            zoo.load_lune(compile_model=True)
            cache = LuneFeatureCache.build(zoo, prompts)
            zoo.unload("lune")  # Free VRAM after extraction

        Args:
            zoo: ModelZoo with clip and lune loaded
            prompts: List of text prompts
            t_buckets: Timestep buckets (default: 10 buckets from 0.05 to 0.95)
            batch_size: Prompts per batch
            dtype: Storage dtype
            base_seed: Base seed for reproducible latent generation
        """
        from tqdm import tqdm

        if zoo.clip is None:
            raise RuntimeError("CLIP not loaded. Call zoo.load_clip() first.")
        if zoo.lune is None:
            raise RuntimeError("Lune not loaded. Call zoo.load_lune() first.")

        if t_buckets is None:
            t_buckets = get_t_buckets(zoo.device)

        n_prompts = len(prompts)
        n_buckets = len(t_buckets)
        device = zoo.device

        all_features = torch.zeros(n_prompts, n_buckets, LUNE_DIM, dtype=dtype)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            for start_idx in tqdm(range(0, n_prompts, batch_size), desc="Extracting Lune"):
                end_idx = min(start_idx + batch_size, n_prompts)
                batch_prompts = prompts[start_idx:end_idx]
                B = len(batch_prompts)

                # Encode CLIP
                clip_inputs = zoo.clip_tokenizer(
                    batch_prompts, return_tensors="pt", padding="max_length",
                    max_length=77, truncation=True
                ).to(device)
                clip_hidden = zoo.clip(**clip_inputs).last_hidden_state

                # Expand for all timesteps: [B, n_buckets, 77, 768] -> [B*n_buckets, 77, 768]
                clip_expanded = clip_hidden.unsqueeze(1).expand(-1, n_buckets, -1, -1)
                clip_expanded = clip_expanded.reshape(B * n_buckets, 77, -1)

                # Timesteps: [B * n_buckets]
                t_expanded = t_buckets.unsqueeze(0).expand(B, -1).reshape(-1)

                # Seeded random latents for reproducibility
                gen = seed_batch_extraction(start_idx, B, n_buckets, device, base_seed)
                latents = torch.randn(
                    B * n_buckets, 4, 64, 64,
                    generator=gen, device=device, dtype=torch.float16
                )

                # Forward through Lune (hook captures mid-block features)
                _ = zoo.lune(latents, t_expanded * 1000, encoder_hidden_states=clip_expanded.half())

                # Reshape: [B * n_buckets, 1280] -> [B, n_buckets, 1280]
                features = zoo.lune_mid_features.reshape(B, n_buckets, -1)
                all_features[start_idx:end_idx] = features.cpu().to(dtype)

        return cls(all_features, t_buckets.cpu(), dtype)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __repr__(self) -> str:
        return f"LuneFeatureCache(samples={len(self)}, buckets={self.n_buckets}, dim={LUNE_DIM})"


class SolFeatureCache:
    """
    Precached Sol attention statistics with timestep interpolation.

    Shapes:
        stats: [N_samples, N_buckets, 4]
        spatial: [N_samples, N_buckets, 8, 8]
    """

    def __init__(
        self,
        stats: torch.Tensor,
        spatial: torch.Tensor,
        t_buckets: torch.Tensor,
        dtype: torch.dtype = torch.float16,
    ):
        self.stats = stats.to(dtype)  # [N, n_buckets, 4]
        self.spatial = spatial.to(dtype)  # [N, n_buckets, 8, 8]
        self.t_buckets = t_buckets.cpu()
        self.t_min = t_buckets[0].item()
        self.t_max = t_buckets[-1].item()
        self.t_step = (t_buckets[1] - t_buckets[0]).item()
        self.n_buckets = len(t_buckets)
        self.dtype = dtype

    def get(
        self,
        indices: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get interpolated features for given samples and timesteps.

        Returns:
            stats: [B, 4]
            spatial: [B, 8, 8]
        """
        device = timesteps.device

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
        """Save cache to file."""
        torch.save({
            "stats": self.stats,
            "spatial": self.spatial,
            "t_buckets": self.t_buckets,
        }, path)
        print(f"Saved SolFeatureCache to {path}")

    @classmethod
    def load(cls, path: str, dtype: torch.dtype = torch.float16) -> "SolFeatureCache":
        """Load cache from file."""
        data = torch.load(path, map_location="cpu")
        return cls(data["stats"], data["spatial"], data["t_buckets"], dtype)

    @classmethod
    def from_geometric(
        cls,
        n_samples: int,
        t_buckets: Optional[torch.Tensor] = None,
        spatial_size: int = SOL_SPATIAL_SIZE,
        dtype: torch.dtype = torch.float16,
    ) -> "SolFeatureCache":
        """
        DEPRECATED: Use build() with a proper zoo instead.

        Create cache using geometric heuristics (no model needed).
        WARNING: These are fake statistics, not real attention patterns.
        """
        import warnings
        warnings.warn(
            "SolFeatureCache.from_geometric() produces fake statistics. "
            "Use SolFeatureCache.build(zoo, prompts) for real attention extraction.",
            DeprecationWarning
        )

        if t_buckets is None:
            t_buckets = get_t_buckets()

        n_buckets = len(t_buckets)
        t_vals = t_buckets.float()

        # Stats: [n_buckets, 3] -> broadcast to [N, n_buckets, 3]
        # Only 3 stats - sparsity dropped (was redundant with locality)
        locality = 1 - t_vals
        entropy = t_vals
        clustering = 0.5 - 0.3 * (t_vals - 0.5).abs()
        stats_per_t = torch.stack([locality, entropy, clustering], dim=-1)
        all_stats = stats_per_t.unsqueeze(0).expand(n_samples, -1, -1)

        # Spatial: [n_buckets, H, W] -> broadcast to [N, n_buckets, H, W]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, spatial_size),
            torch.linspace(-1, 1, spatial_size),
            indexing='ij'
        )
        center_dist = torch.sqrt(x ** 2 + y ** 2)
        t_weight = (1 - t_vals).view(-1, 1, 1)
        center_bias = 1 - center_dist.unsqueeze(0) * t_weight
        center_bias = center_bias / center_bias.sum(dim=[-2, -1], keepdim=True)
        all_spatial = center_bias.unsqueeze(0).expand(n_samples, -1, -1, -1)

        return cls(all_stats, all_spatial, t_buckets, dtype)

    @classmethod
    def build(
        cls,
        zoo: "ModelZoo",
        prompts: List[str],
        t_buckets: Optional[torch.Tensor] = None,
        batch_size: int = 64,
        spatial_size: int = SOL_SPATIAL_SIZE,
        dtype: torch.dtype = torch.float16,
        base_seed: int = EXTRACTION_BASE_SEED,
    ) -> "SolFeatureCache":
        """
        Extract Sol attention statistics for all prompts at all timestep buckets.

        Requires zoo with clip and sol loaded:
            zoo.load_clip()
            zoo.load_sol()
            cache = SolFeatureCache.build(zoo, prompts)
            zoo.unload("sol")  # Free VRAM after extraction

        Args:
            zoo: ModelZoo with clip and sol loaded
            prompts: List of text prompts
            t_buckets: Timestep buckets (default: 10 buckets from 0.05 to 0.95)
            batch_size: Prompts per batch
            spatial_size: Output spatial resolution
            dtype: Storage dtype
            base_seed: Base seed for reproducible latent generation
        """
        from tqdm import tqdm

        if zoo.clip is None:
            raise RuntimeError("CLIP not loaded. Call zoo.load_clip() first.")
        if zoo.sol is None:
            raise RuntimeError("Sol not loaded. Call zoo.load_sol() first.")

        if t_buckets is None:
            t_buckets = get_t_buckets(zoo.device)

        n_prompts = len(prompts)
        n_buckets = len(t_buckets)
        device = zoo.device

        all_stats = torch.zeros(n_prompts, n_buckets, SOL_STATS_DIM, dtype=dtype)
        all_spatial = torch.zeros(n_prompts, n_buckets, spatial_size, spatial_size, dtype=dtype)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            for start_idx in tqdm(range(0, n_prompts, batch_size), desc="Extracting Sol"):
                end_idx = min(start_idx + batch_size, n_prompts)
                batch_prompts = prompts[start_idx:end_idx]
                B = len(batch_prompts)

                # Encode CLIP
                clip_inputs = zoo.clip_tokenizer(
                    batch_prompts, return_tensors="pt", padding="max_length",
                    max_length=77, truncation=True
                ).to(device)
                clip_hidden = zoo.clip(**clip_inputs).last_hidden_state

                # Process each timestep bucket
                for t_idx, t_val in enumerate(t_buckets):
                    t = torch.full((B,), t_val.item(), device=device)

                    # Seeded random latents
                    gen = seed_batch_extraction(start_idx, B, t_idx, device, base_seed)
                    latents = torch.randn(
                        B, 4, 64, 64,
                        generator=gen, device=device, dtype=torch.float16
                    )

                    # Extract attention statistics
                    stats, spatial = zoo.sol_forward(latents, t, clip_hidden.half(), spatial_size)

                    all_stats[start_idx:end_idx, t_idx] = stats.cpu().to(dtype)
                    all_spatial[start_idx:end_idx, t_idx] = spatial.cpu().to(dtype)

        return cls(all_stats, all_spatial, t_buckets.cpu(), dtype)

    def __len__(self) -> int:
        return self.stats.shape[0]

    def __repr__(self) -> str:
        return f"SolFeatureCache(samples={len(self)}, buckets={self.n_buckets})"


# =============================================================================
# Smoke Test
# =============================================================================

def _smoke_test():
    print("Expert Caches Smoke Test")
    print("=" * 50)

    N = 100  # samples
    B = 8    # batch
    t_buckets = get_t_buckets()

    # Test seeding utilities
    print("\n[1] Seeding utilities...")
    seed1 = seed_for_extraction(0, 0)
    seed2 = seed_for_extraction(0, 1)
    seed3 = seed_for_extraction(1, 0)
    assert seed1 != seed2 != seed3, "Seeds should differ"
    assert seed_for_extraction(5, 3) == seed_for_extraction(5, 3), "Same inputs -> same seed"
    print(f"    seed_for_extraction(0,0)={seed1}, (0,1)={seed2}, (1,0)={seed3} ✓")

    gen1 = seed_batch_extraction(0, 8, 10, device="cpu")
    gen2 = seed_batch_extraction(0, 8, 10, device="cpu")
    r1 = torch.randn(3, generator=gen1)
    r2 = torch.randn(3, generator=gen2)
    assert torch.equal(r1, r2), "Same seed -> same randoms"
    print(f"    seed_batch_extraction reproducible ✓")

    # Test Lune cache
    print("\n[2] LuneFeatureCache...")
    lune_features = torch.randn(N, len(t_buckets), LUNE_DIM)
    lune_cache = LuneFeatureCache(lune_features, t_buckets)

    indices = torch.randint(0, N, (B,))
    timesteps = torch.rand(B)
    out = lune_cache.get(indices, timesteps)
    assert out.shape == (B, LUNE_DIM), f"Bad shape: {out.shape}"
    print(f"    {lune_cache}")
    print(f"    get({B} samples) -> {out.shape} ✓")

    # Test Sol cache (geometric)
    print("\n[3] SolFeatureCache (geometric)...")
    sol_cache = SolFeatureCache.from_geometric(N)

    stats, spatial = sol_cache.get(indices, timesteps)
    assert stats.shape == (B, SOL_STATS_DIM), f"Bad stats: {stats.shape}"
    assert spatial.shape == (B, SOL_SPATIAL_SIZE, SOL_SPATIAL_SIZE), f"Bad spatial: {spatial.shape}"
    print(f"    {sol_cache}")
    print(f"    get({B} samples) -> stats {stats.shape}, spatial {spatial.shape} ✓")

    print("\n" + "=" * 50)
    print("✓ All tests passed")


if __name__ == "__main__":
    _smoke_test()