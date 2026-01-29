"""
TinyFlux Data Utilities

- MultiSourceDataset: Combines multiple datasets with separate caches
- ExpertCacheRouter: Routes batch to correct cache by dataset_id
- collate_fn: Batches samples properly
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class MultiSourceDataset(Dataset):
    """
    Dataset combining multiple sources, each with its own encodings.

    Returns sample_idx, local_idx, dataset_id for cache routing.

    Usage:
        dataset = MultiSourceDataset()
        dataset.add_source(
            name="portraits",
            latents=portrait_latents,  # [N, C, H, W]
            t5_embeds=portrait_t5,     # [N, L, 768]
            clip_pooled=portrait_clip, # [N, 768]
            masks=portrait_masks,      # [N, H, W] optional
        )
        dataset.add_source(name="imagenet", ...)
    """

    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        self.dtype = dtype
        self.sources: List[Dict[str, Any]] = []
        self.cumulative: List[int] = [0]
        self.total = 0

    def add_source(
            self,
            name: str,
            latents: torch.Tensor,
            t5_embeds: torch.Tensor,
            clip_pooled: torch.Tensor,
            masks: Optional[torch.Tensor] = None,
    ):
        """
        Add a data source.

        Args:
            name: Source identifier
            latents: [N, C, H, W] encoded images
            t5_embeds: [N, L, 768] T5 embeddings
            clip_pooled: [N, 768] CLIP pooled
            masks: [N, H, W] optional foreground masks
        """
        n = len(latents)
        self.sources.append({
            'name': name,
            'latents': latents,
            't5_embeds': t5_embeds,
            'clip_pooled': clip_pooled,
            'masks': masks,
            'n': n,
            'dataset_id': len(self.sources),
        })
        self.total += n
        self.cumulative.append(self.total)

    def __len__(self) -> int:
        return self.total

    def _find_source(self, idx: int) -> Tuple[int, int]:
        """Find source and local index for global index."""
        for i, (start, end) in enumerate(zip(self.cumulative[:-1], self.cumulative[1:])):
            if start <= idx < end:
                return i, idx - start
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        source_id, local_idx = self._find_source(idx)
        source = self.sources[source_id]

        result = {
            'latent': source['latents'][local_idx].to(self.dtype),
            't5_embed': source['t5_embeds'][local_idx].to(self.dtype),
            'clip_pooled': source['clip_pooled'][local_idx].to(self.dtype),
            'sample_idx': idx,
            'local_idx': local_idx,
            'dataset_id': source['dataset_id'],
        }

        if source['masks'] is not None:
            result['mask'] = source['masks'][local_idx].to(self.dtype)

        return result

    def __repr__(self) -> str:
        lines = [f"MultiSourceDataset({self.total} samples)"]
        for i, src in enumerate(self.sources):
            lines.append(f"  [{i}] {src['name']}: {src['n']}")
        return "\n".join(lines)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for MultiSourceDataset.

    Returns:
        latents: [B, C, H, W]
        t5_embeds: [B, L, 768]
        clip_pooled: [B, 768]
        local_indices: [B] local index within source
        dataset_ids: [B] which source
        masks: [B, H, W] or None
    """
    latents = torch.stack([b['latent'] for b in batch])
    t5_embeds = torch.stack([b['t5_embed'] for b in batch])
    clip_pooled = torch.stack([b['clip_pooled'] for b in batch])
    local_indices = torch.tensor([b['local_idx'] for b in batch], dtype=torch.long)
    dataset_ids = torch.tensor([b['dataset_id'] for b in batch], dtype=torch.long)

    masks = None
    if any('mask' in b for b in batch):
        H = W = 64  # Default
        if 'mask' in batch[0]:
            H, W = batch[0]['mask'].shape
        masks = []
        for b in batch:
            if 'mask' in b:
                masks.append(b['mask'])
            else:
                masks.append(torch.ones(H, W, dtype=latents.dtype))
        masks = torch.stack(masks)

    return {
        'latents': latents,
        't5_embeds': t5_embeds,
        'clip_pooled': clip_pooled,
        'local_indices': local_indices,
        'dataset_ids': dataset_ids,
        'masks': masks,
    }


class ExpertCacheRouter:
    """
    Routes batch samples to correct cache based on dataset_id.

    Usage:
        router = ExpertCacheRouter()
        router.add_lune_cache(dataset_id=0, cache=portrait_lune_cache)
        router.add_lune_cache(dataset_id=1, cache=schnell_lune_cache)
        router.add_sol_cache(dataset_id=0, cache=portrait_sol_cache)
        ...

        # In training loop:
        lune_feats = router.get_lune(local_indices, dataset_ids, timesteps)
        sol_stats, sol_spatial = router.get_sol(local_indices, dataset_ids, timesteps)
    """

    def __init__(
            self,
            lune_dim: int = 1280,
            sol_stats_dim: int = 3,
            sol_spatial_size: int = 8,
            dtype: torch.dtype = torch.bfloat16,
    ):
        self.lune_caches: Dict[int, Any] = {}
        self.sol_caches: Dict[int, Any] = {}
        self.lune_dim = lune_dim
        self.sol_stats_dim = sol_stats_dim
        self.sol_spatial_size = sol_spatial_size
        self.dtype = dtype

    def add_lune_cache(self, dataset_id: int, cache):
        """Add Lune cache for a dataset."""
        self.lune_caches[dataset_id] = cache

    def add_sol_cache(self, dataset_id: int, cache):
        """Add Sol cache for a dataset."""
        self.sol_caches[dataset_id] = cache

    def get_lune(
            self,
            local_indices: torch.Tensor,
            dataset_ids: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Get Lune features for batch, routing to correct cache.

        Args:
            local_indices: [B] local index within each dataset
            dataset_ids: [B] which dataset
            timesteps: [B] timesteps for interpolation

        Returns:
            features: [B, 1280] or None if no caches
        """
        if not self.lune_caches:
            return None

        B = local_indices.shape[0]
        device = timesteps.device
        features = torch.zeros(B, self.lune_dim, device=device, dtype=self.dtype)

        for ds_id, cache in self.lune_caches.items():
            mask = dataset_ids == ds_id
            if not mask.any():
                continue

            ds_indices = local_indices[mask]
            ds_timesteps = timesteps[mask]
            ds_features = cache.get(ds_indices, ds_timesteps)
            features[mask] = ds_features.to(device, dtype=self.dtype)

        return features

    def get_sol(
            self,
            local_indices: torch.Tensor,
            dataset_ids: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get Sol features for batch, routing to correct cache.

        Returns:
            stats: [B, 3] or None
            spatial: [B, 8, 8] or None
        """
        if not self.sol_caches:
            return None, None

        B = local_indices.shape[0]
        device = timesteps.device
        H = W = self.sol_spatial_size

        stats = torch.zeros(B, self.sol_stats_dim, device=device, dtype=self.dtype)
        spatial = torch.zeros(B, H, W, device=device, dtype=self.dtype)

        for ds_id, cache in self.sol_caches.items():
            mask = dataset_ids == ds_id
            if not mask.any():
                continue

            ds_indices = local_indices[mask]
            ds_timesteps = timesteps[mask]
            ds_stats, ds_spatial = cache.get(ds_indices, ds_timesteps)

            # Take first 3 stats (drop sparsity if present)
            stats[mask] = ds_stats[:, :self.sol_stats_dim].to(device, dtype=self.dtype)
            spatial[mask] = ds_spatial.to(device, dtype=self.dtype)

        return stats, spatial

    def __repr__(self) -> str:
        return f"ExpertCacheRouter(lune={list(self.lune_caches.keys())}, sol={list(self.sol_caches.keys())})"