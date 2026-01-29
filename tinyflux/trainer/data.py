"""
TinyFlux Data Utilities

Bridges cache_experts.py (storage) and trainer.py (consumption).

- CachedDataset: torch Dataset wrapping DatasetCache
- MultiCachedDataset: combines multiple CachedDatasets with dataset_id routing
- collate_fn: batches samples for trainer
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any

from .cache_experts import DatasetCache, MultiSourceCache


class CachedDataset(Dataset):
    """
    Dataset wrapping a DatasetCache.

    Yields samples with indices for expert cache lookup.

    Usage:
        cache = DatasetCache.load("cache/portraits/")
        dataset = CachedDataset(cache, dataset_id=0)
        loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    """

    def __init__(
        self,
        cache: DatasetCache,
        dataset_id: int = 0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.cache = cache
        self.dataset_id = dataset_id
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        latent, t5, clip = self.cache.get_encodings(idx)

        return {
            'latent': latent.to(self.dtype),
            't5_embed': t5.to(self.dtype),
            'clip_pooled': clip.to(self.dtype),
            'local_idx': idx,
            'dataset_id': self.dataset_id,
        }


class MultiCachedDataset(Dataset):
    """
    Combines multiple CachedDatasets into one.

    Handles dataset_id routing automatically.

    Usage:
        multi = MultiCachedDataset()
        multi.add(portrait_cache, dataset_id=0)
        multi.add(imagenet_cache, dataset_id=1)

        loader = DataLoader(multi, batch_size=8, collate_fn=collate_fn)
    """

    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        self.caches: List[Tuple[DatasetCache, int]] = []  # (cache, dataset_id)
        self.cumulative: List[int] = [0]
        self.total = 0
        self.dtype = dtype

    def add(self, cache: DatasetCache, dataset_id: int):
        """Add a dataset cache."""
        n = len(cache)
        self.caches.append((cache, dataset_id))
        self.total += n
        self.cumulative.append(self.total)

    def __len__(self) -> int:
        return self.total

    def _find_cache(self, idx: int) -> Tuple[DatasetCache, int, int]:
        """Find cache, dataset_id, and local index for global index."""
        for i, (start, end) in enumerate(zip(self.cumulative[:-1], self.cumulative[1:])):
            if start <= idx < end:
                cache, dataset_id = self.caches[i]
                local_idx = idx - start
                return cache, dataset_id, local_idx
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cache, dataset_id, local_idx = self._find_cache(idx)
        latent, t5, clip = cache.get_encodings(local_idx)

        return {
            'latent': latent.to(self.dtype),
            't5_embed': t5.to(self.dtype),
            'clip_pooled': clip.to(self.dtype),
            'local_idx': local_idx,
            'dataset_id': dataset_id,
        }

    def get_multi_source_cache(self) -> MultiSourceCache:
        """Build MultiSourceCache for expert feature lookup."""
        multi = MultiSourceCache(dtype=self.dtype)
        for cache, dataset_id in self.caches:
            multi.add(cache, dataset_id)
        return multi

    def __repr__(self) -> str:
        parts = [f"{ds_id}:{c.name}({len(c)})" for c, ds_id in self.caches]
        return f"MultiCachedDataset({', '.join(parts)})"


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for CachedDataset / MultiCachedDataset.

    Returns batch format expected by Trainer:
        latents: [B, C, H, W]
        t5_embeds: [B, L, 768]
        clip_pooled: [B, 768]
        local_indices: [B]
        dataset_ids: [B]
        masks: None (not currently cached)
    """
    latents = torch.stack([b['latent'] for b in batch])
    t5_embeds = torch.stack([b['t5_embed'] for b in batch])
    clip_pooled = torch.stack([b['clip_pooled'] for b in batch])
    local_indices = torch.tensor([b['local_idx'] for b in batch], dtype=torch.long)
    dataset_ids = torch.tensor([b['dataset_id'] for b in batch], dtype=torch.long)

    return {
        'latents': latents,
        't5_embeds': t5_embeds,
        'clip_pooled': clip_pooled,
        'local_indices': local_indices,
        'dataset_ids': dataset_ids,
        'masks': None,
    }