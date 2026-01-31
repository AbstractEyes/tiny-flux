"""
Simple Directory Dataset for LoRA Training

Loads images and prompts from a directory structure:

    data_dir/
        image1.png
        image1.txt      # prompt for image1
        image2.jpg
        image2.txt
        ...

Or with a single prompts file:

    data_dir/
        image1.png
        image2.jpg
        prompts.txt     # one prompt per line, matching image order

Supports:
- Repeats to expand small datasets
- Resize from any size to 512x512
- Common image formats (png, jpg, jpeg, webp)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple, Optional, Dict
from pathlib import Path

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


class DirectoryDataset(Dataset):
    """
    Simple dataset from a directory of images + prompts.

    Args:
        data_dir: Directory containing images and prompts
        repeats: Number of times to repeat the dataset
        target_size: Resize images to this size (default 512)
        prompt_file: Optional single file with all prompts (one per line)

    Usage:
        dataset = DirectoryDataset("/content/drive/MyDrive/lora_data", repeats=100)
        print(f"Dataset size: {len(dataset)}")  # 12 images × 100 = 1200

        img, prompt = dataset[0]
    """

    def __init__(
            self,
            data_dir: str,
            repeats: int = 1,
            target_size: int = 512,
            prompt_file: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.repeats = repeats
        self.target_size = target_size

        # Find all images
        self.image_paths: List[Path] = []
        for f in sorted(self.data_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                self.image_paths.append(f)

        if not self.image_paths:
            raise ValueError(f"No images found in {data_dir}")

        # Load prompts
        self.prompts = self._load_prompts(prompt_file)

        if len(self.prompts) != len(self.image_paths):
            raise ValueError(
                f"Mismatch: {len(self.image_paths)} images but {len(self.prompts)} prompts"
            )

        self.base_size = len(self.image_paths)

        print(f"[DirectoryDataset]")
        print(f"  Images: {self.base_size}")
        print(f"  Repeats: {self.repeats}")
        print(f"  Total samples: {len(self)}")
        print(f"  Target size: {self.target_size}x{self.target_size}")

    def _load_prompts(self, prompt_file: Optional[str]) -> List[str]:
        """Load prompts from individual .txt files or single prompts file."""

        # Check for single prompts file
        if prompt_file:
            prompt_path = self.data_dir / prompt_file
        else:
            prompt_path = self.data_dir / "prompts.txt"

        if prompt_path.exists():
            # Single file with all prompts
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            return prompts

        # Individual .txt files per image
        prompts = []
        for img_path in self.image_paths:
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
            else:
                # Fallback: use filename as prompt
                prompt = img_path.stem.replace('_', ' ').replace('-', ' ')
                print(f"  Warning: No prompt for {img_path.name}, using filename")
            prompts.append(prompt)

        return prompts

    def __len__(self) -> int:
        return self.base_size * self.repeats

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        """Returns (PIL Image, prompt string)."""
        base_idx = idx % self.base_size

        # Load and resize image
        img = Image.open(self.image_paths[base_idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if img.size != (self.target_size, self.target_size):
            img = img.resize(
                (self.target_size, self.target_size),
                Image.Resampling.LANCZOS
            )

        return img, self.prompts[base_idx]

    def get_images_and_prompts(self) -> Tuple[List[Image.Image], List[str]]:
        """Get all unique images and prompts (no repeats) for cache building."""
        images = []
        for path in self.image_paths:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if img.size != (self.target_size, self.target_size):
                img = img.resize(
                    (self.target_size, self.target_size),
                    Image.Resampling.LANCZOS
                )
            images.append(img)
        return images, self.prompts.copy()


class CachedDirectoryDataset(Dataset):
    """
    Dataset that wraps a built cache with repeat support.

    Used after building the cache from DirectoryDataset.

    Args:
        cache: DatasetCache instance
        repeats: Number of times to repeat
    """

    def __init__(self, cache, repeats: int = 1):
        self.cache = cache
        self.repeats = repeats
        self.base_size = len(cache)

    def __len__(self) -> int:
        return self.base_size * self.repeats

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns dict with 'index' pointing to base cache index."""
        base_idx = idx % self.base_size
        return {'index': torch.tensor(base_idx)}


def build_cache_from_directory(
        zoo,
        data_dir: str,
        cache_dir: str,
        target_size: int = 512,
        prompt_file: Optional[str] = None,
        build_lune: bool = True,
        build_sol: bool = True,
        batch_size: int = 4,
        compile_experts: bool = False,  # Small dataset, compilation overhead not worth it
) -> "DatasetCache":
    """
    Build cache from a directory of images.

    Args:
        zoo: ModelZoo instance
        data_dir: Directory with images and prompts
        cache_dir: Where to save the cache
        target_size: Resize images to this size
        prompt_file: Optional single prompts file
        build_lune: Extract Lune features
        build_sol: Extract Sol features
        batch_size: Batch size for extraction
        compile_experts: Whether to torch.compile (not worth it for <50 images)

    Returns:
        DatasetCache ready for training
    """
    from .cache_experts import DatasetCache

    # Load images and prompts
    dataset = DirectoryDataset(
        data_dir,
        repeats=1,  # No repeats for cache building
        target_size=target_size,
        prompt_file=prompt_file,
    )

    images, prompts = dataset.get_images_and_prompts()

    # Build cache
    cache = DatasetCache.build(
        zoo=zoo,
        images=images,
        prompts=prompts,
        name=Path(data_dir).name,
        build_lune=build_lune,
        build_sol=build_sol,
        batch_size=batch_size,
        compile_experts=compile_experts,
    )

    # Save
    cache.save(cache_dir)
    print(f"Cache saved to: {cache_dir}")

    return cache


def create_dataloader(
        cache,
        repeats: int = 100,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader from a cache with repeats.

    Args:
        cache: DatasetCache instance
        repeats: How many times to repeat the dataset per epoch
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: DataLoader workers (0 for Colab)

    Returns:
        DataLoader yielding {'index': tensor} batches
    """
    dataset = CachedDirectoryDataset(cache, repeats=repeats)

    def collate_fn(batch):
        indices = torch.stack([b['index'] for b in batch])
        return {'index': indices}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )