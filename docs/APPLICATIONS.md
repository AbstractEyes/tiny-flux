# TinyFlux Application Guide

This guide covers common use cases for the TinyFlux training framework.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training from Scratch](#training-from-scratch)
3. [Fine-tuning an Existing Model](#fine-tuning-an-existing-model)
4. [Training with Custom Datasets](#training-with-custom-datasets)
5. [Multi-Dataset Training](#multi-dataset-training)
6. [Inference and Generation](#inference-and-generation)
7. [Colab/Cloud Training](#colabcloud-training)
8. [Memory Optimization](#memory-optimization)
9. [Monitoring and Debugging](#monitoring-and-debugging)

---

## Quick Start

Minimal example to verify everything works:

```python
import torch
from tinyflux.model.model import TinyFluxConfig, TinyFluxDeep
from tinyflux.model.zoo import ModelZoo
from tinyflux.trainer.sampling import Sampler

# Load a pretrained model
from tinyflux.model.loader import load_model
model = load_model("AbstractPhil/tiny-flux-deep")

# Setup inference
zoo = ModelZoo(device="cuda")
zoo.load_vae()
zoo.load_t5()
zoo.load_clip()

sampler = Sampler(zoo, model)
images = sampler.generate(["a photo of a cat"])
sampler.save_samples(images, ["a photo of a cat"], "./outputs")
```

---

## Training from Scratch

### Step 1: Prepare Data

```python
from datasets import load_dataset
from PIL import Image

# Load your dataset
ds = load_dataset("your/dataset", split="train")
images = [sample["image"] for sample in ds]
prompts = [sample["caption"] for sample in ds]

# Ensure images are PIL and 512x512
images = [img.convert("RGB").resize((512, 512)) for img in images]
```

### Step 2: Build Feature Cache

```python
from tinyflux.model.zoo import ModelZoo
from tinyflux.trainer.cache_experts import DatasetCache

zoo = ModelZoo(device="cuda")

# Build comprehensive cache (one-time, ~10 min for 10k samples)
cache = DatasetCache.build(
    zoo, 
    images, 
    prompts,
    name="my_dataset",
    extract_lune=True,  # Extract Lune teacher features
    extract_sol=True,   # Extract Sol teacher features
)

# Save for reuse
cache.save("my_dataset_cache.pt")
```

### Step 3: Setup Model and Trainer

```python
from tinyflux.model.model import TinyFluxConfig, TinyFluxDeep
from tinyflux.trainer.trainer import Trainer, TrainerConfig
from tinyflux.trainer.data import CachedDataset, collate_fn
from tinyflux.trainer.cache_experts import MultiSourceCache
from tinyflux.trainer.sampling import Sampler
from torch.utils.data import DataLoader

# Create model
config = TinyFluxConfig(
    hidden_size=1536,
    num_double_layers=4,
    num_single_layers=8,
    use_lune_expert=True,
    use_sol_prior=True,
)
model = TinyFluxDeep(config)

# Setup data
dataset = CachedDataset(cache.encodings)
loader = DataLoader(
    dataset, 
    batch_size=8, 
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    drop_last=True,
)

# Setup trainer
trainer_config = TrainerConfig(
    total_steps=100000,
    learning_rate=3e-4,
    gradient_accumulation=4,
    
    # Expert distillation
    enable_lune=True,
    lune_weight=0.1,
    lune_warmup_steps=1000,
    
    enable_sol=True,
    sol_weight=0.05,
    sol_warmup_steps=2000,
    
    # Checkpointing
    checkpoint_dir="./checkpoints",
    save_every_steps=1000,
    
    # Sampling during training
    sample_every=500,
    sample_prompts=["a cat", "a dog", "a landscape"],
    
    # HuggingFace upload
    hf_repo_id="your-username/your-model",
    upload_every_steps=5000,
)

trainer = Trainer(model, trainer_config)

# Create multi-source cache and sampler
multi_cache = MultiSourceCache().add(cache, dataset_id=0)
sampler = Sampler(zoo, model)

trainer.setup(loader, multi_cache, sampler)
```

### Step 4: Train

```python
# Train for N epochs
trainer.train(epochs=10)

# Or train until total_steps
# trainer.train()  # Runs until config.total_steps
```

---

## Fine-tuning an Existing Model

```python
from tinyflux.model.loader import load_model
from tinyflux.trainer.trainer import Trainer, TrainerConfig

# Load pretrained
model = load_model("AbstractPhil/tiny-flux-deep")

# Lower LR for fine-tuning
config = TrainerConfig(
    total_steps=10000,
    learning_rate=1e-5,      # Lower than training from scratch
    warmup_steps=100,        # Short warmup
    
    # Can disable expert distillation if not needed
    enable_lune=False,
    enable_sol=False,
)

trainer = Trainer(model, config)
trainer.setup(loader)  # No cache needed if experts disabled
trainer.train(epochs=3)
```

---

## Training with Custom Datasets

### HuggingFace Dataset with Latents

If your dataset already contains precomputed latents:

```python
from datasets import load_dataset

ds = load_dataset("your/latent-dataset", split="train")

# Expected columns: latent, t5_embed, clip_pooled, prompt
class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, dtype=torch.bfloat16):
        self.ds = hf_dataset
        self.dtype = dtype
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "latents": torch.tensor(item["latent"]).to(self.dtype),
            "t5_embeds": torch.tensor(item["t5_embed"]).to(self.dtype),
            "clip_pooled": torch.tensor(item["clip_pooled"]).to(self.dtype),
            "local_indices": idx,
            "dataset_ids": 0,
        }
```

### Local Image Folder

```python
from pathlib import Path
from PIL import Image

image_dir = Path("./my_images")
images = []
prompts = []

for img_path in image_dir.glob("*.jpg"):
    images.append(Image.open(img_path).convert("RGB").resize((512, 512)))
    # Assume matching .txt file with caption
    txt_path = img_path.with_suffix(".txt")
    prompts.append(txt_path.read_text().strip() if txt_path.exists() else "")

# Then build cache as normal
cache = DatasetCache.build(zoo, images, prompts, name="local_images")
```

### Dataset with Masks (Foreground/Background)

```python
class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, masks, dtype=torch.bfloat16):
        self.encodings = encodings
        self.masks = masks  # [N, 64, 64] tensor
        self.dtype = dtype
    
    def __getitem__(self, idx):
        enc = self.encodings[idx]
        return {
            "latents": enc["latents"].to(self.dtype),
            "t5_embeds": enc["t5_embeds"].to(self.dtype),
            "clip_pooled": enc["clip_pooled"].to(self.dtype),
            "masks": self.masks[idx].to(self.dtype),  # Added mask
            "local_indices": idx,
            "dataset_ids": 0,
        }
```

Then enable masked loss:

```python
config = TrainerConfig(
    # ... other config ...
    use_spatial_weighting=True,  # Use Sol spatial maps for weighting
)
```

---

## Multi-Dataset Training

Combine multiple datasets with different characteristics:

```python
from tinyflux.trainer.cache_experts import DatasetCache, MultiSourceCache

# Build caches for each dataset
portrait_cache = DatasetCache.build(zoo, portrait_images, portrait_prompts, name="portraits")
landscape_cache = DatasetCache.build(zoo, landscape_images, landscape_prompts, name="landscapes")
product_cache = DatasetCache.build(zoo, product_images, product_prompts, name="products")

# Save individually
portrait_cache.save("portrait_cache.pt")
landscape_cache.save("landscape_cache.pt")
product_cache.save("product_cache.pt")

# Combine for training
multi_cache = MultiSourceCache()
multi_cache.add(portrait_cache, dataset_id=0)
multi_cache.add(landscape_cache, dataset_id=1)
multi_cache.add(product_cache, dataset_id=2)

# Create combined dataset
from torch.utils.data import ConcatDataset

class IndexedDataset(torch.utils.data.Dataset):
    """Wrapper that adds dataset_id to samples."""
    def __init__(self, cache, dataset_id):
        self.cache = cache
        self.dataset_id = dataset_id
    
    def __len__(self):
        return len(self.cache.encodings)
    
    def __getitem__(self, idx):
        enc = self.cache.encodings[idx]
        return {
            "latents": enc["latents"],
            "t5_embeds": enc["t5_embeds"],
            "clip_pooled": enc["clip_pooled"],
            "local_indices": idx,
            "dataset_ids": self.dataset_id,
        }

combined = ConcatDataset([
    IndexedDataset(portrait_cache, 0),
    IndexedDataset(landscape_cache, 1),
    IndexedDataset(product_cache, 2),
])

loader = DataLoader(combined, batch_size=8, shuffle=True, collate_fn=collate_fn)
trainer.setup(loader, multi_cache, sampler)
```

---

## Inference and Generation

### Basic Generation

```python
from tinyflux.trainer.sampling import Sampler

sampler = Sampler(zoo, model)

images = sampler.generate(
    prompts=["a majestic lion", "a serene lake at sunset"],
    num_steps=28,
    guidance_scale=5.0,
    seed=42,
)

sampler.save_samples(images, prompts, "./outputs")
```

### Batch Generation

```python
# Generate many images efficiently
all_prompts = ["prompt1", "prompt2", ..., "prompt100"]
batch_size = 4

for i in range(0, len(all_prompts), batch_size):
    batch_prompts = all_prompts[i:i+batch_size]
    images = sampler.generate(batch_prompts, num_steps=28)
    sampler.save_samples(images, batch_prompts, "./outputs", step=i)
```

### Using EMA Weights

```python
# During training, use EMA for sampling
sampler = Sampler(zoo, model, ema=trainer.ema)
images = sampler.generate(prompts, use_ema=True)

# Or manually apply EMA
trainer.ema.apply_shadow(model)
images = sampler.generate(prompts, use_ema=False)  # Model already has EMA weights
trainer.ema.restore(model)
```

### Different Resolutions

```python
# Standard 512x512 (64x64 latents)
images = sampler.generate(prompts, height=64, width=64)

# Landscape 768x512
images = sampler.generate(prompts, height=64, width=96)

# Portrait 512x768
images = sampler.generate(prompts, height=96, width=64)
```

---

## Colab/Cloud Training

### Colab Setup

```python
# Install dependencies
!pip install torch torchvision transformers diffusers safetensors datasets accelerate

# Clone/upload tinyflux package
!git clone https://github.com/your/tinyflux.git
import sys
sys.path.insert(0, "./tinyflux")

# Login to HuggingFace for uploads
from huggingface_hub import login
login(token="your_token")
```

### Resume from Checkpoint

```python
# Download checkpoint from HF
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="AbstractPhil/tiny-flux-deep",
    filename="checkpoints/step_50000.pt"
)
weights_path = hf_hub_download(
    repo_id="AbstractPhil/tiny-flux-deep",
    filename="checkpoints/step_50000.safetensors"
)

# Load and resume
model = load_model(weights_path, config=config)
trainer = Trainer(model, trainer_config)
trainer.load_checkpoint(ckpt_path)
trainer.setup(loader, cache, sampler)
trainer.train()  # Continues from step 50000
```

### Backup Training to HuggingFace

See the `hf-training-backup` skill for automated backups during training.

---

## Memory Optimization

### For Limited VRAM (< 16GB)

```python
config = TrainerConfig(
    # Reduce batch size
    gradient_accumulation=8,  # Effective batch = actual_batch × accum
    
    # Enable gradient checkpointing
    gradient_checkpointing=True,
    
    # Use 8-bit optimizer
    optimizer="adamw_8bit",
    
    # Reduce precision
    dtype=torch.float16,
)

# Use smaller model
model_config = TinyFluxConfig(
    hidden_size=768,         # Smaller hidden
    num_double_layers=2,     # Fewer layers
    num_single_layers=4,
)
```

### Cache Building with Limited VRAM

```python
# Build caches sequentially, not all at once
from tinyflux.trainer.cache_experts import EncodingCache, LuneFeatureCache, SolFeatureCache

# Step 1: Encodings (needs VAE + T5 + CLIP)
zoo.load_vae()
zoo.load_t5()
zoo.load_clip()
encodings = EncodingCache.build(zoo, images, prompts)

# Unload encoders
zoo.unload("vae")
zoo.unload("t5")
zoo.unload("clip")
torch.cuda.empty_cache()

# Step 2: Lune features
zoo.load_lune()
lune = LuneFeatureCache.build(zoo, prompts, batch_timesteps=False)  # Lower VRAM
zoo.unload("lune")
torch.cuda.empty_cache()

# Step 3: Sol features
zoo.load_sol()
sol = SolFeatureCache.build(zoo, prompts, batch_timesteps=True)  # Sol is lighter
zoo.unload("sol")

# Combine
cache = DatasetCache(encodings, lune, sol, name="my_data")
```

---

## Monitoring and Debugging

### TensorBoard

```bash
tensorboard --logdir ./logs
```

Metrics logged:
- `train/loss` - Total loss
- `train/main_loss` - Velocity prediction loss
- `train/lune_loss` - Lune distillation loss
- `train/sol_loss` - Sol distillation loss
- `train/lune_weight` - Current Lune loss weight (with warmup)
- `train/sol_weight` - Current Sol loss weight
- `train/lr` - Learning rate
- `train/grad_norm` - Gradient norm

### Debug Mode

```python
# Verbose training output
import logging
logging.basicConfig(level=logging.DEBUG)

# Check shapes during forward pass
model.eval()
with torch.no_grad():
    result = model(
        hidden_states=torch.randn(2, 4096, 16).cuda(),
        encoder_hidden_states=torch.randn(2, 128, 768).cuda(),
        pooled_projections=torch.randn(2, 768).cuda(),
        timestep=torch.tensor([0.5, 0.5]).cuda(),
        img_ids=TinyFluxDeep.create_img_ids(2, 64, 64, "cuda"),
        return_expert_pred=True,
    )
    v_pred, expert_info = result
    print(f"v_pred: {v_pred.shape}")
    print(f"lune_pred: {expert_info.get('lune_pred', 'None')}")
    print(f"sol_stats_pred: {expert_info.get('sol_stats_pred', 'None')}")
```

### Validate Cache

```python
# Check cache integrity
cache = DatasetCache.load("my_cache.pt")

print(f"Encodings: {len(cache.encodings)} samples")
print(f"  Latents: {cache.encodings.latents.shape}")
print(f"  T5: {cache.encodings.t5_embeds.shape}")
print(f"  CLIP: {cache.encodings.clip_pooled.shape}")

if cache.lune is not None:
    print(f"Lune: {cache.lune.features.shape}")
    print(f"  T buckets: {cache.lune.t_buckets}")

if cache.sol is not None:
    print(f"Sol stats: {cache.sol.stats.shape}")
    print(f"Sol spatial: {cache.sol.spatial.shape}")
```