# Expert Cache System Documentation

This document details the Lune and Sol expert distillation system used in TinyFlux training.

---

## Overview

TinyFlux uses **dual expert distillation** to guide training:

| Expert | Purpose | Source | Feature Shape |
|--------|---------|--------|---------------|
| **Lune** | Trajectory guidance | SD1.5-flow UNet mid-block | `[B, 1280]` |
| **Sol** | Attention structure | SD1.5-flow attention patterns | `[B, 4]` stats + `[B, 8, 8]` spatial |

Both experts provide **teacher signals** during training. The student model learns to:
1. Predict the main velocity target (diffusion objective)
2. Predict what the Lune teacher would output (trajectory)
3. Predict what the Sol teacher would output (attention structure)

At inference time, **no teachers are needed** - the student's internal `lune_predictor` and `sol_prior` modules run standalone.

---

## Why Precache?

Running teacher models during training is expensive:

| Approach | VRAM | Speed |
|----------|------|-------|
| Live extraction | +4-6 GB | ~2x slower |
| Precached features | +0 GB | Full speed |

By extracting features at **10 timestep buckets** and interpolating during training, we get:
- Zero additional VRAM during training
- Full training speed
- Accurate feature approximation (linear interpolation is sufficient)

---

## Timestep Buckets

Features are extracted at 10 evenly-spaced timesteps:

```python
t_buckets = torch.linspace(0.05, 0.95, 10)
# [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
```

During training, when we sample timestep `t=0.42`, we:
1. Find neighboring buckets: `t_low=0.35`, `t_high=0.45`
2. Compute interpolation weight: `alpha = (0.42 - 0.35) / 0.10 = 0.7`
3. Interpolate: `features = (1 - alpha) * f_low + alpha * f_high`

This gives continuous feature values for any timestep.

---

## Lune Expert (Trajectory Guidance)

### What It Captures

Lune extracts the **mid-block features** from an SD1.5 UNet that has been fine-tuned for flow matching. These features encode:

- Global image structure
- Denoising trajectory information
- Semantic content understanding

### Feature Details

| Property | Value |
|----------|-------|
| Source model | `stable-diffusion-v1-5/stable-diffusion-v1-5` UNet |
| Fine-tuned weights | `AbstractPhil/tinyflux-experts/sd15-flow-lune-unet.safetensors` |
| Extraction point | `unet.mid_block` output |
| Pooling | Spatial mean: `[B, 1280, H, W]` → `[B, 1280]` |
| Cached shape | `[N, 10, 1280]` (N samples × 10 timesteps × 1280 dim) |

### Extraction Process

```python
# Hook to capture mid-block features
def hook_fn(module, inp, out):
    mid_features[0] = out.mean(dim=[2, 3])  # Spatial pooling

unet.mid_block.register_forward_hook(hook_fn)

# For each sample and timestep bucket:
# 1. Encode prompt with CLIP
# 2. Generate random latent
# 3. Run UNet forward pass
# 4. Capture hooked features
```

### Training Usage

During training:
1. Look up precached features for batch indices and current timestep
2. Apply **teacher dropout** (10% of samples get `None`)
3. Model outputs `lune_pred` from internal `lune_predictor`
4. Compute loss: `cosine_similarity(lune_pred, lune_features)`

```python
# In trainer
lune_features = cache.get_lune(local_indices, dataset_ids, timesteps)
if random.random() < lune_dropout:
    lune_features = None  # Force predictor independence

# Model forward
result = model(..., lune_features=lune_features, return_expert_pred=True)
v_pred, expert_info = result

# Loss (only when features present)
if lune_features is not None:
    lune_loss = compute_lune_loss(expert_info['lune_pred'], lune_features)
```

### Loss Modes

| Mode | Formula | Use Case |
|------|---------|----------|
| `cosine` | `1 - cos_sim(pred, target)` | Default, normalized |
| `mse` | `mean((pred - target)²)` | Raw L2 |
| `huber` | Huber with δ=0.1 | Robust to outliers |

---

## Sol Expert (Attention Structure)

### What It Captures

Sol extracts **attention statistics** that describe how the model attends to different parts of the image:

| Statistic | Meaning | Range |
|-----------|---------|-------|
| `locality` | How focused attention is | 0 (diffuse) to 1 (focused) |
| `entropy` | Uncertainty in attention | 0 (certain) to 1 (uniform) |
| `clustering` | Grouping of attention | 0 to 1 |
| `sparsity` | How sparse attention is | 0 (dense) to 1 (sparse) |

Plus a **spatial importance map** `[8, 8]` showing where attention concentrates.

### Feature Details

| Property | Value |
|----------|-------|
| Stats shape | `[N, 10, 4]` (N samples × 10 timesteps × 4 stats) |
| Spatial shape | `[N, 10, 8, 8]` (N samples × 10 timesteps × 8×8 map) |
| Generation | Geometric heuristics (no heavy teacher needed) |

### Geometric Heuristic Generation

Sol features are generated using **geometric heuristics** rather than a heavy teacher model:

```python
# Stats vary with timestep
t_vals = t_buckets.float()

locality = 1 - t_vals      # Early = focused, late = diffuse
entropy = t_vals           # Early = certain, late = uncertain
clustering = 0.5 - 0.3 * (t_vals - 0.5).abs()  # Peak at middle
sparsity = 1 - t_vals      # Early = sparse, late = dense

# Spatial: center-biased, strength varies with timestep
y, x = torch.meshgrid(torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 8))
center_dist = torch.sqrt(x² + y²)
t_weight = (1 - t_vals).view(-1, 1, 1)
spatial = 1 - center_dist.unsqueeze(0) * t_weight
spatial = spatial / spatial.sum(dim=[-2,-1], keepdim=True)  # Normalize
```

This captures the intuition that:
- Early timesteps (low t): attention is focused, sparse, certain
- Late timesteps (high t): attention is diffuse, dense, uncertain
- Center bias: important content tends to be centered

### Training Usage

```python
# Get features
sol_stats, sol_spatial = cache.get_sol(local_indices, dataset_ids, timesteps)
if random.random() < sol_dropout:
    sol_stats, sol_spatial = None, None

# Model forward
result = model(..., sol_stats=sol_stats, sol_spatial=sol_spatial, return_expert_pred=True)

# Loss
if sol_stats is not None:
    sol_loss = compute_sol_loss(
        expert_info['sol_stats_pred'], expert_info['sol_spatial_pred'],
        sol_stats, sol_spatial
    )
```

### Optional: Spatial Loss Weighting

Sol's spatial map can also weight the **main velocity loss**:

```python
config = TrainerConfig(
    use_spatial_weighting=True,  # Enable
)

# In loss computation:
# spatial_weights [B, 8, 8] → upsample to [B, 64, 64] → weight main loss
```

This focuses the model on "important" regions as identified by Sol.

---

## Cache File Format

### DatasetCache Save Format

```python
{
    "name": str,
    "encodings": {
        "latents": Tensor[N, 16, 64, 64],
        "t5_embeds": Tensor[N, 128, 768],
        "clip_pooled": Tensor[N, 768],
    },
    "lune": {  # Optional
        "features": Tensor[N, 10, 1280],
        "t_buckets": Tensor[10],
    },
    "sol": {  # Optional
        "stats": Tensor[N, 10, 4],
        "spatial": Tensor[N, 10, 8, 8],
        "t_buckets": Tensor[10],
    },
}
```

### Memory Estimates

For a dataset of N=10,000 samples:

| Component | Shape | Size (fp16) |
|-----------|-------|-------------|
| Latents | `[10000, 16, 64, 64]` | 1.31 GB |
| T5 embeds | `[10000, 128, 768]` | 1.97 GB |
| CLIP pooled | `[10000, 768]` | 15 MB |
| Lune features | `[10000, 10, 1280]` | 256 MB |
| Sol stats | `[10000, 10, 4]` | 800 KB |
| Sol spatial | `[10000, 10, 8, 8]` | 12.8 MB |
| **Total** | | **~3.5 GB** |

---

## Multi-Source Cache

When training on multiple datasets:

```python
multi_cache = MultiSourceCache()
multi_cache.add(portrait_cache, dataset_id=0)
multi_cache.add(landscape_cache, dataset_id=1)
multi_cache.add(product_cache, dataset_id=2)

# During training, batch contains:
# - local_indices: [B] index within each sample's original dataset
# - dataset_ids: [B] which dataset (0, 1, or 2)

# Lookup routes to correct cache:
lune_features = multi_cache.get_lune(local_indices, dataset_ids, timesteps)
```

The router handles:
- Mapping indices to correct cache
- Gathering features across datasets
- Returning `None` if cache doesn't have expert features

---

## Interpolation Details

### Linear Interpolation

```python
def get_features(self, indices, timesteps):
    # Clamp to valid range
    t_clamped = timesteps.clamp(self.t_min, self.t_max)
    
    # Find bucket indices
    t_idx_float = (t_clamped - self.t_min) / self.t_step
    t_idx_low = t_idx_float.long().clamp(0, self.n_buckets - 2)
    t_idx_high = (t_idx_low + 1).clamp(0, self.n_buckets - 1)
    
    # Interpolation weight
    alpha = (t_idx_float - t_idx_low.float()).unsqueeze(-1)
    
    # Gather and interpolate
    f_low = self.features[indices, t_idx_low]
    f_high = self.features[indices, t_idx_high]
    return (1 - alpha) * f_low + alpha * f_high
```

### Edge Cases

| Timestep | Behavior |
|----------|----------|
| `t < 0.05` | Clamped to 0.05, returns bucket 0 features |
| `t > 0.95` | Clamped to 0.95, returns bucket 9 features |
| `t = 0.15` | Exact bucket match, returns bucket 1 features |
| `t = 0.42` | Interpolates between buckets 3 and 4 |

---

## Teacher Dropout

Both Lune and Sol support **teacher dropout** during training:

```python
# 10% of samples get no teacher signal
if random.random() < lune_dropout:
    lune_features = None

if random.random() < sol_dropout:
    sol_stats, sol_spatial = None, None
```

**Why dropout?**

At inference time, teachers are not available. The model's internal predictors must work standalone. Teacher dropout forces the predictors to learn independently rather than just copying teacher signals.

| With Teacher | Without Teacher |
|--------------|-----------------|
| Predictor gets direct supervision | Predictor must infer from context |
| Can "cheat" by passing through | Must generalize |

Recommended dropout: 10% for both Lune and Sol.

---

## Building Caches Efficiently

### Standard Build (High VRAM)

```python
# All models loaded at once
cache = DatasetCache.build(zoo, images, prompts, name="my_data")
```

### Sequential Build (Low VRAM)

```python
# Step 1: Encodings
zoo.load_vae()
zoo.load_t5()
zoo.load_clip()
encodings = EncodingCache.build(zoo, images, prompts)

zoo.offload()
torch.cuda.empty_cache()

# Step 2: Lune (heavy)
zoo.load_lune()
lune = LuneFeatureCache.build(zoo, prompts, batch_timesteps=False)
zoo.unload("lune")
torch.cuda.empty_cache()

# Step 3: Sol (light)
zoo.load_sol()
sol = SolFeatureCache.build(zoo, prompts)
zoo.unload("sol")

# Combine
cache = DatasetCache(encodings, lune, sol, name="my_data")
cache.save("cache.pt")
```

### Batched Timestep Extraction

| Setting | VRAM | Speed |
|---------|------|-------|
| `batch_timesteps=True` | Higher | 10x faster |
| `batch_timesteps=False` | Lower | Slower but fits in 16GB |

```python
# High VRAM (A100): batch all timesteps
lune = LuneFeatureCache.build(zoo, prompts, batch_timesteps=True)

# Low VRAM (T4/V100): one timestep at a time
lune = LuneFeatureCache.build(zoo, prompts, batch_timesteps=False)
```

---

## Alignment Verification

Verify your cache matches the original train_v4.py:

```python
# Original train_v4.py values
LUNE_DIM = 1280
SOL_SPATIAL_SIZE = 8
EXPERT_T_BUCKETS = torch.linspace(0.05, 0.95, 10)

# Check your cache
cache = DatasetCache.load("my_cache.pt")

assert cache.lune.features.shape[-1] == 1280, "Lune dim mismatch"
assert cache.sol.spatial.shape[-2:] == (8, 8), "Sol spatial mismatch"
assert torch.allclose(cache.lune.t_buckets, EXPERT_T_BUCKETS), "T bucket mismatch"
assert torch.allclose(cache.sol.t_buckets, EXPERT_T_BUCKETS), "T bucket mismatch"

print("✓ Cache aligned with train_v4.py")
```

### Feature Value Ranges

| Feature | Expected Range | Notes |
|---------|----------------|-------|
| Lune features | ~[-5, 5] | Varies by sample |
| Sol locality | [0, 1] | Decreases with t |
| Sol entropy | [0, 1] | Increases with t |
| Sol clustering | [0.2, 0.5] | Peaks at middle |
| Sol sparsity | [0, 1] | Decreases with t |
| Sol spatial | [0, 1] | Sums to 1 per sample |