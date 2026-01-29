# TinyFlux-Deep v4.2

Compact diffusion transformer (245M params) with dual expert distillation.

## Architecture

```
TinyFluxDeep (240M)
├── Double blocks (text+image joint attention)
├── Single blocks (image self-attention)
├── Lune predictor (2.5M) - predicts SD1.5 mid-block features
└── Sol prior (0.8M) - predicts attention statistics
```

**Experts (distillation targets, not shipped):**
- **Lune**: SD1.5 UNet mid-block features [1280] - trajectory guidance
- **Sol**: SD1.5 UNet attention statistics [3] + spatial [8,8] - structural prior

## Package Structure

```
tinyflux/
├── model/
│   ├── model.py          # TinyFluxConfig, TinyFluxDeep
│   ├── zoo.py            # ModelZoo (extraction models only)
│   └── loader.py         # Unified model loading
├── trainer/
│   ├── cache_experts.py  # EncodingCache, LuneFeatureCache, SolFeatureCache, DatasetCache, MultiSourceCache
│   ├── data.py           # CachedDataset, MultiCachedDataset, collate_fn
│   ├── trainer.py        # Trainer, TrainerConfig
│   ├── losses.py         # main_loss, lune_loss, sol_loss, min_snr
│   ├── schedules.py      # flux_shift, timestep sampling, warmup
│   ├── ema.py            # EMA
│   └── sampling.py       # CFG sampling, Sampler class
├── util/
│   └── predictions.py    # Rectified flow math
└── test_training.py      # Field test script for Colab
```

## Training Workflow

Training has **two phases**: precaching (one-time) and training loop.

### Phase 1: Precaching

All data is extracted once and saved to disk. Training loop has zero model overhead.

```python
from tinyflux.model.zoo import ModelZoo
from tinyflux.trainer.cache_experts import DatasetCache, MultiSourceCache

# Load all models for extraction
zoo = ModelZoo(device="cuda")
zoo.load_vae()
zoo.load_clip()
zoo.load_t5()
zoo.load_lune()
zoo.load_sol()

# Build complete cache for each dataset
portrait_cache = DatasetCache.build(
    zoo, portrait_images, portrait_prompts,
    name="portraits",
    build_lune=True,
    build_sol=True,
)
portrait_cache.save("cache/portraits/")

imagenet_cache = DatasetCache.build(
    zoo, imagenet_images, imagenet_prompts,
    name="imagenet",
    build_lune=True,
    build_sol=True,
)
imagenet_cache.save("cache/imagenet/")

# Unload everything - not needed during training
zoo.unload("vae")
zoo.unload("clip")
zoo.unload("t5")
zoo.unload("lune")
zoo.unload("sol")
```

**What gets cached per dataset:**
- `encodings.pt`: latents [N, 16, 64, 64], t5 [N, 128, 768], clip [N, 768]
- `lune.pt`: [N, 10, 1280] mid-block features at 10 timestep buckets
- `sol.pt`: stats [N, 10, 3] + spatial [N, 10, 8, 8]

### Phase 2: Training Loop

Only TinyFlux (1GB) on GPU. DataLoader workers move cached tensors.

```python
from tinyflux.trainer.cache_experts import DatasetCache, MultiSourceCache

# Load precached data
portrait_cache = DatasetCache.load("cache/portraits/")
imagenet_cache = DatasetCache.load("cache/imagenet/")

# Combine for multi-source training
cache = MultiSourceCache()
cache.add(portrait_cache, dataset_id=0)
cache.add(imagenet_cache, dataset_id=1)

# Training loop
for batch in loader:
    latents = batch['latents']           # [B, 16, 64, 64]
    t5 = batch['t5_embeds']              # [B, 128, 768]
    clip = batch['clip_pooled']          # [B, 768]
    local_indices = batch['local_indices']
    dataset_ids = batch['dataset_ids']
    
    # Sample timesteps
    t = sigmoid(randn(B))
    t = flux_shift(t, shift=3.0)
    
    # Lookup cached expert features (interpolated by timestep)
    lune_feats = cache.get_lune(local_indices, dataset_ids, t)
    sol_stats, sol_spatial = cache.get_sol(local_indices, dataset_ids, t)
    
    # Rectified flow interpolation
    data = latents.permute(0, 2, 3, 1).reshape(B, H*W, C)
    noise = randn_like(data)
    t_exp = t.view(B, 1, 1)
    x_t = (1 - t_exp) * noise + t_exp * data
    v_target = data - noise
    
    # Forward
    v_pred, expert_info = model(
        hidden_states=x_t,
        encoder_hidden_states=t5,
        pooled_projections=clip,
        timestep=t,
        img_ids=img_ids,
        lune_features=lune_feats,
        sol_stats=sol_stats,
        sol_spatial=sol_spatial,
        return_expert_pred=True,
    )
    
    # Losses
    main_loss = mse(v_pred, v_target)
    lune_loss = cosine(expert_info['lune_pred'], lune_feats)
    sol_loss = mse(expert_info['sol_stats_pred'], sol_stats)
    
    total = main_loss + 0.1*lune_loss + 0.05*sol_loss
```

### Expert Feature Details

**Lune** (trajectory guidance):
- SD1.5 UNet mid-block output after self-attention
- 1280-dim feature capturing "what the image should look like at this timestep"
- Cached at 10 timestep buckets [0.05, 0.15, ..., 0.95], interpolated at runtime

**Sol** (attention prior):
- Extracted from SD1.5 UNet self-attention layers via `SolAttnProcessor`
- **stats [3]**: locality, entropy, clustering
  - locality: do queries attend nearby or far?
  - entropy: focused vs diffuse attention
  - clustering: 1 - entropy
- **spatial [8,8]**: where attention aggregates across layers
- Used to bias TinyFlux's weak 4-head attention

**Note:** Prior versions used geometric heuristics (formulas from timestep) instead of real extraction. This was wrong - Sol distillation was training against noise.

## Inference Workflow

```python
zoo = ModelZoo(device="cuda")
zoo.load_vae()
zoo.load_clip()
zoo.load_t5()
zoo.load_tinyflux("checkpoint_ema.safetensors")

# Encode prompt
t5_embed = zoo.encode_t5(prompt)
clip_pooled = zoo.encode_clip(prompt)

# Sample (Euler, 28 steps, CFG)
x = randn(1, 64*64, 16)
for t_curr, t_next in timestep_pairs:
    v = model(x, t5_embed, clip_pooled, t_curr)
    x = x + v * (t_next - t_curr)

# Decode
latents = x.reshape(1, 64, 64, 16).permute(0, 3, 1, 2)
image = zoo.vae.decode(latents / vae_scale)
```

No experts needed at inference - Lune/Sol predictors learned the behavior.

## Key Files

| File | Purpose |
|------|---------|
| `model.py` | TinyFluxDeep architecture, attention blocks, RoPE |
| `zoo.py` | Load VAE, CLIP, T5, Lune, Sol for extraction; SolAttnProcessor |
| `loader.py` | Unified model loading from HF/file/directory |
| `cache_experts.py` | EncodingCache, LuneFeatureCache, SolFeatureCache, DatasetCache, MultiSourceCache |
| `data.py` | CachedDataset, MultiCachedDataset, collate_fn (bridges cache to DataLoader) |
| `trainer.py` | Training loop, optimizer, EMA, checkpointing, HF upload |
| `losses.py` | `compute_main_loss`, `compute_lune_loss`, `compute_sol_loss`, `min_snr_weight` |
| `schedules.py` | `flux_shift`, `sample_timesteps`, warmup functions |
| `ema.py` | EMA with save/load/copy_to |
| `sampling.py` | CFG sampling with flux shift |
| `predictions.py` | Rectified flow math |
| `test_training.py` | Field test for Colab - verifies full pipeline |

## Batch Format

`collate_fn` from `data.py` produces:
```python
{
    'latents': [B, 16, 64, 64],      # VAE-encoded images
    't5_embeds': [B, 128, 768],      # T5 hidden states
    'clip_pooled': [B, 768],         # CLIP pooled
    'local_indices': [B],            # Index into cache (for expert lookup)
    'dataset_ids': [B],              # Which dataset (for multi-source routing)
    'masks': None,                   # Not currently cached
}
```

## Rectified Flow

```
Forward:  x_t = (1-t)*noise + t*data
Velocity: v = data - noise
Training: predict v from x_t
Sampling: x_{t+dt} = x_t + v*dt
```

t=0 is pure noise, t=1 is data. Flux shift biases sampling toward higher t.

## Checkpoints

Training saves:
- `step_N.pt` - full checkpoint (model, optimizer, scheduler, EMA)
- `step_N_ema.safetensors` - EMA weights only (for inference)
- `best.pt` / `best_ema.safetensors` - best validation loss

## Config Defaults

```python
TrainerConfig(
    learning_rate=1e-4,
    total_steps=100000,
    gradient_accumulation=1,
    ema_decay=0.9999,
    shift=3.0,                    # Flux shift
    use_snr_weighting=True,
    
    # Expert distillation
    enable_lune=True,
    lune_weight=0.1,
    lune_warmup_steps=1000,
    lune_dropout=0.1,
    enable_sol=True,
    sol_weight=0.05,
    sol_warmup_steps=2000,
    
    # Checkpointing
    save_every_steps=5000,        # Step-based checkpoints
    keep_last_n_steps=3,          # Rolling cleanup
    save_every_epochs=1,          # Epoch-based checkpoints
    keep_last_n_epochs=3,
    
    # Logging
    tensorboard_dir="logs",       # TensorBoard output
    log_every=100,
    
    # HuggingFace upload
    hf_repo_id="user/model",      # Optional HF repo
    upload_every_steps=0,         # 0 to disable
    upload_every_epochs=1,        # Upload after each epoch
)
```

## Loading Models

```python
from tinyflux.model.loader import load_model, load_checkpoint, from_pretrained

# From HuggingFace
model = from_pretrained("AbstractPhil/tinyflux-deep", use_ema=True)

# From local file
model = load_model("/path/to/model.safetensors")

# From directory
model = load_model("/path/to/checkpoint/")

# Resume training
ckpt = load_checkpoint("checkpoints/step_5000.pt")
model = ckpt['model']
step = ckpt['step']
```

## Field Testing

Run on Colab to verify full pipeline:

```bash
!git clone https://github.com/AbstractPhil/tinyflux
%cd tinyflux
!pip install -e .
!python test_training.py
```

Test script:
1. Generates random images + prompts
2. Builds all caches (VAE, T5, CLIP, Lune, Sol)
3. Trains TinyFlux for 50 steps
4. Verifies losses decrease

Passes if final loss < 1.0.