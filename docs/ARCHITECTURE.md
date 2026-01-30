# TinyFlux Architecture Blueprint

## Overview

TinyFlux is a modular training framework for flow-matching diffusion models with dual expert distillation. It enables training compact models that internalize knowledge from larger teacher models (Lune for trajectory guidance, Sol for attention priors).

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TinyFlux Pipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │   Dataset   │───▶│  Encoding   │───▶│    Cache    │             │
│  │  (Images +  │    │  (VAE, T5,  │    │  (Latents,  │             │
│  │   Prompts)  │    │    CLIP)    │    │  Encodings) │             │
│  └─────────────┘    └─────────────┘    └──────┬──────┘             │
│                                               │                     │
│  ┌─────────────┐    ┌─────────────┐           │                     │
│  │    Lune     │───▶│ LuneCache   │───────────┤                     │
│  │  (SD1.5)    │    │ [N,10,1280] │           │                     │
│  └─────────────┘    └─────────────┘           │                     │
│                                               │                     │
│  ┌─────────────┐    ┌─────────────┐           │                     │
│  │     Sol     │───▶│  SolCache   │───────────┤                     │
│  │  (SD1.5)    │    │ [N,10,4+64] │           │                     │
│  └─────────────┘    └─────────────┘           │                     │
│                                               ▼                     │
│                                        ┌─────────────┐              │
│                                        │ DatasetCache│              │
│                                        │ (Combined)  │              │
│                                        └──────┬──────┘              │
│                                               │                     │
│                                               ▼                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      Training Loop                           │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │   │
│  │  │ Sample t│─▶│ Flow x_t│─▶│ Forward │─▶│  Loss   │         │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘         │   │
│  │       │                          │            │              │   │
│  │       └──────── Cache Lookup ────┘            │              │   │
│  │                (Lune + Sol)                   │              │   │
│  │                                               ▼              │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │   │
│  │  │Backward │─▶│  Step   │─▶│   EMA   │─▶│Checkpoint│        │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Package Structure

```
tinyflux/
├── __init__.py
├── model/
│   ├── __init__.py
│   ├── model.py          # TinyFluxConfig, TinyFluxDeep
│   ├── zoo.py            # ModelZoo (extraction models)
│   └── loader.py         # Unified model loading
├── trainer/
│   ├── __init__.py
│   ├── cache_experts.py  # EncodingCache, LuneFeatureCache, SolFeatureCache, DatasetCache, MultiSourceCache
│   ├── data.py           # CachedDataset, collate_fn
│   ├── trainer.py        # Trainer, TrainerConfig
│   ├── losses.py         # Loss functions
│   ├── schedules.py      # Timestep sampling, LR schedules
│   ├── ema.py            # Exponential moving average
│   └── sampling.py       # Inference sampling
└── util/
    ├── __init__.py
    └── predictions.py    # Flow matching utilities
```

---

## Hierarchy of Operations

### Phase 1: Data Preparation (Once)

```
1. Load raw data (images + prompts)
         │
         ▼
2. Initialize ModelZoo with extraction models
   ├── VAE (FLUX.1)
   ├── T5 (flan-t5-base)
   ├── CLIP (clip-vit-large-patch14)
   ├── Lune (SD1.5-flow UNet)
   └── Sol (SD1.5-flow UNet with attention hooks)
         │
         ▼
3. Build DatasetCache
   ├── EncodingCache.build() → latents, t5_embeds, clip_pooled
   ├── LuneFeatureCache.build() → [N, 10, 1280] mid-block features
   └── SolFeatureCache.build() → [N, 10, 4] stats + [N, 10, 8, 8] spatial
         │
         ▼
4. Save cache to disk (cache.save())
         │
         ▼
5. Unload extraction models (zoo.unload_all())
```

### Phase 2: Training (Iterative)

```
1. Load cached data (DatasetCache.load())
         │
         ▼
2. Create CachedDataset + DataLoader
         │
         ▼
3. Create MultiSourceCache (routes by dataset_id)
         │
         ▼
4. Initialize TinyFluxDeep model
         │
         ▼
5. Initialize Trainer with TrainerConfig
         │
         ▼
6. Setup (loader, cache, optional sampler)
         │
         ▼
7. Train loop:
   ├── For each batch:
   │   ├── Sample timesteps t ~ logit-normal + flux_shift
   │   ├── Compute x_t = (1-t)*noise + t*data
   │   ├── Compute v_target = data - noise
   │   ├── Lookup cached Lune/Sol features for (indices, t)
   │   ├── Forward: v_pred, expert_info = model(x_t, t5, clip, t, lune, sol)
   │   ├── Compute losses (main + lune_distill + sol_distill)
   │   ├── Backward + step
   │   └── EMA update
   ├── Periodic: log, checkpoint, sample, upload
   └── End epoch: cleanup, epoch checkpoint
```

### Phase 3: Inference (Deployment)

```
1. Load trained model (load_model or from_pretrained)
   └── Optionally load EMA weights
         │
         ▼
2. Initialize Sampler with model + zoo (VAE, T5, CLIP only)
         │
         ▼
3. Generate:
   ├── Encode prompt → t5_embed, clip_pooled
   ├── Initialize x ~ N(0, I)
   ├── For t in schedule:
   │   ├── v = model(x, t5, clip, t)  # No teacher features!
   │   └── x = x + v * dt  # Euler step
   ├── Decode latents → image
   └── Return image
```

---

## Data Flow Diagrams

### Training Batch Format

```python
batch = {
    'latents': Tensor[B, 16, 64, 64],      # VAE-encoded images
    't5_embeds': Tensor[B, 128, 768],      # T5 hidden states
    'clip_pooled': Tensor[B, 768],         # CLIP pooled features
    'local_indices': Tensor[B],            # Index within dataset
    'dataset_ids': Tensor[B],              # Which dataset (for multi-source)
    'masks': Optional[Tensor[B, 64, 64]],  # Foreground masks
}
```

### Cache Structure

```python
DatasetCache:
├── encodings: EncodingCache
│   ├── latents: Tensor[N, 16, 64, 64]
│   ├── t5_embeds: Tensor[N, 128, 768]
│   └── clip_pooled: Tensor[N, 768]
├── lune: LuneFeatureCache
│   ├── features: Tensor[N, 10, 1280]  # 10 timestep buckets
│   └── t_buckets: Tensor[10]          # [0.05, 0.15, ..., 0.95]
└── sol: SolFeatureCache
    ├── stats: Tensor[N, 10, 4]        # locality, entropy, clustering, sparsity
    ├── spatial: Tensor[N, 10, 8, 8]   # Spatial importance maps
    └── t_buckets: Tensor[10]
```

### Model Forward Signature

```python
v_pred, expert_info = model(
    hidden_states=x_t,           # [B, H*W, 16]
    encoder_hidden_states=t5,    # [B, 128, 768]
    pooled_projections=clip,     # [B, 768]
    timestep=t,                  # [B]
    img_ids=img_ids,             # [B, H*W, 3]
    lune_features=lune,          # [B, 1280] or None
    sol_stats=sol_stats,         # [B, 4] or None
    sol_spatial=sol_spatial,     # [B, 8, 8] or None
    return_expert_pred=True,
)

# expert_info = {
#     'lune': {'pred': Tensor, 'target': Tensor},
#     'sol': {'stats_pred': Tensor, 'spatial_pred': Tensor, ...}
# }
```

---

## Configuration Hierarchy

### Model Configuration (TinyFluxConfig)

Controls model architecture - set once, saved with checkpoint.

```python
TinyFluxConfig(
    # Core dimensions
    hidden_size=512,
    num_attention_heads=4,
    attention_head_dim=128,  # hidden_size == heads * head_dim
    num_double_layers=15,
    num_single_layers=25,
    
    # Lune predictor
    use_lune_expert=True,
    lune_expert_dim=1280,
    lune_hidden_dim=512,
    
    # Sol prior
    use_sol_prior=True,
    sol_spatial_size=8,
    sol_hidden_dim=256,
)
```

### Training Configuration (TrainerConfig)

Controls training behavior - can adjust between runs.

```python
TrainerConfig(
    # Optimization
    learning_rate=1e-4,
    optimizer="adamw",
    lr_scheduler="cosine",
    min_lr=0.0,
    
    # Flow matching
    shift=3.0,
    logit_normal_sampling=True,
    
    # Expert distillation
    enable_lune=True,
    lune_weight=0.1,
    lune_dropout=0.1,
    
    enable_sol=True,
    sol_weight=0.05,
    sol_dropout=0.1,
    
    # Checkpointing
    save_every_steps=5000,
    hf_repo_id="user/model",
)
```

---

## Key Design Principles

### 1. Precache Everything

Expert features are expensive to compute but static. Extract once, reuse forever.

### 2. Teacher Dropout

Predictors must work without teachers at inference. Force independence via dropout during training.

### 3. Timestep Interpolation

Cache features at 10 buckets [0.05...0.95], interpolate for continuous t.

### 4. Multi-Source Training

Train on multiple datasets simultaneously with proper cache routing via dataset_id.

### 5. Inference Without Teachers

At inference, only the student model runs. Internal predictors replace external teachers.

---

## Memory Budget (A100 40GB)

| Phase | Components | VRAM |
|-------|-----------|------|
| Extraction | VAE + T5 + CLIP + Lune + Sol | ~6 GB |
| Training | Model + Optimizer + Activations | ~8-12 GB |
| Inference | Model + VAE + T5 + CLIP | ~3 GB |

Cache lives in CPU RAM during training, transfers batch-by-batch.