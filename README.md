# TinyFlux

A modular training framework for compact flow-matching diffusion models with dual expert distillation.

## Features

- **Flow Matching**: Rectified flow training with flux-style timestep shifting
- **Dual Expert Distillation**: Learn from Lune (trajectory) and Sol (attention) teachers
- **Feature Caching**: Extract expensive teacher features once, reuse forever
- **Modular Design**: Swap components, customize losses, extend easily
- **HuggingFace Integration**: Upload checkpoints and samples automatically
- **Memory Efficient**: Gradient checkpointing, mixed precision, model offloading

## Installation

```bash
pip install torch torchvision
pip install transformers diffusers accelerate
pip install safetensors huggingface_hub datasets
```

## Quick Start

### 1. Build Feature Cache

```python
from tinyflux.model.zoo import ModelZoo
from tinyflux.trainer.cache_experts import DatasetCache

# Load extraction models
zoo = ModelZoo(device="cuda")
zoo.load_all()

# Build cache (one-time)
cache = DatasetCache.build(
    zoo=zoo,
    images=my_images,      # List of PIL Images
    prompts=my_prompts,    # List of strings
    name="my_dataset",
)
cache.save("my_cache.pt")
zoo.unload_all()
```

### 2. Train

```python
from tinyflux.model.model import TinyFluxConfig, TinyFluxDeep
from tinyflux.trainer.trainer import Trainer, TrainerConfig
from tinyflux.trainer.cache_experts import DatasetCache, MultiSourceCache
from tinyflux.trainer.data import CachedDataset, collate_fn
from torch.utils.data import DataLoader

# Load cache
cache = DatasetCache.load("my_cache.pt")

# Create model
model = TinyFluxDeep(TinyFluxConfig()).to("cuda")

# Setup data
dataset = CachedDataset(cache)
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

multi_cache = MultiSourceCache()
multi_cache.add(cache, dataset_id=0)

# Train
trainer = Trainer(model, TrainerConfig(total_steps=100000))
trainer.setup(loader, multi_cache)
trainer.train()
```

### 3. Generate

```python
from tinyflux.model.loader import load_model
from tinyflux.trainer.sampling import Sampler

model = load_model("path/to/checkpoint.safetensors")
sampler = Sampler(zoo, model)

images = sampler.generate(
    prompts=["a cat", "a dog"],
    num_steps=28,
    guidance_scale=5.0,
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TinyFlux Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Cache Building (One-Time)                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │ Images  │───▶│ ModelZoo│───▶│  Cache  │                 │
│  │ Prompts │    │ (VAE,   │    │(latents,│                 │
│  └─────────┘    │T5,CLIP, │    │ expert  │                 │
│                 │Lune,Sol)│    │features)│                 │
│                 └─────────┘    └─────────┘                 │
│                                                              │
│  Phase 2: Training (Iterative)                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │DataLoader───▶│ Trainer │───▶│Checkpoint                 │
│  └─────────┘    │(TinyFlux│    │  + EMA  │                 │
│       │         │  Deep)  │    └─────────┘                 │
│       │         └─────────┘                                 │
│       │              ▲                                      │
│       └──── Cache ───┘                                      │
│         (Lune + Sol)                                        │
│                                                              │
│  Phase 3: Inference                                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │ Prompt  │───▶│TinyFlux │───▶│  Image  │                 │
│  └─────────┘    │(no cache│    └─────────┘                 │
│                 │ needed) │                                 │
│                 └─────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

## Expert Distillation

TinyFlux learns from two teacher experts:

| Expert | Source | What it Teaches |
|--------|--------|-----------------|
| **Lune** | SD1.5 mid-block | Trajectory guidance, semantic structure |
| **Sol** | SD1.5 attention | Spatial importance, attention patterns |

During training, teacher features are cached and used as targets. At inference, internal predictors replace the teachers - no external models needed.

## Configuration

### Model Config

```python
TinyFluxConfig(
    hidden_size=512,
    num_attention_heads=4,
    attention_head_dim=128,
    num_double_layers=15,
    num_single_layers=25,
    use_lune_expert=True,
    use_sol_prior=True,
)
```

### Training Config

```python
TrainerConfig(
    learning_rate=1e-4,
    total_steps=100000,
    optimizer="adamw",
    lr_scheduler="cosine",
    
    # Expert distillation
    enable_lune=True,
    lune_weight=0.1,
    lune_dropout=0.1,
    
    enable_sol=True,
    sol_weight=0.05,
    sol_dropout=0.1,
    
    # Checkpointing
    save_every_steps=5000,
    hf_repo_id="username/my-model",
)
```

## Package Structure

```
tinyflux/
├── model/
│   ├── model.py      # TinyFluxConfig, TinyFluxDeep
│   ├── zoo.py        # ModelZoo (extraction models)
│   └── loader.py     # Unified checkpoint loading
├── trainer/
│   ├── cache_experts.py  # Feature caching
│   ├── trainer.py        # Trainer, TrainerConfig
│   ├── losses.py         # Loss functions
│   ├── schedules.py      # Timestep/LR schedules
│   ├── ema.py            # EMA tracking
│   └── sampling.py       # Inference
└── util/
    └── predictions.py    # Flow matching math
```

## Documentation

- [Architecture Blueprint](docs/ARCHITECTURE.md) - System design and data flow
- [Technical Reference](docs/TECHNICAL.md) - Component API documentation
- [Applications Guide](docs/APPLICATIONS.md) - Use cases and examples
- [Expert Cache System](docs/EXPERT_CACHE.md) - Lune/Sol distillation details
- [Conversion Checklist](CONVERSION_CHECKLIST.md) - Migration from train_v4.py

## Key Concepts

### Rectified Flow

TinyFlux uses rectified flow matching:

```python
# Interpolation: straight line from noise to data
x_t = (1 - t) * noise + t * data

# Target: velocity = direction of flow
v_target = data - noise

# Model predicts velocity
v_pred = model(x_t, t)
loss = MSE(v_pred, v_target)
```

### Teacher Dropout

Predictors must work without teachers at inference. Force this during training:

```python
# 10% of steps: drop teacher features
if random.random() < lune_dropout:
    lune_features = None  # Predictor must work alone
```

### Timestep Caching

Teacher features vary with timestep. Cache at 10 points, interpolate:

```python
t_buckets = [0.05, 0.15, 0.25, ..., 0.95]  # 10 buckets
# For any t, linearly interpolate between nearest buckets
```

## Memory Requirements

| Phase | Components | VRAM (A100) |
|-------|-----------|-------------|
| Cache Building | VAE + T5 + CLIP + Lune + Sol | ~6 GB |
| Training | Model + Optimizer + Activations | ~8-12 GB |
| Inference | Model + VAE + T5 + CLIP | ~3 GB |

## License

MIT License - see LICENSE file.

## Acknowledgments

- Flow matching: [Lipman et al.](https://arxiv.org/abs/2210.02747)
- Flux architecture: [Black Forest Labs](https://github.com/black-forest-labs/flux)
- Stable Diffusion: [Stability AI](https://stability.ai/)