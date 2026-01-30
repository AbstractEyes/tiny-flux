# TinyFlux Technical Documentation

## Table of Contents

1. [Model Components](#model-components)
2. [Trainer Components](#trainer-components)
3. [Utility Components](#utility-components)

---

## Model Components

### `model/model.py` - TinyFluxDeep

The student model that learns from teacher distillation.

#### TinyFluxConfig

```python
@dataclass
class TinyFluxConfig:
    # Core architecture
    hidden_size: int = 1536           # Transformer hidden dimension
    num_attention_heads: int = 24     # Must divide hidden_size evenly
    attention_head_dim: int = 64      # hidden_size = heads × head_dim
    num_double_layers: int = 4        # Joint attention layers
    num_single_layers: int = 8        # Self-attention layers
    mlp_ratio: float = 4.0            # MLP hidden = hidden_size × mlp_ratio
    
    # Expert modules
    use_lune_expert: bool = True      # Enable Lune predictor
    lune_expert_dim: int = 1280       # SD1.5 mid-block dimension
    lune_hidden_dim: int = 512        # Predictor hidden size
    freeze_lune: bool = False         # Freeze predictor weights
    
    use_sol_prior: bool = True        # Enable Sol prior
    sol_spatial_size: int = 8         # Spatial map resolution (8×8)
    sol_hidden_dim: int = 256         # Prior hidden size
    sol_geometric_weight: float = 0.7 # Geometric vs learned balance
    freeze_sol: bool = False          # Freeze prior weights
    
    # Text encoding
    t5_layers: Tuple[int] = (0, 4, 8, 12)  # T5 layers to use
    t5_hidden_size: int = 768              # T5-base dimension
    use_cross_attention_sharing: bool = True
    
    # Position encoding
    axes_dims_rope: Tuple[int] = (16, 24, 24)  # RoPE dimensions
    theta: int = 10000                          # RoPE base frequency
```

#### TinyFluxDeep

```python
class TinyFluxDeep(nn.Module):
    def __init__(self, config: TinyFluxConfig): ...
    
    def forward(
        self,
        hidden_states: torch.Tensor,      # [B, N, C] noised latents
        encoder_hidden_states: torch.Tensor,  # [B, L, 768] T5 embeddings
        pooled_projections: torch.Tensor,     # [B, 768] CLIP pooled
        timestep: torch.Tensor,               # [B] timesteps in [0, 1]
        img_ids: torch.Tensor,                # [B, N, 3] position IDs
        lune_features: Optional[torch.Tensor] = None,  # [B, 1280] teacher
        sol_stats: Optional[torch.Tensor] = None,      # [B, 4] teacher
        sol_spatial: Optional[torch.Tensor] = None,    # [B, 8, 8] teacher
        return_expert_pred: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Returns:
            v_pred: [B, N, C] velocity prediction
            expert_info: (if return_expert_pred=True) dict with:
                - lune_pred: [B, 1280] predicted Lune features
                - sol_stats_pred: [B, 4] predicted Sol stats
                - sol_spatial_pred: [B, 8, 8] predicted spatial map
        """
    
    @staticmethod
    def create_img_ids(batch_size: int, h: int, w: int, device) -> torch.Tensor:
        """Create position IDs for image patches."""
```

**Key Methods:**
- `enable_gradient_checkpointing()`: Trade compute for VRAM
- `get_trainable_parameters()`: Returns parameters not frozen

---

### `model/zoo.py` - ModelZoo

Manages loading/unloading of teacher models and encoders.

```python
class ModelZoo:
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        vae_repo: str = "black-forest-labs/FLUX.1-dev",
        t5_repo: str = "google/flan-t5-base",
        clip_repo: str = "openai/clip-vit-large-patch14",
        experts_repo: str = "AbstractPhil/tinyflux-experts",
    ): ...
    
    # Loading methods
    def load_vae(self) -> AutoencoderKL: ...
    def load_t5(self) -> Tuple[T5Tokenizer, T5EncoderModel]: ...
    def load_clip(self) -> Tuple[CLIPTokenizer, CLIPTextModel]: ...
    def load_lune(self, filename: str = "sd15-flow-lune-unet.safetensors") -> UNet2DConditionModel: ...
    def load_sol(self, filename: str = "sd15-flow-sol-unet.safetensors") -> UNet2DConditionModel: ...
    
    # Memory management
    def unload(self, name: str): ...      # Unload specific model
    def offload(self): ...                 # Move all to CPU
    def reload(self): ...                  # Move all back to GPU
    
    # Encoding methods
    def encode_image(self, image: PIL.Image) -> torch.Tensor: ...
    def encode_images_batched(self, images: List, batch_size: int = 32) -> torch.Tensor: ...
    def encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def encode_prompts_batched(self, prompts: List[str], batch_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]: ...
    
    # Properties
    @property
    def vae_scale(self) -> float: ...     # VAE scaling factor (0.3611)
```

**Usage Pattern:**
```python
zoo = ModelZoo(device="cuda")

# Load what you need
zoo.load_vae()
zoo.load_t5()

# Encode
latents = zoo.encode_images_batched(images)
t5_emb, clip_pool = zoo.encode_prompts_batched(prompts)

# Free memory when done
zoo.unload("vae")
```

---

### `model/loader.py` - Unified Loading

Load models and configs from various sources.

```python
def load_model(
    source: str,                    # Path, HF repo, or "repo:subfolder"
    config: Optional[TinyFluxConfig] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> TinyFluxDeep:
    """
    Load model from:
    - Local .safetensors or .pt file
    - Local directory (finds model.safetensors + config.json)
    - HuggingFace repo: "user/repo" or "user/repo:subfolder"
    """

def load_config(source: str) -> TinyFluxConfig:
    """Load config from JSON file or directory containing config.json."""
```

**Examples:**
```python
# From local file
model = load_model("checkpoints/step_10000.safetensors", config=my_config)

# From local directory
model = load_model("./my_model/")  # Loads model.safetensors + config.json

# From HuggingFace
model = load_model("AbstractPhil/tiny-flux-deep")
model = load_model("AbstractPhil/tiny-flux-deep:checkpoint_runs/v4")
```

---

## Trainer Components

### `trainer/trainer.py` - Trainer

Main training orchestration.

#### TrainerConfig

```python
@dataclass
class TrainerConfig:
    # Optimization
    learning_rate: float = 1e-4
    optimizer: str = "adamw"          # "adamw", "adamw_8bit", "adafactor"
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    gradient_accumulation: int = 4
    
    # Schedule
    total_steps: int = 100000
    warmup_steps: int = 1000
    lr_scheduler: str = "cosine"      # "cosine", "linear", "constant"
    min_lr: float = 0.0               # Floor for cosine decay
    warmup_type: str = "linear"       # "linear", "cosine"
    
    # Memory optimization
    gradient_checkpointing: bool = False
    compile_mode: Optional[str] = None  # "reduce-overhead", "max-autotune"
    
    # Flow matching
    shift: float = 3.0                # Flux timestep shift
    logit_normal_sampling: bool = True
    logit_mean: float = 0.0
    logit_std: float = 1.0
    
    # Loss
    use_snr_weighting: bool = True
    snr_gamma: float = 5.0
    use_huber_loss: bool = True
    huber_delta: float = 0.1
    use_spatial_weighting: bool = False  # Weight by Sol spatial map
    
    # Expert distillation
    enable_lune: bool = True
    lune_weight: float = 0.1
    lune_warmup_steps: int = 1000
    lune_dropout: float = 0.1         # Teacher dropout
    lune_mode: str = "cosine"         # "cosine", "mse", "huber"
    
    enable_sol: bool = True
    sol_weight: float = 0.05
    sol_warmup_steps: int = 2000
    sol_dropout: float = 0.1          # Teacher dropout
    
    # Text
    text_dropout: float = 0.1         # For CFG training
    
    # EMA
    ema_decay: float = 0.9999
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_steps: Optional[int] = 1000
    save_every_epochs: Optional[int] = None
    keep_last_n_steps: int = 5
    keep_last_n_epochs: int = 3
    
    # Logging
    log_every: int = 100
    tensorboard_dir: str = "./logs"
    
    # Sampling
    sample_every: Optional[int] = 1000
    sample_prompts: List[str] = field(default_factory=list)
    sample_dir: str = "./samples"
    
    # HuggingFace upload
    hf_repo_id: Optional[str] = None
    upload_every_steps: Optional[int] = None
    upload_every_epochs: Optional[int] = None
    
    # Precision
    dtype: torch.dtype = torch.bfloat16
```

#### Trainer Class

```python
class Trainer:
    def __init__(self, model: TinyFluxDeep, config: TrainerConfig): ...
    
    def setup(
        self,
        dataloader: DataLoader,
        cache: Optional[MultiSourceCache] = None,
        sampler: Optional[Sampler] = None,
    ): ...
    
    def train(self, epochs: int = 1): ...
    
    def load_checkpoint(self, path: str): ...
    
    def save_checkpoint(self, path: str): ...
    
    @property
    def ema(self) -> EMA: ...
    
    @property
    def current_step(self) -> int: ...
```

**Usage:**
```python
trainer = Trainer(model, TrainerConfig(
    total_steps=100000,
    learning_rate=3e-4,
    enable_lune=True,
    enable_sol=True,
))

trainer.setup(dataloader, cache, sampler)
trainer.train(epochs=10)
```

---

### `trainer/cache_experts.py` - Feature Caching

#### EncodingCache

Stores VAE latents and text embeddings.

```python
class EncodingCache:
    def __init__(
        self,
        latents: torch.Tensor,      # [N, C, H, W]
        t5_embeds: torch.Tensor,    # [N, L, 768]
        clip_pooled: torch.Tensor,  # [N, 768]
        dtype: torch.dtype = torch.float16,
    ): ...
    
    @classmethod
    def build(
        cls,
        zoo: ModelZoo,
        images: List[PIL.Image],
        prompts: List[str],
        batch_size: int = 32,
    ) -> "EncodingCache": ...
    
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> Dict: ...
    
    def save(self, path: str): ...
    
    @classmethod
    def load(cls, path: str) -> "EncodingCache": ...
```

#### LuneFeatureCache

Stores Lune teacher features at 10 timestep buckets.

```python
class LuneFeatureCache:
    def __init__(
        self,
        features: torch.Tensor,     # [N, 10, 1280]
        t_buckets: torch.Tensor,    # [10] timesteps
        dtype: torch.dtype = torch.float16,
    ): ...
    
    @classmethod
    def build(
        cls,
        zoo: ModelZoo,
        prompts: List[str],
        t_buckets: torch.Tensor = torch.linspace(0.05, 0.95, 10),
        batch_size: int = 64,
        batch_timesteps: bool = False,  # True = faster but more VRAM
    ) -> "LuneFeatureCache": ...
    
    def get_features(
        self,
        indices: torch.Tensor,      # [B] sample indices
        timesteps: torch.Tensor,    # [B] timesteps
    ) -> torch.Tensor:              # [B, 1280] interpolated
        """Linear interpolation between cached timestep buckets."""
    
    def save(self, path: str): ...
    
    @classmethod
    def load(cls, path: str) -> "LuneFeatureCache": ...
```

#### SolFeatureCache

Stores Sol attention statistics and spatial maps.

```python
class SolFeatureCache:
    def __init__(
        self,
        stats: torch.Tensor,        # [N, 10, 4] locality/entropy/clustering/sparsity
        spatial: torch.Tensor,      # [N, 10, 8, 8] importance maps
        t_buckets: torch.Tensor,    # [10]
        dtype: torch.dtype = torch.float16,
    ): ...
    
    @classmethod
    def build(
        cls,
        zoo: ModelZoo,
        prompts: List[str],
        t_buckets: torch.Tensor = torch.linspace(0.05, 0.95, 10),
        spatial_size: int = 8,
        batch_size: int = 64,
        batch_timesteps: bool = True,
    ) -> "SolFeatureCache": ...
    
    def get_features(
        self,
        indices: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # stats [B,4], spatial [B,8,8]
        """Linear interpolation between cached timestep buckets."""
    
    def save(self, path: str): ...
    
    @classmethod
    def load(cls, path: str) -> "SolFeatureCache": ...
```

#### DatasetCache

Combined cache for a single dataset.

```python
class DatasetCache:
    def __init__(
        self,
        encodings: EncodingCache,
        lune: Optional[LuneFeatureCache] = None,
        sol: Optional[SolFeatureCache] = None,
        name: str = "dataset",
    ): ...
    
    @classmethod
    def build(
        cls,
        zoo: ModelZoo,
        images: List[PIL.Image],
        prompts: List[str],
        name: str = "dataset",
        extract_lune: bool = True,
        extract_sol: bool = True,
    ) -> "DatasetCache": ...
    
    def save(self, path: str): ...
    
    @classmethod
    def load(cls, path: str) -> "DatasetCache": ...
```

#### MultiSourceCache

Routes lookups across multiple dataset caches.

```python
class MultiSourceCache:
    def __init__(self): ...
    
    def add(
        self,
        cache: DatasetCache,
        dataset_id: int,
    ) -> "MultiSourceCache": ...
    
    def get_lune(
        self,
        local_indices: torch.Tensor,
        dataset_ids: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Optional[torch.Tensor]: ...
    
    def get_sol(
        self,
        local_indices: torch.Tensor,
        dataset_ids: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: ...
```

---

### `trainer/losses.py` - Loss Functions

```python
def compute_main_loss(
    pred: torch.Tensor,             # [B, N, C]
    target: torch.Tensor,           # [B, N, C]
    mask: Optional[torch.Tensor] = None,        # [B, H, W] FG mask
    spatial_weights: Optional[torch.Tensor] = None,  # [B, 8, 8] from Sol
    use_huber: bool = True,
    huber_delta: float = 0.1,
    fg_weight: float = 2.0,
    bg_weight: float = 0.5,
    snr_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor: ...

def compute_lune_loss(
    pred: torch.Tensor,             # [B, 1280]
    target: torch.Tensor,           # [B, 1280]
    mode: str = "cosine",           # "cosine", "mse", "huber"
) -> torch.Tensor: ...

def compute_sol_loss(
    pred_stats: torch.Tensor,       # [B, 4]
    pred_spatial: torch.Tensor,     # [B, 8, 8]
    target_stats: torch.Tensor,
    target_spatial: torch.Tensor,
) -> torch.Tensor: ...

def min_snr_weight(
    t: torch.Tensor,                # [B] timesteps
    gamma: float = 5.0,
) -> torch.Tensor: ...

def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.1,
) -> torch.Tensor: ...
```

---

### `trainer/schedules.py` - Schedules

```python
# Timestep sampling
def sample_timesteps(
    batch_size: int,
    device: str = "cuda",
    shift: float = 3.0,
    min_t: float = 1e-4,
    max_t: float = 1 - 1e-4,
    logit_normal: bool = True,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor: ...

def flux_shift(t: torch.Tensor, shift: float = 3.0) -> torch.Tensor:
    """Flux timestep shift: t' = s*t / (1 + (s-1)*t)"""

def flux_unshift(t: torch.Tensor, shift: float = 3.0) -> torch.Tensor:
    """Inverse of flux_shift."""

# Loss weight warmup
def get_lune_weight(step: int, warmup_steps: int, target_weight: float) -> float: ...
def get_sol_weight(step: int, warmup_steps: int, target_weight: float) -> float: ...

# LR schedules
def cosine_schedule(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float: ...
def linear_decay(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float: ...
def constant_with_warmup(step: int, warmup_steps: int) -> float: ...
```

---

### `trainer/ema.py` - Exponential Moving Average

```python
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999): ...
    
    @torch.no_grad()
    def update(self, model: nn.Module): ...
    
    def apply_shadow(self, model: nn.Module): ...
    def restore(self, model: nn.Module): ...
    
    def copy_to(self, model: nn.Module): ...
    
    def state_dict(self) -> Dict: ...
    def load_state_dict(self, state: Dict): ...
    
    def save(self, path: str, dtype: torch.dtype = torch.bfloat16): ...
    
    @classmethod
    def load(cls, path: str, model: nn.Module) -> "EMA": ...
```

---

### `trainer/sampling.py` - Inference

```python
class Sampler:
    def __init__(
        self,
        zoo: ModelZoo,
        model: TinyFluxDeep,
        ema: Optional[EMA] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ): ...
    
    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        num_steps: int = 28,
        guidance_scale: float = 5.0,
        height: int = 64,
        width: int = 64,
        seed: Optional[int] = None,
        use_ema: bool = True,
        negative_prompt: str = "blurry, low quality",
    ) -> torch.Tensor:  # [B, 3, H*8, W*8] images in [0, 1]
        ...
    
    def save_samples(
        self,
        images: torch.Tensor,
        prompts: List[str],
        output_dir: str,
        step: Optional[int] = None,
    ) -> str:  # Returns path to saved grid
        ...
```

---

## Utility Components

### `util/predictions.py` - Flow Math

```python
# Rectified Flow (what TinyFlux uses)
def flow_x_t(data, noise, t):
    """x_t = (1-t)*noise + t*data"""

def flow_velocity(data, noise):
    """v = data - noise"""

def flow_data_from_velocity(x_t, v, t):
    """data = x_t + (1-t)*v"""

def flow_noise_from_velocity(x_t, v, t):
    """noise = x_t - t*v"""

# V-Prediction (for reference/conversion)
def vpred_x_t(data, noise, alpha): ...
def vpred_velocity(data, noise, alpha): ...
def vpred_data_from_velocity(x_t, v, alpha): ...

# Epsilon Prediction (for reference/conversion)
def eps_x_t(data, noise, alpha): ...
def eps_target(noise): ...
def eps_data_from_noise(x_t, eps, alpha): ...

# Conversions
def eps_to_vpred(eps, x_t, alpha): ...
def vpred_to_eps(v, x_t, alpha): ...
```