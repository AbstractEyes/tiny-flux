# ============================================================================
# TinyFlux-Deep v4.1 Training Cell - Dual Expert Distillation (Lune + Sol)
# ============================================================================
# Integrates:
#   - Lune: SD1.5-flow trajectory guidance (mid-block features)
#   - Sol: Geometric attention prior (attention statistics + spatial importance)
#
# Both expert features are PRECACHED at 10 timestep buckets for speed.
# At inference, predictors run standalone - no teachers needed.
#
# USAGE: Run model_v4.py cell first, then this cell
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, concatenate_datasets
from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModel, CLIPTokenizer
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file, load_file
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import numpy as np
import math
import json
import random
from typing import Tuple, Optional, Dict, List
import os
from datetime import datetime
from PIL import Image

# ============================================================================
# CUDA OPTIMIZATIONS
# ============================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

import warnings

warnings.filterwarnings('ignore', message='.*TF32.*')

# ============================================================================
# CONFIG
# ============================================================================
BATCH_SIZE = 8
GRAD_ACCUM = 4
LR = 3e-4
EPOCHS = 10
MAX_SEQ = 128
SHIFT = 3.0
DEVICE = "cuda"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

ALLOW_WEIGHT_UPGRADE = True

# HuggingFace Hub
HF_REPO = "AbstractPhil/tiny-flux-deep"
SAVE_EVERY = 1562
UPLOAD_EVERY = 1562
SAMPLE_EVERY = 781
LOG_EVERY = 200
LOG_UPLOAD_EVERY = 1562

# Checkpoint loading
# v4.1 init checkpoint (converted from v3 step_401434)
# Options:
#   "hub:checkpoint_runs/v4_init/lailah_401434_v4_init" - v4.1 init (no EMA, fresh Sol)
#   "hub:step_401434" - v3 checkpoint (will auto-remap expert_predictor -> lune_predictor)
#   "latest" - latest local checkpoint
#   "none" - start fresh
LOAD_TARGET = "hub:checkpoint_runs/v4_init/lailah_401434_v4_init"
RESUME_STEP = 401434

# ============================================================================
# EXPERT REPOSITORY (both Lune and Sol)
# ============================================================================
EXPERTS_REPO = "AbstractPhil/tinyflux-experts"

# ============================================================================
# LUNE EXPERT DISTILLATION CONFIG (trajectory guidance)
# ============================================================================
ENABLE_LUNE_DISTILLATION = True
LUNE_FILENAME = "sd15-flow-lune-unet.safetensors"
LUNE_DIM = 1280  # SD1.5 mid-block dimension
LUNE_HIDDEN_DIM = 512
LUNE_DROPOUT = 0.1
LUNE_LOSS_WEIGHT = 0.1
LUNE_WARMUP_STEPS = 1000
LUNE_DISTILL_MODE = "cosine"  # "hard", "soft", "cosine", "huber"

# ============================================================================
# SOL ATTENTION PRIOR CONFIG (structural guidance)
# ============================================================================
ENABLE_SOL_DISTILLATION = True
SOL_FILENAME = "sd15-flow-sol-unet.safetensors"
SOL_HIDDEN_DIM = 256
SOL_SPATIAL_SIZE = 8  # 8x8 spatial importance map
SOL_GEOMETRIC_WEIGHT = 0.7  # 70% geometric, 30% learned
SOL_LOSS_WEIGHT = 0.05
SOL_WARMUP_STEPS = 2000  # Start Sol later than Lune

# Timestep buckets for precaching (shared by Lune and Sol)
EXPERT_T_BUCKETS = torch.linspace(0.05, 0.95, 10)

# ============================================================================
# LOSS CONFIG
# ============================================================================
USE_HUBER_LOSS = True
HUBER_DELTA = 0.1
USE_SPATIAL_WEIGHTING = False  # Weight main loss by Sol spatial importance

# ============================================================================
# DATASET CONFIG
# ============================================================================
ENABLE_PORTRAIT = False
ENABLE_SCHNELL = False
ENABLE_SPORTFASHION = False
ENABLE_SYNTHMOCAP = False
ENABLE_IMAGENET = False
ENABLE_OBJECT_RELATIONS = True

PORTRAIT_REPO = "AbstractPhil/ffhq_flux_latents_repaired"
PORTRAIT_NUM_SHARDS = 11
SCHNELL_REPO = "AbstractPhil/flux-schnell-teacher-latents"
SCHNELL_CONFIGS = ["train_512"]
SPORTFASHION_REPO = "Pianokill/SportFashion_512x512"
SYNTHMOCAP_REPO = "toyxyz/SynthMoCap_smpl_512"
IMAGENET_REPO = "AbstractPhil/synthetic-imagenet-1k"
IMAGENET_SUBSET = "schnell_512"
OBJECT_RELATIONS_REPO = "AbstractPhil/synthetic-object-relations"

# Confidence threshold for misprediction filtering
IMAGENET_CONFIDENCE_THRESHOLD = 0.5  # If confident but wrong, remove label

FG_LOSS_WEIGHT = 2.0
BG_LOSS_WEIGHT = 0.5
USE_MASKED_LOSS = False
MIN_SNR_GAMMA = 5.0

# Paths
CHECKPOINT_DIR = "./tiny_flux_deep_checkpoints"
LOG_DIR = "./tiny_flux_deep_logs"
SAMPLE_DIR = "./tiny_flux_deep_samples"
ENCODING_CACHE_DIR = "./encoding_cache"
LATENT_CACHE_DIR = "./latent_cache"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(ENCODING_CACHE_DIR, exist_ok=True)
os.makedirs(LATENT_CACHE_DIR, exist_ok=True)

# ============================================================================
# REGULARIZATION CONFIG
# ============================================================================
TEXT_DROPOUT = 0.1
GUIDANCE_DROPOUT = 0.1
EMA_DECAY = 0.9999


# ============================================================================
# LUNE FEATURE CACHE (SD1.5 mid-block features)
# ============================================================================
class LuneFeatureCache:
    """
    Precached SD1.5-flow Lune features with timestep interpolation.
    Features extracted at 10 timestep buckets [0.05, 0.15, ..., 0.95].
    """

    def __init__(self, features: torch.Tensor, t_buckets: torch.Tensor, dtype=torch.float16):
        self.features = features.to(dtype)  # [N, 10, 1280]
        self.t_buckets = t_buckets
        self.t_min = t_buckets[0].item()
        self.t_max = t_buckets[-1].item()
        self.t_step = (t_buckets[1] - t_buckets[0]).item()
        self.n_buckets = len(t_buckets)
        self.dtype = dtype

    def get_features(self, indices: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        t_clamped = timesteps.float().clamp(self.t_min, self.t_max)
        t_idx_float = (t_clamped - self.t_min) / self.t_step
        t_idx_low = t_idx_float.long().clamp(0, self.n_buckets - 2)
        t_idx_high = (t_idx_low + 1).clamp(0, self.n_buckets - 1)
        alpha = (t_idx_float - t_idx_low.float()).unsqueeze(-1)

        idx_cpu = indices.cpu()
        t_low_cpu = t_idx_low.cpu()
        t_high_cpu = t_idx_high.cpu()

        f_low = self.features[idx_cpu, t_low_cpu]
        f_high = self.features[idx_cpu, t_high_cpu]

        result = (1 - alpha.cpu()) * f_low + alpha.cpu() * f_high
        return result.to(device=device, dtype=self.dtype)


# ============================================================================
# SOL FEATURE CACHE (attention statistics + spatial importance)
# ============================================================================
class SolFeatureCache:
    """
    Precached Sol attention statistics with timestep interpolation.

    Statistics per sample per timestep:
      - stats: [N, 10, 4] - locality, entropy, clustering, sparsity
      - spatial: [N, 10, 8, 8] - spatial importance map
    """

    def __init__(self, stats: torch.Tensor, spatial: torch.Tensor,
                 t_buckets: torch.Tensor, dtype=torch.float16):
        self.stats = stats.to(dtype)  # [N, 10, 4]
        self.spatial = spatial.to(dtype)  # [N, 10, 8, 8]
        self.t_buckets = t_buckets
        self.t_min = t_buckets[0].item()
        self.t_max = t_buckets[-1].item()
        self.t_step = (t_buckets[1] - t_buckets[0]).item()
        self.n_buckets = len(t_buckets)
        self.dtype = dtype

    def get_features(self, indices: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        s_low = self.stats[idx_cpu, t_low_cpu]
        s_high = self.stats[idx_cpu, t_high_cpu]
        stats_result = (1 - alpha_stats.cpu()) * s_low + alpha_stats.cpu() * s_high

        sp_low = self.spatial[idx_cpu, t_low_cpu]
        sp_high = self.spatial[idx_cpu, t_high_cpu]
        spatial_result = (1 - alpha_spatial.cpu()) * sp_low + alpha_spatial.cpu() * sp_high

        return (
            stats_result.to(device=device, dtype=self.dtype),
            spatial_result.to(device=device, dtype=self.dtype)
        )


def load_or_extract_lune_features(cache_path: str, prompts: List[str], name: str,
                                  clip_tok, clip_enc, t_buckets: torch.Tensor,
                                  batch_size: int = 32) -> Optional[LuneFeatureCache]:
    """Load cached Lune features or extract from SD1.5-flow teacher."""
    if not prompts or not ENABLE_LUNE_DISTILLATION:
        return None

    if os.path.exists(cache_path):
        print(f"Loading cached {name} Lune features...")
        cached = torch.load(cache_path, map_location="cpu")
        cache = LuneFeatureCache(cached["features"], cached["t_buckets"], DTYPE)
        print(f"  ✓ Loaded {cache.features.shape[0]} samples × {cache.n_buckets} timesteps")
        return cache

    print(f"Extracting {name} Lune features ({len(prompts)} × {len(t_buckets)} timesteps)...")
    print(f"  This is a one-time operation, will be cached.")

    checkpoint_path = hf_hub_download(
        repo_id=EXPERTS_REPO,
        filename=LUNE_FILENAME,
    )
    print(f"  Loaded Lune from {EXPERTS_REPO}/{LUNE_FILENAME}")

    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        subfolder="unet",
        torch_dtype=torch.float16,
    ).to(DEVICE).eval()

    state_dict = load_file(checkpoint_path)
    unet.load_state_dict(state_dict, strict=False)

    # Convert to fp16 and compile for speed
    unet = unet.half()
    unet = torch.compile(unet, mode="reduce-overhead")
    print(f"  ✓ Lune UNet compiled (fp16)")

    for p in unet.parameters():
        p.requires_grad = False

    mid_features = [None]

    def hook_fn(module, inp, out):
        mid_features[0] = out.mean(dim=[2, 3])

    unet.mid_block.register_forward_hook(hook_fn)

    n_prompts = len(prompts)
    n_buckets = len(t_buckets)
    all_features = torch.zeros(n_prompts, n_buckets, LUNE_DIM, dtype=torch.float16)

    # A100 can handle large batches - 64 prompts × 10 timesteps = 640 UNet forward passes batched
    # SD1.5 UNet at 64x64 latents uses ~2GB for batch of 64, so 640 samples ~10-15GB
    LUNE_BATCH_PROMPTS = 64  # Number of prompts per iteration

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for start_idx in tqdm(range(0, n_prompts, LUNE_BATCH_PROMPTS), desc=f"Extracting {name} Lune"):
            end_idx = min(start_idx + LUNE_BATCH_PROMPTS, n_prompts)
            batch_prompts = prompts[start_idx:end_idx]
            B = len(batch_prompts)

            # Encode CLIP once per prompt batch
            clip_inputs = clip_tok(
                batch_prompts, return_tensors="pt", padding="max_length",
                max_length=77, truncation=True
            ).to(DEVICE)
            clip_hidden = clip_enc(**clip_inputs).last_hidden_state  # [B, 77, 768]

            # Expand for all timesteps: [B * n_buckets, 77, 768]
            clip_expanded = clip_hidden.unsqueeze(1).expand(-1, n_buckets, -1, -1)
            clip_expanded = clip_expanded.reshape(B * n_buckets, 77, -1)

            # Create timesteps for all buckets: [B * n_buckets]
            t_expanded = t_buckets.unsqueeze(0).expand(B, -1).reshape(-1).to(DEVICE)

            # Random latents: [B * n_buckets, 4, 64, 64]
            latents = torch.randn(B * n_buckets, 4, 64, 64, device=DEVICE, dtype=DTYPE)

            # Single batched UNet forward pass
            _ = unet(latents, t_expanded * 1000, encoder_hidden_states=clip_expanded.to(DTYPE))

            # Reshape features back to [B, n_buckets, LUNE_DIM]
            features = mid_features[0].reshape(B, n_buckets, -1)
            all_features[start_idx:end_idx] = features.cpu().to(torch.float16)

    del unet
    torch.cuda.empty_cache()

    torch.save({"features": all_features, "t_buckets": t_buckets}, cache_path)
    print(f"  ✓ Cached to {cache_path}")
    print(f"  Size: {all_features.numel() * 2 / 1e9:.2f} GB")

    return LuneFeatureCache(all_features, t_buckets, DTYPE)


def load_or_extract_sol_features(cache_path: str, prompts: List[str], name: str,
                                 clip_tok, clip_enc, t_buckets: torch.Tensor,
                                 spatial_size: int = 8,
                                 batch_size: int = 32) -> Optional[SolFeatureCache]:
    """Load cached Sol features or generate geometric heuristics."""
    if not prompts or not ENABLE_SOL_DISTILLATION:
        return None

    if os.path.exists(cache_path):
        print(f"Loading cached {name} Sol features...")
        cached = torch.load(cache_path, map_location="cpu")
        cache = SolFeatureCache(
            cached["stats"], cached["spatial"], cached["t_buckets"], DTYPE
        )
        print(f"  ✓ Loaded {cache.stats.shape[0]} samples × {cache.n_buckets} timesteps")
        return cache

    print(f"Generating {name} Sol features ({len(prompts)} × {len(t_buckets)} timesteps)...")
    print(f"  Using geometric heuristics (no teacher needed)")

    n_prompts = len(prompts)
    n_buckets = len(t_buckets)

    # Vectorized generation - no loops needed
    # Stats: [n_buckets, 4] then broadcast to [n_prompts, n_buckets, 4]
    t_vals = t_buckets.float()  # [n_buckets]

    locality = 1 - t_vals  # [n_buckets]
    entropy = t_vals
    clustering = 0.5 - 0.3 * (t_vals - 0.5).abs()
    sparsity = 1 - t_vals

    stats_per_t = torch.stack([locality, entropy, clustering, sparsity], dim=-1)  # [n_buckets, 4]
    all_stats = stats_per_t.unsqueeze(0).expand(n_prompts, -1, -1).to(torch.float16)  # [n_prompts, n_buckets, 4]

    # Spatial: [n_buckets, spatial_size, spatial_size] then broadcast
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, spatial_size),
        torch.linspace(-1, 1, spatial_size),
        indexing='ij'
    )
    center_dist = torch.sqrt(x ** 2 + y ** 2)  # [spatial_size, spatial_size]

    # Vectorized across timesteps: [n_buckets, spatial_size, spatial_size]
    t_weight = (1 - t_vals).view(-1, 1, 1)  # [n_buckets, 1, 1]
    center_bias = 1 - center_dist.unsqueeze(0) * t_weight  # [n_buckets, spatial_size, spatial_size]
    center_bias = center_bias / center_bias.sum(dim=[-2, -1], keepdim=True)  # Normalize per timestep

    all_spatial = center_bias.unsqueeze(0).expand(n_prompts, -1, -1, -1).to(
        torch.float16)  # [n_prompts, n_buckets, 8, 8]

    torch.save({
        "stats": all_stats,
        "spatial": all_spatial,
        "t_buckets": t_buckets
    }, cache_path)
    print(f"  ✓ Cached to {cache_path}")

    return SolFeatureCache(all_stats, all_spatial, t_buckets, DTYPE)


# ============================================================================
# EMA
# ============================================================================
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self._backup = {}
        if hasattr(model, '_orig_mod'):
            state = model._orig_mod.state_dict()
        else:
            state = model.state_dict()
        for k, v in state.items():
            self.shadow[k] = v.clone().detach()

    @torch.no_grad()
    def update(self, model):
        if hasattr(model, '_orig_mod'):
            state = model._orig_mod.state_dict()
        else:
            state = model.state_dict()
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].lerp_(v.to(self.shadow[k].dtype), 1 - self.decay)

    def apply_shadow_for_eval(self, model):
        if hasattr(model, '_orig_mod'):
            self._backup = {k: v.clone() for k, v in model._orig_mod.state_dict().items()}
            model._orig_mod.load_state_dict(self.shadow)
        else:
            self._backup = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(self.shadow)

    def restore(self, model):
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(self._backup)
        else:
            model.load_state_dict(self._backup)
        self._backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def sync_from_model(self, model, pattern=None):
        if hasattr(model, '_orig_mod'):
            model_state = model._orig_mod.state_dict()
        else:
            model_state = model.state_dict()

        synced = 0
        for k, v in model_state.items():
            if pattern is None or pattern in k:
                if k in self.shadow:
                    self.shadow[k] = v.clone().to(self.shadow[k].device)
                    synced += 1

        print(f"  ✓ Synced EMA: {synced} weights" + (f" matching '{pattern}'" if pattern else ""))

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state['shadow'].items()}
        self.decay = state.get('decay', self.decay)

    def load_shadow(self, shadow_state, model=None):
        device = next(iter(self.shadow.values())).device if self.shadow else 'cuda'

        loaded = 0
        skipped_old = 0
        initialized_from_model = 0

        for k, v in shadow_state.items():
            if k in self.shadow:
                self.shadow[k] = v.clone().to(device)
                loaded += 1
            else:
                skipped_old += 1

        if model is not None:
            if hasattr(model, '_orig_mod'):
                model_state = model._orig_mod.state_dict()
            else:
                model_state = model.state_dict()

            for k in self.shadow:
                if k not in shadow_state and k in model_state:
                    self.shadow[k] = model_state[k].clone().to(device)
                    initialized_from_model += 1

        print(f"  ✓ Restored EMA: {loaded} loaded, {skipped_old} deprecated, {initialized_from_model} new (from model)")


# ============================================================================
# REGULARIZATION
# ============================================================================
def apply_text_dropout(t5_embeds, clip_pooled, dropout_prob=0.1):
    B = t5_embeds.shape[0]
    mask = torch.rand(B, device=t5_embeds.device) < dropout_prob
    t5_embeds = t5_embeds.clone()
    clip_pooled = clip_pooled.clone()
    t5_embeds[mask] = 0
    clip_pooled[mask] = 0
    return t5_embeds, clip_pooled, mask


# ============================================================================
# MASKING UTILITIES
# ============================================================================
def detect_background_color(image: Image.Image, sample_size: int = 100) -> Tuple[int, int, int]:
    img = np.array(image)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    h, w = img.shape[:2]
    corners = [
        img[:sample_size, :sample_size],
        img[:sample_size, -sample_size:],
        img[-sample_size:, :sample_size],
        img[-sample_size:, -sample_size:],
    ]
    corner_pixels = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    bg_color = np.median(corner_pixels, axis=0).astype(np.uint8)
    return tuple(bg_color)


def create_product_mask(image: Image.Image, threshold: int = 30) -> np.ndarray:
    img = np.array(image).astype(np.float32)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    bg_color = detect_background_color(image)
    bg_color = np.array(bg_color, dtype=np.float32)
    diff = np.sqrt(np.sum((img - bg_color) ** 2, axis=-1))
    mask = (diff > threshold).astype(np.float32)
    return mask


def create_smpl_mask(conditioning_image: Image.Image, threshold: int = 20) -> np.ndarray:
    img = np.array(conditioning_image).astype(np.float32)
    if len(img.shape) == 2:
        return (img > threshold).astype(np.float32)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    is_background = (g > r + 20) & (g > b + 20)
    mask = (~is_background).astype(np.float32)
    return mask


def downsample_mask_to_latent(mask: np.ndarray, latent_h: int = 64, latent_w: int = 64) -> torch.Tensor:
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_pil = mask_pil.resize((latent_w, latent_h), Image.Resampling.BILINEAR)
    mask_latent = np.array(mask_pil).astype(np.float32) / 255.0
    return torch.from_numpy(mask_latent)


# ============================================================================
# HF HUB SETUP
# ============================================================================
print("Setting up HuggingFace Hub...")
api = HfApi()


# ============================================================================
# FLOW MATCHING HELPERS
# ============================================================================
def flux_shift(t, s=SHIFT):
    return s * t / (1 + (s - 1) * t)


def min_snr_weight(t, gamma=MIN_SNR_GAMMA):
    snr = (t / (1 - t).clamp(min=1e-5)).pow(2)
    return torch.clamp(snr, max=gamma) / snr.clamp(min=1e-5)


# ============================================================================
# LOAD TEXT ENCODERS
# ============================================================================
print("Loading text encoders...")
t5_tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
t5_enc = T5EncoderModel.from_pretrained("google/flan-t5-base", torch_dtype=DTYPE).to(DEVICE).eval()
for p in t5_enc.parameters():
    p.requires_grad = False

clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
clip_enc = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=DTYPE).to(DEVICE).eval()
for p in clip_enc.parameters():
    p.requires_grad = False
print("✓ Text encoders loaded")

# ============================================================================
# LOAD VAE
# ============================================================================
print("Loading VAE...")
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=DTYPE).to(
    DEVICE).eval()
for p in vae.parameters():
    p.requires_grad = False
VAE_SCALE = vae.config.scaling_factor
print(f"✓ VAE loaded (scale={VAE_SCALE})")


# ============================================================================
# ENCODING FUNCTIONS
# ============================================================================
@torch.no_grad()
def encode_prompt(prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
    t5_inputs = t5_tok(prompt, return_tensors="pt", padding="max_length",
                       max_length=MAX_SEQ, truncation=True).to(DEVICE)
    t5_out = t5_enc(**t5_inputs).last_hidden_state
    clip_inputs = clip_tok(prompt, return_tensors="pt", padding="max_length",
                           max_length=77, truncation=True).to(DEVICE)
    clip_out = clip_enc(**clip_inputs).pooler_output
    return t5_out.squeeze(0), clip_out.squeeze(0)


@torch.no_grad()
@torch.no_grad()
def encode_prompts_batched(prompts: List[str], batch_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch encode prompts with T5 and CLIP."""
    all_t5 = []
    all_clip = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Encoding prompts", leave=False):
        batch = prompts[i:i + batch_size]
        t5_inputs = t5_tok(batch, return_tensors="pt", padding="max_length",
                           max_length=MAX_SEQ, truncation=True).to(DEVICE)
        t5_out = t5_enc(**t5_inputs).last_hidden_state
        all_t5.append(t5_out.cpu())
        clip_inputs = clip_tok(batch, return_tensors="pt", padding="max_length",
                               max_length=77, truncation=True).to(DEVICE)
        clip_out = clip_enc(**clip_inputs).pooler_output
        all_clip.append(clip_out.cpu())
    return torch.cat(all_t5, dim=0), torch.cat(all_clip, dim=0)


@torch.no_grad()
def encode_image_to_latent(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (512, 512):
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
    img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor * 2.0 - 1.0).to(DEVICE, dtype=DTYPE)
    latent = vae.encode(img_tensor).latent_dist.sample()
    latent = latent * VAE_SCALE
    return latent.squeeze(0).cpu()


# ============================================================================
# LOAD DATASETS
# ============================================================================

portrait_ds = None
portrait_indices = []
portrait_prompts = []

if ENABLE_PORTRAIT:
    print(f"\n[1/6] Loading portrait dataset from {PORTRAIT_REPO}...")
    portrait_shards = []
    for i in range(PORTRAIT_NUM_SHARDS):
        split_name = f"train_{i:02d}"
        print(f"  Loading {split_name}...")
        shard = load_dataset(PORTRAIT_REPO, split=split_name)
        portrait_shards.append(shard)
    portrait_ds = concatenate_datasets(portrait_shards)
    print(f"✓ Portrait: {len(portrait_ds)} base samples")
    print("  Extracting prompts (columnar)...")
    florence_list = list(portrait_ds["text_florence"])
    llava_list = list(portrait_ds["text_llava"])
    blip_list = list(portrait_ds["text_blip"])
    for i, (f, l, b) in enumerate(zip(florence_list, llava_list, blip_list)):
        if f and f.strip():
            portrait_indices.append(i)
            portrait_prompts.append(f)
        if l and l.strip():
            portrait_indices.append(i)
            portrait_prompts.append(l)
        if b and b.strip():
            portrait_indices.append(i)
            portrait_prompts.append(b)
    print(f"  Expanded: {len(portrait_prompts)} samples (3 prompts/image)")
else:
    print("\n[1/6] Portrait dataset DISABLED")

schnell_ds = None
schnell_prompts = []

if ENABLE_SCHNELL:
    print(f"\n[2/6] Loading schnell teacher dataset from {SCHNELL_REPO}...")
    schnell_datasets = []
    for config in SCHNELL_CONFIGS:
        print(f"  Loading {config}...")
        ds = load_dataset(SCHNELL_REPO, config, split="train")
        schnell_datasets.append(ds)
        print(f"    {len(ds)} samples")
    schnell_ds = concatenate_datasets(schnell_datasets)
    schnell_prompts = list(schnell_ds["prompt"])
    print(f"✓ Schnell: {len(schnell_ds)} samples")
else:
    print("\n[2/6] Schnell dataset DISABLED")

sportfashion_ds = None
sportfashion_prompts = []
sportfashion_latents = None
sportfashion_masks = None

if ENABLE_SPORTFASHION:
    print(f"\n[3/6] Loading SportFashion dataset from {SPORTFASHION_REPO}...")
    sportfashion_ds = load_dataset(SPORTFASHION_REPO, split="train")
    sportfashion_prompts = list(sportfashion_ds["text"])
    print(f"✓ SportFashion: {len(sportfashion_ds)} samples")

    # Precache latents and masks
    sportfashion_latent_cache = os.path.join(LATENT_CACHE_DIR, f"sportfashion_latents_{len(sportfashion_ds)}.pt")
    sportfashion_mask_cache = os.path.join(LATENT_CACHE_DIR, f"sportfashion_masks_{len(sportfashion_ds)}.pt")

    if os.path.exists(sportfashion_latent_cache):
        print(f"  Loading cached SportFashion latents...")
        sportfashion_latents = torch.load(sportfashion_latent_cache)
        print(f"  ✓ Loaded {len(sportfashion_latents)} latents")
        if os.path.exists(sportfashion_mask_cache):
            sportfashion_masks = torch.load(sportfashion_mask_cache)
            print(f"  ✓ Loaded {len(sportfashion_masks)} masks")
    else:
        print(f"  Encoding SportFashion images to latents (one-time)...")
        VAE_BATCH_SIZE = 64  # A100 can handle large batches
        sportfashion_latents = []
        sportfashion_masks = []
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(sportfashion_ds), VAE_BATCH_SIZE), desc="Encoding latents"):
                end_idx = min(start_idx + VAE_BATCH_SIZE, len(sportfashion_ds))
                batch_images = []
                batch_masks = []
                for i in range(start_idx, end_idx):
                    image = sportfashion_ds[i]["image"]
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    if image.size != (512, 512):
                        image = image.resize((512, 512), Image.Resampling.LANCZOS)
                    img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1)
                    batch_images.append(img_tensor)
                    # Create mask
                    pixel_mask = create_product_mask(image)
                    mask = downsample_mask_to_latent(pixel_mask, 64, 64)
                    batch_masks.append(mask)
                batch_tensor = torch.stack(batch_images)
                batch_tensor = (batch_tensor * 2.0 - 1.0).to(DEVICE, dtype=DTYPE)
                latents = vae.encode(batch_tensor).latent_dist.sample()
                latents = latents * VAE_SCALE
                sportfashion_latents.append(latents.cpu())
                sportfashion_masks.extend(batch_masks)
        sportfashion_latents = torch.cat(sportfashion_latents, dim=0)
        sportfashion_masks = torch.stack(sportfashion_masks)
        torch.save(sportfashion_latents, sportfashion_latent_cache)
        torch.save(sportfashion_masks, sportfashion_mask_cache)
        print(f"  ✓ Cached to {sportfashion_latent_cache}")
else:
    print("\n[3/6] SportFashion dataset DISABLED")

synthmocap_ds = None
synthmocap_prompts = []
synthmocap_latents = None
synthmocap_masks = None

if ENABLE_SYNTHMOCAP:
    print(f"\n[4/6] Loading SynthMoCap dataset from {SYNTHMOCAP_REPO}...")
    synthmocap_ds = load_dataset(SYNTHMOCAP_REPO, split="train")
    synthmocap_prompts = list(synthmocap_ds["text"])
    print(f"✓ SynthMoCap: {len(synthmocap_ds)} samples")

    # Precache latents and masks
    synthmocap_latent_cache = os.path.join(LATENT_CACHE_DIR, f"synthmocap_latents_{len(synthmocap_ds)}.pt")
    synthmocap_mask_cache = os.path.join(LATENT_CACHE_DIR, f"synthmocap_masks_{len(synthmocap_ds)}.pt")

    if os.path.exists(synthmocap_latent_cache):
        print(f"  Loading cached SynthMoCap latents...")
        synthmocap_latents = torch.load(synthmocap_latent_cache)
        print(f"  ✓ Loaded {len(synthmocap_latents)} latents")
        if os.path.exists(synthmocap_mask_cache):
            synthmocap_masks = torch.load(synthmocap_mask_cache)
            print(f"  ✓ Loaded {len(synthmocap_masks)} masks")
    else:
        print(f"  Encoding SynthMoCap images to latents (one-time)...")
        VAE_BATCH_SIZE = 64  # A100 can handle large batches
        synthmocap_latents = []
        synthmocap_masks = []
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(synthmocap_ds), VAE_BATCH_SIZE), desc="Encoding latents"):
                end_idx = min(start_idx + VAE_BATCH_SIZE, len(synthmocap_ds))
                batch_images = []
                batch_masks = []
                for i in range(start_idx, end_idx):
                    image = synthmocap_ds[i]["image"]
                    conditioning = synthmocap_ds[i]["conditioning_image"]
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    if image.size != (512, 512):
                        image = image.resize((512, 512), Image.Resampling.LANCZOS)
                    img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1)
                    batch_images.append(img_tensor)
                    # Create mask from conditioning image
                    pixel_mask = create_smpl_mask(conditioning)
                    mask = downsample_mask_to_latent(pixel_mask, 64, 64)
                    batch_masks.append(mask)
                batch_tensor = torch.stack(batch_images)
                batch_tensor = (batch_tensor * 2.0 - 1.0).to(DEVICE, dtype=DTYPE)
                latents = vae.encode(batch_tensor).latent_dist.sample()
                latents = latents * VAE_SCALE
                synthmocap_latents.append(latents.cpu())
                synthmocap_masks.extend(batch_masks)
        synthmocap_latents = torch.cat(synthmocap_latents, dim=0)
        synthmocap_masks = torch.stack(synthmocap_masks)
        torch.save(synthmocap_latents, synthmocap_latent_cache)
        torch.save(synthmocap_masks, synthmocap_mask_cache)
        print(f"  ✓ Cached to {synthmocap_latent_cache}")
else:
    print("\n[4/6] SynthMoCap dataset DISABLED")

# ============================================================================
# IMAGENET DATASET WITH SMART PROMPT FILTERING
# ============================================================================
imagenet_ds = None
imagenet_prompts = []


def build_imagenet_prompt(item):
    semantic_class = item.get("semantic_class", "object")
    semantic_subclass = item.get("semantic_subclass", "")
    label = item.get("label", "").replace("_", " ")
    base_prompt = item.get("prompt", "")
    synset_id = item.get("synset_id", "")

    pred_confidence = item.get("pred_confidence", 0.0)
    top1_correct = item.get("top1_correct", False)
    top5_correct = item.get("top5_correct", False)

    confident_but_wrong = (
            pred_confidence >= IMAGENET_CONFIDENCE_THRESHOLD and
            not top1_correct and
            not top5_correct
    )

    if confident_but_wrong:
        parts = ["subject", semantic_class]
        if semantic_subclass:
            parts.append(semantic_subclass)
        parts.append(base_prompt)
        parts.append(synset_id)
        parts.append("imagenet")
    else:
        parts = ["subject", semantic_class]
        if semantic_subclass:
            parts.append(semantic_subclass)
        if label:
            parts.append(label)
        parts.append(base_prompt)
        parts.append(synset_id)
        parts.append("imagenet")

    return ", ".join(p for p in parts if p)


if ENABLE_IMAGENET:
    print(f"\n[5/6] Loading Synthetic ImageNet from {IMAGENET_REPO}...")
    imagenet_ds = load_dataset(IMAGENET_REPO, IMAGENET_SUBSET, split="train")
    print(f"  Raw samples: {len(imagenet_ds)}")

    # Use columnar access - MUCH faster than row iteration
    print(f"  Building prompts...")
    semantic_classes = imagenet_ds["semantic_class"]
    semantic_subclasses = imagenet_ds.get("semantic_subclass", [""] * len(
        imagenet_ds)) if "semantic_subclass" in imagenet_ds.features else [""] * len(imagenet_ds)
    labels = imagenet_ds["label"]
    base_prompts = imagenet_ds["prompt"]
    synset_ids = imagenet_ds["synset_id"]
    pred_confidences = imagenet_ds.get("pred_confidence", [0.0] * len(
        imagenet_ds)) if "pred_confidence" in imagenet_ds.features else [0.0] * len(imagenet_ds)
    top1_corrects = imagenet_ds.get("top1_correct", [False] * len(
        imagenet_ds)) if "top1_correct" in imagenet_ds.features else [False] * len(imagenet_ds)
    top5_corrects = imagenet_ds.get("top5_correct", [False] * len(
        imagenet_ds)) if "top5_correct" in imagenet_ds.features else [False] * len(imagenet_ds)

    # Handle case where columns might not exist
    if not isinstance(semantic_subclasses, list):
        semantic_subclasses = list(semantic_subclasses) if semantic_subclasses else [""] * len(imagenet_ds)
    if not isinstance(pred_confidences, list):
        pred_confidences = list(pred_confidences) if pred_confidences else [0.0] * len(imagenet_ds)
    if not isinstance(top1_corrects, list):
        top1_corrects = list(top1_corrects) if top1_corrects else [False] * len(imagenet_ds)
    if not isinstance(top5_corrects, list):
        top5_corrects = list(top5_corrects) if top5_corrects else [False] * len(imagenet_ds)

    confident_wrong = 0
    for i in range(len(imagenet_ds)):
        semantic_class = semantic_classes[i] if semantic_classes[i] else "object"
        semantic_subclass = semantic_subclasses[i] if i < len(semantic_subclasses) else ""
        label = labels[i].replace("_", " ") if labels[i] else ""
        base_prompt = base_prompts[i] if base_prompts[i] else ""
        synset_id = synset_ids[i] if synset_ids[i] else ""
        pred_confidence = pred_confidences[i] if i < len(pred_confidences) else 0.0
        top1_correct = top1_corrects[i] if i < len(top1_corrects) else False
        top5_correct = top5_corrects[i] if i < len(top5_corrects) else False

        confident_but_wrong = (
                pred_confidence >= IMAGENET_CONFIDENCE_THRESHOLD and
                not top1_correct and
                not top5_correct
        )

        if confident_but_wrong:
            parts = ["subject", semantic_class]
            if semantic_subclass:
                parts.append(semantic_subclass)
            parts.append(base_prompt)
            parts.append(synset_id)
            parts.append("imagenet")
            confident_wrong += 1
        else:
            parts = ["subject", semantic_class]
            if semantic_subclass:
                parts.append(semantic_subclass)
            if label:
                parts.append(label)
            parts.append(base_prompt)
            parts.append(synset_id)
            parts.append("imagenet")

        imagenet_prompts.append(", ".join(p for p in parts if p))

    print(f"✓ ImageNet: {len(imagenet_ds)} samples")
    print(f"  Confident mispredictions (label removed): {confident_wrong}")

    imagenet_latent_cache = os.path.join(LATENT_CACHE_DIR, f"imagenet_latents_{len(imagenet_ds)}.pt")
    if os.path.exists(imagenet_latent_cache):
        print(f"  Loading cached ImageNet latents...")
        imagenet_latents = torch.load(imagenet_latent_cache)
        print(f"  ✓ Loaded {len(imagenet_latents)} latents")
    else:
        print(f"  Encoding ImageNet images to latents (one-time)...")
        VAE_BATCH_SIZE = 64  # A100 can handle large batches
        imagenet_latents = []
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(imagenet_ds), VAE_BATCH_SIZE), desc="Encoding latents"):
                end_idx = min(start_idx + VAE_BATCH_SIZE, len(imagenet_ds))
                batch_images = []
                for i in range(start_idx, end_idx):
                    image = imagenet_ds[i]["image"]
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    if image.size != (512, 512):
                        image = image.resize((512, 512), Image.Resampling.LANCZOS)
                    img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1)
                    batch_images.append(img_tensor)
                batch_tensor = torch.stack(batch_images)
                batch_tensor = (batch_tensor * 2.0 - 1.0).to(DEVICE, dtype=DTYPE)
                latents = vae.encode(batch_tensor).latent_dist.sample()
                latents = latents * VAE_SCALE
                imagenet_latents.append(latents.cpu())
        imagenet_latents = torch.cat(imagenet_latents, dim=0)
        torch.save(imagenet_latents, imagenet_latent_cache)
        print(f"  ✓ Cached to {imagenet_latent_cache}")
else:
    print("\n[5/6] ImageNet dataset DISABLED")
    imagenet_latents = None

# ============================================================================
# OBJECT RELATIONS DATASET WITH SUBJECT PREFIX
# ============================================================================
object_relations_ds = None
object_relations_prompts = []
object_relations_latents = None


def build_object_relations_prompt(item):
    prompt = item.get("prompt", "")
    if random.random() < 0.5:
        return f"subject, object, {prompt}"
    else:
        return f"subject, {prompt}"


if ENABLE_OBJECT_RELATIONS:
    print(f"\n[6/6] Loading Object Relations from {OBJECT_RELATIONS_REPO}...")
    object_relations_ds = load_dataset(OBJECT_RELATIONS_REPO, "schnell_512_1", split="train")
    print(f"  Raw samples: {len(object_relations_ds)}")

    # Use columnar access - MUCH faster than row iteration
    print(f"  Building prompts...")
    all_prompts = object_relations_ds["prompt"]  # Get entire column at once

    random.seed(42)
    object_relations_prompts = []
    for prompt in all_prompts:
        if random.random() < 0.5:
            object_relations_prompts.append(f"subject, object, {prompt}")
        else:
            object_relations_prompts.append(f"subject, {prompt}")
    random.seed()

    subject_object_count = sum(1 for p in object_relations_prompts if p.startswith("subject, object,"))
    subject_only_count = len(object_relations_prompts) - subject_object_count
    print(f"✓ Object Relations: {len(object_relations_ds)} samples")
    print(f"  'subject, object, ...' prefix: {subject_object_count}")
    print(f"  'subject, ...' prefix: {subject_only_count}")

    object_relations_latent_cache = os.path.join(LATENT_CACHE_DIR,
                                                 f"object_relations_latents_{len(object_relations_ds)}.pt")
    if os.path.exists(object_relations_latent_cache):
        print(f"  Loading cached Object Relations latents...")
        object_relations_latents = torch.load(object_relations_latent_cache)
        print(f"  ✓ Loaded {len(object_relations_latents)} latents")
    else:
        print(f"  Encoding Object Relations images to latents (one-time)...")
        VAE_BATCH_SIZE = 64  # A100 can handle large batches
        object_relations_latents = []
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(object_relations_ds), VAE_BATCH_SIZE), desc="Encoding latents"):
                end_idx = min(start_idx + VAE_BATCH_SIZE, len(object_relations_ds))
                batch_images = []
                for i in range(start_idx, end_idx):
                    image = object_relations_ds[i]["image"]
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    if image.size != (512, 512):
                        image = image.resize((512, 512), Image.Resampling.LANCZOS)
                    img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1)
                    batch_images.append(img_tensor)
                batch_tensor = torch.stack(batch_images)
                batch_tensor = (batch_tensor * 2.0 - 1.0).to(DEVICE, dtype=DTYPE)
                latents = vae.encode(batch_tensor).latent_dist.sample()
                latents = latents * VAE_SCALE
                object_relations_latents.append(latents.cpu())
        object_relations_latents = torch.cat(object_relations_latents, dim=0)
        torch.save(object_relations_latents, object_relations_latent_cache)
        print(f"  ✓ Cached to {object_relations_latent_cache}")
else:
    print("\n[6/6] Object Relations dataset DISABLED")

# ============================================================================
# ENCODE ALL PROMPTS
# ============================================================================
total_samples = len(portrait_prompts) + len(schnell_prompts) + len(sportfashion_prompts) + len(
    synthmocap_prompts) + len(imagenet_prompts) + len(object_relations_prompts)
print(f"\nTotal combined samples: {total_samples}")


def load_or_encode(cache_path, prompts, name):
    if not prompts:
        return None, None
    if os.path.exists(cache_path):
        print(f"Loading cached {name} encodings...")
        cached = torch.load(cache_path)
        return cached["t5_embeds"], cached["clip_pooled"]
    else:
        print(f"Encoding {len(prompts)} {name} prompts...")
        t5, clip = encode_prompts_batched(prompts, batch_size=64)
        torch.save({"t5_embeds": t5, "clip_pooled": clip}, cache_path)
        print(f"✓ Cached to {cache_path}")
        return t5, clip


portrait_t5, portrait_clip = None, None
schnell_t5, schnell_clip = None, None
sportfashion_t5, sportfashion_clip = None, None
synthmocap_t5, synthmocap_clip = None, None

if portrait_prompts:
    portrait_enc_cache = os.path.join(ENCODING_CACHE_DIR, f"portrait_encodings_{len(portrait_prompts)}.pt")
    portrait_t5, portrait_clip = load_or_encode(portrait_enc_cache, portrait_prompts, "portrait")

if schnell_prompts:
    schnell_enc_cache = os.path.join(ENCODING_CACHE_DIR, f"schnell_encodings_{len(schnell_prompts)}.pt")
    schnell_t5, schnell_clip = load_or_encode(schnell_enc_cache, schnell_prompts, "schnell")

if sportfashion_prompts:
    sportfashion_enc_cache = os.path.join(ENCODING_CACHE_DIR, f"sportfashion_encodings_{len(sportfashion_prompts)}.pt")
    sportfashion_t5, sportfashion_clip = load_or_encode(sportfashion_enc_cache, sportfashion_prompts, "sportfashion")

if synthmocap_prompts:
    synthmocap_enc_cache = os.path.join(ENCODING_CACHE_DIR, f"synthmocap_encodings_{len(synthmocap_prompts)}.pt")
    synthmocap_t5, synthmocap_clip = load_or_encode(synthmocap_enc_cache, synthmocap_prompts, "synthmocap")

imagenet_t5, imagenet_clip = None, None
if imagenet_prompts:
    imagenet_enc_cache = os.path.join(ENCODING_CACHE_DIR, f"imagenet_encodings_{len(imagenet_prompts)}.pt")
    imagenet_t5, imagenet_clip = load_or_encode(imagenet_enc_cache, imagenet_prompts, "imagenet")

object_relations_t5, object_relations_clip = None, None
if object_relations_prompts:
    object_relations_enc_cache = os.path.join(ENCODING_CACHE_DIR,
                                              f"object_relations_encodings_{len(object_relations_prompts)}.pt")
    object_relations_t5, object_relations_clip = load_or_encode(object_relations_enc_cache, object_relations_prompts,
                                                                "object_relations")

# ============================================================================
# EXTRACT/LOAD LUNE AND SOL FEATURES (precached)
# ============================================================================
print("\n" + "=" * 60)
print("Expert Feature Caching (Lune + Sol)")
print("=" * 60)

# Lune caches
schnell_lune_cache = None
portrait_lune_cache = None
sportfashion_lune_cache = None
synthmocap_lune_cache = None
imagenet_lune_cache = None
object_relations_lune_cache = None

# Sol caches
schnell_sol_cache = None
portrait_sol_cache = None
sportfashion_sol_cache = None
synthmocap_sol_cache = None
imagenet_sol_cache = None
object_relations_sol_cache = None

if schnell_prompts:
    if ENABLE_LUNE_DISTILLATION:
        schnell_lune_path = os.path.join(ENCODING_CACHE_DIR, f"schnell_lune_{len(schnell_prompts)}.pt")
        schnell_lune_cache = load_or_extract_lune_features(
            schnell_lune_path, schnell_prompts, "schnell",
            clip_tok, clip_enc, EXPERT_T_BUCKETS
        )
    if ENABLE_SOL_DISTILLATION:
        schnell_sol_path = os.path.join(ENCODING_CACHE_DIR, f"schnell_sol_{len(schnell_prompts)}.pt")
        schnell_sol_cache = load_or_extract_sol_features(
            schnell_sol_path, schnell_prompts, "schnell",
            clip_tok, clip_enc, EXPERT_T_BUCKETS, SOL_SPATIAL_SIZE
        )

if portrait_prompts:
    if ENABLE_LUNE_DISTILLATION:
        portrait_lune_path = os.path.join(ENCODING_CACHE_DIR, f"portrait_lune_{len(portrait_prompts)}.pt")
        portrait_lune_cache = load_or_extract_lune_features(
            portrait_lune_path, portrait_prompts, "portrait",
            clip_tok, clip_enc, EXPERT_T_BUCKETS
        )
    if ENABLE_SOL_DISTILLATION:
        portrait_sol_path = os.path.join(ENCODING_CACHE_DIR, f"portrait_sol_{len(portrait_prompts)}.pt")
        portrait_sol_cache = load_or_extract_sol_features(
            portrait_sol_path, portrait_prompts, "portrait",
            clip_tok, clip_enc, EXPERT_T_BUCKETS, SOL_SPATIAL_SIZE
        )

if sportfashion_prompts:
    if ENABLE_LUNE_DISTILLATION:
        sportfashion_lune_path = os.path.join(ENCODING_CACHE_DIR, f"sportfashion_lune_{len(sportfashion_prompts)}.pt")
        sportfashion_lune_cache = load_or_extract_lune_features(
            sportfashion_lune_path, sportfashion_prompts, "sportfashion",
            clip_tok, clip_enc, EXPERT_T_BUCKETS
        )
    if ENABLE_SOL_DISTILLATION:
        sportfashion_sol_path = os.path.join(ENCODING_CACHE_DIR, f"sportfashion_sol_{len(sportfashion_prompts)}.pt")
        sportfashion_sol_cache = load_or_extract_sol_features(
            sportfashion_sol_path, sportfashion_prompts, "sportfashion",
            clip_tok, clip_enc, EXPERT_T_BUCKETS, SOL_SPATIAL_SIZE
        )

if synthmocap_prompts:
    if ENABLE_LUNE_DISTILLATION:
        synthmocap_lune_path = os.path.join(ENCODING_CACHE_DIR, f"synthmocap_lune_{len(synthmocap_prompts)}.pt")
        synthmocap_lune_cache = load_or_extract_lune_features(
            synthmocap_lune_path, synthmocap_prompts, "synthmocap",
            clip_tok, clip_enc, EXPERT_T_BUCKETS
        )
    if ENABLE_SOL_DISTILLATION:
        synthmocap_sol_path = os.path.join(ENCODING_CACHE_DIR, f"synthmocap_sol_{len(synthmocap_prompts)}.pt")
        synthmocap_sol_cache = load_or_extract_sol_features(
            synthmocap_sol_path, synthmocap_prompts, "synthmocap",
            clip_tok, clip_enc, EXPERT_T_BUCKETS, SOL_SPATIAL_SIZE
        )

if imagenet_prompts:
    if ENABLE_LUNE_DISTILLATION:
        imagenet_lune_path = os.path.join(ENCODING_CACHE_DIR, f"imagenet_lune_{len(imagenet_prompts)}.pt")
        imagenet_lune_cache = load_or_extract_lune_features(
            imagenet_lune_path, imagenet_prompts, "imagenet",
            clip_tok, clip_enc, EXPERT_T_BUCKETS
        )
    if ENABLE_SOL_DISTILLATION:
        imagenet_sol_path = os.path.join(ENCODING_CACHE_DIR, f"imagenet_sol_{len(imagenet_prompts)}.pt")
        imagenet_sol_cache = load_or_extract_sol_features(
            imagenet_sol_path, imagenet_prompts, "imagenet",
            clip_tok, clip_enc, EXPERT_T_BUCKETS, SOL_SPATIAL_SIZE
        )

if object_relations_prompts:
    if ENABLE_LUNE_DISTILLATION:
        object_relations_lune_path = os.path.join(ENCODING_CACHE_DIR,
                                                  f"object_relations_lune_{len(object_relations_prompts)}.pt")
        object_relations_lune_cache = load_or_extract_lune_features(
            object_relations_lune_path, object_relations_prompts, "object_relations",
            clip_tok, clip_enc, EXPERT_T_BUCKETS
        )
    if ENABLE_SOL_DISTILLATION:
        object_relations_sol_path = os.path.join(ENCODING_CACHE_DIR,
                                                 f"object_relations_sol_{len(object_relations_prompts)}.pt")
        object_relations_sol_cache = load_or_extract_sol_features(
            object_relations_sol_path, object_relations_prompts, "object_relations",
            clip_tok, clip_enc, EXPERT_T_BUCKETS, SOL_SPATIAL_SIZE
        )


# ============================================================================
# COMBINED DATASET CLASS
# ============================================================================
class CombinedDataset(Dataset):
    """Combined dataset returning sample index for expert feature lookup."""

    def __init__(
            self,
            portrait_ds, portrait_indices, portrait_t5, portrait_clip,
            schnell_ds, schnell_t5, schnell_clip,
            sportfashion_ds, sportfashion_latents, sportfashion_masks, sportfashion_t5, sportfashion_clip,
            synthmocap_ds, synthmocap_latents, synthmocap_masks, synthmocap_t5, synthmocap_clip,
            imagenet_ds, imagenet_latents, imagenet_t5, imagenet_clip,
            object_relations_ds, object_relations_latents, object_relations_t5, object_relations_clip,
            vae, vae_scale, device, dtype,
            compute_masks=True,
    ):
        self.portrait_ds = portrait_ds
        self.portrait_indices = portrait_indices
        self.portrait_t5 = portrait_t5
        self.portrait_clip = portrait_clip

        self.schnell_ds = schnell_ds
        self.schnell_t5 = schnell_t5
        self.schnell_clip = schnell_clip

        self.sportfashion_ds = sportfashion_ds
        self.sportfashion_latents = sportfashion_latents
        self.sportfashion_masks = sportfashion_masks
        self.sportfashion_t5 = sportfashion_t5
        self.sportfashion_clip = sportfashion_clip

        self.synthmocap_ds = synthmocap_ds
        self.synthmocap_latents = synthmocap_latents
        self.synthmocap_masks = synthmocap_masks
        self.synthmocap_t5 = synthmocap_t5
        self.synthmocap_clip = synthmocap_clip

        self.imagenet_ds = imagenet_ds
        self.imagenet_latents = imagenet_latents
        self.imagenet_t5 = imagenet_t5
        self.imagenet_clip = imagenet_clip

        self.object_relations_ds = object_relations_ds
        self.object_relations_latents = object_relations_latents
        self.object_relations_t5 = object_relations_t5
        self.object_relations_clip = object_relations_clip

        self.vae = vae
        self.vae_scale = vae_scale
        self.device = device
        self.dtype = dtype
        self.compute_masks = compute_masks

        self.n_portrait = len(portrait_indices) if portrait_indices else 0
        self.n_schnell = len(schnell_ds) if schnell_ds else 0
        self.n_sportfashion = len(sportfashion_latents) if sportfashion_latents is not None else 0
        self.n_synthmocap = len(synthmocap_latents) if synthmocap_latents is not None else 0
        self.n_imagenet = len(imagenet_latents) if imagenet_latents is not None else 0
        self.n_object_relations = len(object_relations_latents) if object_relations_latents is not None else 0

        self.c1 = self.n_portrait
        self.c2 = self.c1 + self.n_schnell
        self.c3 = self.c2 + self.n_sportfashion
        self.c4 = self.c3 + self.n_synthmocap
        self.c5 = self.c4 + self.n_imagenet
        self.total = self.c5 + self.n_object_relations

    def __len__(self):
        return self.total

    def _get_latent_from_array(self, latent_data):
        if isinstance(latent_data, torch.Tensor):
            return latent_data.to(self.dtype)
        return torch.tensor(np.array(latent_data), dtype=self.dtype)

    def __getitem__(self, idx):
        mask = None

        if idx < self.c1:
            local_idx = idx
            orig_idx = self.portrait_indices[idx]
            item = self.portrait_ds[orig_idx]
            latent = self._get_latent_from_array(item["latent"])
            t5 = self.portrait_t5[idx]
            clip = self.portrait_clip[idx]
            dataset_id = 0

        elif idx < self.c2:
            local_idx = idx - self.c1
            item = self.schnell_ds[local_idx]
            latent = self._get_latent_from_array(item["latent"])
            t5 = self.schnell_t5[local_idx]
            clip = self.schnell_clip[local_idx]
            dataset_id = 1

        elif idx < self.c3:
            local_idx = idx - self.c2
            latent = self.sportfashion_latents[local_idx].to(self.dtype)
            t5 = self.sportfashion_t5[local_idx]
            clip = self.sportfashion_clip[local_idx]
            dataset_id = 2
            if self.compute_masks and self.sportfashion_masks is not None:
                mask = self.sportfashion_masks[local_idx].to(self.dtype)

        elif idx < self.c4:
            local_idx = idx - self.c3
            latent = self.synthmocap_latents[local_idx].to(self.dtype)
            t5 = self.synthmocap_t5[local_idx]
            clip = self.synthmocap_clip[local_idx]
            dataset_id = 3
            if self.compute_masks and self.synthmocap_masks is not None:
                mask = self.synthmocap_masks[local_idx].to(self.dtype)

        elif idx < self.c5:
            local_idx = idx - self.c4
            latent = self.imagenet_latents[local_idx].to(self.dtype)
            t5 = self.imagenet_t5[local_idx]
            clip = self.imagenet_clip[local_idx]
            dataset_id = 4

        else:
            local_idx = idx - self.c5
            latent = self.object_relations_latents[local_idx].to(self.dtype)
            t5 = self.object_relations_t5[local_idx]
            clip = self.object_relations_clip[local_idx]
            dataset_id = 5

        result = {
            "latent": latent,
            "t5_embed": t5.to(self.dtype),
            "clip_pooled": clip.to(self.dtype),
            "sample_idx": idx,
            "local_idx": local_idx,
            "dataset_id": dataset_id,
        }

        if mask is not None:
            result["mask"] = mask.to(self.dtype)

        return result


# ============================================================================
# COLLATE FUNCTION
# ============================================================================
def collate_fn(batch):
    latents = torch.stack([b["latent"] for b in batch])
    t5_embeds = torch.stack([b["t5_embed"] for b in batch])
    clip_pooled = torch.stack([b["clip_pooled"] for b in batch])
    sample_indices = torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long)
    local_indices = torch.tensor([b["local_idx"] for b in batch], dtype=torch.long)
    dataset_ids = torch.tensor([b["dataset_id"] for b in batch], dtype=torch.long)

    masks = None
    if any("mask" in b for b in batch):
        masks = []
        for b in batch:
            if "mask" in b:
                masks.append(b["mask"])
            else:
                masks.append(torch.ones(64, 64, dtype=latents.dtype))
        masks = torch.stack(masks)

    return {
        "latents": latents,
        "t5_embeds": t5_embeds,
        "clip_pooled": clip_pooled,
        "sample_indices": sample_indices,
        "local_indices": local_indices,
        "dataset_ids": dataset_ids,
        "masks": masks,
    }


# ============================================================================
# EXPERT FEATURE LOOKUP (handles multiple datasets, dual experts)
# ============================================================================
def get_lune_features_for_batch(
        local_indices: torch.Tensor,
        dataset_ids: torch.Tensor,
        timesteps: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Get Lune features from the appropriate cache for each sample."""
    caches = [
        portrait_lune_cache, schnell_lune_cache, sportfashion_lune_cache,
        synthmocap_lune_cache, imagenet_lune_cache, object_relations_lune_cache
    ]

    if not any(c is not None for c in caches):
        return None

    B = local_indices.shape[0]
    device = timesteps.device
    features = torch.zeros(B, LUNE_DIM, device=device, dtype=DTYPE)

    for ds_id, cache in enumerate(caches):
        if cache is None:
            continue
        mask = dataset_ids == ds_id
        if not mask.any():
            continue
        ds_local_indices = local_indices[mask]
        ds_timesteps = timesteps[mask]
        ds_features = cache.get_features(ds_local_indices, ds_timesteps)
        features[mask] = ds_features

    return features


def get_sol_features_for_batch(
        local_indices: torch.Tensor,
        dataset_ids: torch.Tensor,
        timesteps: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Get Sol features (stats + spatial) from the appropriate cache."""
    caches = [
        portrait_sol_cache, schnell_sol_cache, sportfashion_sol_cache,
        synthmocap_sol_cache, imagenet_sol_cache, object_relations_sol_cache
    ]

    if not any(c is not None for c in caches):
        return None, None

    B = local_indices.shape[0]
    device = timesteps.device
    stats = torch.zeros(B, 3, device=device, dtype=DTYPE)  # 3 stats: locality, entropy, clustering
    spatial = torch.zeros(B, SOL_SPATIAL_SIZE, SOL_SPATIAL_SIZE, device=device, dtype=DTYPE)

    for ds_id, cache in enumerate(caches):
        if cache is None:
            continue
        mask = dataset_ids == ds_id
        if not mask.any():
            continue
        ds_local_indices = local_indices[mask]
        ds_timesteps = timesteps[mask]
        ds_stats, ds_spatial = cache.get_features(ds_local_indices, ds_timesteps)
        stats[mask] = ds_stats[:, :3]  # Drop redundant sparsity (was copy of locality)
        spatial[mask] = ds_spatial

    return stats, spatial


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
def huber_loss(pred, target, delta=0.1):
    """Huber loss - L2 for small errors, L1 for large."""
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


def compute_main_loss(pred, target, mask=None, spatial_weights=None,
                      fg_weight=2.0, bg_weight=0.5, snr_weights=None):
    """Compute main prediction loss with optional spatial weighting."""
    B, N, C = pred.shape

    if USE_HUBER_LOSS:
        loss_per_elem = huber_loss(pred, target, HUBER_DELTA)
    else:
        loss_per_elem = (pred - target) ** 2

    # Apply spatial weights from Sol if enabled
    if spatial_weights is not None and USE_SPATIAL_WEIGHTING:
        H = W = int(math.sqrt(N))
        # Upsample spatial weights from 8x8 to HxW
        spatial_upsampled = F.interpolate(
            spatial_weights.unsqueeze(1),  # [B, 1, 8, 8]
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [B, H, W]
        # Normalize so mean = 1
        spatial_upsampled = spatial_upsampled / (spatial_upsampled.mean(dim=[1, 2], keepdim=True) + 1e-6)
        spatial_flat = spatial_upsampled.view(B, N, 1)
        loss_per_elem = loss_per_elem * spatial_flat

    # Apply foreground/background mask
    if mask is not None:
        H = W = int(math.sqrt(N))
        mask_flat = mask.view(B, H * W, 1).to(pred.device)
        weights = mask_flat * fg_weight + (1 - mask_flat) * bg_weight
        loss_per_elem = loss_per_elem * weights

    loss_per_sample = loss_per_elem.mean(dim=[1, 2])

    if snr_weights is not None:
        loss_per_sample = loss_per_sample * snr_weights

    return loss_per_sample.mean()


def compute_lune_loss(pred, target, mode="cosine"):
    """Compute Lune distillation loss."""
    if mode == "cosine":
        # Cosine similarity loss (1 - cos_sim)
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        return (1 - (pred_norm * target_norm).sum(dim=-1)).mean()
    elif mode == "huber":
        return huber_loss(pred, target, HUBER_DELTA).mean()
    elif mode == "soft":
        # Soft L2 with temperature
        return F.mse_loss(pred / 10.0, target / 10.0)
    else:  # hard
        return F.mse_loss(pred, target)


def compute_sol_loss(pred_stats, pred_spatial, target_stats, target_spatial):
    """Compute Sol distillation loss (stats + spatial)."""
    stats_loss = F.mse_loss(pred_stats, target_stats)
    spatial_loss = F.mse_loss(pred_spatial, target_spatial)
    return stats_loss + spatial_loss


# ============================================================================
# WEIGHT SCHEDULES
# ============================================================================
def get_lune_weight(step):
    if step < LUNE_WARMUP_STEPS:
        return LUNE_LOSS_WEIGHT * (step / LUNE_WARMUP_STEPS)
    return LUNE_LOSS_WEIGHT


def get_sol_weight(step):
    if step < SOL_WARMUP_STEPS:
        return SOL_LOSS_WEIGHT * (step / SOL_WARMUP_STEPS)
    return SOL_LOSS_WEIGHT


# ============================================================================
# CREATE DATASET
# ============================================================================
print("\nCreating combined dataset...")
combined_ds = CombinedDataset(
    portrait_ds, portrait_indices, portrait_t5, portrait_clip,
    schnell_ds, schnell_t5, schnell_clip,
    sportfashion_ds, sportfashion_latents, sportfashion_masks, sportfashion_t5, sportfashion_clip,
    synthmocap_ds, synthmocap_latents, synthmocap_masks, synthmocap_t5, synthmocap_clip,
    imagenet_ds, imagenet_latents, imagenet_t5, imagenet_clip,
    object_relations_ds, object_relations_latents, object_relations_t5, object_relations_clip,
    vae, VAE_SCALE, DEVICE, DTYPE,
    compute_masks=USE_MASKED_LOSS,
)
print(f"✓ Combined dataset: {len(combined_ds)} samples")
print(f"  - Portraits (3x):    {combined_ds.n_portrait:,}")
print(f"  - Schnell teacher:   {combined_ds.n_schnell:,}")
print(f"  - SportFashion:      {combined_ds.n_sportfashion:,}")
print(f"  - SynthMoCap:        {combined_ds.n_synthmocap:,}")
print(f"  - ImageNet:          {combined_ds.n_imagenet:,}")
print(f"  - Object Relations:  {combined_ds.n_object_relations:,}")
print(f"  - Lune distillation: {ENABLE_LUNE_DISTILLATION}")
print(f"  - Sol distillation:  {ENABLE_SOL_DISTILLATION}")

# ============================================================================
# DATALOADER
# ============================================================================
loader = DataLoader(
    combined_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=collate_fn,
    drop_last=True,
)
print(f"✓ DataLoader: {len(loader)} batches/epoch")


# ============================================================================
# SAMPLING FUNCTION
# ============================================================================
@torch.inference_mode()
def generate_samples(model, prompts, num_steps=28, guidance_scale=5.0, H=64, W=64,
                     use_ema=True, seed=None,
                     negative_prompt="blurry, distorted, low quality"):
    """Generate samples during training with proper CFG support."""
    was_training = model.training
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    model_ref = model._orig_mod if hasattr(model, '_orig_mod') else model

    if use_ema and 'ema' in globals() and ema is not None:
        ema.apply_shadow_for_eval(model)

    B = len(prompts)
    C = 16

    t5_list, clip_list = [], []
    for p in prompts:
        t5, clip = encode_prompt(p)
        t5_list.append(t5)
        clip_list.append(clip)
    t5_cond = torch.stack(t5_list).to(DTYPE)
    clip_cond = torch.stack(clip_list).to(DTYPE)

    if guidance_scale > 1.0:
        t5_uncond, clip_uncond = encode_prompt(negative_prompt)
        t5_uncond = t5_uncond.expand(B, -1, -1)
        clip_uncond = clip_uncond.expand(B, -1)
    else:
        t5_uncond, clip_uncond = None, None

    x = torch.randn(B, H * W, C, device=DEVICE, dtype=DTYPE)
    img_ids = model_ref.create_img_ids(B, H, W, DEVICE)

    t_linear = torch.linspace(0, 1, num_steps + 1, device=DEVICE, dtype=DTYPE)
    timesteps = flux_shift(t_linear, s=SHIFT)

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_curr

        t_batch = t_curr.expand(B).to(DTYPE)

        with torch.autocast("cuda", dtype=DTYPE):
            v_cond = model_ref(
                hidden_states=x,
                encoder_hidden_states=t5_cond,
                pooled_projections=clip_cond,
                timestep=t_batch,
                img_ids=img_ids,
            )
            if isinstance(v_cond, tuple):
                v_cond = v_cond[0]

            if guidance_scale > 1.0 and t5_uncond is not None:
                v_uncond = model_ref(
                    hidden_states=x,
                    encoder_hidden_states=t5_uncond,
                    pooled_projections=clip_uncond,
                    timestep=t_batch,
                    img_ids=img_ids,
                )
                if isinstance(v_uncond, tuple):
                    v_uncond = v_uncond[0]
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond

        x = x + v * dt

    latents = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
    latents = latents / VAE_SCALE

    with torch.autocast("cuda", dtype=DTYPE):
        images = vae.decode(latents.to(vae.dtype)).sample
    images = (images / 2 + 0.5).clamp(0, 1)

    if use_ema and 'ema' in globals() and ema is not None:
        ema.restore(model)

    if was_training:
        model.train()
    return images


def save_samples(images, prompts, step, output_dir):
    from torchvision.utils import save_image
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_path = os.path.join(output_dir, f"samples_step_{step}.png")
    save_image(images, grid_path, nrow=2, padding=2)
    try:
        api.upload_file(
            path_or_fileobj=grid_path,
            path_in_repo=f"samples/{timestamp}_step_{step}.png",
            repo_id=HF_REPO,
        )
    except:
        pass


# ============================================================================
# CHECKPOINT FUNCTIONS
# ============================================================================
def save_checkpoint(model, optimizer, scheduler, step, epoch, loss, path, ema=None):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    if hasattr(model, '_orig_mod'):
        state_dict = model._orig_mod.state_dict()
    else:
        state_dict = model.state_dict()
    state_dict = {k: v.to(DTYPE) if v.is_floating_point() else v for k, v in state_dict.items()}
    weights_path = path.replace(".pt", ".safetensors")
    save_file(state_dict, weights_path)
    if ema is not None:
        ema_weights = {k: v.to(DTYPE) if v.is_floating_point() else v for k, v in ema.shadow.items()}
        ema_weights_path = path.replace(".pt", "_ema.safetensors")
        save_file(ema_weights, ema_weights_path)
    state = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if ema is not None:
        state["ema_decay"] = ema.decay
    torch.save(state, path)
    print(f"  ✓ Saved checkpoint: step {step}")
    return weights_path


def upload_checkpoint(weights_path, step):
    try:
        api.upload_file(
            path_or_fileobj=weights_path,
            path_in_repo=f"checkpoints/step_{step}.safetensors",
            repo_id=HF_REPO,
        )
        ema_path = weights_path.replace(".safetensors", "_ema.safetensors")
        if os.path.exists(ema_path):
            api.upload_file(
                path_or_fileobj=ema_path,
                path_in_repo=f"checkpoints/step_{step}_ema.safetensors",
                repo_id=HF_REPO,
            )
        print(f"  ✓ Uploaded checkpoint to {HF_REPO}")
    except Exception as e:
        print(f"  ⚠ Upload failed: {e}")


def upload_logs():
    try:
        for root, dirs, files in os.walk(LOG_DIR):
            for f in files:
                if f.startswith("events.out.tfevents"):
                    local_path = os.path.join(root, f)
                    rel_path = os.path.relpath(local_path, LOG_DIR)
                    repo_path = f"logs/{rel_path}"
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=repo_path,
                        repo_id=HF_REPO,
                    )
        print(f"  ✓ Uploaded logs to {HF_REPO}")
    except Exception as e:
        print(f"  ⚠ Log upload failed: {e}")


# ============================================================================
# WEIGHT UPGRADE LOADING (v3 -> v4.1)
# ============================================================================


def load_with_weight_upgrade(model, state_dict):
    """Load state dict with bidirectional remapping support.

    Handles:
    - v3 checkpoint (expert_predictor) -> v4 model (lune_predictor)
    - v4 checkpoint (lune_predictor) -> model with (expert_predictor)
    """
    model_state = model.state_dict()

    # Detect which naming the MODEL uses
    model_has_expert = any('expert_predictor' in k for k in model_state.keys())
    model_has_lune = any('lune_predictor' in k for k in model_state.keys())

    # Detect which naming the CHECKPOINT uses
    ckpt_has_expert = any('expert_predictor' in k for k in state_dict.keys())
    ckpt_has_lune = any('lune_predictor' in k for k in state_dict.keys())

    # Build remap based on mismatch
    REMAP = {}
    if model_has_expert and ckpt_has_lune:
        # Checkpoint has lune_predictor, model expects expert_predictor
        print("  Remapping: lune_predictor -> expert_predictor")
        REMAP = {'lune_predictor.': 'expert_predictor.'}
    elif model_has_lune and ckpt_has_expert:
        # Checkpoint has expert_predictor, model expects lune_predictor
        print("  Remapping: expert_predictor -> lune_predictor")
        REMAP = {'expert_predictor.': 'lune_predictor.'}

    # New modules that may not exist in checkpoint
    NEW_WEIGHT_PATTERNS = [
        'expert_predictor.',
        'lune_predictor.',
        'sol_prior.',
        't5_vec_proj.',
        '.norm_q.weight',
        '.norm_k.weight',
        '.norm_added_q.weight',
        '.norm_added_k.weight',
    ]

    # Deprecated keys
    DEPRECATED_PATTERNS = [
        'guidance_in.',
        '.sin_basis',
    ]

    loaded_keys = []
    missing_keys = []
    unexpected_keys = []
    initialized_keys = []
    remapped_keys = []

    # First pass: remap checkpoint keys to match model
    remapped_state = {}
    for k, v in state_dict.items():
        new_k = k
        for old_pat, new_pat in REMAP.items():
            if old_pat in k:
                new_k = k.replace(old_pat, new_pat)
                remapped_keys.append(f"{k} -> {new_k}")
                break
        remapped_state[new_k] = v

    # Second pass: load matching weights
    for key, v in remapped_state.items():
        if key in model_state:
            if v.shape == model_state[key].shape:
                model_state[key] = v
                loaded_keys.append(key)
            else:
                print(f"  ⚠ Shape mismatch for {key}: checkpoint {v.shape} vs model {model_state[key].shape}")
                unexpected_keys.append(key)
        else:
            is_deprecated = any(pat in key for pat in DEPRECATED_PATTERNS)
            if is_deprecated:
                unexpected_keys.append(key)
            else:
                print(f"  ⚠ Unexpected key (not in model): {key}")
                unexpected_keys.append(key)

    # Third pass: handle missing keys
    for key in model_state.keys():
        if key not in loaded_keys:
            is_new = any(pat in key for pat in NEW_WEIGHT_PATTERNS)
            if is_new:
                initialized_keys.append(key)
            else:
                missing_keys.append(key)
                print(f"  ⚠ Missing key (not in checkpoint): {key}")

    model.load_state_dict(model_state, strict=False)

    # Report
    if remapped_keys:
        print(f"  ✓ Remapped v3->v4: {len(remapped_keys)} keys")
        for rk in remapped_keys[:5]:
            print(f"      {rk}")
        if len(remapped_keys) > 5:
            print(f"      ... and {len(remapped_keys) - 5} more")

    if initialized_keys:
        modules = set()
        for k in initialized_keys:
            parts = k.split('.')
            if len(parts) >= 2:
                modules.add(parts[0])
        print(f"  ✓ Initialized new modules (fresh): {sorted(modules)}")

    if unexpected_keys:
        deprecated = [k for k in unexpected_keys if any(p in k for p in DEPRECATED_PATTERNS)]
        if deprecated:
            print(f"  ✓ Ignored deprecated keys: {len(deprecated)}")

    return missing_keys, unexpected_keys


def load_checkpoint(model, optimizer, scheduler, target):
    """Load checkpoint with weight upgrade support for v4.1."""
    start_step = 0
    start_epoch = 0
    ema_state = None

    if target == "none":
        print("Starting fresh (no checkpoint)")
        return start_step, start_epoch, None

    ckpt_path = None
    weights_path = None
    ema_weights_path = None

    if target == "latest":
        if os.path.exists(CHECKPOINT_DIR):
            ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("step_") and f.endswith(".pt")]
            if ckpts:
                steps = [int(f.split("_")[1].split(".")[0]) for f in ckpts]
                latest_step = max(steps)
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{latest_step}.pt")
                weights_path = ckpt_path.replace(".pt", ".safetensors")
                ema_weights_path = ckpt_path.replace(".pt", "_ema.safetensors")

    elif target == "hub" or target.startswith("hub:"):
        try:
            from huggingface_hub import list_repo_files

            if target.startswith("hub:"):
                path_or_name = target.split(":", 1)[1]

                # Check if it's a full path (contains /) or just a step name
                if "/" in path_or_name:
                    # Full path like checkpoint_runs/v4_init/lailah_401434_v4_init
                    weights_path = hf_hub_download(HF_REPO, f"{path_or_name}.safetensors")
                    try:
                        ema_weights_path = hf_hub_download(HF_REPO, f"{path_or_name}_ema.safetensors")
                        print(f"  Found EMA weights on hub")
                    except:
                        ema_weights_path = None
                        print(f"  No EMA weights on hub (will start fresh)")
                    print(f"Downloaded {path_or_name} from hub")
                else:
                    # Simple step name like step_401434
                    step_name = path_or_name
                    weights_path = hf_hub_download(HF_REPO, f"checkpoints/{step_name}.safetensors")
                    try:
                        ema_weights_path = hf_hub_download(HF_REPO, f"checkpoints/{step_name}_ema.safetensors")
                        print(f"  Found EMA weights on hub")
                    except:
                        ema_weights_path = None
                        print(f"  No EMA weights on hub (will start fresh)")
                    start_step = int(step_name.split("_")[1]) if "_" in step_name else 0
                    print(f"Downloaded {step_name} from hub")
            else:
                files = list_repo_files(HF_REPO)
                ckpts = [f for f in files if
                         f.startswith("checkpoints/step_") and f.endswith(".safetensors") and "_ema" not in f]
                if ckpts:
                    steps = [int(f.split("_")[1].split(".")[0]) for f in ckpts]
                    latest = max(steps)
                    weights_path = hf_hub_download(HF_REPO, f"checkpoints/step_{latest}.safetensors")
                    try:
                        ema_weights_path = hf_hub_download(HF_REPO, f"checkpoints/step_{latest}_ema.safetensors")
                        print(f"  Found EMA weights on hub")
                    except:
                        ema_weights_path = None
                        print(f"  No EMA weights on hub (will start fresh)")
                    start_step = latest
                    print(f"Downloaded step_{latest} from hub")
        except Exception as e:
            print(f"Could not download from hub: {e}")
            return start_step, start_epoch, None

    elif target == "best":
        ckpt_path = os.path.join(CHECKPOINT_DIR, "best.pt")
        weights_path = ckpt_path.replace(".pt", ".safetensors")
        ema_weights_path = ckpt_path.replace(".pt", "_ema.safetensors")

    elif os.path.exists(target):
        if target.endswith(".safetensors"):
            weights_path = target
            ckpt_path = target.replace(".safetensors", ".pt")
            ema_weights_path = target.replace(".safetensors", "_ema.safetensors")
        else:
            ckpt_path = target
            weights_path = target.replace(".pt", ".safetensors")
            ema_weights_path = target.replace(".pt", "_ema.safetensors")

    # Load main model weights
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        state_dict = load_file(weights_path)
        state_dict = {k: v.to(DTYPE) if v.is_floating_point() else v for k, v in state_dict.items()}

        model_ref = model._orig_mod if hasattr(model, '_orig_mod') else model

        if ALLOW_WEIGHT_UPGRADE:
            missing, unexpected = load_with_weight_upgrade(model_ref, state_dict)
            if missing:
                print(f"  ⚠ {len(missing)} truly missing parameters")
        else:
            model_ref.load_state_dict(state_dict, strict=True)

        print(f"✓ Loaded model weights")

        # Load EMA weights
        if ema_weights_path and os.path.exists(ema_weights_path):
            ema_state = load_file(ema_weights_path)
            ema_state = {k: v.to(DTYPE) if v.is_floating_point() else v for k, v in ema_state.items()}
            print(f"✓ Loaded EMA weights ({len(ema_state)} params)")
        else:
            print(f"  ℹ No EMA weights found (will initialize fresh)")
    else:
        print(f"  ⚠ Weights file not found: {weights_path}")
        print(f"  Starting with fresh model")
        return start_step, start_epoch, None

    # Load optimizer/scheduler state
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        start_step = state.get("step", 0)
        start_epoch = state.get("epoch", 0)
        try:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            print(f"✓ Loaded optimizer/scheduler state")
        except Exception as e:
            print(f"  ⚠ Could not load optimizer state: {e}")
            print(f"  Will use fresh optimizer")
        print(f"Resuming from step {start_step}, epoch {start_epoch}")

    return start_step, start_epoch, ema_state


# ============================================================================
# CREATE MODEL (v4.1 with dual experts)
# ============================================================================
print("\nCreating TinyFlux v4.1 model with Lune + Sol...")

# Import model - expects model_v4.py to define TinyFluxConfig and TinyFlux
# If running as a notebook cell, ensure model_v4.py cell was run first
# If running as a script, uncomment the import below:
# from model_v4 import TinyFluxConfig, TinyFlux

config = TinyFluxConfig(
    hidden_size=512,
    num_attention_heads=4,
    attention_head_dim=128,
    num_double_layers=15,
    num_single_layers=25,

    # Lune expert (trajectory guidance)
    use_lune_expert=ENABLE_LUNE_DISTILLATION,
    lune_expert_dim=LUNE_DIM,
    lune_hidden_dim=LUNE_HIDDEN_DIM,
    lune_dropout=LUNE_DROPOUT,

    # Sol prior (structural guidance)
    use_sol_prior=ENABLE_SOL_DISTILLATION,
    sol_spatial_size=SOL_SPATIAL_SIZE,
    sol_hidden_dim=SOL_HIDDEN_DIM,
    sol_geometric_weight=SOL_GEOMETRIC_WEIGHT,

    # Other settings
    use_t5_vec=True,
    lune_distill_mode=LUNE_DISTILL_MODE,
    use_huber_loss=USE_HUBER_LOSS,
    huber_delta=HUBER_DELTA,
    guidance_embeds=False,
)
model = TinyFluxDeep(config).to(device=DEVICE, dtype=DTYPE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

if hasattr(model, 'lune_predictor') and model.lune_predictor is not None:
    lune_params = sum(p.numel() for p in model.lune_predictor.parameters())
    print(f"Lune predictor parameters: {lune_params:,}")

if hasattr(model, 'sol_prior') and model.sol_prior is not None:
    sol_params = sum(p.numel() for p in model.sol_prior.parameters())
    print(f"Sol prior parameters: {sol_params:,}")

trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

# ============================================================================
# OPTIMIZER
# ============================================================================
opt = torch.optim.AdamW(trainable_params, lr=LR, betas=(0.9, 0.99), weight_decay=0.01, fused=True)

total_steps = len(loader) * EPOCHS // GRAD_ACCUM
warmup = min(1000, total_steps // 10)


def lr_fn(step):
    if step < warmup:
        return step / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

# ============================================================================
# LOAD CHECKPOINT
# ============================================================================
start_step, start_epoch, ema_state = load_checkpoint(model, opt, sched, LOAD_TARGET)

if RESUME_STEP is not None:
    start_step = RESUME_STEP

# ============================================================================
# COMPILE
# ============================================================================
model = torch.compile(model, mode="default")

# ============================================================================
# EMA
# ============================================================================
print("Initializing EMA...")
ema = EMA(model, decay=EMA_DECAY)
if ema_state is not None:
    # Remap v3 EMA keys to v4
    remapped_ema = {}
    for k, v in ema_state.items():
        # if k in V3_TO_V4_REMAP:
        #    remapped_ema[V3_TO_V4_REMAP[k]] = v
        # else:
        remapped_ema[k] = v
    ema.load_shadow(remapped_ema, model=model)

    # Sync new modules from model
    has_lune_in_ema = any('lune_predictor' in k for k in ema_state.keys())
    has_sol_in_ema = any('sol_prior' in k for k in ema_state.keys())

    if ENABLE_LUNE_DISTILLATION and not has_lune_in_ema:
        # Check if expert_predictor was in the v3 checkpoint (remapped to lune_predictor)
        has_expert_in_v3 = any('expert_predictor' in k for k in ema_state.keys())
        if not has_expert_in_v3:
            ema.sync_from_model(model, pattern='lune_predictor')
            print("  ✓ Force-synced lune_predictor (new weights)")
        else:
            print("  ✓ lune_predictor loaded from remapped v3 checkpoint")

    if ENABLE_SOL_DISTILLATION and not has_sol_in_ema:
        ema.sync_from_model(model, pattern='sol_prior')
        print("  ✓ Force-synced sol_prior (new weights)")
else:
    print("  Starting fresh EMA from current weights")

# ============================================================================
# TENSORBOARD
# ============================================================================
run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(os.path.join(LOG_DIR, run_name))

SAMPLE_PROMPTS = [
    "a photo of a cat sitting on a windowsill",
    "a portrait of a woman with red hair",
    "a black backpack on white background",
    "a person standing in a t-pose",
]

# ============================================================================
# TRAINING LOOP
# ============================================================================
print(f"\n{'=' * 60}")
print(f"Training TinyFlux v4.1 with Dual Expert Distillation")
print(f"{'=' * 60}")
print(f"Total: {len(combined_ds):,} samples")
print(f"Epochs: {EPOCHS}, Steps/epoch: {len(loader)}, Total: {total_steps}")
print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"Lune distillation: {ENABLE_LUNE_DISTILLATION}")
if ENABLE_LUNE_DISTILLATION:
    print(f"  - Mode: {LUNE_DISTILL_MODE}")
    print(f"  - Weight: {LUNE_LOSS_WEIGHT} (warmup: {LUNE_WARMUP_STEPS} steps)")
print(f"Sol distillation: {ENABLE_SOL_DISTILLATION}")
if ENABLE_SOL_DISTILLATION:
    print(f"  - Weight: {SOL_LOSS_WEIGHT} (warmup: {SOL_WARMUP_STEPS} steps)")
print(f"Huber loss: {USE_HUBER_LOSS} (delta={HUBER_DELTA})")
print(f"Spatial weighting: {USE_SPATIAL_WEIGHTING}")
print(f"Resume: step {start_step}, epoch {start_epoch}")

model.train()
step = start_step
best = float("inf")

for ep in range(start_epoch, EPOCHS):
    ep_loss = 0
    ep_main_loss = 0
    ep_lune_loss = 0
    ep_sol_loss = 0
    ep_batches = 0
    pbar = tqdm(loader, desc=f"E{ep + 1}")

    for i, batch in enumerate(pbar):
        latents = batch["latents"].to(DEVICE, non_blocking=True)
        t5 = batch["t5_embeds"].to(DEVICE, non_blocking=True)
        clip = batch["clip_pooled"].to(DEVICE, non_blocking=True)
        local_indices = batch["local_indices"]
        dataset_ids = batch["dataset_ids"]
        masks = batch["masks"]

        if masks is not None:
            masks = masks.to(DEVICE, non_blocking=True)

        B, C, H, W = latents.shape
        data = latents.permute(0, 2, 3, 1).reshape(B, H * W, C)
        noise = torch.randn_like(data)

        if TEXT_DROPOUT > 0:
            t5, clip, _ = apply_text_dropout(t5, clip, TEXT_DROPOUT)

        t = torch.sigmoid(torch.randn(B, device=DEVICE))
        t = flux_shift(t, s=SHIFT).to(DTYPE).clamp(1e-4, 1 - 1e-4)

        t_expanded = t.view(B, 1, 1)
        x_t = (1 - t_expanded) * noise + t_expanded * data
        v_target = data - noise

        img_ids = TinyFluxDeep.create_img_ids(B, H, W, DEVICE)

        # Get expert features from CACHE
        lune_features = None
        sol_stats = None
        sol_spatial = None

        if ENABLE_LUNE_DISTILLATION:
            lune_features = get_lune_features_for_batch(local_indices, dataset_ids, t)
            if lune_features is not None and random.random() < LUNE_DROPOUT:
                lune_features = None

        if ENABLE_SOL_DISTILLATION:
            sol_stats, sol_spatial = get_sol_features_for_batch(local_indices, dataset_ids, t)

        with torch.autocast("cuda", dtype=DTYPE):
            result = model(
                hidden_states=x_t,
                encoder_hidden_states=t5,
                pooled_projections=clip,
                timestep=t,
                img_ids=img_ids,
                lune_features=lune_features,
                sol_stats=sol_stats,
                sol_spatial=sol_spatial,
                return_expert_pred=True,
            )

            if isinstance(result, tuple):
                v_pred, expert_info = result
            else:
                v_pred = result
                expert_info = {}

        # Compute losses
        snr_weights = min_snr_weight(t)

        # Main loss with optional spatial weighting from Sol
        spatial_weights = sol_spatial if USE_SPATIAL_WEIGHTING else None
        main_loss = compute_main_loss(
            v_pred, v_target,
            mask=masks if USE_MASKED_LOSS else None,
            spatial_weights=spatial_weights,
            fg_weight=FG_LOSS_WEIGHT,
            bg_weight=BG_LOSS_WEIGHT,
            snr_weights=snr_weights
        )

        # Lune distillation loss
        lune_loss = torch.tensor(0.0, device=DEVICE)
        if lune_features is not None and expert_info.get('lune_pred') is not None:
            lune_loss = compute_lune_loss(
                expert_info['lune_pred'], lune_features, mode=LUNE_DISTILL_MODE
            )

        # Sol distillation loss
        sol_loss = torch.tensor(0.0, device=DEVICE)
        if sol_stats is not None and expert_info.get('sol_stats_pred') is not None:
            sol_loss = compute_sol_loss(
                expert_info['sol_stats_pred'], expert_info.get('sol_spatial_pred'),
                sol_stats, sol_spatial
            )

        # Total loss with warmup weights
        total_loss = main_loss
        total_loss = total_loss + get_lune_weight(step) * lune_loss
        total_loss = total_loss + get_sol_weight(step) * sol_loss

        loss = total_loss / GRAD_ACCUM
        loss.backward()

        if (i + 1) % GRAD_ACCUM == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            ema.update(model)
            step += 1

            if step % LOG_EVERY == 0:
                writer.add_scalar("train/loss", total_loss.item(), step)
                writer.add_scalar("train/main_loss", main_loss.item(), step)
                if ENABLE_LUNE_DISTILLATION:
                    writer.add_scalar("train/lune_loss", lune_loss.item(), step)
                    writer.add_scalar("train/lune_weight", get_lune_weight(step), step)
                if ENABLE_SOL_DISTILLATION:
                    writer.add_scalar("train/sol_loss", sol_loss.item(), step)
                    writer.add_scalar("train/sol_weight", get_sol_weight(step), step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), step)

            if step % SAMPLE_EVERY == 0:
                print(f"\n  Generating samples at step {step}...")
                images = generate_samples(
                    model, SAMPLE_PROMPTS,
                    num_steps=28,
                    guidance_scale=5.0,
                    use_ema=True,
                    negative_prompt="blurry, distorted, low quality, deformed",
                )
                save_samples(images, SAMPLE_PROMPTS, step, SAMPLE_DIR)

            if step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{step}.pt")
                weights_path = save_checkpoint(model, opt, sched, step, ep, total_loss.item(), ckpt_path, ema=ema)
                if step % UPLOAD_EVERY == 0:
                    upload_checkpoint(weights_path, step)
                if step % LOG_UPLOAD_EVERY == 0:
                    writer.flush()
                    upload_logs()

        ep_loss += total_loss.item()
        ep_main_loss += main_loss.item()
        ep_lune_loss += lune_loss.item()
        ep_sol_loss += sol_loss.item()
        ep_batches += 1

        pbar.set_postfix(
            loss=f"{total_loss.item():.4f}",
            main=f"{main_loss.item():.4f}",
            lune=f"{lune_loss.item():.4f}" if ENABLE_LUNE_DISTILLATION else "-",
            sol=f"{sol_loss.item():.4f}" if ENABLE_SOL_DISTILLATION else "-",
            step=step
        )

    avg = ep_loss / max(ep_batches, 1)
    avg_main = ep_main_loss / max(ep_batches, 1)
    avg_lune = ep_lune_loss / max(ep_batches, 1)
    avg_sol = ep_sol_loss / max(ep_batches, 1)

    print(f"Epoch {ep + 1} - total: {avg:.4f}, main: {avg_main:.4f}, lune: {avg_lune:.4f}, sol: {avg_sol:.4f}")

    if avg < best:
        best = avg
        weights_path = save_checkpoint(model, opt, sched, step, ep, avg, os.path.join(CHECKPOINT_DIR, "best.pt"),
                                       ema=ema)
        try:
            api.upload_file(path_or_fileobj=weights_path, path_in_repo="model.safetensors", repo_id=HF_REPO)
        except:
            pass

print(f"\n✓ Training complete! Best loss: {best:.4f}")
writer.close()