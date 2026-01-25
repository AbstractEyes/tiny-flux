# ============================================================================
# TinyFlux-Deep Training Cell - With Expert Distillation (Precached)
# ============================================================================
# Integrates SD1.5-flow-lune as a frozen timestep expert.
# Expert features are PRECACHED at 10 timestep buckets for speed.
# The ExpertPredictor learns to emulate expert features from (t, CLIP).
# At inference, no expert needed - predictor runs standalone.
#
# USAGE: Run model.py cell first, then this cell
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
BATCH_SIZE = 16
GRAD_ACCUM = 2
LR = 3e-4
EPOCHS = 50
MAX_SEQ = 128
SHIFT = 3.0
DEVICE = "cuda"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

ALLOW_WEIGHT_UPGRADE = True

# HuggingFace Hub
HF_REPO = "AbstractPhil/tiny-flux-deep"
SAVE_EVERY = 625
UPLOAD_EVERY = 625
SAMPLE_EVERY = 312
LOG_EVERY = 10
LOG_UPLOAD_EVERY = 625

# Checkpoint loading
LOAD_TARGET = "latest"
RESUME_STEP = None

# ============================================================================
# EXPERT DISTILLATION CONFIG
# ============================================================================
ENABLE_EXPERT_DISTILLATION = True
EXPERT_CHECKPOINT = "AbstractPhil/sd15-flow-lune-flux"
EXPERT_CHECKPOINT_PATH = "flux_t2_6_pose_t4_6_port_t1_4/checkpoint-00018765/unet/diffusion_pytorch_model.safetensors"
EXPERT_DIM = 1280
EXPERT_HIDDEN_DIM = 512
EXPERT_DROPOUT = 0.1  # Prob of forcing predictor (applied outside model)
DISTILL_LOSS_WEIGHT = 0.1
DISTILL_WARMUP_STEPS = 1000

# Timestep buckets for precaching
EXPERT_T_BUCKETS = torch.linspace(0.05, 0.95, 10)

# ============================================================================
# DATASET CONFIG
# ============================================================================
ENABLE_PORTRAIT = False
ENABLE_SCHNELL = True
ENABLE_SPORTFASHION = False
ENABLE_SYNTHMOCAP = False

PORTRAIT_REPO = "AbstractPhil/ffhq_flux_latents_repaired"
PORTRAIT_NUM_SHARDS = 11
SCHNELL_REPO = "AbstractPhil/flux-schnell-teacher-latents"
SCHNELL_CONFIGS = ["train_2_512"]
SPORTFASHION_REPO = "Pianokill/SportFashion_512x512"
SYNTHMOCAP_REPO = "toyxyz/SynthMoCap_smpl_512"

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
# EXPERT FEATURE CACHE (precached, fast lookup + interpolation)
# ============================================================================

class ExpertFeatureCache:
    """
    Precached SD1.5-flow expert features with timestep interpolation.

    Features extracted at 10 timestep buckets [0.05, 0.15, ..., 0.95].
    At runtime, interpolates between nearest buckets.
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
        """
        Get interpolated expert features.

        Args:
            indices: [B] sample indices into dataset
            timesteps: [B] timesteps in [0, 1]

        Returns:
            [B, 1280] interpolated features
        """
        device = timesteps.device

        # Clamp to valid range
        t_clamped = timesteps.float().clamp(self.t_min, self.t_max)

        # Find bucket indices
        t_idx_float = (t_clamped - self.t_min) / self.t_step
        t_idx_low = t_idx_float.long().clamp(0, self.n_buckets - 2)
        t_idx_high = (t_idx_low + 1).clamp(0, self.n_buckets - 1)

        # Interpolation alpha
        alpha = (t_idx_float - t_idx_low.float()).unsqueeze(-1)  # [B, 1]

        # Gather (on CPU for large caches)
        idx_cpu = indices.cpu()
        t_low_cpu = t_idx_low.cpu()
        t_high_cpu = t_idx_high.cpu()

        f_low = self.features[idx_cpu, t_low_cpu]  # [B, 1280]
        f_high = self.features[idx_cpu, t_high_cpu]  # [B, 1280]

        # Interpolate and move to device
        result = (1 - alpha.cpu()) * f_low + alpha.cpu() * f_high
        return result.to(device=device, dtype=self.dtype)


def load_or_extract_expert_features(cache_path: str, prompts: List[str], name: str,
                                    clip_tok, clip_enc, t_buckets: torch.Tensor,
                                    batch_size: int = 32) -> Optional[ExpertFeatureCache]:
    """
    Load cached expert features or extract them from SD1.5-flow.
    Follows same pattern as load_or_encode for text embeddings.
    """
    if not prompts or not ENABLE_EXPERT_DISTILLATION:
        return None

    # Check cache
    if os.path.exists(cache_path):
        print(f"Loading cached {name} expert features...")
        cached = torch.load(cache_path, map_location="cpu")
        cache = ExpertFeatureCache(cached["features"], cached["t_buckets"], DTYPE)
        print(f"  ✓ Loaded {cache.features.shape[0]} samples × {cache.n_buckets} timesteps")
        return cache

    # Extract features
    print(f"Extracting {name} expert features ({len(prompts)} × {len(t_buckets)} timesteps)...")
    print(f"  This is a one-time operation, will be cached for future runs.")

    # Load expert model temporarily
    checkpoint_path = hf_hub_download(
        repo_id=EXPERT_CHECKPOINT,
        filename=EXPERT_CHECKPOINT_PATH,
    )

    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        subfolder="unet",
        torch_dtype=DTYPE,
    ).to(DEVICE).eval()

    state_dict = load_file(checkpoint_path)
    unet.load_state_dict(state_dict, strict=False)

    for p in unet.parameters():
        p.requires_grad = False

    # Hook for mid-block features
    mid_features = [None]

    def hook_fn(module, inp, out):
        mid_features[0] = out.mean(dim=[2, 3])

    unet.mid_block.register_forward_hook(hook_fn)

    # Extract
    n_prompts = len(prompts)
    n_buckets = len(t_buckets)
    all_features = torch.zeros(n_prompts, n_buckets, EXPERT_DIM, dtype=torch.float16)

    with torch.no_grad():
        for start_idx in tqdm(range(0, n_prompts, batch_size), desc=f"Extracting {name}"):
            end_idx = min(start_idx + batch_size, n_prompts)
            batch_prompts = prompts[start_idx:end_idx]
            B = len(batch_prompts)

            # Encode CLIP hidden states
            clip_inputs = clip_tok(
                batch_prompts, return_tensors="pt", padding="max_length",
                max_length=77, truncation=True
            ).to(DEVICE)
            clip_hidden = clip_enc(**clip_inputs).last_hidden_state  # [B, 77, 768]

            # Extract at each timestep bucket
            for t_idx, t_val in enumerate(t_buckets):
                timesteps = torch.full((B,), t_val.item(), device=DEVICE)
                latents = torch.randn(B, 4, 64, 64, device=DEVICE, dtype=DTYPE)

                _ = unet(latents, timesteps * 1000, encoder_hidden_states=clip_hidden.to(DTYPE))

                all_features[start_idx:end_idx, t_idx] = mid_features[0].cpu().to(torch.float16)

    # Cleanup
    del unet
    torch.cuda.empty_cache()

    # Save cache
    torch.save({"features": all_features, "t_buckets": t_buckets}, cache_path)
    print(f"  ✓ Cached to {cache_path}")
    print(f"  Size: {all_features.numel() * 2 / 1e9:.2f} GB")

    return ExpertFeatureCache(all_features, t_buckets, DTYPE)


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

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state['shadow'].items()}
        self.decay = state.get('decay', self.decay)

    def load_shadow(self, shadow_state):
        """Load EMA shadow weights, handling architecture changes gracefully."""
        device = next(iter(self.shadow.values())).device if self.shadow else 'cuda'

        loaded = 0
        skipped_old = 0
        kept_new = 0

        for k, v in shadow_state.items():
            if k in self.shadow:
                # Key exists in current model - load it
                self.shadow[k] = v.clone().to(device)
                loaded += 1
            else:
                # Key doesn't exist (deprecated like guidance_in)
                skipped_old += 1

        # Count new keys not in checkpoint
        for k in self.shadow:
            if k not in shadow_state:
                kept_new += 1

        print(f"  ✓ Restored EMA: {loaded} loaded, {skipped_old} deprecated skipped, {kept_new} new (fresh init)")


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
def encode_prompts_batched(prompts: List[str], batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    all_t5 = []
    all_clip = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Encoding", leave=False):
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
    print(f"\n[1/4] Loading portrait dataset from {PORTRAIT_REPO}...")
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
    print("\n[1/4] Portrait dataset DISABLED")

schnell_ds = None
schnell_prompts = []

if ENABLE_SCHNELL:
    print(f"\n[2/4] Loading schnell teacher dataset from {SCHNELL_REPO}...")
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
    print("\n[2/4] Schnell dataset DISABLED")

sportfashion_ds = None
sportfashion_prompts = []

if ENABLE_SPORTFASHION:
    print(f"\n[3/4] Loading SportFashion dataset from {SPORTFASHION_REPO}...")
    sportfashion_ds = load_dataset(SPORTFASHION_REPO, split="train")
    sportfashion_prompts = list(sportfashion_ds["text"])
    print(f"✓ SportFashion: {len(sportfashion_ds)} samples")
else:
    print("\n[3/4] SportFashion dataset DISABLED")

synthmocap_ds = None
synthmocap_prompts = []

if ENABLE_SYNTHMOCAP:
    print(f"\n[4/4] Loading SynthMoCap dataset from {SYNTHMOCAP_REPO}...")
    synthmocap_ds = load_dataset(SYNTHMOCAP_REPO, split="train")
    synthmocap_prompts = list(synthmocap_ds["text"])
    print(f"✓ SynthMoCap: {len(synthmocap_ds)} samples")
else:
    print("\n[4/4] SynthMoCap dataset DISABLED")

# ============================================================================
# ENCODE ALL PROMPTS
# ============================================================================
total_samples = len(portrait_prompts) + len(schnell_prompts) + len(sportfashion_prompts) + len(synthmocap_prompts)
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


# Standard text encodings
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

# ============================================================================
# EXTRACT/LOAD EXPERT FEATURES (precached)
# ============================================================================
print("\n" + "=" * 60)
print("Expert Feature Caching")
print("=" * 60)

schnell_expert_cache = None
portrait_expert_cache = None
sportfashion_expert_cache = None
synthmocap_expert_cache = None

if schnell_prompts and ENABLE_EXPERT_DISTILLATION:
    schnell_expert_path = os.path.join(ENCODING_CACHE_DIR, f"schnell_expert_{len(schnell_prompts)}.pt")
    schnell_expert_cache = load_or_extract_expert_features(
        schnell_expert_path, schnell_prompts, "schnell",
        clip_tok, clip_enc, EXPERT_T_BUCKETS
    )

if portrait_prompts and ENABLE_EXPERT_DISTILLATION:
    portrait_expert_path = os.path.join(ENCODING_CACHE_DIR, f"portrait_expert_{len(portrait_prompts)}.pt")
    portrait_expert_cache = load_or_extract_expert_features(
        portrait_expert_path, portrait_prompts, "portrait",
        clip_tok, clip_enc, EXPERT_T_BUCKETS
    )

if sportfashion_prompts and ENABLE_EXPERT_DISTILLATION:
    sportfashion_expert_path = os.path.join(ENCODING_CACHE_DIR, f"sportfashion_expert_{len(sportfashion_prompts)}.pt")
    sportfashion_expert_cache = load_or_extract_expert_features(
        sportfashion_expert_path, sportfashion_prompts, "sportfashion",
        clip_tok, clip_enc, EXPERT_T_BUCKETS
    )

if synthmocap_prompts and ENABLE_EXPERT_DISTILLATION:
    synthmocap_expert_path = os.path.join(ENCODING_CACHE_DIR, f"synthmocap_expert_{len(synthmocap_prompts)}.pt")
    synthmocap_expert_cache = load_or_extract_expert_features(
        synthmocap_expert_path, synthmocap_prompts, "synthmocap",
        clip_tok, clip_enc, EXPERT_T_BUCKETS
    )


# ============================================================================
# COMBINED DATASET CLASS (with sample_idx for expert lookup)
# ============================================================================
class CombinedDataset(Dataset):
    """Combined dataset returning sample index for expert feature lookup."""

    def __init__(
            self,
            portrait_ds, portrait_indices, portrait_t5, portrait_clip,
            schnell_ds, schnell_t5, schnell_clip,
            sportfashion_ds, sportfashion_t5, sportfashion_clip,
            synthmocap_ds, synthmocap_t5, synthmocap_clip,
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
        self.sportfashion_t5 = sportfashion_t5
        self.sportfashion_clip = sportfashion_clip

        self.synthmocap_ds = synthmocap_ds
        self.synthmocap_t5 = synthmocap_t5
        self.synthmocap_clip = synthmocap_clip

        self.vae = vae
        self.vae_scale = vae_scale
        self.device = device
        self.dtype = dtype
        self.compute_masks = compute_masks

        self.n_portrait = len(portrait_indices) if portrait_indices else 0
        self.n_schnell = len(schnell_ds) if schnell_ds else 0
        self.n_sportfashion = len(sportfashion_ds) if sportfashion_ds else 0
        self.n_synthmocap = len(synthmocap_ds) if synthmocap_ds else 0

        self.c1 = self.n_portrait
        self.c2 = self.c1 + self.n_schnell
        self.c3 = self.c2 + self.n_sportfashion
        self.total = self.c3 + self.n_synthmocap

    def __len__(self):
        return self.total

    def _get_latent_from_array(self, latent_data):
        if isinstance(latent_data, torch.Tensor):
            return latent_data.to(self.dtype)
        return torch.tensor(np.array(latent_data), dtype=self.dtype)

    @torch.no_grad()
    def _encode_image(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
        img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        img_tensor = (img_tensor * 2.0 - 1.0).to(self.device, dtype=self.dtype)
        latent = self.vae.encode(img_tensor).latent_dist.sample()
        latent = latent * self.vae_scale
        return latent.squeeze(0).cpu()

    def __getitem__(self, idx):
        mask = None

        # Determine which dataset and local index
        if idx < self.c1:
            # Portrait
            local_idx = idx
            orig_idx = self.portrait_indices[idx]
            item = self.portrait_ds[orig_idx]
            latent = self._get_latent_from_array(item["latent"])
            t5 = self.portrait_t5[idx]
            clip = self.portrait_clip[idx]
            dataset_id = 0

        elif idx < self.c2:
            # Schnell
            local_idx = idx - self.c1
            item = self.schnell_ds[local_idx]
            latent = self._get_latent_from_array(item["latent"])
            t5 = self.schnell_t5[local_idx]
            clip = self.schnell_clip[local_idx]
            dataset_id = 1

        elif idx < self.c3:
            # SportFashion
            local_idx = idx - self.c2
            item = self.sportfashion_ds[local_idx]
            image = item["image"]
            latent = self._encode_image(image)
            t5 = self.sportfashion_t5[local_idx]
            clip = self.sportfashion_clip[local_idx]
            dataset_id = 2
            if self.compute_masks:
                pixel_mask = create_product_mask(image)
                mask = downsample_mask_to_latent(pixel_mask, 64, 64)

        else:
            # SynthMoCap
            local_idx = idx - self.c3
            item = self.synthmocap_ds[local_idx]
            image = item["image"]
            conditioning = item["conditioning_image"]
            latent = self._encode_image(image)
            t5 = self.synthmocap_t5[local_idx]
            clip = self.synthmocap_clip[local_idx]
            dataset_id = 3
            if self.compute_masks:
                pixel_mask = create_smpl_mask(conditioning)
                mask = downsample_mask_to_latent(pixel_mask, 64, 64)

        result = {
            "latent": latent,
            "t5_embed": t5.to(self.dtype),
            "clip_pooled": clip.to(self.dtype),
            "sample_idx": idx,  # Global index for expert cache lookup
            "local_idx": local_idx,  # Local index within dataset
            "dataset_id": dataset_id,  # Which dataset (0=portrait, 1=schnell, etc)
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
# EXPERT FEATURE LOOKUP (handles multiple datasets)
# ============================================================================
def get_expert_features_for_batch(
        local_indices: torch.Tensor,
        dataset_ids: torch.Tensor,
        timesteps: torch.Tensor,
        portrait_cache: Optional[ExpertFeatureCache],
        schnell_cache: Optional[ExpertFeatureCache],
        sportfashion_cache: Optional[ExpertFeatureCache],
        synthmocap_cache: Optional[ExpertFeatureCache],
) -> Optional[torch.Tensor]:
    """Get expert features from the appropriate cache for each sample."""

    caches = [portrait_cache, schnell_cache, sportfashion_cache, synthmocap_cache]

    # Check if any cache is available
    if not any(c is not None for c in caches):
        return None

    B = local_indices.shape[0]
    device = timesteps.device
    features = torch.zeros(B, EXPERT_DIM, device=device, dtype=DTYPE)

    for ds_id, cache in enumerate(caches):
        if cache is None:
            continue

        # Find samples from this dataset
        mask = dataset_ids == ds_id
        if not mask.any():
            continue

        # Get features for these samples
        ds_local_indices = local_indices[mask]
        ds_timesteps = timesteps[mask]
        ds_features = cache.get_features(ds_local_indices, ds_timesteps)
        features[mask] = ds_features

    return features


# ============================================================================
# MASKED LOSS FUNCTION
# ============================================================================
def masked_mse_loss(pred, target, mask=None, fg_weight=2.0, bg_weight=0.5, snr_weights=None):
    B, N, C = pred.shape
    if mask is None:
        loss_per_sample = ((pred - target) ** 2).mean(dim=[1, 2])
    else:
        H = W = int(math.sqrt(N))
        mask_flat = mask.view(B, H * W, 1).to(pred.device)
        sq_error = (pred - target) ** 2
        weights = mask_flat * fg_weight + (1 - mask_flat) * bg_weight
        weighted_error = sq_error * weights
        loss_per_sample = weighted_error.mean(dim=[1, 2])
    if snr_weights is not None:
        loss_per_sample = loss_per_sample * snr_weights
    return loss_per_sample.mean()


# ============================================================================
# CREATE DATASET
# ============================================================================
print("\nCreating combined dataset...")
combined_ds = CombinedDataset(
    portrait_ds, portrait_indices, portrait_t5, portrait_clip,
    schnell_ds, schnell_t5, schnell_clip,
    sportfashion_ds, sportfashion_t5, sportfashion_clip,
    synthmocap_ds, synthmocap_t5, synthmocap_clip,
    vae, VAE_SCALE, DEVICE, DTYPE,
    compute_masks=USE_MASKED_LOSS,
)
print(f"✓ Combined dataset: {len(combined_ds)} samples")
print(f"  - Portraits (3x):    {combined_ds.n_portrait:,}")
print(f"  - Schnell teacher:   {combined_ds.n_schnell:,}")
print(f"  - SportFashion:      {combined_ds.n_sportfashion:,}")
print(f"  - SynthMoCap:        {combined_ds.n_synthmocap:,}")
print(f"  - Expert distillation: {ENABLE_EXPERT_DISTILLATION}")

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
                     use_ema=True, use_expert=None, seed=None,
                     negative_prompt="blurry, distorted, low quality"):
    """
    Generate samples during training with proper CFG support.

    Args:
        guidance_scale: CFG scale (1.0 = no CFG, 5.0-7.0 typical)
        use_expert: None = auto, True = force expert on, False = force expert off
                    During early training, ExpertPredictor is untrained - disable it.
        negative_prompt: Text for unconditional branch of CFG
    """
    was_training = model.training
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    # Get uncompiled model reference - torch.compile bakes the graph,
    # so we can't dynamically disable expert_predictor on compiled model
    model_ref = model._orig_mod if hasattr(model, '_orig_mod') else model

    if use_ema and 'ema' in globals() and ema is not None:
        ema.apply_shadow_for_eval(model)

    # Optionally disable expert predictor for cleaner samples during early training
    expert_backup = None
    if use_expert is False and hasattr(model_ref, 'expert_predictor') and model_ref.expert_predictor is not None:
        expert_backup = model_ref.expert_predictor
        model_ref.expert_predictor = None

    B = len(prompts)
    C = 16

    # Encode conditional prompts
    t5_list, clip_list = [], []
    for p in prompts:
        t5, clip = encode_prompt(p)
        t5_list.append(t5)
        clip_list.append(clip)
    t5_cond = torch.stack(t5_list).to(DTYPE)
    clip_cond = torch.stack(clip_list).to(DTYPE)

    # Encode unconditional prompt for CFG
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
            # Use model_ref (uncompiled) so expert_predictor=None actually works
            v_cond = model_ref(
                hidden_states=x,
                encoder_hidden_states=t5_cond,
                pooled_projections=clip_cond,
                timestep=t_batch,
                img_ids=img_ids,
            )
            if isinstance(v_cond, tuple):
                v_cond = v_cond[0]

            # CFG: unconditional prediction
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

    # Restore expert predictor if we disabled it
    if expert_backup is not None:
        model_ref.expert_predictor = expert_backup

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


def load_with_weight_upgrade(model, state_dict):
    """
    Load state dict with automatic handling of:
      - Missing ExpertPredictor weights → initialize fresh
      - Missing Q/K norm weights → initialize to ones (identity)
      - Unexpected keys → ignore (e.g., old guidance_in, sin_basis caches)
    """
    model_state = model.state_dict()

    # Patterns for new weights that may not exist in old checkpoints
    NEW_WEIGHT_PATTERNS = [
        'expert_predictor.',  # New ExpertPredictor module
        '.norm_q.weight',
        '.norm_k.weight',
        '.norm_added_q.weight',
        '.norm_added_k.weight',
    ]

    # Keys that may exist in old checkpoints but not new model
    DEPRECATED_PATTERNS = [
        'guidance_in.',  # Replaced by expert_predictor
        '.sin_basis',  # Old cached sin embeddings
    ]

    loaded_keys = []
    missing_keys = []
    unexpected_keys = []
    initialized_keys = []

    # First pass: load matching weights
    for key in state_dict.keys():
        if key in model_state:
            if state_dict[key].shape == model_state[key].shape:
                model_state[key] = state_dict[key]
                loaded_keys.append(key)
            else:
                print(
                    f"  ⚠ Shape mismatch for {key}: checkpoint {state_dict[key].shape} vs model {model_state[key].shape}")
                unexpected_keys.append(key)
        else:
            is_deprecated = any(pat in key for pat in DEPRECATED_PATTERNS)
            if is_deprecated:
                unexpected_keys.append(key)
            else:
                print(f"  ⚠ Unexpected key (not in model): {key}")
                unexpected_keys.append(key)

    # Second pass: handle missing keys
    for key in model_state.keys():
        if key not in loaded_keys:
            is_new = any(pat in key for pat in NEW_WEIGHT_PATTERNS)

            if is_new:
                # Keep default initialization for new modules
                initialized_keys.append(key)
            else:
                missing_keys.append(key)
                print(f"  ⚠ Missing key (not in checkpoint): {key}")

    # Load the updated state
    model.load_state_dict(model_state, strict=False)

    # Report
    if initialized_keys:
        # Group by module for cleaner output
        modules = set()
        for k in initialized_keys:
            parts = k.split('.')
            if len(parts) >= 2:
                modules.add(parts[0] + '.' + parts[1] if parts[0] == 'expert_predictor' else parts[0])
        print(f"  ✓ Initialized new modules (fresh): {sorted(modules)}")

    if unexpected_keys:
        deprecated = [k for k in unexpected_keys if any(p in k for p in DEPRECATED_PATTERNS)]
        if deprecated:
            print(f"  ✓ Ignored deprecated keys: {len(deprecated)} (guidance_in, etc)")

    return missing_keys, unexpected_keys


def load_checkpoint(model, optimizer, scheduler, target):
    """
    Load checkpoint with weight upgrade support for ExpertPredictor.

    When ALLOW_WEIGHT_UPGRADE=True:
      - Missing ExpertPredictor weights are initialized fresh
      - Old guidance_in weights are ignored
      - Model continues training with new architecture
    """
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
                step_name = target.split(":")[1]
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

        # Get model reference (handle torch.compile wrapper)
        model_ref = model._orig_mod if hasattr(model, '_orig_mod') else model

        if ALLOW_WEIGHT_UPGRADE:
            # Flexible loading with weight upgrade
            missing, unexpected = load_with_weight_upgrade(model_ref, state_dict)

            if missing:
                print(f"  ⚠ {len(missing)} truly missing parameters (may need attention)")
        else:
            # Strict loading - must match exactly
            model_ref.load_state_dict(state_dict, strict=True)

        print(f"✓ Loaded model weights")

        # Load EMA weights if they exist
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
            print(f"  Will use fresh optimizer (this is fine for architecture changes)")
        print(f"Resuming from step {start_step}, epoch {start_epoch}")

    return start_step, start_epoch, ema_state


# ============================================================================
# CREATE MODEL
# ============================================================================
print("\nCreating TinyFluxDeep model with ExpertPredictor...")

config = TinyFluxDeepConfig(
    use_expert_predictor=ENABLE_EXPERT_DISTILLATION,
    expert_dim=EXPERT_DIM,
    expert_hidden_dim=EXPERT_HIDDEN_DIM,
    expert_dropout=EXPERT_DROPOUT,
    guidance_embeds=False,
)
model = TinyFluxDeep(config).to(device=DEVICE, dtype=DTYPE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

if hasattr(model, 'expert_predictor') and model.expert_predictor is not None:
    expert_params = sum(p.numel() for p in model.expert_predictor.parameters())
    print(f"Expert predictor parameters: {expert_params:,}")

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
    ema.load_shadow(ema_state)
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
# DISTILLATION WEIGHT SCHEDULE
# ============================================================================
def get_distill_weight(step):
    if step < DISTILL_WARMUP_STEPS:
        return DISTILL_LOSS_WEIGHT * (step / DISTILL_WARMUP_STEPS)
    return DISTILL_LOSS_WEIGHT


# ============================================================================
# TRAINING LOOP
# ============================================================================
print(f"\n{'=' * 60}")
print(f"Training TinyFlux-Deep with Expert Distillation (Precached)")
print(f"{'=' * 60}")
print(f"Total: {len(combined_ds):,} samples")
print(f"Epochs: {EPOCHS}, Steps/epoch: {len(loader)}, Total: {total_steps}")
print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"Expert distillation: {ENABLE_EXPERT_DISTILLATION} (PRECACHED)")
if ENABLE_EXPERT_DISTILLATION:
    print(f"  - Expert: {EXPERT_CHECKPOINT}")
    print(f"  - Timestep buckets: {len(EXPERT_T_BUCKETS)}")
    print(f"  - Distill weight: {DISTILL_LOSS_WEIGHT} (warmup: {DISTILL_WARMUP_STEPS} steps)")
    print(f"  - Expert dropout: {EXPERT_DROPOUT}")
print(f"Masked loss: {USE_MASKED_LOSS}")
print(f"Min-SNR gamma: {MIN_SNR_GAMMA}")
print(f"Resume: step {start_step}, epoch {start_epoch}")

model.train()
step = start_step
best = float("inf")

for ep in range(start_epoch, EPOCHS):
    ep_loss = 0
    ep_main_loss = 0
    ep_distill_loss = 0
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

        # Get expert features from CACHE (fast!)
        expert_features = None
        if ENABLE_EXPERT_DISTILLATION:
            expert_features = get_expert_features_for_batch(
                local_indices, dataset_ids, t,
                portrait_expert_cache, schnell_expert_cache,
                sportfashion_expert_cache, synthmocap_expert_cache,
            )

            # Apply dropout OUTSIDE model (no graph break)
            if expert_features is not None and random.random() < EXPERT_DROPOUT:
                expert_features = None

        with torch.autocast("cuda", dtype=DTYPE):
            v_pred, expert_info = model(
                hidden_states=x_t,
                encoder_hidden_states=t5,
                pooled_projections=clip,
                timestep=t,
                img_ids=img_ids,
                expert_features=expert_features,
                return_expert_pred=True,
            )

        # Compute losses
        snr_weights = min_snr_weight(t)

        main_loss = masked_mse_loss(
            v_pred, v_target,
            mask=masks if USE_MASKED_LOSS else None,
            fg_weight=FG_LOSS_WEIGHT,
            bg_weight=BG_LOSS_WEIGHT,
            snr_weights=snr_weights
        )

        # Distillation loss
        distill_loss = torch.tensor(0.0, device=DEVICE)
        if expert_features is not None and expert_info is not None and 'expert_pred' in expert_info:
            distill_weight = get_distill_weight(step)
            distill_loss = F.mse_loss(expert_info['expert_pred'], expert_features)
            total_loss = main_loss + distill_weight * distill_loss
        else:
            total_loss = main_loss

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
                if ENABLE_EXPERT_DISTILLATION:
                    writer.add_scalar("train/distill_loss", distill_loss.item(), step)
                    writer.add_scalar("train/distill_weight", get_distill_weight(step), step)
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

        ep_loss += total_loss.item()
        ep_main_loss += main_loss.item()
        ep_distill_loss += distill_loss.item()
        ep_batches += 1

        pbar.set_postfix(
            loss=f"{total_loss.item():.4f}",
            main=f"{main_loss.item():.4f}",
            dist=f"{distill_loss.item():.4f}" if ENABLE_EXPERT_DISTILLATION else "off",
            step=step
        )

    avg = ep_loss / max(ep_batches, 1)
    avg_main = ep_main_loss / max(ep_batches, 1)
    avg_distill = ep_distill_loss / max(ep_batches, 1)

    print(f"Epoch {ep + 1} - total: {avg:.4f}, main: {avg_main:.4f}, distill: {avg_distill:.4f}")

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