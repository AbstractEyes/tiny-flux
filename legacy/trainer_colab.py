# ============================================================================
# TinyFlux-Deep Training Cell
# ============================================================================
# Trains the deep variant with frozen ported layers
# Config: 25 single blocks, 15 double blocks, 4 attention heads
# hidden_size: 512 (4 heads * 128 head_dim)
# Repo: AbstractPhil/tiny-flux-deep
#
# USAGE: Run model.py cell first, then this cell
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModel, CLIPTokenizer
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file, load_file
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import numpy as np
import math
import json
from typing import Tuple, Optional, Dict
import os
from datetime import datetime
from dataclasses import dataclass

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
BATCH_SIZE = 4
GRAD_ACCUM = 1    # Effective batch = 32
LR = 1e-4         # Lower LR for fine-tuning frozen model
EPOCHS = 20
MAX_SEQ = 128
MIN_SNR = 5.0
SHIFT = 3.0
DEVICE = "cuda"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# HuggingFace Hub
HF_REPO = "AbstractPhil/tiny-flux-deep"
SAVE_EVERY = 2500
UPLOAD_EVERY = 2500
SAMPLE_EVERY = 2500
LOG_EVERY = 10
LOG_UPLOAD_EVERY = 2500  # Upload logs every N steps

# Checkpoint loading
LOAD_TARGET = "latest"  # "hub", "latest", "best", "none"
RESUME_STEP = 162500

# Dataset
DATASET_REPO = "AbstractPhil/flux-schnell-teacher-latents"
DATASET_CONFIG = "train_512"

# Paths
CHECKPOINT_DIR = "./tiny_flux_deep_checkpoints"
LOG_DIR = "./tiny_flux_deep_logs"
SAMPLE_DIR = "./tiny_flux_deep_samples"
ENCODING_CACHE_DIR = "./encoding_cache"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(ENCODING_CACHE_DIR, exist_ok=True)

# ============================================================================
# FROZEN LAYER POSITIONS (from porting)
# ============================================================================
# Single blocks: old 0→0, old 1→{8,12,16}, old 2→24
FROZEN_SINGLE_POSITIONS = {}

# Double blocks: old 0→0, old 1→{4,7,10}, old 2→14
FROZEN_DOUBLE_POSITIONS = {}

# ============================================================================
# MODEL CONFIG
# ============================================================================
@dataclass
class TinyFluxDeepConfig:
    """Deep variant: 512 hidden, 4 heads, 25 single, 15 double."""
    hidden_size: int = 512
    num_attention_heads: int = 4
    attention_head_dim: int = 128
    in_channels: int = 16
    patch_size: int = 1
    joint_attention_dim: int = 768
    pooled_projection_dim: int = 768
    num_double_layers: int = 15
    num_single_layers: int = 25
    mlp_ratio: float = 4.0
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)
    guidance_embeds: bool = True

# ============================================================================
# HF HUB SETUP
# ============================================================================
print("Setting up HuggingFace Hub...")
api = HfApi()
try:
    api.create_repo(repo_id=HF_REPO, exist_ok=True, repo_type="model")
    print(f"✓ Repo ready: {HF_REPO}")
except Exception as e:
    print(f"Note: {e}")

# ============================================================================
# TENSORBOARD
# ============================================================================
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))
print(f"✓ Tensorboard: {LOG_DIR}/{run_name}")

# ============================================================================
# LOAD DATASET
# ============================================================================
print("\nLoading dataset...")
ds = load_dataset(DATASET_REPO, DATASET_CONFIG, split="train")
print(f"Samples: {len(ds)} ({DATASET_CONFIG})")

# ============================================================================
# LOAD TEXT ENCODERS
# ============================================================================
print("\nLoading flan-t5-base...")
t5_tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
t5_enc = T5EncoderModel.from_pretrained("google/flan-t5-base", torch_dtype=DTYPE).to(DEVICE).eval()

print("Loading CLIP-L...")
clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
clip_enc = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=DTYPE).to(DEVICE).eval()

for p in t5_enc.parameters(): p.requires_grad = False
for p in clip_enc.parameters(): p.requires_grad = False

# ============================================================================
# LOAD VAE
# ============================================================================
print("Loading Flux VAE...")
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    subfolder="vae",
    torch_dtype=DTYPE
).to(DEVICE).eval()
for p in vae.parameters(): p.requires_grad = False

# ============================================================================
# BATCHED ENCODING
# ============================================================================
@torch.inference_mode()
def encode_prompts_batched(prompts: list) -> tuple:
    t5_in = t5_tok(prompts, max_length=MAX_SEQ, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
    t5_out = t5_enc(input_ids=t5_in.input_ids, attention_mask=t5_in.attention_mask).last_hidden_state

    clip_in = clip_tok(prompts, max_length=77, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
    clip_out = clip_enc(input_ids=clip_in.input_ids, attention_mask=clip_in.attention_mask)

    return t5_out, clip_out.pooler_output

# ============================================================================
# PRE-ENCODE PROMPTS
# ============================================================================
print("\nPre-encoding prompts...")
PRECOMPUTE_ENCODINGS = True
cache_file = os.path.join(ENCODING_CACHE_DIR, f"encodings_{DATASET_CONFIG}_{len(ds)}.pt")

if PRECOMPUTE_ENCODINGS:
    if os.path.exists(cache_file):
        print(f"Loading cached encodings from {cache_file}...")
        cached = torch.load(cache_file, weights_only=True)
        all_t5_embeds = cached["t5_embeds"]
        all_clip_pooled = cached["clip_pooled"]
        print(f"✓ Loaded cached encodings")
    else:
        print("Encoding prompts (will cache)...")
        all_prompts = ds["prompt"]

        encode_batch_size = 64
        all_t5_embeds = []
        all_clip_pooled = []

        for i in tqdm(range(0, len(all_prompts), encode_batch_size), desc="Encoding"):
            batch_prompts = all_prompts[i:i+encode_batch_size]
            t5_out, clip_out = encode_prompts_batched(batch_prompts)
            all_t5_embeds.append(t5_out.cpu())
            all_clip_pooled.append(clip_out.cpu())

        all_t5_embeds = torch.cat(all_t5_embeds, dim=0)
        all_clip_pooled = torch.cat(all_clip_pooled, dim=0)

        torch.save({"t5_embeds": all_t5_embeds, "clip_pooled": all_clip_pooled}, cache_file)
        print(f"✓ Saved encoding cache")

# ============================================================================
# FLOW MATCHING HELPERS
# ============================================================================
def flux_shift(t, s=SHIFT):
    return s * t / (1 + (s - 1) * t)

def min_snr_weight(t, gamma=MIN_SNR):
    snr = (t / (1 - t).clamp(min=1e-5)).pow(2)
    return torch.clamp(snr, max=gamma) / snr.clamp(min=1e-5)

# ============================================================================
# SAMPLING FUNCTION
# ============================================================================
@torch.inference_mode()
def generate_samples(model, prompts, num_steps=28, guidance_scale=3.5, H=64, W=64):
    model.eval()
    B = len(prompts)
    C = 16

    t5_embeds, clip_pooleds = encode_prompts_batched(prompts)
    t5_embeds = t5_embeds.to(DTYPE)
    clip_pooleds = clip_pooleds.to(DTYPE)

    x = torch.randn(B, H * W, C, device=DEVICE, dtype=DTYPE)
    img_ids = TinyFluxDeep.create_img_ids(B, H, W, DEVICE)

    t_linear = torch.linspace(0, 1, num_steps + 1, device=DEVICE, dtype=DTYPE)
    timesteps = flux_shift(t_linear, s=SHIFT)

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_curr

        t_batch = t_curr.expand(B).to(DTYPE)
        guidance = torch.full((B,), guidance_scale, device=DEVICE, dtype=DTYPE)

        v_cond = model(
            hidden_states=x,
            encoder_hidden_states=t5_embeds,
            pooled_projections=clip_pooleds,
            timestep=t_batch,
            img_ids=img_ids,
            guidance=guidance,
        )
        x = x + v_cond * dt

    latents = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents.to(vae.dtype)).sample
    images = (images / 2 + 0.5).clamp(0, 1)

    model.train()
    return images


def save_samples(images, prompts, step, save_dir, upload=True):
    from torchvision.utils import make_grid, save_image

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, (img, prompt) in enumerate(zip(images, prompts)):
        safe_prompt = prompt[:50].replace(" ", "_").replace("/", "-")
        path = os.path.join(save_dir, f"step{step}_{i}_{safe_prompt}.png")
        save_image(img, path)

    grid = make_grid(images, nrow=2, normalize=False)
    grid_path = os.path.join(save_dir, f"step{step}_grid.png")
    save_image(grid, grid_path)

    writer.add_image("samples", grid, step)

    if upload:
        try:
            api.upload_file(
                path_or_fileobj=grid_path,
                path_in_repo=f"samples/{timestamp}_step_{step}.png",
                repo_id=HF_REPO,
            )
            print(f"  ✓ Saved & uploaded {len(images)} samples")
        except Exception as e:
            print(f"  ✓ Saved {len(images)} samples (upload failed: {e})")

# ============================================================================
# COLLATE
# ============================================================================
class IndexedDataset:
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = dict(self.ds[idx])
        item["__index__"] = idx
        return item

def collate_preencoded(batch):
    indices = [b["__index__"] for b in batch]
    latents = torch.stack([torch.tensor(np.array(b["latent"]), dtype=DTYPE) for b in batch])
    return {
        "latents": latents,
        "t5_embeds": all_t5_embeds[indices].to(DTYPE),
        "clip_pooled": all_clip_pooled[indices].to(DTYPE),
    }

ds = IndexedDataset(ds)
num_workers = 8

# ============================================================================
# FREEZE PORTED LAYERS
# ============================================================================
def freeze_ported_layers(model):
    """Freeze layers that were ported from TinyFlux."""
    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        should_freeze = False

        # Check single blocks
        for pos in FROZEN_SINGLE_POSITIONS:
            if f"single_blocks.{pos}." in name:
                should_freeze = True
                break

        # Check double blocks
        for pos in FROZEN_DOUBLE_POSITIONS:
            if f"double_blocks.{pos}." in name:
                should_freeze = True
                break

        if should_freeze:
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            param.requires_grad = True
            trainable_count += param.numel()

    print(f"\nFrozen params: {frozen_count:,}")
    print(f"Trainable params: {trainable_count:,}")
    print(f"Total: {frozen_count + trainable_count:,}")
    print(f"Trainable ratio: {trainable_count / (frozen_count + trainable_count) * 100:.1f}%")

    return model

# ============================================================================
# CHECKPOINT FUNCTIONS
# ============================================================================
EXPECTED_MISSING = {'time_in.sin_basis', 'guidance_in.sin_basis',
                    'rope.freqs_0', 'rope.freqs_1', 'rope.freqs_2'}

def load_weights(path):
    if path.endswith(".safetensors"):
        state_dict = load_file(path)
    else:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    return state_dict


def save_checkpoint(model, optimizer, scheduler, step, epoch, loss, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    state_dict = model.state_dict()
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    weights_path = path.replace(".pt", ".safetensors")
    save_file(state_dict, weights_path)

    torch.save({
        "step": step, "epoch": epoch, "loss": loss,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    print(f"  ✓ Saved checkpoint: step {step}")
    return weights_path


def upload_checkpoint(weights_path, step):
    try:
        api.upload_file(path_or_fileobj=weights_path, path_in_repo=f"checkpoints/step_{step}.safetensors", repo_id=HF_REPO)
        print(f"  ✓ Uploaded step {step}")
    except Exception as e:
        print(f"  ⚠ Upload failed: {e}")


def load_checkpoint(model, target):
    if target == "none" or target is None:
        print("Starting from scratch (no checkpoint)")
        return 0, 0

    if target == "hub":
        try:
            weights_path = hf_hub_download(repo_id=HF_REPO, filename="model.safetensors")
            weights = load_weights(weights_path)
            missing, unexpected = model.load_state_dict(weights, strict=False)
            actual_missing = set(missing) - EXPECTED_MISSING
            if actual_missing:
                print(f"  ⚠ Missing: {list(actual_missing)[:5]}...")
            else:
                print(f"  ✓ Missing only precomputed buffers (OK)")
            if unexpected:
                print(f"  ⚠ Unexpected: {unexpected[:5]}...")
            print(f"✓ Loaded from hub: {HF_REPO}")
            return 0, 0
        except Exception as e:
            print(f"Hub load failed: {e}")
            return 0, 0

    if target == "latest":
        # Find latest local checkpoint
        ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("step_") and f.endswith(".safetensors")]
        if ckpts:
            latest = sorted(ckpts, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
            weights_path = os.path.join(CHECKPOINT_DIR, latest)
            weights = load_weights(weights_path)
            model.load_state_dict(weights, strict=False)
            step = int(latest.split("_")[1].split(".")[0])
            print(f"✓ Loaded local: {latest}")
            return step, 0

    return 0, 0

# ============================================================================
# DATALOADER
# ============================================================================
loader = DataLoader(
    ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_preencoded,
    num_workers=num_workers, pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

# ============================================================================
# MODEL (assumes TinyFluxDeep is defined - run model cell first)
# ============================================================================
print("\nCreating TinyFlux-Deep model...")
config = TinyFluxDeepConfig()
model = TinyFluxDeep(config).to(DEVICE).to(DTYPE)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Upload config.json
config_dict = {
    "hidden_size": config.hidden_size,
    "num_attention_heads": config.num_attention_heads,
    "attention_head_dim": config.attention_head_dim,
    "in_channels": config.in_channels,
    "joint_attention_dim": config.joint_attention_dim,
    "pooled_projection_dim": config.pooled_projection_dim,
    "num_double_layers": config.num_double_layers,
    "num_single_layers": config.num_single_layers,
    "mlp_ratio": config.mlp_ratio,
    "axes_dims_rope": list(config.axes_dims_rope),
    "guidance_embeds": config.guidance_embeds,
}
config_path = os.path.join(CHECKPOINT_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(config_dict, f, indent=2)
try:
    api.upload_file(path_or_fileobj=config_path, path_in_repo="config.json", repo_id=HF_REPO)
    print("✓ Uploaded config.json")
except Exception as e:
    print(f"⚠ Config upload failed: {e}")

# ============================================================================
# LOAD & FREEZE
# ============================================================================
print(f"\nLoad target: {LOAD_TARGET}")
start_step, start_epoch = load_checkpoint(model, LOAD_TARGET)

#print("\nFreezing ported layers...")
#model = freeze_ported_layers(model)
#print(f"Frozen single blocks: {sorted(FROZEN_SINGLE_POSITIONS)}")
#print(f"Frozen double blocks: {sorted(FROZEN_DOUBLE_POSITIONS)}")
#
## Upload frozen_params.json
#frozen_params_list = [name for name, p in model.named_parameters() if not p.requires_grad]
#frozen_path = os.path.join(CHECKPOINT_DIR, "frozen_params.json")
#with open(frozen_path, "w") as f:
#    json.dump({
#        "frozen_single_positions": sorted(FROZEN_SINGLE_POSITIONS),
#        "frozen_double_positions": sorted(FROZEN_DOUBLE_POSITIONS),
#        "frozen_param_names": frozen_params_list,
#        "num_frozen": len(frozen_params_list),
#    }, f, indent=2)
#try:
#    api.upload_file(path_or_fileobj=frozen_path, path_in_repo="frozen_params.json", repo_id=HF_REPO)
#    print("✓ Uploaded frozen_params.json")
#except:
#    pass
#
# Only optimize trainable params
trainable_params = [p for p in model.parameters() if p.requires_grad]
#print(f"Optimizing {len(trainable_params)} parameter groups")

# ============================================================================
# OPTIMIZER
# ============================================================================
opt = torch.optim.AdamW(trainable_params, lr=LR, betas=(0.9, 0.99), weight_decay=0.01, fused=True)

total_steps = len(loader) * EPOCHS // GRAD_ACCUM
warmup = min(500, total_steps // 10)

def lr_fn(step):
    if step < warmup:
        return step / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

if RESUME_STEP is not None:
    start_step = RESUME_STEP

# ============================================================================
# COMPILE (after freezing)
# ============================================================================
model = torch.compile(model, mode="default")

# Sample prompts
SAMPLE_PROMPTS = [
    "a photo of a cat sitting on a windowsill",
    "a beautiful sunset over mountains",
    "a portrait of a woman with red hair",
    "a futuristic cityscape at night",
]

# ============================================================================
# TRAINING LOOP
# ============================================================================
print(f"\n{'='*60}")
print(f"Training TinyFlux-Deep")
print(f"{'='*60}")
print(f"Epochs: {EPOCHS}, Steps: {total_steps}")
print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"LR: {LR}, Warmup: {warmup}")

model.train()
step = start_step
best = float("inf")

for ep in range(start_epoch, EPOCHS):
    ep_loss = 0
    ep_batches = 0
    pbar = tqdm(loader, desc=f"E{ep + 1}")

    for i, batch in enumerate(pbar):
        latents = batch["latents"].to(DEVICE, non_blocking=True)
        t5 = batch["t5_embeds"].to(DEVICE, non_blocking=True)
        clip = batch["clip_pooled"].to(DEVICE, non_blocking=True)

        B, C, H, W = latents.shape
        data = latents.permute(0, 2, 3, 1).reshape(B, H * W, C)
        noise = torch.randn_like(data)

        # Logit-normal timesteps with flux shift
        t = torch.sigmoid(torch.randn(B, device=DEVICE))
        t = flux_shift(t, s=SHIFT).to(DTYPE).clamp(1e-4, 1 - 1e-4)

        t_expanded = t.view(B, 1, 1)
        x_t = (1 - t_expanded) * noise + t_expanded * data
        v_target = data - noise

        img_ids = TinyFluxDeep.create_img_ids(B, H, W, DEVICE)
        guidance = torch.rand(B, device=DEVICE, dtype=DTYPE) * 4 + 1

        with torch.autocast("cuda", dtype=DTYPE):
            v_pred = model(
                hidden_states=x_t,
                encoder_hidden_states=t5,
                pooled_projections=clip,
                timestep=t,
                img_ids=img_ids,
                guidance=guidance,
            )

        loss_raw = F.mse_loss(v_pred, v_target, reduction="none").mean(dim=[1, 2])
        snr_weights = min_snr_weight(t)
        loss = (loss_raw * snr_weights).mean() / GRAD_ACCUM
        loss.backward()

        if (i + 1) % GRAD_ACCUM == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % LOG_EVERY == 0:
                writer.add_scalar("train/loss", loss.item() * GRAD_ACCUM, step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), step)

            if step % SAMPLE_EVERY == 0:
                print(f"\n  Generating samples at step {step}...")
                images = generate_samples(model, SAMPLE_PROMPTS, num_steps=20)  # Faster during training
                save_samples(images, SAMPLE_PROMPTS, step, SAMPLE_DIR)

            if step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{step}.pt")
                weights_path = save_checkpoint(model, opt, sched, step, ep, loss.item(), ckpt_path)
                if step % UPLOAD_EVERY == 0:
                    upload_checkpoint(weights_path, step)

            # Upload logs periodically
            if step % LOG_UPLOAD_EVERY == 0:
                writer.flush()
                log_run_dir = os.path.join(LOG_DIR, run_name)
                for fname in os.listdir(log_run_dir):
                    if fname.startswith("events."):
                        try:
                            api.upload_file(
                                path_or_fileobj=os.path.join(log_run_dir, fname),
                                path_in_repo=f"logs/{run_name}/{fname}",
                                repo_id=HF_REPO
                            )
                            print(f"  ✓ Uploaded logs")
                        except:
                            pass
                        break  # Only need to upload once per dir

        ep_loss += loss.item() * GRAD_ACCUM
        ep_batches += 1
        pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM:.4f}", step=step)

    avg = ep_loss / max(ep_batches, 1)
    print(f"Epoch {ep + 1} loss: {avg:.4f}")

    if avg < best:
        best = avg
        weights_path = save_checkpoint(model, opt, sched, step, ep, avg, os.path.join(CHECKPOINT_DIR, "best.pt"))
        try:
            api.upload_file(path_or_fileobj=weights_path, path_in_repo="model.safetensors", repo_id=HF_REPO)
            print(f"  ✓ Uploaded best model")
        except:
            pass

# ============================================================================
# FINAL
# ============================================================================
print(f"\n✓ Training complete! Best loss: {best:.4f}")
writer.close()

# Upload tensorboard logs
def upload_logs():
    """Upload tensorboard logs to HF Hub."""
    import shutil
    log_run_dir = os.path.join(LOG_DIR, run_name)
    if os.path.exists(log_run_dir):
        # Create a zip of logs
        zip_path = f"{log_run_dir}.zip"
        shutil.make_archive(log_run_dir, 'zip', LOG_DIR, run_name)
        try:
            api.upload_file(
                path_or_fileobj=zip_path,
                path_in_repo=f"logs/{run_name}.zip",
                repo_id=HF_REPO
            )
            print(f"✓ Uploaded logs: logs/{run_name}.zip")
        except Exception as e:
            print(f"⚠ Log upload failed: {e}")

        # Also upload individual event files
        for fname in os.listdir(log_run_dir):
            if fname.startswith("events."):
                try:
                    api.upload_file(
                        path_or_fileobj=os.path.join(log_run_dir, fname),
                        path_in_repo=f"logs/{run_name}/{fname}",
                        repo_id=HF_REPO
                    )
                except:
                    pass

print("\nUploading logs...")
upload_logs()

# Final samples
print("\nGenerating final samples...")
images = generate_samples(model, SAMPLE_PROMPTS, num_steps=28)
save_samples(images, SAMPLE_PROMPTS, step, SAMPLE_DIR)