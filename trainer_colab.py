# ============================================================================
# TinyFlux Training Cell - Full Featured
# ============================================================================
# Run the model cell before this one (defines TinyFlux, TinyFluxConfig)
# Dataset: AbstractPhil/flux-schnell-teacher-latents
# Uploads checkpoints to: AbstractPhil/tiny-flux
# ============================================================================

import torch
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
import os
import json
from datetime import datetime

# ============================================================================
# CONFIG
# ============================================================================
BATCH_SIZE = 16
GRAD_ACCUM = 1
LR = 1e-4
EPOCHS = 10
MAX_SEQ = 128
MIN_SNR = 5.0
SHIFT = 3.0
DEVICE = "cuda"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# HuggingFace Hub
HF_REPO = "AbstractPhil/tiny-flux"
SAVE_EVERY = 1000  # steps - local save
UPLOAD_EVERY = 1000  # steps - hub upload
SAMPLE_EVERY = 500  # steps - generate samples
LOG_EVERY = 10  # steps - tensorboard

# Checkpoint loading target
# Options:
#   None or "latest" - load most recent checkpoint
#   "best" - load best model
#   int (e.g. 1500) - load specific step
#   "hub:step_1000" - load specific checkpoint from hub
#   "local:path/to/checkpoint.safetensors" or "local:path/to/checkpoint.pt"
#   "none" - start fresh, ignore existing checkpoints
LOAD_TARGET = "hub:step_18500"

# Manual resume step (set to override step from checkpoint, or None to use checkpoint's step)
# Useful when checkpoint doesn't contain step info
RESUME_STEP = None  # e.g., 5000 to resume from step 5000

# Local paths
CHECKPOINT_DIR = "./tiny_flux_checkpoints"
LOG_DIR = "./tiny_flux_logs"
SAMPLE_DIR = "./tiny_flux_samples"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

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
ds = load_dataset("AbstractPhil/flux-schnell-teacher-latents", "train_2_512", split="train")
print(f"Samples: {len(ds)}")

# ============================================================================
# LOAD TEXT ENCODERS
# ============================================================================
print("\nLoading flan-t5-base (768 dim)...")
t5_tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
t5_enc = T5EncoderModel.from_pretrained("google/flan-t5-base", torch_dtype=DTYPE).to(DEVICE).eval()

print("Loading CLIP-L...")
clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
clip_enc = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=DTYPE).to(DEVICE).eval()

for p in t5_enc.parameters(): p.requires_grad = False
for p in clip_enc.parameters(): p.requires_grad = False

# ============================================================================
# LOAD VAE FOR SAMPLE GENERATION
# ============================================================================
print("Loading Flux VAE for samples...")
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    subfolder="vae",
    torch_dtype=DTYPE
).to(DEVICE).eval()
for p in vae.parameters(): p.requires_grad = False


# ============================================================================
# ENCODING HELPERS
# ============================================================================
@torch.no_grad()
def encode_prompt(prompt):
    t5_in = t5_tok(prompt, max_length=MAX_SEQ, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
    t5_out = t5_enc(input_ids=t5_in.input_ids, attention_mask=t5_in.attention_mask).last_hidden_state

    clip_in = clip_tok(prompt, max_length=77, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
    clip_out = clip_enc(input_ids=clip_in.input_ids, attention_mask=clip_in.attention_mask)
    return t5_out, clip_out.pooler_output


# ============================================================================
# FLOW MATCHING HELPERS
# ============================================================================
# Rectified Flow / Flow Matching formulation:
#   x_t = (1-t) * x_0 + t * x_1
#   where x_0 = noise, x_1 = data
#   t=0: pure noise, t=1: pure data
#   velocity v = x_1 - x_0 = data - noise
#
# Training: model learns to predict v given (x_t, t)
# Inference: start from noise (t=0), integrate to data (t=1)
#   x_{t+dt} = x_t + v_pred * dt
# ============================================================================

def flux_shift(t, s=SHIFT):
    """Flux timestep shift for training distribution.

    Shifts timesteps towards higher values (closer to data),
    making training focus more on refining details.

    s=3.0 (default): flux_shift(0.5) ≈ 0.75
    """
    return s * t / (1 + (s - 1) * t)


def flux_shift_inverse(t_shifted, s=SHIFT):
    """Inverse of flux_shift."""
    return t_shifted / (s - (s - 1) * t_shifted)


def min_snr_weight(t, gamma=MIN_SNR):
    """Min-SNR weighting to balance loss across timesteps.

    Downweights very easy timesteps (near t=0 or t=1).
    gamma=5.0 is typical.
    """
    snr = (t / (1 - t).clamp(min=1e-5)).pow(2)
    return torch.clamp(snr, max=gamma) / snr.clamp(min=1e-5)


# ============================================================================
# SAMPLING FUNCTION
# ============================================================================
@torch.no_grad()
def generate_samples(model, prompts, num_steps=20, guidance_scale=3.5, H=64, W=64):
    """Generate sample images using Euler sampling.

    Flow matching: x_t = (1-t)*noise + t*data, v = data - noise
    At t=0: pure noise. At t=1: pure data.
    We integrate from t=0 to t=1.
    """
    model.eval()
    B = len(prompts)
    C = 16  # VAE channels

    # Encode prompts
    t5_embeds, clip_pooleds = [], []
    for p in prompts:
        t5_out, clip_pooled = encode_prompt(p)
        t5_embeds.append(t5_out.squeeze(0))
        clip_pooleds.append(clip_pooled.squeeze(0))
    t5_embeds = torch.stack(t5_embeds)
    clip_pooleds = torch.stack(clip_pooleds)

    # Start from pure noise (t=0)
    x = torch.randn(B, H * W, C, device=DEVICE, dtype=DTYPE)

    # Create image IDs
    img_ids = TinyFlux.create_img_ids(B, H, W, DEVICE)

    # Euler sampling: t goes from 0 (noise) to 1 (data)
    timesteps = torch.linspace(0, 1, num_steps + 1, device=DEVICE, dtype=DTYPE)

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_curr  # positive

        t_batch = t_curr.expand(B)

        # Conditional prediction
        guidance = torch.full((B,), guidance_scale, device=DEVICE, dtype=DTYPE)
        v_cond = model(
            hidden_states=x,
            encoder_hidden_states=t5_embeds,
            pooled_projections=clip_pooleds,
            timestep=t_batch,
            img_ids=img_ids,
            guidance=guidance,
        )

        # Euler step: x_{t+dt} = x_t + v * dt
        x = x + v_cond * dt

    # Reshape to image format: (B, H*W, C) -> (B, C, H, W)
    latents = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

    # Decode with VAE (match VAE dtype)
    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents.to(vae.dtype)).sample
    images = (images / 2 + 0.5).clamp(0, 1)

    model.train()
    return images


def save_samples(images, prompts, step, save_dir):
    """Save sample images and log to tensorboard."""
    from torchvision.utils import make_grid, save_image

    # Save individual images
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        safe_prompt = prompt[:50].replace(" ", "_").replace("/", "-")
        path = os.path.join(save_dir, f"step{step}_{i}_{safe_prompt}.png")
        save_image(img, path)

    # Log grid to tensorboard
    grid = make_grid(images, nrow=2, normalize=False)
    writer.add_image("samples", grid, step)

    # Log prompts
    writer.add_text("sample_prompts", "\n".join(prompts), step)

    print(f"  ✓ Saved {len(images)} samples")


# ============================================================================
# COLLATE
# ============================================================================
def collate(batch):
    latents, t5_embeds, clip_embeds, prompts = [], [], [], []
    for b in batch:
        latents.append(torch.tensor(np.array(b["latent"]), dtype=DTYPE))
        t5_out, clip_pooled = encode_prompt(b["prompt"])
        t5_embeds.append(t5_out.squeeze(0))
        clip_embeds.append(clip_pooled.squeeze(0))
        prompts.append(b["prompt"])
    return {
        "latents": torch.stack(latents).to(DEVICE),
        "t5_embeds": torch.stack(t5_embeds),
        "clip_pooled": torch.stack(clip_embeds),
        "prompts": prompts,
    }


# ============================================================================
# CHECKPOINT FUNCTIONS
# ============================================================================
def load_weights(path):
    """Load weights from .safetensors or .pt file."""
    if path.endswith(".safetensors"):
        return load_file(path)
    elif path.endswith(".pt"):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                return ckpt["model"]
            elif "state_dict" in ckpt:
                return ckpt["state_dict"]
            else:
                # Check if it looks like a state dict (has tensor values)
                first_val = next(iter(ckpt.values()), None)
                if isinstance(first_val, torch.Tensor):
                    return ckpt
                # Otherwise might have optimizer etc, look for model keys
                return ckpt
        return ckpt
    else:
        # Try safetensors first, then pt
        try:
            return load_file(path)
        except:
            return torch.load(path, map_location=DEVICE, weights_only=False)


def save_checkpoint(model, optimizer, scheduler, step, epoch, loss, path):
    """Save checkpoint locally."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    weights_path = path.replace(".pt", ".safetensors")
    save_file(model.state_dict(), weights_path)

    state = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(state, path)
    print(f"  ✓ Saved checkpoint: step {step}")
    return weights_path


def upload_checkpoint(weights_path, step, config, include_logs=True):
    """Upload checkpoint to HuggingFace Hub."""
    try:
        # Upload weights
        api.upload_file(
            path_or_fileobj=weights_path,
            path_in_repo=f"checkpoints/step_{step}.safetensors",
            repo_id=HF_REPO,
            commit_message=f"Checkpoint step {step}",
        )

        # Upload config
        config_path = os.path.join(CHECKPOINT_DIR, "config.json")
        with open(config_path, "w") as f:
            json.dump(config.__dict__, f, indent=2)
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.json",
            repo_id=HF_REPO,
        )

        # Upload tensorboard logs
        if include_logs and os.path.exists(LOG_DIR):
            api.upload_folder(
                folder_path=LOG_DIR,
                path_in_repo="logs",
                repo_id=HF_REPO,
                commit_message=f"Logs at step {step}",
            )

        # Upload samples
        if os.path.exists(SAMPLE_DIR) and os.listdir(SAMPLE_DIR):
            api.upload_folder(
                folder_path=SAMPLE_DIR,
                path_in_repo="samples",
                repo_id=HF_REPO,
                commit_message=f"Samples at step {step}",
            )

        print(f"  ✓ Uploaded to {HF_REPO}")
    except Exception as e:
        print(f"  ⚠ Upload failed: {e}")


def load_checkpoint(model, optimizer, scheduler, target):
    """
    Load checkpoint based on target specification.

    Args:
        target:
            None, "latest" - most recent checkpoint
            "best" - best model
            int (1500) - specific step
            "hub:step_1000" - specific hub checkpoint
            "local:/path/to/file.safetensors" or "local:/path/to/file.pt" - specific local file
            "none" - skip loading, start fresh
    """
    if target == "none":
        print("Starting fresh (no checkpoint loading)")
        return 0, 0

    start_step, start_epoch = 0, 0

    # Parse target
    if target is None or target == "latest":
        load_mode = "latest"
        load_path = None
    elif target == "best":
        load_mode = "best"
        load_path = None
    elif isinstance(target, int):
        load_mode = "step"
        load_path = target
    elif target.startswith("hub:"):
        load_mode = "hub"
        load_path = target[4:]  # Remove "hub:" prefix
    elif target.startswith("local:"):
        load_mode = "local"
        load_path = target[6:]  # Remove "local:" prefix
    else:
        print(f"Unknown target format: {target}, trying as step number")
        try:
            load_mode = "step"
            load_path = int(target)
        except:
            load_mode = "latest"
            load_path = None

    # Load based on mode
    if load_mode == "local":
        # Direct local file (.pt or .safetensors)
        if os.path.exists(load_path):
            weights = load_weights(load_path)
            model.load_state_dict(weights)

            # Try to find associated state file for optimizer/scheduler
            if load_path.endswith(".safetensors"):
                state_path = load_path.replace(".safetensors", ".pt")
            elif load_path.endswith(".pt"):
                # The .pt file might contain everything
                ckpt = torch.load(load_path, map_location=DEVICE, weights_only=False)
                if isinstance(ckpt, dict):
                    # Debug: show what keys are in the checkpoint
                    non_tensor_keys = [k for k in ckpt.keys() if not isinstance(ckpt.get(k), torch.Tensor)]
                    if non_tensor_keys:
                        print(f"  Checkpoint keys: {non_tensor_keys}")

                    # Extract step/epoch - try multiple common key names
                    start_step = ckpt.get("step", ckpt.get("global_step", ckpt.get("iteration", 0)))
                    start_epoch = ckpt.get("epoch", 0)

                    # Also check for nested state dict
                    if "state" in ckpt and isinstance(ckpt["state"], dict):
                        start_step = ckpt["state"].get("step", start_step)
                        start_epoch = ckpt["state"].get("epoch", start_epoch)

                    # Try to load optimizer/scheduler if present
                    if "optimizer" in ckpt:
                        try:
                            optimizer.load_state_dict(ckpt["optimizer"])
                            if "scheduler" in ckpt:
                                scheduler.load_state_dict(ckpt["scheduler"])
                        except Exception as e:
                            print(f"  Note: Could not load optimizer state: {e}")
                state_path = None
            else:
                state_path = load_path + ".pt"

            if state_path and os.path.exists(state_path):
                state = torch.load(state_path, map_location=DEVICE, weights_only=False)
                try:
                    start_step = state.get("step", start_step)
                    start_epoch = state.get("epoch", start_epoch)
                    if "optimizer" in state:
                        optimizer.load_state_dict(state["optimizer"])
                    if "scheduler" in state:
                        scheduler.load_state_dict(state["scheduler"])
                except Exception as e:
                    print(f"  Note: Could not load optimizer state: {e}")

            print(f"✓ Loaded local: {load_path} (step {start_step})")
            return start_step, start_epoch
        else:
            print(f"⚠ Local file not found: {load_path}")

    elif load_mode == "hub":
        # Specific hub checkpoint - try both extensions
        for ext in [".safetensors", ".pt", ""]:
            try:
                if load_path.endswith((".safetensors", ".pt")):
                    filename = load_path if "/" in load_path else f"checkpoints/{load_path}"
                else:
                    filename = f"checkpoints/{load_path}{ext}"
                local_path = hf_hub_download(repo_id=HF_REPO, filename=filename)
                weights = load_weights(local_path)
                model.load_state_dict(weights)
                # Extract step from filename
                if "step_" in load_path:
                    start_step = int(load_path.split("step_")[-1].replace(".safetensors", "").replace(".pt", ""))
                print(f"✓ Loaded from Hub: {filename} (step {start_step})")
                return start_step, start_epoch
            except Exception as e:
                continue
        print(f"⚠ Could not load from hub: {load_path}")

    elif load_mode == "best":
        # Try hub best first (try both extensions)
        for ext in [".safetensors", ".pt"]:
            try:
                filename = f"model{ext}" if ext else "model.safetensors"
                local_path = hf_hub_download(repo_id=HF_REPO, filename=filename)
                weights = load_weights(local_path)
                model.load_state_dict(weights)
                print(f"✓ Loaded best model from Hub")
                return start_step, start_epoch
            except:
                continue

        # Try local best (both extensions)
        for ext in [".safetensors", ".pt"]:
            best_path = os.path.join(CHECKPOINT_DIR, f"best{ext}")
            if os.path.exists(best_path):
                weights = load_weights(best_path)
                model.load_state_dict(weights)
                # Try to load optimizer state
                state_path = best_path.replace(ext, ".pt") if ext == ".safetensors" else best_path
                if os.path.exists(state_path):
                    state = torch.load(state_path, map_location=DEVICE, weights_only=False)
                    if isinstance(state, dict) and "step" in state:
                        start_step = state.get("step", 0)
                        start_epoch = state.get("epoch", 0)
                print(f"✓ Loaded local best (step {start_step})")
                return start_step, start_epoch

    elif load_mode == "step":
        # Specific step number
        step_num = load_path
        # Try hub (both extensions)
        for ext in [".safetensors", ".pt"]:
            try:
                filename = f"checkpoints/step_{step_num}{ext}"
                local_path = hf_hub_download(repo_id=HF_REPO, filename=filename)
                weights = load_weights(local_path)
                model.load_state_dict(weights)
                start_step = step_num
                print(f"✓ Loaded step {step_num} from Hub")
                return start_step, start_epoch
            except:
                continue

        # Try local (both extensions)
        for ext in [".safetensors", ".pt"]:
            local_path = os.path.join(CHECKPOINT_DIR, f"step_{step_num}{ext}")
            if os.path.exists(local_path):
                weights = load_weights(local_path)
                model.load_state_dict(weights)
                state_path = local_path.replace(".safetensors", ".pt") if ext == ".safetensors" else local_path
                if os.path.exists(state_path):
                    state = torch.load(state_path, map_location=DEVICE, weights_only=False)
                    if isinstance(state, dict):
                        try:
                            if "optimizer" in state:
                                optimizer.load_state_dict(state["optimizer"])
                            if "scheduler" in state:
                                scheduler.load_state_dict(state["scheduler"])
                            start_epoch = state.get("epoch", 0)
                        except:
                            pass
                start_step = step_num
                print(f"✓ Loaded local step {step_num}")
                return start_step, start_epoch
        print(f"⚠ Step {step_num} not found")

    # Default: latest
    # Try Hub first (both extensions)
    try:
        files = api.list_repo_files(repo_id=HF_REPO)
        checkpoints = [f for f in files if
                       f.startswith("checkpoints/step_") and (f.endswith(".safetensors") or f.endswith(".pt"))]
        if checkpoints:
            # Sort by step number
            def get_step(f):
                return int(f.split("step_")[-1].replace(".safetensors", "").replace(".pt", ""))

            checkpoints.sort(key=get_step)
            latest = checkpoints[-1]
            step = get_step(latest)
            local_path = hf_hub_download(repo_id=HF_REPO, filename=latest)
            weights = load_weights(local_path)
            model.load_state_dict(weights)
            start_step = step
            print(f"✓ Loaded latest from Hub: step {step}")
            return start_step, start_epoch
    except Exception as e:
        print(f"Hub check: {e}")

    # Try local (both extensions)
    if os.path.exists(CHECKPOINT_DIR):
        local_ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if
                       f.startswith("step_") and (f.endswith(".safetensors") or f.endswith(".pt"))]
        # Filter to just weights files (not state .pt files that pair with .safetensors)
        local_ckpts = [f for f in local_ckpts if
                       not (f.endswith(".pt") and f.replace(".pt", ".safetensors") in local_ckpts)]
        if local_ckpts:
            def get_step(f):
                return int(f.split("step_")[-1].replace(".safetensors", "").replace(".pt", ""))

            local_ckpts.sort(key=get_step)
            latest = local_ckpts[-1]
            step = get_step(latest)
            weights_path = os.path.join(CHECKPOINT_DIR, latest)
            weights = load_weights(weights_path)
            model.load_state_dict(weights)
            # Try to load optimizer state
            state_path = weights_path.replace(".safetensors", ".pt") if weights_path.endswith(
                ".safetensors") else weights_path
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location=DEVICE, weights_only=False)
                if isinstance(state, dict):
                    try:
                        if "optimizer" in state:
                            optimizer.load_state_dict(state["optimizer"])
                        if "scheduler" in state:
                            scheduler.load_state_dict(state["scheduler"])
                        start_epoch = state.get("epoch", 0)
                    except:
                        pass
            start_step = step
            print(f"✓ Loaded latest local: step {step}")
            return start_step, start_epoch

    print("No checkpoint found, starting fresh")
    return 0, 0


# ============================================================================
# DATALOADER
# ============================================================================
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=0)

# ============================================================================
# MODEL
# ============================================================================
config = TinyFluxConfig()
model = TinyFlux(config).to(DEVICE).to(DTYPE)
print(f"\nParams: {sum(p.numel() for p in model.parameters()):,}")
model = torch.compile(model, mode="reduce-overhead")

# ============================================================================
# OPTIMIZER & SCHEDULER
# ============================================================================
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.01)
total_steps = len(loader) * EPOCHS // GRAD_ACCUM
warmup = min(500, total_steps // 10)


def lr_fn(step):
    if step < warmup: return step / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

# ============================================================================
# LOAD CHECKPOINT
# ============================================================================
print(f"\nLoad target: {LOAD_TARGET}")
start_step, start_epoch = load_checkpoint(model, opt, sched, LOAD_TARGET)

# Override start_step if RESUME_STEP is set
if RESUME_STEP is not None:
    print(f"Overriding start_step: {start_step} -> {RESUME_STEP}")
    start_step = RESUME_STEP

# Log config to tensorboard
writer.add_text("config", json.dumps(config.__dict__, indent=2), 0)
writer.add_text("training_config", json.dumps({
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "lr": LR,
    "epochs": EPOCHS,
    "min_snr": MIN_SNR,
    "shift": SHIFT,
}, indent=2), 0)

# ============================================================================
# SAMPLE PROMPTS FOR PERIODIC GENERATION
# ============================================================================
SAMPLE_PROMPTS = [
    "a photo of a cat sitting on a windowsill",
    "a beautiful sunset over mountains",
    "a portrait of a woman with red hair",
    "a futuristic cityscape at night",
]

# ============================================================================
# TRAINING
# ============================================================================
print(f"\nTraining {EPOCHS} epochs, {total_steps} total steps")
print(f"Resuming from step {start_step}, epoch {start_epoch}")
print(f"Save: {SAVE_EVERY}, Upload: {UPLOAD_EVERY}, Sample: {SAMPLE_EVERY}, Log: {LOG_EVERY}")

model.train()
step = start_step
best = float("inf")

for ep in range(start_epoch, EPOCHS):
    ep_loss = 0
    ep_batches = 0
    pbar = tqdm(loader, desc=f"E{ep + 1}")

    for i, batch in enumerate(pbar):
        latents = batch["latents"]  # Ground truth data (VAE encoded images)
        t5 = batch["t5_embeds"]
        clip = batch["clip_pooled"]

        B, C, H, W = latents.shape

        # ================================================================
        # FLOW MATCHING FORMULATION
        # ================================================================
        # x_1 = data (what we want to generate)
        # x_0 = noise (where we start at inference)
        # x_t = (1-t)*x_0 + t*x_1  (linear interpolation)
        #
        # At t=0: x_t = x_0 (pure noise)
        # At t=1: x_t = x_1 (pure data)
        #
        # Velocity field: v = dx/dt = x_1 - x_0
        # Model learns to predict v given (x_t, t)
        #
        # At inference: start from noise, integrate v from t=0 to t=1
        # ================================================================

        # Reshape data to sequence format: (B, C, H, W) -> (B, H*W, C)
        data = latents.permute(0, 2, 3, 1).reshape(B, H * W, C)  # x_1
        noise = torch.randn_like(data)  # x_0

        # Sample timesteps with logit-normal distribution + Flux shift
        # This biases training towards higher t (closer to data)
        t = torch.sigmoid(torch.randn(B, device=DEVICE))
        t = flux_shift(t, s=SHIFT).to(DTYPE).clamp(1e-4, 1 - 1e-4)

        # Create noisy samples via linear interpolation
        t_expanded = t.view(B, 1, 1)
        x_t = (1 - t_expanded) * noise + t_expanded * data  # Noisy sample at time t

        # Target velocity: direction from noise to data
        v_target = data - noise

        # Create position IDs for RoPE
        img_ids = TinyFlux.create_img_ids(B, H, W, DEVICE)

        # Random guidance scale (for CFG training)
        guidance = torch.rand(B, device=DEVICE, dtype=DTYPE) * 4 + 1  # [1, 5]

        # Forward pass: predict velocity
        with torch.autocast("cuda", dtype=DTYPE):
            v_pred = model(
                hidden_states=x_t,
                encoder_hidden_states=t5,
                pooled_projections=clip,
                timestep=t,
                img_ids=img_ids,
                guidance=guidance,
            )

        # Loss: MSE between predicted and target velocity
        loss_raw = F.mse_loss(v_pred, v_target, reduction="none").mean(dim=[1, 2])

        # Min-SNR weighting: downweight easy timesteps (near t=0 or t=1)
        snr_weights = min_snr_weight(t)
        loss = (loss_raw * snr_weights).mean() / GRAD_ACCUM
        loss.backward()

        if (i + 1) % GRAD_ACCUM == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1

            # Tensorboard logging
            if step % LOG_EVERY == 0:
                writer.add_scalar("train/loss", loss.item() * GRAD_ACCUM, step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), step)
                writer.add_scalar("train/t_mean", t.mean().item(), step)
                writer.add_scalar("train/snr_weight_mean", snr_weights.mean().item(), step)

            # Generate samples
            if step % SAMPLE_EVERY == 0:
                print(f"\n  Generating samples at step {step}...")
                images = generate_samples(model, SAMPLE_PROMPTS, num_steps=20)
                save_samples(images, SAMPLE_PROMPTS, step, SAMPLE_DIR)

            # Save checkpoint
            if step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{step}.pt")
                weights_path = save_checkpoint(model, opt, sched, step, ep, loss.item(), ckpt_path)

                # Upload
                if step % UPLOAD_EVERY == 0:
                    upload_checkpoint(weights_path, step, config, include_logs=True)

        ep_loss += loss.item() * GRAD_ACCUM
        ep_batches += 1
        pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM:.4f}", lr=f"{sched.get_last_lr()[0]:.1e}", step=step)

    avg = ep_loss / max(ep_batches, 1)
    print(f"Epoch {ep + 1} loss: {avg:.4f}")
    writer.add_scalar("train/epoch_loss", avg, ep + 1)

    if avg < best:
        best = avg
        best_path = os.path.join(CHECKPOINT_DIR, "best.pt")
        weights_path = save_checkpoint(model, opt, sched, step, ep, avg, best_path)

        try:
            api.upload_file(
                path_or_fileobj=weights_path,
                path_in_repo="model.safetensors",
                repo_id=HF_REPO,
                commit_message=f"Best model (epoch {ep + 1}, loss {avg:.4f})",
            )
            print(f"  ✓ Uploaded best to {HF_REPO}")
        except Exception as e:
            print(f"  ⚠ Upload failed: {e}")

# ============================================================================
# FINAL
# ============================================================================
print("\nSaving final model...")
final_path = os.path.join(CHECKPOINT_DIR, "final.pt")
weights_path = save_checkpoint(model, opt, sched, step, EPOCHS, best, final_path)

# Final samples
print("Generating final samples...")
images = generate_samples(model, SAMPLE_PROMPTS, num_steps=20)
save_samples(images, SAMPLE_PROMPTS, step, SAMPLE_DIR)

# Final upload
try:
    api.upload_file(path_or_fileobj=weights_path, path_in_repo="model.safetensors", repo_id=HF_REPO)
    config_path = os.path.join(CHECKPOINT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)
    api.upload_file(path_or_fileobj=config_path, path_in_repo="config.json", repo_id=HF_REPO)
    api.upload_folder(folder_path=LOG_DIR, path_in_repo="logs", repo_id=HF_REPO)
    api.upload_folder(folder_path=SAMPLE_DIR, path_in_repo="samples", repo_id=HF_REPO)
    print(f"\n✓ Training complete! https://huggingface.co/{HF_REPO}")
except Exception as e:
    print(f"\n⚠ Final upload failed: {e}")

writer.close()
print(f"Best loss: {best:.4f}")