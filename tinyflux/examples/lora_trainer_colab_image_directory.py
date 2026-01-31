"""
TinyFlux LoRA Training - Colab Edition

Simple setup for testing LoRA with a small local dataset.

Directory structure expected:
    /content/drive/MyDrive/lora_dataset/
        image1.png
        image1.txt  (caption)
        image2.jpg
        image2.txt
        ...

Or with a single prompts file:
    /content/drive/MyDrive/lora_dataset/
        image1.png
        image2.jpg
        prompts.txt  (one line per image, alphabetical order)

Usage:
    from tinyflux.examples.train_lora_colab import train_lora, LoRAConfig

    config = LoRAConfig(
        data_dir="/content/drive/MyDrive/lora_dataset",
        output_dir="/content/lora_output",
        hf_repo="AbstractPhil/tiny-flux-lora",
        hf_subdir="my_lora_v1",
        repeats=100,
        steps=1000,
    )

    train_lora(config)
"""

import os
import torch
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """Configuration for LoRA training."""

    # Data
    data_dir: str = "/content/drive/MyDrive/lora_dataset"
    output_dir: str = "/content/lora_output"

    # Dataset inflation
    repeats: int = 100  # Repeat each image N times per epoch

    # LoRA
    rank: int = 16
    alpha: float = 16.0

    # Training
    steps: int = 1000
    batch_size: int = 1
    lr: float = 1e-4
    warmup_steps: int = 50

    # Checkpoints
    save_every: int = 250  # Save checkpoint every N steps

    # HuggingFace upload
    hf_repo: Optional[str] = None  # e.g., "AbstractPhil/tiny-flux-lora"
    hf_subdir: str = "lora_v1"     # Subdirectory in repo
    upload_every: int = 500        # Upload checkpoint every N steps

    # Sampling
    sample_prompts: List[str] = field(default_factory=lambda: [
        "a red cube on a blue sphere",
        "a cat sitting on a table",
    ])
    sample_every_epoch: bool = True
    sample_steps: int = 20
    sample_cfg: float = 4.0
    sample_seed: int = 42

    # Experts
    build_lune: bool = True
    build_sol: bool = True

    # Base model
    base_repo: str = "AbstractPhil/tiny-flux-deep"
    base_weights: str = "checkpoint_runs/v4_init/lailah_401434_v4_init.safetensors"


def upload_to_hf(
    local_path: str,
    repo_id: str,
    subdir: str,
    filename: Optional[str] = None,
):
    """Upload file to HuggingFace repo."""
    from huggingface_hub import HfApi

    api = HfApi()

    if filename is None:
        filename = os.path.basename(local_path)

    path_in_repo = f"{subdir}/{filename}" if subdir else filename

    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  ✓ Uploaded to {repo_id}/{path_in_repo}")
    except Exception as e:
        print(f"  ✗ Upload failed: {e}")


def train_lora(config: Optional[LoRAConfig] = None, **kwargs):
    """
    Main training function for Colab.

    Args:
        config: LoRAConfig instance, or pass kwargs directly
    """
    import torch.nn.functional as F
    from tqdm.auto import tqdm

    # Build config from kwargs if not provided
    if config is None:
        config = LoRAConfig(**kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 60)
    print("TinyFlux LoRA Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data: {config.data_dir}")
    print(f"Repeats: {config.repeats}")
    print(f"Steps: {config.steps}, Rank: {config.rank}, LR: {config.lr}")
    if config.hf_repo:
        print(f"HF Upload: {config.hf_repo}/{config.hf_subdir} every {config.upload_every} steps")

    os.makedirs(config.output_dir, exist_ok=True)
    cache_dir = os.path.join(config.output_dir, "cache")
    samples_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # =========================================================================
    # 1. Load dataset
    # =========================================================================
    print("\n[1/6] Loading images...")

    from tinyflux.trainer.data_directory import (
        DirectoryDataset,
        create_dataloader,
    )

    raw_dataset = DirectoryDataset(config.data_dir, repeats=1, target_size=512)
    images, prompts = raw_dataset.get_images_and_prompts()
    n_images = len(images)

    # Calculate epoch size
    epoch_size = n_images * config.repeats
    steps_per_epoch = epoch_size // config.batch_size

    print(f"  {n_images} images × {config.repeats} repeats = {epoch_size} samples/epoch")
    print(f"  ~{steps_per_epoch} steps/epoch")

    # =========================================================================
    # 2. Build cache
    # =========================================================================
    print("\n[2/6] Building cache...")

    from tinyflux.model.zoo import ModelZoo
    from tinyflux.trainer.cache_experts import DatasetCache

    zoo = ModelZoo(device=device, dtype=dtype)

    cache_meta = os.path.join(cache_dir, "meta.pt")
    if os.path.exists(cache_meta):
        print("  Loading existing cache...")
        cache = DatasetCache.load(cache_dir)
    else:
        print("  Building new cache (this takes a few minutes)...")
        cache = DatasetCache.build(
            zoo,
            images,
            prompts,
            name="lora_dataset",
            build_lune=config.build_lune,
            build_sol=config.build_sol,
            batch_size=min(4, n_images),
            sol_batch_size=1,
            dtype=torch.float16,
            compile_experts=False,
        )
        cache.save(cache_dir)

    print(f"  Cache: {len(cache)} samples")

    # Free cache-building memory
    del images, raw_dataset
    zoo.unload("lune")
    zoo.unload("sol")
    torch.cuda.empty_cache()

    # =========================================================================
    # 3. Load model + inject LoRA
    # =========================================================================
    print("\n[3/6] Loading model...")

    from tinyflux.model.lora import DoubleStreamLoRA, DoubleStreamLoRAConfig

    model = zoo.load_tinyflux(
        source=config.base_repo,
        ema_path=config.base_weights,
        train_mode=True,
    )

    print("\n[4/6] Injecting LoRA...")

    lora_config = DoubleStreamLoRAConfig(
        rank=config.rank,
        alpha=config.alpha,
    )
    lora = DoubleStreamLoRA(model, config=lora_config)

    # =========================================================================
    # 4. Setup sampler
    # =========================================================================
    print("\n[5/6] Setting up sampler...")

    from tinyflux.trainer.sampling import Sampler, save_samples

    # Keep VAE/T5/CLIP loaded for sampling
    if zoo.vae is None:
        zoo.load_vae()
    if zoo.t5 is None:
        zoo.load_t5()
    # CLIP should already be loaded

    sampler = Sampler(
        zoo=zoo,
        model=model,
        ema=None,  # No EMA for LoRA
        num_steps=config.sample_steps,
        guidance_scale=config.sample_cfg,
        shift=3.0,
        device=device,
        dtype=dtype,
    )

    # =========================================================================
    # 5. Training loop
    # =========================================================================
    print("\n[6/6] Training...")

    from tinyflux.trainer.schedules import sample_timesteps
    from tinyflux.utils.predictions import flow_x_t, flow_velocity
    from tinyflux.model.model import TinyFluxDeep

    loader = create_dataloader(
        cache,
        repeats=config.repeats,
        batch_size=config.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(lora.parameters(), lr=config.lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    step = 0
    epoch = 0
    running_loss = 0.0

    pbar = tqdm(total=config.steps, desc="Training LoRA")

    while step < config.steps:
        epoch += 1
        epoch_start_step = step

        for batch in loader:
            if step >= config.steps:
                break

            indices = batch['index']
            B = len(indices)

            # Get cached encodings
            latents, t5_embed, clip_embed = cache.get_encodings_batch(indices)
            latents = latents.to(device, dtype=dtype)
            t5_embed = t5_embed.to(device, dtype=dtype)
            clip_embed = clip_embed.to(device, dtype=dtype)

            H = W = latents.shape[-1]

            # Sample timesteps
            t = sample_timesteps(B, device, dtype, shift=3.0)

            # Get expert features
            lune_features = cache.get_lune(indices, t)
            if lune_features is not None:
                lune_features = lune_features.to(device, dtype=dtype)

            sol_stats, sol_spatial = cache.get_sol(indices, t)
            if sol_stats is not None:
                sol_stats = sol_stats.to(device, dtype=dtype)
                sol_spatial = sol_spatial.to(device, dtype=dtype)

            # Flow matching
            noise = torch.randn_like(latents)
            x_t = flow_x_t(latents, noise, t)
            v_target = flow_velocity(latents, noise)

            # Reshape for model
            x_t_seq = x_t.flatten(2).transpose(1, 2)
            v_target_seq = v_target.flatten(2).transpose(1, 2)

            # Position IDs
            img_ids = TinyFluxDeep.create_img_ids(B, H, W, device)

            # Forward
            optimizer.zero_grad()

            with torch.autocast(device, dtype=dtype):
                v_pred = model(
                    hidden_states=x_t_seq,
                    encoder_hidden_states=t5_embed,
                    pooled_projections=clip_embed,
                    timestep=t,
                    img_ids=img_ids,
                    lune_features=lune_features,
                    sol_stats=sol_stats,
                    sol_spatial=sol_spatial,
                )

                loss = F.mse_loss(v_pred, v_target_seq)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Logging
            running_loss += loss.item()
            step += 1

            if step % 10 == 0:
                avg_loss = running_loss / 10
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    epoch=epoch,
                )
                running_loss = 0.0

            # Local checkpoint
            if step % config.save_every == 0:
                ckpt_path = os.path.join(config.output_dir, f"lora_step_{step}.safetensors")
                lora.save(ckpt_path)

            # HuggingFace upload
            if config.hf_repo and step % config.upload_every == 0:
                ckpt_path = os.path.join(config.output_dir, f"lora_step_{step}.safetensors")
                if not os.path.exists(ckpt_path):
                    lora.save(ckpt_path)
                upload_to_hf(ckpt_path, config.hf_repo, config.hf_subdir)

            pbar.update(1)

        # End of epoch: sample
        if config.sample_every_epoch and config.sample_prompts:
            print(f"\n  [Epoch {epoch}] Generating samples...")
            model.eval()

            with torch.no_grad():
                sample_images = sampler.generate(
                    config.sample_prompts,
                    seed=config.sample_seed,
                )
                sample_path = save_samples(
                    sample_images,
                    config.sample_prompts,
                    step,
                    samples_dir,
                )
                print(f"  Saved: {sample_path}")

                # Upload sample to HF
                if config.hf_repo:
                    upload_to_hf(
                        sample_path,
                        config.hf_repo,
                        f"{config.hf_subdir}/samples",
                    )

            model.train()

    pbar.close()

    # Final save
    final_path = os.path.join(config.output_dir, "lora_final.safetensors")
    lora.save(final_path)

    # Final upload
    if config.hf_repo:
        upload_to_hf(final_path, config.hf_repo, config.hf_subdir, "lora_final.safetensors")

    # Final sample
    if config.sample_prompts:
        print("\nGenerating final samples...")
        model.eval()
        with torch.no_grad():
            final_images = sampler.generate(config.sample_prompts, seed=config.sample_seed)
            final_sample_path = save_samples(final_images, config.sample_prompts, step, samples_dir)
            print(f"  Saved: {final_sample_path}")
            if config.hf_repo:
                upload_to_hf(final_sample_path, config.hf_repo, f"{config.hf_subdir}/samples")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Final LoRA: {final_path}")
    if config.hf_repo:
        print(f"  HF Repo: https://huggingface.co/{config.hf_repo}/tree/main/{config.hf_subdir}")
    print("=" * 60)

    return model, lora


# =============================================================================
# Colab cell helper
# =============================================================================

COLAB_SETUP = """
# Cell 1: Mount Drive and install
from google.colab import drive
drive.mount('/content/drive')

!pip install -q safetensors accelerate huggingface_hub
!pip install -q git+https://github.com/AbstractPhil/tinyflux.git

# Cell 2: Login to HuggingFace (for uploads)
from huggingface_hub import login
login()

# Cell 3: Train!
from tinyflux.examples.train_lora_colab import train_lora, LoRAConfig

config = LoRAConfig(
    # Data
    data_dir="/content/drive/MyDrive/lora_dataset",
    output_dir="/content/lora_output",
    repeats=100,
    
    # Training
    steps=1000,
    rank=16,
    batch_size=1,
    lr=1e-4,
    
    # HuggingFace
    hf_repo="YourUsername/tiny-flux-lora",  # Your repo
    hf_subdir="my_character_v1",            # Subdirectory for this LoRA
    upload_every=500,
    
    # Sampling
    sample_prompts=[
        "a red cube on a blue sphere",
        "your custom prompt here",
    ],
    sample_every_epoch=True,
)

model, lora = train_lora(config)
"""

if __name__ == "__main__":
    print("Colab setup:\n")
    print(COLAB_SETUP)