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

    # LoRA type and parameters
    lora_type: str = "single"  # "double", "single", or "combined"
    rank: int = 16
    alpha: float = 16.0
    adapt_mlp: bool = True  # For single/combined: include MLP layers

    # Training (epoch-based)
    epochs: int = 10  # Number of epochs to train
    batch_size: int = 1
    lr: float = 1e-4
    warmup_epochs: float = 0.5  # Warmup for first half epoch
    train_resolution: int = 512  # 512 for A100, 256 for T4

    # Checkpoints
    save_every_epoch: int = 1  # Save checkpoint every N epochs

    # HuggingFace upload
    hf_repo: Optional[str] = "AbstractPhil/tinyflux-lailah-loras"
    hf_subdir: str = "lora_v1_man_wearing_brown_cap"
    upload_every_epoch: int = 2  # Upload checkpoint every N epochs

    # Sampling
    sample_prompts: List[str] = field(default_factory=lambda: [
        "a red cube on a blue sphere",
        "a cat sitting on a table",
        "A man wearing a brown cap looking sitting at his computer with a black and brown dog resting next to him on the couch."
    ])
    sample_every_epoch: bool = True
    sample_steps: int = 50
    sample_cfg: float = 7.5
    sample_seed: int = 42

    # Experts
    build_lune: bool = False
    build_sol: bool = False

    # Base model
    base_repo: str = "AbstractPhil/tiny-flux-deep"
    base_weights: str = "step_417054.pt"


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
    print(f"LoRA type: {config.lora_type}")
    print(f"Epochs: {config.epochs}, Rank: {config.rank}, LR: {config.lr}")
    print(f"Train resolution: {config.train_resolution}x{config.train_resolution}")

    # Memory estimate
    latent_size = config.train_resolution // 8
    tokens = latent_size * latent_size
    print(f"  Latent: {latent_size}x{latent_size} = {tokens} tokens")
    if tokens > 2048:
        print(f"  ⚠️  Warning: {tokens} tokens may OOM on T4. Try train_resolution=256")

    if config.hf_repo:
        print(f"HF Upload: {config.hf_repo}/{config.hf_subdir} every {config.upload_every_epoch} epochs")

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

    print(f"  Found {n_images} images")

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

    # Free cache-building memory - unload ALL models
    del images, raw_dataset
    zoo.unload("vae")
    zoo.unload("t5")
    zoo.unload("clip")
    zoo.unload("lune")
    zoo.unload("sol")
    torch.cuda.empty_cache()

    # =========================================================================
    # 3. Load model + inject LoRA
    # =========================================================================
    print("\n[3/6] Loading model...")

    from tinyflux.model.lora import (
        DoubleStreamLoRA, DoubleStreamLoRAConfig,
        SingleStreamLoRA, SingleStreamLoRAConfig,
        CombinedLoRA, CombinedLoRAConfig,
    )

    model = zoo.load_tinyflux(
        source=config.base_repo,
        ema_path=config.base_weights,
        train_mode=True,
    )

    # Memory optimizations for T4/Colab
    # Enable memory efficient attention
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    print("  Memory-efficient attention enabled")

    print(f"\n[4/6] Injecting LoRA ({config.lora_type})...")

    # Create appropriate LoRA based on type
    if config.lora_type == "double":
        lora_cfg = DoubleStreamLoRAConfig(
            rank=config.rank,
            alpha=config.alpha,
        )
        lora = DoubleStreamLoRA(model, config=lora_cfg)
    elif config.lora_type == "single":
        lora_cfg = SingleStreamLoRAConfig(
            rank=config.rank,
            alpha=config.alpha,
            adapt_mlp=config.adapt_mlp,
        )
        lora = SingleStreamLoRA(model, config=lora_cfg)
    elif config.lora_type == "combined":
        lora_cfg = CombinedLoRAConfig(
            rank=config.rank,
            alpha=config.alpha,
            adapt_mlp=config.adapt_mlp,
        )
        lora = CombinedLoRA(model, config=lora_cfg)
    else:
        raise ValueError(f"Unknown lora_type: {config.lora_type}. Use 'double', 'single', or 'combined'")

    # =========================================================================
    # 4. Setup sampler (lazy - will load encoders only when sampling)
    # =========================================================================
    print("\n[5/6] Setting up sampler...")

    from tinyflux.trainer.sampling import Sampler, save_samples

    # Don't load encoders yet - will load on demand for sampling
    # This saves ~3GB VRAM during training
    sampler = None  # Created lazily

    def do_sample(epoch_num: int) -> Optional[str]:
        """Generate and save samples, loading encoders as needed."""
        nonlocal sampler

        if not config.sample_prompts:
            return None

        # Ensure encoders are loaded and on GPU
        if zoo.vae is None:
            zoo.load_vae()
        else:
            zoo.onload("vae")

        if zoo.t5 is None:
            zoo.load_t5()
        else:
            zoo.onload("t5")

        if zoo.clip is None:
            zoo.load_clip()
        else:
            zoo.onload("clip")

        # Create sampler if needed
        if sampler is None:
            print("  Initializing sampler...")
            sampler = Sampler(
                zoo=zoo,
                model=model,
                ema=None,
                num_steps=config.sample_steps,
                guidance_scale=config.sample_cfg,
                shift=3.0,
                device=device,
                dtype=dtype,
            )

        model.eval()
        with torch.no_grad():
            sample_images = sampler.generate(
                config.sample_prompts,
                seed=config.sample_seed,
            )
            sample_path = save_samples(
                sample_images,
                config.sample_prompts,
                epoch_num,
                samples_dir,
            )
            print(f"  Saved: {sample_path}")

            if config.hf_repo:
                upload_to_hf(
                    sample_path,
                    config.hf_repo,
                    f"{config.hf_subdir}/samples",
                )

        model.train()

        # On A100 (40GB+), don't offload - plenty of VRAM
        # Only offload on smaller GPUs to fit training
        if torch.cuda.get_device_properties(0).total_memory < 20e9:
            zoo.offload("vae")
            zoo.offload("t5")
            zoo.offload("clip")
            torch.cuda.empty_cache()

        return sample_path

    # =========================================================================
    # 5. Training loop (epoch-based)
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

    # Calculate training metrics
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(config.warmup_epochs * steps_per_epoch)

    print(f"  {n_images} images × {config.repeats} repeats = {steps_per_epoch} steps/epoch")
    print(f"  {config.epochs} epochs = {total_steps} total steps")
    print(f"  Warmup: {warmup_steps} steps ({config.warmup_epochs} epochs)")

    optimizer = torch.optim.AdamW(lora.parameters(), lr=config.lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    global_step = 0
    running_loss = 0.0
    log_every = max(1, steps_per_epoch // 10)  # Log ~10 times per epoch

    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.epochs}")

        for batch in pbar:
            indices = batch['index']
            B = len(indices)

            # Get cached encodings
            latents, t5_embed, clip_embed = cache.get_encodings_batch(indices)
            latents = latents.to(device, dtype=dtype)
            t5_embed = t5_embed.to(device, dtype=dtype)
            clip_embed = clip_embed.to(device, dtype=dtype)

            # Resize latents if training at different resolution
            target_latent_size = config.train_resolution // 8
            if latents.shape[-1] != target_latent_size:
                latents = torch.nn.functional.interpolate(
                    latents,
                    size=(target_latent_size, target_latent_size),
                    mode='bilinear',
                    align_corners=False,
                )

            H = W = latents.shape[-1]

            # Sample timesteps
            t = sample_timesteps(B, device=device, dtype=dtype, shift=3.0)

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
            loss_val = loss.item()
            running_loss += loss_val
            epoch_loss += loss_val
            global_step += 1
            epoch_steps += 1

            if global_step % log_every == 0:
                avg_loss = running_loss / log_every
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
                running_loss = 0.0

        # End of epoch
        avg_epoch_loss = epoch_loss / epoch_steps
        print(f"  Epoch {epoch} complete | Loss: {avg_epoch_loss:.4f}")

        # Checkpoint every N epochs
        if epoch % config.save_every_epoch == 0:
            ckpt_path = os.path.join(config.output_dir, f"lora_epoch_{epoch}.safetensors")
            lora.save(ckpt_path)
            print(f"  Saved: {ckpt_path}")

        # Upload every N epochs
        if config.hf_repo and epoch % config.upload_every_epoch == 0:
            ckpt_path = os.path.join(config.output_dir, f"lora_epoch_{epoch}.safetensors")
            if not os.path.exists(ckpt_path):
                lora.save(ckpt_path)
            upload_to_hf(ckpt_path, config.hf_repo, config.hf_subdir)

        # Sample every epoch
        if config.sample_every_epoch and config.sample_prompts:
            print(f"  Generating samples...")
            do_sample(epoch)

    # Final save
    final_path = os.path.join(config.output_dir, "lora_final.safetensors")
    lora.save(final_path)

    # Final upload
    if config.hf_repo:
        upload_to_hf(final_path, config.hf_repo, config.hf_subdir, "lora_final.safetensors")

    # Final sample
    if config.sample_prompts:
        print("\nGenerating final samples...")
        do_sample(config.epochs)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Epochs: {config.epochs}")
    print(f"  Total steps: {total_steps}")
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
from google.colab import userdata
login(userdata.get("HF_TOKEN"))

# Cell 3: Train!
from tinyflux.examples.train_lora_colab import train_lora, LoRAConfig

config = LoRAConfig(
    # Data
    data_dir="/content/drive/MyDrive/test_1024",
    output_dir="/content/lora_output",
    repeats=100,  # 10 images × 100 repeats = 1000 steps/epoch
    
    # LoRA type: "double", "single", or "combined"
    lora_type="single",  # single-stream for detail refinement
    
    # Training (epoch-based)
    epochs=10,
    rank=16,
    batch_size=1,
    lr=1e-4,
    train_resolution=512,  # 512 for A100, 256 for T4
    
    # HuggingFace
    hf_repo="AbstractPhil/tinyflux-lailah-loras",
    hf_subdir="my_character_single_v1",
    upload_every_epoch=2,
    
    # Sampling
    sample_prompts=[
        "a red cube on a blue sphere",
        "A man wearing a brown cap sitting at his computer with a black and brown dog resting next to him on the couch.",
    ],
    sample_every_epoch=True,
)

model, lora = train_lora(config)
"""

if __name__ == "__main__":
    from huggingface_hub import login
    from google.colab import userdata
    login(userdata.get("HF_TOKEN"))

    config = LoRAConfig(
        data_dir="/content/drive/MyDrive/test_1024",
        output_dir="/content/lora_output",
        repeats=100,
        epochs=10,
        lora_type="single",  # "double", "single", or "combined"
        train_resolution=512,
    )

    model, lora = train_lora(config)