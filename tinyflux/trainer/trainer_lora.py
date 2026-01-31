#!/usr/bin/env python3
"""
Prototype 1: Double-Stream LoRA Training

Tests whether cross-modal attention (txt↔img) is sufficient for concept adaptation.

Expert handling: Lune/Sol run in INFERENCE mode only (frozen, no distillation loss).
We're testing if we can adapt the model without touching the expert system.

Usage:
    python examples/prototype_double_lora.py
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PrototypeConfig:
    # LoRA
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0

    # Training
    steps: int = 2000
    batch_size: int = 4
    lr: float = 1e-4
    warmup_steps: int = 100

    # Data
    num_samples: int = 2000

    # Checkpoints
    save_every: int = 500
    output_dir: str = "./prototype_double_lora"

    # Model
    base_model: str = "AbstractPhil/tiny-flux-deep"
    base_weights: str = "checkpoint_runs/v4_init/lailah_401434_v4_init.safetensors"


def main():
    cfg = PrototypeConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 60)
    print("Prototype 1: Double-Stream LoRA")
    print("=" * 60)
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Rank: {cfg.rank}, Steps: {cfg.steps}, LR: {cfg.lr}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # =========================================================================
    # 1. Load model
    # =========================================================================
    print("\n[1/4] Loading model...")

    from tinyflux.model.zoo import ModelZoo

    zoo = ModelZoo(device=device, dtype=dtype)
    model = zoo.load_tinyflux(
        source=cfg.base_model,
        ema_path=cfg.base_weights,
        train_mode=True,
    )

    # =========================================================================
    # 2. Inject LoRA
    # =========================================================================
    print("\n[2/4] Injecting LoRA...")

    from tinyflux.trainer.lora_double import DoubleStreamLoRA, DoubleStreamLoRAConfig

    lora_config = DoubleStreamLoRAConfig(
        rank=cfg.rank,
        alpha=cfg.alpha,
        dropout=cfg.dropout,
    )

    lora = DoubleStreamLoRA(model, config=lora_config)

    # =========================================================================
    # 3. Setup training
    # =========================================================================
    print("\n[3/4] Setting up training...")

    # Load cache (assumes already built)
    from tinyflux.trainer.cache_experts import MultiExpertCache

    cache_dir = os.path.join(cfg.output_dir, "cache")

    if os.path.exists(os.path.join(cache_dir, "state.json")):
        print("  Loading existing cache...")
        cache = MultiExpertCache.load(cache_dir)
    else:
        print("  Building cache (this may take a while)...")
        from datasets import load_dataset

        zoo.load_vae()
        zoo.load_t5()
        zoo.load_clip()

        ds = load_dataset("AbstractPhil/synthetic-object-relations", "schnell_512_1", split="train")
        if cfg.num_samples < len(ds):
            ds = ds.select(range(cfg.num_samples))

        cache = MultiExpertCache.build_from_hf_dataset(
            ds,
            zoo=zoo,
            output_dir=cache_dir,
            chunk_size=500,
            compile_experts=True,
        )

    print(f"  Cache: {len(cache)} samples")

    # Dataloader
    indices = torch.arange(len(cache))
    dataset = TensorDataset(indices)

    def collate_fn(batch):
        return {'index': torch.stack([b[0] for b in batch])}

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Optimizer (only LoRA params)
    optimizer = torch.optim.AdamW(
        lora.parameters(),
        lr=cfg.lr,
        weight_decay=0.01,
        betas=(0.9, 0.99),
    )

    # Simple linear warmup then constant
    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # =========================================================================
    # 4. Training loop
    # =========================================================================
    print("\n[4/4] Training...")
    print("=" * 60)

    model.train()
    step = 0
    running_loss = 0.0

    pbar = tqdm(total=cfg.steps, desc="Training")

    while step < cfg.steps:
        for batch in loader:
            if step >= cfg.steps:
                break

            indices = batch['index'].tolist()

            # Get cached encodings
            latents = cache.get_latents(indices).to(device, dtype=dtype)
            t5_embed = cache.get_t5(indices).to(device, dtype=dtype)
            clip_embed = cache.get_clip(indices).to(device, dtype=dtype)

            # Get expert features (for inference-mode feeding)
            lune_features = cache.get_lune(indices).to(device, dtype=dtype)
            sol_stats = cache.get_sol_stats(indices).to(device, dtype=dtype)
            sol_spatial = cache.get_sol_spatial(indices).to(device, dtype=dtype)

            B = latents.shape[0]
            H = W = latents.shape[-1]

            # Sample timesteps (flux-shifted toward t=1)
            u = torch.rand(B, device=device)
            t = 1.0 - (1.0 - u) ** 2  # Shift toward t=1 (clean)
            t = t.to(dtype)

            # Sample noise
            noise = torch.randn_like(latents)

            # Flow matching interpolation: x_t = (1-t)*noise + t*data
            # t=0 is pure noise, t=1 is clean data
            from ..utils.predictions import flow_x_t, flow_velocity
            x_t = flow_x_t(latents, noise, t)  # (1-t)*noise + t*data

            # Velocity target: v = data - noise
            v_target = flow_velocity(latents, noise)  # data - noise

            # Reshape for model
            x_t_seq = x_t.flatten(2).transpose(1, 2)  # [B, H*W, C]

            # Position IDs
            from ..model.model import TinyFluxDeep
            img_ids = TinyFluxDeep.create_img_ids(B, H, W, device)

            # Forward (experts in inference mode - just fed, not trained)
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

                # v_pred is [B, N, C], v_target needs same shape
                v_target_seq = v_target.flatten(2).transpose(1, 2)  # [B, H*W, C]

                # Flow matching loss (simple MSE on velocity)
                loss = F.mse_loss(v_pred, v_target_seq)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(lora.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Logging
            running_loss += loss.item()
            step += 1

            if step % 50 == 0:
                avg_loss = running_loss / 50
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
                running_loss = 0.0

            # Save checkpoint
            if step % cfg.save_every == 0:
                path = os.path.join(cfg.output_dir, f"lora_step_{step}.safetensors")
                lora.save(path)

            pbar.update(1)

    pbar.close()

    # Final save
    final_path = os.path.join(cfg.output_dir, "lora_final.safetensors")
    lora.save(final_path)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Final LoRA: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()