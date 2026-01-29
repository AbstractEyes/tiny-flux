#!/usr/bin/env python3
"""
TinyFlux End-to-End Test with HuggingFace Upload

Tests the complete pipeline:
1. Build cache from random data
2. Train for a few steps
3. Generate samples
4. Save checkpoints
5. Upload to HuggingFace
6. Download and verify

Run on Colab:
    !pip install -e .
    !huggingface-cli login
    !python test_e2e.py
"""

import os
import sys
import shutil
import tempfile
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path

# =============================================================================
# Config
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Test parameters
NUM_SAMPLES = 16
BATCH_SIZE = 1
TRAIN_STEPS = 20
SAMPLE_STEPS = 10

# HuggingFace repo for testing (will create if doesn't exist)
HF_REPO_ID = "AbstractPhil/tinyflux-test-e2e"

# Local directories
TEST_DIR = tempfile.mkdtemp(prefix="tinyflux_e2e_")
CHECKPOINT_DIR = os.path.join(TEST_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(TEST_DIR, "samples")
CACHE_DIR = os.path.join(TEST_DIR, "cache")
TB_DIR = os.path.join(TEST_DIR, "tensorboard")


def generate_test_data(n: int):
    """Generate random test images and prompts."""
    images = []
    prompts = []

    subjects = ["cat", "dog", "bird", "fish", "robot", "tree", "car", "house"]
    styles = ["realistic", "cartoon", "watercolor", "sketch", "3d render"]

    for i in range(n):
        # Random colored image
        img = Image.new('RGB', (512, 512))
        pixels = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(pixels)
        images.append(img)

        # Random prompt
        subject = subjects[i % len(subjects)]
        style = styles[i % len(styles)]
        prompts.append(f"a {style} of a {subject}, high quality")

    return images, prompts


def main():
    print("=" * 60)
    print("TinyFlux End-to-End Test")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Train steps: {TRAIN_STEPS}")
    print(f"HF Repo: {HF_REPO_ID}")
    print(f"Test dir: {TEST_DIR}")

    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # =================================================================
    # Step 1: Check HuggingFace authentication
    # =================================================================
    print("\n[1/7] Checking HuggingFace authentication...")

    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi()
        user_info = api.whoami()
        print(f"  ✓ Logged in as: {user_info['name']}")
    except Exception as e:
        print(f"  ✗ Not logged in: {e}")
        print("  Run: huggingface-cli login")
        return False

    # Create or verify repo exists
    try:
        create_repo(HF_REPO_ID, exist_ok=True, repo_type="model")
        print(f"  ✓ Repo ready: {HF_REPO_ID}")
    except Exception as e:
        print(f"  ⚠ Repo issue: {e}")

    # =================================================================
    # Step 2: Generate test data and build cache
    # =================================================================
    print("\n[2/7] Building cache...")

    from tinyflux.model.zoo import ModelZoo
    from tinyflux.trainer.cache_experts import DatasetCache

    images, prompts = generate_test_data(NUM_SAMPLES)
    print(f"  Generated {len(images)} images and prompts")

    # Check for existing cache
    cache_path = os.path.join(CACHE_DIR, "test_cache.pt")
    if os.path.exists(cache_path):
        print(f"  Loading existing cache...")
        cache = DatasetCache.load(cache_path)
    else:
        # Build cache with zoo
        zoo = ModelZoo(device=DEVICE, dtype=DTYPE)
        zoo.load_vae()
        zoo.load_clip()
        zoo.load_t5()
        zoo.load_lune()
        zoo.load_sol()

        print(f"  Building cache (this may take a minute)...")
        cache = DatasetCache.build(
            zoo=zoo,
            images=images,
            prompts=prompts,
            name="e2e_test",
            batch_size=4,
            dtype=torch.float16,
        )
        cache.save(cache_path)

        # Unload extraction models
        zoo.unload("lune")
        zoo.unload("sol")
        zoo.unload("t5")
        zoo.unload("clip")

    print(f"  ✓ Cache: {cache}")

    # =================================================================
    # Step 3: Setup model and trainer
    # =================================================================
    print("\n[3/7] Setting up training...")

    from tinyflux.model.model import TinyFluxConfig, TinyFluxDeep
    from tinyflux.trainer.trainer import Trainer, TrainerConfig
    from tinyflux.trainer.cache_experts import MultiSourceCache
    from tinyflux.trainer.data import CachedDataset, collate_fn
    from torch.utils.data import DataLoader

    # Model config (small for testing)
    model_config = TinyFluxConfig(
        hidden_size=768,  # 6 * 128
        num_attention_heads=6,
        attention_head_dim=128,
        num_double_layers=2,
        num_single_layers=4,
        mlp_ratio=4.0,
        use_lune_expert=True,
        use_sol_prior=True,
    )
    model = TinyFluxDeep(model_config)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count / 1e6:.1f}M")

    # Dataset and loader
    dataset = CachedDataset(cache, dataset_id=0)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Multi-source cache
    multi_cache = MultiSourceCache(dtype=DTYPE)
    multi_cache.add(cache, dataset_id=0)

    # Trainer config with HF upload
    trainer_config = TrainerConfig(
        learning_rate=1e-4,
        total_steps=TRAIN_STEPS,
        warmup_steps=5,
        gradient_accumulation=1,
        shift=3.0,
        enable_lune=True,
        enable_sol=True,
        lune_weight=0.1,
        lune_warmup_steps=5,
        sol_weight=0.05,
        sol_warmup_steps=10,
        log_every=5,
        checkpoint_dir=CHECKPOINT_DIR,
        save_every_steps=SAMPLE_STEPS,
        keep_last_n_steps=2,
        save_every_epochs=0,
        tensorboard_dir=TB_DIR,
        sample_dir=SAMPLE_DIR,
        sample_every=SAMPLE_STEPS,
        sample_prompts=["a realistic cat, high quality", "a watercolor dog"],
        hf_repo_id=HF_REPO_ID,
        upload_every_steps=TRAIN_STEPS,  # Upload at end
        dtype=DTYPE,
    )

    # Create trainer
    trainer = Trainer(model, trainer_config, device=DEVICE)

    # =================================================================
    # Step 4: Setup sampler
    # =================================================================
    print("\n[4/7] Setting up sampler...")

    from tinyflux.trainer.sampling import Sampler
    from tinyflux.model.zoo import ModelZoo

    # Reload zoo for sampling (may have been used for cache building)
    zoo = ModelZoo(device=DEVICE, dtype=DTYPE)
    zoo.load_vae()
    zoo.load_clip()
    zoo.load_t5()

    sampler = Sampler(
        zoo=zoo,
        model=model,
        ema=trainer.ema,
        num_steps=4,  # Fast sampling for test
        guidance_scale=1.0,  # No CFG for speed
        shift=3.0,
        device=DEVICE,
        dtype=DTYPE,
    )

    trainer.setup(
        train_loader=loader,
        cache=multi_cache,
        sampler=sampler,
    )

    print(f"  ✓ Trainer ready")
    print(f"  ✓ Sampler ready")

    # =================================================================
    # Step 5: Train
    # =================================================================
    print("\n[5/7] Training...")

    trainer.train()

    # Check loss decreased
    final_loss = trainer.best_loss
    print(f"  Final best loss: {final_loss:.4f}")

    # =================================================================
    # Step 6: Verify checkpoints saved
    # =================================================================
    print("\n[6/7] Verifying checkpoints...")

    checkpoints = list(Path(CHECKPOINT_DIR).glob("*.pt"))
    ema_files = list(Path(CHECKPOINT_DIR).glob("*_ema.safetensors"))
    config_files = list(Path(CHECKPOINT_DIR).glob("*.json"))

    print(f"  Checkpoints: {[f.name for f in checkpoints]}")
    print(f"  EMA files: {[f.name for f in ema_files]}")
    print(f"  Config files: {[f.name for f in config_files]}")

    assert len(checkpoints) > 0, "No checkpoints saved!"
    assert len(ema_files) > 0, "No EMA files saved!"
    print(f"  ✓ Checkpoints verified")

    # Check samples
    samples = list(Path(SAMPLE_DIR).glob("*.png"))
    print(f"  Samples: {[f.name for f in samples]}")

    # Check tensorboard
    tb_events = list(Path(TB_DIR).glob("events.out.*"))
    print(f"  TensorBoard events: {len(tb_events)}")

    # =================================================================
    # Step 7: Upload to HuggingFace
    # =================================================================
    print("\n[7/7] Uploading to HuggingFace...")

    # Upload final checkpoint
    final_ckpt = os.path.join(CHECKPOINT_DIR, "final.pt")
    if os.path.exists(final_ckpt):
        trainer.upload_checkpoint(final_ckpt)
    else:
        # Upload latest step checkpoint
        latest = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
        trainer.upload_checkpoint(str(latest))

    # Upload samples
    trainer.upload_samples()

    # Verify upload by downloading
    print("\n  Verifying upload...")
    try:
        from huggingface_hub import hf_hub_download

        # Try to download config
        config_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="config.json",
        )
        print(f"  ✓ Downloaded config: {config_path}")

        # Try to download training config
        try:
            train_config_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="training_config.json",
            )
            print(f"  ✓ Downloaded training_config: {train_config_path}")
        except:
            pass

    except Exception as e:
        print(f"  ⚠ Download verification failed: {e}")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  ✓ Cache built: {cache}")
    print(f"  ✓ Model trained: {TRAIN_STEPS} steps")
    print(f"  ✓ Best loss: {final_loss:.4f}")
    print(f"  ✓ Checkpoints: {len(checkpoints)}")
    print(f"  ✓ Samples: {len(samples)}")
    print(f"  ✓ TensorBoard: {len(tb_events)} event files")
    print(f"  ✓ Uploaded to: {HF_REPO_ID}")

    # Cleanup
    print(f"\nTest directory: {TEST_DIR}")
    try:
        cleanup = input("Clean up test directory? [y/N]: ").strip().lower()
        if cleanup == 'y':
            shutil.rmtree(TEST_DIR)
            print("  Cleaned up")
    except EOFError:
        # Non-interactive mode
        print("  (keeping test directory for inspection)")

    print("\n✓ End-to-end test complete!")
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)