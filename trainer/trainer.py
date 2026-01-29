"""
TinyFlux Trainer

Training loop for TinyFlux-Deep with Lune/Sol distillation.

Expected batch format from DataLoader (using collate_fn from data.py):
    {
        'latents': [B, C, H, W] encoded images,
        't5_embeds': [B, L, 768] T5 hidden states,
        'clip_pooled': [B, 768] CLIP pooled features,
        'local_indices': [B] local index within dataset,
        'dataset_ids': [B] which dataset (for cache routing),
        'masks': [B, H, W] optional foreground masks,
    }
"""

import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict, Any, Tuple, List, Callable
from dataclasses import dataclass, field
from tqdm import tqdm

from .losses import compute_main_loss, compute_lune_loss, compute_sol_loss, min_snr_weight
from .schedules import sample_timesteps, flux_shift, get_lune_weight, get_sol_weight, make_cosine_schedule
from .ema import EMA
from .cache_experts import MultiSourceCache


@dataclass
class TrainerConfig:
    """Training configuration."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 1000
    total_steps: int = 100000

    # Batching
    gradient_accumulation: int = 1

    # EMA
    ema_decay: float = 0.9999

    # Timestep sampling
    shift: float = 3.0

    # Loss
    use_snr_weighting: bool = True
    snr_gamma: float = 5.0
    use_huber_loss: bool = False
    huber_delta: float = 0.1

    # Lune distillation
    enable_lune: bool = True
    lune_weight: float = 0.1
    lune_warmup_steps: int = 1000
    lune_dropout: float = 0.1
    lune_mode: str = "cosine"

    # Sol distillation
    enable_sol: bool = True
    sol_weight: float = 0.05
    sol_warmup_steps: int = 2000
    use_spatial_weighting: bool = False  # Weight main loss by Sol spatial

    # Text dropout
    text_dropout: float = 0.0

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5000
    keep_last_n: int = 3

    # Logging
    log_every: int = 100

    # Sampling
    sample_every: int = 2000
    sample_prompts: List[str] = None
    sample_dir: str = "samples"

    # Mixed precision
    dtype: torch.dtype = torch.bfloat16


def apply_text_dropout(
    t5_embeds: torch.Tensor,
    clip_pooled: torch.Tensor,
    dropout_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply text dropout - zero out embeddings for some samples.

    Helps model learn unconditional generation for CFG.
    """
    if dropout_prob <= 0:
        return t5_embeds, clip_pooled

    B = t5_embeds.shape[0]
    mask = torch.rand(B, device=t5_embeds.device) > dropout_prob
    mask = mask.view(B, 1, 1)

    t5_embeds = t5_embeds * mask.float()
    clip_pooled = clip_pooled * mask.view(B, 1).float()

    return t5_embeds, clip_pooled


class Trainer:
    """
    TinyFlux trainer with Lune/Sol distillation.

    Usage:
        trainer = Trainer(model, config)
        trainer.setup(train_loader, lune_cache, sol_cache)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer (only trainable params)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        # Scheduler
        schedule_fn = make_cosine_schedule(config.total_steps, config.warmup_steps)
        self.scheduler = LambdaLR(self.optimizer, schedule_fn)

        # EMA
        self.ema = EMA(model, decay=config.ema_decay)

        # Data (set via setup())
        self.train_loader: Optional[DataLoader] = None
        self.cache: Optional[MultiSourceCache] = None

        # Sampling (set via setup())
        self.sampler = None

        # State
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Logging
        self.writer = None

        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        if config.sample_prompts:
            os.makedirs(config.sample_dir, exist_ok=True)

    def setup(
        self,
        train_loader: DataLoader,
        cache: Optional[MultiSourceCache] = None,
        sampler=None,
        tensorboard_dir: Optional[str] = None,
    ):
        """
        Setup training data and optional components.

        Args:
            train_loader: DataLoader yielding batches
            cache: MultiSourceCache with all precached data
            sampler: Optional Sampler for generating samples during training
            tensorboard_dir: Optional path for TensorBoard logs
        """
        self.train_loader = train_loader
        self.cache = cache
        self.sampler = sampler

        if tensorboard_dir:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(tensorboard_dir)

    def train(self):
        """Main training loop."""
        cfg = self.config

        print(f"\nStarting training:")
        print(f"  Total steps: {cfg.total_steps}")
        print(f"  Gradient accumulation: {cfg.gradient_accumulation}")
        print(f"  Learning rate: {cfg.learning_rate}")
        print(f"  Lune: {cfg.enable_lune} (weight={cfg.lune_weight}, warmup={cfg.lune_warmup_steps})")
        print(f"  Sol: {cfg.enable_sol} (weight={cfg.sol_weight}, warmup={cfg.sol_warmup_steps})")
        print(f"  Text dropout: {cfg.text_dropout}")

        while self.step < cfg.total_steps:
            self._train_epoch()

        # Final save
        self._save_checkpoint(final=True)

        if self.writer:
            self.writer.close()

        print(f"\n✓ Training complete! Best loss: {self.best_loss:.4f}")

    def _train_epoch(self):
        """Train for one epoch."""
        cfg = self.config
        self.model.train()

        ep_loss = 0.0
        ep_main = 0.0
        ep_lune = 0.0
        ep_sol = 0.0
        ep_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            losses = self._train_step(batch)

            # Gradient accumulation
            if (batch_idx + 1) % cfg.gradient_accumulation == 0:
                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    cfg.grad_clip
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.ema.update(self.model)
                self.step += 1

                # Logging
                if self.step % cfg.log_every == 0 and self.writer:
                    self.writer.add_scalar('train/loss', losses['total'], self.step)
                    self.writer.add_scalar('train/main_loss', losses['main'], self.step)
                    self.writer.add_scalar('train/lune_loss', losses['lune'], self.step)
                    self.writer.add_scalar('train/sol_loss', losses['sol'], self.step)
                    self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.step)
                    self.writer.add_scalar('train/grad_norm', grad_norm.item(), self.step)
                    self.writer.add_scalar('train/lune_weight', losses['lune_w'], self.step)
                    self.writer.add_scalar('train/sol_weight', losses['sol_w'], self.step)

                # Checkpointing
                if self.step % cfg.save_every == 0:
                    self._save_checkpoint()

                # Sampling
                if cfg.sample_prompts and self.step % cfg.sample_every == 0:
                    self._generate_samples()

            # Accumulate for epoch stats
            ep_loss += losses['total']
            ep_main += losses['main']
            ep_lune += losses['lune']
            ep_sol += losses['sol']
            ep_batches += 1

            pbar.set_postfix(
                loss=f"{losses['total']:.4f}",
                main=f"{losses['main']:.4f}",
                lune=f"{losses['lune']:.4f}",
                sol=f"{losses['sol']:.4f}",
                step=self.step,
            )

            if self.step >= cfg.total_steps:
                break

        self.epoch += 1

        # Epoch summary
        n = max(ep_batches, 1)
        avg_loss = ep_loss / n
        print(f"\nEpoch {self.epoch}: loss={avg_loss:.4f}, main={ep_main/n:.4f}, "
              f"lune={ep_lune/n:.4f}, sol={ep_sol/n:.4f}")

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self._save_checkpoint(best=True)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        cfg = self.config

        # Unpack batch
        latents = batch['latents'].to(self.device, non_blocking=True)
        t5 = batch['t5_embeds'].to(self.device, non_blocking=True)
        clip = batch['clip_pooled'].to(self.device, non_blocking=True)
        local_indices = batch.get('local_indices')
        dataset_ids = batch.get('dataset_ids')
        masks = batch.get('masks')

        if masks is not None:
            masks = masks.to(self.device, non_blocking=True)

        B, C, H, W = latents.shape

        # Reshape: [B, C, H, W] -> [B, H*W, C]
        data = latents.permute(0, 2, 3, 1).reshape(B, H * W, C)
        noise = torch.randn_like(data)

        # Text dropout
        if cfg.text_dropout > 0:
            t5, clip = apply_text_dropout(t5, clip, cfg.text_dropout)

        # Sample timesteps
        t = torch.sigmoid(torch.randn(B, device=self.device))
        t = flux_shift(t, shift=cfg.shift).to(cfg.dtype).clamp(1e-4, 1 - 1e-4)

        # Rectified flow interpolation
        t_exp = t.view(B, 1, 1)
        x_t = (1 - t_exp) * noise + t_exp * data
        v_target = data - noise

        # Position IDs
        from ..model.model import TinyFluxDeep
        img_ids = TinyFluxDeep.create_img_ids(B, H, W, self.device)

        # Get expert features from cache
        lune_features = None
        sol_stats = None
        sol_spatial = None

        if self.cache is not None and local_indices is not None and dataset_ids is not None:
            if cfg.enable_lune:
                lune_features = self.cache.get_lune(local_indices, dataset_ids, t)
                # Teacher dropout - forces model to use predictor
                if lune_features is not None and random.random() < cfg.lune_dropout:
                    lune_features = None

            if cfg.enable_sol:
                sol_stats, sol_spatial = self.cache.get_sol(local_indices, dataset_ids, t)

        # Forward
        with torch.autocast(self.device, dtype=cfg.dtype):
            result = self.model(
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

        # === Losses ===

        # SNR weighting
        snr_weights = min_snr_weight(t, cfg.snr_gamma) if cfg.use_snr_weighting else None

        # Spatial weighting from Sol
        spatial_w = sol_spatial if cfg.use_spatial_weighting and sol_spatial is not None else None

        # Main loss
        main_loss = compute_main_loss(
            v_pred, v_target,
            mask=masks,
            spatial_weights=spatial_w,
            snr_weights=snr_weights,
            use_huber=cfg.use_huber_loss,
            huber_delta=cfg.huber_delta,
        )

        # Lune distillation
        lune_loss = torch.tensor(0.0, device=self.device)
        if lune_features is not None and 'lune_pred' in expert_info:
            lune_loss = compute_lune_loss(
                expert_info['lune_pred'],
                lune_features,
                mode=cfg.lune_mode,
            )

        # Sol distillation
        sol_loss = torch.tensor(0.0, device=self.device)
        if sol_stats is not None and 'sol_stats_pred' in expert_info:
            sol_loss = compute_sol_loss(
                expert_info['sol_stats_pred'],
                expert_info.get('sol_spatial_pred'),
                sol_stats,
                sol_spatial,
            )

        # Total with warmup weights
        lune_w = get_lune_weight(self.step, cfg.lune_warmup_steps, cfg.lune_weight)
        sol_w = get_sol_weight(self.step, cfg.sol_warmup_steps, cfg.sol_weight)

        total_loss = main_loss + lune_w * lune_loss + sol_w * sol_loss

        # Backward (scaled for gradient accumulation)
        (total_loss / cfg.gradient_accumulation).backward()

        return {
            'total': total_loss.item(),
            'main': main_loss.item(),
            'lune': lune_loss.item(),
            'sol': sol_loss.item(),
            'lune_w': lune_w,
            'sol_w': sol_w,
        }

    def _save_checkpoint(self, best: bool = False, final: bool = False):
        """Save checkpoint."""
        cfg = self.config

        # Get model state (handle compiled)
        if hasattr(self.model, '_orig_mod'):
            model_state = self.model._orig_mod.state_dict()
        else:
            model_state = self.model.state_dict()

        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'model': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ema': self.ema.state_dict(),
            'config': cfg,
        }

        # Determine path
        if final:
            name = 'final'
        elif best:
            name = 'best'
        else:
            name = f'step_{self.step}'

        path = os.path.join(cfg.checkpoint_dir, f'{name}.pt')
        torch.save(checkpoint, path)

        # Save EMA weights as safetensors
        ema_path = os.path.join(cfg.checkpoint_dir, f'{name}_ema.safetensors')
        try:
            from safetensors.torch import save_file
            save_file(self.ema.shadow, ema_path)
        except ImportError:
            pass

        print(f"  ✓ Saved: {path}")

        # Cleanup old
        if not best and not final:
            self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Keep only last N checkpoints."""
        cfg = self.config

        checkpoints = sorted([
            f for f in os.listdir(cfg.checkpoint_dir)
            if f.startswith('step_') and f.endswith('.pt')
        ], key=lambda x: int(x.split('_')[1].split('.')[0]))

        while len(checkpoints) > cfg.keep_last_n:
            old = checkpoints.pop(0)
            old_path = os.path.join(cfg.checkpoint_dir, old)
            os.remove(old_path)
            # Also EMA
            ema_path = old_path.replace('.pt', '_ema.safetensors')
            if os.path.exists(ema_path):
                os.remove(ema_path)

    def _generate_samples(self):
        """Generate samples during training."""
        cfg = self.config

        if self.sampler is None or not cfg.sample_prompts:
            return

        print(f"\n  Generating samples at step {self.step}...")

        try:
            images = self.sampler.generate(
                cfg.sample_prompts,
                seed=self.step,  # Reproducible per step
            )
            path = self.sampler.save(images, cfg.sample_prompts, self.step, cfg.sample_dir)
            print(f"  ✓ Saved: {path}")

            # Log to tensorboard
            if self.writer is not None:
                from torchvision.utils import make_grid
                grid = make_grid(images, nrow=2, padding=2)
                self.writer.add_image('samples', grid, self.step)
        except Exception as e:
            print(f"  ✗ Sample generation failed: {e}")

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device)

        self.step = ckpt['step']
        self.epoch = ckpt['epoch']
        self.best_loss = ckpt.get('best_loss', float('inf'))

        # Model
        if hasattr(self.model, '_orig_mod'):
            self.model._orig_mod.load_state_dict(ckpt['model'])
        else:
            self.model.load_state_dict(ckpt['model'])

        # Optimizer/scheduler
        if load_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])

        # EMA
        self.ema.load_state_dict(ckpt['ema'])

        print(f"  ✓ Loaded: {path} (step={self.step}, epoch={self.epoch})")


# =============================================================================
# Convenience: Simple training function
# =============================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    cache: Optional[MultiSourceCache] = None,
    sampler=None,
    config: Optional[TrainerConfig] = None,
    device: str = "cuda",
    tensorboard_dir: Optional[str] = None,
) -> Trainer:
    """
    Convenience function to train a model.

    Args:
        model: TinyFluxDeep model
        train_loader: DataLoader
        cache: MultiSourceCache with precached data
        sampler: Optional Sampler for generating samples
        config: Training config
        device: Device
        tensorboard_dir: Optional TensorBoard directory

    Returns the Trainer for further use.
    """
    cfg = config or TrainerConfig()
    trainer = Trainer(model, cfg, device)
    trainer.setup(train_loader, cache, sampler, tensorboard_dir)
    trainer.train()
    return trainer