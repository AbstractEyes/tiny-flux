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
from .schedules import sample_timesteps, get_lune_weight, get_sol_weight, make_cosine_schedule
from .ema import EMA
from .cache_experts import MultiSourceCache
from tinyflux.utils.predictions import flow_x_t, flow_velocity


@dataclass
class TrainerConfig:
    """Training configuration."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    optimizer: str = "adamw"  # "adamw", "adamw_8bit", "adafactor"

    # Schedule
    warmup_steps: int = 1000
    total_steps: int = 100000
    max_epochs: Optional[int] = None  # None = unlimited, stops at total_steps
    lr_scheduler: str = "cosine"  # "cosine", "linear", "constant"
    min_lr: float = 0.0  # Minimum LR for cosine decay
    warmup_type: str = "linear"  # "linear", "cosine"

    # Batching
    gradient_accumulation: int = 1

    # Memory optimization
    gradient_checkpointing: bool = False
    compile_mode: Optional[str] = None  # "reduce-overhead", "max-autotune", None

    # EMA
    ema_decay: float = 0.9999

    # Timestep sampling
    shift: float = 3.0

    # Loss
    use_snr_weighting: bool = True
    snr_gamma: float = 5.0
    use_huber_loss: bool = False
    huber_delta: float = 0.1

    # Flow matching options
    logit_normal_sampling: bool = True  # Sample t from logit-normal (vs uniform)
    logit_mean: float = 0.0  # Mean for logit-normal sampling
    logit_std: float = 1.0  # Std for logit-normal sampling

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
    sol_dropout: float = 0.1  # Drop teacher attention to force predictor
    use_spatial_weighting: bool = False  # Weight main loss by Sol spatial

    # Text dropout
    text_dropout: float = 0.0

    # Checkpointing - Step based
    checkpoint_dir: str = "checkpoints"
    save_every_steps: int = 5000
    keep_last_n_steps: int = 3

    # Checkpointing - Epoch based
    save_every_epochs: int = 1  # 0 to disable
    keep_last_n_epochs: int = 3

    # Logging
    log_every: int = 100
    tensorboard_dir: Optional[str] = None

    # Sampling
    sample_every: int = 2000
    sample_prompts: List[str] = None
    sample_dir: str = "samples"

    # Mixed precision
    dtype: torch.dtype = torch.bfloat16

    # HuggingFace upload
    hf_repo_id: Optional[str] = None  # e.g. "AbstractPhil/tinyflux-deep"
    upload_every_steps: int = 0  # 0 to disable step uploads
    upload_every_epochs: int = 0  # 0 to disable epoch uploads

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.dtype):
                d[k] = str(v)
            elif isinstance(v, tuple):
                d[k] = list(v)
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainerConfig":
        """Create from dict."""
        # Handle dtype conversion
        if 'dtype' in d and isinstance(d['dtype'], str):
            dtype_map = {
                'torch.float32': torch.float32,
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
            }
            d['dtype'] = dtype_map.get(d['dtype'], torch.bfloat16)
        # Handle betas tuple
        if 'betas' in d and isinstance(d['betas'], list):
            d['betas'] = tuple(d['betas'])
        return cls(**d)

    def save(self, path: str):
        """Save config to JSON."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainerConfig":
        """Load config from JSON."""
        import json
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


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

    Features:
        - Step and epoch-based checkpointing
        - Rolling checkpoint cleanup
        - TensorBoard logging
        - HuggingFace upload integration
        - Resume from checkpoint

    Usage:
        trainer = Trainer(model, config)
        trainer.setup(train_loader, cache)
        trainer.train()

        # Resume
        trainer.load_checkpoint("checkpoints/step_5000.pt")
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device

        # Gradient checkpointing
        if config.gradient_checkpointing:
            if hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
            elif hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            print("  ✓ Gradient checkpointing enabled")

        self.model = model.to(device)

        # Compile model if requested
        if config.compile_mode:
            self.model = torch.compile(self.model, mode=config.compile_mode)
            print(f"  ✓ Model compiled with mode={config.compile_mode}")

        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = self._create_optimizer(trainable_params, config)

        # Scheduler
        self.scheduler = self._create_scheduler(self.optimizer, config)

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
        self.epoch_losses: List[float] = []

        # Logging
        self.writer = None

        # Model config for saving
        self.model_config: Optional[Dict[str, Any]] = None
        if hasattr(model, 'config'):
            from dataclasses import asdict, is_dataclass
            if is_dataclass(model.config):
                cfg_dict = asdict(model.config)
                # Convert tuple to list for JSON serialization
                if 'axes_dims_rope' in cfg_dict:
                    cfg_dict['axes_dims_rope'] = list(cfg_dict['axes_dims_rope'])
                self.model_config = cfg_dict
            elif hasattr(model.config, '__dict__'):
                self.model_config = model.config.__dict__

        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        if config.sample_prompts:
            os.makedirs(config.sample_dir, exist_ok=True)

    def _create_optimizer(self, params, config: TrainerConfig):
        """Create optimizer based on config."""
        if config.optimizer == "adamw":
            return AdamW(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=config.betas,
            )
        elif config.optimizer == "adamw_8bit":
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(
                    params,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    betas=config.betas,
                )
            except ImportError:
                print("  ⚠ bitsandbytes not installed, falling back to AdamW")
                return AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay, betas=config.betas)
        elif config.optimizer == "adafactor":
            from transformers import Adafactor
            return Adafactor(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                scale_parameter=False,
                relative_step=False,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

    def _create_scheduler(self, optimizer, config: TrainerConfig):
        """Create LR scheduler based on config."""
        total_steps = config.total_steps
        warmup_steps = config.warmup_steps
        min_lr_ratio = config.min_lr / config.learning_rate if config.learning_rate > 0 else 0

        if config.lr_scheduler == "cosine":
            def schedule_fn(step):
                # Warmup
                if step < warmup_steps:
                    if config.warmup_type == "cosine":
                        return 0.5 * (1 - math.cos(math.pi * step / warmup_steps))
                    else:  # linear
                        return step / warmup_steps
                # Cosine decay to min_lr
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
        elif config.lr_scheduler == "linear":
            def schedule_fn(step):
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return max(min_lr_ratio, 1 - progress * (1 - min_lr_ratio))
        elif config.lr_scheduler == "constant":
            def schedule_fn(step):
                if step < warmup_steps:
                    return step / warmup_steps
                return 1.0
        else:
            raise ValueError(f"Unknown lr_scheduler: {config.lr_scheduler}")

        return LambdaLR(optimizer, schedule_fn)

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
            tensorboard_dir: Optional path for TensorBoard logs (overrides config)
        """
        self.train_loader = train_loader
        self.cache = cache
        self.sampler = sampler

        # TensorBoard
        tb_dir = tensorboard_dir or self.config.tensorboard_dir
        if tb_dir:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(tb_dir)

        # Save configs
        self._save_configs()

    def _save_configs(self):
        """Save training and model configs to checkpoint directory."""
        import json
        cfg = self.config

        # Training config
        trainer_config_path = os.path.join(cfg.checkpoint_dir, "training_config.json")
        cfg.save(trainer_config_path)

        # Model config
        if self.model_config:
            model_config_path = os.path.join(cfg.checkpoint_dir, "config.json")
            with open(model_config_path, 'w') as f:
                json.dump(self.model_config, f, indent=2)

    def _log_step(self, losses: Dict[str, float], lr: float, grad_norm: Optional[float] = None):
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return

        # Losses
        for name, value in losses.items():
            self.writer.add_scalar(f'loss/{name}', value, self.step)

        # Learning rate
        self.writer.add_scalar('train/lr', lr, self.step)

        # Gradient norm
        if grad_norm is not None:
            self.writer.add_scalar('train/grad_norm', grad_norm, self.step)

        # Step/epoch
        self.writer.add_scalar('train/epoch', self.epoch, self.step)

    def _log_epoch(self, epoch_loss: float):
        """Log epoch-level metrics."""
        if self.writer is None:
            return

        self.writer.add_scalar('epoch/loss', epoch_loss, self.epoch)
        self.writer.add_scalar('epoch/step', self.step, self.epoch)

    def train(self):
        """Main training loop. Runs until total_steps or max_epochs (whichever first)."""
        cfg = self.config

        print(f"\nStarting training:")
        print(f"  Total steps: {cfg.total_steps}")
        if cfg.max_epochs:
            print(f"  Max epochs: {cfg.max_epochs}")
        print(f"  Gradient accumulation: {cfg.gradient_accumulation}")
        print(f"  Learning rate: {cfg.learning_rate} (scheduler={cfg.lr_scheduler}, min_lr={cfg.min_lr})")
        print(f"  Flow: shift={cfg.shift}, logit_normal={cfg.logit_normal_sampling} (μ={cfg.logit_mean}, σ={cfg.logit_std})")
        print(f"  Lune: {cfg.enable_lune} (weight={cfg.lune_weight}, warmup={cfg.lune_warmup_steps}, dropout={cfg.lune_dropout})")
        print(f"  Sol: {cfg.enable_sol} (weight={cfg.sol_weight}, warmup={cfg.sol_warmup_steps}, dropout={cfg.sol_dropout})")
        print(f"  Text dropout: {cfg.text_dropout}")

        while self.step < cfg.total_steps:
            if cfg.max_epochs and self.epoch >= cfg.max_epochs:
                print(f"\n✓ Reached max_epochs={cfg.max_epochs}")
                break
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
                if self.step % cfg.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self._log_step(
                        {
                            'total': losses['total'],
                            'main': losses['main'],
                            'lune': losses['lune'],
                            'sol': losses['sol'],
                            'lune_weight': losses['lune_w'],
                            'sol_weight': losses['sol_w'],
                        },
                        lr=lr,
                        grad_norm=grad_norm.item(),
                    )

                # Step-based checkpointing
                if cfg.save_every_steps > 0 and self.step % cfg.save_every_steps == 0:
                    path = self._save_checkpoint()

                    # HF upload at step intervals
                    if cfg.upload_every_steps > 0 and self.step % cfg.upload_every_steps == 0:
                        self.upload_checkpoint(path)

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
        self.epoch_losses.append(avg_loss)
        print(f"\nEpoch {self.epoch}: loss={avg_loss:.4f}, main={ep_main/n:.4f}, "
              f"lune={ep_lune/n:.4f}, sol={ep_sol/n:.4f}")

        # Log epoch metrics
        self._log_epoch(avg_loss)

        # Best checkpoint
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self._save_checkpoint(best=True)

        # Epoch-based checkpointing
        if cfg.save_every_epochs > 0 and self.epoch % cfg.save_every_epochs == 0:
            path = self._save_checkpoint(epoch_checkpoint=True)

            # HF upload at epoch intervals
            if cfg.upload_every_epochs > 0 and self.epoch % cfg.upload_every_epochs == 0:
                self.upload_checkpoint(path)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        cfg = self.config

        # Unpack batch and convert to training dtype
        latents = batch['latents'].to(self.device, dtype=cfg.dtype, non_blocking=True)
        t5 = batch['t5_embeds'].to(self.device, dtype=cfg.dtype, non_blocking=True)
        clip = batch['clip_pooled'].to(self.device, dtype=cfg.dtype, non_blocking=True)
        local_indices = batch.get('local_indices')
        dataset_ids = batch.get('dataset_ids')
        masks = batch.get('masks')

        if masks is not None:
            masks = masks.to(self.device, dtype=cfg.dtype, non_blocking=True)

        B, C, H, W = latents.shape

        # Reshape: [B, C, H, W] -> [B, H*W, C]
        data = latents.permute(0, 2, 3, 1).reshape(B, H * W, C)
        noise = torch.randn_like(data)

        # Text dropout
        if cfg.text_dropout > 0:
            t5, clip = apply_text_dropout(t5, clip, cfg.text_dropout)

        # Sample timesteps
        t = sample_timesteps(
            B,
            device=self.device,
            shift=cfg.shift,
            logit_normal=cfg.logit_normal_sampling,
            logit_mean=cfg.logit_mean,
            logit_std=cfg.logit_std,
            dtype=cfg.dtype,
        )

        # Rectified flow matching
        x_t = flow_x_t(data, noise, t)  # x_t = (1-t)*noise + t*data
        v_target = flow_velocity(data, noise)  # v = data - noise

        # Position IDs
        from tinyflux.model.model import TinyFluxDeep
        img_ids = TinyFluxDeep.create_img_ids(B, H, W, self.device)

        # Get expert features from cache
        lune_features = None
        sol_stats = None
        sol_spatial = None

        if self.cache is not None and local_indices is not None and dataset_ids is not None:
            if cfg.enable_lune:
                lune_features = self.cache.get_lune(local_indices, dataset_ids, t)
                if lune_features is not None:
                    lune_features = lune_features.to(dtype=cfg.dtype)
                    # Teacher dropout - forces model to use predictor
                    if random.random() < cfg.lune_dropout:
                        lune_features = None

            if cfg.enable_sol:
                sol_stats, sol_spatial = self.cache.get_sol(local_indices, dataset_ids, t)
                if sol_stats is not None:
                    sol_stats = sol_stats.to(dtype=cfg.dtype)
                if sol_spatial is not None:
                    sol_spatial = sol_spatial.to(dtype=cfg.dtype)
                # Teacher dropout - forces model to use predictor
                if sol_stats is not None and random.random() < cfg.sol_dropout:
                    sol_stats = None
                    sol_spatial = None

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

        # Extract nested expert info
        lune_info = expert_info.get('lune') or {}
        sol_info = expert_info.get('sol') or {}

        # Lune distillation
        lune_loss = torch.tensor(0.0, device=self.device)
        if lune_features is not None and 'expert_pred' in lune_info:
            lune_loss = compute_lune_loss(
                lune_info['expert_pred'],
                lune_features,
                mode=cfg.lune_mode,
            )

        # Sol distillation
        sol_loss = torch.tensor(0.0, device=self.device)
        if sol_stats is not None and 'pred_stats' in sol_info:
            sol_loss = compute_sol_loss(
                sol_info['pred_stats'],
                sol_info.get('pred_spatial'),
                sol_stats,
                sol_spatial,
            )

        # Total with warmup weights
        lune_w = get_lune_weight(self.step, cfg.lune_warmup_steps, cfg.lune_weight)
        sol_w = get_sol_weight(self.step, cfg.sol_warmup_steps, cfg.sol_weight)

        # Debug: print on first step
        if not hasattr(self, '_sol_debug'):
            print(f"\n  [DEBUG] step={self.step}")
            print(f"  [DEBUG] sol_w={sol_w:.6f}, lune_w={lune_w:.6f}")
            print(f"  [DEBUG] sol_loss={sol_loss.item():.4f} (before weighting)")
            if self.model.sol_prior is not None:
                sol_params = list(self.model.sol_prior.parameters())
                sol_req = sum(p.requires_grad for p in sol_params)
                sol_total = len(sol_params)
                print(f"  [DEBUG] sol_prior: {sol_req}/{sol_total} params require_grad")
                # Check optimizer
                opt_params = sum(len(g['params']) for g in self.optimizer.param_groups)
                print(f"  [DEBUG] optimizer has {opt_params} params")
            self._sol_debug = True

        total_loss = main_loss + lune_w * lune_loss + sol_w * sol_loss

        # Backward (scaled for gradient accumulation)
        (total_loss / cfg.gradient_accumulation).backward()

        # Debug: check gradient flow on first backward
        if not hasattr(self, '_grad_debug'):
            if self.model.sol_prior is not None:
                sol_grads = []
                for name, p in self.model.sol_prior.named_parameters():
                    if p.grad is not None:
                        sol_grads.append((name, p.grad.abs().mean().item()))
                if sol_grads:
                    print(f"  [DEBUG] sol_prior gradients: {len(sol_grads)} params have grads")
                    for name, g in sol_grads[:3]:
                        print(f"    {name}: grad_mean={g:.6f}")
                else:
                    print(f"  [DEBUG] sol_prior: NO GRADIENTS!")
            self._grad_debug = True

        return {
            'total': total_loss.item(),
            'main': main_loss.item(),
            'lune': lune_loss.item(),
            'sol': sol_loss.item(),
            'lune_w': lune_w,
            'sol_w': sol_w,
        }

    def _save_checkpoint(
        self,
        best: bool = False,
        final: bool = False,
        epoch_checkpoint: bool = False,
    ):
        """
        Save checkpoint.

        Args:
            best: Save as best.pt
            final: Save as final.pt
            epoch_checkpoint: Save as epoch_N.pt (vs step_N.pt)
        """
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
            'trainer_config': cfg.to_dict(),
            'model_config': self.model_config,
        }

        # Determine name
        if final:
            name = 'final'
        elif best:
            name = 'best'
        elif epoch_checkpoint:
            name = f'epoch_{self.epoch}'
        else:
            name = f'step_{self.step}'

        path = os.path.join(cfg.checkpoint_dir, f'{name}.pt')
        torch.save(checkpoint, path)

        # Save EMA weights as safetensors
        ema_path = os.path.join(cfg.checkpoint_dir, f'{name}_ema.safetensors')
        self.ema.save(ema_path)

        print(f"  ✓ Saved: {path}")

        # Cleanup old checkpoints
        if epoch_checkpoint:
            self._cleanup_checkpoints(epoch_based=True)
        elif not best and not final:
            self._cleanup_checkpoints(epoch_based=False)

        return path

    def _cleanup_checkpoints(self, epoch_based: bool = False):
        """
        Keep only last N checkpoints.

        Args:
            epoch_based: Clean epoch_* checkpoints (vs step_*)
        """
        cfg = self.config

        if epoch_based:
            prefix = 'epoch_'
            keep_n = cfg.keep_last_n_epochs
        else:
            prefix = 'step_'
            keep_n = cfg.keep_last_n_steps

        # Find matching checkpoints
        checkpoints = sorted([
            f for f in os.listdir(cfg.checkpoint_dir)
            if f.startswith(prefix) and f.endswith('.pt')
        ], key=lambda x: int(x.split('_')[1].split('.')[0]))

        # Remove excess
        while len(checkpoints) > keep_n:
            old = checkpoints.pop(0)
            old_path = os.path.join(cfg.checkpoint_dir, old)
            os.remove(old_path)

            # Also EMA
            ema_path = old_path.replace('.pt', '_ema.safetensors')
            if os.path.exists(ema_path):
                os.remove(ema_path)

    def _upload_to_hf(self, files: Optional[List[str]] = None, commit_message: Optional[str] = None):
        """
        Upload files to HuggingFace.

        Args:
            files: List of file paths to upload (or upload whole checkpoint_dir)
            commit_message: Commit message
        """
        cfg = self.config

        if not cfg.hf_repo_id:
            return

        try:
            from huggingface_hub import HfApi
            api = HfApi()

            if files:
                for file_path in files:
                    if os.path.exists(file_path):
                        api.upload_file(
                            path_or_fileobj=file_path,
                            path_in_repo=os.path.basename(file_path),
                            repo_id=cfg.hf_repo_id,
                            commit_message=commit_message or f"Upload {os.path.basename(file_path)}",
                        )
            else:
                # Upload entire checkpoint directory
                api.upload_folder(
                    folder_path=cfg.checkpoint_dir,
                    repo_id=cfg.hf_repo_id,
                    commit_message=commit_message or f"Training step {self.step}",
                )

            print(f"  ✓ Uploaded to {cfg.hf_repo_id}")
        except Exception as e:
            print(f"  ✗ HF upload failed: {e}")

    def upload_checkpoint(self, checkpoint_path: str):
        """Upload a specific checkpoint to HuggingFace."""
        cfg = self.config

        if not cfg.hf_repo_id:
            print("  ✗ No hf_repo_id configured")
            return

        files = [checkpoint_path]

        # Also upload EMA
        ema_path = checkpoint_path.replace('.pt', '_ema.safetensors')
        if os.path.exists(ema_path):
            files.append(ema_path)

        # Also upload configs
        config_files = ['config.json', 'training_config.json']
        for cf in config_files:
            cf_path = os.path.join(cfg.checkpoint_dir, cf)
            if os.path.exists(cf_path):
                files.append(cf_path)

        self._upload_to_hf(files, f"Checkpoint step {self.step}")

    def upload_samples(self):
        """Upload samples directory to HuggingFace."""
        cfg = self.config

        if not cfg.hf_repo_id or not os.path.exists(cfg.sample_dir):
            return

        try:
            from huggingface_hub import HfApi
            api = HfApi()

            api.upload_folder(
                folder_path=cfg.sample_dir,
                path_in_repo="samples",
                repo_id=cfg.hf_repo_id,
                commit_message=f"Samples at step {self.step}",
            )
            print(f"  ✓ Uploaded samples to {cfg.hf_repo_id}")
        except Exception as e:
            print(f"  ✗ Sample upload failed: {e}")

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