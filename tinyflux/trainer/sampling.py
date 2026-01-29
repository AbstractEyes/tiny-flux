"""
TinyFlux Sampling Utilities

- generate_samples: CFG sampling with flux shift
- save_samples: Save sample grids
"""

import os
import torch
import torch.nn as nn
from typing import List, Optional, Callable, Tuple
from datetime import datetime

from .schedules import flux_shift


@torch.inference_mode()
def generate_samples(
        model: nn.Module,
        prompts: List[str],
        encode_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
        vae: nn.Module,
        vae_scale: float,
        num_steps: int = 28,
        guidance_scale: float = 5.0,
        H: int = 64,
        W: int = 64,
        shift: float = 3.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        seed: Optional[int] = None,
        negative_prompt: str = "blurry, distorted, low quality",
        ema=None,
) -> torch.Tensor:
    """
    Generate samples with classifier-free guidance.

    Args:
        model: TinyFluxDeep model
        prompts: List of text prompts
        encode_fn: Function (prompt) -> (t5_embed, clip_pooled)
        vae: VAE for decoding
        vae_scale: VAE scaling factor
        num_steps: Number of sampling steps
        guidance_scale: CFG scale (1.0 = no guidance)
        H, W: Latent height/width (64 for 512px)
        shift: Flux shift parameter
        device: Device
        dtype: Compute dtype
        seed: Optional seed for reproducibility
        negative_prompt: Negative prompt for CFG
        ema: Optional EMA to use for sampling

    Returns:
        images: [B, 3, H*8, W*8] generated images in [0, 1]
    """
    was_training = model.training
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    # Handle compiled models
    model_ref = model._orig_mod if hasattr(model, '_orig_mod') else model

    # Apply EMA if provided
    if ema is not None:
        ema.apply_shadow(model)

    B = len(prompts)
    C = 16  # Latent channels

    # Encode prompts
    t5_list, clip_list = [], []
    for p in prompts:
        t5, clip = encode_fn(p)
        t5_list.append(t5)
        clip_list.append(clip)
    t5_cond = torch.stack(t5_list).to(device, dtype=dtype)
    clip_cond = torch.stack(clip_list).to(device, dtype=dtype)

    # Encode negative prompt for CFG
    if guidance_scale > 1.0:
        t5_uncond, clip_uncond = encode_fn(negative_prompt)
        t5_uncond = t5_uncond.unsqueeze(0).expand(B, -1, -1).to(device, dtype=dtype)
        clip_uncond = clip_uncond.unsqueeze(0).expand(B, -1).to(device, dtype=dtype)
    else:
        t5_uncond = clip_uncond = None

    # Initialize noise
    x = torch.randn(B, H * W, C, device=device, dtype=dtype)
    img_ids = model_ref.create_img_ids(B, H, W, device)

    # Timestep schedule with flux shift
    t_linear = torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)
    timesteps = flux_shift(t_linear, shift=shift)

    # Sampling loop
    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_curr

        t_batch = t_curr.expand(B).to(dtype)

        with torch.autocast(device, dtype=dtype):
            # Conditional prediction
            v_cond = model_ref(
                hidden_states=x,
                encoder_hidden_states=t5_cond,
                pooled_projections=clip_cond,
                timestep=t_batch,
                img_ids=img_ids,
            )
            if isinstance(v_cond, tuple):
                v_cond = v_cond[0]

            # CFG
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

        # Euler step
        x = x + v * dt

    # Decode latents
    latents = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
    latents = latents / vae_scale

    with torch.autocast(device, dtype=dtype):
        images = vae.decode(latents.to(vae.dtype)).sample

    # Normalize to [0, 1]
    images = (images / 2 + 0.5).clamp(0, 1)

    # Restore EMA
    if ema is not None:
        ema.restore(model)

    if was_training:
        model.train()

    return images


def save_samples(
        images: torch.Tensor,
        prompts: List[str],
        step: int,
        output_dir: str,
        nrow: int = 2,
) -> str:
    """
    Save sample images as grid.

    Args:
        images: [B, 3, H, W] images in [0, 1]
        prompts: List of prompts (for logging)
        step: Training step
        output_dir: Output directory
        nrow: Images per row in grid

    Returns:
        path: Path to saved grid
    """
    from torchvision.utils import save_image

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"samples_step_{step}.png")

    save_image(images, path, nrow=nrow, padding=2)

    # Also save prompts
    prompts_path = os.path.join(output_dir, f"prompts_step_{step}.txt")
    with open(prompts_path, 'w') as f:
        for i, p in enumerate(prompts):
            f.write(f"{i}: {p}\n")

    return path


class Sampler:
    """
    Convenient sampler wrapper for training.

    Usage:
        sampler = Sampler(zoo, model, ema)
        images = sampler.generate(prompts)
        sampler.save(images, prompts, step, output_dir)
    """

    def __init__(
            self,
            zoo,  # ModelZoo with vae, clip, t5 loaded
            model: nn.Module,
            ema=None,
            num_steps: int = 28,
            guidance_scale: float = 5.0,
            shift: float = 3.0,
            device: str = "cuda",
            dtype: torch.dtype = torch.bfloat16,
    ):
        self.zoo = zoo
        self.model = model
        self.ema = ema
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.shift = shift
        self.device = device
        self.dtype = dtype

    def _encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt using zoo's encoders."""
        # T5
        t5_inputs = self.zoo.t5_tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        ).to(self.device)
        t5_out = self.zoo.t5(**t5_inputs).last_hidden_state.squeeze(0)

        # CLIP
        clip_inputs = self.zoo.clip_tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        ).to(self.device)
        clip_out = self.zoo.clip(**clip_inputs).pooler_output.squeeze(0)

        return t5_out, clip_out

    def generate(
            self,
            prompts: List[str],
            seed: Optional[int] = None,
            negative_prompt: str = "blurry, distorted, low quality",
    ) -> torch.Tensor:
        """Generate images from prompts."""
        return generate_samples(
            model=self.model,
            prompts=prompts,
            encode_fn=self._encode_prompt,
            vae=self.zoo.vae,
            vae_scale=self.zoo.vae.config.scaling_factor,
            num_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            shift=self.shift,
            device=self.device,
            dtype=self.dtype,
            seed=seed,
            negative_prompt=negative_prompt,
            ema=self.ema,
        )

    def save(
            self,
            images: torch.Tensor,
            prompts: List[str],
            step: int,
            output_dir: str,
    ) -> str:
        """Save sample grid."""
        return save_samples(images, prompts, step, output_dir)