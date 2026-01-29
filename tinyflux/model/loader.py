"""
TinyFlux Model Loader

Unified loading from:
- HuggingFace repo: "AbstractPhil/tinyflux-deep"
- Local directory: "/path/to/checkpoint/"
- Local file: "/path/to/model.safetensors" or ".pt"
- Separate model/ema targets

Handles:
- safetensors vs pt format
- EMA weights detection
- Config loading
- Compiled model unwrapping
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple

from .model import TinyFluxConfig, TinyFluxDeep


def _is_hf_repo(path: str) -> bool:
    """Check if path looks like a HuggingFace repo ID."""
    return '/' in path and not os.path.exists(path) and not path.startswith('.')


def _find_weights_in_dir(directory: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Find model and EMA weights in directory.

    Returns:
        (model_path, ema_path) - either can be None
    """
    directory = Path(directory)

    model_path = None
    ema_path = None

    # Priority order for model weights
    model_candidates = [
        'model.safetensors',
        'pytorch_model.safetensors',
        'weights.safetensors',
        'model.pt',
        'pytorch_model.pt',
        'weights.pt',
        'checkpoint.pt',
    ]

    # Priority order for EMA weights
    ema_candidates = [
        'ema.safetensors',
        'model_ema.safetensors',
        'ema_weights.safetensors',
        'ema.pt',
        'model_ema.pt',
    ]

    for candidate in model_candidates:
        path = directory / candidate
        if path.exists():
            model_path = str(path)
            break

    for candidate in ema_candidates:
        path = directory / candidate
        if path.exists():
            ema_path = str(path)
            break

    # Also check for *_ema.safetensors pattern
    if ema_path is None:
        for f in directory.glob('*_ema.safetensors'):
            ema_path = str(f)
            break
        for f in directory.glob('*_ema.pt'):
            ema_path = str(f)
            break

    return model_path, ema_path


def _find_config_in_dir(directory: str) -> Optional[str]:
    """Find config.json in directory."""
    directory = Path(directory)

    candidates = ['config.json', 'model_config.json']
    for candidate in candidates:
        path = directory / candidate
        if path.exists():
            return str(path)

    return None


def _load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    """Load safetensors file."""
    from safetensors.torch import load_file
    return load_file(path)


def _load_pt(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load .pt file, handling checkpoint vs raw state dict."""
    data = torch.load(path, map_location=map_location)

    # If it's a full checkpoint, extract model state
    if isinstance(data, dict):
        if 'model' in data:
            return data['model']
        if 'state_dict' in data:
            return data['state_dict']
        if 'model_state_dict' in data:
            return data['model_state_dict']

    return data


def _load_weights(path: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load weights from safetensors or pt file."""
    if path.endswith('.safetensors'):
        return _load_safetensors(path)
    else:
        return _load_pt(path, map_location)


def _download_from_hf(
    repo_id: str,
    filename: str,
    cache_dir: Optional[str] = None,
) -> str:
    """Download file from HuggingFace."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


def load_config(
    path: str,
    cache_dir: Optional[str] = None,
) -> TinyFluxConfig:
    """
    Load TinyFluxConfig from path or HF repo.

    Args:
        path: HF repo ID, directory, or config.json file
        cache_dir: Cache directory for HF downloads

    Returns:
        TinyFluxConfig
    """
    if _is_hf_repo(path):
        config_path = _download_from_hf(path, "config.json", cache_dir)
    elif os.path.isdir(path):
        config_path = _find_config_in_dir(path)
        if config_path is None:
            raise FileNotFoundError(f"No config.json found in {path}")
    else:
        config_path = path

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    return TinyFluxConfig(**config_dict)


def load_model(
    path: str,
    config: Optional[TinyFluxConfig] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    load_ema: bool = False,
    ema_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    compile_model: bool = False,
    strict: bool = True,
) -> TinyFluxDeep:
    """
    Load TinyFluxDeep model from various sources.

    Args:
        path: One of:
            - HF repo ID: "AbstractPhil/tinyflux-deep"
            - Local directory: "/path/to/checkpoint/"
            - Local file: "/path/to/model.safetensors"
        config: Model config (loaded from path if None)
        device: Target device
        dtype: Model dtype
        load_ema: If True, load EMA weights instead of model weights
        ema_path: Explicit path to EMA weights (overrides auto-detection)
        cache_dir: Cache directory for HF downloads
        compile_model: Whether to torch.compile the model
        strict: Strict state dict loading

    Returns:
        TinyFluxDeep model
    """
    # Determine source type and locate files
    if _is_hf_repo(path):
        # HuggingFace repo
        if config is None:
            config = load_config(path, cache_dir)

        if load_ema:
            filename = ema_path or "ema.safetensors"
        else:
            filename = "model.safetensors"

        weights_path = _download_from_hf(path, filename, cache_dir)

    elif os.path.isdir(path):
        # Local directory
        if config is None:
            config_path = _find_config_in_dir(path)
            if config_path:
                config = load_config(config_path)
            else:
                config = TinyFluxConfig()  # Default

        model_path, auto_ema_path = _find_weights_in_dir(path)

        if load_ema:
            weights_path = ema_path or auto_ema_path
            if weights_path is None:
                raise FileNotFoundError(f"No EMA weights found in {path}")
        else:
            weights_path = model_path
            if weights_path is None:
                raise FileNotFoundError(f"No model weights found in {path}")

    else:
        # Local file
        if config is None:
            # Try to find config in same directory
            parent = os.path.dirname(path)
            config_path = _find_config_in_dir(parent) if parent else None
            if config_path:
                config = load_config(config_path)
            else:
                config = TinyFluxConfig()

        if load_ema and ema_path:
            weights_path = ema_path
        else:
            weights_path = path

    # Create model
    model = TinyFluxDeep(config)

    # Load weights
    state_dict = _load_weights(weights_path)

    # Handle _orig_mod prefix from compiled models
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)

    # Move to device/dtype
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Compile if requested
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")

    return model


def load_checkpoint(
    path: str,
    model: Optional[TinyFluxDeep] = None,
    config: Optional[TinyFluxConfig] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """
    Load full training checkpoint.

    Args:
        path: Path to checkpoint.pt
        model: Existing model to load into (created if None)
        config: Model config (loaded from checkpoint if None)
        device: Target device
        dtype: Model dtype

    Returns:
        Dict with:
            - 'model': TinyFluxDeep
            - 'step': int
            - 'epoch': int
            - 'best_loss': float
            - 'optimizer': optimizer state dict
            - 'scheduler': scheduler state dict
            - 'ema': EMA shadow dict
            - 'trainer_config': TrainerConfig dict
            - 'model_config': TinyFluxConfig dict
    """
    checkpoint = torch.load(path, map_location='cpu')

    # Get or create model
    if model is None:
        if config is None:
            if 'model_config' in checkpoint:
                config = TinyFluxConfig(**checkpoint['model_config'])
            else:
                config = TinyFluxConfig()
        model = TinyFluxDeep(config).to(device=device, dtype=dtype)

    # Load model weights
    model_state = checkpoint.get('model', checkpoint.get('state_dict', {}))
    if any(k.startswith('_orig_mod.') for k in model_state.keys()):
        model_state = {k.replace('_orig_mod.', ''): v for k, v in model_state.items()}
    model.load_state_dict(model_state)

    return {
        'model': model,
        'step': checkpoint.get('step', 0),
        'epoch': checkpoint.get('epoch', 0),
        'best_loss': checkpoint.get('best_loss', float('inf')),
        'optimizer': checkpoint.get('optimizer'),
        'scheduler': checkpoint.get('scheduler'),
        'ema': checkpoint.get('ema'),
        'trainer_config': checkpoint.get('trainer_config'),
        'model_config': checkpoint.get('model_config'),
    }


# =============================================================================
# Convenience functions
# =============================================================================

def from_pretrained(
    repo_id: str = "AbstractPhil/tinyflux-deep",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    use_ema: bool = True,
    compile_model: bool = False,
) -> TinyFluxDeep:
    """
    Load pretrained TinyFlux model from HuggingFace.

    Args:
        repo_id: HuggingFace repo ID
        device: Target device
        dtype: Model dtype
        use_ema: Load EMA weights (recommended for inference)
        compile_model: Whether to torch.compile

    Returns:
        TinyFluxDeep model ready for inference
    """
    return load_model(
        repo_id,
        device=device,
        dtype=dtype,
        load_ema=use_ema,
        compile_model=compile_model,
    )