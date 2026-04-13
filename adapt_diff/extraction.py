"""
Activation extraction utilities for diffusion models.

Shared utilities for:
- Flattening multi-layer activations
- Loading/saving activation files (.npz, .npy)
- Fast format conversion for large datasets

These utilities are designed to be used by downstream packages
like diffviews (visualization) and yodal (attribution).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Re-export ActivationExtractor from generation module
from .generation import ActivationExtractor

__all__ = [
    "ActivationExtractor",
    "flatten_activations",
    "load_activations",
    "save_activations",
    "convert_to_fast_format",
    "load_fast_activations",
]


def flatten_activations(activations: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Flatten all layer activations to single vector per sample.

    Concatenates activations from multiple layers in sorted order.

    Args:
        activations: Dict of layer_name -> array (B, features) or (B, C, H, W)

    Returns:
        Flattened array (B, total_features)

    Example:
        >>> acts = {"layer1": np.zeros((10, 512)), "layer2": np.zeros((10, 256))}
        >>> flat = flatten_activations(acts)
        >>> flat.shape
        (10, 768)
    """
    all_features = []
    for layer_name in sorted(activations.keys()):
        act = activations[layer_name]
        if len(act.shape) > 2:
            # Flatten spatial dims: (B, C, H, W) -> (B, C*H*W)
            act = act.reshape(act.shape[0], -1)
        all_features.append(act)

    return np.concatenate(all_features, axis=1)


def save_activations(
    activations: Dict[str, np.ndarray],
    output_path: Path,
    metadata: Optional[Dict] = None,
    compress: bool = True,
) -> Path:
    """
    Save activations to disk.

    Args:
        activations: Dict of layer_name -> array (B, features)
        output_path: Path to save (without extension)
        metadata: Optional metadata dict (saved as .json)
        compress: Use compressed .npz format (slower but smaller)

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten spatial dims if present
    activation_dict = {}
    for name, activation in activations.items():
        if len(activation.shape) == 4:
            batch_size = activation.shape[0]
            activation_dict[name] = activation.reshape(batch_size, -1)
        else:
            activation_dict[name] = activation

    # Save activations
    npz_path = output_path.with_suffix(".npz")
    if compress:
        np.savez_compressed(str(npz_path), **activation_dict)
    else:
        np.savez(str(npz_path), **activation_dict)

    # Save metadata
    if metadata:
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return npz_path


def load_activations(activation_path: Path) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load activations and metadata from disk.

    Args:
        activation_path: Path to .npz file (or without extension)

    Returns:
        (activations_dict, metadata_dict)
    """
    activation_path = Path(activation_path)

    # Handle path with or without extension
    npz_path = activation_path.with_suffix(".npz")
    if not npz_path.exists() and activation_path.suffix == ".npz":
        npz_path = activation_path

    data = np.load(str(npz_path))
    activations = {key: data[key] for key in data.keys()}

    metadata = {}
    metadata_path = activation_path.with_suffix(".json")
    if not metadata_path.exists():
        # Try alongside .npz
        metadata_path = npz_path.with_suffix(".json")
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return activations, metadata


def convert_to_fast_format(
    npz_path: Path,
    output_path: Optional[Path] = None,
    layers: Optional[List[str]] = None,
    dtype: str = "float32",
) -> Path:
    """
    Convert compressed .npz to fast-loading .npy format.

    Concatenates specified layers into single pre-flattened array
    for ~30x faster loading via memory mapping.

    Args:
        npz_path: Path to .npz file
        output_path: Output .npy path (default: same name with .npy)
        layers: Layer names to include (default: all, sorted)
        dtype: Output dtype (default: float32)

    Returns:
        Path to created .npy file
    """
    npz_path = Path(npz_path)
    if output_path is None:
        output_path = npz_path.with_suffix(".npy")
    else:
        output_path = Path(output_path)

    data = np.load(str(npz_path))
    layer_names = layers or sorted(data.keys())

    # Concatenate layers in sorted order
    arrays = [data[name] for name in layer_names]
    combined = np.concatenate(arrays, axis=1).astype(dtype)

    # Save combined array
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), combined)

    # Save layer info for reconstruction
    info_path = Path(str(output_path) + ".json")
    info = {
        "layers": layer_names,
        "shapes": {name: list(data[name].shape) for name in layer_names},
        "total_features": combined.shape[1],
        "num_samples": combined.shape[0],
        "dtype": dtype,
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return output_path


def load_fast_activations(
    npy_path: Path,
    mmap_mode: Optional[str] = "r",
) -> np.ndarray:
    """
    Load pre-concatenated activations with optional memory mapping.

    ~30x faster than loading .npz for large datasets.

    Args:
        npy_path: Path to .npy file
        mmap_mode: Memory map mode ('r' for read-only, None for full load)

    Returns:
        Activation matrix (N, D)
    """
    return np.load(str(npy_path), mmap_mode=mmap_mode)


def get_fast_format_info(npy_path: Path) -> Dict:
    """
    Load metadata for fast-format .npy file.

    Args:
        npy_path: Path to .npy file

    Returns:
        Info dict with layers, shapes, total_features, num_samples
    """
    info_path = Path(str(npy_path) + ".json")
    if not info_path.exists():
        raise FileNotFoundError(f"Metadata not found: {info_path}")

    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)
