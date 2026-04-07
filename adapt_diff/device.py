"""
Device detection and management utilities.
Supports CUDA, MPS (Apple Silicon), and CPU.
"""

import gc

import torch


def get_device(prefer_device: str = None) -> str:
    """
    Get best available device.

    Args:
        prefer_device: Optional preferred device ('cuda', 'mps', 'cpu')

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if prefer_device:
        if prefer_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif prefer_device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif prefer_device == "cpu":
            return "cpu"

    # Auto-detect
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_info(device: str) -> dict:
    """Get information about the device."""
    info = {"device": device}

    if device == "cuda":
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_allocated_gb"] = torch.cuda.memory_allocated(0) / 1024**3
        info["memory_reserved_gb"] = torch.cuda.memory_reserved(0) / 1024**3
    elif device == "mps":
        info["device_name"] = "Apple Silicon MPS"
    else:
        info["device_name"] = "CPU"

    return info


def clear_cache(device: str = None):
    """
    Clear device memory cache and run garbage collection.

    Args:
        device: Device string ('cuda', 'mps', 'cpu', or None for all)
    """
    gc.collect()

    if device is None or "mps" in str(device):
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    if device is None or "cuda" in str(device):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
