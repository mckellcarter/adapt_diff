"""adapt_diff - Model-agnostic adapter interface for diffusion models."""

from .base import GeneratorAdapter
from .hooks import HookMixin
from .registry import (
    discover_adapters,
    get_adapter,
    list_adapters,
    register_adapter,
    register_adapter_class,
    unregister_adapter,
)
from .generation import generate, GenerationResult, ActivationExtractor
from .extraction import (
    flatten_activations,
    load_activations,
    save_activations,
    convert_to_fast_format,
    load_fast_activations,
    get_fast_format_info,
)
from .device import get_device, get_device_info, clear_cache

# Import adapters to register them
from .adapters import dmd2_imagenet, edm_imagenet  # noqa: F401

__all__ = [
    # Base
    "GeneratorAdapter",
    "HookMixin",
    # Registry
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "register_adapter_class",
    "unregister_adapter",
    "discover_adapters",
    # Generation
    "generate",
    "GenerationResult",
    # Extraction
    "ActivationExtractor",
    "flatten_activations",
    "load_activations",
    "save_activations",
    "convert_to_fast_format",
    "load_fast_activations",
    "get_fast_format_info",
    # Device
    "get_device",
    "get_device_info",
    "clear_cache",
]

__version__ = "0.3.0"
