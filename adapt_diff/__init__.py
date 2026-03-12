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

# Import adapters to register them
from .adapters import dmd2_imagenet, edm_imagenet  # noqa: F401

__all__ = [
    "GeneratorAdapter",
    "HookMixin",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "register_adapter_class",
    "unregister_adapter",
    "discover_adapters",
]

__version__ = "0.1.0"
