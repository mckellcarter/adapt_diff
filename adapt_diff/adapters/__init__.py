"""Model-specific adapter implementations."""

from .dmd2_imagenet import DMD2ImageNetAdapter
from .edm_imagenet import EDMImageNetAdapter

__all__ = [
    "DMD2ImageNetAdapter",
    "EDMImageNetAdapter",
]
