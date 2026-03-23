"""Model-specific adapter implementations."""

from .dmd2_imagenet import DMD2ImageNetAdapter
from .edm_imagenet import EDMImageNetAdapter
from .mscoco_t2i import MSCOCOT2IAdapter

__all__ = [
    "DMD2ImageNetAdapter",
    "EDMImageNetAdapter",
    "MSCOCOT2IAdapter",
]
