"""Hook management utilities for adapters."""

from typing import Callable, Dict, List, Optional

import torch


class HookMixin:
    """
    Mixin providing reusable forward hook management for adapters.

    Provides common patterns for:
    - Extracting activations during forward pass
    - Masking activations with fixed values
    - Managing hook lifecycle

    Usage:
        class MyAdapter(HookMixin, GeneratorAdapter):
            def __init__(self, model):
                HookMixin.__init__(self)
                self._model = model
    """

    def __init__(self):
        self._activations: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._masks: Dict[str, torch.Tensor] = {}

    def make_extraction_hook(self, layer_name: str) -> Callable:
        """
        Create forward hook that extracts layer output.

        Args:
            layer_name: Key to store activation under

        Returns:
            Hook function for register_forward_hook
        """
        def forward_hook(module, input, output):
            # Handle tuple outputs (some layers return (output, extras))
            if isinstance(output, tuple):
                output = output[0]
            self._activations[layer_name] = output.detach().cpu()
        return forward_hook

    def make_mask_hook(self, layer_name: str, mask: torch.Tensor) -> Callable:
        """
        Create forward hook that replaces layer output with mask value.

        Args:
            layer_name: For logging/debugging
            mask: Tensor to replace output with (broadcasts batch dim)

        Returns:
            Hook function for register_forward_hook
        """
        def forward_hook(module, input, output):
            target = mask.to(output.device, output.dtype)
            # Broadcast if mask has batch=1 but output has larger batch
            if target.shape[0] == 1 and output.shape[0] > 1:
                target = target.expand(output.shape[0], -1, -1, -1)
            return target
        return forward_hook

    def set_mask(self, layer_name: str, value: torch.Tensor):
        """Store activation mask for a layer."""
        self._masks[layer_name] = value

    def get_mask(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get stored mask for a layer."""
        return self._masks.get(layer_name)

    def clear_mask(self, layer_name: str):
        """Remove mask for a layer."""
        self._masks.pop(layer_name, None)

    def clear_masks(self):
        """Remove all masks."""
        self._masks.clear()

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Return copy of extracted activations."""
        return self._activations.copy()

    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get single activation by layer name."""
        return self._activations.get(layer_name)

    def clear_activations(self):
        """Clear all extracted activations."""
        self._activations.clear()

    def remove_hooks(self):
        """Remove all registered forward hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def add_handle(self, handle: torch.utils.hooks.RemovableHandle):
        """Track a hook handle for later removal."""
        self._handles.append(handle)

    @property
    def num_hooks(self) -> int:
        """Number of active forward hooks."""
        return len(self._handles)

    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()
