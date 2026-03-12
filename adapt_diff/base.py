"""Abstract base class for diffusion model adapters."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch


class GeneratorAdapter(ABC):
    """
    Abstract interface for diffusion generator models.

    Implementations provide model-specific logic for:
    - Forward pass (denoising)
    - Hook registration on internal layers
    - Checkpoint loading

    This allows the visualizer to work with any diffusion architecture
    (EDM, DDPM, Stable Diffusion, etc.) through a common interface.
    """

    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        Model identifier string.

        Examples: 'imagenet-64', 'sdxl', 'sd-v1.5'
        """
        pass

    @property
    @abstractmethod
    def resolution(self) -> int:
        """Output image resolution (assumes square)."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of classes (0 for unconditional/text-conditioned models)."""
        pass

    @property
    @abstractmethod
    def hookable_layers(self) -> List[str]:
        """
        List of layer names available for hook registration.

        These names are adapter-specific but should be consistent
        (e.g., 'encoder_bottleneck', 'midblock', 'decoder_block_0').
        """
        pass

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for denoising.

        Args:
            x: Noisy input tensor (B, C, H, W)
            sigma: Noise levels (B,) or scalar
            class_labels: One-hot class labels (B, num_classes) or None
            **kwargs: Model-specific options (e.g., text embeddings)

        Returns:
            Denoised output tensor (B, C, H, W)
        """
        pass

    @abstractmethod
    def register_activation_hooks(
        self,
        layer_names: List[str],
        hook_fn: callable
    ) -> List[torch.utils.hooks.RemovableHandle]:
        """
        Register forward hooks on specified layers.

        The hook_fn should have signature:
            hook_fn(module, input, output) -> None or modified_output

        Args:
            layer_names: Layer names from hookable_layers
            hook_fn: Hook function to register

        Returns:
            List of hook handles (call handle.remove() to unregister)
        """
        pass

    @abstractmethod
    def get_layer_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """
        Return activation shapes for hookable layers.

        Returns:
            Dict mapping layer_name -> (C, H, W) shape tuple
        """
        pass

    @classmethod
    @abstractmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = 'cuda',
        **kwargs
    ) -> 'GeneratorAdapter':
        """
        Load adapter from checkpoint file.

        Args:
            checkpoint_path: Path to model weights (.pth, .safetensors, or dir)
            device: Target device ('cuda', 'mps', 'cpu')
            **kwargs: Model-specific options (e.g., label_dropout, cfg_scale)

        Returns:
            Initialized adapter instance ready for inference
        """
        pass

    @classmethod
    @abstractmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Return default configuration dict for this model type.

        Useful for documentation and adapter instantiation.
        """
        pass

    def to(self, device: str) -> 'GeneratorAdapter':
        """Move model to device. Override if model attribute differs."""
        if hasattr(self, '_model'):
            self._model = self._model.to(device)
        return self

    def eval(self) -> 'GeneratorAdapter':
        """Set model to eval mode. Override if model attribute differs."""
        if hasattr(self, '_model'):
            self._model.eval()
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_type='{self.model_type}', "
            f"resolution={self.resolution}, "
            f"num_classes={self.num_classes}, "
            f"layers={len(self.hookable_layers)})"
        )
