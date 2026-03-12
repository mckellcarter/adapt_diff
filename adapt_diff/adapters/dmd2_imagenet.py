"""DMD2 ImageNet 64x64 adapter implementing the GeneratorAdapter interface."""

from typing import Any, Dict, List, Optional, Tuple

import torch

from adapt_diff.base import GeneratorAdapter
from adapt_diff.hooks import HookMixin
from adapt_diff.registry import register_adapter


@register_adapter('dmd2-imagenet-64')
class DMD2ImageNetAdapter(HookMixin, GeneratorAdapter):
    """
    Adapter for DMD2 ImageNet 64x64 model (EDMPrecond + DhariwalUNet).

    Layer naming:
        - encoder_block_N: N-th encoder block (0-indexed)
        - encoder_bottleneck: Last encoder block (bottleneck)
        - midblock: First decoder block (processes bottleneck)
        - decoder_block_N: N-th decoder block (0-indexed)
    """

    def __init__(self, model, device: str = 'cuda'):
        HookMixin.__init__(self)
        self._model = model
        self._device = device
        self._layer_shapes: Optional[Dict[str, Tuple[int, ...]]] = None

    @property
    def model_type(self) -> str:
        return 'dmd2-imagenet-64'

    @property
    def resolution(self) -> int:
        return 64

    @property
    def num_classes(self) -> int:
        return 1000

    @property
    def hookable_layers(self) -> List[str]:
        """Return list of available layer names for hooks."""
        unet = self._model.model
        layers = []

        # Encoder blocks
        enc_keys = list(unet.enc.keys())
        for i in range(len(enc_keys) - 1):
            layers.append(f'encoder_block_{i}')
        layers.append('encoder_bottleneck')

        # Decoder blocks
        layers.append('midblock')
        dec_keys = list(unet.dec.keys())
        for i in range(1, len(dec_keys)):
            layers.append(f'decoder_block_{i}')

        return layers

    def _get_layer_module(self, layer_name: str):
        """Get the actual PyTorch module for a layer name."""
        unet = self._model.model
        enc_keys = list(unet.enc.keys())
        dec_keys = list(unet.dec.keys())

        if layer_name == 'encoder_bottleneck':
            return unet.enc[enc_keys[-1]]
        elif layer_name == 'midblock':
            return unet.dec[dec_keys[0]]
        elif layer_name.startswith('encoder_block_'):
            idx = int(layer_name.split('_')[-1])
            if idx < len(enc_keys) - 1:
                return unet.enc[enc_keys[idx]]
        elif layer_name.startswith('decoder_block_'):
            idx = int(layer_name.split('_')[-1])
            if idx < len(dec_keys):
                return unet.dec[dec_keys[idx]]

        raise ValueError(f"Unknown layer: {layer_name}")

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
            x: Noisy input (B, C, H, W)
            sigma: Noise levels (B,) or scalar
            class_labels: One-hot class labels (B, 1000) or None
        """
        if class_labels is None:
            class_labels = torch.zeros(x.shape[0], 1000, device=x.device)
        return self._model(x, sigma, class_labels)

    def register_activation_hooks(
        self,
        layer_names: List[str],
        hook_fn: callable
    ) -> List[torch.utils.hooks.RemovableHandle]:
        """Register forward hooks on specified layers."""
        handles = []
        for name in layer_names:
            module = self._get_layer_module(name)
            handle = module.register_forward_hook(hook_fn)
            handles.append(handle)
            self.add_handle(handle)
        return handles

    def get_layer_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Return activation shapes for hookable layers (runs dummy forward on first call)."""
        if self._layer_shapes is not None:
            return self._layer_shapes

        self._layer_shapes = {}
        layer_names = self.hookable_layers

        def make_shape_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self._layer_shapes[name] = tuple(output.shape[1:])
            return hook

        temp_handles = []
        for name in layer_names:
            module = self._get_layer_module(name)
            handle = module.register_forward_hook(make_shape_hook(name))
            temp_handles.append(handle)

        with torch.no_grad():
            dummy_x = torch.randn(1, 3, 64, 64, device=self._device)
            dummy_sigma = torch.ones(1, device=self._device) * 80.0
            dummy_labels = torch.zeros(1, 1000, device=self._device)
            dummy_labels[0, 0] = 1.0
            self._model(dummy_x * 80.0, dummy_sigma, dummy_labels)

        for h in temp_handles:
            h.remove()

        return self._layer_shapes

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = 'cuda',
        label_dropout: float = 0.0,
        **kwargs
    ) -> 'DMD2ImageNetAdapter':
        """Load adapter from pickle checkpoint file.

        Expects pickle format with 'ema' key containing the full model,
        same as EDM checkpoints. Use scripts/convert_checkpoint.py to
        convert from safetensors format.
        """
        import pickle

        # Ensure vendored NVIDIA modules are importable for pickle
        from adapt_diff.vendor.nvidia_compat import ensure_nvidia_modules
        ensure_nvidia_modules()

        print(f"Loading DMD2 from {checkpoint_path}...")
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)

        # EMA weights are preferred
        model = data['ema'].to(device)
        model.eval()

        print(f"Loaded DMD2: {model.img_resolution}x{model.img_resolution}")
        return cls(model, device)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Return default configuration for ImageNet 64x64."""
        return {
            "img_resolution": 64,
            "img_channels": 3,
            "label_dim": 1000,
            "use_fp16": False,
            "sigma_min": 0,
            "sigma_max": float("inf"),
            "sigma_data": 0.5,
            "model_type": "DhariwalUNet"
        }

    def to(self, device: str) -> 'DMD2ImageNetAdapter':
        self._model = self._model.to(device)
        self._device = device
        return self

    def eval(self) -> 'DMD2ImageNetAdapter':
        self._model.eval()
        return self
