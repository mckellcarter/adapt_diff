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

    def get_timesteps(
        self,
        num_steps: int,
        device: str = 'cuda',
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        **kwargs  # Accept rho etc for API consistency, ignored
    ) -> torch.Tensor:
        """
        Return logarithmic sigma schedule for DMD2.

        DMD2 is a distilled model designed for 1-10 steps with log-spaced sigmas.

        Args:
            num_steps: Number of denoising steps (typically 1-10)
            device: Target device
            sigma_max: Maximum noise level (default: 80.0)
            sigma_min: Minimum noise level (default: 0.002)
            **kwargs: Ignored (for API consistency with other adapters)

        Returns:
            Sigma tensor (num_steps + 1,) from sigma_max to 0
            Example for 6 steps: [80, 33, 10, 3, 1, 0.5, 0]
        """
        # Log-linear interpolation in log space, then append 0
        sigmas = torch.logspace(
            torch.log10(torch.tensor(sigma_max)),
            torch.log10(torch.tensor(sigma_min)),
            num_steps,
            device=device
        )
        # Append final sigma = 0
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
        return_x0: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Euler step for DMD2 sampling.

        Args:
            x_t: Current noisy sample (B, C, H, W)
            t: Current sigma value (scalar or (B,))
            model_output: Denoised output from forward() (B, C, H, W) - this IS x0
            t_next: Next sigma value (required)
            return_x0: If True, return (x_{t-1}, x0) tuple for visualization
            **kwargs: Unused

        Returns:
            x_{t-1}: Next (less noisy) sample, or (x_{t-1}, x0) if return_x0=True
        """
        if t_next is None:
            raise ValueError("DMD2 step() requires t_next parameter")

        # Euler step: x_next = x + (t_next - t) * dx/dt
        d_cur = (x_t - model_output) / t
        x_next = x_t + (t_next - t) * d_cur
        if return_x0:
            return x_next, model_output  # model_output is predicted x0
        return x_next

    def get_initial_noise(
        self,
        batch_size: int,
        device: str = 'cuda',
        generator: Optional[torch.Generator] = None,
        sigma_max: float = 80.0
    ) -> torch.Tensor:
        """
        Generate initial noise for DMD2 sampling.

        Args:
            batch_size: Number of samples
            device: Target device
            generator: Optional RNG for reproducibility
            sigma_max: Maximum noise level (default: 80.0)

        Returns:
            Noise tensor (B, 3, 64, 64) scaled by sigma_max
        """
        noise = torch.randn(
            batch_size, 3, self.resolution, self.resolution,
            device=device,
            generator=generator
        )
        return noise * sigma_max

    def prepare_conditioning(
        self,
        class_label: Optional[int] = None,
        batch_size: int = 1,
        device: str = 'cuda',
        **kwargs
    ) -> torch.Tensor:
        """
        Prepare class conditioning for DMD2.

        Args:
            class_label: Single class ID (0-999) or None for random
            batch_size: Number of samples
            device: Target device
            **kwargs: Unused

        Returns:
            One-hot class labels (B, 1000)
        """
        if class_label is not None:
            labels = torch.full((batch_size,), class_label, device=device, dtype=torch.long)
        else:
            labels = torch.randint(0, self.num_classes, (batch_size,), device=device)

        return torch.eye(self.num_classes, device=device)[labels]

    @property
    def prediction_type(self) -> str:
        """DMD2 predicts denoised sample x0."""
        return 'sample'

    @property
    def uses_latent(self) -> bool:
        """DMD2 operates in pixel space."""
        return False

    @property
    def in_channels(self) -> int:
        """DMD2 uses 3-channel RGB input."""
        return 3

    @property
    def conditioning_type(self) -> str:
        """DMD2 uses class conditioning."""
        return 'class'

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        uncond: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward with classifier-free guidance for class conditioning.

        Args:
            x: Noisy input (B, C, H, W)
            t: Sigma value (scalar or (B,))
            cond: Conditional one-hot class labels (B, 1000)
            uncond: Unconditional class labels (B, 1000), defaults to zeros
            guidance_scale: CFG scale (1.0 = no guidance)
            **kwargs: Passed to forward()

        Returns:
            Guided denoised output (B, C, H, W)
        """
        if guidance_scale == 1.0:
            return self.forward(x, t, class_labels=cond, **kwargs)

        if uncond is None:
            uncond = torch.zeros_like(cond)

        # Unconditional forward (null class)
        uncond_out = self.forward(x, t, class_labels=uncond, **kwargs)

        # Conditional forward
        cond_out = self.forward(x, t, class_labels=cond, **kwargs)

        # CFG: uncond + scale * (cond - uncond)
        return uncond_out + guidance_scale * (cond_out - uncond_out)

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
            "sigma_data": 0.5,
            "model_type": "DhariwalUNet",
            # Sampling defaults (DMD2 is distilled for few-step generation)
            "sigma_max": 80.0,
            "sigma_min": 0.5,
            "default_steps": 5,
        }

    def to(self, device: str) -> 'DMD2ImageNetAdapter':
        self._model = self._model.to(device)
        self._device = device
        return self

    def eval(self) -> 'DMD2ImageNetAdapter':
        self._model.eval()
        return self
