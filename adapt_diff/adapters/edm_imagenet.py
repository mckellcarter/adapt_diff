"""EDM ImageNet 64x64 adapter for visualization.

This adapter wraps the original EDM (Elucidating Diffusion Models) checkpoint
from Karras et al. Unlike DMD2's single-step distilled model, EDM requires
multi-step iterative sampling.

Reference:
- Paper: https://arxiv.org/abs/2206.00364
- Pretrained: https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/
"""

import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from adapt_diff.base import GeneratorAdapter
from adapt_diff.hooks import HookMixin
from adapt_diff.registry import register_adapter


@register_adapter('edm-imagenet-64')
class EDMImageNetAdapter(HookMixin, GeneratorAdapter):
    """
    Adapter for original EDM ImageNet 64x64 model (EDMPrecond + DhariwalUNet).

    Unlike DMD2, EDM requires multi-step iterative sampling. The forward()
    method performs a single denoising step, while sample() runs the full
    EDM sampler.

    Checkpoint formats supported:
        - .pkl: Original EDM pickle format (via dnnlib)
        - .pth/.bin: Converted PyTorch state dict

    Layer naming (same as DMD2):
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
        return 'edm-imagenet-64'

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
        if layer_name == 'midblock':
            return unet.dec[dec_keys[0]]
        if layer_name.startswith('encoder_block_'):
            idx = int(layer_name.split('_')[-1])
            if idx < len(enc_keys) - 1:
                return unet.enc[enc_keys[idx]]
        if layer_name.startswith('decoder_block_'):
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
        Single denoising step.

        For EDM, this performs one D_theta(x; sigma) call. Unlike DMD2,
        you typically need to call this multiple times in a sampling loop.

        Args:
            x: Noisy input (B, C, H, W) - should be scaled by sigma
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
        rho: float = 7.0,
        **kwargs  # For API extensibility
    ) -> torch.Tensor:
        """
        Return Karras sigma schedule for EDM.

        Args:
            num_steps: Number of denoising steps
            device: Target device
            sigma_max: Maximum noise level (default: 80.0)
            sigma_min: Minimum noise level (default: 0.002)
            rho: Schedule curvature parameter (default: 7.0)
            **kwargs: Ignored (for API extensibility)

        Returns:
            Sigma tensor (num_steps + 1,) from sigma_max to 0
        """
        # MPS doesn't support float64, use float32 instead
        high_prec_dtype = torch.float64 if device != 'mps' else torch.float32

        step_indices = torch.arange(num_steps, dtype=high_prec_dtype, device=device)
        t_steps = (
            sigma_max ** (1 / rho) +
            step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho

        # Append t_N = 0
        return torch.cat([
            self._model.round_sigma(t_steps),
            torch.zeros_like(t_steps[:1])
        ])

    def step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Euler step for EDM sampling.

        Args:
            x_t: Current noisy sample (B, C, H, W)
            t: Current sigma value (scalar or (B,))
            model_output: Denoised output from forward() (B, C, H, W)
            t_next: Next sigma value (required for EDM)
            **kwargs: Unused

        Returns:
            x_{t-1}: Next (less noisy) sample
        """
        if t_next is None:
            raise ValueError("EDM step() requires t_next parameter")

        # Euler step: x_next = x + (t_next - t) * dx/dt
        d_cur = (x_t - model_output) / t
        return x_t + (t_next - t) * d_cur

    def get_initial_noise(
        self,
        batch_size: int,
        device: str = 'cuda',
        generator: Optional[torch.Generator] = None,
        sigma_max: float = 80.0
    ) -> torch.Tensor:
        """
        Generate initial noise for EDM sampling.

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
        Prepare class conditioning for EDM.

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
        """EDM predicts denoised sample x0."""
        return 'sample'

    def convert_latent_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor
    ) -> torch.Tensor:
        """EDM predicts x0 directly, no conversion needed."""
        return model_output

    @property
    def uses_latent(self) -> bool:
        """EDM operates in pixel space."""
        return False

    @property
    def in_channels(self) -> int:
        """EDM uses 3-channel RGB input."""
        return 3

    @property
    def conditioning_type(self) -> str:
        """EDM uses class conditioning."""
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

    def sample(
        self,
        num_samples: int,
        class_label: Optional[int] = None,
        num_steps: int = 256,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        rho: float = 7.0,
        S_churn: float = 40.0,
        S_min: float = 0.05,
        S_max: float = 50.0,
        S_noise: float = 1.003,
        device: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full EDM sampling with stochastic sampler (Karras et al. settings).

        Default parameters are the recommended ImageNet-64 settings from
        the EDM paper which achieve FID=1.36 with NFE=511.

        Args:
            num_samples: Number of images to generate
            class_label: Single class ID or None for random classes
            num_steps: Number of denoising steps (default 256 for ImageNet)
            sigma_max: Maximum noise level (start of trajectory)
            sigma_min: Minimum noise level (end of trajectory)
            rho: Schedule curvature parameter
            S_churn: Amount of noise injection per step
            S_min: Minimum sigma for noise injection
            S_max: Maximum sigma for noise injection
            S_noise: Noise multiplier for stochastic steps
            device: Target device (uses adapter device if None)

        Returns:
            Tuple of (images, labels) where:
                - images: (B, 3, 64, 64) normalized to [-1, 1]
                - labels: (B,) class label indices
        """
        device = device or self._device

        # MPS doesn't support float64, use float32 instead
        high_prec_dtype = torch.float64 if device != 'mps' else torch.float32

        # Class labels
        if class_label is not None:
            labels = torch.full((num_samples,), class_label, device=device)
        else:
            labels = torch.randint(0, self.num_classes, (num_samples,), device=device)
        one_hot = torch.eye(self.num_classes, device=device)[labels]

        # Karras sigma schedule
        step_indices = torch.arange(num_steps, dtype=high_prec_dtype, device=device)
        t_steps = (
            sigma_max ** (1 / rho) +
            step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho

        # Append t_N = 0
        t_steps = torch.cat([
            self._model.round_sigma(t_steps),
            torch.zeros_like(t_steps[:1])
        ])

        # Start from noise
        x = torch.randn(num_samples, 3, self.resolution, self.resolution, device=device)
        x = x.to(high_prec_dtype) * t_steps[0]

        # Sampling loop with 2nd order Heun correction
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # Stochastic churn
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self._model.round_sigma(t_cur + gamma * t_cur)
            x_hat = x + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x)

            # Euler step
            denoised = self.forward(x_hat.float(), t_hat.expand(num_samples), one_hot)
            denoised = denoised.to(high_prec_dtype)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # 2nd order Heun correction
            if i < num_steps - 1:
                denoised = self.forward(x_next.float(), t_next.expand(num_samples), one_hot)
                denoised = denoised.to(high_prec_dtype)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            x = x_next

        # Output images in [-1, 1]
        images = x.float().clamp(-1, 1)
        return images, labels

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
        """Return activation shapes for hookable layers."""
        if self._layer_shapes is not None:
            return self._layer_shapes

        self._layer_shapes = {}
        layer_names = self.hookable_layers

        def make_shape_hook(name):
            def hook(module, inp, output):  # pylint: disable=unused-argument
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
    ) -> 'EDMImageNetAdapter':
        """Load adapter from pickle checkpoint file.

        Expects pickle format with 'ema' key containing the full model.
        This is the standard format for EDM checkpoints like
        `edm-imagenet-64x64-cond-adm.pkl`.
        """
        # Ensure vendored NVIDIA modules are importable for pickle
        from adapt_diff.vendor.nvidia_compat import ensure_nvidia_modules
        ensure_nvidia_modules()

        print(f"Loading EDM from {checkpoint_path}...")
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)

        # EMA weights are preferred
        model = data['ema'].to(device)
        model.eval()

        print(f"Loaded EDM: {model.img_resolution}x{model.img_resolution}, "
              f"{model.label_dim} classes")
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
            # Sampling defaults
            "sigma_max": 80.0,
            "sigma_min": 0.002,
            "default_steps": 50,
            "rho": 7.0,
        }

    def to(self, device: str) -> 'EDMImageNetAdapter':
        self._model = self._model.to(device)
        self._device = device
        return self

    def eval(self) -> 'EDMImageNetAdapter':
        self._model.eval()
        return self
