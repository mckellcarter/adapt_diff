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

    # DMD2 native sigma range
    SIGMA_MAX = 80.0
    SIGMA_MIN = 0.002

    def noise_level_to_native(
        self,
        noise_level: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert noise_level (0-100) to DMD2 sigma.

        Uses log-space interpolation:
        - noise_level 0 → sigma_min (0.002)
        - noise_level 100 → sigma_max (80.0)

        Args:
            noise_level: Noise level as percentage (0-100)

        Returns:
            sigma: DMD2 sigma value
        """
        t = noise_level / 100.0
        # Log-space interpolation: sigma = sigma_min^(1-t) * sigma_max^t
        sigma = self.SIGMA_MIN ** (1 - t) * self.SIGMA_MAX ** t
        return sigma

    def native_to_noise_level(
        self,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert DMD2 sigma to noise_level (0-100).

        Inverse of noise_level_to_native:
        - sigma_min (0.002) → noise_level 0
        - sigma_max (80.0) → noise_level 100

        Args:
            sigma: DMD2 sigma value

        Returns:
            noise_level: Noise level as percentage (0-100)
        """
        import math
        # Inverse of: sigma = sigma_min^(1-t) * sigma_max^t
        # log(sigma) = (1-t)*log(sigma_min) + t*log(sigma_max)
        # t = (log(sigma) - log(sigma_min)) / (log(sigma_max) - log(sigma_min))
        log_sigma = torch.log(sigma)
        log_min = math.log(self.SIGMA_MIN)
        log_max = math.log(self.SIGMA_MAX)
        t = (log_sigma - log_min) / (log_max - log_min)
        return t * 100.0

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
        target_noise_max: float = 100.0,
        target_noise_min: float = 0.0,
        rho: float = 7.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Return Karras sigma schedule for DMD2.

        Uses Karras schedule (same as EDM/diffviews) for compatibility with
        attribution indices built using the original diffviews generator.

        Args:
            num_steps: Number of denoising steps (typically 1-10)
            device: Target device
            target_noise_max: Target starting noise level (0-100), default 100 (pure noise)
            target_noise_min: Target ending noise level (0-100), default 0 (clean)
            rho: Karras schedule parameter, default 7.0
            **kwargs: Ignored (for API consistency)

        Returns:
            Sigma tensor (num_steps + 1,) from sigma_max to 0
        """
        # Convert target noise levels to sigma
        sigma_max = float(self.noise_level_to_native(torch.tensor(target_noise_max)))
        sigma_min = float(self.noise_level_to_native(torch.tensor(target_noise_min)))
        # Ensure sigma_min is not zero for schedule
        sigma_min = max(sigma_min, self.SIGMA_MIN)

        # Karras schedule: (max^(1/rho) + ramp * (min^(1/rho) - max^(1/rho)))^rho
        ramp = torch.linspace(0, 1, num_steps, device=device)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

        # Return raw Karras schedule (no rounding) to match diffviews index training
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
        step_noise: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Ancestral sampling step for DMD2.

        DMD2 predicts x0 directly, so we use ancestral sampling:
        x_next = x0_pred + sigma_next * noise

        This matches the working diffviews generator and avoids
        numerical instability from Euler's division by small sigma.

        Args:
            x_t: Current noisy sample (B, C, H, W)
            t: Current sigma value (scalar or (B,))
            model_output: Denoised x0 prediction from forward() (B, C, H, W)
            t_next: Next sigma value (0 for final step)
            step_noise: Pre-generated noise (None=fresh randn)
            **kwargs: Unused

        Returns:
            x_{t-1}: Next (less noisy) sample
        """
        # Final step or no next sigma: return prediction directly
        if t_next is None or float(t_next) == 0:
            return model_output

        # Ancestral sampling: x0_pred + sigma_next * noise
        noise = step_noise if step_noise is not None else torch.randn_like(model_output)
        return model_output + t_next * noise

    def get_initial_noise(
        self,
        batch_size: int,
        device: str = 'cuda',
        generator: Optional[torch.Generator] = None,
        noise_level: float = 100.0
    ) -> torch.Tensor:
        """
        Generate initial noise for DMD2 sampling.

        Args:
            batch_size: Number of samples
            device: Target device
            generator: Optional RNG for reproducibility
            noise_level: Starting noise level (0-100), default 100

        Returns:
            Noise tensor (B, 3, 64, 64) scaled by corresponding sigma
        """
        sigma = float(self.noise_level_to_native(torch.tensor(noise_level)))
        noise = torch.randn(
            batch_size, 3, self.resolution, self.resolution,
            device=device,
            generator=generator
        )
        return noise * sigma

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
    def timestep_label(self) -> str:
        """Label for native timestep values in UI (σ for sigma-based models)."""
        return "σ"

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

    @property
    def training_data_id(self) -> str:
        """Training dataset identifier for yodal-train-items."""
        return 'imagenet-64x64'

    @property
    def default_checkpoint_key(self) -> str:
        """Default R2 key for DMD2 ImageNet 64x64 checkpoint."""
        return 'data/dmd2/checkpoints/dmd2-imagenet-64-10step.pkl'

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
            # Sampling defaults (noise_level 0-100 scale, DMD2 distilled for few-step)
            "noise_max": 100.0,
            "noise_min": 0.5,
            "default_steps": 5,
        }

    def to(self, device: str) -> 'DMD2ImageNetAdapter':
        self._model = self._model.to(device)
        self._device = device
        return self

    def eval(self) -> 'DMD2ImageNetAdapter':
        self._model.eval()
        return self
