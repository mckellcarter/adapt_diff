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

    Noise Level Abstraction:
        All adapters use a universal `noise_level` parameter (0-100) representing
        the percentage of the noise schedule:
        - 0 = fully denoised (clean image)
        - 100 = fully noised (pure noise)

        Each adapter translates this to its native format:
        - EDM/DMD2: sigma values (log-space interpolation)
        - DDPM/SD: timesteps (0-999)

        This abstraction allows model-agnostic code to work with any diffusion model.
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

    @property
    @abstractmethod
    def prediction_type(self) -> str:
        """
        Model's prediction target.

        Returns:
            One of: 'epsilon', 'sample', 'v_prediction'
            - epsilon: Predicts noise (most common)
            - sample: Predicts denoised x0
            - v_prediction: Predicts velocity
        """
        pass

    @property
    @abstractmethod
    def uses_latent(self) -> bool:
        """
        Whether model operates in latent space or pixel space.

        Returns:
            True if latent-space model (e.g., Stable Diffusion)
            False if pixel-space model (e.g., EDM, DMD2)
        """
        pass

    @property
    @abstractmethod
    def in_channels(self) -> int:
        """
        Number of input channels for the model.

        Returns:
            Channel count: 3 (pixel RGB), 4 (SD latent), etc.
        """
        pass

    @property
    @abstractmethod
    def conditioning_type(self) -> str:
        """
        Type of conditioning the model uses.

        Returns:
            One of: 'class', 'text', 'unconditional'
        """
        pass

    @property
    def latent_scale_factor(self) -> int:
        """
        Spatial downsampling factor for latent models.

        Returns:
            Downsampling factor (e.g., 8 for SD: 512->64)
            Returns 1 for pixel-space models
        """
        return 8 if self.uses_latent else 1

    @abstractmethod
    def noise_level_to_native(
        self,
        noise_level: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert universal noise_level (0-100) to model's native format.

        Args:
            noise_level: Noise level as percentage (0-100)
                - 0 = fully denoised (clean)
                - 100 = fully noised (pure noise)

        Returns:
            Native noise parameter:
                - EDM/DMD2: sigma value
                - DDPM/SD: timestep (integer 0-999)

        Each adapter implements this based on its noise parameterization.
        """
        pass

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for denoising.

        Args:
            x: Noisy input tensor (B, C, H, W)
            t: Native timestep/sigma (B,) or scalar. Use noise_level_to_native()
                to convert from noise_level (0-100) if needed.
            class_labels: One-hot class labels (B, num_classes) or None
            **kwargs: Model-specific options (e.g., text embeddings)

        Returns:
            Model prediction tensor (B, C, H, W). The prediction type varies:
            - 'sample': returns denoised x0
            - 'epsilon': returns predicted noise
            - 'v_prediction': returns velocity
            Use pred_to_sample() to convert any prediction type to x0.
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

    def pred_to_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert model output to estimated clean sample (x0).

        This method handles the prediction_type conversion so that callers
        can always get an x0 estimate regardless of what the model predicts.

        Args:
            x_t: Current noisy sample (B, C, H, W)
            t: Current native timestep/sigma (from get_timesteps(), not noise_level)
            model_output: Raw output from forward() (B, C, H, W)

        Returns:
            x0: Estimated clean sample (B, C, H, W)

        Default implementation returns model_output unchanged (for sample-prediction).
        Override in epsilon/v_prediction adapters.
        """
        # Default: assume model predicts x0 directly (sample prediction)
        return model_output

    @abstractmethod
    def get_timesteps(
        self,
        num_steps: int,
        device: str = 'cuda',
        noise_level_max: float = 100.0,
        noise_level_min: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Return noise schedule for sampling with num_steps.

        Args:
            num_steps: Number of denoising steps
            device: Target device for tensor
            noise_level_max: Starting noise level (0-100), default 100 (pure noise)
            noise_level_min: Ending noise level (0-100), default 0 (clean)
            **kwargs: Model-specific schedule parameters (e.g., rho for EDM)

        Returns:
            Schedule tensor in the adapter's native format (num_steps,) or (num_steps+1,):
            - For sigma-based (EDM/DMD2): floats representing sigma values
            - For timestep-based (DDPM): integers [0-999]

            The returned values are in native format, not noise_level. Use
            noise_level_to_native() to convert noise_level to native format.
        """
        pass

    @abstractmethod
    def step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Single denoising step from x_t to x_{t-1}.

        Handles prediction type conversion internally (epsilon -> x0 if needed).

        Args:
            x_t: Current noisy sample (B, C, H, W)
            t: Current native timestep/sigma (B,) or scalar (from get_timesteps())
            model_output: Raw model output (B, C, H, W)
            **kwargs: Scheduler-specific options (e.g., eta, generator, t_next)

        Returns:
            x_{t-1}: Less noisy sample (B, C, H, W)

        Note: The `t` parameter is in native format (sigma or timestep), not noise_level.
        This is because step() is called in a loop using values from get_timesteps().
        """
        pass

    @abstractmethod
    def get_initial_noise(
        self,
        batch_size: int,
        device: str = 'cuda',
        generator: Optional[torch.Generator] = None,
        noise_level: float = 100.0
    ) -> torch.Tensor:
        """
        Generate correctly shaped and scaled initial noise.

        Args:
            batch_size: Number of samples
            device: Target device
            generator: Optional RNG for reproducibility
            noise_level: Starting noise level (0-100), default 100 (pure noise).
                Each adapter scales the noise appropriately for its native format.

        Returns:
            Initial noise tensor with correct shape and scaling:
            - Latent models: (B, 4, H/8, W/8)
            - Pixel models: (B, 3, H, W)
            - Scaled appropriately for the noise_level
        """
        pass

    @abstractmethod
    def prepare_conditioning(
        self,
        text: Optional[str] = None,
        class_label: Optional[int] = None,
        batch_size: int = 1,
        device: str = 'cuda',
        **kwargs
    ) -> Any:
        """
        Prepare model-ready conditioning.

        Args:
            text: Text prompt (for text-conditioned models)
            class_label: Class index (for class-conditioned models)
            batch_size: Number of samples to condition
            device: Target device
            **kwargs: Model-specific (e.g., negative_prompt, truncation)

        Returns:
            Conditioning in model-specific format:
            - Text models: encoder_hidden_states (B, seq_len, dim)
            - Class models: one-hot labels (B, num_classes)
            - Unconditional: None or empty tensor

        Raises:
            ValueError: If required conditioning not provided
        """
        pass

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to model's internal representation.

        Args:
            images: Pixel-space images (B, 3, H, W), range [-1, 1]

        Returns:
            Internal representation:
            - Latent models: (B, 4, H/8, W/8) latent codes
            - Pixel models: Identity, returns input unchanged
        """
        return images

    def decode(self, representation: torch.Tensor) -> torch.Tensor:
        """
        Decode internal representation to pixel space.

        Args:
            representation: Model's internal representation

        Returns:
            Images (B, 3, H, W), range [-1, 1]
        """
        return representation

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Any,
        uncond: Optional[Any] = None,
        guidance_scale: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward with classifier-free guidance.

        Args:
            x: Noisy input (B, C, H, W)
            t: Native timestep/sigma (same as forward())
            cond: Conditional input from prepare_conditioning()
            uncond: Unconditional input (None = use empty/null conditioning)
            guidance_scale: CFG scale (1.0 = no guidance)
            **kwargs: Passed to forward()

        Returns:
            Guided model output (B, C, H, W)

        Formula: output = uncond_out + scale * (cond_out - uncond_out)

        Note: Models without CFG support should just call forward(cond) and
        ignore guidance_scale. This provides a unified interface.
        """
        if guidance_scale == 1.0 or uncond is None:
            # No CFG, just forward with conditioning
            if isinstance(cond, dict):
                return self.forward(x, t, **cond, **kwargs)
            else:
                return self.forward(x, t, class_labels=cond, **kwargs)

        raise NotImplementedError(
            f"{self.__class__.__name__} does not support classifier-free guidance"
        )

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
