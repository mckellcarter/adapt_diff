"""
Image generation with trajectory extraction using adapter interface.

Ported from diffviews.core.generator for model-agnostic generation.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from .base import GeneratorAdapter


class GenerationResult(NamedTuple):
    """Result of generate() call."""
    images: torch.Tensor  # (B, H, W, 3) uint8
    labels: torch.Tensor  # (B,) class labels
    trajectory: Optional[List[np.ndarray]] = None  # activations per step
    intermediates: Optional[List[torch.Tensor]] = None  # images per step
    timesteps: Optional[List[float]] = None  # native timesteps
    noised_inputs: Optional[List[torch.Tensor]] = None  # noised latents per step


class ActivationExtractor:
    """
    Extract activations from a model during forward pass.

    Uses the adapter interface for model-agnostic hook registration.
    """

    def __init__(self, adapter: GeneratorAdapter, layers: List[str]):
        """
        Args:
            adapter: GeneratorAdapter instance
            layers: Layer names to extract
        """
        self.adapter = adapter
        self.layers = layers
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles = []

    def _make_hook(self, name: str):
        """Create forward hook that stores activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations[name] = output.detach().cpu()
        return hook

    def register_hooks(self):
        """Register extraction hooks on specified layers."""
        for name in self.layers:
            hook_fn = self._make_hook(name)
            handles = self.adapter.register_activation_hooks([name], hook_fn)
            self._handles.extend(handles)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self):
        """Clear stored activations."""
        self.activations.clear()

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get extracted activations."""
        return self.activations.copy()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()


def tensor_to_uint8_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor in [-1, 1] to uint8 in [0, 255].

    Args:
        tensor: Image tensor (B, C, H, W) in range [-1, 1]

    Returns:
        uint8 tensor (B, H, W, C) in range [0, 255]
    """
    images = ((tensor + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    images = images.permute(0, 2, 3, 1).cpu()
    return images


def _prepare_conditioning(
    adapter: GeneratorAdapter,
    class_label: Optional[int],
    caption: Optional[str],
    num_samples: int,
    device: str
) -> Tuple[Any, Any, torch.Tensor]:
    """
    Prepare conditioning and unconditioning for CFG.

    Returns:
        (cond, uncond, labels) - format depends on model type
    """
    num_classes = adapter.num_classes
    labels = torch.zeros(num_samples, device=device, dtype=torch.long)

    if caption is not None:
        # T2I model - use adapter's prepare_conditioning
        cond = adapter.prepare_conditioning(text=caption, batch_size=num_samples, device=device)
        uncond = adapter.prepare_conditioning(text="", batch_size=num_samples, device=device)
        return cond, uncond, labels

    # Class-conditioned model
    if class_label is not None and class_label < 0:
        # Uniform distribution (unconditional)
        labels = torch.tensor([-1], device=device).repeat(num_samples)
        one_hot = torch.ones((num_samples, num_classes), device=device) / num_classes
        uncond = one_hot.clone()
    elif class_label is None and num_classes > 0:
        # Random class
        labels = torch.randint(0, num_classes, (num_samples,), device=device)
        one_hot = torch.eye(num_classes, device=device)[labels]
        uncond = torch.zeros_like(one_hot)
    elif class_label is not None:
        # Specific class
        labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
        one_hot = torch.eye(num_classes, device=device)[labels]
        uncond = torch.zeros_like(one_hot)
    else:
        one_hot = None
        uncond = None

    return one_hot, uncond, labels


@torch.no_grad()
def generate(
    adapter: GeneratorAdapter,
    class_label: Optional[int] = None,
    caption: Optional[str] = None,
    num_steps: int = 6,
    noise_level_max: float = 100.0,
    noise_level_min: float = 0.0,
    target_noise_max: Optional[float] = None,
    target_noise_min: Optional[float] = None,
    guidance_scale: float = 1.0,
    num_samples: int = 1,
    device: str = 'cuda',
    seed: Optional[int] = None,
    # Masking (optional)
    masker: Optional[Any] = None,
    mask_image: Optional[torch.Tensor] = None,
    # Trajectory extraction (optional)
    extract_layers: Optional[List[str]] = None,
    return_trajectory: bool = False,
    return_intermediates: bool = False,
    return_noised_inputs: bool = False,
    # Schedule params
    rho: float = 7.0,
    **schedule_kwargs
) -> GenerationResult:
    """
    Generate images using multi-step denoising with optional trajectory extraction.

    Args:
        adapter: GeneratorAdapter instance
        class_label: Class label (0-999), random if None, -1 for uniform
        caption: Text caption for T2I models (overrides class_label)
        num_steps: Number of denoising steps
        noise_level_max: Absolute max noise level (0-100), default 100
        noise_level_min: Absolute min noise level (0-100), default 0
        target_noise_max: Target starting noise level (defaults to noise_level_max)
        target_noise_min: Target ending noise level (defaults to noise_level_min)
        guidance_scale: CFG scale (1.0=no guidance)
        num_samples: Number of images
        device: Device for generation
        seed: Random seed
        masker: Optional masker object with apply_mask(x, t, mask) method
        mask_image: Optional mask tensor for inpainting
        extract_layers: Layers to extract for trajectory
        return_trajectory: Return activations at each step
        return_intermediates: Return intermediate images
        return_noised_inputs: Return noised latents at each step
        rho: Karras schedule parameter (for EDM-style models)
        **schedule_kwargs: Additional schedule params

    Returns:
        GenerationResult with images, labels, and optional trajectory/intermediates/noised_inputs
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Get timesteps from adapter using universal noise_level (0-100)
    timesteps = adapter.get_timesteps(
        num_steps,
        device=device,
        noise_level_max=noise_level_max,
        noise_level_min=noise_level_min,
        target_noise_max=target_noise_max,
        target_noise_min=target_noise_min,
        rho=rho,
        **schedule_kwargs
    )

    # Prepare conditioning
    cond, uncond, labels = _prepare_conditioning(
        adapter, class_label, caption, num_samples, device
    )

    # Setup trajectory extraction
    trajectory_activations = []
    intermediate_images = []
    noised_input_images = []
    extractor = None

    if return_trajectory and extract_layers:
        extractor = ActivationExtractor(adapter, extract_layers)
        extractor.register_hooks()

    # Generate initial noise
    x = adapter.get_initial_noise(
        batch_size=num_samples,
        device=device,
        noise_level=noise_level_max
    )

    # Iterative denoising
    for i, t in enumerate(timesteps[:-1]):
        # Capture noised input before denoising
        if return_noised_inputs:
            noised_input_images.append(tensor_to_uint8_image(adapter.decode(x)))

        # Forward with CFG
        pred = adapter.forward_with_cfg(x, t, cond, uncond, guidance_scale)

        # Extract trajectory activations
        if extractor is not None:
            acts = extractor.get_activations()
            layer_acts = []
            for layer_name in sorted(extract_layers):
                act = acts.get(layer_name)
                if act is not None:
                    if len(act.shape) == 4:
                        B, C, H, W = act.shape
                        act = act.reshape(B, -1)
                    layer_acts.append(act.numpy())
            if layer_acts:
                concat_act = np.concatenate(layer_acts, axis=1)
                trajectory_activations.append(concat_act)
            extractor.clear()

        # Capture intermediate image
        if return_intermediates:
            x0_estimate = adapter.pred_to_sample(x, t, pred)
            intermediate_images.append(tensor_to_uint8_image(adapter.decode(x0_estimate)))

        # Single denoising step
        t_next = timesteps[i + 1]
        x = adapter.step(x, t, pred, t_next=t_next)

        # Apply masking if provided
        if masker is not None and mask_image is not None:
            x = masker.apply_mask(x, t_next, mask_image)

    if extractor is not None:
        extractor.remove_hooks()

    # Decode and convert to uint8
    x = adapter.decode(x)
    images = tensor_to_uint8_image(x)

    # Append final image to intermediates
    if return_intermediates:
        intermediate_images.append(images)

    return GenerationResult(
        images=images,
        labels=labels.cpu(),
        trajectory=trajectory_activations if return_trajectory else None,
        intermediates=intermediate_images if return_intermediates else None,
        timesteps=timesteps.cpu().tolist(),
        noised_inputs=noised_input_images if return_noised_inputs else None
    )
