"""
Generation with trajectory extraction using adapter interface.

Supports both diffusion (iterative denoising) and autoregressive (token-by-token)
generation modes. Ported from diffviews.core.generator for model-agnostic generation.
"""

from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from .base import GeneratorAdapter

if TYPE_CHECKING:
    from .extraction import ActivationMasker


class GenerationResult(NamedTuple):
    """Result of diffusion generate() call."""
    images: torch.Tensor  # (B, H, W, 3) uint8
    labels: torch.Tensor  # (B,) class labels
    trajectory: Optional[List[np.ndarray]] = None  # activations per step
    intermediates: Optional[List[torch.Tensor]] = None  # images per step
    timesteps: Optional[List[float]] = None  # native timesteps
    noised_inputs: Optional[List[torch.Tensor]] = None  # noised latents per step


@dataclass
class TextGenerationResult:
    """Result from autoregressive text generation."""
    tokens: torch.Tensor  # (B, seq_len) token IDs
    text: List[str]  # Decoded strings
    trajectory: Optional[List[np.ndarray]] = None  # Per-token activations
    token_probs: Optional[torch.Tensor] = None  # Per-token probabilities


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


def _flatten_activations(
    acts: Dict[str, torch.Tensor],
    layer_names: List[str]
) -> Optional[np.ndarray]:
    """
    Flatten activations dict to single array for trajectory storage.

    Args:
        acts: Dict mapping layer_name -> activation tensor
        layer_names: Ordered list of layer names to include

    Returns:
        Concatenated flattened array (B, total_features) or None if empty
    """
    layer_acts = []
    for layer_name in sorted(layer_names):
        act = acts.get(layer_name)
        if act is not None:
            if len(act.shape) >= 3:
                # Flatten spatial/sequence dims: (B, ...) -> (B, -1)
                act = act.reshape(act.shape[0], -1)
            layer_acts.append(act.numpy())
    if layer_acts:
        return np.concatenate(layer_acts, axis=1)
    return None


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
    # Activation masking (optional)
    activation_masker: Optional["ActivationMasker"] = None,
    mask_steps: Optional[int] = None,
    # Noise control
    noise_mode: str = "stochastic",
    # Legacy masking (for inpainting)
    masker: Optional[Any] = None,
    mask_image: Optional[torch.Tensor] = None,
    # Trajectory extraction (optional)
    extract_layers: Optional[List[str]] = None,
    return_trajectory: bool = False,
    return_intermediates: bool = False,
    return_noised_inputs: bool = False,
    # Schedule params
    rho: float = 7.0,
    # Autoregressive params
    temperature: float = 1.0,
    top_p: float = 0.95,
    **schedule_kwargs
) -> Union[GenerationResult, TextGenerationResult]:
    """
    Generate samples using adapter. Branches on adapter.generation_mode.

    For diffusion models (generation_mode='diffusion'):
        Iterative denoising from noise to clean image.

    For autoregressive models (generation_mode='autoregressive'):
        Token-by-token text generation.

    Args:
        adapter: GeneratorAdapter instance
        class_label: Class label (0-999), random if None, -1 for uniform
        caption: Text caption for T2I models (overrides class_label)
        num_steps: Number of denoising steps (diffusion) or max new tokens (autoregressive)
        noise_level_max: Absolute max noise level (0-100), default 100
        noise_level_min: Absolute min noise level (0-100), default 0
        target_noise_max: Target starting noise level (defaults to noise_level_max)
        target_noise_min: Target ending noise level (defaults to noise_level_min)
        guidance_scale: CFG scale (1.0=no guidance)
        num_samples: Number of images
        device: Device for generation
        seed: Random seed
        activation_masker: ActivationMasker to replace layer outputs during generation
        mask_steps: Number of steps to keep masker active (default: all steps)
        noise_mode: Noise mode for stochastic samplers:
            - "stochastic": Fresh random noise at each step (default)
            - "fixed": Same pre-generated noise sequence (reproducible)
            - "zero": No noise added (deterministic)
        masker: Optional masker object with apply_mask(x, t, mask) for inpainting
        mask_image: Optional mask tensor for inpainting
        extract_layers: Layers to extract for trajectory
        return_trajectory: Return activations at each step
        return_intermediates: Return intermediate images
        return_noised_inputs: Return noised latents at each step
        rho: Karras schedule parameter (for EDM-style models)
        temperature: Sampling temperature (autoregressive only)
        top_p: Nucleus sampling threshold (autoregressive only)
        **schedule_kwargs: Additional schedule params

    Returns:
        GenerationResult (diffusion) or TextGenerationResult (autoregressive)
    """
    # Branch on generation mode
    if adapter.generation_mode == 'autoregressive':
        return _generate_autoregressive(
            adapter=adapter,
            num_steps=num_steps,
            seed=seed,
            extract_layers=extract_layers,
            return_trajectory=return_trajectory,
            temperature=temperature,
            top_p=top_p,
            device=device,
            **schedule_kwargs
        )

    # Diffusion generation follows
    if seed is not None:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Get timesteps from adapter using model-agnostic noise_level interface
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

    # Pre-generate step noises based on mode
    noise_shape = (num_samples, adapter.in_channels, adapter.resolution, adapter.resolution)
    if noise_mode == "zero":
        step_noises = [torch.zeros(noise_shape, device=device) for _ in range(num_steps - 1)]
    elif noise_mode == "fixed":
        rng = torch.Generator(device=device).manual_seed(seed if seed is not None else 42)
        step_noises = [torch.randn(noise_shape, device=device, generator=rng)
                       for _ in range(num_steps - 1)]
    else:  # stochastic (default)
        step_noises = [None] * (num_steps - 1)

    # Register activation masker if provided
    if activation_masker is not None:
        activation_masker.register_hooks()
        if mask_steps is None:
            mask_steps = num_steps  # Keep masker active for all steps by default

    # Setup trajectory extraction
    trajectory_activations = []
    intermediate_images = []
    noised_input_images = []
    extractor = None

    if return_trajectory and extract_layers:
        extractor = ActivationExtractor(adapter, extract_layers)
        extractor.register_hooks()

f    # Generate initial noise using adapter's model-agnostic interface
    x = adapter.get_initial_noise(
        batch_size=num_samples,
        device=device,
        noise_level=noise_level_max
    )

    # Iterative denoising with activation masker management
    try:
        for i, t in enumerate(timesteps[:-1]):
            # Remove activation masker after mask_steps
            if activation_masker is not None and i == mask_steps:
                activation_masker.remove_hooks()

            # Capture noised input before denoising
            if return_noised_inputs:
                noised_input_images.append(tensor_to_uint8_image(adapter.decode(x)))

            # Create batched sigma tensor to match diffviews behavior
            t_batched = torch.ones(num_samples, device=device) * t

            # Forward with CFG
            pred = adapter.forward_with_cfg(x, t_batched, cond, uncond, guidance_scale)

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
                x0_estimate = adapter.pred_to_sample(x, t_batched, pred)
                intermediate_images.append(tensor_to_uint8_image(adapter.decode(x0_estimate)))

            # Single denoising step with noise control
            t_next = timesteps[i + 1]
            t_next_batched = torch.ones(num_samples, device=device) * t_next
            step_noise = step_noises[i] if i < len(step_noises) else None
            x = adapter.step(x, t_batched, pred, t_next=t_next_batched, step_noise=step_noise)

            # Apply inpainting mask if provided
            if masker is not None and mask_image is not None:
                x = masker.apply_mask(x, t_next, mask_image)
    finally:
        # Ensure cleanup of hooks
        if extractor is not None:
            extractor.remove_hooks()
        if activation_masker is not None:
            activation_masker.remove_hooks()

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


@torch.no_grad()
def _generate_autoregressive(
    adapter: GeneratorAdapter,
    num_steps: int = 256,
    seed: Optional[int] = None,
    extract_layers: Optional[List[str]] = None,
    return_trajectory: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: str = 'cuda',
    **kwargs
) -> TextGenerationResult:
    """
    Token-by-token generation with optional activation extraction.

    Args:
        adapter: GeneratorAdapter with generation_mode='autoregressive'
        num_steps: Max new tokens to generate
        seed: Random seed
        extract_layers: Layers to extract for trajectory
        return_trajectory: Return activations at each step
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        device: Target device
        **kwargs: Passed to adapter.forward()

    Returns:
        TextGenerationResult with tokens, text, and optional trajectory
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Get initial sequence (tokenized prompt)
    x = adapter.get_initial_noise(1, device)

    # Get generation steps (token positions)
    timesteps = adapter.get_timesteps(num_steps, device)

    # Setup hooks for trajectory extraction
    trajectory = []
    extractor = None
    if return_trajectory and extract_layers:
        extractor = ActivationExtractor(adapter, extract_layers)
        extractor.register_hooks()

    try:
        for t in timesteps:
            # Forward pass - get logits for next token
            logits = adapter.forward(x, t, **kwargs)

            # Capture activations
            if extractor is not None:
                acts = extractor.get_activations()
                flat = _flatten_activations(acts, extract_layers)
                if flat is not None:
                    trajectory.append(flat)
                extractor.clear()

            # Sample next token and extend sequence
            x = adapter.step(x, t, logits, temperature=temperature, top_p=top_p)

            # Check for EOS
            if hasattr(adapter, 'tokenizer'):
                eos_id = adapter.tokenizer.eos_token_id
                if eos_id is not None and x[0, -1].item() == eos_id:
                    break
    finally:
        if extractor is not None:
            extractor.remove_hooks()

    # Decode tokens to text
    text = adapter.tokenizer.batch_decode(x, skip_special_tokens=True)

    return TextGenerationResult(
        tokens=x,
        text=text,
        trajectory=trajectory if return_trajectory else None
    )
