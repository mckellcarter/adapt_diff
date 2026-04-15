"""Tests for generation module with activation masking and noise control."""

import numpy as np
import pytest
import torch

from adapt_diff import GeneratorAdapter
from adapt_diff.extraction import ActivationMasker
from adapt_diff.generation import generate, GenerationResult


class MockModule(torch.nn.Module):
    """Mock module for testing hooks."""

    def __init__(self):
        super().__init__()
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        return x


class MockAdapter(GeneratorAdapter):
    """Mock adapter for testing generation."""

    def __init__(self):
        self._modules = {
            "encoder_bottleneck": MockModule(),
            "midblock": MockModule(),
            "decoder_block_0": MockModule(),
        }
        self._step_noises = []  # Track noise values passed to step()

    @property
    def model_type(self) -> str:
        return "mock"

    @property
    def resolution(self) -> int:
        return 64

    @property
    def num_classes(self) -> int:
        return 1000

    @property
    def hookable_layers(self):
        return list(self._modules.keys())

    @property
    def prediction_type(self) -> str:
        return "sample"

    @property
    def uses_latent(self) -> bool:
        return False

    @property
    def conditioning_type(self) -> str:
        return "class"

    @property
    def in_channels(self) -> int:
        return 3

    def forward(self, x, sigma, class_labels=None, **kwargs):
        return x * 0.9  # Simple denoising mock

    def forward_with_cfg(self, x, t, cond, uncond=None, guidance_scale=1.0, **kwargs):
        return self.forward(x, t, class_labels=cond)

    def register_activation_hooks(self, layer_names, hook_fn):
        handles = []
        for name in layer_names:
            if name in self._modules:
                handle = self._modules[name].register_forward_hook(hook_fn)
                handles.append(handle)
        return handles

    def get_layer_shapes(self):
        return {
            "encoder_bottleneck": (256, 8, 8),
            "midblock": (512, 4, 4),
            "decoder_block_0": (256, 8, 8),
        }

    def get_timesteps(self, num_steps, device='cuda', **kwargs):
        # Simple linear schedule
        sigma_max = kwargs.get('sigma_max', 80.0) or 80.0
        sigma_min = kwargs.get('sigma_min', 0.002) or 0.002
        sigmas = torch.linspace(sigma_max, sigma_min, num_steps, device=device)
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def get_initial_noise(self, batch_size, device="cpu", generator=None, noise_level=100.0):
        return torch.randn(batch_size, 3, 64, 64, device=device)

    def step(self, x_t, t, model_output, t_next=None, step_noise=None, **kwargs):
        # Track step_noise for testing
        self._step_noises.append(step_noise)
        if t_next is None or float(t_next.flatten()[0]) == 0:
            return model_output
        noise = step_noise if step_noise is not None else torch.randn_like(model_output)
        t_scale = t_next.flatten()[0] if t_next.numel() > 1 else t_next
        return model_output + t_scale * 0.1 * noise

    def prepare_conditioning(self, class_label=None, text=None, batch_size=1, device='cpu', **kwargs):
        if class_label is not None:
            labels = torch.full((batch_size,), class_label, device=device, dtype=torch.long)
        else:
            labels = torch.randint(0, self.num_classes, (batch_size,), device=device)
        return torch.eye(self.num_classes, device=device)[labels]

    def noise_level_to_native(self, noise_level):
        return noise_level * 0.8

    def native_to_noise_level(self, native):
        return native / 0.8

    def decode(self, representation):
        return representation

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device="cpu", **kwargs):
        return cls()

    @classmethod
    def get_default_config(cls):
        return {}


class TestGenerateWithActivationMasker:
    """Tests for generate() with ActivationMasker."""

    def test_masker_hooks_registered_and_removed(self):
        """ActivationMasker hooks should be registered and cleaned up."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        masker.set_mask("midblock", torch.ones(1, 512, 4, 4))

        result = generate(
            adapter,
            class_label=0,
            num_steps=4,
            num_samples=1,
            device='cpu',
            sigma_max=80.0,
            sigma_min=0.002,
            activation_masker=masker,
        )

        assert isinstance(result, GenerationResult)
        # Hooks should be removed after generation
        assert len(masker._handles) == 0

    def test_masker_hooks_removed_on_exception(self):
        """Hooks should be cleaned up even if generation fails."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        masker.set_mask("midblock", torch.ones(1, 512, 4, 4))

        # Make forward fail
        original_forward = adapter.forward_with_cfg
        def failing_forward(*args, **kwargs):
            raise RuntimeError("Test error")
        adapter.forward_with_cfg = failing_forward

        with pytest.raises(RuntimeError, match="Test error"):
            generate(
                adapter,
                class_label=0,
                num_steps=4,
                num_samples=1,
                device='cpu',
                sigma_max=80.0,
                sigma_min=0.002,
                activation_masker=masker,
            )

        # Hooks should still be removed
        assert len(masker._handles) == 0


class TestGenerateMaskSteps:
    """Tests for mask_steps parameter."""

    def test_mask_steps_removes_hooks_at_step(self):
        """Masker hooks should be removed after mask_steps."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        masker.set_mask("midblock", torch.ones(1, 512, 4, 4))

        # Track hook count during generation
        hook_counts = []
        original_forward = adapter.forward_with_cfg
        def tracking_forward(*args, **kwargs):
            hook_counts.append(len(masker._handles))
            return original_forward(*args, **kwargs)
        adapter.forward_with_cfg = tracking_forward

        generate(
            adapter,
            class_label=0,
            num_steps=4,
            num_samples=1,
            device='cpu',
            sigma_max=80.0,
            sigma_min=0.002,
            activation_masker=masker,
            mask_steps=2,
        )

        # First 2 steps should have hooks, rest should not
        assert hook_counts[0] > 0  # Step 0: hooks active
        assert hook_counts[1] > 0  # Step 1: hooks active
        assert hook_counts[2] == 0  # Step 2: hooks removed
        if len(hook_counts) > 3:
            assert hook_counts[3] == 0  # Step 3: hooks still removed


class TestGenerateNoiseMode:
    """Tests for noise_mode parameter."""

    def test_noise_mode_stochastic_uses_none(self):
        """Stochastic mode should pass None to step() (fresh noise)."""
        adapter = MockAdapter()

        generate(
            adapter,
            class_label=0,
            num_steps=4,
            num_samples=1,
            device='cpu',
            sigma_max=80.0,
            sigma_min=0.002,
            noise_mode="stochastic",
        )

        # All step_noise values should be None
        assert all(n is None for n in adapter._step_noises)

    def test_noise_mode_zero_uses_zeros(self):
        """Zero mode should pass zero tensors to step()."""
        adapter = MockAdapter()

        generate(
            adapter,
            class_label=0,
            num_steps=4,
            num_samples=1,
            device='cpu',
            sigma_max=80.0,
            sigma_min=0.002,
            noise_mode="zero",
        )

        # All step_noise values should be zero tensors
        for noise in adapter._step_noises[:-1]:  # Last step may be None (final step)
            if noise is not None:
                assert torch.allclose(noise, torch.zeros_like(noise))

    def test_noise_mode_fixed_is_reproducible(self):
        """Fixed mode should produce same noise sequence with same seed."""
        adapter1 = MockAdapter()
        adapter2 = MockAdapter()

        generate(
            adapter1,
            class_label=0,
            num_steps=4,
            num_samples=1,
            device='cpu',
            sigma_max=80.0,
            sigma_min=0.002,
            noise_mode="fixed",
            seed=42,
        )

        generate(
            adapter2,
            class_label=0,
            num_steps=4,
            num_samples=1,
            device='cpu',
            sigma_max=80.0,
            sigma_min=0.002,
            noise_mode="fixed",
            seed=42,
        )

        # Same seed should produce same noise tensors
        for n1, n2 in zip(adapter1._step_noises, adapter2._step_noises):
            if n1 is not None and n2 is not None:
                assert torch.allclose(n1, n2)

    def test_noise_mode_fixed_different_seeds_differ(self):
        """Fixed mode with different seeds should produce different noise."""
        adapter1 = MockAdapter()
        adapter2 = MockAdapter()

        generate(
            adapter1,
            class_label=0,
            num_steps=4,
            num_samples=1,
            device='cpu',
            sigma_max=80.0,
            sigma_min=0.002,
            noise_mode="fixed",
            seed=42,
        )

        generate(
            adapter2,
            class_label=0,
            num_steps=4,
            num_samples=1,
            device='cpu',
            sigma_max=80.0,
            sigma_min=0.002,
            noise_mode="fixed",
            seed=123,
        )

        # Different seeds should produce different noise
        for n1, n2 in zip(adapter1._step_noises, adapter2._step_noises):
            if n1 is not None and n2 is not None:
                assert not torch.allclose(n1, n2)


class TestGenerateBasic:
    """Basic tests for generate() function."""

    def test_returns_generation_result(self):
        """generate() should return a GenerationResult."""
        adapter = MockAdapter()
        result = generate(
            adapter,
            class_label=0,
            num_steps=4,
            num_samples=1,
            device='cpu',
            sigma_max=80.0,
            sigma_min=0.002,
        )

        assert isinstance(result, GenerationResult)
        assert result.images is not None
        assert result.labels is not None

    def test_output_shapes(self):
        """Output should have correct shapes."""
        adapter = MockAdapter()
        result = generate(
            adapter,
            class_label=5,
            num_steps=4,
            num_samples=2,
            device='cpu',
            sigma_max=80.0,
            sigma_min=0.002,
        )

        assert result.images.shape == (2, 64, 64, 3)  # (B, H, W, C) uint8
        assert result.labels.shape == (2,)
