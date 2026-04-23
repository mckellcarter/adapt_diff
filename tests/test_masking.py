"""Tests for ActivationMasker and related utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from adapt_diff import GeneratorAdapter
from adapt_diff.extraction import (
    ActivationMasker,
    load_activation_from_npz,
    unflatten_activation,
)


class MockModule(torch.nn.Module):
    """Mock module for testing hooks."""

    def forward(self, x):
        return x


class MockAdapter(GeneratorAdapter):
    """Mock adapter for testing."""

    def __init__(self):
        self._modules = {
            "encoder_bottleneck": MockModule(),
            "midblock": MockModule(),
            "decoder_block_0": MockModule(),
        }

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
        return "epsilon"

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
        return torch.randn_like(x)

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

    def get_timesteps(self, num_steps, sigma_max=80.0, sigma_min=0.002, rho=7.0):
        return torch.linspace(sigma_max, sigma_min, num_steps)

    def get_initial_noise(self, batch_size, device="cuda", seed=None):
        return torch.randn(batch_size, 3, 64, 64, device=device)

    def step(self, x, pred, sigma, sigma_next, **kwargs):
        return x - pred * (sigma - sigma_next)

    def prepare_conditioning(self, class_labels=None, text=None, **kwargs):
        return {"class_labels": class_labels}

    def noise_level_to_native(self, noise_level):
        return noise_level * 0.8  # Mock sigma conversion

    def native_to_noise_level(self, native):
        return native / 0.8  # Mock reverse conversion

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device="cuda", **kwargs):
        return cls()

    @classmethod
    def get_default_config(cls):
        return {}


class TestActivationMasker:
    """Tests for ActivationMasker class."""

    def test_init(self):
        """ActivationMasker should initialize correctly."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)

        assert masker.adapter is adapter
        assert masker.masks == {}
        assert masker._handles == []

    def test_set_mask(self):
        """set_mask should store tensor on CPU."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        activation = torch.randn(1, 256, 8, 8)

        masker.set_mask("midblock", activation)

        assert "midblock" in masker.masks
        assert masker.masks["midblock"].device.type == "cpu"

    def test_clear_mask(self):
        """clear_mask should remove specific mask."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        masker.set_mask("layer1", torch.randn(1, 128, 16, 16))
        masker.set_mask("layer2", torch.randn(1, 256, 8, 8))

        masker.clear_mask("layer1")

        assert "layer1" not in masker.masks
        assert "layer2" in masker.masks

    def test_clear_masks(self):
        """clear_masks should remove all masks."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        masker.set_mask("layer1", torch.randn(1, 128, 16, 16))
        masker.set_mask("layer2", torch.randn(1, 256, 8, 8))

        masker.clear_masks()

        assert len(masker.masks) == 0

    def test_hook_replaces_output(self):
        """Hook should replace output with mask value."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        fixed = torch.ones(1, 64, 16, 16)
        masker.set_mask("test_layer", fixed)

        hook_fn = masker._make_hook("test_layer")
        original = torch.zeros(1, 64, 16, 16)
        result = hook_fn(None, None, original)

        assert torch.allclose(result, fixed)

    def test_hook_passthrough_without_mask(self):
        """Hook should pass through when no mask set."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)

        hook_fn = masker._make_hook("test_layer")
        original = torch.randn(1, 64, 16, 16)
        result = hook_fn(None, None, original)

        assert result is original

    def test_hook_batch_expansion(self):
        """Hook should expand batch dim 1 to match output batch."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        fixed = torch.ones(1, 64, 16, 16)
        masker.set_mask("test_layer", fixed)

        hook_fn = masker._make_hook("test_layer")
        batch_output = torch.zeros(4, 64, 16, 16)
        result = hook_fn(None, None, batch_output)

        assert result.shape == (4, 64, 16, 16)
        assert torch.allclose(result, torch.ones(4, 64, 16, 16))

    def test_hook_tuple_output(self):
        """Hook should handle tuple outputs (replace first element)."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        fixed = torch.ones(1, 64, 16, 16) * 42
        masker.set_mask("test_layer", fixed)

        hook_fn = masker._make_hook("test_layer")
        extra = torch.randn(1, 32)
        original = (torch.zeros(1, 64, 16, 16), extra)
        result = hook_fn(None, None, original)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.allclose(result[0], fixed)
        assert result[1] is extra

    def test_register_hooks(self):
        """register_hooks should register hooks for masked layers."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        masker.set_mask("encoder_bottleneck", torch.randn(1, 256, 8, 8))

        masker.register_hooks()

        assert len(masker._handles) > 0

        masker.remove_hooks()
        assert len(masker._handles) == 0

    def test_context_manager(self):
        """Context manager should register and remove hooks."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        masker.set_mask("encoder_bottleneck", torch.randn(1, 256, 8, 8))

        with masker as m:
            assert len(m._handles) > 0

        assert len(masker._handles) == 0


class TestUnflattenActivation:
    """Tests for unflatten_activation."""

    def test_basic_unflatten(self):
        """Should reshape (1, C*H*W) to (1, C, H, W)."""
        flat = torch.randn(1, 256 * 8 * 8)
        result = unflatten_activation(flat, (256, 8, 8))
        assert result.shape == (1, 256, 8, 8)

    def test_1d_input(self):
        """Should handle 1D input by adding batch dim."""
        flat = torch.randn(128 * 16 * 16)
        result = unflatten_activation(flat, (128, 16, 16))
        assert result.shape == (1, 128, 16, 16)

    def test_preserves_values(self):
        """Should preserve tensor values through flatten/unflatten."""
        original = torch.arange(64 * 4 * 4).float().reshape(1, 64, 4, 4)
        flat = original.reshape(1, -1)
        restored = unflatten_activation(flat, (64, 4, 4))
        assert torch.equal(restored, original)

    def test_batch_size_preserved(self):
        """Should preserve batch size > 1."""
        flat = torch.randn(4, 128 * 8 * 8)
        result = unflatten_activation(flat, (128, 8, 8))
        assert result.shape == (4, 128, 8, 8)


class TestLoadActivationFromNpz:
    """Tests for load_activation_from_npz."""

    def test_load_single_layer(self):
        """Should load single layer from NPZ file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            data = np.random.randn(1, 256 * 8 * 8).astype(np.float32)
            np.savez(path, midblock=data)

            loaded = load_activation_from_npz(path, "midblock")

            assert isinstance(loaded, torch.Tensor)
            assert loaded.shape == (1, 256 * 8 * 8)

    def test_load_adds_batch_dim(self):
        """Should add batch dim for 1D data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            data = np.random.randn(256 * 8 * 8).astype(np.float32)
            np.savez(path, layer=data)

            loaded = load_activation_from_npz(path, "layer")

            assert loaded.shape == (1, 256 * 8 * 8)

    def test_missing_layer_raises(self):
        """Should raise ValueError for missing layer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            np.savez(path, encoder=np.zeros((1, 64)))

            with pytest.raises(ValueError, match="not found"):
                load_activation_from_npz(path, "midblock")

    def test_error_lists_available(self):
        """Error should list available layers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            np.savez(path, layer1=np.zeros((1, 64)), layer2=np.zeros((1, 128)))

            with pytest.raises(ValueError, match="layer1"):
                load_activation_from_npz(path, "missing")
