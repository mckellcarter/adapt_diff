"""Tests for HookMixin functionality."""

import torch

from adapt_diff import HookMixin


class MockModule(torch.nn.Module):
    """Mock module for testing hooks."""
    def forward(self, x):
        return x * 2


class TestHookMixin:
    """Tests for HookMixin."""

    def test_init(self):
        """HookMixin should initialize empty storage."""
        mixin = HookMixin()
        assert mixin._activations == {}
        assert mixin._handles == []
        assert mixin._masks == {}
        assert mixin.num_hooks == 0

    def test_set_and_get_mask(self):
        """set_mask and get_mask should work correctly."""
        mixin = HookMixin()
        tensor = torch.randn(1, 64, 8, 8)

        mixin.set_mask('layer1', tensor)
        assert mixin.get_mask('layer1') is tensor
        assert mixin.get_mask('nonexistent') is None

    def test_clear_mask(self):
        """clear_mask should remove specific mask."""
        mixin = HookMixin()
        mixin.set_mask('layer1', torch.randn(1, 64, 8, 8))
        mixin.set_mask('layer2', torch.randn(1, 128, 4, 4))

        mixin.clear_mask('layer1')
        assert mixin.get_mask('layer1') is None
        assert mixin.get_mask('layer2') is not None

    def test_clear_masks(self):
        """clear_masks should remove all masks."""
        mixin = HookMixin()
        mixin.set_mask('layer1', torch.randn(1, 64, 8, 8))
        mixin.set_mask('layer2', torch.randn(1, 128, 4, 4))

        mixin.clear_masks()
        assert mixin.get_mask('layer1') is None
        assert mixin.get_mask('layer2') is None

    def test_extraction_hook(self):
        """make_extraction_hook should capture activations."""
        mixin = HookMixin()
        module = MockModule()

        hook = mixin.make_extraction_hook('test_layer')
        handle = module.register_forward_hook(hook)
        mixin.add_handle(handle)

        x = torch.randn(2, 64)
        _ = module(x)

        activations = mixin.get_activations()
        assert 'test_layer' in activations
        assert activations['test_layer'].shape == (2, 64)

        mixin.remove_hooks()
        assert mixin.num_hooks == 0

    def test_mask_hook(self):
        """make_mask_hook should replace output with 4D image tensors."""
        mixin = HookMixin()

        # Mock module that outputs 4D tensors (like conv layers)
        class Mock4DModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        module = Mock4DModule()

        # Mask hook is designed for 4D image tensors (B, C, H, W)
        mask = torch.ones(1, 3, 8, 8) * 42
        hook = mixin.make_mask_hook('test_layer', mask)
        handle = module.register_forward_hook(hook)
        mixin.add_handle(handle)

        x = torch.randn(2, 3, 8, 8)
        output = module(x)

        # Output should be replaced with mask (broadcast to batch size)
        assert output.shape == (2, 3, 8, 8)
        assert torch.allclose(output, torch.ones(2, 3, 8, 8) * 42)

        mixin.remove_hooks()

    def test_clear_activations(self):
        """clear_activations should empty activation dict."""
        mixin = HookMixin()
        mixin._activations['layer1'] = torch.randn(1, 64)
        mixin._activations['layer2'] = torch.randn(1, 128)

        mixin.clear_activations()
        assert mixin.get_activations() == {}

    def test_get_activation(self):
        """get_activation should return single activation."""
        mixin = HookMixin()
        tensor = torch.randn(1, 64)
        mixin._activations['layer1'] = tensor

        assert mixin.get_activation('layer1') is tensor
        assert mixin.get_activation('nonexistent') is None

    def test_num_hooks(self):
        """num_hooks should track handle count."""
        mixin = HookMixin()
        module = MockModule()

        assert mixin.num_hooks == 0

        hook = mixin.make_extraction_hook('layer1')
        handle = module.register_forward_hook(hook)
        mixin.add_handle(handle)

        assert mixin.num_hooks == 1

        hook2 = mixin.make_extraction_hook('layer2')
        handle2 = module.register_forward_hook(hook2)
        mixin.add_handle(handle2)

        assert mixin.num_hooks == 2

        mixin.remove_hooks()
        assert mixin.num_hooks == 0
