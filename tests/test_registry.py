"""Tests for adapter registry."""

import pytest

from adapt_diff import (
    GeneratorAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
    register_adapter_class,
    unregister_adapter,
)


def test_list_adapters_includes_builtin():
    """Built-in adapters should be registered on import."""
    adapters = list_adapters()
    assert 'dmd2-imagenet-64' in adapters
    assert 'edm-imagenet-64' in adapters


def test_get_adapter_returns_class():
    """get_adapter should return a class, not instance."""
    adapter_cls = get_adapter('dmd2-imagenet-64')
    assert isinstance(adapter_cls, type)
    assert issubclass(adapter_cls, GeneratorAdapter)


def test_get_adapter_unknown_raises():
    """Unknown adapter names should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown adapter"):
        get_adapter('nonexistent-adapter')


def test_register_adapter_decorator():
    """@register_adapter decorator should register adapters."""
    @register_adapter('test-adapter')
    class TestAdapter(GeneratorAdapter):
        @property
        def model_type(self): return 'test'
        @property
        def resolution(self): return 64
        @property
        def num_classes(self): return 10
        @property
        def hookable_layers(self): return []
        def forward(self, x, sigma, class_labels=None, **kwargs): pass
        def register_activation_hooks(self, layers, hook_fn): return []
        def get_layer_shapes(self): return {}
        @classmethod
        def from_checkpoint(cls, path, device='cuda', **kwargs): pass
        @classmethod
        def get_default_config(cls): return {}

    assert 'test-adapter' in list_adapters()
    assert get_adapter('test-adapter') is TestAdapter

    # Cleanup
    unregister_adapter('test-adapter')


def test_register_adapter_class_manual():
    """register_adapter_class should work for manual registration."""
    class ManualAdapter(GeneratorAdapter):
        @property
        def model_type(self): return 'manual'
        @property
        def resolution(self): return 64
        @property
        def num_classes(self): return 10
        @property
        def hookable_layers(self): return []
        def forward(self, x, sigma, class_labels=None, **kwargs): pass
        def register_activation_hooks(self, layers, hook_fn): return []
        def get_layer_shapes(self): return {}
        @classmethod
        def from_checkpoint(cls, path, device='cuda', **kwargs): pass
        @classmethod
        def get_default_config(cls): return {}

    register_adapter_class('manual-adapter', ManualAdapter)
    assert 'manual-adapter' in list_adapters()

    # Cleanup
    removed = unregister_adapter('manual-adapter')
    assert removed is ManualAdapter


def test_unregister_adapter():
    """unregister_adapter should remove and return the adapter."""
    @register_adapter('temp-adapter')
    class TempAdapter(GeneratorAdapter):
        @property
        def model_type(self): return 'temp'
        @property
        def resolution(self): return 64
        @property
        def num_classes(self): return 10
        @property
        def hookable_layers(self): return []
        def forward(self, x, sigma, class_labels=None, **kwargs): pass
        def register_activation_hooks(self, layers, hook_fn): return []
        def get_layer_shapes(self): return {}
        @classmethod
        def from_checkpoint(cls, path, device='cuda', **kwargs): pass
        @classmethod
        def get_default_config(cls): return {}

    assert 'temp-adapter' in list_adapters()
    removed = unregister_adapter('temp-adapter')
    assert removed is TempAdapter
    assert 'temp-adapter' not in list_adapters()


def test_unregister_nonexistent_returns_none():
    """unregister_adapter on nonexistent adapter should return None."""
    result = unregister_adapter('does-not-exist')
    assert result is None
