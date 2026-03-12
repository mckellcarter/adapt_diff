"""Adapter registration and discovery."""

from typing import Dict, Optional, Type

from .base import GeneratorAdapter

_ADAPTERS: Dict[str, Type[GeneratorAdapter]] = {}


def register_adapter(name: str):
    """
    Decorator to register an adapter class.

    Usage:
        @register_adapter('imagenet-64')
        class MyAdapter(GeneratorAdapter):
            ...
    """
    def decorator(cls: Type[GeneratorAdapter]):
        if not issubclass(cls, GeneratorAdapter):
            raise TypeError(f"{cls} must be a GeneratorAdapter subclass")
        _ADAPTERS[name] = cls
        return cls
    return decorator


def get_adapter(name: str) -> Type[GeneratorAdapter]:
    """
    Get adapter class by registered name.

    Args:
        name: Adapter name (e.g., 'imagenet-64', 'sdxl')

    Returns:
        Adapter class (not instance)

    Raises:
        ValueError: If adapter not found
    """
    # Try discovery first if not found
    if name not in _ADAPTERS:
        discover_adapters()

    if name not in _ADAPTERS:
        available = list(_ADAPTERS.keys())
        raise ValueError(
            f"Unknown adapter: '{name}'. "
            f"Available: {available or '(none registered)'}"
        )
    return _ADAPTERS[name]


def list_adapters() -> list:
    """List all registered adapter names."""
    discover_adapters()
    return list(_ADAPTERS.keys())


def discover_adapters():
    """
    Auto-discover adapters from entry points.

    External packages can register adapters via setup.py/pyproject.toml:

        [project.entry-points."adapt_diff.adapters"]
        imagenet-64 = "my_package.adapters:ImageNetAdapter"
    """
    try:
        # Python 3.10+
        from importlib.metadata import entry_points
        eps = entry_points(group='adapt_diff.adapters')
    except TypeError:
        # Python 3.9
        from importlib.metadata import entry_points
        all_eps = entry_points()
        eps = all_eps.get('adapt_diff.adapters', [])
    except ImportError:
        return

    for ep in eps:
        if ep.name not in _ADAPTERS:
            try:
                cls = ep.load()
                _ADAPTERS[ep.name] = cls
            except Exception as e:
                print(f"Warning: Failed to load adapter '{ep.name}': {e}")


def register_adapter_class(name: str, cls: Type[GeneratorAdapter]):
    """Manually register an adapter class (alternative to decorator)."""
    if not issubclass(cls, GeneratorAdapter):
        raise TypeError(f"{cls} must be a GeneratorAdapter subclass")
    _ADAPTERS[name] = cls


def unregister_adapter(name: str) -> Optional[Type[GeneratorAdapter]]:
    """Remove adapter from registry. Returns removed class or None."""
    return _ADAPTERS.pop(name, None)
