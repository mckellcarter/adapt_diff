"""NVIDIA EDM/DMD2 compatibility - ensures vendored modules are importable."""

import sys
from pathlib import Path

_setup_done = False


def ensure_nvidia_modules():
    """Ensure torch_utils and dnnlib are importable for pickle loading.

    Adds the vendored modules to sys.path if not already available.
    This must be called before pickle.load() on EDM/DMD2 checkpoints.
    """
    global _setup_done

    # Already available?
    try:
        import torch_utils
        import dnnlib
        return True
    except ImportError:
        pass

    if _setup_done:
        return False

    # Add vendor directory to sys.path (this file is inside vendor/)
    vendor_dir = Path(__file__).parent
    vendor_str = str(vendor_dir)

    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
        _setup_done = True

    # Verify import works
    try:
        import torch_utils
        import dnnlib
        return True
    except ImportError as e:
        print(f"ERROR: Failed to import vendored modules: {e}")
        return False
