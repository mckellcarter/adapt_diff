# Vendored Dependencies

This directory contains vendored code from NVIDIA's EDM repository required for
loading EDM/DMD2 checkpoint files.

## Attribution

The `torch_utils/` and `dnnlib/` packages are derived from:

- **Repository**: https://github.com/NVlabs/edm
- **Authors**: NVIDIA CORPORATION & AFFILIATES
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

These files enable unpickling of EDM-format checkpoint files which embed source
code references to `torch_utils.persistence`.

## License

See LICENSE file in this directory for the full CC BY-NC-SA 4.0 license text.

## Modifications

The vendored code has been minimized to include only the essential components
needed for checkpoint loading:

- `torch_utils/persistence.py` - Pickle reconstruction for persistent classes
- `dnnlib/util.py` - EasyDict class (minimal subset)

Non-essential utilities, CUDA extensions, and training-specific code have been
removed to reduce dependencies.
