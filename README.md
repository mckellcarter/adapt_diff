# adapt_diff

Model-agnostic adapter interface for diffusion models. Provides a common interface for forward pass, hook registration, and checkpoint loading across different diffusion architectures.

## Installation

```bash
pip install adapt_diff
```

Or from source:
```bash
git clone https://github.com/mckellcarter/adapt_diff
cd adapt_diff
pip install -e .
```

## Structure

```
adapt_diff/
├── LICENSE                    # MIT
├── README.md
├── pyproject.toml
├── adapt_diff/
│   ├── __init__.py           # Re-exports
│   ├── base.py               # GeneratorAdapter ABC
│   ├── hooks.py              # HookMixin utilities
│   ├── registry.py           # Registration + entry-point discovery
│   ├── adapters/
│   │   ├── dmd2_imagenet.py  # DMD2 ImageNet 64x64
│   │   ├── edm_imagenet.py   # EDM ImageNet 64x64
│   │   └── mscoco_t2i.py     # MSCOCO T2I 128x128
│   ├── scripts/
│   │   ├── cli.py            # CLI (download, list)
│   │   └── downloaders.py    # Model-specific download functions
│   └── vendor/               # Licenses and attributions
│       └── LICENSE
└── tests/
    ├── adapters/             # Adapter-specific tests
    │   └── test_mscoco_t2i.py
    ├── test_hooks.py
    └── test_registry.py
```

## Quick Start

```python
from adapt_diff import get_adapter, list_adapters

# List available adapters
print(list_adapters())  # ['dmd2-imagenet-64', 'edm-imagenet-64', 'mscoco-t2i-128']

# Load adapter from checkpoint
AdapterClass = get_adapter('edm-imagenet-64')
adapter = AdapterClass.from_checkpoint('path/to/edm.pkl', device='cuda')

# Forward pass
output = adapter.forward(noisy_input, sigma, class_labels)

# Get layer shapes
shapes = adapter.get_layer_shapes()
print(shapes)  # {'encoder_bottleneck': (512, 8, 8), ...}
```

## Activation Extraction

```python
# Register hooks for activation extraction
def extraction_hook(module, input, output):
    activations[name] = output.detach().cpu()

handles = adapter.register_activation_hooks(['encoder_bottleneck'], extraction_hook)

# Run forward pass
output = adapter.forward(x, sigma, class_labels)

# Clean up
for h in handles:
    h.remove()
```

## Creating a Custom Adapter

```python
from adapt_diff import GeneratorAdapter, HookMixin, register_adapter

@register_adapter('my-model')
class MyModelAdapter(HookMixin, GeneratorAdapter):
    def __init__(self, model, device='cuda'):
        HookMixin.__init__(self)
        self._model = model
        self._device = device

    @property
    def model_type(self): return 'my-model'

    @property
    def resolution(self): return 256

    @property
    def num_classes(self): return 1000

    @property
    def hookable_layers(self):
        return ['encoder', 'decoder', 'mid']

    def forward(self, x, sigma, class_labels=None, **kwargs):
        return self._model(x, sigma, class_labels)

    def register_activation_hooks(self, layer_names, hook_fn):
        handles = []
        for name in layer_names:
            module = self._get_module(name)
            h = module.register_forward_hook(hook_fn)
            handles.append(h)
            self.add_handle(h)
        return handles

    def get_layer_shapes(self):
        return {'encoder': (512, 16, 16), 'decoder': (256, 32, 32), 'mid': (512, 8, 8)}

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cuda', **kwargs):
        model = load_my_model(checkpoint_path)
        return cls(model.to(device).eval(), device)

    @classmethod
    def get_default_config(cls):
        return {'resolution': 256, 'channels': 3}
```

## Supported Models

| Model | Adapter Name | Resolution | Description |
|-------|--------------|------------|-------------|
| DMD2 | `dmd2-imagenet-64` | 64x64 | Distribution Matching Distillation (1-10 steps) |
| EDM | `edm-imagenet-64` | 64x64 | Elucidating Diffusion Models (50-256 steps) |
| MSCOCO T2I | `mscoco-t2i-128` | 128x128 | Text-to-image diffusion (1000 steps, latent space) |

## Checkpoints

Checkpoints are not included (too large). Use the CLI to download:

```bash
# Download all checkpoints
adapt_diff download

# Download specific model
adapt_diff download --models edm
adapt_diff download --models dmd2
adapt_diff download --models mscoco

# Custom output directory
adapt_diff download --output-dir ./my_checkpoints
```

Or download manually:
- **EDM**: https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/
- **DMD2**: https://huggingface.co/mckell/diffviews-dmd2-checkpoint
- **MSCOCO T2I**: https://huggingface.co/datasets/sywang/AttributeByUnlearning

**Note on DMD2 checkpoint**: The hosted DMD2 checkpoint has been fine-tuned from the original single-step model to support up to 10 diffusion steps, enabling trajectory visualization and intermediate step analysis.

**Note on MSCOCO T2I**: Requires `huggingface_hub` and `7z` for download/extraction. Model operates in latent space; VAE loaded automatically from HuggingFace.

**Security Note**: Some checkpoints are pickle (`.pkl`) files which can execute arbitrary code. Only load checkpoints from trusted sources.

## External Adapter Registration

External packages can register adapters via entry points:

```toml
# In your package's pyproject.toml
[project.entry-points."adapt_diff.adapters"]
my-model = "my_package.adapters:MyModelAdapter"
```

## License

| Component | License |
|-----------|---------|
| Core adapter framework | MIT |
| NVIDIA vendored code (`adapt_diff/vendor/`) | CC BY-NC-SA 4.0 |
| EDM checkpoints | CC BY-NC-SA 4.0 |
| DMD2 checkpoints | MIT |
| MSCOCO T2I checkpoints | CC BY-NC-SA 4.0 |
| MSCOCO dataset | CC BY 4.0 |

The code in `adapt_diff/vendor/` is derived from [NVIDIA's EDM repository](https://github.com/NVlabs/edm) and is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

See `adapt_diff/vendor/LICENSE` for full license details and citations.

## Acknowledgments

- NVIDIA for the EDM codebase and pretrained checkpoints
- The DMD2 team for the distillation work
- Wang et al. for [AttributeByUnlearning](https://github.com/PeterWang512/AttributeByUnlearning) (NeurIPS 2024)
- Lin et al. for the [MSCOCO dataset](https://cocodataset.org/)
- Ronneberger et al. for the [U-Net architecture](https://arxiv.org/abs/1505.04597)
