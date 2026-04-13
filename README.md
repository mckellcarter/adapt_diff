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
│   ├── generation.py         # High-level generate() API
│   ├── extraction.py         # ActivationExtractor + utilities
│   ├── adapters/
│   │   ├── abu_custom_sd.py    # AbU Custom SD 512x512
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
from adapt_diff import get_adapter, generate

# Load adapter from checkpoint
AdapterClass = get_adapter('dmd2-imagenet-64')
adapter = AdapterClass.from_checkpoint('path/to/dmd2.pkl', device='cuda')

# Generate images (high-level API)
result = generate(
    adapter=adapter,
    class_label=281,  # tabby cat
    num_steps=10,     # Use 10 steps to match training data
    num_samples=4,
    device='cuda',
    seed=42,
)
images = result.images  # (4, 64, 64, 3) uint8

# With trajectory extraction for attribution
result = generate(
    adapter=adapter,
    class_label=281,
    num_steps=10,
    extract_layers=['encoder_bottleneck'],
    return_trajectory=True,
    return_intermediates=True,
)
activations = result.trajectory  # list of (B, D) arrays per step
intermediates = result.intermediates  # list of images per step

# Direct sigma mode (for diffviews compatibility)
# Bypasses noise_level conversion, computes Karras schedule directly
result = generate(
    adapter=adapter,
    class_label=281,
    num_steps=10,
    sigma_max=80.0,   # Direct sigma values
    sigma_min=0.5,    # Must match training extraction sigma
    extract_layers=['encoder_bottleneck'],
    return_trajectory=True,
)
```

### Low-level API

```python
from adapt_diff import get_adapter, list_adapters

# List available adapters
print(list_adapters())  # ['abu-custom-sd14', 'dmd2-imagenet-64', ...]

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

Use `ActivationExtractor` for hook-based activation capture:

```python
from adapt_diff import get_adapter, ActivationExtractor

adapter = get_adapter('dmd2-imagenet-64').from_checkpoint('dmd2.pkl', device='cuda')

# Context manager handles hook registration/cleanup
with ActivationExtractor(adapter, ['encoder_bottleneck']) as extractor:
    adapter.forward(x_noisy, sigma, class_labels)
    activations = extractor.get_activations()  # Dict[str, Tensor]

# Or manual lifecycle
extractor = ActivationExtractor(adapter, ['encoder_bottleneck', 'decoder'])
extractor.register_hooks()
adapter.forward(x_noisy, sigma, class_labels)
activations = extractor.get_activations()
extractor.clear()  # Clear for next batch
# ... more forward passes ...
extractor.remove_hooks()
```

### Extraction Utilities

Save, load, and convert activation files:

```python
from adapt_diff import (
    flatten_activations,
    save_activations,
    load_activations,
    convert_to_fast_format,
    load_fast_activations,
)

# Flatten multi-layer activations to single vector
# Dict[str, (B, D_i)] -> (B, sum(D_i))
flat = flatten_activations(activations)

# Save to .npz with metadata
save_activations(activations, 'output/acts', metadata={'sigma': 0.5})

# Load back
activations, metadata = load_activations('output/acts')

# Convert to fast .npy format (~30x faster loading)
convert_to_fast_format('output/acts.npz', 'output/acts_fast.npy')

# Memory-mapped loading for large datasets
acts = load_fast_activations('output/acts_fast.npy', mmap_mode='r')
```

### Low-level Hook API

For custom hook logic:

```python
# Manual hook registration
def extraction_hook(module, input, output):
    activations[name] = output.detach().cpu()

handles = adapter.register_activation_hooks(['encoder_bottleneck'], extraction_hook)
output = adapter.forward(x, sigma, class_labels)
for h in handles:
    h.remove()
```

## Diffusion Sampling Interface

Adapters expose a unified interface for diffusion sampling. Noise levels use a universal 0-100 scale (100=pure noise, 0=clean) that each adapter translates to its native format.

```python
# Get sampling defaults from adapter
config = adapter.get_default_config()
num_steps = config.get('default_steps', 50)
noise_max = config.get('noise_max', 100.0)  # Starting noise level (0-100)
noise_min = config.get('noise_min', 0.0)    # Ending noise level (0-100)

# Generate noise schedule (returns native format: sigmas or timesteps)
timesteps = adapter.get_timesteps(
    num_steps=num_steps,
    device='cuda',
    noise_level_max=noise_max,
    noise_level_min=noise_min
)

# Initialize noise (scaled for starting noise level)
x = adapter.get_initial_noise(batch_size=4, device='cuda', noise_level=noise_max)

# Prepare conditioning
cond = adapter.prepare_conditioning(text="a cat", batch_size=4)       # text models
# or: cond = adapter.prepare_conditioning(class_label=281, batch_size=4)  # class models

# Sampling loop with classifier-free guidance
for i, t in enumerate(timesteps[:-1]):
    # CFG works for both text and class-conditional models
    pred = adapter.forward_with_cfg(x, t, cond, guidance_scale=7.5)

    # Optional: get x0 estimate for visualization (handles epsilon/sample/v_prediction)
    # x0_estimate = adapter.pred_to_sample(x, t, pred)

    # Single denoising step
    x = adapter.step(x, t, pred, t_next=timesteps[i+1])

# Decode to pixel space (latent models) or identity (pixel models)
images = adapter.decode(x)
```

### Sampling Defaults

Each adapter provides sensible defaults via `get_default_config()`. Noise levels use a universal 0-100 scale where 100=pure noise, 0=clean. Each adapter translates internally to native format (sigma for EDM/DMD2, timesteps for DDPM).

| Adapter | `default_steps` | `noise_max` | `noise_min` | Notes |
|---------|----------------|-------------|-------------|-------|
| EDM | 50 | 100.0 | 0.0 | Karras schedule, `rho=7.0` |
| DMD2 | 10 | 100.0 | 0.5 | Karras schedule, ancestral sampling |
| MSCOCO T2I | 20 | 100.0 | 0.0 | DDPM timesteps, `guidance_scale=7.5` |
| AbU Custom SD | 50 | 100.0 | 0.0 | SD v1.4 latent space, `guidance_scale=7.5` |

**Note on noise_level scale**: The 0-100 noise_level maps to native sigma via log interpolation. For attribution at a specific sigma (e.g., σ=0.5), use `adapter.native_to_noise_level(sigma)` to get the corresponding noise_level (~52.1 for DMD2).

**Note on num_steps for attribution**: When extracting activations for attribution, `num_steps` must match the number of steps used to build the training index. The activation at a given sigma depends on the entire denoising trajectory, not just the final sigma value. For DMD2 with yodal training data, use `num_steps=10`.

**Direct sigma mode**: For exact control over the sigma schedule (e.g., to match diffviews), use `sigma_max` and `sigma_min` parameters instead of `noise_level_max/min`. This bypasses noise_level conversion and computes the Karras schedule directly from sigma values.

### Adapter Properties

All adapters expose metadata about their architecture:

```python
adapter.prediction_type      # 'epsilon', 'sample', or 'v_prediction'
adapter.uses_latent          # True for latent-space models (SD), False for pixel-space
adapter.in_channels          # 3 (RGB) or 4 (SD latent)
adapter.conditioning_type    # 'text', 'class', or 'unconditional'
adapter.latent_scale_factor  # 8 for SD (512→64), 1 for pixel models
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

    # ============ Basic Properties ============
    @property
    def model_type(self): return 'my-model'

    @property
    def resolution(self): return 256

    @property
    def num_classes(self): return 1000

    @property
    def hookable_layers(self):
        return ['encoder', 'decoder', 'mid']

    # ============ Diffusion Properties ============
    @property
    def prediction_type(self): return 'epsilon'  # or 'sample', 'v_prediction'

    @property
    def uses_latent(self): return False  # True if latent-space model

    @property
    def in_channels(self): return 3  # or 4 for latent models

    @property
    def conditioning_type(self): return 'class'  # or 'text', 'unconditional'

    # ============ Core Methods ============
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

    # ============ Noise Level Translation ============
    def noise_level_to_native(self, noise_level):
        """Convert universal noise_level (0-100) to model's native format.

        Args:
            noise_level: 0-100 scale (0=clean, 100=pure noise)
        Returns:
            Native format (sigma for EDM/DMD2, timestep for DDPM)
        """
        # For sigma models (EDM/DMD2): log-space interpolation
        t = noise_level / 100.0
        return self.SIGMA_MIN ** (1 - t) * self.SIGMA_MAX ** t
        # For timestep models (DDPM): linear mapping
        # return (noise_level / 100.0 * 999).long()

    # ============ Diffusion Methods ============
    def get_timesteps(self, num_steps, device='cuda', noise_level_max=100.0, noise_level_min=0.0, **kwargs):
        """Return noise schedule in native format.

        Args:
            num_steps: Number of denoising steps
            noise_level_max: Starting noise (0-100), default 100
            noise_level_min: Ending noise (0-100), default 0
        Returns:
            Schedule tensor in native format (sigma or timesteps)
        """
        # For timestep models (DDPM):
        self._scheduler.set_timesteps(num_steps, device=device)
        return self._scheduler.timesteps
        # For sigma models (EDM): use noise_level_to_native for conversion

    def step(self, x_t, t, model_output, **kwargs):
        """Single denoising step: x_t → x_{t-1}."""
        # For timestep models:
        return self._scheduler.step(model_output, t, x_t, **kwargs).prev_sample
        # For sigma models with t_next in kwargs:
        # d = (x_t - model_output) / t
        # return x_t + (kwargs['t_next'] - t) * d

    def get_initial_noise(self, batch_size, device='cuda', generator=None, noise_level=100.0):
        """Generate initial noise with correct shape and scaling.

        Args:
            noise_level: Starting noise (0-100), default 100 (pure noise)
        """
        noise = torch.randn(batch_size, self.in_channels,
                           self.resolution, self.resolution,
                           device=device, generator=generator)
        # Scale by native sigma if needed (EDM/DMD2)
        # sigma = self.noise_level_to_native(torch.tensor(noise_level))
        # return noise * sigma
        return noise

    def pred_to_sample(self, x_t, t, model_output):
        """Convert model output to estimated clean sample (x0).

        Override for epsilon-prediction models:
            alpha_t = scheduler.alphas_cumprod[t]
            x0 = (x_t - sqrt(1-alpha_t) * model_output) / sqrt(alpha_t)
        Default returns model_output unchanged (for sample-prediction).
        """
        return model_output

    def prepare_conditioning(self, text=None, class_label=None,
                           batch_size=1, device='cuda', **kwargs):
        """Prepare model-ready conditioning."""
        # For class-conditional:
        if class_label is not None:
            labels = torch.full((batch_size,), class_label, device=device)
        else:
            labels = torch.randint(0, self.num_classes, (batch_size,), device=device)
        return torch.eye(self.num_classes, device=device)[labels]
        # For text-conditional: encode text with CLIP and return embeddings dict

    # Optional: Override for latent-space models
    def encode(self, images):
        """Encode images to latent space (override for VAE models)."""
        return self._vae.encode(images).latent_dist.sample() * 0.18215

    def decode(self, latents):
        """Decode latents to pixel space (override for VAE models)."""
        return self._vae.decode(latents / 0.18215).sample

    # Optional: Override for CFG-capable models
    def forward_with_cfg(self, x, t, cond, uncond=None, guidance_scale=1.0, **kwargs):
        """Classifier-free guidance (override for text models)."""
        if guidance_scale == 1.0 or uncond is None:
            return self.forward(x, t, **cond, **kwargs)
        uncond_out = self.forward(x, t, **uncond, **kwargs)
        cond_out = self.forward(x, t, **cond, **kwargs)
        return uncond_out + guidance_scale * (cond_out - uncond_out)

    # ============ Class Methods ============
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cuda', **kwargs):
        model = load_my_model(checkpoint_path)
        return cls(model.to(device).eval(), device)

    @classmethod
    def get_default_config(cls):
        return {
            'img_resolution': 256,
            'in_channels': 3,
            'prediction_type': 'epsilon',
            'uses_latent': False
        }
```

## Supported Models

| Model | Adapter Name | Resolution | Conditioning | CFG | Description |
|-------|--------------|------------|--------------|-----|-------------|
| AbU Custom SD | `abu-custom-sd14` | 512x512 | text | ✓ | AttributeByUnlearning SD v1.4 |
| DMD2 | `dmd2-imagenet-64` | 64x64 | class | ✓ | Distribution Matching Distillation (1-10 steps) |
| EDM | `edm-imagenet-64` | 64x64 | class | ✓ | Elucidating Diffusion Models (50-256 steps) |
| MSCOCO T2I | `mscoco-t2i-128` | 128x128 | text | ✓ | Text-to-image diffusion (latent space) |

## Checkpoints

Checkpoints are not included (too large). Use the CLI to download:

```bash
# Download all checkpoints
adapt_diff download

# Download specific model
adapt_diff download --models edm
adapt_diff download --models dmd2
adapt_diff download --models mscoco
adapt_diff download --models abu_custom_sd

# Custom output directory
adapt_diff download --output-dir ./my_checkpoints
```

Or download manually:
- **EDM**: https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/
- **DMD2**: https://huggingface.co/mckell/diffviews-dmd2-checkpoint
- **MSCOCO T2I**: https://huggingface.co/datasets/sywang/AttributeByUnlearning
- **AbU Custom SD**: Base SD v1.4 auto-downloads via diffusers; AbC benchmark from above

**Note on DMD2 checkpoint**: The hosted DMD2 checkpoint has been fine-tuned from the original single-step model to support up to 10 diffusion steps, enabling trajectory visualization and intermediate step analysis.

**Note on MSCOCO T2I / AbU Custom SD**: Requires `huggingface_hub` for download. Models operate in latent space; VAE loaded automatically from HuggingFace.

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
| AbU Custom SD checkpoints | CC BY-NC-SA 4.0 |
| MSCOCO dataset | CC BY 4.0 |

The code in `adapt_diff/vendor/` is derived from [NVIDIA's EDM repository](https://github.com/NVlabs/edm) and is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

See `adapt_diff/vendor/LICENSE` for full license details and citations.

## Acknowledgments

- NVIDIA for the EDM codebase and pretrained checkpoints
- The DMD2 team for the distillation work
- Wang et al. for [AttributeByUnlearning](https://github.com/PeterWang512/AttributeByUnlearning) (NeurIPS 2024)
- Lin et al. for the [MSCOCO dataset](https://cocodataset.org/)
- Ronneberger et al. for the [U-Net architecture](https://arxiv.org/abs/1505.04597)
