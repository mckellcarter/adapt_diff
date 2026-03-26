# Adapter Expansion Plan: Add Diffusion-Specific Methods

## Overview

Expand `GeneratorAdapter` base class to encapsulate diffusion sampling logic, making adapters self-contained and enabling `diffviews` to use a unified interface across models.

## Requirements Summary

User wants adapters to expose:
1. **Noise schedule** - timesteps/sigmas for sampling
2. **Stepping logic** - x_t → x_{t-1} denoising steps
3. **Latent conversion** - encode/decode between pixel and latent space
4. **Conditioning prep** - text/class → model-ready format
5. **Metadata** - prediction_type, uses_latent, training data pointers
6. **Config files** - JSON metadata with training data info

## Critical Files

- `/Users/mckell/Documents/GitHub/adapt_diff/adapt_diff/base.py` - Add abstract methods/properties
- `/Users/mckell/Documents/GitHub/adapt_diff/adapt_diff/adapters/edm_imagenet.py` - Extract from `sample()` method
- `/Users/mckell/Documents/GitHub/adapt_diff/adapt_diff/adapters/mscoco_t2i.py` - Rename encode/decode, add CFG
- `/Users/mckell/Documents/GitHub/adapt_diff/adapt_diff/adapters/abu_custom_sd.py` - Similar to MSCOCO
- `/Users/mckell/Documents/GitHub/adapt_diff/adapt_diff/adapters/dmd2_imagenet.py` - Similar to EDM

## Implementation Steps

### 1. Extend Base Class (`base.py`)

Add abstract methods to `GeneratorAdapter`:

```python
@abstractmethod
def get_timesteps(self, num_steps: int, device: str = 'cuda') -> torch.Tensor:
    """Return noise schedule (timesteps or sigmas) for num_steps."""
    pass

@abstractmethod
def step(self, x_t: torch.Tensor, t: torch.Tensor,
         model_output: torch.Tensor, **kwargs) -> torch.Tensor:
    """Single denoising step: x_t → x_{t-1}."""
    pass

@abstractmethod
def get_initial_noise(self, batch_size: int, device: str = 'cuda',
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Generate correctly shaped/scaled initial noise."""
    pass

@abstractmethod
def prepare_conditioning(self, text: Optional[str] = None,
                        class_label: Optional[int] = None,
                        batch_size: int = 1, device: str = 'cuda',
                        **kwargs) -> Any:
    """Prepare model-ready conditioning (text embeddings or class labels)."""
    pass

def encode(self, images: torch.Tensor) -> torch.Tensor:
    """Encode to internal representation (default: identity for pixel models)."""
    return images

def decode(self, representation: torch.Tensor) -> torch.Tensor:
    """Decode to pixel space (default: identity for pixel models)."""
    return representation

def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor,
                    cond: Any, uncond: Optional[Any] = None,
                    guidance_scale: float = 1.0, **kwargs) -> torch.Tensor:
    """Forward with CFG. Default: raises NotImplementedError (opt-in)."""
    if guidance_scale == 1.0 or uncond is None:
        # No CFG, just forward with conditioning
        return self.forward(x, t, **cond if isinstance(cond, dict) else {'class_labels': cond}, **kwargs)
    raise NotImplementedError(f"{self.__class__.__name__} does not support CFG")
```

Add abstract properties:

```python
@property
@abstractmethod
def prediction_type(self) -> str:
    """'epsilon', 'sample', or 'v_prediction'."""
    pass

@property
@abstractmethod
def uses_latent(self) -> bool:
    """True if latent-space model, False if pixel-space."""
    pass

@property
@abstractmethod
def in_channels(self) -> int:
    """Input channels (3 for pixel RGB, 4 for SD latent)."""
    pass

@property
@abstractmethod
def conditioning_type(self) -> str:
    """'class', 'text', or 'unconditional'."""
    pass

@property
def latent_scale_factor(self) -> int:
    """Spatial downsampling factor (8 for SD, 1 for pixel models)."""
    return 8 if self.uses_latent else 1
```

### 2. Update EDM Adapter (`edm_imagenet.py`)

Extract from existing `sample()` method:

```python
def get_timesteps(self, num_steps: int, device: str = 'cuda') -> torch.Tensor:
    """Karras sigma schedule."""
    rho, sigma_max, sigma_min = 7.0, 80.0, 0.002
    high_prec = torch.float64 if device != 'mps' else torch.float32
    step_indices = torch.arange(num_steps, dtype=high_prec, device=device)
    t_steps = (
        sigma_max ** (1/rho) +
        step_indices / (num_steps - 1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))
    ) ** rho
    return torch.cat([self._model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

def step(self, x_t: torch.Tensor, t_cur: torch.Tensor,
         model_output: torch.Tensor, t_next: torch.Tensor, **kwargs) -> torch.Tensor:
    """Euler step for EDM."""
    d_cur = (x_t - model_output) / t_cur
    return x_t + (t_next - t_cur) * d_cur

def get_initial_noise(self, batch_size: int, device: str = 'cuda',
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Generate (B, 3, 64, 64) noise scaled by sigma_max."""
    noise = torch.randn(batch_size, 3, self.resolution, self.resolution,
                       device=device, generator=generator)
    return noise * 80.0  # sigma_max

def prepare_conditioning(self, class_label: Optional[int] = None,
                        batch_size: int = 1, device: str = 'cuda', **kwargs) -> torch.Tensor:
    """Convert class label to one-hot."""
    if class_label is not None:
        labels = torch.full((batch_size,), class_label, device=device)
    else:
        labels = torch.randint(0, self.num_classes, (batch_size,), device=device)
    return torch.eye(self.num_classes, device=device)[labels]

# Properties
@property
def prediction_type(self) -> str:
    return 'sample'

@property
def uses_latent(self) -> bool:
    return False

@property
def in_channels(self) -> int:
    return 3

@property
def conditioning_type(self) -> str:
    return 'class'
```

Keep existing `sample()` method but refactor to use new methods internally.

### 3. Update DMD2 Adapter (`dmd2_imagenet.py`)

Similar to EDM but simpler schedule (1-10 steps):

```python
def get_timesteps(self, num_steps: int, device: str = 'cuda') -> torch.Tensor:
    """Simple linear schedule for distilled model."""
    return torch.linspace(80.0, 0.0, num_steps + 1, device=device)

# step(), get_initial_noise(), prepare_conditioning() same as EDM
# Properties same as EDM
```

### 4. Update MSCOCO Adapter (`mscoco_t2i.py`)

Rename encode/decode, add new methods:

```python
def encode(self, images: torch.Tensor) -> torch.Tensor:
    """Delegate to VAE encoding."""
    return self.encode_images(images)  # Keep old method for backward compat

def decode(self, representation: torch.Tensor) -> torch.Tensor:
    """Delegate to VAE decoding."""
    return self.decode_latents(representation)  # Keep old method

# Add deprecation warnings to encode_images() and decode_latents()

def get_timesteps(self, num_steps: int, device: str = 'cuda') -> torch.Tensor:
    """DDPM timesteps."""
    self._scheduler.set_timesteps(num_steps, device=device)
    return self._scheduler.timesteps

def step(self, x_t: torch.Tensor, t: torch.Tensor,
         model_output: torch.Tensor, **kwargs) -> torch.Tensor:
    """DDPM step via scheduler."""
    return self._scheduler.step(model_output, t, x_t, **kwargs).prev_sample

def get_initial_noise(self, batch_size: int, device: str = 'cuda',
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Generate (B, 4, 16, 16) latent noise."""
    return torch.randn(batch_size, 4, 16, 16, device=device, generator=generator)

def prepare_conditioning(self, text: Optional[str] = None, batch_size: int = 1,
                        device: str = 'cuda', **kwargs) -> Dict[str, torch.Tensor]:
    """Encode text with CLIP."""
    if text is None:
        raise ValueError("Text prompt required for MSCOCO T2I")
    # Load CLIP tokenizer/encoder, encode text, return embeddings
    # Return dict with 'encoder_hidden_states' key for forward()
    pass

def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, cond: Dict,
                    uncond: Optional[Dict] = None, guidance_scale: float = 7.5,
                    **kwargs) -> torch.Tensor:
    """CFG for text-to-image."""
    if guidance_scale == 1.0 or uncond is None:
        return self.forward(x, t, **cond, **kwargs)
    # Unconditional forward
    uncond_out = self.forward(x, t, **uncond, **kwargs)
    # Conditional forward
    cond_out = self.forward(x, t, **cond, **kwargs)
    # CFG: uncond + scale * (cond - uncond)
    return uncond_out + guidance_scale * (cond_out - uncond_out)

# Properties
@property
def prediction_type(self) -> str:
    return 'epsilon'

@property
def uses_latent(self) -> bool:
    return True

@property
def in_channels(self) -> int:
    return 4

@property
def conditioning_type(self) -> str:
    return 'text'
```

### 5. Update AbU Custom SD Adapter (`abu_custom_sd.py`)

Identical to MSCOCO but different dimensions (512x512, 64x64 latent).

### 6. Create Config Files

**Schema** (`checkpoints/<model>/config.json`):

```json
{
  "model_metadata": {
    "name": "abu-custom-sd14",
    "version": "1.0",
    "description": "AbU Custom Diffusion on SD v1.4",
    "citation": "Wang et al., AttributeByUnlearning, NeurIPS 2024",
    "license": "CC BY-NC-SA 4.0"
  },
  "architecture": {
    "model_type": "UNet2DConditionModel",
    "img_resolution": 512,
    "latent_resolution": 64,
    "in_channels": 4,
    "prediction_type": "epsilon",
    "uses_latent": true,
    "latent_scale_factor": 8
  },
  "conditioning": {
    "type": "text",
    "supports_cfg": true,
    "default_cfg_scale": 7.5
  },
  "scheduler": {
    "type": "DDPMScheduler",
    "num_train_timesteps": 1000
  },
  "training_data": {
    "dataset": "custom_diffusion_concepts",
    "base_model": "CompVis/stable-diffusion-v1-4",
    "hf_repo": "sywang/AttributeByUnlearning"
  }
}
```

Create configs for all 4 adapters in `checkpoints/<model_dir>/config.json`.

Update `get_default_config()` to include new metadata fields.

### 7. Update Config Loading

Modify `from_checkpoint()` in each adapter:

```python
@classmethod
def from_checkpoint(cls, checkpoint_path, device='cuda', **kwargs):
    # Try loading config from checkpoint dir
    config_path = Path(checkpoint_path).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = cls.get_default_config()

    # Use config for model construction...
```

## Implementation Order

1. **Base class** - Add abstract methods/properties
2. **EDM adapter** - Simplest, extract from existing `sample()`
3. **DMD2 adapter** - Similar to EDM
4. **MSCOCO adapter** - Add encode/decode delegation, CFG
5. **AbU Custom SD** - Follow MSCOCO pattern
6. **Config files** - Create JSONs, update loaders
7. **Tests** - Verify all methods work, test backward compat

## Key Design Decisions

- **Unified `get_timesteps()`** - Returns tensor (timesteps OR sigmas), caller doesn't need to know which
- **Abstract `step()`** - Each model handles prediction_type conversion internally
- **Optional CFG** - `forward_with_cfg()` raises NotImplementedError by default (opt-in)
- **Backward compat** - Keep old `encode_images()`/`decode_latents()` with deprecation warnings
- **Config location** - Alongside checkpoints, fallback to `get_default_config()`

## Testing Strategy

1. Test each adapter's new methods individually
2. Test backward compatibility (old encode_images still works)
3. Test unified sampling loop across all adapters:
   ```python
   timesteps = adapter.get_timesteps(50)
   x = adapter.get_initial_noise(1, device)
   cond = adapter.prepare_conditioning(text="cat")

   for t in timesteps[:-1]:
       pred = adapter.forward_with_cfg(x, t, cond, guidance_scale=7.5)
       x = adapter.step(x, t, pred, t_next=timesteps[i+1])

   images = adapter.decode(x)
   ```
4. Verify config loading from JSON files

## Success Criteria

- All 4 adapters implement new interface
- Backward compatibility maintained
- Config JSONs created with training data pointers
- Tests pass for all adapters
- `diffviews` can use unified interface across models
