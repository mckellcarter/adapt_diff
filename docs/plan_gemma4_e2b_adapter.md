# Gemma 4 E2B Adapter Plan

## Overview

Add Gemma 4 E2B (2B param multimodal LLM) support to adapt_diff. First text-output model using unified abstraction: **token position as time**.

## Core Insight: Token Step = Time Step

Autoregressive generation maps cleanly to `GeneratorAdapter` by treating token position as timestep:

| Concept | Diffusion | Autoregressive |
|---------|-----------|----------------|
| timestep/t | noise level T→0 | token position 0→N |
| step() | denoise x_t → x_{t-1} | generate token at position t |
| forward() | predict noise/sample | predict logits for position t |
| get_timesteps() | noise schedule | token positions to generate |
| get_initial_noise() | random tensor | tokenized prompt (seed) |
| noise_level | 0=clean, 100=noisy | 0=prompt start, 100=max_tokens |
| output | image | token sequence |

## Critical Files

- `/Users/mckell/Documents/GitHub/adapt_diff/adapt_diff/base.py` - Add `output_type`, `generation_mode` properties
- `/Users/mckell/Documents/GitHub/adapt_diff/adapt_diff/generation.py` - Branch on generation_mode
- `/Users/mckell/Documents/GitHub/adapt_diff/adapt_diff/device.py` - Add `get_backend()` for MLX
- `/Users/mckell/Documents/GitHub/adapt_diff/adapt_diff/adapters/gemma4_e2b.py` - NEW adapter

## Implementation Steps

### 1. Extend Base Class (`base.py`) ✅ DONE

> Committed in `c1e8d2f` on `feature/gemma4-e2b-adapter`

Add optional properties with defaults (existing adapters unchanged):

```python
@property
def output_type(self) -> str:
    """Output modality: 'image' (default) or 'text'."""
    return 'image'

@property
def generation_mode(self) -> str:
    """Generation paradigm: 'diffusion' (default) or 'autoregressive'."""
    return 'diffusion'
```

### 2. Create Gemma 4 E2B Adapter (`adapters/gemma4_e2b.py`) ✅ DONE

> Committed in `c1e8d2f` on `feature/gemma4-e2b-adapter`

```python
@register_adapter('gemma4-e2b')
class Gemma4E2BAdapter(HookMixin, GeneratorAdapter):
    """
    Adapter for Gemma 4 E2B (2B params, multimodal).

    Supports: text + image + audio input → text output
    Backends: torch (cuda/cpu) or MLX (mps)
    """

    def __init__(self, model, tokenizer, device='cuda', backend='torch',
                 vision_encoder=None, audio_encoder=None, text_embedder=None):
        HookMixin.__init__(self)
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._backend = backend
        self._vision_encoder = vision_encoder
        self._audio_encoder = audio_encoder
        self._text_embedder = text_embedder  # EmbeddingGemma
        self._current_prompt = None
        self._temperature = 1.0
        self._top_p = 1.0

    # === Properties ===

    @property
    def model_type(self) -> str:
        return 'gemma4-e2b'

    @property
    def output_type(self) -> str:
        return 'text'

    @property
    def generation_mode(self) -> str:
        return 'autoregressive'

    @property
    def resolution(self) -> int:
        """Context length (tokens) - analogous to image resolution."""
        return 128000  # 128K context

    @property
    def num_classes(self) -> int:
        """Vocab size - analogous to num_classes."""
        return 256000

    @property
    def prediction_type(self) -> str:
        return 'logits'  # Model predicts token logits

    @property
    def uses_latent(self) -> bool:
        return False  # Direct token space

    @property
    def in_channels(self) -> int:
        return 1  # Token sequence (1D)

    @property
    def conditioning_type(self) -> str:
        return 'multimodal'  # text + image + audio

    @property
    def hookable_layers(self) -> List[str]:
        layers = []
        for i in range(35):  # 35 decoder layers
            layers.extend([f'layer_{i}', f'layer_{i}_attn', f'layer_{i}_mlp'])
        layers.append('final_norm')
        return layers

    # === Core Interface (mapped from GeneratorAdapter) ===

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                class_labels=None, **kwargs) -> torch.Tensor:
        """
        Forward pass returning logits at current position.

        Args:
            x: Token sequence (B, seq_len)
            t: Current generation step (position) - unused for autoregressive
            **kwargs: image_embeds, audio_embeds, attention_mask

        Returns:
            Logits for next token (B, vocab_size)
        """
        outputs = self._model(
            input_ids=x,
            image_embeds=kwargs.get('image_embeds'),
            audio_embeds=kwargs.get('audio_embeds'),
            attention_mask=kwargs.get('attention_mask'),
        )
        return outputs.logits[:, -1, :]  # Last position logits

    def step(self, x_t: torch.Tensor, t: torch.Tensor,
             model_output: torch.Tensor, t_next=None,
             step_noise=None, **kwargs) -> torch.Tensor:
        """
        Sample next token and append to sequence.

        Args:
            x_t: Current token sequence (B, seq_len)
            t: Current step (unused)
            model_output: Logits from forward() (B, vocab_size)
            **kwargs: temperature, top_p, top_k

        Returns:
            Extended sequence (B, seq_len + 1)
        """
        temperature = kwargs.get('temperature', self._temperature)
        top_p = kwargs.get('top_p', self._top_p)

        # Sample from logits
        next_token = self._sample_token(model_output, temperature, top_p)

        # Append to sequence
        return torch.cat([x_t, next_token.unsqueeze(-1)], dim=-1)

    def _sample_token(self, logits: torch.Tensor, temperature: float,
                      top_p: float) -> torch.Tensor:
        """Sample token from logits with temperature and nucleus sampling."""
        if temperature > 0:
            logits = logits / temperature

        if top_p < 1.0:
            # Nucleus sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            mask = cumsum - probs > top_p
            sorted_logits[mask] = float('-inf')
            logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def get_timesteps(self, num_steps: int, device: str = 'cuda',
                      noise_level_max: float = 100.0,
                      noise_level_min: float = 0.0, **kwargs) -> torch.Tensor:
        """
        Return token positions to generate.

        Args:
            num_steps: Max new tokens to generate

        Returns:
            Range tensor (0, 1, 2, ..., num_steps-1)
        """
        return torch.arange(num_steps, device=device)

    def get_initial_noise(self, batch_size: int, device: str = 'cuda',
                          generator: Optional[torch.Generator] = None,
                          noise_level: float = 100.0) -> torch.Tensor:
        """
        Return tokenized prompt as "initial state".

        Note: Must call prepare_conditioning() first to set prompt.
        """
        if self._current_prompt is None:
            raise ValueError("Call prepare_conditioning(text=...) first")

        tokens = self._tokenizer(
            self._current_prompt,
            return_tensors='pt',
            padding=True
        ).input_ids.to(device)

        if batch_size > 1:
            tokens = tokens.repeat(batch_size, 1)

        return tokens

    def noise_level_to_native(self, noise_level: torch.Tensor) -> torch.Tensor:
        """Map noise_level (0-100) to token position."""
        # noise_level 0 = start of generation
        # noise_level 100 = max_tokens reached
        max_new = 256  # Default max new tokens
        return (noise_level / 100.0 * max_new).long()

    def native_to_noise_level(self, native: torch.Tensor) -> torch.Tensor:
        """Map token position to noise_level (0-100)."""
        max_new = 256
        return native.float() / max_new * 100.0

    def prepare_conditioning(self, text: Optional[str] = None,
                             class_label: Optional[int] = None,
                             batch_size: int = 1, device: str = 'cuda',
                             images: Optional[List] = None,
                             audio: Optional[torch.Tensor] = None,
                             **kwargs) -> Dict[str, Any]:
        """
        Prepare multimodal inputs.

        Args:
            text: Text prompt (required)
            images: List of image tensors
            audio: Audio waveform tensor

        Returns:
            Dict with input_ids, image_embeds, audio_embeds
        """
        if text is None:
            raise ValueError("Text prompt required")

        self._current_prompt = text
        cond = {}

        # Encode images if provided
        if images and self._vision_encoder:
            embeds = []
            for img in images:
                embeds.append(self._vision_encoder(img.to(device)))
            cond['image_embeds'] = torch.cat(embeds, dim=1)

        # Encode audio if provided
        if audio is not None and self._audio_encoder:
            cond['audio_embeds'] = self._audio_encoder(audio.to(device))

        return cond

    def get_text_embedding(self, text: str, dim: int = 768) -> torch.Tensor:
        """
        Get text embedding via EmbeddingGemma.

        Args:
            text: Input text
            dim: Output dimension (128, 256, 384, 512, or 768)

        Returns:
            Embedding tensor (1, dim)
        """
        if self._text_embedder is None:
            raise ValueError("EmbeddingGemma not loaded")
        return self._text_embedder.encode(text, output_dim=dim)

    # === Hooks ===

    def _get_layer_module(self, layer_name: str):
        """Get PyTorch module by layer name."""
        if layer_name == 'final_norm':
            return self._model.model.norm

        parts = layer_name.split('_')
        idx = int(parts[1])
        layer = self._model.model.layers[idx]

        if len(parts) == 2:
            return layer
        elif parts[2] == 'attn':
            return layer.self_attn
        elif parts[2] == 'mlp':
            return layer.mlp

    def register_activation_hooks(self, layer_names: List[str],
                                   hook_fn: callable) -> List:
        handles = []
        for name in layer_names:
            module = self._get_layer_module(name)
            handle = module.register_forward_hook(hook_fn)
            handles.append(handle)
            self.add_handle(handle)
        return handles

    def get_layer_shapes(self) -> Dict[str, Tuple[int, ...]]:
        # Run dummy forward to capture shapes
        if hasattr(self, '_layer_shapes'):
            return self._layer_shapes
        # ... shape detection logic
        return {}

    # === Checkpoint Loading ===

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = 'cuda',
                        backend: str = 'auto',
                        load_vision: bool = True,
                        load_audio: bool = True,
                        embedding_model: str = 'google/embeddinggemma-300m',
                        **kwargs) -> 'Gemma4E2BAdapter':
        """
        Load Gemma 4 E2B with optional embedders.

        Args:
            checkpoint_path: HF repo ID or local path
            device: 'cuda', 'mps', 'cpu'
            backend: 'torch', 'mlx', or 'auto' (MLX on mps)
            load_vision: Load 150M ViT vision encoder
            load_audio: Load Conformer audio encoder
            embedding_model: EmbeddingGemma model ID
        """
        if backend == 'auto':
            backend = 'mlx' if device == 'mps' else 'torch'

        if backend == 'mlx':
            return cls._load_mlx(checkpoint_path, **kwargs)
        else:
            return cls._load_torch(
                checkpoint_path, device, load_vision, load_audio,
                embedding_model, **kwargs
            )

    @classmethod
    def _load_torch(cls, path, device, load_vision, load_audio, embed_model, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading Gemma 4 E2B (torch) from {path}...")
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        model.eval()

        vision_encoder = None
        if load_vision:
            # Load ViT 150M (internal to Gemma 4)
            vision_encoder = cls._load_vision_encoder(model, device)

        audio_encoder = None
        if load_audio:
            # Load Conformer audio encoder
            audio_encoder = cls._load_audio_encoder(model, device)

        text_embedder = None
        if embed_model:
            text_embedder = cls._load_embedding_gemma(embed_model, device)

        return cls(model, tokenizer, device, 'torch',
                   vision_encoder, audio_encoder, text_embedder)

    @classmethod
    def _load_mlx(cls, path, **kwargs):
        from mlx_lm import load

        print(f"Loading Gemma 4 E2B (MLX) from {path}...")
        model, tokenizer = load(path)

        return cls(model, tokenizer, 'mps', 'mlx')

    @classmethod
    def _load_embedding_gemma(cls, model_id: str, device: str):
        """Load EmbeddingGemma 308M for text embeddings."""
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading EmbeddingGemma from {model_id}...")

        class EmbeddingWrapper:
            def __init__(self, model, tokenizer, device):
                self.model = model.to(device)
                self.tokenizer = tokenizer
                self.device = device
                self.model.eval()

            def encode(self, text: str, output_dim: int = 768) -> torch.Tensor:
                tokens = self.tokenizer(
                    text, return_tensors='pt', truncation=True, max_length=2048
                ).to(self.device)
                with torch.no_grad():
                    out = self.model(**tokens)
                    emb = out.last_hidden_state.mean(dim=1)
                return emb[:, :output_dim]  # MRL truncation

        model = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return EmbeddingWrapper(model, tokenizer, device)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "model_type": "gemma4-e2b",
            "context_length": 128000,
            "vocab_size": 256000,
            "hidden_dim": 2304,
            "num_layers": 35,
            "input_modalities": ["text", "image", "audio"],
            "output_modality": "text",
            "generation_mode": "autoregressive",
            "default_max_tokens": 256,
            "default_temperature": 1.0,
            "default_top_p": 0.95,
        }
```

### 3. Extend Generation Module (`generation.py`)

Add autoregressive branch and result type:

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch

@dataclass
class TextGenerationResult:
    """Result from autoregressive text generation."""
    tokens: torch.Tensor  # (B, seq_len) token IDs
    text: List[str]  # Decoded strings
    trajectory: Optional[List[np.ndarray]] = None  # Per-token activations
    token_probs: Optional[torch.Tensor] = None  # Per-token probabilities


@torch.no_grad()
def generate(
    adapter: GeneratorAdapter,
    # ... existing params ...
) -> Union[GenerationResult, TextGenerationResult]:
    """
    Generate samples using adapter.

    Automatically detects generation_mode and branches:
    - 'diffusion': iterative denoising (existing logic)
    - 'autoregressive': token-by-token generation
    """
    if adapter.generation_mode == 'autoregressive':
        return _generate_autoregressive(adapter, ...)
    else:
        return _generate_diffusion(adapter, ...)  # Existing logic


def _generate_autoregressive(
    adapter: GeneratorAdapter,
    num_steps: int = 256,  # max_new_tokens
    seed: Optional[int] = None,
    extract_layers: Optional[List[str]] = None,
    return_trajectory: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
    **kwargs
) -> TextGenerationResult:
    """Token-by-token generation with inline activation extraction."""

    device = kwargs.get('device', 'cuda')
    if seed is not None:
        torch.manual_seed(seed)

    # Get initial sequence (tokenized prompt)
    x = adapter.get_initial_noise(1, device)

    # Get generation steps
    timesteps = adapter.get_timesteps(num_steps, device)

    # Setup hooks for trajectory extraction
    trajectory = []
    if return_trajectory and extract_layers:
        extractor = ActivationExtractor(adapter, extract_layers)
        extractor.register_hooks()

    try:
        for t in timesteps:
            # Forward pass
            logits = adapter.forward(x, t, **kwargs)

            # Capture activations
            if return_trajectory and extract_layers:
                acts = extractor.get_activations()
                trajectory.append(_flatten_acts(acts))
                extractor.clear()

            # Sample next token and extend sequence
            x = adapter.step(x, t, logits, temperature=temperature, top_p=top_p)

            # Check for EOS
            if x[0, -1].item() == adapter._tokenizer.eos_token_id:
                break
    finally:
        if return_trajectory and extract_layers:
            extractor.remove_hooks()

    # Decode tokens to text
    text = adapter._tokenizer.batch_decode(x, skip_special_tokens=True)

    return TextGenerationResult(
        tokens=x,
        text=text,
        trajectory=trajectory if return_trajectory else None
    )
```

### 4. Extend Device Module (`device.py`)

```python
def get_backend(device: str = None, prefer_backend: str = 'auto') -> str:
    """
    Select optimal backend for device.

    Args:
        device: Target device (cuda, mps, cpu)
        prefer_backend: 'torch', 'mlx', or 'auto'

    Returns:
        'torch' or 'mlx'
    """
    if prefer_backend != 'auto':
        return prefer_backend

    device = device or get_device()

    if device == 'mps' and supports_mlx():
        return 'mlx'

    return 'torch'


def supports_mlx() -> bool:
    """Check if MLX is available."""
    try:
        import mlx
        return True
    except ImportError:
        return False
```

## Key Design Decisions

1. **Unified GeneratorAdapter** - No new base class. Add `output_type` and `generation_mode` properties with backward-compatible defaults.

2. **Token position as time** - Autoregressive generation maps cleanly:
   - `get_timesteps()` → range(max_new_tokens)
   - `step()` → sample + append token
   - `forward()` → predict next token logits
   - `get_initial_noise()` → tokenized prompt

3. **Embedders bundled** - Vision (ViT 150M), audio (Conformer), text (EmbeddingGemma 308M) loaded inside adapter via `from_checkpoint()`, consistent with MSCOCO pattern.

4. **MLX for LLM only** - Vision/audio encoders stay on torch (works on MPS). MLX backend only for main Gemma model.

5. **Inline activation hooks** - Capture activations during generation loop, single pass.

## Testing Strategy

1. Unit tests with mock model (no GPU required)
2. Verify existing diffusion adapter tests still pass
3. Test torch backend on CUDA
4. Test MLX backend on Apple Silicon (if available)
5. Test multimodal inputs (text + image, text + audio)
6. Run pylint before commit

## Success Criteria

- Gemma 4 E2B adapter implements full `GeneratorAdapter` interface
- Text generation via `generate()` works
- Activation extraction during generation works
- Torch and MLX backends functional
- Backward compatibility: all existing diffusion adapters unchanged
- Tests pass

## Dependencies

```toml
[project.optional-dependencies]
gemma = ["transformers>=4.40.0", "sentencepiece", "accelerate", "torchaudio"]
mlx = ["mlx>=0.15.0", "mlx-lm>=0.15.0"]
```

## References

- [Gemma 4 Overview - Google AI](https://ai.google.dev/gemma/docs/core)
- [EmbeddingGemma - Google AI](https://ai.google.dev/gemma/docs/embeddinggemma)
- [HuggingFace: google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)

---

## Progress

- [x] Phase 1: Extend `base.py` with `output_type`/`generation_mode` properties
- [x] Phase 1: Create `adapters/gemma4_e2b.py` adapter
- [ ] Phase 2: Extend `generation.py` with autoregressive branch + `TextGenerationResult`
- [ ] Phase 2: Extend `device.py` with `get_backend()` and `supports_mlx()`
- [ ] Phase 3: Add unit tests for Gemma4E2BAdapter
- [ ] Phase 3: Verify existing diffusion tests still pass

---

## Continuation Prompt

```
Continue Gemma 4 E2B adapter implementation per docs/plan_gemma4_e2b_adapter.md. Phase 1 complete. Now implement Phase 2: extend generation.py with TextGenerationResult dataclass and _generate_autoregressive() function that branches on adapter.generation_mode. Also add get_backend() and supports_mlx() to device.py.
```
