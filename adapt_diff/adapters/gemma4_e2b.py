"""Gemma 4 E2B adapter implementing the GeneratorAdapter interface.

Gemma 4 E2B is a 2B parameter multimodal LLM supporting text, image, and audio input
with text output. This adapter unifies autoregressive generation under the same
abstraction as diffusion models by treating token position as timestep.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from adapt_diff.base import GeneratorAdapter
from adapt_diff.hooks import HookMixin
from adapt_diff.registry import register_adapter


@register_adapter('gemma4-e2b')
class Gemma4E2BAdapter(HookMixin, GeneratorAdapter):
    """
    Adapter for Gemma 4 E2B (2B params, multimodal).

    Supports: text + image + audio input → text output
    Backends: torch (cuda/cpu) or MLX (mps)

    Token position as time:
        - get_timesteps() → token positions to generate
        - step() → sample + append next token
        - forward() → predict next token logits
        - get_initial_noise() → tokenized prompt
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda',
        backend: str = 'torch',
        vision_encoder=None,
        audio_encoder=None
    ):
        HookMixin.__init__(self)
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._backend = backend
        self._vision_encoder = vision_encoder
        self._audio_encoder = audio_encoder
        self._current_prompt = None
        self._temperature = 1.0
        self._top_p = 1.0
        self._layer_shapes: Optional[Dict[str, Tuple[int, ...]]] = None

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
        # Gemma 2 uses 256k vocab, actual size may vary by model
        if hasattr(self, '_tokenizer') and hasattr(self._tokenizer, 'vocab_size'):
            return self._tokenizer.vocab_size
        return 256000

    @property
    def prediction_type(self) -> str:
        return 'logits'

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
    def timestep_label(self) -> str:
        return "pos"  # Token position

    @property
    def hookable_layers(self) -> List[str]:
        """Return list of available layer names for hooks."""
        layers = []
        for i in range(35):  # 35 decoder layers
            layers.extend([f'layer_{i}', f'layer_{i}_attn', f'layer_{i}_mlp'])
        layers.append('final_norm')
        return layers

    # === Core Interface ===

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass returning logits at current position.

        Args:
            x: Token sequence (B, seq_len)
            t: Current generation step - unused for autoregressive
            class_labels: Unused
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

    def step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
        step_noise: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample next token and append to sequence.

        Args:
            x_t: Current token sequence (B, seq_len)
            t: Current step (unused)
            model_output: Logits from forward() (B, vocab_size)
            t_next: Unused
            step_noise: Unused
            **kwargs: temperature, top_p, top_k

        Returns:
            Extended sequence (B, seq_len + 1)
        """
        temperature = kwargs.get('temperature', self._temperature)
        top_p = kwargs.get('top_p', self._top_p)

        next_token = self._sample_token(model_output, temperature, top_p)
        return torch.cat([x_t, next_token.unsqueeze(-1)], dim=-1)

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float
    ) -> torch.Tensor:
        """Sample token from logits with temperature and nucleus sampling."""
        if temperature > 0:
            logits = logits / temperature

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            mask = cumsum - probs > top_p
            sorted_logits[mask] = float('-inf')
            # Scatter back to original order
            logits = torch.zeros_like(logits).scatter(-1, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def get_timesteps(
        self,
        num_steps: int,
        device: str = 'cuda',
        noise_level_max: float = 100.0,
        noise_level_min: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Return token positions to generate.

        Args:
            num_steps: Max new tokens to generate

        Returns:
            Range tensor (0, 1, 2, ..., num_steps-1)
        """
        return torch.arange(num_steps, device=device)

    def get_initial_noise(
        self,
        batch_size: int,
        device: str = 'cuda',
        generator: Optional[torch.Generator] = None,
        noise_level: float = 100.0
    ) -> torch.Tensor:
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

    def noise_level_to_native(
        self,
        noise_level: torch.Tensor
    ) -> torch.Tensor:
        """
        Map noise_level (0-100) to token position.

        For autoregressive:
        - noise_level 0 = start of generation
        - noise_level 100 = max_tokens reached
        """
        max_new = 256  # Default max new tokens
        return (noise_level / 100.0 * max_new).long()

    def native_to_noise_level(
        self,
        native: torch.Tensor
    ) -> torch.Tensor:
        """Map token position to noise_level (0-100)."""
        max_new = 256
        return native.float() / max_new * 100.0

    def prepare_conditioning(
        self,
        text: Optional[str] = None,
        class_label: Optional[int] = None,
        batch_size: int = 1,
        device: str = 'cuda',
        images: Optional[List] = None,
        audio: Optional[torch.Tensor] = None,
        apply_chat_template: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare multimodal inputs.

        Args:
            text: Text prompt (required)
            class_label: Unused
            batch_size: Number of samples
            device: Target device
            images: List of image tensors
            audio: Audio waveform tensor
            apply_chat_template: Apply tokenizer's chat template (default True)

        Returns:
            Dict with image_embeds, audio_embeds if provided

        Raises:
            ValueError: If text is None
        """
        if text is None:
            raise ValueError("Text prompt required")

        # Apply chat template if available and requested
        if apply_chat_template and hasattr(self._tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": text}]
            self._current_prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
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
        if parts[2] == 'attn':
            return layer.self_attn
        if parts[2] == 'mlp':
            return layer.mlp
        raise ValueError(f"Unknown layer: {layer_name}")

    def register_activation_hooks(
        self,
        layer_names: List[str],
        hook_fn: callable
    ) -> List[torch.utils.hooks.RemovableHandle]:
        """Register forward hooks on specified layers."""
        handles = []
        for name in layer_names:
            module = self._get_layer_module(name)
            handle = module.register_forward_hook(hook_fn)
            handles.append(handle)
            self.add_handle(handle)
        return handles

    def get_layer_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Return activation shapes for hookable layers."""
        if self._layer_shapes is not None:
            return self._layer_shapes

        self._layer_shapes = {}
        # Shape detection requires forward pass with actual model
        # Return empty for now; actual shapes populated on first forward
        return self._layer_shapes

    # === Checkpoint Loading ===

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = 'cuda',
        backend: str = 'auto',
        load_vision: bool = True,
        load_audio: bool = True,
        **kwargs
    ) -> 'Gemma4E2BAdapter':
        """
        Load Gemma 4 E2B with optional encoders.

        Args:
            checkpoint_path: HF repo ID or local path
            device: 'cuda', 'mps', 'cpu'
            backend: 'torch', 'mlx', or 'auto' (MLX on mps)
            load_vision: Load ViT vision encoder if available
            load_audio: Load Conformer audio encoder if available
        """
        if backend == 'auto':
            backend = 'mlx' if device == 'mps' else 'torch'

        if backend == 'mlx':
            return cls._load_mlx(checkpoint_path, **kwargs)
        return cls._load_torch(
            checkpoint_path, device, load_vision, load_audio, **kwargs
        )

    @classmethod
    def _load_torch(
        cls,
        path: str,
        device: str,
        load_vision: bool,
        load_audio: bool,
        **kwargs
    ) -> 'Gemma4E2BAdapter':
        """Load model with torch backend."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading Gemma 4 E2B (torch) from {path}...")
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        model.eval()

        vision_encoder = None
        if load_vision:
            vision_encoder = cls._load_vision_encoder(model, device)

        audio_encoder = None
        if load_audio:
            audio_encoder = cls._load_audio_encoder(model, device)

        return cls(model, tokenizer, device, 'torch',
                   vision_encoder, audio_encoder)

    @classmethod
    def _load_mlx(cls, path: str, **kwargs) -> 'Gemma4E2BAdapter':
        """Load model with MLX backend."""
        try:
            from mlx_lm import load
        except ImportError as e:
            raise ImportError(
                "MLX backend requires: pip install mlx>=0.15.0 mlx-lm>=0.15.0"
            ) from e

        print(f"Loading Gemma 4 E2B (MLX) from {path}...")
        try:
            model, tokenizer = load(path)
        except ModuleNotFoundError as e:
            if 'gemma' in str(e).lower():
                raise NotImplementedError(
                    f"MLX doesn't support this model architecture yet. "
                    f"Use backend='torch' instead: "
                    f"Gemma4E2BAdapter.from_checkpoint('{path}', device='mps', backend='torch')"
                ) from e
            raise

        return cls(model, tokenizer, 'mps', 'mlx')

    @classmethod
    def _load_vision_encoder(cls, model, device: str):
        """Extract/load ViT vision encoder."""
        # Vision encoder internal to Gemma 4 if available
        if hasattr(model, 'vision_tower'):
            return model.vision_tower
        return None

    @classmethod
    def _load_audio_encoder(cls, model, device: str):
        """Extract/load Conformer audio encoder."""
        # Audio encoder internal to Gemma 4 if available
        if hasattr(model, 'audio_tower'):
            return model.audio_tower
        return None

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Return default configuration for Gemma 4 E2B."""
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

    def to(self, device: str) -> 'Gemma4E2BAdapter':
        """Move model to device."""
        if hasattr(self._model, 'to'):
            self._model = self._model.to(device)
        self._device = device
        return self

    def eval(self) -> 'Gemma4E2BAdapter':
        """Set model to eval mode."""
        if hasattr(self._model, 'eval'):
            self._model.eval()
        return self

    @property
    def tokenizer(self):
        """Access the tokenizer."""
        return self._tokenizer

    @property
    def backend(self) -> str:
        """Return backend type ('torch' or 'mlx')."""
        return self._backend
