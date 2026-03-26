"""MSCOCO Text-to-Image adapter implementing the GeneratorAdapter interface.

Based on AttributeByUnlearning (Wang et al., NeurIPS 2024):
https://github.com/PeterWang512/AttributeByUnlearning

Licensed under CC BY-NC-SA 4.0.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from adapt_diff.base import GeneratorAdapter
from adapt_diff.hooks import HookMixin
from adapt_diff.registry import register_adapter


@register_adapter('mscoco-t2i-128')
class MSCOCOT2IAdapter(HookMixin, GeneratorAdapter):
    """
    Adapter for MSCOCO Text-to-Image diffusion model.

    This adapter wraps a UNet2DConditionModel from diffusers trained on
    MSCOCO for text-to-image generation at 128x128 resolution.

    The model operates in latent space (4 channels at 16x16) and requires:
    - Pre-computed text embeddings (1024-dim, typically from CLIP ViT-L/14)
    - Optional: VAE for pixel-space decoding

    Layer naming:
        - down_block_N: N-th down block (0-indexed)
        - mid_block: Middle UNet block
        - up_block_N: N-th up block (0-indexed)
    """

    def __init__(
        self,
        model,
        scheduler,
        device: str = 'cuda',
        vae=None
    ):
        HookMixin.__init__(self)
        self._model = model
        self._scheduler = scheduler
        self._device = device
        self._vae = vae
        self._layer_shapes: Optional[Dict[str, Tuple[int, ...]]] = None

    @property
    def model_type(self) -> str:
        return 'mscoco-t2i-128'

    @property
    def resolution(self) -> int:
        return 128

    @property
    def latent_resolution(self) -> int:
        """Resolution in latent space (128 / 8 = 16)."""
        return 16

    @property
    def num_classes(self) -> int:
        # Text-conditioned, not class-conditioned
        return 0

    @property
    def hookable_layers(self) -> List[str]:
        """Return list of available layer names for hooks."""
        layers = []

        # Down blocks
        for i, _ in enumerate(self._model.down_blocks):
            layers.append(f'down_block_{i}')

        # Mid block
        layers.append('mid_block')

        # Up blocks
        for i, _ in enumerate(self._model.up_blocks):
            layers.append(f'up_block_{i}')

        return layers

    def _get_layer_module(self, layer_name: str):
        """Get the actual PyTorch module for a layer name."""
        if layer_name == 'mid_block':
            return self._model.mid_block

        if layer_name.startswith('down_block_'):
            idx = int(layer_name.split('_')[-1])
            return self._model.down_blocks[idx]

        if layer_name.startswith('up_block_'):
            idx = int(layer_name.split('_')[-1])
            return self._model.up_blocks[idx]

        raise ValueError(f"Unknown layer: {layer_name}")

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for denoising.

        Args:
            x: Noisy latent input (B, 4, 16, 16)
            timestep: Diffusion timestep (B,) or scalar, range [0, 999]
            class_labels: Unused (text-conditioned model)
            encoder_hidden_states: Text embeddings (B, seq_len, 1024)
        """
        if encoder_hidden_states is None:
            raise ValueError(
                "encoder_hidden_states (text embeddings) required for MSCOCO T2I model"
            )

        return self._model(
            x,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]

    def get_timesteps(self, num_steps: int, device: str = 'cuda', **kwargs) -> torch.Tensor:
        """
        Return DDPM timestep schedule.

        Args:
            num_steps: Number of denoising steps
            device: Target device
            **kwargs: Ignored (sigma params not applicable to DDPM)

        Returns:
            Timesteps tensor (num_steps,) in descending order [999, ..., 0]
        """
        self._scheduler.set_timesteps(num_steps, device=device)
        return self._scheduler.timesteps

    def step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        DDPM denoising step via scheduler.

        Args:
            x_t: Current noisy latent (B, 4, 16, 16)
            t: Current timestep (scalar or (B,))
            model_output: Predicted noise from forward() (B, 4, 16, 16)
            t_next: Unused (accepted for API consistency with sigma-based models)
            **kwargs: Passed to scheduler.step() (e.g., generator, eta)

        Returns:
            x_{t-1}: Less noisy latent
        """
        # t_next is ignored - scheduler handles timestep progression internally
        return self._scheduler.step(model_output, t, x_t, **kwargs).prev_sample

    def get_initial_noise(
        self,
        batch_size: int,
        device: str = 'cuda',
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generate initial latent noise.

        Args:
            batch_size: Number of samples
            device: Target device
            generator: Optional RNG for reproducibility

        Returns:
            Noise tensor (B, 4, 16, 16) with unit variance
        """
        return torch.randn(
            batch_size, 4, 16, 16,
            device=device,
            generator=generator
        )

    def prepare_conditioning(
        self,
        text: Optional[str] = None,
        batch_size: int = 1,
        device: str = 'cuda',
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare text conditioning for MSCOCO T2I.

        Args:
            text: Text prompt
            batch_size: Number of samples (text is repeated batch_size times)
            device: Target device
            **kwargs: Unused

        Returns:
            Dict with 'encoder_hidden_states' key containing text embeddings

        Raises:
            ValueError: If text is None
        """
        if text is None:
            raise ValueError("Text prompt required for MSCOCO T2I model")

        # Load CLIP model if not already loaded
        if not hasattr(self, '_clip_tokenizer'):
            from transformers import CLIPTokenizer, CLIPTextModel
            self._clip_tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self._clip_text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(device)
            self._clip_text_encoder.eval()

        # Tokenize and encode text
        tokens = self._clip_tokenizer(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(device)

        with torch.no_grad():
            embeddings = self._clip_text_encoder(tokens)[0]

        # Repeat for batch
        if batch_size > 1:
            embeddings = embeddings.repeat(batch_size, 1, 1)

        return {"encoder_hidden_states": embeddings}

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]] = None,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward with classifier-free guidance.

        Args:
            x: Noisy latent (B, 4, 16, 16)
            t: Timestep
            cond: Conditional dict with 'encoder_hidden_states'
            uncond: Unconditional dict (if None, uses empty text "")
            guidance_scale: CFG scale (1.0 = no guidance)
            **kwargs: Passed to forward()

        Returns:
            Guided noise prediction
        """
        if guidance_scale == 1.0 or uncond is None:
            return self.forward(x, t, **cond, **kwargs)

        # Unconditional forward
        uncond_out = self.forward(x, t, **uncond, **kwargs)

        # Conditional forward
        cond_out = self.forward(x, t, **cond, **kwargs)

        # CFG: uncond + scale * (cond - uncond)
        return uncond_out + guidance_scale * (cond_out - uncond_out)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space.

        Args:
            images: Pixel-space images (B, 3, 128, 128), range [-1, 1]

        Returns:
            Latent tensor (B, 4, 16, 16)
        """
        return self.encode_images(images)

    def decode(self, representation: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to pixel space.

        Args:
            representation: Latent tensor (B, 4, 16, 16)

        Returns:
            Images (B, 3, 128, 128), range [-1, 1]
        """
        return self.decode_latents(representation)

    @property
    def prediction_type(self) -> str:
        """MSCOCO T2I predicts noise (epsilon)."""
        return 'epsilon'

    @property
    def uses_latent(self) -> bool:
        """MSCOCO T2I operates in latent space."""
        return True

    @property
    def in_channels(self) -> int:
        """MSCOCO T2I uses 4-channel latent input."""
        return 4

    @property
    def conditioning_type(self) -> str:
        """MSCOCO T2I uses text conditioning."""
        return 'text'

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
        """Return activation shapes for hookable layers (runs dummy forward on first call)."""
        if self._layer_shapes is not None:
            return self._layer_shapes

        self._layer_shapes = {}
        layer_names = self.hookable_layers

        def make_shape_hook(name):
            def hook(_module, _input, output):
                # Handle different output types
                if isinstance(output, tuple):
                    out = output[0]
                elif hasattr(output, 'sample'):
                    out = output.sample
                else:
                    out = output
                self._layer_shapes[name] = tuple(out.shape[1:])
            return hook

        temp_handles = []
        for name in layer_names:
            module = self._get_layer_module(name)
            handle = module.register_forward_hook(make_shape_hook(name))
            temp_handles.append(handle)

        with torch.no_grad():
            # Dummy forward pass
            dummy_x = torch.randn(1, 4, 16, 16, device=self._device)
            dummy_t = torch.tensor([500], device=self._device)
            # Single token embedding for shape detection
            dummy_emb = torch.randn(1, 1, 1024, device=self._device)
            self._model(dummy_x, dummy_t, encoder_hidden_states=dummy_emb)

        for h in temp_handles:
            h.remove()

        return self._layer_shapes

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to pixel space using VAE.

        .. deprecated::
            Use :meth:`decode` instead.

        Args:
            latents: Latent tensor (B, 4, 16, 16)

        Returns:
            Images in pixel space (B, 3, 128, 128), range [-1, 1]
        """
        import warnings
        warnings.warn(
            "decode_latents() is deprecated, use decode() instead",
            DeprecationWarning,
            stacklevel=2
        )

        if self._vae is None:
            raise ValueError("VAE not provided. Pass vae to from_checkpoint().")

        # Scale latents (standard SD VAE scaling)
        latents = latents / 0.18215

        with torch.no_grad():
            images = self._vae.decode(latents).sample

        return images

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space using VAE.

        .. deprecated::
            Use :meth:`encode` instead.

        Args:
            images: Images in pixel space (B, 3, 128, 128), range [-1, 1]

        Returns:
            Latent tensor (B, 4, 16, 16)
        """
        import warnings
        warnings.warn(
            "encode_images() is deprecated, use encode() instead",
            DeprecationWarning,
            stacklevel=2
        )

        if self._vae is None:
            raise ValueError("VAE not provided. Pass vae to from_checkpoint().")

        with torch.no_grad():
            latents = self._vae.encode(images).latent_dist.sample()

        # Scale latents (standard SD VAE scaling)
        latents = latents * 0.18215

        return latents

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = 'cuda',
        vae_id: Optional[str] = 'stabilityai/sd-vae-ft-mse',
        **kwargs
    ) -> 'MSCOCOT2IAdapter':
        """
        Load adapter from checkpoint file.

        Args:
            checkpoint_path: Path to model weights (.bin or .safetensors)
            device: Target device
            vae_id: HuggingFace VAE model ID for latent decoding (None to skip)

        Returns:
            Initialized adapter
        """
        from diffusers import DDPMScheduler, UNet2DConditionModel
        from safetensors.torch import load_file

        print(f"Loading MSCOCO T2I from {checkpoint_path}...")

        # Create model architecture
        model = UNet2DConditionModel(
            sample_size=16,  # 128 // 8 = 16
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 256, 256),
            cross_attention_dim=1024,
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            attention_head_dim=8,
        )

        # Load weights
        if str(checkpoint_path).endswith('.safetensors'):
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # Scheduler
        scheduler = DDPMScheduler(num_train_timesteps=1000)

        # Optional VAE for latent decoding
        vae = None
        if vae_id:
            try:
                from diffusers import AutoencoderKL
                print(f"Loading VAE from {vae_id}...")
                vae = AutoencoderKL.from_pretrained(vae_id).to(device)
                vae.eval()
            except Exception as e:
                print(f"Warning: Could not load VAE: {e}")

        print(f"Loaded MSCOCO T2I: {model.config.sample_size * 8}x{model.config.sample_size * 8}")

        return cls(model, scheduler, device, vae)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Return default configuration for MSCOCO T2I."""
        return {
            "img_resolution": 128,
            "latent_resolution": 16,
            "latent_channels": 4,
            "cross_attention_dim": 1024,
            "num_train_timesteps": 1000,
            "block_out_channels": [128, 256, 256, 256],
            # Sampling defaults
            "default_steps": 20,
            "guidance_scale": 7.5,
        }

    def to(self, device: str) -> 'MSCOCOT2IAdapter':
        self._model = self._model.to(device)
        if self._vae is not None:
            self._vae = self._vae.to(device)
        self._device = device
        return self

    def eval(self) -> 'MSCOCOT2IAdapter':
        self._model.eval()
        if self._vae is not None:
            self._vae.eval()
        return self

    @property
    def scheduler(self):
        """Access the noise scheduler."""
        return self._scheduler

    @property
    def vae(self):
        """Access the VAE (may be None)."""
        return self._vae
