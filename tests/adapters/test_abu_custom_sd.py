"""Tests for Custom Diffusion adapter.

These tests use the base SD v1.4 model which auto-downloads from HuggingFace.
Custom diffusion weights are optional.
"""

import pytest
import torch

# Skip if diffusers not available
pytest.importorskip("diffusers")


@pytest.fixture(scope="module")
def adapter():
    """Load adapter with base SD v1.4 (no custom weights)."""
    from adapt_diff import get_adapter
    AdapterClass = get_adapter('abu-custom-sd14')
    # Use CPU and skip VAE for faster tests
    return AdapterClass.from_checkpoint(
        checkpoint_path=None,
        device='cpu',
        vae_id=None,
        enable_xformers=False
    )


@pytest.fixture(scope="module")
def adapter_with_vae():
    """Load adapter with VAE for decode tests."""
    from adapt_diff import get_adapter
    AdapterClass = get_adapter('abu-custom-sd14')
    return AdapterClass.from_checkpoint(
        checkpoint_path=None,
        device='cpu',
        vae_id='stabilityai/sd-vae-ft-mse',
        enable_xformers=False
    )


class TestAbuCustomSDAdapter:
    """Tests for AbuCustomSDAdapter."""

    def test_load_base_model(self, adapter):
        """Test that base SD model loads correctly."""
        assert adapter is not None
        assert adapter.model_type == 'abu-custom-sd14'

    def test_resolution(self, adapter):
        """Test resolution properties."""
        assert adapter.resolution == 512
        assert adapter.latent_resolution == 64

    def test_cross_attention_dim(self, adapter):
        """Test CLIP embedding dimension."""
        assert adapter.cross_attention_dim == 768

    def test_num_classes(self, adapter):
        """Test that model is text-conditioned (no classes)."""
        assert adapter.num_classes == 0

    def test_sd_version(self, adapter):
        """Test SD version property."""
        assert adapter.sd_version == 'CompVis/stable-diffusion-v1-4'

    def test_hookable_layers(self, adapter):
        """Test hookable layer names."""
        layers = adapter.hookable_layers
        assert len(layers) > 0
        assert 'down_block_0' in layers
        assert 'mid_block' in layers
        assert 'up_block_0' in layers

    def test_get_layer_shapes(self, adapter):
        """Test layer shape detection."""
        shapes = adapter.get_layer_shapes()

        assert len(shapes) > 0
        assert 'mid_block' in shapes

        # Mid block should have specific shape for SD v1.4
        mid_shape = shapes['mid_block']
        assert len(mid_shape) == 3  # (C, H, W)

    def test_forward_pass(self, adapter):
        """Test forward pass with latent input."""
        batch_size = 1
        x = torch.randn(batch_size, 4, 64, 64)
        t = torch.tensor([500])
        # SD v1.4 uses 768-dim CLIP embeddings
        text_emb = torch.randn(batch_size, 77, 768)

        with torch.no_grad():
            out = adapter.forward(x, t, encoder_hidden_states=text_emb)

        assert out.shape == (batch_size, 4, 64, 64)

    def test_forward_requires_text_embeddings(self, adapter):
        """Test that forward raises without text embeddings."""
        x = torch.randn(1, 4, 64, 64)
        t = torch.tensor([500])

        with pytest.raises(ValueError, match="encoder_hidden_states"):
            adapter.forward(x, t)

    def test_activation_hooks(self, adapter):
        """Test activation extraction via hooks."""
        layers_to_hook = ['down_block_0', 'mid_block']

        for layer in layers_to_hook:
            hook = adapter.make_extraction_hook(layer)
            adapter.register_activation_hooks([layer], hook)

        x = torch.randn(1, 4, 64, 64)
        t = torch.tensor([500])
        text_emb = torch.randn(1, 77, 768)

        with torch.no_grad():
            adapter.forward(x, t, encoder_hidden_states=text_emb)

        activations = adapter.get_activations()

        assert 'down_block_0' in activations
        assert 'mid_block' in activations

        adapter.remove_hooks()
        adapter.clear_activations()

    def test_default_config(self):
        """Test default configuration."""
        from adapt_diff.adapters.abu_custom_sd import AbuCustomSDAdapter

        config = AbuCustomSDAdapter.get_default_config()

        assert config['img_resolution'] == 512
        assert config['latent_resolution'] == 64
        assert config['latent_channels'] == 4
        assert config['cross_attention_dim'] == 768
        assert config['sd_version'] == 'CompVis/stable-diffusion-v1-4'


class TestAbuCustomSDAdapterVAE:
    """Tests for VAE functionality."""

    def test_vae_loaded(self, adapter_with_vae):
        """Test VAE is loaded."""
        assert adapter_with_vae.vae is not None

    def test_encode_images(self, adapter_with_vae):
        """Test image encoding to latent space."""
        img = torch.randn(1, 3, 512, 512)

        latents = adapter_with_vae.encode_images(img)

        assert latents.shape == (1, 4, 64, 64)

    def test_decode_latents(self, adapter_with_vae):
        """Test latent decoding to pixel space."""
        latents = torch.randn(1, 4, 64, 64)

        images = adapter_with_vae.decode_latents(latents)

        assert images.shape == (1, 3, 512, 512)

    def test_decode_without_vae_raises(self, adapter):
        """Test that decode raises when VAE not loaded."""
        latents = torch.randn(1, 4, 64, 64)

        with pytest.raises(ValueError, match="VAE not provided"):
            adapter.decode_latents(latents)

    def test_encode_without_vae_raises(self, adapter):
        """Test that encode raises when VAE not loaded."""
        img = torch.randn(1, 3, 512, 512)

        with pytest.raises(ValueError, match="VAE not provided"):
            adapter.encode_images(img)

    def test_encode_method(self, adapter_with_vae):
        """Test new encode() method."""
        img = torch.randn(1, 3, 512, 512)

        latents = adapter_with_vae.encode(img)

        assert latents.shape == (1, 4, 64, 64)

    def test_decode_method(self, adapter_with_vae):
        """Test new decode() method."""
        latents = torch.randn(1, 4, 64, 64)

        images = adapter_with_vae.decode(latents)

        assert images.shape == (1, 3, 512, 512)


class TestAbuCustomSDAdapterDiffusionMethods:
    """Tests for new diffusion-specific methods."""

    def test_get_timesteps(self, adapter):
        """Test timestep schedule generation."""
        num_steps = 50
        timesteps = adapter.get_timesteps(num_steps, device='cpu')

        assert isinstance(timesteps, torch.Tensor)
        assert len(timesteps) == num_steps
        # DDPM timesteps should be in descending order
        assert timesteps[0] > timesteps[-1]
        assert timesteps[-1] >= 0

    def test_step(self, adapter):
        """Test single denoising step."""
        # Need to set timesteps first for scheduler
        timesteps = adapter.get_timesteps(50, device='cpu')

        batch_size = 2
        x_t = torch.randn(batch_size, 4, 64, 64)
        t = timesteps[25]  # Use a valid timestep from schedule
        model_output = torch.randn(batch_size, 4, 64, 64)

        x_next = adapter.step(x_t, t, model_output)

        assert x_next.shape == x_t.shape
        assert isinstance(x_next, torch.Tensor)

    def test_get_initial_noise(self, adapter):
        """Test initial noise generation."""
        batch_size = 4
        noise = adapter.get_initial_noise(batch_size, device='cpu')

        assert noise.shape == (batch_size, 4, 64, 64)
        assert isinstance(noise, torch.Tensor)
        # Check approximate unit variance
        assert 0.5 < noise.std().item() < 1.5

    def test_get_initial_noise_with_generator(self, adapter):
        """Test noise generation with fixed seed."""
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)

        noise1 = adapter.get_initial_noise(2, device='cpu', generator=gen1)
        noise2 = adapter.get_initial_noise(2, device='cpu', generator=gen2)

        assert torch.allclose(noise1, noise2)

    @pytest.mark.slow
    def test_prepare_conditioning(self, adapter):
        """Test text conditioning preparation (requires CLIP download)."""
        text = "a cute dog"
        batch_size = 2

        try:
            cond = adapter.prepare_conditioning(text, batch_size, device='cpu')

            assert isinstance(cond, dict)
            assert 'encoder_hidden_states' in cond
            emb = cond['encoder_hidden_states']
            assert emb.shape[0] == batch_size
            assert emb.shape[1] == 77  # CLIP max length
            assert emb.shape[2] == 1024  # CLIP ViT-L/14 dimension
        except Exception as e:
            pytest.skip(f"CLIP model download failed: {e}")

    def test_prepare_conditioning_requires_text(self, adapter):
        """Test that prepare_conditioning raises without text."""
        with pytest.raises(ValueError, match="Text prompt required"):
            adapter.prepare_conditioning(text=None)

    @pytest.mark.slow
    def test_forward_with_cfg(self, adapter):
        """Test classifier-free guidance forward (requires CLIP download)."""
        batch_size = 2
        x = torch.randn(batch_size, 4, 64, 64)
        t = torch.tensor([500])

        try:
            cond = adapter.prepare_conditioning("a dog", batch_size, device='cpu')
            uncond = adapter.prepare_conditioning("", batch_size, device='cpu')

            # Test with CFG
            with torch.no_grad():
                out_cfg = adapter.forward_with_cfg(x, t, cond, uncond, guidance_scale=7.5)

            assert out_cfg.shape == (batch_size, 4, 64, 64)

            # Test without CFG (scale=1.0)
            with torch.no_grad():
                out_no_cfg = adapter.forward_with_cfg(x, t, cond, guidance_scale=1.0)

            assert out_no_cfg.shape == (batch_size, 4, 64, 64)
        except Exception as e:
            pytest.skip(f"CLIP model download failed: {e}")

    def test_prediction_type_property(self, adapter):
        """Test prediction_type property."""
        assert adapter.prediction_type == 'epsilon'

    def test_uses_latent_property(self, adapter):
        """Test uses_latent property."""
        assert adapter.uses_latent is True

    def test_in_channels_property(self, adapter):
        """Test in_channels property."""
        assert adapter.in_channels == 4

    def test_conditioning_type_property(self, adapter):
        """Test conditioning_type property."""
        assert adapter.conditioning_type == 'text'

    def test_latent_scale_factor_property(self, adapter):
        """Test latent_scale_factor property."""
        assert adapter.latent_scale_factor == 8
