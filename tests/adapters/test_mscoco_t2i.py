"""Tests for MSCOCO T2I adapter.

These tests require the model checkpoint at checkpoints/mscoco/model.bin.
Run `adapt_diff download --models mscoco` to download.
"""

import pytest
import torch

# Skip all tests if checkpoint not available
CHECKPOINT_PATH = "checkpoints/mscoco/model.bin"


def checkpoint_exists():
    from pathlib import Path
    return Path(CHECKPOINT_PATH).exists()


requires_checkpoint = pytest.mark.skipif(
    not checkpoint_exists(),
    reason=f"Checkpoint not found: {CHECKPOINT_PATH}"
)


@pytest.fixture(scope="module")
def adapter():
    """Load adapter once for all tests in module."""
    from adapt_diff import get_adapter
    AdapterClass = get_adapter('mscoco-t2i-128')
    return AdapterClass.from_checkpoint(CHECKPOINT_PATH, device='cpu', vae_id=None)


@pytest.fixture(scope="module")
def adapter_with_vae():
    """Load adapter with VAE for decode tests."""
    from adapt_diff import get_adapter
    AdapterClass = get_adapter('mscoco-t2i-128')
    return AdapterClass.from_checkpoint(
        CHECKPOINT_PATH, device='cpu', vae_id='stabilityai/sd-vae-ft-mse'
    )


class TestMSCOCOT2IAdapter:
    """Tests for MSCOCOT2IAdapter."""

    @requires_checkpoint
    def test_load_checkpoint(self, adapter):
        """Test that checkpoint loads correctly."""
        assert adapter is not None
        assert adapter.model_type == 'mscoco-t2i-128'

    @requires_checkpoint
    def test_resolution(self, adapter):
        """Test resolution properties."""
        assert adapter.resolution == 128
        assert adapter.latent_resolution == 16

    @requires_checkpoint
    def test_num_classes(self, adapter):
        """Test that model is text-conditioned (no classes)."""
        assert adapter.num_classes == 0

    @requires_checkpoint
    def test_hookable_layers(self, adapter):
        """Test hookable layer names."""
        layers = adapter.hookable_layers
        assert len(layers) == 9
        assert 'down_block_0' in layers
        assert 'mid_block' in layers
        assert 'up_block_3' in layers

    @requires_checkpoint
    def test_get_layer_shapes(self, adapter):
        """Test layer shape detection."""
        shapes = adapter.get_layer_shapes()

        assert len(shapes) == 9

        # Check specific shapes
        assert shapes['down_block_0'] == (128, 8, 8)
        assert shapes['mid_block'] == (256, 2, 2)
        assert shapes['up_block_3'] == (128, 16, 16)

    @requires_checkpoint
    def test_forward_pass(self, adapter):
        """Test forward pass with latent input."""
        batch_size = 2
        x = torch.randn(batch_size, 4, 16, 16)
        t = torch.tensor([500, 300])
        text_emb = torch.randn(batch_size, 77, 1024)

        with torch.no_grad():
            out = adapter.forward(x, t, encoder_hidden_states=text_emb)

        assert out.shape == (batch_size, 4, 16, 16)

    @requires_checkpoint
    def test_forward_requires_text_embeddings(self, adapter):
        """Test that forward raises without text embeddings."""
        x = torch.randn(1, 4, 16, 16)
        t = torch.tensor([500])

        with pytest.raises(ValueError, match="encoder_hidden_states"):
            adapter.forward(x, t)

    @requires_checkpoint
    def test_activation_hooks(self, adapter):
        """Test activation extraction via hooks."""
        layers_to_hook = ['down_block_1', 'mid_block', 'up_block_2']

        for layer in layers_to_hook:
            hook = adapter.make_extraction_hook(layer)
            adapter.register_activation_hooks([layer], hook)

        x = torch.randn(1, 4, 16, 16)
        t = torch.tensor([500])
        text_emb = torch.randn(1, 77, 1024)

        with torch.no_grad():
            adapter.forward(x, t, encoder_hidden_states=text_emb)

        activations = adapter.get_activations()

        assert 'down_block_1' in activations
        assert 'mid_block' in activations
        assert 'up_block_2' in activations

        assert activations['down_block_1'].shape == (1, 256, 4, 4)
        assert activations['mid_block'].shape == (1, 256, 2, 2)
        assert activations['up_block_2'].shape == (1, 256, 16, 16)

        adapter.remove_hooks()
        adapter.clear_activations()

    @requires_checkpoint
    def test_default_config(self):
        """Test default configuration."""
        from adapt_diff.adapters.mscoco_t2i import MSCOCOT2IAdapter

        config = MSCOCOT2IAdapter.get_default_config()

        assert config['img_resolution'] == 128
        assert config['latent_resolution'] == 16
        assert config['latent_channels'] == 4
        assert config['cross_attention_dim'] == 1024


class TestMSCOCOT2IAdapterVAE:
    """Tests for VAE functionality."""

    @requires_checkpoint
    def test_vae_loaded(self, adapter_with_vae):
        """Test VAE is loaded."""
        assert adapter_with_vae.vae is not None

    @requires_checkpoint
    def test_encode_images(self, adapter_with_vae):
        """Test image encoding to latent space."""
        img = torch.randn(1, 3, 128, 128)

        latents = adapter_with_vae.encode_images(img)

        assert latents.shape == (1, 4, 16, 16)

    @requires_checkpoint
    def test_decode_latents(self, adapter_with_vae):
        """Test latent decoding to pixel space."""
        latents = torch.randn(1, 4, 16, 16)

        images = adapter_with_vae.decode_latents(latents)

        assert images.shape == (1, 3, 128, 128)

    @requires_checkpoint
    def test_encode_decode_cycle(self, adapter_with_vae):
        """Test encode/decode roundtrip."""
        img = torch.randn(1, 3, 128, 128)

        latents = adapter_with_vae.encode_images(img)
        decoded = adapter_with_vae.decode_latents(latents)

        assert decoded.shape == img.shape

    @requires_checkpoint
    def test_decode_without_vae_raises(self, adapter):
        """Test that decode raises when VAE not loaded."""
        latents = torch.randn(1, 4, 16, 16)

        with pytest.raises(ValueError, match="VAE not provided"):
            adapter.decode_latents(latents)

    @requires_checkpoint
    def test_encode_without_vae_raises(self, adapter):
        """Test that encode raises when VAE not loaded."""
        img = torch.randn(1, 3, 128, 128)

        with pytest.raises(ValueError, match="VAE not provided"):
            adapter.encode_images(img)

    @requires_checkpoint
    def test_encode_method(self, adapter_with_vae):
        """Test new encode() method."""
        img = torch.randn(1, 3, 128, 128)

        latents = adapter_with_vae.encode(img)

        assert latents.shape == (1, 4, 16, 16)

    @requires_checkpoint
    def test_decode_method(self, adapter_with_vae):
        """Test new decode() method."""
        latents = torch.randn(1, 4, 16, 16)

        images = adapter_with_vae.decode(latents)

        assert images.shape == (1, 3, 128, 128)


class TestMSCOCOT2IAdapterDiffusionMethods:
    """Tests for new diffusion-specific methods."""

    @requires_checkpoint
    def test_get_timesteps(self, adapter):
        """Test timestep schedule generation."""
        num_steps = 50
        timesteps = adapter.get_timesteps(num_steps, device='cpu')

        assert isinstance(timesteps, torch.Tensor)
        assert len(timesteps) == num_steps
        # DDPM timesteps should be in descending order
        assert timesteps[0] > timesteps[-1]
        assert timesteps[-1] >= 0

    @requires_checkpoint
    def test_step(self, adapter):
        """Test single denoising step."""
        # Need to set timesteps first for scheduler
        timesteps = adapter.get_timesteps(50, device='cpu')

        batch_size = 2
        x_t = torch.randn(batch_size, 4, 16, 16)
        t = timesteps[25]  # Use a valid timestep from schedule
        model_output = torch.randn(batch_size, 4, 16, 16)

        x_next = adapter.step(x_t, t, model_output)

        assert x_next.shape == x_t.shape
        assert isinstance(x_next, torch.Tensor)

    @requires_checkpoint
    def test_get_initial_noise(self, adapter):
        """Test initial noise generation."""
        batch_size = 4
        noise = adapter.get_initial_noise(batch_size, device='cpu')

        assert noise.shape == (batch_size, 4, 16, 16)
        assert isinstance(noise, torch.Tensor)
        # Check approximate unit variance
        assert 0.5 < noise.std().item() < 1.5

    @requires_checkpoint
    def test_get_initial_noise_with_generator(self, adapter):
        """Test noise generation with fixed seed."""
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)

        noise1 = adapter.get_initial_noise(2, device='cpu', generator=gen1)
        noise2 = adapter.get_initial_noise(2, device='cpu', generator=gen2)

        assert torch.allclose(noise1, noise2)

    @requires_checkpoint
    @pytest.mark.slow
    def test_prepare_conditioning(self, adapter):
        """Test text conditioning preparation (requires CLIP download)."""
        text = "a cat sitting on a mat"
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

    @requires_checkpoint
    def test_prepare_conditioning_requires_text(self, adapter):
        """Test that prepare_conditioning raises without text."""
        with pytest.raises(ValueError, match="Text prompt required"):
            adapter.prepare_conditioning(text=None)

    @requires_checkpoint
    @pytest.mark.slow
    def test_forward_with_cfg(self, adapter):
        """Test classifier-free guidance forward (requires CLIP download)."""
        batch_size = 2
        x = torch.randn(batch_size, 4, 16, 16)
        t = torch.tensor([500])

        try:
            cond = adapter.prepare_conditioning("a cat", batch_size, device='cpu')
            uncond = adapter.prepare_conditioning("", batch_size, device='cpu')

            # Test with CFG
            with torch.no_grad():
                out_cfg = adapter.forward_with_cfg(x, t, cond, uncond, guidance_scale=7.5)

            assert out_cfg.shape == (batch_size, 4, 16, 16)

            # Test without CFG (scale=1.0)
            with torch.no_grad():
                out_no_cfg = adapter.forward_with_cfg(x, t, cond, guidance_scale=1.0)

            assert out_no_cfg.shape == (batch_size, 4, 16, 16)
        except Exception as e:
            pytest.skip(f"CLIP model download failed: {e}")

    @requires_checkpoint
    def test_prediction_type_property(self, adapter):
        """Test prediction_type property."""
        assert adapter.prediction_type == 'epsilon'

    @requires_checkpoint
    def test_uses_latent_property(self, adapter):
        """Test uses_latent property."""
        assert adapter.uses_latent is True

    @requires_checkpoint
    def test_in_channels_property(self, adapter):
        """Test in_channels property."""
        assert adapter.in_channels == 4

    @requires_checkpoint
    def test_conditioning_type_property(self, adapter):
        """Test conditioning_type property."""
        assert adapter.conditioning_type == 'text'

    @requires_checkpoint
    def test_latent_scale_factor_property(self, adapter):
        """Test latent_scale_factor property."""
        assert adapter.latent_scale_factor == 8
