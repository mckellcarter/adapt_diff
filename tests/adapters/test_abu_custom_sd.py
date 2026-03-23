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
