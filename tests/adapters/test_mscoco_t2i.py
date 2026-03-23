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
