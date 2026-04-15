"""Tests for DMD2 ImageNet adapter.

These tests verify the new diffusion-specific methods without requiring
the actual checkpoint.
"""

import pytest
import torch


@pytest.fixture
def mock_adapter():
    """Create mock DMD2 adapter for testing API."""
    from adapt_diff.adapters.dmd2_imagenet import DMD2ImageNetAdapter

    # Create minimal mock model
    class MockDMD2Model:
        def __init__(self):
            self.model = self._create_mock_unet()

        def _create_mock_unet(self):
            class MockUNet:
                def __init__(self):
                    self.enc = {f'enc_{i}': torch.nn.Identity() for i in range(4)}
                    self.dec = {f'dec_{i}': torch.nn.Identity() for i in range(4)}
            return MockUNet()

        def __call__(self, x, sigma, class_labels):
            # Return denoised output (model predicts x0)
            return torch.randn_like(x)

    model = MockDMD2Model()
    return DMD2ImageNetAdapter(model, device='cpu')


class TestDMD2ImageNetAdapterProperties:
    """Test adapter properties."""

    def test_prediction_type_property(self, mock_adapter):
        """Test prediction_type property."""
        assert mock_adapter.prediction_type == 'sample'

    def test_uses_latent_property(self, mock_adapter):
        """Test uses_latent property."""
        assert mock_adapter.uses_latent is False

    def test_in_channels_property(self, mock_adapter):
        """Test in_channels property."""
        assert mock_adapter.in_channels == 3

    def test_conditioning_type_property(self, mock_adapter):
        """Test conditioning_type property."""
        assert mock_adapter.conditioning_type == 'class'

    def test_latent_scale_factor_property(self, mock_adapter):
        """Test latent_scale_factor property."""
        assert mock_adapter.latent_scale_factor == 1

    def test_resolution_property(self, mock_adapter):
        """Test resolution property."""
        assert mock_adapter.resolution == 64

    def test_num_classes_property(self, mock_adapter):
        """Test num_classes property."""
        assert mock_adapter.num_classes == 1000


class TestDMD2ImageNetAdapterDiffusionMethods:
    """Tests for new diffusion-specific methods."""

    def test_get_timesteps(self, mock_adapter):
        """Test Karras sigma schedule generation."""
        num_steps = 6
        sigmas = mock_adapter.get_timesteps(num_steps, device='cpu')

        assert isinstance(sigmas, torch.Tensor)
        assert len(sigmas) == num_steps + 1  # Includes final 0
        # Sigmas should be in descending order (monotonically decreasing)
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1]
        # Last sigma should be 0
        assert sigmas[-1] == 0.0
        # First sigma should be close to SIGMA_MAX (80.0)
        assert 70.0 < sigmas[0] < 90.0

    def test_get_timesteps_custom_params(self, mock_adapter):
        """Test timestep generation with custom noise_level parameters."""
        # noise_level 50-10 maps to partial noise schedule
        sigmas = mock_adapter.get_timesteps(
            10, device='cpu', noise_level_max=50.0, noise_level_min=10.0
        )

        assert len(sigmas) == 11
        # noise_level 50 → sigma between min and max
        assert sigmas[0] < mock_adapter.SIGMA_MAX
        assert sigmas[0] > mock_adapter.SIGMA_MIN

    def test_get_timesteps_single_step(self, mock_adapter):
        """Test single-step schedule (DMD2's specialty)."""
        sigmas = mock_adapter.get_timesteps(1, device='cpu')

        assert len(sigmas) == 2
        assert sigmas[0] > 0
        assert sigmas[1] == 0.0

    def test_get_timesteps_target_noise(self, mock_adapter):
        """Test timestep generation with target_noise parameters for attribution."""
        # Compute noise_level for sigma=0.5 (attribution target)
        import math
        target_sigma = 0.5
        log_sigma = math.log(target_sigma)
        log_min = math.log(mock_adapter.SIGMA_MIN)
        log_max = math.log(mock_adapter.SIGMA_MAX)
        target_noise_min = (log_sigma - log_min) / (log_max - log_min) * 100.0

        sigmas = mock_adapter.get_timesteps(
            6, device='cpu',
            target_noise_max=100.0,
            target_noise_min=target_noise_min
        )

        assert len(sigmas) == 7  # 6 steps + final 0
        # First sigma should be sigma_max (80)
        assert abs(sigmas[0].item() - 80.0) < 0.01
        # Last non-zero sigma should be close to 0.5
        assert abs(sigmas[-2].item() - 0.5) < 0.01
        assert sigmas[-1] == 0.0

    def test_step(self, mock_adapter):
        """Test Euler stepping."""
        batch_size = 2
        x_t = torch.randn(batch_size, 3, 64, 64)
        t_cur = torch.tensor([10.0])
        t_next = torch.tensor([5.0])
        model_output = torch.randn(batch_size, 3, 64, 64)

        x_next = mock_adapter.step(x_t, t_cur, model_output, t_next=t_next)

        assert x_next.shape == x_t.shape
        assert isinstance(x_next, torch.Tensor)
        # Verify it's not just returning input
        assert not torch.allclose(x_next, x_t)

    def test_step_without_t_next_returns_prediction(self, mock_adapter):
        """Test that step returns prediction when t_next is None (final step)."""
        x_t = torch.randn(1, 3, 64, 64)
        t = torch.tensor([10.0])
        model_output = torch.randn(1, 3, 64, 64)

        # DMD2 step() returns model_output when t_next is None (final step)
        result = mock_adapter.step(x_t, t, model_output)
        assert torch.allclose(result, model_output)

    def test_get_initial_noise(self, mock_adapter):
        """Test initial noise generation."""
        batch_size = 4
        noise = mock_adapter.get_initial_noise(batch_size, device='cpu')

        assert noise.shape == (batch_size, 3, 64, 64)
        assert isinstance(noise, torch.Tensor)
        # Check it's scaled by SIGMA_MAX (80.0 for DMD2)
        assert 60.0 < noise.std().item() < 100.0

    def test_get_initial_noise_custom_noise_level(self, mock_adapter):
        """Test noise generation with custom noise_level."""
        # noise_level 50 → sigma ~6.3 (geometric mean of 80 and 0.5)
        noise = mock_adapter.get_initial_noise(
            1, device='cpu', noise_level=50.0
        )

        # Lower noise_level = lower sigma = smaller std
        assert noise.std().item() < 20.0

    def test_get_initial_noise_with_generator(self, mock_adapter):
        """Test reproducible noise generation."""
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)

        noise1 = mock_adapter.get_initial_noise(2, device='cpu', generator=gen1)
        noise2 = mock_adapter.get_initial_noise(2, device='cpu', generator=gen2)

        assert torch.allclose(noise1, noise2)

    def test_prepare_conditioning(self, mock_adapter):
        """Test class conditioning preparation."""
        class_label = 207  # golden retriever
        batch_size = 2

        cond = mock_adapter.prepare_conditioning(
            class_label=class_label, batch_size=batch_size, device='cpu'
        )

        assert cond.shape == (batch_size, 1000)  # One-hot
        # Check it's one-hot
        assert cond.sum(dim=1).allclose(torch.ones(batch_size))
        assert cond[0, class_label] == 1.0

    def test_prepare_conditioning_random(self, mock_adapter):
        """Test random class sampling."""
        cond = mock_adapter.prepare_conditioning(batch_size=3, device='cpu')

        assert cond.shape == (3, 1000)
        # Check it's one-hot
        assert cond.sum(dim=1).allclose(torch.ones(3))

    def test_encode_decode_identity(self, mock_adapter):
        """Test encode/decode are identity for pixel models."""
        img = torch.randn(1, 3, 64, 64)

        encoded = mock_adapter.encode(img)
        decoded = mock_adapter.decode(img)

        assert torch.allclose(encoded, img)
        assert torch.allclose(decoded, img)

    def test_forward_with_cfg(self, mock_adapter):
        """Test that CFG works for class-conditional models."""
        x = torch.randn(1, 3, 64, 64)
        t = torch.tensor([10.0])
        cond = torch.eye(1000)[207:208]  # class 207
        uncond = torch.zeros(1, 1000)

        # CFG with guidance_scale > 1.0 should work
        output = mock_adapter.forward_with_cfg(
            x, t, cond, uncond, guidance_scale=2.0
        )
        assert output.shape == (1, 3, 64, 64)

    def test_forward_with_cfg_default_uncond(self, mock_adapter):
        """Test that CFG uses zeros as default uncond."""
        x = torch.randn(1, 3, 64, 64)
        t = torch.tensor([10.0])
        cond = torch.eye(1000)[207:208]  # class 207

        # Should work without explicit uncond (defaults to zeros)
        output = mock_adapter.forward_with_cfg(
            x, t, cond, guidance_scale=2.0
        )
        assert output.shape == (1, 3, 64, 64)

    def test_forward_with_cfg_scale_1(self, mock_adapter):
        """Test that scale=1.0 works (no CFG)."""
        x = torch.randn(1, 3, 64, 64)
        t = torch.tensor([10.0])
        cond = torch.eye(1000)[207:208]

        # Should work with scale=1.0 (no CFG needed)
        output = mock_adapter.forward_with_cfg(x, t, cond, guidance_scale=1.0)
        assert output.shape == (1, 3, 64, 64)
