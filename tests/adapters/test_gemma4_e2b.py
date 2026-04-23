"""Tests for Gemma 4 E2B adapter.

These tests verify the adapter API without requiring the actual model checkpoint.
"""

import pytest
import torch


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0

    def __call__(self, text, return_tensors='pt', padding=True, **kwargs):
        # Return mock token IDs
        class MockOutput:
            def __init__(self):
                self.input_ids = torch.tensor([[1, 5, 10, 15, 20]])

            def to(self, device):
                self.input_ids = self.input_ids.to(device)
                return self
        return MockOutput()

    def batch_decode(self, tokens, skip_special_tokens=True):
        return ["mock decoded text"] * tokens.shape[0]


class MockGemmaModel:
    """Mock Gemma model for testing."""

    def __init__(self):
        self.model = self._create_mock_layers()

    def _create_mock_layers(self):
        class MockLayers:
            def __init__(self):
                self.norm = torch.nn.Identity()
                self.layers = [self._create_layer() for _ in range(35)]

            def _create_layer(self):
                class MockLayer:
                    def __init__(self):
                        self.self_attn = torch.nn.Identity()
                        self.mlp = torch.nn.Identity()
                return MockLayer()
        return MockLayers()

    def __call__(self, input_ids, image_embeds=None, audio_embeds=None,
                 attention_mask=None):
        batch_size = input_ids.shape[0]
        vocab_size = 256000

        class MockOutput:
            def __init__(self):
                self.logits = torch.randn(batch_size, input_ids.shape[1], vocab_size)
        return MockOutput()

    def to(self, device):
        return self

    def eval(self):
        return self


@pytest.fixture
def mock_adapter():
    """Create mock Gemma 4 E2B adapter for testing API."""
    from adapt_diff.adapters.gemma4_e2b import Gemma4E2BAdapter

    model = MockGemmaModel()
    tokenizer = MockTokenizer()
    return Gemma4E2BAdapter(model, tokenizer, device='cpu', backend='torch')


class TestGemma4E2BAdapterProperties:
    """Test adapter properties."""

    def test_model_type(self, mock_adapter):
        assert mock_adapter.model_type == 'gemma4-e2b'

    def test_output_type(self, mock_adapter):
        assert mock_adapter.output_type == 'text'

    def test_generation_mode(self, mock_adapter):
        assert mock_adapter.generation_mode == 'autoregressive'

    def test_resolution(self, mock_adapter):
        assert mock_adapter.resolution == 128000

    def test_num_classes(self, mock_adapter):
        assert mock_adapter.num_classes == 256000

    def test_prediction_type(self, mock_adapter):
        assert mock_adapter.prediction_type == 'logits'

    def test_uses_latent(self, mock_adapter):
        assert mock_adapter.uses_latent is False

    def test_in_channels(self, mock_adapter):
        assert mock_adapter.in_channels == 1

    def test_conditioning_type(self, mock_adapter):
        assert mock_adapter.conditioning_type == 'multimodal'

    def test_timestep_label(self, mock_adapter):
        assert mock_adapter.timestep_label == 'pos'

    def test_backend(self, mock_adapter):
        assert mock_adapter.backend == 'torch'

    def test_tokenizer_property(self, mock_adapter):
        assert mock_adapter.tokenizer is not None


class TestGemma4E2BAdapterHookableLayers:
    """Test hookable layers property."""

    def test_hookable_layers_count(self, mock_adapter):
        layers = mock_adapter.hookable_layers
        # 35 layers * 3 (layer, attn, mlp) + final_norm
        assert len(layers) == 35 * 3 + 1

    def test_hookable_layers_contains_final_norm(self, mock_adapter):
        assert 'final_norm' in mock_adapter.hookable_layers

    def test_hookable_layers_format(self, mock_adapter):
        layers = mock_adapter.hookable_layers
        assert 'layer_0' in layers
        assert 'layer_0_attn' in layers
        assert 'layer_0_mlp' in layers
        assert 'layer_34' in layers


class TestGemma4E2BAdapterForward:
    """Test forward pass."""

    def test_forward_returns_logits(self, mock_adapter):
        mock_adapter.prepare_conditioning(text="Hello")
        x = mock_adapter.get_initial_noise(1, device='cpu')
        t = torch.tensor([0])

        logits = mock_adapter.forward(x, t)

        assert logits.shape == (1, 256000)

    def test_forward_batch(self, mock_adapter):
        mock_adapter.prepare_conditioning(text="Hello")
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        t = torch.tensor([0])

        logits = mock_adapter.forward(x, t)

        assert logits.shape == (2, 256000)


class TestGemma4E2BAdapterStep:
    """Test step (token sampling)."""

    def test_step_extends_sequence(self, mock_adapter):
        x_t = torch.tensor([[1, 2, 3, 4, 5]])
        t = torch.tensor([0])
        logits = torch.randn(1, 256000)

        x_next = mock_adapter.step(x_t, t, logits)

        assert x_next.shape[0] == 1
        assert x_next.shape[1] == 6  # Extended by 1

    def test_step_with_temperature(self, mock_adapter):
        x_t = torch.tensor([[1, 2, 3]])
        t = torch.tensor([0])
        logits = torch.randn(1, 256000)

        x_next = mock_adapter.step(x_t, t, logits, temperature=0.5)

        assert x_next.shape[1] == 4

    def test_step_with_top_p(self, mock_adapter):
        x_t = torch.tensor([[1, 2, 3]])
        t = torch.tensor([0])
        logits = torch.randn(1, 256000)

        x_next = mock_adapter.step(x_t, t, logits, top_p=0.9)

        assert x_next.shape[1] == 4


class TestGemma4E2BAdapterTimesteps:
    """Test timestep generation."""

    def test_get_timesteps(self, mock_adapter):
        num_steps = 10
        timesteps = mock_adapter.get_timesteps(num_steps, device='cpu')

        assert isinstance(timesteps, torch.Tensor)
        assert len(timesteps) == num_steps
        assert timesteps[0] == 0
        assert timesteps[-1] == num_steps - 1

    def test_get_timesteps_is_range(self, mock_adapter):
        timesteps = mock_adapter.get_timesteps(5, device='cpu')
        expected = torch.arange(5)
        assert torch.equal(timesteps, expected)


class TestGemma4E2BAdapterNoiseLevelConversion:
    """Test noise level to/from native conversion."""

    def test_noise_level_to_native(self, mock_adapter):
        noise_level = torch.tensor([50.0])
        native = mock_adapter.noise_level_to_native(noise_level)
        # 50% of 256 max tokens = 128
        assert native.item() == 128

    def test_noise_level_to_native_boundaries(self, mock_adapter):
        assert mock_adapter.noise_level_to_native(torch.tensor([0.0])).item() == 0
        assert mock_adapter.noise_level_to_native(torch.tensor([100.0])).item() == 256

    def test_native_to_noise_level(self, mock_adapter):
        native = torch.tensor([128])
        noise_level = mock_adapter.native_to_noise_level(native)
        assert noise_level.item() == 50.0

    def test_roundtrip_conversion(self, mock_adapter):
        original = torch.tensor([75.0])
        native = mock_adapter.noise_level_to_native(original)
        back = mock_adapter.native_to_noise_level(native.float())
        assert torch.isclose(back, original, atol=1.0)


class TestGemma4E2BAdapterConditioning:
    """Test conditioning preparation."""

    def test_prepare_conditioning_requires_text(self, mock_adapter):
        with pytest.raises(ValueError, match="Text prompt required"):
            mock_adapter.prepare_conditioning()

    def test_prepare_conditioning_sets_prompt(self, mock_adapter):
        mock_adapter.prepare_conditioning(text="Hello world")
        assert mock_adapter._current_prompt == "Hello world"  # pylint: disable=protected-access

    def test_prepare_conditioning_returns_dict(self, mock_adapter):
        cond = mock_adapter.prepare_conditioning(text="Hello")
        assert isinstance(cond, dict)


class TestGemma4E2BAdapterInitialNoise:
    """Test initial noise (tokenized prompt) generation."""

    def test_get_initial_noise_requires_prompt(self, mock_adapter):
        with pytest.raises(ValueError, match="prepare_conditioning"):
            mock_adapter.get_initial_noise(1, device='cpu')

    def test_get_initial_noise_returns_tokens(self, mock_adapter):
        mock_adapter.prepare_conditioning(text="Hello world")
        tokens = mock_adapter.get_initial_noise(1, device='cpu')

        assert isinstance(tokens, torch.Tensor)
        assert tokens.dim() == 2
        assert tokens.shape[0] == 1

    def test_get_initial_noise_batch(self, mock_adapter):
        mock_adapter.prepare_conditioning(text="Hello")
        tokens = mock_adapter.get_initial_noise(3, device='cpu')

        assert tokens.shape[0] == 3


class TestGemma4E2BAdapterDefaultConfig:
    """Test default config."""

    def test_get_default_config(self, mock_adapter):
        config = mock_adapter.get_default_config()

        assert config['model_type'] == 'gemma4-e2b'
        assert config['generation_mode'] == 'autoregressive'
        assert config['vocab_size'] == 256000
        assert config['context_length'] == 128000
        assert 'text' in config['input_modalities']


class TestGemma4E2BAdapterDeviceMethods:
    """Test device movement methods."""

    def test_to_returns_self(self, mock_adapter):
        result = mock_adapter.to('cpu')
        assert result is mock_adapter

    def test_eval_returns_self(self, mock_adapter):
        result = mock_adapter.eval()
        assert result is mock_adapter
