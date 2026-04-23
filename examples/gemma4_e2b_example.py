#!/usr/bin/env python3
"""
Example: Gemma 4 E2B text generation with activation extraction.

Prerequisites:
    pip install transformers>=4.40.0 sentencepiece accelerate
    huggingface-cli login  # Gemma requires agreement to terms (for HF download)

For Apple Silicon with MLX:
    pip install mlx>=0.15.0 mlx-lm>=0.15.0

Usage:
    # From HuggingFace
    python examples/gemma4_e2b_example.py

    # From local path (download first with: huggingface-cli download google/gemma-2-2b-it)
    python examples/gemma4_e2b_example.py --model /path/to/gemma-2-2b-it

    # Apple Silicon
    python examples/gemma4_e2b_example.py --device mps --backend torch

    # Custom prompt
    python examples/gemma4_e2b_example.py --prompt "Write a haiku about coding"
"""

import argparse

from adapt_diff.adapters.gemma4_e2b import Gemma4E2BAdapter
from adapt_diff.generation import generate
from adapt_diff.device import get_device, get_backend


def main():
    parser = argparse.ArgumentParser(description='Gemma 4 E2B generation example')
    parser.add_argument(
        '--model', default='google/gemma-2-2b-it',
        help='HuggingFace model ID or local path to downloaded model'
    )
    parser.add_argument(
        '--device', default=None,
        help='Device (cuda, mps, cpu). Auto-detected if not specified'
    )
    parser.add_argument(
        '--backend', default='torch',
        help='Backend (torch, mlx). Note: MLX support for Gemma is limited'
    )
    parser.add_argument(
        '--prompt', default='Explain quantum computing in one sentence:',
        help='Text prompt for generation'
    )
    parser.add_argument(
        '--max-tokens', type=int, default=64,
        help='Maximum new tokens to generate'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-p', type=float, default=0.95,
        help='Nucleus sampling threshold'
    )
    parser.add_argument(
        '--extract-layers', nargs='*', default=None,
        help='Layers to extract activations from (e.g., layer_0 layer_17_attn)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()

    # Auto-detect device and backend
    device = args.device or get_device()
    backend = get_backend(device, args.backend)
    print(f"Using device: {device}, backend: {backend}")

    # Load adapter
    print(f"Loading model from {args.model}...")
    adapter = Gemma4E2BAdapter.from_checkpoint(
        args.model,
        device=device,
        backend=backend,
        load_vision=False,  # Text-only for this example
        load_audio=False,
        embedding_model=None  # Skip EmbeddingGemma
    )
    print(f"Loaded: {adapter}")

    # Prepare conditioning
    print(f"\nPrompt: {args.prompt}")
    adapter.prepare_conditioning(text=args.prompt)

    # Generate
    print(f"Generating up to {args.max_tokens} tokens...")
    result = generate(
        adapter,
        num_steps=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        seed=args.seed,
        extract_layers=args.extract_layers,
        return_trajectory=args.extract_layers is not None
    )

    # Print results
    print("\n" + "=" * 60)
    print("Generated text:")
    print("=" * 60)
    print(result.text[0])
    print("=" * 60)

    print(f"\nTokens generated: {result.tokens.shape[1]}")

    if result.trajectory:
        print(f"Trajectory steps: {len(result.trajectory)}")
        print(f"Activation shape per step: {result.trajectory[0].shape}")


def example_with_trajectory():
    """Example showing activation extraction during generation."""
    device = get_device()

    adapter = Gemma4E2BAdapter.from_checkpoint(
        'google/gemma-2-2b-it',
        device=device,
        backend='auto',
        load_vision=False,
        load_audio=False,
        embedding_model=None
    )

    adapter.prepare_conditioning(text="The meaning of life is")

    # Extract activations from middle layer
    result = generate(
        adapter,
        num_steps=32,
        temperature=0.8,
        device=device,
        extract_layers=['layer_17', 'layer_17_attn'],
        return_trajectory=True
    )

    print(f"Generated: {result.text[0]}")
    print(f"Trajectory shape: {len(result.trajectory)} steps")

    # Trajectory can be used for visualization, analysis, etc.
    import numpy as np
    trajectory = np.stack(result.trajectory)  # (steps, batch, features)
    print(f"Stacked trajectory: {trajectory.shape}")


def example_batch_generation():
    """Example showing batch generation."""
    device = get_device()

    adapter = Gemma4E2BAdapter.from_checkpoint(
        'google/gemma-2-2b-it',
        device=device,
        backend='auto',
        load_vision=False,
        load_audio=False,
        embedding_model=None
    )

    prompts = [
        "Write a one-line joke about programming:",
        "Describe the ocean in three words:",
        "What is 2 + 2? Answer:"
    ]

    for prompt in prompts:
        adapter.prepare_conditioning(text=prompt)
        result = generate(
            adapter,
            num_steps=32,
            temperature=0.7,
            device=device
        )
        print(f"Q: {prompt}")
        print(f"A: {result.text[0]}\n")


if __name__ == '__main__':
    main()
