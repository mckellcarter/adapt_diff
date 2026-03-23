"""
CLI for adapt_diff package.

Usage:
    adapt_diff download [--output-dir DIR] [--models dmd2|edm|mscoco|all]
    adapt_diff list
"""

import argparse
import sys
from pathlib import Path

from .downloaders import DOWNLOADERS

CHECKPOINT_INFO = {
    "dmd2": {
        "filename": "dmd2-imagenet-64-10step.pkl",
        "size_mb": 296,
        "license": "MIT",
        "description": "DMD2 ImageNet 64x64 (1-10 step, fine-tuned for multi-step)",
        "note": "Fine-tuned from original DMD2 for multi-step (up to 10 steps)",
    },
    "edm": {
        "filename": "edm-imagenet-64x64-cond-adm.pkl",
        "size_mb": 296,
        "license": "CC BY-NC-SA 4.0",
        "description": "EDM ImageNet 64x64 (multi-step)",
    },
    "mscoco": {
        "filename": "model.bin",
        "size_mb": 150,
        "license": "CC BY-NC-SA 4.0",
        "description": "MSCOCO T2I 128x128 (text-to-image diffusion)",
        "note": "From AttributeByUnlearning (NeurIPS 2024). Requires 7z.",
        "hf_repo": "sywang/AttributeByUnlearning",
    },
}


def download_command(args):
    """Download model checkpoints."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = args.models
    if "all" in models:
        models = list(DOWNLOADERS.keys())

    print(f"Output directory: {output_dir.absolute()}")
    print()

    for model in models:
        if model not in DOWNLOADERS:
            print(f"Unknown model: {model}")
            continue

        info = CHECKPOINT_INFO[model]

        print(f"Model: {model}")
        print(f"  {info['description']}")
        if "note" in info:
            print(f"  Note: {info['note']}")
        print(f"  License: {info['license']}")
        print(f"  Size: ~{info['size_mb']} MB")

        # Call model-specific downloader
        downloader = DOWNLOADERS[model]
        downloader(output_dir, info)
        print()

    print("=" * 50)
    print("Download complete!")
    print("=" * 50)
    print()
    print("Usage example:")
    print("  from adapt_diff import get_adapter")
    print("  adapter = get_adapter('edm-imagenet-64').from_checkpoint(")
    print(f"      '{output_dir}/edm-imagenet-64x64-cond-adm.pkl')")


def list_command(_args):
    """List available adapters."""
    from adapt_diff import list_adapters
    adapters = list_adapters()
    print("Available adapters:")
    for name in adapters:
        print(f"  - {name}")


def main():
    parser = argparse.ArgumentParser(
        prog="adapt_diff",
        description="Model-agnostic adapter interface for diffusion models"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download subcommand
    download_parser = subparsers.add_parser(
        "download",
        help="Download model checkpoints"
    )
    download_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="checkpoints",
        help="Output directory (default: checkpoints)"
    )
    download_parser.add_argument(
        "--models", "-m",
        nargs="*",
        choices=list(DOWNLOADERS.keys()) + ["all"],
        default=["all"],
        help="Models to download (default: all)"
    )

    # List subcommand
    subparsers.add_parser(
        "list",
        help="List available adapters"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "download":
        download_command(args)
    elif args.command == "list":
        list_command(args)


if __name__ == "__main__":
    main()
