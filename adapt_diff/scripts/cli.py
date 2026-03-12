"""
CLI for adapt_diff package.

Usage:
    adapt_diff download [--output-dir DIR] [--models dmd2|edm|all]
    adapt_diff list
"""

import argparse
import sys
import urllib.request
from pathlib import Path

CHECKPOINT_URLS = {
    "dmd2": "https://huggingface.co/mckell/diffviews-dmd2-checkpoint/resolve/main/dmd2-imagenet-64-10step.pkl",
    "edm": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl",
}

CHECKPOINT_INFO = {
    "dmd2": {
        "filename": "dmd2-imagenet-64-10step.pkl",
        "size_mb": 296,
        "license": "MIT",
        "description": "DMD2 ImageNet 64x64 (1-10 step, fine-tuned for multi-step)",
        "note": "Fine-tuned from original DMD2 to support up to 10 diffusion steps (for visualization)",
    },
    "edm": {
        "filename": "edm-imagenet-64x64-cond-adm.pkl",
        "size_mb": 296,
        "license": "CC BY-NC-SA 4.0",
        "description": "EDM ImageNet 64x64 (multi-step)",
    },
}


def download_command(args):
    """Download model checkpoints."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = args.models
    if "all" in models:
        models = list(CHECKPOINT_URLS.keys())

    print(f"Output directory: {output_dir.absolute()}")
    print()

    for model in models:
        if model not in CHECKPOINT_URLS:
            print(f"Unknown model: {model}")
            continue

        info = CHECKPOINT_INFO[model]
        filename = info["filename"]
        filepath = output_dir / filename

        print(f"Model: {model}")
        print(f"  {info['description']}")
        if "note" in info:
            print(f"  Note: {info['note']}")
        print(f"  License: {info['license']}")
        print(f"  Size: ~{info['size_mb']} MB")

        if filepath.exists():
            print(f"  Status: Already exists at {filepath}")
            print()
            continue

        url = CHECKPOINT_URLS[model]
        print(f"  Downloading from: {url}")

        try:
            # Download with progress
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(100, downloaded * 100 / total_size)
                    mb = downloaded / 1e6
                    print(f"\r  Progress: {pct:.1f}% ({mb:.1f} MB)", end="", flush=True)

            urllib.request.urlretrieve(url, filepath, reporthook=report_progress)
            print()  # newline after progress
            print(f"  Saved: {filepath} ({filepath.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"  Error: {e}")

        print()

    print("=" * 50)
    print("Download complete!")
    print("=" * 50)
    print()
    print("Usage example:")
    print("  from adapt_diff import get_adapter")
    print(f"  adapter = get_adapter('edm-imagenet-64').from_checkpoint('{output_dir}/edm-imagenet-64x64-cond-adm.pkl')")


def list_command(args):
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
        choices=["dmd2", "edm", "all"],
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
