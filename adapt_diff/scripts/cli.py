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
    "dmd2": (
        "https://huggingface.co/mckell/diffviews-dmd2-checkpoint"
        "/resolve/main/dmd2-imagenet-64-10step.pkl"
    ),
    "edm": (
        "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/"
        "edm-imagenet-64x64-cond-adm.pkl"
    ),
    "mscoco": "huggingface://sywang/AttributeByUnlearning/mscoco/model_fisher.7z",
}

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
        "hf_files": ["mscoco/model_fisher.7z"],
    },
}


def download_mscoco(output_dir: Path, info: dict):
    """Download MSCOCO model from HuggingFace dataset repo."""
    import subprocess
    import shutil

    mscoco_dir = output_dir / "mscoco"
    model_path = mscoco_dir / "model.bin"

    if model_path.exists():
        print(f"  Status: Already exists at {model_path}")
        return

    mscoco_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        print("  Downloading from HuggingFace...")
        archive_path = hf_hub_download(
            repo_id=info["hf_repo"],
            filename="mscoco/model_fisher.7z",
            repo_type="dataset",
        )

        # Find 7z executable
        sevenz = shutil.which("7z") or shutil.which("7zz")
        if not sevenz:
            print("  Error: 7z not found. Install p7zip:")
            print("    brew install p7zip (mac) / apt install p7zip-full (linux)")
            print(f"  Archive downloaded to: {archive_path}")
            print("  Extract manually: 7z x model_fisher.7z -o<output_dir>/mscoco")
            return

        print(f"  Extracting with {sevenz}...")
        subprocess.run([sevenz, "x", archive_path, f"-o{mscoco_dir}", "-y"], check=True)

        if model_path.exists():
            print(f"  Saved: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  Extracted to: {mscoco_dir}")

    except ImportError:
        print("  Error: huggingface_hub not installed. Run: pip install huggingface_hub")
    except Exception as e:
        print(f"  Error: {e}")


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

        print(f"Model: {model}")
        print(f"  {info['description']}")
        if "note" in info:
            print(f"  Note: {info['note']}")
        print(f"  License: {info['license']}")
        print(f"  Size: ~{info['size_mb']} MB")

        # Handle MSCOCO specially (HuggingFace dataset repo)
        if model == "mscoco":
            download_mscoco(output_dir, info)
            print()
            continue

        filepath = output_dir / filename

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
        choices=["dmd2", "edm", "mscoco", "all"],
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
