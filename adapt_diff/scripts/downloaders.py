"""Model-specific download functions."""

import urllib.request
from pathlib import Path


def download_url(url: str, filepath: Path, name: str = "file"):
    """Download file from URL with progress reporting."""
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / 1e6
            print(f"\r  Progress: {pct:.1f}% ({mb:.1f} MB)", end="", flush=True)

    print(f"  Downloading {name}...")
    urllib.request.urlretrieve(url, filepath, reporthook=report_progress)
    print()  # newline after progress
    print(f"  Saved: {filepath} ({filepath.stat().st_size / 1e6:.1f} MB)")


def download_dmd2(output_dir: Path, info: dict) -> bool:
    """Download DMD2 checkpoint."""
    filepath = output_dir / info["filename"]

    if filepath.exists():
        print(f"  Status: Already exists at {filepath}")
        return True

    url = (
        "https://huggingface.co/mckell/diffviews-dmd2-checkpoint"
        "/resolve/main/dmd2-imagenet-64-10step.pkl"
    )
    try:
        download_url(url, filepath, "DMD2 checkpoint")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_edm(output_dir: Path, info: dict) -> bool:
    """Download EDM checkpoint."""
    filepath = output_dir / info["filename"]

    if filepath.exists():
        print(f"  Status: Already exists at {filepath}")
        return True

    url = (
        "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/"
        "edm-imagenet-64x64-cond-adm.pkl"
    )
    try:
        download_url(url, filepath, "EDM checkpoint")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_mscoco(output_dir: Path, info: dict) -> bool:
    """Download MSCOCO model from HuggingFace dataset repo."""
    import shutil
    import subprocess

    mscoco_dir = output_dir / "mscoco"
    model_path = mscoco_dir / "model.bin"

    if model_path.exists():
        print(f"  Status: Already exists at {model_path}")
        return True

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
            return False

        print(f"  Extracting with {sevenz}...")
        subprocess.run(
            [sevenz, "x", archive_path, f"-o{mscoco_dir}", "-y"],
            check=True
        )

        if model_path.exists():
            print(f"  Saved: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
            return True
        else:
            print(f"  Extracted to: {mscoco_dir}")
            return True

    except ImportError:
        print("  Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_custom_diffusion(output_dir: Path, info: dict) -> bool:
    """Download Custom Diffusion model weights from HuggingFace.

    Base SD v1.4 UNet is auto-downloaded by diffusers on first use.
    This downloads the AbC benchmark custom diffusion weights.
    """
    import shutil
    import subprocess

    abc_dir = output_dir / "custom_diffusion" / "abc"
    marker_file = abc_dir / ".download_complete"

    if marker_file.exists():
        print(f"  Status: Already exists at {abc_dir}")
        return True

    abc_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        print("  Base SD v1.4 UNet auto-downloads on first use via diffusers")
        print("  Downloading AbC benchmark data from HuggingFace...")

        # Download the models archive
        archive_path = hf_hub_download(
            repo_id=info["hf_repo"],
            filename="custom_diffusion/abc/models_test/models_test.7z.001",
            repo_type="dataset",
        )

        # Find 7z executable
        sevenz = shutil.which("7z") or shutil.which("7zz")
        if not sevenz:
            print("  Error: 7z not found. Install p7zip:")
            print("    brew install p7zip (mac) / apt install p7zip-full (linux)")
            print(f"  Archive downloaded to: {archive_path}")
            return False

        print(f"  Extracting with {sevenz}...")
        subprocess.run(
            [sevenz, "x", archive_path, f"-o{abc_dir}", "-y"],
            check=True
        )

        # Create marker file
        marker_file.touch()
        print(f"  Extracted to: {abc_dir}")
        return True

    except ImportError:
        print("  Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


# Registry of download functions
DOWNLOADERS = {
    "custom_diffusion": download_custom_diffusion,
    "dmd2": download_dmd2,
    "edm": download_edm,
    "mscoco": download_mscoco,
}
