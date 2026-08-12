"""Thin subprocess wrapper around the official ACE-Step training CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_REPOSITORY = "https://github.com/ace-step/ACE-Step-1.5.git"
DEFAULT_REVISION = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0"


def _run(command: Iterable[str], *, cwd: str | Path | None = None) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"$ {printable}")
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def ensure_acestep_repo(
    repo_dir: str | Path,
    *,
    repository: str = DEFAULT_REPOSITORY,
    revision: str | None = None,
) -> Path:
    destination = Path(repo_dir).expanduser().resolve()
    if not (destination / ".git").is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", repository, str(destination)])
    if revision:
        _run(["git", "checkout", revision], cwd=destination)
    if not (destination / "train.py").is_file():
        raise FileNotFoundError(f"train.py was not found in {destination}")
    return destination


def apply_acestep_patch(repo_dir: str | Path, patch_file: str | Path) -> None:
    """Apply the MOHIM dual-stream patch once to an ACE-Step checkout.

    Args:
        repo_dir: ACE-Step repository checked out at ``DEFAULT_REVISION``.
        patch_file: Patch containing the MOHIM dual-stream training changes.

    Raises:
        FileNotFoundError: If the patch file is missing.
        subprocess.CalledProcessError: If the patch cannot be applied cleanly.
    """
    root = Path(repo_dir).expanduser().resolve()
    patch = Path(patch_file).expanduser().resolve()
    if not patch.is_file():
        raise FileNotFoundError(f"ACE-Step patch was not found: {patch}")

    reverse_check = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if reverse_check.returncode == 0:
        return
    _run(["git", "apply", "--check", str(patch)], cwd=root)
    _run(["git", "apply", str(patch)], cwd=root)


def install_acestep(repo_dir: str | Path) -> None:
    """Install the patched ACE-Step checkout without rewriting its requirements file."""

    root = Path(repo_dir).expanduser().resolve()
    requirements = root / "requirements.txt"
    if requirements.is_file():
        filtered = Path("/tmp/mohim_acestep_requirements.txt")
        lines = [line for line in requirements.read_text(encoding="utf-8").splitlines() if "flash-attn" not in line]
        filtered.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _run([sys.executable, "-m", "pip", "install", "-r", str(filtered)])
    _run([sys.executable, "-m", "pip", "install", "-e", str(root), "--no-deps"])
    nano_vllm = root / "acestep" / "third_parts" / "nano-vllm"
    if nano_vllm.is_dir():
        _run([sys.executable, "-m", "pip", "install", "--no-deps", str(nano_vllm)])


def preprocess_dual_stream(
    *,
    repo_dir: str | Path,
    manifest_path: str | Path,
    checkpoint_dir: str | Path,
    tensor_dir: str | Path,
    model_variant: str = "base",
    max_duration: float = 300.0,
    device: str = "cuda",
    precision: str = "bf16",
) -> None:
    root = Path(repo_dir).expanduser().resolve()
    command = [
        sys.executable,
        "train.py",
        "fixed",
        "--checkpoint-dir",
        str(Path(checkpoint_dir).expanduser().resolve()),
        "--model-variant",
        model_variant,
        "--preprocess",
        "--dual-stream",
        "--dataset-json",
        str(Path(manifest_path).expanduser().resolve()),
        "--tensor-output",
        str(Path(tensor_dir).expanduser().resolve()),
        "--max-duration",
        str(max_duration),
        "--device",
        device,
        "--precision",
        precision,
    ]
    _run(command, cwd=root)


def train_lora(
    *,
    repo_dir: str | Path,
    checkpoint_dir: str | Path,
    tensor_dir: str | Path,
    output_dir: str | Path,
    model_variant: str = "base",
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    batch_size: int = 1,
    gradient_accumulation: int = 4,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    save_every: int = 1,
    device: str = "cuda",
    precision: str = "bf16",
) -> None:
    root = Path(repo_dir).expanduser().resolve()
    command = [
        sys.executable,
        "train.py",
        "fixed",
        "--checkpoint-dir",
        str(Path(checkpoint_dir).expanduser().resolve()),
        "--model-variant",
        model_variant,
        "--dataset-dir",
        str(Path(tensor_dir).expanduser().resolve()),
        "--output-dir",
        str(Path(output_dir).expanduser().resolve()),
        "--dual-stream",
        "--attention-type",
        "cross",
        "--rank",
        str(rank),
        "--alpha",
        str(alpha),
        "--dropout",
        str(dropout),
        "--batch-size",
        str(batch_size),
        "--gradient-accumulation",
        str(gradient_accumulation),
        "--epochs",
        str(epochs),
        "--save-every",
        str(save_every),
        "--lr",
        str(learning_rate),
        "--num-workers",
        "0",
        "--device",
        device,
        "--precision",
        precision,
    ]
    _run(command, cwd=root)
