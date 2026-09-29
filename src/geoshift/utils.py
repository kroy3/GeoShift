"""Configuration, seeding and provenance helpers."""

from __future__ import annotations

import json
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def load_config(path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch; optionally request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def environment_info() -> Dict[str, Any]:
    """Software versions and git revision, stored with every experiment."""
    import torch_geometric

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        repo = Path(__file__).resolve().parent
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    return info
