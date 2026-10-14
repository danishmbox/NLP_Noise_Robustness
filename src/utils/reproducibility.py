"""
Reproducibility utilities for NLP Noise Robustness research.

This module provides deterministic seed configuration across Python, NumPy,
and PyTorch, along with runtime environment metadata collection and documentation
of hardware-level reproducibility limitations.
"""

import os
import sys
import random
import platform
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np


def set_seed(seed: int = 42) -> Dict[str, Any]:
    """
    Sets random seeds across Python, NumPy, and PyTorch (if available).

    Args:
        seed: Integer random seed (default: 42).

    Returns:
        Dict detailing the components configured.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    status = {
        "seed": seed,
        "python_random": True,
        "numpy_random": True,
        "os_pythonhashseed": True,
        "pytorch_seeded": False,
        "cuda_deterministic": False
    }

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            status["cuda_deterministic"] = True
        status["pytorch_seeded"] = True
    except ImportError:
        # PyTorch not installed in current environment
        pass

    return status


def get_git_commit_hash() -> Optional[str]:
    """
    Attempts to retrieve the current git commit hash safely.

    Returns:
        Short commit hash string, or None if git is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def get_runtime_metadata() -> Dict[str, Any]:
    """
    Collects runtime environment, hardware, and dependency metadata.

    Returns:
        Dict of environment details.
    """
    meta: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "git_commit": get_git_commit_hash()
    }

    # Record installed library versions
    packages = [
        "scikit-learn",
        "numpy",
        "pandas",
        "joblib",
        "scipy",
        "matplotlib",
        "seaborn",
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "evaluate"
    ]

    installed_versions: Dict[str, str] = {}
    for pkg in packages:
        module_name = pkg.replace("-", "_")
        try:
            mod = __import__(module_name)
            installed_versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            installed_versions[pkg] = "not_installed"

    meta["dependencies"] = installed_versions

    # CUDA device info if available
    try:
        import torch
        if torch.cuda.is_available():
            meta["cuda_available"] = True
            meta["device_name"] = torch.cuda.get_device_name(0)
            meta["device_count"] = torch.cuda.device_count()
        else:
            meta["cuda_available"] = False
    except ImportError:
        meta["cuda_available"] = False

    return meta


def get_reproducibility_caveats() -> Dict[str, str]:
    """
    Returns documented limitations to bitwise reproducibility.
    """
    return {
        "gpu_non_determinism": (
            "Certain PyTorch CUDA convolution and reduction algorithms have non-deterministic "
            "atomic operations even when torch.backends.cudnn.deterministic is enabled."
        ),
        "mixed_precision": (
            "FP16 and BF16 mixed-precision floating-point arithmetic can introduce minor rounding "
            "divergence compared to standard FP32 operations."
        ),
        "hardware_variation": (
            "Executing on different GPU microarchitectures (e.g. Tesla T4 vs A100 vs V100) "
            "yields numerical precision differences in matrix multiplication."
        )
    }
