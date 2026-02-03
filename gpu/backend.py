"""
GPU Backend Detection
=====================

Detects CuPy availability and provides a unified array interface.
All GPU kernels use `xp` (the active array module) so the same
code runs on GPU (CuPy) or CPU (NumPy) with zero changes.

Runtime GPU Toggle:
    The GPU can be disabled at runtime via environment variable or
    the set_gpu_enabled() function. This allows switching between
    GPU and CPU mode without restarting the application.

    # Disable GPU at runtime:
    from cognisom.gpu.backend import set_gpu_enabled
    set_gpu_enabled(False)

    # Re-enable:
    set_gpu_enabled(True)
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Global backend state ─────────────────────────────────────────

_backend: Optional["GPUBackend"] = None
_gpu_enabled: Optional[bool] = None  # None = auto-detect, True/False = forced


@dataclass
class GPUBackend:
    """Holds the active array module and device info."""

    xp: object  # numpy or cupy module
    has_gpu: bool
    device_name: str
    device_memory_gb: float
    cupy_version: str

    def to_numpy(self, arr) -> np.ndarray:
        """Convert any array to numpy (no-op if already numpy)."""
        if self.has_gpu:
            import cupy as cp
            if isinstance(arr, cp.ndarray):
                return cp.asnumpy(arr)
        return np.asarray(arr)

    def to_device(self, arr):
        """Move numpy array to GPU (no-op if no GPU)."""
        if self.has_gpu:
            import cupy as cp
            if isinstance(arr, np.ndarray):
                return cp.asarray(arr)
        return arr

    def synchronize(self):
        """Wait for all GPU operations to complete."""
        if self.has_gpu:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()

    def summary(self) -> str:
        if self.has_gpu:
            return (
                f"GPU: {self.device_name} ({self.device_memory_gb:.1f} GB) "
                f"— CuPy {self.cupy_version}"
            )
        return "CPU only (NumPy) — install CuPy for GPU acceleration"


def get_backend(force_cpu: bool = False) -> GPUBackend:
    """Detect and return the GPU backend (cached singleton).

    Args:
        force_cpu: If True, use NumPy even if CuPy is available.

    Returns:
        GPUBackend instance with `xp` set to cupy or numpy.
    """
    global _backend, _gpu_enabled

    # Check if we need to reinitialize due to toggle change
    if _backend is not None:
        # If GPU was toggled off and we're on GPU, reinitialize
        if _gpu_enabled is False and _backend.has_gpu:
            _backend = None
        # If GPU was toggled on and we're on CPU, reinitialize
        elif _gpu_enabled is True and not _backend.has_gpu:
            _backend = None
        else:
            return _backend

    # Determine if GPU should be used
    use_gpu = True
    if force_cpu:
        use_gpu = False
    elif _gpu_enabled is False:
        use_gpu = False
    elif os.environ.get("COGNISOM_GPU", "").lower() in ("0", "false", "no", "off"):
        use_gpu = False

    if not use_gpu:
        _backend = _make_cpu_backend()
        logger.info("GPU backend: CPU mode (GPU disabled)")
        return _backend

    try:
        import cupy as cp  # noqa: F401
        # Verify a device is actually available
        device = cp.cuda.Device(0)
        mem = device.mem_info
        free_gb = mem[0] / (1024 ** 3)
        total_gb = mem[1] / (1024 ** 3)
        device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()

        _backend = GPUBackend(
            xp=cp,
            has_gpu=True,
            device_name=device_name,
            device_memory_gb=total_gb,
            cupy_version=cp.__version__,
        )
        logger.info(f"GPU backend: {_backend.summary()}")
    except Exception as e:
        logger.info(f"GPU not available ({e}), using CPU backend")
        _backend = _make_cpu_backend()

    return _backend


def _make_cpu_backend() -> GPUBackend:
    return GPUBackend(
        xp=np,
        has_gpu=False,
        device_name="CPU",
        device_memory_gb=0.0,
        cupy_version="",
    )


def reset_backend():
    """Reset the cached backend (useful for testing)."""
    global _backend
    _backend = None


def set_gpu_enabled(enabled: bool) -> GPUBackend:
    """Enable or disable GPU at runtime.

    This allows switching between GPU and CPU mode without restarting.
    The setting is stored in a persistent file for cross-session use.

    Args:
        enabled: True to enable GPU (if available), False to force CPU

    Returns:
        The new GPUBackend instance

    Example:
        >>> from cognisom.gpu.backend import set_gpu_enabled, get_backend
        >>> set_gpu_enabled(False)  # Switch to CPU
        >>> backend = get_backend()
        >>> print(backend.has_gpu)  # False
    """
    global _backend, _gpu_enabled
    _gpu_enabled = enabled
    _backend = None  # Force reinit

    # Persist the setting
    _save_gpu_setting(enabled)

    return get_backend()


def get_gpu_enabled() -> bool:
    """Get the current GPU enabled state.

    Returns:
        True if GPU is enabled (or auto-detect), False if disabled
    """
    global _gpu_enabled
    if _gpu_enabled is None:
        _gpu_enabled = _load_gpu_setting()
    return _gpu_enabled is not False


def is_gpu_available() -> bool:
    """Check if a GPU is actually available (hardware check).

    Returns:
        True if CuPy can access a CUDA device
    """
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


def _get_settings_path() -> Path:
    """Get the path to the GPU settings file."""
    data_dir = Path(os.environ.get(
        "COGNISOM_DATA_DIR",
        Path(__file__).resolve().parent.parent.parent / "data"
    ))
    return data_dir / "settings" / "gpu_config.txt"


def _save_gpu_setting(enabled: bool):
    """Persist GPU setting to disk."""
    try:
        path = _get_settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("enabled" if enabled else "disabled")
    except Exception as e:
        logger.warning(f"Could not save GPU setting: {e}")


def _load_gpu_setting() -> Optional[bool]:
    """Load GPU setting from disk. Returns None if not set."""
    try:
        path = _get_settings_path()
        if path.exists():
            content = path.read_text().strip().lower()
            if content == "disabled":
                return False
            elif content == "enabled":
                return True
    except Exception:
        pass
    return None  # Auto-detect
