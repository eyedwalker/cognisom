"""
GPU Backend Detection
=====================

Detects CuPy availability and provides a unified array interface.
All GPU kernels use `xp` (the active array module) so the same
code runs on GPU (CuPy) or CPU (NumPy) with zero changes.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Global backend state ─────────────────────────────────────────

_backend: Optional["GPUBackend"] = None


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
    global _backend
    if _backend is not None:
        return _backend

    if force_cpu:
        _backend = _make_cpu_backend()
        logger.info("GPU backend: forced CPU mode")
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
