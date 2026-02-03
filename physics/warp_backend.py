"""
NVIDIA Warp Backend
===================

Wrapper for NVIDIA Warp providing:
- Device detection and management
- Array conversion utilities
- Automatic fallback to CuPy/NumPy
- Gradient tape management for autodiff

Warp is a Python framework for high-performance simulation and graphics code.
It supports autodifferentiation through GPU kernels, enabling gradient-based
optimization of simulation parameters.

Installation:
    pip install warp-lang

Phase A.1 of the Strategic Implementation Plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

# Global state
_warp_backend: Optional["WarpBackend"] = None
_warp_available: Optional[bool] = None


class WarpDeviceType(str, Enum):
    """Warp device types."""
    CUDA = "cuda"
    CPU = "cpu"


@dataclass
class WarpConfig:
    """Configuration for Warp backend."""
    device: str = "cuda:0"  # Device string (cuda:0, cuda:1, cpu)
    enable_backward: bool = True  # Enable gradient computation
    kernel_cache_dir: str = ""  # Custom kernel cache directory
    verbose: bool = False  # Verbose kernel compilation
    verify_fp: bool = False  # Verify floating point operations
    print_launches: bool = False  # Print kernel launches (debug)


def check_warp_available() -> bool:
    """
    Check if NVIDIA Warp is available.

    Returns
    -------
    bool
        True if Warp is installed and a CUDA device is available
    """
    global _warp_available

    if _warp_available is not None:
        return _warp_available

    try:
        import warp as wp
        wp.init()

        # Check for CUDA device
        if wp.is_cuda_available():
            _warp_available = True
            log.info(f"Warp available: CUDA device detected")
        else:
            _warp_available = True  # Warp works on CPU too
            log.info("Warp available: CPU mode (no CUDA)")

    except ImportError:
        _warp_available = False
        log.info("Warp not available: install with 'pip install warp-lang'")
    except Exception as e:
        _warp_available = False
        log.warning(f"Warp initialization failed: {e}")

    return _warp_available


def get_warp_device() -> str:
    """
    Get the best available Warp device.

    Returns
    -------
    str
        Device string (e.g., "cuda:0" or "cpu")
    """
    if not check_warp_available():
        return "cpu"

    try:
        import warp as wp
        if wp.is_cuda_available():
            return "cuda:0"
    except Exception:
        pass

    return "cpu"


class WarpBackend:
    """
    Unified backend for NVIDIA Warp.

    Provides:
    - Device management
    - Array creation and conversion
    - Gradient tape for autodiff
    - Kernel launching utilities

    Examples
    --------
    >>> backend = WarpBackend()
    >>> arr = backend.zeros((100, 3), dtype=float)
    >>> with backend.tape() as tape:
    ...     result = my_kernel(arr)
    >>> grads = tape.gradients
    """

    def __init__(self, config: Optional[WarpConfig] = None):
        """
        Initialize Warp backend.

        Parameters
        ----------
        config : WarpConfig, optional
            Backend configuration
        """
        self.config = config or WarpConfig()
        self._initialized = False
        self._device = None
        self._wp = None  # Warp module reference

        self._initialize()

    def _initialize(self):
        """Initialize Warp runtime."""
        if not check_warp_available():
            log.warning("Warp not available, using fallback mode")
            self._initialized = False
            return

        try:
            import warp as wp

            # Initialize Warp
            wp.init()

            # Configure options
            if self.config.kernel_cache_dir:
                wp.config.kernel_cache_dir = self.config.kernel_cache_dir
            wp.config.verbose = self.config.verbose
            wp.config.verify_fp = self.config.verify_fp
            wp.config.print_launches = self.config.print_launches

            # Set device
            if self.config.device.startswith("cuda") and wp.is_cuda_available():
                self._device = self.config.device
            else:
                self._device = "cpu"
                if self.config.device.startswith("cuda"):
                    log.warning(f"CUDA not available, using CPU")

            self._wp = wp
            self._initialized = True

            log.info(f"Warp backend initialized on {self._device}")

        except Exception as e:
            log.error(f"Warp initialization failed: {e}")
            self._initialized = False

    @property
    def is_available(self) -> bool:
        """Check if backend is initialized."""
        return self._initialized

    @property
    def device(self) -> str:
        """Get current device string."""
        return self._device or "cpu"

    @property
    def wp(self):
        """Get Warp module reference."""
        if not self._initialized:
            raise RuntimeError("Warp backend not initialized")
        return self._wp

    def zeros(
        self,
        shape: Tuple[int, ...],
        dtype: type = float,
        requires_grad: bool = False,
    ):
        """
        Create a zero-filled Warp array.

        Parameters
        ----------
        shape : tuple
            Array shape
        dtype : type
            Data type (float, int, etc.)
        requires_grad : bool
            Whether to track gradients

        Returns
        -------
        wp.array
            Warp array
        """
        if not self._initialized:
            return np.zeros(shape, dtype=np.float32 if dtype == float else dtype)

        wp = self._wp
        wp_dtype = self._to_warp_dtype(dtype)

        return wp.zeros(
            shape,
            dtype=wp_dtype,
            device=self._device,
            requires_grad=requires_grad and self.config.enable_backward,
        )

    def array(
        self,
        data: np.ndarray,
        dtype: Optional[type] = None,
        requires_grad: bool = False,
    ):
        """
        Create a Warp array from numpy data.

        Parameters
        ----------
        data : np.ndarray
            Input numpy array
        dtype : type, optional
            Override data type
        requires_grad : bool
            Whether to track gradients

        Returns
        -------
        wp.array
            Warp array
        """
        if not self._initialized:
            return data.astype(np.float32 if dtype == float else dtype or data.dtype)

        wp = self._wp
        wp_dtype = self._to_warp_dtype(dtype or data.dtype)

        return wp.array(
            data,
            dtype=wp_dtype,
            device=self._device,
            requires_grad=requires_grad and self.config.enable_backward,
        )

    def to_numpy(self, arr) -> np.ndarray:
        """
        Convert Warp array to numpy.

        Parameters
        ----------
        arr : wp.array
            Warp array

        Returns
        -------
        np.ndarray
            Numpy array
        """
        if not self._initialized:
            return np.asarray(arr)

        wp = self._wp
        if isinstance(arr, wp.array):
            return arr.numpy()
        return np.asarray(arr)

    def synchronize(self):
        """Synchronize device (wait for all operations to complete)."""
        if self._initialized:
            self._wp.synchronize()

    def _to_warp_dtype(self, dtype):
        """Convert Python/numpy dtype to Warp dtype."""
        wp = self._wp

        if dtype in (float, np.float32, np.float64):
            return wp.float32
        elif dtype in (int, np.int32, np.int64):
            return wp.int32
        elif dtype == np.uint32:
            return wp.uint32
        elif dtype == bool:
            return wp.bool
        else:
            return wp.float32

    def tape(self) -> "GradientTape":
        """
        Create a gradient tape for autodiff.

        Returns
        -------
        GradientTape
            Context manager for gradient computation

        Examples
        --------
        >>> with backend.tape() as tape:
        ...     loss = simulate(params)
        >>> grads = tape.gradients[params]
        """
        return GradientTape(self)

    def launch_kernel(
        self,
        kernel: Callable,
        dim: int,
        inputs: List[Any],
        outputs: List[Any] = None,
    ):
        """
        Launch a Warp kernel.

        Parameters
        ----------
        kernel : Callable
            Warp kernel function
        dim : int
            Number of threads
        inputs : List
            Input arrays
        outputs : List, optional
            Output arrays (for in-place kernels, same as inputs)
        """
        if not self._initialized:
            raise RuntimeError("Warp backend not initialized")

        wp = self._wp
        wp.launch(
            kernel,
            dim=dim,
            inputs=inputs,
            outputs=outputs or [],
            device=self._device,
        )

    def get_memory_info(self) -> Dict[str, float]:
        """
        Get device memory information.

        Returns
        -------
        dict
            Memory info with 'free_gb' and 'total_gb' keys
        """
        if not self._initialized or not self._device.startswith("cuda"):
            return {"free_gb": 0, "total_gb": 0}

        try:
            import cupy as cp
            mem = cp.cuda.Device(0).mem_info
            return {
                "free_gb": mem[0] / (1024**3),
                "total_gb": mem[1] / (1024**3),
            }
        except Exception:
            return {"free_gb": 0, "total_gb": 0}

    def summary(self) -> str:
        """Get backend summary string."""
        if not self._initialized:
            return "Warp backend: not initialized (fallback to NumPy)"

        mem = self.get_memory_info()
        if mem["total_gb"] > 0:
            return f"Warp backend: {self._device} ({mem['total_gb']:.1f} GB)"
        return f"Warp backend: {self._device}"


class GradientTape:
    """
    Context manager for gradient computation.

    Records operations and enables backward pass through
    Warp kernels for autodifferentiation.

    Examples
    --------
    >>> with backend.tape() as tape:
    ...     # Forward pass
    ...     positions = simulate(params, steps=100)
    ...     loss = compute_loss(positions, target)
    ...
    >>> # Backward pass
    >>> tape.backward(loss)
    >>> param_grads = tape.gradients[params]
    """

    def __init__(self, backend: WarpBackend):
        self.backend = backend
        self._tape = None
        self._loss = None

    def __enter__(self) -> "GradientTape":
        if self.backend.is_available and self.backend.config.enable_backward:
            wp = self.backend.wp
            self._tape = wp.Tape()
            self._tape.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._tape is not None:
            self._tape.__exit__(exc_type, exc_val, exc_tb)

    def backward(self, loss=None):
        """
        Compute gradients via backward pass.

        Parameters
        ----------
        loss : wp.array, optional
            Loss value to differentiate (if not recorded in tape)
        """
        if self._tape is None:
            log.warning("No tape recorded, gradients not available")
            return

        if loss is not None:
            self._loss = loss

        self._tape.backward(self._loss)

    @property
    def gradients(self) -> Dict[Any, Any]:
        """
        Get computed gradients.

        Returns
        -------
        dict
            Mapping of arrays to their gradients
        """
        if self._tape is None:
            return {}
        return self._tape.gradients

    def zero_gradients(self):
        """Zero all gradients in the tape."""
        if self._tape is not None:
            self._tape.zero()


def get_warp_backend(config: Optional[WarpConfig] = None) -> WarpBackend:
    """
    Get the global Warp backend instance (singleton).

    Parameters
    ----------
    config : WarpConfig, optional
        Configuration (only used on first call)

    Returns
    -------
    WarpBackend
        Global backend instance
    """
    global _warp_backend

    if _warp_backend is None:
        _warp_backend = WarpBackend(config)

    return _warp_backend
