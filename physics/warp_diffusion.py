"""
Warp-Accelerated 3D Diffusion Solver
====================================

Port of the CuPy diffusion kernel to NVIDIA Warp with autodifferentiation
support. This enables gradient computation through diffusion for:

- Parameter optimization (diffusion coefficients)
- Inverse problems (reconstruct sources from observations)
- Sensitivity analysis (how parameter changes affect outcomes)

The kernel computes the 7-point Laplacian stencil on a 3D grid and applies
explicit Euler integration for the diffusion equation:

    ∂C/∂t = D * ∇²C

Phase A.1 of the Strategic Implementation Plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .warp_backend import WarpBackend, get_warp_backend, check_warp_available

log = logging.getLogger(__name__)

# Check if Warp is available at module load time
_WARP_AVAILABLE = check_warp_available()

if _WARP_AVAILABLE:
    import warp as wp

    # ── Warp Kernels ─────────────────────────────────────────────────

    @wp.kernel
    def laplacian_3d_kernel(
        C: wp.array3d(dtype=wp.float32),
        L: wp.array3d(dtype=wp.float32),
    ):
        """
        Compute 3D Laplacian using 7-point stencil.

        L[i,j,k] = C[i+1,j,k] + C[i-1,j,k] + C[i,j+1,k] + C[i,j-1,k]
                 + C[i,j,k+1] + C[i,j,k-1] - 6*C[i,j,k]

        Boundary voxels are skipped (remain zero).
        """
        i, j, k = wp.tid()

        # Get grid dimensions
        nx = C.shape[0]
        ny = C.shape[1]
        nz = C.shape[2]

        # Skip boundary voxels
        if i < 1 or i >= nx - 1:
            return
        if j < 1 or j >= ny - 1:
            return
        if k < 1 or k >= nz - 1:
            return

        center = C[i, j, k]

        lap = (
            C[i + 1, j, k] + C[i - 1, j, k] +
            C[i, j + 1, k] + C[i, j - 1, k] +
            C[i, j, k + 1] + C[i, j, k - 1] -
            6.0 * center
        )

        L[i, j, k] = lap

    @wp.kernel
    def diffusion_step_kernel(
        C: wp.array3d(dtype=wp.float32),
        L: wp.array3d(dtype=wp.float32),
        C_out: wp.array3d(dtype=wp.float32),
        D_over_dx2: wp.float32,
        dt: wp.float32,
    ):
        """
        Apply explicit Euler diffusion update with Laplacian.

        C_out = max(C + D/dx² * L * dt, 0)
        """
        i, j, k = wp.tid()

        nx = C.shape[0]
        ny = C.shape[1]
        nz = C.shape[2]

        if i >= nx or j >= ny or k >= nz:
            return

        new_val = C[i, j, k] + D_over_dx2 * L[i, j, k] * dt
        C_out[i, j, k] = wp.max(new_val, 0.0)

    @wp.kernel
    def add_source_kernel(
        C: wp.array3d(dtype=wp.float32),
        source_idx: wp.array(dtype=wp.vec3i),
        source_rate: wp.array(dtype=wp.float32),
        dt: wp.float32,
    ):
        """Add sources to concentration field."""
        tid = wp.tid()

        idx = source_idx[tid]
        rate = source_rate[tid]

        i, j, k = idx[0], idx[1], idx[2]

        # Bounds check
        if i < 0 or i >= C.shape[0]:
            return
        if j < 0 or j >= C.shape[1]:
            return
        if k < 0 or k >= C.shape[2]:
            return

        wp.atomic_add(C, i, j, k, rate * dt)

    @wp.kernel
    def apply_sink_kernel(
        C: wp.array3d(dtype=wp.float32),
        sink_idx: wp.array(dtype=wp.vec3i),
        sink_rate: wp.array(dtype=wp.float32),
        dt: wp.float32,
    ):
        """Apply sinks to concentration field."""
        tid = wp.tid()

        idx = sink_idx[tid]
        rate = sink_rate[tid]

        i, j, k = idx[0], idx[1], idx[2]

        if i < 0 or i >= C.shape[0]:
            return
        if j < 0 or j >= C.shape[1]:
            return
        if k < 0 or k >= C.shape[2]:
            return

        current = C[i, j, k]
        removal = wp.min(rate * dt, current)
        C[i, j, k] = current - removal


@dataclass
class WarpDiffusionConfig:
    """Configuration for Warp diffusion solver."""
    device: str = "cuda:0"
    enable_gradients: bool = True


class WarpDiffusionSolver:
    """
    GPU-accelerated 3D diffusion solver using NVIDIA Warp.

    Features:
    - 7-point Laplacian stencil
    - Explicit Euler time integration
    - Source and sink terms
    - Autodifferentiation through simulation

    Examples
    --------
    >>> solver = WarpDiffusionSolver(grid_shape=(200, 200, 100))
    >>> solver.set_diffusion_coeff(2000.0)  # um²/s
    >>> solver.set_resolution(10.0)  # um
    >>>
    >>> # Run diffusion
    >>> for step in range(100):
    ...     solver.step(dt=0.001)
    >>>
    >>> # Get concentration
    >>> C = solver.get_concentration()
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        config: Optional[WarpDiffusionConfig] = None,
        backend: Optional[WarpBackend] = None,
    ):
        """
        Initialize diffusion solver.

        Parameters
        ----------
        grid_shape : tuple
            Grid dimensions (nx, ny, nz)
        config : WarpDiffusionConfig, optional
            Solver configuration
        backend : WarpBackend, optional
            Warp backend instance
        """
        self.grid_shape = grid_shape
        self.config = config or WarpDiffusionConfig()
        self.backend = backend or get_warp_backend()

        self.diffusion_coeff = 2000.0  # um²/s
        self.resolution = 10.0  # um

        # Arrays
        self._C: Optional[wp.array3d] = None  # Concentration
        self._L: Optional[wp.array3d] = None  # Laplacian
        self._C_temp: Optional[wp.array3d] = None  # Temp for updates

        self._initialized = False

        if self.backend.is_available:
            self._initialize()
        else:
            log.warning("Warp not available, using NumPy fallback")

    def _initialize(self):
        """Initialize Warp arrays."""
        nx, ny, nz = self.grid_shape
        requires_grad = self.config.enable_gradients

        self._C = self.backend.wp.zeros(
            (nx, ny, nz),
            dtype=wp.float32,
            device=self.backend.device,
            requires_grad=requires_grad,
        )
        self._L = self.backend.wp.zeros(
            (nx, ny, nz),
            dtype=wp.float32,
            device=self.backend.device,
        )
        self._C_temp = self.backend.wp.zeros(
            (nx, ny, nz),
            dtype=wp.float32,
            device=self.backend.device,
        )

        self._initialized = True
        log.debug(f"WarpDiffusionSolver initialized: {self.grid_shape}")

    def set_diffusion_coeff(self, D: float):
        """Set diffusion coefficient (um²/s)."""
        self.diffusion_coeff = D

    def set_resolution(self, dx: float):
        """Set voxel resolution (um)."""
        self.resolution = dx

    def set_concentration(self, C: np.ndarray):
        """
        Set concentration field from numpy array.

        Parameters
        ----------
        C : np.ndarray
            Concentration field (nx, ny, nz)
        """
        if not self._initialized:
            self._C = C.astype(np.float32).copy()
            return

        C_np = np.ascontiguousarray(C, dtype=np.float32)
        self._C = self.backend.wp.array(
            C_np,
            dtype=wp.float32,
            device=self.backend.device,
            requires_grad=self.config.enable_gradients,
        )

    def get_concentration(self) -> np.ndarray:
        """
        Get concentration field as numpy array.

        Returns
        -------
        np.ndarray
            Concentration field
        """
        if not self._initialized:
            return np.asarray(self._C)
        return self._C.numpy()

    def step(self, dt: float):
        """
        Advance diffusion by one time step.

        Parameters
        ----------
        dt : float
            Time step (hours)
        """
        if not self._initialized:
            self._step_numpy(dt)
            return

        # Compute Laplacian
        wp.launch(
            laplacian_3d_kernel,
            dim=self.grid_shape,
            inputs=[self._C, self._L],
            device=self.backend.device,
        )

        # Apply diffusion update
        D_over_dx2 = self.diffusion_coeff / (self.resolution ** 2)

        wp.launch(
            diffusion_step_kernel,
            dim=self.grid_shape,
            inputs=[self._C, self._L, self._C_temp, D_over_dx2, dt],
            device=self.backend.device,
        )

        # Swap buffers
        self._C, self._C_temp = self._C_temp, self._C

    def _step_numpy(self, dt: float):
        """Fallback numpy implementation."""
        C = np.asarray(self._C)

        # Laplacian
        L = np.zeros_like(C)
        L[1:-1, 1:-1, 1:-1] = (
            C[2:, 1:-1, 1:-1] + C[:-2, 1:-1, 1:-1] +
            C[1:-1, 2:, 1:-1] + C[1:-1, :-2, 1:-1] +
            C[1:-1, 1:-1, 2:] + C[1:-1, 1:-1, :-2] -
            6 * C[1:-1, 1:-1, 1:-1]
        )

        # Update
        D_over_dx2 = self.diffusion_coeff / (self.resolution ** 2)
        C += D_over_dx2 * L * dt
        C = np.maximum(C, 0)

        self._C = C

    def add_source(self, position: Tuple[int, int, int], rate: float, dt: float):
        """Add a source term at a position."""
        if not self._initialized:
            i, j, k = position
            if 0 <= i < self._C.shape[0] and 0 <= j < self._C.shape[1] and 0 <= k < self._C.shape[2]:
                self._C[i, j, k] += rate * dt
            return

        # For efficiency with many sources, use batch add_sources
        C_np = self._C.numpy()
        i, j, k = position
        if 0 <= i < C_np.shape[0] and 0 <= j < C_np.shape[1] and 0 <= k < C_np.shape[2]:
            C_np[i, j, k] += rate * dt
        self.set_concentration(C_np)

    def add_sink(self, position: Tuple[int, int, int], rate: float, dt: float):
        """Add a sink term at a position."""
        if not self._initialized:
            i, j, k = position
            if 0 <= i < self._C.shape[0] and 0 <= j < self._C.shape[1] and 0 <= k < self._C.shape[2]:
                removal = min(rate * dt, self._C[i, j, k])
                self._C[i, j, k] -= removal
            return

        C_np = self._C.numpy()
        i, j, k = position
        if 0 <= i < C_np.shape[0] and 0 <= j < C_np.shape[1] and 0 <= k < C_np.shape[2]:
            removal = min(rate * dt, C_np[i, j, k])
            C_np[i, j, k] -= removal
        self.set_concentration(C_np)

    def synchronize(self):
        """Wait for all GPU operations to complete."""
        if self._initialized:
            self.backend.synchronize()


def laplacian_3d_warp(
    concentration: np.ndarray,
    backend: Optional[WarpBackend] = None,
) -> np.ndarray:
    """
    Compute 3D Laplacian using Warp.

    Parameters
    ----------
    concentration : np.ndarray
        3D concentration field
    backend : WarpBackend, optional
        Warp backend

    Returns
    -------
    np.ndarray
        Laplacian field
    """
    backend = backend or get_warp_backend()

    if not backend.is_available:
        # Fallback to numpy
        L = np.zeros_like(concentration)
        C = concentration
        L[1:-1, 1:-1, 1:-1] = (
            C[2:, 1:-1, 1:-1] + C[:-2, 1:-1, 1:-1] +
            C[1:-1, 2:, 1:-1] + C[1:-1, :-2, 1:-1] +
            C[1:-1, 1:-1, 2:] + C[1:-1, 1:-1, :-2] -
            6 * C[1:-1, 1:-1, 1:-1]
        )
        return L

    # Warp implementation
    C_np = np.ascontiguousarray(concentration, dtype=np.float32)
    C_wp = wp.array(C_np, dtype=wp.float32, device=backend.device)
    L_wp = wp.zeros_like(C_wp)

    wp.launch(
        laplacian_3d_kernel,
        dim=C_wp.shape,
        inputs=[C_wp, L_wp],
        device=backend.device,
    )

    return L_wp.numpy()


def diffusion_step(
    concentration: np.ndarray,
    diffusion_coeff: float,
    dt: float,
    dx: float,
    backend: Optional[WarpBackend] = None,
) -> np.ndarray:
    """
    Single diffusion step using Warp.

    Parameters
    ----------
    concentration : np.ndarray
        Current concentration field
    diffusion_coeff : float
        Diffusion coefficient (um²/s)
    dt : float
        Time step (hours)
    dx : float
        Voxel resolution (um)
    backend : WarpBackend, optional
        Warp backend

    Returns
    -------
    np.ndarray
        Updated concentration field
    """
    backend = backend or get_warp_backend()

    if not backend.is_available:
        # Fallback
        C = concentration.copy()
        L = laplacian_3d_warp(C)
        D_over_dx2 = diffusion_coeff / (dx ** 2)
        C += D_over_dx2 * L * dt
        return np.maximum(C, 0)

    C_np = np.ascontiguousarray(concentration, dtype=np.float32)
    C_wp = wp.array(C_np, dtype=wp.float32, device=backend.device)
    L_wp = wp.zeros_like(C_wp)
    C_out = wp.zeros_like(C_wp)

    # Laplacian
    wp.launch(
        laplacian_3d_kernel,
        dim=C_wp.shape,
        inputs=[C_wp, L_wp],
        device=backend.device,
    )

    # Diffusion update
    D_over_dx2 = diffusion_coeff / (dx ** 2)
    wp.launch(
        diffusion_step_kernel,
        dim=C_wp.shape,
        inputs=[C_wp, L_wp, C_out, D_over_dx2, dt],
        device=backend.device,
    )

    return C_out.numpy()
