"""
Cognisom Physics Package
========================

GPU-accelerated physics simulation using NVIDIA Warp.

This package provides:
- Differentiable physics kernels (autodiff through simulations)
- Cell mechanics (repulsion, adhesion, chemotaxis)
- Soft-body dynamics (cell deformation)
- Fluid-structure interaction (blood flow, interstitial fluid)

Backends:
- Warp: Primary backend with autodifferentiation support
- CuPy: Fallback for systems without Warp
- NumPy: CPU fallback

Phase A of the Strategic Implementation Plan.

Usage::

    from cognisom.physics import WarpBackend, CellMechanics

    # Initialize Warp backend
    backend = WarpBackend()

    # Create cell mechanics solver
    mechanics = CellMechanics(backend)
    mechanics.add_cells(positions, radii)

    # Simulate with force computation
    for step in range(1000):
        mechanics.compute_forces()
        mechanics.integrate(dt=0.001)

    # Get gradients (for parameter optimization)
    grads = mechanics.compute_gradients(loss_fn)
"""

from .warp_backend import (
    WarpBackend,
    WarpConfig,
    check_warp_available,
    get_warp_device,
)

from .cell_mechanics import (
    CellMechanics,
    CellMechanicsConfig,
    ForceType,
)

from .warp_diffusion import (
    WarpDiffusionSolver,
    laplacian_3d_warp,
)

from .warp_ssa import (
    WarpSSASolver,
    tau_leap_warp,
)

from .integrators import (
    verlet_integrate,
    xpbd_integrate,
)

__version__ = "0.1.0"

__all__ = [
    # Backend
    "WarpBackend",
    "WarpConfig",
    "check_warp_available",
    "get_warp_device",
    # Cell mechanics
    "CellMechanics",
    "CellMechanicsConfig",
    "ForceType",
    # Diffusion
    "WarpDiffusionSolver",
    "laplacian_3d_warp",
    # SSA
    "WarpSSASolver",
    "tau_leap_warp",
    # Integrators
    "verlet_integrate",
    "xpbd_integrate",
]
