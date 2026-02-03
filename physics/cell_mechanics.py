"""
Cell Mechanics Module
=====================

GPU-accelerated force-based cell mechanics using NVIDIA Warp.

Forces modeled:
- F_repulsion: Soft-sphere collision avoidance
- F_adhesion: E-cadherin mediated cell-cell adhesion
- F_chemotaxis: Gradient-following (CXCL12, etc.)
- F_random: Brownian motion (Langevin dynamics)

Physics:
    F_total = F_repulsion + F_adhesion + F_chemotaxis + F_random
    m * dv/dt = F_total - γ * v  (overdamped Langevin)

This module is central to the prostate cancer metastasis simulation,
where cells must physically migrate, adhere, and respond to chemical
gradients (CXCR4/CXCL12 for bone homing).

Phase A.2 of the Strategic Implementation Plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from .warp_backend import WarpBackend, get_warp_backend, check_warp_available

log = logging.getLogger(__name__)

_WARP_AVAILABLE = check_warp_available()

if _WARP_AVAILABLE:
    import warp as wp

    # ── Force computation kernels ─────────────────────────────────────

    @wp.kernel
    def compute_pairwise_forces_kernel(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        forces: wp.array(dtype=wp.vec3),
        radii: wp.array(dtype=wp.float32),
        cell_types: wp.array(dtype=wp.int32),
        k_repulsion: wp.float32,
        k_adhesion: wp.float32,
        adhesion_range: wp.float32,
        n_cells: wp.int32,
    ):
        """
        Compute pairwise forces between all cells (O(N²)).

        For larger simulations, use spatial hashing or neighbor lists.
        """
        i = wp.tid()

        if i >= n_cells:
            return

        pos_i = positions[i]
        r_i = radii[i]
        type_i = cell_types[i]

        f_total = wp.vec3(0.0, 0.0, 0.0)

        for j in range(n_cells):
            if i == j:
                continue

            pos_j = positions[j]
            r_j = radii[j]
            type_j = cell_types[j]

            # Vector from j to i
            d = pos_i - pos_j
            dist = wp.length(d)

            if dist < 1e-6:
                continue

            n = d / dist  # unit normal

            # Contact distance
            contact = r_i + r_j

            # Repulsion (soft-sphere)
            if dist < contact:
                overlap = contact - dist
                f_rep = k_repulsion * overlap * n
                f_total = f_total + f_rep

            # Adhesion (within adhesion range, beyond contact)
            if dist < contact + adhesion_range and dist >= contact:
                # Adhesion pulls cells together
                # Strength depends on cell types (E-cadherin expression)
                # For now, use uniform adhesion
                stretch = dist - contact
                f_adh = -k_adhesion * stretch * n
                f_total = f_total + f_adh

        forces[i] = f_total

    @wp.kernel
    def compute_chemotaxis_force_kernel(
        positions: wp.array(dtype=wp.vec3),
        forces: wp.array(dtype=wp.vec3),
        gradient_field: wp.array3d(dtype=wp.vec3),  # 3D vector field
        chemotaxis_strength: wp.array(dtype=wp.float32),  # per-cell sensitivity
        grid_origin: wp.vec3,
        grid_spacing: wp.float32,
        n_cells: wp.int32,
    ):
        """
        Compute chemotactic force from gradient field.

        F_chemotaxis = μ * ∇[concentration]
        """
        i = wp.tid()

        if i >= n_cells:
            return

        pos = positions[i]
        mu = chemotaxis_strength[i]

        if mu <= 0.0:
            return

        # Convert position to grid indices
        grid_pos = (pos - grid_origin) / grid_spacing

        # Get grid dimensions
        nx = gradient_field.shape[0]
        ny = gradient_field.shape[1]
        nz = gradient_field.shape[2]

        # Integer indices (clamped)
        ix = wp.clamp(wp.int32(grid_pos[0]), 0, nx - 1)
        iy = wp.clamp(wp.int32(grid_pos[1]), 0, ny - 1)
        iz = wp.clamp(wp.int32(grid_pos[2]), 0, nz - 1)

        # Sample gradient at cell position
        grad = gradient_field[ix, iy, iz]

        # Chemotactic force
        f_chem = mu * grad

        # Add to forces
        forces[i] = forces[i] + f_chem

    @wp.kernel
    def add_brownian_force_kernel(
        forces: wp.array(dtype=wp.vec3),
        rng_state: wp.array(dtype=wp.uint64),
        noise_strength: wp.float32,
        n_cells: wp.int32,
    ):
        """
        Add Brownian (thermal) noise to forces.

        F_random = √(2kT/γ) * ξ(t)

        where ξ is Gaussian white noise.
        """
        i = wp.tid()

        if i >= n_cells:
            return

        state = rng_state[i]

        # Generate 3 uniform random numbers
        def next_rand():
            nonlocal state
            state = state ^ (state << wp.uint64(13))
            state = state ^ (state >> wp.uint64(7))
            state = state ^ (state << wp.uint64(17))
            return wp.float32(state & wp.uint64(0xFFFFFFFF)) / 4294967296.0

        u1 = next_rand()
        u2 = next_rand()
        u3 = next_rand()
        u4 = next_rand()
        u5 = next_rand()
        u6 = next_rand()

        # Box-Muller transform for 3 Gaussian random numbers
        def box_muller(u1_: wp.float32, u2_: wp.float32) -> wp.float32:
            if u1_ < 1e-15:
                u1_ = wp.float32(1e-15)
            return wp.sqrt(-2.0 * wp.log(u1_)) * wp.cos(6.283185307 * u2_)

        noise = wp.vec3(
            box_muller(u1, u2),
            box_muller(u3, u4),
            box_muller(u5, u6),
        )

        forces[i] = forces[i] + noise_strength * noise

        # Update RNG state
        rng_state[i] = state


class ForceType(Flag):
    """Types of forces to include in simulation."""
    NONE = 0
    REPULSION = auto()
    ADHESION = auto()
    CHEMOTAXIS = auto()
    BROWNIAN = auto()
    ALL = REPULSION | ADHESION | CHEMOTAXIS | BROWNIAN


@dataclass
class CellMechanicsConfig:
    """Configuration for cell mechanics simulation."""
    # Device
    device: str = "cuda:0"

    # Force parameters
    k_repulsion: float = 100.0      # Repulsion stiffness (pN/um)
    k_adhesion: float = 10.0        # Adhesion stiffness (pN/um)
    adhesion_range: float = 2.0     # Adhesion distance beyond contact (um)
    chemotaxis_strength: float = 1.0  # Default chemotaxis sensitivity
    noise_strength: float = 0.1     # Brownian noise strength

    # Dynamics
    damping: float = 1.0            # Viscous damping coefficient (pN·s/um)
    mass: float = 1.0               # Cell mass (for inertial dynamics)
    overdamped: bool = True         # Use overdamped (Langevin) dynamics

    # Integration
    dt: float = 0.001               # Default time step (hours)

    # Forces to include
    force_types: ForceType = ForceType.ALL


class CellMechanics:
    """
    GPU-accelerated cell mechanics solver.

    Computes forces between cells and integrates equations of motion.

    Examples
    --------
    >>> mechanics = CellMechanics(n_cells=1000)
    >>> mechanics.set_positions(initial_positions)
    >>> mechanics.set_radii(np.full(1000, 5.0))  # 5 um radius
    >>>
    >>> for _ in range(10000):
    ...     mechanics.step(dt=0.001)
    >>>
    >>> final_positions = mechanics.get_positions()
    """

    def __init__(
        self,
        n_cells: int,
        config: Optional[CellMechanicsConfig] = None,
        backend: Optional[WarpBackend] = None,
    ):
        """
        Initialize cell mechanics solver.

        Parameters
        ----------
        n_cells : int
            Number of cells
        config : CellMechanicsConfig, optional
            Solver configuration
        backend : WarpBackend, optional
            Warp backend
        """
        self.n_cells = n_cells
        self.config = config or CellMechanicsConfig()
        self.backend = backend or get_warp_backend()

        # State arrays
        self._positions: Optional[np.ndarray] = None
        self._velocities: Optional[np.ndarray] = None
        self._forces: Optional[np.ndarray] = None
        self._radii: Optional[np.ndarray] = None
        self._cell_types: Optional[np.ndarray] = None
        self._chemotaxis_strength: Optional[np.ndarray] = None

        # Warp arrays
        self._pos_wp = None
        self._vel_wp = None
        self._forces_wp = None
        self._radii_wp = None
        self._types_wp = None
        self._chem_wp = None
        self._rng_state = None

        # Gradient field (for chemotaxis)
        self._gradient_field = None
        self._grid_origin = np.array([0.0, 0.0, 0.0])
        self._grid_spacing = 10.0

        self._initialized = False
        self._time = 0.0

        self._initialize()

    def _initialize(self):
        """Initialize arrays."""
        # NumPy arrays (always available)
        self._positions = np.zeros((self.n_cells, 3), dtype=np.float32)
        self._velocities = np.zeros((self.n_cells, 3), dtype=np.float32)
        self._forces = np.zeros((self.n_cells, 3), dtype=np.float32)
        self._radii = np.full(self.n_cells, 5.0, dtype=np.float32)  # Default 5 um
        self._cell_types = np.zeros(self.n_cells, dtype=np.int32)
        self._chemotaxis_strength = np.full(
            self.n_cells,
            self.config.chemotaxis_strength,
            dtype=np.float32
        )

        if self.backend.is_available:
            self._initialize_warp()
        else:
            log.debug("CellMechanics initialized (NumPy fallback)")

        self._initialized = True

    def _initialize_warp(self):
        """Initialize Warp arrays."""
        self._pos_wp = wp.array(
            self._positions,
            dtype=wp.vec3,
            device=self.backend.device,
        )
        self._vel_wp = wp.array(
            self._velocities,
            dtype=wp.vec3,
            device=self.backend.device,
        )
        self._forces_wp = wp.zeros(
            self.n_cells,
            dtype=wp.vec3,
            device=self.backend.device,
        )
        self._radii_wp = wp.array(
            self._radii,
            dtype=wp.float32,
            device=self.backend.device,
        )
        self._types_wp = wp.array(
            self._cell_types,
            dtype=wp.int32,
            device=self.backend.device,
        )
        self._chem_wp = wp.array(
            self._chemotaxis_strength,
            dtype=wp.float32,
            device=self.backend.device,
        )

        # RNG state
        rng_seeds = np.random.randint(1, 2**32 - 1, size=self.n_cells, dtype=np.uint64)
        self._rng_state = wp.array(rng_seeds, dtype=wp.uint64, device=self.backend.device)

        log.debug(f"CellMechanics initialized on {self.backend.device}")

    def set_positions(self, positions: np.ndarray):
        """Set cell positions."""
        self._positions = positions.astype(np.float32)
        if self.backend.is_available:
            self._pos_wp = wp.array(
                self._positions,
                dtype=wp.vec3,
                device=self.backend.device,
            )

    def get_positions(self) -> np.ndarray:
        """Get cell positions."""
        if self.backend.is_available and self._pos_wp is not None:
            return self._pos_wp.numpy()
        return self._positions

    def set_velocities(self, velocities: np.ndarray):
        """Set cell velocities."""
        self._velocities = velocities.astype(np.float32)
        if self.backend.is_available:
            self._vel_wp = wp.array(
                self._velocities,
                dtype=wp.vec3,
                device=self.backend.device,
            )

    def get_velocities(self) -> np.ndarray:
        """Get cell velocities."""
        if self.backend.is_available and self._vel_wp is not None:
            return self._vel_wp.numpy()
        return self._velocities

    def set_radii(self, radii: np.ndarray):
        """Set cell radii."""
        self._radii = radii.astype(np.float32)
        if self.backend.is_available:
            self._radii_wp = wp.array(
                self._radii,
                dtype=wp.float32,
                device=self.backend.device,
            )

    def set_cell_types(self, types: np.ndarray):
        """Set cell types (for type-dependent adhesion)."""
        self._cell_types = types.astype(np.int32)
        if self.backend.is_available:
            self._types_wp = wp.array(
                self._cell_types,
                dtype=wp.int32,
                device=self.backend.device,
            )

    def set_chemotaxis_strength(self, strength: np.ndarray):
        """Set per-cell chemotaxis sensitivity."""
        self._chemotaxis_strength = strength.astype(np.float32)
        if self.backend.is_available:
            self._chem_wp = wp.array(
                self._chemotaxis_strength,
                dtype=wp.float32,
                device=self.backend.device,
            )

    def step(self, dt: Optional[float] = None):
        """
        Advance simulation by one time step.

        Parameters
        ----------
        dt : float, optional
            Time step (defaults to config.dt)
        """
        dt = dt or self.config.dt

        if self.backend.is_available:
            self._step_warp(dt)
        else:
            self._step_numpy(dt)

        self._time += dt

    def _step_warp(self, dt: float):
        """Warp-accelerated step."""
        cfg = self.config

        # Zero forces
        self._forces_wp.zero_()

        # Compute pairwise forces (repulsion + adhesion)
        if cfg.force_types & (ForceType.REPULSION | ForceType.ADHESION):
            wp.launch(
                compute_pairwise_forces_kernel,
                dim=self.n_cells,
                inputs=[
                    self._pos_wp,
                    self._vel_wp,
                    self._forces_wp,
                    self._radii_wp,
                    self._types_wp,
                    cfg.k_repulsion,
                    cfg.k_adhesion,
                    cfg.adhesion_range,
                    self.n_cells,
                ],
                device=self.backend.device,
            )

        # Add Brownian noise
        if cfg.force_types & ForceType.BROWNIAN:
            wp.launch(
                add_brownian_force_kernel,
                dim=self.n_cells,
                inputs=[
                    self._forces_wp,
                    self._rng_state,
                    cfg.noise_strength,
                    self.n_cells,
                ],
                device=self.backend.device,
            )

        # Integrate (overdamped or inertial)
        if cfg.overdamped:
            self._integrate_overdamped_warp(dt)
        else:
            self._integrate_verlet_warp(dt)

    def _integrate_overdamped_warp(self, dt: float):
        """Overdamped Langevin integration on GPU."""
        # dx/dt = F/γ  ->  x += F/γ * dt
        gamma = self.config.damping

        # Get numpy, modify, set back (simple approach)
        # For production, use a Warp kernel
        pos = self._pos_wp.numpy()
        forces = self._forces_wp.numpy()

        pos += (forces / gamma) * dt

        self._pos_wp = wp.array(pos, dtype=wp.vec3, device=self.backend.device)

    def _integrate_verlet_warp(self, dt: float):
        """Velocity Verlet integration on GPU."""
        gamma = self.config.damping
        mass = self.config.mass

        pos = self._pos_wp.numpy()
        vel = self._vel_wp.numpy()
        forces = self._forces_wp.numpy()

        # Velocity Verlet with damping
        # a = (F - γv) / m
        acc = (forces - gamma * vel) / mass

        # v += a * dt
        vel += acc * dt

        # x += v * dt
        pos += vel * dt

        self._pos_wp = wp.array(pos, dtype=wp.vec3, device=self.backend.device)
        self._vel_wp = wp.array(vel, dtype=wp.vec3, device=self.backend.device)

    def _step_numpy(self, dt: float):
        """NumPy fallback step."""
        cfg = self.config
        n_active = len(self._positions)

        # Ensure forces array matches positions
        if len(self._forces) != n_active:
            self._forces = np.zeros((n_active, 3), dtype=np.float32)
        else:
            self._forces.fill(0)

        # Pairwise forces
        if cfg.force_types & (ForceType.REPULSION | ForceType.ADHESION):
            self._compute_pairwise_forces_numpy()

        # Brownian noise
        if cfg.force_types & ForceType.BROWNIAN:
            noise = np.random.randn(n_active, 3).astype(np.float32)
            self._forces += cfg.noise_strength * noise

        # Integrate
        if cfg.overdamped:
            self._positions += (self._forces / cfg.damping) * dt
        else:
            if len(self._velocities) != n_active:
                self._velocities = np.zeros((n_active, 3), dtype=np.float32)
            acc = (self._forces - cfg.damping * self._velocities) / cfg.mass
            self._velocities += acc * dt
            self._positions += self._velocities * dt

    def _compute_pairwise_forces_numpy(self):
        """Compute pairwise forces (O(N²) - use spatial hashing for large N)."""
        cfg = self.config
        n_active = len(self._positions)

        for i in range(n_active):
            pos_i = self._positions[i]
            r_i = self._radii[i]

            for j in range(i + 1, n_active):
                pos_j = self._positions[j]
                r_j = self._radii[j]

                d = pos_i - pos_j
                dist = np.linalg.norm(d)

                if dist < 1e-6:
                    continue

                n = d / dist
                contact = r_i + r_j

                # Repulsion
                if dist < contact:
                    overlap = contact - dist
                    f_rep = cfg.k_repulsion * overlap * n
                    self._forces[i] += f_rep
                    self._forces[j] -= f_rep

                # Adhesion
                if dist < contact + cfg.adhesion_range and dist >= contact:
                    stretch = dist - contact
                    f_adh = -cfg.k_adhesion * stretch * n
                    self._forces[i] += f_adh
                    self._forces[j] -= f_adh

    def get_forces(self) -> np.ndarray:
        """Get current forces on cells."""
        if self.backend.is_available and self._forces_wp is not None:
            return self._forces_wp.numpy()
        return self._forces

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._time

    def synchronize(self):
        """Wait for GPU operations to complete."""
        if self.backend.is_available:
            self.backend.synchronize()

    def seed_random(
        self,
        n_cells: int,
        region: Tuple[float, float, float, float, float, float],
        radii: float = 5.0,
    ):
        """
        Seed cells randomly in a region.

        Parameters
        ----------
        n_cells : int
            Number of cells to place
        region : tuple
            Bounding box (x_min, y_min, z_min, x_max, y_max, z_max)
        radii : float
            Cell radius
        """
        x_min, y_min, z_min, x_max, y_max, z_max = region

        positions = np.random.uniform(
            [x_min, y_min, z_min],
            [x_max, y_max, z_max],
            size=(n_cells, 3)
        ).astype(np.float32)

        self.set_positions(positions)
        self.set_radii(np.full(n_cells, radii, dtype=np.float32))
