"""
Physics Integrators
===================

Time integration schemes for particle dynamics.

Integrators provided:
- Explicit Euler (simple, first-order)
- Velocity Verlet (symplectic, second-order)
- XPBD (eXtended Position Based Dynamics for constraints)
- Overdamped Langevin (Brownian dynamics)

For cell mechanics, we typically use:
- Overdamped Langevin for viscous environments (cells in tissue)
- Velocity Verlet for inertial dynamics (circulating tumor cells)

Phase A.2 of the Strategic Implementation Plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .warp_backend import WarpBackend, get_warp_backend, check_warp_available

log = logging.getLogger(__name__)

_WARP_AVAILABLE = check_warp_available()

if _WARP_AVAILABLE:
    import warp as wp

    @wp.kernel
    def euler_integrate_kernel(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        forces: wp.array(dtype=wp.vec3),
        masses: wp.array(dtype=wp.float32),
        dt: wp.float32,
    ):
        """Explicit Euler integration."""
        i = wp.tid()

        m = masses[i]
        if m <= 0.0:
            m = wp.float32(1.0)

        # a = F/m
        a = forces[i] / m

        # v += a * dt
        velocities[i] = velocities[i] + a * dt

        # x += v * dt
        positions[i] = positions[i] + velocities[i] * dt

    @wp.kernel
    def verlet_integrate_kernel(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        forces: wp.array(dtype=wp.vec3),
        masses: wp.array(dtype=wp.float32),
        damping: wp.float32,
        dt: wp.float32,
    ):
        """
        Velocity Verlet integration with damping.

        dx/dt = v
        m * dv/dt = F - γv
        """
        i = wp.tid()

        m = masses[i]
        if m <= 0.0:
            m = wp.float32(1.0)

        v = velocities[i]
        f = forces[i]

        # Acceleration with damping
        a = (f - damping * v) / m

        # Update velocity (half step)
        v_half = v + 0.5 * a * dt

        # Update position
        x = positions[i]
        x = x + v_half * dt

        # Store for next step
        positions[i] = x
        velocities[i] = v + a * dt

    @wp.kernel
    def overdamped_integrate_kernel(
        positions: wp.array(dtype=wp.vec3),
        forces: wp.array(dtype=wp.vec3),
        gamma: wp.float32,
        dt: wp.float32,
    ):
        """
        Overdamped (Langevin) integration.

        In the overdamped limit (high viscosity):
        γ * dx/dt = F  ->  dx = F/γ * dt
        """
        i = wp.tid()

        f = forces[i]
        x = positions[i]

        # dx = F/γ * dt
        dx = (f / gamma) * dt
        positions[i] = x + dx

    @wp.kernel
    def xpbd_project_distance_kernel(
        positions: wp.array(dtype=wp.vec3),
        constraint_i: wp.array(dtype=wp.int32),
        constraint_j: wp.array(dtype=wp.int32),
        rest_lengths: wp.array(dtype=wp.float32),
        inv_masses: wp.array(dtype=wp.float32),
        compliance: wp.float32,
        dt: wp.float32,
    ):
        """
        XPBD distance constraint projection.

        Projects particles to satisfy distance constraints.
        Used for rigid bonds, membranes, etc.
        """
        c = wp.tid()

        i = constraint_i[c]
        j = constraint_j[c]

        x_i = positions[i]
        x_j = positions[j]

        w_i = inv_masses[i]
        w_j = inv_masses[j]
        w_sum = w_i + w_j

        if w_sum <= 0.0:
            return

        # Current distance
        d = x_i - x_j
        dist = wp.length(d)

        if dist < 1e-6:
            return

        n = d / dist

        # Rest length
        L = rest_lengths[c]

        # Constraint value
        C = dist - L

        # XPBD compliance
        alpha = compliance / (dt * dt)

        # Lagrange multiplier update
        delta_lambda = -C / (w_sum + alpha)

        # Position correction
        dx = delta_lambda * n

        positions[i] = x_i + w_i * dx
        positions[j] = x_j - w_j * dx


def euler_integrate(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    dt: float,
    backend: Optional[WarpBackend] = None,
) -> tuple:
    """
    Explicit Euler integration.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) positions
    velocities : np.ndarray
        (N, 3) velocities
    forces : np.ndarray
        (N, 3) forces
    masses : np.ndarray
        (N,) masses
    dt : float
        Time step
    backend : WarpBackend, optional
        Warp backend

    Returns
    -------
    tuple
        (new_positions, new_velocities)
    """
    backend = backend or get_warp_backend()

    if not backend.is_available:
        # NumPy fallback
        a = forces / masses[:, np.newaxis]
        velocities = velocities + a * dt
        positions = positions + velocities * dt
        return positions, velocities

    # Warp
    n = len(positions)
    pos_wp = wp.array(positions.astype(np.float32), dtype=wp.vec3, device=backend.device)
    vel_wp = wp.array(velocities.astype(np.float32), dtype=wp.vec3, device=backend.device)
    forces_wp = wp.array(forces.astype(np.float32), dtype=wp.vec3, device=backend.device)
    masses_wp = wp.array(masses.astype(np.float32), dtype=wp.float32, device=backend.device)

    wp.launch(
        euler_integrate_kernel,
        dim=n,
        inputs=[pos_wp, vel_wp, forces_wp, masses_wp, dt],
        device=backend.device,
    )

    return pos_wp.numpy(), vel_wp.numpy()


def verlet_integrate(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    damping: float,
    dt: float,
    backend: Optional[WarpBackend] = None,
) -> tuple:
    """
    Velocity Verlet integration with damping.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) positions
    velocities : np.ndarray
        (N, 3) velocities
    forces : np.ndarray
        (N, 3) forces
    masses : np.ndarray
        (N,) masses
    damping : float
        Damping coefficient
    dt : float
        Time step
    backend : WarpBackend, optional
        Warp backend

    Returns
    -------
    tuple
        (new_positions, new_velocities)
    """
    backend = backend or get_warp_backend()

    if not backend.is_available:
        # NumPy fallback
        a = (forces - damping * velocities) / masses[:, np.newaxis]
        v_half = velocities + 0.5 * a * dt
        positions = positions + v_half * dt
        velocities = velocities + a * dt
        return positions, velocities

    n = len(positions)
    pos_wp = wp.array(positions.astype(np.float32), dtype=wp.vec3, device=backend.device)
    vel_wp = wp.array(velocities.astype(np.float32), dtype=wp.vec3, device=backend.device)
    forces_wp = wp.array(forces.astype(np.float32), dtype=wp.vec3, device=backend.device)
    masses_wp = wp.array(masses.astype(np.float32), dtype=wp.float32, device=backend.device)

    wp.launch(
        verlet_integrate_kernel,
        dim=n,
        inputs=[pos_wp, vel_wp, forces_wp, masses_wp, damping, dt],
        device=backend.device,
    )

    return pos_wp.numpy(), vel_wp.numpy()


def overdamped_integrate(
    positions: np.ndarray,
    forces: np.ndarray,
    gamma: float,
    dt: float,
    backend: Optional[WarpBackend] = None,
) -> np.ndarray:
    """
    Overdamped (Langevin) integration.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) positions
    forces : np.ndarray
        (N, 3) forces
    gamma : float
        Damping coefficient
    dt : float
        Time step
    backend : WarpBackend, optional
        Warp backend

    Returns
    -------
    np.ndarray
        New positions
    """
    backend = backend or get_warp_backend()

    if not backend.is_available:
        return positions + (forces / gamma) * dt

    n = len(positions)
    pos_wp = wp.array(positions.astype(np.float32), dtype=wp.vec3, device=backend.device)
    forces_wp = wp.array(forces.astype(np.float32), dtype=wp.vec3, device=backend.device)

    wp.launch(
        overdamped_integrate_kernel,
        dim=n,
        inputs=[pos_wp, forces_wp, gamma, dt],
        device=backend.device,
    )

    return pos_wp.numpy()


@dataclass
class XPBDConstraint:
    """Distance constraint for XPBD."""
    i: int              # First particle index
    j: int              # Second particle index
    rest_length: float  # Target distance
    compliance: float = 0.0  # Stiffness (0 = rigid)


def xpbd_integrate(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    constraints: list,
    dt: float,
    n_substeps: int = 10,
    backend: Optional[WarpBackend] = None,
) -> tuple:
    """
    XPBD (eXtended Position Based Dynamics) integration.

    Combines velocity update with constraint projection.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) positions
    velocities : np.ndarray
        (N, 3) velocities
    forces : np.ndarray
        (N, 3) forces
    masses : np.ndarray
        (N,) masses
    constraints : list
        List of XPBDConstraint objects
    dt : float
        Time step
    n_substeps : int
        Number of substeps for constraint projection
    backend : WarpBackend, optional
        Warp backend

    Returns
    -------
    tuple
        (new_positions, new_velocities)
    """
    backend = backend or get_warp_backend()

    h = dt / n_substeps  # Substep size

    # Predict positions
    inv_masses = 1.0 / masses
    inv_masses[masses <= 0] = 0.0

    x_pred = positions.copy()
    v = velocities.copy()

    for _ in range(n_substeps):
        # Apply external forces
        a = forces * inv_masses[:, np.newaxis]
        v += a * h
        x_pred += v * h

        # Project constraints
        if constraints and backend.is_available:
            x_pred = _project_constraints_warp(
                x_pred, constraints, inv_masses, h, backend
            )
        elif constraints:
            x_pred = _project_constraints_numpy(
                x_pred, constraints, inv_masses, h
            )

    # Update velocities from position change
    velocities = (x_pred - positions) / dt

    return x_pred, velocities


def _project_constraints_numpy(
    positions: np.ndarray,
    constraints: list,
    inv_masses: np.ndarray,
    dt: float,
    n_iterations: int = 4,
) -> np.ndarray:
    """Project constraints using Gauss-Seidel iteration."""
    x = positions.copy()

    for _ in range(n_iterations):
        for c in constraints:
            i, j = c.i, c.j
            w_i, w_j = inv_masses[i], inv_masses[j]
            w_sum = w_i + w_j

            if w_sum <= 0:
                continue

            d = x[i] - x[j]
            dist = np.linalg.norm(d)

            if dist < 1e-6:
                continue

            n = d / dist
            C = dist - c.rest_length

            alpha = c.compliance / (dt * dt)
            delta_lambda = -C / (w_sum + alpha)

            x[i] += w_i * delta_lambda * n
            x[j] -= w_j * delta_lambda * n

    return x


def _project_constraints_warp(
    positions: np.ndarray,
    constraints: list,
    inv_masses: np.ndarray,
    dt: float,
    backend: WarpBackend,
    n_iterations: int = 4,
) -> np.ndarray:
    """Project constraints using Warp kernel."""
    n_constraints = len(constraints)

    if n_constraints == 0:
        return positions

    # Build constraint arrays
    constraint_i = np.array([c.i for c in constraints], dtype=np.int32)
    constraint_j = np.array([c.j for c in constraints], dtype=np.int32)
    rest_lengths = np.array([c.rest_length for c in constraints], dtype=np.float32)
    compliance = constraints[0].compliance  # Assume uniform

    pos_wp = wp.array(positions.astype(np.float32), dtype=wp.vec3, device=backend.device)
    ci_wp = wp.array(constraint_i, dtype=wp.int32, device=backend.device)
    cj_wp = wp.array(constraint_j, dtype=wp.int32, device=backend.device)
    rest_wp = wp.array(rest_lengths, dtype=wp.float32, device=backend.device)
    inv_m_wp = wp.array(inv_masses.astype(np.float32), dtype=wp.float32, device=backend.device)

    for _ in range(n_iterations):
        wp.launch(
            xpbd_project_distance_kernel,
            dim=n_constraints,
            inputs=[pos_wp, ci_wp, cj_wp, rest_wp, inv_m_wp, compliance, dt],
            device=backend.device,
        )

    return pos_wp.numpy()
