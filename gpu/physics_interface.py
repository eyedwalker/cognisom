"""
Physics Model Interface
=======================

Protocol-based interface for GPU physics backends. This enables pluggable
physics engines (NVIDIA Warp, Newton, custom CuPy kernels) to be registered
and used interchangeably.

The interface is designed to support:
- Rigid body dynamics (collision detection, contact resolution)
- Soft body dynamics (cell deformation, VBD solver)
- Particle systems (molecular diffusion, chemotaxis)
- Fluid dynamics (blood flow, interstitial fluid)

Usage::

    from cognisom.gpu.physics_interface import physics_registry, PhysicsModel

    @physics_registry.register("warp_softbody")
    class WarpSoftBodyPhysics(PhysicsModel):
        def initialize(self, state):
            # Setup Warp simulation
            ...

        def step(self, dt: float, state):
            # Advance physics
            ...

Phase 0 of the Strategic Implementation Plan.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Type, runtime_checkable

from cognisom.core.registry import Registry, registry_manager

log = logging.getLogger(__name__)


class PhysicsBackendType(str, Enum):
    """Available physics backend types."""
    CUPY = "cupy"         # Custom CuPy kernels
    WARP = "warp"         # NVIDIA Warp
    NEWTON = "newton"     # NVIDIA Newton (future)
    NUMPY = "numpy"       # CPU fallback


class PhysicsModelType(str, Enum):
    """Types of physics models."""
    PARTICLE = "particle"       # Point particles with forces
    RIGID_BODY = "rigid_body"   # Rigid collision shapes
    SOFT_BODY = "soft_body"     # Deformable meshes (cells)
    FLUID = "fluid"             # Eulerian/SPH fluid
    DIFFUSION = "diffusion"     # Concentration fields
    HYBRID = "hybrid"           # Combined approaches


@dataclass
class PhysicsState:
    """
    Physics simulation state container.

    Holds positions, velocities, forces, and other physics data.
    Uses numpy/cupy arrays for GPU compatibility.
    """
    # Particle positions (N x 3)
    positions: Any = None
    # Particle velocities (N x 3)
    velocities: Any = None
    # Forces on particles (N x 3)
    forces: Any = None
    # Particle radii (N,)
    radii: Any = None
    # Particle masses (N,)
    masses: Any = None
    # Particle types/groups (N,) - for different interaction rules
    particle_types: Any = None
    # Additional custom state
    custom: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_particles(self) -> int:
        if self.positions is not None:
            return len(self.positions)
        return 0


@runtime_checkable
class PhysicsModel(Protocol):
    """
    Protocol defining the interface for physics backends.

    All physics models must implement these methods to be usable
    by the simulation engine.
    """

    @property
    def name(self) -> str:
        """Return the model name."""
        ...

    @property
    def backend_type(self) -> PhysicsBackendType:
        """Return the backend type (cupy, warp, newton)."""
        ...

    @property
    def model_type(self) -> PhysicsModelType:
        """Return the physics model type."""
        ...

    def initialize(self, state: PhysicsState, **config) -> None:
        """
        Initialize the physics model with initial state.

        Parameters
        ----------
        state : PhysicsState
            Initial physics state
        **config
            Model-specific configuration
        """
        ...

    def step(self, dt: float, state: PhysicsState) -> PhysicsState:
        """
        Advance physics simulation by one time step.

        Parameters
        ----------
        dt : float
            Time step in seconds
        state : PhysicsState
            Current physics state

        Returns
        -------
        PhysicsState
            Updated physics state
        """
        ...

    def apply_forces(self, forces: Any, state: PhysicsState) -> None:
        """
        Apply external forces to particles.

        Parameters
        ----------
        forces : array
            Force vectors (N x 3)
        state : PhysicsState
            Current state to modify
        """
        ...

    def get_collisions(self, state: PhysicsState) -> List[tuple]:
        """
        Get list of collision pairs.

        Returns
        -------
        List[tuple]
            List of (i, j, distance) collision tuples
        """
        ...


class BasePhysicsModel(ABC):
    """
    Abstract base class for physics models.

    Provides default implementations and utilities. Concrete
    implementations can inherit from this or implement the
    PhysicsModel protocol directly.
    """

    def __init__(self, name: str, config: Optional[Dict] = None):
        self._name = name
        self.config = config or {}
        self._initialized = False

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def backend_type(self) -> PhysicsBackendType:
        """Return the backend type."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> PhysicsModelType:
        """Return the model type."""
        pass

    @abstractmethod
    def initialize(self, state: PhysicsState, **config) -> None:
        """Initialize the model."""
        pass

    @abstractmethod
    def step(self, dt: float, state: PhysicsState) -> PhysicsState:
        """Advance one time step."""
        pass

    def apply_forces(self, forces: Any, state: PhysicsState) -> None:
        """Default: accumulate forces into state."""
        if state.forces is not None and forces is not None:
            state.forces += forces

    def get_collisions(self, state: PhysicsState) -> List[tuple]:
        """Default: no collision detection."""
        return []

    def cleanup(self) -> None:
        """Release resources (override if needed)."""
        pass


# ── Physics Model Registry ───────────────────────────────────────────

physics_registry = Registry(
    name="physics",
    base_class=None,  # Accept any class implementing PhysicsModel protocol
    allow_override=False,
)
registry_manager.add_registry("physics", physics_registry)


def register_physics(
    name: str,
    version: str = "1.0.0",
    backend: PhysicsBackendType = PhysicsBackendType.CUPY,
    model_type: PhysicsModelType = PhysicsModelType.PARTICLE,
    **metadata
):
    """
    Decorator to register a physics model.

    Parameters
    ----------
    name : str
        Model name (e.g., "warp_softbody", "cupy_diffusion")
    version : str
        Version string
    backend : PhysicsBackendType
        The GPU backend used
    model_type : PhysicsModelType
        Type of physics simulation
    **metadata
        Additional metadata

    Examples
    --------
    >>> @register_physics("cell_softbody", backend=PhysicsBackendType.WARP)
    ... class CellSoftBodyPhysics(BasePhysicsModel):
    ...     @property
    ...     def backend_type(self):
    ...         return PhysicsBackendType.WARP
    ...     @property
    ...     def model_type(self):
    ...         return PhysicsModelType.SOFT_BODY
    ...     def initialize(self, state, **config):
    ...         pass
    ...     def step(self, dt, state):
    ...         return state
    """
    return physics_registry.register(
        name=name,
        version=version,
        backend=backend.value,
        model_type=model_type.value,
        **metadata
    )


def get_physics_model(name: str) -> Type:
    """Get a physics model class by name."""
    return physics_registry.get(name)


def list_physics_models() -> List[str]:
    """List all registered physics model names."""
    return physics_registry.list_names()


def list_physics_by_backend(backend: PhysicsBackendType) -> List[str]:
    """List physics models that use a specific backend."""
    return [
        name for name, entry in physics_registry.items()
        if entry.metadata.get("backend") == backend.value
    ]


def create_physics_model(name: str, **config) -> Any:
    """
    Create a physics model instance by name.

    Parameters
    ----------
    name : str
        Model name
    **config
        Configuration passed to constructor

    Returns
    -------
    PhysicsModel
        Configured physics model instance
    """
    cls = physics_registry.get(name)
    return cls(name=name, config=config)


# ── Built-in Physics Models ──────────────────────────────────────────

@register_physics(
    "cupy_diffusion",
    version="1.0.0",
    backend=PhysicsBackendType.CUPY,
    model_type=PhysicsModelType.DIFFUSION,
)
class CuPyDiffusionPhysics(BasePhysicsModel):
    """
    CuPy-based 3D diffusion solver using finite differences.

    Uses the existing cognisom/gpu/diffusion.py implementation.
    """

    @property
    def backend_type(self) -> PhysicsBackendType:
        return PhysicsBackendType.CUPY

    @property
    def model_type(self) -> PhysicsModelType:
        return PhysicsModelType.DIFFUSION

    def initialize(self, state: PhysicsState, **config) -> None:
        self.diffusion_coeff = config.get("diffusion_coeff", 2000.0)
        self.grid_shape = config.get("grid_shape", (200, 200, 100))
        self.dx = config.get("dx", 10.0)
        self._initialized = True

    def step(self, dt: float, state: PhysicsState) -> PhysicsState:
        # Delegate to existing diffusion kernel
        from cognisom.gpu.diffusion import diffusion_step

        if "concentration" in state.custom:
            state.custom["concentration"] = diffusion_step(
                state.custom["concentration"],
                self.diffusion_coeff,
                dt,
                self.dx,
            )
        return state


@register_physics(
    "cupy_particle",
    version="1.0.0",
    backend=PhysicsBackendType.CUPY,
    model_type=PhysicsModelType.PARTICLE,
)
class CuPyParticlePhysics(BasePhysicsModel):
    """
    CuPy-based particle dynamics with simple integration.

    Supports position/velocity updates with external forces.
    """

    @property
    def backend_type(self) -> PhysicsBackendType:
        return PhysicsBackendType.CUPY

    @property
    def model_type(self) -> PhysicsModelType:
        return PhysicsModelType.PARTICLE

    def initialize(self, state: PhysicsState, **config) -> None:
        self.damping = config.get("damping", 0.99)
        self._initialized = True

    def step(self, dt: float, state: PhysicsState) -> PhysicsState:
        from cognisom.gpu.backend import get_backend
        xp = get_backend().xp

        if state.velocities is not None and state.forces is not None:
            if state.masses is not None:
                # F = ma -> a = F/m
                accel = state.forces / state.masses[:, None]
            else:
                accel = state.forces

            state.velocities += accel * dt
            state.velocities *= self.damping

        if state.positions is not None and state.velocities is not None:
            state.positions += state.velocities * dt

        # Clear forces for next step
        if state.forces is not None:
            state.forces *= 0

        return state


@register_physics(
    "numpy_fallback",
    version="1.0.0",
    backend=PhysicsBackendType.NUMPY,
    model_type=PhysicsModelType.PARTICLE,
)
class NumPyFallbackPhysics(BasePhysicsModel):
    """
    NumPy-based fallback for systems without GPU.

    Provides basic particle dynamics on CPU.
    """

    @property
    def backend_type(self) -> PhysicsBackendType:
        return PhysicsBackendType.NUMPY

    @property
    def model_type(self) -> PhysicsModelType:
        return PhysicsModelType.PARTICLE

    def initialize(self, state: PhysicsState, **config) -> None:
        self.damping = config.get("damping", 0.99)
        self._initialized = True

    def step(self, dt: float, state: PhysicsState) -> PhysicsState:
        import numpy as np

        if state.velocities is not None and state.forces is not None:
            if state.masses is not None:
                accel = state.forces / state.masses[:, np.newaxis]
            else:
                accel = state.forces

            state.velocities += accel * dt
            state.velocities *= self.damping

        if state.positions is not None and state.velocities is not None:
            state.positions += state.velocities * dt

        if state.forces is not None:
            state.forces *= 0

        return state


# ── Placeholder for Future Physics Backends ──────────────────────────

# These will be implemented in Phase A when Warp/Newton are integrated

class WarpPhysicsStub(BasePhysicsModel):
    """
    Placeholder for NVIDIA Warp soft-body physics.

    Will be implemented in Phase A: Physics Foundation.
    Requires: pip install warp-lang

    Features planned:
    - VBD (Vertex Block Descent) solver for cell deformation
    - GPU-accelerated contact detection
    - Differentiable physics for learning cell mechanics
    """

    @property
    def backend_type(self) -> PhysicsBackendType:
        return PhysicsBackendType.WARP

    @property
    def model_type(self) -> PhysicsModelType:
        return PhysicsModelType.SOFT_BODY

    def initialize(self, state: PhysicsState, **config) -> None:
        raise NotImplementedError(
            "Warp physics will be implemented in Phase A. "
            "Install warp-lang and check back after Phase A completion."
        )

    def step(self, dt: float, state: PhysicsState) -> PhysicsState:
        raise NotImplementedError("Warp physics not yet implemented")


class NewtonPhysicsStub(BasePhysicsModel):
    """
    Placeholder for NVIDIA Newton physics.

    Will be implemented when Newton becomes publicly available.
    Expected features:
    - Unified physics for rigid/soft/fluid
    - Advanced contact handling
    - Fluid-structure interaction
    """

    @property
    def backend_type(self) -> PhysicsBackendType:
        return PhysicsBackendType.NEWTON

    @property
    def model_type(self) -> PhysicsModelType:
        return PhysicsModelType.HYBRID

    def initialize(self, state: PhysicsState, **config) -> None:
        raise NotImplementedError(
            "Newton physics will be implemented when publicly available. "
            "Currently in preview/beta at NVIDIA."
        )

    def step(self, dt: float, state: PhysicsState) -> PhysicsState:
        raise NotImplementedError("Newton physics not yet implemented")


log.debug(f"Physics registry initialized with {len(physics_registry)} models")
