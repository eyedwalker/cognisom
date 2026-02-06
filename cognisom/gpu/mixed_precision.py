"""
Mixed-Precision Simulation Bridge
=================================

Phase B6: Optimal precision/performance trade-off for GPU simulation.

The Trade-off:
- Consumer GPUs: FP64 is 1/32 to 1/64 of FP32 throughput
- Data center GPUs (H100): FP64 is ~1/2 of FP32
- Biological accuracy demands FP64 for positions (prevents drift)
- Force calculations can often use FP32 for local interactions

Strategy:
- Position integration: MUST be float64 (prevents accumulated drift)
- Force calculation: float32 for local interactions (upcast when needed)
- Rendering: float32 for GPU compatibility

This module provides:
- Hardware-aware precision configuration
- Mixed-precision kernels for simulation
- Precision conversion utilities
- Drift monitoring and correction

Usage::

    from cognisom.gpu.mixed_precision import (
        MixedPrecisionConfig,
        MixedPrecisionSimulator,
        detect_optimal_precision,
    )

    # Auto-detect optimal configuration
    config = MixedPrecisionConfig.auto_detect()

    # Create simulator with mixed precision
    simulator = MixedPrecisionSimulator(config)
    simulator.initialize(positions, velocities, masses)

    # Simulation step (positions stay FP64, forces use FP32)
    simulator.step(dt, forces_fp32)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

# Try to import GPU libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    import warp as wp
    WARP_AVAILABLE = True
except ImportError:
    wp = None
    WARP_AVAILABLE = False


class PrecisionLevel(str, Enum):
    """Available precision levels."""
    FLOAT16 = "float16"   # Half precision (for storage/rendering)
    FLOAT32 = "float32"   # Single precision (standard GPU)
    FLOAT64 = "float64"   # Double precision (scientific)


class GPUCapability(str, Enum):
    """GPU capability tiers for precision selection."""
    CONSUMER = "consumer"       # GTX/RTX consumer cards (FP64 = 1/32 FP32)
    PROFESSIONAL = "professional"  # Quadro/RTX Pro (FP64 = 1/2 FP32)
    DATACENTER = "datacenter"   # A100/H100 (FP64 = 1/2 FP32)
    CPU_ONLY = "cpu_only"       # No GPU available


@dataclass
class MixedPrecisionConfig:
    """
    Configuration for mixed-precision simulation.

    Attributes
    ----------
    position_precision : PrecisionLevel
        Precision for position integration (recommend float64)
    velocity_precision : PrecisionLevel
        Precision for velocity storage
    force_precision : PrecisionLevel
        Precision for force calculations
    render_precision : PrecisionLevel
        Precision for visualization output
    accumulator_precision : PrecisionLevel
        Precision for force accumulation
    gpu_capability : GPUCapability
        Detected GPU capability tier
    enable_drift_correction : bool
        Whether to apply drift correction
    drift_correction_interval : int
        Steps between drift corrections
    """
    position_precision: PrecisionLevel = PrecisionLevel.FLOAT64
    velocity_precision: PrecisionLevel = PrecisionLevel.FLOAT64
    force_precision: PrecisionLevel = PrecisionLevel.FLOAT32
    render_precision: PrecisionLevel = PrecisionLevel.FLOAT32
    accumulator_precision: PrecisionLevel = PrecisionLevel.FLOAT64
    gpu_capability: GPUCapability = GPUCapability.CPU_ONLY
    enable_drift_correction: bool = True
    drift_correction_interval: int = 100

    @classmethod
    def auto_detect(cls) -> "MixedPrecisionConfig":
        """
        Auto-detect optimal precision configuration based on hardware.

        Returns
        -------
        MixedPrecisionConfig
            Optimized configuration for detected hardware
        """
        gpu_cap = detect_gpu_capability()

        if gpu_cap == GPUCapability.DATACENTER:
            # H100/A100: Can afford FP64 everywhere
            return cls(
                position_precision=PrecisionLevel.FLOAT64,
                velocity_precision=PrecisionLevel.FLOAT64,
                force_precision=PrecisionLevel.FLOAT64,
                render_precision=PrecisionLevel.FLOAT32,
                accumulator_precision=PrecisionLevel.FLOAT64,
                gpu_capability=gpu_cap,
            )
        elif gpu_cap == GPUCapability.PROFESSIONAL:
            # Quadro/RTX Pro: FP64 for positions, FP32 for forces
            return cls(
                position_precision=PrecisionLevel.FLOAT64,
                velocity_precision=PrecisionLevel.FLOAT64,
                force_precision=PrecisionLevel.FLOAT32,
                render_precision=PrecisionLevel.FLOAT32,
                accumulator_precision=PrecisionLevel.FLOAT64,
                gpu_capability=gpu_cap,
            )
        elif gpu_cap == GPUCapability.CONSUMER:
            # Consumer GPU: Minimize FP64 usage
            return cls(
                position_precision=PrecisionLevel.FLOAT64,
                velocity_precision=PrecisionLevel.FLOAT32,
                force_precision=PrecisionLevel.FLOAT32,
                render_precision=PrecisionLevel.FLOAT32,
                accumulator_precision=PrecisionLevel.FLOAT64,
                gpu_capability=gpu_cap,
                drift_correction_interval=50,  # More frequent correction
            )
        else:
            # CPU fallback: FP64 is cheap on CPU
            return cls(
                position_precision=PrecisionLevel.FLOAT64,
                velocity_precision=PrecisionLevel.FLOAT64,
                force_precision=PrecisionLevel.FLOAT64,
                render_precision=PrecisionLevel.FLOAT32,
                accumulator_precision=PrecisionLevel.FLOAT64,
                gpu_capability=gpu_cap,
            )

    @classmethod
    def high_accuracy(cls) -> "MixedPrecisionConfig":
        """Configuration prioritizing accuracy over performance."""
        return cls(
            position_precision=PrecisionLevel.FLOAT64,
            velocity_precision=PrecisionLevel.FLOAT64,
            force_precision=PrecisionLevel.FLOAT64,
            render_precision=PrecisionLevel.FLOAT32,
            accumulator_precision=PrecisionLevel.FLOAT64,
            gpu_capability=detect_gpu_capability(),
        )

    @classmethod
    def high_performance(cls) -> "MixedPrecisionConfig":
        """Configuration prioritizing performance over accuracy."""
        return cls(
            position_precision=PrecisionLevel.FLOAT64,  # Still FP64 to prevent drift
            velocity_precision=PrecisionLevel.FLOAT32,
            force_precision=PrecisionLevel.FLOAT32,
            render_precision=PrecisionLevel.FLOAT32,
            accumulator_precision=PrecisionLevel.FLOAT32,
            gpu_capability=detect_gpu_capability(),
            drift_correction_interval=25,  # Very frequent correction
        )


def detect_gpu_capability() -> GPUCapability:
    """
    Detect GPU capability tier.

    Returns
    -------
    GPUCapability
        Detected capability tier
    """
    if not CUPY_AVAILABLE:
        return GPUCapability.CPU_ONLY

    try:
        device = cp.cuda.Device(0)
        compute_capability = device.compute_capability

        # Get device name for classification
        with device:
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else props['name']

        # Classify based on compute capability and name
        major, minor = compute_capability

        # H100/A100 detection
        if major >= 9 or (major == 8 and minor >= 0):
            if 'H100' in name or 'A100' in name or 'H200' in name:
                return GPUCapability.DATACENTER

        # Quadro/Professional detection
        if 'Quadro' in name or 'RTX A' in name or 'A6000' in name:
            return GPUCapability.PROFESSIONAL

        # L40S is datacenter-class
        if 'L40' in name:
            return GPUCapability.DATACENTER

        # Default to consumer
        return GPUCapability.CONSUMER

    except Exception as e:
        log.warning(f"GPU detection failed: {e}")
        return GPUCapability.CPU_ONLY


def precision_to_dtype(precision: PrecisionLevel) -> np.dtype:
    """Convert precision level to numpy dtype."""
    mapping = {
        PrecisionLevel.FLOAT16: np.float16,
        PrecisionLevel.FLOAT32: np.float32,
        PrecisionLevel.FLOAT64: np.float64,
    }
    return np.dtype(mapping[precision])


@dataclass
class SimulationState:
    """State container for mixed-precision simulation."""
    # High-precision state (positions, velocities)
    positions: np.ndarray  # (N, 3) float64
    velocities: np.ndarray  # (N, 3) float64 or float32
    masses: np.ndarray  # (N,) float64

    # Force accumulator
    forces: np.ndarray  # (N, 3) float32 or float64

    # Render-ready data (downcast from positions)
    render_positions: Optional[np.ndarray] = None  # (N, 3) float32

    # Drift tracking
    initial_center_of_mass: Optional[np.ndarray] = None
    drift_accumulated: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Statistics
    step_count: int = 0
    last_drift_correction: int = 0


class MixedPrecisionSimulator:
    """
    Mixed-precision particle/atom simulator.

    Manages precision trade-offs for optimal accuracy/performance balance.

    Parameters
    ----------
    config : MixedPrecisionConfig
        Precision configuration

    Example
    -------
    >>> config = MixedPrecisionConfig.auto_detect()
    >>> sim = MixedPrecisionSimulator(config)
    >>>
    >>> # Initialize with double-precision positions
    >>> sim.initialize(positions_fp64, velocities_fp64, masses_fp64)
    >>>
    >>> # Step with single-precision forces
    >>> sim.step(dt=0.001, forces=forces_fp32)
    >>>
    >>> # Get render-ready positions
    >>> render_pos = sim.get_render_positions()  # float32
    """

    def __init__(self, config: Optional[MixedPrecisionConfig] = None):
        self.config = config or MixedPrecisionConfig.auto_detect()
        self._state: Optional[SimulationState] = None
        self._initialized = False

        # GPU arrays (if available)
        self._gpu_positions = None
        self._gpu_velocities = None
        self._gpu_forces = None

        # Kernels
        self._kernels_compiled = False

        log.info(
            f"MixedPrecisionSimulator: positions={self.config.position_precision.value}, "
            f"forces={self.config.force_precision.value}, "
            f"gpu={self.config.gpu_capability.value}"
        )

    def initialize(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        masses: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize simulation state.

        Parameters
        ----------
        positions : np.ndarray
            Initial positions (N, 3)
        velocities : np.ndarray, optional
            Initial velocities (N, 3), zeros if not provided
        masses : np.ndarray, optional
            Particle masses (N,), ones if not provided
        """
        n_particles = len(positions)

        # Ensure correct precision for positions (always float64)
        pos_dtype = precision_to_dtype(self.config.position_precision)
        positions = positions.astype(pos_dtype)

        # Velocities
        vel_dtype = precision_to_dtype(self.config.velocity_precision)
        if velocities is None:
            velocities = np.zeros((n_particles, 3), dtype=vel_dtype)
        else:
            velocities = velocities.astype(vel_dtype)

        # Masses
        if masses is None:
            masses = np.ones(n_particles, dtype=np.float64)
        else:
            masses = masses.astype(np.float64)

        # Forces
        force_dtype = precision_to_dtype(self.config.force_precision)
        forces = np.zeros((n_particles, 3), dtype=force_dtype)

        # Create state
        self._state = SimulationState(
            positions=positions,
            velocities=velocities,
            masses=masses,
            forces=forces,
            initial_center_of_mass=self._compute_center_of_mass(positions, masses),
        )

        # Update render positions
        self._update_render_positions()

        # Transfer to GPU if available
        if CUPY_AVAILABLE and self.config.gpu_capability != GPUCapability.CPU_ONLY:
            self._transfer_to_gpu()

        self._initialized = True
        log.info(f"Initialized with {n_particles} particles")

    def step(
        self,
        dt: float,
        forces: Optional[np.ndarray] = None,
    ) -> None:
        """
        Advance simulation by one timestep.

        Uses velocity Verlet integration with mixed precision:
        1. Update positions (FP64): x = x + v*dt + 0.5*a*dt²
        2. Update velocities (FP32/64): v = v + 0.5*(a_old + a_new)*dt

        Parameters
        ----------
        dt : float
            Timestep
        forces : np.ndarray, optional
            External forces (N, 3). If None, uses stored forces.
        """
        if not self._initialized:
            raise RuntimeError("Simulator not initialized")

        state = self._state

        # Update forces if provided
        if forces is not None:
            force_dtype = precision_to_dtype(self.config.force_precision)
            state.forces = forces.astype(force_dtype)

        # Compute acceleration (upcast forces to FP64 for integration)
        accel = state.forces.astype(np.float64) / state.masses[:, np.newaxis]

        # Velocity Verlet integration (in FP64)
        # x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        state.positions += state.velocities.astype(np.float64) * dt
        state.positions += 0.5 * accel * (dt * dt)

        # v(t+dt) = v(t) + a(t)*dt
        # (Simplified - full Verlet needs force at t+dt)
        vel_dtype = precision_to_dtype(self.config.velocity_precision)
        state.velocities = (state.velocities.astype(np.float64) + accel * dt).astype(vel_dtype)

        state.step_count += 1

        # Drift correction
        if (self.config.enable_drift_correction and
            state.step_count - state.last_drift_correction >= self.config.drift_correction_interval):
            self._apply_drift_correction()

        # Update render positions
        self._update_render_positions()

    def _compute_center_of_mass(
        self,
        positions: np.ndarray,
        masses: np.ndarray
    ) -> np.ndarray:
        """Compute center of mass."""
        total_mass = np.sum(masses)
        if total_mass == 0:
            return np.zeros(3)
        return np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass

    def _apply_drift_correction(self) -> None:
        """Apply drift correction to prevent accumulated floating point errors."""
        state = self._state

        if state.initial_center_of_mass is None:
            return

        current_com = self._compute_center_of_mass(state.positions, state.masses)
        drift = current_com - state.initial_center_of_mass

        # Only correct if drift is significant
        if np.linalg.norm(drift) > 1e-10:
            state.positions -= drift
            state.drift_accumulated += drift
            state.last_drift_correction = state.step_count

            log.debug(
                f"Drift correction applied: {np.linalg.norm(drift):.2e} "
                f"(total: {np.linalg.norm(state.drift_accumulated):.2e})"
            )

    def _update_render_positions(self) -> None:
        """Update float32 render positions from float64 simulation positions."""
        state = self._state
        render_dtype = precision_to_dtype(self.config.render_precision)
        state.render_positions = state.positions.astype(render_dtype)

    def _transfer_to_gpu(self) -> None:
        """Transfer state arrays to GPU."""
        if not CUPY_AVAILABLE:
            return

        state = self._state
        self._gpu_positions = cp.asarray(state.positions)
        self._gpu_velocities = cp.asarray(state.velocities)
        self._gpu_forces = cp.asarray(state.forces)

    def get_positions(self) -> np.ndarray:
        """Get high-precision positions (float64)."""
        return self._state.positions.copy()

    def get_render_positions(self) -> np.ndarray:
        """Get render-ready positions (float32)."""
        return self._state.render_positions.copy()

    def get_velocities(self) -> np.ndarray:
        """Get velocities."""
        return self._state.velocities.copy()

    def set_forces(self, forces: np.ndarray) -> None:
        """Set forces for next integration step."""
        force_dtype = precision_to_dtype(self.config.force_precision)
        self._state.forces = forces.astype(force_dtype)

    def get_state_dict(self) -> Dict[str, Any]:
        """Get simulation state as dictionary."""
        state = self._state
        return {
            'positions': state.positions.copy(),
            'velocities': state.velocities.copy(),
            'masses': state.masses.copy(),
            'step_count': state.step_count,
            'drift_accumulated': state.drift_accumulated.copy(),
            'config': {
                'position_precision': self.config.position_precision.value,
                'force_precision': self.config.force_precision.value,
                'gpu_capability': self.config.gpu_capability.value,
            }
        }

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return len(self._state.positions) if self._state else 0

    @property
    def step_count(self) -> int:
        """Current step count."""
        return self._state.step_count if self._state else 0


# ── CUDA Kernels for Mixed Precision ────────────────────────────────────

_INTEGRATE_KERNEL = r"""
extern "C" __global__
void integrate_positions_mixed(
    double* positions,      // FP64 positions (N*3)
    const float* velocities, // FP32 velocities (N*3)
    const float* forces,     // FP32 forces (N*3)
    const double* masses,    // FP64 masses (N)
    double dt,
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    double m = masses[tid];
    double inv_m = 1.0 / m;

    for (int d = 0; d < 3; d++) {
        int idx = tid * 3 + d;

        // Upcast FP32 to FP64 for integration
        double v = (double)velocities[idx];
        double f = (double)forces[idx];
        double a = f * inv_m;

        // Position update in FP64
        positions[idx] += v * dt + 0.5 * a * dt * dt;
    }
}

extern "C" __global__
void integrate_velocities_mixed(
    float* velocities,       // FP32 velocities (N*3)
    const float* forces,     // FP32 forces (N*3)
    const double* masses,    // FP64 masses (N)
    double dt,
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    double m = masses[tid];
    double inv_m = 1.0 / m;

    for (int d = 0; d < 3; d++) {
        int idx = tid * 3 + d;

        double v = (double)velocities[idx];
        double f = (double)forces[idx];
        double a = f * inv_m;

        // Velocity update, then cast back to FP32
        velocities[idx] = (float)(v + a * dt);
    }
}

extern "C" __global__
void downcast_positions(
    float* render_positions,  // FP32 output (N*3)
    const double* positions,  // FP64 input (N*3)
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles * 3) return;

    render_positions[tid] = (float)positions[tid];
}
"""


class GPUMixedPrecisionKernels:
    """GPU kernels for mixed-precision simulation."""

    def __init__(self):
        self._compiled = False
        self._integrate_pos_kernel = None
        self._integrate_vel_kernel = None
        self._downcast_kernel = None

    def compile(self) -> bool:
        """Compile CUDA kernels."""
        if not CUPY_AVAILABLE:
            return False

        try:
            self._integrate_pos_kernel = cp.RawKernel(
                _INTEGRATE_KERNEL, "integrate_positions_mixed"
            )
            self._integrate_vel_kernel = cp.RawKernel(
                _INTEGRATE_KERNEL, "integrate_velocities_mixed"
            )
            self._downcast_kernel = cp.RawKernel(
                _INTEGRATE_KERNEL, "downcast_positions"
            )
            self._compiled = True
            log.info("Mixed-precision GPU kernels compiled")
            return True
        except Exception as e:
            log.error(f"Failed to compile GPU kernels: {e}")
            return False

    def integrate_positions(
        self,
        positions: Any,  # cp.ndarray FP64
        velocities: Any,  # cp.ndarray FP32
        forces: Any,  # cp.ndarray FP32
        masses: Any,  # cp.ndarray FP64
        dt: float,
    ) -> None:
        """Integrate positions on GPU."""
        if not self._compiled:
            self.compile()

        n = len(positions)
        block = 256
        grid = (n + block - 1) // block

        self._integrate_pos_kernel(
            (grid,), (block,),
            (positions, velocities, forces, masses, np.float64(dt), np.int32(n))
        )

    def integrate_velocities(
        self,
        velocities: Any,
        forces: Any,
        masses: Any,
        dt: float,
    ) -> None:
        """Integrate velocities on GPU."""
        if not self._compiled:
            self.compile()

        n = len(velocities)
        block = 256
        grid = (n + block - 1) // block

        self._integrate_vel_kernel(
            (grid,), (block,),
            (velocities, forces, masses, np.float64(dt), np.int32(n))
        )


# ── Warp Integration ────────────────────────────────────────────────────

if WARP_AVAILABLE:
    @wp.kernel
    def integrate_positions_fp64(
        positions: wp.array(dtype=wp.vec3d),
        velocities: wp.array(dtype=wp.vec3d),
        forces: wp.array(dtype=wp.vec3f),
        masses: wp.array(dtype=wp.float64),
        dt: wp.float64,
    ):
        """Warp kernel for FP64 position integration with FP32 forces."""
        tid = wp.tid()

        # Upcast force to FP64
        f = wp.vec3d(
            wp.float64(forces[tid][0]),
            wp.float64(forces[tid][1]),
            wp.float64(forces[tid][2]),
        )

        # Acceleration
        acceleration = f / masses[tid]

        # Velocity Verlet position update
        positions[tid] = positions[tid] + velocities[tid] * dt + 0.5 * acceleration * dt * dt

        # Velocity update
        velocities[tid] = velocities[tid] + acceleration * dt


# ── Convenience Functions ───────────────────────────────────────────────

def detect_optimal_precision() -> MixedPrecisionConfig:
    """Detect and return optimal precision configuration."""
    return MixedPrecisionConfig.auto_detect()


def create_mixed_precision_simulator(
    accuracy_priority: bool = True,
) -> MixedPrecisionSimulator:
    """
    Create a mixed-precision simulator with appropriate configuration.

    Parameters
    ----------
    accuracy_priority : bool
        If True, prioritize accuracy; otherwise prioritize performance

    Returns
    -------
    MixedPrecisionSimulator
        Configured simulator
    """
    if accuracy_priority:
        config = MixedPrecisionConfig.high_accuracy()
    else:
        config = MixedPrecisionConfig.high_performance()

    return MixedPrecisionSimulator(config)
