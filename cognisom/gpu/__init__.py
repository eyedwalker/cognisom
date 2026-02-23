"""
GPU Acceleration Package
========================

Provides CuPy-accelerated kernels for Cognisom simulation modules.
Falls back to NumPy on machines without NVIDIA GPUs.

Modules:
    backend              — GPU/CPU detection and array abstraction
    diffusion            — 3D Laplacian stencil solver
    cell_ops             — Vectorized cell state updates (Structure-of-Arrays)
    spatial_ops          — Pairwise distance, neighbor search, force calculations
    orchestrator         — GPU-aware wrapper for simulation modules
    fba_solver           — Flux Balance Analysis (metabolic modeling)
    ssa_kernel           — CUDA SSA / tau-leaping for gene expression
    domain_decomposition — Multi-GPU slab decomposition with ghost exchange
    physics_interface    — Protocol for pluggable physics backends (Warp, Newton)

Extensibility:
    New physics backends can be registered at runtime:

        from cognisom.gpu import register_physics, BasePhysicsModel

        @register_physics("my_physics", backend=PhysicsBackendType.WARP)
        class MyPhysics(BasePhysicsModel):
            ...
"""

from .backend import get_backend, GPUBackend
from .physics_interface import (
    # Types
    PhysicsBackendType,
    PhysicsModelType,
    PhysicsState,
    PhysicsModel,
    BasePhysicsModel,
    # Registry
    physics_registry,
    register_physics,
    get_physics_model,
    list_physics_models,
    list_physics_by_backend,
    create_physics_model,
    # Built-in models
    CuPyDiffusionPhysics,
    CuPyParticlePhysics,
    NumPyFallbackPhysics,
)
from .mixed_precision import (
    MixedPrecisionConfig,
    MixedPrecisionSimulator,
    PrecisionLevel,
    GPUCapability,
    SimulationState,
    detect_gpu_capability,
    detect_optimal_precision,
    create_mixed_precision_simulator,
    precision_to_dtype,
)
from .multi_gpu_backend import MultiGPUBackend, DeviceContext
from .nccl_ghost_exchange import GhostExchanger
from .tissue_distributor import TissueDistributor, TissueCellArrays
from .multi_gpu_diffusion import MultiGPUDiffusionSolver
from .multi_gpu_mechanics import MultiGPUCellMechanics, MechanicsConfig, SpatialHash

__all__ = [
    # Backend
    "get_backend",
    "GPUBackend",
    # Physics types
    "PhysicsBackendType",
    "PhysicsModelType",
    "PhysicsState",
    "PhysicsModel",
    "BasePhysicsModel",
    # Physics registry
    "physics_registry",
    "register_physics",
    "get_physics_model",
    "list_physics_models",
    "list_physics_by_backend",
    "create_physics_model",
    # Built-in physics
    "CuPyDiffusionPhysics",
    "CuPyParticlePhysics",
    "NumPyFallbackPhysics",
    # Mixed precision (Phase B6)
    "MixedPrecisionConfig",
    "MixedPrecisionSimulator",
    "PrecisionLevel",
    "GPUCapability",
    "SimulationState",
    "detect_gpu_capability",
    "detect_optimal_precision",
    "create_mixed_precision_simulator",
    "precision_to_dtype",
    # Multi-GPU tissue-scale (Phase C)
    "MultiGPUBackend",
    "DeviceContext",
    "GhostExchanger",
    "TissueDistributor",
    "TissueCellArrays",
    "MultiGPUDiffusionSolver",
    "MultiGPUCellMechanics",
    "MechanicsConfig",
    "SpatialHash",
]
