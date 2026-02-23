"""
Omniverse Integration (Phase 8)
================================

Real-time simulation integration with NVIDIA Omniverse and Isaac Sim.

This package provides:
- ``connector``: Connection manager for Omniverse/Isaac Sim
- ``realtime_sim``: Real-time simulation loop with live updates
- ``scene_manager``: Dynamic scene management for biological entities
- ``physics_bridge``: Bridge between Cognisom physics and PhysX/Isaac

The integration enables:
1. Live visualization of cell simulations in Omniverse
2. Physics-based interactions using Isaac Sim
3. Bidirectional parameter editing
4. Real-time data streaming

Requirements:
    - NVIDIA Omniverse Kit SDK (for extension development)
    - Isaac Sim (optional, for advanced physics)
    - omni.client Python package

Usage::

    from cognisom.omniverse import OmniverseConnector, RealtimeSimulation

    # Connect to Omniverse
    connector = OmniverseConnector()
    connector.connect("omniverse://localhost/cognisom")

    # Run real-time simulation
    sim = RealtimeSimulation(engine, connector)
    sim.start()  # Starts live update loop
"""

from .connector import OmniverseConnector, ConnectionStatus
from .realtime_sim import RealtimeSimulation, SimulationMode
from .scene_manager import SceneManager, EntityVisual
from .physics_bridge import PhysicsBridge
from .real_connector import RealOmniverseConnector, CellVisualization, SimulationFrame
try:
    from .precision import (
        PrecisionTransformManager,
        BioPrecisePointAPI,
        BiologicalScale,
        ScaleMetadata,
        SCALE_CONFIGS,
        configure_stage_for_biology,
        centroid_and_center,
        verify_no_jitter,
    )
    from .instancing import (
        NestedInstancingManager,
        InstancingStrategy,
        PrototypeDefinition,
        InstanceData,
        create_sphere_prototype,
        create_capsule_prototype,
        create_ellipsoid_prototype,
        estimate_instancing_strategy,
    )
    from .prototype_library import (
        PrototypeLibrary,
        PrototypeCategory,
        PrototypeSpec,
        ALL_PROTOTYPES,
        MOLECULE_PROTOTYPES,
        ORGANELLE_PROTOTYPES,
        CELL_PROTOTYPES,
        PARTICLE_PROTOTYPES,
        STRUCTURE_PROTOTYPES,
        create_biological_scene,
    )
    from .payload_manager import (
        DynamicPayloadManager,
        PayloadManagerConfig,
        PayloadManagerStats,
        PayloadInfo,
        PayloadState,
        LoadPriority,
        create_payload_manager,
        open_stage_for_streaming,
    )
    from .semantic_zoom import (
        SemanticZoomController,
        SemanticZoomConfig,
        ZoomThresholds,
        ZoomableEntity,
        RepresentationLevel,
        create_semantic_zoom_controller,
        setup_biological_zoom_hierarchy,
    )
except (ImportError, NameError):
    # OpenUSD (pxr) not available — these modules require it
    pass

__all__ = [
    # Original exports
    "OmniverseConnector",
    "ConnectionStatus",
    "RealtimeSimulation",
    "SimulationMode",
    "SceneManager",
    "EntityVisual",
    "PhysicsBridge",
    # Real USD connector
    "RealOmniverseConnector",
    "CellVisualization",
    "SimulationFrame",
    # Precision pipeline (Phase B1)
    "PrecisionTransformManager",
    "BioPrecisePointAPI",
    "BiologicalScale",
    "ScaleMetadata",
    "SCALE_CONFIGS",
    "configure_stage_for_biology",
    "centroid_and_center",
    "verify_no_jitter",
    # Nested Instancing (Phase B2)
    "NestedInstancingManager",
    "InstancingStrategy",
    "PrototypeDefinition",
    "InstanceData",
    "create_sphere_prototype",
    "create_capsule_prototype",
    "create_ellipsoid_prototype",
    "estimate_instancing_strategy",
    # Prototype Library (Phase B2)
    "PrototypeLibrary",
    "PrototypeCategory",
    "PrototypeSpec",
    "ALL_PROTOTYPES",
    "MOLECULE_PROTOTYPES",
    "ORGANELLE_PROTOTYPES",
    "CELL_PROTOTYPES",
    "PARTICLE_PROTOTYPES",
    "STRUCTURE_PROTOTYPES",
    "create_biological_scene",
    # Dynamic Payload Manager (Phase B3)
    "DynamicPayloadManager",
    "PayloadManagerConfig",
    "PayloadManagerStats",
    "PayloadInfo",
    "PayloadState",
    "LoadPriority",
    "create_payload_manager",
    "open_stage_for_streaming",
    # Semantic Zooming (Phase B4)
    "SemanticZoomController",
    "SemanticZoomConfig",
    "ZoomThresholds",
    "ZoomableEntity",
    "RepresentationLevel",
    "create_semantic_zoom_controller",
    "setup_biological_zoom_hierarchy",
]
