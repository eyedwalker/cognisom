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

__all__ = [
    "OmniverseConnector",
    "ConnectionStatus",
    "RealtimeSimulation",
    "SimulationMode",
    "SceneManager",
    "EntityVisual",
    "PhysicsBridge",
]
