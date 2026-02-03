"""
cognisom Core
=============

Master simulation engine and core infrastructure.
"""

from .simulation_engine import SimulationEngine, SimulationConfig
from .event_bus import EventBus, EventTypes
from .module_base import SimulationModule
from .registry import (
    Registry,
    RegistryEntry,
    RegistryManager,
    RegistryError,
    DuplicateRegistrationError,
    NotFoundError,
    ValidationError,
    registry_manager,
    get_registry,
)

__all__ = [
    'SimulationEngine',
    'SimulationConfig',
    'EventBus',
    'EventTypes',
    'SimulationModule',
    # Registry system
    'Registry',
    'RegistryEntry',
    'RegistryManager',
    'RegistryError',
    'DuplicateRegistrationError',
    'NotFoundError',
    'ValidationError',
    'registry_manager',
    'get_registry',
]
