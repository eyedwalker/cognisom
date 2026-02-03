"""
cognisom Modules
================

Simulation modules for cognisom platform.

Each module encapsulates a specific biological process (cellular dynamics,
immune response, vascular network, etc.) and can be composed into
simulation pipelines.

Extensibility:
    New simulation modules can be registered at runtime:

        from cognisom.modules import register_module
        from cognisom.core import SimulationModule

        @register_module("my_custom_module")
        class MyCustomModule(SimulationModule):
            def initialize(self):
                ...
            def update(self, dt):
                ...
            def get_state(self):
                ...

    Modules can also be discovered via entry points:
        [project.entry-points."cognisom.modules"]
        my_module = "my_package.modules:MyModule"
"""

import logging
from typing import Dict, List, Type

from cognisom.core.registry import Registry, registry_manager
from cognisom.core.module_base import SimulationModule

log = logging.getLogger(__name__)

# Import built-in modules
from .molecular_module import MolecularModule
from .cellular_module import CellularModule
from .immune_module import ImmuneModule
from .vascular_module import VascularModule
from .lymphatic_module import LymphaticModule
from .spatial_module import SpatialModule
from .epigenetic_module import EpigeneticModule
from .circadian_module import CircadianModule
from .morphogen_module import MorphogenModule
from .cell_mechanics_module import CellMechanicsModule


# ── Module Registry ──────────────────────────────────────────────────

# Create the module registry
# Note: base_class is None because some legacy modules use relative imports
# which causes Python to see them as different classes. The modules all
# inherit from SimulationModule in practice.
module_registry = Registry(
    name="modules",
    base_class=None,  # Disable strict validation for legacy compatibility
    allow_override=False,
)
registry_manager.add_registry("modules", module_registry)


def register_module(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    **metadata
):
    """
    Decorator to register a simulation module.

    Parameters
    ----------
    name : str
        Module name (e.g., "custom_physics", "virus_dynamics")
    version : str
        Version string
    description : str
        Human-readable description
    **metadata
        Additional metadata

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> @register_module("virus_module", version="1.0.0")
    ... class VirusModule(SimulationModule):
    ...     def initialize(self):
    ...         self.virus_count = 0
    ...     def update(self, dt):
    ...         # virus dynamics
    ...         pass
    ...     def get_state(self):
    ...         return {"virus_count": self.virus_count}
    """
    return module_registry.register(
        name=name,
        version=version,
        description=description,
        **metadata
    )


def get_module_class(module_name: str) -> Type[SimulationModule]:
    """
    Get a module class by name.

    Parameters
    ----------
    module_name : str
        Module name (e.g., "cellular", "immune")

    Returns
    -------
    Type[SimulationModule]
        The module class
    """
    return module_registry.get(module_name)


def list_modules() -> List[str]:
    """List all registered module names."""
    return module_registry.list_names()


def create_module(module_name: str, config: Dict = None) -> SimulationModule:
    """
    Create a module instance by name.

    Parameters
    ----------
    module_name : str
        Module name
    config : Dict, optional
        Module configuration

    Returns
    -------
    SimulationModule
        New module instance
    """
    return module_registry.create(module_name, config=config)


def discover_plugins(entry_point_group: str = "cognisom.modules") -> int:
    """
    Discover and register modules from entry points.

    Parameters
    ----------
    entry_point_group : str
        Entry point group name

    Returns
    -------
    int
        Number of plugins discovered
    """
    return module_registry.discover_plugins(entry_point_group)


# ── Built-in Module Registration ─────────────────────────────────────

# Map of module names to classes (for backward compatibility)
_BUILTIN_MODULES = {
    "molecular": MolecularModule,
    "cellular": CellularModule,
    "immune": ImmuneModule,
    "vascular": VascularModule,
    "lymphatic": LymphaticModule,
    "spatial": SpatialModule,
    "epigenetic": EpigeneticModule,
    "circadian": CircadianModule,
    "morphogen": MorphogenModule,
    "cell_mechanics": CellMechanicsModule,
}


def _register_builtin_modules():
    """Register all built-in modules."""
    for name, cls in _BUILTIN_MODULES.items():
        module_registry.register_class(
            name,
            cls,
            version="1.0.0",
            tags=["builtin"],
            description=cls.__doc__.strip().split("\n")[0] if cls.__doc__ else "",
        )
    log.debug(f"Module registry initialized with {len(module_registry)} modules")


# Register built-in modules on import
_register_builtin_modules()


# ── Exports ──────────────────────────────────────────────────────────

__all__ = [
    # Built-in modules
    'MolecularModule',
    'CellularModule',
    'ImmuneModule',
    'VascularModule',
    'LymphaticModule',
    'SpatialModule',
    'EpigeneticModule',
    'CircadianModule',
    'MorphogenModule',
    'CellMechanicsModule',  # Phase A
    # Registry API
    'module_registry',
    'register_module',
    'get_module_class',
    'list_modules',
    'create_module',
    'discover_plugins',
]
