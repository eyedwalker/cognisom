"""
Cell Mechanics Simulation Module
================================

Simulation module that integrates GPU-accelerated cell mechanics
with the Cognisom simulation engine.

This module bridges:
- CellularModule (cell state, cycle, metabolism)
- SpatialModule (concentration fields)
- CellMechanics (force-based dynamics)

Features:
- Cell-cell collision and adhesion
- Chemotaxis toward attractant gradients
- Brownian motion for stochastic movement
- Integration with spatial fields for chemotaxis

Phase A.2 of the Strategic Implementation Plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cognisom.core.module_base import SimulationModule
from cognisom.physics.cell_mechanics import (
    CellMechanics,
    CellMechanicsConfig,
    ForceType,
)

log = logging.getLogger(__name__)


@dataclass
class CellMechanicsModuleConfig:
    """Configuration for cell mechanics module."""
    # Physics parameters
    k_repulsion: float = 100.0      # Repulsion stiffness (pN/um)
    k_adhesion: float = 10.0        # Adhesion stiffness (pN/um)
    adhesion_range: float = 2.0     # Adhesion distance (um)
    noise_strength: float = 0.1     # Brownian noise
    damping: float = 1.0            # Viscous damping

    # Chemotaxis
    chemotaxis_field: str = ""      # Name of concentration field to follow
    chemotaxis_strength: float = 1.0

    # Integration
    substeps_per_update: int = 10   # Physics substeps per simulation step
    overdamped: bool = True         # Use overdamped dynamics

    # Force types
    enable_repulsion: bool = True
    enable_adhesion: bool = True
    enable_chemotaxis: bool = True
    enable_brownian: bool = True


class CellMechanicsModule(SimulationModule):
    """
    Simulation module for cell mechanics.

    Integrates force-based cell dynamics with the simulation engine.
    Cells experience:
    - Repulsion (soft-sphere collision avoidance)
    - Adhesion (E-cadherin mediated cell-cell adhesion)
    - Chemotaxis (gradient following)
    - Brownian motion (thermal noise)

    Events Emitted:
    - CELL_COLLISION: When cells collide
    - CELL_ADHESION_FORMED: When adhesion forms between cells
    - CELL_MOVED: After significant cell displacement

    Events Subscribed:
    - CELL_CREATED: Add new cell to physics simulation
    - CELL_DIED: Remove cell from physics simulation
    - CELL_DIVIDED: Add daughter cells
    """

    def __init__(self, config: Dict = None):
        """
        Initialize cell mechanics module.

        Parameters
        ----------
        config : Dict, optional
            Module configuration
        """
        super().__init__(config)

        # Parse config
        self.module_config = CellMechanicsModuleConfig(
            **{k: v for k, v in (config or {}).items()
               if hasattr(CellMechanicsModuleConfig, k)}
        )

        # Cell mechanics solver (created during initialize)
        self._mechanics: Optional[CellMechanics] = None

        # Cell registry (cell_id -> index mapping)
        self._cell_ids: List[int] = []
        self._cell_id_to_index: Dict[int, int] = {}

        # Reference to other modules (set during engine setup)
        self._cellular_module = None
        self._spatial_module = None

        # Statistics
        self._total_collisions = 0
        self._total_adhesions = 0

    def initialize(self):
        """Initialize the cell mechanics solver."""
        # Build force types from config
        force_types = ForceType.NONE
        if self.module_config.enable_repulsion:
            force_types |= ForceType.REPULSION
        if self.module_config.enable_adhesion:
            force_types |= ForceType.ADHESION
        if self.module_config.enable_chemotaxis:
            force_types |= ForceType.CHEMOTAXIS
        if self.module_config.enable_brownian:
            force_types |= ForceType.BROWNIAN

        # Create physics config
        physics_config = CellMechanicsConfig(
            k_repulsion=self.module_config.k_repulsion,
            k_adhesion=self.module_config.k_adhesion,
            adhesion_range=self.module_config.adhesion_range,
            noise_strength=self.module_config.noise_strength,
            damping=self.module_config.damping,
            overdamped=self.module_config.overdamped,
            force_types=force_types,
        )

        # Start with capacity for 1000 cells (will grow as needed)
        self._mechanics = CellMechanics(
            n_cells=1000,
            config=physics_config,
        )

        # Subscribe to events
        self.subscribe("CELL_CREATED", self._on_cell_created)
        self.subscribe("CELL_DIED", self._on_cell_died)
        self.subscribe("CELL_DIVIDED", self._on_cell_divided)

        log.info("CellMechanicsModule initialized")

    def update(self, dt: float):
        """
        Update cell positions using force-based dynamics.

        Parameters
        ----------
        dt : float
            Time step in hours
        """
        if not self.enabled or self._mechanics is None:
            return

        n_active = len(self._cell_ids)
        if n_active == 0:
            return

        # Sync positions from cellular module
        self._sync_from_cellular()

        # Update chemotaxis from spatial fields
        if self.module_config.enable_chemotaxis and self._spatial_module:
            self._update_chemotaxis_gradient()

        # Run physics substeps
        physics_dt = dt / self.module_config.substeps_per_update

        for _ in range(self.module_config.substeps_per_update):
            self._mechanics.step(physics_dt)

        # Sync positions back to cellular module
        self._sync_to_cellular()

    def _sync_from_cellular(self):
        """Sync cell positions from cellular module."""
        if self._cellular_module is None:
            return

        # Get cell data from cellular module
        cells = getattr(self._cellular_module, 'cells', [])

        positions = []
        radii = []
        cell_types = []

        for cell_id in self._cell_ids:
            idx = self._cell_id_to_index.get(cell_id)
            if idx is None or idx >= len(cells):
                continue

            cell = cells[idx]

            pos = getattr(cell, 'position', (0, 0, 0))
            if isinstance(pos, (list, tuple)):
                pos = list(pos)
            else:
                pos = [pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)]

            positions.append(pos)

            # Get radius from volume (assuming spherical cells)
            volume = getattr(cell, 'volume', 1.0)
            radius = (3 * volume / (4 * np.pi)) ** (1/3) * 5  # Scale factor
            radii.append(radius)

            # Cell type for adhesion
            ctype = getattr(cell, 'cell_type', 0)
            if hasattr(ctype, 'value'):
                ctype = hash(ctype.value) % 100
            cell_types.append(int(ctype))

        if positions:
            self._mechanics.set_positions(np.array(positions, dtype=np.float32))
            self._mechanics.set_radii(np.array(radii, dtype=np.float32))
            self._mechanics.set_cell_types(np.array(cell_types, dtype=np.int32))

    def _sync_to_cellular(self):
        """Sync updated positions back to cellular module."""
        if self._cellular_module is None:
            return

        cells = getattr(self._cellular_module, 'cells', [])
        positions = self._mechanics.get_positions()

        for i, cell_id in enumerate(self._cell_ids):
            idx = self._cell_id_to_index.get(cell_id)
            if idx is None or idx >= len(cells):
                continue

            cell = cells[idx]
            if i < len(positions):
                pos = positions[i]
                if hasattr(cell, 'position'):
                    cell.position = (float(pos[0]), float(pos[1]), float(pos[2]))

    def _update_chemotaxis_gradient(self):
        """Update chemotaxis strength based on spatial gradients."""
        if not self.module_config.chemotaxis_field:
            return

        # TODO: Get gradient field from spatial module and pass to mechanics
        # This requires computing ∇[concentration] from the spatial field

    def _on_cell_created(self, data: Dict):
        """Handle new cell creation."""
        cell_id = data.get("cell_id", len(self._cell_ids))
        position = data.get("position", (0, 0, 0))

        # Add to registry
        index = len(self._cell_ids)
        self._cell_ids.append(cell_id)
        self._cell_id_to_index[cell_id] = index

        log.debug(f"Cell {cell_id} added to mechanics at index {index}")

    def _on_cell_died(self, data: Dict):
        """Handle cell death."""
        cell_id = data.get("cell_id")
        if cell_id in self._cell_id_to_index:
            # Remove from registry (compact later if needed)
            del self._cell_id_to_index[cell_id]
            if cell_id in self._cell_ids:
                self._cell_ids.remove(cell_id)

    def _on_cell_divided(self, data: Dict):
        """Handle cell division."""
        parent_id = data.get("parent_id")
        daughter_ids = data.get("daughter_ids", [])

        for d_id in daughter_ids:
            self._on_cell_created({"cell_id": d_id})

    def set_cellular_module(self, module):
        """Set reference to cellular module."""
        self._cellular_module = module

    def set_spatial_module(self, module):
        """Set reference to spatial module."""
        self._spatial_module = module

    def get_state(self) -> Dict[str, Any]:
        """Return current module state."""
        return {
            "n_cells": len(self._cell_ids),
            "total_collisions": self._total_collisions,
            "total_adhesions": self._total_adhesions,
            "physics_time": self._mechanics.time if self._mechanics else 0,
            "config": {
                "k_repulsion": self.module_config.k_repulsion,
                "k_adhesion": self.module_config.k_adhesion,
                "chemotaxis_field": self.module_config.chemotaxis_field,
            }
        }

    def get_cell_positions(self) -> np.ndarray:
        """Get current cell positions."""
        if self._mechanics:
            return self._mechanics.get_positions()
        return np.array([])

    def get_cell_velocities(self) -> np.ndarray:
        """Get current cell velocities."""
        if self._mechanics:
            return self._mechanics.get_velocities()
        return np.array([])

    def get_cell_forces(self) -> np.ndarray:
        """Get current forces on cells."""
        if self._mechanics:
            return self._mechanics.get_forces()
        return np.array([])

    def add_cell(
        self,
        cell_id: int,
        position: Tuple[float, float, float],
        radius: float = 5.0,
        cell_type: int = 0,
    ):
        """
        Add a cell to the mechanics simulation.

        Parameters
        ----------
        cell_id : int
            Unique cell identifier
        position : tuple
            (x, y, z) position
        radius : float
            Cell radius
        cell_type : int
            Cell type for adhesion rules
        """
        index = len(self._cell_ids)
        self._cell_ids.append(cell_id)
        self._cell_id_to_index[cell_id] = index

        # Add to mechanics (resize if needed)
        if self._mechanics and index < self._mechanics.n_cells:
            positions = self._mechanics.get_positions()
            if index < len(positions):
                positions[index] = position
                self._mechanics.set_positions(positions)

    def seed_cells(
        self,
        n_cells: int,
        region: Tuple[float, float, float, float, float, float],
        radius: float = 5.0,
    ):
        """
        Seed random cells in a region.

        Parameters
        ----------
        n_cells : int
            Number of cells
        region : tuple
            (x_min, y_min, z_min, x_max, y_max, z_max)
        radius : float
            Cell radius
        """
        x_min, y_min, z_min, x_max, y_max, z_max = region

        positions = np.random.uniform(
            [x_min, y_min, z_min],
            [x_max, y_max, z_max],
            size=(n_cells, 3)
        ).astype(np.float32)

        # Ensure mechanics has enough capacity
        if self._mechanics is None or n_cells > self._mechanics.n_cells:
            physics_config = CellMechanicsConfig(
                k_repulsion=self.module_config.k_repulsion,
                k_adhesion=self.module_config.k_adhesion,
                adhesion_range=self.module_config.adhesion_range,
                noise_strength=self.module_config.noise_strength,
                damping=self.module_config.damping,
                overdamped=self.module_config.overdamped,
            )
            self._mechanics = CellMechanics(n_cells=n_cells, config=physics_config)

        self._mechanics.set_positions(positions)
        self._mechanics.set_radii(np.full(n_cells, radius, dtype=np.float32))

        # Register cells
        self._cell_ids = list(range(n_cells))
        self._cell_id_to_index = {i: i for i in range(n_cells)}

        log.info(f"Seeded {n_cells} cells in region")
