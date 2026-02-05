"""
Scene Manager (Phase 8)
=======================

Dynamic scene management for biological entities in Omniverse.

Handles:
- Entity creation/deletion in USD scene
- Visual representation mapping (cell type -> geometry/material)
- Efficient batch updates for large simulations
- LOD (Level of Detail) management
- Spatial partitioning for culling

Usage::

    from cognisom.omniverse import SceneManager

    manager = SceneManager(connector)
    manager.initialize()

    # Update entity in scene
    manager.update_entity("cell_001", position=[10, 5, 3], radius=5.0)

    # Remove entity
    manager.remove_entity("cell_001")
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Types of biological entities for visualization."""
    CELL = "cell"
    TUMOR_CELL = "tumor_cell"
    IMMUNE_CELL = "immune_cell"
    STROMAL_CELL = "stromal_cell"
    BLOOD_VESSEL = "blood_vessel"
    ECM = "ecm"                    # Extracellular matrix
    DRUG_PARTICLE = "drug_particle"
    SIGNALING_MOLECULE = "signaling_molecule"


class EntityState(str, Enum):
    """Entity lifecycle states."""
    NORMAL = "normal"
    DIVIDING = "dividing"
    APOPTOTIC = "apoptotic"
    NECROTIC = "necrotic"
    QUIESCENT = "quiescent"
    MIGRATING = "migrating"
    STRESSED = "stressed"


@dataclass
class EntityVisual:
    """Visual representation of an entity."""
    entity_id: str = ""
    entity_type: EntityType = EntityType.CELL
    state: EntityState = EntityState.NORMAL
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # Quaternion
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    radius: float = 5.0
    color: Tuple[float, float, float] = (0.8, 0.2, 0.2)  # RGB normalized
    opacity: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    prim_path: str = ""           # USD prim path
    lod_level: int = 0            # Level of detail (0 = highest)
    visible: bool = True
    last_updated: float = 0.0

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = time.time()


@dataclass
class MaterialConfig:
    """Material configuration for entity types."""
    base_color: Tuple[float, float, float] = (0.8, 0.2, 0.2)
    metallic: float = 0.0
    roughness: float = 0.7
    emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    opacity: float = 1.0
    subsurface: float = 0.5      # For cell membrane effect


# Default materials by entity type and state
ENTITY_MATERIALS: Dict[Tuple[EntityType, EntityState], MaterialConfig] = {
    # Normal cells
    (EntityType.CELL, EntityState.NORMAL): MaterialConfig(
        base_color=(0.9, 0.7, 0.6), roughness=0.8, subsurface=0.6
    ),
    (EntityType.TUMOR_CELL, EntityState.NORMAL): MaterialConfig(
        base_color=(0.8, 0.2, 0.2), roughness=0.6, emissive=(0.1, 0.0, 0.0)
    ),
    (EntityType.IMMUNE_CELL, EntityState.NORMAL): MaterialConfig(
        base_color=(0.2, 0.6, 0.9), roughness=0.7, subsurface=0.5
    ),
    (EntityType.STROMAL_CELL, EntityState.NORMAL): MaterialConfig(
        base_color=(0.6, 0.8, 0.6), roughness=0.9
    ),

    # Dividing state
    (EntityType.TUMOR_CELL, EntityState.DIVIDING): MaterialConfig(
        base_color=(1.0, 0.4, 0.2), roughness=0.5, emissive=(0.2, 0.1, 0.0)
    ),
    (EntityType.CELL, EntityState.DIVIDING): MaterialConfig(
        base_color=(0.9, 0.9, 0.5), roughness=0.6
    ),

    # Apoptotic state
    (EntityType.TUMOR_CELL, EntityState.APOPTOTIC): MaterialConfig(
        base_color=(0.4, 0.2, 0.3), roughness=0.9, opacity=0.7
    ),
    (EntityType.CELL, EntityState.APOPTOTIC): MaterialConfig(
        base_color=(0.5, 0.4, 0.4), roughness=0.9, opacity=0.6
    ),

    # Blood vessels
    (EntityType.BLOOD_VESSEL, EntityState.NORMAL): MaterialConfig(
        base_color=(0.8, 0.1, 0.1), roughness=0.3, subsurface=0.8
    ),

    # Drug particles
    (EntityType.DRUG_PARTICLE, EntityState.NORMAL): MaterialConfig(
        base_color=(0.2, 0.9, 0.3), roughness=0.2, emissive=(0.0, 0.2, 0.0), opacity=0.8
    ),
}


class SceneManager:
    """Manages biological entity visualization in USD scene.

    Handles creation, updates, and removal of entities with efficient
    batch processing and LOD management.
    """

    # Scene hierarchy paths
    ROOT_PATH = "/World"
    CELLS_PATH = "/World/Cells"
    VESSELS_PATH = "/World/Vessels"
    PARTICLES_PATH = "/World/Particles"
    LIGHTS_PATH = "/World/Lights"

    # LOD thresholds (distance from camera)
    LOD_THRESHOLDS = [50.0, 150.0, 500.0]

    def __init__(self, connector) -> None:
        """Initialize scene manager.

        Args:
            connector: OmniverseConnector instance
        """
        self._connector = connector
        self._stage = None
        self._entities: Dict[str, EntityVisual] = {}
        self._materials: Dict[str, Any] = {}
        self._pending_updates: List[EntityVisual] = []
        self._pending_removals: Set[str] = set()

        # Spatial partitioning for culling
        self._spatial_grid: Dict[Tuple[int, int, int], Set[str]] = {}
        self._grid_size = 50.0  # Grid cell size

        # Performance tracking
        self._last_batch_time = 0.0
        self._batch_count = 0

    def initialize(self) -> bool:
        """Initialize the scene structure."""
        if not self._connector or not self._connector.is_connected:
            log.warning("Connector not available or not connected")
            return False

        self._stage = self._connector.get_stage()
        if not self._stage:
            log.error("Failed to get USD stage")
            return False

        # Create scene hierarchy
        self._create_hierarchy()

        # Create default materials
        self._create_materials()

        log.info("Scene manager initialized")
        return True

    def _create_hierarchy(self) -> None:
        """Create default scene hierarchy."""
        paths = [
            self.ROOT_PATH,
            self.CELLS_PATH,
            self.VESSELS_PATH,
            self.PARTICLES_PATH,
            self.LIGHTS_PATH,
        ]

        for path in paths:
            try:
                if not self._stage.GetPrimAtPath(path):
                    self._stage.DefinePrim(path, "Xform")
            except Exception as e:
                log.warning("Failed to create path %s: %s", path, e)

    def _create_materials(self) -> None:
        """Create materials for entity types."""
        for (etype, state), config in ENTITY_MATERIALS.items():
            mat_name = f"Mat_{etype.value}_{state.value}"
            mat_path = f"/World/Materials/{mat_name}"

            try:
                # Create material prim
                mat_prim = self._stage.DefinePrim(mat_path, "Material")

                # Set material properties (simplified - real impl uses UsdShade)
                if hasattr(mat_prim, "CreateAttribute"):
                    mat_prim.CreateAttribute("inputs:diffuseColor").Set(config.base_color)
                    mat_prim.CreateAttribute("inputs:roughness").Set(config.roughness)
                    mat_prim.CreateAttribute("inputs:metallic").Set(config.metallic)
                    mat_prim.CreateAttribute("inputs:opacity").Set(config.opacity)

                self._materials[(etype, state)] = mat_path

            except Exception as e:
                log.debug("Material creation (mock mode): %s", e)
                self._materials[(etype, state)] = mat_path

    # ── Entity Management ───────────────────────────────────────────────

    def update_entity(
        self,
        entity_id: str,
        position: Optional[List[float]] = None,
        radius: Optional[float] = None,
        state: Optional[str] = None,
        entity_type: Optional[str] = None,
        color: Optional[Tuple[float, float, float]] = None,
        metadata: Optional[Dict] = None,
    ) -> EntityVisual:
        """Update or create an entity in the scene.

        Args:
            entity_id: Unique entity identifier
            position: [x, y, z] position
            radius: Entity radius
            state: Entity state (normal, dividing, etc.)
            entity_type: Type of entity (cell, tumor_cell, etc.)
            color: Override color (RGB normalized)
            metadata: Additional metadata

        Returns:
            Updated EntityVisual
        """
        # Get or create entity
        if entity_id in self._entities:
            entity = self._entities[entity_id]
        else:
            entity = EntityVisual(entity_id=entity_id)
            self._entities[entity_id] = entity

        # Update properties
        if position is not None:
            old_pos = entity.position
            entity.position = tuple(position)
            self._update_spatial_grid(entity_id, old_pos, entity.position)

        if radius is not None:
            entity.radius = radius
            entity.scale = (radius / 5.0, radius / 5.0, radius / 5.0)

        if state is not None:
            try:
                entity.state = EntityState(state)
            except ValueError:
                entity.state = EntityState.NORMAL

        if entity_type is not None:
            try:
                entity.entity_type = EntityType(entity_type)
            except ValueError:
                entity.entity_type = EntityType.CELL

        if color is not None:
            entity.color = color

        if metadata is not None:
            entity.metadata.update(metadata)

        entity.last_updated = time.time()

        # Queue for batch update
        self._pending_updates.append(entity)

        # Create/update USD prim if not batching
        if len(self._pending_updates) == 1:
            self._create_or_update_prim(entity)

        return entity

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity from the scene.

        Args:
            entity_id: Entity to remove

        Returns:
            True if entity was removed
        """
        if entity_id not in self._entities:
            return False

        entity = self._entities[entity_id]

        # Remove from spatial grid
        grid_key = self._get_grid_key(entity.position)
        if grid_key in self._spatial_grid:
            self._spatial_grid[grid_key].discard(entity_id)

        # Queue for removal
        self._pending_removals.add(entity_id)

        # Remove USD prim
        if entity.prim_path and self._stage:
            try:
                self._stage.RemovePrim(entity.prim_path)
            except Exception as e:
                log.debug("Prim removal error: %s", e)

        del self._entities[entity_id]
        return True

    def _create_or_update_prim(self, entity: EntityVisual) -> None:
        """Create or update USD prim for entity."""
        if not self._stage:
            return

        # Determine parent path based on type
        if entity.entity_type in (EntityType.BLOOD_VESSEL,):
            parent_path = self.VESSELS_PATH
        elif entity.entity_type in (EntityType.DRUG_PARTICLE, EntityType.SIGNALING_MOLECULE):
            parent_path = self.PARTICLES_PATH
        else:
            parent_path = self.CELLS_PATH

        prim_path = f"{parent_path}/{entity.entity_id}"
        entity.prim_path = prim_path

        try:
            # Get or create prim
            prim = self._stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                # Create sphere for cell
                prim = self._stage.DefinePrim(prim_path, "Sphere")

            # Set transform
            if hasattr(prim, "GetAttribute"):
                # Position
                translate_attr = prim.GetAttribute("xformOp:translate")
                if not translate_attr:
                    translate_attr = prim.CreateAttribute("xformOp:translate")
                translate_attr.Set(entity.position)

                # Scale
                scale_attr = prim.GetAttribute("xformOp:scale")
                if not scale_attr:
                    scale_attr = prim.CreateAttribute("xformOp:scale")
                scale_attr.Set(entity.scale)

                # Radius (for Sphere)
                radius_attr = prim.GetAttribute("radius")
                if radius_attr:
                    radius_attr.Set(entity.radius)

            # Apply material
            mat_key = (entity.entity_type, entity.state)
            if mat_key in self._materials:
                self._bind_material(prim, self._materials[mat_key])

        except Exception as e:
            log.debug("Prim update (mock mode): %s", e)

    def _bind_material(self, prim, material_path: str) -> None:
        """Bind material to prim."""
        # In real USD, this uses UsdShade.MaterialBindingAPI
        try:
            mat_attr = prim.GetAttribute("material:binding")
            if not mat_attr:
                mat_attr = prim.CreateAttribute("material:binding")
            mat_attr.Set(material_path)
        except Exception:
            pass

    # ── Batch Processing ────────────────────────────────────────────────

    def flush_updates(self) -> int:
        """Process all pending updates.

        Returns:
            Number of entities updated
        """
        t0 = time.time()
        count = 0

        # Process updates
        for entity in self._pending_updates:
            self._create_or_update_prim(entity)
            count += 1

        self._pending_updates.clear()

        # Process removals
        for entity_id in self._pending_removals:
            if entity_id in self._entities:
                entity = self._entities[entity_id]
                if entity.prim_path and self._stage:
                    try:
                        self._stage.RemovePrim(entity.prim_path)
                    except Exception:
                        pass

        self._pending_removals.clear()

        self._last_batch_time = time.time() - t0
        self._batch_count += 1

        return count

    def batch_update(self, entities: List[Dict]) -> int:
        """Batch update multiple entities.

        Args:
            entities: List of entity update dictionaries

        Returns:
            Number of entities updated
        """
        for entity_data in entities:
            self.update_entity(**entity_data)

        return self.flush_updates()

    # ── Spatial Partitioning ────────────────────────────────────────────

    def _get_grid_key(self, position: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Get spatial grid key for position."""
        return (
            int(position[0] / self._grid_size),
            int(position[1] / self._grid_size),
            int(position[2] / self._grid_size),
        )

    def _update_spatial_grid(
        self,
        entity_id: str,
        old_pos: Tuple[float, float, float],
        new_pos: Tuple[float, float, float]
    ) -> None:
        """Update entity position in spatial grid."""
        old_key = self._get_grid_key(old_pos)
        new_key = self._get_grid_key(new_pos)

        if old_key != new_key:
            # Remove from old cell
            if old_key in self._spatial_grid:
                self._spatial_grid[old_key].discard(entity_id)

            # Add to new cell
            if new_key not in self._spatial_grid:
                self._spatial_grid[new_key] = set()
            self._spatial_grid[new_key].add(entity_id)

    def get_entities_in_region(
        self,
        center: Tuple[float, float, float],
        radius: float
    ) -> List[EntityVisual]:
        """Get entities within a spherical region.

        Args:
            center: Center of region
            radius: Radius of region

        Returns:
            List of entities in region
        """
        results = []
        grid_radius = int(radius / self._grid_size) + 1

        center_key = self._get_grid_key(center)

        # Check all grid cells in range
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                for dz in range(-grid_radius, grid_radius + 1):
                    key = (center_key[0] + dx, center_key[1] + dy, center_key[2] + dz)
                    if key in self._spatial_grid:
                        for entity_id in self._spatial_grid[key]:
                            entity = self._entities.get(entity_id)
                            if entity:
                                # Check actual distance
                                dist = math.sqrt(
                                    (entity.position[0] - center[0]) ** 2 +
                                    (entity.position[1] - center[1]) ** 2 +
                                    (entity.position[2] - center[2]) ** 2
                                )
                                if dist <= radius:
                                    results.append(entity)

        return results

    # ── LOD Management ──────────────────────────────────────────────────

    def update_lod(self, camera_position: Tuple[float, float, float]) -> None:
        """Update LOD levels based on camera position.

        Args:
            camera_position: Current camera position
        """
        for entity in self._entities.values():
            dist = math.sqrt(
                (entity.position[0] - camera_position[0]) ** 2 +
                (entity.position[1] - camera_position[1]) ** 2 +
                (entity.position[2] - camera_position[2]) ** 2
            )

            # Determine LOD level
            new_lod = 0
            for i, threshold in enumerate(self.LOD_THRESHOLDS):
                if dist > threshold:
                    new_lod = i + 1

            if new_lod != entity.lod_level:
                entity.lod_level = new_lod
                self._apply_lod(entity)

    def _apply_lod(self, entity: EntityVisual) -> None:
        """Apply LOD settings to entity."""
        if not self._stage or not entity.prim_path:
            return

        try:
            prim = self._stage.GetPrimAtPath(entity.prim_path)
            if not prim:
                return

            # Adjust complexity based on LOD
            if entity.lod_level == 0:
                # Full detail
                entity.visible = True
            elif entity.lod_level == 1:
                # Medium detail - could swap geometry
                entity.visible = True
            elif entity.lod_level == 2:
                # Low detail - simplified geometry
                entity.visible = True
            else:
                # Very far - could hide or use billboards
                entity.visible = entity.lod_level < 4

            # Set visibility
            vis_attr = prim.GetAttribute("visibility")
            if vis_attr:
                vis_attr.Set("inherited" if entity.visible else "invisible")

        except Exception:
            pass

    # ── Culling ─────────────────────────────────────────────────────────

    def cull_entities(
        self,
        camera_position: Tuple[float, float, float],
        view_direction: Tuple[float, float, float],
        fov: float = 90.0,
        far_plane: float = 1000.0
    ) -> int:
        """Cull entities outside view frustum.

        Args:
            camera_position: Camera position
            view_direction: Normalized view direction
            fov: Field of view in degrees
            far_plane: Far clipping plane distance

        Returns:
            Number of entities culled
        """
        culled = 0
        half_fov = math.radians(fov / 2)
        cos_half_fov = math.cos(half_fov)

        for entity in self._entities.values():
            # Vector from camera to entity
            to_entity = (
                entity.position[0] - camera_position[0],
                entity.position[1] - camera_position[1],
                entity.position[2] - camera_position[2],
            )

            dist = math.sqrt(to_entity[0]**2 + to_entity[1]**2 + to_entity[2]**2)

            if dist > far_plane:
                # Beyond far plane
                entity.visible = False
                culled += 1
                continue

            if dist > 0:
                # Normalize
                to_entity = (to_entity[0]/dist, to_entity[1]/dist, to_entity[2]/dist)

                # Dot product with view direction
                dot = (
                    to_entity[0] * view_direction[0] +
                    to_entity[1] * view_direction[1] +
                    to_entity[2] * view_direction[2]
                )

                if dot < cos_half_fov:
                    # Outside FOV
                    entity.visible = False
                    culled += 1
                else:
                    entity.visible = True
            else:
                entity.visible = True

        return culled

    # ── Status ──────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get scene statistics."""
        type_counts = {}
        state_counts = {}

        for entity in self._entities.values():
            t = entity.entity_type.value
            s = entity.state.value
            type_counts[t] = type_counts.get(t, 0) + 1
            state_counts[s] = state_counts.get(s, 0) + 1

        return {
            "total_entities": len(self._entities),
            "visible_entities": sum(1 for e in self._entities.values() if e.visible),
            "pending_updates": len(self._pending_updates),
            "pending_removals": len(self._pending_removals),
            "grid_cells": len(self._spatial_grid),
            "type_counts": type_counts,
            "state_counts": state_counts,
            "last_batch_time_ms": self._last_batch_time * 1000,
            "batch_count": self._batch_count,
        }

    def clear(self) -> None:
        """Clear all entities from scene."""
        for entity_id in list(self._entities.keys()):
            self.remove_entity(entity_id)

        self._spatial_grid.clear()
        self.flush_updates()
        log.info("Scene cleared")
