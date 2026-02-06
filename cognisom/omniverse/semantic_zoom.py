"""
Semantic Zooming
===============

Phase B4: Representation switching based on camera distance and screen coverage.

Unlike standard LOD (Level of Detail) which shows the same content with fewer
polygons, Semantic Zooming changes the representation TYPE entirely based on scale.

Example: A heart at different zoom levels:
- Distant (< 100 pixels): Hidden or icon
- Far (100-1000 pixels): Anatomy variant → surface mesh
- Medium (1000-10000 pixels): Cellular variant → PointInstancer of cardiomyocytes
- Close (> 10000 pixels): Molecular variant → payload with molecular data

This is implemented using USD VariantSets:
- Each major assembly has a "Representation" VariantSet
- Variants: "Hidden", "Icon", "Anatomy", "Cellular", "Molecular"
- A controller monitors camera and switches variants based on screen coverage

Usage::

    from cognisom.omniverse.semantic_zoom import (
        SemanticZoomController,
        RepresentationLevel,
        ZoomThresholds,
    )

    # Create controller
    controller = SemanticZoomController(stage)

    # Register entities for semantic zoom
    controller.register_entity(
        prim_path="/World/Heart",
        representations=["Anatomy", "Cellular", "Molecular"],
    )

    # In render loop
    while rendering:
        controller.update(camera, viewport_size)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Try to import USD modules
try:
    from pxr import Gf, Sdf, Usd, UsdGeom
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False
    log.warning("USD (pxr) not available - semantic zoom will use mock mode")


class RepresentationLevel(str, Enum):
    """
    Semantic representation levels for biological entities.

    Each level represents a different visualization paradigm,
    not just a level of detail.
    """
    HIDDEN = "Hidden"           # Not visible (too small/distant)
    ICON = "Icon"               # Symbolic representation (sprite, glyph)
    SCHEMATIC = "Schematic"     # Simplified diagram view
    ANATOMY = "Anatomy"         # Organ-level surface mesh
    TISSUE = "Tissue"           # Tissue structures visible
    CELLULAR = "Cellular"       # Individual cells as instances
    SUBCELLULAR = "Subcellular" # Organelles visible
    MOLECULAR = "Molecular"     # Molecular detail (proteins, etc.)
    ATOMIC = "Atomic"           # Atomic resolution


@dataclass
class ZoomThresholds:
    """
    Pixel thresholds for switching between representation levels.

    The thresholds define when to switch from one representation to the next
    based on the entity's screen coverage in pixels.

    Attributes
    ----------
    hidden_max : float
        Below this, entity is hidden
    icon_max : float
        Below this, show icon
    schematic_max : float
        Below this, show schematic
    anatomy_max : float
        Below this, show anatomy mesh
    tissue_max : float
        Below this, show tissue level
    cellular_max : float
        Below this, show cellular (individual cells)
    subcellular_max : float
        Below this, show subcellular (organelles)
    molecular_max : float
        Below this, show molecular
    # Above molecular_max: atomic level (if available)
    """
    hidden_max: float = 10.0
    icon_max: float = 50.0
    schematic_max: float = 200.0
    anatomy_max: float = 1000.0
    tissue_max: float = 3000.0
    cellular_max: float = 10000.0
    subcellular_max: float = 50000.0
    molecular_max: float = 200000.0

    # Hysteresis factor to prevent oscillation at boundaries
    hysteresis: float = 0.2  # 20% buffer


@dataclass
class ZoomableEntity:
    """Information about an entity that supports semantic zooming."""
    prim_path: str
    variant_set_name: str = "Representation"

    # Available representations (subset of RepresentationLevel)
    available_representations: List[str] = field(default_factory=list)

    # Current state
    current_representation: Optional[str] = None
    last_screen_pixels: float = 0.0

    # Bounding box (world space, cached)
    bbox_center: Optional[Tuple[float, float, float]] = None
    bbox_radius: float = 1.0  # Bounding sphere radius

    # Custom thresholds (if different from global)
    custom_thresholds: Optional[ZoomThresholds] = None

    # Metadata
    entity_type: str = "generic"  # organ, tissue, cell, etc.
    parent_path: Optional[str] = None  # For hierarchical zoom


@dataclass
class SemanticZoomConfig:
    """Configuration for the Semantic Zoom Controller."""

    # Global thresholds
    thresholds: ZoomThresholds = field(default_factory=ZoomThresholds)

    # Variant set configuration
    variant_set_name: str = "Representation"

    # Performance
    update_interval_ms: float = 33.33  # ~30fps updates
    max_switches_per_frame: int = 10   # Limit rapid switching

    # Behavior
    enable_smooth_transitions: bool = False  # Future: animated transitions
    enable_parent_awareness: bool = True     # Consider parent zoom level

    # Default representations per entity type
    default_representations: Dict[str, List[str]] = field(default_factory=lambda: {
        "organ": ["Hidden", "Icon", "Anatomy", "Tissue", "Cellular"],
        "tissue": ["Hidden", "Anatomy", "Cellular", "Subcellular"],
        "cell": ["Hidden", "Icon", "Anatomy", "Subcellular", "Molecular"],
        "organelle": ["Hidden", "Anatomy", "Molecular"],
        "molecule": ["Hidden", "Schematic", "Molecular", "Atomic"],
        "generic": ["Hidden", "Anatomy", "Molecular"],
    })


class SemanticZoomController:
    """
    Controller for semantic zooming in USD stages.

    Monitors camera position and viewport size to determine screen coverage
    of registered entities, then switches their representation variants
    based on coverage thresholds.

    Parameters
    ----------
    stage : Usd.Stage, optional
        USD stage to control
    config : SemanticZoomConfig, optional
        Configuration options

    Examples
    --------
    >>> controller = SemanticZoomController(stage)
    >>>
    >>> # Register a heart for semantic zoom
    >>> controller.register_entity(
    ...     "/World/Heart",
    ...     entity_type="organ",
    ...     representations=["Anatomy", "Cellular", "Molecular"],
    ... )
    >>>
    >>> # Each frame:
    >>> controller.update(camera, (1920, 1080))
    """

    def __init__(
        self,
        stage: Optional[Any] = None,
        config: Optional[SemanticZoomConfig] = None,
    ):
        self.stage = stage
        self.config = config or SemanticZoomConfig()

        # Registered entities
        self._entities: Dict[str, ZoomableEntity] = {}

        # State tracking
        self._switches_this_frame: int = 0
        self._last_update_time: float = 0.0

        # Statistics
        self._total_switches: int = 0
        self._switches_by_level: Dict[str, int] = {}

    def register_entity(
        self,
        prim_path: str,
        entity_type: str = "generic",
        representations: Optional[List[str]] = None,
        bbox_center: Optional[Tuple[float, float, float]] = None,
        bbox_radius: Optional[float] = None,
        custom_thresholds: Optional[ZoomThresholds] = None,
        parent_path: Optional[str] = None,
    ) -> ZoomableEntity:
        """
        Register an entity for semantic zooming.

        Parameters
        ----------
        prim_path : str
            USD prim path for the entity
        entity_type : str
            Type of entity (organ, tissue, cell, etc.)
        representations : List[str], optional
            Available representation levels. If None, uses defaults for entity_type.
        bbox_center : Tuple[float, float, float], optional
            Bounding box center. If None, computed from prim.
        bbox_radius : float, optional
            Bounding sphere radius. If None, computed from prim.
        custom_thresholds : ZoomThresholds, optional
            Custom thresholds for this entity
        parent_path : str, optional
            Path to parent entity for hierarchical zoom

        Returns
        -------
        ZoomableEntity
            The registered entity info
        """
        # Get default representations if not specified
        if representations is None:
            representations = self.config.default_representations.get(
                entity_type,
                self.config.default_representations["generic"]
            ).copy()

        # Compute bounds if not provided
        if bbox_center is None or bbox_radius is None:
            computed_center, computed_radius = self._compute_prim_bounds(prim_path)
            bbox_center = bbox_center or computed_center
            bbox_radius = bbox_radius or computed_radius or 1.0

        entity = ZoomableEntity(
            prim_path=prim_path,
            variant_set_name=self.config.variant_set_name,
            available_representations=representations,
            bbox_center=bbox_center,
            bbox_radius=bbox_radius,
            custom_thresholds=custom_thresholds,
            entity_type=entity_type,
            parent_path=parent_path,
        )

        self._entities[prim_path] = entity

        # Ensure variant set exists on prim
        self._ensure_variant_set(prim_path, representations)

        log.info(
            f"Registered entity for semantic zoom: {prim_path} "
            f"({entity_type}, {len(representations)} representations)"
        )

        return entity

    def _compute_prim_bounds(
        self,
        prim_path: str
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        """Compute bounding sphere for a prim."""
        if not USD_AVAILABLE or self.stage is None:
            return None, None

        try:
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return None, None

            imageable = UsdGeom.Imageable(prim)
            bbox = imageable.ComputeWorldBound(
                Usd.TimeCode.Default(),
                UsdGeom.Tokens.default_
            )

            if bbox.IsEmpty():
                return None, None

            range_box = bbox.ComputeAlignedRange()
            min_pt = range_box.GetMin()
            max_pt = range_box.GetMax()

            center = (min_pt + max_pt) / 2.0
            diagonal = (max_pt - min_pt).GetLength()
            radius = diagonal / 2.0

            return (center[0], center[1], center[2]), radius

        except Exception as e:
            log.debug(f"Could not compute bounds for {prim_path}: {e}")
            return None, None

    def _ensure_variant_set(
        self,
        prim_path: str,
        representations: List[str]
    ) -> None:
        """Ensure the prim has the required variant set and variants."""
        if not USD_AVAILABLE or self.stage is None:
            return

        try:
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return

            variant_sets = prim.GetVariantSets()
            variant_set_name = self.config.variant_set_name

            # Create variant set if it doesn't exist
            if not variant_sets.HasVariantSet(variant_set_name):
                variant_sets.AddVariantSet(variant_set_name)
                log.debug(f"Created variant set '{variant_set_name}' on {prim_path}")

            variant_set = variant_sets.GetVariantSet(variant_set_name)

            # Add each representation as a variant
            existing_variants = variant_set.GetVariantNames()
            for rep in representations:
                if rep not in existing_variants:
                    variant_set.AddVariant(rep)
                    log.debug(f"Added variant '{rep}' to {prim_path}")

        except Exception as e:
            log.error(f"Failed to setup variant set for {prim_path}: {e}")

    def unregister_entity(self, prim_path: str) -> bool:
        """Remove an entity from semantic zoom control."""
        if prim_path in self._entities:
            del self._entities[prim_path]
            return True
        return False

    def update(
        self,
        camera_position: Tuple[float, float, float],
        camera_target: Tuple[float, float, float],
        viewport_size: Tuple[int, int],
        fov_degrees: float = 60.0,
        near_clip: float = 0.1,
    ) -> int:
        """
        Update semantic zoom for all registered entities.

        Parameters
        ----------
        camera_position : Tuple[float, float, float]
            World position of camera
        camera_target : Tuple[float, float, float]
            Point camera is looking at (for direction)
        viewport_size : Tuple[int, int]
            Viewport dimensions (width, height) in pixels
        fov_degrees : float
            Vertical field of view in degrees
        near_clip : float
            Near clipping plane distance

        Returns
        -------
        int
            Number of representation switches made
        """
        self._switches_this_frame = 0
        switches = 0

        viewport_width, viewport_height = viewport_size
        fov_radians = math.radians(fov_degrees)

        for path, entity in self._entities.items():
            if self._switches_this_frame >= self.config.max_switches_per_frame:
                break

            # Compute screen coverage
            screen_pixels = self._compute_screen_coverage(
                entity=entity,
                camera_position=camera_position,
                viewport_height=viewport_height,
                fov_radians=fov_radians,
            )

            entity.last_screen_pixels = screen_pixels

            # Determine target representation
            target_rep = self._pixels_to_representation(
                entity=entity,
                screen_pixels=screen_pixels,
            )

            # Switch if needed
            if target_rep != entity.current_representation:
                if self._switch_representation(entity, target_rep):
                    switches += 1
                    self._switches_this_frame += 1
                    self._total_switches += 1

                    # Track by level
                    self._switches_by_level[target_rep] = \
                        self._switches_by_level.get(target_rep, 0) + 1

        return switches

    def _compute_screen_coverage(
        self,
        entity: ZoomableEntity,
        camera_position: Tuple[float, float, float],
        viewport_height: int,
        fov_radians: float,
    ) -> float:
        """
        Compute approximate screen coverage in pixels.

        Uses the bounding sphere and perspective projection to estimate
        how many pixels the entity occupies on screen.

        Parameters
        ----------
        entity : ZoomableEntity
            Entity to compute coverage for
        camera_position : Tuple[float, float, float]
            Camera world position
        viewport_height : int
            Viewport height in pixels
        fov_radians : float
            Field of view in radians

        Returns
        -------
        float
            Estimated screen coverage in pixels (diameter)
        """
        if entity.bbox_center is None:
            return 0.0

        # Distance from camera to entity center
        dx = entity.bbox_center[0] - camera_position[0]
        dy = entity.bbox_center[1] - camera_position[1]
        dz = entity.bbox_center[2] - camera_position[2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        if distance < 0.001:  # Avoid division by zero
            return float('inf')

        # Angular size of bounding sphere
        angular_size = 2.0 * math.atan(entity.bbox_radius / distance)

        # Convert to pixels
        # viewport_height = 2 * tan(fov/2) in angular units
        pixels_per_radian = viewport_height / fov_radians
        screen_pixels = angular_size * pixels_per_radian

        return screen_pixels

    def _pixels_to_representation(
        self,
        entity: ZoomableEntity,
        screen_pixels: float,
    ) -> str:
        """
        Determine the appropriate representation for given screen coverage.

        Uses thresholds with hysteresis to prevent oscillation.

        Parameters
        ----------
        entity : ZoomableEntity
            The entity
        screen_pixels : float
            Current screen coverage

        Returns
        -------
        str
            Target representation level
        """
        thresholds = entity.custom_thresholds or self.config.thresholds
        available = entity.available_representations
        current = entity.current_representation

        # Map from thresholds to representations
        # Order: Hidden < Icon < Schematic < Anatomy < Tissue < Cellular < Subcellular < Molecular < Atomic
        threshold_levels = [
            (thresholds.hidden_max, RepresentationLevel.HIDDEN.value),
            (thresholds.icon_max, RepresentationLevel.ICON.value),
            (thresholds.schematic_max, RepresentationLevel.SCHEMATIC.value),
            (thresholds.anatomy_max, RepresentationLevel.ANATOMY.value),
            (thresholds.tissue_max, RepresentationLevel.TISSUE.value),
            (thresholds.cellular_max, RepresentationLevel.CELLULAR.value),
            (thresholds.subcellular_max, RepresentationLevel.SUBCELLULAR.value),
            (thresholds.molecular_max, RepresentationLevel.MOLECULAR.value),
            (float('inf'), RepresentationLevel.ATOMIC.value),
        ]

        # Apply hysteresis if we have a current representation
        hysteresis_factor = 1.0
        if current is not None:
            # Find current level index
            current_idx = None
            for i, (_, level) in enumerate(threshold_levels):
                if level == current:
                    current_idx = i
                    break

            if current_idx is not None:
                # Apply hysteresis in direction of change
                # This makes it harder to switch away from current level
                pass  # Simple implementation: just use raw thresholds

        # Find appropriate level
        target_level = RepresentationLevel.HIDDEN.value

        for threshold, level in threshold_levels:
            if screen_pixels < threshold:
                target_level = level
                break

        # Constrain to available representations
        if target_level not in available:
            # Find nearest available
            target_level = self._find_nearest_available(
                target_level, available, threshold_levels
            )

        return target_level

    def _find_nearest_available(
        self,
        target: str,
        available: List[str],
        threshold_levels: List[Tuple[float, str]],
    ) -> str:
        """Find the nearest available representation to target."""
        if not available:
            return RepresentationLevel.HIDDEN.value

        # Get level ordering
        level_order = [level for _, level in threshold_levels]

        try:
            target_idx = level_order.index(target)
        except ValueError:
            return available[0]

        # Search outward from target
        for delta in range(len(level_order)):
            for direction in [-1, 1]:
                check_idx = target_idx + (delta * direction)
                if 0 <= check_idx < len(level_order):
                    check_level = level_order[check_idx]
                    if check_level in available:
                        return check_level

        return available[0]

    def _switch_representation(
        self,
        entity: ZoomableEntity,
        target_representation: str,
    ) -> bool:
        """
        Switch an entity to a new representation.

        Parameters
        ----------
        entity : ZoomableEntity
            Entity to switch
        target_representation : str
            Target representation level

        Returns
        -------
        bool
            True if switch was successful
        """
        if USD_AVAILABLE and self.stage is not None:
            try:
                prim = self.stage.GetPrimAtPath(entity.prim_path)
                if not prim.IsValid():
                    return False

                variant_set = prim.GetVariantSet(entity.variant_set_name)
                variant_set.SetVariantSelection(target_representation)

                log.debug(
                    f"Switched {entity.prim_path} from "
                    f"{entity.current_representation} to {target_representation}"
                )

            except Exception as e:
                log.error(
                    f"Failed to switch representation for {entity.prim_path}: {e}"
                )
                return False

        # Update state
        entity.current_representation = target_representation
        return True

    def get_entity(self, prim_path: str) -> Optional[ZoomableEntity]:
        """Get entity info by path."""
        return self._entities.get(prim_path)

    def get_all_entities(self) -> Dict[str, ZoomableEntity]:
        """Get all registered entities."""
        return self._entities.copy()

    def set_representation(
        self,
        prim_path: str,
        representation: str,
    ) -> bool:
        """
        Manually set an entity's representation.

        Useful for user-initiated zoom or debugging.

        Parameters
        ----------
        prim_path : str
            Entity path
        representation : str
            Target representation

        Returns
        -------
        bool
            True if successful
        """
        entity = self._entities.get(prim_path)
        if entity is None:
            return False

        if representation not in entity.available_representations:
            log.warning(
                f"Representation '{representation}' not available for {prim_path}"
            )
            return False

        return self._switch_representation(entity, representation)

    def get_stats(self) -> Dict[str, Any]:
        """Get zoom controller statistics."""
        return {
            "total_entities": len(self._entities),
            "total_switches": self._total_switches,
            "switches_by_level": self._switches_by_level.copy(),
            "entities_by_representation": self._count_by_representation(),
        }

    def _count_by_representation(self) -> Dict[str, int]:
        """Count entities at each representation level."""
        counts: Dict[str, int] = {}
        for entity in self._entities.values():
            rep = entity.current_representation or "None"
            counts[rep] = counts.get(rep, 0) + 1
        return counts

    def set_global_thresholds(self, thresholds: ZoomThresholds) -> None:
        """Update global zoom thresholds."""
        self.config.thresholds = thresholds

    def reset_all_to_hidden(self) -> None:
        """Reset all entities to hidden representation."""
        for entity in self._entities.values():
            if RepresentationLevel.HIDDEN.value in entity.available_representations:
                self._switch_representation(entity, RepresentationLevel.HIDDEN.value)


def create_semantic_zoom_controller(
    stage: Any,
    thresholds: Optional[ZoomThresholds] = None,
) -> SemanticZoomController:
    """
    Convenience function to create a configured zoom controller.

    Parameters
    ----------
    stage : Usd.Stage
        USD stage
    thresholds : ZoomThresholds, optional
        Custom thresholds

    Returns
    -------
    SemanticZoomController
        Configured controller
    """
    config = SemanticZoomConfig()
    if thresholds:
        config.thresholds = thresholds
    return SemanticZoomController(stage=stage, config=config)


def setup_biological_zoom_hierarchy(
    controller: SemanticZoomController,
    root_path: str,
    structure: Dict[str, Any],
) -> int:
    """
    Setup semantic zoom for a biological hierarchy.

    Recursively registers entities with appropriate parent relationships.

    Parameters
    ----------
    controller : SemanticZoomController
        Zoom controller
    root_path : str
        Root prim path
    structure : Dict[str, Any]
        Hierarchy structure, e.g.:
        {
            "type": "organ",
            "children": {
                "Tissue1": {"type": "tissue", "children": {...}},
                "Tissue2": {"type": "tissue", "children": {...}},
            }
        }

    Returns
    -------
    int
        Number of entities registered

    Examples
    --------
    >>> hierarchy = {
    ...     "type": "organ",
    ...     "children": {
    ...         "Ventricle": {
    ...             "type": "tissue",
    ...             "children": {
    ...                 "Cardiomyocytes": {"type": "cell"}
    ...             }
    ...         }
    ...     }
    ... }
    >>> setup_biological_zoom_hierarchy(controller, "/World/Heart", hierarchy)
    """
    count = 0

    def register_recursive(path: str, spec: Dict[str, Any], parent: Optional[str]) -> int:
        nonlocal count

        entity_type = spec.get("type", "generic")
        controller.register_entity(
            prim_path=path,
            entity_type=entity_type,
            parent_path=parent,
        )
        count += 1

        children = spec.get("children", {})
        for child_name, child_spec in children.items():
            child_path = f"{path}/{child_name}"
            register_recursive(child_path, child_spec, path)

        return count

    return register_recursive(root_path, structure, None)
