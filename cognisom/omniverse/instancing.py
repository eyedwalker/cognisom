"""
Nested Instancing Architecture (Phase B2)
==========================================

Implements hierarchical USD instancing for massive biological assemblies.

The Challenge:
- Human liver: 100 billion hepatocytes
- Each hepatocyte: thousands of mitochondria
- Each mitochondrion: thousands of ATP synthase
- Unique geometry: impossible (exceeds global compute capacity)

Solution: Hierarchical Instance Graph
-------------------------------------
ATP Synthase (authored once)
    ↑ instanceable=true
Mitochondrion (references ATP Synthase N times → 1 prototype)
    ↑ instanceable=true
Hepatocyte (references Mitochondrion M times → 1 prototype)
    ↑ instanceable=true
Liver Lobule (references Hepatocyte K times → 1 prototype)

Memory Cost: O(1) for geometry, O(N) for transform matrices only.

Key Concepts:
1. Scenegraph Instancing (instanceable=true): For complex assemblies (organelles)
2. Point Instancing (UsdGeomPointInstancer): For particles (water, lipids, RBCs)
3. Primvars for per-instance variation (not scenegraph overrides)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

log = logging.getLogger(__name__)

# Import USD modules
try:
    from pxr import Usd, UsdGeom, Gf, Sdf, Vt, UsdShade
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False
    log.warning("OpenUSD not available for instancing module")


class InstancingStrategy(str, Enum):
    """Instancing strategies based on entity count and complexity."""
    SCENEGRAPH = "scenegraph"      # instanceable=true, for complex assemblies
    POINT_INSTANCER = "point"      # UsdGeomPointInstancer, for particles
    HYBRID = "hybrid"              # Scenegraph for structure, point for details


@dataclass
class PrototypeDefinition:
    """Definition for a reusable prototype."""
    name: str
    prim_path: str
    category: str = "generic"          # organelle, molecule, particle, etc.
    suggested_strategy: InstancingStrategy = InstancingStrategy.SCENEGRAPH
    max_instances: int = 1000000       # Performance hint
    lod_variants: List[str] = field(default_factory=list)  # LOD variant names
    semantic_variants: List[str] = field(default_factory=list)  # Representation variants
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InstanceData:
    """Data for a single instance."""
    prototype_name: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # Quaternion
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    primvar_data: Dict[str, Any] = field(default_factory=dict)  # Per-instance variation


class NestedInstancingManager:
    """
    Manages hierarchical USD instancing for biological assemblies.

    Key Features:
    1. Prototype registration and caching
    2. Scenegraph instancing for complex structures
    3. Point instancing for particle systems
    4. Per-instance primvars for state coloring (preserves instance sharing)
    5. Nested hierarchy support (instances containing instances)
    """

    # Standard paths
    PROTOTYPES_PATH = "/Prototypes"
    INSTANCES_PATH = "/World/Instances"

    # Thresholds for instancing strategy
    POINT_INSTANCER_THRESHOLD = 10000  # Use point instancer above this count

    def __init__(self, stage: "Usd.Stage"):
        """
        Initialize the instancing manager.

        Args:
            stage: USD stage to manage
        """
        if not USD_AVAILABLE:
            raise RuntimeError("OpenUSD required for NestedInstancingManager")

        self.stage = stage
        self._prototypes: Dict[str, PrototypeDefinition] = {}
        self._prototype_prims: Dict[str, "Usd.Prim"] = {}
        self._instance_counts: Dict[str, int] = {}
        self._point_instancers: Dict[str, "UsdGeom.PointInstancer"] = {}

        # Initialize prototype layer
        self._initialize_prototype_layer()

    def _initialize_prototype_layer(self) -> None:
        """Create the prototype container layer."""
        # Create prototypes container (invisible, not rendered directly)
        proto_prim = self.stage.DefinePrim(self.PROTOTYPES_PATH, "Scope")
        proto_prim.SetMetadata("kind", "group")

        # Mark as not directly visible
        imageable = UsdGeom.Imageable(proto_prim)
        if imageable:
            imageable.MakeInvisible()

        # Create instances container
        self.stage.DefinePrim(self.INSTANCES_PATH, "Scope")

        log.info("Initialized prototype and instance containers")

    def register_prototype(
        self,
        name: str,
        geometry_creator: Optional[callable] = None,
        from_usd_path: Optional[str] = None,
        category: str = "generic",
        suggested_strategy: InstancingStrategy = InstancingStrategy.SCENEGRAPH,
        make_instanceable: bool = True
    ) -> PrototypeDefinition:
        """
        Register a reusable prototype.

        Prototypes can be created either by:
        1. A geometry_creator function that builds the prim
        2. Referencing an existing USD file

        Args:
            name: Unique prototype name
            geometry_creator: Function(stage, path) -> creates geometry
            from_usd_path: Path to external USD file to reference
            category: Prototype category (organelle, molecule, etc.)
            suggested_strategy: Recommended instancing strategy
            make_instanceable: Mark the prototype as instanceable

        Returns:
            PrototypeDefinition for the registered prototype
        """
        if name in self._prototypes:
            log.warning(f"Prototype '{name}' already registered, returning existing")
            return self._prototypes[name]

        proto_path = f"{self.PROTOTYPES_PATH}/{name}"

        if from_usd_path:
            # Reference external USD
            prim = self.stage.DefinePrim(proto_path, "Xform")
            prim.GetReferences().AddReference(from_usd_path)
        elif geometry_creator:
            # Create geometry via callback
            geometry_creator(self.stage, proto_path)
            prim = self.stage.GetPrimAtPath(proto_path)
        else:
            # Create empty xform placeholder
            prim = self.stage.DefinePrim(proto_path, "Xform")

        if not prim or not prim.IsValid():
            raise RuntimeError(f"Failed to create prototype at {proto_path}")

        # Mark as instanceable for scenegraph instancing
        if make_instanceable:
            prim.SetInstanceable(True)

        # Store definition
        definition = PrototypeDefinition(
            name=name,
            prim_path=proto_path,
            category=category,
            suggested_strategy=suggested_strategy
        )
        self._prototypes[name] = definition
        self._prototype_prims[name] = prim
        self._instance_counts[name] = 0

        log.info(f"Registered prototype: {name} at {proto_path}")
        return definition

    def create_instance(
        self,
        prototype_name: str,
        instance_name: str,
        position: Tuple[float, float, float],
        rotation: Optional[Tuple[float, float, float, float]] = None,
        scale: Optional[Tuple[float, float, float]] = None,
        parent_path: str = "/World/Instances"
    ) -> Optional["Usd.Prim"]:
        """
        Create a single scenegraph instance of a prototype.

        Uses USD's native instancing (references with instanceable=true)
        for memory-efficient repeated structures.

        Args:
            prototype_name: Name of registered prototype
            instance_name: Unique name for this instance
            position: World position (uses double precision)
            rotation: Quaternion rotation (w, x, y, z)
            scale: Scale factors

        Returns:
            The instance prim, or None on failure
        """
        if prototype_name not in self._prototypes:
            log.error(f"Prototype '{prototype_name}' not registered")
            return None

        definition = self._prototypes[prototype_name]
        proto_prim = self._prototype_prims[prototype_name]

        instance_path = f"{parent_path}/{instance_name}"

        # Create instance prim that references the prototype
        instance_prim = self.stage.DefinePrim(instance_path, "Xform")

        # Add internal reference to prototype
        instance_prim.GetReferences().AddInternalReference(proto_prim.GetPath())

        # Set transform with double precision
        xform = UsdGeom.Xformable(instance_prim)

        # Translate (double precision for global coords)
        translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(*position))

        # Rotate (if provided)
        if rotation:
            rotate_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat)
            rotate_op.Set(Gf.Quatf(*rotation))

        # Scale (if provided)
        if scale:
            scale_op = xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat)
            scale_op.Set(Gf.Vec3f(*scale))

        self._instance_counts[prototype_name] += 1

        return instance_prim

    def create_instances_batch(
        self,
        prototype_name: str,
        instances: List[InstanceData],
        parent_path: str = "/World/Instances"
    ) -> int:
        """
        Create multiple instances efficiently.

        Automatically selects strategy based on count:
        - < POINT_INSTANCER_THRESHOLD: Individual scenegraph instances
        - >= POINT_INSTANCER_THRESHOLD: UsdGeomPointInstancer

        Args:
            prototype_name: Name of registered prototype
            instances: List of instance data
            parent_path: Parent path for instances

        Returns:
            Number of instances created
        """
        if prototype_name not in self._prototypes:
            log.error(f"Prototype '{prototype_name}' not registered")
            return 0

        count = len(instances)
        definition = self._prototypes[prototype_name]

        # Choose strategy based on count
        if count >= self.POINT_INSTANCER_THRESHOLD:
            return self._create_point_instances(prototype_name, instances, parent_path)
        else:
            return self._create_scenegraph_instances(prototype_name, instances, parent_path)

    def _create_scenegraph_instances(
        self,
        prototype_name: str,
        instances: List[InstanceData],
        parent_path: str
    ) -> int:
        """Create instances using scenegraph instancing."""
        created = 0

        for i, inst in enumerate(instances):
            instance_name = f"{prototype_name}_{i:06d}"
            prim = self.create_instance(
                prototype_name,
                instance_name,
                inst.position,
                inst.rotation,
                inst.scale,
                parent_path
            )
            if prim:
                # Apply per-instance primvars
                if inst.primvar_data:
                    self._apply_primvars(prim, inst.primvar_data)
                created += 1

        log.info(f"Created {created} scenegraph instances of '{prototype_name}'")
        return created

    def _create_point_instances(
        self,
        prototype_name: str,
        instances: List[InstanceData],
        parent_path: str
    ) -> int:
        """Create instances using UsdGeomPointInstancer for high counts."""
        instancer_path = f"{parent_path}/{prototype_name}_Instancer"

        # Get or create point instancer
        if prototype_name in self._point_instancers:
            instancer = self._point_instancers[prototype_name]
        else:
            instancer = UsdGeom.PointInstancer.Define(self.stage, instancer_path)
            self._point_instancers[prototype_name] = instancer

            # Set up prototype relationship
            proto_path = self._prototypes[prototype_name].prim_path
            instancer.CreatePrototypesRel().SetTargets([Sdf.Path(proto_path)])

        # Build arrays for all instances
        n = len(instances)
        positions = Vt.Vec3fArray(n)
        orientations = Vt.QuathArray(n)
        scales = Vt.Vec3fArray(n)
        proto_indices = Vt.IntArray(n)

        for i, inst in enumerate(instances):
            positions[i] = Gf.Vec3f(*inst.position)
            orientations[i] = Gf.Quath(
                inst.rotation[0],  # w
                inst.rotation[1],  # x
                inst.rotation[2],  # y
                inst.rotation[3]   # z
            )
            scales[i] = Gf.Vec3f(*inst.scale)
            proto_indices[i] = 0  # All use prototype 0

        # Set instance attributes
        instancer.CreatePositionsAttr().Set(positions)
        instancer.CreateOrientationsAttr().Set(orientations)
        instancer.CreateScalesAttr().Set(scales)
        instancer.CreateProtoIndicesAttr().Set(proto_indices)

        self._instance_counts[prototype_name] += n
        log.info(f"Created {n} point instances of '{prototype_name}'")

        return n

    def _apply_primvars(self, prim: "Usd.Prim", primvar_data: Dict[str, Any]) -> None:
        """
        Apply per-instance primvars for variation.

        CRITICAL: Use primvars for state coloring to preserve instance sharing.
        Overriding materials directly would break instancing.
        """
        primvars_api = UsdGeom.PrimvarsAPI(prim)

        for name, value in primvar_data.items():
            if isinstance(value, (list, tuple)) and len(value) == 3:
                # Color value
                primvar = primvars_api.CreatePrimvar(
                    f"primvars:{name}",
                    Sdf.ValueTypeNames.Color3f,
                    UsdGeom.Tokens.constant
                )
                primvar.Set(Gf.Vec3f(*value))
            elif isinstance(value, float):
                primvar = primvars_api.CreatePrimvar(
                    f"primvars:{name}",
                    Sdf.ValueTypeNames.Float,
                    UsdGeom.Tokens.constant
                )
                primvar.Set(value)
            elif isinstance(value, int):
                primvar = primvars_api.CreatePrimvar(
                    f"primvars:{name}",
                    Sdf.ValueTypeNames.Int,
                    UsdGeom.Tokens.constant
                )
                primvar.Set(value)

    def create_nested_assembly(
        self,
        assembly_name: str,
        component_definitions: List[Dict],
        assembly_path: Optional[str] = None
    ) -> Optional["Usd.Prim"]:
        """
        Create a nested assembly (instances containing instances).

        Example: Mitochondrion containing ATP synthase instances

        Args:
            assembly_name: Name for the assembly
            component_definitions: List of {prototype, count, positions, ...}
            assembly_path: Path for the assembly (default: prototypes)

        Returns:
            Assembly prim
        """
        if assembly_path is None:
            assembly_path = f"{self.PROTOTYPES_PATH}/{assembly_name}"

        # Create the assembly container
        assembly_prim = self.stage.DefinePrim(assembly_path, "Xform")

        for comp in component_definitions:
            proto_name = comp["prototype"]
            positions = comp.get("positions", [])

            # Create instances within the assembly
            for i, pos in enumerate(positions):
                instance_data = InstanceData(
                    prototype_name=proto_name,
                    position=pos,
                    rotation=comp.get("rotation", (1.0, 0.0, 0.0, 0.0)),
                    scale=comp.get("scale", (1.0, 1.0, 1.0))
                )

                instance_name = f"{proto_name}_{i:04d}"
                self.create_instance(
                    proto_name,
                    instance_name,
                    pos,
                    parent_path=assembly_path
                )

        # Mark the assembly itself as instanceable
        assembly_prim.SetInstanceable(True)

        # Register as a prototype
        self._prototypes[assembly_name] = PrototypeDefinition(
            name=assembly_name,
            prim_path=assembly_path,
            category="assembly",
            suggested_strategy=InstancingStrategy.SCENEGRAPH
        )
        self._prototype_prims[assembly_name] = assembly_prim
        self._instance_counts[assembly_name] = 0

        log.info(f"Created nested assembly: {assembly_name}")
        return assembly_prim

    def update_point_instancer_positions(
        self,
        prototype_name: str,
        positions: np.ndarray,
        time_code: float = Usd.TimeCode.Default()
    ) -> bool:
        """
        Update positions for a point instancer (for animation).

        Args:
            prototype_name: Name of the prototype
            positions: Nx3 array of new positions
            time_code: USD time code for animation

        Returns:
            True if successful
        """
        if prototype_name not in self._point_instancers:
            log.error(f"No point instancer for '{prototype_name}'")
            return False

        instancer = self._point_instancers[prototype_name]

        # Convert to Vt array
        vt_positions = Vt.Vec3fArray.FromNumpy(positions.astype(np.float32))
        instancer.GetPositionsAttr().Set(vt_positions, time_code)

        return True

    def get_prototype_info(self, name: str) -> Optional[Dict]:
        """Get information about a registered prototype."""
        if name not in self._prototypes:
            return None

        definition = self._prototypes[name]
        return {
            "name": definition.name,
            "path": definition.prim_path,
            "category": definition.category,
            "strategy": definition.suggested_strategy.value,
            "instance_count": self._instance_counts.get(name, 0),
            "has_point_instancer": name in self._point_instancers
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get instancing statistics."""
        total_instances = sum(self._instance_counts.values())
        total_prototypes = len(self._prototypes)

        # Estimate memory savings
        # Each instance only needs transform (12 floats = 48 bytes)
        # vs full geometry (could be 100KB+ each)
        estimated_memory_saved_mb = total_instances * 0.1  # Conservative estimate

        return {
            "total_prototypes": total_prototypes,
            "total_instances": total_instances,
            "point_instancers": len(self._point_instancers),
            "instance_counts": dict(self._instance_counts),
            "estimated_memory_saved_mb": estimated_memory_saved_mb,
            "prototypes": list(self._prototypes.keys())
        }


# ── Helper Functions ─────────────────────────────────────────────────────

def create_sphere_prototype(stage: "Usd.Stage", path: str, radius: float = 1.0) -> None:
    """Create a simple sphere prototype."""
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.GetRadiusAttr().Set(radius)


def create_capsule_prototype(
    stage: "Usd.Stage",
    path: str,
    radius: float = 0.5,
    height: float = 2.0
) -> None:
    """Create a capsule prototype (for rod-shaped bacteria, etc.)."""
    capsule = UsdGeom.Capsule.Define(stage, path)
    capsule.GetRadiusAttr().Set(radius)
    capsule.GetHeightAttr().Set(height)


def create_ellipsoid_prototype(
    stage: "Usd.Stage",
    path: str,
    radii: Tuple[float, float, float] = (1.0, 0.5, 0.5)
) -> None:
    """Create an ellipsoid prototype (stretched sphere)."""
    # Use a sphere with non-uniform scale
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.GetRadiusAttr().Set(1.0)

    xform = UsdGeom.Xformable(sphere)
    scale_op = xform.AddScaleOp()
    scale_op.Set(Gf.Vec3f(*radii))


def estimate_instancing_strategy(
    entity_count: int,
    geometry_complexity: int = 100,  # Estimated vertex count
    needs_individual_updates: bool = False
) -> InstancingStrategy:
    """
    Recommend an instancing strategy based on requirements.

    Args:
        entity_count: Number of instances needed
        geometry_complexity: Approximate vertex count per entity
        needs_individual_updates: Whether instances need individual transforms

    Returns:
        Recommended InstancingStrategy
    """
    if entity_count > 1000000:
        return InstancingStrategy.POINT_INSTANCER

    if entity_count > 10000 and not needs_individual_updates:
        return InstancingStrategy.POINT_INSTANCER

    if geometry_complexity > 10000:
        return InstancingStrategy.SCENEGRAPH

    return InstancingStrategy.SCENEGRAPH
