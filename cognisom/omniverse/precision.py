"""
Precision Transform Pipeline
============================

Implements the "Local-Float, Global-Double" coordinate architecture for
universe-scale Bio-Digital Twin visualization.

The Challenge:
- Human body: ~2 meters
- Carbon-hydrogen bond: 1.09 Ångströms = 1.09×10⁻¹⁰ meters
- Dynamic range: 11 orders of magnitude
- float32 precision: 7 significant digits → vertex jitter at scale

Solution: Hierarchical Coordinate System
- Level 0 (Atoms): Local coords relative to molecule, float32
- Level 1 (Molecules): Local coords relative to organelle, float32
- Level 2 (Organelles): Global coords relative to cell, double3
- Level 3 (Cells): Global coords relative to tissue, double3
- Level 4 (Organs): World space, double3

All transforms use double3 (float64), while mesh vertex data uses float32.
This eliminates vertex jitter while maintaining GPU performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

log = logging.getLogger(__name__)

# Import USD modules
try:
    from pxr import Usd, UsdGeom, Gf, Sdf, Vt
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False
    log.warning("OpenUSD not available for precision module")


class BiologicalScale(Enum):
    """Biological hierarchy scales with typical size ranges."""
    ATOMIC = "atomic"           # 0.1-10 Å (1e-11 to 1e-9 m)
    MOLECULAR = "molecular"     # 1-100 nm (1e-9 to 1e-7 m)
    ORGANELLE = "organelle"     # 100 nm - 10 μm (1e-7 to 1e-5 m)
    CELLULAR = "cellular"       # 10-100 μm (1e-5 to 1e-4 m)
    TISSUE = "tissue"           # 100 μm - 10 mm (1e-4 to 1e-2 m)
    ORGAN = "organ"             # 1 cm - 1 m (1e-2 to 1 m)
    ORGANISM = "organism"       # 1-2 m


@dataclass
class ScaleMetadata:
    """Metadata for a biological scale level."""
    scale: BiologicalScale
    meters_per_unit: float      # USD metersPerUnit for this scale
    origin_shift: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    precision: str = "double"   # "float" for local mesh, "double" for transforms


# Standard scale configurations
SCALE_CONFIGS: Dict[BiologicalScale, ScaleMetadata] = {
    BiologicalScale.ATOMIC: ScaleMetadata(
        scale=BiologicalScale.ATOMIC,
        meters_per_unit=1e-10,  # 1 unit = 1 Ångström
        precision="float"      # Local coords use float32
    ),
    BiologicalScale.MOLECULAR: ScaleMetadata(
        scale=BiologicalScale.MOLECULAR,
        meters_per_unit=1e-9,   # 1 unit = 1 nanometer
        precision="float"
    ),
    BiologicalScale.ORGANELLE: ScaleMetadata(
        scale=BiologicalScale.ORGANELLE,
        meters_per_unit=1e-6,   # 1 unit = 1 micrometer
        precision="double"
    ),
    BiologicalScale.CELLULAR: ScaleMetadata(
        scale=BiologicalScale.CELLULAR,
        meters_per_unit=1e-6,   # 1 unit = 1 micrometer
        precision="double"
    ),
    BiologicalScale.TISSUE: ScaleMetadata(
        scale=BiologicalScale.TISSUE,
        meters_per_unit=1e-3,   # 1 unit = 1 millimeter
        precision="double"
    ),
    BiologicalScale.ORGAN: ScaleMetadata(
        scale=BiologicalScale.ORGAN,
        meters_per_unit=1e-2,   # 1 unit = 1 centimeter
        precision="double"
    ),
    BiologicalScale.ORGANISM: ScaleMetadata(
        scale=BiologicalScale.ORGANISM,
        meters_per_unit=1.0,    # 1 unit = 1 meter
        precision="double"
    ),
}


class PrecisionTransformManager:
    """
    Manages double-precision transforms for hierarchical biological scenes.

    Key responsibilities:
    1. Set stage-level metersPerUnit for biological scale
    2. Create transform hierarchies with proper precision
    3. Convert between local and global coordinates
    4. Validate precision requirements at each level
    """

    def __init__(self, stage: "Usd.Stage", default_scale: BiologicalScale = BiologicalScale.CELLULAR):
        """
        Initialize the precision manager.

        Args:
            stage: USD stage to manage
            default_scale: Default biological scale for the scene
        """
        if not USD_AVAILABLE:
            raise RuntimeError("OpenUSD required for PrecisionTransformManager")

        self.stage = stage
        self.default_scale = default_scale
        self._scale_metadata = SCALE_CONFIGS[default_scale]

        # Configure stage for precision
        self._configure_stage()

    def _configure_stage(self) -> None:
        """Configure stage metadata for biological scale."""
        # Set metersPerUnit for the default scale
        UsdGeom.SetStageMetersPerUnit(self.stage, self._scale_metadata.meters_per_unit)

        # Set up axis (Y-up is standard for biology visualization)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)

        # Add custom metadata for biological scale
        root = self.stage.GetPseudoRoot()
        root.SetCustomDataByKey("cognisom:biologicalScale", self.default_scale.value)
        root.SetCustomDataByKey("cognisom:metersPerUnit", self._scale_metadata.meters_per_unit)
        root.SetCustomDataByKey("cognisom:precisionMode", "localFloat_globalDouble")

        log.info(f"Stage configured for {self.default_scale.value} scale "
                 f"(metersPerUnit={self._scale_metadata.meters_per_unit})")

    def create_hierarchy_xform(
        self,
        path: str,
        scale: BiologicalScale,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        use_double_precision: bool = True
    ) -> "UsdGeom.Xform":
        """
        Create an Xform prim with proper precision for the biological scale.

        For organelle level and above, uses double-precision transforms.
        For molecular/atomic level, uses single-precision for mesh vertices
        but still uses double for the container transform.

        Args:
            path: USD prim path
            scale: Biological scale level
            position: World position (in stage units)
            use_double_precision: Force double precision (default True for hierarchy)

        Returns:
            UsdGeom.Xform prim
        """
        xform = UsdGeom.Xform.Define(self.stage, path)

        # Add scale metadata
        prim = xform.GetPrim()
        prim.SetCustomDataByKey("cognisom:scale", scale.value)

        # Create translate op with appropriate precision
        if use_double_precision:
            # Double precision for global positioning
            translate_op = xform.AddTranslateOp(
                opSuffix="",
                precision=UsdGeom.XformOp.PrecisionDouble
            )
            translate_op.Set(Gf.Vec3d(*position))
        else:
            # Single precision for local positioning
            translate_op = xform.AddTranslateOp(
                opSuffix="",
                precision=UsdGeom.XformOp.PrecisionFloat
            )
            translate_op.Set(Gf.Vec3f(*position))

        return xform

    def set_world_position(
        self,
        prim: "Usd.Prim",
        position: Tuple[float, float, float],
        time_code: float = None
    ) -> bool:
        """
        Set world position using double-precision transform.

        This is the primary method for positioning entities at global scale.
        Always uses double precision to prevent vertex jitter.

        Args:
            prim: USD prim to position
            position: World position (x, y, z) in stage units
            time_code: USD time code for animation

        Returns:
            True if successful
        """
        if time_code is None:
            time_code = Usd.TimeCode.Default()
        try:
            xformable = UsdGeom.Xformable(prim)

            # Check if translate op exists
            xform_ops = xformable.GetOrderedXformOps()
            translate_op = None

            for op in xform_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break

            if translate_op is None:
                # Create new translate op with double precision
                translate_op = xformable.AddTranslateOp(
                    precision=UsdGeom.XformOp.PrecisionDouble
                )

            # Set position with double precision
            translate_op.Set(Gf.Vec3d(*position), time_code)
            return True

        except Exception as e:
            log.error(f"Failed to set world position: {e}")
            return False

    def set_local_position(
        self,
        prim: "Usd.Prim",
        position: Tuple[float, float, float],
        time_code: float = None
    ) -> bool:
        """
        Set local position using single-precision transform.

        Use this for positioning entities within their local coordinate frame
        (e.g., atoms within a molecule, organelles within a cell).

        Args:
            prim: USD prim to position
            position: Local position (x, y, z)
            time_code: USD time code for animation

        Returns:
            True if successful
        """
        if time_code is None:
            time_code = Usd.TimeCode.Default()
        try:
            xformable = UsdGeom.Xformable(prim)
            xform_ops = xformable.GetOrderedXformOps()

            translate_op = None
            for op in xform_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break

            if translate_op is None:
                # Create with single precision for local coords
                translate_op = xformable.AddTranslateOp(
                    precision=UsdGeom.XformOp.PrecisionFloat
                )

            # Use Vec3f for local coordinates
            translate_op.Set(Gf.Vec3f(*position), time_code)
            return True

        except Exception as e:
            log.error(f"Failed to set local position: {e}")
            return False

    def get_world_position(self, prim: "Usd.Prim", time_code: float = None) -> Tuple[float, float, float]:
        """
        Get world position from a prim's transform.

        Returns full double-precision coordinates.

        Args:
            prim: USD prim
            time_code: USD time code

        Returns:
            World position (x, y, z) as float64
        """
        if time_code is None:
            time_code = Usd.TimeCode.Default()
        try:
            xformable = UsdGeom.Xformable(prim)
            world_transform = xformable.ComputeLocalToWorldTransform(time_code)
            translation = world_transform.ExtractTranslation()
            return (translation[0], translation[1], translation[2])
        except Exception as e:
            log.error(f"Failed to get world position: {e}")
            return (0.0, 0.0, 0.0)

    def convert_to_scale(
        self,
        position: Tuple[float, float, float],
        from_scale: BiologicalScale,
        to_scale: BiologicalScale
    ) -> Tuple[float, float, float]:
        """
        Convert position between biological scales.

        Args:
            position: Position in from_scale units
            from_scale: Source scale
            to_scale: Target scale

        Returns:
            Position in to_scale units
        """
        from_meters = SCALE_CONFIGS[from_scale].meters_per_unit
        to_meters = SCALE_CONFIGS[to_scale].meters_per_unit

        scale_factor = from_meters / to_meters

        return (
            position[0] * scale_factor,
            position[1] * scale_factor,
            position[2] * scale_factor
        )

    def create_biological_hierarchy(
        self,
        root_path: str = "/World",
        include_scales: Optional[List[BiologicalScale]] = None
    ) -> Dict[BiologicalScale, str]:
        """
        Create a standard biological hierarchy with proper precision at each level.

        Creates nested Xforms for each biological scale level, each with
        double-precision transforms ready for positioning.

        Args:
            root_path: Root path for the hierarchy
            include_scales: Scales to include (default: all)

        Returns:
            Dict mapping scale to USD path
        """
        if include_scales is None:
            include_scales = [
                BiologicalScale.ORGANISM,
                BiologicalScale.ORGAN,
                BiologicalScale.TISSUE,
                BiologicalScale.CELLULAR,
                BiologicalScale.ORGANELLE,
            ]

        paths = {}
        current_path = root_path

        for scale in include_scales:
            scale_path = f"{current_path}/{scale.value.capitalize()}"
            self.create_hierarchy_xform(
                scale_path,
                scale,
                position=(0.0, 0.0, 0.0),
                use_double_precision=True
            )
            paths[scale] = scale_path
            current_path = scale_path

        return paths

    def validate_precision(self, prim: "Usd.Prim") -> Dict[str, Any]:
        """
        Validate that a prim has appropriate precision for its scale.

        Returns diagnostic info about precision issues.

        Args:
            prim: USD prim to validate

        Returns:
            Validation result dict
        """
        result = {
            "prim_path": str(prim.GetPath()),
            "valid": True,
            "issues": [],
            "recommendations": []
        }

        try:
            xformable = UsdGeom.Xformable(prim)

            # Check transform precision
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    precision = op.GetPrecision()

                    # Get scale from metadata
                    scale_str = prim.GetCustomDataByKey("cognisom:scale")
                    if scale_str:
                        try:
                            scale = BiologicalScale(scale_str)
                            expected_precision = SCALE_CONFIGS[scale].precision

                            # Check if precision matches expectation
                            is_double = (precision == UsdGeom.XformOp.PrecisionDouble)
                            expects_double = (expected_precision == "double")

                            if expects_double and not is_double:
                                result["valid"] = False
                                result["issues"].append(
                                    f"Scale '{scale.value}' expects double precision but has float"
                                )
                                result["recommendations"].append(
                                    "Use set_world_position() for global coordinates"
                                )
                        except ValueError:
                            pass

            # Check for large coordinates that need double precision
            world_pos = self.get_world_position(prim)
            max_coord = max(abs(c) for c in world_pos)

            # If coordinates exceed float32 precision threshold (~10^6 for sub-unit accuracy)
            if max_coord > 1e6:
                xform_ops = xformable.GetOrderedXformOps()
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        if op.GetPrecision() != UsdGeom.XformOp.PrecisionDouble:
                            result["valid"] = False
                            result["issues"].append(
                                f"Large coordinates ({max_coord:.2e}) require double precision"
                            )
                            result["recommendations"].append(
                                "Upgrade to double-precision transforms to prevent jitter"
                            )

        except Exception as e:
            result["valid"] = False
            result["issues"].append(f"Validation error: {e}")

        return result


class BioPrecisePointAPI:
    """
    API schema for storing high-precision (float64) coordinates alongside
    renderable float32 mesh data.

    The visualization layer reads float32 `points` for rendering.
    The simulation layer reads float64 `precise:positions` for physics.

    Usage:
        api = BioPrecisePointAPI(prim)
        api.set_precise_positions(positions_float64)
        api.set_precise_velocities(velocities_float64)
    """

    POSITIONS_ATTR = "precise:positions"
    VELOCITIES_ATTR = "precise:velocities"
    MASSES_ATTR = "precise:masses"
    FORCES_ATTR = "precise:forces"

    def __init__(self, prim: "Usd.Prim"):
        """
        Initialize the API schema.

        Args:
            prim: USD prim to apply the schema to
        """
        if not USD_AVAILABLE:
            raise RuntimeError("OpenUSD required for BioPrecisePointAPI")

        self.prim = prim
        self._apply_schema()

    def _apply_schema(self) -> None:
        """Mark prim as having the BioPrecisePointAPI applied."""
        schemas = self.prim.GetAppliedSchemas()
        if "BioPrecisePointAPI" not in schemas:
            self.prim.AddAppliedSchema("BioPrecisePointAPI")

    def set_precise_positions(self, positions: np.ndarray, time_code: float = None) -> None:
        """
        Set double-precision positions for simulation.

        Args:
            positions: Nx3 array of float64 positions
            time_code: USD time code
        """
        if time_code is None:
            time_code = Usd.TimeCode.Default()
        if positions.dtype != np.float64:
            positions = positions.astype(np.float64)

        attr = self.prim.GetAttribute(self.POSITIONS_ATTR)
        if not attr:
            attr = self.prim.CreateAttribute(
                self.POSITIONS_ATTR,
                Sdf.ValueTypeNames.Double3Array
            )

        # Convert to Vt array
        vt_positions = Vt.Vec3dArray.FromNumpy(positions.reshape(-1, 3))
        attr.Set(vt_positions, time_code)

    def get_precise_positions(self, time_code: float = None) -> np.ndarray:
        """
        Get double-precision positions.

        Args:
            time_code: USD time code

        Returns:
            Nx3 array of float64 positions
        """
        if time_code is None:
            time_code = Usd.TimeCode.Default()
        attr = self.prim.GetAttribute(self.POSITIONS_ATTR)
        if not attr or not attr.HasValue():
            return np.array([], dtype=np.float64).reshape(0, 3)

        vt_positions = attr.Get(time_code)
        return np.array(vt_positions, dtype=np.float64)

    def set_precise_velocities(self, velocities: np.ndarray, time_code: float = None) -> None:
        """
        Set double-precision velocities.

        Args:
            velocities: Nx3 array of float64 velocities
            time_code: USD time code
        """
        if time_code is None:
            time_code = Usd.TimeCode.Default()
        if velocities.dtype != np.float64:
            velocities = velocities.astype(np.float64)

        attr = self.prim.GetAttribute(self.VELOCITIES_ATTR)
        if not attr:
            attr = self.prim.CreateAttribute(
                self.VELOCITIES_ATTR,
                Sdf.ValueTypeNames.Double3Array
            )

        vt_velocities = Vt.Vec3dArray.FromNumpy(velocities.reshape(-1, 3))
        attr.Set(vt_velocities, time_code)

    def get_precise_velocities(self, time_code: float = None) -> np.ndarray:
        """
        Get double-precision velocities.

        Args:
            time_code: USD time code

        Returns:
            Nx3 array of float64 velocities
        """
        if time_code is None:
            time_code = Usd.TimeCode.Default()
        attr = self.prim.GetAttribute(self.VELOCITIES_ATTR)
        if not attr or not attr.HasValue():
            return np.array([], dtype=np.float64).reshape(0, 3)

        vt_velocities = attr.Get(time_code)
        return np.array(vt_velocities, dtype=np.float64)

    def set_precise_masses(self, masses: np.ndarray, time_code: float = None) -> None:
        """
        Set double-precision masses.

        Args:
            masses: N array of float64 masses
            time_code: USD time code
        """
        if time_code is None:
            time_code = Usd.TimeCode.Default()
        if masses.dtype != np.float64:
            masses = masses.astype(np.float64)

        attr = self.prim.GetAttribute(self.MASSES_ATTR)
        if not attr:
            attr = self.prim.CreateAttribute(
                self.MASSES_ATTR,
                Sdf.ValueTypeNames.DoubleArray
            )

        vt_masses = Vt.DoubleArray.FromNumpy(masses.flatten())
        attr.Set(vt_masses, time_code)

    def get_precise_masses(self, time_code: float = None) -> np.ndarray:
        """
        Get double-precision masses.

        Args:
            time_code: USD time code

        Returns:
            N array of float64 masses
        """
        if time_code is None:
            time_code = Usd.TimeCode.Default()
        attr = self.prim.GetAttribute(self.MASSES_ATTR)
        if not attr or not attr.HasValue():
            return np.array([], dtype=np.float64)

        vt_masses = attr.Get(time_code)
        return np.array(vt_masses, dtype=np.float64)


def centroid_and_center(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute centroid and center positions to local coordinates.

    This is critical for the Local-Float/Global-Double architecture:
    - Store original centroid for global positioning (double3 transform)
    - Use centered coords for local mesh (float32 points)

    Args:
        positions: Nx3 array of positions

    Returns:
        Tuple of (centroid, centered_positions)
    """
    centroid = positions.mean(axis=0).astype(np.float64)
    centered = (positions - centroid).astype(np.float32)
    return centroid, centered


# ── Convenience Functions ────────────────────────────────────────────────

def configure_stage_for_biology(
    stage: "Usd.Stage",
    scale: BiologicalScale = BiologicalScale.CELLULAR
) -> PrecisionTransformManager:
    """
    Configure a USD stage for biological visualization with proper precision.

    Args:
        stage: USD stage to configure
        scale: Primary biological scale for the scene

    Returns:
        PrecisionTransformManager for the stage
    """
    return PrecisionTransformManager(stage, default_scale=scale)


def verify_no_jitter(
    position: Tuple[float, float, float],
    required_precision_meters: float = 1e-9
) -> bool:
    """
    Verify that float32 would have sufficient precision at this position.

    Args:
        position: World position
        required_precision_meters: Required precision in meters

    Returns:
        True if float32 is sufficient, False if double needed
    """
    max_coord = max(abs(c) for c in position)

    # float32 has ~7 significant digits
    # At max_coord, the precision is roughly max_coord * 1e-7
    float32_precision = max_coord * 1e-7

    return float32_precision <= required_precision_meters
