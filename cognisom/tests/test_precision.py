"""
Tests for the Double-Precision Pipeline (Phase B1)
===================================================

Tests the Local-Float/Global-Double coordinate architecture for
universe-scale Bio-Digital Twin visualization.
"""

import pytest
import numpy as np

# Skip all tests if USD is not available
pytest.importorskip("pxr")

from cognisom.omniverse.precision import (
    BiologicalScale,
    ScaleMetadata,
    SCALE_CONFIGS,
    PrecisionTransformManager,
    BioPrecisePointAPI,
    centroid_and_center,
    verify_no_jitter,
    configure_stage_for_biology,
)

from pxr import Usd, UsdGeom, Gf


class TestBiologicalScale:
    """Test BiologicalScale enum and configurations."""

    def test_scale_enum_values(self):
        """Verify all expected scales exist."""
        assert BiologicalScale.ATOMIC.value == "atomic"
        assert BiologicalScale.MOLECULAR.value == "molecular"
        assert BiologicalScale.ORGANELLE.value == "organelle"
        assert BiologicalScale.CELLULAR.value == "cellular"
        assert BiologicalScale.TISSUE.value == "tissue"
        assert BiologicalScale.ORGAN.value == "organ"
        assert BiologicalScale.ORGANISM.value == "organism"

    def test_scale_configs_exist(self):
        """Verify all scales have configurations."""
        for scale in BiologicalScale:
            assert scale in SCALE_CONFIGS
            config = SCALE_CONFIGS[scale]
            assert isinstance(config, ScaleMetadata)
            assert config.meters_per_unit > 0

    def test_scale_hierarchy(self):
        """Verify scales are ordered from smallest to largest."""
        meters = [SCALE_CONFIGS[s].meters_per_unit for s in [
            BiologicalScale.ATOMIC,
            BiologicalScale.MOLECULAR,
            BiologicalScale.ORGANELLE,
            BiologicalScale.CELLULAR,
            BiologicalScale.TISSUE,
            BiologicalScale.ORGAN,
            BiologicalScale.ORGANISM,
        ]]
        # Each should be larger than the previous
        for i in range(1, len(meters)):
            assert meters[i] >= meters[i-1], f"Scale order violated at index {i}"


class TestPrecisionTransformManager:
    """Test PrecisionTransformManager for double-precision transforms."""

    @pytest.fixture
    def stage(self, tmp_path):
        """Create a temporary USD stage."""
        stage_path = tmp_path / "test_precision.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        yield stage

    def test_stage_configuration(self, stage):
        """Test that stage is properly configured for biology."""
        manager = PrecisionTransformManager(stage, BiologicalScale.CELLULAR)

        # Check metersPerUnit is set
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        assert meters_per_unit == SCALE_CONFIGS[BiologicalScale.CELLULAR].meters_per_unit

        # Check custom metadata
        root = stage.GetPseudoRoot()
        assert root.GetCustomDataByKey("cognisom:biologicalScale") == "cellular"
        assert root.GetCustomDataByKey("cognisom:precisionMode") == "localFloat_globalDouble"

    def test_create_hierarchy_xform_double_precision(self, stage):
        """Test that hierarchy xforms use double precision."""
        manager = PrecisionTransformManager(stage, BiologicalScale.CELLULAR)

        xform = manager.create_hierarchy_xform(
            "/World/TestCell",
            BiologicalScale.CELLULAR,
            position=(1000000.0, 2000000.0, 3000000.0),
            use_double_precision=True
        )

        # Verify transform op precision
        xformable = UsdGeom.Xformable(xform.GetPrim())
        ops = xformable.GetOrderedXformOps()
        assert len(ops) > 0

        translate_op = ops[0]
        assert translate_op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        assert translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble

    def test_set_world_position(self, stage):
        """Test setting world position with double precision."""
        manager = PrecisionTransformManager(stage, BiologicalScale.CELLULAR)

        # Create a prim
        xform = UsdGeom.Xform.Define(stage, "/World/TestPrim")

        # Set a large position that would lose precision in float32
        large_pos = (1e8, 2e8, 3e8)
        manager.set_world_position(xform.GetPrim(), large_pos)

        # Retrieve and verify
        retrieved = manager.get_world_position(xform.GetPrim())
        assert abs(retrieved[0] - large_pos[0]) < 1.0
        assert abs(retrieved[1] - large_pos[1]) < 1.0
        assert abs(retrieved[2] - large_pos[2]) < 1.0

    def test_set_local_position(self, stage):
        """Test setting local position with single precision."""
        manager = PrecisionTransformManager(stage, BiologicalScale.CELLULAR)

        xform = UsdGeom.Xform.Define(stage, "/World/LocalPrim")

        # Small local position
        local_pos = (1.5, 2.5, 3.5)
        manager.set_local_position(xform.GetPrim(), local_pos)

        # Verify the op exists (local uses float precision)
        xformable = UsdGeom.Xformable(xform.GetPrim())
        ops = xformable.GetOrderedXformOps()
        assert len(ops) > 0

    def test_convert_between_scales(self, stage):
        """Test coordinate conversion between scales."""
        manager = PrecisionTransformManager(stage, BiologicalScale.CELLULAR)

        # 1 micrometer in cellular scale (meters_per_unit = 1e-6)
        pos_cellular = (1.0, 1.0, 1.0)

        # Convert to nanometer scale (meters_per_unit = 1e-9)
        pos_molecular = manager.convert_to_scale(
            pos_cellular,
            BiologicalScale.CELLULAR,
            BiologicalScale.MOLECULAR
        )

        # 1 μm = 1000 nm
        assert abs(pos_molecular[0] - 1000.0) < 0.001

    def test_create_biological_hierarchy(self, stage):
        """Test creation of biological hierarchy with proper paths."""
        manager = PrecisionTransformManager(stage, BiologicalScale.CELLULAR)

        paths = manager.create_biological_hierarchy("/World")

        # Verify hierarchy was created
        assert BiologicalScale.ORGANISM in paths
        assert BiologicalScale.CELLULAR in paths

        # Verify prims exist
        for scale, path in paths.items():
            prim = stage.GetPrimAtPath(path)
            assert prim.IsValid()


class TestBioPrecisePointAPI:
    """Test BioPrecisePointAPI for storing float64 simulation data."""

    @pytest.fixture
    def stage_with_prim(self, tmp_path):
        """Create a stage with a test prim."""
        stage_path = tmp_path / "test_bio_api.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        prim = stage.DefinePrim("/World/TestPoints", "Xform")
        return stage, prim

    def test_set_and_get_precise_positions(self, stage_with_prim):
        """Test storing and retrieving float64 positions."""
        stage, prim = stage_with_prim
        api = BioPrecisePointAPI(prim)

        # Create test positions with high precision values
        positions = np.array([
            [1.123456789012345, 2.234567890123456, 3.345678901234567],
            [4.456789012345678, 5.567890123456789, 6.678901234567890],
        ], dtype=np.float64)

        api.set_precise_positions(positions)
        retrieved = api.get_precise_positions()

        assert retrieved.dtype == np.float64
        assert retrieved.shape == positions.shape
        np.testing.assert_array_almost_equal(retrieved, positions, decimal=10)

    def test_set_and_get_precise_velocities(self, stage_with_prim):
        """Test storing and retrieving float64 velocities."""
        stage, prim = stage_with_prim
        api = BioPrecisePointAPI(prim)

        velocities = np.array([
            [0.001, 0.002, 0.003],
            [-0.001, -0.002, -0.003],
        ], dtype=np.float64)

        api.set_precise_velocities(velocities)
        retrieved = api.get_precise_velocities()

        assert retrieved.dtype == np.float64
        np.testing.assert_array_almost_equal(retrieved, velocities, decimal=10)

    def test_set_and_get_precise_masses(self, stage_with_prim):
        """Test storing and retrieving float64 masses."""
        stage, prim = stage_with_prim
        api = BioPrecisePointAPI(prim)

        masses = np.array([1.008, 12.011, 14.007], dtype=np.float64)

        api.set_precise_masses(masses)
        retrieved = api.get_precise_masses()

        assert retrieved.dtype == np.float64
        np.testing.assert_array_almost_equal(retrieved, masses, decimal=10)

    def test_empty_retrieval(self, stage_with_prim):
        """Test retrieving data when nothing has been set."""
        stage, prim = stage_with_prim
        api = BioPrecisePointAPI(prim)

        positions = api.get_precise_positions()
        assert positions.shape == (0, 3)

        velocities = api.get_precise_velocities()
        assert velocities.shape == (0, 3)


class TestHelperFunctions:
    """Test helper functions."""

    def test_centroid_and_center(self):
        """Test centroiding and centering of positions."""
        positions = np.array([
            [10.0, 20.0, 30.0],
            [20.0, 30.0, 40.0],
            [30.0, 40.0, 50.0],
        ], dtype=np.float64)

        centroid, centered = centroid_and_center(positions)

        # Centroid should be the mean
        expected_centroid = np.array([20.0, 30.0, 40.0])
        np.testing.assert_array_almost_equal(centroid, expected_centroid)

        # Centered positions should have zero mean
        assert abs(centered.mean()) < 1e-6

        # Centroid should be float64
        assert centroid.dtype == np.float64

        # Centered should be float32 (for rendering)
        assert centered.dtype == np.float32

    def test_verify_no_jitter_small_position(self):
        """Test jitter verification for small positions."""
        # Small position: float32 is sufficient
        small_pos = (100.0, 100.0, 100.0)
        assert verify_no_jitter(small_pos, required_precision_meters=1e-9)

    def test_verify_no_jitter_large_position(self):
        """Test jitter verification for large positions."""
        # Large position: float32 loses precision
        large_pos = (1e8, 1e8, 1e8)
        assert not verify_no_jitter(large_pos, required_precision_meters=1e-9)

    def test_configure_stage_for_biology(self, tmp_path):
        """Test the convenience function."""
        stage_path = tmp_path / "test_configure.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))

        manager = configure_stage_for_biology(stage, BiologicalScale.ORGANELLE)

        assert isinstance(manager, PrecisionTransformManager)
        assert UsdGeom.GetStageMetersPerUnit(stage) == SCALE_CONFIGS[BiologicalScale.ORGANELLE].meters_per_unit


class TestPrecisionValidation:
    """Test precision validation functionality."""

    @pytest.fixture
    def stage(self, tmp_path):
        """Create a temporary USD stage."""
        stage_path = tmp_path / "test_validation.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        yield stage

    def test_validate_correct_precision(self, stage):
        """Test validation passes for correct precision."""
        manager = PrecisionTransformManager(stage, BiologicalScale.CELLULAR)

        # Create hierarchy with proper precision
        xform = manager.create_hierarchy_xform(
            "/World/ValidCell",
            BiologicalScale.CELLULAR,
            position=(0.0, 0.0, 0.0),
            use_double_precision=True
        )

        result = manager.validate_precision(xform.GetPrim())
        assert result["valid"]
        assert len(result["issues"]) == 0

    def test_validate_large_coords_need_double(self, stage):
        """Test validation catches large coords that need double precision."""
        manager = PrecisionTransformManager(stage, BiologicalScale.CELLULAR)

        # Create an xform manually with float precision but large coords
        xform = UsdGeom.Xform.Define(stage, "/World/LargeCoords")
        xformable = UsdGeom.Xformable(xform.GetPrim())

        # Add translate with FLOAT precision (not double)
        translate_op = xformable.AddTranslateOp(
            precision=UsdGeom.XformOp.PrecisionFloat
        )
        # Set a very large position
        translate_op.Set(Gf.Vec3f(1e8, 1e8, 1e8))

        result = manager.validate_precision(xform.GetPrim())
        # Should flag that large coords need double precision
        assert not result["valid"] or len(result["issues"]) > 0
