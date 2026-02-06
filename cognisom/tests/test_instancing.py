"""
Tests for Nested Instancing Architecture (Phase B2)
====================================================

Tests hierarchical USD instancing for massive biological assemblies.
"""

import pytest
import numpy as np

# Skip all tests if USD is not available
pytest.importorskip("pxr")

from cognisom.omniverse.instancing import (
    NestedInstancingManager,
    InstancingStrategy,
    PrototypeDefinition,
    InstanceData,
    create_sphere_prototype,
    create_capsule_prototype,
    create_ellipsoid_prototype,
    estimate_instancing_strategy,
)

from cognisom.omniverse.prototype_library import (
    PrototypeLibrary,
    PrototypeCategory,
    PrototypeSpec,
    ALL_PROTOTYPES,
    CELL_PROTOTYPES,
    ORGANELLE_PROTOTYPES,
)

from pxr import Usd, UsdGeom, Gf


class TestInstancingStrategy:
    """Test instancing strategy selection."""

    def test_small_count_uses_scenegraph(self):
        """Small counts should use scenegraph instancing."""
        strategy = estimate_instancing_strategy(100)
        assert strategy == InstancingStrategy.SCENEGRAPH

    def test_large_count_uses_point_instancer(self):
        """Large counts should use point instancer."""
        strategy = estimate_instancing_strategy(1000000)
        assert strategy == InstancingStrategy.POINT_INSTANCER

    def test_complex_geometry_prefers_scenegraph(self):
        """Complex geometry should prefer scenegraph even with higher counts."""
        strategy = estimate_instancing_strategy(
            entity_count=5000,
            geometry_complexity=50000
        )
        assert strategy == InstancingStrategy.SCENEGRAPH


class TestNestedInstancingManager:
    """Test NestedInstancingManager functionality."""

    @pytest.fixture
    def stage(self, tmp_path):
        """Create a temporary USD stage."""
        stage_path = tmp_path / "test_instancing.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        yield stage

    @pytest.fixture
    def manager(self, stage):
        """Create an instancing manager."""
        return NestedInstancingManager(stage)

    def test_initialization(self, manager, stage):
        """Test manager initialization creates proper hierarchy."""
        # Check prototypes container exists
        proto_prim = stage.GetPrimAtPath("/Prototypes")
        assert proto_prim.IsValid()

        # Check instances container exists
        inst_prim = stage.GetPrimAtPath("/World/Instances")
        assert inst_prim.IsValid()

    def test_register_prototype(self, manager):
        """Test prototype registration."""
        def create_test_geom(stage, path):
            UsdGeom.Sphere.Define(stage, path)

        definition = manager.register_prototype(
            name="test_cell",
            geometry_creator=create_test_geom,
            category="cell"
        )

        assert definition is not None
        assert definition.name == "test_cell"
        assert definition.prim_path == "/Prototypes/test_cell"

        # Check prototype is instanceable
        prim = manager.stage.GetPrimAtPath("/Prototypes/test_cell")
        assert prim.IsInstanceable()

    def test_create_single_instance(self, manager):
        """Test creating a single instance."""
        # Register prototype
        def create_sphere(stage, path):
            sphere = UsdGeom.Sphere.Define(stage, path)
            sphere.GetRadiusAttr().Set(5.0)

        manager.register_prototype(
            name="cell",
            geometry_creator=create_sphere
        )

        # Create instance
        position = (100.0, 200.0, 300.0)
        prim = manager.create_instance(
            "cell",
            "cell_001",
            position
        )

        assert prim is not None
        assert prim.IsValid()

        # Check position uses double precision
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        assert len(ops) > 0

        translate_op = ops[0]
        assert translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble

    def test_create_batch_scenegraph(self, manager):
        """Test batch creation with scenegraph instancing."""
        def create_sphere(stage, path):
            UsdGeom.Sphere.Define(stage, path)

        manager.register_prototype("cell", create_sphere)

        # Create instances (below point instancer threshold)
        instances = [
            InstanceData(
                prototype_name="cell",
                position=(float(i * 10), 0.0, 0.0)
            )
            for i in range(100)
        ]

        count = manager.create_instances_batch("cell", instances)
        assert count == 100

        # Check instance count
        assert manager._instance_counts["cell"] == 100

    def test_create_batch_point_instancer(self, manager):
        """Test batch creation with point instancer for large counts."""
        def create_sphere(stage, path):
            UsdGeom.Sphere.Define(stage, path)

        manager.register_prototype("particle", create_sphere)

        # Create many instances (above threshold)
        n = 15000
        instances = [
            InstanceData(
                prototype_name="particle",
                position=(float(i % 100), float(i // 100), 0.0)
            )
            for i in range(n)
        ]

        count = manager.create_instances_batch("particle", instances)
        assert count == n

        # Check point instancer was used
        assert "particle" in manager._point_instancers

    def test_nested_assembly(self, manager):
        """Test creating nested assemblies."""
        # Create component prototypes
        def create_small(stage, path):
            sphere = UsdGeom.Sphere.Define(stage, path)
            sphere.GetRadiusAttr().Set(0.1)

        manager.register_prototype("ribosome", create_small)

        # Create assembly with components
        assembly_prim = manager.create_nested_assembly(
            "mitochondrion",
            [
                {
                    "prototype": "ribosome",
                    "positions": [(0, 0, 0), (0.5, 0, 0), (1.0, 0, 0)]
                }
            ]
        )

        assert assembly_prim is not None
        assert assembly_prim.IsInstanceable()

        # Assembly should be registered as prototype
        assert "mitochondrion" in manager._prototypes

    def test_update_point_instancer_positions(self, manager):
        """Test updating positions for point instancer."""
        def create_sphere(stage, path):
            UsdGeom.Sphere.Define(stage, path)

        manager.register_prototype("particle", create_sphere)

        # Create instances
        n = 15000
        instances = [
            InstanceData(
                prototype_name="particle",
                position=(float(i), 0.0, 0.0)
            )
            for i in range(n)
        ]
        manager.create_instances_batch("particle", instances)

        # Update positions
        new_positions = np.random.randn(n, 3).astype(np.float32)
        success = manager.update_point_instancer_positions("particle", new_positions)

        assert success

    def test_get_stats(self, manager):
        """Test statistics gathering."""
        def create_sphere(stage, path):
            UsdGeom.Sphere.Define(stage, path)

        manager.register_prototype("cell", create_sphere)

        instances = [
            InstanceData(prototype_name="cell", position=(i, 0, 0))
            for i in range(50)
        ]
        manager.create_instances_batch("cell", instances)

        stats = manager.get_stats()

        assert stats["total_prototypes"] == 1
        assert stats["total_instances"] == 50
        assert "cell" in stats["instance_counts"]


class TestPrototypeLibrary:
    """Test PrototypeLibrary functionality."""

    @pytest.fixture
    def stage(self, tmp_path):
        """Create a temporary USD stage."""
        stage_path = tmp_path / "test_library.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        yield stage

    @pytest.fixture
    def library(self, stage):
        """Create a prototype library."""
        manager = NestedInstancingManager(stage)
        return PrototypeLibrary(manager)

    def test_standard_prototypes_exist(self):
        """Test that standard prototypes are defined."""
        assert len(ALL_PROTOTYPES) > 0
        assert "generic_cell" in CELL_PROTOTYPES
        assert "mitochondrion" in ORGANELLE_PROTOTYPES

    def test_prototype_specs_valid(self):
        """Test that all prototype specs are valid."""
        for name, spec in ALL_PROTOTYPES.items():
            assert spec.name == name
            assert isinstance(spec.category, PrototypeCategory)
            assert spec.default_size > 0
            assert 0 <= spec.opacity <= 1

    def test_register_standard_prototype(self, library):
        """Test registering a standard prototype."""
        definition = library.register_standard_prototype("generic_cell")

        assert definition is not None
        assert definition.name == "generic_cell"

    def test_register_all_by_category(self, library):
        """Test registering all prototypes in a category."""
        count = library.register_all_standard(
            categories=[PrototypeCategory.CELL]
        )

        assert count > 0
        assert count == len([
            s for s in ALL_PROTOTYPES.values()
            if s.category == PrototypeCategory.CELL
        ])

    def test_list_available(self, library):
        """Test listing available prototypes."""
        all_available = library.list_available()
        assert len(all_available) == len(ALL_PROTOTYPES)

        cells_only = library.list_available(PrototypeCategory.CELL)
        assert all(
            ALL_PROTOTYPES[name].category == PrototypeCategory.CELL
            for name in cells_only
        )

    def test_get_spec(self, library):
        """Test getting prototype specs."""
        spec = library.get_spec("red_blood_cell")

        assert spec is not None
        assert spec.name == "red_blood_cell"
        assert spec.category == PrototypeCategory.CELL

    def test_material_creation(self, library, stage):
        """Test that materials are created for prototypes."""
        library.register_standard_prototype("generic_cell")

        mat_path = "/Prototypes/Materials/generic_cell_Material"
        mat_prim = stage.GetPrimAtPath(mat_path)

        assert mat_prim.IsValid()


class TestGeometryCreators:
    """Test geometry creator helper functions."""

    @pytest.fixture
    def stage(self, tmp_path):
        """Create a temporary USD stage."""
        stage_path = tmp_path / "test_geom.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        yield stage

    def test_create_sphere_prototype(self, stage):
        """Test sphere prototype creation."""
        create_sphere_prototype(stage, "/Test/Sphere", radius=5.0)

        prim = stage.GetPrimAtPath("/Test/Sphere")
        assert prim.IsValid()

        sphere = UsdGeom.Sphere(prim)
        assert sphere.GetRadiusAttr().Get() == 5.0

    def test_create_capsule_prototype(self, stage):
        """Test capsule prototype creation."""
        create_capsule_prototype(stage, "/Test/Capsule", radius=1.0, height=3.0)

        prim = stage.GetPrimAtPath("/Test/Capsule")
        assert prim.IsValid()

        capsule = UsdGeom.Capsule(prim)
        assert capsule.GetRadiusAttr().Get() == 1.0
        assert capsule.GetHeightAttr().Get() == 3.0

    def test_create_ellipsoid_prototype(self, stage):
        """Test ellipsoid prototype creation."""
        create_ellipsoid_prototype(stage, "/Test/Ellipsoid", radii=(2.0, 1.0, 0.5))

        prim = stage.GetPrimAtPath("/Test/Ellipsoid")
        assert prim.IsValid()

        # Should have scale transform
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        assert len(ops) > 0
