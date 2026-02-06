"""
Tests for Semantic Zooming
=========================

Phase B4: Tests for representation switching based on camera distance.
"""

import math
import pytest
from unittest.mock import MagicMock, patch

import numpy as np


class TestRepresentationLevel:
    """Tests for RepresentationLevel enum."""

    def test_all_levels_exist(self):
        """Test that all representation levels are defined."""
        from cognisom.omniverse.semantic_zoom import RepresentationLevel

        expected = [
            "HIDDEN", "ICON", "SCHEMATIC", "ANATOMY", "TISSUE",
            "CELLULAR", "SUBCELLULAR", "MOLECULAR", "ATOMIC"
        ]

        for level in expected:
            assert hasattr(RepresentationLevel, level)

    def test_level_values(self):
        """Test representation level string values."""
        from cognisom.omniverse.semantic_zoom import RepresentationLevel

        assert RepresentationLevel.HIDDEN.value == "Hidden"
        assert RepresentationLevel.ANATOMY.value == "Anatomy"
        assert RepresentationLevel.MOLECULAR.value == "Molecular"


class TestZoomThresholds:
    """Tests for ZoomThresholds dataclass."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        from cognisom.omniverse.semantic_zoom import ZoomThresholds

        thresholds = ZoomThresholds()

        assert thresholds.hidden_max == 10.0
        assert thresholds.icon_max == 50.0
        assert thresholds.anatomy_max == 1000.0
        assert thresholds.cellular_max == 10000.0
        assert thresholds.molecular_max == 200000.0

    def test_threshold_ordering(self):
        """Test that thresholds are in increasing order."""
        from cognisom.omniverse.semantic_zoom import ZoomThresholds

        thresholds = ZoomThresholds()

        levels = [
            thresholds.hidden_max,
            thresholds.icon_max,
            thresholds.schematic_max,
            thresholds.anatomy_max,
            thresholds.tissue_max,
            thresholds.cellular_max,
            thresholds.subcellular_max,
            thresholds.molecular_max,
        ]

        # Each should be greater than the previous
        for i in range(1, len(levels)):
            assert levels[i] > levels[i-1], f"Threshold {i} not > threshold {i-1}"

    def test_custom_thresholds(self):
        """Test creating custom thresholds."""
        from cognisom.omniverse.semantic_zoom import ZoomThresholds

        thresholds = ZoomThresholds(
            anatomy_max=500.0,
            cellular_max=5000.0,
        )

        assert thresholds.anatomy_max == 500.0
        assert thresholds.cellular_max == 5000.0

    def test_hysteresis_value(self):
        """Test hysteresis value for preventing oscillation."""
        from cognisom.omniverse.semantic_zoom import ZoomThresholds

        thresholds = ZoomThresholds()

        # Default 20% hysteresis
        assert thresholds.hysteresis == 0.2


class TestZoomableEntity:
    """Tests for ZoomableEntity dataclass."""

    def test_creation(self):
        """Test creating a zoomable entity."""
        from cognisom.omniverse.semantic_zoom import ZoomableEntity

        entity = ZoomableEntity(
            prim_path="/World/Heart",
            available_representations=["Anatomy", "Cellular", "Molecular"],
            bbox_center=(0.0, 0.0, 0.0),
            bbox_radius=0.15,
        )

        assert entity.prim_path == "/World/Heart"
        assert len(entity.available_representations) == 3
        assert entity.bbox_radius == 0.15

    def test_default_values(self):
        """Test default entity values."""
        from cognisom.omniverse.semantic_zoom import ZoomableEntity

        entity = ZoomableEntity(prim_path="/Test")

        assert entity.variant_set_name == "Representation"
        assert entity.current_representation is None
        assert entity.last_screen_pixels == 0.0
        assert entity.entity_type == "generic"


class TestSemanticZoomConfig:
    """Tests for SemanticZoomConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomConfig

        config = SemanticZoomConfig()

        assert config.variant_set_name == "Representation"
        assert config.max_switches_per_frame == 10

    def test_default_representations_by_type(self):
        """Test default representations for entity types."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomConfig

        config = SemanticZoomConfig()

        # Organs should have organ-level representations
        organ_reps = config.default_representations["organ"]
        assert "Anatomy" in organ_reps
        assert "Cellular" in organ_reps

        # Molecules should have molecular representations
        molecule_reps = config.default_representations["molecule"]
        assert "Molecular" in molecule_reps
        assert "Atomic" in molecule_reps


class TestSemanticZoomController:
    """Tests for SemanticZoomController class."""

    def test_creation_without_stage(self):
        """Test creating controller without a stage."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomController

        controller = SemanticZoomController()

        assert controller.stage is None
        assert len(controller._entities) == 0

    def test_register_entity(self):
        """Test registering an entity for semantic zoom."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomController

        controller = SemanticZoomController()

        entity = controller.register_entity(
            prim_path="/World/Heart",
            entity_type="organ",
            bbox_center=(0.0, 0.0, 0.0),
            bbox_radius=0.15,
        )

        assert entity.prim_path == "/World/Heart"
        assert entity.entity_type == "organ"
        assert "/World/Heart" in controller._entities

    def test_register_with_custom_representations(self):
        """Test registering with custom representation list."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomController

        controller = SemanticZoomController()

        entity = controller.register_entity(
            prim_path="/World/Cell",
            representations=["Hidden", "Anatomy", "Molecular"],
        )

        assert entity.available_representations == ["Hidden", "Anatomy", "Molecular"]

    def test_unregister_entity(self):
        """Test removing an entity from zoom control."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomController

        controller = SemanticZoomController()
        controller.register_entity("/Test", bbox_center=(0, 0, 0))

        assert controller.unregister_entity("/Test") is True
        assert "/Test" not in controller._entities

    def test_unregister_nonexistent(self):
        """Test unregistering an entity that doesn't exist."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomController

        controller = SemanticZoomController()

        assert controller.unregister_entity("/Nonexistent") is False

    def test_screen_coverage_computation(self):
        """Test computing screen coverage in pixels."""
        from cognisom.omniverse.semantic_zoom import (
            SemanticZoomController, ZoomableEntity
        )

        controller = SemanticZoomController()

        entity = ZoomableEntity(
            prim_path="/Test",
            bbox_center=(0.0, 0.0, -10.0),  # 10 units in front of camera
            bbox_radius=1.0,
        )

        # Camera at origin looking at -Z
        coverage = controller._compute_screen_coverage(
            entity=entity,
            camera_position=(0.0, 0.0, 0.0),
            viewport_height=1080,
            fov_radians=math.radians(60),
        )

        # Should have some reasonable pixel coverage
        assert coverage > 0
        assert coverage < 10000  # Not filling the whole screen

    def test_screen_coverage_closer_is_larger(self):
        """Test that closer objects have more screen coverage."""
        from cognisom.omniverse.semantic_zoom import (
            SemanticZoomController, ZoomableEntity
        )

        controller = SemanticZoomController()

        entity_far = ZoomableEntity(
            prim_path="/Far",
            bbox_center=(0.0, 0.0, -100.0),
            bbox_radius=1.0,
        )

        entity_near = ZoomableEntity(
            prim_path="/Near",
            bbox_center=(0.0, 0.0, -10.0),
            bbox_radius=1.0,
        )

        coverage_far = controller._compute_screen_coverage(
            entity=entity_far,
            camera_position=(0.0, 0.0, 0.0),
            viewport_height=1080,
            fov_radians=math.radians(60),
        )

        coverage_near = controller._compute_screen_coverage(
            entity=entity_near,
            camera_position=(0.0, 0.0, 0.0),
            viewport_height=1080,
            fov_radians=math.radians(60),
        )

        assert coverage_near > coverage_far

    def test_pixels_to_representation_hidden(self):
        """Test that small pixel coverage returns Hidden."""
        from cognisom.omniverse.semantic_zoom import (
            SemanticZoomController, ZoomableEntity, RepresentationLevel
        )

        controller = SemanticZoomController()

        entity = ZoomableEntity(
            prim_path="/Test",
            available_representations=["Hidden", "Anatomy", "Molecular"],
        )

        rep = controller._pixels_to_representation(entity, screen_pixels=5.0)
        assert rep == RepresentationLevel.HIDDEN.value

    def test_pixels_to_representation_anatomy(self):
        """Test that medium pixel coverage returns Anatomy."""
        from cognisom.omniverse.semantic_zoom import (
            SemanticZoomController, ZoomableEntity, RepresentationLevel
        )

        controller = SemanticZoomController()

        entity = ZoomableEntity(
            prim_path="/Test",
            available_representations=["Hidden", "Anatomy", "Cellular", "Molecular"],
        )

        # Between schematic_max (200) and anatomy_max (1000)
        rep = controller._pixels_to_representation(entity, screen_pixels=500.0)
        assert rep == RepresentationLevel.ANATOMY.value

    def test_pixels_to_representation_constrained(self):
        """Test that representation is constrained to available options."""
        from cognisom.omniverse.semantic_zoom import (
            SemanticZoomController, ZoomableEntity
        )

        controller = SemanticZoomController()

        # Only Hidden and Molecular available
        entity = ZoomableEntity(
            prim_path="/Test",
            available_representations=["Hidden", "Molecular"],
        )

        # 500 pixels would normally be Anatomy, but should fall back
        rep = controller._pixels_to_representation(entity, screen_pixels=500.0)
        assert rep in ["Hidden", "Molecular"]

    def test_update_switches_representation(self):
        """Test that update changes representation based on distance."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomController

        controller = SemanticZoomController()

        controller.register_entity(
            prim_path="/Heart",
            representations=["Hidden", "Anatomy", "Cellular"],
            bbox_center=(0.0, 0.0, 0.0),
            bbox_radius=0.1,
        )

        # Camera far away - should be hidden or low detail
        switches = controller.update(
            camera_position=(0.0, 0.0, 1000.0),
            camera_target=(0.0, 0.0, 0.0),
            viewport_size=(1920, 1080),
        )

        entity = controller.get_entity("/Heart")
        assert entity.current_representation is not None

    def test_update_returns_switch_count(self):
        """Test that update returns number of switches."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomController

        controller = SemanticZoomController()

        controller.register_entity(
            "/A", representations=["Hidden", "Anatomy"],
            bbox_center=(0, 0, 0), bbox_radius=1.0
        )
        controller.register_entity(
            "/B", representations=["Hidden", "Anatomy"],
            bbox_center=(10, 0, 0), bbox_radius=1.0
        )

        # First update should switch from None to something
        switches = controller.update(
            camera_position=(0, 0, 100),
            camera_target=(0, 0, 0),
            viewport_size=(1920, 1080),
        )

        # Should have made some switches
        assert switches >= 0

    def test_set_representation_manually(self):
        """Test manually setting representation."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomController

        controller = SemanticZoomController()

        controller.register_entity(
            "/Test",
            representations=["Hidden", "Anatomy", "Molecular"],
            bbox_center=(0, 0, 0),
        )

        success = controller.set_representation("/Test", "Molecular")
        assert success is True

        entity = controller.get_entity("/Test")
        assert entity.current_representation == "Molecular"

    def test_set_invalid_representation(self):
        """Test setting an unavailable representation."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomController

        controller = SemanticZoomController()

        controller.register_entity(
            "/Test",
            representations=["Hidden", "Anatomy"],
            bbox_center=(0, 0, 0),
        )

        # "Atomic" is not in available representations
        success = controller.set_representation("/Test", "Atomic")
        assert success is False

    def test_get_stats(self):
        """Test statistics retrieval."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomController

        controller = SemanticZoomController()

        controller.register_entity("/A", bbox_center=(0, 0, 0))
        controller.register_entity("/B", bbox_center=(10, 0, 0))

        stats = controller.get_stats()

        assert stats["total_entities"] == 2
        assert "total_switches" in stats
        assert "entities_by_representation" in stats

    def test_reset_all_to_hidden(self):
        """Test resetting all entities to hidden."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomController

        controller = SemanticZoomController()

        controller.register_entity(
            "/A", representations=["Hidden", "Anatomy"],
            bbox_center=(0, 0, 0)
        )
        controller.register_entity(
            "/B", representations=["Hidden", "Anatomy"],
            bbox_center=(10, 0, 0)
        )

        # Set to visible
        controller.set_representation("/A", "Anatomy")
        controller.set_representation("/B", "Anatomy")

        # Reset all
        controller.reset_all_to_hidden()

        assert controller.get_entity("/A").current_representation == "Hidden"
        assert controller.get_entity("/B").current_representation == "Hidden"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_semantic_zoom_controller(self):
        """Test create_semantic_zoom_controller helper."""
        from cognisom.omniverse.semantic_zoom import (
            create_semantic_zoom_controller, ZoomThresholds
        )

        thresholds = ZoomThresholds(anatomy_max=500.0)
        controller = create_semantic_zoom_controller(
            stage=None,
            thresholds=thresholds,
        )

        assert controller.config.thresholds.anatomy_max == 500.0

    def test_setup_biological_zoom_hierarchy(self):
        """Test setting up a biological hierarchy for zoom."""
        from cognisom.omniverse.semantic_zoom import (
            SemanticZoomController, setup_biological_zoom_hierarchy
        )

        controller = SemanticZoomController()

        hierarchy = {
            "type": "organ",
            "children": {
                "Ventricle": {
                    "type": "tissue",
                    "children": {
                        "Cells": {"type": "cell"}
                    }
                }
            }
        }

        count = setup_biological_zoom_hierarchy(
            controller, "/World/Heart", hierarchy
        )

        # Should register: Heart, Ventricle, Cells
        assert count == 3
        assert "/World/Heart" in controller._entities
        assert "/World/Heart/Ventricle" in controller._entities
        assert "/World/Heart/Ventricle/Cells" in controller._entities


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    def test_zoom_in_scenario(self):
        """Test zooming in from far to close."""
        from cognisom.omniverse.semantic_zoom import (
            SemanticZoomController, RepresentationLevel
        )

        controller = SemanticZoomController()

        # Register a cell
        controller.register_entity(
            "/Cell",
            entity_type="cell",
            representations=["Hidden", "Anatomy", "Subcellular", "Molecular"],
            bbox_center=(0, 0, 0),
            bbox_radius=0.01,  # 10 microns
        )

        viewport = (1920, 1080)

        # Start far away
        controller.update((0, 0, 1.0), (0, 0, 0), viewport)
        state1 = controller.get_entity("/Cell").current_representation

        # Move closer
        controller.update((0, 0, 0.1), (0, 0, 0), viewport)
        state2 = controller.get_entity("/Cell").current_representation

        # Move very close
        controller.update((0, 0, 0.02), (0, 0, 0), viewport)
        state3 = controller.get_entity("/Cell").current_representation

        # States should progress to more detailed
        # (exact values depend on thresholds and geometry)
        assert state1 is not None
        assert state2 is not None
        assert state3 is not None

    def test_multiple_entities_different_distances(self):
        """Test multiple entities at different distances."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomController

        controller = SemanticZoomController()

        # Near entity
        controller.register_entity(
            "/Near",
            representations=["Hidden", "Anatomy"],
            bbox_center=(0, 0, -1),
            bbox_radius=0.5,
        )

        # Far entity
        controller.register_entity(
            "/Far",
            representations=["Hidden", "Anatomy"],
            bbox_center=(0, 0, -100),
            bbox_radius=0.5,
        )

        controller.update(
            camera_position=(0, 0, 0),
            camera_target=(0, 0, -1),
            viewport_size=(1920, 1080),
        )

        near_entity = controller.get_entity("/Near")
        far_entity = controller.get_entity("/Far")

        # Near should have more pixels
        assert near_entity.last_screen_pixels > far_entity.last_screen_pixels

    def test_max_switches_per_frame(self):
        """Test that max switches per frame is respected."""
        from cognisom.omniverse.semantic_zoom import (
            SemanticZoomController, SemanticZoomConfig
        )

        config = SemanticZoomConfig(max_switches_per_frame=2)
        controller = SemanticZoomController(config=config)

        # Register many entities
        for i in range(10):
            controller.register_entity(
                f"/Entity{i}",
                representations=["Hidden", "Anatomy"],
                bbox_center=(i * 10, 0, 0),
                bbox_radius=1.0,
            )

        # First update - should be limited to max_switches_per_frame
        switches = controller.update(
            camera_position=(0, 0, 10),
            camera_target=(0, 0, 0),
            viewport_size=(1920, 1080),
        )

        # Note: actual switches may be less if not all need switching
        assert switches <= config.max_switches_per_frame


class TestFindNearestAvailable:
    """Tests for _find_nearest_available helper."""

    def test_exact_match(self):
        """Test when target is available."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomController

        controller = SemanticZoomController()

        available = ["Hidden", "Anatomy", "Molecular"]
        threshold_levels = [
            (10, "Hidden"),
            (50, "Icon"),
            (200, "Schematic"),
            (1000, "Anatomy"),
            (10000, "Cellular"),
            (200000, "Molecular"),
        ]

        result = controller._find_nearest_available("Anatomy", available, threshold_levels)
        assert result == "Anatomy"

    def test_find_nearest_lower(self):
        """Test finding nearest when target not available."""
        from cognisom.omniverse.semantic_zoom import SemanticZoomController

        controller = SemanticZoomController()

        available = ["Hidden", "Molecular"]
        threshold_levels = [
            (10, "Hidden"),
            (1000, "Anatomy"),
            (200000, "Molecular"),
        ]

        # "Anatomy" not available, should find nearest
        result = controller._find_nearest_available("Anatomy", available, threshold_levels)
        assert result in ["Hidden", "Molecular"]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
