"""
Tests for Dynamic Payload Manager
================================

Phase B3: Tests for proximity-based payload streaming.
"""

import math
import pytest
from unittest.mock import MagicMock, patch

import numpy as np


class TestPayloadManagerConfig:
    """Tests for PayloadManagerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from cognisom.omniverse.payload_manager import PayloadManagerConfig

        config = PayloadManagerConfig()

        assert config.load_distance == 1000.0
        assert config.unload_distance == 1500.0
        assert config.max_concurrent_loads == 4
        assert config.target_memory_mb == 4096.0

    def test_hysteresis_valid(self):
        """Test that unload > load for hysteresis."""
        from cognisom.omniverse.payload_manager import PayloadManagerConfig

        config = PayloadManagerConfig(
            load_distance=100.0,
            unload_distance=150.0,
        )

        assert config.unload_distance > config.load_distance

    def test_priority_factors(self):
        """Test priority distance factors."""
        from cognisom.omniverse.payload_manager import PayloadManagerConfig

        config = PayloadManagerConfig(load_distance=1000.0)

        critical_dist = config.load_distance * config.critical_distance_factor
        high_dist = config.load_distance * config.high_priority_factor

        # Critical should be closest
        assert critical_dist < high_dist < config.load_distance


class TestPayloadInfo:
    """Tests for PayloadInfo dataclass."""

    def test_creation(self):
        """Test creating payload info."""
        from cognisom.omniverse.payload_manager import PayloadInfo, PayloadState

        info = PayloadInfo(
            path="/World/Organ",
            bbox_center=(100.0, 200.0, 300.0),
        )

        assert info.path == "/World/Organ"
        assert info.state == PayloadState.UNLOADED
        assert info.bbox_center == (100.0, 200.0, 300.0)
        assert info.last_distance == float('inf')

    def test_state_transitions(self):
        """Test payload state transitions."""
        from cognisom.omniverse.payload_manager import PayloadInfo, PayloadState

        info = PayloadInfo(path="/Test")

        # Initial state
        assert info.state == PayloadState.UNLOADED

        # Simulate loading
        info.state = PayloadState.LOADING
        assert info.state == PayloadState.LOADING

        info.state = PayloadState.LOADED
        assert info.state == PayloadState.LOADED


class TestLoadPriority:
    """Tests for LoadPriority enum."""

    def test_priority_ordering(self):
        """Test priority levels."""
        from cognisom.omniverse.payload_manager import LoadPriority

        # These should exist
        assert LoadPriority.CRITICAL
        assert LoadPriority.HIGH
        assert LoadPriority.NORMAL
        assert LoadPriority.LOW
        assert LoadPriority.DEFERRED


class TestDynamicPayloadManager:
    """Tests for DynamicPayloadManager class."""

    def test_creation_without_stage(self):
        """Test creating manager without a stage."""
        from cognisom.omniverse.payload_manager import DynamicPayloadManager

        manager = DynamicPayloadManager()

        assert manager.stage is None
        assert manager.config.load_distance == 1000.0

    def test_creation_with_custom_distances(self):
        """Test creating manager with custom distances."""
        from cognisom.omniverse.payload_manager import DynamicPayloadManager

        manager = DynamicPayloadManager(
            load_distance=500.0,
            unload_distance=800.0,
        )

        assert manager.config.load_distance == 500.0
        assert manager.config.unload_distance == 800.0

    def test_hysteresis_adjustment(self):
        """Test automatic hysteresis adjustment when unload <= load."""
        from cognisom.omniverse.payload_manager import DynamicPayloadManager

        # Invalid: unload distance <= load distance
        manager = DynamicPayloadManager(
            load_distance=500.0,
            unload_distance=400.0,  # Should be corrected
        )

        # Should be auto-adjusted to 1.5x load distance
        assert manager.config.unload_distance == 750.0  # 500 * 1.5

    def test_distance_computation(self):
        """Test distance calculation to payload center."""
        from cognisom.omniverse.payload_manager import (
            DynamicPayloadManager, PayloadInfo
        )

        manager = DynamicPayloadManager()

        info = PayloadInfo(
            path="/Test",
            bbox_center=(100.0, 0.0, 0.0),
        )

        camera_pos = (0.0, 0.0, 0.0)
        distance = manager._compute_distance(camera_pos, info)

        assert distance == 100.0

    def test_distance_3d(self):
        """Test 3D distance calculation."""
        from cognisom.omniverse.payload_manager import (
            DynamicPayloadManager, PayloadInfo
        )

        manager = DynamicPayloadManager()

        info = PayloadInfo(
            path="/Test",
            bbox_center=(3.0, 4.0, 0.0),
        )

        camera_pos = (0.0, 0.0, 0.0)
        distance = manager._compute_distance(camera_pos, info)

        # 3-4-5 triangle
        assert distance == 5.0

    def test_priority_computation_critical(self):
        """Test critical priority for very close payloads."""
        from cognisom.omniverse.payload_manager import (
            DynamicPayloadManager, LoadPriority
        )

        manager = DynamicPayloadManager(load_distance=1000.0)

        # Within 30% of load distance
        priority = manager._compute_priority(200.0)
        assert priority == LoadPriority.CRITICAL

    def test_priority_computation_high(self):
        """Test high priority for moderately close payloads."""
        from cognisom.omniverse.payload_manager import (
            DynamicPayloadManager, LoadPriority
        )

        manager = DynamicPayloadManager(load_distance=1000.0)

        # Between 30% and 60% of load distance
        priority = manager._compute_priority(500.0)
        assert priority == LoadPriority.HIGH

    def test_priority_computation_normal(self):
        """Test normal priority for payloads within load distance."""
        from cognisom.omniverse.payload_manager import (
            DynamicPayloadManager, LoadPriority
        )

        manager = DynamicPayloadManager(load_distance=1000.0)

        # Beyond 60% but within load distance
        priority = manager._compute_priority(800.0)
        assert priority == LoadPriority.NORMAL

    def test_priority_computation_low(self):
        """Test low priority for payloads beyond load distance."""
        from cognisom.omniverse.payload_manager import (
            DynamicPayloadManager, LoadPriority
        )

        manager = DynamicPayloadManager(load_distance=1000.0)

        # Beyond load distance
        priority = manager._compute_priority(1500.0)
        assert priority == LoadPriority.LOW

    def test_register_payload(self):
        """Test manual payload registration."""
        from cognisom.omniverse.payload_manager import DynamicPayloadManager

        manager = DynamicPayloadManager()

        manager.register_payload(
            path="/World/CustomOrgan",
            bbox_center=(1000.0, 2000.0, 500.0),
            memory_estimate_mb=100.0,
        )

        info = manager.get_payload_info("/World/CustomOrgan")
        assert info is not None
        assert info.bbox_center == (1000.0, 2000.0, 500.0)
        assert info.memory_estimate_mb == 100.0

    def test_update_loads_nearby_payloads(self):
        """Test that update triggers loading of nearby payloads."""
        from cognisom.omniverse.payload_manager import (
            DynamicPayloadManager, PayloadState
        )

        manager = DynamicPayloadManager(load_distance=100.0)

        # Register a payload
        manager.register_payload(
            path="/World/NearbyOrgan",
            bbox_center=(50.0, 0.0, 0.0),
        )

        # Camera at origin - payload is within load distance
        manager.update(camera_position=(0.0, 0.0, 0.0))

        info = manager.get_payload_info("/World/NearbyOrgan")
        # Should attempt to load (in mock mode, may not fully complete)
        assert info.state in [PayloadState.LOADING, PayloadState.LOADED]

    def test_update_unloads_distant_payloads(self):
        """Test that update triggers unloading of distant payloads."""
        from cognisom.omniverse.payload_manager import (
            DynamicPayloadManager, PayloadState
        )

        manager = DynamicPayloadManager(
            load_distance=100.0,
            unload_distance=150.0,
        )

        # Register and manually set as loaded
        manager.register_payload(
            path="/World/DistantOrgan",
            bbox_center=(200.0, 0.0, 0.0),
        )

        # Force load state
        manager._payloads["/World/DistantOrgan"].state = PayloadState.LOADED
        manager._loaded_paths.add("/World/DistantOrgan")

        # Camera at origin - payload is beyond unload distance
        manager.update(camera_position=(0.0, 0.0, 0.0))

        info = manager.get_payload_info("/World/DistantOrgan")
        assert info.state == PayloadState.UNLOADED

    def test_get_nearby_payloads(self):
        """Test getting payloads within a radius."""
        from cognisom.omniverse.payload_manager import DynamicPayloadManager

        manager = DynamicPayloadManager()

        # Register several payloads at different distances
        manager.register_payload("/Near1", bbox_center=(50.0, 0.0, 0.0))
        manager.register_payload("/Near2", bbox_center=(70.0, 0.0, 0.0))
        manager.register_payload("/Far", bbox_center=(500.0, 0.0, 0.0))

        nearby = manager.get_nearby_payloads(
            position=(0.0, 0.0, 0.0),
            radius=100.0,
        )

        paths = [p.path for p in nearby]
        assert "/Near1" in paths
        assert "/Near2" in paths
        assert "/Far" not in paths

    def test_load_within_radius(self):
        """Test loading all payloads within a radius."""
        from cognisom.omniverse.payload_manager import DynamicPayloadManager

        manager = DynamicPayloadManager()

        manager.register_payload("/A", bbox_center=(10.0, 0.0, 0.0))
        manager.register_payload("/B", bbox_center=(20.0, 0.0, 0.0))
        manager.register_payload("/C", bbox_center=(200.0, 0.0, 0.0))

        loaded = manager.load_within_radius(
            center=(0.0, 0.0, 0.0),
            radius=50.0,
        )

        # Should load A and B, not C
        assert loaded == 2

    def test_get_stats(self):
        """Test statistics retrieval."""
        from cognisom.omniverse.payload_manager import DynamicPayloadManager

        manager = DynamicPayloadManager()

        manager.register_payload("/A", bbox_center=(10.0, 0.0, 0.0))
        manager.register_payload("/B", bbox_center=(20.0, 0.0, 0.0))

        stats = manager.get_stats()

        assert stats.total_payloads == 2
        assert stats.unloaded_payloads >= 0

    def test_set_load_distance(self):
        """Test updating load distance."""
        from cognisom.omniverse.payload_manager import DynamicPayloadManager

        manager = DynamicPayloadManager(load_distance=100.0)

        manager.set_load_distance(200.0)

        assert manager.config.load_distance == 200.0
        # Unload distance should be auto-adjusted
        assert manager.config.unload_distance == 300.0

    def test_set_memory_budget(self):
        """Test updating memory budget."""
        from cognisom.omniverse.payload_manager import DynamicPayloadManager

        manager = DynamicPayloadManager()

        manager.set_memory_budget(8192.0)

        assert manager.config.target_memory_mb == 8192.0


class TestPayloadManagerStats:
    """Tests for PayloadManagerStats."""

    def test_default_stats(self):
        """Test default statistics values."""
        from cognisom.omniverse.payload_manager import PayloadManagerStats

        stats = PayloadManagerStats()

        assert stats.total_payloads == 0
        assert stats.loaded_payloads == 0
        assert stats.estimated_memory_mb == 0.0

    def test_memory_utilization(self):
        """Test memory utilization calculation."""
        from cognisom.omniverse.payload_manager import PayloadManagerStats

        stats = PayloadManagerStats(
            estimated_memory_mb=2048.0,
            memory_budget_mb=4096.0,
        )

        # Would be calculated during update
        stats.memory_utilization = stats.estimated_memory_mb / stats.memory_budget_mb

        assert stats.memory_utilization == 0.5


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_payload_manager(self):
        """Test create_payload_manager helper."""
        from cognisom.omniverse.payload_manager import create_payload_manager

        manager = create_payload_manager(
            stage=None,
            load_distance=500.0,
            memory_budget_mb=2048.0,
        )

        assert manager.config.load_distance == 500.0
        assert manager.config.target_memory_mb == 2048.0

    def test_create_payload_manager_auto_unload(self):
        """Test auto-calculated unload distance."""
        from cognisom.omniverse.payload_manager import create_payload_manager

        manager = create_payload_manager(
            stage=None,
            load_distance=100.0,
            # unload_distance not specified
        )

        # Should be 1.5x load distance
        assert manager.config.unload_distance == 150.0


class TestIntegrationScenarios:
    """Integration tests for common usage scenarios."""

    def test_camera_approaching_payload(self):
        """Test payload loading as camera approaches."""
        from cognisom.omniverse.payload_manager import (
            DynamicPayloadManager, PayloadState
        )

        manager = DynamicPayloadManager(load_distance=100.0)

        # Payload at fixed position
        manager.register_payload("/Target", bbox_center=(0.0, 0.0, 0.0))

        # Start far away
        manager.update(camera_position=(500.0, 0.0, 0.0))
        info = manager.get_payload_info("/Target")
        assert info.state == PayloadState.UNLOADED

        # Move closer (within load distance)
        manager.update(camera_position=(50.0, 0.0, 0.0))
        info = manager.get_payload_info("/Target")
        # Should be loading or loaded
        assert info.state in [PayloadState.LOADING, PayloadState.LOADED]

    def test_camera_leaving_payload(self):
        """Test payload unloading as camera moves away."""
        from cognisom.omniverse.payload_manager import (
            DynamicPayloadManager, PayloadState
        )

        manager = DynamicPayloadManager(
            load_distance=100.0,
            unload_distance=150.0,
        )

        manager.register_payload("/Target", bbox_center=(0.0, 0.0, 0.0))

        # Start close and load
        manager.update(camera_position=(50.0, 0.0, 0.0))

        # Force to loaded state
        manager._payloads["/Target"].state = PayloadState.LOADED
        manager._loaded_paths.add("/Target")

        # Move beyond unload distance
        manager.update(camera_position=(200.0, 0.0, 0.0))
        info = manager.get_payload_info("/Target")
        assert info.state == PayloadState.UNLOADED

    def test_hysteresis_prevents_thrashing(self):
        """Test that hysteresis prevents load/unload oscillation."""
        from cognisom.omniverse.payload_manager import (
            DynamicPayloadManager, PayloadState
        )

        manager = DynamicPayloadManager(
            load_distance=100.0,
            unload_distance=150.0,
        )

        manager.register_payload("/Target", bbox_center=(0.0, 0.0, 0.0))

        # Load the payload
        manager.update(camera_position=(50.0, 0.0, 0.0))
        manager._payloads["/Target"].state = PayloadState.LOADED
        manager._loaded_paths.add("/Target")

        # Move to just beyond load distance but within unload distance
        # (in the hysteresis zone)
        manager.update(camera_position=(120.0, 0.0, 0.0))

        info = manager.get_payload_info("/Target")
        # Should remain loaded (hysteresis)
        assert info.state == PayloadState.LOADED

    def test_multiple_payloads_priority_sorting(self):
        """Test that closer payloads load first."""
        from cognisom.omniverse.payload_manager import (
            DynamicPayloadManager, LoadPriority
        )

        manager = DynamicPayloadManager(load_distance=100.0)

        # Register payloads at different distances
        manager.register_payload("/Far", bbox_center=(80.0, 0.0, 0.0))
        manager.register_payload("/Near", bbox_center=(20.0, 0.0, 0.0))
        manager.register_payload("/Medium", bbox_center=(50.0, 0.0, 0.0))

        # Update
        manager.update(camera_position=(0.0, 0.0, 0.0))

        # Check priorities were assigned correctly
        near_info = manager.get_payload_info("/Near")
        far_info = manager.get_payload_info("/Far")

        # Near should have higher (or equal) priority
        priority_order = {
            LoadPriority.CRITICAL: 0,
            LoadPriority.HIGH: 1,
            LoadPriority.NORMAL: 2,
            LoadPriority.LOW: 3,
        }

        assert priority_order[near_info.priority] <= priority_order[far_info.priority]


class TestPayloadStateTransitions:
    """Tests for payload state machine."""

    def test_unloaded_to_loading(self):
        """Test transition from unloaded to loading."""
        from cognisom.omniverse.payload_manager import (
            DynamicPayloadManager, PayloadState
        )

        manager = DynamicPayloadManager()
        manager.register_payload("/Test", bbox_center=(0.0, 0.0, 0.0))

        info = manager.get_payload_info("/Test")
        assert info.state == PayloadState.UNLOADED

        # Trigger load (will transition through loading)
        manager._load_payload("/Test")

        info = manager.get_payload_info("/Test")
        assert info.state in [PayloadState.LOADING, PayloadState.LOADED]

    def test_loaded_to_unloaded(self):
        """Test transition from loaded to unloaded."""
        from cognisom.omniverse.payload_manager import (
            DynamicPayloadManager, PayloadState
        )

        manager = DynamicPayloadManager()
        manager.register_payload("/Test", bbox_center=(0.0, 0.0, 0.0))

        # Force to loaded
        manager._payloads["/Test"].state = PayloadState.LOADED
        manager._loaded_paths.add("/Test")

        manager._unload_payload("/Test")

        info = manager.get_payload_info("/Test")
        assert info.state == PayloadState.UNLOADED


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
