"""
Tests for Mixed-Precision Simulation Bridge
==========================================

Phase B6: Tests for precision trade-off management.
"""

import pytest
import numpy as np


class TestPrecisionLevel:
    """Tests for PrecisionLevel enum."""

    def test_precision_levels_exist(self):
        """Test that all precision levels are defined."""
        from cognisom.gpu.mixed_precision import PrecisionLevel

        assert PrecisionLevel.FLOAT16
        assert PrecisionLevel.FLOAT32
        assert PrecisionLevel.FLOAT64


class TestGPUCapability:
    """Tests for GPUCapability enum."""

    def test_capability_tiers(self):
        """Test GPU capability tiers."""
        from cognisom.gpu.mixed_precision import GPUCapability

        assert GPUCapability.CONSUMER
        assert GPUCapability.PROFESSIONAL
        assert GPUCapability.DATACENTER
        assert GPUCapability.CPU_ONLY


class TestMixedPrecisionConfig:
    """Tests for MixedPrecisionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from cognisom.gpu.mixed_precision import MixedPrecisionConfig, PrecisionLevel

        config = MixedPrecisionConfig()

        # Positions should always be FP64
        assert config.position_precision == PrecisionLevel.FLOAT64
        # Render should be FP32
        assert config.render_precision == PrecisionLevel.FLOAT32

    def test_auto_detect(self):
        """Test auto-detection of optimal config."""
        from cognisom.gpu.mixed_precision import MixedPrecisionConfig, PrecisionLevel

        config = MixedPrecisionConfig.auto_detect()

        # Positions must always be FP64 regardless of hardware
        assert config.position_precision == PrecisionLevel.FLOAT64

    def test_high_accuracy(self):
        """Test high-accuracy configuration."""
        from cognisom.gpu.mixed_precision import MixedPrecisionConfig, PrecisionLevel

        config = MixedPrecisionConfig.high_accuracy()

        assert config.position_precision == PrecisionLevel.FLOAT64
        assert config.velocity_precision == PrecisionLevel.FLOAT64
        assert config.force_precision == PrecisionLevel.FLOAT64

    def test_high_performance(self):
        """Test high-performance configuration."""
        from cognisom.gpu.mixed_precision import MixedPrecisionConfig, PrecisionLevel

        config = MixedPrecisionConfig.high_performance()

        # Even high-perf keeps positions in FP64
        assert config.position_precision == PrecisionLevel.FLOAT64
        # But forces can be FP32
        assert config.force_precision == PrecisionLevel.FLOAT32


class TestPrecisionConversion:
    """Tests for precision conversion utilities."""

    def test_precision_to_dtype(self):
        """Test converting precision level to numpy dtype."""
        from cognisom.gpu.mixed_precision import precision_to_dtype, PrecisionLevel

        assert precision_to_dtype(PrecisionLevel.FLOAT16) == np.float16
        assert precision_to_dtype(PrecisionLevel.FLOAT32) == np.float32
        assert precision_to_dtype(PrecisionLevel.FLOAT64) == np.float64


class TestSimulationState:
    """Tests for SimulationState dataclass."""

    def test_state_creation(self):
        """Test creating simulation state."""
        from cognisom.gpu.mixed_precision import SimulationState

        n = 100
        state = SimulationState(
            positions=np.zeros((n, 3), dtype=np.float64),
            velocities=np.zeros((n, 3), dtype=np.float64),
            masses=np.ones(n, dtype=np.float64),
            forces=np.zeros((n, 3), dtype=np.float32),
        )

        assert state.positions.shape == (100, 3)
        assert state.positions.dtype == np.float64
        assert state.forces.dtype == np.float32


class TestMixedPrecisionSimulator:
    """Tests for MixedPrecisionSimulator class."""

    def test_creation(self):
        """Test creating simulator."""
        from cognisom.gpu.mixed_precision import MixedPrecisionSimulator

        sim = MixedPrecisionSimulator()
        assert sim.config is not None
        assert not sim._initialized

    def test_initialize(self):
        """Test initializing with positions."""
        from cognisom.gpu.mixed_precision import MixedPrecisionSimulator

        sim = MixedPrecisionSimulator()

        positions = np.random.rand(100, 3).astype(np.float64)
        sim.initialize(positions)

        assert sim._initialized
        assert sim.n_particles == 100

    def test_initialize_with_velocities(self):
        """Test initializing with positions and velocities."""
        from cognisom.gpu.mixed_precision import MixedPrecisionSimulator

        sim = MixedPrecisionSimulator()

        positions = np.random.rand(50, 3).astype(np.float64)
        velocities = np.random.rand(50, 3).astype(np.float64) * 0.1

        sim.initialize(positions, velocities)

        assert sim.n_particles == 50
        np.testing.assert_array_almost_equal(
            sim.get_velocities(), velocities, decimal=5
        )

    def test_initialize_with_masses(self):
        """Test initializing with custom masses."""
        from cognisom.gpu.mixed_precision import MixedPrecisionSimulator

        sim = MixedPrecisionSimulator()

        positions = np.random.rand(25, 3)
        masses = np.random.rand(25) * 10 + 1  # 1-11 mass units

        sim.initialize(positions, masses=masses)

        # Mass should be preserved
        state = sim.get_state_dict()
        np.testing.assert_array_almost_equal(state['masses'], masses)

    def test_step(self):
        """Test simulation step."""
        from cognisom.gpu.mixed_precision import MixedPrecisionSimulator

        sim = MixedPrecisionSimulator()

        positions = np.zeros((10, 3), dtype=np.float64)
        velocities = np.ones((10, 3), dtype=np.float64)  # Moving at 1 unit/time

        sim.initialize(positions, velocities)

        # Step with no forces
        forces = np.zeros((10, 3), dtype=np.float32)
        sim.step(dt=1.0, forces=forces)

        # Positions should have moved by velocity * dt
        new_positions = sim.get_positions()
        np.testing.assert_array_almost_equal(new_positions, np.ones((10, 3)))

    def test_step_with_forces(self):
        """Test step with forces applied."""
        from cognisom.gpu.mixed_precision import MixedPrecisionSimulator

        sim = MixedPrecisionSimulator()

        positions = np.zeros((1, 3), dtype=np.float64)
        velocities = np.zeros((1, 3), dtype=np.float64)
        masses = np.array([1.0])  # Unit mass

        sim.initialize(positions, velocities, masses)

        # Apply force in x direction
        forces = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        sim.step(dt=1.0, forces=forces)

        # x = 0.5 * a * t^2 = 0.5 * 1 * 1 = 0.5
        new_positions = sim.get_positions()
        assert np.isclose(new_positions[0, 0], 0.5, atol=1e-10)

    def test_get_render_positions(self):
        """Test getting render-ready positions."""
        from cognisom.gpu.mixed_precision import MixedPrecisionSimulator

        sim = MixedPrecisionSimulator()

        positions = np.random.rand(100, 3).astype(np.float64)
        sim.initialize(positions)

        render_pos = sim.get_render_positions()

        # Should be float32
        assert render_pos.dtype == np.float32
        # Values should be close
        np.testing.assert_array_almost_equal(render_pos, positions, decimal=5)

    def test_positions_stay_fp64(self):
        """Test that internal positions remain float64."""
        from cognisom.gpu.mixed_precision import MixedPrecisionSimulator

        sim = MixedPrecisionSimulator()

        # Initialize with float32 input
        positions = np.random.rand(10, 3).astype(np.float32)
        sim.initialize(positions)

        # Internal positions should be float64
        internal_pos = sim.get_positions()
        assert internal_pos.dtype == np.float64

    def test_step_count(self):
        """Test step counter."""
        from cognisom.gpu.mixed_precision import MixedPrecisionSimulator

        sim = MixedPrecisionSimulator()
        sim.initialize(np.random.rand(10, 3))

        assert sim.step_count == 0

        forces = np.zeros((10, 3), dtype=np.float32)
        sim.step(dt=0.01, forces=forces)
        assert sim.step_count == 1

        sim.step(dt=0.01, forces=forces)
        sim.step(dt=0.01, forces=forces)
        assert sim.step_count == 3

    def test_get_state_dict(self):
        """Test getting state as dictionary."""
        from cognisom.gpu.mixed_precision import MixedPrecisionSimulator

        sim = MixedPrecisionSimulator()
        sim.initialize(np.random.rand(50, 3))

        state = sim.get_state_dict()

        assert 'positions' in state
        assert 'velocities' in state
        assert 'masses' in state
        assert 'step_count' in state
        assert 'config' in state


class TestDriftCorrection:
    """Tests for drift correction functionality."""

    def test_drift_correction_config(self):
        """Test drift correction configuration."""
        from cognisom.gpu.mixed_precision import MixedPrecisionConfig

        config = MixedPrecisionConfig()

        assert config.enable_drift_correction is True
        assert config.drift_correction_interval > 0

    def test_drift_accumulation(self):
        """Test that drift is tracked."""
        from cognisom.gpu.mixed_precision import MixedPrecisionSimulator, MixedPrecisionConfig

        # Create config with very frequent correction
        config = MixedPrecisionConfig(drift_correction_interval=5)
        sim = MixedPrecisionSimulator(config)

        positions = np.random.rand(10, 3) * 100
        velocities = np.random.rand(10, 3) * 0.1

        sim.initialize(positions, velocities)

        # Run several steps
        forces = np.zeros((10, 3), dtype=np.float32)
        for _ in range(10):
            sim.step(dt=0.1, forces=forces)

        state = sim.get_state_dict()
        # Drift should have been tracked
        assert 'drift_accumulated' in state


class TestGPUDetection:
    """Tests for GPU detection."""

    def test_detect_gpu_capability(self):
        """Test GPU capability detection."""
        from cognisom.gpu.mixed_precision import detect_gpu_capability, GPUCapability

        cap = detect_gpu_capability()

        # Should return a valid capability
        assert cap in GPUCapability


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_detect_optimal_precision(self):
        """Test detect_optimal_precision function."""
        from cognisom.gpu.mixed_precision import detect_optimal_precision

        config = detect_optimal_precision()
        assert config is not None

    def test_create_mixed_precision_simulator(self):
        """Test create_mixed_precision_simulator function."""
        from cognisom.gpu.mixed_precision import create_mixed_precision_simulator

        sim_accuracy = create_mixed_precision_simulator(accuracy_priority=True)
        sim_perf = create_mixed_precision_simulator(accuracy_priority=False)

        assert sim_accuracy is not None
        assert sim_perf is not None


class TestIntegrationNumerics:
    """Tests for numerical accuracy of integration."""

    def test_constant_velocity_trajectory(self):
        """Test that constant velocity produces linear trajectory."""
        from cognisom.gpu.mixed_precision import MixedPrecisionSimulator

        sim = MixedPrecisionSimulator()

        # Single particle moving at constant velocity
        positions = np.array([[0.0, 0.0, 0.0]])
        velocities = np.array([[1.0, 2.0, 3.0]])

        sim.initialize(positions, velocities)

        # Step with no forces
        dt = 0.1
        forces = np.zeros((1, 3), dtype=np.float32)

        for _ in range(100):
            sim.step(dt, forces)

        # After 100 steps of dt=0.1, t=10
        # x = v*t = [10, 20, 30]
        final_pos = sim.get_positions()[0]
        expected = np.array([10.0, 20.0, 30.0])

        np.testing.assert_array_almost_equal(final_pos, expected, decimal=5)

    def test_constant_acceleration(self):
        """Test motion under constant force."""
        from cognisom.gpu.mixed_precision import MixedPrecisionSimulator

        sim = MixedPrecisionSimulator()

        positions = np.array([[0.0, 0.0, 0.0]])
        velocities = np.array([[0.0, 0.0, 0.0]])
        masses = np.array([2.0])  # m=2

        sim.initialize(positions, velocities, masses)

        # Constant force F=2 in x, so a=1
        dt = 0.01
        forces = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)

        for _ in range(100):
            sim.step(dt, forces)

        # After t=1: x = 0.5*a*t^2 = 0.5*1*1 = 0.5
        # With discrete integration, should be close
        final_pos = sim.get_positions()[0]
        assert np.isclose(final_pos[0], 0.5, rtol=0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
