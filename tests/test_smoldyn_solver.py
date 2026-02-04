#!/usr/bin/env python3
"""
Unit Tests for Smoldyn Spatial Stochastic Solver
=================================================

Tests for the GPU-accelerated Smoldyn-style particle-based simulation.

Run with:
    cd cognisom && python -m pytest tests/test_smoldyn_solver.py -v
"""

import sys
import os

# Ensure cognisom package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from typing import Dict, List


# ────────────────────────────────────────────────────────────────
# Test Data Classes
# ────────────────────────────────────────────────────────────────

class TestSmoldynSpecies:
    """Tests for SmoldynSpecies dataclass."""

    def test_species_creation(self):
        """Test basic species creation."""
        from gpu.smoldyn_solver import SmoldynSpecies

        species = SmoldynSpecies(
            name="A",
            diffusion_coeff=1.0,
        )
        assert species.name == "A"
        assert species.diffusion_coeff == 1.0

    def test_species_defaults(self):
        """Test species default values."""
        from gpu.smoldyn_solver import SmoldynSpecies

        species = SmoldynSpecies(name="B", diffusion_coeff=0.5)
        assert species.color == (1.0, 1.0, 1.0)
        assert species.radius == 0.01


class TestSmoldynReaction:
    """Tests for SmoldynReaction dataclass."""

    def test_unimolecular_reaction(self):
        """Test first-order reaction creation."""
        from gpu.smoldyn_solver import SmoldynReaction, ReactionType

        rxn = SmoldynReaction(
            name="decay",
            reaction_type=ReactionType.FIRST,
            rate=0.1,
            reactants=["A"],
            products=[],
        )
        assert rxn.name == "decay"
        assert rxn.reaction_type == ReactionType.FIRST
        assert rxn.binding_radius == 0.0

    def test_bimolecular_reaction(self):
        """Test second-order reaction creation."""
        from gpu.smoldyn_solver import SmoldynReaction, ReactionType

        rxn = SmoldynReaction(
            name="binding",
            reaction_type=ReactionType.SECOND,
            rate=1e6,
            reactants=["A", "B"],
            products=["C"],
            binding_radius=0.01,
        )
        assert rxn.reaction_type == ReactionType.SECOND
        assert rxn.binding_radius == 0.01


class TestSmoldynSystem:
    """Tests for SmoldynSystem dataclass and factory methods."""

    def test_simple_binding_system(self):
        """Test creation of simple A+B->C system."""
        from gpu.smoldyn_solver import SmoldynSystem

        system = SmoldynSystem.simple_binding()

        # Check species
        species_names = [s.name for s in system.species]
        assert "A" in species_names
        assert "B" in species_names
        assert "C" in species_names

        # Check reactions
        assert len(system.reactions) >= 1
        binding_rxn = next(r for r in system.reactions if r.name == "bind")
        assert binding_rxn.reactants == ["A", "B"]
        assert binding_rxn.products == ["C"]

    def test_enzyme_kinetics_system(self):
        """Test creation of enzyme kinetics system."""
        from gpu.smoldyn_solver import SmoldynSystem

        system = SmoldynSystem.enzyme_kinetics()

        species_names = [s.name for s in system.species]
        assert "E" in species_names  # Enzyme
        assert "S" in species_names  # Substrate
        assert "ES" in species_names  # Complex
        assert "P" in species_names  # Product

        # Should have binding, unbinding, and catalysis reactions
        assert len(system.reactions) >= 3

    def test_min_oscillator_system(self):
        """Test creation of MinDE oscillator system."""
        from gpu.smoldyn_solver import SmoldynSystem

        system = SmoldynSystem.min_oscillator()

        species_names = [s.name for s in system.species]
        assert "MinD_ATP" in species_names
        assert "MinE" in species_names


# ────────────────────────────────────────────────────────────────
# Test Particle System
# ────────────────────────────────────────────────────────────────

class TestParticleSystem:
    """Tests for ParticleSystem class."""

    def test_particle_system_creation(self):
        """Test particle system initialization."""
        from gpu.smoldyn_solver import ParticleSystem

        ps = ParticleSystem(n_max=10000, n_species=3)

        assert ps.n_max == 10000
        assert ps.positions.shape == (10000, 3)
        assert ps.species.shape == (10000,)
        assert ps.alive.shape == (10000,)
        assert ps.n_alive == 0

    def test_add_particles(self):
        """Test adding particles to system."""
        from gpu.smoldyn_solver import ParticleSystem

        ps = ParticleSystem(n_max=1000, n_species=3)

        positions = np.random.uniform(0, 10, (100, 3)).astype(np.float32)
        ps.add_particles(species_idx=0, positions=positions)

        assert ps.n_alive == 100

    def test_add_multiple_species(self):
        """Test adding particles of different species."""
        from gpu.smoldyn_solver import ParticleSystem

        ps = ParticleSystem(n_max=1000, n_species=3)

        pos_a = np.random.uniform(0, 10, (50, 3)).astype(np.float32)
        pos_b = np.random.uniform(0, 10, (30, 3)).astype(np.float32)

        ps.add_particles(species_idx=0, positions=pos_a)
        ps.add_particles(species_idx=1, positions=pos_b)

        assert ps.n_alive == 80

    def test_remove_particles(self):
        """Test removing particles."""
        from gpu.smoldyn_solver import ParticleSystem

        ps = ParticleSystem(n_max=1000, n_species=3)
        positions = np.random.uniform(0, 10, (100, 3)).astype(np.float32)
        ps.add_particles(species_idx=0, positions=positions)

        # Remove some particles
        ps.alive[10:20] = 0

        assert ps.n_alive == 90


# ────────────────────────────────────────────────────────────────
# Test Smoldyn Solver
# ────────────────────────────────────────────────────────────────

class TestSmoldynSolver:
    """Tests for main SmoldynSolver class."""

    def test_solver_initialization(self):
        """Test solver initialization."""
        from gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

        system = SmoldynSystem.simple_binding()
        solver = SmoldynSolver(
            system=system,
            n_max_particles=10000,
        )

        # Add some particles
        positions = np.random.uniform(0, 100, (500, 3)).astype(np.float32)
        solver.add_particles("A", positions)

        assert solver.particles.n_alive == 500

    def test_single_step(self):
        """Test single simulation step."""
        from gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

        system = SmoldynSystem.simple_binding()
        solver = SmoldynSolver(
            system=system,
            n_max_particles=10000,
        )

        # Add particles
        pos_a = np.random.uniform(0, 100, (500, 3)).astype(np.float32)
        pos_b = np.random.uniform(0, 100, (500, 3)).astype(np.float32)
        solver.add_particles("A", pos_a)
        solver.add_particles("B", pos_b)

        solver.step(dt=0.001)

        # Particles should have moved, system should be valid
        assert solver.particles.n_alive >= 0
        assert solver.particles.n_alive <= solver.n_max

    def test_multiple_steps(self):
        """Test multiple simulation steps."""
        from gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

        system = SmoldynSystem.simple_binding()
        solver = SmoldynSolver(
            system=system,
            n_max_particles=10000,
        )

        pos_a = np.random.uniform(0, 100, (500, 3)).astype(np.float32)
        pos_b = np.random.uniform(0, 100, (500, 3)).astype(np.float32)
        solver.add_particles("A", pos_a)
        solver.add_particles("B", pos_b)

        for _ in range(100):
            solver.step(dt=0.001)

        # System should still be valid
        assert solver.particles.n_alive >= 0

    def test_boundary_conditions_reflective(self):
        """Test reflective boundary conditions."""
        from gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

        system = SmoldynSystem.simple_binding()
        solver = SmoldynSolver(
            system=system,
            n_max_particles=10000,
        )

        # Add particles
        pos_a = np.random.uniform(0, 100, (500, 3)).astype(np.float32)
        solver.add_particles("A", pos_a)

        # Run several steps
        for _ in range(100):
            solver.step(dt=0.01)

        # All particles should be within domain (0-100 from simple_binding)
        alive_mask = solver.particles.alive.astype(bool)
        positions = solver.particles.positions[alive_mask]

        if len(positions) > 0:
            assert np.all(positions >= 0)
            assert np.all(positions[:, 0] <= 100.0)
            assert np.all(positions[:, 1] <= 100.0)
            assert np.all(positions[:, 2] <= 100.0)

    def test_particle_count(self):
        """Test species counting."""
        from gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

        system = SmoldynSystem.simple_binding()
        solver = SmoldynSolver(system=system, n_max_particles=10000)

        pos_a = np.random.uniform(0, 100, (300, 3)).astype(np.float32)
        pos_b = np.random.uniform(0, 100, (200, 3)).astype(np.float32)
        solver.add_particles("A", pos_a)
        solver.add_particles("B", pos_b)

        assert solver.count_species("A") == 300
        assert solver.count_species("B") == 200


class TestBrownianMotion:
    """Tests for Brownian motion implementation."""

    def test_mean_squared_displacement(self):
        """Test MSD follows 6Dt for 3D diffusion."""
        from gpu.smoldyn_solver import (
            SmoldynSolver, SmoldynSystem, SmoldynSpecies,
            SmoldynCompartment, BoundaryType
        )

        # Create system with single species, no reactions
        species = SmoldynSpecies(
            name="A",
            diffusion_coeff=1.0,  # 1 um^2/s
        )
        compartment = SmoldynCompartment(
            name="large_domain",
            bounds=(0, 1000, 0, 1000, 0, 1000),  # Large domain
            boundary_type=BoundaryType.PERIODIC,
        )
        system = SmoldynSystem(
            species=[species],
            reactions=[],
            compartment=compartment,
            name="diffusion_test",
        )

        solver = SmoldynSolver(
            system=system,
            n_max_particles=10000,
        )

        # Add particles in center of domain
        n_particles = 5000
        positions = np.random.uniform(400, 600, (n_particles, 3)).astype(np.float32)
        solver.add_particles("A", positions)

        # Record initial positions
        initial_positions = solver.particles.positions[:n_particles].copy()

        # Run for 1 second
        dt = 0.001
        n_steps = 1000
        for _ in range(n_steps):
            solver.step(dt)

        t = dt * n_steps  # 1 second

        # Calculate MSD
        alive_mask = solver.particles.alive[:n_particles].astype(bool)
        final_positions = solver.particles.positions[:n_particles]
        displacements = final_positions[alive_mask] - initial_positions[alive_mask]
        msd = np.mean(np.sum(displacements**2, axis=1))

        # Expected MSD = 6Dt = 6 * 1.0 * 1.0 = 6.0
        expected_msd = 6.0 * 1.0 * t

        # Allow 50% tolerance for stochastic variation
        assert abs(msd - expected_msd) / expected_msd < 0.50, \
            f"MSD {msd:.2f} differs from expected {expected_msd:.2f}"


class TestReactions:
    """Tests for reaction execution."""

    def test_unimolecular_decay(self):
        """Test first-order decay reaction."""
        from gpu.smoldyn_solver import (
            SmoldynSolver, SmoldynSystem, SmoldynSpecies, SmoldynReaction,
            SmoldynCompartment, BoundaryType, ReactionType
        )

        # A -> (nothing) with rate k
        species = SmoldynSpecies(
            name="A",
            diffusion_coeff=1.0,
        )
        decay_rxn = SmoldynReaction(
            name="decay",
            reaction_type=ReactionType.FIRST,
            rate=1.0,  # 1/s
            reactants=["A"],
            products=[],
        )
        compartment = SmoldynCompartment(
            name="domain",
            bounds=(0, 10, 0, 10, 0, 10),
            boundary_type=BoundaryType.REFLECT,
        )
        system = SmoldynSystem(
            species=[species],
            reactions=[decay_rxn],
            compartment=compartment,
            name="decay_test",
        )

        solver = SmoldynSolver(
            system=system,
            n_max_particles=5000,
        )

        # Add initial particles
        positions = np.random.uniform(0, 10, (1000, 3)).astype(np.float32)
        solver.add_particles("A", positions)

        initial_count = solver.particles.n_alive

        # Run for 1 second (should decay to ~1000 * exp(-1) ~ 368)
        for _ in range(1000):
            solver.step(dt=0.001)

        final_count = solver.particles.n_alive

        # Expected: N(t) = N0 * exp(-kt) = 1000 * exp(-1) ≈ 368
        expected = 1000 * np.exp(-1.0)

        # Allow 50% tolerance for stochastic variation
        assert abs(final_count - expected) / expected < 0.50, \
            f"Final count {final_count} differs from expected {expected:.0f}"

    def test_bimolecular_binding(self):
        """Test second-order binding reaction."""
        from gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

        system = SmoldynSystem.simple_binding()
        solver = SmoldynSolver(
            system=system,
            n_max_particles=10000,
        )

        # Add initial particles (smaller domain region for more collisions)
        pos_a = np.random.uniform(40, 60, (500, 3)).astype(np.float32)
        pos_b = np.random.uniform(40, 60, (500, 3)).astype(np.float32)
        solver.add_particles("A", pos_a)
        solver.add_particles("B", pos_b)

        initial_c_count = solver.count_species("C")

        # Run simulation
        for _ in range(500):
            solver.step(dt=0.001)

        final_c_count = solver.count_species("C")

        # Product C should have formed (or stayed same if no collisions)
        assert final_c_count >= initial_c_count, \
            "Product C count should not decrease"


# ────────────────────────────────────────────────────────────────
# Test Smoldyn Module
# ────────────────────────────────────────────────────────────────

class TestSmoldynModule:
    """Tests for SmoldynModule wrapper."""

    def test_module_initialization(self):
        """Test module initialization."""
        from modules.smoldyn_module import SmoldynModule

        module = SmoldynModule({
            'system': 'simple_binding',
            'domain_size': (10.0, 10.0, 10.0),
            'max_particles': 5000,
            'species_counts': {'A': 200, 'B': 200},
        })
        module.initialize()

        state = module.get_state()
        assert state['initialized']
        assert state['n_particles'] > 0

    def test_module_update(self):
        """Test module update step."""
        from modules.smoldyn_module import SmoldynModule

        module = SmoldynModule({
            'system': 'simple_binding',
            'domain_size': (5.0, 5.0, 5.0),
            'max_particles': 10000,
            'species_counts': {'A': 100, 'B': 100},
        })
        module.initialize()

        module.update(dt=0.001)
        final_state = module.get_state()

        assert final_state['total_steps'] == 1
        assert final_state['current_time'] > 0

    def test_get_species_counts(self):
        """Test species count retrieval."""
        from modules.smoldyn_module import SmoldynModule

        module = SmoldynModule({
            'system': 'simple_binding',
            'domain_size': (10.0, 10.0, 10.0),
            'species_counts': {'A': 300, 'B': 200, 'C': 0},
        })
        module.initialize()

        counts = module.get_species_counts()
        assert "A" in counts
        assert "B" in counts
        assert counts["A"] == 300
        assert counts["B"] == 200

    def test_get_particle_positions(self):
        """Test position retrieval."""
        from modules.smoldyn_module import SmoldynModule

        module = SmoldynModule({
            'system': 'simple_binding',
            'domain_size': (10.0, 10.0, 10.0),
            'species_counts': {'A': 100, 'B': 100},
        })
        module.initialize()

        positions = module.get_particle_positions("A")
        assert positions.shape[1] == 3
        assert len(positions) > 0

    def test_add_particles(self):
        """Test particle addition."""
        from modules.smoldyn_module import SmoldynModule

        module = SmoldynModule({
            'system': 'simple_binding',
            'domain_size': (10.0, 10.0, 10.0),
            'species_counts': {'A': 100, 'B': 100},
        })
        module.initialize()

        initial_count = module.get_species_counts()["A"]
        module.add_particles("A", 50)
        final_count = module.get_species_counts()["A"]

        assert final_count == initial_count + 50

    def test_get_concentration(self):
        """Test concentration calculation."""
        from modules.smoldyn_module import SmoldynModule

        module = SmoldynModule({
            'system': 'simple_binding',
            'domain_size': (10.0, 10.0, 10.0),
            'species_counts': {'A': 1000, 'B': 0, 'C': 0},
        })
        module.initialize()

        conc = module.get_concentration("A")
        count = module.get_species_counts()["A"]
        volume = 10.0 * 10.0 * 10.0

        assert abs(conc - count / volume) < 1e-6


# ────────────────────────────────────────────────────────────────
# Test GPU Acceleration
# ────────────────────────────────────────────────────────────────

class TestGPUAcceleration:
    """Tests for GPU acceleration features."""

    def test_cupy_availability_check(self):
        """Test CuPy availability detection."""
        from gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

        system = SmoldynSystem.simple_binding()
        solver = SmoldynSolver(
            system=system,
            n_max_particles=10000,
        )

        # Add particles
        positions = np.random.uniform(0, 100, (500, 3)).astype(np.float32)
        solver.add_particles("A", positions)

        # Solver should work regardless of CuPy availability
        solver.step(dt=0.001)
        assert solver.particles.n_alive >= 0

    def test_large_scale_performance(self):
        """Test solver handles large particle counts."""
        from gpu.smoldyn_solver import (
            SmoldynSolver, SmoldynSystem, SmoldynSpecies,
            SmoldynCompartment, BoundaryType
        )

        # Large system
        species = SmoldynSpecies(
            name="A",
            diffusion_coeff=1.0,
        )
        compartment = SmoldynCompartment(
            name="large_domain",
            bounds=(0, 50, 0, 50, 0, 50),
            boundary_type=BoundaryType.REFLECT,
        )
        system = SmoldynSystem(
            species=[species],
            reactions=[],
            compartment=compartment,
            name="large_test",
        )

        solver = SmoldynSolver(
            system=system,
            n_max_particles=100000,
        )

        # Add 50k particles
        positions = np.random.uniform(0, 50, (50000, 3)).astype(np.float32)
        solver.add_particles("A", positions)

        # Should handle 50k particles
        assert solver.particles.n_alive == 50000

        # Should complete steps without error
        for _ in range(10):
            solver.step(dt=0.001)


# ────────────────────────────────────────────────────────────────
# Test Physics Registration
# ────────────────────────────────────────────────────────────────

class TestPhysicsRegistration:
    """Tests for PhysicsModel registration."""

    def test_cupy_smoldyn_registration(self):
        """Test CuPySmoldynPhysics is registered."""
        from gpu.physics_interface import get_physics_model

        try:
            model = get_physics_model("cupy_smoldyn")
            assert model is not None
        except (KeyError, ImportError):
            pytest.skip("cupy_smoldyn not registered")


# ────────────────────────────────────────────────────────────────
# Integration Tests
# ────────────────────────────────────────────────────────────────

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_enzyme_kinetics_simulation(self):
        """Test full enzyme kinetics simulation."""
        from modules.smoldyn_module import SmoldynModule

        module = SmoldynModule({
            'system': 'enzyme_kinetics',
            'domain_size': (5.0, 5.0, 5.0),
            'max_particles': 20000,
            'species_counts': {'E': 50, 'S': 500, 'ES': 0, 'P': 0},
        })
        module.initialize()

        # Run for 1000 steps
        for _ in range(1000):
            module.update(dt=0.001)

        state = module.get_state()

        # Simulation should have run
        assert state['total_steps'] == 1000

    def test_spatial_distribution(self):
        """Test spatial distribution analysis."""
        from modules.smoldyn_module import SmoldynModule

        module = SmoldynModule({
            'system': 'simple_binding',
            'domain_size': (10.0, 10.0, 10.0),
            'species_counts': {'A': 1000, 'B': 0, 'C': 0},
        })
        module.initialize()

        edges, counts = module.get_spatial_distribution("A", bins=10, axis=0)

        assert len(edges) == 11  # bins + 1
        assert len(counts) == 10
        assert np.sum(counts) == module.get_species_counts()["A"]


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
