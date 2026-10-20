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
        from cognisom.gpu.smoldyn_solver import SmoldynSpecies

        species = SmoldynSpecies(
            name="A",
            diffusion_coeff=1.0,
        )
        assert species.name == "A"
        assert species.diffusion_coeff == 1.0

    def test_species_defaults(self):
        """Test species default values."""
        from cognisom.gpu.smoldyn_solver import SmoldynSpecies

        species = SmoldynSpecies(name="B", diffusion_coeff=0.5)
        assert species.color == (1.0, 1.0, 1.0)
        assert species.radius == 0.01


class TestSmoldynReaction:
    """Tests for SmoldynReaction dataclass."""

    def test_unimolecular_reaction(self):
        """Test first-order reaction creation."""
        from cognisom.gpu.smoldyn_solver import SmoldynReaction, ReactionType

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
        from cognisom.gpu.smoldyn_solver import SmoldynReaction, ReactionType

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
        from cognisom.gpu.smoldyn_solver import SmoldynSystem

        system = SmoldynSystem.simple_binding()

        # Check species
        species_names = [s.name for s in system.species]
        assert "A" in species_names
        assert "B" in species_names
        assert "C" in species_names

        # Check reactions
        assert len(system.reactions) >= 1
        # Bare next() raising StopIteration inside a generator context surfaces
        # as an opaque "generator raised StopIteration"; assert on the lookup
        # instead so a renamed reaction names itself in the failure.
        binding_rxn = next(
            (r for r in system.reactions if r.name == "A_B_binding"), None
        )
        assert binding_rxn is not None, (
            f"no 'A_B_binding' reaction; found {[r.name for r in system.reactions]}"
        )
        assert binding_rxn.reactants == ["A", "B"]
        assert binding_rxn.products == ["C"]

    def test_enzyme_kinetics_system(self):
        """Test creation of enzyme kinetics system."""
        from cognisom.gpu.smoldyn_solver import SmoldynSystem

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
        from cognisom.gpu.smoldyn_solver import SmoldynSystem

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
        from cognisom.gpu.smoldyn_solver import ParticleSystem

        ps = ParticleSystem(n_max=10000, n_species=3)

        assert ps.n_max == 10000
        assert ps.positions.shape == (10000, 3)
        assert ps.species.shape == (10000,)
        assert ps.alive.shape == (10000,)
        assert ps.n_alive == 0

    def test_add_particles(self):
        """Test adding particles to system."""
        from cognisom.gpu.smoldyn_solver import ParticleSystem

        ps = ParticleSystem(n_max=1000, n_species=3)

        positions = np.random.uniform(0, 10, (100, 3)).astype(np.float32)
        ps.add_particles(species_idx=0, positions=positions)

        assert ps.n_alive == 100

    def test_add_multiple_species(self):
        """Test adding particles of different species."""
        from cognisom.gpu.smoldyn_solver import ParticleSystem

        ps = ParticleSystem(n_max=1000, n_species=3)

        pos_a = np.random.uniform(0, 10, (50, 3)).astype(np.float32)
        pos_b = np.random.uniform(0, 10, (30, 3)).astype(np.float32)

        ps.add_particles(species_idx=0, positions=pos_a)
        ps.add_particles(species_idx=1, positions=pos_b)

        assert ps.n_alive == 80

    def test_remove_particles(self):
        """Test removing particles."""
        from cognisom.gpu.smoldyn_solver import ParticleSystem

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
        from cognisom.gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

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
        from cognisom.gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

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
        from cognisom.gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

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
        from cognisom.gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

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
        from cognisom.gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

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
        from cognisom.gpu.smoldyn_solver import (
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
        from cognisom.gpu.smoldyn_solver import (
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
        from cognisom.gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

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
        from cognisom.modules.smoldyn_module import SmoldynModule

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
        from cognisom.modules.smoldyn_module import SmoldynModule

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
        from cognisom.modules.smoldyn_module import SmoldynModule

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
        from cognisom.modules.smoldyn_module import SmoldynModule

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
        from cognisom.modules.smoldyn_module import SmoldynModule

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
        from cognisom.modules.smoldyn_module import SmoldynModule

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
        from cognisom.gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

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
        from cognisom.gpu.smoldyn_solver import (
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
        from cognisom.gpu.physics_interface import get_physics_model

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
        from cognisom.modules.smoldyn_module import SmoldynModule

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
        from cognisom.modules.smoldyn_module import SmoldynModule

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
# Vectorised CPU Step
# ────────────────────────────────────────────────────────────────
#
# The CPU step is vectorised over the live particles and finds reaction
# partners with a neighbour search. It is stochastic, so these tests check it
# three ways: structurally against a brute-force O(n^2) reference written the
# slow obvious way, statistically against closed-form expectations (6*D*t for
# diffusion, the analytic pair count for binding), and by invariant
# (conservation, boundary containment, determinism under a fixed seed).


def _reference_step(solver, dt):
    """Brute-force O(n^2) reference for ``SmoldynSolver._step_cpu``.

    One Python loop per stage and a full pair scan, mutating the solver's
    arrays in place. It draws from ``np.random`` in the same order and the same
    quantity as the vectorised step, so with one seeding the two can be
    compared element by element.

    One deliberate difference from the pre-vectorisation code: a candidate pair
    is considered whichever way round the two particles' slot indices fall. The
    old scan only looked at ``j > i``, so an A sitting in a higher slot than its
    B partner could never react with it - the reaction rate depended on
    allocation order.
    """
    particles = solver.particles
    positions = particles.positions
    species = particles.species
    alive = particles.alive
    n = particles.n_slots
    b = solver._bounds
    axes = [(b['x_min'], b['x_max']), (b['y_min'], b['y_max']), (b['z_min'], b['z_max'])]

    # 1. Brownian displacement
    for i in range(n):
        if not alive[i]:
            continue
        sigma = np.sqrt(2 * solver._diffusion[species[i]] * dt)
        positions[i] += np.random.randn(3) * sigma

    # 2. Reflect off the walls
    for i in range(n):
        if not alive[i]:
            continue
        for dim, (lo, hi) in enumerate(axes):
            if positions[i, dim] < lo:
                positions[i, dim] = 2 * lo - positions[i, dim]
            if positions[i, dim] > hi:
                positions[i, dim] = 2 * hi - positions[i, dim]

    # 3. Unimolecular reactions
    for i in range(n):
        if not alive[i]:
            continue
        rate = solver._uni_rates[species[i]]
        if rate > 0:
            prob = 1 - np.exp(-rate * dt)
            if np.random.rand() < prob:
                product = solver._uni_products[species[i]]
                if product < 0:
                    alive[i] = 0
                else:
                    species[i] = product

    # 4. Bimolecular reactions
    for r in range(solver._n_bimol):
        sp1 = solver._bimol_reactant1[r]
        sp2 = solver._bimol_reactant2[r]
        bind_r2 = solver._bimol_bind_r[r] ** 2
        prod1 = solver._bimol_product1[r]
        prod2 = solver._bimol_product2[r]
        for i in range(n):
            if not alive[i] or species[i] != sp1:
                continue
            # A self-reaction (A + A -> B) must see each unordered pair once.
            partners = range(i + 1, n) if sp1 == sp2 else range(n)
            for j in partners:
                if j == i or not alive[j] or species[j] != sp2:
                    continue
                if np.sum((positions[i] - positions[j]) ** 2) >= bind_r2:
                    continue
                alive[i] = 0
                alive[j] = 0
                midpoint = (positions[i] + positions[j]) / 2
                if prod1 >= 0:
                    species[i] = prod1
                    positions[i] = midpoint
                    alive[i] = 1
                if prod2 >= 0:
                    species[j] = prod2
                    positions[j] = midpoint
                    alive[j] = 1
                break


def _twin(solver):
    """A second solver holding exactly the same particle state."""
    from cognisom.gpu.smoldyn_solver import SmoldynSolver

    twin = SmoldynSolver(solver.system, n_max_particles=solver.n_max)
    twin.particles.positions[...] = solver.particles.positions
    twin.particles.species[...] = solver.particles.species
    twin.particles.alive[...] = solver.particles.alive
    twin.particles._next_free = solver.particles.n_slots
    return twin


def _snapshot(solver):
    p = solver.particles
    n = p.n_slots
    return (p.positions[:n].copy(), p.species[:n].copy(), p.alive[:n].copy())


#: How far apart the vectorised step and the reference may round, in units
#: of float32 epsilon. Measured worst case on numpy 1.26 over four steps:
#: 1 ULP. Eight leaves room for longer runs and still sits far below any
#: real divergence.
ULP_TOLERANCE = 8


def _assert_same_state(a, b, label):
    pos_a, sp_a, al_a = a
    pos_b, sp_b, al_b = b
    assert np.array_equal(al_a, al_b), f"{label}: alive differs"
    assert np.array_equal(sp_a, sp_b), f"{label}: species differs"
    live = al_a == 1
    want, got = pos_b[live], pos_a[live]
    # Positions must agree to float32 round-off, not bit for bit. Bit
    # identity is not a property either path guarantees: on numpy 1.26 --
    # which is what CI runs, because TensorFlow pins numpy below 2 -- the
    # vectorised (n, 3) loop and the reference's per-row (3,) loop round
    # the same expression differently, measured at up to 1 ULP, while
    # numpy 2.x happens to agree. The assertion was exact, so CI failed on
    # five of these and printed two arrays identical to repr precision,
    # which said nothing.
    #
    # The tolerance does not weaken what the test is for. Everything that
    # carries the semantics -- which particles are alive, what species they
    # are -- is still compared exactly above, and any real divergence (a
    # different random draw, a missed reaction, a wrong pairing) moves a
    # particle by about one displacement, sigma ~ 0.1 um, four orders of
    # magnitude more than this bound.
    rtol = ULP_TOLERANCE * np.finfo(np.float32).eps
    atol = np.finfo(np.float32).eps
    allowed = rtol * np.abs(want.astype(np.float64)) + atol
    diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
    if (diff <= allowed).all():
        return
    # Say what differs, and by how much against what was allowed. Printing
    # the two arrays is useless here: they agree to repr precision, so a
    # failure reads as two identical arrays, which is how this went
    # undiagnosed in CI.
    worst = int(np.argmax(diff / allowed))
    row, col = divmod(worst, got.shape[1])
    raise AssertionError(
        f"{label}: positions differ beyond float32 round-off in "
        f"{int((diff > allowed).sum())} of {diff.size} components "
        f"(tolerance {ULP_TOLERANCE} ULP). Worst at live row {row} axis "
        f"{col}: vectorised {got[row, col]!r} vs reference "
        f"{want[row, col]!r}, |diff| {diff[row, col]:.3e} against an "
        f"allowance of {allowed[row, col]:.3e} "
        f"({diff[row, col] / allowed[row, col]:.1f}x)"
    )


def _binding_system(radius=0.2, domain=10.0, diffusion=1.0):
    from cognisom.gpu.smoldyn_solver import (
        BoundaryType, ReactionType, SmoldynCompartment, SmoldynReaction,
        SmoldynSpecies, SmoldynSystem,
    )

    return SmoldynSystem(
        species=[
            SmoldynSpecies("A", diffusion_coeff=diffusion),
            SmoldynSpecies("B", diffusion_coeff=diffusion),
            SmoldynSpecies("C", diffusion_coeff=diffusion),
        ],
        reactions=[
            SmoldynReaction(
                name="bind",
                reaction_type=ReactionType.SECOND,
                rate=1e6,
                reactants=["A", "B"],
                products=["C"],
                binding_radius=radius,
            ),
        ],
        compartment=SmoldynCompartment(
            name="box",
            bounds=(0, domain, 0, domain, 0, domain),
            boundary_type=BoundaryType.REFLECT,
        ),
        name="binding_test",
    )


class TestNeighborSearch:
    """The neighbour searches must find exactly the brute-force pairs."""

    @staticmethod
    def _brute(pos_a, pos_b, radius):
        d2 = ((pos_a[:, None, :] - pos_b[None, :, :]) ** 2).sum(-1)
        ia, ib = np.nonzero(d2 < radius * radius)
        return set(zip(ia.tolist(), ib.tolist()))

    @pytest.mark.parametrize("mode", ["kdtree", "cells"])
    @pytest.mark.parametrize(
        "n_a,n_b,domain,radius",
        [
            (300, 300, 100.0, 0.1),   # dilute: essentially no pairs
            (400, 400, 10.0, 0.3),    # a handful of pairs
            (600, 600, 5.0, 0.5),     # dense: many pairs, several per particle
            (200, 50, 4.0, 1.0),      # radius comparable to the domain
        ],
    )
    def test_matches_brute_force(self, mode, n_a, n_b, domain, radius):
        from cognisom.gpu.smoldyn_solver import SmoldynSolver

        solver = SmoldynSolver(
            _binding_system(radius=radius, domain=domain),
            n_max_particles=10,
            neighbor_search=mode,
        )
        rng = np.random.default_rng(11)
        pos_a = rng.uniform(0, domain, (n_a, 3)).astype(np.float32)
        pos_b = rng.uniform(0, domain, (n_b, 3)).astype(np.float32)

        ia, ib = solver._find_pairs(pos_a, pos_b, radius)
        found = set(zip(np.asarray(ia).tolist(), np.asarray(ib).tolist()))
        assert found == self._brute(pos_a, pos_b, radius)

    @pytest.mark.parametrize("mode", ["kdtree", "cells"])
    def test_same_set_pairs(self, mode):
        """A self-reaction searches one set against itself."""
        from cognisom.gpu.smoldyn_solver import SmoldynSolver

        radius, domain = 0.4, 8.0
        solver = SmoldynSolver(
            _binding_system(radius=radius, domain=domain),
            n_max_particles=10,
            neighbor_search=mode,
        )
        rng = np.random.default_rng(12)
        pos = rng.uniform(0, domain, (500, 3)).astype(np.float32)

        ia, ib = solver._find_pairs(pos, pos, radius)
        ia, ib = np.asarray(ia), np.asarray(ib)
        found = {(int(i), int(j)) for i, j in zip(ia, ib) if i < j}
        expected = {p for p in self._brute(pos, pos, radius) if p[0] < p[1]}
        assert found == expected
        assert found, "no pairs found, test is vacuous"


class TestVectorisedStepEquivalence:
    """The vectorised step must reproduce the brute-force reference exactly."""

    @staticmethod
    def _compare(solver, dt, n_steps, seed=4242):
        twin = _twin(solver)
        np.random.seed(seed)
        for step in range(n_steps):
            solver.step(dt)
        np.random.seed(seed)
        for step in range(n_steps):
            _reference_step(twin, dt)
        _assert_same_state(_snapshot(solver), _snapshot(twin), "vectorised vs reference")

    def test_diffusion_only(self):
        from cognisom.gpu.smoldyn_solver import (
            BoundaryType, SmoldynCompartment, SmoldynSolver, SmoldynSpecies,
            SmoldynSystem,
        )

        system = SmoldynSystem(
            species=[SmoldynSpecies("A", diffusion_coeff=5.0)],
            reactions=[],
            compartment=SmoldynCompartment(
                name="box", bounds=(0, 20, 0, 20, 0, 20),
                boundary_type=BoundaryType.REFLECT,
            ),
            name="diffusion",
        )
        solver = SmoldynSolver(system, n_max_particles=2000)
        rng = np.random.default_rng(3)
        solver.add_particles("A", rng.uniform(1, 19, (400, 3)).astype(np.float32))
        self._compare(solver, dt=0.001, n_steps=5)

    def test_unimolecular_decay(self):
        from cognisom.gpu.smoldyn_solver import SmoldynSolver, SmoldynSystem

        # enzyme_kinetics has first-order reactions on ES as well as a
        # bimolecular E + S step, so this exercises death, conversion and
        # binding in the same comparison.
        system = SmoldynSystem.enzyme_kinetics()
        solver = SmoldynSolver(system, n_max_particles=3000)
        rng = np.random.default_rng(5)
        solver.add_particles("E", rng.uniform(0, 50, (60, 3)).astype(np.float32))
        solver.add_particles("S", rng.uniform(0, 50, (200, 3)).astype(np.float32))
        solver.add_particles("ES", rng.uniform(0, 50, (120, 3)).astype(np.float32))
        self._compare(solver, dt=0.002, n_steps=4)

    def test_bimolecular_with_interleaved_slots(self):
        """A and B interleaved across slots, so slot order cannot help."""
        from cognisom.gpu.smoldyn_solver import SmoldynSolver

        solver = SmoldynSolver(_binding_system(radius=0.3, domain=4.0),
                               n_max_particles=4000)
        rng = np.random.default_rng(7)
        for _ in range(10):
            solver.add_particles("A", rng.uniform(0, 4, (15, 3)).astype(np.float32))
            solver.add_particles("B", rng.uniform(0, 4, (15, 3)).astype(np.float32))
        self._compare(solver, dt=0.001, n_steps=4)
        assert solver.count_species("C") > 0, "no binding happened, test is vacuous"

    def test_dense_binding_many_contested_pairs(self):
        """Dense enough that particles compete for partners every step."""
        from cognisom.gpu.smoldyn_solver import SmoldynSolver

        solver = SmoldynSolver(_binding_system(radius=0.5, domain=3.0),
                               n_max_particles=3000)
        rng = np.random.default_rng(9)
        solver.add_particles("A", rng.uniform(0, 3, (150, 3)).astype(np.float32))
        solver.add_particles("B", rng.uniform(0, 3, (150, 3)).astype(np.float32))
        self._compare(solver, dt=0.0005, n_steps=3)
        assert solver.count_species("C") > 30, "expected plenty of binding"

    def test_self_reaction(self):
        """A + A -> B, where both reactants come from one set."""
        from cognisom.gpu.smoldyn_solver import (
            BoundaryType, ReactionType, SmoldynCompartment, SmoldynReaction,
            SmoldynSolver, SmoldynSpecies, SmoldynSystem,
        )

        system = SmoldynSystem(
            species=[
                SmoldynSpecies("A", diffusion_coeff=1.0),
                SmoldynSpecies("B", diffusion_coeff=1.0),
            ],
            reactions=[
                SmoldynReaction(
                    name="dimerise",
                    reaction_type=ReactionType.SECOND,
                    rate=1e5,
                    reactants=["A", "A"],
                    products=["B"],
                    binding_radius=0.4,
                ),
            ],
            compartment=SmoldynCompartment(
                name="box", bounds=(0, 3, 0, 3, 0, 3),
                boundary_type=BoundaryType.REFLECT,
            ),
            name="dimerisation",
        )
        solver = SmoldynSolver(system, n_max_particles=2000)
        rng = np.random.default_rng(13)
        solver.add_particles("A", rng.uniform(0, 3, (200, 3)).astype(np.float32))
        self._compare(solver, dt=0.0005, n_steps=3)
        assert solver.count_species("B") > 20, "expected dimerisation to fire"

    def test_cell_list_and_kdtree_agree(self):
        """The two neighbour searches drive the solver identically."""
        from cognisom.gpu.smoldyn_solver import SmoldynSolver

        solver = SmoldynSolver(_binding_system(radius=0.4, domain=6.0),
                               n_max_particles=3000, neighbor_search="kdtree")
        rng = np.random.default_rng(17)
        solver.add_particles("A", rng.uniform(0, 6, (500, 3)).astype(np.float32))
        solver.add_particles("B", rng.uniform(0, 6, (500, 3)).astype(np.float32))

        twin = _twin(solver)
        twin.neighbor_search = "cells"

        np.random.seed(99)
        for _ in range(10):
            solver.step(0.001)
        np.random.seed(99)
        for _ in range(10):
            twin.step(0.001)

        _assert_same_state(_snapshot(solver), _snapshot(twin), "kdtree vs cells")
        assert solver.count_species("C") > 0


class TestStepInvariants:
    """Properties that must hold whatever the random draws are."""

    def test_seeded_determinism(self):
        from cognisom.gpu.smoldyn_solver import SmoldynSolver

        states, c_counts = [], []
        for _ in range(2):
            np.random.seed(2024)
            solver = SmoldynSolver(_binding_system(radius=0.4, domain=6.0),
                                   n_max_particles=5000)
            solver.add_particles(
                "A", np.random.uniform(0, 6, (400, 3)).astype(np.float32))
            solver.add_particles(
                "B", np.random.uniform(0, 6, (400, 3)).astype(np.float32))
            for _ in range(25):
                solver.step(0.001)
            states.append(_snapshot(solver))
            c_counts.append(solver.count_species("C"))
        _assert_same_state(states[0], states[1], "run 1 vs run 2")
        assert c_counts[0] == c_counts[1]
        assert c_counts[0] > 0, "no C formed, test is vacuous"

    def test_dead_slots_do_not_change_the_trajectory(self):
        """Spare capacity must not be stepped, and must not consume randoms."""
        from cognisom.gpu.smoldyn_solver import SmoldynSolver

        states = []
        for n_max in (1000, 100000):
            solver = SmoldynSolver(_binding_system(radius=0.4, domain=6.0),
                                   n_max_particles=n_max)
            rng = np.random.default_rng(21)
            solver.add_particles("A", rng.uniform(0, 6, (400, 3)).astype(np.float32))
            solver.add_particles("B", rng.uniform(0, 6, (400, 3)).astype(np.float32))
            np.random.seed(77)
            for _ in range(20):
                solver.step(0.001)
            states.append(_snapshot(solver))
        _assert_same_state(states[0], states[1], "n_max=1000 vs n_max=100000")

    def test_conservation_across_bimolecular_reactions(self):
        """A + B -> C destroys one A and one B per C, and nothing else."""
        from cognisom.gpu.smoldyn_solver import SmoldynSolver

        solver = SmoldynSolver(_binding_system(radius=0.4, domain=5.0),
                               n_max_particles=5000)
        rng = np.random.default_rng(23)
        n_a, n_b = 700, 500
        solver.add_particles("A", rng.uniform(0, 5, (n_a, 3)).astype(np.float32))
        solver.add_particles("B", rng.uniform(0, 5, (n_b, 3)).astype(np.float32))

        np.random.seed(31)
        for _ in range(40):
            solver.step(0.001)
            a = solver.count_species("A")
            b = solver.count_species("B")
            c = solver.count_species("C")
            assert a + c == n_a, f"A + C = {a + c}, expected {n_a}"
            assert b + c == n_b, f"B + C = {b + c}, expected {n_b}"
            assert solver.particles.n_alive == a + b + c

        assert solver.count_species("C") > 0, "no reactions fired, test is vacuous"

    def test_unimolecular_conversion_conserves_count(self):
        """A -> B conversions never create or destroy a particle."""
        from cognisom.gpu.smoldyn_solver import (
            BoundaryType, ReactionType, SmoldynCompartment, SmoldynReaction,
            SmoldynSolver, SmoldynSpecies, SmoldynSystem,
        )

        system = SmoldynSystem(
            species=[
                SmoldynSpecies("A", diffusion_coeff=1.0),
                SmoldynSpecies("B", diffusion_coeff=1.0),
            ],
            reactions=[
                SmoldynReaction(
                    name="convert",
                    reaction_type=ReactionType.FIRST,
                    rate=20.0,
                    reactants=["A"],
                    products=["B"],
                ),
            ],
            compartment=SmoldynCompartment(
                name="box", bounds=(0, 10, 0, 10, 0, 10),
                boundary_type=BoundaryType.REFLECT,
            ),
            name="conversion",
        )
        solver = SmoldynSolver(system, n_max_particles=3000)
        rng = np.random.default_rng(29)
        solver.add_particles("A", rng.uniform(0, 10, (1000, 3)).astype(np.float32))

        for _ in range(50):
            solver.step(0.001)
            assert solver.count_species("A") + solver.count_species("B") == 1000
            assert solver.particles.n_alive == 1000

        # ~1 - exp(-20 * 0.05) = 63% converted
        assert 400 < solver.count_species("B") < 850

    @pytest.mark.parametrize("dt", [0.001, 1.0, 100.0])
    def test_no_particle_escapes_the_boundary(self, dt):
        """Even a step far larger than the domain leaves everyone inside."""
        from cognisom.gpu.smoldyn_solver import (
            BoundaryType, SmoldynCompartment, SmoldynSolver, SmoldynSpecies,
            SmoldynSystem,
        )

        lo = np.array([2.0, -5.0, 0.0])
        hi = np.array([7.0, -1.0, 3.0])
        system = SmoldynSystem(
            species=[SmoldynSpecies("A", diffusion_coeff=50.0)],
            reactions=[],
            compartment=SmoldynCompartment(
                name="offset_box",
                bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]),
                boundary_type=BoundaryType.REFLECT,
            ),
            name="boundary",
        )
        solver = SmoldynSolver(system, n_max_particles=3000)
        rng = np.random.default_rng(37)
        start = rng.uniform(lo, hi, (1000, 3)).astype(np.float32)
        solver.add_particles("A", start)

        np.random.seed(41)
        for _ in range(20):
            solver.step(dt)
            pos = solver.get_positions("A")
            assert len(pos) == 1000
            assert np.all(pos >= lo), f"escaped below: {pos.min(axis=0)}"
            assert np.all(pos <= hi), f"escaped above: {pos.max(axis=0)}"

    def test_mean_squared_displacement_unbounded(self):
        """MSD must track 6*D*t while the walls are out of reach."""
        from cognisom.gpu.smoldyn_solver import (
            BoundaryType, SmoldynCompartment, SmoldynSolver, SmoldynSpecies,
            SmoldynSystem,
        )

        D = 2.0
        system = SmoldynSystem(
            species=[SmoldynSpecies("A", diffusion_coeff=D)],
            reactions=[],
            compartment=SmoldynCompartment(
                name="huge", bounds=(0, 10000, 0, 10000, 0, 10000),
                boundary_type=BoundaryType.REFLECT,
            ),
            name="msd",
        )
        solver = SmoldynSolver(system, n_max_particles=20000)
        n = 20000
        start = np.full((n, 3), 5000.0, dtype=np.float32)
        solver.add_particles("A", start)

        np.random.seed(43)
        dt, n_steps = 0.001, 200
        for _ in range(n_steps):
            solver.step(dt)

        t = dt * n_steps
        disp = solver.get_positions("A") - start
        msd = float(np.mean(np.sum(disp.astype(np.float64) ** 2, axis=1)))
        expected = 6.0 * D * t
        # 3n independent Gaussians, so the relative standard error on the MSD
        # is sqrt(2 / (3n)) ~ 0.6%. 5% is ~8 sigma.
        assert abs(msd - expected) / expected < 0.05, \
            f"MSD {msd:.4f} vs expected {expected:.4f}"

        # Per-axis variance = 2*D*t, and the mean displacement stays at zero.
        per_axis = np.var(disp.astype(np.float64), axis=0)
        assert np.all(np.abs(per_axis - 2.0 * D * t) / (2.0 * D * t) < 0.05)
        assert np.all(np.abs(np.mean(disp, axis=0)) < 5.0 * np.sqrt(2 * D * t / n))

    def test_bimolecular_rate_matches_analytic_pair_count(self):
        """Reactions per step must match the expected number of close pairs.

        For uniformly distributed reactants the expected number of A-B pairs
        within the binding radius is N_A * N_B * (4/3 pi R^3) / V. In this
        dilute regime nearly every such pair is a distinct reaction, so the
        first step from a uniform configuration should fire that many times.
        This is the check that catches a neighbour search which silently misses
        partners - including the old scan's dependence on slot order, which
        halves the rate once the species are interleaved.
        """
        from cognisom.gpu.smoldyn_solver import SmoldynSolver

        radius, domain = 0.2, 10.0
        n_a = n_b = 1000
        n_trials = 30

        expected_per_trial = (
            n_a * n_b * (4.0 / 3.0 * np.pi * radius ** 3) / domain ** 3
        )

        rng = np.random.default_rng(47)
        fired = []
        for _ in range(n_trials):
            solver = SmoldynSolver(_binding_system(radius=radius, domain=domain),
                                   n_max_particles=2 * (n_a + n_b))
            # Interleave the species across slots: with the old index-ordered
            # scan this alone would cost half the reactions.
            for _ in range(10):
                solver.add_particles(
                    "A", rng.uniform(0, domain, (n_a // 10, 3)).astype(np.float32))
                solver.add_particles(
                    "B", rng.uniform(0, domain, (n_b // 10, 3)).astype(np.float32))
            solver.step(1e-9)  # displacement negligible; measure the pair count
            fired.append(solver.count_species("C"))

        mean_fired = float(np.mean(fired))
        # Greedy matching drops the odd pair that shares a particle, and the
        # box walls clip a little neighbourhood volume, so the measured rate
        # sits a few percent under the analytic value. The standard error over
        # 30 trials is ~3%, so 20% is a wide but still diagnostic band.
        assert abs(mean_fired - expected_per_trial) / expected_per_trial < 0.20, \
            f"{mean_fired:.1f} reactions/step vs analytic {expected_per_trial:.1f}"

    def test_product_sits_between_its_reactants(self):
        """The C from A + B -> C appears at the pair midpoint, inside the box."""
        from cognisom.gpu.smoldyn_solver import SmoldynSolver

        radius, domain = 0.5, 4.0
        solver = SmoldynSolver(_binding_system(radius=radius, domain=domain),
                               n_max_particles=2000)
        rng = np.random.default_rng(53)
        solver.add_particles("A", rng.uniform(0, domain, (300, 3)).astype(np.float32))
        solver.add_particles("B", rng.uniform(0, domain, (300, 3)).astype(np.float32))

        before = {name: solver.get_positions(name).copy() for name in ("A", "B")}
        np.random.seed(59)
        solver.step(1e-9)
        products = solver.get_positions("C")
        assert len(products) > 0

        # Every product must be within radius/2 of both an A and a B position
        # from before the step.
        for p in products:
            d_a = np.min(np.linalg.norm(before["A"] - p, axis=1))
            d_b = np.min(np.linalg.norm(before["B"] - p, axis=1))
            assert d_a <= radius / 2 + 1e-3, f"product {p} has no A nearby ({d_a})"
            assert d_b <= radius / 2 + 1e-3, f"product {p} has no B nearby ({d_b})"
        assert np.all(products >= 0) and np.all(products <= domain)


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
