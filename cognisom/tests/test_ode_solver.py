"""
Unit Tests for ODE Solver
=========================

Tests for the GPU-accelerated ODE solver (VCell Parity Phase 1).
"""

import numpy as np
import pytest
from unittest.mock import patch


class TestODESystem:
    """Tests for ODESystem dataclass."""

    def test_gene_expression_2species(self):
        """Test 2-species gene expression model creation."""
        from cognisom.gpu.ode_solver import ODESystem

        system = ODESystem.gene_expression_2species()

        assert system.n_species == 2
        assert system.species_names == ['mRNA', 'Protein']
        assert 'k_prod' in system.parameters
        assert 'k_deg' in system.parameters
        assert system.rhs_func is not None

    def test_ar_signaling_pathway(self):
        """Test AR signaling pathway model creation."""
        from cognisom.gpu.ode_solver import ODESystem

        system = ODESystem.ar_signaling_pathway()

        assert system.n_species == 6
        assert 'AR_mRNA' in system.species_names
        assert 'PSA' in system.species_names
        assert system.stiff is True
        assert 'k_bind' in system.parameters

    def test_rhs_function(self):
        """Test RHS function evaluation."""
        from cognisom.gpu.ode_solver import ODESystem

        system = ODESystem.gene_expression_2species()

        # Single cell
        y = np.array([[10.0, 100.0]])  # (1, 2)
        dydt = system.rhs_func(0, y, system.parameters)

        assert dydt.shape == (1, 2)
        # dmRNA = k_prod - k_deg * mRNA = 10 - 1*10 = 0
        assert np.isclose(dydt[0, 0], 0.0, atol=1e-6)
        # dProtein = k_trans * mRNA - k_deg_prot * Protein = 50*10 - 0.5*100 = 450
        assert np.isclose(dydt[0, 1], 450.0, atol=1e-6)

    def test_rhs_batched(self):
        """Test RHS function with multiple cells."""
        from cognisom.gpu.ode_solver import ODESystem

        system = ODESystem.gene_expression_2species()

        # 100 cells with varying states
        n_cells = 100
        y = np.random.rand(n_cells, 2) * 100
        dydt = system.rhs_func(0, y, system.parameters)

        assert dydt.shape == (n_cells, 2)
        # Check production term is always present
        assert np.all(dydt[:, 0] <= system.parameters['k_prod'])


class TestBatchedODEIntegrator:
    """Tests for BatchedODEIntegrator."""

    def test_creation(self):
        """Test integrator creation."""
        from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem

        system = ODESystem.gene_expression_2species()
        integrator = BatchedODEIntegrator(system, n_cells=1000)

        assert integrator.n_cells == 1000
        assert integrator.method == 'bdf'
        assert integrator.system == system

    def test_rk45_integration(self):
        """Test explicit RK45 integration."""
        from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem

        system = ODESystem.gene_expression_2species()
        integrator = BatchedODEIntegrator(system, n_cells=100, method='rk45')

        y0 = np.ones((100, 2)) * 10
        solution = integrator.integrate(t_span=(0, 1), y0=y0)

        assert solution.success
        assert len(solution.t) > 1
        assert solution.y.shape[0] == len(solution.t)
        assert solution.y.shape[1] == 100
        assert solution.y.shape[2] == 2

    def test_bdf_integration(self):
        """Test implicit BDF integration."""
        from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem

        system = ODESystem.gene_expression_2species()
        integrator = BatchedODEIntegrator(system, n_cells=50, method='bdf')

        y0 = np.ones((50, 2)) * 10
        solution = integrator.integrate(t_span=(0, 0.5), y0=y0)

        assert solution.success
        assert solution.n_steps > 0
        assert solution.n_rhs_evals > 0

    def test_steady_state_convergence(self):
        """Test that gene expression reaches steady state."""
        from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem

        system = ODESystem.gene_expression_2species()
        integrator = BatchedODEIntegrator(system, n_cells=10, method='rk45')

        # Start from zero
        y0 = np.zeros((10, 2))
        solution = integrator.integrate(t_span=(0, 50), y0=y0)

        # Check steady state approached
        final_mrna = solution.y[-1, :, 0].mean()
        final_protein = solution.y[-1, :, 1].mean()

        # Analytical: mRNA_ss = k_prod/k_deg = 10/1 = 10
        # Protein_ss = k_trans * mRNA_ss / k_deg_prot = 50*10/0.5 = 1000
        assert np.isclose(final_mrna, 10.0, rtol=0.1)
        assert np.isclose(final_protein, 1000.0, rtol=0.1)

    def test_cell_heterogeneity(self):
        """Test that heterogeneous parameters produce varied results."""
        from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem

        system = ODESystem.gene_expression_2species()
        integrator = BatchedODEIntegrator(system, n_cells=100, method='rk45')

        # Set heterogeneous parameters
        base_params = np.array(list(system.parameters.values()), dtype=np.float32)
        noise = np.random.lognormal(0, 0.3, (100, len(base_params)))
        cell_params = base_params * noise.astype(np.float32)
        integrator.set_cell_parameters(cell_params)

        y0 = np.ones((100, 2)) * 10
        solution = integrator.integrate(t_span=(0, 10), y0=y0)

        # Final states should have variance from parameter heterogeneity
        final_protein = solution.y[-1, :, 1]
        cv = final_protein.std() / final_protein.mean()

        assert cv > 0.1  # Should see >10% CV from parameter noise

    def test_step_function(self):
        """Test single-step integration."""
        from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem, ODEState

        system = ODESystem.gene_expression_2species()
        integrator = BatchedODEIntegrator(system, n_cells=10, method='rk45')

        y0 = np.ones((10, 2)) * 10
        integrator._state = ODEState(y=y0, t=0.0, dt=0.01, order=1)

        initial_y = integrator.get_state().copy()
        integrator.step(0.1)

        assert integrator._state.t == 0.1
        # State should have changed
        assert not np.allclose(integrator.get_state(), initial_y)

    def test_get_species(self):
        """Test species extraction by name."""
        from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem, ODEState

        system = ODESystem.gene_expression_2species()
        integrator = BatchedODEIntegrator(system, n_cells=50, method='rk45')

        y0 = np.random.rand(50, 2) * 100
        integrator._state = ODEState(y=y0.astype(np.float32), t=0.0, dt=0.01, order=1)

        mrna = integrator.get_species('mRNA')
        protein = integrator.get_species('Protein')

        assert mrna.shape == (50,)
        assert protein.shape == (50,)
        np.testing.assert_array_equal(mrna, y0[:, 0])
        np.testing.assert_array_equal(protein, y0[:, 1])


class TestCuPyAdaptiveODESolver:
    """Tests for registered physics model."""

    def test_registration(self):
        """Test that solver is registered in physics registry."""
        from cognisom.gpu.physics_interface import physics_registry

        assert 'cupy_ode_adaptive' in physics_registry.list_names()

    def test_protocol_compliance(self):
        """Test that solver implements PhysicsModel protocol."""
        from cognisom.gpu.ode_solver import CuPyAdaptiveODESolver
        from cognisom.gpu.physics_interface import (
            PhysicsModel,
            PhysicsBackendType,
            PhysicsModelType,
        )

        solver = CuPyAdaptiveODESolver()

        assert hasattr(solver, 'backend_type')
        assert hasattr(solver, 'model_type')
        assert hasattr(solver, 'initialize')
        assert hasattr(solver, 'step')

        assert solver.backend_type == PhysicsBackendType.CUPY
        assert solver.model_type == PhysicsModelType.ODE

    def test_initialize_and_step(self):
        """Test full initialization and stepping workflow."""
        from cognisom.gpu.ode_solver import CuPyAdaptiveODESolver
        from cognisom.gpu.physics_interface import PhysicsState

        solver = CuPyAdaptiveODESolver()
        state = PhysicsState()

        solver.initialize(state, system='gene_expression', n_cells=100)

        assert solver._initialized
        assert solver._integrator is not None

        # Step
        state = solver.step(0.1, state)

        assert 'ode_y' in state.custom
        assert state.custom['ode_y'].shape == (100, 2)


class TestODEModule:
    """Tests for ODEModule SimulationModule wrapper."""

    def test_initialization(self):
        """Test module initialization."""
        from cognisom.modules.ode_module import ODEModule

        module = ODEModule({'n_cells': 100, 'system': 'gene_expression'})
        module.initialize()

        assert module.integrator is not None
        assert module.n_cells == 100

    def test_update(self):
        """Test module update step."""
        from cognisom.modules.ode_module import ODEModule

        module = ODEModule({'n_cells': 50})
        module.initialize()

        initial_time = module.current_time
        module.update(0.1)

        assert module.current_time > initial_time
        assert module.total_steps > 0

    def test_get_state(self):
        """Test state retrieval."""
        from cognisom.modules.ode_module import ODEModule

        module = ODEModule({'n_cells': 100})
        module.initialize()
        module.update(0.1)

        state = module.get_state()

        assert state['initialized']
        assert state['n_cells'] == 100
        assert 'mRNA_mean' in state
        assert 'Protein_mean' in state

    def test_get_species(self):
        """Test species value retrieval."""
        from cognisom.modules.ode_module import ODEModule

        module = ODEModule({'n_cells': 100, 'system': 'gene_expression'})
        module.initialize()

        mrna = module.get_species('mRNA')
        protein = module.get_species('Protein')

        assert mrna.shape == (100,)
        assert protein.shape == (100,)

    def test_heterogeneity_config(self):
        """Test parameter heterogeneity configuration."""
        from cognisom.modules.ode_module import ODEModule

        module = ODEModule({
            'n_cells': 1000,
            'heterogeneity': 0.3,  # 30% CV
        })
        module.initialize()

        # Check that cell parameters have variance
        params = module.cell_params
        cv_per_param = params.std(axis=0) / params.mean(axis=0)

        # Should see ~30% CV (with some variation)
        assert np.all(cv_per_param > 0.1)
        assert np.all(cv_per_param < 0.6)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_ode_solver_string(self):
        """Test creating solver from string system name."""
        from cognisom.gpu.ode_solver import create_ode_solver

        solver = create_ode_solver('gene_expression', n_cells=100)

        assert solver.n_cells == 100
        assert solver.system.n_species == 2

    def test_create_ode_solver_ar(self):
        """Test creating AR signaling solver."""
        from cognisom.gpu.ode_solver import create_ode_solver

        solver = create_ode_solver('ar_signaling', n_cells=50, method='bdf')

        assert solver.n_cells == 50
        assert solver.method == 'bdf'
        assert solver.system.n_species == 6


class TestNumericalAccuracy:
    """Tests for numerical accuracy of ODE solutions."""

    def test_conservation(self):
        """Test mass conservation in closed system."""
        from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem

        # For gene expression: total mass not conserved, but we can check bounds
        system = ODESystem.gene_expression_2species()
        integrator = BatchedODEIntegrator(system, n_cells=10, method='rk45')

        y0 = np.ones((10, 2)) * 10
        solution = integrator.integrate(t_span=(0, 10), y0=y0)

        # All values should be non-negative
        assert np.all(solution.y >= -1e-6)  # Allow small numerical error

    def test_exponential_decay(self):
        """Test solver on simple exponential decay (exact solution known)."""
        from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem

        # Create a simple decay system: dy/dt = -k*y
        def decay_rhs(t, y, params):
            return -params['k'] * y

        system = ODESystem(
            n_species=1,
            species_names=['y'],
            rhs_func=decay_rhs,
            parameters={'k': 1.0},
            stiff=False,
        )

        integrator = BatchedODEIntegrator(system, n_cells=10, method='rk45', rtol=1e-6)

        y0 = np.ones((10, 1)) * 100
        solution = integrator.integrate(t_span=(0, 5), y0=y0)

        # Analytical solution: y(t) = y0 * exp(-k*t)
        t_final = solution.t[-1]
        y_analytical = 100 * np.exp(-1.0 * t_final)
        y_numerical = solution.y[-1, :, 0].mean()

        assert np.isclose(y_numerical, y_analytical, rtol=0.01)


# Skip GPU-specific tests if no GPU available
def gpu_available():
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


@pytest.mark.skipif(not gpu_available(), reason="GPU not available")
class TestGPUAcceleration:
    """Tests that require GPU."""

    def test_gpu_vs_cpu_parity(self):
        """Test that GPU and CPU produce similar results."""
        from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem
        from cognisom.gpu.backend import set_gpu_enabled

        system = ODESystem.gene_expression_2species()
        y0 = np.random.rand(100, 2) * 100

        # GPU run
        set_gpu_enabled(True)
        integrator_gpu = BatchedODEIntegrator(system, n_cells=100, method='rk45')
        solution_gpu = integrator_gpu.integrate(t_span=(0, 1), y0=y0.copy())

        # CPU run
        set_gpu_enabled(False)
        integrator_cpu = BatchedODEIntegrator(system, n_cells=100, method='rk45')
        solution_cpu = integrator_cpu.integrate(t_span=(0, 1), y0=y0.copy())

        # Re-enable GPU
        set_gpu_enabled(True)

        # Results should be close
        np.testing.assert_allclose(
            solution_gpu.y[-1],
            solution_cpu.y[-1],
            rtol=0.01,
            atol=1e-6
        )

    def test_gpu_kernel_compilation(self):
        """Test that GPU kernels compile successfully."""
        from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem

        system = ODESystem.gene_expression_2species()
        integrator = BatchedODEIntegrator(system, n_cells=100, method='bdf')

        # BDF method should compile CUDA kernels
        if integrator._backend.has_gpu:
            assert integrator._rhs_kernel is not None or integrator._newton_kernel is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
