"""
Regression tests for correctness fixes
======================================

Each test here pins a specific defect that previously produced silently wrong
results. See docs/review/2026-08-06_SIMULATION_QUALITY_AND_BUSINESS_REVIEW.md.
"""

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════
# Digital twin — per-patient RNG
# ═══════════════════════════════════════════════════════════════════════

class TestTreatmentSimulatorRNG:
    """The tumour trajectory must depend on the patient, not just the drug."""

    @staticmethod
    def _twin(patient_id: str, tmb: float):
        from cognisom.genomics.twin_config import DigitalTwinConfig
        from cognisom.genomics.patient_profile import PatientProfile

        return DigitalTwinConfig(
            patient=PatientProfile(patient_id=patient_id),
            tumor_mutational_burden=tmb,
            immune_score="hot",
            mean_exhaustion=0.4,
        )

    def test_different_patients_get_different_trajectories(self):
        """Seeding on the drug name alone made every patient identical."""
        from cognisom.genomics.treatment_simulator import TreatmentSimulator

        sim = TreatmentSimulator()
        a = sim.simulate("pembrolizumab", self._twin("PT-A", 12.0), duration_days=120)
        b = sim.simulate("pembrolizumab", self._twin("PT-B", 3.0), duration_days=120)

        assert a.tumor_response_curve != b.tumor_response_curve

    def test_same_patient_reproduces_exactly(self):
        from cognisom.genomics.treatment_simulator import TreatmentSimulator

        sim = TreatmentSimulator()
        first = sim.simulate("pembrolizumab", self._twin("PT-A", 12.0), duration_days=120)
        second = sim.simulate("pembrolizumab", self._twin("PT-A", 12.0), duration_days=120)

        assert first.tumor_response_curve == second.tumor_response_curve

    def test_does_not_disturb_global_numpy_rng(self):
        """The simulator used np.random.seed(), clobbering caller RNG state."""
        from cognisom.genomics.treatment_simulator import TreatmentSimulator

        np.random.seed(1234)
        expected = np.random.random(5).tolist()

        np.random.seed(1234)
        TreatmentSimulator().simulate(
            "pembrolizumab", self._twin("PT-A", 12.0), duration_days=60
        )
        after = np.random.random(5).tolist()

        assert after == expected


# ═══════════════════════════════════════════════════════════════════════
# Spatial diffusion — sub-cycling
# ═══════════════════════════════════════════════════════════════════════

class TestDiffusionSubCycling:
    """Diffusion must advance the full requested dt, not one stable step."""

    @staticmethod
    def _grid():
        from cognisom.engine.py.spatial.grid import SpatialGrid, GridConfig
        return SpatialGrid(GridConfig(size=(12, 12, 12), resolution=10.0))

    def test_large_dt_diffuses_further_than_small_dt(self):
        """Previously both clamped to the same single sub-step, so a 100x longer
        interval produced an identical field."""
        grid = self._grid()

        field = np.zeros((12, 12, 12), dtype=float)
        field[6, 6, 6] = 1000.0

        short = grid.diffuse_field(field.copy(), D=2000.0, dt=1e-6)
        long = grid.diffuse_field(field.copy(), D=2000.0, dt=1e-4)

        # More elapsed time => more spreading => lower peak at the source voxel.
        assert long[6, 6, 6] < short[6, 6, 6]

    def test_mass_is_conserved(self):
        """No-flux boundaries must conserve total mass exactly (to round-off).

        The old np.roll + boundary-copy scheme *created* mass: a 1000-unit point
        source grew to ~1634 units over a single 1e-4 h call.
        """
        grid = self._grid()

        field = np.zeros((12, 12, 12), dtype=float)
        field[6, 6, 6] = 1000.0
        before = field.sum()

        after = grid.diffuse_field(field, D=2000.0, dt=1e-4).sum()

        assert after == pytest.approx(before, rel=1e-9)

    def test_mass_conserved_over_many_substeps(self):
        """Conservation must hold when the step is sub-cycled many times."""
        grid = self._grid()

        field = np.zeros((12, 12, 12), dtype=float)
        field[3, 4, 5] = 500.0
        field[8, 7, 6] = 250.0
        before = field.sum()

        after = grid.diffuse_field(field, D=2000.0, dt=0.01).sum()

        assert after == pytest.approx(before, rel=1e-6)

    def test_remains_bounded_and_finite(self):
        """Sub-steps must each satisfy the stability criterion (no blow-up)."""
        grid = self._grid()

        field = np.zeros((12, 12, 12), dtype=float)
        field[6, 6, 6] = 1000.0

        out = grid.diffuse_field(field, D=2000.0, dt=0.01)

        assert np.all(np.isfinite(out))
        assert out.max() <= 1000.0 + 1e-6
        assert out.min() >= -1e-6


# ═══════════════════════════════════════════════════════════════════════
# Intracellular — ribosome pool
# ═══════════════════════════════════════════════════════════════════════

class TestRibosomePool:
    """Ribosome availability must be a real constraint, not a dead flag."""

    def test_ribosome_pool_limits_translation(self):
        from cognisom.engine.py.intracellular import IntracellularModel, mRNA

        model = IntracellularModel()
        model.num_ribosomes = 0
        model.mrnas["GAPDH"] = mRNA(gene_name="GAPDH", sequence_length=1200,
                                    copy_number=500)

        assert model.translate("GAPDH", dt=1.0) == 0

    def test_pool_is_shared_across_mrnas(self):
        from cognisom.engine.py.intracellular import IntracellularModel, mRNA

        model = IntracellularModel()
        model.num_ribosomes = 10
        model._release_ribosomes()

        for name in ("GAPDH", "ACTB"):
            model.mrnas[name] = mRNA(gene_name=name, sequence_length=1000,
                                     copy_number=100)

        model.translate("GAPDH", dt=0.01)
        first_busy = model._ribosomes_busy
        model.translate("ACTB", dt=0.01)

        # Second mRNA cannot exceed what the first left behind.
        assert first_busy == 10
        assert model._ribosomes_busy <= model.num_ribosomes

    def test_pool_resets_each_step(self):
        from cognisom.engine.py.intracellular import IntracellularModel

        model = IntracellularModel()
        model.step(dt=0.01)
        busy_after_first = model._ribosomes_busy

        model.step(dt=0.01)

        assert busy_after_first <= model.num_ribosomes
        assert model._ribosomes_busy <= model.num_ribosomes


# ═══════════════════════════════════════════════════════════════════════
# ODE solver — GPU BDF gating
# ═══════════════════════════════════════════════════════════════════════

class TestGPUBDFGating:
    """The 2-species-only GPU kernels must never silently no-op a larger system."""

    def test_multispecies_system_is_not_gpu_bdf_compatible(self):
        from cognisom.gpu.ode_solver import ODESystem

        ar = ODESystem.ar_signaling_pathway()

        assert ar.n_species == 6
        assert ar.gpu_bdf_compatible is False

    def test_builtin_two_species_system_opts_in(self):
        from cognisom.gpu.ode_solver import ODESystem

        gene = ODESystem.gene_expression_2species()

        assert gene.n_species == 2
        assert gene.gpu_bdf_compatible is True

    def test_ar_pathway_actually_integrates_with_default_bdf(self):
        """The 6-species AR system previously returned success with an unchanged
        state on the GPU BDF path. It must now integrate for real."""
        from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem

        system = ODESystem.ar_signaling_pathway()
        solver = BatchedODEIntegrator(system, n_cells=4, method='bdf')

        y0 = np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), (4, 1))
        solution = solver.integrate(t_span=(0.0, 0.5), y0=y0)

        final = np.asarray(solution.y_final if hasattr(solution, "y_final")
                           else solution.y)
        assert np.all(np.isfinite(final))
        assert not np.allclose(final, y0), "state did not change — solver no-opped"

    def test_gpu_bdf_gate_rejects_incompatible_system(self):
        from cognisom.gpu.ode_solver import BatchedODEIntegrator, ODESystem

        solver = BatchedODEIntegrator(
            ODESystem.ar_signaling_pathway(), n_cells=2, method='bdf'
        )

        # Regardless of GPU availability, an incompatible system must not take
        # the hardcoded-kernel path.
        assert solver._can_use_gpu_bdf() is False
