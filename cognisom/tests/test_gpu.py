"""
Tests for the GPU acceleration module.

All tests run on CPU (NumPy fallback) since the test environment
may not have an NVIDIA GPU. The GPU code paths use the same logic,
just backed by CuPy arrays instead of NumPy.
"""

import numpy as np
import pytest

from cognisom.gpu.backend import get_backend, reset_backend, GPUBackend


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def cpu_backend():
    """Force CPU backend for all tests."""
    reset_backend()
    backend = get_backend(force_cpu=True)
    yield backend
    reset_backend()


@pytest.fixture
def small_field():
    """Small 3D field for diffusion tests."""
    field = np.zeros((10, 10, 6), dtype=np.float32)
    field[5, 5, 3] = 1.0  # Point source in center
    return field


@pytest.fixture
def cell_dict():
    """Simulate CellularModule.cells dict with dataclass-like objects."""
    class FakeCell:
        def __init__(self, cid, pos, ctype, oxygen=0.2, glucose=5.0, atp=1000, lactate=0.0, age=0.0, mhc1=1.0, alive=True, phase="G1"):
            self.cell_id = cid
            self.position = np.array(pos, dtype=np.float32)
            self.cell_type = ctype
            self.oxygen = oxygen
            self.glucose = glucose
            self.atp = atp
            self.lactate = lactate
            self.age = age
            self.mhc1_expression = mhc1
            self.alive = alive
            self.phase = phase

    cells = {}
    # 5 normal cells
    for i in range(5):
        cells[i] = FakeCell(i, [50 + i * 10, 50, 50], "normal")
    # 3 cancer cells
    for i in range(5, 8):
        cells[i] = FakeCell(i, [100 + (i - 5) * 10, 100, 50], "cancer",
                            oxygen=0.08, glucose=2.0, atp=600, mhc1=0.3)
    return cells


# ── Backend Tests ────────────────────────────────────────────────

class TestBackend:
    def test_cpu_backend_detected(self, cpu_backend):
        assert not cpu_backend.has_gpu
        assert cpu_backend.device_name == "CPU"
        assert cpu_backend.xp is np

    def test_to_numpy_noop(self, cpu_backend):
        arr = np.array([1.0, 2.0, 3.0])
        result = cpu_backend.to_numpy(arr)
        np.testing.assert_array_equal(result, arr)

    def test_to_device_noop(self, cpu_backend):
        arr = np.array([1.0, 2.0, 3.0])
        result = cpu_backend.to_device(arr)
        np.testing.assert_array_equal(result, arr)

    def test_synchronize_noop(self, cpu_backend):
        cpu_backend.synchronize()  # Should not raise

    def test_summary(self, cpu_backend):
        s = cpu_backend.summary()
        assert "CPU" in s

    def test_reset_and_redetect(self):
        reset_backend()
        b1 = get_backend(force_cpu=True)
        reset_backend()
        b2 = get_backend(force_cpu=True)
        assert b1 is not b2  # Different instances after reset


# ── Diffusion Tests ──────────────────────────────────────────────

class TestDiffusion:
    def test_laplacian_zero_field(self):
        from cognisom.gpu.diffusion import compute_laplacian
        field = np.zeros((10, 10, 6), dtype=np.float32)
        lap = compute_laplacian(field)
        np.testing.assert_array_equal(lap, np.zeros_like(field))

    def test_laplacian_point_source(self, small_field):
        from cognisom.gpu.diffusion import compute_laplacian
        lap = compute_laplacian(small_field)
        # Center should have negative Laplacian (diffusing outward)
        assert lap[5, 5, 3] < 0
        # Neighbors should have positive Laplacian
        assert lap[4, 5, 3] > 0
        assert lap[6, 5, 3] > 0

    def test_laplacian_shape_preserved(self, small_field):
        from cognisom.gpu.diffusion import compute_laplacian
        lap = compute_laplacian(small_field)
        assert lap.shape == small_field.shape

    def test_laplacian_boundaries_zero(self, small_field):
        from cognisom.gpu.diffusion import compute_laplacian
        lap = compute_laplacian(small_field)
        # Boundary slices should be zero
        assert lap[0, :, :].sum() == 0
        assert lap[-1, :, :].sum() == 0
        assert lap[:, 0, :].sum() == 0
        assert lap[:, -1, :].sum() == 0
        assert lap[:, :, 0].sum() == 0
        assert lap[:, :, -1].sum() == 0

    def test_diffuse_field_conserves_mass_approx(self, small_field):
        from cognisom.gpu.diffusion import diffuse_field
        initial_sum = small_field.sum()
        result = diffuse_field(
            small_field, diffusion_coeff=100.0, dt=0.01, resolution=10.0,
        )
        # Mass should be approximately conserved (no sources/sinks)
        assert abs(result.sum() - initial_sum) < 0.1

    def test_diffuse_field_with_source(self, small_field):
        from cognisom.gpu.diffusion import diffuse_field
        sources = [((2, 2, 2), 10.0)]
        result = diffuse_field(
            small_field, diffusion_coeff=100.0, dt=0.1, resolution=10.0,
            sources=sources,
        )
        # Source should add concentration (rate=10 * dt=0.1 = 1.0 added)
        assert result[2, 2, 2] > small_field[2, 2, 2]

    def test_diffuse_field_non_negative(self, small_field):
        from cognisom.gpu.diffusion import diffuse_field
        result = diffuse_field(
            small_field, diffusion_coeff=100.0, dt=0.01, resolution=10.0,
        )
        assert (result >= 0).all()

    def test_diffuse_fields_batch(self):
        from cognisom.gpu.diffusion import diffuse_fields_batch
        f1 = np.random.rand(10, 10, 6).astype(np.float32)
        f2 = np.random.rand(10, 10, 6).astype(np.float32)
        fields = {
            "o2": (f1, 2000.0, [], []),
            "glucose": (f2, 600.0, [], []),
        }
        results = diffuse_fields_batch(fields, dt=0.01, resolution=10.0)
        assert "o2" in results
        assert "glucose" in results
        assert results["o2"].shape == (10, 10, 6)


# ── Cell Operations Tests ────────────────────────────────────────

class TestCellOps:
    def test_cell_arrays_from_dict(self, cell_dict):
        from cognisom.gpu.cell_ops import CellArrays
        arrays = CellArrays.from_cell_dict(cell_dict)
        assert arrays.n == 8
        assert arrays.positions.shape == (8, 3)
        assert arrays.oxygen.shape == (8,)

    def test_cell_arrays_types_correct(self, cell_dict):
        from cognisom.gpu.cell_ops import CellArrays
        arrays = CellArrays.from_cell_dict(cell_dict)
        # First 5 are normal (0), last 3 are cancer (1)
        assert (arrays.cell_types[:5] == 0).all()
        assert (arrays.cell_types[5:] == 1).all()

    def test_metabolism_update(self, cell_dict):
        from cognisom.gpu.cell_ops import CellArrays, update_metabolism_vectorized
        arrays = CellArrays.from_cell_dict(cell_dict)
        initial_glucose = arrays.glucose.copy()

        update_metabolism_vectorized(arrays, dt=1.0)

        # Glucose should decrease
        assert (arrays.glucose < initial_glucose).all()
        # All values should be non-negative
        assert (arrays.glucose >= 0).all()
        assert (arrays.oxygen >= 0).all()

    def test_metabolism_cancer_consumes_more(self, cell_dict):
        from cognisom.gpu.cell_ops import CellArrays, update_metabolism_vectorized
        arrays = CellArrays.from_cell_dict(cell_dict)

        # Set all to same glucose
        arrays.glucose[:] = 5.0
        update_metabolism_vectorized(arrays, dt=1.0)

        # Cancer cells (idx 5-7) should have less glucose than normal (idx 0-4)
        normal_glucose = float(arrays.glucose[0])
        cancer_glucose = float(arrays.glucose[5])
        assert cancer_glucose < normal_glucose

    def test_metabolism_ages_cells(self, cell_dict):
        from cognisom.gpu.cell_ops import CellArrays, update_metabolism_vectorized
        arrays = CellArrays.from_cell_dict(cell_dict)
        assert arrays.ages.sum() == 0

        update_metabolism_vectorized(arrays, dt=0.5)
        assert (arrays.ages > 0).all()

    def test_write_back(self, cell_dict):
        from cognisom.gpu.cell_ops import CellArrays, update_metabolism_vectorized
        arrays = CellArrays.from_cell_dict(cell_dict)
        update_metabolism_vectorized(arrays, dt=1.0)
        arrays.write_back(cell_dict)

        # Original dict should now have updated values
        assert cell_dict[0].glucose < 5.0
        assert cell_dict[5].glucose < 2.0

    def test_detect_death_candidates(self, cell_dict):
        from cognisom.gpu.cell_ops import CellArrays, detect_death_candidates
        arrays = CellArrays.from_cell_dict(cell_dict)
        # Set some cells to very low O2
        arrays.oxygen[0] = 0.01  # Normal cell -> should die
        arrays.oxygen[5] = 0.01  # Cancer cell -> stochastic

        normal_die, cancer_die = detect_death_candidates(arrays, seed=42)
        # Normal cell 0 should die
        assert 0 in normal_die

    def test_detect_division_candidates(self, cell_dict):
        from cognisom.gpu.cell_ops import CellArrays, detect_division_candidates
        arrays = CellArrays.from_cell_dict(cell_dict)
        # Set one cell's age past division time
        arrays.ages[0] = 25.0  # Normal, div time = 24
        arrays.ages[5] = 13.0  # Cancer, div time = 12

        div_ids = detect_division_candidates(arrays)
        assert 0 in div_ids  # Normal cell at age 25 > 24
        assert 5 in div_ids  # Cancer cell at age 13 > 12


# ── Spatial Operations Tests ─────────────────────────────────────

class TestSpatialOps:
    def test_pairwise_distances_identity(self):
        from cognisom.gpu.spatial_ops import pairwise_distances
        A = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
        B = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
        D = pairwise_distances(A, B)
        assert D.shape == (2, 2)
        np.testing.assert_almost_equal(D[0, 0], 0.0, decimal=3)
        np.testing.assert_almost_equal(D[0, 1], 10.0, decimal=3)
        np.testing.assert_almost_equal(D[1, 0], 10.0, decimal=3)
        np.testing.assert_almost_equal(D[1, 1], 0.0, decimal=3)

    def test_pairwise_distances_3d(self):
        from cognisom.gpu.spatial_ops import pairwise_distances
        A = np.array([[0, 0, 0]], dtype=np.float32)
        B = np.array([[3, 4, 0]], dtype=np.float32)
        D = pairwise_distances(A, B)
        np.testing.assert_almost_equal(D[0, 0], 5.0, decimal=3)

    def test_pairwise_empty(self):
        from cognisom.gpu.spatial_ops import pairwise_distances
        A = np.zeros((0, 3), dtype=np.float32)
        B = np.array([[1, 2, 3]], dtype=np.float32)
        D = pairwise_distances(A, B)
        assert D.shape == (0, 1)

    def test_find_neighbors(self):
        from cognisom.gpu.spatial_ops import find_neighbors
        A = np.array([[0, 0, 0], [100, 100, 100]], dtype=np.float32)
        B = np.array([[5, 0, 0], [105, 100, 100]], dtype=np.float32)
        idx_a, idx_b, dists = find_neighbors(A, B, radius=10.0)
        # (0,0) should be neighbors (dist=5), (1,1) should be neighbors (dist=5)
        assert len(idx_a) == 2
        assert 0 in idx_a
        assert 1 in idx_a

    def test_nearest_neighbor(self):
        from cognisom.gpu.spatial_ops import nearest_neighbor
        A = np.array([[0, 0, 0], [90, 90, 90]], dtype=np.float32)
        B = np.array([[1, 0, 0], [100, 100, 100]], dtype=np.float32)
        nearest_idx, nearest_dist = nearest_neighbor(A, B)
        assert nearest_idx[0] == 0  # (0,0,0) is nearest to (1,0,0)
        assert nearest_idx[1] == 1  # (90,90,90) is nearest to (100,100,100)

    def test_compute_directions(self):
        from cognisom.gpu.spatial_ops import compute_directions
        A = np.array([[0, 0, 0]], dtype=np.float32)
        B = np.array([[10, 0, 0]], dtype=np.float32)
        dirs, dists = compute_directions(A, B)
        np.testing.assert_almost_equal(dirs[0], [1, 0, 0], decimal=3)
        np.testing.assert_almost_equal(dists[0], 10.0, decimal=3)

    def test_immune_detection_batch(self):
        from cognisom.gpu.spatial_ops import immune_detection_batch
        immune_pos = np.array([[100, 100, 50]], dtype=np.float32)
        cancer_pos = np.array([[105, 100, 50]], dtype=np.float32)  # 5um away
        cancer_mhc = np.array([0.5], dtype=np.float32)
        immune_types = np.array([0], dtype=np.int8)  # T_cell

        detections = immune_detection_batch(
            immune_pos, cancer_pos,
            detection_radius=10.0,
            cancer_mhc1=cancer_mhc,
            immune_types=immune_types,
        )
        # T cell should detect cancer (MHC-I 0.5 > threshold 0.2)
        assert len(detections) == 1
        assert detections[0][0] == 0  # immune idx
        assert detections[0][1] == 0  # cancer idx

    def test_immune_detection_nk_missing_self(self):
        from cognisom.gpu.spatial_ops import immune_detection_batch
        immune_pos = np.array([[100, 100, 50]], dtype=np.float32)
        cancer_pos = np.array([[105, 100, 50]], dtype=np.float32)
        cancer_mhc = np.array([0.1], dtype=np.float32)  # Low MHC-I
        immune_types = np.array([1], dtype=np.int8)  # NK_cell

        detections = immune_detection_batch(
            immune_pos, cancer_pos,
            detection_radius=10.0,
            cancer_mhc1=cancer_mhc,
            immune_types=immune_types,
        )
        # NK cell should detect (MHC-I 0.1 < threshold 0.4)
        assert len(detections) == 1

    def test_immune_detection_out_of_range(self):
        from cognisom.gpu.spatial_ops import immune_detection_batch
        immune_pos = np.array([[0, 0, 0]], dtype=np.float32)
        cancer_pos = np.array([[100, 100, 100]], dtype=np.float32)  # Far away
        cancer_mhc = np.array([0.5], dtype=np.float32)
        immune_types = np.array([0], dtype=np.int8)

        detections = immune_detection_batch(
            immune_pos, cancer_pos,
            detection_radius=10.0,
            cancer_mhc1=cancer_mhc,
            immune_types=immune_types,
        )
        assert len(detections) == 0


# ── Orchestrator Tests ───────────────────────────────────────────

class TestOrchestrator:
    def test_create_orchestrator(self):
        from cognisom.gpu.orchestrator import GPUOrchestrator
        gpu = GPUOrchestrator(force_cpu=True)
        assert not gpu.backend.has_gpu

    def test_get_report(self):
        from cognisom.gpu.orchestrator import GPUOrchestrator
        gpu = GPUOrchestrator(force_cpu=True)
        report = gpu.get_report()
        assert "backend" in report
        assert report["has_gpu"] is False

    def test_acceleration_report(self):
        from cognisom.gpu.orchestrator import AccelerationReport
        report = AccelerationReport(
            backend="CPU only",
            modules_accelerated=["spatial", "cellular"],
            modules_skipped=["molecular"],
        )
        s = report.summary()
        assert "spatial" in s
        assert "molecular" in s
