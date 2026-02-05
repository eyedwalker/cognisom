"""
Integration Tests: Validation Framework
========================================

Tests that the validation framework can:
1. Load benchmark datasets
2. Run simulations against benchmarks
3. Produce scored results
4. Calibrate parameters
5. Profile simulation performance
6. Generate synthetic reference data

Run: pytest cognisom/tests/test_validation.py -v
"""

import sys
from pathlib import Path

# Ensure project root on path
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest
import numpy as np


class TestBenchmarks:
    """Test benchmark data loading and structure."""

    def test_all_benchmarks_load(self):
        from cognisom.validation.benchmarks import get_all_benchmarks
        bms = get_all_benchmarks()
        assert len(bms) > 0
        for cat, datasets in bms.items():
            assert len(datasets) > 0
            for ds in datasets:
                assert ds.name
                assert ds.category == cat
                assert len(ds.data_points) >= 2

    def test_benchmark_categories(self):
        from cognisom.validation.benchmarks import get_all_benchmarks
        bms = get_all_benchmarks()
        expected = {"tumor_growth", "immune", "metabolic"}
        assert set(bms.keys()) == expected

    def test_benchmark_count(self):
        from cognisom.validation.benchmarks import summary
        counts = summary()
        assert counts["tumor_growth"] == 3
        assert counts["immune"] == 3
        assert counts["metabolic"] == 3

    def test_benchmark_by_name(self):
        from cognisom.validation.benchmarks import get_benchmark_by_name
        bm = get_benchmark_by_name("PC3 Spheroid Growth Curve")
        assert bm is not None
        assert bm.category == "tumor_growth"
        assert len(bm.data_points) == 15

    def test_benchmark_interpolation(self):
        from cognisom.validation.benchmarks import get_benchmark_by_name
        bm = get_benchmark_by_name("LNCaP Doubling Time")
        # Interpolation at a known point
        val = bm.interpolate(0)
        assert val == pytest.approx(100.0)
        # Between points
        val = bm.interpolate(36)
        assert 130 < val < 170

    def test_tolerance_check(self):
        from cognisom.validation.benchmarks import get_benchmark_by_name
        bm = get_benchmark_by_name("LNCaP Doubling Time")
        # Within tolerance (30%)
        assert bm.within_tolerance(0, 120.0)  # 20% off
        # Outside tolerance
        assert not bm.within_tolerance(0, 200.0)  # 100% off

    def test_x_y_arrays(self):
        from cognisom.validation.benchmarks import get_benchmark_by_name
        bm = get_benchmark_by_name("PC3 Spheroid Growth Curve")
        assert isinstance(bm.x_values, np.ndarray)
        assert isinstance(bm.y_values, np.ndarray)
        assert len(bm.x_values) == len(bm.y_values)


class TestSyntheticData:
    """Test synthetic data generation."""

    def test_tumor_growth_shape(self):
        from cognisom.validation.data_sources import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=42)
        data = gen.tumor_spheroid_growth(n_days=14)
        assert "cell_counts" in data
        assert "diameters_um" in data
        assert len(data["cell_counts"]) == len(data["time_hours"])
        # Cells should grow
        assert data["cell_counts"][-1] > data["cell_counts"][0]

    def test_tumor_growth_gompertz(self):
        from cognisom.validation.data_sources import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=42)
        data = gen.tumor_spheroid_growth(n_days=30, carrying_capacity=5000)
        counts = data["cell_counts"]
        # Should approach carrying capacity
        assert counts[-1] < 5500  # within noise of carrying capacity

    def test_immune_infiltration(self):
        from cognisom.validation.data_sources import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=42)
        data = gen.immune_infiltration(n_days=28)
        assert "cd8_density_per_mm2" in data
        assert "nk_density_per_mm2" in data
        assert "pd1_positive_fraction" in data
        # PD-1 should increase over time
        pd1 = data["pd1_positive_fraction"]
        assert pd1[-1] > pd1[0]

    def test_metabolic_profile(self):
        from cognisom.validation.data_sources import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=42)
        data = gen.metabolic_profile(n_hours=48)
        # Cancer should consume glucose faster
        normal_gluc = data["normal"]["glucose_mM"]
        cancer_gluc = data["cancer"]["glucose_mM"]
        assert cancer_gluc[-1] < normal_gluc[-1]
        # Cancer should produce more lactate
        normal_lac = data["normal"]["lactate_mM"]
        cancer_lac = data["cancer"]["lactate_mM"]
        assert cancer_lac[-1] > normal_lac[-1]

    def test_cell_type_proportions(self):
        from cognisom.validation.data_sources import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=42)
        data = gen.cell_type_proportions()
        normal = data["normal_tissue"]
        tumor = data["tumor_tissue"]
        # Proportions should sum to ~1.0
        assert abs(sum(normal.values()) - 1.0) < 0.01
        assert abs(sum(tumor.values()) - 1.0) < 0.01

    def test_generate_all(self, tmp_path):
        from cognisom.validation.data_sources import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=42)
        paths = gen.generate_all(output_dir=str(tmp_path))
        assert len(paths) == 4
        for name, path in paths.items():
            assert path.exists()


class TestValidator:
    """Test the validation runner."""

    def test_validator_creates(self, tmp_path):
        from cognisom.validation.validator import ValidationRunner
        runner = ValidationRunner(output_dir=str(tmp_path))
        assert runner.output_dir.exists()

    def test_run_single_benchmark(self, tmp_path):
        from cognisom.validation.validator import ValidationRunner
        from cognisom.validation.benchmarks import get_benchmark_by_name

        bm = get_benchmark_by_name("LNCaP Doubling Time")
        runner = ValidationRunner(output_dir=str(tmp_path))
        result = runner.run_benchmark(bm)

        assert result.benchmark_name == "LNCaP Doubling Time"
        assert result.category == "tumor_growth"
        assert 0 <= result.score <= 100
        assert result.elapsed_sec >= 0

    def test_generate_report(self, tmp_path):
        from cognisom.validation.validator import ValidationRunner
        from cognisom.validation.benchmarks import TUMOR_GROWTH_BENCHMARKS

        runner = ValidationRunner(output_dir=str(tmp_path))
        # Run just the first benchmark for speed
        results = [runner.run_benchmark(TUMOR_GROWTH_BENCHMARKS[0])]
        report = runner.generate_report(results)

        assert "status" in report
        assert "benchmarks_run" in report
        assert report["benchmarks_run"] == 1
        assert "by_category" in report


class TestCalibrator:
    """Test parameter calibration."""

    def test_default_parameters(self):
        from cognisom.validation.calibrator import ParameterCalibrator
        params = ParameterCalibrator.get_default_parameters()
        assert len(params) > 0
        for p in params:
            assert p.min_val < p.max_val
            assert p.min_val <= p.initial <= p.max_val

    def test_module_filter(self):
        from cognisom.validation.calibrator import ParameterCalibrator
        cellular_params = ParameterCalibrator.get_default_parameters(module="cellular")
        assert all(p.module == "cellular" for p in cellular_params)
        assert len(cellular_params) > 0

    def test_parameter_bounds(self):
        from cognisom.validation.calibrator import ParameterCalibrator
        params = ParameterCalibrator.get_default_parameters()
        for p in params:
            lo, hi = p.bounds
            assert lo == p.min_val
            assert hi == p.max_val


class TestProfiler:
    """Test simulation profiler."""

    def test_profiler_creates(self, tmp_path):
        from cognisom.validation.profiler import SimulationProfiler
        profiler = SimulationProfiler(output_dir=str(tmp_path))
        assert profiler.output_dir.exists()

    def test_profile_short_run(self, tmp_path):
        from cognisom.validation.profiler import SimulationProfiler

        profiler = SimulationProfiler(output_dir=str(tmp_path))
        report = profiler.profile(
            duration=1.0,
            dt=0.1,
            modules=["cellular"],
            n_cells=20,
        )

        assert report.total_steps == 10
        assert report.total_time_sec > 0
        assert report.steps_per_second > 0
        assert "cellular" in report.modules
        assert report.bottleneck_module == "cellular"

    def test_profile_report_summary(self, tmp_path):
        from cognisom.validation.profiler import SimulationProfiler

        profiler = SimulationProfiler(output_dir=str(tmp_path))
        report = profiler.profile(
            duration=1.0, dt=0.1, modules=["cellular"], n_cells=10,
        )

        summary = report.summary()
        assert "cellular" in summary
        assert "steps/sec" in summary

    def test_profile_to_dict(self, tmp_path):
        from cognisom.validation.profiler import SimulationProfiler

        profiler = SimulationProfiler(output_dir=str(tmp_path))
        report = profiler.profile(
            duration=1.0, dt=0.1, modules=["cellular"], n_cells=10,
        )

        d = report.to_dict()
        assert "total_time_sec" in d
        assert "modules" in d
        assert "cellular" in d["modules"]
        assert "avg_step_ms" in d["modules"]["cellular"]
