"""
Validation Framework
====================

Benchmark, validate, calibrate, and profile Cognisom simulations
against published experimental data.
"""

from .benchmarks import (
    BenchmarkDataset,
    get_all_benchmarks,
    get_benchmark_by_name,
    TUMOR_GROWTH_BENCHMARKS,
    IMMUNE_INFILTRATION_BENCHMARKS,
    METABOLIC_FLUX_BENCHMARKS,
)


def _lazy_import(name):
    """Lazy import to avoid scipy dependency at module load time."""
    import importlib
    mod = importlib.import_module(f".{name}", __package__)
    return mod


def __getattr__(name):
    """Lazy-load heavy modules only when accessed."""
    _map = {
        "ValidationRunner": ("validator", "ValidationRunner"),
        "ValidationResult": ("validator", "ValidationResult"),
        "ParameterCalibrator": ("calibrator", "ParameterCalibrator"),
        "CalibrationResult": ("calibrator", "CalibrationResult"),
        "SimulationProfiler": ("profiler", "SimulationProfiler"),
        "ProfileReport": ("profiler", "ProfileReport"),
    }
    if name in _map:
        module_name, attr_name = _map[name]
        mod = _lazy_import(module_name)
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BenchmarkDataset",
    "get_all_benchmarks",
    "get_benchmark_by_name",
    "TUMOR_GROWTH_BENCHMARKS",
    "IMMUNE_INFILTRATION_BENCHMARKS",
    "METABOLIC_FLUX_BENCHMARKS",
    "ValidationRunner",
    "ValidationResult",
    "ParameterCalibrator",
    "CalibrationResult",
    "SimulationProfiler",
    "ProfileReport",
]
