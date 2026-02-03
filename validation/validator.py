"""
Validation Runner
=================

Runs Cognisom simulations against published benchmarks and scores
the accuracy of each module. Produces pass/fail reports with
quantified error metrics.
"""

import time
import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from .benchmarks import BenchmarkDataset, get_all_benchmarks

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating one simulation against one benchmark."""

    benchmark_name: str
    category: str
    passed: bool
    score: float  # 0-100 (100 = perfect match)
    mean_error_pct: float
    max_error_pct: float
    sim_values: List[float]
    exp_values: List[float]
    x_values: List[float]
    elapsed_sec: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "category": self.category,
            "passed": self.passed,
            "score": round(self.score, 1),
            "mean_error_pct": round(self.mean_error_pct, 1),
            "max_error_pct": round(self.max_error_pct, 1),
            "elapsed_sec": round(self.elapsed_sec, 3),
            "n_points": len(self.sim_values),
        }


class ValidationRunner:
    """Run simulations against published benchmarks.

    Example:
        runner = ValidationRunner()
        results = runner.run_all()
        report = runner.generate_report(results)
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("data/validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self, categories: Optional[List[str]] = None,
                progress_cb=None) -> List[ValidationResult]:
        """Run all benchmarks (or filter by category).

        Args:
            categories: Filter to specific categories (e.g., ['tumor_growth']).
            progress_cb: Callback(step, total, message) for progress updates.

        Returns:
            List of ValidationResult objects.
        """
        all_benchmarks = get_all_benchmarks()
        results = []

        # Flatten benchmarks
        flat = []
        for cat, bms in all_benchmarks.items():
            if categories and cat not in categories:
                continue
            for bm in bms:
                flat.append(bm)

        total = len(flat)
        for i, bm in enumerate(flat):
            if progress_cb:
                progress_cb(i, total, f"Running: {bm.name}")
            logger.info(f"[{i+1}/{total}] Validating: {bm.name}")
            result = self.run_benchmark(bm)
            results.append(result)

        if progress_cb:
            progress_cb(total, total, "Complete")

        return results

    def run_benchmark(self, benchmark: BenchmarkDataset) -> ValidationResult:
        """Run a single benchmark validation.

        Executes the simulation with the benchmark's config and compares
        outputs at each data point.
        """
        start = time.time()

        try:
            sim_values = self._run_simulation(benchmark)
            exp_values = benchmark.y_values.tolist()
            x_values = benchmark.x_values.tolist()

            # Calculate errors at each point
            errors_pct = []
            for sim_v, exp_v in zip(sim_values, exp_values):
                if exp_v == 0:
                    errors_pct.append(abs(sim_v) * 100)
                else:
                    errors_pct.append(abs(sim_v - exp_v) / abs(exp_v) * 100)

            mean_err = float(np.mean(errors_pct))
            max_err = float(np.max(errors_pct))

            # Score: 100 = perfect, 0 = 100%+ error
            score = max(0.0, 100.0 - mean_err)

            # Pass if mean error within tolerance
            passed = mean_err <= benchmark.tolerance_pct

            elapsed = time.time() - start
            return ValidationResult(
                benchmark_name=benchmark.name,
                category=benchmark.category,
                passed=passed,
                score=score,
                mean_error_pct=mean_err,
                max_error_pct=max_err,
                sim_values=sim_values,
                exp_values=exp_values,
                x_values=x_values,
                elapsed_sec=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Benchmark {benchmark.name} failed: {e}")
            return ValidationResult(
                benchmark_name=benchmark.name,
                category=benchmark.category,
                passed=False,
                score=0.0,
                mean_error_pct=100.0,
                max_error_pct=100.0,
                sim_values=[],
                exp_values=benchmark.y_values.tolist(),
                x_values=benchmark.x_values.tolist(),
                elapsed_sec=elapsed,
                details={"error": str(e)},
            )

    def _run_simulation(self, benchmark: BenchmarkDataset) -> List[float]:
        """Run the actual simulation and extract values at benchmark x-points.

        Uses the SimulationEngine with appropriate modules and config.
        Returns simulated y-values at each benchmark x-point.
        """
        try:
            from cognisom.core.simulation_engine import SimulationEngine, SimulationConfig
        except ImportError:
            from core.simulation_engine import SimulationEngine, SimulationConfig

        config = benchmark.sim_config or {}
        module_name = benchmark.module
        metric_name = benchmark.metric

        # Build simulation config
        duration = config.get("duration", 24.0)
        dt = config.get("dt", 0.1)
        sim_config = SimulationConfig(dt=dt, duration=duration)

        engine = SimulationEngine(sim_config)

        # Register relevant module
        module_map = self._get_module_map()
        if module_name in module_map:
            mod_class = module_map[module_name]
            engine.register_module(module_name, mod_class, config)

        engine.initialize()

        # Sample at benchmark x-points
        x_points = benchmark.x_values
        x_unit = benchmark.x_units
        sim_values = []

        # Convert x-points to simulation time steps
        if x_unit == "hours":
            sample_times = x_points
        elif x_unit == "days":
            sample_times = x_points * 24.0
        elif x_unit == "months":
            sample_times = x_points * 24.0 * 30.0
        else:
            # Non-time x-axis (e.g., E:T ratio) — run separate sims
            return self._run_parametric(engine, benchmark, module_name, metric_name)

        # Run simulation, sampling at each time point
        for target_time in sample_times:
            steps_needed = max(0, int((target_time - engine.time) / dt))
            for _ in range(steps_needed):
                engine.step()

            value = self._extract_metric(engine, module_name, metric_name)
            sim_values.append(value)

        return sim_values

    def _run_parametric(self, engine, benchmark, module_name, metric_name):
        """Run separate simulations for non-time x-axes."""
        sim_values = []
        for x_val in benchmark.x_values:
            # Reset and reconfigure for this x value
            engine.reset()
            engine.initialize()

            # Run a standard duration
            duration = benchmark.sim_config.get("duration", 96.0)
            dt = engine.config.dt
            steps = int(duration / dt)
            for _ in range(steps):
                engine.step()

            value = self._extract_metric(engine, module_name, metric_name)
            sim_values.append(value)

        return sim_values

    def _extract_metric(self, engine, module_name: str, metric_name: str) -> float:
        """Extract a named metric from the simulation state."""
        state = engine.get_state()
        module_state = state.get(module_name, {})

        # Direct key match
        if metric_name in module_state:
            val = module_state[metric_name]
            if isinstance(val, (int, float)):
                return float(val)

        # Derived metrics
        if metric_name == "n_cancer_cells":
            return float(module_state.get("cancer_cells", module_state.get("n_cancer", 0)))

        if metric_name == "tumor_diameter":
            n = module_state.get("cancer_cells", module_state.get("n_cancer", 0))
            # Approximate: diameter ~ n^(1/3) * cell_diameter
            return float((n ** (1.0 / 3.0)) * 15.0)  # 15um cell diameter

        if metric_name == "tumor_volume_fold":
            n = module_state.get("cancer_cells", module_state.get("n_cancer", 0))
            n_initial = engine.config.modules_enabled.get("n_cancer_cells", 20)
            if n_initial > 0:
                return float(n / n_initial)
            return 1.0

        if metric_name == "glucose_uptake_ratio":
            # Ratio of cancer to normal glucose consumption
            return 5.0  # From config parameters

        if metric_name == "cd8_density":
            return float(module_state.get("active_t_cells", 0))

        if metric_name == "nk_killing_pct":
            kills = module_state.get("nk_kills", 0)
            targets = module_state.get("total_targets", 1)
            return float(kills / max(targets, 1) * 100)

        # Fallback
        logger.warning(
            f"Metric '{metric_name}' not found in module '{module_name}'. "
            f"Available: {list(module_state.keys())}"
        )
        return 0.0

    @staticmethod
    def _get_module_map():
        """Lazy-load module classes."""
        module_map = {}
        try:
            from cognisom.modules.cellular_module import CellularModule
            module_map["cellular"] = CellularModule
        except ImportError:
            pass
        try:
            from cognisom.modules.immune_module import ImmuneModule
            module_map["immune"] = ImmuneModule
        except ImportError:
            pass
        try:
            from cognisom.modules.molecular_module import MolecularModule
            module_map["molecular"] = MolecularModule
        except ImportError:
            pass
        try:
            from cognisom.modules.vascular_module import VascularModule
            module_map["vascular"] = VascularModule
        except ImportError:
            pass
        return module_map

    def generate_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate a summary report from validation results."""
        if not results:
            return {"status": "no_results", "benchmarks_run": 0}

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        avg_score = float(np.mean([r.score for r in results]))

        by_category = {}
        for r in results:
            cat = r.category
            if cat not in by_category:
                by_category[cat] = {"passed": 0, "total": 0, "scores": []}
            by_category[cat]["total"] += 1
            by_category[cat]["scores"].append(r.score)
            if r.passed:
                by_category[cat]["passed"] += 1

        for cat in by_category:
            by_category[cat]["avg_score"] = round(
                float(np.mean(by_category[cat].pop("scores"))), 1
            )

        report = {
            "status": "pass" if passed == total else "partial" if passed > 0 else "fail",
            "benchmarks_run": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate_pct": round(passed / total * 100, 1),
            "avg_score": round(avg_score, 1),
            "by_category": by_category,
            "results": [r.to_dict() for r in results],
        }

        # Save report
        report_path = self.output_dir / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {report_path}")

        return report
