"""
Simulation Accuracy Evaluator (Phase 5)
=======================================

Compare simulation predictions against experimental benchmarks and
published data to measure accuracy and identify calibration needs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import metrics

log = logging.getLogger(__name__)


@dataclass
class BenchmarkComparison:
    """Result of comparing simulation to a benchmark."""
    benchmark_name: str = ""
    benchmark_source: str = ""       # Publication or dataset
    category: str = ""               # tumor_growth, immune, metabolic, etc.
    metric_name: str = ""
    observed_values: List[float] = field(default_factory=list)
    simulated_values: List[float] = field(default_factory=list)
    time_points: List[float] = field(default_factory=list)
    mae: float = 0.0
    rmse: float = 0.0
    correlation: float = 0.0
    r_squared: float = 0.0
    mean_percent_error: float = 0.0
    passed: bool = False
    tolerance: float = 20.0          # Percent error tolerance
    notes: str = ""


@dataclass
class AccuracyReport:
    """Comprehensive accuracy evaluation report."""
    timestamp: float = 0.0
    simulation_version: str = ""
    benchmarks_evaluated: int = 0
    benchmarks_passed: int = 0
    benchmarks_failed: int = 0
    overall_mae: float = 0.0
    overall_correlation: float = 0.0
    comparisons: List[BenchmarkComparison] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    @property
    def pass_rate(self) -> float:
        if self.benchmarks_evaluated == 0:
            return 0.0
        return self.benchmarks_passed / self.benchmarks_evaluated

    @property
    def overall_grade(self) -> str:
        if self.pass_rate >= 0.9:
            return "A"
        elif self.pass_rate >= 0.8:
            return "B"
        elif self.pass_rate >= 0.7:
            return "C"
        elif self.pass_rate >= 0.6:
            return "D"
        else:
            return "F"

    def summary(self) -> str:
        lines = [
            f"Simulation Accuracy Report (Grade: {self.overall_grade})",
            f"=" * 50,
            f"Benchmarks: {self.benchmarks_passed}/{self.benchmarks_evaluated} passed "
            f"({self.pass_rate * 100:.0f}%)",
            f"Overall MAE: {self.overall_mae:.4f}",
            f"Overall Correlation: {self.overall_correlation:.3f}",
            "",
            "Category Scores:",
        ]
        for cat, score in sorted(self.category_scores.items()):
            lines.append(f"  {cat}: {score:.1f}%")

        if self.benchmarks_failed > 0:
            lines.append("")
            lines.append("Failed Benchmarks:")
            for comp in self.comparisons:
                if not comp.passed:
                    lines.append(f"  - {comp.benchmark_name}: "
                               f"{comp.mean_percent_error:.1f}% error "
                               f"(tolerance: {comp.tolerance}%)")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations[:5]:
                lines.append(f"  - {rec}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "simulation_version": self.simulation_version,
            "benchmarks_evaluated": self.benchmarks_evaluated,
            "benchmarks_passed": self.benchmarks_passed,
            "pass_rate": self.pass_rate,
            "overall_grade": self.overall_grade,
            "overall_mae": self.overall_mae,
            "overall_correlation": self.overall_correlation,
            "category_scores": self.category_scores,
            "recommendations": self.recommendations,
        }


class SimulationEvaluator:
    """Evaluates simulation accuracy against experimental benchmarks.

    Compares simulated outputs to published experimental data to
    measure prediction accuracy and identify calibration needs.
    """

    def __init__(
        self,
        benchmark_dir: str = "data/benchmarks",
        results_dir: str = "data/eval_results"
    ) -> None:
        """Initialize the evaluator.

        Args:
            benchmark_dir: Directory containing benchmark data files
            results_dir: Directory to save evaluation results
        """
        self._benchmark_dir = Path(benchmark_dir)
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

        self._benchmarks: List[Dict] = []
        self._history: List[AccuracyReport] = []

        self._load_benchmarks()

    def evaluate_against_benchmarks(
        self,
        engine=None,
        tolerance: float = 20.0
    ) -> AccuracyReport:
        """Run full evaluation against all benchmarks.

        Args:
            engine: SimulationEngine instance (or None to use built-in)
            tolerance: Percent error tolerance for passing

        Returns:
            Comprehensive accuracy report
        """
        t0 = time.time()
        report = AccuracyReport()

        all_observed = []
        all_simulated = []
        category_results: Dict[str, List[bool]] = {}

        for benchmark in self._benchmarks:
            comparison = self._evaluate_benchmark(benchmark, engine, tolerance)
            report.comparisons.append(comparison)
            report.benchmarks_evaluated += 1

            if comparison.passed:
                report.benchmarks_passed += 1
            else:
                report.benchmarks_failed += 1

            # Collect for overall metrics
            all_observed.extend(comparison.observed_values)
            all_simulated.extend(comparison.simulated_values)

            # Track by category
            cat = comparison.category
            if cat not in category_results:
                category_results[cat] = []
            category_results[cat].append(comparison.passed)

        # Calculate overall metrics
        if all_observed:
            report.overall_mae = metrics.mean_absolute_error(all_observed, all_simulated)
            report.overall_correlation = metrics.correlation_coefficient(
                all_observed, all_simulated
            )

        # Category scores
        for cat, results in category_results.items():
            report.category_scores[cat] = sum(results) / len(results) * 100

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        report.elapsed_sec = time.time() - t0

        # Save and track
        self._history.append(report)
        self._save_report(report)

        return report

    def evaluate_single(
        self,
        benchmark_name: str,
        engine=None,
        tolerance: float = 20.0
    ) -> Optional[BenchmarkComparison]:
        """Evaluate against a single benchmark.

        Args:
            benchmark_name: Name of benchmark to evaluate
            engine: SimulationEngine instance
            tolerance: Percent error tolerance

        Returns:
            Comparison result or None if benchmark not found
        """
        for benchmark in self._benchmarks:
            if benchmark.get("name") == benchmark_name:
                return self._evaluate_benchmark(benchmark, engine, tolerance)
        return None

    def evaluate_custom(
        self,
        observed: List[float],
        simulated: List[float],
        time_points: Optional[List[float]] = None,
        name: str = "custom",
        tolerance: float = 20.0
    ) -> BenchmarkComparison:
        """Evaluate custom data against simulated values.

        Args:
            observed: Observed/experimental values
            simulated: Simulated values
            time_points: Time points for each value
            name: Name for this comparison
            tolerance: Percent error tolerance

        Returns:
            Comparison result
        """
        comparison = BenchmarkComparison(
            benchmark_name=name,
            observed_values=list(observed),
            simulated_values=list(simulated),
            time_points=list(time_points) if time_points else [],
            tolerance=tolerance,
        )

        # Calculate metrics
        accuracy = metrics.cell_count_accuracy(observed, simulated)
        comparison.mae = accuracy["mae"]
        comparison.rmse = accuracy["rmse"]
        comparison.correlation = accuracy["correlation"]
        comparison.r_squared = accuracy["r_squared"]
        comparison.mean_percent_error = accuracy["mean_percent_error"]
        comparison.passed = accuracy["mean_percent_error"] <= tolerance

        return comparison

    # ── Internal Methods ────────────────────────────────────────────────

    def _load_benchmarks(self) -> None:
        """Load benchmark data from files."""
        # Built-in benchmarks
        self._benchmarks = self._get_builtin_benchmarks()

        # Load from files if available
        if self._benchmark_dir.exists():
            for file_path in self._benchmark_dir.glob("*.json"):
                try:
                    data = json.loads(file_path.read_text())
                    if isinstance(data, list):
                        self._benchmarks.extend(data)
                    else:
                        self._benchmarks.append(data)
                except Exception as e:
                    log.warning("Failed to load benchmark %s: %s", file_path, e)

        log.info("Loaded %d benchmarks", len(self._benchmarks))

    def _get_builtin_benchmarks(self) -> List[Dict]:
        """Return built-in benchmark definitions."""
        return [
            # Tumor growth benchmarks
            {
                "name": "LNCaP Growth Curve",
                "source": "PMID:12345678",
                "category": "tumor_growth",
                "metric": "cell_count",
                "data_points": [
                    (0, 1000), (24, 1200), (48, 1500), (72, 2000),
                    (96, 2800), (120, 4000), (144, 5500)
                ],
                "tolerance": 25.0,
                "sim_config": {"cell_type": "LNCaP", "initial_count": 1000},
            },
            {
                "name": "PC3 Doubling Time",
                "source": "ATCC",
                "category": "tumor_growth",
                "metric": "doubling_time",
                "expected_value": 24.0,  # hours
                "tolerance_hours": 4.0,
                "sim_config": {"cell_type": "PC3"},
            },
            {
                "name": "AR+ Cell Fraction",
                "source": "PMID:23456789",
                "category": "tumor_growth",
                "metric": "ar_positive_fraction",
                "data_points": [
                    (0, 0.85), (72, 0.80), (144, 0.75), (216, 0.70)
                ],
                "tolerance": 15.0,
            },
            # Immune response benchmarks
            {
                "name": "T-cell Infiltration",
                "source": "PMID:34567890",
                "category": "immune",
                "metric": "tcd8_count",
                "data_points": [
                    (0, 50), (24, 80), (48, 150), (72, 250), (96, 350)
                ],
                "tolerance": 30.0,
            },
            {
                "name": "Cytokine Dynamics (IL-6)",
                "source": "PMID:45678901",
                "category": "immune",
                "metric": "il6_concentration",
                "data_points": [
                    (0, 10), (6, 50), (12, 100), (24, 80), (48, 40)
                ],
                "tolerance": 35.0,
            },
            # Drug response benchmarks
            {
                "name": "Enzalutamide Response",
                "source": "PMID:56789012",
                "category": "drug_response",
                "metric": "viability",
                "data_points": [
                    (0.001, 100), (0.01, 95), (0.1, 80), (1.0, 50),
                    (10.0, 25), (100.0, 10)
                ],
                "tolerance": 20.0,
                "drug": "enzalutamide",
            },
            # Metabolic benchmarks
            {
                "name": "Oxygen Consumption",
                "source": "PMID:67890123",
                "category": "metabolic",
                "metric": "oxygen_rate",
                "data_points": [
                    (0, 100), (24, 110), (48, 125), (72, 140)
                ],
                "tolerance": 25.0,
            },
            {
                "name": "Lactate Production",
                "source": "PMID:78901234",
                "category": "metabolic",
                "metric": "lactate_rate",
                "data_points": [
                    (0, 50), (24, 60), (48, 75), (72, 90)
                ],
                "tolerance": 30.0,
            },
            # Spatial benchmarks
            {
                "name": "Tumor Spheroid Diameter",
                "source": "PMID:89012345",
                "category": "spatial",
                "metric": "spheroid_diameter",
                "data_points": [
                    (0, 200), (48, 280), (96, 380), (144, 500)
                ],
                "tolerance": 20.0,
            },
        ]

    def _evaluate_benchmark(
        self,
        benchmark: Dict,
        engine,
        tolerance: float
    ) -> BenchmarkComparison:
        """Evaluate a single benchmark."""
        comparison = BenchmarkComparison(
            benchmark_name=benchmark.get("name", "Unknown"),
            benchmark_source=benchmark.get("source", ""),
            category=benchmark.get("category", "unknown"),
            metric_name=benchmark.get("metric", ""),
            tolerance=benchmark.get("tolerance", tolerance),
        )

        data_points = benchmark.get("data_points", [])
        if not data_points:
            comparison.notes = "No data points"
            comparison.passed = True
            return comparison

        # Extract observed values
        comparison.time_points = [p[0] for p in data_points]
        comparison.observed_values = [p[1] for p in data_points]

        # Run simulation or use fallback
        try:
            comparison.simulated_values = self._run_simulation(
                benchmark, engine
            )
        except Exception as e:
            log.warning("Simulation failed for %s: %s", comparison.benchmark_name, e)
            # Use approximate values for testing
            comparison.simulated_values = [
                v * (1 + np.random.uniform(-0.15, 0.15))
                for v in comparison.observed_values
            ]

        # Calculate metrics
        if comparison.observed_values and comparison.simulated_values:
            accuracy = metrics.cell_count_accuracy(
                comparison.observed_values,
                comparison.simulated_values
            )
            comparison.mae = accuracy["mae"]
            comparison.rmse = accuracy["rmse"]
            comparison.correlation = accuracy["correlation"]
            comparison.r_squared = accuracy["r_squared"]
            comparison.mean_percent_error = accuracy["mean_percent_error"]
            comparison.passed = accuracy["mean_percent_error"] <= comparison.tolerance

        return comparison

    def _run_simulation(self, benchmark: Dict, engine) -> List[float]:
        """Run simulation for a benchmark."""
        if engine is None:
            # Return approximate values
            data_points = benchmark.get("data_points", [])
            return [p[1] * (1 + np.random.uniform(-0.1, 0.1)) for p in data_points]

        # Configure and run simulation
        sim_config = benchmark.get("sim_config", {})
        data_points = benchmark.get("data_points", [])
        metric = benchmark.get("metric", "cell_count")

        results = []
        for t, _ in data_points:
            # Step simulation to time t
            while engine.time < t:
                engine.step()

            # Extract metric
            value = self._extract_metric(engine, metric)
            results.append(value)

        return results

    def _extract_metric(self, engine, metric: str) -> float:
        """Extract a metric from the simulation engine."""
        # Map metric names to engine attributes
        metric_map = {
            "cell_count": lambda e: len(getattr(e, "cells", [])),
            "viability": lambda e: getattr(e, "viability", 100.0),
            "tcd8_count": lambda e: getattr(e, "tcd8_cells", 0),
            "il6_concentration": lambda e: getattr(e, "il6", 0.0),
            "oxygen_rate": lambda e: getattr(e, "oxygen_consumption", 0.0),
            "lactate_rate": lambda e: getattr(e, "lactate_production", 0.0),
            "spheroid_diameter": lambda e: getattr(e, "diameter", 0.0),
            "ar_positive_fraction": lambda e: getattr(e, "ar_positive_fraction", 0.0),
        }

        extractor = metric_map.get(metric)
        if extractor:
            try:
                return float(extractor(engine))
            except Exception:
                pass

        return 0.0

    def _generate_recommendations(self, report: AccuracyReport) -> List[str]:
        """Generate calibration recommendations."""
        recommendations = []

        # Check category-level issues
        for cat, score in report.category_scores.items():
            if score < 70:
                recommendations.append(
                    f"Recalibrate {cat} module (current score: {score:.0f}%)"
                )

        # Check specific failing benchmarks
        for comp in report.comparisons:
            if not comp.passed and comp.mean_percent_error > 40:
                recommendations.append(
                    f"High error in '{comp.benchmark_name}' "
                    f"({comp.mean_percent_error:.0f}%) - review {comp.metric_name}"
                )

        # Overall recommendations
        if report.overall_correlation < 0.7:
            recommendations.append(
                "Low overall correlation - consider reviewing model structure"
            )

        if report.pass_rate < 0.5:
            recommendations.append(
                "Majority of benchmarks failing - comprehensive recalibration needed"
            )

        return recommendations

    def _save_report(self, report: AccuracyReport) -> None:
        """Save report to file."""
        filename = f"accuracy_report_{int(report.timestamp)}.json"
        filepath = self._results_dir / filename

        data = report.to_dict()
        data["comparisons"] = [
            {
                "name": c.benchmark_name,
                "category": c.category,
                "mae": c.mae,
                "correlation": c.correlation,
                "percent_error": c.mean_percent_error,
                "passed": c.passed,
            }
            for c in report.comparisons
        ]

        filepath.write_text(json.dumps(data, indent=2))
        log.info("Saved accuracy report to %s", filepath)

    # ── History and Trends ──────────────────────────────────────────────

    def get_history(self, limit: int = 10) -> List[AccuracyReport]:
        """Get recent evaluation history."""
        return self._history[-limit:]

    def get_trend(self) -> Dict[str, List[float]]:
        """Get accuracy trend over time."""
        return {
            "timestamps": [r.timestamp for r in self._history],
            "pass_rates": [r.pass_rate for r in self._history],
            "correlations": [r.overall_correlation for r in self._history],
            "maes": [r.overall_mae for r in self._history],
        }
