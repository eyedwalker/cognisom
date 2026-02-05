"""
Parameter Calibrator
====================

Bayesian-inspired parameter calibration that fits simulation parameters
to match experimental benchmark data. Uses scipy.optimize for the
optimization loop and supports multiple fitting strategies.
"""

import time
import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np

# Scipy imports - optional for environments with numpy version conflicts
try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError as e:
    SCIPY_AVAILABLE = False
    minimize = None
    differential_evolution = None
    import warnings
    warnings.warn(f"scipy.optimize not available: {e}. Calibration will be limited.")

from .benchmarks import BenchmarkDataset

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpec:
    """Definition of a tunable simulation parameter."""

    name: str  # e.g., 'division_time_cancer'
    module: str  # e.g., 'cellular'
    min_val: float
    max_val: float
    initial: float
    description: str = ""
    units: str = ""

    @property
    def bounds(self) -> Tuple[float, float]:
        return (self.min_val, self.max_val)


@dataclass
class CalibrationResult:
    """Result of a parameter calibration run."""

    benchmark_name: str
    parameters: Dict[str, float]  # name -> optimized value
    initial_error: float
    final_error: float
    improvement_pct: float
    iterations: int
    converged: bool
    elapsed_sec: float
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "parameters": {k: round(v, 6) for k, v in self.parameters.items()},
            "initial_error": round(self.initial_error, 3),
            "final_error": round(self.final_error, 3),
            "improvement_pct": round(self.improvement_pct, 1),
            "iterations": self.iterations,
            "converged": self.converged,
            "elapsed_sec": round(self.elapsed_sec, 2),
        }


# Pre-defined tunable parameters for each module
DEFAULT_PARAMETERS: List[ParameterSpec] = [
    # Cellular module
    ParameterSpec(
        name="division_time_cancer",
        module="cellular",
        min_val=8.0, max_val=120.0, initial=12.0,
        description="Cancer cell division time",
        units="hours",
    ),
    ParameterSpec(
        name="division_time_normal",
        module="cellular",
        min_val=18.0, max_val=72.0, initial=24.0,
        description="Normal cell division time",
        units="hours",
    ),
    ParameterSpec(
        name="glucose_consumption",
        module="cellular",
        min_val=0.05, max_val=1.0, initial=0.2,
        description="Normal cell glucose consumption rate",
        units="mmol/h",
    ),
    ParameterSpec(
        name="glucose_consumption_cancer",
        module="cellular",
        min_val=0.1, max_val=2.0, initial=0.5,
        description="Cancer cell glucose consumption rate",
        units="mmol/h",
    ),
    # Immune module
    ParameterSpec(
        name="nk_kill_probability",
        module="immune",
        min_val=0.01, max_val=0.5, initial=0.1,
        description="NK cell per-encounter kill probability",
        units="probability",
    ),
    ParameterSpec(
        name="t_cell_speed",
        module="immune",
        min_val=1.0, max_val=20.0, initial=5.0,
        description="T cell migration speed in tissue",
        units="um/min",
    ),
    ParameterSpec(
        name="exhaustion_rate",
        module="immune",
        min_val=0.001, max_val=0.1, initial=0.02,
        description="Rate of T cell exhaustion marker increase",
        units="per_hour",
    ),
    # Vascular module
    ParameterSpec(
        name="o2_diffusion_rate",
        module="vascular",
        min_val=0.5, max_val=5.0, initial=2.0,
        description="Oxygen diffusion coefficient in tissue",
        units="um2/s",
    ),
    ParameterSpec(
        name="vegf_threshold",
        module="vascular",
        min_val=0.01, max_val=0.5, initial=0.1,
        description="VEGF concentration threshold for angiogenesis",
        units="nM",
    ),
]


class ParameterCalibrator:
    """Calibrate simulation parameters against experimental data.

    Supports two optimization strategies:
    - Nelder-Mead (fast, local search, good for fine-tuning)
    - Differential Evolution (slower, global search, better for exploration)

    Example:
        calibrator = ParameterCalibrator()
        result = calibrator.calibrate(
            benchmark=my_benchmark,
            parameters=[param1, param2],
            method="differential_evolution",
        )
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("data/calibration")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._eval_count = 0

    def calibrate(
        self,
        benchmark: BenchmarkDataset,
        parameters: Optional[List[ParameterSpec]] = None,
        method: str = "nelder-mead",
        max_iterations: int = 100,
        progress_cb: Optional[Callable] = None,
    ) -> CalibrationResult:
        """Run parameter calibration for a benchmark.

        Args:
            benchmark: The experimental data to fit against.
            parameters: Parameters to tune (defaults filtered by benchmark.module).
            method: 'nelder-mead' or 'differential_evolution'.
            max_iterations: Maximum optimization iterations.
            progress_cb: Callback(iteration, max_iter, current_error).

        Returns:
            CalibrationResult with optimized parameter values.
        """
        start = time.time()
        self._eval_count = 0
        history = []

        # Check if scipy is available
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, returning empty calibration result")
            return CalibrationResult(
                benchmark_name=benchmark.name,
                parameters={},
                initial_error=0.0,
                final_error=0.0,
                improvement_pct=0.0,
                iterations=0,
                converged=False,
                elapsed_sec=time.time() - start,
            )

        # Select parameters relevant to this benchmark's module
        if parameters is None:
            parameters = [p for p in DEFAULT_PARAMETERS if p.module == benchmark.module]

        if not parameters:
            logger.warning(f"No tunable parameters for module '{benchmark.module}'")
            return CalibrationResult(
                benchmark_name=benchmark.name,
                parameters={},
                initial_error=0.0,
                final_error=0.0,
                improvement_pct=0.0,
                iterations=0,
                converged=False,
                elapsed_sec=time.time() - start,
            )

        # Initial values and bounds
        x0 = np.array([p.initial for p in parameters])
        bounds = [p.bounds for p in parameters]

        # Compute initial error
        initial_error = self._objective(x0, parameters, benchmark)

        # Objective with history tracking
        def tracked_objective(x):
            err = self._objective(x, parameters, benchmark)
            self._eval_count += 1
            entry = {
                "iteration": self._eval_count,
                "error": float(err),
                "params": {p.name: float(v) for p, v in zip(parameters, x)},
            }
            history.append(entry)
            if progress_cb and self._eval_count % 5 == 0:
                progress_cb(self._eval_count, max_iterations, err)
            return err

        # Optimize
        if method == "differential_evolution":
            result = differential_evolution(
                tracked_objective,
                bounds=bounds,
                maxiter=max_iterations,
                seed=42,
                tol=1e-4,
                polish=True,
            )
            converged = result.success
            x_opt = result.x
            final_error = result.fun
            n_iter = result.nit
        else:
            result = minimize(
                tracked_objective,
                x0=x0,
                method="Nelder-Mead",
                options={"maxiter": max_iterations, "xatol": 1e-4, "fatol": 1e-4},
            )
            converged = result.success
            x_opt = result.x
            final_error = result.fun
            n_iter = result.nit

        # Clamp to bounds
        for i, (lo, hi) in enumerate(bounds):
            x_opt[i] = np.clip(x_opt[i], lo, hi)

        opt_params = {p.name: float(v) for p, v in zip(parameters, x_opt)}

        improvement = 0.0
        if initial_error > 0:
            improvement = (1.0 - final_error / initial_error) * 100

        elapsed = time.time() - start

        cal_result = CalibrationResult(
            benchmark_name=benchmark.name,
            parameters=opt_params,
            initial_error=float(initial_error),
            final_error=float(final_error),
            improvement_pct=float(improvement),
            iterations=int(n_iter),
            converged=converged,
            elapsed_sec=elapsed,
            history=history,
        )

        # Save result
        result_path = self.output_dir / f"cal_{benchmark.name.replace(' ', '_')}.json"
        with open(result_path, "w") as f:
            json.dump(cal_result.to_dict(), f, indent=2)

        logger.info(
            f"Calibration complete: {benchmark.name} — "
            f"error {initial_error:.3f} -> {final_error:.3f} "
            f"({improvement:.1f}% improvement)"
        )

        return cal_result

    def _objective(self, x: np.ndarray, parameters: List[ParameterSpec],
                   benchmark: BenchmarkDataset) -> float:
        """Objective function: mean squared error between sim and experiment."""
        try:
            from cognisom.core.simulation_engine import SimulationEngine, SimulationConfig
        except ImportError:
            from core.simulation_engine import SimulationEngine, SimulationConfig

        # Build config from parameter values
        config = dict(benchmark.sim_config) if benchmark.sim_config else {}
        for param, val in zip(parameters, x):
            config[param.name] = float(val)

        duration = config.pop("duration", 24.0)
        dt = config.pop("dt", 0.1)
        sim_config = SimulationConfig(dt=dt, duration=duration)

        engine = SimulationEngine(sim_config)

        # Register module
        module_map = self._get_module_map()
        if benchmark.module in module_map:
            engine.register_module(benchmark.module, module_map[benchmark.module], config)

        engine.initialize()

        # Run and sample
        x_points = benchmark.x_values
        exp_values = benchmark.y_values
        sim_values = []

        if benchmark.x_units in ("hours", "days", "months"):
            multiplier = {"hours": 1.0, "days": 24.0, "months": 720.0}
            sample_times = x_points * multiplier[benchmark.x_units]

            for target_time in sample_times:
                steps_needed = max(0, int((target_time - engine.time) / dt))
                for _ in range(steps_needed):
                    engine.step()

                state = engine.get_state()
                module_state = state.get(benchmark.module, {})
                val = self._extract_simple(module_state, benchmark.metric)
                sim_values.append(val)
        else:
            # Non-time axis — use initial values
            sim_values = [0.0] * len(x_points)

        sim_arr = np.array(sim_values, dtype=float)
        exp_arr = np.array(exp_values, dtype=float)

        # Normalized MSE
        scale = np.maximum(np.abs(exp_arr), 1.0)
        mse = float(np.mean(((sim_arr - exp_arr) / scale) ** 2))
        return mse

    @staticmethod
    def _extract_simple(module_state: Dict, metric: str) -> float:
        """Quick metric extraction for optimization loop."""
        if metric in module_state:
            v = module_state[metric]
            return float(v) if isinstance(v, (int, float)) else 0.0

        if metric == "n_cancer_cells":
            return float(module_state.get("cancer_cells", module_state.get("n_cancer", 0)))
        if metric == "tumor_diameter":
            n = module_state.get("cancer_cells", module_state.get("n_cancer", 0))
            return float((n ** (1.0 / 3.0)) * 15.0)
        return 0.0

    @staticmethod
    def _get_module_map():
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
        return module_map

    @staticmethod
    def get_default_parameters(module: Optional[str] = None) -> List[ParameterSpec]:
        """Get default tunable parameters, optionally filtered by module."""
        if module:
            return [p for p in DEFAULT_PARAMETERS if p.module == module]
        return list(DEFAULT_PARAMETERS)
