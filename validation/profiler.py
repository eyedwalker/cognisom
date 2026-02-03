"""
Simulation Profiler
===================

Per-module performance profiling for Cognisom simulations.
Tracks wall-clock time, step throughput, memory usage, and
identifies bottleneck modules.
"""

import time
import logging
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModuleProfile:
    """Profile data for a single simulation module."""

    name: str
    total_time_sec: float = 0.0
    step_count: int = 0
    min_step_ms: float = float("inf")
    max_step_ms: float = 0.0
    step_times_ms: List[float] = field(default_factory=list)

    @property
    def avg_step_ms(self) -> float:
        if not self.step_times_ms:
            return 0.0
        return float(np.mean(self.step_times_ms))

    @property
    def std_step_ms(self) -> float:
        if len(self.step_times_ms) < 2:
            return 0.0
        return float(np.std(self.step_times_ms))

    @property
    def p95_step_ms(self) -> float:
        if not self.step_times_ms:
            return 0.0
        return float(np.percentile(self.step_times_ms, 95))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_time_sec": round(self.total_time_sec, 4),
            "step_count": self.step_count,
            "avg_step_ms": round(self.avg_step_ms, 3),
            "min_step_ms": round(self.min_step_ms, 3) if self.min_step_ms != float("inf") else 0.0,
            "max_step_ms": round(self.max_step_ms, 3),
            "p95_step_ms": round(self.p95_step_ms, 3),
            "std_step_ms": round(self.std_step_ms, 3),
        }


@dataclass
class ProfileReport:
    """Complete profiling report for a simulation run."""

    total_time_sec: float
    total_steps: int
    modules: Dict[str, ModuleProfile]
    event_bus_time_sec: float = 0.0
    overhead_time_sec: float = 0.0
    memory_peak_mb: float = 0.0
    memory_current_mb: float = 0.0
    steps_per_second: float = 0.0
    bottleneck_module: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        module_data = {}
        for name, mp in self.modules.items():
            module_data[name] = mp.to_dict()
            # Add percentage of total
            if self.total_time_sec > 0:
                module_data[name]["pct_of_total"] = round(
                    mp.total_time_sec / self.total_time_sec * 100, 1
                )

        return {
            "total_time_sec": round(self.total_time_sec, 3),
            "total_steps": self.total_steps,
            "steps_per_second": round(self.steps_per_second, 1),
            "memory_peak_mb": round(self.memory_peak_mb, 1),
            "memory_current_mb": round(self.memory_current_mb, 1),
            "event_bus_time_sec": round(self.event_bus_time_sec, 4),
            "overhead_time_sec": round(self.overhead_time_sec, 4),
            "bottleneck_module": self.bottleneck_module,
            "modules": module_data,
            "config": self.config,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Simulation Profile: {self.total_steps} steps in {self.total_time_sec:.2f}s "
            f"({self.steps_per_second:.0f} steps/sec)",
            f"Memory: {self.memory_current_mb:.0f} MB current, {self.memory_peak_mb:.0f} MB peak",
            "",
            "Module Breakdown:",
        ]

        # Sort by time descending
        sorted_mods = sorted(
            self.modules.values(), key=lambda m: m.total_time_sec, reverse=True
        )
        for mp in sorted_mods:
            pct = mp.total_time_sec / max(self.total_time_sec, 0.001) * 100
            bar_len = int(pct / 2)
            bar = "#" * bar_len + "-" * (50 - bar_len)
            lines.append(
                f"  {mp.name:20s} {mp.avg_step_ms:8.2f} ms/step  "
                f"[{bar}] {pct:5.1f}%"
            )

        if self.event_bus_time_sec > 0:
            pct = self.event_bus_time_sec / max(self.total_time_sec, 0.001) * 100
            lines.append(f"  {'event_bus':20s} {'':>8s}           {pct:5.1f}%")

        lines.append("")
        lines.append(f"Bottleneck: {self.bottleneck_module}")
        return "\n".join(lines)


class SimulationProfiler:
    """Profile simulation performance at the module level.

    Wraps the SimulationEngine step loop to measure per-module
    execution time, memory usage, and throughput.

    Example:
        profiler = SimulationProfiler()
        report = profiler.profile(
            duration=24.0,
            modules=['cellular', 'immune', 'vascular'],
            n_cells=100,
        )
        print(report.summary())
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("data/profiling")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def profile(
        self,
        duration: float = 24.0,
        dt: float = 0.1,
        modules: Optional[List[str]] = None,
        module_configs: Optional[Dict[str, Dict]] = None,
        n_cells: int = 100,
        progress_cb=None,
    ) -> ProfileReport:
        """Run a profiled simulation.

        Args:
            duration: Simulation duration in hours.
            dt: Time step in hours.
            modules: Which modules to enable (None = all available).
            module_configs: Per-module config dicts.
            n_cells: Number of initial cells.
            progress_cb: Callback(step, total_steps, elapsed_sec).

        Returns:
            ProfileReport with per-module timing data.
        """
        try:
            from cognisom.core.simulation_engine import SimulationEngine, SimulationConfig
        except ImportError:
            from core.simulation_engine import SimulationEngine, SimulationConfig

        module_configs = module_configs or {}
        sim_config = SimulationConfig(dt=dt, duration=duration)

        engine = SimulationEngine(sim_config)

        # Register modules
        available = self._get_module_map()
        if modules is None:
            modules = list(available.keys())

        for mod_name in modules:
            if mod_name in available:
                cfg = module_configs.get(mod_name, {})
                if "n_normal_cells" not in cfg and "n_cancer_cells" not in cfg:
                    cfg.setdefault("n_normal_cells", max(1, n_cells * 4 // 5))
                    cfg.setdefault("n_cancer_cells", max(1, n_cells // 5))
                engine.register_module(mod_name, available[mod_name], cfg)

        engine.initialize()

        # Prepare profiling state
        total_steps = int(duration / dt)
        mod_profiles: Dict[str, ModuleProfile] = {
            name: ModuleProfile(name=name) for name in engine.modules
        }
        event_bus_total = 0.0

        # Get initial memory
        mem_before = self._get_memory_mb()

        # Profiled step loop
        wall_start = time.time()

        for step_i in range(total_steps):
            # Pre-step (not profiled individually)
            for name, mod in engine.modules.items():
                if mod.enabled:
                    mod.pre_step(dt)

            # Update (profiled per module)
            for name, mod in engine.modules.items():
                if not mod.enabled:
                    continue
                t0 = time.perf_counter()
                mod.update(dt)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                mp = mod_profiles[name]
                mp.step_times_ms.append(elapsed_ms)
                mp.total_time_sec += elapsed_ms / 1000
                mp.step_count += 1
                mp.min_step_ms = min(mp.min_step_ms, elapsed_ms)
                mp.max_step_ms = max(mp.max_step_ms, elapsed_ms)

            # Post-step
            for name, mod in engine.modules.items():
                if mod.enabled:
                    mod.post_step(dt)

            # Event bus
            t0 = time.perf_counter()
            engine.event_bus.process_events()
            event_bus_total += time.perf_counter() - t0

            engine.time += dt
            engine.step_count += 1

            if progress_cb and step_i % 50 == 0:
                progress_cb(step_i, total_steps, time.time() - wall_start)

        wall_total = time.time() - wall_start
        mem_after = self._get_memory_mb()

        # Calculate overhead
        module_total = sum(mp.total_time_sec for mp in mod_profiles.values())
        overhead = wall_total - module_total - event_bus_total

        # Find bottleneck
        bottleneck = max(mod_profiles.values(), key=lambda m: m.total_time_sec)

        report = ProfileReport(
            total_time_sec=wall_total,
            total_steps=total_steps,
            modules=mod_profiles,
            event_bus_time_sec=event_bus_total,
            overhead_time_sec=max(0, overhead),
            memory_peak_mb=max(mem_before, mem_after),
            memory_current_mb=mem_after,
            steps_per_second=total_steps / max(wall_total, 0.001),
            bottleneck_module=bottleneck.name,
            config={
                "duration": duration,
                "dt": dt,
                "modules": modules,
                "n_cells": n_cells,
            },
        )

        # Save report
        report_path = self.output_dir / "profile_report.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        return report

    @staticmethod
    def _get_memory_mb() -> float:
        """Get current process memory usage in MB."""
        try:
            import resource
            # maxrss is in bytes on Linux, KB on macOS
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if os.uname().sysname == "Darwin":
                return usage.ru_maxrss / (1024 * 1024)  # bytes -> MB
            return usage.ru_maxrss / 1024  # KB -> MB
        except Exception:
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
        try:
            from cognisom.modules.lymphatic_module import LymphaticModule
            module_map["lymphatic"] = LymphaticModule
        except ImportError:
            pass
        try:
            from cognisom.modules.spatial_module import SpatialModule
            module_map["spatial"] = SpatialModule
        except ImportError:
            pass
        try:
            from cognisom.modules.epigenetic_module import EpigeneticModule
            module_map["epigenetic"] = EpigeneticModule
        except ImportError:
            pass
        try:
            from cognisom.modules.circadian_module import CircadianModule
            module_map["circadian"] = CircadianModule
        except ImportError:
            pass
        try:
            from cognisom.modules.morphogen_module import MorphogenModule
            module_map["morphogen"] = MorphogenModule
        except ImportError:
            pass
        return module_map
