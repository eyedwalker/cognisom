"""Run Manager — create, execute, and persist simulation runs.

Bridges the gap between scenarios (what to simulate) and persisted results
(what happened).  Each run freezes the scenario config at creation time,
executes via EngineRunner, serializes all artifacts, auto-evaluates
against benchmarks, and stores a lightweight summary in the EntityStore.

Usage::

    store = EntityStore()
    artifacts = ArtifactStore()
    mgr = RunManager(store, artifacts)

    run = mgr.create_run(scenario_id="...", random_seed=42)
    run = mgr.execute_run(run.entity_id, progress_callback=my_progress_fn)
    print(run.accuracy_grade, run.final_metrics)

    runs = mgr.list_runs(status="completed")
    comparison = mgr.compare_runs([run1.entity_id, run2.entity_id])
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import platform
import subprocess
import time
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from cognisom.library.models import (
    EntityType,
    RelationshipType,
    SimulationRun,
    SimulationScenario,
)
from cognisom.library.store import EntityStore

from .artifact_store import ArtifactStore

log = logging.getLogger(__name__)


class RunManager:
    """Create, execute, and persist simulation runs."""

    def __init__(self, store: EntityStore, artifact_store: ArtifactStore):
        self._store = store
        self._artifacts = artifact_store

    # ── Create ───────────────────────────────────────────────────────

    def create_run(
        self,
        scenario_id: str,
        overrides: Optional[Dict] = None,
        random_seed: Optional[int] = None,
        run_name: str = "",
        project_id: str = "",
    ) -> SimulationRun:
        """Create a new SimulationRun from a scenario.

        Freezes the scenario config (with optional overrides) and collects
        hardware/software metadata for reproducibility.

        Parameters
        ----------
        scenario_id
            ID of the SimulationScenario entity to run.
        overrides
            Optional dict of parameter overrides (module -> params dict).
        random_seed
            Random seed for reproducibility.  Auto-generated if None.
        run_name
            Human-readable name for the run.
        project_id
            Optional ResearchProject ID to link this run to.

        Returns
        -------
        SimulationRun
            The newly created (status=pending) run entity.
        """
        # Load scenario
        scenario = self._store.get_entity(scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario {scenario_id} not found")
        if not isinstance(scenario, SimulationScenario):
            raise TypeError(f"Entity {scenario_id} is not a SimulationScenario")

        # Freeze config
        config_snapshot = {
            "scenario_type": scenario.scenario_type,
            "grid_shape": scenario.grid_shape,
            "resolution_um": scenario.resolution_um,
            "cell_type_ids": scenario.cell_type_ids,
            "gene_ids": scenario.gene_ids,
            "drug_ids": scenario.drug_ids,
            "parameter_set_ids": scenario.parameter_set_ids,
            "initial_cell_counts": scenario.initial_cell_counts,
            "initial_field_values": scenario.initial_field_values,
        }
        if overrides:
            config_snapshot["overrides"] = overrides

        seed = random_seed if random_seed is not None else int(time.time() * 1000) % (2**31)

        run = SimulationRun(
            entity_id=str(uuid.uuid4()),
            name=run_name or f"Run {time.strftime('%Y-%m-%d %H:%M')}",
            entity_type=EntityType.SIMULATION_RUN,
            scenario_id=scenario_id,
            project_id=project_id,
            run_status="pending",
            config_snapshot=config_snapshot,
            modules_enabled={},  # Populated at execution
            dt=scenario.time_step_hours,
            duration_hours=scenario.duration_hours,
            random_seed=seed,
            hardware_info=self._detect_hardware(),
            software_versions=self._detect_software(),
            git_commit=self._detect_git_commit(),
            created_by="researcher",
        )
        run.artifacts_dir = str(self._artifacts.run_dir(run.entity_id))

        self._store.add_entity(run)

        # Create relationship: run -> scenario
        from cognisom.library.models import Relationship
        self._store.add_relationship(Relationship(
            source_id=run.entity_id,
            target_id=scenario_id,
            rel_type=RelationshipType.EXECUTES,
        ))

        # Link to project if specified
        if project_id:
            self._store.add_relationship(Relationship(
                source_id=run.entity_id,
                target_id=project_id,
                rel_type=RelationshipType.BELONGS_TO,
            ))

        log.info("Created run %s for scenario %s (seed=%d)", run.entity_id, scenario_id, seed)
        return run

    # ── Execute ──────────────────────────────────────────────────────

    def execute_run(
        self,
        run_id: str,
        modules_enabled: Optional[Dict[str, bool]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> SimulationRun:
        """Execute a pending simulation run and persist all results.

        Steps:
        1. Build EngineRunner from frozen config
        2. Run simulation with progress callback
        3. Serialize artifacts (CSV, NPZ, JSONL)
        4. Auto-evaluate against benchmarks
        5. Auto-generate figures
        6. Update run entity with results

        Parameters
        ----------
        run_id
            ID of a pending SimulationRun.
        modules_enabled
            Which modules to enable.  All enabled by default.
        progress_callback
            Called with (current_step, total_steps).

        Returns
        -------
        SimulationRun
            The updated run entity with status=completed (or failed).
        """
        run = self._get_run(run_id)
        if run.run_status != "pending":
            raise ValueError(f"Run {run_id} status is '{run.run_status}', expected 'pending'")

        # Update status
        run.run_status = "running"
        run.started_at = time.time()
        if modules_enabled:
            run.modules_enabled = modules_enabled
        self._update_run(run)

        try:
            # Seed for reproducibility
            np.random.seed(run.random_seed)

            # Build engine
            from cognisom.dashboard.engine_runner import EngineRunner
            runner = EngineRunner(
                dt=run.dt,
                duration=run.duration_hours,
                scenario="Baseline",  # Scenario preset name (config applied via overrides)
                modules_enabled=modules_enabled or {
                    "cellular": True, "immune": True, "vascular": True,
                    "lymphatic": True, "molecular": True, "spatial": True,
                    "epigenetic": True, "circadian": True, "morphogen": True,
                },
                overrides=run.config_snapshot.get("overrides"),
            )
            run.modules_enabled = runner.modules_enabled

            runner.build()
            runner.run(progress_callback=progress_callback)

            # Serialize artifacts
            ts_data = runner.get_time_series()
            self._artifacts.save_time_series(run.entity_id, ts_data)

            cell_snaps = runner.serialize_cell_snapshots()
            if cell_snaps:
                self._artifacts.save_cell_snapshots(run.entity_id, cell_snaps)

            self._artifacts.save_events(run.entity_id, runner.event_log)

            # Extract final metrics and event summary
            run.final_metrics = runner.get_final_metrics()
            run.event_summary = runner.get_event_summary()

            # Auto-evaluate
            self._auto_evaluate(run, runner)

            # Auto-generate figures
            try:
                from cognisom.workflow.figure_generator import FigureGenerator
                fig_gen = FigureGenerator()
                fig_gen.generate_all(run.entity_id, self._artifacts)
            except Exception as e:
                log.warning("Figure generation failed (non-fatal): %s", e)

            # Mark complete
            run.run_status = "completed"
            run.completed_at = time.time()
            run.elapsed_seconds = run.completed_at - run.started_at

        except Exception as e:
            log.error("Run %s failed: %s", run_id, e)
            run.run_status = "failed"
            run.completed_at = time.time()
            run.elapsed_seconds = run.completed_at - run.started_at
            run.final_metrics["error"] = str(e)

        self._update_run(run)
        log.info(
            "Run %s %s in %.1fs (accuracy=%s, fidelity=%s)",
            run_id, run.run_status, run.elapsed_seconds,
            run.accuracy_grade or "N/A", run.fidelity_grade or "N/A",
        )
        return run

    # ── Query ────────────────────────────────────────────────────────

    def get_run(self, run_id: str) -> Optional[SimulationRun]:
        """Get a run by ID, or None if not found."""
        entity = self._store.get_entity(run_id)
        if isinstance(entity, SimulationRun):
            return entity
        return None

    def list_runs(
        self,
        scenario_id: Optional[str] = None,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[SimulationRun]:
        """List simulation runs, optionally filtered."""
        results, _ = self._store.search(
            query="",
            entity_type=EntityType.SIMULATION_RUN.value,
            limit=limit,
        )
        runs = [r for r in results if isinstance(r, SimulationRun)]

        if scenario_id:
            runs = [r for r in runs if r.scenario_id == scenario_id]
        if project_id:
            runs = [r for r in runs if r.project_id == project_id]
        if status:
            runs = [r for r in runs if r.run_status == status]

        # Sort by creation time, newest first
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs side-by-side.

        Returns a dict with:
        - ``runs``: list of run summaries
        - ``metrics_table``: dict of metric_name -> [value_per_run]
        - ``deltas``: dict of metric_name -> [delta_from_first_run]
        """
        runs = []
        for rid in run_ids:
            run = self.get_run(rid)
            if run:
                runs.append(run)

        if len(runs) < 2:
            return {"runs": [], "metrics_table": {}, "deltas": {}}

        # Collect all metric keys
        all_keys = set()
        for run in runs:
            all_keys.update(run.final_metrics.keys())

        # Build comparison table
        metrics_table: Dict[str, list] = {}
        for key in sorted(all_keys):
            metrics_table[key] = [run.final_metrics.get(key) for run in runs]

        # Calculate deltas from first run
        deltas: Dict[str, list] = {}
        for key, values in metrics_table.items():
            base = values[0]
            if isinstance(base, (int, float)):
                deltas[key] = [
                    round(v - base, 4) if isinstance(v, (int, float)) else None
                    for v in values
                ]

        return {
            "runs": [
                {
                    "run_id": r.entity_id,
                    "name": r.name,
                    "scenario_id": r.scenario_id,
                    "status": r.run_status,
                    "elapsed_seconds": r.elapsed_seconds,
                    "accuracy_grade": r.accuracy_grade,
                    "fidelity_grade": r.fidelity_grade,
                }
                for r in runs
            ],
            "metrics_table": metrics_table,
            "deltas": deltas,
        }

    # ── Delete ───────────────────────────────────────────────────────

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and its artifacts."""
        self._artifacts.cleanup_run(run_id)
        return self._store.hard_delete_entity(run_id)

    # ── Internal helpers ─────────────────────────────────────────────

    def _get_run(self, run_id: str) -> SimulationRun:
        entity = self._store.get_entity(run_id)
        if not isinstance(entity, SimulationRun):
            raise ValueError(f"Run {run_id} not found")
        return entity

    def _update_run(self, run: SimulationRun):
        self._store.update_entity(run, changed_by="run_manager")

    def _auto_evaluate(self, run: SimulationRun, runner):
        """Run accuracy and fidelity evaluation after simulation."""
        eval_report = {}

        try:
            from cognisom.eval.simulation_accuracy import SimulationEvaluator
            evaluator = SimulationEvaluator()
            accuracy = evaluator.evaluate_against_benchmarks(runner.engine)
            run.accuracy_grade = accuracy.overall_grade
            eval_report["accuracy"] = accuracy.to_dict()
        except Exception as e:
            log.warning("Accuracy evaluation failed: %s", e)

        try:
            from cognisom.eval.tissue_fidelity import TissueFidelityEvaluator
            evaluator = TissueFidelityEvaluator()
            state = runner.get_final_state()
            fidelity = evaluator.evaluate(state)
            run.fidelity_grade = fidelity.grade
            eval_report["fidelity"] = fidelity.to_dict() if hasattr(fidelity, "to_dict") else {}
        except Exception as e:
            log.warning("Fidelity evaluation failed: %s", e)

        if eval_report:
            self._artifacts.save_eval_report(run.entity_id, eval_report)

    def _detect_hardware(self) -> Dict[str, Any]:
        """Auto-detect hardware info for reproducibility."""
        info = {
            "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
        }
        # Try to detect GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                info["gpu"] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return info

    def _detect_software(self) -> Dict[str, str]:
        """Auto-detect Python package versions."""
        packages = ["numpy", "scipy", "matplotlib", "streamlit", "flask"]
        versions = {"python": platform.python_version()}
        for pkg in packages:
            try:
                versions[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                pass
        return versions

    def _detect_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=os.path.dirname(os.path.dirname(__file__)),
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return ""
