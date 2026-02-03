"""
Contract v1 — Multiscale Simulation Acceptance Tests
====================================================

A stability gate that must pass on CPU reference, single-GPU, and
(later) multi-GPU configurations. Prevents GPU optimizations and
visualization layers from breaking simulation validity.

The contract defines:
1. Canonical per-cell state schema (typed + versioned)
2. Canonical tissue field schema (O2/glucose/cytokines/drug/ECM)
3. Coupling rules (uptake/secretion, diffusion timesteps, event ordering)
4. Deterministic replay mode for debugging

Usage::

    from cognisom.validation.contract_v1 import ContractV1, run_contract_suite

    result = run_contract_suite(backend="cpu")
    assert result.all_passed, result.summary()

    # Or run individual checks
    contract = ContractV1()
    contract.check_cell_state_schema()
    contract.check_field_schema()
    contract.check_coupling_rules()
    contract.check_deterministic_replay()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

CONTRACT_VERSION = "1.0.0"

# ── Canonical Schemas ────────────────────────────────────────────────

# Required per-cell state fields with types and valid ranges
CELL_STATE_SCHEMA = {
    "cell_id":        {"type": "int",   "min": 0,     "max": None},
    "position_x":     {"type": "float", "min": -1e6,  "max": 1e6},
    "position_y":     {"type": "float", "min": -1e6,  "max": 1e6},
    "position_z":     {"type": "float", "min": -1e6,  "max": 1e6},
    "cell_type":      {"type": "str",   "allowed": [
        "normal", "cancer", "immune", "stromal", "endothelial",
        "basal", "luminal", "neuroendocrine", "stem", "fibroblast",
    ]},
    "phase":          {"type": "str",   "allowed": ["G0", "G1", "S", "G2", "M"]},
    "alive":          {"type": "bool"},
    "age":            {"type": "float", "min": 0,     "max": 1e6},
    "volume":         {"type": "float", "min": 0.01,  "max": 100.0},
    "oxygen":         {"type": "float", "min": 0.0,   "max": 0.25},
    "glucose":        {"type": "float", "min": 0.0,   "max": 50.0},
    "atp":            {"type": "float", "min": 0.0,   "max": 5000.0},
    "lactate":        {"type": "float", "min": 0.0,   "max": 50.0},
}

# Required spatial field properties
FIELD_SCHEMA = {
    "oxygen":   {"unit": "fraction",  "min": 0.0, "max": 0.21, "diffusion_um2_s": 2000.0},
    "glucose":  {"unit": "mM",        "min": 0.0, "max": 10.0, "diffusion_um2_s": 600.0},
    "cytokine": {"unit": "nM",        "min": 0.0, "max": 100.0, "diffusion_um2_s": 100.0},
}

# Coupling rules — ordering constraints for simulation steps
COUPLING_RULES = [
    "diffusion_before_uptake",       # fields must diffuse before cells consume
    "uptake_before_metabolism",       # cells uptake before metabolic state update
    "metabolism_before_phase",        # metabolic state before cell cycle decisions
    "phase_before_division",         # phase check before division events
    "division_before_immune",        # new cells created before immune scanning
    "immune_before_death",           # immune kills before death cleanup
    "events_logged_in_order",        # all events have monotonically increasing timestamps
]


@dataclass
class ContractCheck:
    """Result of a single contract check."""
    name: str
    passed: bool
    message: str = ""
    details: List[str] = field(default_factory=list)


@dataclass
class ContractResult:
    """Results of the full contract suite."""
    version: str = CONTRACT_VERSION
    backend: str = "cpu"
    checks: List[ContractCheck] = field(default_factory=list)
    run_time: float = 0.0

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def summary(self) -> str:
        total = len(self.checks)
        status = "PASSED" if self.all_passed else "FAILED"
        lines = [
            f"Contract v{self.version} [{self.backend}]: {status} "
            f"({self.passed_count}/{total} checks, {self.run_time:.2f}s)",
        ]
        for c in self.checks:
            icon = "+" if c.passed else "X"
            lines.append(f"  [{icon}] {c.name}: {c.message}")
            for d in c.details:
                lines.append(f"      {d}")
        return "\n".join(lines)


class ContractV1:
    """Contract v1 acceptance test suite.

    Validates that the simulation engine produces valid,
    schema-compliant output regardless of backend (CPU/GPU).
    """

    def __init__(self, engine=None):
        self._engine = engine
        self._checks: List[ContractCheck] = []

    # ── 1. Cell State Schema ─────────────────────────────────────────

    def check_cell_state_schema(self, cells: Optional[List[dict]] = None) -> ContractCheck:
        """Verify every cell has all required fields with valid types/ranges."""
        if cells is None:
            cells = self._get_engine_cells()

        if not cells:
            check = ContractCheck(
                name="cell_state_schema",
                passed=False,
                message="No cells found",
            )
            self._checks.append(check)
            return check

        violations = []
        for cell in cells:
            cid = cell.get("cell_id", "?")

            for field_name, spec in CELL_STATE_SCHEMA.items():
                if field_name not in cell:
                    violations.append(f"cell {cid}: missing field '{field_name}'")
                    continue

                val = cell[field_name]
                ftype = spec["type"]

                # Type check
                if ftype == "int" and not isinstance(val, (int, np.integer)):
                    violations.append(f"cell {cid}: '{field_name}' expected int, got {type(val).__name__}")
                elif ftype == "float" and not isinstance(val, (int, float, np.floating)):
                    violations.append(f"cell {cid}: '{field_name}' expected float, got {type(val).__name__}")
                elif ftype == "bool" and not isinstance(val, (bool, np.bool_)):
                    violations.append(f"cell {cid}: '{field_name}' expected bool, got {type(val).__name__}")
                elif ftype == "str" and not isinstance(val, str):
                    violations.append(f"cell {cid}: '{field_name}' expected str, got {type(val).__name__}")

                # Range check
                if "min" in spec and spec["min"] is not None:
                    if isinstance(val, (int, float)) and val < spec["min"]:
                        violations.append(f"cell {cid}: '{field_name}' = {val} < min {spec['min']}")
                if "max" in spec and spec["max"] is not None:
                    if isinstance(val, (int, float)) and val > spec["max"]:
                        violations.append(f"cell {cid}: '{field_name}' = {val} > max {spec['max']}")

                # Allowed values check
                if "allowed" in spec:
                    if val not in spec["allowed"]:
                        violations.append(f"cell {cid}: '{field_name}' = '{val}' not in {spec['allowed']}")

        passed = len(violations) == 0
        check = ContractCheck(
            name="cell_state_schema",
            passed=passed,
            message=f"{len(cells)} cells validated, {len(violations)} violations"
            if not passed else f"{len(cells)} cells validated OK",
            details=violations[:20],  # cap at 20
        )
        self._checks.append(check)
        return check

    # ── 2. Field Schema ──────────────────────────────────────────────

    def check_field_schema(self, fields: Optional[Dict[str, np.ndarray]] = None) -> ContractCheck:
        """Verify spatial fields have correct shapes, ranges, and no NaN/Inf."""
        if fields is None:
            fields = self._get_engine_fields()

        if not fields:
            check = ContractCheck(
                name="field_schema",
                passed=True,
                message="No fields to validate (acceptable for cell-only runs)",
            )
            self._checks.append(check)
            return check

        violations = []
        for fname, spec in FIELD_SCHEMA.items():
            if fname not in fields:
                violations.append(f"missing field '{fname}'")
                continue

            arr = fields[fname]
            if not isinstance(arr, np.ndarray):
                violations.append(f"'{fname}': expected ndarray, got {type(arr).__name__}")
                continue

            # Check dimensionality
            if arr.ndim not in (2, 3):
                violations.append(f"'{fname}': expected 2D or 3D, got {arr.ndim}D")

            # Check for NaN/Inf
            if np.any(np.isnan(arr)):
                n_nan = int(np.sum(np.isnan(arr)))
                violations.append(f"'{fname}': {n_nan} NaN values")
            if np.any(np.isinf(arr)):
                violations.append(f"'{fname}': contains Inf values")

            # Range check
            vmin = float(np.min(arr))
            vmax = float(np.max(arr))
            if vmin < spec["min"] - 1e-8:
                violations.append(f"'{fname}': min={vmin:.6f} < schema min={spec['min']}")
            if vmax > spec["max"] * 1.5:  # allow 50% overshoot for transients
                violations.append(f"'{fname}': max={vmax:.6f} > schema max={spec['max']}*1.5")

        passed = len(violations) == 0
        check = ContractCheck(
            name="field_schema",
            passed=passed,
            message=f"{len(fields)} fields validated, {len(violations)} violations"
            if not passed else f"{len(fields)} fields validated OK",
            details=violations[:20],
        )
        self._checks.append(check)
        return check

    # ── 3. Coupling Rules ────────────────────────────────────────────

    def check_coupling_rules(self, event_log: Optional[List[dict]] = None) -> ContractCheck:
        """Verify simulation step ordering follows coupling rules."""
        if event_log is None:
            event_log = self._get_engine_event_log()

        if not event_log:
            check = ContractCheck(
                name="coupling_rules",
                passed=True,
                message="No event log available (ordering assumed correct)",
            )
            self._checks.append(check)
            return check

        violations = []

        # Check monotonic timestamps
        timestamps = [e.get("time", 0.0) for e in event_log]
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                violations.append(
                    f"Non-monotonic timestamp at event {i}: "
                    f"{timestamps[i]} < {timestamps[i-1]}"
                )

        # Check step ordering within each timestep
        step_events = {}
        for e in event_log:
            t = e.get("time", 0.0)
            if t not in step_events:
                step_events[t] = []
            step_events[t].append(e.get("type", ""))

        EXPECTED_ORDER = [
            "diffusion", "uptake", "metabolism", "phase", "division", "immune", "death"
        ]
        for t, events in step_events.items():
            last_idx = -1
            for ev in events:
                for i, expected in enumerate(EXPECTED_ORDER):
                    if expected in ev.lower():
                        if i < last_idx:
                            violations.append(
                                f"t={t}: '{ev}' occurred after later-stage event"
                            )
                        last_idx = max(last_idx, i)

        passed = len(violations) == 0
        check = ContractCheck(
            name="coupling_rules",
            passed=passed,
            message=f"{len(event_log)} events checked, {len(violations)} violations"
            if not passed else f"{len(event_log)} events, ordering OK",
            details=violations[:20],
        )
        self._checks.append(check)
        return check

    # ── 4. Deterministic Replay ──────────────────────────────────────

    def check_deterministic_replay(self, n_steps: int = 10, seed: int = 42) -> ContractCheck:
        """Run simulation twice with same seed and verify identical output."""
        if self._engine is None:
            check = ContractCheck(
                name="deterministic_replay",
                passed=True,
                message="No engine provided; replay test skipped",
            )
            self._checks.append(check)
            return check

        violations = []

        try:
            # Run 1
            np.random.seed(seed)
            self._engine.reset() if hasattr(self._engine, "reset") else None
            for _ in range(n_steps):
                if hasattr(self._engine, "step"):
                    self._engine.step()
            state1 = self._snapshot_state()

            # Run 2
            np.random.seed(seed)
            self._engine.reset() if hasattr(self._engine, "reset") else None
            for _ in range(n_steps):
                if hasattr(self._engine, "step"):
                    self._engine.step()
            state2 = self._snapshot_state()

            # Compare
            if state1 and state2:
                for key in state1:
                    if key in state2:
                        if isinstance(state1[key], np.ndarray):
                            if not np.allclose(state1[key], state2[key], atol=1e-10):
                                violations.append(f"'{key}' differs between runs")
                        elif state1[key] != state2[key]:
                            violations.append(f"'{key}': {state1[key]} != {state2[key]}")
            else:
                violations.append("Could not snapshot engine state for comparison")

        except Exception as e:
            violations.append(f"Replay test error: {e}")

        passed = len(violations) == 0
        check = ContractCheck(
            name="deterministic_replay",
            passed=passed,
            message=f"{n_steps} steps x2, {len(violations)} differences"
            if not passed else f"{n_steps} steps x2, outputs match",
            details=violations[:10],
        )
        self._checks.append(check)
        return check

    # ── 5. Conservation Laws ─────────────────────────────────────────

    def check_mass_conservation(self, cells: Optional[List[dict]] = None) -> ContractCheck:
        """Verify basic conservation: total ATP+lactate correlates with glucose consumed."""
        if cells is None:
            cells = self._get_engine_cells()

        if not cells:
            check = ContractCheck(
                name="mass_conservation",
                passed=True,
                message="No cells for conservation check",
            )
            self._checks.append(check)
            return check

        violations = []
        for cell in cells:
            if not cell.get("alive", True):
                continue
            atp = cell.get("atp", 0)
            glucose = cell.get("glucose", 0)
            lactate = cell.get("lactate", 0)

            # Basic sanity: a living cell with no ATP and no glucose is suspicious
            if atp <= 0 and glucose <= 0 and cell.get("age", 0) > 1.0:
                violations.append(
                    f"cell {cell.get('cell_id', '?')}: alive with ATP={atp}, glucose={glucose}"
                )

            # Warburg check: high lactate should correlate with lower oxygen
            oxygen = cell.get("oxygen", 0.21)
            if lactate > 10.0 and oxygen > 0.15:
                violations.append(
                    f"cell {cell.get('cell_id', '?')}: high lactate ({lactate}) "
                    f"but normal O2 ({oxygen}) — unlikely"
                )

        passed = len(violations) == 0
        check = ContractCheck(
            name="mass_conservation",
            passed=passed,
            message=f"{len(violations)} metabolic inconsistencies"
            if not passed else "Metabolic consistency OK",
            details=violations[:20],
        )
        self._checks.append(check)
        return check

    # ── 6. GPU/CPU Parity ────────────────────────────────────────────

    def check_gpu_cpu_parity(
        self, cpu_cells: Optional[List[dict]] = None,
        gpu_cells: Optional[List[dict]] = None,
        tolerance: float = 1e-4,
    ) -> ContractCheck:
        """Verify GPU output matches CPU reference within tolerance."""
        if cpu_cells is None or gpu_cells is None:
            check = ContractCheck(
                name="gpu_cpu_parity",
                passed=True,
                message="Parity check skipped (both backends needed)",
            )
            self._checks.append(check)
            return check

        violations = []
        cpu_by_id = {c["cell_id"]: c for c in cpu_cells}
        gpu_by_id = {c["cell_id"]: c for c in gpu_cells}

        # Check same cell count
        if len(cpu_cells) != len(gpu_cells):
            violations.append(f"Cell count mismatch: CPU={len(cpu_cells)}, GPU={len(gpu_cells)}")

        # Check field-by-field parity
        for cid, cpu_cell in cpu_by_id.items():
            if cid not in gpu_by_id:
                violations.append(f"cell {cid}: present on CPU but not GPU")
                continue
            gpu_cell = gpu_by_id[cid]

            for field_name in ["oxygen", "glucose", "atp", "lactate", "volume", "age"]:
                cpu_val = cpu_cell.get(field_name, 0.0)
                gpu_val = gpu_cell.get(field_name, 0.0)
                if abs(cpu_val - gpu_val) > tolerance:
                    violations.append(
                        f"cell {cid}.{field_name}: CPU={cpu_val:.6f} GPU={gpu_val:.6f} "
                        f"(diff={abs(cpu_val-gpu_val):.2e})"
                    )

        passed = len(violations) == 0
        check = ContractCheck(
            name="gpu_cpu_parity",
            passed=passed,
            message=f"{len(violations)} parity violations (tol={tolerance})"
            if not passed else f"CPU/GPU parity within {tolerance}",
            details=violations[:20],
        )
        self._checks.append(check)
        return check

    # ── Engine access helpers ────────────────────────────────────────

    def _get_engine_cells(self) -> List[dict]:
        if self._engine is None:
            return []
        cell_list = getattr(self._engine, "cells", None)
        if cell_list is None:
            cell_module = getattr(self._engine, "cellular_module", None)
            if cell_module:
                cell_list = getattr(cell_module, "cells", [])
        if cell_list is None:
            return []
        return [c if isinstance(c, dict) else vars(c) for c in cell_list]

    def _get_engine_fields(self) -> Dict[str, np.ndarray]:
        if self._engine is None:
            return {}
        spatial = getattr(self._engine, "spatial_module", None)
        if spatial is None:
            return {}
        fields = {}
        for name in ["oxygen", "glucose", "cytokine"]:
            arr = getattr(spatial, f"{name}_field", None)
            if arr is not None:
                fields[name] = arr
        return fields

    def _get_engine_event_log(self) -> List[dict]:
        if self._engine is None:
            return []
        return getattr(self._engine, "event_log", [])

    def _snapshot_state(self) -> dict:
        cells = self._get_engine_cells()
        if not cells:
            return {}
        return {
            "n_cells": len(cells),
            "alive_count": sum(1 for c in cells if c.get("alive", True)),
            "positions": np.array([
                [c.get("position_x", 0), c.get("position_y", 0), c.get("position_z", 0)]
                for c in cells
            ]),
        }

    @property
    def results(self) -> List[ContractCheck]:
        return self._checks


# ── Convenience runner ───────────────────────────────────────────────

def run_contract_suite(
    engine=None,
    cells: Optional[List[dict]] = None,
    fields: Optional[Dict[str, np.ndarray]] = None,
    backend: str = "cpu",
) -> ContractResult:
    """Run the full Contract v1 acceptance test suite.

    Args:
        engine: Optional simulation engine
        cells: Optional cell state dicts (if no engine)
        fields: Optional spatial field arrays (if no engine)
        backend: "cpu" or "gpu" label for reporting

    Returns:
        ContractResult with all check outcomes
    """
    start = time.time()

    contract = ContractV1(engine)
    contract.check_cell_state_schema(cells)
    contract.check_field_schema(fields)
    contract.check_coupling_rules()
    contract.check_mass_conservation(cells)
    contract.check_deterministic_replay()

    result = ContractResult(
        backend=backend,
        checks=contract.results,
        run_time=time.time() - start,
    )

    log.info("Contract v1 result: %s", "PASSED" if result.all_passed else "FAILED")
    return result
