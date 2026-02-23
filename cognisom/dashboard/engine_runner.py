"""Engine Runner - Bridge between the 9-module simulation engine and the dashboard.

Provides:
- Easy engine setup with configurable modules and parameters
- Per-step time-series data collection for plotting
- Cell/immune/vessel position snapshots for 3D visualization
- Event log capture for timeline views
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import numpy as np

# Ensure project root is on path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cognisom.core import SimulationEngine, SimulationConfig, EventBus, EventTypes
from cognisom.modules import (
    MolecularModule, CellularModule, ImmuneModule,
    VascularModule, LymphaticModule, SpatialModule,
    EpigeneticModule, CircadianModule, MorphogenModule,
)


# ── Scenario presets ─────────────────────────────────────────────────

SCENARIOS = {
    "Baseline": {
        "desc": "Normal tissue with a small tumor seed. All 9 modules active.",
        "cellular": {"n_normal_cells": 80, "n_cancer_cells": 10},
        "immune": {"n_t_cells": 12, "n_nk_cells": 8, "n_macrophages": 6},
        "vascular": {"n_capillaries": 8},
        "lymphatic": {"n_vessels": 4, "metastasis_probability": 0.001},
    },
    "Aggressive Tumor": {
        "desc": "Fast-dividing tumor with reduced immune response.",
        "cellular": {
            "n_normal_cells": 60, "n_cancer_cells": 30,
            "division_time_cancer": 8.0,
        },
        "immune": {"n_t_cells": 6, "n_nk_cells": 4, "n_macrophages": 3},
        "vascular": {"n_capillaries": 6},
        "lymphatic": {"n_vessels": 4, "metastasis_probability": 0.005},
    },
    "Immunotherapy": {
        "desc": "Enhanced immune surveillance (anti-PD1 effect simulated via more T cells).",
        "cellular": {"n_normal_cells": 80, "n_cancer_cells": 15},
        "immune": {
            "n_t_cells": 30, "n_nk_cells": 15, "n_macrophages": 10,
            "kill_probability": 0.95,
        },
        "vascular": {"n_capillaries": 8},
        "lymphatic": {"n_vessels": 4, "metastasis_probability": 0.001},
    },
    "Hypoxia": {
        "desc": "Poor vasculature — tumor core becomes hypoxic quickly.",
        "cellular": {"n_normal_cells": 60, "n_cancer_cells": 20},
        "immune": {"n_t_cells": 10, "n_nk_cells": 6, "n_macrophages": 5},
        "vascular": {"n_capillaries": 3, "exchange_rate": 0.4},
        "lymphatic": {"n_vessels": 3, "metastasis_probability": 0.002},
    },
    "Metastasis Risk": {
        "desc": "Tumor near lymphatic vessels with elevated metastasis probability.",
        "cellular": {"n_normal_cells": 60, "n_cancer_cells": 25},
        "immune": {"n_t_cells": 8, "n_nk_cells": 5, "n_macrophages": 4},
        "vascular": {"n_capillaries": 6},
        "lymphatic": {"n_vessels": 6, "metastasis_probability": 0.01},
    },
}


# ── Data collection ──────────────────────────────────────────────────

@dataclass
class SimulationSnapshot:
    """One time-point of simulation data (for time-series plots)."""
    time: float = 0.0
    step: int = 0

    # Cellular
    n_cells: int = 0
    n_cancer: int = 0
    n_normal: int = 0
    total_divisions: int = 0
    total_deaths: int = 0
    total_transformations: int = 0
    avg_oxygen: float = 0.0
    avg_glucose: float = 0.0

    # Immune
    n_immune: int = 0
    n_activated: int = 0
    n_t_cells: int = 0
    n_nk_cells: int = 0
    n_macrophages: int = 0
    total_kills: int = 0

    # Vascular
    avg_cell_O2: float = 0.0
    avg_cell_glucose: float = 0.0
    hypoxic_regions: int = 0

    # Lymphatic
    total_metastases: int = 0
    immune_in_vessels: int = 0
    cancer_in_vessels: int = 0

    # Epigenetic
    avg_methylation: float = 0.0
    silenced_genes: int = 0

    # Circadian
    master_phase: float = 0.0
    synchrony: float = 0.0

    # Morphogen
    total_fates_determined: int = 0


@dataclass
class CellSnapshot:
    """Snapshot of all cell & immune positions for 3D viz."""
    cell_positions: np.ndarray = None   # (N, 3)
    cell_types: list = field(default_factory=list)         # 'normal' | 'cancer'
    cell_phases: list = field(default_factory=list)        # 'G1'/'S'/'G2'/'M'
    cell_oxygen: list = field(default_factory=list)
    cell_mhc1: list = field(default_factory=list)
    cell_ids: list = field(default_factory=list)

    immune_positions: np.ndarray = None  # (M, 3)
    immune_types: list = field(default_factory=list)       # 'T_cell' | 'NK_cell' | 'macrophage'
    immune_activated: list = field(default_factory=list)

    capillary_starts: list = field(default_factory=list)   # list of (x,y,z)
    capillary_ends: list = field(default_factory=list)

    lymph_starts: list = field(default_factory=list)
    lymph_ends: list = field(default_factory=list)


# ── Runner ───────────────────────────────────────────────────────────

class EngineRunner:
    """Thin wrapper that sets up, runs, and collects data from the real engine."""

    def __init__(
        self,
        dt: float = 0.05,
        duration: float = 6.0,
        scenario: str = "Baseline",
        modules_enabled: Optional[Dict[str, bool]] = None,
        overrides: Optional[Dict[str, Dict]] = None,
        record_interval: int = 10,
    ):
        self.dt = dt
        self.duration = duration
        self.scenario_name = scenario
        self.record_interval = record_interval  # record snapshot every N steps

        # Merge scenario defaults with any user overrides
        scenario_cfg = SCENARIOS.get(scenario, SCENARIOS["Baseline"])
        self.module_configs = {
            "cellular": dict(scenario_cfg.get("cellular", {})),
            "immune": dict(scenario_cfg.get("immune", {})),
            "vascular": dict(scenario_cfg.get("vascular", {})),
            "lymphatic": dict(scenario_cfg.get("lymphatic", {})),
            "molecular": {},
            "spatial": {},
            "epigenetic": {},
            "circadian": {},
            "morphogen": {},
        }
        if overrides:
            for mod, params in overrides.items():
                if mod in self.module_configs:
                    self.module_configs[mod].update(params)

        # Which modules to enable (all by default)
        self.modules_enabled = modules_enabled or {
            "cellular": True,
            "immune": True,
            "vascular": True,
            "lymphatic": True,
            "molecular": True,
            "spatial": True,
            "epigenetic": True,
            "circadian": True,
            "morphogen": True,
        }

        # Collected data
        self.history: List[SimulationSnapshot] = []
        self.cell_snapshots: List[CellSnapshot] = []
        self.event_log: List[dict] = []

        # Engine reference (populated on build)
        self.engine: Optional[SimulationEngine] = None

    # ── Setup ────────────────────────────────────────────────────────

    def build(self):
        """Instantiate engine, register and link modules."""
        import io
        from contextlib import redirect_stdout

        # Suppress the engine's print() output during build
        buf = io.StringIO()
        with redirect_stdout(buf):
            config = SimulationConfig(dt=self.dt, duration=self.duration)
            self.engine = SimulationEngine(config)

            # Register enabled modules
            module_classes = {
                "molecular": MolecularModule,
                "cellular": CellularModule,
                "immune": ImmuneModule,
                "vascular": VascularModule,
                "lymphatic": LymphaticModule,
                "spatial": SpatialModule,
                "epigenetic": EpigeneticModule,
                "circadian": CircadianModule,
                "morphogen": MorphogenModule,
            }

            for name, cls in module_classes.items():
                if self.modules_enabled.get(name, False):
                    self.engine.register_module(name, cls, self.module_configs.get(name, {}))

            # Initialize
            self.engine.initialize()

            # Link modules
            self._link_modules()

        # Subscribe to events we want to capture
        self._subscribe_events()

    def _link_modules(self):
        """Wire inter-module dependencies."""
        mods = self.engine.modules

        cellular = mods.get("cellular")
        immune = mods.get("immune")
        vascular = mods.get("vascular")
        lymphatic = mods.get("lymphatic")
        molecular = mods.get("molecular")
        epigenetic = mods.get("epigenetic")
        circadian = mods.get("circadian")
        morphogen = mods.get("morphogen")

        if immune and cellular:
            immune.set_cellular_module(cellular)
        if vascular and cellular:
            vascular.set_cellular_module(cellular)
        if lymphatic and cellular:
            lymphatic.set_cellular_module(cellular)
        if lymphatic and immune:
            lymphatic.set_immune_module(immune)

        # Register cells in supporting modules
        if cellular:
            for cell_id, cell in cellular.cells.items():
                if molecular:
                    molecular.add_cell(cell_id)
                if epigenetic:
                    epigenetic.add_cell(cell_id, cell.cell_type)
                if circadian:
                    circadian.add_cell(cell_id)
                if morphogen:
                    morphogen.add_cell(cell_id, cell.position)

    def _subscribe_events(self):
        """Subscribe to key events for the dashboard event log."""
        bus = self.engine.event_bus

        important_types = [
            EventTypes.CELL_DIVIDED, EventTypes.CELL_DIED,
            EventTypes.CELL_TRANSFORMED, EventTypes.CANCER_KILLED,
            EventTypes.IMMUNE_ACTIVATED, EventTypes.IMMUNE_RECRUITED,
            EventTypes.METASTASIS_OCCURRED, EventTypes.HYPOXIA_DETECTED,
            EventTypes.EXOSOME_RELEASED, EventTypes.MUTATION_OCCURRED,
        ]

        def _make_handler(etype):
            def handler(data):
                self.event_log.append({
                    "time": self.engine.time,
                    "step": self.engine.step_count,
                    "event": etype,
                    "data": data,
                })
            return handler

        for etype in important_types:
            bus.subscribe(etype, _make_handler(etype))

    # ── Run ──────────────────────────────────────────────────────────

    def run(self, progress_callback=None):
        """Run the full simulation, collecting time-series data.

        Parameters
        ----------
        progress_callback : callable, optional
            Called with (current_step, total_steps) for Streamlit progress bars.
        """
        import io
        from contextlib import redirect_stdout

        if self.engine is None:
            self.build()

        total_steps = int(self.duration / self.dt)
        self.history.clear()
        self.cell_snapshots.clear()
        self.event_log.clear()

        # Record initial state
        self._record_snapshot()
        self._record_cell_snapshot()

        buf = io.StringIO()
        with redirect_stdout(buf):
            for step_i in range(total_steps):
                self.engine.step()

                if step_i % self.record_interval == 0 or step_i == total_steps - 1:
                    self._record_snapshot()
                    self._record_cell_snapshot()

                if progress_callback and step_i % 5 == 0:
                    progress_callback(step_i + 1, total_steps)

        if progress_callback:
            progress_callback(total_steps, total_steps)

    # ── Data capture ─────────────────────────────────────────────────

    def _record_snapshot(self):
        """Capture scalar time-series data."""
        state = self.engine.get_state()
        snap = SimulationSnapshot(
            time=state["time"],
            step=state["step_count"],
        )

        # Cellular
        cs = state.get("cellular", {})
        snap.n_cells = cs.get("n_cells", 0)
        snap.n_cancer = cs.get("n_cancer", 0)
        snap.n_normal = cs.get("n_normal", 0)
        snap.total_divisions = cs.get("total_divisions", 0)
        snap.total_deaths = cs.get("total_deaths", 0)
        snap.total_transformations = cs.get("total_transformations", 0)
        snap.avg_oxygen = cs.get("avg_oxygen", 0)
        snap.avg_glucose = cs.get("avg_glucose", 0)

        # Immune
        ims = state.get("immune", {})
        snap.n_immune = ims.get("n_immune_cells", 0)
        snap.n_activated = ims.get("n_activated", 0)
        snap.n_t_cells = ims.get("n_t_cells", 0)
        snap.n_nk_cells = ims.get("n_nk_cells", 0)
        snap.n_macrophages = ims.get("n_macrophages", 0)
        snap.total_kills = ims.get("total_kills", 0)

        # Vascular
        vs = state.get("vascular", {})
        snap.avg_cell_O2 = vs.get("avg_cell_O2", 0)
        snap.avg_cell_glucose = vs.get("avg_cell_glucose", 0)
        snap.hypoxic_regions = vs.get("hypoxic_regions", 0)

        # Lymphatic
        ls = state.get("lymphatic", {})
        snap.total_metastases = ls.get("total_metastases", 0)
        snap.immune_in_vessels = ls.get("immune_in_vessels", 0)
        snap.cancer_in_vessels = ls.get("cancer_in_vessels", 0)

        # Epigenetic
        es = state.get("epigenetic", {})
        snap.avg_methylation = es.get("avg_methylation", 0)
        snap.silenced_genes = es.get("silenced_genes", 0)

        # Circadian
        crs = state.get("circadian", {})
        snap.master_phase = crs.get("master_phase", 0)
        snap.synchrony = crs.get("synchrony", 0)

        # Morphogen
        ms = state.get("morphogen", {})
        snap.total_fates_determined = ms.get("total_fates_determined", 0)

        self.history.append(snap)

    def _record_cell_snapshot(self):
        """Capture cell and immune positions for 3D rendering."""
        snap = CellSnapshot()
        mods = self.engine.modules

        # Cells
        cellular = mods.get("cellular")
        if cellular:
            alive = [c for c in cellular.cells.values() if c.alive]
            if alive:
                snap.cell_positions = np.array([c.position for c in alive])
                snap.cell_types = [c.cell_type for c in alive]
                snap.cell_phases = [c.phase for c in alive]
                snap.cell_oxygen = [float(c.oxygen) for c in alive]
                snap.cell_mhc1 = [float(c.mhc1_expression) for c in alive]
                snap.cell_ids = [c.cell_id for c in alive]

        # Immune
        immune = mods.get("immune")
        if immune:
            active = [ic for ic in immune.immune_cells.values() if not ic.in_blood]
            if active:
                snap.immune_positions = np.array([ic.position for ic in active])
                snap.immune_types = [ic.cell_type for ic in active]
                snap.immune_activated = [ic.activated for ic in active]

        # Capillaries
        vascular = mods.get("vascular")
        if vascular:
            for cap in vascular.capillaries.values():
                snap.capillary_starts.append(cap.start.tolist())
                snap.capillary_ends.append(cap.end.tolist())

        # Lymphatics
        lymphatic = mods.get("lymphatic")
        if lymphatic:
            for vessel in lymphatic.vessels.values():
                snap.lymph_starts.append(vessel.start.tolist())
                snap.lymph_ends.append(vessel.end.tolist())

        self.cell_snapshots.append(snap)

    # ── Convenience accessors ────────────────────────────────────────

    def get_time_series(self) -> Dict[str, list]:
        """Return dict of lists suitable for Plotly / DataFrame."""
        keys = [
            "time", "step",
            "n_cells", "n_cancer", "n_normal",
            "total_divisions", "total_deaths", "total_transformations",
            "avg_oxygen", "avg_glucose",
            "n_immune", "n_activated", "n_t_cells", "n_nk_cells", "n_macrophages",
            "total_kills",
            "avg_cell_O2", "avg_cell_glucose", "hypoxic_regions",
            "total_metastases", "immune_in_vessels", "cancer_in_vessels",
            "avg_methylation", "silenced_genes",
            "master_phase", "synchrony",
            "total_fates_determined",
        ]
        result = {k: [] for k in keys}
        for snap in self.history:
            for k in keys:
                result[k].append(getattr(snap, k, 0))
        return result

    def get_final_state(self) -> Dict[str, Any]:
        """Return final engine state dict."""
        if self.engine:
            return self.engine.get_state()
        return {}

    def get_event_summary(self) -> Dict[str, int]:
        """Count events by type."""
        from collections import Counter
        counts = Counter(e["event"] for e in self.event_log)
        return dict(counts)

    def get_key_events(self) -> List[dict]:
        """Return high-importance events (kills, metastases, transformations)."""
        important = {
            EventTypes.CANCER_KILLED, EventTypes.METASTASIS_OCCURRED,
            EventTypes.CELL_TRANSFORMED, EventTypes.HYPOXIA_DETECTED,
            EventTypes.IMMUNE_ACTIVATED,
        }
        return [e for e in self.event_log if e["event"] in important]

    def get_cell_snapshot_at(self, index: int = -1) -> Optional[CellSnapshot]:
        """Get cell positions snapshot at given recording index."""
        if self.cell_snapshots:
            return self.cell_snapshots[index]
        return None

    # ── Serialization helpers (for RunManager persistence) ──────────

    def serialize_cell_snapshots(self) -> List[Dict[str, Any]]:
        """Convert cell snapshots to a list of dicts with numpy arrays.

        Used by RunManager to persist snapshots via ArtifactStore.
        """
        serialized = []
        for i, snap in enumerate(self.cell_snapshots):
            entry: Dict[str, Any] = {"index": i}
            if snap.cell_positions is not None:
                entry["cell_positions"] = snap.cell_positions
                entry["cell_types"] = snap.cell_types
                entry["cell_phases"] = snap.cell_phases
                entry["cell_oxygen"] = np.array(snap.cell_oxygen, dtype=np.float32)
            if snap.immune_positions is not None:
                entry["immune_positions"] = snap.immune_positions
                entry["immune_types"] = snap.immune_types
                entry["immune_activated"] = snap.immune_activated
            serialized.append(entry)
        return serialized

    def get_final_metrics(self) -> Dict[str, Any]:
        """Extract final snapshot metrics for storage in SimulationRun entity."""
        if not self.history:
            return {}
        final = self.history[-1]
        return {
            "final_cancer": final.n_cancer,
            "final_normal": final.n_normal,
            "final_immune": final.n_immune,
            "total_kills": final.total_kills,
            "total_divisions": final.total_divisions,
            "total_deaths": final.total_deaths,
            "total_transformations": final.total_transformations,
            "total_metastases": final.total_metastases,
            "avg_oxygen": final.avg_oxygen,
            "avg_glucose": final.avg_glucose,
            "total_steps": final.step,
            "final_time": final.time,
        }
