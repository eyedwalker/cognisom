"""Configuration dataclasses for tissue-scale simulation.

Defines all parameters for running a 1M-cell tissue simulation
across multiple GPUs, including grid shape, cell composition,
time stepping, diffusion fields, and safety limits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DiffusionFieldConfig:
    """Configuration for a single diffusion field."""
    name: str
    diffusion_coeff: float    # um^2/s
    initial_value: float      # Initial uniform concentration
    # NOTE: neither of the next two is read by any solver -- they are declared
    # here and only ever serialised by to_dict. The diffusion solvers write the
    # interior and leave every face at its current value, which is a Dirichlet
    # condition, not the Neumann this claims. Left as-is rather than silently
    # reinterpreted; changing the boundary condition is a modelling decision.
    boundary_type: str = "neumann"  # neumann or dirichlet -- NOT IMPLEMENTED
    degradation_rate: float = 0.0   # 1/s, exponential decay -- NOT IMPLEMENTED

    #: First-order uptake per cell, in 1/h, following the convention
    #: ``spatial_module.add_uptake`` argues for: k = q_volumetric / C_arterial,
    #: so that at the arterial concentration the consumption equals the
    #: measured zeroth-order rate. A zeroth-order sink cannot be used in a
    #: steady-state solve -- it drives the concentration negative wherever
    #: demand exceeds supply -- which is why this is first order.
    #:
    #: Rates accumulate within a voxel, so a voxel holding several cells
    #: consumes proportionally more.
    #:
    #: Only used by the implicit solver. The explicit ``step()`` ignores
    #: sources and sinks entirely, which is why tissue diffusion was a no-op:
    #: uniform initial fields with nothing consuming leave the Laplacian
    #: identically zero at every step.
    uptake_per_cell_per_hour: float = 0.0

    @staticmethod
    def oxygen() -> DiffusionFieldConfig:
        # q = 0.15 (the archetypes' highest oxygen_consumption), C_art = 0.21.
        return DiffusionFieldConfig(
            "oxygen", diffusion_coeff=2000.0, initial_value=0.21,
            uptake_per_cell_per_hour=0.15 / 0.21)

    @staticmethod
    def glucose() -> DiffusionFieldConfig:
        # q = 0.2 (archetypes' glucose_consumption), C_art = 5.0.
        return DiffusionFieldConfig(
            "glucose", diffusion_coeff=600.0, initial_value=5.0,
            uptake_per_cell_per_hour=0.2 / 5.0)

    @staticmethod
    def cxcl12() -> DiffusionFieldConfig:
        # A chemokine is secreted, not consumed, so it carries no uptake. The
        # secretion side is still not modelled, so this field stays uniform.
        return DiffusionFieldConfig(
            "cxcl12", diffusion_coeff=100.0, initial_value=0.0)


# Default prostate tissue cell type composition
DEFAULT_COMPOSITION: Dict[str, float] = {
    "luminal_secretory": 0.30,
    "basal": 0.15,
    "fibroblast": 0.15,
    "smooth_muscle": 0.10,
    "endothelial": 0.08,
    "cancer_epithelial": 0.10,
    "t_cell": 0.05,
    "macrophage": 0.04,
    "nk_cell": 0.02,
    "neuroendocrine": 0.01,
}


#: Archetype -> the entity-library row its parameters are read from. The
#: store matches names exactly and case-sensitively, so without this table no
#: archetype key finds a row. The immune archetypes read the rows the immune
#: module reads (modules/immune_module.LINEAGE_ENTITY), the macrophage the
#: unpolarized cell_type row. smooth_muscle and cancer_epithelial have no row
#: -- "Prostate cancer stem cell" is a stem cell, not bulk tumour epithelium
#: -- so they always take the config default.
ARCHETYPE_ENTITY: Dict[str, Tuple[str, str]] = {
    "luminal_secretory": ("Luminal epithelial cell", "cell_type"),
    "basal": ("Basal epithelial cell", "cell_type"),
    "fibroblast": ("Stromal fibroblast", "cell_type"),
    "endothelial": ("Endothelial cell", "cell_type"),
    "neuroendocrine": ("Neuroendocrine cell", "cell_type"),
    "macrophage": ("Macrophage", "cell_type"),
    "t_cell": ("CD8+ Effector T cell", "immune_cell"),
    "nk_cell": ("NK cell", "immune_cell"),
}


@dataclass
class TissueScaleConfig:
    """Master configuration for tissue-scale simulation."""

    # ── Scale ──────────────────────────────────────────────────
    n_cells: int = 1_000_000
    grid_shape: Tuple[int, int, int] = (500, 500, 500)
    resolution_um: float = 10.0       # um per voxel
    tissue_size_um: Tuple[float, float, float] = (5000.0, 5000.0, 5000.0)

    # ── Multi-GPU ──────────────────────────────────────────────
    n_gpus: int = 8
    ghost_width_voxels: int = 2       # Ghost layers for diffusion stencil
    ghost_width_cells_um: float = 30.0  # Ghost region for cell mechanics

    # ── Time stepping ──────────────────────────────────────────
    dt: float = 0.01                   # hours per step
    duration: float = 24.0             # total simulation hours
    diffusion_substeps: int = 5        # per main step

    #: Use the unconditionally stable implicit (ADI) diffusion solver.
    #:
    #: The explicit scheme cannot integrate this configuration correctly at
    #: any cost: with dt converted to seconds as the coefficients require,
    #: D*dt/dx^2 = 720 for oxygen against an explicit 3-D limit of 1/6, so
    #: stability needs 4320 substeps per step rather than 5 -- about 1286 h
    #: for a 24 h run against max_runtime_hours below. See
    #: cognisom/gpu/implicit_diffusion.py.
    #:
    #: With the implicit solver, diffusion_substeps above is an ACCURACY knob
    #: rather than a stability requirement: per-voxel uptake is operator-split
    #: and carries an error of order k*dt, measured at 0.71% with one substep
    #: and 0.14% with five.
    implicit_diffusion: bool = True
    mechanics_substeps: int = 10       # per main step

    # ── Diffusion fields ───────────────────────────────────────
    fields: Dict[str, DiffusionFieldConfig] = field(default_factory=lambda: {
        "oxygen": DiffusionFieldConfig.oxygen(),
        "glucose": DiffusionFieldConfig.glucose(),
        "cxcl12": DiffusionFieldConfig.cxcl12(),
    })

    # ── Cell biology ───────────────────────────────────────────
    initial_composition: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_COMPOSITION)
    )
    cell_radius_um: float = 5.0       # Radius of any archetype without its own
    #: Per-archetype radius in um, for archetypes with something better than
    #: the default. Filled by resolve_radii_from_library when the config is
    #: built, or set directly, and serialised with the config so a remote run
    #: uses the same radii without the library.
    archetype_radius_um: Dict[str, float] = field(default_factory=dict)
    #: archetype -> where its radius came from: "config", the library row
    #: with the value's grade and citation, or why the library gave none
    #: (and cell_radius_um applies).
    archetype_radius_source: Dict[str, str] = field(default_factory=dict)
    division_time_normal_hr: float = 48.0
    division_time_cancer_hr: float = 12.0

    # ── Module toggles ─────────────────────────────────────────
    enable_ode: bool = True            # Intracellular ODE (AR signaling)
    enable_ssa: bool = False           # Stochastic gene expression
    enable_mechanics: bool = True      # Cell-cell forces
    enable_diffusion: bool = True      # 3D concentration fields
    enable_cell_cycle: bool = True     # Division/death
    enable_immune: bool = False        # Immune response module

    # ── Visualization ──────────────────────────────────────────
    snapshot_interval_steps: int = 100
    max_viz_cells: int = 100_000       # Subsample for rendering
    stream_positions: bool = True
    stream_fields: bool = True

    # ── Execution ──────────────────────────────────────────────
    execution_mode: str = "local"      # "local" (CPU/single GPU) or "remote" (Lambda)

    # ── Safety (Lambda Labs) ───────────────────────────────────
    max_runtime_hours: float = 2.0     # Hard auto-terminate limit
    budget_usd: float = 80.0           # Max spend per session
    disconnect_timeout_min: int = 10   # Auto-terminate if dashboard disconnects

    @property
    def total_steps(self) -> int:
        return int(self.duration / self.dt)

    def radius_for(self, archetype: str) -> float:
        """Radius of ``archetype`` in um."""
        return float(self.archetype_radius_um.get(archetype, self.cell_radius_um))

    def resolve_radii_from_library(self, resolver=None) -> "TissueScaleConfig":
        """Take each archetype's radius from the entity library where it is evidence.

        A value is used only if the resolver passes it -- simulation grade,
        in um, not a placeholder. Every cell row's radius except a few
        curated ones is a placeholder halved from a scale_um nobody
        measured, and reading those would resize most of the tissue on no
        evidence. An archetype already in archetype_radius_um keeps its
        value: explicit config outranks the library. Fails soft -- with no
        library the config is unchanged and says so. Returns self.
        """
        if resolver is None:
            try:
                from cognisom.genomics import parameter_resolver
                resolver = parameter_resolver.shared_resolver()
            except Exception:                              # pragma: no cover
                resolver = None
        for archetype in self.initial_composition:
            if archetype in self.archetype_radius_um:
                self.archetype_radius_source.setdefault(archetype, "config")
                continue
            row = ARCHETYPE_ENTITY.get(archetype)
            if row is None:
                self.archetype_radius_source[archetype] = "no library row"
                continue
            if resolver is None:
                self.archetype_radius_source[archetype] = "library unavailable"
                continue
            name, entity_type = row
            pv, note = resolver.sourced_value(name, entity_type, "radius_um")
            if pv is not None:
                self.archetype_radius_um[archetype] = float(pv.value)
            self.archetype_radius_source[archetype] = f"{name}: {note}"
        return self

    @property
    def n_fields(self) -> int:
        return len(self.fields)

    def to_json(self) -> str:
        """Serialize to JSON for remote execution."""
        d = {
            "n_cells": self.n_cells,
            "grid_shape": list(self.grid_shape),
            "resolution_um": self.resolution_um,
            "tissue_size_um": list(self.tissue_size_um),
            "n_gpus": self.n_gpus,
            "ghost_width_voxels": self.ghost_width_voxels,
            "ghost_width_cells_um": self.ghost_width_cells_um,
            "dt": self.dt,
            "duration": self.duration,
            "diffusion_substeps": self.diffusion_substeps,
            "implicit_diffusion": self.implicit_diffusion,
            "mechanics_substeps": self.mechanics_substeps,
            "fields": {
                name: {
                    "name": fc.name,
                    "diffusion_coeff": fc.diffusion_coeff,
                    "initial_value": fc.initial_value,
                    "boundary_type": fc.boundary_type,
                    "uptake_per_cell_per_hour": fc.uptake_per_cell_per_hour,
                    "degradation_rate": fc.degradation_rate,
                }
                for name, fc in self.fields.items()
            },
            "initial_composition": self.initial_composition,
            "cell_radius_um": self.cell_radius_um,
            "archetype_radius_um": dict(self.archetype_radius_um),
            "archetype_radius_source": dict(self.archetype_radius_source),
            "enable_ode": self.enable_ode,
            "enable_ssa": self.enable_ssa,
            "enable_mechanics": self.enable_mechanics,
            "enable_diffusion": self.enable_diffusion,
            "enable_cell_cycle": self.enable_cell_cycle,
            "enable_immune": self.enable_immune,
            "snapshot_interval_steps": self.snapshot_interval_steps,
            "max_viz_cells": self.max_viz_cells,
            "execution_mode": self.execution_mode,
            "max_runtime_hours": self.max_runtime_hours,
            "budget_usd": self.budget_usd,
            "disconnect_timeout_min": self.disconnect_timeout_min,
        }
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str) -> TissueScaleConfig:
        """Deserialize from JSON."""
        d = json.loads(data)
        config = cls(
            n_cells=d.get("n_cells", 1_000_000),
            grid_shape=tuple(d.get("grid_shape", [500, 500, 500])),
            resolution_um=d.get("resolution_um", 10.0),
            tissue_size_um=tuple(d.get("tissue_size_um", [5000, 5000, 5000])),
            n_gpus=d.get("n_gpus", 8),
            ghost_width_voxels=d.get("ghost_width_voxels", 2),
            ghost_width_cells_um=d.get("ghost_width_cells_um", 30.0),
            dt=d.get("dt", 0.01),
            duration=d.get("duration", 24.0),
            diffusion_substeps=d.get("diffusion_substeps", 5),
            implicit_diffusion=d.get("implicit_diffusion", True),
            mechanics_substeps=d.get("mechanics_substeps", 10),
            initial_composition=d.get("initial_composition", dict(DEFAULT_COMPOSITION)),
            cell_radius_um=d.get("cell_radius_um", 5.0),
            archetype_radius_um=dict(d.get("archetype_radius_um") or {}),
            archetype_radius_source=dict(d.get("archetype_radius_source") or {}),
            enable_ode=d.get("enable_ode", True),
            enable_ssa=d.get("enable_ssa", False),
            enable_mechanics=d.get("enable_mechanics", True),
            enable_diffusion=d.get("enable_diffusion", True),
            enable_cell_cycle=d.get("enable_cell_cycle", True),
            enable_immune=d.get("enable_immune", False),
            snapshot_interval_steps=d.get("snapshot_interval_steps", 100),
            max_viz_cells=d.get("max_viz_cells", 100_000),
            execution_mode=d.get("execution_mode", "local"),
            max_runtime_hours=d.get("max_runtime_hours", 2.0),
            budget_usd=d.get("budget_usd", 80.0),
            disconnect_timeout_min=d.get("disconnect_timeout_min", 10),
        )
        # Parse fields
        if "fields" in d:
            config.fields = {
                name: DiffusionFieldConfig(**fc)
                for name, fc in d["fields"].items()
            }
        return config

    # ── Presets ─────────────────────────────────────────────────

    @classmethod
    def small_test(cls) -> TissueScaleConfig:
        """10K cells for local CPU testing."""
        return cls(
            n_cells=10_000,
            grid_shape=(100, 100, 100),
            tissue_size_um=(1000, 1000, 1000),
            n_gpus=8,  # Simulated partitions
            duration=1.0,
            snapshot_interval_steps=10,
            execution_mode="local",
        )

    @classmethod
    def medium_test(cls) -> TissueScaleConfig:
        """100K cells for single-GPU testing."""
        return cls(
            n_cells=100_000,
            grid_shape=(200, 200, 200),
            tissue_size_um=(2000, 2000, 2000),
            n_gpus=8,
            duration=4.0,
            execution_mode="local",
        )

    @classmethod
    def full_tissue(cls) -> TissueScaleConfig:
        """1M cells on 8x B200 (Lambda Labs)."""
        return cls(
            n_cells=1_000_000,
            grid_shape=(500, 500, 500),
            tissue_size_um=(5000, 5000, 5000),
            n_gpus=8,
            duration=24.0,
            execution_mode="remote",
            max_runtime_hours=2.0,
            budget_usd=80.0,
        )
