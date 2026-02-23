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
    boundary_type: str = "neumann"  # neumann or dirichlet
    degradation_rate: float = 0.0   # 1/s, exponential decay

    @staticmethod
    def oxygen() -> DiffusionFieldConfig:
        return DiffusionFieldConfig("oxygen", diffusion_coeff=2000.0, initial_value=0.21)

    @staticmethod
    def glucose() -> DiffusionFieldConfig:
        return DiffusionFieldConfig("glucose", diffusion_coeff=600.0, initial_value=5.0)

    @staticmethod
    def cxcl12() -> DiffusionFieldConfig:
        return DiffusionFieldConfig("cxcl12", diffusion_coeff=100.0, initial_value=0.0)


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
    cell_radius_um: float = 5.0       # Default cell radius
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
            "mechanics_substeps": self.mechanics_substeps,
            "fields": {
                name: {
                    "name": fc.name,
                    "diffusion_coeff": fc.diffusion_coeff,
                    "initial_value": fc.initial_value,
                    "boundary_type": fc.boundary_type,
                    "degradation_rate": fc.degradation_rate,
                }
                for name, fc in self.fields.items()
            },
            "initial_composition": self.initial_composition,
            "cell_radius_um": self.cell_radius_um,
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
            mechanics_substeps=d.get("mechanics_substeps", 10),
            initial_composition=d.get("initial_composition", dict(DEFAULT_COMPOSITION)),
            cell_radius_um=d.get("cell_radius_um", 5.0),
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
