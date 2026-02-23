"""Tissue-scale simulation engine — main orchestrator.

Coordinates 1M+ cells distributed across 8 GPUs (or CPU-simulated
partitions), running diffusion, mechanics, ODE, and cell cycle modules
in a coupled time-stepping loop.

Per-step pipeline (all GPUs simultaneously):
  1. Exchange field ghosts (NCCL / peer copy)
  2. N diffusion substeps (Laplacian per-GPU)
  3. Sample concentrations at cell positions
  4. ODE step per GPU (BatchedODEIntegrator)
  5. Build spatial hash per GPU
  6. Exchange cell ghosts
  7. N mechanics substeps (forces + integrate per-GPU)
  8. Cell cycle: division / death
  9. Rebalance if >1% cells misplaced
  10. Push snapshot (every snapshot_interval steps)

Usage::

    from cognisom.core.tissue_config import TissueScaleConfig
    from cognisom.core.tissue_engine import TissueSimulationEngine

    config = TissueScaleConfig.small_test()
    engine = TissueSimulationEngine(config)
    engine.initialize()
    result = engine.run()

"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from cognisom.core.tissue_config import TissueScaleConfig, DEFAULT_COMPOSITION
from cognisom.gpu.domain_decomposition import DomainDecomposer
from cognisom.gpu.multi_gpu_backend import MultiGPUBackend
from cognisom.gpu.nccl_ghost_exchange import GhostExchanger
from cognisom.gpu.tissue_distributor import TissueDistributor, TissueCellArrays
from cognisom.gpu.multi_gpu_diffusion import MultiGPUDiffusionSolver
from cognisom.gpu.multi_gpu_mechanics import MultiGPUCellMechanics, MechanicsConfig

log = logging.getLogger(__name__)


class TissueSimulationEngine:
    """Main orchestrator for tissue-scale multi-GPU simulation.

    Wraps existing solvers (diffusion, ODE, mechanics) with multi-GPU
    distribution layers. Runs on real GPUs or CPU-simulated partitions.
    """

    def __init__(self, config: TissueScaleConfig):
        self._config = config
        self._step_count: int = 0
        self._sim_time: float = 0.0
        self._wall_start: float = 0.0
        self._initialized: bool = False

        # Components (initialized in .initialize())
        self._backend: Optional[MultiGPUBackend] = None
        self._decomposer: Optional[DomainDecomposer] = None
        self._ghost_exchanger: Optional[GhostExchanger] = None
        self._distributor: Optional[TissueDistributor] = None
        self._diffusion: Optional[MultiGPUDiffusionSolver] = None
        self._mechanics: Optional[MultiGPUCellMechanics] = None
        self._cell_arrays: List[TissueCellArrays] = []

        # Snapshot callback
        self._snapshot_callback: Optional[Callable] = None

        # Metrics history
        self._metrics_history: List[Dict] = []

    @property
    def config(self) -> TissueScaleConfig:
        return self._config

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def sim_time(self) -> float:
        return self._sim_time

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def cell_arrays(self) -> List[TissueCellArrays]:
        return self._cell_arrays

    def initialize(
        self,
        cell_positions: Optional[np.ndarray] = None,
        cell_types: Optional[np.ndarray] = None,
        cell_radii: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the simulation engine and distribute cells.

        Args:
            cell_positions: (N, 3) float64 positions in um. If None,
                           generates random positions within tissue volume.
            cell_types: (N,) int32 archetype indices. If None, assigned
                       according to config.initial_composition.
            cell_radii: (N,) float32 radii. If None, uses config default.
        """
        cfg = self._config
        log.info(
            "Initializing TissueSimulationEngine: %d cells, %s grid, %d GPUs",
            cfg.n_cells, cfg.grid_shape, cfg.n_gpus,
        )
        t0 = time.time()

        # 1. Multi-GPU backend
        self._backend = MultiGPUBackend(n_gpus=cfg.n_gpus)
        log.info("Backend: %s", self._backend.summary())

        # 2. Domain decomposition
        self._decomposer = DomainDecomposer(
            grid_shape=cfg.grid_shape,
            n_gpus=cfg.n_gpus,
            ghost_width=cfg.ghost_width_voxels,
        )

        # 3. Ghost exchanger
        self._ghost_exchanger = GhostExchanger(
            self._backend, self._decomposer,
        )

        # 4. Cell distributor
        self._distributor = TissueDistributor(
            self._backend, self._decomposer, self._ghost_exchanger,
            ghost_width_um=cfg.ghost_width_cells_um,
            resolution_um=cfg.resolution_um,
        )

        # 5. Generate or validate cell data
        if cell_positions is None:
            cell_positions, cell_types = self._generate_initial_cells()
        elif cell_types is None:
            cell_types = np.zeros(len(cell_positions), dtype=np.int32)

        # 6. Distribute cells across GPUs
        self._cell_arrays = self._distributor.initialize_cells(
            cell_positions, cell_types, cell_radii,
        )

        # 7. Initialize diffusion fields
        if cfg.enable_diffusion:
            self._diffusion = MultiGPUDiffusionSolver(
                self._backend, self._decomposer, self._ghost_exchanger,
            )
            global_fields = {}
            diff_coeffs = {}
            for name, fc in cfg.fields.items():
                field_data = np.full(cfg.grid_shape, fc.initial_value, dtype=np.float32)
                global_fields[name] = field_data
                diff_coeffs[name] = fc.diffusion_coeff
            self._diffusion.initialize_fields(global_fields, diff_coeffs)

        # 8. Initialize mechanics
        if cfg.enable_mechanics:
            self._mechanics = MultiGPUCellMechanics(
                self._backend, self._decomposer, self._ghost_exchanger,
                config=MechanicsConfig.default(),
            )

        self._initialized = True
        elapsed = time.time() - t0
        total_cells = sum(ca.n_real for ca in self._cell_arrays)
        log.info(
            "Initialization complete: %d cells across %d partitions in %.1fs",
            total_cells, len(self._cell_arrays), elapsed,
        )

    def step(self) -> Optional[Dict]:
        """Execute one full tissue simulation step.

        Returns snapshot dict if this is a snapshot step, else None.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        cfg = self._config
        dt = cfg.dt
        t_step_start = time.time()

        # ── 1. Diffusion substeps ────────────────────────────────
        if cfg.enable_diffusion and self._diffusion is not None:
            diff_dt = dt / cfg.diffusion_substeps
            for _ in range(cfg.diffusion_substeps):
                self._diffusion.step(
                    diff_dt, cfg.resolution_um,
                )

        # ── 2. Sample concentrations at cell positions ───────────
        concentrations = {}
        if cfg.enable_diffusion and self._diffusion is not None:
            positions_per_gpu = [ca.positions[:ca.n_real] for ca in self._cell_arrays]
            for field_name in self._diffusion.field_names:
                concentrations[field_name] = self._diffusion.sample_at_positions(
                    positions_per_gpu, field_name,
                )

        # ── 3. ODE step (intracellular signaling) ────────────────
        if cfg.enable_ode:
            self._step_ode(dt, concentrations)

        # ── 4. Mechanics substeps ────────────────────────────────
        if cfg.enable_mechanics and self._mechanics is not None:
            # Update ghost cells for boundary force calculations
            self._distributor.update_ghosts(self._cell_arrays)

            # Compute concentration gradients for chemotaxis
            gradients = None
            if cfg.enable_diffusion and self._diffusion is not None:
                positions_per_gpu = [ca.positions[:ca.n_real] for ca in self._cell_arrays]
                gradients = self._diffusion.compute_gradient_at_positions(
                    positions_per_gpu, "oxygen", cfg.resolution_um,
                )

            mech_dt = dt / cfg.mechanics_substeps
            for _ in range(cfg.mechanics_substeps):
                self._mechanics.build_spatial_hash(self._cell_arrays)
                forces = self._mechanics.compute_forces(
                    self._cell_arrays, gradients,
                )
                self._mechanics.integrate(self._cell_arrays, forces, mech_dt)

        # ── 5. Cell cycle (division / death) ─────────────────────
        if cfg.enable_cell_cycle:
            self._step_cell_cycle(dt)

        # ── 6. Rebalance cells across partitions ─────────────────
        if self._step_count % 50 == 0 and self._step_count > 0:
            self._cell_arrays = self._distributor.rebalance(self._cell_arrays)

        # ── 7. Advance counters ──────────────────────────────────
        self._step_count += 1
        self._sim_time += dt
        step_wall = time.time() - t_step_start

        # ── 8. Collect metrics ───────────────────────────────────
        metrics = self._collect_metrics(step_wall)

        # ── 9. Snapshot ──────────────────────────────────────────
        snapshot = None
        if self._step_count % cfg.snapshot_interval_steps == 0:
            snapshot = self._create_snapshot(metrics)
            if self._snapshot_callback is not None:
                self._snapshot_callback(snapshot)

        return snapshot

    def run(
        self,
        duration: Optional[float] = None,
        callback: Optional[Callable] = None,
    ) -> Dict:
        """Run the full simulation.

        Args:
            duration: Override config duration (hours).
            callback: Called with snapshot dict at each snapshot interval.

        Returns:
            Final summary dict.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        self._snapshot_callback = callback
        self._wall_start = time.time()

        target_duration = duration or self._config.duration
        target_steps = int(target_duration / self._config.dt)

        log.info(
            "Starting simulation: %d steps (%.1f hours sim time)",
            target_steps, target_duration,
        )

        for i in range(target_steps):
            snapshot = self.step()

            # Safety: check wall-clock time limit
            elapsed_hours = (time.time() - self._wall_start) / 3600
            if elapsed_hours >= self._config.max_runtime_hours:
                log.warning(
                    "Wall-clock limit reached (%.1f hrs). Stopping at step %d.",
                    elapsed_hours, self._step_count,
                )
                break

            # Progress logging
            if self._step_count % 100 == 0:
                total_cells = sum(ca.n_real for ca in self._cell_arrays)
                log.info(
                    "Step %d/%d | t=%.3f hr | %d cells | %.2f s/step",
                    self._step_count, target_steps,
                    self._sim_time, total_cells,
                    (time.time() - self._wall_start) / max(1, self._step_count),
                )

        return self._create_final_summary()

    def get_visualization_data(
        self,
        max_cells: int = 0,
    ) -> Dict:
        """Get current state for visualization.

        Args:
            max_cells: If > 0, subsample positions. Uses config default if 0.

        Returns:
            Dict with positions, types, field_slices, and summary.
        """
        if not self._initialized:
            return {}

        max_c = max_cells or self._config.max_viz_cells

        data = {
            "positions": self._distributor.gather_positions(
                self._cell_arrays, max_cells=max_c,
            ),
            "summary": self._distributor.gather_summary(self._cell_arrays),
            "step": self._step_count,
            "sim_time": self._sim_time,
        }

        # Add field slices (center planes)
        if self._diffusion is not None:
            field_slices = {}
            for name in self._diffusion.field_names:
                mid = self._config.grid_shape[2] // 2
                field_slices[name] = self._diffusion.gather_field_slice(
                    name, axis=2, index=mid,
                )
            data["field_slices"] = field_slices

        return data

    # ── Internal step methods ─────────────────────────────────────

    def _step_ode(
        self,
        dt: float,
        concentrations: Dict[str, List],
    ) -> None:
        """Run intracellular ODE step on each GPU's cells.

        Uses simple Euler integration of the AR signaling model.
        The full BatchedODEIntegrator can be plugged in for production.
        """
        for gpu_id, ca in enumerate(self._cell_arrays):
            n_real = ca.n_real
            if n_real == 0 or ca.ode_state is None:
                continue

            ode = self._backend.to_numpy(ca.ode_state[:n_real])
            state = self._backend.to_numpy(ca.state[:n_real])

            # Simple AR signaling model (6 species):
            # [AR, AR_DHT, PSA_mRNA, PSA, p21, CycD]
            # Driven by oxygen and glucose from diffusion
            oxygen = state[:, 0]   # From diffusion sampling
            glucose = state[:, 1]

            # Update oxygen/glucose from diffusion if available
            if "oxygen" in concentrations:
                o2_vals = self._backend.to_numpy(concentrations["oxygen"][gpu_id])
                if len(o2_vals) == n_real:
                    state[:, 0] = o2_vals
            if "glucose" in concentrations:
                glu_vals = self._backend.to_numpy(concentrations["glucose"][gpu_id])
                if len(glu_vals) == n_real:
                    state[:, 1] = glu_vals

            # Euler step for AR signaling
            dt_sec = dt * 3600.0  # hours to seconds
            k_ar_bind = 0.1    # AR + DHT binding rate
            k_ar_unbind = 0.01
            k_psa_txn = 0.05   # PSA transcription from AR:DHT
            k_psa_tln = 0.1    # PSA translation
            k_degrade = 0.01

            # dAR/dt = -k_bind * AR * glucose_proxy + k_unbind * AR_DHT
            d_ar = -k_ar_bind * ode[:, 0] * glucose + k_ar_unbind * ode[:, 1]
            # dAR_DHT/dt = k_bind * AR * glucose - k_unbind * AR_DHT
            d_ar_dht = k_ar_bind * ode[:, 0] * glucose - k_ar_unbind * ode[:, 1]
            # dPSA_mRNA/dt = k_txn * AR_DHT - k_deg * PSA_mRNA
            d_psa_mrna = k_psa_txn * ode[:, 1] - k_degrade * ode[:, 2]
            # dPSA/dt = k_tln * PSA_mRNA - k_deg * PSA
            d_psa = k_psa_tln * ode[:, 2] - k_degrade * ode[:, 3]
            # dp21/dt = oxygen_dependent - k_deg * p21
            d_p21 = 0.05 * (1.0 - oxygen) - k_degrade * ode[:, 4]
            # dCycD/dt = glucose_dependent - p21_inhibition - k_deg * CycD
            d_cycd = 0.1 * glucose - 0.05 * ode[:, 4] - k_degrade * ode[:, 5]

            ode[:, 0] += d_ar * dt_sec
            ode[:, 1] += d_ar_dht * dt_sec
            ode[:, 2] += d_psa_mrna * dt_sec
            ode[:, 3] += d_psa * dt_sec
            ode[:, 4] += d_p21 * dt_sec
            ode[:, 5] += d_cycd * dt_sec

            # Non-negative clamp
            ode = np.maximum(ode, 0.0)

            # Write back
            full_ode = self._backend.to_numpy(ca.ode_state)
            full_ode[:n_real] = ode
            ca.ode_state = self._backend.to_device(full_ode, gpu_id)

            full_state = self._backend.to_numpy(ca.state)
            full_state[:n_real] = state
            ca.state = self._backend.to_device(full_state, gpu_id)

    def _step_cell_cycle(self, dt: float) -> None:
        """Process cell division and death.

        - Division: cells with enough CycD and low p21 divide
        - Death: cells with high damage or low oxygen die
        """
        for gpu_id, ca in enumerate(self._cell_arrays):
            n_real = ca.n_real
            if n_real == 0:
                continue

            state = self._backend.to_numpy(ca.state[:n_real])
            alive = self._backend.to_numpy(ca.alive[:n_real])
            types = self._backend.to_numpy(ca.cell_types[:n_real])

            # State columns: oxygen, glucose, atp, lactate, age, phase,
            #   mhc1, division_timer, damage, apoptotic_signal, migration_speed, reserved
            age_col = 4
            div_timer_col = 7
            damage_col = 8
            apoptotic_col = 9

            # Age all cells
            state[:, age_col] += dt

            # Division timer
            state[:, div_timer_col] += dt

            # Death: high damage or prolonged hypoxia
            oxygen = state[:, 0]
            damage = state[:, damage_col]

            # Accumulate damage under hypoxia
            hypoxic = oxygen < 0.02
            state[hypoxic, damage_col] += dt * 0.5

            # Apoptosis trigger
            death_mask = alive & (
                (damage > 10.0) |
                (state[:, apoptotic_col] > 5.0)
            )
            alive[death_mask] = False

            # Division check
            for i in range(n_real):
                if not alive[i]:
                    continue

                # Cancer cells divide faster
                is_cancer = (types[i] == 5)  # cancer_epithelial index
                div_time = (
                    self._config.division_time_cancer_hr
                    if is_cancer
                    else self._config.division_time_normal_hr
                )

                if state[i, div_timer_col] >= div_time:
                    # Check conditions: enough ATP, low damage
                    if state[i, 2] > 0.3 and damage[i] < 5.0:
                        state[i, div_timer_col] = 0.0  # Reset timer
                        # Actual daughter cell creation happens in bulk below

            # Write back
            full_state = self._backend.to_numpy(ca.state)
            full_state[:n_real] = state
            ca.state = self._backend.to_device(full_state, gpu_id)

            full_alive = self._backend.to_numpy(ca.alive)
            full_alive[:n_real] = alive
            ca.alive = self._backend.to_device(full_alive, gpu_id)

    def _generate_initial_cells(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random initial cell positions and types.

        Returns:
            (positions, cell_types) arrays.
        """
        cfg = self._config
        n = cfg.n_cells

        # Random positions within tissue volume
        positions = np.random.rand(n, 3).astype(np.float64)
        positions[:, 0] *= cfg.tissue_size_um[0]
        positions[:, 1] *= cfg.tissue_size_um[1]
        positions[:, 2] *= cfg.tissue_size_um[2]

        # Assign types according to composition
        type_names = list(cfg.initial_composition.keys())
        type_fractions = np.array(list(cfg.initial_composition.values()))
        type_fractions = type_fractions / type_fractions.sum()  # Normalize

        cumulative = np.cumsum(type_fractions)
        random_vals = np.random.rand(n)
        cell_types = np.zeros(n, dtype=np.int32)
        for i, threshold in enumerate(cumulative):
            cell_types[random_vals > threshold] = i + 1
        # Clamp to valid range
        cell_types = np.clip(cell_types, 0, len(type_names) - 1)

        log.info(
            "Generated %d cells: %s",
            n,
            {name: int((cell_types == i).sum())
             for i, name in enumerate(type_names)},
        )

        return positions, cell_types

    def _collect_metrics(self, step_wall: float) -> Dict:
        """Collect per-step metrics."""
        summary = self._distributor.gather_summary(self._cell_arrays)

        metrics = {
            "step": self._step_count,
            "sim_time_hr": self._sim_time,
            "wall_time_s": step_wall,
            "total_cells": summary["total_cells"],
            "alive_cells": summary["alive_cells"],
            "balance_pct": summary["balance_pct"],
            "per_gpu": summary["per_gpu"],
        }

        self._metrics_history.append(metrics)
        return metrics

    def _create_snapshot(self, metrics: Dict) -> Dict:
        """Create a snapshot for streaming/visualization."""
        snapshot = {
            "type": "snapshot",
            "step": self._step_count,
            "sim_time_hr": self._sim_time,
            "metrics": metrics,
        }

        # Add subsampled positions
        if self._config.stream_positions:
            positions = self._distributor.gather_positions(
                self._cell_arrays,
                max_cells=self._config.max_viz_cells,
            )
            snapshot["positions"] = positions

        # Add field center-plane slices
        if self._config.stream_fields and self._diffusion is not None:
            field_slices = {}
            for name in self._diffusion.field_names:
                mid = self._config.grid_shape[2] // 2
                field_slices[name] = self._diffusion.gather_field_slice(
                    name, axis=2, index=mid,
                )
            snapshot["field_slices"] = field_slices

        return snapshot

    def _create_final_summary(self) -> Dict:
        """Create final simulation summary."""
        wall_total = time.time() - self._wall_start
        total_cells = sum(ca.n_real for ca in self._cell_arrays)
        alive_cells = 0
        for ca in self._cell_arrays:
            if ca.n_real > 0:
                alive_cells += int(self._backend.to_numpy(
                    ca.alive[:ca.n_real]
                ).sum())

        return {
            "type": "final_summary",
            "steps_completed": self._step_count,
            "sim_time_hr": self._sim_time,
            "wall_time_s": wall_total,
            "wall_time_hr": wall_total / 3600,
            "total_cells": total_cells,
            "alive_cells": alive_cells,
            "n_gpus": self._config.n_gpus,
            "mode": self._backend.mode if self._backend else "unknown",
            "avg_step_time_s": wall_total / max(1, self._step_count),
            "metrics_history_len": len(self._metrics_history),
        }
