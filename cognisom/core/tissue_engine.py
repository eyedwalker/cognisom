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

#: cfg.dt is hours; diffusion coefficients are um^2/s.
_SECONDS_PER_HOUR = 3600.0
from cognisom.gpu.multi_gpu_mechanics import (
    ADHESION_RANGE_UM, MultiGPUCellMechanics, MechanicsConfig,
)

log = logging.getLogger(__name__)


class TissueSimulationEngine:
    """Main orchestrator for tissue-scale multi-GPU simulation.

    Wraps existing solvers (diffusion, ODE, mechanics) with multi-GPU
    distribution layers. Runs on real GPUs or CPU-simulated partitions.
    """

    #: Index of cancer_epithelial in the archetype table.
    _CANCER_TYPE_INDEX = 5
    #: Spawn keys of the random streams derived from the run seed.
    _MECHANICS_STREAM = 1
    _VIZ_STREAM = 2
    #: A cell must have this much ATP to commit to division.
    _DIVISION_MIN_ATP = 0.3
    #: Damage above this blocks division (checkpoint arrest).
    _DIVISION_MAX_DAMAGE = 5.0
    #: Default growth ceiling as a multiple of the initial population,
    #: used when the config does not set max_cells_per_partition.
    _GROWTH_HEADROOM = 4.0

    def __init__(self, config: TissueScaleConfig):
        self._config = config
        self._steady_solves = {}
        # Resolved HERE rather than at the one call site that used to do
        # it. resolve_radii_from_library was called from the tissue-scale
        # dashboard page and nowhere else, so every other entry point --
        # tissue_runner.main, which is what the container runs, and every
        # script building a config from_json or from a preset -- gave
        # EVERY archetype the 5.0 um default, macrophage included, whose
        # measured radius is 10.6. This constructor is the one place all
        # of them pass through. It is idempotent (an archetype already in
        # archetype_radius_um is left alone and marked "config") and it
        # fails soft when no library is reachable.
        try:
            config.resolve_radii_from_library()
        except Exception:                                  # pragma: no cover
            log.debug("radii not resolved from the library", exc_info=True)
        self._step_count: int = 0
        self._sim_time: float = 0.0
        self._wall_start: float = 0.0
        self._initialized: bool = False
        # Seeded explicitly so a run is reproducible, and local so it does
        # not disturb the global NumPy RNG. This stream lays out the initial
        # tissue and places daughters; mechanics noise and the
        # visualisation subsample draw from their own streams of the same
        # seed, so taking snapshots more or less often cannot change the
        # physics.
        self._seed = int(getattr(config, "seed", 0) or 0)
        self._rng = np.random.default_rng(self._seed)
        self._viz_rng = np.random.default_rng(
            np.random.SeedSequence(self._seed, spawn_key=(self._VIZ_STREAM,)))
        self._division_capacity_warned = False

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


    #: Persistence time for a cell migrating in tissue, minutes. The
    #: library has no sourced persistence for any lineage yet, so this is
    #: an explicit ESTIMATE and is named as one rather than buried: it
    #: sits in the 10-90 min range PMID:25679760 measures for fibroblasts
    #: across its adhesion series (Figure 2H), toward the low-adhesion
    #: end because a cell in tissue is not on a glass coverslip.
    DEFAULT_PERSISTENCE_MIN = 15.0

    #: Where the motility coefficient came from, so a run can say.
    _motility_source: str = "not resolved"

    #: The last steady-state solve per field, as a SteadyResult. Carries the
    #: Newton and CG counts, the depletion, and whether it converged, so a run
    #: can report how well its fields are solved rather than only that a
    #: solve was attempted. Set per instance in __init__: a dict here would be
    #: one dict shared by every engine ever built.
    _steady_solves: Dict[str, Any]

    #: Seconds per voxel for one warm steady-state solve of one field, on a
    #: CPU. Measured at 0.15-0.17 us/voxel across 24^3, 48^3 and 100^3 --
    #: flat, because the vessel lattice keeps the CG iteration count
    #: independent of grid size. Machine-dependent, and used only to warn, so
    #: an order of magnitude is what matters rather than the third figure.
    #:
    #: It was 0.19-0.25 before the solver stopped allocating a full grid per
    #: operator application; removing that bought 23%, which is all there is
    #: to get that way -- the solve is memory-bandwidth bound, about ten
    #: full-grid passes per CG iteration and 37 iterations. Closing the
    #: remaining gap to the 500^3 preset's budget needs the GPU or a
    #: multigrid preconditioner, not more of this.
    _STEADY_SECONDS_PER_VOXEL = 0.16e-6

    def _warn_if_the_solves_will_not_fit(self, cfg) -> None:
        """Say so at startup if the steady solves exceed the runtime budget.

        A correct field is worth little if the run is killed before it
        finishes. Solving oxygen and glucose on the 500^3 ``full_tissue``
        preset is about 47 s per step, roughly 32 h over a 2400-step run,
        against a ``max_runtime_hours`` of 2 -- so the preset the headline
        cell count comes from no longer fits, where the stepped path it
        replaces did at about 1.25 h. That is the price of a field that is not
        312x out on its own depletion, and the user should meet it at startup
        rather than at the auto-terminate.
        """
        steady = [n for n, fc in cfg.fields.items()
                  if fc.steady_state and fc.diffusion_coeff > 0]
        if not steady:
            return
        voxels = 1
        for dim in cfg.grid_shape:
            voxels *= int(dim)
        interval = max(1, int(getattr(cfg, "steady_interval", 1)))
        per_step = (voxels * self._STEADY_SECONDS_PER_VOXEL
                    * len(steady) / interval)
        steps = int(cfg.duration / cfg.dt) if cfg.dt else 0
        hours = per_step * steps / 3600.0
        budget = float(getattr(cfg, "max_runtime_hours", 0.0) or 0.0)
        if budget <= 0 or hours <= budget:
            return
        needed = int(-(-hours // budget)) * interval
        log.warning(
            "steady-state solves for %s will take about %.1f h over %d steps "
            "(%.1f s per step on %s voxels), against max_runtime_hours of "
            "%.1f, so the run will be auto-terminated first. Raise "
            "steady_interval to about %d -- which leaves the field roughly "
            "%d%% stale, measured -- or allow a longer budget. A coarser "
            "field grid is NOT an option: a vessel is one voxel wide, so "
            "coarsening widens every capillary and changes the supply "
            "(tests/test_a_vessel_is_one_voxel_wide.py).",
            ", ".join(steady), hours, steps, per_step, f"{voxels:,}",
            budget, needed, needed,
        )

    def _resolve_motility_diffusion(self, cfg) -> float:
        """D for the tissue, from the library's migration speed.

        Returns 0.0 when the library has no evidenced speed, which
        leaves the diffusive term switched off rather than inventing
        motion. That is the honest default: before this, motility came
        from a random FORCE whose effect depended on how finely the step
        was subdivided.
        """
        from cognisom.library.seed_migration_modes import (
            diffusion_coefficient_um2_per_min)

        explicit = getattr(cfg, "motility_diffusion_um2_per_min", None)
        if explicit is not None:
            return float(explicit)

        speed = getattr(cfg, "migration_speed_um_per_min", None)
        if speed is None:
            # Straight from the library, the way resolve_radii does it,
            # and only when the value is EVIDENCE. 42 of the 44 stored
            # migration speeds are placeholders; reading one of those
            # would be inventing motion, so the term stays off.
            try:
                import contextlib
                import io

                from cognisom.library.parameters import (Provenance,
                                                         get_parameter)
                from cognisom.library.store import EntityStore

                with contextlib.redirect_stdout(io.StringIO()):
                    store = EntityStore()
                entity = store.find_entity_by_name("Stromal fibroblast")
                value = (get_parameter(entity, "migration_speed_um_per_min")
                         if entity is not None else None)
                if value is not None and \
                        value.provenance is not Provenance.PLACEHOLDER:
                    speed = float(value.value)
                    self._motility_source = (
                        f"Stromal fibroblast {speed} um/min "
                        f"({value.provenance.value}: {value.citation}), "
                        f"P={self.DEFAULT_PERSISTENCE_MIN:g} min estimated")
            except Exception:                              # pragma: no cover
                log.debug("motility not resolved from the library",
                          exc_info=True)
                speed = None
        if not speed:
            self._motility_source = "none: no evidenced migration speed"
            return 0.0
        return diffusion_coefficient_um2_per_min(
            float(speed), self.DEFAULT_PERSISTENCE_MIN)


    #: Centre-to-centre spacing of capillaries, um. Tissue is organised
    #: so that no cell is further from a vessel than oxygen can diffuse
    #: against consumption -- Krogh's observation -- which puts
    #: intercapillary distance in well-perfused tissue in the tens of
    #: micrometres. 60 um is inside that range and is an ESTIMATE: this
    #: library has no sourced value for any tissue, and a tumour's
    #: vasculature is irregular besides, which is part of why tumours go
    #: hypoxic at all.
    VESSEL_SPACING_UM = 60.0

    def _place_vessels(self, cfg) -> None:
        """Hold a lattice of voxels at arterial concentration.

        A capillary is a boundary CONDITION on the tissue field, not a
        source term: the vessel wall sits at whatever the blood carries,
        and the mass that implies is whatever the surrounding tissue
        draws out of it. That is the Krogh-Erlang inner boundary, and it
        introduces no free parameter beyond the arterial level each
        field already declares as its initial value.

        Without it there is nothing for a gradient to form between. The
        field was uniform and consumed uniformly, so oxygen varied by
        0.4% across 240 um of tissue, and correcting the uptake rate
        alone would only have lowered the whole field together -- which
        is why a rate fix on its own was measured to kill 84% of the
        tissue rather than produce a hypoxic core.
        """
        import numpy as np

        if self._diffusion is None:
            return
        resolution = float(cfg.resolution_um)
        stride = max(1, int(round(self.VESSEL_SPACING_UM / resolution)))
        shape = tuple(cfg.grid_shape)

        lattice = np.zeros(shape, dtype=bool)
        # Offset by half a spacing so no vessel sits on the domain face:
        # a vessel plane on the boundary would pin a whole face and make
        # the gradient one-dimensional.
        offset = stride // 2
        lattice[offset::stride, offset::stride, :] = True
        self._vessel_fraction = float(lattice.mean())

        for name, fc in cfg.fields.items():
            # Only the blood-borne fields. A chemokine is secreted by
            # cells into tissue; pinning it at a vessel would invent a
            # source the vasculature does not provide.
            if name not in ("oxygen", "glucose"):
                continue
            masks = self._decomposer.scatter_field(lattice) \
                if hasattr(self._decomposer, "scatter_field") else None
            if masks is None:
                masks = self._scatter_mask(lattice)
            try:
                self._diffusion.set_vessels(name, masks, fc.initial_value)
            except Exception:                              # pragma: no cover
                log.warning("vessels not applied to %r", name, exc_info=True)

    def _scatter_mask(self, lattice):
        """Split a global boolean grid into the solver's partitions."""
        import numpy as np

        out = []
        for gpu_id, field in enumerate(
                self._diffusion._fields[next(iter(self._diffusion._fields))]):
            partition = self._decomposer.partitions[gpu_id]
            axis = self._decomposer.split_axis
            gw = self._decomposer._ghost_width
            mask = np.zeros(tuple(field.shape), dtype=bool)
            lo, hi = partition.start, partition.end
            src = [slice(None)] * 3
            src[axis] = slice(lo, hi)
            dst = [slice(None)] * 3
            dst[axis] = slice(gw, gw + (hi - lo))
            mask[tuple(dst)] = lattice[tuple(src)]
            out.append(mask)
        return out

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
            cell_radii: (N,) float32 radii. If None, each cell takes its
                       archetype's radius (config.radius_for), which is
                       cell_radius_um unless the config carries better.
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

        # 6. Distribute cells across GPUs. The distributor's own default was
        # a literal 5.0, so cfg.cell_radius_um was never read and no
        # archetype could have its own size.
        if cell_radii is None:
            cell_radii = self._radius_by_type(int(np.max(cell_types, initial=0)) + 1)[
                np.asarray(cell_types, dtype=np.int64)]
        self._cell_arrays = self._distributor.initialize_cells(
            cell_positions, cell_types, cell_radii,
        )
        if cfg.asynchronous_cell_cycle:
            self._stagger_division_timers()

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
            self._place_vessels(cfg)
            self._warn_if_the_solves_will_not_fit(cfg)
            # A saturating sink where the field config asks for one. Set once:
            # the per-voxel rates are rebuilt every step, but what shape the
            # sink has is not a function of where the cells are.
            for name, fc in cfg.fields.items():
                if fc.uptake_km > 0.0 and fc.uptake_per_cell_per_hour > 0.0:
                    self._diffusion.set_saturating_uptake(
                        name, fc.uptake_km, fc.initial_value,
                    )

        # 8. Initialize mechanics
        if cfg.enable_mechanics:
            mechanics_config = MechanicsConfig.default()
            # The neighbour search sees every contact only while its bucket
            # is at least the contact range of the largest cell; below that
            # it falls back to the original 27-bucket stencil, misses
            # included. 15 um covers 5 um cells and nothing above 6.5 um, so
            # size it to the largest radius present. Radii only shrink after
            # this (a daughter is smaller than its parent).
            r_max = float(np.max(cell_radii, initial=cfg.cell_radius_um))
            contact_range = 2.0 * r_max + ADHESION_RANGE_UM
            mechanics_config.bucket_size_um = max(
                mechanics_config.bucket_size_um, contact_range)
            if contact_range > cfg.ghost_width_cells_um:
                log.warning(
                    "cells of radius %.1f um have a %.1f um contact range, "
                    "beyond the %.1f um ghost region: contacts across "
                    "partition boundaries will be missed. Raise "
                    "ghost_width_cells_um.", r_max, contact_range,
                    cfg.ghost_width_cells_um)
            # Motility from the library, as a DIFFUSION coefficient.
            # The library carries migration SPEED; a speed is not a
            # diffusion coefficient, because a fast cell that turns
            # constantly goes nowhere. The persistent-random-walk
            # relation D = S^2 P / 2 (PMID:25679760) is the conversion,
            # and it needs the persistence time as well.
            mechanics_config.motility_diffusion_um2_per_min = \
                self._resolve_motility_diffusion(cfg)
            self._mechanics = MultiGPUCellMechanics(
                self._backend, self._decomposer, self._ghost_exchanger,
                config=mechanics_config,
                seed=np.random.SeedSequence(
                    self._seed, spawn_key=(self._MECHANICS_STREAM,)),
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
            if cfg.implicit_diffusion:
                # cfg.dt is in HOURS; the field coefficients are in um^2/s.
                # The explicit branch below passes hours straight through, so
                # it advances 0.01 SECONDS of diffusion per step -- 3600x too
                # slow, a 60x short diffusion length for oxygen. Converting
                # here is the fix; it is also why the explicit branch cannot
                # simply be corrected in place (see implicit_diffusion.py:
                # correct units put D*dt/dx^2 at 720 against a limit of 1/6).
                dt_seconds = dt * _SECONDS_PER_HOUR

                # Cells consume. Without this the fields stay at their uniform
                # initial values, the Laplacian is identically zero, and the
                # whole diffusion stage is a no-op that still costs ~0.4-1.5 h
                # per run. Rebuilt every step because cells move, divide and
                # die.
                uptake = {}
                for name, fc in cfg.fields.items():
                    if fc.uptake_per_cell_per_hour > 0.0:
                        uptake[name] = self._diffusion.build_uptake_from_cells(
                            self._cell_arrays,
                            fc.uptake_per_cell_per_hour / _SECONDS_PER_HOUR,
                            cfg.resolution_um,
                        )
                self._diffusion.set_uptake(uptake or None)

                # Fields that relax far faster than one step are SOLVED,
                # not stepped. Substepping them to where the split resolves
                # the depletion would need ~7,000 substeps -- the same order
                # as the explicit scheme's stability requirement -- for a
                # transient nobody observes. See DiffusionFieldConfig.
                steady = [n for n, fc in cfg.fields.items()
                          if fc.steady_state and fc.diffusion_coeff > 0]
                interval = max(1, int(getattr(cfg, "steady_interval", 1)))
                if self._step_count % interval == 0:
                    for name in steady:
                        outcome = self._diffusion.solve_steady(
                            name, cfg.resolution_um)
                        self._steady_solves[name] = outcome

                # Unconditionally stable, so substeps buy accuracy only: the
                # per-voxel uptake split carries an error of order k*dt,
                # measured 0.71% at one substep and 0.14% at five.
                sub_dt = dt_seconds / cfg.diffusion_substeps
                for _ in range(cfg.diffusion_substeps):
                    self._diffusion.step_implicit(
                        sub_dt, cfg.resolution_um, exclude=steady)
            else:
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
                    positions_per_gpu, field_name, cfg.resolution_um,
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

        # ── 4b. Metabolism ───────────────────────────────────────
        # After the fields are sampled and before division and death read
        # atp and damage, so a cell is judged on this step's substrate
        # levels rather than the previous step's.
        if cfg.enable_ode:
            self._step_metabolism(dt)

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
                self._cell_arrays, max_cells=max_c, rng=self._viz_rng,
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

    #: Rate at which a cell that cannot pay its maintenance ATP accumulates
    #: an apoptotic signal, per hour. With the death threshold at 5.0 this is
    #: about two hours of total metabolic failure. NOT sourced -- it is a
    #: plausible time for an ATP-starved cell and nothing more, and it is the
    #: one number in this path that is not either stoichiometry or a measured
    #: substrate level.
    _APOPTOSIS_RATE_PER_HOUR = 2.5

    def _step_metabolism(self, dt: float) -> None:
        """Turn local oxygen and glucose into ATP, lactate and a death signal.

        Both ``atp`` and ``lactate`` had no writer: ATP was set to 1.0 at
        initialisation and read by the division gate forever after, and
        lactate was never touched, so there was no metabolism and no Warburg
        effect on a platform built to study one.

        The stoichiometry comes from ``cognisom.gpu.fba_solver``, which was
        already in the repo and used by nothing. ``core.metabolism`` solves it
        once per substrate level on a grid and interpolates, because the
        linear program depends on a cell only through its local oxygen and
        glucose -- solving it per cell would be a million linear programs per
        step.

        Two death routes, with different meanings, both feeding the existing
        test in :meth:`_step_cell_cycle`. ``damage`` accumulates under
        prolonged hypoxia, which is sublethal stress. ``apoptotic_signal``
        accumulates when the flux problem is INFEASIBLE, which with a lower
        bound on maintenance means precisely "this cell cannot pay its upkeep
        at these substrate levels" -- a stronger and better-defined condition
        than a threshold on oxygen.
        """
        from cognisom.core import antigen_presentation, metabolism

        for gpu_id, ca in enumerate(self._cell_arrays):
            n_real = ca.n_real
            if n_real == 0:
                continue
            state = self._backend.to_numpy(ca.state[:n_real])

            atp, lactate, feasible = metabolism.evaluate(
                state[:, 0], state[:, 1])
            state[:, 2] = atp
            state[:, 3] = lactate
            starving = ~feasible
            state[starving, 9] += dt * self._APOPTOSIS_RATE_PER_HOUR

            # MHC-I, from the archetype baseline and the local oxygen. Every
            # number behind this is an ESTIMATE and antigen_presentation.py
            # says so at length -- it is here because the column had
            # consumers and no producer, not because anyone measured it.
            types = self._backend.to_numpy(ca.cell_types[:n_real])
            state[:, 6] = antigen_presentation.surface_level(
                state[:, 0], types)

            full = self._backend.to_numpy(ca.state)
            full[:n_real] = state
            ca.state = self._backend.to_device(full, gpu_id)

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

            # ── Division ─────────────────────────────────────────
            # Vectorised: the per-cell Python loop this replaces ran once
            # per cell per step, which at the configured 1e6 cells is a
            # million interpreter iterations every step.
            #
            # It also never actually divided anything. The loop reset
            # div_timer to 0 and left a comment saying "Actual daughter
            # cell creation happens in bulk below" -- there was no bulk
            # below; the function ended. The tissue population was fixed
            # for the entire run, so growth, clonal expansion and the
            # tumour/immune ratio were all static.
            is_cancer = (types == self._CANCER_TYPE_INDEX)
            div_time = np.where(
                is_cancer,
                self._config.division_time_cancer_hr,
                self._config.division_time_normal_hr,
            )
            atp = state[:, 2]
            ready = (
                alive
                & (state[:, div_timer_col] >= div_time)
                & (atp > self._DIVISION_MIN_ATP)
                & (damage < self._DIVISION_MAX_DAMAGE)
            )
            dividing = np.flatnonzero(ready)

            # Bound total growth. The arrays are allocated to exactly the
            # initial cell count, so growth means reallocating; this cap
            # keeps a fast-dividing tumour from exhausting memory.
            max_cells = getattr(
                self._config, "max_cells_per_partition", None
            ) or max(16, int(self._config.n_cells * self._GROWTH_HEADROOM))
            room = max(0, max_cells - n_real)
            if len(dividing) > room:
                if not self._division_capacity_warned:
                    log.warning(
                        "Partition %d reached the growth cap of %d cells; "
                        "%d divisions dropped this step. Raise n_cells or "
                        "set max_cells_per_partition for more headroom.",
                        gpu_id, max_cells, len(dividing) - room,
                    )
                    self._division_capacity_warned = True
                dividing = dividing[:room]

            # Parents reset their timer whether or not a daughter fit.
            state[ready, div_timer_col] = 0.0

            n_new = len(dividing)
            if n_new:
                # Slice to the real region: ghost cells sit immediately
                # after it and are rebuilt by `update_ghosts` at the start
                # of the next step's mechanics phase, so they are dropped
                # here rather than shifted.
                positions = self._backend.to_numpy(ca.positions)[:n_real]
                radii = self._backend.to_numpy(ca.radii)[:n_real]
                velocities = self._backend.to_numpy(ca.velocities)[:n_real]
                cell_types = self._backend.to_numpy(ca.cell_types)[:n_real]

                # Daughters are placed one radius away along a random
                # unit vector, so the pair starts in contact rather than
                # co-located -- perfectly overlapping cells produce an
                # enormous repulsive force on the next mechanics substep.
                directions = self._rng.normal(size=(n_new, 3))
                directions /= np.maximum(
                    np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
                )

                # Volume splits between the two cells: r -> r * 2^(-1/3).
                new_radii = radii[dividing] * (0.5 ** (1.0 / 3.0))
                radii[dividing] = new_radii

                daughter_positions = (
                    positions[dividing] + directions * new_radii[:, None]
                ).astype(positions.dtype)

                # Daughter inherits the parent's metabolic state, then
                # starts its own cycle: timer and age at zero.
                daughter_state = state[dividing].copy()
                daughter_state[:, div_timer_col] = 0.0
                daughter_state[:, age_col] = 0.0

                ca.positions = self._backend.to_device(
                    np.concatenate([positions, daughter_positions]), gpu_id)
                ca.radii = self._backend.to_device(
                    np.concatenate([radii, new_radii.astype(radii.dtype)]),
                    gpu_id)
                ca.velocities = self._backend.to_device(
                    np.concatenate([velocities, velocities[dividing]]), gpu_id)
                ca.cell_types = self._backend.to_device(
                    np.concatenate([cell_types, cell_types[dividing]]), gpu_id)
                ca.alive = self._backend.to_device(
                    np.concatenate([alive, np.ones(n_new, dtype=alive.dtype)]),
                    gpu_id)
                ca.state = self._backend.to_device(
                    np.concatenate([state, daughter_state.astype(state.dtype)]),
                    gpu_id)

                if ca.ode_state is not None:
                    ode = self._backend.to_numpy(ca.ode_state)[:n_real]
                    # Species are concentrations (intensive), so both
                    # cells carry the parent's values.
                    ca.ode_state = self._backend.to_device(
                        np.concatenate([ode, ode[dividing]]), gpu_id)

                ca.n_real = n_real + n_new
                ca.n_ghost = 0
            else:
                # Write back
                full_state = self._backend.to_numpy(ca.state)
                full_state[:n_real] = state
                ca.state = self._backend.to_device(full_state, gpu_id)

                full_alive = self._backend.to_numpy(ca.alive)
                full_alive[:n_real] = alive
                ca.alive = self._backend.to_device(full_alive, gpu_id)

    def _stagger_division_timers(self) -> None:
        """Start each cell at a random point in its cycle.

        Every division timer (and age) used to start at 0, so all cancer
        cells divided together at 12 h and every other cell at 48 h. Drawn
        from the steady-state age distribution (core.cell_cycle), from the
        layout stream after the layout, so the positions and types of a
        given seed are unchanged.
        """
        from cognisom.core.cell_cycle import steady_state_ages

        cfg = self._config
        for gpu_id, ca in enumerate(self._cell_arrays):
            n_real = ca.n_real
            if n_real == 0:
                continue
            state = self._backend.to_numpy(ca.state)
            types = self._backend.to_numpy(ca.cell_types)[:n_real]
            periods = np.where(types == self._CANCER_TYPE_INDEX,
                               cfg.division_time_cancer_hr,
                               cfg.division_time_normal_hr)
            ages = steady_state_ages(periods, self._rng.random(n_real))
            state[:n_real, 7] = ages          # division timer
            state[:n_real, 4] = ages          # age since birth
            ca.state = self._backend.to_device(state, gpu_id)

    def _radius_by_type(self, n_types: int) -> np.ndarray:
        """Radius per archetype index, in composition order, in um.

        Type indices follow list(initial_composition) -- the order
        _generate_initial_cells assigns them in. An index past the end of
        the composition (caller-supplied types) takes cell_radius_um.
        """
        cfg = self._config
        names = list(cfg.initial_composition)
        return np.array(
            [cfg.radius_for(names[i]) if i < len(names) else cfg.cell_radius_um
             for i in range(max(n_types, len(names)))],
            dtype=np.float32,
        )

    def _generate_initial_cells(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random initial cell positions and types.

        Returns:
            (positions, cell_types) arrays.
        """
        cfg = self._config
        n = cfg.n_cells

        # Random positions within tissue volume. Drawn from the engine's
        # own seeded generator: `np.random.rand` gave a different starting
        # tissue on every run -- different cell positions AND a different
        # cancer/normal split -- so two runs of the same configuration
        # were not comparable, and it mutated the global NumPy RNG.
        positions = self._rng.random((n, 3)).astype(np.float64)
        positions[:, 0] *= cfg.tissue_size_um[0]
        positions[:, 1] *= cfg.tissue_size_um[1]
        positions[:, 2] *= cfg.tissue_size_um[2]

        # Assign types according to composition
        type_names = list(cfg.initial_composition.keys())
        type_fractions = np.array(list(cfg.initial_composition.values()))
        type_fractions = type_fractions / type_fractions.sum()  # Normalize

        cumulative = np.cumsum(type_fractions)
        random_vals = self._rng.random(n)
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
                rng=self._viz_rng,
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
