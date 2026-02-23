"""
Diapedesis Simulation Engine
=============================

GPU-accelerated simulation of the 6-step leukocyte extravasation cascade
through postcapillary venule endothelium (Abbas Fig 3.3):

1. Cytokine production → E-selectin expression on endothelium
2. Selectin-mediated rolling (low-affinity, fast on/off PSGL-1↔E-selectin)
3. Chemokine-mediated integrin activation (inside-out signaling)
4. Integrin-mediated firm adhesion (LFA-1↔ICAM-1, VLA-4↔VCAM-1)
5. Transmigration through endothelial junctions (PECAM-1, VE-cadherin)
6. Chemotaxis to infection site through ECM

Physics:
- Cylindrical vessel geometry with Poiseuille flow profile
- Stochastic selectin bind/unbind kinetics
- State-machine-driven leukocyte behavior
- Chemokine diffusion field with gradient computation
- CPU (NumPy) with optional Warp GPU acceleration
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Try importing Warp GPU kernels
try:
    from cognisom.physics.diapedesis_kernels import DiapedesisGPU
    _GPU_CLASS_AVAILABLE = True
except ImportError:
    _GPU_CLASS_AVAILABLE = False

try:
    import warp as wp
    WARP_AVAILABLE = True
except ImportError:
    WARP_AVAILABLE = False


# ── Enums ──────────────────────────────────────────────────────────────

class LeukocyteState(IntEnum):
    """State machine for leukocyte diapedesis progression."""
    FLOWING = 0         # In bloodstream, no contact
    MARGINATING = 1     # Near vessel wall
    ROLLING = 2         # Selectin-mediated rolling on endothelium
    ACTIVATING = 3      # Chemokine binding, integrin conformational change
    ARRESTED = 4        # Firm integrin-mediated adhesion
    CRAWLING = 5        # Crawling to nearest junction
    TRANSMIGRATING = 6  # Squeezing through endothelial junction
    MIGRATED = 7        # In extravascular tissue, chemotaxing


class ParticleType(IntEnum):
    """Types of particles in the simulation."""
    LEUKOCYTE = 0
    RED_BLOOD_CELL = 1
    ENDOTHELIAL = 2     # Fixed on vessel wall
    CHEMOKINE = 3       # Diffusing signaling molecules


# ── Configuration ──────────────────────────────────────────────────────

@dataclass
class DiapedesisConfig:
    """Configuration for diapedesis simulation parameters.

    All spatial units in micrometers (μm), time in seconds (s),
    force in piconewtons (pN).
    """
    # Vessel geometry
    vessel_length: float = 200.0        # μm (along x-axis)
    vessel_radius: float = 25.0         # μm (postcapillary venule ~20-50)
    vessel_wall_thickness: float = 2.0  # μm

    # Particle counts
    n_leukocytes: int = 20
    n_rbc: int = 200
    n_chemokine_sources: int = 5

    # Blood flow (Poiseuille)
    shear_stress: float = 2.0           # dyn/cm² (venular: 1-10)
    flow_velocity_max: float = 500.0    # μm/s at vessel center

    # Selectin binding (rolling)
    selectin_on_rate: float = 10.0      # s⁻¹ when near wall
    selectin_off_rate: float = 5.0      # s⁻¹ (fast off → rolling)
    selectin_bond_force: float = 50.0   # pN (catch-bond peak)
    rolling_velocity: float = 30.0      # μm/s (typical: 20-40)

    # Integrin activation
    chemokine_activation_threshold: float = 0.3  # normalized concentration
    integrin_activation_time: float = 2.0         # seconds to full activation

    # Integrin adhesion (arrest)
    integrin_on_rate: float = 1.0       # s⁻¹ (slower but stable)
    integrin_off_rate: float = 0.01     # s⁻¹ (very slow → firm)
    integrin_bond_strength: float = 200.0  # pN

    # Transmigration
    junction_disruption_time: float = 20.0  # seconds (accelerated for visualization)
    transmigration_speed: float = 5.0       # μm/min

    # Chemotaxis
    chemotaxis_strength: float = 2.0    # μm/min per gradient unit
    chemokine_diffusion: float = 100.0  # μm²/s
    tissue_migration_speed: float = 20.0  # μm/min (neutrophil speed in tissue)

    # Bacteria / phagocytosis
    n_bacteria: int = 6
    bacteria_kill_distance: float = 10.0  # μm to start phagocytosis
    phagocytosis_time: float = 12.0       # seconds to fully engulf

    # Inflammation (0-1, controls E-selectin/ICAM-1 expression)
    inflammation_level: float = 0.7

    # Physics
    dt: float = 0.01                    # seconds
    damping: float = 10.0               # viscous drag coefficient
    brownian_strength: float = 0.5      # thermal noise amplitude
    leukocyte_radius: float = 6.0       # μm (diameter ~12)
    rbc_radius: float = 3.75            # μm (diameter ~7.5)

    # Margination threshold: distance from wall to trigger margination
    margination_distance: float = 15.0  # μm

    # Crawling
    crawl_speed: float = 10.0           # μm/min on endothelial surface
    crawl_to_junction_time: float = 10.0  # seconds average


# ── Chemokine Field (simplified 2D radial) ─────────────────────────────

class ChemokineField:
    """Simplified chemokine concentration field for the vessel environment.

    Uses a 2D grid (x along vessel, r radial distance from wall)
    to model chemokine diffusion from tissue infection site through
    the vessel wall.
    """

    def __init__(self, vessel_length: float, vessel_radius: float,
                 nx: int = 40, nr: int = 20,
                 diffusion_coeff: float = 100.0):
        self.vessel_length = vessel_length
        self.vessel_radius = vessel_radius
        self.nx = nx
        self.nr = nr
        self.diffusion_coeff = diffusion_coeff
        self.dx = vessel_length / nx
        self.dr = (vessel_radius * 2) / nr  # covers inside + outside

        # Concentration grid: x × r
        # r=0 is vessel center, r=nr is far tissue
        self.concentration = np.zeros((nx, nr), dtype=np.float32)

        # Sources: infection sites below vessel
        self.sources: List[Tuple[int, int, float]] = []

    def add_source(self, x_pos: float, rate: float):
        """Add chemokine source at x position, outside vessel wall."""
        ix = int(np.clip(x_pos / self.dx, 0, self.nx - 1))
        # Source in tissue, just outside vessel wall
        ir = int(self.nr * 0.75)  # ~75% of radial range = tissue
        self.sources.append((ix, ir, rate))

    def update(self, dt: float):
        """Diffuse chemokines for one timestep."""
        laplacian = np.zeros_like(self.concentration)
        c = self.concentration

        # Interior diffusion (2D)
        laplacian[1:-1, 1:-1] = (
            (c[2:, 1:-1] + c[:-2, 1:-1] - 2 * c[1:-1, 1:-1]) / (self.dx ** 2) +
            (c[1:-1, 2:] + c[1:-1, :-2] - 2 * c[1:-1, 1:-1]) / (self.dr ** 2)
        )

        self.concentration += self.diffusion_coeff * laplacian * dt

        # Apply sources
        for ix, ir, rate in self.sources:
            self.concentration[ix, ir] += rate * dt

        # Clamp
        np.clip(self.concentration, 0.0, 10.0, out=self.concentration)

    def get_concentration_at(self, x: float, r_from_center: float) -> float:
        """Get chemokine concentration at (x, radial_distance)."""
        ix = int(np.clip(x / self.dx, 0, self.nx - 1))
        # Map radial distance to grid: center=0 → index=0, far=nr-1
        ir = int(np.clip(r_from_center / self.dr, 0, self.nr - 1))
        return float(self.concentration[ix, ir])

    def get_gradient_x_at(self, x: float, r_from_center: float) -> float:
        """Get x-component of chemokine gradient."""
        ix = int(np.clip(x / self.dx, 1, self.nx - 2))
        ir = int(np.clip(r_from_center / self.dr, 0, self.nr - 1))
        return float((self.concentration[ix + 1, ir] - self.concentration[ix - 1, ir])
                      / (2 * self.dx))

    def get_gradient_radial_at(self, x: float, r_from_center: float) -> float:
        """Get radial gradient (toward tissue = positive)."""
        ix = int(np.clip(x / self.dx, 0, self.nx - 1))
        ir = int(np.clip(r_from_center / self.dr, 1, self.nr - 2))
        return float((self.concentration[ix, ir + 1] - self.concentration[ix, ir - 1])
                      / (2 * self.dr))


# ── Simulation Engine ──────────────────────────────────────────────────

class DiapedesisSim:
    """GPU-accelerated diapedesis simulation engine.

    Simulates leukocyte extravasation through a postcapillary venule
    with biologically accurate binding kinetics and state transitions.

    Usage::

        sim = DiapedesisSim(DiapedesisConfig(inflammation_level=0.8))
        sim.initialize()
        frames = sim.run(duration=120.0, fps=30)
        # frames is a list of snapshot dicts for visualization
    """

    def __init__(self, config: DiapedesisConfig = None):
        self.config = config or DiapedesisConfig()
        self._initialized = False
        self._time = 0.0
        self._step_count = 0
        self._rng = np.random.default_rng(42)

        # Particle arrays (initialized in initialize())
        self.positions: np.ndarray = None      # (N, 3)
        self.velocities: np.ndarray = None     # (N, 3)
        self.forces: np.ndarray = None         # (N, 3)
        self.particle_types: np.ndarray = None # (N,) int
        self.radii: np.ndarray = None          # (N,)

        # Leukocyte-specific arrays
        self.leukocyte_states: np.ndarray = None          # (n_leuko,) LeukocyteState
        self.selectin_bonds: np.ndarray = None            # (n_leuko,) bool
        self.integrin_activation: np.ndarray = None       # (n_leuko,) 0-1
        self.transmigration_progress: np.ndarray = None   # (n_leuko,) 0-1
        self.state_timers: np.ndarray = None              # (n_leuko,) seconds in current state
        self.leukocyte_indices: np.ndarray = None          # indices into main arrays

        # RBC indices
        self.rbc_indices: np.ndarray = None

        # Endothelial arrays
        self.endo_positions: np.ndarray = None            # (n_endo, 3)
        self.endo_selectin_expr: np.ndarray = None        # (n_endo,) 0-1
        self.endo_icam1_expr: np.ndarray = None           # (n_endo,) 0-1
        self.endo_vcam1_expr: np.ndarray = None           # (n_endo,) 0-1
        self.endo_junction_integrity: np.ndarray = None   # (n_endo,) 0-1
        self.endo_indices: np.ndarray = None

        # Chemokine field
        self.chemokine_field: ChemokineField = None

        # Vessel geometry (precomputed)
        self._vessel_center = np.array([0.0, 0.0, 0.0])  # yz center
        self._vessel_axis = np.array([1.0, 0.0, 0.0])    # along x

        # Bacteria (infection site)
        self.bacteria_positions: np.ndarray = None      # (n_bacteria, 3)
        self.bacteria_alive: np.ndarray = None          # (n_bacteria,) bool
        self.bacteria_phagocytosis: np.ndarray = None   # (n_bacteria,) 0-1
        self.leukocyte_target: np.ndarray = None        # (n_leuko,) int, -1=none

        # Metrics
        self._metrics_history: List[Dict] = []

        # GPU accelerator (force computation + integration + boundary)
        self._gpu = None
        if _GPU_CLASS_AVAILABLE:
            try:
                gpu = DiapedesisGPU()
                if gpu.available:
                    self._gpu = gpu
                    log.info("DiapedesisSim: GPU acceleration enabled (Warp)")
            except Exception as e:
                log.info(f"DiapedesisSim: GPU init failed ({e}), using CPU")
        if self._gpu is None:
            log.info("DiapedesisSim: using CPU (NumPy) path")

    @property
    def gpu_enabled(self) -> bool:
        """Whether GPU acceleration is active."""
        return self._gpu is not None and self._gpu.available

    @property
    def n_total(self) -> int:
        return (self.config.n_leukocytes + self.config.n_rbc +
                len(self.endo_positions) if self.endo_positions is not None else 0)

    def initialize(self) -> None:
        """Set up vessel geometry, place particles, create chemokine field."""
        cfg = self.config

        # Vessel center at y=0, z=0, extending along x
        self._vessel_center = np.array([cfg.vessel_length / 2, 0.0, 0.0])

        # Place endothelial cells on vessel wall (bottom half for cutaway view)
        self._place_endothelial_cells()

        # Place leukocytes inside vessel
        self._place_leukocytes()

        # Place RBCs inside vessel
        self._place_rbc()

        # Build combined particle arrays
        self._build_particle_arrays()

        # Place bacteria at infection site
        self._place_bacteria()

        # Set up chemokine field (sources at bacteria positions)
        self._setup_chemokine_field()

        self._initialized = True
        self._time = 0.0
        self._step_count = 0

        log.info(f"DiapedesisSim initialized: {cfg.n_leukocytes} leukocytes, "
                 f"{cfg.n_rbc} RBCs, {len(self.endo_positions)} endothelial cells, "
                 f"inflammation={cfg.inflammation_level:.1f}")

    def _place_endothelial_cells(self):
        """Tile endothelial cells on the inner vessel wall."""
        cfg = self.config
        R = cfg.vessel_radius
        L = cfg.vessel_length

        # Tile endothelial cells on the cylindrical wall
        # ~25 μm cells → spacing of ~20 μm
        spacing = 20.0
        n_along = max(2, int(L / spacing))
        n_around = max(4, int(2 * math.pi * R / spacing))

        positions = []
        for i in range(n_along):
            x = (i + 0.5) * (L / n_along)
            for j in range(n_around):
                theta = 2 * math.pi * j / n_around
                y = R * math.cos(theta)
                z = R * math.sin(theta)
                positions.append([x, y, z])

        self.endo_positions = np.array(positions, dtype=np.float32)
        n_endo = len(self.endo_positions)

        # Expression levels driven by inflammation
        infl = cfg.inflammation_level
        self.endo_selectin_expr = np.clip(
            infl * (0.7 + 0.3 * self._rng.random(n_endo)), 0, 1
        ).astype(np.float32)
        self.endo_icam1_expr = np.clip(
            0.2 + infl * (0.6 + 0.2 * self._rng.random(n_endo)), 0, 1
        ).astype(np.float32)
        self.endo_vcam1_expr = np.clip(
            infl * (0.5 + 0.3 * self._rng.random(n_endo)), 0, 1
        ).astype(np.float32)
        self.endo_junction_integrity = np.ones(n_endo, dtype=np.float32)

    def _place_leukocytes(self):
        """Place leukocytes randomly inside the vessel lumen."""
        cfg = self.config
        n = cfg.n_leukocytes
        R = cfg.vessel_radius
        L = cfg.vessel_length

        positions = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            x = self._rng.uniform(0, L)
            # Random position inside cylinder, biased away from center
            r = R * math.sqrt(self._rng.uniform(0, 1)) * 0.85  # 85% of radius
            theta = self._rng.uniform(0, 2 * math.pi)
            y = r * math.cos(theta)
            z = r * math.sin(theta)
            positions[i] = [x, y, z]

        self._leukocyte_positions = positions

        # State arrays
        self.leukocyte_states = np.full(n, LeukocyteState.FLOWING, dtype=np.int32)
        self.selectin_bonds = np.zeros(n, dtype=np.bool_)
        self.integrin_activation = np.zeros(n, dtype=np.float32)
        self.transmigration_progress = np.zeros(n, dtype=np.float32)
        self.state_timers = np.zeros(n, dtype=np.float32)

    def _place_rbc(self):
        """Place RBCs throughout the vessel lumen."""
        cfg = self.config
        n = cfg.n_rbc
        R = cfg.vessel_radius
        L = cfg.vessel_length

        positions = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            x = self._rng.uniform(0, L)
            r = R * math.sqrt(self._rng.uniform(0, 1)) * 0.9
            theta = self._rng.uniform(0, 2 * math.pi)
            y = r * math.cos(theta)
            z = r * math.sin(theta)
            positions[i] = [x, y, z]

        self._rbc_positions = positions

    def _place_bacteria(self):
        """Place bacteria at infection site below vessel."""
        cfg = self.config
        n = cfg.n_bacteria
        R = cfg.vessel_radius
        L = cfg.vessel_length

        self.bacteria_positions = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            x = self._rng.uniform(L * 0.3, L * 0.7)
            y = -(R * 1.5 + self._rng.uniform(0, R * 0.5))
            z = self._rng.uniform(-R * 0.3, R * 0.3)
            self.bacteria_positions[i] = [x, y, z]

        self.bacteria_alive = np.ones(n, dtype=np.bool_)
        self.bacteria_phagocytosis = np.zeros(n, dtype=np.float32)
        self.leukocyte_target = np.full(cfg.n_leukocytes, -1, dtype=np.int32)

    def _build_particle_arrays(self):
        """Combine leukocytes + RBCs into unified arrays."""
        cfg = self.config
        n_leuko = cfg.n_leukocytes
        n_rbc = cfg.n_rbc
        n_total = n_leuko + n_rbc

        self.positions = np.zeros((n_total, 3), dtype=np.float32)
        self.velocities = np.zeros((n_total, 3), dtype=np.float32)
        self.forces = np.zeros((n_total, 3), dtype=np.float32)
        self.particle_types = np.zeros(n_total, dtype=np.int32)
        self.radii = np.zeros(n_total, dtype=np.float32)

        # Leukocytes first
        self.leukocyte_indices = np.arange(0, n_leuko, dtype=np.int32)
        self.positions[:n_leuko] = self._leukocyte_positions
        self.particle_types[:n_leuko] = ParticleType.LEUKOCYTE
        self.radii[:n_leuko] = cfg.leukocyte_radius

        # RBCs
        self.rbc_indices = np.arange(n_leuko, n_leuko + n_rbc, dtype=np.int32)
        self.positions[n_leuko:n_leuko + n_rbc] = self._rbc_positions
        self.particle_types[n_leuko:n_leuko + n_rbc] = ParticleType.RED_BLOOD_CELL
        self.radii[n_leuko:n_leuko + n_rbc] = cfg.rbc_radius

        # Initialize velocities with flow velocity
        for i in range(n_total):
            v = self._flow_velocity_at(self.positions[i])
            self.velocities[i] = [v, 0.0, 0.0]

    def _setup_chemokine_field(self):
        """Create chemokine field with sources at infection site."""
        cfg = self.config
        self.chemokine_field = ChemokineField(
            vessel_length=cfg.vessel_length,
            vessel_radius=cfg.vessel_radius,
            diffusion_coeff=cfg.chemokine_diffusion,
        )

        # Place chemokine sources at bacteria positions (infection site)
        for i in range(min(cfg.n_bacteria, cfg.n_chemokine_sources)):
            rate = cfg.inflammation_level * 0.5
            self.chemokine_field.add_source(self.bacteria_positions[i, 0], rate)

        # Pre-diffuse for some initial spread
        for _ in range(100):
            self.chemokine_field.update(0.1)

    # ── Flow profile ───────────────────────────────────────────────────

    def _radial_distance(self, pos: np.ndarray) -> float:
        """Compute radial distance from vessel axis (along x)."""
        return math.sqrt(pos[1] ** 2 + pos[2] ** 2)

    def _flow_velocity_at(self, pos: np.ndarray) -> float:
        """Poiseuille parabolic velocity profile: v(r) = v_max * (1 - (r/R)²)."""
        r = self._radial_distance(pos)
        R = self.config.vessel_radius
        ratio = min(r / R, 1.0)
        return self.config.flow_velocity_max * (1.0 - ratio ** 2)

    def _wall_normal_at(self, pos: np.ndarray) -> np.ndarray:
        """Unit vector pointing radially inward (from wall toward center)."""
        yz = np.array([0.0, pos[1], pos[2]])
        dist = np.linalg.norm(yz)
        if dist < 1e-6:
            return np.array([0.0, 0.0, 1.0])
        return -yz / dist  # inward toward center

    def _distance_to_wall(self, pos: np.ndarray) -> float:
        """Distance from position to vessel wall (positive = inside)."""
        r = self._radial_distance(pos)
        return self.config.vessel_radius - r

    # ── Physics steps ──────────────────────────────────────────────────

    def _apply_shear_flow(self):
        """Apply Poiseuille flow force to all particles."""
        cfg = self.config
        n = len(self.positions)

        for i in range(n):
            v_flow = self._flow_velocity_at(self.positions[i])
            # Drag force toward flow velocity
            target_vx = v_flow
            self.forces[i, 0] += cfg.damping * (target_vx - self.velocities[i, 0])

    def _apply_brownian(self):
        """Apply Brownian (thermal) noise to all particles."""
        cfg = self.config
        n = len(self.positions)
        noise = self._rng.standard_normal((n, 3)).astype(np.float32)
        # Scale: sqrt(2 * kT * damping / dt) — simplified with tunable strength
        self.forces += cfg.brownian_strength * noise * math.sqrt(cfg.damping / cfg.dt)

    def _update_selectin_binding(self):
        """Stochastic selectin bind/unbind for rolling leukocytes."""
        cfg = self.config
        dt = cfg.dt
        n_leuko = cfg.n_leukocytes

        for i in range(n_leuko):
            idx = self.leukocyte_indices[i]
            state = self.leukocyte_states[i]
            d_wall = self._distance_to_wall(self.positions[idx])

            if state == LeukocyteState.MARGINATING and not self.selectin_bonds[i]:
                # Try to bind: probability depends on selectin expression
                # Find nearest endothelial cell
                expr = self._nearest_selectin_expression(self.positions[idx])
                p_bind = cfg.selectin_on_rate * expr * dt
                if self._rng.random() < p_bind:
                    self.selectin_bonds[i] = True

            elif self.selectin_bonds[i] and state in (LeukocyteState.ROLLING,
                                                        LeukocyteState.MARGINATING):
                # Try to unbind
                p_unbind = cfg.selectin_off_rate * dt
                if self._rng.random() < p_unbind:
                    self.selectin_bonds[i] = False

            # Apply rolling force when selectin-bound
            if self.selectin_bonds[i]:
                # Pull toward wall
                wall_norm = self._wall_normal_at(self.positions[idx])
                self.forces[idx] -= wall_norm * cfg.selectin_bond_force * 0.1

                # Rolling: reduce flow velocity to rolling speed
                v_flow = self._flow_velocity_at(self.positions[idx])
                target_vx = min(cfg.rolling_velocity, v_flow)
                self.forces[idx, 0] += cfg.damping * 2.0 * (target_vx - self.velocities[idx, 0])

    def _nearest_selectin_expression(self, pos: np.ndarray) -> float:
        """Get selectin expression of nearest endothelial cell."""
        if self.endo_positions is None or len(self.endo_positions) == 0:
            return 0.0
        dists = np.linalg.norm(self.endo_positions - pos, axis=1)
        nearest = np.argmin(dists)
        if dists[nearest] < 30.0:  # within ~30 μm
            return float(self.endo_selectin_expr[nearest])
        return 0.0

    def _update_integrin_activation(self):
        """Chemokine-triggered integrin conformational change (inside-out signaling)."""
        cfg = self.config
        dt = cfg.dt

        for i in range(cfg.n_leukocytes):
            idx = self.leukocyte_indices[i]
            state = self.leukocyte_states[i]

            if state in (LeukocyteState.ROLLING, LeukocyteState.ACTIVATING):
                # Get chemokine concentration at leukocyte position
                pos = self.positions[idx]
                r = self._radial_distance(pos)
                conc = self.chemokine_field.get_concentration_at(pos[0], r)

                if conc > cfg.chemokine_activation_threshold:
                    # Increase integrin activation
                    rate = dt / cfg.integrin_activation_time
                    self.integrin_activation[i] = min(1.0, self.integrin_activation[i] + rate)

    def _update_integrin_adhesion(self):
        """Apply firm adhesion force for activated integrins."""
        cfg = self.config

        for i in range(cfg.n_leukocytes):
            idx = self.leukocyte_indices[i]
            activation = self.integrin_activation[i]
            state = self.leukocyte_states[i]

            if activation > 0.5 and state in (LeukocyteState.ACTIVATING,
                                                LeukocyteState.ARRESTED,
                                                LeukocyteState.CRAWLING):
                # Strong adhesion toward wall
                d_wall = self._distance_to_wall(self.positions[idx])
                wall_norm = self._wall_normal_at(self.positions[idx])

                # Spring force to wall contact point
                adhesion_force = activation * cfg.integrin_bond_strength
                self.forces[idx] -= wall_norm * adhesion_force * 0.05

                # Also resist flow (firm arrest)
                self.forces[idx, 0] += cfg.damping * 5.0 * (0.0 - self.velocities[idx, 0])

    def _update_transmigration(self):
        """Handle transmigration through endothelial junction."""
        cfg = self.config
        dt = cfg.dt

        for i in range(cfg.n_leukocytes):
            idx = self.leukocyte_indices[i]
            state = self.leukocyte_states[i]

            if state == LeukocyteState.TRANSMIGRATING:
                # Progress through junction
                rate = dt / cfg.junction_disruption_time
                self.transmigration_progress[i] = min(1.0,
                    self.transmigration_progress[i] + rate)

                # Push radially outward through wall
                pos = self.positions[idx]
                yz = np.array([0.0, pos[1], pos[2]])
                dist = np.linalg.norm(yz)
                if dist > 1e-6:
                    outward = yz / dist
                    speed = cfg.transmigration_speed / 60.0  # μm/s
                    self.forces[idx, 1:3] += outward[1:3] * speed * cfg.damping

                # Weaken nearby junction
                self._weaken_junction_near(pos, self.transmigration_progress[i])

    def _weaken_junction_near(self, pos: np.ndarray, progress: float):
        """Reduce junction integrity near transmigrating leukocyte."""
        if self.endo_positions is None:
            return
        dists = np.linalg.norm(self.endo_positions - pos, axis=1)
        nearby = dists < 15.0  # within 15 μm
        weakening = progress * 0.5  # max 50% weakening
        self.endo_junction_integrity[nearby] = np.maximum(
            self.endo_junction_integrity[nearby] - weakening * self.config.dt,
            0.1  # never fully zero
        )

    def _update_tissue_migration(self):
        """Chemotaxis toward bacteria in extravascular tissue + phagocytosis."""
        cfg = self.config

        for i in range(cfg.n_leukocytes):
            idx = self.leukocyte_indices[i]
            state = self.leukocyte_states[i]

            if state == LeukocyteState.MIGRATED:
                pos = self.positions[idx]

                # Find or validate target bacterium
                target = int(self.leukocyte_target[i])
                if target < 0 or target >= cfg.n_bacteria or not self.bacteria_alive[target]:
                    alive_mask = self.bacteria_alive
                    if not np.any(alive_mask):
                        continue  # All bacteria destroyed
                    alive_idx = np.where(alive_mask)[0]
                    dists = np.linalg.norm(
                        self.bacteria_positions[alive_idx] - pos, axis=1)
                    target = int(alive_idx[np.argmin(dists)])
                    self.leukocyte_target[i] = target

                # Direction and distance to target bacterium
                bact_pos = self.bacteria_positions[target]
                direction = bact_pos - pos
                dist = np.linalg.norm(direction)

                if dist < cfg.bacteria_kill_distance:
                    # Phagocytosis: engulf the bacterium
                    self.bacteria_phagocytosis[target] = min(
                        1.0,
                        self.bacteria_phagocytosis[target] + cfg.dt / cfg.phagocytosis_time,
                    )
                    if self.bacteria_phagocytosis[target] >= 1.0:
                        self.bacteria_alive[target] = False
                        self.leukocyte_target[i] = -1
                    # Gentle force to stay near bacterium
                    if dist > 2.0:
                        speed = cfg.tissue_migration_speed / 60.0
                        self.forces[idx] += (direction / dist) * speed * cfg.damping
                else:
                    # Chemotax toward bacterium
                    speed = cfg.tissue_migration_speed / 60.0
                    self.forces[idx] += (direction / max(dist, 1e-6)) * speed * cfg.damping

    # ── State machine ──────────────────────────────────────────────────

    def _update_state_machine(self):
        """Transition leukocytes between diapedesis states."""
        cfg = self.config
        dt = cfg.dt

        for i in range(cfg.n_leukocytes):
            idx = self.leukocyte_indices[i]
            state = self.leukocyte_states[i]
            self.state_timers[i] += dt
            pos = self.positions[idx]
            d_wall = self._distance_to_wall(pos)

            if state == LeukocyteState.FLOWING:
                # → MARGINATING when close to wall
                if d_wall < cfg.margination_distance:
                    self.leukocyte_states[i] = LeukocyteState.MARGINATING
                    self.state_timers[i] = 0.0

            elif state == LeukocyteState.MARGINATING:
                # → ROLLING when selectin bonds form
                if self.selectin_bonds[i]:
                    self.leukocyte_states[i] = LeukocyteState.ROLLING
                    self.state_timers[i] = 0.0
                # → FLOWING if drifts away from wall
                elif d_wall > cfg.margination_distance * 1.5:
                    self.leukocyte_states[i] = LeukocyteState.FLOWING
                    self.state_timers[i] = 0.0

            elif state == LeukocyteState.ROLLING:
                # → ACTIVATING when chemokine detected
                r = self._radial_distance(pos)
                conc = self.chemokine_field.get_concentration_at(pos[0], r)
                if conc > cfg.chemokine_activation_threshold:
                    self.leukocyte_states[i] = LeukocyteState.ACTIVATING
                    self.state_timers[i] = 0.0
                # → FLOWING if selectin bond breaks
                elif not self.selectin_bonds[i]:
                    self.leukocyte_states[i] = LeukocyteState.FLOWING
                    self.state_timers[i] = 0.0

            elif state == LeukocyteState.ACTIVATING:
                # → ARRESTED when integrins fully activated
                if self.integrin_activation[i] > 0.9:
                    self.leukocyte_states[i] = LeukocyteState.ARRESTED
                    self.state_timers[i] = 0.0

            elif state == LeukocyteState.ARRESTED:
                # → CRAWLING after brief delay
                if self.state_timers[i] > 5.0 + self._rng.uniform(0, 5):
                    self.leukocyte_states[i] = LeukocyteState.CRAWLING
                    self.state_timers[i] = 0.0

            elif state == LeukocyteState.CRAWLING:
                # → TRANSMIGRATING when junction found and integrity low enough
                nearest_junction_integrity = self._nearest_junction_integrity(pos)
                crawl_done = self.state_timers[i] > cfg.crawl_to_junction_time
                if crawl_done and nearest_junction_integrity < 0.8:
                    self.leukocyte_states[i] = LeukocyteState.TRANSMIGRATING
                    self.state_timers[i] = 0.0
                elif crawl_done:
                    # Weaken junction while waiting
                    self._weaken_junction_near(pos, 0.3)

            elif state == LeukocyteState.TRANSMIGRATING:
                # → MIGRATED when fully through wall
                if self.transmigration_progress[i] >= 1.0:
                    self.leukocyte_states[i] = LeukocyteState.MIGRATED
                    self.state_timers[i] = 0.0

    def _nearest_junction_integrity(self, pos: np.ndarray) -> float:
        """Get junction integrity near position."""
        if self.endo_positions is None or len(self.endo_positions) == 0:
            return 1.0
        dists = np.linalg.norm(self.endo_positions - pos, axis=1)
        nearest = np.argmin(dists)
        if dists[nearest] < 20.0:
            return float(self.endo_junction_integrity[nearest])
        return 1.0

    # ── Integration ────────────────────────────────────────────────────

    def _integrate_positions(self):
        """Euler integration of positions from forces."""
        cfg = self.config
        dt = cfg.dt

        # Overdamped: v = F / damping
        self.velocities = self.forces / cfg.damping

        # Clamp velocities to prevent numerical blowup
        max_v = cfg.flow_velocity_max * 3.0
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        mask = speeds > max_v
        if np.any(mask):
            scale = np.where(mask, max_v / np.maximum(speeds, 1e-10), 1.0)
            self.velocities *= scale

        self.positions += self.velocities * dt

    def _enforce_vessel_boundary(self):
        """Keep particles inside/outside vessel as appropriate."""
        cfg = self.config
        R = cfg.vessel_radius
        L = cfg.vessel_length

        for i in range(len(self.positions)):
            # X boundary: wrap around (periodic)
            if self.positions[i, 0] < 0:
                self.positions[i, 0] += L
            elif self.positions[i, 0] > L:
                self.positions[i, 0] -= L

            # Radial boundary
            r = self._radial_distance(self.positions[i])
            ptype = self.particle_types[i]

            if ptype == ParticleType.RED_BLOOD_CELL:
                # RBCs stay inside vessel
                if r > R - self.radii[i]:
                    # Push back inside
                    scale = (R - self.radii[i]) / max(r, 1e-6)
                    self.positions[i, 1] *= scale
                    self.positions[i, 2] *= scale
                    # Kill radial velocity
                    self.velocities[i, 1] *= -0.3
                    self.velocities[i, 2] *= -0.3

            elif ptype == ParticleType.LEUKOCYTE:
                # Find leukocyte index
                leuko_idx = i  # leukocytes are first in array
                if leuko_idx < cfg.n_leukocytes:
                    state = self.leukocyte_states[leuko_idx]

                    if state <= LeukocyteState.CRAWLING:
                        # Keep inside vessel, but allow wall contact
                        max_r = R - self.radii[i] * 0.3
                        if r > max_r:
                            scale = max_r / max(r, 1e-6)
                            self.positions[i, 1] *= scale
                            self.positions[i, 2] *= scale

                    elif state == LeukocyteState.TRANSMIGRATING:
                        # Allow crossing wall
                        pass

                    elif state == LeukocyteState.MIGRATED:
                        # Keep outside vessel
                        min_r = R + self.radii[i]
                        if r < min_r:
                            scale = min_r / max(r, 1e-6)
                            self.positions[i, 1] *= scale
                            self.positions[i, 2] *= scale

    def _update_chemokine_field(self):
        """Update chemokine diffusion."""
        self.chemokine_field.update(self.config.dt)

    # ── Main step ──────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance simulation by one timestep.

        Uses GPU (Warp) kernels for force computation, integration, and
        boundary enforcement when available. State machine logic and
        stochastic binding decisions stay on CPU.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before step()")

        cfg = self.config

        if self._gpu is not None:
            self._step_gpu()
        else:
            self._step_cpu()

        # Update chemokine field (every 10 steps for performance)
        if self._step_count % 10 == 0:
            self._update_chemokine_field()

        self._time += cfg.dt
        self._step_count += 1

    def _step_cpu(self) -> None:
        """CPU (NumPy) simulation step."""
        # Zero forces
        self.forces[:] = 0.0

        # Apply forces
        self._apply_shear_flow()
        self._apply_brownian()
        self._update_selectin_binding()
        self._update_integrin_activation()
        self._update_integrin_adhesion()
        self._update_transmigration()
        self._update_tissue_migration()

        # State transitions
        self._update_state_machine()

        # Integrate
        self._integrate_positions()
        self._enforce_vessel_boundary()

    def _step_gpu(self) -> None:
        """GPU (Warp) simulation step — forces + integration + boundary on GPU."""
        cfg = self.config
        gpu = self._gpu
        n_leuko = cfg.n_leukocytes

        # CPU: stochastic binding / state logic (branchy, not GPU-friendly)
        self._update_selectin_binding_stochastic()
        self._update_integrin_activation()
        self._update_transmigration_progress()
        self._update_state_machine()

        # Upload current state to GPU
        gpu.upload(
            self.positions, self.velocities, self.forces,
            self.particle_types, self.radii,
            self.selectin_bonds.astype(np.int32),
            self.leukocyte_states.astype(np.int32),
            self.integrin_activation.astype(np.float32),
            self.transmigration_progress.astype(np.float32),
            n_leuko,
        )

        # GPU: parallel force computation
        gpu.zero_forces()
        gpu.apply_shear_flow(cfg.vessel_radius, cfg.flow_velocity_max, cfg.damping)
        gpu.apply_brownian(
            cfg.brownian_strength * math.sqrt(cfg.damping / cfg.dt),
            self._rng,
        )
        gpu.apply_selectin_rolling(
            cfg.vessel_radius, cfg.selectin_bond_force,
            cfg.rolling_velocity, cfg.damping,
        )
        gpu.apply_integrin_adhesion(
            cfg.vessel_radius, cfg.integrin_bond_strength, cfg.damping,
        )
        gpu.apply_transmigration(
            cfg.vessel_radius, cfg.transmigration_speed, cfg.damping,
        )

        # GPU: integration + boundary
        gpu.integrate(cfg.damping, cfg.flow_velocity_max * 3.0, cfg.dt)
        gpu.apply_boundary(cfg.vessel_radius, cfg.vessel_length)
        gpu.sync()

        # Download results back to CPU
        gpu.download(self.positions, self.velocities)

        # CPU: tissue migration (needs chemokine field gradient, not worth GPU)
        self._update_tissue_migration()

    def _update_selectin_binding_stochastic(self):
        """CPU-only stochastic selectin bind/unbind (no force computation).

        Separated from _update_selectin_binding so GPU path can handle
        forces while CPU handles stochastic decisions.
        """
        cfg = self.config
        dt = cfg.dt

        for i in range(cfg.n_leukocytes):
            idx = self.leukocyte_indices[i]
            state = self.leukocyte_states[i]

            if state not in (LeukocyteState.MARGINATING, LeukocyteState.ROLLING):
                continue

            d_wall = self._distance_to_wall(self.positions[idx])
            near_wall = d_wall < self.radii[idx] * 3

            if not self.selectin_bonds[i] and near_wall:
                sel_expr = self._nearest_selectin_expression(self.positions[idx])
                p_bind = cfg.selectin_on_rate * sel_expr * dt
                if self._rng.random() < p_bind:
                    self.selectin_bonds[i] = True

            elif self.selectin_bonds[i]:
                p_unbind = cfg.selectin_off_rate * dt
                if self._rng.random() < p_unbind:
                    self.selectin_bonds[i] = False

    def _update_transmigration_progress(self):
        """CPU: update transmigration progress counter and junction weakening."""
        cfg = self.config
        dt = cfg.dt

        for i in range(cfg.n_leukocytes):
            idx = self.leukocyte_indices[i]
            state = self.leukocyte_states[i]

            if state == LeukocyteState.TRANSMIGRATING:
                rate = dt / cfg.junction_disruption_time
                self.transmigration_progress[i] = min(
                    1.0, self.transmigration_progress[i] + rate)
                self._weaken_junction_near(
                    self.positions[idx], self.transmigration_progress[i])

    def get_snapshot(self) -> Dict[str, Any]:
        """Get current state for visualization.

        Returns dict with arrays suitable for Three.js InstancedMesh rendering.
        """
        cfg = self.config
        n_leuko = cfg.n_leukocytes
        n_rbc = cfg.n_rbc

        # Leukocyte colors by state
        state_colors = {
            LeukocyteState.FLOWING: [0.9, 0.9, 0.9],       # White
            LeukocyteState.MARGINATING: [1.0, 1.0, 0.7],    # Light yellow
            LeukocyteState.ROLLING: [1.0, 0.85, 0.0],       # Yellow
            LeukocyteState.ACTIVATING: [1.0, 0.55, 0.0],    # Orange
            LeukocyteState.ARRESTED: [1.0, 0.3, 0.0],       # Red-orange
            LeukocyteState.CRAWLING: [0.9, 0.1, 0.1],       # Red
            LeukocyteState.TRANSMIGRATING: [0.8, 0.0, 0.5], # Magenta
            LeukocyteState.MIGRATED: [0.2, 0.8, 0.2],       # Green
        }

        leuko_colors = np.array([
            state_colors[LeukocyteState(s)] for s in self.leukocyte_states
        ], dtype=np.float32)

        rbc_colors = np.full((n_rbc, 3), [0.7, 0.1, 0.1], dtype=np.float32)

        # Endothelial colors: pink → red with inflammation
        n_endo = len(self.endo_positions)
        endo_colors = np.zeros((n_endo, 3), dtype=np.float32)
        for j in range(n_endo):
            infl = self.endo_selectin_expr[j]
            endo_colors[j] = [0.9, 0.75 - 0.4 * infl, 0.75 - 0.5 * infl]

        metrics = self._compute_metrics()

        return {
            "time": self._time,
            "step": self._step_count,
            # Leukocytes
            "leukocyte_positions": self.positions[:n_leuko].tolist(),
            "leukocyte_colors": leuko_colors.tolist(),
            "leukocyte_radii": self.radii[:n_leuko].tolist(),
            "leukocyte_states": self.leukocyte_states.tolist(),
            "integrin_activation": self.integrin_activation.tolist(),
            "transmigration_progress": self.transmigration_progress.tolist(),
            # RBCs
            "rbc_positions": self.positions[n_leuko:n_leuko + n_rbc].tolist(),
            "rbc_colors": rbc_colors.tolist(),
            # Endothelial cells
            "endo_positions": self.endo_positions.tolist(),
            "endo_colors": endo_colors.tolist(),
            "endo_selectin_expr": self.endo_selectin_expr.tolist(),
            "endo_junction_integrity": self.endo_junction_integrity.tolist(),
            # Vessel geometry
            "vessel_length": cfg.vessel_length,
            "vessel_radius": cfg.vessel_radius,
            # Bacteria
            "bacteria_positions": self.bacteria_positions.tolist(),
            "bacteria_alive": self.bacteria_alive.tolist(),
            "bacteria_phagocytosis": self.bacteria_phagocytosis.tolist(),
            "leukocyte_target": self.leukocyte_target.tolist(),
            # Metrics
            "metrics": metrics,
        }

    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute summary metrics for current state."""
        states = self.leukocyte_states
        n = len(states)

        state_counts = {}
        for s in LeukocyteState:
            state_counts[s.name.lower()] = int(np.sum(states == s.value))

        # Average rolling velocity
        rolling_mask = states == LeukocyteState.ROLLING
        if np.any(rolling_mask):
            rolling_indices = self.leukocyte_indices[rolling_mask]
            avg_rolling_v = float(np.mean(np.abs(self.velocities[rolling_indices, 0])))
        else:
            avg_rolling_v = 0.0

        # Transmigration rate (cells that reached MIGRATED)
        n_migrated = state_counts.get("migrated", 0)

        n_bacteria_alive = int(np.sum(self.bacteria_alive)) if self.bacteria_alive is not None else 0
        n_bacteria_total = len(self.bacteria_alive) if self.bacteria_alive is not None else 0

        metrics = {
            "state_counts": state_counts,
            "avg_rolling_velocity": avg_rolling_v,
            "n_migrated": n_migrated,
            "time": self._time,
            "avg_integrin_activation": float(np.mean(self.integrin_activation)),
            "avg_junction_integrity": float(np.mean(self.endo_junction_integrity)),
            "bacteria_alive": n_bacteria_alive,
            "bacteria_total": n_bacteria_total,
        }

        return metrics

    def run(self, duration: float, fps: int = 30) -> List[Dict]:
        """Run simulation and collect frames.

        Args:
            duration: Total simulation time in seconds
            fps: Frames per second for output

        Returns:
            List of snapshot dicts, one per frame
        """
        if not self._initialized:
            self.initialize()

        dt = self.config.dt
        steps_per_frame = max(1, int(1.0 / (fps * dt)))
        total_steps = int(duration / dt)
        frames = []

        for step_i in range(total_steps):
            self.step()

            if step_i % steps_per_frame == 0:
                frames.append(self.get_snapshot())

        log.info(f"DiapedesisSim: {total_steps} steps, {len(frames)} frames, "
                 f"final metrics: {self._compute_metrics()['state_counts']}")

        return frames

    # ── Preset Scenarios ───────────────────────────────────────────────

    @staticmethod
    def healthy_vessel() -> DiapedesisSim:
        """Healthy vessel: low inflammation, minimal recruitment."""
        return DiapedesisSim(DiapedesisConfig(
            inflammation_level=0.1,
            n_leukocytes=15,
            n_rbc=200,
        ))

    @staticmethod
    def mild_inflammation() -> DiapedesisSim:
        """Mild inflammation: moderate rolling, some arrests."""
        return DiapedesisSim(DiapedesisConfig(
            inflammation_level=0.5,
            n_leukocytes=20,
            n_rbc=200,
        ))

    @staticmethod
    def severe_inflammation() -> DiapedesisSim:
        """Severe inflammation: rapid recruitment, many transmigrate."""
        return DiapedesisSim(DiapedesisConfig(
            inflammation_level=0.9,
            n_leukocytes=30,
            n_rbc=200,
            chemokine_activation_threshold=0.15,
        ))

    @staticmethod
    def lad1_no_lfa1() -> DiapedesisSim:
        """LAD-1 (LFA-1 deficiency): normal rolling, NO firm adhesion."""
        return DiapedesisSim(DiapedesisConfig(
            inflammation_level=0.7,
            n_leukocytes=20,
            n_rbc=200,
            integrin_on_rate=0.0,       # No integrin engagement
            integrin_bond_strength=0.0,
        ))

    @staticmethod
    def lad2_no_selectin_ligand() -> DiapedesisSim:
        """LAD-2 (selectin ligand deficiency): NO rolling at all."""
        return DiapedesisSim(DiapedesisConfig(
            inflammation_level=0.7,
            n_leukocytes=20,
            n_rbc=200,
            selectin_on_rate=0.0,  # Cannot bind selectins
        ))

    @staticmethod
    def lad3_no_kindlin3() -> DiapedesisSim:
        """LAD-3 (kindlin-3 deficiency): rolling but integrin activation fails."""
        return DiapedesisSim(DiapedesisConfig(
            inflammation_level=0.7,
            n_leukocytes=20,
            n_rbc=200,
            integrin_activation_time=1e6,  # Effectively never activates
        ))
