#!/usr/bin/env python3
"""
Spatial Module
==============

Handles 3D spatial fields and diffusion.

Features:
- 3D grid for diffusible molecules
- Diffusion solver (explicit)
- Multiple fields (O2, glucose, cytokines)
- Gradient calculation
- Source/sink management
"""

import sys
sys.path.insert(0, '..')

import logging

import numpy as np
from typing import Dict, List, Any, Tuple

from cognisom.core.module_base import SimulationModule
from cognisom.core.event_bus import EventTypes

_log = logging.getLogger(__name__)


#: First-order decay of the chemokine field, 1/h, from a half-life of 30
#: minutes. That figure is CXCL12's half-life in the BLOODSTREAM (review,
#: PMC6582792: protease degradation, GAG binding, sequestration), and is
#: the only measured persistence available; tissue-bound chemokine very
#: likely persists longer, and CD26 -- the protease that clips it -- is
#: lost on activated CAFs (PMC6639198). With D = 100 um^2/s the decay
#: length sqrt(D/k) is 510 um, so across this 200 um tissue the gradient
#: is shallow but present. Lengthening the half-life lengthens it.
#: Lactate (90 Da) in interstitium, um^2/s. ~1,000 in water at 37 C by
#: Stokes-Einstein; interstitial hindrance roughly halves it. ESTIMATE.
#: It does not matter here, and that is the finding: see
#: _lactate_vessel_level.
LACTATE_DIFFUSION_UM2_S = 700.0

#: IFN-gamma (a 34 kDa dimer) in interstitium, um^2/s. ESTIMATE from size:
#: a protein of this mass moves at ~50-100 um^2/s in water and slower in
#: tissue. What sets its RANGE is not this number but consumption --
#: Oyler-Yaniv et al., Immunity 2017;46:609-20 (PMID 28389069) show
#: cytokine propagation is a diffusion-CONSUMPTION mechanism whose length
#: scale sqrt(D/k) shrinks as the consuming cell density rises. That is
#: why the range is tens of micrometres and why it is tunable, and it is
#: the reason this field is worth solving rather than counting neighbours.
IFN_GAMMA_DIFFUSION_UM2_S = 50.0

#: First-order consumption of IFN-gamma by one cell, 1/h, applied at the
#: cell's voxel like every other uptake here. No measured per-cell rate
#: was found, so this is an ESTIMATE -- but it is pinned to the one thing
#: that IS measured: the signal's range in REAL tissue. The field obeys
#: L = sqrt(D / k_eff) with k_eff the volume-averaged uptake, verified
#: against the solver (see tests); at 200/h that gives
#:
#:     39% packed (near real tissue)   L = 48 um   -- Oyler-Yaniv's tens
#:      1% packed (these presets)      L = 276 um  -- a broad field
#:
#: so the value reproduces the measured paracrine range at the density
#: the measurement was made in, and the presets' sparse block gets a
#: broad field as a CONSEQUENCE, stated rather than tuned away. A
#: sourced per-cell uptake replaces this outright.
IFN_GAMMA_UPTAKE_PER_H = 200.0

#: Testosterone (288 Da) in interstitium, um^2/s. No measured tissue value
#: was found. Stokes-Einstein for a ~290 Da solute gives ~650 in water at
#: 37 C (glucose, 180 Da, is 600 above and ~670 measured), and interstitial
#: hindrance roughly halves that; 500 is the midpoint: ESTIMATE. The
#: field's Thiele modulus is << 1 on the shipped grid either way, so the
#: value moves nothing measurable here -- the field is flat at the wall
#: level, which is itself a prediction: uniform intraprostatic free T.
TESTOSTERONE_DIFFUSION_UM2_S = 500.0

CHEMOKINE_HALF_LIFE_HOURS = 0.5
CHEMOKINE_DECAY_PER_HOUR = float(np.log(2.0) / CHEMOKINE_HALF_LIFE_HOURS)


class SpatialField:
    """3D diffusible field"""
    
    def __init__(self, name: str, grid_size: Tuple[int, int, int], 
                 resolution: float, diffusion_coeff: float,
                 decay_rate: float = 0.0):
        self.name = name
        self.grid_size = grid_size
        self.resolution = resolution  # μm per voxel
        self.diffusion_coeff = diffusion_coeff  # μm²/s
        # Uniform first-order decay, 1/h. A secreted signal that is
        # never degraded accumulates without bound; this is what gives
        # a chemokine a finite decay length and the steady state a
        # non-singular operator even with no vessel and no uptake.
        self.decay_rate = float(decay_rate)
        
        # Concentration field
        self.concentration = np.zeros(grid_size, dtype=np.float32)
        
        # Sources and sinks
        self.sources = []  # [(position, rate), ...]
        self.sinks = []    # [(position, rate), ...]

        # First-order uptake, [(index, k), ...] with k in 1/h. This is
        # the physically right sink for a metabolising cell -- see
        # add_uptake -- and unlike `sinks` it keeps the steady-state
        # operator non-singular and its solution non-negative.
        self.uptake = []

        # Dirichlet voxels, [(index, value), ...]: a capillary holds the
        # interstitium at the plasma level it carries. This is the
        # classical Krogh-Erlang vessel-wall condition.
        self.dirichlet = []

        # Cached sparse factorisation of the steady-state operator, with
        # the key it was built for. The operator depends on the Dirichlet
        # pattern (capillaries, which never move) and the uptake pattern
        # (cells, which move only on division or death), so one
        # factorisation serves many solves: MEASURED at 6 distinct
        # patterns across a 240-update 24 h Baseline run.
        self._lu = None
        self._lu_key = None
        self._lu_is_dir = None
        self.lu_rebuilds = 0
        self.last_solver = 'none'

        # Each diagnostic warns ONCE per field, not once per update. The
        # under-diffusion warning fired 720 times in a 24 h run (240
        # updates x 3 fields), all of it describing the transport of an
        # identically zero field.
        self._cfl_warned = False
        self._solver_warned = False
    
    def add_source(self, position: np.ndarray, rate: float):
        """Add source"""
        idx = self._position_to_index(position)
        self.sources.append((idx, rate))
    
    def add_sink(self, position: np.ndarray, rate: float):
        """Add sink"""
        idx = self._position_to_index(position)
        self.sinks.append((idx, rate))
    
    def add_uptake(self, position: np.ndarray, k: float):
        """Add FIRST-ORDER uptake of strength ``k`` (1/h) at ``position``.

        A metabolising cell consumes at a near-constant rate: O2 uptake is
        Michaelis-Menten with Km ~ 1 mmHg, which is far below every
        concentration in this model, so the kinetics are effectively
        zeroth order. That is exactly why Krogh and Erlang (1919) assume
        constant consumption, and why this engine's CellularModule
        subtracts a flat rate per hour.

        A zeroth-order sink cannot be used in a steady-state solve: it is
        unbounded below, so it drives the concentration negative wherever
        demand exceeds local supply, with no mechanism to stop it. This
        method therefore takes the SECANT LINEARISATION of the
        zeroth-order rate through the arterial operating point,

            U(C) = k * C,    k = q_volumetric / C_arterial

        so that at C = C_arterial the volumetric consumption is exactly
        the cell's demand. Three consequences, all of them deliberate:

        1. The operator stays LINEAR, so one LU factorisation serves every
           solve that shares an occupancy pattern.
        2. The operator is an M-matrix (see _assemble_matrix), so the
           discrete maximum principle gives 0 <= C <= C_blood outright.
           No clipping is applied, so the solve cannot destroy mass the
           way the clipped explicit path did.
        3. It UNDER-consumes wherever C < C_arterial, so it is
           conservative: it cannot manufacture a hypoxia the zeroth-order
           model would not show.

        If the tissue extent is ever enlarged until uptake actually bites
        (see thiele_modulus), the upgrade is Michaelis-Menten uptake under
        a damped Newton iteration -- the standard treatment (McGuire &
        Secomb 2001; Goldman 2008). On the shipped 200x200x100 um grid it
        would make no difference: measured phi << 1.
        """
        if k <= 0.0:
            return
        idx = self._position_to_index(position)
        self.uptake.append((idx, float(k)))

    def add_dirichlet(self, position: np.ndarray, value: float):
        """Pin the field at ``position`` to ``value`` (a Dirichlet voxel).

        Used for capillaries: the vessel wall is held at the plasma level
        the blood carries. This is the Krogh-Erlang inner boundary
        condition and introduces no free parameter -- the value is the
        arterial level the vascular module already uses.
        """
        idx = self._position_to_index(position)
        self.dirichlet.append((idx, float(value)))

    def clear_sources_sinks(self):
        """Clear all sources, sinks, uptake and Dirichlet voxels.

        Callers that re-derive their terms from a moving cell population
        must CLEAR THEN RE-REGISTER every step. ``add_*`` append, and
        ``update`` applies every entry, so re-registering without
        clearing grows the lists without bound and multiplies the
        deposited amount by the step count -- the hazard
        MorphogenModule._update_gradients_spatial guards against with a
        ``_registered_sources`` set. That set-based guard is right for a
        static organizer and WRONG for cells: it cannot retract the entry
        of a cell that has died or changed voxel. Clearing first makes
        the operation idempotent instead, which is what a moving
        population needs.
        """
        self.sources.clear()
        self.sinks.clear()
        self.uptake.clear()
        self.dirichlet.clear()
    
    #: Stability limit for the explicit 3D 7-point Laplacian:
    #: D*dt/dx^2 <= 1/6. Above it the scheme amplifies high-frequency
    #: modes every step instead of damping them.
    CFL_LIMIT_3D = 1.0 / 6.0
    #: Fraction of the limit to actually run at.
    CFL_SAFETY = 0.9
    #: Cap on sub-steps per update, so a pathological diffusion number
    #: degrades into a warning rather than an unbounded loop.
    MAX_SUBSTEPS = 512

    SECONDS_PER_HOUR = 3600.0

    def update(self, dt: float):
        """Update field via diffusion.

        ``dt`` is in HOURS (the simulation's time unit) while
        ``diffusion_coeff`` is in um^2/s, so the conversion below is
        required. It was missing: ``D * lap * dt / dx^2`` was evaluated
        with mixed units, understating the transported amount 3600-fold.

        The step is then sub-divided to satisfy the explicit-Euler
        stability limit. Previously there was no check at all -- with the
        oxygen field's D = 2000 um^2/s on a 10 um grid the diffusion
        number is ~7200 against a limit of 1/6, so the scheme was
        unconditionally unstable. It did not visibly blow up only because
        ``np.maximum(concentration, 0)`` clipped the resulting oscillation
        every step, which also silently destroys mass.
        """
        dt_seconds = dt * self.SECONDS_PER_HOUR
        dx2 = self.resolution ** 2

        # A driven field whose diffusion time across the domain is short
        # against dt has already reached its steady state by the end of
        # the step; integrating towards it with sub-stepped explicit
        # Euler is both wrong (the sub-step cap applies only ~1% of the
        # oxygen transport) and more expensive than solving for it
        # directly. Try the steady state first.
        # A field with no terms AND no material in it has nothing to
        # transport. Diffusing identically zero is exactly zero, so this
        # only skips work -- but it skips a LOT of it: the cytokine field
        # is driven by nobody, and it was running 512 explicit sub-steps
        # per update, 240 times a run, on an all-zero array, and emitting
        # an under-diffusion warning about it.
        if (not self.sources and not self.sinks and not self.uptake
                and not self.dirichlet and not self.concentration.any()):
            self.last_solver = 'idle'
            return

        if self._steady_state_applies(dt):
            if self._solve_steady_state():
                self.last_solver = 'steady_state'
                return

        # Undriven, decaying, and too diffusive for the explicit path: a
        # secreted field after its last source is gone. The explicit
        # step would apply a fifth of the transport and warn; the steady
        # state is zero, which throws the residual away. Between those,
        # a field whose diffusion number says it crosses the domain
        # within the step is well described as fully mixed, decaying by
        # exp(-k dt). Mass is conserved up to the decay, which neither
        # alternative manages.
        if (not self.dirichlet and not self.sources and not self.uptake
                and self.decay_rate > 0.0 and self._explicit_is_capped(dt)):
            level = (float(self.concentration.mean())
                     * float(np.exp(-self.decay_rate * dt)))
            self.concentration.fill(np.float32(level))
            self.last_solver = 'mixed'
            return

        self.last_solver = 'explicit'

        # Sources and sinks act over the whole interval, once.
        for idx, rate in self.sources:
            if self._is_valid_index(idx):
                self.concentration[idx] += rate * dt

        for idx, rate in self.sinks:
            if self._is_valid_index(idx):
                removal = min(rate * dt, self.concentration[idx])
                self.concentration[idx] -= removal

        # First-order uptake over the interval, applied EXACTLY as
        # exp(-k*dt) rather than as a linearised 1 - k*dt. This branch
        # exists because the explicit path is reachable with uptake
        # registered -- a field driven by cells but with no vessel
        # (spatial + cellular enabled, vascular disabled) is not
        # quasi-steady by _steady_state_applies' DRIVEN test, and
        # without this loop its uptake was silently dropped and the
        # field sat at its initial level for the whole run. The
        # exponential form is unconditionally stable and cannot go
        # negative for any k or dt, unlike the sink loop above, which
        # needs its min() clamp.
        for idx, rate in self.uptake:
            if self._is_valid_index(idx) and rate > 0.0:
                self.concentration[idx] *= np.exp(-rate * dt)
        if self.decay_rate > 0.0:
            self.concentration *= np.float32(np.exp(-self.decay_rate * dt))

        if self.diffusion_coeff <= 0 or dt_seconds <= 0:
            np.maximum(self.concentration, 0, out=self.concentration)
            return

        diffusion_number = self.diffusion_coeff * dt_seconds / dx2
        max_stable = self.CFL_LIMIT_3D * self.CFL_SAFETY

        n_sub = max(1, int(np.ceil(diffusion_number / max_stable)))
        coeff = diffusion_number / n_sub

        if n_sub > self.MAX_SUBSTEPS:
            self._warn_under_diffused(diffusion_number, n_sub, max_stable)
            n_sub = self.MAX_SUBSTEPS
            # Each sub-step must still satisfy the stability limit --
            # spreading an over-limit coefficient across capped sub-steps
            # leaves every one of them unstable, which overflows to NaN
            # rather than merely being inaccurate.
            coeff = max_stable

        # Pin the Dirichlet voxels around the transport, not just before it.
        # A vessel voxel held at arterial level is a boundary CONDITION: if
        # the sub-step loop is allowed to diffuse it away, the source leaks
        # into the field and then vanishes, and on a coarse grid (where the
        # explicit path is actually taken) the field relaxes to identically
        # zero -- which is exactly the symptom this coupling work set out to
        # remove, reappearing only in explicit mode.
        self._apply_dirichlet()
        for _ in range(n_sub):
            self.concentration += coeff * self._compute_laplacian()
            self._apply_dirichlet()

        # Non-negative
        np.maximum(self.concentration, 0, out=self.concentration)

    def _apply_dirichlet(self):
        """Hold every pinned voxel at its boundary value."""
        for idx, value in self.dirichlet:
            if self._is_valid_index(idx):
                self.concentration[idx] = value

    def _warn_under_diffused(self, diffusion_number, n_sub, max_stable):
        """Warn ONCE per field that the explicit path is under-diffusing.

        This used to fire from inside update(), i.e. once per update: 240
        updates x 3 fields = 720 identical records in a 24 h run. The
        condition is a property of (D, dx, dt), which do not change
        between updates, so one record per field says everything the 720
        said.
        """
        if self._cfl_warned:
            return
        self._cfl_warned = True
        _log.warning(
            "Field '%s': D*dt/dx^2 = %.1f needs %d explicit sub-steps to "
                "stay stable, above the cap of %d. Running %d stable "
                "sub-steps instead, so the field is UNDER-DIFFUSED over this "
                "interval (about %.1f%% of the transport applied). At this "
                "diffusion number the field is effectively at steady state "
                "within one step, so an implicit or steady-state solve is the "
                "right method here, not explicit Euler.",
                self.name, diffusion_number, n_sub, self.MAX_SUBSTEPS,
                self.MAX_SUBSTEPS,
                100.0 * self.MAX_SUBSTEPS * max_stable / diffusion_number,
            )

    # ── Quasi-steady-state solve ────────────────────────────────────

    #: Minimum Fourier number D*dt/L^2 at which the field is treated as
    #: having equilibrated within one update. L is the LARGEST domain
    #: extent, because the slowest-decaying diffusive mode spans the
    #: longest dimension and so sets the relaxation time; taking the
    #: smallest extent would OVERstate Fo and could call an elongated
    #: domain quasi-steady when its long axis is still relaxing.
    #:
    #: On the shipped configuration (dt = 0.1 h, dx = 10 um, 20x20x10
    #: voxels so L = 200 um): oxygen Fo = 18, glucose 5.4, cytokine 0.9.
    #: Oxygen and glucose are solved directly; cytokine's long axis is
    #: genuinely still relaxing within a step, so it is integrated.
    QUASI_STEADY_FOURIER = 1.0

    def fourier_number(self, dt: float) -> float:
        """D*dt / L^2 with L the largest domain extent, dimensionless."""
        if self.diffusion_coeff <= 0 or dt <= 0:
            return 0.0
        extent = max(n * self.resolution for n in self.grid_size)
        if extent <= 0:
            return 0.0
        return float(self.diffusion_coeff * self.SECONDS_PER_HOUR * dt
                     / (extent * extent))

    def thiele_modulus(self) -> float:
        """L * sqrt(k_mean / D): diffusion time over uptake time.

        L is the LARGEST domain extent, so phi is exactly the domain
        length measured in diffusion lengths, L / sqrt(D/k) -- the
        standard Thiele modulus.

        This is the number that decides whether a gradient can exist at
        all. phi << 1 means transport is far faster than consumption, so
        the field is uniform to O(phi^2) NO MATTER how fine the grid is;
        phi >~ 1 means the geometry is capable of producing a real
        gradient, and therefore of producing hypoxia by geometry alone.

        Reported in SpatialModule.get_state() precisely so that the
        answer is visible rather than assumed. MEASURED on the shipped
        200x200x100 um grid, all nine modules, 24 h Baseline: oxygen
        phi = 0.0119, and the vessel-to-corner drop is 0.002% -- i.e.
        this tissue extent is NOT diffusion-limited for O2 or glucose,
        and no correct diffusion solve can produce a gradient on it.
        That is why cell O2 stays on VascularModule's lumped Krogh
        conductance rather than being read from this field: a
        field-sourced cell O2 would sit at the arterial level
        everywhere and no preset could produce hypoxia.
        """
        if self.diffusion_coeff <= 0 or not self.uptake:
            return 0.0
        n_vox = int(np.prod(self.grid_size))
        n_dir = len({idx for idx, _ in self.dirichlet})
        n_free = max(n_vox - n_dir, 1)
        k_mean = sum(k for _, k in self.uptake) / n_free
        if k_mean <= 0:
            return 0.0
        extent = max(n * self.resolution for n in self.grid_size)
        return float(extent * np.sqrt(
            k_mean / (self.diffusion_coeff * self.SECONDS_PER_HOUR)))

    def _steady_state_applies(self, dt: float) -> bool:
        """Whether the steady-state solve is both valid and wanted.

        Three conditions, each necessary:

        * DRIVEN. There must be a Dirichlet voxel or a rate source.
          A field carrying only an initial blob has no steady state but
          the trivial one, and replacing its transient with zero would
          discard exactly the thing being modelled. (This is also what
          keeps the shipped stability tests on the explicit path: they
          seed a blob and register nothing.)
        * REGULAR. There must be a Dirichlet voxel or positive uptake,
          otherwise the pure-Neumann operator is singular -- the constant
          vector is in its null space and the steady state is undefined.
        * QUASI-STEADY. Fo >= QUASI_STEADY_FOURIER, i.e. the field really
          does equilibrate within the step. Below that the transient is
          the physical answer and explicit integration is correct.
        """
        driven = bool(self.dirichlet) or bool(self.sources)
        regular = (bool(self.dirichlet) or bool(self.uptake)
                   or self.decay_rate > 0.0)
        if not (driven and regular):
            return False
        if self.fourier_number(dt) >= self.QUASI_STEADY_FOURIER:
            return True
        # Not quasi-steady at the DOMAIN scale, but if the explicit path
        # would hit its sub-step cap it applies only a fraction of the
        # transport, and a steady state taken a little early is the
        # smaller error by far. Measured on the cytokine field (D = 100
        # um^2/s, 200 um box, 0.1 h): domain Fo = 0.9, just under the
        # threshold, so it went explicit -- where the voxel-scale
        # diffusion number of 360 needed 2,400 sub-steps against a cap
        # of 512 and 21% of the transport was applied, with a warning
        # every run. The warning's own text says a steady solve is the
        # right method there. Now it is used.
        return self._explicit_is_capped(dt)

    def _explicit_is_capped(self, dt: float) -> bool:
        """Would the explicit path need more sub-steps than it is allowed?"""
        if self.diffusion_coeff <= 0.0 or dt <= 0.0:
            return False
        diffusion_number = (self.diffusion_coeff * dt * self.SECONDS_PER_HOUR
                            / self.resolution ** 2)
        max_stable = self.CFL_LIMIT_3D * self.CFL_SAFETY
        return int(np.ceil(diffusion_number / max_stable)) > self.MAX_SUBSTEPS

    def _assemble_matrix(self):
        """Assemble the steady-state operator; return (A_csc, is_dirichlet).

        Discretises, on the voxel grid,

            D grad^2 C - k(x) C + s(x) = 0

        with two boundary conditions:

        * Homogeneous NEUMANN (no-flux) on the domain boundary. A tissue
          sample exchanges with the body only through its vessels, and
          this is the same outer closure as the Krogh cylinder's no-flux
          tissue radius. It is imposed by simply omitting a boundary
          voxel's missing neighbours from BOTH the off-diagonals and the
          degree counted in the diagonal, which is the standard
          ghost-node-free form of zero normal flux.
        * DIRICHLET C = C_blood on every voxel a capillary passes
          through (Krogh-Erlang vessel wall).

        Note the explicit path's _compute_laplacian writes interior
        points only and so leaves the boundary SHELL frozen at its
        initial value -- a different, absorbing closure. That was
        harmless while every field was identically zero, and the shipped
        stability tests now pin it (test_mass_is_conserved... asserts the
        shell stays at zero), so it is deliberately left alone. The two
        paths differ in their boundary treatment and the docstring of
        each says which it uses.

        M-MATRIX PROPERTY, which is what makes the result trustworthy:
        off-diagonals are +alpha >= 0, the diagonal is
        -(alpha*degree + k) <= 0, and every row is weakly diagonally
        dominant with STRICT dominance wherever k > 0 or the row is
        Dirichlet. _steady_state_applies guarantees at least one such
        row, and the 7-point grid graph is connected, so -A is an
        irreducibly diagonally dominant M-matrix. Hence (-A)^-1 >= 0
        entrywise and the solution obeys the discrete maximum principle:
        0 <= C <= max(C_dirichlet, max(s/k)). Non-negativity is a
        theorem here, not something enforced by clipping.
        """
        import scipy.sparse as sp

        nx, ny, nz = self.grid_size
        n = nx * ny * nz
        ids = np.arange(n, dtype=np.int64).reshape(nx, ny, nz)
        alpha = (self.diffusion_coeff * self.SECONDS_PER_HOUR
                 / (self.resolution ** 2))

        k = np.zeros(n, dtype=np.float64)
        for idx, rate in self.uptake:
            if self._is_valid_index(idx):
                k[ids[idx]] += rate
        # Decay is uptake everywhere: it lands on the diagonal of every
        # row, which is what makes a secreted field's operator regular.
        k += self.decay_rate

        is_dir = np.zeros(n, dtype=bool)
        for idx, _value in self.dirichlet:
            if self._is_valid_index(idx):
                is_dir[ids[idx]] = True

        # The six in-domain neighbour relations. A pair (a, b) means
        # "voxel a has neighbour b"; absent pairs at the boundary are
        # exactly homogeneous Neumann.
        pairs = (
            (ids[:-1, :, :], ids[1:, :, :]),
            (ids[1:, :, :], ids[:-1, :, :]),
            (ids[:, :-1, :], ids[:, 1:, :]),
            (ids[:, 1:, :], ids[:, :-1, :]),
            (ids[:, :, :-1], ids[:, :, 1:]),
            (ids[:, :, 1:], ids[:, :, :-1]),
        )
        rows = np.concatenate([a.ravel() for a, _ in pairs])
        cols = np.concatenate([b.ravel() for _, b in pairs])
        degree = np.bincount(rows, minlength=n).astype(np.float64)

        keep = ~is_dir[rows]
        off_rows, off_cols = rows[keep], cols[keep]
        off_vals = np.full(off_rows.size, alpha, dtype=np.float64)

        diag_rows = np.arange(n, dtype=np.int64)
        diag_vals = np.where(is_dir, 1.0, -(alpha * degree + k))

        A = sp.csc_matrix(
            (np.concatenate([off_vals, diag_vals]),
             (np.concatenate([off_rows, diag_rows]),
              np.concatenate([off_cols, diag_rows]))),
            shape=(n, n),
        )
        return A, is_dir

    def _assemble_rhs(self, is_dir):
        """Right-hand side: the Dirichlet value on pinned rows, -s
        elsewhere. Cheap, and rebuilt every solve, because only the
        matrix is worth caching."""
        nx, ny, nz = self.grid_size
        n = nx * ny * nz
        ids = np.arange(n, dtype=np.int64).reshape(nx, ny, nz)

        s = np.zeros(n, dtype=np.float64)
        for idx, rate in self.sources:
            if self._is_valid_index(idx):
                s[ids[idx]] += rate
        for idx, rate in self.sinks:
            if self._is_valid_index(idx):
                s[ids[idx]] -= rate

        b = -s
        for idx, value in self.dirichlet:
            if self._is_valid_index(idx):
                b[ids[idx]] = value
        return b

    def _operator_key(self):
        """Identity of the steady-state operator, so the factorisation is
        reused whenever it has not actually changed."""
        return (
            tuple(sorted({(int(i), int(j), int(kk))
                          for (i, j, kk), _ in self.dirichlet})),
            tuple(sorted((int(i), int(j), int(kk), round(float(r), 12))
                         for (i, j, kk), r in self.uptake)),
            round(float(self.diffusion_coeff), 9),
            round(float(self.decay_rate), 12),
            round(float(self.resolution), 9),
            tuple(self.grid_size),
        )

    def _solve_steady_state(self) -> bool:
        """Solve for the steady state in place. True on success.

        Returns False -- rather than writing a number nobody can trust --
        if scipy is unavailable, the factorisation fails, or the result
        violates the maximum principle the operator is supposed to
        guarantee. The caller then falls back to explicit integration.
        """
        try:
            import scipy.sparse.linalg as spla
        except Exception:
            if not self._solver_warned:
                self._solver_warned = True
                _log.warning(
                    "Field '%s': scipy is unavailable, so the quasi-steady "
                    "state cannot be solved and the field falls back to "
                    "explicit sub-stepping, which at D*dt/dx^2 = %.1f "
                    "applies only a few percent of the transport.",
                    self.name, self.diffusion_coeff * self.SECONDS_PER_HOUR
                    * 0.1 / self.resolution ** 2)
            return False

        try:
            key = self._operator_key()
            if self._lu is None or key != self._lu_key:
                A, is_dir = self._assemble_matrix()
                self._lu = spla.splu(A)
                self._lu_is_dir = is_dir
                self._lu_key = key
                self.lu_rebuilds += 1
            b = self._assemble_rhs(self._lu_is_dir)
            c = self._lu.solve(b)
        except Exception as exc:
            if not self._solver_warned:
                self._solver_warned = True
                _log.warning(
                    "Field '%s': the steady-state solve failed (%s); falling "
                    "back to explicit sub-stepping.", self.name, exc)
            self._lu = None
            self._lu_key = None
            return False

        # The M-matrix property makes 0 <= C a theorem. Check it rather
        # than assume it: a violation beyond round-off means the operator
        # is not what this code believes it is, and a wrong field is worse
        # than a slow one.
        if not np.isfinite(c).all() or c.min() < -1e-9:
            if not self._solver_warned:
                self._solver_warned = True
                _log.warning(
                    "Field '%s': the steady-state solution violates the "
                    "maximum principle (min %.3e, finite=%s). Refusing it "
                    "and falling back to explicit sub-stepping.",
                    self.name, float(np.min(c)), bool(np.isfinite(c).all()))
            return False

        self.concentration[...] = np.maximum(
            c.reshape(self.grid_size), 0.0).astype(np.float32)
        return True

    def _compute_laplacian(self) -> np.ndarray:
        """Compute Laplacian using finite differences"""
        laplacian = np.zeros_like(self.concentration)
        
        # 3D Laplacian: ∇²C = ∂²C/∂x² + ∂²C/∂y² + ∂²C/∂z²
        # Interior points only
        laplacian[1:-1, 1:-1, 1:-1] = (
            self.concentration[2:, 1:-1, 1:-1] + self.concentration[:-2, 1:-1, 1:-1] +
            self.concentration[1:-1, 2:, 1:-1] + self.concentration[1:-1, :-2, 1:-1] +
            self.concentration[1:-1, 1:-1, 2:] + self.concentration[1:-1, 1:-1, :-2] -
            6 * self.concentration[1:-1, 1:-1, 1:-1]
        )
        
        return laplacian
    
    def get_concentration_at(self, position: np.ndarray) -> float:
        """Get concentration at position"""
        idx = self._position_to_index(position)
        if self._is_valid_index(idx):
            return self.concentration[idx]
        return 0.0
    
    def get_gradient_at(self, position: np.ndarray) -> np.ndarray:
        """Get gradient at position"""
        idx = self._position_to_index(position)
        i, j, k = idx
        
        if not self._is_valid_index((i, j, k)):
            return np.zeros(3)
        
        # Central differences
        if i > 0 and i < self.grid_size[0] - 1:
            dx = (self.concentration[i+1, j, k] - self.concentration[i-1, j, k]) / (2 * self.resolution)
        else:
            dx = 0
        
        if j > 0 and j < self.grid_size[1] - 1:
            dy = (self.concentration[i, j+1, k] - self.concentration[i, j-1, k]) / (2 * self.resolution)
        else:
            dy = 0
        
        if k > 0 and k < self.grid_size[2] - 1:
            dz = (self.concentration[i, j, k+1] - self.concentration[i, j, k-1]) / (2 * self.resolution)
        else:
            dz = 0
        
        return np.array([dx, dy, dz])
    
    def _position_to_index(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Convert position to grid index"""
        idx = tuple((position / self.resolution).astype(int))
        return idx
    
    def _is_valid_index(self, idx: Tuple[int, int, int]) -> bool:
        """Check if index is valid"""
        return (0 <= idx[0] < self.grid_size[0] and
                0 <= idx[1] < self.grid_size[1] and
                0 <= idx[2] < self.grid_size[2])


class SpatialModule(SimulationModule):
    """
    Spatial simulation module
    
    Manages:
    - 3D grid
    - Multiple diffusible fields
    - Diffusion solver
    - Gradient calculation
    
    Events Emitted:
    - None (provides service to other modules)
    
    Events Subscribed:
    - CELL_DIVIDED: Update sources/sinks
    - CELL_DIED: Remove sources/sinks
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Grid configuration
        self.grid_size = self.config.get('grid_size', (20, 20, 10))
        self.resolution = self.config.get('resolution', 10.0)  # μm per voxel
        
        # Fields
        self.fields: Dict[str, SpatialField] = {}
        
        # Parameters
        self.update_interval = self.config.get('update_interval', 0.1)  # hours
        self.time_since_update = 0.0

        # Coupled modules. Without these the module allocated three
        # fields and diffused zeros: add_source/add_sink had no callers
        # outside this file's own __main__, so after 24 h all three
        # fields were identically zero while the under-diffusion warning
        # had fired 720 times describing the transport of nothing.
        self.cellular_module = None
        self.vascular_module = None
        # field name -> callables returning [(position, rate), ...]; see
        # register_secretion.
        self._secretion_providers: Dict[str, list] = {}

        # Cell radius used to turn a per-cell demand into a volumetric
        # uptake density. 5.0 um is this project's own default --
        # cognisom/core/tissue_config.py: cell_radius_um -- not a number
        # invented here.
        self.cell_radius_um = float(self.config.get('cell_radius_um', 5.0))

        # Reference levels the first-order uptake coefficients are
        # normalised by. Overridden from the vascular module when it is
        # linked, so the two cannot drift.
        self.arterial_O2 = float(self.config.get('arterial_O2', 0.21))
        self.arterial_testosterone_nM = 0.0   # set from the vascular module
        self.blood_lactate_mM = 1.0           # set from the vascular module
        self.perfusion_ml_per_ml_min = 0.15   # set from the vascular module
        self.lactate_budget: Dict[str, float] = {}
        # Production the FIELD sources from, mM/h on the cell's volume.
        # None = the cellular module's pool rates (0.3 / 0.1), which are
        # ~300x below a glycolytic tumour's; a dense preset that wants
        # Walenta's tissue lactate sets these to glycolytic values.
        self.lactate_production_cancer_per_h = self.config.get('lactate_production_cancer_per_h')
        self.lactate_production_normal_per_h = self.config.get('lactate_production_normal_per_h')
        self.arterial_glucose = float(
            self.config.get('arterial_glucose', 5.0))
    
    def initialize(self):
        """Initialize spatial system"""
        print("  Creating spatial fields...")
        
        # Create fields
        self._create_field('oxygen', diffusion_coeff=2000.0)  # μm²/s
        self._create_field('glucose', diffusion_coeff=600.0)
        self._create_field('cytokine', diffusion_coeff=100.0,
                           decay_rate=CHEMOKINE_DECAY_PER_HOUR)
        self._create_field('testosterone',
                           diffusion_coeff=TESTOSTERONE_DIFFUSION_UM2_S)
        self._create_field('lactate', diffusion_coeff=LACTATE_DIFFUSION_UM2_S)
        self._create_field('ifn_gamma',
                           diffusion_coeff=IFN_GAMMA_DIFFUSION_UM2_S)
        
        # Subscribe to events
        self.subscribe(EventTypes.CELL_DIVIDED, self.on_cell_divided)
        self.subscribe(EventTypes.CELL_DIED, self.on_cell_died)
        
        print(f"    ✓ Grid: {self.grid_size} voxels")
        print(f"    ✓ Resolution: {self.resolution} μm/voxel")
        print(f"    ✓ {len(self.fields)} fields created")
    
    def _create_field(self, name: str, diffusion_coeff: float,
                      decay_rate: float = 0.0):
        """Create diffusible field"""
        field = SpatialField(name, self.grid_size, self.resolution,
                             diffusion_coeff, decay_rate=decay_rate)
        self.fields[name] = field

    def register_secretion(self, field_name: str, provider) -> None:
        """Let another module supply sources for ``field_name``.

        ``provider()`` returns an iterable of ``(position, rate)`` for
        the cells secreting right now. It is called inside
        _sync_sources_sinks, AFTER the field's terms are cleared, so a
        cell that has died or moved stops or moves with it. Registering
        a source once with add_source cannot do that: the terms are
        rebuilt every update, and a one-time entry would be wiped on
        the first.
        """
        self._secretion_providers.setdefault(field_name, []).append(provider)
    
    #: Per-cell O2 demand in field units per hour, matched to
    #: CellularModule._update_metabolism, which hard-codes them (0.1 /h
    #: for cancer, 0.15 /h otherwise). Glucose is read off the cellular
    #: module instead, which exposes it as a configurable attribute.
    O2_DEMAND_CANCER = 0.10
    O2_DEMAND_NORMAL = 0.15

    def set_cellular_module(self, cellular_module):
        """Link the cell population that consumes from these fields."""
        self.cellular_module = cellular_module

    def set_vascular_module(self, vascular_module):
        """Link the capillary network that supplies these fields."""
        self.vascular_module = vascular_module
        for name, attr in (('arterial_O2', 'arterial_O2'),
                           ('arterial_glucose', 'arterial_glucose'),
                           ('arterial_testosterone_nM', 'arterial_testosterone_nM'),
                           ('blood_lactate_mM', 'blood_lactate_mM'),
                           ('perfusion_ml_per_ml_min', 'perfusion_ml_per_ml_min')):
            value = getattr(vascular_module, attr, None)
            if value is not None and float(value) > 0.0:
                setattr(self, name, float(value))

    def _sync_sources_sinks(self):
        """Re-derive every field's terms from the current cell and vessel
        state. Clear-then-register, so the operation is idempotent --
        see SpatialField.clear_sources_sinks for why a set-based guard
        would be wrong for a population that moves and dies.
        """
        if (self.cellular_module is None and self.vascular_module is None
                and not self._secretion_providers):
            return

        for field in self.fields.values():
            field.clear_sources_sinks()
        for field_name, providers in self._secretion_providers.items():
            target = self.fields.get(field_name)
            if target is None:
                continue
            for provider in providers:
                for position, rate in provider():
                    if rate > 0.0:
                        target.add_source(np.asarray(position, dtype=float),
                                          float(rate))

        oxygen = self.fields.get('oxygen')
        glucose = self.fields.get('glucose')
        testosterone = self.fields.get('testosterone')
        lactate = self.fields.get('lactate')
        lactate_vessel = self._lactate_vessel_level() if lactate is not None else 0.0
        ifn_gamma = self.fields.get('ifn_gamma')

        # Capillaries supply the interstitium: a Dirichlet voxel wherever
        # a vessel passes. Sampled at half-voxel spacing along the
        # segment so no voxel it meaningfully traverses is skipped.
        if self.vascular_module is not None:
            caps = getattr(self.vascular_module, 'capillaries', {}) or {}
            step = 0.5 * self.resolution
            for cap in caps.values():
                length = float(np.linalg.norm(
                    np.asarray(cap.end, dtype=float)
                    - np.asarray(cap.start, dtype=float)))
                n_samples = max(2, int(np.ceil(length / step)) + 1)
                for t in np.linspace(0.0, 1.0, n_samples):
                    pos = cap.get_position(t)
                    if oxygen is not None:
                        oxygen.add_dirichlet(pos, float(cap.oxygen))
                    if glucose is not None:
                        glucose.add_dirichlet(pos, float(cap.glucose))
                    if testosterone is not None and float(getattr(cap, 'testosterone', 0.0)) > 0.0:
                        testosterone.add_dirichlet(pos, float(cap.testosterone))
                    if lactate is not None:
                        lactate.add_dirichlet(pos, lactate_vessel)

        # Cells consume: first-order uptake, k = q_volumetric / C_arterial.
        # See SpatialField.add_uptake for the derivation and for why the
        # linearisation is the conservative direction.
        if self.cellular_module is not None:
            v_cell = (4.0 / 3.0) * np.pi * self.cell_radius_um ** 3
            v_voxel = self.resolution ** 3
            frac = v_cell / v_voxel if v_voxel > 0 else 0.0
            glu_normal = float(getattr(
                self.cellular_module, 'glucose_consumption_normal', 0.2))
            glu_cancer = float(getattr(
                self.cellular_module, 'glucose_consumption_cancer', 0.5))
            cells = getattr(self.cellular_module, 'cells', {}) or {}
            for cell in cells.values():
                if not getattr(cell, 'alive', False):
                    continue
                is_cancer = getattr(cell, 'cell_type', '') == 'cancer'
                if oxygen is not None and self.arterial_O2 > 0:
                    q = (self.O2_DEMAND_CANCER if is_cancer
                         else self.O2_DEMAND_NORMAL) * frac
                    oxygen.add_uptake(cell.position, q / self.arterial_O2)
                if glucose is not None and self.arterial_glucose > 0:
                    q = (glu_cancer if is_cancer else glu_normal) * frac
                    glucose.add_uptake(cell.position,
                                       q / self.arterial_glucose)
                if lactate is not None:
                    q_c, q_n = self._lactate_rates()
                    lactate.add_source(cell.position, (q_c if is_cancer else q_n) * frac)
                if ifn_gamma is not None:
                    # Every cell consumes IFN-gamma: receptor binding and
                    # internalisation. This is the term that sets the
                    # signal's range (Oyler-Yaniv 2017), so a tissue that
                    # is denser has a shorter-ranged signal -- which is
                    # the mechanism, not a side effect. Scaled by the
                    # cell's share of its voxel, like every other uptake
                    # here, so one cell in a big voxel consumes less of it.
                    ifn_gamma.add_uptake(cell.position,
                                         IFN_GAMMA_UPTAKE_PER_H * frac)
                if testosterone is not None and self.arterial_testosterone_nM > 0:
                    # The cell's draw on free T is what it turns into DHT:
                    # the circulating share of the reference tissue DHT,
                    # turned over once per relaxation time. Same secant
                    # linearisation as oxygen (see add_uptake).
                    # First-order for real, not a secant: 5a-reductase
                    # sits far below saturation at 0.34 nM, so the rate
                    # constant is the EUGONADAL draw over the EUGONADAL
                    # level and does not change when the wall level does.
                    # Dividing by the current wall level made a castrate
                    # tissue consume 17x harder per nM, which is the
                    # opposite of what a sub-saturated enzyme does.
                    from cognisom.modules.vascular_module import EUGONADAL_FREE_TESTOSTERONE_NM
                    q = self.T_UPTAKE_NM_PER_H * frac
                    testosterone.add_uptake(cell.position,
                                            q / EUGONADAL_FREE_TESTOSTERONE_NM)

    def update(self, dt: float):
        """Update spatial fields"""
        self.time_since_update += dt

        # Update at intervals (diffusion is expensive)
        if self.time_since_update >= self.update_interval:
            # Re-register BEFORE solving, so the field sees this step's
            # cells and not the previous step's.
            self._sync_sources_sinks()

            for field in self.fields.values():
                field.update(self.time_since_update)

            self.time_since_update = 0.0
            self._push_fields_to_cells()
    
    #: field -> cell attribute, written for every live cell after every
    #: solve. This is how a signal reaches a consumer: the cell reads its
    #: own attribute, and never has to know a field exists.
    FIELDS_TO_CELLS = (('testosterone', 'testosterone_nM'),
                       ('lactate', 'lactate_mM'),
                       ('ifn_gamma', 'ifn_gamma_level'))

    #: Per-cell volumetric draw on free testosterone, nM/h: the share of the
    #: reference tissue DHT that comes from the blood, turned over once per
    #: relaxation time (CellularModule constants).
    from cognisom.modules.cellular_module import (  # noqa: E402
        DHT_INTRACRINE_FRACTION as _F, DHT_RELAXATION_H as _TAU,
        TISSUE_DHT_REFERENCE_NM as _DHT)
    T_UPTAKE_NM_PER_H = (1.0 - _F) * _DHT / _TAU
    del _F, _TAU, _DHT

    def _lactate_rates(self):
        """(cancer, normal) lactate production, mM/h on the cell volume."""
        from cognisom.modules.cellular_module import (
            LACTATE_PRODUCTION_CANCER_PER_H, LACTATE_PRODUCTION_NORMAL_PER_H)
        q_c = self.lactate_production_cancer_per_h
        q_n = self.lactate_production_normal_per_h
        return (float(q_c if q_c is not None else LACTATE_PRODUCTION_CANCER_PER_H),
                float(q_n if q_n is not None else LACTATE_PRODUCTION_NORMAL_PER_H))

    def _lactate_vessel_level(self) -> float:
        """The lactate the blood carries past the cells, mM: a budget.

        A Dirichlet wall at ARTERIAL lactate would be wrong by two orders
        of magnitude, and the reason is worth having in one place. With
        vessels 60-100 um apart, diffusion equilibrates a 200 um block in
        ~15 s (L^2/D), so a diffusion-limited gradient of 9 mM needs
        q R^2 / 4D = 9 -- a radius of about 1.6 mm at a glycolytic
        tumour's production. Tumour lactate of 6-10 umol/g (Walenta 2000,
        PMID 10706105) is not a diffusion gradient; it is what the BLOOD
        has picked up, and that is set by perfusion, exactly as the
        oxygen budget found consumption was. Fick's principle:

            C_tissue = C_arterial + P / F

        with P the volumetric production (mM per minute of tissue) and F
        the perfusion (mL per mL per minute). The wall is held at that
        level, uniform along the vessels (the blood enriches along a
        capillary and this uses the mean), and the cell sources on top
        add the small diffusive structure that is really there.

        On the shipped presets, at ~1% of a cell per voxel, P/F is a few
        hundredths of a millimolar: the field is flat at blood level and
        the immune gate is inert, which is the correct answer for a
        sparse, perfused block. A 70%-packed tumour at prostate perfusion
        reaches Walenta's range only if the per-cell rate is a glycolytic
        one (~100 mM/h on the cell's volume), not the 0.3/h the pool
        model uses -- recorded in lactate_budget, not papered over.
        """
        cells = getattr(self.cellular_module, 'cells', None) or {}
        q_cancer, q_normal = self._lactate_rates()
        v_cell = (4.0 / 3.0) * np.pi * self.cell_radius_um ** 3
        nx, ny, nz = self.grid_size
        v_domain = float(nx * ny * nz) * self.resolution ** 3
        production_mm_per_h = 0.0
        n_alive = 0
        for cell in cells.values():
            if not getattr(cell, 'alive', False):
                continue
            n_alive += 1
            q = q_cancer if getattr(cell, 'cell_type', '') == 'cancer' else q_normal
            production_mm_per_h += q * v_cell
        production_mm_per_h /= v_domain if v_domain > 0 else 1.0
        p_per_min = production_mm_per_h / 60.0
        f = max(float(self.perfusion_ml_per_ml_min), 1e-9)
        level = float(self.blood_lactate_mM) + p_per_min / f
        self.lactate_budget = {
            'arterial_mM': float(self.blood_lactate_mM),
            'production_mM_per_min': p_per_min,
            'rate_cancer_mM_per_h': q_cancer,
            'perfusion_ml_per_ml_min': f,
            'vessel_mM': level,
            'packing_fraction': (n_alive * v_cell / v_domain) if v_domain > 0 else 0.0,
        }
        return level

    def _push_fields_to_cells(self):
        if self.cellular_module is None:
            return
        cells = getattr(self.cellular_module, 'cells', {}) or {}
        for field_name, attr in self.FIELDS_TO_CELLS:
            field = self.fields.get(field_name)
            if field is None:
                continue
            for cell in cells.values():
                if getattr(cell, 'alive', False):
                    setattr(cell, attr, float(field.get_concentration_at(cell.position)))

    def get_concentration(self, field_name: str, position: np.ndarray) -> float:
        """Get concentration at position"""
        if field_name in self.fields:
            return self.fields[field_name].get_concentration_at(position)
        return 0.0
    
    def get_gradient(self, field_name: str, position: np.ndarray) -> np.ndarray:
        """Get gradient at position"""
        if field_name in self.fields:
            return self.fields[field_name].get_gradient_at(position)
        return np.zeros(3)
    
    def add_source(self, field_name: str, position: np.ndarray, rate: float):
        """Add source to field"""
        if field_name in self.fields:
            self.fields[field_name].add_source(position, rate)
    
    def add_sink(self, field_name: str, position: np.ndarray, rate: float):
        """Add sink to field"""
        if field_name in self.fields:
            self.fields[field_name].add_sink(position, rate)
    
    def get_state(self) -> Dict[str, Any]:
        """Return current spatial state"""
        field_stats = {}
        for name, field in self.fields.items():
            field_stats[name] = {
                'mean': float(np.mean(field.concentration)),
                'max': float(np.max(field.concentration)),
                'min': float(np.min(field.concentration)),
                'n_sources': len(field.sources),
                'n_sinks': len(field.sinks),
                # What the field is actually coupled to, and how it was
                # solved. All three were silently absent while the fields
                # were identically zero, which is why nothing reported
                # that the module had no inputs at all.
                'n_uptake': len(field.uptake),
                'n_dirichlet': len(field.dirichlet),
                'solver': field.last_solver,
                'lu_rebuilds': field.lu_rebuilds,
                # The dimensionless number that says whether a gradient
                # can exist on this grid at all: phi << 1 means transport
                # beats consumption and the field is uniform to O(phi^2)
                # however fine the mesh.
                'thiele_modulus': field.thiele_modulus(),
                'fourier_number': field.fourier_number(self.update_interval),
            }

        state = {
            'lactate_budget': dict(self.lactate_budget),
            'grid_size': self.grid_size,
            'resolution': self.resolution,
            'n_fields': len(self.fields),
            'coupled_to_cells': self.cellular_module is not None,
            'coupled_to_vessels': self.vascular_module is not None,
            'fields': field_stats
        }

        # Interstitial levels sampled AT THE CELLS, which is the quantity
        # a consumer would actually want, and the one that shows whether
        # the field is uniform.
        if self.cellular_module is not None:
            cells = getattr(self.cellular_module, 'cells', {}) or {}
            alive = [c for c in cells.values()
                     if getattr(c, 'alive', False)]
            for field_name, key in (('oxygen', 'interstitial_O2'),
                                    ('glucose', 'interstitial_glucose'),
                                    ('testosterone', 'interstitial_testosterone_nM'),
                                    ('lactate', 'interstitial_lactate_mM'),
                                    ('ifn_gamma', 'interstitial_ifn_gamma')):
                if field_name not in self.fields or not alive:
                    continue
                values = [self.get_concentration(field_name, c.position)
                          for c in alive]
                state[key + '_mean'] = float(np.mean(values))
                state[key + '_min'] = float(np.min(values))

        return state
    
    # Event handlers
    def on_cell_divided(self, data):
        """Handle cell division"""
        pass
    
    def on_cell_died(self, data):
        """Handle cell death"""
        pass


# Test
if __name__ == '__main__':
    from cognisom.core import SimulationEngine, SimulationConfig
    
    print("=" * 70)
    print("Spatial Module Test")
    print("=" * 70)
    print()
    
    # Create engine
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=0.5))
    
    # Register spatial module
    engine.register_module('spatial', SpatialModule, {
        'grid_size': (20, 20, 10),
        'resolution': 10.0,
        'update_interval': 0.1
    })
    
    # Initialize
    engine.initialize()
    
    # Add some sources
    spatial = engine.modules['spatial']
    
    # Add O2 source (blood vessel)
    spatial.add_source('oxygen', np.array([100, 100, 50]), rate=0.5)
    
    # Add glucose source
    spatial.add_source('glucose', np.array([100, 100, 50]), rate=0.3)
    
    # Add cytokine source (immune cell)
    spatial.add_source('cytokine', np.array([80, 80, 50]), rate=0.1)
    
    print("Added sources")
    print()
    
    # Run simulation
    engine.run()
    
    # Check concentrations
    print("\nConcentrations at different positions:")
    positions = [
        np.array([100, 100, 50]),  # At source
        np.array([120, 120, 50]),  # Near source
        np.array([150, 150, 50])   # Far from source
    ]
    
    for pos in positions:
        O2 = spatial.get_concentration('oxygen', pos)
        glucose = spatial.get_concentration('glucose', pos)
        print(f"  Position {pos}: O2={O2:.3f}, glucose={glucose:.3f}")
    
    print()
    
    # Get results
    print("Spatial State:")
    spatial_state = spatial.get_state()
    for key, value in spatial_state.items():
        if key != 'fields':
            print(f"  {key}: {value}")
    
    print("\nField Statistics:")
    for field_name, stats in spatial_state['fields'].items():
        print(f"  {field_name}:")
        for stat_name, stat_value in stats.items():
            print(f"    {stat_name}: {stat_value}")
    
    print()
    print("=" * 70)
    print("✓ Spatial module working!")
    print("=" * 70)
