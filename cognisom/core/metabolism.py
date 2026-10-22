"""What a cell's local oxygen and glucose let it do, from the FBA model.

The tissue engine carries ``atp`` and ``lactate`` per cell and, until now,
wrote neither: ATP was set to 1.0 at initialisation, read by the division
gate, and never updated, and lactate had no writer at all -- so there was no
metabolism and no Warburg effect to study on a platform built to study one.

The stoichiometry to do it with was already in the repo, in
``cognisom.gpu.fba_solver``, unused by anything but a re-export. This module
is the coupling. It does not reimplement the biochemistry; it asks that model
what is achievable at a given pair of substrate levels.

**Why a table.** The flux balance problem is a linear program, and the CPU
path solves one LP per cell in a Python loop. At the configured million cells
that is a million LPs per step, which is the same shape of mistake as the
per-cell force loop this repo already removed. But the LP depends on the cell
only through two numbers, its local oxygen and glucose, so it is solved once
per point on a small grid and interpolated per cell. The interpolation is
vectorised and the table is built once. ``test_oxygen_feeds_metabolism``
checks the interpolated answer against a direct LP at points off the grid.

**Why maximise ATP rather than biomass.** The shipped objective maximises
biomass, whose upper bound is 2.0, and the model can reach that bound across
almost the whole substrate range -- so ATP throughput came out between 20 and
21 everywhere and carried no information. Asking instead for the largest ATP
production the location can support is a well-posed question with real
dynamic range, and it is what an energy charge in [0, 1] should be built on.
The objective is read off the stoichiometric matrix rather than written down
again here: it is the positive part of the ATP row.

**What is a modelling choice rather than a measurement.** Two things, and
neither is sourced. The uptake capacities are whatever ``fba_solver``
shipped. And the map from a concentration to an uptake bound is saturating
for oxygen, with the same half-saturation constant the diffusion sink uses --
so the two agree about when oxygen stops being available -- and linear for
glucose, because GLUT1's Km is of the same order as blood glucose itself and
picking a value would be a guess dressed as a correction. Both are flagged in
``CAVEATS`` and reported by :func:`describe`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

#: Half-saturation for oxygen uptake, in the oxygen field's own units. The
#: same 1 mmHg the diffusion sink saturates at, so a cell and the field it
#: draws from agree about when oxygen has run out.
O2_KM = 1.0 / 760.0

#: The substrate levels the capacities correspond to: arterial oxygen as the
#: field expresses it, and blood glucose in mM.
O2_REFERENCE = 0.21
GLUCOSE_REFERENCE = 5.0

#: Grid resolution per axis. The response is smooth in both substrates, so a
#: coarse grid interpolates well; 32 keeps the build under a second.
GRID = 32

CAVEATS: Tuple[str, ...] = (
    "uptake capacities are fba_solver's shipped bounds and are not sourced",
    "oxygen uptake saturates at km = 1 mmHg, shared with the diffusion sink",
    "glucose uptake is taken as linear in concentration, which is a choice: "
    "GLUT1's Km is of the same order as blood glucose, so neither linear nor "
    "saturating is clearly right and no value for it is sourced here",
    "the ATP charge is normalised to the reference substrate levels, so it is "
    "relative to well-perfused tissue and not an absolute nucleotide pool",
)


@dataclass
class MetabolicTable:
    """FBA answers on a grid of (oxygen, glucose), ready to interpolate."""

    o2_levels: np.ndarray
    glucose_levels: np.ndarray
    #: Largest ATP production the location supports, normalised so that the
    #: reference substrate levels give 1.0.
    atp_charge: np.ndarray
    #: Lactate export flux at that optimum, normalised the same way. Rises as
    #: oxygen falls, which is the Warburg shift and is not imposed anywhere --
    #: it falls out of the NADH balance, because without oxygen the only way
    #: to reoxidise NADH is to make lactate.
    lactate: np.ndarray
    #: False where the cell cannot satisfy its own maintenance requirement.
    feasible: np.ndarray
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"{self.atp_charge.shape[0]}x{self.atp_charge.shape[1]} FBA table, "
            f"{int(self.feasible.sum())}/{self.feasible.size} feasible; "
            f"ATP charge {self.atp_charge[self.feasible].min():.3f}"
            f"-{self.atp_charge[self.feasible].max():.3f}, "
            f"lactate {self.lactate[self.feasible].min():.3f}"
            f"-{self.lactate[self.feasible].max():.3f}"
        )


def _atp_objective(model) -> np.ndarray:
    """Maximise ATP production: the positive part of the ATP row of S.

    Read from the matrix so that a change to the stoichiometry cannot leave a
    stale objective behind.
    """
    atp_row = model.metabolite_names.index("ATP")
    return np.maximum(model.S[atp_row], 0.0)


def _bounds_for(model, o2: float, glucose: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell uptake ceilings from local substrate levels."""
    lb, ub = model.lb.copy(), model.ub.copy()
    o2_rxn = model.reaction_names.index("oxygen_consumption")
    glc_rxn = model.reaction_names.index("glucose_uptake")
    o2_cap, glc_cap = model.ub[o2_rxn], model.ub[glc_rxn]

    saturating = o2 / (O2_KM + o2) if o2 > 0 else 0.0
    reference = O2_REFERENCE / (O2_KM + O2_REFERENCE)
    ub[o2_rxn] = o2_cap * saturating / reference
    ub[glc_rxn] = glc_cap * max(glucose, 0.0) / GLUCOSE_REFERENCE
    return lb, ub


def build_table(grid: int = GRID, model=None) -> MetabolicTable:
    """Solve the FBA model across a grid of substrate levels.

    The oxygen axis is spaced on a square law rather than uniformly: the
    response turns over near km = 1 mmHg, which is 160x below arterial, and a
    uniform grid would put almost no points where the answer changes.
    """
    from cognisom.gpu.fba_solver import FBASolver, prostate_cancer_model

    model = model or prostate_cancer_model()
    objective = _atp_objective(model)
    solver = FBASolver(
        type(model)(
            S=model.S, lb=model.lb, ub=model.ub, objective=objective,
            metabolite_names=list(model.metabolite_names),
            reaction_names=list(model.reaction_names),
        ),
        use_gpu=False,
    )
    lactate_rxn = model.reaction_names.index("lactate_export")

    o2_levels = O2_REFERENCE * np.linspace(0.0, 1.0, grid) ** 2
    glucose_levels = GLUCOSE_REFERENCE * np.linspace(0.0, 1.0, grid)

    atp = np.zeros((grid, grid))
    lactate = np.zeros((grid, grid))
    feasible = np.zeros((grid, grid), dtype=bool)
    for i, o2 in enumerate(o2_levels):
        for j, glucose in enumerate(glucose_levels):
            lb, ub = _bounds_for(model, float(o2), float(glucose))
            fluxes = solver.solve_single(lb, ub)
            if fluxes is None:
                continue
            feasible[i, j] = True
            atp[i, j] = float(np.dot(objective,
                                     [fluxes[n] for n in model.reaction_names]))
            lactate[i, j] = fluxes[model.reaction_names[lactate_rxn]]

    reference = atp[-1, -1]
    if reference <= 0:
        raise RuntimeError(
            "the FBA model produces no ATP at reference substrate levels, so "
            "there is nothing to normalise against; check fba_solver's bounds"
        )
    scale = lactate[feasible].max() or 1.0
    return MetabolicTable(
        o2_levels=o2_levels,
        glucose_levels=glucose_levels,
        atp_charge=atp / reference,
        lactate=lactate / scale,
        feasible=feasible,
        notes=list(CAVEATS),
    )


_CACHE: Dict[int, MetabolicTable] = {}


def table(grid: int = GRID) -> MetabolicTable:
    """The table, built once per grid size."""
    hit = _CACHE.get(grid)
    if hit is None:
        hit = _CACHE[grid] = build_table(grid)
        log.info("metabolism: %s", hit.describe())
    return hit


def _interp_axis(levels: np.ndarray, values: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Index below and weight, for one axis of a bilinear interpolation."""
    idx = np.clip(np.searchsorted(levels, values) - 1, 0, len(levels) - 2)
    lo, hi = levels[idx], levels[idx + 1]
    span = np.where(hi > lo, hi - lo, 1.0)
    return idx, np.clip((values - lo) / span, 0.0, 1.0)


def evaluate(o2: np.ndarray, glucose: np.ndarray,
             tbl: Optional[MetabolicTable] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell ``(atp_charge, lactate, feasible)``, bilinearly interpolated.

    Vectorised over cells, which is the point: the underlying linear program
    is solved once per grid point and never per cell.

    Feasibility is interpolated with AND rather than averaged. A cell between
    a feasible and an infeasible grid point is reported infeasible, because
    the alternative -- a fractional feasibility -- is not a thing a caller can
    act on, and rounding it towards survival is the direction that hides
    deaths.
    """
    tbl = tbl or table()
    o2 = np.asarray(o2, dtype=np.float64)
    glucose = np.asarray(glucose, dtype=np.float64)

    i, u = _interp_axis(tbl.o2_levels, np.clip(o2, 0.0, tbl.o2_levels[-1]))
    j, v = _interp_axis(tbl.glucose_levels,
                        np.clip(glucose, 0.0, tbl.glucose_levels[-1]))

    def blend(grid_values):
        return (grid_values[i, j] * (1 - u) * (1 - v)
                + grid_values[i + 1, j] * u * (1 - v)
                + grid_values[i, j + 1] * (1 - u) * v
                + grid_values[i + 1, j + 1] * u * v)

    feasible = (tbl.feasible[i, j] & tbl.feasible[i + 1, j]
                & tbl.feasible[i, j + 1] & tbl.feasible[i + 1, j + 1])
    atp = np.where(feasible, blend(tbl.atp_charge), 0.0)
    lactate = np.where(feasible, blend(tbl.lactate), 0.0)
    return atp, lactate, feasible


def describe() -> str:
    """One line on what the coupling is and what it is not."""
    tbl = table()
    return (f"{tbl.describe()}. Caveats: "
            + "; ".join(tbl.notes))
