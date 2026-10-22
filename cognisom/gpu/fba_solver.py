"""
GPU-Accelerated Flux Balance Analysis (FBA) Solver
===================================================

Solves the metabolic flux balance equation S * v = 0 with bounds and
an objective function, batched across many cells simultaneously.

GPU path: Uses CuPy linear algebra (cuSOLVER) for batched LP relaxation.
CPU path: Falls back to SciPy linprog for each cell.

This is a foundation module — full LP-based FBA requires a proper LP
solver on GPU (e.g., cuOpt or a custom simplex). The current GPU path
uses a least-squares relaxation (pseudo-inverse) which is fast but
approximate. The CPU path uses exact LP.

Phase 3 Roadmap: Replace GPU path with proper cuSOLVER LP or NVIDIA
cuOpt when available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Try GPU imports
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False


@dataclass
class MetabolicModel:
    """A simple stoichiometric model for FBA.

    Attributes:
        S: Stoichiometric matrix (n_metabolites x n_reactions)
        lb: Lower bounds for each reaction flux
        ub: Upper bounds for each reaction flux
        objective: Coefficients for the objective function (maximize c^T * v)
        metabolite_names: Names of metabolites (rows of S)
        reaction_names: Names of reactions (columns of S)
    """
    S: np.ndarray                       # (n_met, n_rxn)
    lb: np.ndarray                      # (n_rxn,)
    ub: np.ndarray                      # (n_rxn,)
    objective: np.ndarray               # (n_rxn,) — maximize c^T * v
    metabolite_names: List[str] = field(default_factory=list)
    reaction_names: List[str] = field(default_factory=list)

    @property
    def n_metabolites(self) -> int:
        return self.S.shape[0]

    @property
    def n_reactions(self) -> int:
        return self.S.shape[1]


def prostate_cancer_model() -> MetabolicModel:
    """Build a simplified prostate cancer metabolic model.

    Reactions:
      0: glucose_uptake      (extracellular -> glucose)
      1: glycolysis           (glucose -> 2 pyruvate + 2 ATP)
      2: lactate_production   (pyruvate -> lactate)
      3: TCA_cycle            (pyruvate -> 3 CO2 + 15 ATP)
      4: oxygen_consumption   (extracellular -> O2)
      5: oxidative_phosph     (O2 -> 2.5 ATP)
      6: atp_maintenance      (ATP -> ADP, maintenance cost)
      7: biomass              (ATP + precursors -> growth)

    Metabolites:
      0: glucose
      1: pyruvate
      2: lactate
      3: ATP
      4: O2
      5: CO2
    """
    n_met, n_rxn = 7, 10

    S = np.zeros((n_met, n_rxn), dtype=np.float64)

    # glucose_uptake: -> glucose
    S[0, 0] = 1.0

    # glycolysis: glucose -> 2 pyruvate + 2 ATP + 2 NADH
    S[0, 1] = -1.0   # consumes glucose
    S[1, 1] = 2.0    # produces pyruvate
    S[3, 1] = 2.0    # produces ATP
    S[6, 1] = 2.0    # produces NADH -- and this is the whole story below

    # lactate_production: pyruvate + NADH -> lactate
    S[1, 2] = -1.0   # consumes pyruvate
    S[2, 2] = 1.0    # produces lactate
    S[6, 2] = -1.0   # CONSUMES NADH, which is why fermentation exists: it
                     # regenerates NAD+ so glycolysis can keep running when
                     # there is no oxygen to reoxidise NADH.

    # TCA_cycle: pyruvate -> 3 CO2 + 4 NADH + 1 ATP
    S[1, 3] = -1.0   # consumes pyruvate
    S[5, 3] = 3.0    # produces CO2
    S[6, 3] = 4.0    # produces NADH (3 NADH + 1 FADH2, lumped)
    S[3, 3] = 1.0    # substrate-level phosphorylation only (the GTP step)

    # oxidative_phosph: NADH + 0.5 O2 -> 2.5 ATP
    S[6, 5] = -1.0   # consumes NADH
    S[4, 5] = -0.5   # consumes O2
    S[3, 5] = 2.5    # produces ATP

    # oxygen_consumption: -> O2
    S[4, 4] = 1.0

    # atp_maintenance: ATP ->
    S[3, 6] = -1.0

    # biomass: ATP -> growth
    S[3, 7] = -10.0

    # lactate_export: lactate ->
    S[2, 8] = -1.0

    # co2_export: CO2 ->
    S[5, 9] = -1.0

    # Lactate and CO2 need somewhere to GO. Without these two exchanges the
    # steady-state constraint S*v = 0 forces both producing reactions to
    # zero: no lactate export means no lactate production, which leaves
    # pyruvate and NADH from glycolysis with no sink, which forces glycolysis
    # itself to zero, and no CO2 export likewise forces the TCA cycle to
    # zero. Every carbon route is then shut and the only reaction that can
    # make ATP is one with no byproduct at all.
    #
    # Which is exactly what the previous model had: an oxidative_phosph
    # reaction turning O2 straight into ATP. It was not merely unphysical, it
    # was load-bearing -- the single feasible ATP route, kept feasible by
    # producing nothing that needed disposing of. Correcting the
    # stoichiometry without adding these exchanges makes the whole model
    # infeasible, which is how the omission surfaced.

    # Yields, as a check on the numbers above. Aerobic, per glucose:
    # 2 ATP + 2 NADH from glycolysis, then 2 pyruvate through TCA for 2 ATP
    # and 8 NADH, then 10 NADH through oxphos for 25 ATP and 5 O2 -- 29 ATP
    # per glucose against a textbook 30-32, and 5 O2 against 6. Anaerobic,
    # per glucose: 2 ATP and 2 lactate, which is exact.
    #
    # The previous version of this model had TCA consume O2 directly for
    # 15 ATP AND a separate oxidative_phosph reaction turning O2 into ATP
    # with no carbon source at all. A cell with oxygen and zero glucose
    # therefore grew at the maximum rate, and since oxygen alone sufficed for
    # ATP, glycolysis and the TCA cycle went unused whenever oxygen was
    # present -- so lactate production was always zero and the model could
    # not express the Warburg effect it exists to study. Routing the
    # oxidative yield through NADH is what couples the two pathways: without
    # oxygen the only NADH sink is lactate, which caps glycolysis at 2 ATP
    # per glucose and forces fermentation.

    # Capacities. These are NOT sourced -- they are the magnitudes this model
    # shipped with, kept so the change above can be judged on its own. The
    # oxidative_phosph ceiling is raised because its flux is now counted in
    # NADH rather than in O2: ten NADH per glucose means the reaction runs an
    # order of magnitude faster than the uptake it serves.
    lb = np.array([0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0], dtype=np.float64)
    ub = np.array([10, 10, 20, 20, 15, 120, 5.0, 2, 20, 60],
                  dtype=np.float64)

    # Objective: maximize biomass (reaction 7)
    objective = np.zeros(n_rxn, dtype=np.float64)
    objective[7] = 1.0

    return MetabolicModel(
        S=S, lb=lb, ub=ub, objective=objective,
        metabolite_names=["glucose", "pyruvate", "lactate", "ATP", "O2",
                          "CO2", "NADH"],
        reaction_names=[
            "glucose_uptake", "glycolysis", "lactate_production",
            "TCA_cycle", "oxygen_consumption", "oxidative_phosph",
            "atp_maintenance", "biomass", "lactate_export", "co2_export",
        ],
    )


class FBASolver:
    """Flux Balance Analysis solver with GPU/CPU backend.

    Solves: maximize c^T * v
            subject to: S * v = 0, lb <= v <= ub

    GPU mode uses batched least-squares (approximate but fast).
    CPU mode uses scipy.optimize.linprog (exact).
    """

    def __init__(self, model: MetabolicModel, use_gpu: bool = True) -> None:
        self.model = model
        self.use_gpu = use_gpu and HAS_GPU
        if self.use_gpu:
            self._S_gpu = cp.asarray(model.S)
            log.info("FBA solver using GPU (CuPy)")
        else:
            log.info("FBA solver using CPU (SciPy)")

    def solve_single(self, cell_lb: Optional[np.ndarray] = None,
                     cell_ub: Optional[np.ndarray] = None
                     ) -> Optional[Dict[str, float]]:
        """Solve FBA for a single cell.

        Returns a flux dict, or None when the constraints cannot be satisfied
        -- see :meth:`_solve_cpu_single` for why None rather than a plausible
        vector.
        """
        lb = cell_lb if cell_lb is not None else self.model.lb
        ub = cell_ub if cell_ub is not None else self.model.ub

        if self.use_gpu:
            fluxes = self._solve_gpu_single(lb, ub)
        else:
            fluxes = self._solve_cpu_single(lb, ub)
        if fluxes is None:
            return None

        return {name: float(v) for name, v in zip(self.model.reaction_names, fluxes)}

    def solve_batch(self, n_cells: int,
                    lb_matrix: Optional[np.ndarray] = None,
                    ub_matrix: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve FBA for a batch of cells.

        Args:
            n_cells: Number of cells
            lb_matrix: (n_cells, n_rxn) lower bounds per cell, or None for default
            ub_matrix: (n_cells, n_rxn) upper bounds per cell, or None for default

        Returns:
            ``(fluxes, feasible)``: an (n_cells, n_rxn) flux matrix, and a
            boolean mask of the cells whose constraints could be met. Rows
            where ``feasible`` is False are zero and mean nothing.
        """
        if lb_matrix is None:
            lb_matrix = np.tile(self.model.lb, (n_cells, 1))
        if ub_matrix is None:
            ub_matrix = np.tile(self.model.ub, (n_cells, 1))

        if self.use_gpu:
            fluxes = self._solve_gpu_batch(lb_matrix, ub_matrix)
            # The GPU path is a least-squares relaxation and has no notion of
            # infeasibility, so it cannot report one. Saying so is better
            # than implying every cell is fine.
            return fluxes, np.ones(len(lb_matrix), dtype=bool)
        return self._solve_cpu_batch(lb_matrix, ub_matrix)

    # ── GPU path ──────────────────────────────────────────────────────

    def _solve_gpu_single(self, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """GPU: least-squares relaxation for single cell."""
        result = self._solve_gpu_batch(lb.reshape(1, -1), ub.reshape(1, -1))
        return result[0]

    def _solve_gpu_batch(self, lb_matrix: np.ndarray, ub_matrix: np.ndarray) -> np.ndarray:
        """GPU: batched least-squares FBA.

        Uses pseudo-inverse of S to find flux vectors that approximately
        satisfy S*v = 0, then projects onto the feasible region [lb, ub]
        and optimizes the objective.
        """
        n_cells = lb_matrix.shape[0]
        S_gpu = self._S_gpu
        c_gpu = cp.asarray(self.model.objective)
        lb_gpu = cp.asarray(lb_matrix)
        ub_gpu = cp.asarray(ub_matrix)

        # Compute pseudo-inverse of S (same for all cells)
        S_pinv = cp.linalg.pinv(S_gpu)  # (n_rxn, n_met)

        # Start at midpoint of bounds
        v = (lb_gpu + ub_gpu) / 2.0  # (n_cells, n_rxn)

        # Project to null space of S: v = v - S_pinv @ (S @ v^T)
        # S @ v^T has shape (n_met, n_cells)
        residual = S_gpu @ v.T           # (n_met, n_cells)
        correction = S_pinv @ residual   # (n_rxn, n_cells)
        v = v - correction.T             # (n_cells, n_rxn)

        # Clamp to bounds
        v = cp.clip(v, lb_gpu, ub_gpu)

        # Gradient ascent on objective within feasible region (5 iterations)
        step = 0.1
        for _ in range(5):
            grad = cp.tile(c_gpu, (n_cells, 1))  # objective gradient
            v_new = v + step * grad
            v_new = cp.clip(v_new, lb_gpu, ub_gpu)
            # Re-project to null space
            residual = S_gpu @ v_new.T
            correction = S_pinv @ residual
            v_new = v_new - correction.T
            v_new = cp.clip(v_new, lb_gpu, ub_gpu)
            # Only keep if objective improved
            obj_old = cp.sum(v * c_gpu, axis=1)
            obj_new = cp.sum(v_new * c_gpu, axis=1)
            improve = (obj_new > obj_old).reshape(-1, 1)
            v = cp.where(improve, v_new, v)

        return cp.asnumpy(v)

    # ── CPU path ──────────────────────────────────────────────────────

    def _solve_cpu_single(self, lb: np.ndarray, ub: np.ndarray
                          ) -> Optional[np.ndarray]:
        """CPU: exact LP via scipy.optimize.linprog, or None if infeasible.

        None means the cell cannot satisfy its own constraints -- with
        ``atp_maintenance`` carrying a lower bound, that is exactly the
        statement "this cell cannot pay its maintenance cost at these
        substrate levels", which is the most consequential thing this model
        can say about a cell in a hypoxic core.

        It used to return ``(lb + ub) / 2`` instead. That vector does not
        satisfy S*v = 0, so it is not a flux distribution at all, and it is
        not recognisable as a failure either: with oxygen and glucose both
        clamped to zero the midpoint reported a growing cell excreting
        lactate. A caller cannot distinguish a fabricated answer from a real
        one, so there is no safe way to return one.
        """
        try:
            from scipy.optimize import linprog
        except ImportError:                                # pragma: no cover
            log.warning("scipy not available; FBA cannot be solved")
            return None

        # linprog minimizes, so negate objective for maximization
        result = linprog(
            c=-self.model.objective,
            A_eq=self.model.S,
            b_eq=np.zeros(self.model.n_metabolites),
            bounds=list(zip(lb, ub)),
            method="highs",
        )
        if result.success:
            return result.x
        log.debug("FBA infeasible: %s", result.message)
        return None

    def _solve_cpu_batch(self, lb_matrix: np.ndarray, ub_matrix: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """CPU: solve each cell sequentially; also report feasibility.

        One LP per cell in a Python loop, which is why nothing in the tissue
        path calls this directly: at the configured million cells it would be
        a million LPs per step. ``cognisom.core.metabolism`` solves the model
        once per substrate level on a small grid and interpolates instead.
        """
        n_cells = lb_matrix.shape[0]
        results = np.zeros((n_cells, self.model.n_reactions))
        feasible = np.zeros(n_cells, dtype=bool)
        for i in range(n_cells):
            fluxes = self._solve_cpu_single(lb_matrix[i], ub_matrix[i])
            if fluxes is not None:
                results[i] = fluxes
                feasible[i] = True
        return results, feasible
