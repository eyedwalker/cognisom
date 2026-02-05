"""
GPU-Accelerated Cell Operations
================================

Vectorized cell state updates using Structure-of-Arrays (SoA) layout.
Instead of iterating over a dict of CellState objects, we pack cell
properties into contiguous arrays and update them in bulk.

On GPU: CuPy element-wise kernels.
On CPU: NumPy vectorized operations (still ~5-10x faster than the
        original per-cell Python loop).
"""

import logging
from typing import Dict, List, Any, Tuple, Optional

import numpy as np

from .backend import get_backend

logger = logging.getLogger(__name__)


# ── Structure-of-Arrays cell packing ────────────────────────────

class CellArrays:
    """Packed cell state in Structure-of-Arrays format.

    Instead of:  cells = {id: CellState(oxygen=0.2, glucose=5.0, ...)}
    We store:    oxygen = np.array([0.2, 0.18, ...])  # one element per cell

    This enables vectorized updates across all cells simultaneously.
    """

    def __init__(self, n: int = 0, backend=None):
        self._backend = backend or get_backend()
        xp = self._backend.xp

        self.n = n
        self.cell_ids = np.zeros(n, dtype=np.int32)
        self.positions = xp.zeros((n, 3), dtype=np.float32)
        self.cell_types = np.zeros(n, dtype=np.int8)  # 0=normal, 1=cancer, 2=immune
        self.phases = np.zeros(n, dtype=np.int8)  # 0=G1, 1=S, 2=G2, 3=M, 4=G0
        self.alive = xp.ones(n, dtype=np.bool_)
        self.ages = xp.zeros(n, dtype=np.float32)

        # Metabolic state
        self.oxygen = xp.full(n, 0.21, dtype=np.float32)
        self.glucose = xp.full(n, 5.0, dtype=np.float32)
        self.atp = xp.full(n, 1000.0, dtype=np.float32)
        self.lactate = xp.zeros(n, dtype=np.float32)

        # Cancer properties
        self.mhc1 = xp.ones(n, dtype=np.float32)

    # Cell type constants
    TYPE_NORMAL = 0
    TYPE_CANCER = 1
    TYPE_IMMUNE = 2

    # Phase constants
    PHASE_G1 = 0
    PHASE_S = 1
    PHASE_G2 = 2
    PHASE_M = 3
    PHASE_G0 = 4

    @classmethod
    def from_cell_dict(cls, cells: Dict[int, Any]) -> "CellArrays":
        """Pack a dict of CellState objects into arrays.

        Args:
            cells: Dict of cell_id -> CellState (from CellularModule.cells).

        Returns:
            CellArrays instance with all cells packed.
        """
        alive_cells = [(cid, c) for cid, c in cells.items() if getattr(c, "alive", True)]
        n = len(alive_cells)

        arr = cls(n)
        type_map = {"normal": 0, "cancer": 1, "immune": 2}
        phase_map = {"G1": 0, "S": 1, "G2": 2, "M": 3, "G0": 4}

        for i, (cid, c) in enumerate(alive_cells):
            arr.cell_ids[i] = cid
            pos = getattr(c, "position", np.zeros(3))
            arr.positions[i] = pos if len(pos) == 3 else np.zeros(3)
            arr.cell_types[i] = type_map.get(getattr(c, "cell_type", "normal"), 0)
            arr.phases[i] = phase_map.get(getattr(c, "phase", "G1"), 0)
            arr.oxygen[i] = getattr(c, "oxygen", 0.21)
            arr.glucose[i] = getattr(c, "glucose", 5.0)
            arr.atp[i] = getattr(c, "atp", 1000.0)
            arr.lactate[i] = getattr(c, "lactate", 0.0)
            arr.mhc1[i] = getattr(c, "mhc1_expression", 1.0)

        return arr

    def write_back(self, cells: Dict[int, Any]):
        """Write array values back to CellState objects.

        Args:
            cells: Same dict that was used in from_cell_dict().
        """
        backend = self._backend
        # Convert to numpy if on GPU
        positions = backend.to_numpy(self.positions)
        oxygen = backend.to_numpy(self.oxygen)
        glucose = backend.to_numpy(self.glucose)
        atp = backend.to_numpy(self.atp)
        lactate = backend.to_numpy(self.lactate)
        ages = backend.to_numpy(self.ages)

        for i in range(self.n):
            cid = int(self.cell_ids[i])
            if cid not in cells:
                continue
            c = cells[cid]
            c.position[:] = positions[i]
            c.oxygen = float(oxygen[i])
            c.glucose = float(glucose[i])
            c.atp = float(atp[i])
            c.lactate = float(lactate[i])
            c.age = float(ages[i])


# ── Vectorized metabolism update ─────────────────────────────────

def update_metabolism_vectorized(
    arrays: CellArrays,
    dt: float,
    glucose_rate_normal: float = 0.2,
    glucose_rate_cancer: float = 0.5,
    o2_rate_normal: float = 0.15,
    o2_rate_cancer: float = 0.1,
    lactate_rate_normal: float = 0.1,
    lactate_rate_cancer: float = 0.3,
    atp_rate_normal: float = 100.0,
    atp_rate_cancer: float = 50.0,
) -> None:
    """Vectorized metabolism update for all cells simultaneously.

    Replaces CellularModule._update_metabolism() per-cell loop.

    Args:
        arrays: CellArrays with current state.
        dt: Time step in hours.
        glucose_rate_*: Glucose consumption rates.
        o2_rate_*: Oxygen consumption rates.
        lactate_rate_*: Lactate production rates.
        atp_rate_*: ATP production rates.
    """
    xp = arrays._backend.xp

    is_cancer = xp.asarray(arrays.cell_types == CellArrays.TYPE_CANCER)
    is_normal = xp.asarray(arrays.cell_types == CellArrays.TYPE_NORMAL)
    alive = arrays.alive

    # Rates: element-wise select based on cell type
    g_rate = xp.where(is_cancer, glucose_rate_cancer, glucose_rate_normal)
    o_rate = xp.where(is_cancer, o2_rate_cancer, o2_rate_normal)
    l_rate = xp.where(is_cancer, lactate_rate_cancer, lactate_rate_normal)
    a_rate = xp.where(is_cancer, atp_rate_cancer, atp_rate_normal)

    # Only update alive cells
    mask = alive.astype(xp.float32)

    arrays.glucose -= g_rate * dt * mask
    arrays.oxygen -= o_rate * dt * mask
    arrays.lactate += l_rate * dt * mask
    arrays.atp += a_rate * dt * mask
    arrays.ages += dt * mask

    # Clamp to non-negative
    arrays.glucose = xp.maximum(arrays.glucose, 0.0)
    arrays.oxygen = xp.maximum(arrays.oxygen, 0.0)
    arrays.lactate = xp.maximum(arrays.lactate, 0.0)
    arrays.atp = xp.maximum(arrays.atp, 0.0)


def detect_death_candidates(
    arrays: CellArrays,
    o2_threshold: float = 0.02,
    glucose_threshold: float = 0.5,
    cancer_death_prob: float = 0.01,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect cells that should die (vectorized).

    Returns indices of cells to kill, separated by type.

    Args:
        arrays: CellArrays.
        o2_threshold: O2 level below which normal cells die.
        glucose_threshold: Glucose level below which normal cells die.
        cancer_death_prob: Probability of cancer cell death under stress.
        seed: Random seed for reproducibility.

    Returns:
        (normal_death_ids, cancer_death_ids): arrays of cell_ids to kill.
    """
    xp = arrays._backend.xp
    rng = np.random.default_rng(seed)

    hypoxic = arrays.oxygen < o2_threshold
    starving = arrays.glucose < glucose_threshold
    stressed = xp.logical_or(hypoxic, starving)

    is_normal = xp.asarray(arrays.cell_types == CellArrays.TYPE_NORMAL)
    is_cancer = xp.asarray(arrays.cell_types == CellArrays.TYPE_CANCER)

    # Normal cells die deterministically under stress
    normal_die = xp.logical_and(stressed, xp.logical_and(is_normal, arrays.alive))

    # Cancer cells die stochastically under stress
    rand = xp.asarray(rng.random(arrays.n).astype(np.float32))
    cancer_die = xp.logical_and(
        stressed,
        xp.logical_and(is_cancer, xp.logical_and(arrays.alive, rand < cancer_death_prob)),
    )

    # Convert to numpy for cell ID lookup
    normal_mask = arrays._backend.to_numpy(normal_die)
    cancer_mask = arrays._backend.to_numpy(cancer_die)

    normal_ids = arrays.cell_ids[normal_mask]
    cancer_ids = arrays.cell_ids[cancer_mask]

    return normal_ids, cancer_ids


def detect_division_candidates(
    arrays: CellArrays,
    division_time_normal: float = 24.0,
    division_time_cancer: float = 12.0,
) -> np.ndarray:
    """Detect cells ready to divide (vectorized).

    Args:
        arrays: CellArrays.
        division_time_normal: Division time for normal cells (hours).
        division_time_cancer: Division time for cancer cells (hours).

    Returns:
        Array of cell_ids ready to divide.
    """
    xp = arrays._backend.xp

    is_cancer = xp.asarray(arrays.cell_types == CellArrays.TYPE_CANCER)
    div_time = xp.where(is_cancer, division_time_cancer, division_time_normal)

    ready = xp.logical_and(arrays.ages >= div_time, arrays.alive)
    ready_mask = arrays._backend.to_numpy(ready)

    return arrays.cell_ids[ready_mask]
