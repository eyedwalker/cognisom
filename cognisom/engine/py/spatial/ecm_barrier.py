"""
ECM (extracellular matrix) barrier: spatial T-cell exclusion.

Models the desmoplastic stroma -- the dense, collagen-rich matrix
that surrounds many solid tumors (paradigmatically PDAC: 80% of
tumor mass) and physically excludes infiltrating lymphocytes. The
lecture by V. Chen (CU Anschutz 2026-05-03, slides 34-35, 49-50)
flagged this as the single biggest mechanistic gap in our pipeline:

  * Only 1% of PDAC patients respond to ICB despite having
    targetable mutations. The reason is not absence of neoantigens
    but absence of *physical* T-cell access.
  * Tumor-associated fibrosis impairs immune surveillance in NSCLC
    (Herzog et al., Sci Transl Med 2023).
  * Inhibiting desmoplasia (e.g., Cdh11 knockout) increases CD8
    infiltration and extends survival in KPC mouse models of PDAC
    (PMID PMC7956114).

This module exposes pure functions:

  ecm_density_at(position, cancer_cells, sample_radius)
      Sample the local ECM density at a 3D position by averaging
      per-cell pdl1-style local_ecm_density across nearby cancer
      cells. Returns a scalar in [0, 1].

  motility_attenuation(base_speed, ecm_density, blocking_factor)
      Scale a T cell's motility speed by ECM. With blocking_factor
      = 0.9 and ecm_density = 1.0, the T cell moves at 10% of base
      speed -- effectively excluded.

  detection_attenuation(base_radius, ecm_density, blocking_factor)
      Scale a T cell's detection radius by ECM. High ECM compresses
      the radius until the T cell can no longer "see" antigen.

The implementation is intentionally cell-grained (per-cancer-cell
``local_ecm_density``) rather than grid-based, so it composes with
the existing cellular module without requiring a new ECM
concentration solver. The patent claim is the *gating* of immune
function on stromal density, not the exact spatial diffusion.

Patent-evidence claim surface (Upgrade 6):
  Given a patient VCF + tumor type, the simulation predicts ICB
  efficacy modulated by stromal density. The pipeline distinguishes:
    * Type II (immunological ignorance, ECM-low): no neoantigens
        -> recommend cancer vaccine
    * Type II (ECM-excluded, ECM-high): neoantigens present but
        TILs blocked -> recommend anti-fibrotic + ICB combo
  This split is invisible to rule-based prior art that does not
  model spatial stromal barriers.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


# Default tuning. These constants are exposed so the cellular and
# immune modules can override them per simulation regime (e.g.,
# PDAC vs melanoma have different stromal densities).
DEFAULT_SAMPLE_RADIUS_UM: float = 30.0
DEFAULT_MOTILITY_BLOCK: float = 0.9    # 1.0 ECM -> 10% speed
DEFAULT_DETECTION_BLOCK: float = 0.8   # 1.0 ECM -> 20% radius
MIN_RETAINED_FRACTION: float = 0.05    # speeds / radii never drop below 5%


def ecm_density_at(
    position: np.ndarray,
    cancer_cells: Iterable,
    sample_radius_um: float = DEFAULT_SAMPLE_RADIUS_UM,
) -> float:
    """Average local ECM density at a 3D position.

    Sampled as the mean of ``local_ecm_density`` across cancer cells
    within ``sample_radius_um`` of ``position``. Returns 0.0 if no
    cancer cells are in range (open tissue, no stromal pressure).

    Result is clamped to [0, 1].
    """
    pos = np.asarray(position, dtype=np.float32)
    densities = []
    for cell in cancer_cells:
        cell_pos = np.asarray(
            getattr(cell, "position", None), dtype=np.float32
        )
        if cell_pos is None or cell_pos.shape != pos.shape:
            continue
        if float(np.linalg.norm(cell_pos - pos)) <= sample_radius_um:
            densities.append(
                float(getattr(cell, "local_ecm_density", 0.0))
            )
    if not densities:
        return 0.0
    avg = float(np.mean(densities))
    return max(0.0, min(1.0, avg))


def motility_attenuation(
    base_speed: float,
    ecm_density: float,
    blocking_factor: float = DEFAULT_MOTILITY_BLOCK,
) -> float:
    """Effective T-cell speed after ECM impedance.

    ``blocking_factor`` is how much of the base speed ECM can take
    away at maximum density. A factor of 0.9 means a fully fibrotic
    region cuts speed to 10%; factor 1.0 would block completely
    (we floor at MIN_RETAINED_FRACTION so T cells never get stuck
    forever).
    """
    d = max(0.0, min(1.0, float(ecm_density)))
    b = max(0.0, min(1.0, float(blocking_factor)))
    retained = max(MIN_RETAINED_FRACTION, 1.0 - d * b)
    return float(base_speed) * retained


def detection_attenuation(
    base_radius: float,
    ecm_density: float,
    blocking_factor: float = DEFAULT_DETECTION_BLOCK,
) -> float:
    """Effective T-cell detection radius after ECM occlusion.

    The biology: dense collagen scatters chemokine gradients and
    occludes paracrine signalling, so the T cell's effective
    sensing radius shrinks. Same shape as motility_attenuation.
    """
    d = max(0.0, min(1.0, float(ecm_density)))
    b = max(0.0, min(1.0, float(blocking_factor)))
    retained = max(MIN_RETAINED_FRACTION, 1.0 - d * b)
    return float(base_radius) * retained
