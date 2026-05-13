"""
Tumor microenvironment 4-type classifier (Teng et al., Cancer Res 2015).

Translates cognisom's per-cell simulation state into the clinically
familiar 4-type TME classification used to predict immune checkpoint
blockade response:

    Type  | TIL | PD-L1 | Mechanism             | ICB response
    ------+-----+-------+-----------------------+-------------
    I     |  +  |   +   | Adaptive immune       | HIGH
          |     |       | resistance            |
    II    |  -  |   -   | Immunological         | MINIMAL
          |     |       | ignorance (cold)      |
    III   |  -  |   +   | Intrinsic PD-L1       | LOW
          |     |       | induction (oncogenic) |
    IV    |  +  |   -   | Tolerance / other     | MODERATE
          |     |       | suppression           |

Reference: Teng MW, Ngiow SF, Ribas A, Smyth MJ. "Classifying Cancers
Based on T-cell Infiltration and PD-L1." Cancer Res. 2015;75(11):
2139-45. PMID: 25977340.

Patent-evidence claim surface (Upgrade 4 / TME output):
  Per-patient VCF -> simulation -> (TIL density, PD-L1 fraction) ->
  TME type -> predicted ICB response category.

A clinical pipeline (IHC / RNA panel) infers the same four types from
biopsy staining; cognisom infers them from the closed-loop simulation
output. The novelty vs. clinical state of the art is that cognisom
computes the classification *prospectively* from the patient's
genome rather than from a tumor biopsy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Tuple

import numpy as np


# Default thresholds. Calibrated from clinical IHC norms:
#   TILs/HPF >100 = "hot" (per Galon immunoscore literature), which
#   maps roughly to >= 0.5 TILs per cancer cell in our cell-grained
#   simulation. PD-L1+ in clinical practice is typically defined as
#   >=1% TPS (tumor proportion score) for some indications and >=50%
#   for others; we default to 25% as a middle ground.
DEFAULT_TIL_RATIO_POS: float = 0.5
DEFAULT_PDL1_FRAC_POS: float = 0.25
DEFAULT_PDL1_PER_CELL_POS: float = 0.5  # a single cell is PD-L1+ at this level
DEFAULT_TIL_PROXIMITY_UM: float = 20.0  # within this radius of a cancer cell

# ECM threshold above which the tumor is classified "stromally dense"
# enough to plausibly exclude TILs (Upgrade 6). 0.4 picks up tumors in
# the moderately-fibrotic regime; PDAC presets typically land at 0.7+.
DEFAULT_ECM_EXCLUDED_THRESHOLD: float = 0.4


class TMEType(str, Enum):
    """Four TME categories from Teng et al. 2015."""
    TYPE_I = "I"        # TIL+ PDL1+  -- adaptive immune resistance
    TYPE_II = "II"      # TIL- PDL1-  -- immunological ignorance
    TYPE_III = "III"    # TIL- PDL1+  -- intrinsic induction
    TYPE_IV = "IV"      # TIL+ PDL1-  -- tolerance / other suppression


# Predicted ICB response per Teng et al. + downstream clinical evidence
# (e.g., Tumeh / Herbst / KEYNOTE-158): Type I responds best, Type II
# essentially does not respond (no antigens), Type III responds poorly
# (intrinsic PD-L1 without TILs to be unblocked), Type IV intermediate.
_ICB_RESPONSE: dict = {
    TMEType.TYPE_I: "high",
    TMEType.TYPE_II: "minimal",
    TMEType.TYPE_III: "low",
    TMEType.TYPE_IV: "moderate",
}


_DESCRIPTION: dict = {
    TMEType.TYPE_I: (
        "Adaptive immune resistance: TILs are present and the tumor "
        "has up-regulated PD-L1 in response to IFN-gamma signalling. "
        "Strongest candidate for anti-PD-1/PD-L1 checkpoint blockade."
    ),
    TMEType.TYPE_II: (
        "Immunological ignorance: no TILs and no PD-L1. The tumor is "
        "'cold' -- either no neoantigens, HLA mismatch, or physical "
        "exclusion by ECM/fibrosis. Checkpoint blockade unlikely to "
        "help; consider neoantigen vaccine or oncolytic virus to "
        "convert to Type I."
    ),
    TMEType.TYPE_III: (
        "Intrinsic PD-L1 induction: PD-L1 expressed via oncogenic "
        "signalling (PI3K, MYC, EGFR pathways) without TIL pressure. "
        "Checkpoint blockade typically ineffective because there are "
        "no TILs to un-block; consider combination with TIL-priming "
        "agents."
    ),
    TMEType.TYPE_IV: (
        "Tolerance / other suppression: TILs present but no PD-L1 "
        "expression -- T cells likely held in check by Treg / MDSC "
        "/ TGF-beta / M2 macrophage axes rather than PD-1/PD-L1. "
        "Checkpoint blockade alone less effective; consider Treg "
        "depletion or macrophage repolarization."
    ),
}


@dataclass(frozen=True)
class TMEClassification:
    """One snapshot of TME state + the classification it yields.

    n_cancer_cells, n_til:
        Raw populations the classifier saw.
    til_ratio:
        ``n_til / n_cancer_cells``. Higher = more infiltrated.
    pdl1_positive_fraction:
        Fraction of cancer cells with pdl1_expression >= the per-cell
        threshold. The tumor as a whole is classified PD-L1+ when this
        is at least ``pdl1_fraction_threshold``.
    mean_ecm_density:
        Average ``local_ecm_density`` across cancer cells. Drives the
        ``ecm_excluded`` flag below.
    ecm_excluded:
        True iff the tumor is TIL-negative AND has high stromal
        density AND its cancer cells display at least one MHC-I
        neoantigen. This is a clinically actionable sub-classification
        of Teng Type II: instead of "no antigens / cold and unfixable",
        the patient has *antigens behind a wall* -- a candidate for
        anti-fibrotic + ICB combo (Upgrade 6).
    tme_type:
        One of the four Teng categories.
    predicted_icb_response:
        ``"high" | "moderate" | "low" | "minimal"`` based on the type.
    description:
        Human-readable mechanism and clinical implication.
    """
    n_cancer_cells: int
    n_til: int
    til_ratio: float
    pdl1_positive_fraction: float
    til_ratio_threshold: float
    pdl1_fraction_threshold: float
    tme_type: TMEType
    predicted_icb_response: str
    description: str
    mean_ecm_density: float = 0.0
    ecm_excluded: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _position(obj: Any) -> Optional[np.ndarray]:
    """Best-effort extraction of an object's 3D position."""
    pos = getattr(obj, "position", None)
    if pos is None:
        return None
    return np.asarray(pos, dtype=np.float32)


def _count_tils(
    cancer_cells: Iterable,
    immune_cells: Iterable,
    proximity_um: float,
    immune_type_filter: Optional[Tuple[str, ...]] = ("T_cell",),
) -> int:
    """Count immune cells (default: T cells only) within proximity_um
    of *any* cancer cell. Each immune cell counted at most once."""
    cancer_positions = []
    for cell in cancer_cells:
        p = _position(cell)
        if p is not None:
            cancer_positions.append(p)
    if not cancer_positions:
        return 0
    cancer_stack = np.stack(cancer_positions)

    count = 0
    for ic in immune_cells:
        if immune_type_filter is not None:
            if getattr(ic, "cell_type", None) not in immune_type_filter:
                continue
        if getattr(ic, "in_blood", False):
            continue
        p = _position(ic)
        if p is None:
            continue
        # Closest distance to any cancer cell
        dist = float(np.min(np.linalg.norm(cancer_stack - p, axis=1)))
        if dist <= proximity_um:
            count += 1
    return count


def _count_pdl1_positive(
    cancer_cells: Iterable,
    per_cell_threshold: float,
) -> int:
    """Count cancer cells with pdl1_expression >= per_cell_threshold."""
    n = 0
    for cell in cancer_cells:
        if getattr(cell, "pdl1_expression", 0.0) >= per_cell_threshold:
            n += 1
    return n


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_tme(
    cancer_cells: Iterable,
    immune_cells: Iterable,
    *,
    til_ratio_threshold: float = DEFAULT_TIL_RATIO_POS,
    pdl1_fraction_threshold: float = DEFAULT_PDL1_FRAC_POS,
    pdl1_per_cell_threshold: float = DEFAULT_PDL1_PER_CELL_POS,
    til_proximity_um: float = DEFAULT_TIL_PROXIMITY_UM,
    immune_type_filter: Optional[Tuple[str, ...]] = ("T_cell",),
    ecm_excluded_threshold: float = DEFAULT_ECM_EXCLUDED_THRESHOLD,
) -> TMEClassification:
    """Classify the TME state into one of Teng's four types.

    Parameters
    ----------
    cancer_cells, immune_cells
        Iterables of cell-like objects with ``.position`` (3-vector)
        and, for cancer cells, ``.pdl1_expression`` (float in [0, 1]).
    til_ratio_threshold
        Minimum (TILs / cancer cells) for the tumor to be classified
        TIL+. Default 0.5 (one T cell per two cancer cells).
    pdl1_fraction_threshold
        Minimum fraction of cancer cells expressing PD-L1 at the
        per-cell threshold for the tumor to be classified PD-L1+.
        Default 0.25 (25% of cancer cells must be PD-L1+).
    pdl1_per_cell_threshold
        A single cell is considered PD-L1+ when its
        ``pdl1_expression`` >= this value. Default 0.5.
    til_proximity_um
        Distance within which an immune cell is counted as a TIL.
        Default 20 um (typical T-cell detection radius in our sim).
    immune_type_filter
        Which immune cell types to count as TILs. Default
        ``("T_cell",)`` -- the clinical IHC definition counts CD8s.
        Pass ``None`` to count any immune cell type.
    """
    cancer_list = list(cancer_cells)
    n_cancer = len(cancer_list)
    immune_list = list(immune_cells)

    if n_cancer == 0:
        # Degenerate: no tumor -> nothing to classify. We still return
        # a record so the caller can log it.
        return TMEClassification(
            n_cancer_cells=0,
            n_til=0,
            til_ratio=0.0,
            pdl1_positive_fraction=0.0,
            til_ratio_threshold=til_ratio_threshold,
            pdl1_fraction_threshold=pdl1_fraction_threshold,
            tme_type=TMEType.TYPE_II,
            predicted_icb_response=_ICB_RESPONSE[TMEType.TYPE_II],
            description="No cancer cells observed; TME classification "
                        "is degenerate (defaulting to Type II)",
        )

    n_til = _count_tils(
        cancer_list, immune_list,
        proximity_um=til_proximity_um,
        immune_type_filter=immune_type_filter,
    )
    til_ratio = n_til / n_cancer

    n_pdl1_pos = _count_pdl1_positive(cancer_list, pdl1_per_cell_threshold)
    pdl1_frac = n_pdl1_pos / n_cancer

    # Mean ECM density across the tumor + any-neoantigen-displayed
    # check (Upgrade 6). The exclusion flag fires when the tumor is
    # cold *despite* carrying neoantigens behind a fibrotic wall.
    ecm_values = [
        float(getattr(c, "local_ecm_density", 0.0)) for c in cancer_list
    ]
    mean_ecm = float(np.mean(ecm_values)) if ecm_values else 0.0
    any_antigens = any(
        getattr(c, "mhc1_displayed_peptides", None) for c in cancer_list
    )

    til_positive = til_ratio >= til_ratio_threshold
    pdl1_positive = pdl1_frac >= pdl1_fraction_threshold
    ecm_excluded = (
        not til_positive
        and mean_ecm >= ecm_excluded_threshold
        and any_antigens
    )

    if til_positive and pdl1_positive:
        t = TMEType.TYPE_I
    elif not til_positive and not pdl1_positive:
        t = TMEType.TYPE_II
    elif not til_positive and pdl1_positive:
        t = TMEType.TYPE_III
    else:  # til_positive and not pdl1_positive
        t = TMEType.TYPE_IV

    description = _DESCRIPTION[t]
    if ecm_excluded:
        # Override the Type-II / Type-III description with the
        # actionable ECM-exclusion narrative when the flag fires.
        description = (
            "ECM-excluded (Upgrade 6 sub-classification of Type "
            f"{t.value}): TILs are absent and mean stromal density is "
            f"{mean_ecm:.2f}, yet the tumor carries neoantigens that "
            "would be MHC-presentable. The barrier is physical, not "
            "antigenic -- candidate for anti-fibrotic + ICB combination "
            "(e.g., PDAC-style desmoplasia, Herzog et al. 2023 in NSCLC)."
        )

    return TMEClassification(
        n_cancer_cells=n_cancer,
        n_til=n_til,
        til_ratio=til_ratio,
        pdl1_positive_fraction=pdl1_frac,
        til_ratio_threshold=til_ratio_threshold,
        pdl1_fraction_threshold=pdl1_fraction_threshold,
        tme_type=t,
        predicted_icb_response=_ICB_RESPONSE[t],
        description=description,
        mean_ecm_density=mean_ecm,
        ecm_excluded=ecm_excluded,
    )
