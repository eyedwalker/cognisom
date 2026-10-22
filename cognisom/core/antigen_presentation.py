"""MHC class I on the cell surface, as a level between 0 and 1.

``mhc1`` is column 6 of the tissue cell-state array and nothing wrote it. On a
cancer immunology platform that is the variable the antigen-presentation story
runs through: whether a CD8 T cell can see a tumour cell at all. The display
path has existed for a while -- ``cell_renderer`` offers ``color_by="mhc1"``
and ``exporters`` writes an ``mhc1_expression`` column -- fed by synthetic
demo values.

**Every number in this module is an ESTIMATE and none of them is measured.**
That is unusual for this repo and it is deliberate, so read why before
trusting an immune-escape rate that comes out of it.

What is solid is the direction and the mechanism. Hypoxia suppresses MHC class
I, HIF-dependently, and also downregulates TAP1/2 and LMP7, which dominate the
surface display of peptide-MHC complexes: Sethumadhavan et al., PLoS One 2017,
PMID 29155844, with p = 0.002 to 0.013 across tumours in vivo and 3D spheroids
in vitro. Later work attributes part of the effect to autophagy through the
PERK arm of the unfolded protein response, and part to HIF1A-AS2
destabilising MHC-I protein; the mechanisms differ and all point the same way.

What is missing is any effect SIZE. Those papers report p-values in their text
and put the magnitudes in figure images, and the one study with an oxygen
dose-response (21% down to below 0.1%) is behind a publisher login. So the two
numbers below were chosen, not found:

* :data:`HIF_HALF_O2` -- where the downregulation is half-complete. Set at
  1.5% O2 because HIF-1alpha stabilises progressively below about 5% O2 and is
  strongly stabilised at 1%, which brackets it. Note this is an order of
  magnitude ABOVE the respiratory half-saturation of 1 mmHg: a cell can be
  respiring at full rate and already losing surface MHC-I, which is the whole
  reason this is a separate curve and not a function of the ATP charge.
* :data:`HYPOXIC_FLOOR` -- how much survives at zero oxygen. Set at 0.35
  because the literature describes downregulation and not abolition; a floor
  of zero would make anoxic cells perfectly invisible, which is a stronger
  claim than anyone has made.

And the baselines in :data:`BASELINE` are the values ``cell_renderer``'s demo
data already used, adopted so that the simulation and the picture agree rather
than because anyone measured them. The library's B2M abundance looks like the
right source for them and is not: its 20 cell types are 5 bulk samples, and
it implies CD4+ T cells carry 2.79x the MHC-I of CD8+ T cells, which is
spectral-counting variation on a draft proteome rather than biology. See
``tests/test_mhc1_has_a_consumer_and_no_producer.py``, which this module does
not delete, because the reasons it records are still true.

One methodological note, kept because it bears on whether this coupling is
observable at all. Sethumadhavan found the effect in 3D spheroids and not in
2D monolayers for MCA205, though EL4 showed it in both -- so it is
line-dependent, not a universal property of 2D culture. Where it does need 3D,
it needs real oxygen gradients, which this model only acquired once the
diffusion field became trustworthy.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

#: Half-saturation of the hypoxic downregulation, as a fraction of an
#: atmosphere. 1.5% O2 = 11 mmHg. ESTIMATED -- see the module docstring.
HIF_HALF_O2 = 0.015

#: Fraction of the normoxic level that survives at zero oxygen. ESTIMATED.
HYPOXIC_FLOOR = 0.35

#: Normoxic surface level by archetype index, 0 to 1. ESTIMATED; these are
#: the values cell_renderer's demo data already used. Index 5 is
#: cancer_epithelial and 6-8 are the immune archetypes.
BASELINE: Dict[int, float] = {
    0: 0.9,    # luminal_secretory
    1: 0.9,    # basal
    2: 0.9,    # fibroblast
    3: 0.9,    # smooth_muscle
    4: 0.9,    # endothelial
    5: 0.3,    # cancer_epithelial -- downregulated, the evasion route
    6: 1.0,    # t_cell
    7: 1.0,    # macrophage
    8: 1.0,    # nk_cell
    9: 0.9,    # neuroendocrine
}

#: What a caller is entitled to conclude from these numbers, and what not.
CAVEATS: Tuple[str, ...] = (
    "every value here is ESTIMATED; none is measured",
    "the direction and mechanism are sourced (PMID 29155844) but no published "
    "effect size was reachable, so HIF_HALF_O2 and HYPOXIC_FLOOR were chosen",
    "the baselines are cell_renderer's demo values, adopted for consistency "
    "and not measured",
    "the library's B2M abundance is NOT a usable source for the baselines: 20 "
    "cell types are 5 bulk samples and the implied CD4/CD8 ratio is an "
    "artefact",
    "an immune-escape rate derived from this is an assumption, not a result",
)

#: Default for an archetype index with no entry, matching the non-cancer
#: somatic baseline rather than the immune one.
DEFAULT_BASELINE = 0.9


def hypoxic_factor(oxygen: Any) -> Any:
    """Fraction of the normoxic MHC-I level retained at this oxygen level.

    Saturating in oxygen, with a floor: ``floor + (1 - floor) * C/(C + km)``.
    Runs from :data:`HYPOXIC_FLOOR` at zero oxygen to 1 as oxygen rises.

    The half-point is an order of magnitude above the respiratory one on
    purpose. A cell at 1% O2 is still respiring at nearly full rate -- the
    mitochondrial Km is 1 mmHg -- and is already losing surface MHC-I, so this
    cannot be a function of the ATP charge and has to be its own curve.
    """
    o2 = np.clip(np.asarray(oxygen, dtype=np.float64), 0.0, None)
    return HYPOXIC_FLOOR + (1.0 - HYPOXIC_FLOOR) * o2 / (o2 + HIF_HALF_O2)


def baseline_for(cell_types: Any) -> Any:
    """Normoxic level per cell, from the archetype index."""
    types = np.asarray(cell_types, dtype=np.int64)
    out = np.full(types.shape, DEFAULT_BASELINE, dtype=np.float64)
    for index, value in BASELINE.items():
        out[types == index] = value
    return out


def surface_level(oxygen: Any, cell_types: Any) -> Any:
    """MHC-I surface level per cell, in [0, 1].

    The product of an archetype baseline and the hypoxic factor, so a tumour
    cell in an anoxic core is doubly suppressed -- which is the mechanism the
    platform exists to study, and is also why the numbers behind it are worth
    treating with suspicion until someone sources them.
    """
    return baseline_for(cell_types) * hypoxic_factor(oxygen)


def describe() -> str:
    """One line, carrying the caveats with the numbers."""
    return (f"MHC-I: baseline by archetype (cancer {BASELINE[5]}, somatic "
            f"{DEFAULT_BASELINE}), hypoxic half-point {HIF_HALF_O2:.3f} "
            f"({HIF_HALF_O2 * 760:.0f} mmHg), floor {HYPOXIC_FLOOR}. "
            + "; ".join(CAVEATS))
