"""MHC class I on the cell surface, as a level between 0 and 1.

``mhc1`` is column 6 of the tissue cell-state array. On a cancer immunology
platform it is the variable the antigen-presentation story runs through:
whether a CD8 T cell can see a tumour cell at all.

The oxygen response is **measured**, digitised from Estephan et al., EMBO J
2025;44:903-922, doi 10.1038/s44318-024-00319-7, Figure 2C. HT29 colorectal
cancer cells were held at 21, 2, 1, 0.5 and <0.1% O2 for 24 h and surface
MHC I was read by flow cytometry, normalised to 21% O2:

    21% O2   1.00        0.5% O2   0.52
     2% O2   0.96       <0.1% O2   0.37
     1% O2   0.75

Three things that measurement settles, two of which contradict what this
module previously assumed:

**The response is cooperative, not hyperbolic.** A Hill curve fits with
n = 2.6 (residual sum of squares 0.0006); a Michaelis-Menten curve is forty
times worse (0.025) and the hyperbola this module shipped is a hundred times
worse (0.079). The practical difference is at the top of the range: the old
curve had a cell at 2% O2 down to 0.72 of its presentation, where the
measurement says 0.96. Most of the loss happens in a narrow band below 1%.

**It is NOT HIF-mediated.** RKO cells lacking HIF1alpha and HT29 cells with
ARNT/HIF1beta knocked down both still lose MHC I under hypoxia. The pathway
is the PERK arm of the unfolded protein response, through ATF4 to SUSD6,
which recruits the E3 ligase STUB1; ubiquitinated MHC I is then bound by the
cargo receptor NBR1 and degraded in the lysosome. This module's half-point
used to be justified by where HIF-1alpha stabilises, which was the wrong
mechanism arriving at roughly the right number.

**The loss is entirely post-transcriptional.** MHC I mRNA goes UP under
hypoxia -- HLA*A1 and HLA*C4 rise two- to threefold in HT29 -- while the
protein falls. So a model that drives presentation from transcript abundance
would get the sign backwards.

What is still an estimate, and flagged in :data:`CAVEATS`:

* The archetype baselines in :data:`BASELINE`. These remain
  ``cell_renderer``'s demo values, adopted so the simulation and the picture
  agree. The paper measures the oxygen RESPONSE of a tumour line relative to
  its own normoxic level; it says nothing about where a tumour cell starts
  relative to a healthy one. The library's B2M abundance is still not a usable
  source for that -- its 20 cell types are 5 bulk samples, implying CD4+
  T cells carry 2.79x the MHC-I of CD8+, which is spectral-counting variation
  on a draft proteome.
* The curve values themselves are read off a published figure, not from the
  source table. Exact data exist at BioStudies
  ``S-SCDT-10_1038-S44318-024-00319-7`` and would refine the fit.

And two things this module does not represent at all. The loss takes hours --
significant within 8 h, measured at 24 h -- while this treats presentation as
an instantaneous function of local oxygen, so a cell that has just entered a
hypoxic region is shown as already suppressed. And the two papers on this
disagree about the rest of the machinery: Sethumadhavan et al. (PLoS One
2017, PMID 29155844) report hypoxia downregulating TAP1/2 and LMP7, while
Estephan finds TAP2, LMP2 and LMP7 simply not induced by hypoxia, responding
to IFN-gamma instead. Neither is modelled here.

For scale, the downstream consequence Estephan measured: hypoxia cut unique
presented peptides from 2521 to 184, and total MHC I peptides roughly in half
(p = 0.0006).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

#: Oxygen at which surface MHC I is halfway between its floor and its
#: normoxic level, as a fraction of an atmosphere. 0.822% O2 = 6.2 mmHg,
#: fitted to Estephan Fig 2C. MEASURED, within the precision of reading a
#: figure. This was 1.5% O2 when it was a guess justified by HIF biology,
#: which is the wrong pathway.
HYPOXIC_HALF_O2 = 0.00822

#: Hill coefficient of that response. The measurement is markedly
#: cooperative, and a hyperbola (n = 1) fits forty times worse. Biologically
#: this is a signalling cascade -- PERK to ATF4 to SUSD6 to lysosome -- not a
#: binding equilibrium, so there is no reason to expect n = 1.
HILL_COEFFICIENT = 2.56

#: Fraction of the normoxic level surviving at zero oxygen, from the <0.1% O2
#: point. MEASURED. The previous estimate of 0.35 was close.
HYPOXIC_FLOOR = 0.373

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
    "the OXYGEN RESPONSE is measured -- Estephan et al., EMBO J 2025, "
    "doi 10.1038/s44318-024-00319-7, Fig 2C, HT29 at 24 h -- but digitised "
    "from the figure rather than read from the source table",
    "the BASELINES are still estimates: that paper measures a tumour line "
    "against its own normoxic level and says nothing about tumour versus "
    "healthy",
    "the library's B2M abundance is NOT a usable source for the baselines: "
    "20 cell types are 5 bulk samples and the implied CD4/CD8 ratio is an "
    "artefact",
    "the real loss takes 8-24 h; this applies it instantaneously, so a cell "
    "newly arrived in a hypoxic region is shown as already suppressed",
    "the mechanism is HIF-INDEPENDENT (PERK/UPR to autophagy), and the two "
    "papers disagree about TAP and LMP; neither is modelled",
)

#: Default for an archetype index with no entry, matching the non-cancer
#: somatic baseline rather than the immune one.
DEFAULT_BASELINE = 0.9


def hypoxic_factor(oxygen: Any) -> Any:
    """Fraction of the normoxic MHC-I level retained at this oxygen level.

    ``floor + (1 - floor) * C^n / (C^n + K^n)``, running from
    :data:`HYPOXIC_FLOOR` at zero oxygen towards 1. Fitted to Estephan
    Fig 2C; see the module docstring for why the exponent is not 1.

    The half-point is far above the respiratory one. A cell at 1% O2 is still
    respiring at nearly full rate -- the mitochondrial Km is 1 mmHg, eight
    times lower -- and has already lost a quarter of its surface MHC I. So
    this cannot be derived from the ATP charge and has to be its own curve.
    """
    o2 = np.clip(np.asarray(oxygen, dtype=np.float64), 0.0, None)
    powered = o2 ** HILL_COEFFICIENT
    half = HYPOXIC_HALF_O2 ** HILL_COEFFICIENT
    return HYPOXIC_FLOOR + (1.0 - HYPOXIC_FLOOR) * powered / (powered + half)


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
            f"{DEFAULT_BASELINE}), measured hypoxic half-point "
            f"{HYPOXIC_HALF_O2:.5f} ({HYPOXIC_HALF_O2 * 760:.1f} mmHg), "
            f"Hill n {HILL_COEFFICIENT}, floor {HYPOXIC_FLOOR}. "
            + "; ".join(CAVEATS))
