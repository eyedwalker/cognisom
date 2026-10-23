"""Tumour lactic acid blunts the killers.

Fischer et al. 2007 (Blood 109:3812, PMID 17255361): lactic acid
suppressed the proliferation and cytokine production of human cytotoxic
T lymphocytes by up to 95% and cut their cytotoxic activity by 50%; a
24 h recovery in acid-free medium restored it. Brand et al. 2016 (Cell
Metab 24:657, PMID 27641098): LDHA-driven lactic acid accumulation in
melanomas inhibits tumour surveillance by T AND NK cells, through NFAT
and IFN-gamma; Ldha-low tumours grew slower only when lymphocytes were
present.

What is sourced and what is not: the 50% ceiling on cytotoxic activity
is Fischer's measurement. The concentration scale is not in either
abstract: the full effect is placed at 20 mM (the upper concentration
Fischer tested, and the upper end of Walenta's tumour range, 6-10
umol/g in cervical cancers, PMID 10706105) and the onset at 2 mM, the
upper end of normal tissue, so that a normally perfused tissue is
untouched. Both are ESTIMATES. Linear between them, which is the
simplest shape that has an onset and a ceiling.

Same shape as sympathetic_attenuation and arginase_attenuation, on
purpose: a final multiplicative gate on the kill path, exactly 1.0 in a
tissue at normal lactate, so every existing configuration is a
provable no-op.
"""

from __future__ import annotations

#: Fischer 2007: cytotoxic activity fell 50%. MEASURED (the ceiling).
DEFAULT_LACTATE_MAX_SUPPRESSION = 0.5
#: Normal tissue sits at 1-2 mM; no effect below. ESTIMATE.
DEFAULT_LACTATE_FLOOR_MM = 2.0
#: Full effect at 20 mM. ESTIMATE (the top concentration Fischer used).
DEFAULT_LACTATE_FULL_MM = 20.0


def lactate_attenuation(lactate_mM: float,
                        floor_mM: float = DEFAULT_LACTATE_FLOOR_MM,
                        full_mM: float = DEFAULT_LACTATE_FULL_MM,
                        max_suppression: float = DEFAULT_LACTATE_MAX_SUPPRESSION) -> float:
    """Fraction of kill function retained at an interstitial lactate.

    1.0 at or below ``floor_mM``; falls linearly to ``1 - max_suppression``
    at ``full_mM``; flat beyond. Applies to T cells and NK cells alike
    (Brand 2016).
    """
    if lactate_mM is None:
        return 1.0
    lac = float(lactate_mM)
    if lac <= floor_mM or full_mM <= floor_mM:
        return 1.0
    frac = min(1.0, (lac - floor_mM) / (full_mM - floor_mM))
    m = min(1.0, max(0.0, float(max_suppression)))
    return 1.0 - frac * m
