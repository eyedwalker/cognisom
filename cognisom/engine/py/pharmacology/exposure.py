"""How much of a drug reaches a cell, given where the cell is.

The assumption this module makes visible
----------------------------------------
``ARSignalingModule.activity_of`` returned ONE number for the whole
tumour. Every cancer cell saw the same enzalutamide, whether it sat
against a capillary or 200 um away in the middle of a nodule. Nothing
said so: uniform exposure was not a stated modelling choice, it was the
absence of one, and a simulation that cannot represent "the drug did not
reach those cells" cannot offer that as a reason a treatment failed --
which is one of the leading reasons treatments do fail.

What is measured
----------------
Primeau 2005 (PMID 16361566, abstract; full text not open) imaged
doxorubicin against CD31-stained vessels in murine 16C and EMT6 tumours
AND human prostate PC-3 xenografts: "The concentration of doxorubicin
decreases exponentially with distance from tumor blood vessels,
decreasing to half its perivascular concentration at a distance of about
40 to 50 um", with "the mean distance from blood vessels to regions of
hypoxia ... 90 to 140 um", so that "many viable tumor cells are not
exposed to detectable concentrations of drug following a single
injection". Minchinton & Tannock 2006 (PMID 16862189) is the review of
that literature.

That gives the SHAPE -- an exponential in distance from the vessel --
and one drug's half-distance. It is the number for DOXORUBICIN, a
DNA-binding anthracycline that tissue sequesters avidly, measured
FOLLOWING A SINGLE INJECTION.

Why the default is still uniform, and why that is now a claim
-------------------------------------------------------------
Enzalutamide is the opposite dosing regime, and its pharmacokinetics are
measured in full: Gibbons 2015 (PMID 25917876, PMC4580721, full text
read) reports a half-life of 5.8 days, steady state by day 28, 8.3-fold
accumulation on once-daily dosing and a mean peak-to-trough ratio of
1.25 -- "a relatively flat concentration profile over the dose
interval". Its exposure-response analysis found every C-trough quartile
uniformly beneficial with no threshold.

Diffusion alone cannot hold a gradient against that. The time for a
small molecule to diffuse 200 um is about L^2 / 2D: at the diffusion
coefficient this engine already uses for testosterone, a molecule of
similar size, that is minutes (``equilibration_hours`` below computes
it). Against a flat profile sustained for weeks past day 28, minutes is
four orders of magnitude of margin.

So a steady-state gradient would have to be held by a SINK -- binding,
sequestration, metabolism or efflux -- which is exactly what doxorubicin
has and what makes its 40-50 um real. Whether enzalutamide has one in
tumour tissue is NOT KNOWN HERE: searches for an intratumoral
enzalutamide or AR-inhibitor concentration returned nothing. Uniform
exposure is therefore the defensible default for the drug this platform
ships, and it is now a stated claim with its reasoning attached rather
than an unexamined one -- and ``drug_half_distance_um`` is the switch
that tests how much it matters.

Shape
-----
``exposure_fraction`` is the fraction of the perivascular concentration
at a distance, ``2 ** (-distance / half_distance)``: 1.0 at the vessel
wall, 0.5 at one half-distance, and exactly 1.0 everywhere when no
half-distance is given. It scales what the drug DOES rather than the
cell's state, so it composes with a blockade and with the agonist switch
alike -- see ``ARSignalingModule.activity_of``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

#: Half-distance for a BOLUS-dosed, tissue-sequestered drug, in um.
#: Doxorubicin. Not a property of tumours in general and not this
#: platform's drug -- see the module docstring before using it.
BOLUS_HALF_DISTANCE_UM: float = 45.0
BOLUS_HALF_DISTANCE_RANGE = (40.0, 50.0)
BOLUS_SOURCE = (
    "Primeau 2005, PMID 16361566 (abstract): doxorubicin halves every "
    "40-50 um from tumour vessels, murine 16C/EMT6 and human PC-3 "
    "xenografts, FOLLOWING A SINGLE INJECTION"
)

#: Distance at which Primeau's tumours became hypoxic, in um. Recorded
#: because it is the same geometry this engine's oxygen path solves, so
#: it is a cross-check on the vessel spacing rather than a drug number.
HYPOXIA_DISTANCE_RANGE_UM = (90.0, 140.0)

#: Enzalutamide's pharmacokinetics, all from Gibbons 2015 (PMID
#: 25917876, PMC4580721), read in full text. These are what justify a
#: uniform default; none of them is a tissue-distribution measurement.
ENZALUTAMIDE_PK: Dict[str, object] = {
    "half_life_days": 5.8,
    "steady_state_day": 28,
    "accumulation_fold": 8.3,
    "peak_to_trough_ratio": 1.25,
    "intersubject_cv_max": 0.30,
    "source": "Gibbons 2015, PMID 25917876 (full text)",
}


@dataclass(frozen=True)
class DrugPenetration:
    """A drug's tissue penetration, and how far it can be trusted."""

    #: None means uniform exposure: every cell sees the plasma level.
    half_distance_um: Optional[float]
    status: str          # 'sourced', 'derived' or 'ESTIMATE'
    source: str

    def __post_init__(self) -> None:
        if self.half_distance_um is not None and self.half_distance_um <= 0.0:
            raise ValueError(
                "a half-distance must be positive; None means uniform")

    @property
    def is_uniform(self) -> bool:
        return self.half_distance_um is None


#: What the platform's own drug runs on, and why.
ENZALUTAMIDE_PENETRATION = DrugPenetration(
    half_distance_um=None, status='derived',
    source=(
        "Uniform at steady state, derived from Gibbons 2015 (PMID "
        "25917876, full text): half-life 5.8 d, steady state by day 28, "
        "peak-to-trough 1.25. Diffusion equilibrates 200 um in minutes, "
        "so a gradient would need a sink, and no intratumoral "
        "enzalutamide measurement was found. NOT a measurement of "
        "uniformity"))

#: The alternative, for a drug given as a bolus and held by the tissue.
BOLUS_PENETRATION = DrugPenetration(
    half_distance_um=BOLUS_HALF_DISTANCE_UM, status='sourced',
    source=BOLUS_SOURCE)


def exposure_fraction(distance_um: float,
                      half_distance_um: Optional[float]) -> float:
    """Fraction of the perivascular drug level at ``distance_um``.

    ``2 ** (-distance / half_distance)``, and exactly 1.0 when
    ``half_distance_um`` is None, so a run that declares no penetration
    limit is identical to one from before this module existed -- to the
    bit, not approximately.
    """
    if half_distance_um is None:
        return 1.0
    if half_distance_um <= 0.0:
        raise ValueError("a half-distance must be positive")
    d = max(0.0, float(distance_um))
    if d == float('inf'):
        # No vessel in the run at all. A cell no vessel serves gets no
        # drug, which is the same answer the oxygen path gives it.
        return 0.0
    return float(2.0 ** (-d / float(half_distance_um)))


def scale_activity(activity: float, exposure: float) -> float:
    """What a drug does to a cell that sees ``exposure`` of it.

    ``activity`` is ARSignalingModule's multiplier on AR-driven
    transcription: below 1 for a blockade, above 1 for the agonist case
    a receptor variant can turn an anti-androgen into. Exposure scales
    the DEPARTURE FROM UNTREATED, so

        exposure 1 -> activity, unchanged
        exposure 0 -> 1.0, the untreated baseline

    which is right for both signs at once: a carrier far from a vessel
    gets less of the agonist drive, exactly as a wild-type cell there
    gets less of the blockade. Doing this by scaling the drug's effect
    rather than the cell's state is what keeps one rule for both.
    """
    e = max(0.0, min(1.0, float(exposure)))
    return 1.0 + (float(activity) - 1.0) * e


def equilibration_hours(distance_um: float,
                        diffusion_um2_s: float) -> float:
    """Rough time for diffusion to cross ``distance_um``: L^2 / 2D.

    Used in the docstring's argument and asserted in the tests, so the
    claim "diffusion equilibrates this in minutes" is computed from the
    engine's own diffusion coefficient rather than asserted in prose.
    """
    if diffusion_um2_s <= 0.0:
        raise ValueError("a diffusion coefficient must be positive")
    seconds = (float(distance_um) ** 2) / (2.0 * float(diffusion_um2_s))
    return seconds / 3600.0


def penetration_report() -> Dict[str, Dict[str, object]]:
    """Every penetration number, for a run to say what it stood on."""
    return {
        'enzalutamide': {
            'half_distance_um': ENZALUTAMIDE_PENETRATION.half_distance_um,
            'status': ENZALUTAMIDE_PENETRATION.status,
            'source': ENZALUTAMIDE_PENETRATION.source,
        },
        'bolus_sequestered': {
            'half_distance_um': BOLUS_PENETRATION.half_distance_um,
            'range': BOLUS_HALF_DISTANCE_RANGE,
            'status': BOLUS_PENETRATION.status,
            'source': BOLUS_PENETRATION.source,
        },
    }
