"""
Treatment Simulator
===================

Simulate immunotherapy and targeted therapy response using the
personalized digital twin configuration.

Models checkpoint inhibitor effects (PD-1, CTLA-4 blockade),
PARP inhibitors, and combination therapies based on the patient's
genomic profile and immune landscape.
"""

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .twin_config import DigitalTwinConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Response criteria, in the units the criteria are defined in
# ─────────────────────────────────────────────────────────────────────────
#
# `tumor_response_curve` is a relative *volume*: `_simulate_tumor_dynamics`
# integrates a multiplicative growth/kill law on tumour cell number, so the
# state variable is a volume.
#
# RECIST 1.1 (Eisenhauer et al., Eur J Cancer 2009;45:228-247) is defined on
# the sum of longest *diameters* of the target lesions. For a lesion that
# changes approximately isotropically, V is proportional to D**3, so
#
#     D / D_0 = (V / V_0) ** (1/3)
#
# A -30% diameter change is therefore a volume ratio of 0.70**3 = 0.343, and
# a +20% diameter change is a volume ratio of 1.20**3 = 1.728. Comparing the
# volume curve directly against 0.70 and 1.20 -- which this file used to do --
# called an 11.2% diameter shrink a partial response, and put the progression
# threshold at a 6.3% diameter increase. Both constants below are diameter
# ratios, and `_classify_response` applies them to the cube root of a volume.
RECIST_PR_DIAMETER_RATIO = 0.70
"""PR: >= 30% decrease in the sum of longest diameters, referenced to BASELINE."""

RECIST_PD_DIAMETER_RATIO = 1.20
"""PD: >= 20% increase in the sum of longest diameters, referenced to the NADIR.

RECIST 1.1 references progression to the smallest sum recorded on study, not
to baseline, and additionally requires an absolute increase of at least 5 mm.
A relative-volume model carries no absolute sizes, so the 5 mm rule is not
representable here; see `MODEL_CAVEATS`.
"""

DAYS_PER_MONTH = 365.25 / 12.0
"""Average Gregorian month. Published medians are in months; this model steps in days."""


# ─────────────────────────────────────────────────────────────────────────
# Parameter provenance: what is calibrated, and what is not
# ─────────────────────────────────────────────────────────────────────────
#
# Exactly one parameter in this file is identified by a published endpoint.
# Two more are fixed by a published standard. Every other number is
# ILLUSTRATIVE -- plausible, chosen, and fitted to nothing. The model card
# used to assert that the parameters were "derived from published clinical
# trial endpoints", which was not borne out anywhere in this file;
# `PARAMETER_PROVENANCE` is the file's own enumeration of which ones are, and
# `cognisom/mad/model_cards.py` now reads it instead of asserting.

PROFOUND_COHORT_A_CONTROL_RPFS_MONTHS = 3.6
"""PROfound (NCT02987543) cohort A, enzalutamide-or-abiraterone control arm.

Median radiographic PFS by blinded independent central review in men with
BRCA1, BRCA2 or ATM alterations: 3.6 months (3.55 unrounded), against 7.4
months (7.39) on olaparib, HR 0.34 (95% CI 0.25-0.47, p<0.0001).
de Bono et al., N Engl J Med 2020;382:2091-2102.
"""

CALIBRATED_GROWTH_RATE_PER_DAY = math.log(
    RECIST_PD_DIAMETER_RATIO ** 3
) / (PROFOUND_COHORT_A_CONTROL_RPFS_MONTHS * DAYS_PER_MONTH)
"""Untreated exponential growth rate (per day) -- derived, not chosen.

Identification. Of every constant in this file, the baseline growth rate is
the only one that acts alone on the no-effect trajectory, so one published
control-arm endpoint pins it. Require that the median patient on the PROfound
cohort A control arm reaches the RECIST 1.1 progression threshold exactly at
the published median rPFS. With exponential growth V(t) = V_0 exp(g t), and
progression at a 20% diameter increase over a nadir which -- with no
effective therapy -- is the baseline:

    exp(g T) = 1.20 ** 3    =>    g = 3 ln(1.20) / T

T = 3.6 months = 109.58 days gives g = 0.004992 /day. The unrounded BICR
median of 3.55 months gives 0.005062 /day instead -- a 1.4% spread, and the
two bracket the value this file shipped, 0.005. The published abstract's 3.6
is used, and the constant is computed from the citation here so that it
cannot drift away from it silently.

Assumptions a reader should weigh before calling this a fit:
  * The control arm stands in for no therapy. Those men had already
    progressed on a second-generation androgen-receptor pathway inhibitor and
    were re-challenged with one, which is the closest available
    approximation to untreated growth -- not untreated growth.
  * A deterministic single-rate model has one trajectory, so "the median
    patient progresses at the median rPFS" is an identifying assumption, not
    something the published data establishes.
  * PROfound scored progression by RECIST 1.1 in soft tissue and PCWG3 in
    bone. Only the RECIST component is representable here.
  * It is one parameter against one endpoint: exactly determined, with no
    degrees of freedom left over to test the fit. There is no interval.
  * PROfound imaged every 8 weeks (+/- 7 days), so the published median is
    interval-censored on a 56-day grid. The continuous crossing time this
    derivation solves for is a finer quantity than the trial could observe,
    and the 1.4% spread quoted above understates the real uncertainty
    accordingly.
  * THE MODEL DOES NOT REPRODUCE THE ENDPOINT THIS CONSTANT IS FITTED TO.
    Measured: running the shipped `enzalutamide` regimen on 245
    cohort-A-like twins returns ORR 0.788 and 245/245 right-censored PFS at
    180 days, against the published 2% ORR and 3.6-month median rPFS. The
    round trip that closes below closes against a synthetic zero-effectiveness
    arm, not against PROfound's control arm -- so this derivation is
    self-consistent but cannot be checked against the trial it cites. Every
    other constant in this file is declared illustrative for the same reason:
    a calibration that cannot reproduce its own source arm is not evidence
    that the model predicts anything.
"""


@dataclass(frozen=True)
class ParameterProvenance:
    """Where one group of this model's numbers comes from."""

    name: str
    count: int
    status: str
    """'derived' (fitted to a published endpoint), 'standard' (fixed by a
    published criterion) or 'illustrative' (chosen; fitted to nothing)."""
    basis: str


PARAMETER_PROVENANCE: Tuple[ParameterProvenance, ...] = (
    ParameterProvenance(
        "growth_rate_per_day", 1, "derived",
        "3 ln(1.20) / 109.58 d from PROfound (NCT02987543) cohort A control-arm "
        "median rPFS 3.6 months; see CALIBRATED_GROWTH_RATE_PER_DAY.",
    ),
    ParameterProvenance(
        "recist_pr_diameter_ratio, recist_pd_diameter_ratio", 2, "standard",
        "RECIST 1.1, Eisenhauer et al., Eur J Cancer 2009;45:228-247.",
    ),
    ParameterProvenance(
        "effectiveness_baseline", 1, "illustrative",
        "The 0.5 every regimen starts from. Not a response probability and not "
        "anchored to any arm's ORR.",
    ),
    ParameterProvenance(
        "effectiveness_biomarker_deltas", 20, "illustrative",
        "Every += / -= in _calculate_effectiveness: the PD-L1 coefficient, the "
        "exhaustion-window bonus, the TMB term, hot/cold, the Treg coefficient "
        "and priming bonus, the HRD reward and penalty, the AR terms, the six "
        "vaccine terms and the MSI-H bonus.",
    ),
    ParameterProvenance(
        "effectiveness_biomarker_thresholds", 7, "illustrative",
        "The 0.2-0.8 exhaustion window, the TMB divisor 50.0, the vaccine "
        "candidate counts 10/5/3 and the strong-binder count 5.",
    ),
    ParameterProvenance(
        "max_kill_rate_coefficient", 1, "illustrative",
        "effectiveness * 0.025 /day. Sets how fast any regimen can shrink a "
        "tumour and is the single largest determinant of the predicted "
        "response depth. Nothing fits it.",
    ),
    ParameterProvenance(
        "effect_ramp_days", 1, "illustrative",
        "The 21-day linear ramp from treatment onset to full kill rate.",
    ),
    ParameterProvenance(
        "noise_sd_per_day", 1, "illustrative",
        "sigma = 0.002 per day on the growth rate. Not an estimate of "
        "measurement error or of between-visit variability.",
    ),
    ParameterProvenance(
        "volume_floor", 1, "illustrative",
        "Relative volume is floored at 0.01, which is why RECIST CR is not "
        "representable; see MODEL_CAVEATS.",
    ),
    ParameterProvenance(
        "growth_multipliers_pten_ar", 2, "illustrative",
        "x1.3 for PTEN loss and x1.1 for an AR mutation. Directionally "
        "supported by the literature; the magnitudes are chosen.",
    ),
    ParameterProvenance(
        "resistance_ramp", 3, "illustrative",
        "Onset day 90, divisor 500.0, coefficient 0.001. Reaches 1.8e-4 /day at "
        "day 180 against a kill rate of about 1.2e-2 /day -- roughly 1.4% of "
        "it -- so it cannot produce acquired resistance; see MODEL_CAVEATS.",
    ),
    ParameterProvenance(
        "per_drug_constants", 36, "illustrative",
        "9 regimens x exhaustion_reversal, treg_effect, irae_base_risk and "
        "effect_onset_days. No regimen's four constants were fitted to that "
        "drug's published endpoints.",
    ),
    ParameterProvenance(
        "confidence_weights", 7, "illustrative",
        "The data-completeness score in _estimate_confidence. It measures how "
        "much input was present, not how accurate the prediction is.",
    ),
)

DERIVED_PARAMETER_COUNT = sum(
    p.count for p in PARAMETER_PROVENANCE if p.status == "derived")
STANDARD_PARAMETER_COUNT = sum(
    p.count for p in PARAMETER_PROVENANCE if p.status == "standard")
ILLUSTRATIVE_PARAMETER_COUNT = sum(
    p.count for p in PARAMETER_PROVENANCE if p.status == "illustrative")

CALIBRATION_STATEMENT = (
    "The constants in this simulator are illustrative, with two exceptions. "
    f"{DERIVED_PARAMETER_COUNT} parameter is derived from a published "
    "endpoint: the untreated growth rate, 0.00499/day, is the exponential rate "
    "at which the median PROfound (NCT02987543) cohort A control-arm patient "
    "reaches the RECIST 1.1 progression threshold at the published median rPFS "
    f"of 3.6 months. {STANDARD_PARAMETER_COUNT} more are the RECIST 1.1 "
    "response thresholds, which a published criterion fixes rather than a fit. "
    f"The other {ILLUSTRATIVE_PARAMETER_COUNT} constants -- enumerated in "
    "PARAMETER_PROVENANCE, and covering every per-drug parameter, every "
    "biomarker effectiveness delta, the kill-rate coefficient, the noise "
    "amplitude, the resistance ramp and the confidence weights -- are "
    "ILLUSTRATIVE: plausible values that were chosen, fitted to nothing. Read "
    "the response curves as an illustration of the model's structure, not as a "
    "prediction of what a patient will do. One per-drug input is evidence "
    "rather than a choice: when a course stops before the end of the window, "
    "a regimen the entity library covers washes out at its FDA-label "
    "elimination half-life, a combination component by component, each "
    "share weighted by that drug's single-agent effectiveness for the "
    "patient (a structural assumption).\n"
    "Calibration against the six reference trials is DEFERRED, and three "
    "things block it. (1) Median rPFS lies outside the model's output range: "
    "the dynamics carry no acquired-resistance mechanism able to reverse net "
    "growth after a response, so PFS is either an early progression day or is "
    "right-censored at the end of the simulated window, and PROfound's 7.4 "
    "months on olaparib is unreachable. (2) ORR is a cohort statistic while "
    "this model is deterministic per (patient, regimen), so an ORR needs a "
    "cohort; the SU2C/PCF mCRPC 2019 set is not pinned locally and carries no "
    "per-drug response outcomes, so a predicted ORR could only be matched "
    "against a published ORR from a different cohort. (3) Only 3 of the 9 "
    "regimens have a same-drug, same-population published endpoint: "
    "pembrolizumab (KEYNOTE-199), and olaparib and enzalutamide (PROfound "
    "cohort A). KEYNOTE-942 is adjuvant resected melanoma with no measurable "
    "disease, so the two vaccine regimens -- which this platform treats as its "
    "most interesting output -- have no usable endpoint at all."
)
"""Plain statement of what is fitted here. Read verbatim by the model card
and by the dashboard pages that show a response table, so the claim and the
code cannot drift apart."""


CR_NOT_REPRESENTABLE = (
    "RECIST complete response is never reported. The dynamics floor relative "
    "volume at 0.01, a 78% diameter reduction, so disappearance of all target "
    "lesions cannot be represented and a 'CR' would be an extrapolation past "
    "the floor."
)

PD_MEANS_NEVER_RESPONDED = (
    "PD here means the tumour progressed without ever reaching a partial "
    "response. Response followed by progression is NOT representable: the "
    "resistance term reaches 1.8e-4 /day by day 180 against a kill rate near "
    "1.2e-2 /day, so an effective regimen never re-crosses its own nadir "
    "inside any plausible horizon."
)

NO_ABSOLUTE_SIZES = (
    "RECIST 1.1 also requires an absolute increase of at least 5 mm for "
    "progression, and measures nodal lesions by short axis. A relative-volume "
    "model carries neither, so both rules are omitted."
)

PD_NEEDS_A_VERY_LOW_EFFECTIVENESS = (
    "Progressive disease requires a predicted effectiveness at or below "
    "0.096 on a 180-day run -- 0.163 at one year, 0.201 at two -- because "
    "the kill term has to stay under the growth rate for the whole window. "
    "Six of the nine shipped regimens have an effectiveness floor above "
    "that, so they cannot return PD on a 180-day run however poor the "
    "patient's biomarkers are. PD is reachable, but it reflects the "
    "effectiveness model's floor rather than a clinical progression rate."
)

MODEL_CAVEATS: Tuple[str, ...] = (
    CR_NOT_REPRESENTABLE,
    PD_MEANS_NEVER_RESPONDED,
    PD_NEEDS_A_VERY_LOW_EFFECTIVENESS,
    NO_ABSOLUTE_SIZES,
)


# ─────────────────────────────────────────────────────────────────────────
# Mechanism dispatch
# ─────────────────────────────────────────────────────────────────────────
#
# `_calculate_effectiveness` used to dispatch on substrings of the display
# class string, which is not a classifier. `"AR" in "PARP inhibitor"` is True,
# because "PARP" contains "AR", so both PARP arms took the androgen-receptor
# branch: olaparib on a twin with no DNA-repair defect scored
# 0.5 - 0.3 (no HRD) + 0.2 (wild-type AR responds well) = 0.4, a 38% volume
# reduction reported as a partial response -- a PARP inhibitor predicted
# active in a patient with no synthetic-lethal target. In the other direction
# `pembro_ipi_combo` has class "combination checkpoint", which matches neither
# "PD-1" nor "CTLA-4", so the arm the recommender escalates to for cold
# tumours received no PD-1 and no CTLA-4 term and took exactly two values ever
# (0.5, or 0.7 with MSI-H), invariant to every biomarker.
#
# Mechanisms are now declared per regimen, and both this file and
# `cognisom/mad/explainability.py` dispatch on the same declaration -- the
# explainer had its own hand-written guard (`"AR" in cls and "PARP" not in
# cls`), so the audit decomposition disagreed with the score it claimed to
# decompose by 0.20 on both PARP arms.
MECH_PD1 = "pd1"
MECH_CTLA4 = "ctla4"
MECH_PARP = "parp"
MECH_AR_ANTAGONIST = "ar_antagonist"
MECH_VACCINE = "vaccine"

KNOWN_MECHANISMS: frozenset = frozenset({
    MECH_PD1, MECH_CTLA4, MECH_PARP, MECH_AR_ANTAGONIST, MECH_VACCINE,
})

#: Exact treatment-class string -> mechanism set, for profiles that carry no
#: explicit `mechanisms` key. Entity-library rows are built by
#: `parameter_resolver.get_treatment_profile`, which emits exactly these
#: labels. Lookup is exact, never a substring test, and the classes this
#: effectiveness model has no terms for are listed with an empty set so that
#: "no mechanism" stays distinguishable from "class not recognised".
CLASS_MECHANISMS: Dict[str, frozenset] = {
    "anti-PD-1": frozenset({MECH_PD1}),
    "anti-PD-L1": frozenset({MECH_PD1}),
    "anti-CTLA-4": frozenset({MECH_CTLA4}),
    "combination checkpoint": frozenset({MECH_PD1, MECH_CTLA4}),
    "PARP inhibitor": frozenset({MECH_PARP}),
    "AR antagonist": frozenset({MECH_AR_ANTAGONIST}),
    "PARP + checkpoint": frozenset({MECH_PARP, MECH_PD1}),
    "neoantigen vaccine": frozenset({MECH_VACCINE}),
    "vaccine + checkpoint": frozenset({MECH_VACCINE, MECH_PD1}),
    # Recognised, and this effectiveness model has no term for them.
    "anti-LAG-3": frozenset(),
    "anti-TIM-3": frozenset(),
    "checkpoint inhibitor": frozenset(),
    "chemotherapy": frozenset(),
    "radiotherapy": frozenset(),
    "PI3K/AKT inhibitor": frozenset(),
    "JAK inhibitor": frozenset(),
    "anti-TNF": frozenset(),
    "anti-CD20": frozenset(),
    "cytokine therapy": frozenset(),
    "unknown": frozenset(),
}

#: Mechanisms by international nonproprietary name, consulted before the
#: class string. The entity library's own class labels are not sufficient:
#: `SimulationParameterResolver.get_treatment_profile` passes an unmapped
#: `drug_class` straight through, so pembrolizumab arrives as class
#: "immunotherapy", and `_infer_checkpoint_class` falls back to the bare
#: "checkpoint inhibitor" for nivolumab, ipilimumab, atezolizumab and
#: relatlimab. Neither string can be assigned a mechanism safely -- an
#: anti-PD-1, an anti-CTLA-4 and an anti-LAG-3 all carry the second one --
#: so the drug itself has to be named. Measured before this table existed:
#: `TreatmentSimulator(store=EntityStore())` scored pembrolizumab,
#: nivolumab and ipilimumab at exactly the 0.5 baseline for every patient,
#: which is to say the documented-as-preferred parameter source was the one
#: that ignored the twin.
#:
#: Lookup is on the lowercased first whitespace-separated token of the
#: profile's display name, which is the INN for every row the library
#: carries. Drugs whose mechanism this effectiveness model has no term for
#: are listed with an empty set, so "recognised, not modelled" stays
#: distinguishable from "not recognised".
DRUG_MECHANISMS: Dict[str, frozenset] = {
    "pembrolizumab": frozenset({MECH_PD1}),
    "nivolumab": frozenset({MECH_PD1}),
    "atezolizumab": frozenset({MECH_PD1}),
    "ipilimumab": frozenset({MECH_CTLA4}),
    "olaparib": frozenset({MECH_PARP}),
    "rucaparib": frozenset({MECH_PARP}),
    "talazoparib": frozenset({MECH_PARP}),
    "enzalutamide": frozenset({MECH_AR_ANTAGONIST}),
    "apalutamide": frozenset({MECH_AR_ANTAGONIST}),
    "darolutamide": frozenset({MECH_AR_ANTAGONIST}),
    "abiraterone": frozenset({MECH_AR_ANTAGONIST}),
    # Recognised, and this effectiveness model has no term for them. An
    # anti-LAG-3 is not an anti-PD-1 and must not inherit its terms.
    "relatlimab": frozenset(),
    "docetaxel": frozenset(),
    "cabazitaxel": frozenset(),
    "lutetium-177": frozenset(),
    "ipatasertib": frozenset(),
    "ruxolitinib": frozenset(),
    "tofacitinib": frozenset(),
    "rituximab": frozenset(),
}


def _inn_of(profile: Dict) -> str:
    """The lowercased INN token of a profile's display name, or ''."""
    name = str(profile.get("name", "")).strip().lower()
    if not name:
        return ""
    return name.split()[0].strip("(),.-")


_UNMAPPED_CLASSES_REPORTED: set = set()

#: Profiles already reported as stopping with no library half-life.
_NO_HALF_LIFE_REPORTED: set = set()


def elimination_half_life_days(profile: Dict) -> Optional[float]:
    """The profile's library half-life in days, or None.

    The resolver puts a half-life on a profile only when it is evidence --
    in practice the FDA label's, cited by SPL set id -- and in the spec's
    unit, hours; this is the one conversion, hours to the loop's days.
    Hardcoded profiles carry no ``pk`` block and return None.
    """
    hours = (profile.get("pk") or {}).get("half_life_hours")
    return None if hours is None else float(hours) / 24.0


def regimen_exposure(day: int, last_dose_day: int,
                     parts: List[Tuple[float, Optional[float]]]) -> float:
    """Relative exposure of a regimen of one or more drugs on ``day``.

    ``parts`` is (share, half-life in days) per component, shares summing to
    1. Exactly 1.0 while dosed, whatever the shares, so a regimen given
    throughout is unchanged to the bit. After the last dose each share
    washes out at its own component's half-life (drug_exposure) and the
    shares add; a single agent is one part of share 1.
    """
    if day <= last_dose_day:
        return 1.0
    return sum(share * drug_exposure(day, last_dose_day, half_life)
               for share, half_life in parts)


def drug_exposure(day: int, last_dose_day: int,
                  half_life_days: Optional[float]) -> float:
    """Relative drug exposure on ``day``: 1 while dosed, washing out after.

    Washout is first-order at the label's terminal half-life, which governs
    the fall after a last dose whatever the dosing pattern was. Exposure
    while dosed is held at 1, steady state, deliberately: the climb to
    steady state depends on dose, interval and route, none of which the
    library holds. Modelling it as a continuous infusion would put
    pembrolizumab's first-week exposure at a few percent, when a 200 mg
    bolus is at about two thirds of the steady-state average from the first
    day, and it would stack a PK delay on the illustrative effect onset that
    already stands in for one. With no half-life the effect stops at the
    last dose, which is what this model did before it had one.

    Consumes no random draws, so the noise sequence -- and every curve
    with no course end -- is unchanged.
    """
    if day <= last_dose_day:
        return 1.0
    if half_life_days is None:
        return 0.0
    return math.exp(-math.log(2.0) * (day - last_dose_day) / half_life_days)


def treatment_mechanisms(profile: Dict) -> frozenset:
    """The mechanism set this regimen's effectiveness terms dispatch on.

    Resolution order, every step an exact lookup and none of them a
    substring test: the profile's own `mechanisms` declaration, then the
    INN (the first token of the display name, which is what the entity
    library carries), then the full display class string. Returns the empty
    set -- with one warning -- when none of the three resolves. An empty set
    means the effectiveness score carries no mechanism-specific term and so
    is not patient-specific; `simulate` says so in the result's caveats
    rather than letting a baseline pass for a prediction.
    """
    declared = profile.get("mechanisms")
    if declared is not None:
        unknown = frozenset(declared) - KNOWN_MECHANISMS
        if unknown:
            raise ValueError(
                f"Treatment profile {profile.get('name', '?')!r} declares "
                f"unknown mechanism(s) {sorted(unknown)}; known: "
                f"{sorted(KNOWN_MECHANISMS)}"
            )
        return frozenset(declared)

    inn = _inn_of(profile)
    if inn in DRUG_MECHANISMS:
        return DRUG_MECHANISMS[inn]

    treatment_class = profile.get("class", "")
    if treatment_class in CLASS_MECHANISMS:
        return CLASS_MECHANISMS[treatment_class]

    if treatment_class not in _UNMAPPED_CLASSES_REPORTED:
        _UNMAPPED_CLASSES_REPORTED.add(treatment_class)
        logger.warning(
            "Neither the drug name %r nor the treatment class %r maps to a "
            "mechanism; no mechanism-specific effectiveness term will be "
            "applied and the score will not be patient-specific",
            profile.get("name", "?"), treatment_class,
        )
    return frozenset()


def format_progression_free_survival(days: int, censored: bool) -> str:
    """Render a PFS value, refusing to print a censored one as months.

    A right-censored PFS is the length of the observation window, not a
    survival time. `agents.py` used to print "PFS: 180 days (6.0 months)"
    for every arm of a 180-day run, which is administrative end of
    follow-up presented to a clinician as a six-month PFS.
    """
    if censored:
        return (f"{days} days (right-censored: no RECIST progression within "
                f"the {days}-day simulated window)")
    return f"{days} days ({days / DAYS_PER_MONTH:.1f} months)"


def progression_free_months(days: int, censored: bool) -> float:
    """PFS in months, or ValueError when the value is right-censored.

    The single days-to-months conversion in this package. A censored PFS
    is the length of follow-up; converting it to months produces a figure
    that reads as a clinical endpoint and is not one. `agents.py` printed
    exactly that -- "PFS: 180 days (6.0 months)" -- for every arm of every
    180-day run, so the failure mode is not hypothetical. Refusing here
    means a future caller gets a traceback instead of a plausible number.

    Raises:
        ValueError: `censored` is True.
    """
    if censored:
        raise ValueError(
            f"a right-censored progression-free interval ({days} days) has "
            "no median in months: it is the length of the observation "
            "window, not a survival time. Report the pair (days, censored) "
            "or use format_progression_free_survival()."
        )
    return days / DAYS_PER_MONTH


def _patient_seed_key(twin) -> str:
    """A stable per-patient key for the noise seed.

    The seed used to digest the drug name alone, so the stochastic component
    was byte-identical for every patient: two twins with different IDs and
    different TMB drew the same five noise values. Seeding from patient and
    regimen keeps a report reproducible while letting it differ between
    patients.

    With no PatientProfile attached, the model's entire notion of "this
    patient" is the twin's field values, so those are digested instead. Two
    twins that differ in any modelled field get different noise; two
    identical twins get the same noise, which is what reproducibility means
    here.
    """
    patient = getattr(twin, "patient", None)
    patient_id = getattr(patient, "patient_id", None) if patient is not None else None
    if patient_id:
        return f"patient_id:{patient_id}"
    fields = tuple(
        repr(getattr(twin, name, None)) for name in (
            "tumor_mutational_burden", "microsatellite_instability",
            "has_dna_repair_defect", "has_ar_mutation", "has_pten_loss",
            "immune_score", "t_cell_exhaustion_fraction", "mean_exhaustion",
            "treg_fraction", "immune_cell_fraction", "pd_l1_expression",
            "mhc1_downregulation", "vaccine_candidate_count",
            "strong_binder_count",
        )
    )
    return "twin:" + "|".join(fields)


@dataclass
class TreatmentResult:
    """Result of a simulated treatment."""
    treatment_name: str
    treatment_class: str
    duration_days: int

    # Tumor response
    tumor_response_curve: List[float]
    #: Relative tumour VOLUME over time, 1.0 = baseline. Not a RECIST sum of
    #: diameters: see RECIST_PR_DIAMETER_RATIO for the conversion the
    #: classifier applies.

    best_response: float
    #: Nadir relative volume. Cannot exceed 1.0, because the curve starts at
    #: 1.0 -- so this field cannot express progression, and never could.
    #: Progression is reported by `response_category` and by
    #: `progression_free_days` / `progression_free_censored`.

    time_to_best_response_days: int

    progression_free_days: int
    #: Day of the first RECIST 1.1 progression event -- a 20% diameter
    #: increase over the running nadir -- or `duration_days` when none
    #: occurred, in which case `progression_free_censored` is True. Read the
    #: pair together, via `progression_free_survival`; a censored value is
    #: the length of the observation window, not a survival time.

    progression_free_censored: bool
    #: True when no progression event was observed inside the simulated
    #: window. Required, not defaulted, so that a caller constructing a
    #: result cannot silently omit the censoring status of its PFS.

    # Immune effects
    t_cell_reactivation_fraction: float  # Fraction of exhausted T-cells reactivated
    final_exhaustion: Optional[float]  # Post-treatment exhaustion, None if unmeasured
    immune_related_adverse_events: float  # 0-1 risk score

    # Classification
    response_category: str
    #: RECIST 1.1 best overall response, on diameter ratios: "PR", "SD" or
    #: "PD". "CR" is never returned -- see CR_NOT_REPRESENTABLE -- and "PD"
    #: means progression without ever reaching a PR, see
    #: PD_MEANS_NEVER_RESPONDED.

    confidence: float
    #: Input-completeness score, NOT a probability that the prediction is
    #: right. Nothing in this model has been validated against observed
    #: patient outcomes.

    # Details
    mechanism: str
    rationale: str

    #: Twin fields this prediction wanted but could not read, because no
    #: cell-state classifier ran. Every effectiveness term that depended on
    #: one was skipped rather than defaulted, so a reader can tell which
    #: inputs the number actually rests on. Empty means all were measured.
    unmeasured_inputs: List[str] = field(default_factory=list)

    #: What this prediction structurally cannot express, carried on the
    #: result so that a reader meets the limits next to the number rather
    #: than having to find the model card. Always non-empty: see
    #: MODEL_CAVEATS.
    caveats: List[str] = field(default_factory=list)

    #: Day of the last dose, or None when the drug was given throughout.
    last_dose_day: Optional[int] = None
    #: The label half-life the washout used, in days, or None.
    elimination_half_life_days: Optional[float] = None
    #: How exposure was modelled, in words, for display next to the curve.
    exposure_model: str = ""
    #: Where each pharmacokinetic parameter came from: a grade and citation
    #: ("measured:FDA-SPL:<set id>") or why the library gave none. For a
    #: combination, keyed "<component>: <parameter>".
    pk_sources: Dict[str, str] = field(default_factory=dict)
    #: The drugs exposure is made of: [{"name", "share", "half_life_days"}].
    #: One entry of share 1 for a single agent.
    exposure_components: List[Dict] = field(default_factory=list)

    @property
    def inputs_fully_measured(self) -> bool:
        """False when any immune input was absent rather than measured."""
        return not self.unmeasured_inputs

    @property
    def progression_free_survival(self) -> Tuple[int, bool]:
        """(days, censored). The only honest way to read this quantity."""
        return (self.progression_free_days, self.progression_free_censored)

    @property
    def best_response_diameter_ratio(self) -> float:
        """Nadir as a RECIST-comparable diameter ratio, (V/V0) ** (1/3)."""
        return self.best_response ** (1.0 / 3.0)

    @property
    def progression_free_months(self) -> float:
        """PFS in months, raising when the value is right-censored."""
        return progression_free_months(
            self.progression_free_days, self.progression_free_censored,
        )

    def format_progression_free_survival(self) -> str:
        """PFS as text, never printing a censored value as a months figure."""
        return format_progression_free_survival(
            self.progression_free_days, self.progression_free_censored,
        )

    def summary(self) -> Dict:
        diameter_reduction = (1 - self.best_response_diameter_ratio) * 100
        return {
            "treatment": self.treatment_name,
            "class": self.treatment_class,
            "response": self.response_category,
            # Both units, labelled. The volume figure sitting unlabelled next
            # to a RECIST category read as a RECIST diameter change, which it
            # is not: a 38% volume reduction is a 15% diameter reduction.
            "best_response_volume_pct": f"{(1 - self.best_response) * 100:.0f}% volume reduction",
            "best_response_diameter_pct": f"{diameter_reduction:.0f}% diameter reduction (RECIST)",
            "pfs_days": self.progression_free_days,
            "pfs_censored": self.progression_free_censored,
            "pfs": self.format_progression_free_survival(),
            "confidence": round(self.confidence, 2),
            "irae_risk": round(self.immune_related_adverse_events, 2),
            "unmeasured_inputs": list(self.unmeasured_inputs),
            "caveats": list(self.caveats),
            "last_dose_day": self.last_dose_day,
            "exposure_model": self.exposure_model,
            "pk_sources": dict(self.pk_sources),
        }


# Immunotherapy drug profiles
TREATMENT_PROFILES = {
    "pembrolizumab": {
        "name": "Pembrolizumab (Keytruda)",
        "class": "anti-PD-1",
        "mechanisms": frozenset({MECH_PD1}),
        "target": "PD-1",
        "mechanism": "Blocks PD-1/PD-L1 interaction, reactivating exhausted T-cells",
        "exhaustion_reversal": 0.55,  # Fraction of exhausted T-cells potentially reactivated
        "treg_effect": 0.0,  # No direct Treg effect
        "irae_base_risk": 0.15,
        "effect_onset_days": 14,
        "requires_tmb_high": False,
        "requires_msi_h": False,
        "best_for": ["hot", "suppressed"],
    },
    "nivolumab": {
        "name": "Nivolumab (Opdivo)",
        "class": "anti-PD-1",
        "mechanisms": frozenset({MECH_PD1}),
        "target": "PD-1",
        "mechanism": "Blocks PD-1/PD-L1 interaction, reactivating exhausted T-cells",
        "exhaustion_reversal": 0.50,
        "treg_effect": 0.0,
        "irae_base_risk": 0.12,
        "effect_onset_days": 14,
        "requires_tmb_high": False,
        "requires_msi_h": False,
        "best_for": ["hot", "suppressed"],
    },
    "ipilimumab": {
        "name": "Ipilimumab (Yervoy)",
        "class": "anti-CTLA-4",
        "mechanisms": frozenset({MECH_CTLA4}),
        "target": "CTLA-4",
        "mechanism": "Depletes Tregs and enhances T-cell priming in lymph nodes",
        "exhaustion_reversal": 0.20,
        "treg_effect": -0.40,  # Depletes 40% of Tregs
        "irae_base_risk": 0.35,  # Higher toxicity
        "effect_onset_days": 21,
        "requires_tmb_high": False,
        "requires_msi_h": False,
        "best_for": ["suppressed", "cold"],
    },
    "pembro_ipi_combo": {
        "name": "Pembrolizumab + Ipilimumab",
        # Single-agent regimens this one is made of, as keys into the
        # active profiles; see TreatmentSimulator._exposure_parts.
        "components": ("pembrolizumab", "ipilimumab"),
        "class": "combination checkpoint",
        "mechanisms": frozenset({MECH_PD1, MECH_CTLA4}),
        "target": "PD-1 + CTLA-4",
        "mechanism": "Dual checkpoint blockade: reactivate T-cells + deplete Tregs",
        "exhaustion_reversal": 0.70,
        "treg_effect": -0.35,
        "irae_base_risk": 0.45,
        "effect_onset_days": 14,
        "requires_tmb_high": False,
        "requires_msi_h": False,
        "best_for": ["suppressed", "cold", "excluded"],
    },
    "olaparib": {
        "name": "Olaparib (Lynparza)",
        "class": "PARP inhibitor",
        "mechanisms": frozenset({MECH_PARP}),
        "target": "PARP1/2",
        "mechanism": "Synthetic lethality in DNA repair-deficient cells (BRCA1/2, ATM)",
        "exhaustion_reversal": 0.0,
        "treg_effect": 0.0,
        "irae_base_risk": 0.05,
        "effect_onset_days": 28,
        "requires_dna_repair_defect": True,
        "best_for": ["any"],
    },
    "enzalutamide": {
        "name": "Enzalutamide (Xtandi)",
        "class": "AR antagonist",
        "mechanisms": frozenset({MECH_AR_ANTAGONIST}),
        "target": "AR",
        "mechanism": "Competitively inhibits androgen receptor signaling",
        "exhaustion_reversal": 0.0,
        "treg_effect": 0.0,
        "irae_base_risk": 0.05,
        "effect_onset_days": 30,
        "requires_ar_sensitivity": True,
        "best_for": ["any"],
    },
    "olaparib_pembro_combo": {
        "name": "Olaparib + Pembrolizumab",
        # Single-agent regimens this one is made of, as keys into the
        # active profiles; see TreatmentSimulator._exposure_parts.
        "components": ("olaparib", "pembrolizumab"),
        "class": "PARP + checkpoint",
        "mechanisms": frozenset({MECH_PARP, MECH_PD1}),
        "target": "PARP + PD-1",
        "mechanism": "PARP-induced DNA damage increases neoantigens + checkpoint blockade",
        "exhaustion_reversal": 0.50,
        "treg_effect": 0.0,
        "irae_base_risk": 0.25,
        "effect_onset_days": 21,
        "requires_dna_repair_defect": True,
        "best_for": ["any_with_hrd"],
    },
    "neoantigen_vaccine": {
        "name": "Personalized mRNA Neoantigen Vaccine",
        "class": "neoantigen vaccine",
        "mechanisms": frozenset({MECH_VACCINE}),
        "target": "Tumor neoantigens",
        "mechanism": (
            "mRNA vaccine encoding patient-specific tumor neoantigens. "
            "LNP-encapsulated mRNA is translated by APCs, processed, and "
            "presented on MHC-I to prime neoantigen-specific CD8+ T-cells"
        ),
        "exhaustion_reversal": 0.10,  # Modest direct effect on exhaustion
        "treg_effect": 0.0,
        "irae_base_risk": 0.08,  # Low toxicity profile
        "effect_onset_days": 28,  # 4 weeks for T-cell priming
        "requires_neoantigens": True,
        "best_for": ["hot", "suppressed", "excluded"],
    },
    "neoantigen_vaccine_pembro": {
        "name": "Neoantigen Vaccine + Pembrolizumab",
        # Single-agent regimens this one is made of, as keys into the
        # active profiles; see TreatmentSimulator._exposure_parts.
        "components": ("neoantigen_vaccine", "pembrolizumab"),
        "class": "vaccine + checkpoint",
        "mechanisms": frozenset({MECH_VACCINE, MECH_PD1}),
        "target": "Neoantigens + PD-1",
        "mechanism": (
            "Personalized mRNA neoantigen vaccine primes tumor-specific T-cells, "
            "while pembrolizumab removes PD-1/PD-L1 checkpoint brake — "
            "synergistic anti-tumor immunity (mRNA-4157/V940 paradigm)"
        ),
        "exhaustion_reversal": 0.60,
        "treg_effect": 0.0,
        "irae_base_risk": 0.20,
        "effect_onset_days": 21,
        "requires_neoantigens": True,
        "best_for": ["hot", "suppressed", "excluded", "cold"],
    },
}


class TreatmentSimulator:
    """Simulate treatment response using the digital twin.

    Models tumor dynamics under different therapies based on the
    patient's genomic profile and immune microenvironment.

    Supports two parameter sources:
    1. Entity library (preferred): Drug entities with physics_params
       drive the simulation. Add a drug → automatically available.
    2. Hardcoded fallback: TREATMENT_PROFILES dict for backward compatibility.

    Example:
        twin = DigitalTwinConfig.from_profile_and_classification(profile, classification)

        # Entity-driven (preferred):
        from cognisom.library.store import EntityStore
        sim = TreatmentSimulator(store=EntityStore())

        # Or legacy (hardcoded):
        sim = TreatmentSimulator()

        result = sim.simulate("pembrolizumab", twin, duration_days=180)
    """

    def __init__(self, store=None):
        """Initialize with optional EntityStore for entity-driven parameters.

        Args:
            store: EntityStore instance. If provided, drug treatment profiles
                   are loaded from the entity library instead of TREATMENT_PROFILES.
        """
        self._entity_profiles = None
        if store is not None:
            try:
                from .parameter_resolver import SimulationParameterResolver
                resolver = SimulationParameterResolver(store)
                self._entity_profiles = resolver.get_all_treatment_profiles()
                if self._entity_profiles:
                    logger.info(
                        "Loaded %d treatment profiles from entity library",
                        len(self._entity_profiles),
                    )
            except Exception as e:
                logger.warning("Entity profile loading failed, using hardcoded: %s", e)

    @property
    def _profiles(self) -> dict:
        """Get active treatment profiles (entity-driven or hardcoded)."""
        if self._entity_profiles:
            # Merge: entity profiles + hardcoded (entity takes precedence)
            merged = dict(TREATMENT_PROFILES)
            merged.update(self._entity_profiles)
            return merged
        return TREATMENT_PROFILES

    def get_available_treatments(self) -> list:
        """List all available treatment keys."""
        return sorted(self._profiles.keys())

    def simulate(self, treatment_key: str,
                 twin: DigitalTwinConfig,
                 duration_days: int = 180,
                 last_dose_day: Optional[int] = None) -> TreatmentResult:
        """Simulate a treatment and predict response.

        Args:
            treatment_key: Key into treatment profiles (entity or hardcoded).
            twin: Personalized DigitalTwinConfig.
            duration_days: Simulation duration in days.
            last_dose_day: Day of the last dose. None (the default) doses
                throughout the window. After it the effect washes out at the
                drug's label half-life when the library has one.

        Returns:
            TreatmentResult with tumor response curve and predictions.
        """
        profile = self._profiles.get(treatment_key)
        if not profile:
            raise ValueError(f"Unknown treatment: {treatment_key}. "
                           f"Available: {list(self._profiles.keys())}")

        # Calculate base effectiveness
        effectiveness = self._calculate_effectiveness(profile, twin)

        if last_dose_day is not None:
            if last_dose_day < 0:
                raise ValueError(f"last_dose_day must be >= 0, got {last_dose_day}")
            if last_dose_day >= duration_days:
                last_dose_day = None      # dosed through the window

        # Simulate tumor dynamics
        response_curve = self._simulate_tumor_dynamics(
            effectiveness, profile, twin, duration_days,
            last_dose_day=last_dose_day,
        )
        parts = self._exposure_parts(profile, twin)
        half_life = (parts[0]["half_life_days"] if len(parts) == 1 else None)
        exposure_model = self._describe_exposure(parts, duration_days,
                                                 last_dose_day)
        if len(parts) == 1:
            pk_sources = dict(parts[0]["pk_sources"])
        else:
            pk_sources = {f"{part['name']}: {name}": note
                          for part in parts
                          for name, note in part["pk_sources"].items()}

        # Find best response and PFS
        best_response = min(response_curve)
        best_day = response_curve.index(best_response)

        # PFS. The event is RECIST 1.1 progression: a 20% increase in the sum
        # of longest diameters over the smallest sum recorded on study so
        # far. The previous test -- first day the VOLUME exceeds baseline,
        # scanned from the nadir -- was wrong twice over. It compared a
        # volume against a diameter threshold, and it referenced baseline
        # rather than the running nadir, so on a 180-day run it returned
        # either day 1 (the growth of the first day before onset, read as
        # progression) or exactly 180, and nothing in between.
        progression_day = self._first_progression_day(response_curve)
        if progression_day is None:
            # Right-censored: the window ended with no progression event.
            # This is the length of follow-up, not a survival time, and the
            # flag travels with it so that no consumer can print it as a
            # clinical months figure.
            pfs = duration_days
            pfs_censored = True
        else:
            pfs = progression_day
            pfs_censored = False

        # Classify response (RECIST 1.1 best overall response)
        response_category = self._classify_response(
            best_response, progression_day is not None,
        )

        # T-cell reactivation
        reversal = profile.get("exhaustion_reversal", 0) * effectiveness
        unmeasured: List[str] = []
        if twin.mean_exhaustion is None:
            # With no measured pre-treatment exhaustion there is no
            # post-treatment exhaustion to report.
            final_exhaustion = None
            unmeasured.append("mean_exhaustion")
        else:
            final_exhaustion = max(0, twin.mean_exhaustion * (1 - reversal))

        # irAE risk. The patient-specific term needs a measured immune
        # fraction. Without one, report the drug's base risk and name the
        # missing input, rather than coalescing the absence to zero and
        # presenting a population number as patient-specific.
        irae_base = profile.get("irae_base_risk", 0.1)
        if twin.immune_cell_fraction is None:
            irae = irae_base
            unmeasured.append("immune_cell_fraction")
        else:
            irae = irae_base * (1 + twin.immune_cell_fraction)
        unmeasured.extend(self._unmeasured_for_mechanism(profile, twin))

        caveats = list(MODEL_CAVEATS)
        if not treatment_mechanisms(profile):
            # No mechanism-specific term fired, so the effectiveness score is
            # the bare baseline and is identical for every patient. Saying so
            # is the difference between a model and a constant.
            caveats.insert(0, (
                f"No mechanism-specific effectiveness term exists for class "
                f"{profile.get('class', '')!r}: this score is the model "
                f"baseline and does not depend on the patient."
            ))

        return TreatmentResult(
            treatment_name=profile["name"],
            treatment_class=profile["class"],
            duration_days=duration_days,
            tumor_response_curve=response_curve,
            best_response=best_response,
            time_to_best_response_days=best_day,
            progression_free_days=pfs,
            progression_free_censored=pfs_censored,
            t_cell_reactivation_fraction=reversal,
            final_exhaustion=final_exhaustion,
            immune_related_adverse_events=min(1.0, irae),
            response_category=response_category,
            confidence=self._estimate_confidence(profile, twin),
            mechanism=profile["mechanism"],
            rationale=self._generate_rationale(profile, twin, effectiveness),
            unmeasured_inputs=sorted(set(unmeasured)),
            caveats=caveats,
            last_dose_day=last_dose_day,
            elimination_half_life_days=half_life,
            exposure_model=exposure_model,
            pk_sources=pk_sources,
            exposure_components=[
                {"name": part["name"], "share": part["share"],
                 "half_life_days": part["half_life_days"]}
                for part in parts
            ],
        )

    @staticmethod
    def _unmeasured_for_mechanism(profile: Dict, twin) -> List[str]:
        """Twin fields the effectiveness model for this regimen wanted.

        Dispatches on the same mechanism set as `_calculate_effectiveness`,
        so the list names inputs that genuinely would have moved this score.
        The class-substring version missed `treg_fraction` for
        `pembro_ipi_combo`, whose class string contains neither "PD-1" nor
        "CTLA-4", and so reported a dual-checkpoint prediction built on an
        unmeasured Treg fraction as fully measured.
        """
        mechanisms = treatment_mechanisms(profile)
        missing: List[str] = []
        if MECH_PD1 in mechanisms:
            if twin.pd_l1_expression is None:
                missing.append("pd_l1_expression")
            if twin.t_cell_exhaustion_fraction is None:
                missing.append("t_cell_exhaustion_fraction")
        if MECH_CTLA4 in mechanisms and twin.treg_fraction is None:
            missing.append("treg_fraction")
        if MECH_VACCINE in mechanisms and twin.mhc1_downregulation is None:
            missing.append("mhc1_downregulation")
        return missing

    def _exposure_parts(self, profile: Dict, twin) -> List[Dict]:
        """The drugs a regimen's exposure is made of, each with its share.

        A single agent is one part of share 1. A combination lists its
        components (keys into the active profiles), and after the last dose
        each component's share of the effect washes out at that component's
        own label half-life -- pembrolizumab's for weeks, olaparib's within
        a day. The shares are the components' single-agent effectiveness for
        this patient, normalised: the effectiveness model adds a term per
        mechanism, so this is its own split of the combined effect, and it
        is patient-specific (olaparib's share is large with a DNA-repair
        defect and small without one). It is a structural assumption, not
        a fitted one, and it introduces no constant.
        """
        def part(component: Dict, share: float) -> Dict:
            return {
                "name": str(component.get("name", "")),
                "share": share,
                "half_life_days": elimination_half_life_days(component),
                "pk_sources": dict(
                    (component.get("pk") or {}).get("_sources") or {}),
                "vaccine": MECH_VACCINE in treatment_mechanisms(component),
            }

        keys = profile.get("components") or ()
        if not keys:
            return [part(profile, 1.0)]
        components = []
        for key in keys:
            component = self._profiles.get(key)
            if component is None:
                raise ValueError(
                    f"{profile.get('name')!r} lists component {key!r}, "
                    "which is not a known regimen")
            components.append(component)
        strengths = [self._calculate_effectiveness(c, twin) for c in components]
        total = sum(strengths)
        shares = ([x / total for x in strengths] if total > 0
                  else [1.0 / len(components)] * len(components))
        return [part(c, share) for c, share in zip(components, shares)]

    @staticmethod
    def _describe_exposure(parts: List[Dict], duration_days: int,
                           last_dose_day: Optional[int]) -> str:
        """How exposure was modelled, in words, for display with the curve."""
        if last_dose_day is None:
            return (f"Dosed throughout the {duration_days}-day window; "
                    "exposure held at steady state.")

        def fate(part: Dict) -> str:
            if part["vaccine"]:
                return ("a vaccine's effect is T-cell memory, which outlasts "
                        "the vaccine and is not modelled, so it stops at the "
                        "last dose (a lower bound)")
            if part["half_life_days"] is None:
                return "no library half-life, so it stops at the last dose"
            return f"label half-life {part['half_life_days']:.3g} days"

        if len(parts) == 1:
            if parts[0]["half_life_days"] is None:
                return (f"Last dose on day {last_dose_day}. "
                        + fate(parts[0])[0].upper() + fate(parts[0])[1:] + ".")
            return (f"Last dose on day {last_dose_day}, then first-order "
                    f"washout at the label half-life of "
                    f"{parts[0]['half_life_days']:.3g} days; exposure held at "
                    "steady state while dosed. Active metabolites are not "
                    "modelled.")
        each = "; ".join(f"{p['name']} (share {p['share']:.2f}): {fate(p)}"
                         for p in parts)
        return (f"Last dose on day {last_dose_day}, then each component's "
                f"share of the effect washes out on its own -- {each}. Shares "
                "are each drug's single-agent effectiveness for this patient; "
                "exposure held at steady state while dosed.")

    def compare_treatments(self, treatment_keys: List[str],
                           twin: DigitalTwinConfig,
                           duration_days: int = 180,
                           last_dose_day: Optional[int] = None,
                           ) -> List[TreatmentResult]:
        """Compare multiple treatments.

        Returns list of TreatmentResult sorted by best response.
        """
        results = []
        for key in treatment_keys:
            result = self.simulate(key, twin, duration_days, last_dose_day)
            results.append(result)

        results.sort(key=lambda r: r.best_response)
        return results

    def get_recommended_treatments(self, twin: DigitalTwinConfig) -> List[str]:
        """Recommend treatments based on the digital twin profile."""
        recommended = []

        # Checkpoint inhibitors for hot/suppressed tumors or TMB-high
        if (twin.immune_score in ("hot", "suppressed") or
                twin.tumor_mutational_burden >= 10.0 or
                twin.microsatellite_instability == "MSI-H"):
            recommended.append("pembrolizumab")
            if twin.treg_fraction is not None and twin.treg_fraction > 0.2:
                recommended.append("pembro_ipi_combo")

        # PARP for DNA repair defects
        if twin.has_dna_repair_defect:
            recommended.append("olaparib")
            # Escalating to the checkpoint combination needs positive
            # evidence of infiltration. `!= "cold"` treated an unmeasured
            # microenvironment as grounds to escalate, which is the same
            # defect this change exists to remove.
            if twin.immune_score in ("hot", "suppressed"):
                recommended.append("olaparib_pembro_combo")

        # AR therapy if AR mutation
        if twin.has_ar_mutation:
            recommended.append("enzalutamide")

        # Neoantigen vaccine if sufficient targets
        if twin.neoantigen_vaccine_candidate:
            recommended.append("neoantigen_vaccine")
            # Vaccine + checkpoint combo is the strongest approach
            if twin.immune_score in ("hot", "suppressed"):
                recommended.append("neoantigen_vaccine_pembro")

        # Combo for cold tumors
        if twin.immune_score == "cold" and "pembro_ipi_combo" not in recommended:
            recommended.append("pembro_ipi_combo")

        return recommended or ["pembrolizumab"]  # Default

    def _calculate_effectiveness(self, profile: Dict,
                                 twin: DigitalTwinConfig) -> float:
        """Calculate treatment effectiveness (0-1) based on twin parameters."""
        eff = 0.5  # ILLUSTRATIVE baseline; see PARAMETER_PROVENANCE.

        # Dispatch on the declared mechanism set, never on substrings of the
        # display class string: `"AR" in "PARP inhibitor"` is True.
        mechanisms = treatment_mechanisms(profile)

        if MECH_PD1 in mechanisms:
            # Anti-PD-1 effectiveness depends on:
            # - PD-L1 expression (higher → more responsive)
            # An unmeasured biomarker must neither raise nor lower the
            # predicted effectiveness, so its term is skipped entirely.
            if twin.pd_l1_expression is not None:
                eff += 0.2 * twin.pd_l1_expression
            # - Exhausted T-cells to reactivate (need some, but not all)
            if (twin.t_cell_exhaustion_fraction is not None
                    and 0.2 < twin.t_cell_exhaustion_fraction < 0.8):
                eff += 0.15
            # - TMB (more neoantigens → more T-cell targets)
            eff += min(0.2, twin.tumor_mutational_burden / 50.0)
            # - Immune score
            if twin.immune_score == "hot":
                eff += 0.15
            elif twin.immune_score == "cold":
                eff -= 0.25

        if MECH_CTLA4 in mechanisms:
            # Anti-CTLA-4 works by depleting Tregs
            if twin.treg_fraction is not None:
                eff += 0.2 * twin.treg_fraction  # More Tregs → more effect
            eff += 0.1  # Priming enhancement

        if MECH_PARP in mechanisms:
            if twin.has_dna_repair_defect:
                eff += 0.35  # Major boost for HRD
            else:
                eff -= 0.3  # Not effective without HRD

        if MECH_AR_ANTAGONIST in mechanisms:
            if twin.has_ar_mutation:
                eff -= 0.1  # Resistance mutations reduce effectiveness
            else:
                eff += 0.2  # Wild-type AR responds well

        # Neoantigen vaccine effectiveness
        if MECH_VACCINE in mechanisms:
            # Effectiveness scales with number of vaccine-quality neoantigens
            if twin.vaccine_candidate_count >= 10:
                eff += 0.30
            elif twin.vaccine_candidate_count >= 5:
                eff += 0.20
            elif twin.vaccine_candidate_count >= 3:
                eff += 0.10
            else:
                eff -= 0.20  # Too few targets

            # Strong binders contribute more
            if twin.strong_binder_count >= 5:
                eff += 0.15

            # Immune microenvironment matters
            if twin.immune_score == "hot":
                eff += 0.15
            elif twin.immune_score == "cold":
                eff -= 0.15  # Hard to activate cold tumors

            # MHC-I downregulation reduces vaccine efficacy
            if twin.mhc1_downregulation is not None:
                eff -= twin.mhc1_downregulation * 0.3

        # MSI-H boosts checkpoint response
        if twin.microsatellite_instability == "MSI-H":
            if MECH_PD1 in mechanisms:
                eff += 0.2

        return max(0.0, min(1.0, eff))

    def _simulate_tumor_dynamics(self, effectiveness: float,
                                 profile: Dict,
                                 twin: DigitalTwinConfig,
                                 duration_days: int,
                                 last_dose_day: Optional[int] = None,
                                 ) -> List[float]:
        """Simulate tumor volume changes over time.

        Returns list of relative tumor volumes (1.0 = baseline).

        The kill rate is scaled by drug_exposure, which is exactly 1.0 on
        every day when the drug is given throughout (``last_dose_day`` None),
        so those curves are the same floats they were before exposure was
        modelled.

        The trajectory is deterministic given the patient AND the regimen:
        the same pair must produce the same curve on every run, or two
        viewings of one report disagree -- while two different patients must
        not share a noise realisation.
        """
        # `hash()` on a str is salted per interpreter process (PYTHONHASHSEED),
        # so seeding from it produced a *different* trajectory every time the
        # process restarted -- the one thing a seed exists to prevent. And
        # np.random.seed() mutates the global RNG, so calling this reached
        # out and changed unrelated draws elsewhere in the process.
        # A stable digest plus a local Generator fixes both.
        #
        # The digest covered the drug name ALONE, so the stochastic component
        # was byte-identical for every patient -- two twins with different
        # IDs and different TMB drew the same noise, and the model card's
        # claim that "results vary with random seed" was false in both
        # directions. Patient and regimen together now seed it.
        seed = int.from_bytes(
            hashlib.sha256(
                (_patient_seed_key(twin) + "\x00"
                 + str(profile.get("name", ""))).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(seed)

        onset = profile.get("effect_onset_days", 14)
        volumes = [1.0]
        volume = 1.0

        dosed_until = duration_days if last_dose_day is None else last_dose_day
        parts = self._exposure_parts(profile, twin)
        exposure_parts = [(p["share"], p["half_life_days"]) for p in parts]
        if dosed_until < duration_days:
            for p in parts:
                if p["half_life_days"] is None and p["name"] not in _NO_HALF_LIFE_REPORTED:
                    _NO_HALF_LIFE_REPORTED.add(p["name"])
                    logger.warning(
                        "%s: no half-life in the library, so after the last "
                        "dose its share of the effect stops at once rather "
                        "than washing out", p["name"])

        # Base tumor growth rate (per day). The one calibrated parameter in
        # this file: derived from the PROfound cohort A control-arm median
        # rPFS, see CALIBRATED_GROWTH_RATE_PER_DAY. 0.004992/day, where the
        # chosen value had been 0.005.
        growth_rate = CALIBRATED_GROWTH_RATE_PER_DAY
        # ILLUSTRATIVE multipliers: directionally supported, magnitudes chosen.
        if twin.has_pten_loss:
            growth_rate *= 1.3
        if twin.has_ar_mutation:
            growth_rate *= 1.1

        # Treatment kill rate (per day, after onset). ILLUSTRATIVE, and the
        # single largest determinant of predicted response depth: nothing
        # fits the 0.025. See PARAMETER_PROVENANCE.
        max_kill_rate = effectiveness * 0.025  # Up to 2.5% per day

        for day in range(1, duration_days + 1):
            # Ramp up treatment effect
            if day < onset:
                treatment_factor = 0
            else:
                ramp = min(1.0, (day - onset) / 21.0)  # 3-week ramp
                treatment_factor = (max_kill_rate * ramp
                                    * regimen_exposure(day, dosed_until,
                                                       exposure_parts))

            # Net growth = growth - treatment kill
            net_rate = growth_rate - treatment_factor

            # Add stochastic noise
            noise = rng.normal(0, 0.002)

            volume *= (1.0 + net_rate + noise)
            volume = max(0.01, volume)  # Floor at 1% (not full eradication)

            # Resistance development over time (gradual)
            if day > 90:
                resistance = (day - 90) / 500.0  # Slow resistance
                volume *= (1.0 + resistance * 0.001)

            volumes.append(round(volume, 4))

        return volumes

    @staticmethod
    def _first_progression_day(curve: List[float]) -> Optional[int]:
        """First day of a RECIST 1.1 progression event, or None.

        RECIST 1.1 references progression to the smallest sum of diameters
        recorded on study so far, not to baseline, so the comparison is
        against a RUNNING nadir. Working in volumes, a 20% diameter increase
        is a factor of 1.20 ** 3 on the volume ratio.

        Returns None when the window ends with no event -- the caller must
        then report the PFS as right-censored.
        """
        threshold = RECIST_PD_DIAMETER_RATIO ** 3
        nadir = curve[0]
        for day, volume in enumerate(curve):
            if volume >= threshold * nadir:
                return day
            if volume < nadir:
                nadir = volume
        return None

    @staticmethod
    def _classify_response(best_response: float, progressed: bool) -> str:
        """RECIST 1.1 best overall response: "PR", "SD" or "PD".

        Thresholds are DIAMETER ratios applied to the cube root of a volume
        ratio; see RECIST_PR_DIAMETER_RATIO. The previous version compared
        volumes against diameter thresholds, so a volume ratio of 0.70 -- an
        11.2% diameter shrink, well inside stable disease -- was reported as
        a partial response.

        PR is tested first because that is what "best overall response"
        means: a confirmed PR followed by progression is still a PR, with
        the progression carried by `progression_free_days`. PD is therefore
        returned only for a tumour that progressed without ever reaching a
        PR. The previous version could return PD for nothing at all: it
        tested `best_response > 1.20` against a nadir that includes day 0's
        1.0, so the branch was unreachable and a tumour 59 times its
        baseline volume at two years was classified stable disease.

        CR is never returned; see CR_NOT_REPRESENTABLE.
        """
        nadir_diameter_ratio = best_response ** (1.0 / 3.0)
        if nadir_diameter_ratio <= RECIST_PR_DIAMETER_RATIO:
            return "PR"
        if progressed:
            return "PD"
        return "SD"

    def _estimate_confidence(self, profile: Dict,
                             twin: DigitalTwinConfig) -> float:
        """Score how complete the INPUT was -- not how accurate the output is.

        Every weight below is illustrative (PARAMETER_PROVENANCE), and
        nothing in this model has been validated against observed patient
        outcomes, so this number must not be read as a probability that the
        prediction is correct.
        """
        confidence = 0.5

        # More data → higher confidence
        if twin.patient:
            confidence += 0.1
        if twin.immune_classification:
            confidence += 0.15
        if twin.tumor_mutational_burden > 0:
            confidence += 0.1
        if twin.microsatellite_instability != "unknown":
            confidence += 0.05

        # Strong biomarker matches boost confidence
        if profile.get("requires_dna_repair_defect") and twin.has_dna_repair_defect:
            confidence += 0.1

        return min(0.95, confidence)

    def _generate_rationale(self, profile: Dict,
                            twin: DigitalTwinConfig,
                            effectiveness: float) -> str:
        """Generate human-readable rationale for the prediction."""
        parts = [f"{profile['name']}: {profile['mechanism']}."]

        if effectiveness > 0.7:
            parts.append("Strong predicted response based on:")
        elif effectiveness > 0.4:
            parts.append("Moderate predicted response based on:")
        else:
            parts.append("Limited predicted response:")

        if twin.immune_score == "hot":
            parts.append("- Hot tumor microenvironment with immune infiltration")
        elif twin.immune_score == "cold":
            parts.append("- Cold tumor — limited immune infiltration may reduce response")

        if twin.tumor_mutational_burden >= 10:
            parts.append(f"- High TMB ({twin.tumor_mutational_burden:.1f}/Mb) — more neoantigens")

        if twin.t_cell_exhaustion_fraction is None:
            parts.append(
                "- T-cell exhaustion not measured (no cell-state data); "
                "no exhaustion term contributed to this estimate"
            )
        elif twin.t_cell_exhaustion_fraction > 0.3:
            parts.append(f"- {twin.t_cell_exhaustion_fraction:.0%} exhausted T-cells available for reactivation")

        mechanisms = treatment_mechanisms(profile)

        if twin.has_dna_repair_defect and MECH_PARP in mechanisms:
            parts.append("- DNA repair deficiency — synthetic lethality with PARP inhibition")
        elif MECH_PARP in mechanisms:
            parts.append(
                "- No DNA repair defect detected — PARP inhibition has no "
                "synthetic-lethal target in this patient"
            )

        if MECH_VACCINE in mechanisms:
            parts.append(f"- {twin.vaccine_candidate_count} neoantigen vaccine targets identified")
            if twin.strong_binder_count > 0:
                parts.append(f"- {twin.strong_binder_count} strong MHC-I binders (<50 nM)")
            if twin.hla_alleles:
                parts.append(f"- HLA typing: {len(twin.hla_alleles)} alleles characterized")

        return " ".join(parts)
