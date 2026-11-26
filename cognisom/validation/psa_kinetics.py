"""Serum PSA kinetics under enzalutamide: a pre-specified outcome test.

WHAT IS TESTED. Whether the engine's serum PSA trajectory under
enzalutamide looks like the one measured in men. The thresholds below
were written on 2026-11-26 BEFORE the multi-seed runs they judge, from
PubMed records read directly that day. They are constants in this file
so that a later change to them is a diff someone has to explain.

WHAT HAD ALREADY BEEN SEEN, stated because it limits the claim. One
12-week run of the carrier arm (seed 7, 2026-11-25) had been read
before these thresholds were written: tumour-total PSA fell 59% by day
23 and was back to a 29% fall at day 84. So the carrier arm is expected
to fail, and this is not a blind test of it. The carrier-FREE arm had
never been run for longer than a week when this was written; for that
arm the test is blind.

WHICH PATIENTS THE ENGINE'S TUMOUR RESEMBLES. This decides which trial
is the fair comparison and it is easy to get wrong. The famous
enzalutamide trials (AFFIRM, PREVAIL) enrolled CASTRATION-RESISTANT
men: castrate testosterone, tumours already growing without androgen.
The engine's default tumour is the opposite on both counts -- a
eugonadal androgen supply, and cells that stop dividing and die when
AR signalling is withdrawn. Its clinical counterpart is enzalutamide
MONOTHERAPY in HORMONE-NAIVE men with non-castrate testosterone, which
is exactly Tombal 2014. That is the PRIMARY comparison. The CRPC
numbers are recorded as secondary context and are not scored, because
a pass or a fail against the wrong population says nothing.

THE ARMS.
  naive    demo patient with the AR F877L variant removed: a tumour
           never exposed to an AR antagonist. F877L is described as
           arising UNDER such treatment, so a hormone-naive tumour is
           modelled without it. (That description is from general
           knowledge, not re-read for this file.)
  carrier  the shipped demo: 10% of cancer cells carry AR F877L. The
           10% is a modelling choice, not a measurement.
  none     untreated, 10% carriers.

WHAT A PASS WOULD AND WOULD NOT MEAN. One patient genome, a handful of
seeds, thresholds from abstracts. A pass says the engine's serum PSA
curve is not inconsistent with the typical hormone-naive response; it
does not validate the mechanism that produced it. A fail is more
informative: it says which part of the curve the engine cannot make.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

# ── clinical anchors, each from a PubMed record read on 2026-11-26 ──

#: PRIMARY. Tombal 2014, Lancet Oncol 15:592, PMID 24739897 (abstract):
#: hormone-naive prostate cancer, non-castrate testosterone,
#: enzalutamide 160 mg/day alone. "62 patients (92.5%, 95% CI
#: 86.2-98.8) had a decline in PSA of 80% or greater at week 25."
TOMBAL_DECLINE = 0.80
TOMBAL_WEEK = 25
TOMBAL_RESPONDER_FRACTION = 0.925
TOMBAL_SOURCE = (
    "Tombal 2014, PMID 24739897 (PubMed abstract, read directly): "
    "92.5% of 67 hormone-naive, non-castrate men on enzalutamide "
    "monotherapy had a PSA decline of 80% or more at week 25")
#: Durability of the same cohort. Tombal 2018, J Urol 199:459, PMID
#: 28867562 (abstract): of 42 men assessed at 3 years, 38 (90.5%)
#: maintained the >=80% response. So the response does not rebound
#: within the horizon any run here reaches.
TOMBAL_DURABILITY_SOURCE = (
    "Tombal 2018, PMID 28867562 (PubMed abstract, read directly): "
    "90.5% of men assessed at 3 years maintained the PSA response")

#: SECONDARY, castration-resistant populations. Recorded, not scored.
CRPC_CONTEXT: Dict[str, str] = {
    "psa50_post_chemo": "54% vs 2% placebo; Scher 2012 (AFFIRM), PMID "
                        "22894553, abstract",
    "psa50_chemo_naive": "78% vs 3% placebo; Beer 2014 (PREVAIL), PMID "
                         "24881730, abstract",
    "time_to_psa_progression": "median 8.3 vs 3.0 months, AFFIRM (PMID "
                               "22894553); 8.31 vs 2.86 months, Pu 2022, "
                               "PMID 35397772; both abstracts",
    "time_to_nadir": "mean 14 weeks on enzalutamide among men with any "
                     "PSA fall; Miyake 2017, PMID 28188090, abstract",
    "minimum_exposure": "treatment continued at least 12 weeks before "
                        "acting on PSA; Scher 2008 (PCWG2), PMID "
                        "18309951, abstract",
}

# ── the pre-specified criteria ─────────────────────────────────────

#: A PSA rise of this much above the nadir counts as progression. The
#: 25% figure is PCWG2's as generally described; the abstract read here
#: does not state it, and PCWG2's accompanying absolute threshold
#: cannot be applied to a relative index at all. So this is the
#: RELATIVE HALF of a clinical definition, and labelled as that.
PROGRESSION_RISE = 0.25

#: Seeds per arm that must be present before a verdict is given.
MIN_SEEDS = 3


def _clean(time_h: Sequence[float],
           serum: Sequence[float]) -> Tuple[List[float], List[float]]:
    if len(time_h) != len(serum) or len(serum) < 2:
        raise ValueError("need matching time and serum series of length >= 2")
    if serum[0] <= 0.0:
        raise ValueError("serum PSA must start positive to take ratios")
    return [float(t) for t in time_h], [float(s) / float(serum[0])
                                        for s in serum]


def decline_at(time_h: Sequence[float], serum: Sequence[float],
               week: float) -> Optional[float]:
    """Fractional fall from baseline at ``week``; None if the run is
    shorter. Negative means PSA is above baseline."""
    t, s = _clean(time_h, serum)
    target = week * 7.0 * 24.0
    if t[-1] < target - 1e-6:
        return None
    for i in range(1, len(t)):
        if t[i] >= target - 1e-9:
            span = t[i] - t[i - 1]
            w = 0.0 if span <= 0 else (target - t[i - 1]) / span
            return 1.0 - (s[i - 1] + w * (s[i] - s[i - 1]))
    return 1.0 - s[-1]


def best_decline(time_h: Sequence[float], serum: Sequence[float]) -> float:
    _, s = _clean(time_h, serum)
    return 1.0 - min(s)


def nadir(time_h: Sequence[float],
          serum: Sequence[float]) -> Tuple[float, float]:
    """(week of the lowest reading, that reading relative to baseline)."""
    t, s = _clean(time_h, serum)
    i = min(range(len(s)), key=lambda j: s[j])
    return t[i] / (7.0 * 24.0), s[i]


def weeks_to_progression(time_h: Sequence[float], serum: Sequence[float],
                         rise: float = PROGRESSION_RISE) -> Optional[float]:
    """First week serum PSA stands ``rise`` above its lowest value so
    far. None if it never does within the run -- which is censoring,
    not evidence that it never would."""
    t, s = _clean(time_h, serum)
    low = s[0]
    for ti, si in zip(t, s):
        low = min(low, si)
        if si >= low * (1.0 + rise) and si > low:
            return ti / (7.0 * 24.0)
    return None


def doubling_time_days(time_h: Sequence[float],
                       serum: Sequence[float]) -> Optional[float]:
    """Log-linear fit over the whole series; None if not rising."""
    t, s = _clean(time_h, serum)
    pts = [(ti / 24.0, math.log(si)) for ti, si in zip(t, s) if si > 0]
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 0:
        return None
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    return math.log(2.0) / slope if slope > 0 else None


@dataclass
class ArmSummary:
    arm: str
    n_seeds: int
    decline_week_25: List[Optional[float]] = field(default_factory=list)
    decline_week_12: List[Optional[float]] = field(default_factory=list)
    best: List[float] = field(default_factory=list)
    nadir_week: List[float] = field(default_factory=list)
    progression_week: List[Optional[float]] = field(default_factory=list)
    doubling_days: List[Optional[float]] = field(default_factory=list)


def summarise(arm: str,
              runs: Sequence[Tuple[Sequence[float], Sequence[float]]]
              ) -> ArmSummary:
    out = ArmSummary(arm=arm, n_seeds=len(runs))
    for t, s in runs:
        out.decline_week_25.append(decline_at(t, s, TOMBAL_WEEK))
        out.decline_week_12.append(decline_at(t, s, 12))
        out.best.append(best_decline(t, s))
        out.nadir_week.append(nadir(t, s)[0])
        out.progression_week.append(weeks_to_progression(t, s))
        out.doubling_days.append(doubling_time_days(t, s))
    return out


def primary_verdict(summary: ArmSummary) -> Dict[str, object]:
    """The pre-specified primary test, applied to one treated arm.

    PASS needs both, in the MEDIAN seed, because 92.5% of men met the
    clinical criterion and so the typical simulated patient must:

      P1  serum PSA at week 25 is at least 80% below baseline;
      P2  no PSA progression (a 25% rise from nadir) by week 25, since
          the clinical response was still held at 3 years.

    A run shorter than 25 weeks cannot be judged and is reported as
    such rather than judged on what it has.
    """
    if summary.n_seeds < MIN_SEEDS:
        return {"verdict": "NOT JUDGED",
                "why": f"{summary.n_seeds} seeds, {MIN_SEEDS} required"}
    at25 = summary.decline_week_25
    if any(d is None for d in at25):
        return {"verdict": "NOT JUDGED",
                "why": "at least one run is shorter than 25 weeks"}
    med = median(at25)  # type: ignore[arg-type]
    progressed = [w is not None and w <= TOMBAL_WEEK
                  for w in summary.progression_week]
    p1 = med >= TOMBAL_DECLINE
    p2 = sum(progressed) * 2 < len(progressed)
    return {
        "verdict": "PASS" if (p1 and p2) else "FAIL",
        "P1_decline_at_week_25": {"median": med, "required": TOMBAL_DECLINE,
                                  "passed": p1, "per_seed": list(at25)},
        "P2_no_progression_by_week_25": {
            "seeds_progressed": int(sum(progressed)),
            "of": len(progressed), "passed": p2,
            "weeks": list(summary.progression_week)},
        "source": TOMBAL_SOURCE,
        "durability_source": TOMBAL_DURABILITY_SOURCE,
    }
