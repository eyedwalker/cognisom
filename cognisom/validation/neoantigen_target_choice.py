"""Ranking neoantigen targets by simulated outcome.

WHAT THIS IS FOR. The platform predicts a patient's neoantigens and
ranks them by predicted MHC binding. Binding is not the same question
as "which of these is worth targeting", and until now the engine could
not be asked the second one: measured on 2026-11-27, the TCR
recognition gate recognised every peptide it was shown -- all 50 of
the demo patient's candidates, 200 random 9-mers, a non-mutant self
peptide -- because receptor and peptide are both hashed to random
vectors and the best match over a repertoire is an order statistic of
noise. Two runs targeting different peptides were the same run. See
the SPECIFICITY notes in ``engine/py/immune/tcr_repertoire.py``.

With cognate specificity a run names the peptides its T cells are
specific for, so arms that differ only in the target differ only in
the target. This module holds the first question put to that: does a
target's CELL COVERAGE -- the share of cancer cells displaying it --
decide the outcome?

THE PREDICTION, WRITTEN BEFORE THE RUNS (2026-11-27). Pre-specified
because the alternative is reading whatever came out as a result.

  C1  Targeting a CLONAL peptide (displayed by every cancer cell)
      leaves fewer cancer cells at 24 h than naming no target at all,
      in the median seed.
  C2  Targeting a SUBCLONAL peptide (displayed by 30% of founders)
      leaves MORE cancer cells than the clonal target, in the median
      seed. Ordering only: clonal better than subclonal.
  C3  Under the subclonal target the surviving tumour is enriched for
      NON-CARRIERS against the 70% they started at, because the cells
      displaying the target are the ones being removed.

WHAT IT IS COMPARED WITH. The direction is the one measured in
patients: McGranahan 2016, Science 351:1463, PMID 26940869, abstract
read directly on 2026-11-27 -- clonal neoantigen burden relates to
overall survival in lung adenocarcinoma, sensitivity to PD-1 and
CTLA-4 blockade "was enhanced in tumors enriched for clonal
neoantigens", and chemotherapy-induced SUBCLONAL neoantigens "were
enriched in certain poor responders". That paper supports "therapeutic
developments targeting clonal neoantigens", which is the ordering C2
states.

WHAT A PASS WOULD NOT MEAN, and this limit is the important part. The
engine kills a cell when a T cell recognises a peptide that cell
displays, and a subclonal peptide is displayed by fewer cells BY
CONSTRUCTION. So C1-C3 are close to arithmetic: they check that the
chain from clonal fraction through per-cell presentation to kill is
wired and carries the right sign, not that the engine discovered
anything about immunology. It is a self-consistency test whose
direction happens to agree with a measured one. It is NOT external
validation and must not be counted as a validated mechanism.

What it does buy, and what the capability is actually for: the engine
can now rank candidate targets by a simulated outcome rather than by
predicted binding alone, and the first thing that ranking says is that
coverage dominates affinity. A candidate list sorted by IC50 -- which
is what the platform's neoantigen page shows -- does not carry that.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Sequence

#: Direction anchor, read directly.
MCGRANAHAN_SOURCE = (
    "McGranahan 2016, Science 351:1463, PMID 26940869 (PubMed abstract, "
    "read directly): checkpoint-blockade sensitivity enhanced in "
    "tumours enriched for CLONAL neoantigens, chemotherapy-induced "
    "SUBCLONAL neoantigens enriched in certain poor responders"
)

#: The subclonal target's share of founding cancer cells in the runs
#: this module judges. A modelling choice, not a measurement.
SUBCLONAL_FRACTION = 0.30

#: Seeds required before a verdict is given.
MIN_SEEDS = 3


@dataclass
class Arm:
    """One arm's per-seed outcome."""

    name: str
    cancer_end: List[int]
    carriers_end: List[Optional[int]]
    non_carriers_end: List[Optional[int]]

    @property
    def n_seeds(self) -> int:
        return len(self.cancer_end)

    def median_cancer(self) -> float:
        return float(median(self.cancer_end))

    def median_non_carrier_share(self) -> Optional[float]:
        shares = []
        for carriers, others in zip(self.carriers_end,
                                    self.non_carriers_end):
            if carriers is None or others is None:
                continue
            total = carriers + others
            if total:
                shares.append(others / total)
        return float(median(shares)) if shares else None


def arm_from_records(name: str, records: Sequence[Dict[str, object]]) -> Arm:
    return Arm(
        name=name,
        cancer_end=[int(r["cancer"]) for r in records],      # type: ignore[arg-type]
        carriers_end=[r.get("carriers_end") for r in records],  # type: ignore[misc]
        non_carriers_end=[r.get("non_carriers_end") for r in records],  # type: ignore[misc]
    )


def coverage_verdict(none: Arm, clonal: Arm,
                     subclonal: Arm) -> Dict[str, object]:
    """Apply C1-C3 to three arms of the same configuration."""
    arms = (none, clonal, subclonal)
    if any(a.n_seeds < MIN_SEEDS for a in arms):
        return {"verdict": "NOT JUDGED",
                "why": {a.name: a.n_seeds for a in arms},
                "required": MIN_SEEDS}
    c1 = clonal.median_cancer() < none.median_cancer()
    c2 = subclonal.median_cancer() > clonal.median_cancer()
    share = subclonal.median_non_carrier_share()
    started = 1.0 - SUBCLONAL_FRACTION
    c3 = share is not None and share > started
    return {
        "verdict": "PASS" if (c1 and c2 and c3) else "FAIL",
        "C1_clonal_beats_no_target": {
            "passed": c1, "clonal": clonal.median_cancer(),
            "none": none.median_cancer()},
        "C2_clonal_beats_subclonal": {
            "passed": c2, "clonal": clonal.median_cancer(),
            "subclonal": subclonal.median_cancer()},
        "C3_subclonal_arm_enriches_non_carriers": {
            "passed": c3, "non_carrier_share": share,
            "started_at": started},
        "direction_source": MCGRANAHAN_SOURCE,
        "not_external_validation": (
            "a subclonal peptide is displayed by fewer cells by "
            "construction, so this checks the wiring and its sign, not "
            "a discovered fact"),
    }


# ── the recorded result ────────────────────────────────────────────
#
# WHAT HAPPENED, 2026-11-27. Ring, 24 h, 5 seeds, every T cell
# tumour-reactive, cognate specificity, F13B T192I set to 30% of
# founders. Clonal target VVPYPILAL (ZNF362 S217L, displayed by 10/10
# cells); subclonal target KKIEEVECL (F13B T192I, 3/10 cells).
#
#   cancer cells left   no target  [4, 23, 14, 9, 9]   median 9
#                       clonal     [0,  1,  0, 0, 0]   median 0
#                       subclonal  [4, 15,  7, 0, 1]   median 4
#
# C1, C2 and C3 all pass. The sharpest number is not in the medians:
# the subclonal arm ended with ZERO carriers in ALL FIVE seeds and
# every survivor a non-carrier, so the target was not merely less
# effective -- it cleared exactly the cells displaying it and left the
# rest untouched. That is selection by the target, and it is the thing
# a ranking by predicted binding cannot see.
#
# Read with the limits in this module's docstring: 10 founding cancer
# cells, one genome, one tissue layout, and an effect that follows from
# how presentation is wired. Two of the five subclonal seeds happened
# to clear as well (0 and 1 cells left), which at this population size
# is noise, not a finding.

RESULTS_FILE = "results/neoantigen_target_choice_2026_11_27.json"


def stored_arms() -> Dict[str, Arm]:
    """The recorded runs, one Arm per target choice."""
    import json
    from pathlib import Path

    data = json.loads((Path(__file__).parent / RESULTS_FILE).read_text())
    return {name: arm_from_records(name, records)
            for name, records in data["arms"].items()}


def stored_verdict() -> Dict[str, object]:
    """Re-apply C1-C3 to the recorded runs."""
    arms = stored_arms()
    return coverage_verdict(arms["none"], arms["clonal"],
                            arms["subclonal"])

