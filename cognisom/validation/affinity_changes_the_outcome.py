"""Does predicted binding affinity change what targeting a peptide does?

WHAT THIS IS FOR. The three-arm experiment of 2026-11-27
(``neoantigen_target_choice.py``) found that a target's CELL COVERAGE
decides the outcome: a peptide on every cancer cell cleared the tumour,
one on 30% removed exactly its carriers. It could not say anything about
affinity, because under cognate specificity the kill signal used a fixed
affinity for an assigned pair -- a peptide binding at 12 nM and one
barely binding at 12,000 nM killed identically. Worse, the two targets
in that experiment differed in affinity as well as coverage
(``VVPYPILAL`` 117.4 nM, ``KKIEEVECL`` 168.7 nM), so the two variables
were confounded even though only coverage could act.

Phase 2 added peptide occupancy: how much of a cell's class I carries
one peptide, from its predicted affinity (single-site model,
``immune/mhc_loading.occupancy_from_ic50``). This module holds the
question put to it, with coverage held constant so affinity is the only
thing that moves.

WHY NOT A GRADED AFFINITY, since that is the obvious design. Measured
2026-11-27: the kill signal is ``affinity * mhc_level * costimulation``
through a Hill curve with EC50 0.3 and slope 4, so above the 0.7
recognition threshold it is saturated -- affinity 0.70 gives kill
probability 0.967 and 0.95 gives 0.990. Grading inside the recognised
band changes nothing; grading below it flips ``is_recognized`` off and
makes a weakly-presented peptide invisible instead. Occupancy
multiplies into the same signal and leaves recognition to the
assignment.

THE TARGETS, chosen from a measurement and not by hope. In a ring run
on the demo patient every one of the 101 displayed peptides is
displayed by every cancer cell, so coverage is uniform unless a
subclone is configured. Among them:

  tight   YLVSCKITV   FBXL17    12.1 nM   occupancy 1.000
  loose   MLENYRNLF   ZNF845   480.3 nM   occupancy 0.189

Same coverage, 39.7x apart in predicted affinity.

THE PREDICTION, WRITTEN BEFORE THE RUNS (2026-11-27).

  A1  With occupancy ON, targeting the tight binder leaves FEWER cancer
      cells at 24 h than targeting the loose binder, in the median seed.
  A2  With occupancy OFF, the two arms are INDISTINGUISHABLE -- equal
      medians. This is the one that can fail informatively: the clones
      assigned to a peptide are chosen by hashing its sequence, so two
      targets differ in which clones carry them even when affinity
      cannot act. If the arms differ with occupancy off, something other
      than affinity separates these peptides and A1 proves nothing.
  A3  With occupancy ON, the tight-binder arm leaves no more cancer
      cells than naming no target at all.

WHAT A PASS WOULD NOT MEAN. A1 is close to arithmetic once occupancy
exists: a lower occupancy multiplies the kill signal down, so fewer
kills follow by construction. What is NOT arithmetic is A2 -- that the
two peptides are otherwise interchangeable -- and that is what makes A1
readable as an affinity effect rather than a peptide-identity effect.
Neither is external validation, and neither is counted among this
engine's validated mechanisms, which remain two mechanisms across four
tests. The occupancy model's magnitude is an ESTIMATE; its direction is
sourced (Wells 2020, PMID 33038342).

WHAT IT BUYS. With coverage computable from a real callset (Phase 1)
and affinity now acting on the outcome, a simulated target ranking
depends on both of the things a candidate list is actually chosen on.
Before Phase 2 it depended on coverage alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Sequence

#: Direction anchor for the occupancy model, read directly.
WELLS_SOURCE = (
    "Wells 2020, Cell 183:818, PMID 33038342 (PubMed abstract read "
    "directly 2026-11-27): peptides T cells recognised had "
    "significantly stronger MHC binding affinity; counting the paper's "
    "Table S4 gives 25 of 224 validated below 50 nM measured affinity "
    "(11.2%) against 37 of 608 overall (6.1%)"
)

#: The two targets, and what the engine predicted for each.
TIGHT_PEPTIDE = "YLVSCKITV"
TIGHT_GENE = "FBXL17"
TIGHT_IC50_NM = 12.1

LOOSE_PEPTIDE = "MLENYRNLF"
LOOSE_GENE = "ZNF845"
LOOSE_IC50_NM = 480.3

#: Seeds required before a verdict is given.
MIN_SEEDS = 5


@dataclass
class Arm:
    """One arm's per-seed cancer-cell count at the end of the run."""

    name: str
    occupancy: bool
    cancer_end: List[int]

    @property
    def n_seeds(self) -> int:
        return len(self.cancer_end)

    def median_cancer(self) -> float:
        return float(median(self.cancer_end))


def arm_from_records(name: str, occupancy: bool,
                     records: Sequence[Dict[str, object]]) -> Arm:
    return Arm(name=name, occupancy=occupancy,
               cancer_end=[int(r["cancer"]) for r in records])


def affinity_verdict(
    tight_on: Arm,
    loose_on: Arm,
    tight_off: Arm,
    loose_off: Arm,
    none_on: Optional[Arm] = None,
) -> Dict[str, object]:
    """Apply A1-A3 to the four arms, plus the no-target control."""
    arms = [tight_on, loose_on, tight_off, loose_off]
    if none_on is not None:
        arms.append(none_on)
    if any(a.n_seeds < MIN_SEEDS for a in arms):
        return {"verdict": "NOT JUDGED",
                "why": {a.name: a.n_seeds for a in arms},
                "required": MIN_SEEDS}

    a1 = tight_on.median_cancer() < loose_on.median_cancer()
    a2 = tight_off.median_cancer() == loose_off.median_cancer()
    a3 = (none_on is None
          or tight_on.median_cancer() <= none_on.median_cancer())

    return {
        "verdict": "PASS" if (a1 and a2 and a3) else "FAIL",
        "A1_tight_beats_loose_with_occupancy": {
            "passed": a1,
            "tight": tight_on.median_cancer(),
            "loose": loose_on.median_cancer()},
        "A2_the_peptides_are_otherwise_interchangeable": {
            "passed": a2,
            "tight": tight_off.median_cancer(),
            "loose": loose_off.median_cancer(),
            "why_it_matters": (
                "cognate clones are chosen by hashing the peptide "
                "sequence, so this is the check that A1 measures "
                "affinity and not peptide identity")},
        "A3_the_tight_target_is_no_worse_than_no_target": {
            "passed": a3,
            "tight": tight_on.median_cancer(),
            "none": none_on.median_cancer() if none_on else None},
        "coverage_held_constant": (
            "both peptides are displayed by every cancer cell in this "
            "preset, so the arms differ only in predicted affinity"),
        "direction_source": WELLS_SOURCE,
        "occupancy_magnitude": (
            "ESTIMATE -- the effective peptide concentration in the "
            "single-site model has no source"),
        "not_external_validation": (
            "A1 follows from occupancy multiplying the kill signal; what "
            "is not arithmetic is A2, which is why it is scored"),
    }


# ── the recorded result ────────────────────────────────────────────
#
# WHAT HAPPENED, 2026-11-27. Ring, 24 h, 5 seeds, every T cell
# tumour-reactive, cognate specificity, no subclone, so both peptides
# are displayed by all 10 founding cancer cells in every run.
#
#   cancer cells left      no target            [4, 23, 14, 9, 9]  median 9
#                          tight, occupancy on  [0,  1,  0, 0, 0]  median 0
#                          loose, occupancy on  [1, 10,  7, 2, 10] median 7
#                          tight, occupancy off [0,  1,  0, 0, 0]  median 0
#                          loose, occupancy off [0,  1,  0, 0, 0]  median 0
#
# A1, A2 and A3 all pass, and A2 passes more cleanly than it had to:
# with occupancy off the two peptides are identical SEED FOR SEED, not
# merely equal in median. So the clone-assignment hash does not
# distinguish them and the whole of the A1 difference is attributable to
# predicted affinity.
#
# Two internal checks fall out of the same table. The tight arm is
# identical with the flag on and off, which is what a peptide at
# occupancy 1.000 must do. And the loose arm with occupancy on (median
# 7) lands most of the way back toward naming no target at all (median
# 9): on this model a 480 nM target is nearly useless, while a 12 nM
# target on the same cells clears the tumour.
#
# The honest reading: this is the engine's ranking now responding to
# affinity, at a magnitude set by an ESTIMATE. It is a wiring-and-sign
# result, like the coverage experiment before it, and it is NOT counted
# among the two externally validated mechanisms. The TESLA table
# (608 peptides with measured affinity and measured T-cell recognition,
# free in the Cell supplement) is the external test that could
# discipline the magnitude; see Phase 3.

RESULTS_FILE = "results/affinity_changes_the_outcome_2026_11_27.json"


def stored_arms() -> Dict[str, Arm]:
    """The recorded runs, one Arm per configuration."""
    import json
    from pathlib import Path

    data = json.loads((Path(__file__).parent / RESULTS_FILE).read_text())
    out: Dict[str, Arm] = {}
    for name, records in data["arms"].items():
        occupancy = name.endswith("_on")
        out[name] = arm_from_records(name, occupancy, records)
    return out


def stored_verdict() -> Dict[str, object]:
    """Re-apply A1-A3 to the recorded runs."""
    a = stored_arms()
    return affinity_verdict(
        tight_on=a["tight_on"], loose_on=a["loose_on"],
        tight_off=a["tight_off"], loose_off=a["loose_off"],
        none_on=a["none"],
    )
