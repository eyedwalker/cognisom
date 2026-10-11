"""The cited trial endpoints must match what the trials reported.

`cognisom/mad/evidence.py` is the evidence chain every board recommendation
carries, and `mad/compliance.py` builds the FDA non-device CDS framing on top
of it. A wrong endpoint there is not a modelling approximation; it is a
misquotation of a publication, presented to a clinician as a citation.

Two were wrong, both found by checking the publications rather than the code:

  * **KEYNOTE-158** carried `median_PFS_months: 4.1`. The TMB-high median is
    **2.1 months** (95% CI 2.1-3.7) -- notably the same median as the TMB-low
    group, with the separation appearing in the 12-month (24.3% vs 14.0%) and
    24-month (18.9% vs 6.5%) rates instead. 4.1 is not a published
    KEYNOTE-158 figure. The ORR of 29% (95% CI 21-39) was correct: that is the
    FDA-approval analysis over the 102 TMB-high patients among 755 with
    evaluable TMB.

  * **CheckMate-650** had its two cohorts assigned backwards -- "ORR 25% in
    post-chemo, 10% in pre-chemo". Sharma et al. report 25% in the
    PRE-chemotherapy cohort (cohort 1, chemo-naive) and 10% POST-chemotherapy
    (cohort 2), with median rPFS 5.5 vs 3.8 months and median OS 19.0 vs 15.2.
    The inversion reverses the clinical message: it implied the combination
    works better after chemotherapy has failed, when the published result is
    the opposite.

These assertions are deliberately literal. A test that recomputed the values
from the same constants would pass either way; the point is to pin them
against an external source, so the docstring above carries the citation and a
future edit has to argue with it.
"""
from __future__ import annotations

import pytest

from cognisom.mad.evidence import CHECKPOINT_TRIALS, PARP_TRIALS, VACCINE_TRIALS


# ── KEYNOTE-158 (NCT02628067), pembrolizumab in TMB-high solid tumours ───

def test_keynote_158_identifiers():
    trial = CHECKPOINT_TRIALS["keynote_158"]

    assert trial.source_id == "NCT02628067"
    assert trial.source_name == "KEYNOTE-158"


def test_keynote_158_median_pfs_is_the_published_figure():
    trial = CHECKPOINT_TRIALS["keynote_158"]

    assert trial.supporting_data["median_PFS_months"] == 2.1, (
        "TMB-high median PFS in KEYNOTE-158 is 2.1 months (95% CI 2.1-3.7); "
        "4.1 is not a published figure for this trial"
    )


def test_keynote_158_orr_is_the_fda_approval_analysis():
    trial = CHECKPOINT_TRIALS["keynote_158"]

    assert trial.supporting_data["ORR_TMB_high"] == 0.29


# ── CheckMate-650 (NCT02985957), nivolumab + ipilimumab in mCRPC ─────────

def test_checkmate_650_identifiers():
    trial = CHECKPOINT_TRIALS["checkmate_650"]

    assert trial.source_id == "NCT02985957"
    assert trial.source_name == "CheckMate-650"


def test_checkmate_650_cohorts_are_the_right_way_round():
    """The pre-chemotherapy cohort responded better, not worse."""
    data = CHECKPOINT_TRIALS["checkmate_650"].supporting_data

    assert data["ORR_pre_chemo"] == 0.25
    assert data["ORR_post_chemo"] == 0.10
    assert data["ORR_pre_chemo"] > data["ORR_post_chemo"], (
        "CheckMate-650 reported a HIGHER response rate in the chemo-naive "
        "cohort (25%) than after chemotherapy (10%); inverting this reverses "
        "the clinical message the evidence chain carries"
    )


def test_checkmate_650_claim_text_matches_the_data():
    """The prose a clinician reads must agree with the numbers beside it."""
    trial = CHECKPOINT_TRIALS["checkmate_650"]
    claim = trial.claim.lower()

    pre = claim.index("pre-chemo")
    post = claim.index("post-chemo")
    assert claim.index("25%") < pre < post, (
        f"the claim text assigns the cohorts differently from "
        f"supporting_data: {trial.claim!r}"
    )


def test_checkmate_650_rpfs_is_recorded_in_cohort_order():
    data = CHECKPOINT_TRIALS["checkmate_650"].supporting_data

    assert data["median_rPFS_months_pre_chemo"] == 5.5
    assert data["median_rPFS_months_post_chemo"] == 3.8


# ── the ones that were already right, pinned so they stay right ──────────

@pytest.mark.parametrize("key,ncts", [
    ("keynote_199", "NCT02787005"),
    ("keynote_158", "NCT02628067"),
    ("checkmate_650", "NCT02985957"),
])
def test_checkpoint_trial_registrations(key, ncts):
    assert CHECKPOINT_TRIALS[key].source_id == ncts


def test_profound_endpoints():
    """PROfound: rPFS 7.4 vs 3.6 months, HR 0.34."""
    trial = PARP_TRIALS["profound"]

    assert trial.source_id == "NCT02987543"
    data = trial.supporting_data
    assert data.get("rPFS_olaparib_months") == 7.4 or "7.4" in trial.claim
    assert data.get("hazard_ratio") == 0.34 or "0.34" in trial.claim


def test_every_cited_trial_has_a_registration_id():
    """A citation without an NCT number cannot be checked by a reviewer."""
    missing = []
    for name, trials in (("checkpoint", CHECKPOINT_TRIALS),
                         ("parp", PARP_TRIALS),
                         ("vaccine", VACCINE_TRIALS)):
        for key, item in trials.items():
            if not (item.source_id or "").startswith("NCT"):
                missing.append(f"{name}/{key}: {item.source_id!r}")

    assert not missing, f"cited trials with no NCT id: {missing}"
