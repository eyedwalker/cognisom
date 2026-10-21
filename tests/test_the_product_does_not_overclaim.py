"""Strings a reader would take as a measurement.

This is a research-use-only scientific tool. Its credibility rests on
never presenting a guess, a default or somebody else's published figure
as its own result. Three surfaces were doing that.

Each test below pins the exact string that was wrong, so the claim
cannot reappear by someone restoring an old copy of a page.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DASHBOARD = REPO_ROOT / "cognisom" / "dashboard"
PAGES = DASHBOARD / "_pages"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── the accuracy the repository does not have ────────────────────────

def test_no_surface_claims_an_unvalidated_accuracy_figure():
    """The home page said ">90% accuracy" and the investor page ">90%
    binding prediction accuracy". The product's own model card records
    in_repo_validation 0.0 and lists "Published metrics are not
    reproduced by any test in this repository" as a known limitation."""
    offenders = []
    for path in [DASHBOARD / "app.py", *PAGES.glob("*.py")]:
        for n, line in enumerate(_text(path).splitlines(), 1):
            if re.search(r">\s*9\d\s*%\s*(binding|accuracy)", line, re.I):
                if "as published" in line.lower() or "not revalidated" in line.lower():
                    continue
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert not offenders, "unvalidated accuracy claim:\n  " + "\n  ".join(offenders)


def test_the_model_card_still_says_it_was_never_revalidated():
    """If this ever stops being true the strings above may change. Until
    then they may not."""
    from cognisom.mad.model_cards import MHCFLURRY_PREDICTOR_CARD as card

    assert card.validation_metrics.get("in_repo_validation") == 0.0
    assert "not reproduced by any test" in " ".join(
        card.known_limitations).lower()


def test_a_published_benchmark_is_labelled_as_published():
    """The pipeline guide may quote MHCflurry's AUC -- it is a technical
    explainer and the number is real. It must say whose number it is."""
    text = _text(PAGES / "39_pipeline_guide.py")
    assert "AUC >0.95" in text
    window = text[text.index("AUC >0.95"):text.index("AUC >0.95") + 240]
    assert "as published" in window and "not revalidated" in window


# ── the column that invited a decision ───────────────────────────────

def test_the_completeness_score_is_not_labelled_confidence():
    """_estimate_confidence's own docstring: it scores "how complete the
    INPUT was -- not how accurate the output is", and its seven weights
    are recorded as illustrative. Beside a PFS in months, a column
    headed Confidence reads as the probability the prediction holds."""
    text = _text(PAGES / "28_digital_twin.py")
    assert '"Confidence":' not in text
    assert '"Input Completeness"' in text
    assert "Not the probability" in text or "NOT the probability" in text


def test_the_simulator_still_describes_that_score_the_same_way():
    """The rename is only honest while the thing being renamed is still
    an input-completeness score."""
    from cognisom.genomics.treatment_simulator import PARAMETER_PROVENANCE

    entry = next(p for p in PARAMETER_PROVENANCE
                 if p.name == "confidence_weights")
    assert entry.status == "illustrative"
    assert "not how accurate" in entry.basis.lower()


def test_the_board_metric_says_what_it_measures():
    """A different construction from the simulator's: hardcoded
    per-agent priors plus a consensus bonus. Agreement, not confidence."""
    text = _text(PAGES / "37_mad_agent.py")
    assert '"Board Agreement"' in text
    assert 'st.metric("Confidence"' not in text


def test_the_board_confidence_is_still_a_heuristic():
    """Pins the construction the label describes: fixed priors, a
    consensus adjustment, and a gap bonus -- nothing fitted."""
    source = _text(REPO_ROOT / "cognisom" / "mad" / "board.py")
    block = source[source.index("def _compute_confidence"):]
    block = block[:block.index("def _generate_rationale")]
    assert "0.10" in block and "0.15" in block
    assert "sum(o.confidence for o in opinions) / len(opinions)" in block
