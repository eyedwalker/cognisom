"""The caveats are written, stored, exported -- and were never shown.

A parameter's evidence is five fields and they are not alternatives.
``citation`` names the paper. ``conditions`` says what was measured, in
what buffer, in what organism. ``rationale`` says how the stored number
was got from the paper -- whether it is the paper's own figure, a median
over a spread, a midpoint of a range, or one column of a table.

Both the dashboard and the model's own formatter joined them with
``or``. Python short-circuits, so whenever a citation existed everything
after it was dropped. IDO1's Km rendered as ``20.0 uM - derived
(PMID:20178337)``, which reads as "that paper reports 20 uM". The
suppressed rationale says it is the median of 41 measurements spanning
0.72 to 6809 uM, and that the PMID is merely the paper contributing the
most records.

Every DERIVED value is REQUIRED to carry a rationale, because for a
derived number the rationale is the evidence. All of them were showing a
citation instead.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PAGE = REPO_ROOT / "cognisom" / "dashboard" / "_pages" / "14_entity_library.py"


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


def test_evidence_lines_keeps_every_field_that_is_set():
    from cognisom.library.parameters import ParameterValue, Provenance

    pv = ParameterValue(
        value=1.0, unit="uM", provenance=Provenance.DERIVED,
        citation="PMID:1", source="BRENDA", conditions="pH 7.4",
        rationale="median of 41, spanning 0.72 to 6809")
    fields = [f for f, _ in pv.evidence_lines()]
    assert fields == ["citation", "source", "conditions", "rationale"]


def test_describe_names_what_it_cannot_fit():
    """A one-liner may not carry a 400-character rationale. It may not
    pretend the rationale is absent either."""
    from cognisom.library.parameters import ParameterValue, Provenance

    pv = ParameterValue(
        value=20.0, unit="uM", provenance=Provenance.DERIVED,
        citation="PMID:20178337", rationale="median over 0.72-6809 uM")
    line = pv.describe()
    assert "PMID:20178337" in line
    assert "+rationale" in line, (
        f"the rationale vanished from the one-line summary: {line}")


def test_a_value_with_no_evidence_still_says_so():
    from cognisom.library.parameters import ParameterValue, Provenance

    pv = ParameterValue(value=1.0, unit="um",
                        provenance=Provenance.PLACEHOLDER)
    assert pv.evidence_lines() == []
    assert "no evidence" in pv.describe()


def test_the_page_no_longer_joins_evidence_with_or():
    """The specific expression that caused it, so it cannot come back."""
    text = PAGE.read_text()
    assert "pv.citation or pv.source" not in text, (
        "the short-circuit is back on the entity library page")
    assert "evidence_lines()" in text


def test_the_derived_values_that_most_needed_this(store):
    """Every DERIVED value carries a rationale by construction. Check a
    real one now reaches a reader rather than being hidden behind its
    citation."""
    from cognisom.library.parameters import get_parameter

    pv = get_parameter(store.find_entity_by_name("IDO1"), "km_um")
    assert pv is not None and pv.citation
    rendered = dict(pv.evidence_lines())
    assert "rationale" in rendered
    assert "6809" in rendered["rationale"], (
        "the spread this median was taken over is not reachable")


def test_every_evidenced_value_in_the_library_has_more_than_a_citation(store):
    """The census that made this worth fixing: a citation alone was all a
    reader ever saw, and essentially every value has more behind it."""
    from cognisom.library.parameters import Provenance, iter_parameters

    entities, _ = store.search(limit=10000)
    cited = beyond = 0
    for entity in entities:
        for _name, pv in iter_parameters(entity):
            if pv.provenance is Provenance.PLACEHOLDER or not pv.citation:
                continue
            cited += 1
            if len(pv.evidence_lines()) > 1:
                beyond += 1
    assert cited >= 700, f"only {cited} cited values found"
    assert beyond / cited > 0.95, (
        f"only {beyond} of {cited} cited values carry more than a "
        "citation; the short-circuit hid that much")
