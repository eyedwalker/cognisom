"""The public roadmap cannot claim more than the repository shows.

``/?page=roadmap`` tells anyone who visits what Cognisom simulates, what it
only partly does, and what it does not do yet. A page like that drifts
toward generosity: a capability gets marked done on the strength of a
sentence, and nobody re-reads it when the code underneath changes.

So the page is generated from ``cognisom/dashboard/capabilities.py`` and
this file holds the rules: a row marked BUILT must name its evidence, the
evidence must exist in the repository, and a PARTIAL row must say what
limits it. When a capability lands, the honest move is to add its test to
the row and flip the status.

The two public pages are also RENDERED here, not just parsed: the roadmap
must show both videos and every layer, and the Bio-USD page must run its
own example through the real validator and print a conforming verdict --
because that page tells strangers the code works.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cognisom.dashboard.capabilities import (
    ABSENT, BUILT, LAYERS, PARTIAL, STATUSES, TARGETS, all_rows, counts,
)

REPO = Path(__file__).resolve().parent.parent
PAGES = REPO / "cognisom" / "dashboard" / "_pages"
ROADMAP = PAGES / "44_roadmap.py"
BIOUSD = PAGES / "45_biousd.py"


def _ids():
    return [f"{layer.title}: {row.name}" for layer, row in all_rows()]


@pytest.mark.parametrize("layer,row", all_rows(), ids=_ids())
def test_a_built_row_names_evidence_that_exists(layer, row):
    if row.status != BUILT:
        pytest.skip("only BUILT rows must cite evidence")
    assert row.evidence, (
        f"{row.name!r} is marked built but cites no test or script. Add the "
        "evidence, or mark it partial -- do not leave the claim unbacked.")
    missing = [p for p in row.evidence if not (REPO / p).is_file()]
    assert not missing, (
        f"{row.name!r} cites files that are not in the repository: {missing}")


def test_every_cited_file_exists_whatever_the_status():
    missing = [(row.name, p) for _, row in all_rows()
               for p in row.evidence if not (REPO / p).is_file()]
    assert not missing, missing


def test_a_partial_row_says_what_limits_it():
    silent = [row.name for _, row in all_rows()
              if row.status == PARTIAL and not row.note.strip()]
    assert not silent, (
        f"partial rows with no note: {silent}. 'Partial' with no reason "
        "reads as 'nearly done', which is the claim this page must not make.")


def test_statuses_are_the_known_three_and_the_counts_add_up():
    assert {row.status for _, row in all_rows()} <= set(STATUSES)
    n = counts()
    assert sum(n.values()) == len(all_rows())
    # The page is not a brochure: there must be gaps on it.
    assert n[ABSENT] > 0 and n[PARTIAL] > 0


def test_no_row_appears_twice():
    names = [row.name for _, row in all_rows()]
    assert len(names) == len(set(names))


def test_there_are_targets_and_each_says_why_it_matters():
    assert len(TARGETS) >= 3
    for title, body in TARGETS:
        assert title.strip() and len(body) > 60, title


def test_both_pages_are_reachable_without_an_account():
    app = (REPO / "cognisom" / "dashboard" / "app.py").read_text()
    for route, page in (("roadmap", ROADMAP), ("biousd", BIOUSD)):
        assert f'_query.get("page") == "{route}"' in app, route
        assert page.name in app, page.name
        assert "streamlit_page_gate" not in page.read_text(), (
            f"{page.name} is public; it must not call the login gate")


# ── rendered, not just parsed ────────────────────────────────────────

def _render(path: Path):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(path), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _all_markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_the_roadmap_renders_both_videos_and_every_layer():
    at = _render(ROADMAP)
    assert len(at.get("video")) == 2, "both demo videos belong on the roadmap"
    text = _all_markdown(at)
    for layer in LAYERS:
        assert layer.title in text, layer.title
        for row in layer.rows:
            assert row.name in text, row.name
    # Every built row's evidence is linked, so a visitor can open the test.
    for _, row in all_rows():
        for p in row.evidence:
            assert f"cognisom/blob/main/{p}" in text, p


def test_the_biousd_page_runs_its_own_example_and_it_conforms():
    at = _render(BIOUSD)
    code = "\n".join(c.value for c in at.code)
    assert "CONFORMS to Bio-USD" in code, (
        "the page's first example did not produce a conforming file when "
        "the page ran it")
    assert "DOES NOT CONFORM" not in code
    assert not at.error, [e.value for e in at.error]


def test_the_biousd_page_shows_the_spec_the_validator_uses():
    from cognisom.biousd import standard as biousd
    at = _render(BIOUSD)
    spec = biousd.render_spec_markdown()
    assert spec in _all_markdown(at), (
        "the page must render the generated specification verbatim, not a "
        "copy that can drift from vocabulary.py")
