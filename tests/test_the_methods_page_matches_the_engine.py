"""The methods page cannot describe an engine other than the one that runs.

``/?page=methods`` tells a scientist how each quantity is computed and
how far to trust it. A page like that fails in two directions, and both
are quiet:

* it goes STALE -- a module is added, or its behaviour changes, and the
  description keeps saying the old thing;
* it goes GENEROUS -- a mechanism that computes numbers nobody reads is
  described as though it drives the simulation, because the sentence
  describing it was written when someone intended to wire it.

So the page is generated from ``cognisom/dashboard/methods.py``, its
signal, kill and validation tables are READ from the engine rather than
retyped, and this file holds the rules. The sharpest of them is
``test_the_inert_modules_really_are_inert``: the page states, in public,
that methylation is computed and read by nothing, and that claim is
checked against the repository by AST rather than taken on trust.

If that test fails, something was WIRED. The fix is to flip the row to
``CHANGES`` and say what it now drives -- never to loosen the test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cognisom.dashboard.methods import (
    CHANGES, LIMITATIONS, MODULES, NARRATIVE, OFF, REPORTS_ONLY, WIRING,
    cited_paths, kill_rows, modules_by_wiring, signal_rows, status_counts,
    validation_rows,
)

REPO = Path(__file__).resolve().parent.parent
PAGES = REPO / "cognisom" / "dashboard" / "_pages"
METHODS = PAGES / "47_methods.py"


# ── the inventory matches the engine ─────────────────────────────────

def _default_modules() -> dict:
    """What an ordinary run enables, read from EngineRunner itself.

    Not a list copied into this file: that is exactly how a new module
    would ship undescribed.
    """
    from cognisom.dashboard.engine_runner import EngineRunner

    return EngineRunner(duration=0.0).modules_enabled


#: The four solver back-ends are alternative integrators, not biology,
#: and the page says so rather than giving each a row.
SOLVERS = {"ode", "bngl", "hybrid", "smoldyn"}


def test_every_module_an_ordinary_run_enables_has_a_row():
    described = {m.key for m in MODULES}
    enabled = {k for k, on in _default_modules().items()
               if on and k not in SOLVERS}
    missing = sorted(enabled - described)
    assert not missing, (
        f"these modules run in a default simulation and the methods page "
        f"does not describe them: {missing}. Add a row to "
        "cognisom/dashboard/methods.py saying what each computes and "
        "whether anything reads it.")


def test_no_row_describes_a_module_the_engine_cannot_register():
    from cognisom.dashboard import engine_runner

    source = (REPO / "cognisom" / "dashboard" / "engine_runner.py").read_text()
    for row in MODULES:
        assert f'"{row.key}": ' in source, (
            f"methods.py has a row for {row.key!r}, which EngineRunner "
            "does not register under that name")
    assert engine_runner  # the import is the point: the names must resolve


def test_a_row_marked_off_by_default_really_is_off():
    enabled = _default_modules()
    for row in modules_by_wiring(OFF):
        assert not enabled.get(row.key, False), (
            f"{row.key!r} is described as off by default and a default run "
            "enables it")


def test_a_row_marked_wired_is_enabled_in_a_default_run():
    enabled = _default_modules()
    for row in MODULES:
        if row.wiring == OFF:
            continue
        assert enabled.get(row.key, False), (
            f"{row.key!r} is described as running in a default run and it "
            "is not enabled")


def test_every_cited_path_is_in_the_repository():
    missing = [p for p in cited_paths() if not (REPO / p).is_file()]
    assert not missing, f"methods.py cites files that are not here: {missing}"


def test_the_narrative_document_exists():
    assert (REPO / NARRATIVE).is_file()


def test_statuses_are_the_known_three_and_no_module_appears_twice():
    assert {m.wiring for m in MODULES} <= set(WIRING)
    keys = [m.key for m in MODULES]
    assert len(keys) == len(set(keys))


# ── the honest column has to mean something ──────────────────────────

def test_the_page_is_not_a_brochure():
    """A methods page on which everything is load-bearing and nothing is
    wrong is a page nobody checked."""
    assert modules_by_wiring(REPORTS_ONLY), (
        "no module is marked 'reported only'. Either something was wired "
        "-- in which case say what it drives -- or the honest column has "
        "quietly become decoration.")
    assert len(LIMITATIONS) >= 8


def test_the_claim_about_the_demo_matches_the_demo_that_shipped():
    """A limitation says the recorded demo sets the tumour-reactive T
    fraction to 10% rather than running on the generous default. That
    is a claim about an artefact, and artefacts get re-recorded."""
    import json

    facts = json.loads(
        (REPO / "cognisom" / "dashboard" / "assets"
         / "demo_run.json").read_text())
    assert facts.get("reactive_t_fraction") == 0.1, (
        f"the shipped recording used reactive_t_fraction="
        f"{facts.get('reactive_t_fraction')!r}, and the methods page "
        "tells readers it is 10%")
    # And the engine default really is the generous one, which is the
    # other half of the same sentence.
    from cognisom.modules.immune_module import ImmuneModule

    mod = ImmuneModule({})
    assert mod.tcell_tumour_reactive_fraction is None, (
        "the tumour-reactive fraction is no longer opt-in; the page says "
        "every T cell is reactive unless a run sets it")


def test_every_limitation_says_what_would_lift_it():
    for lim in LIMITATIONS:
        assert lim.what.strip() and lim.why_it_matters.strip()
        assert len(lim.to_lift_it) > 40, (
            f"{lim.what!r} has no real remedy; 'more work' is not one")
        assert "more work" not in lim.to_lift_it.lower()


def test_every_module_row_that_changes_a_run_shows_its_work():
    for row in modules_by_wiring(CHANGES):
        assert len(row.worked_example) > 80, (
            f"{row.key!r} changes a run and the page does not say how it "
            "reaches its answer, which is the whole question this page "
            "exists to answer")


#: The functions by which an "inert" module could change a cell's fate.
#: Each maps a module key to the call that would do it.
_INERT_ESCAPE_HATCHES = {
    "epigenetic": "get_expression_modifier",
}


@pytest.mark.parametrize("key,func", sorted(_INERT_ESCAPE_HATCHES.items()))
def test_the_inert_modules_really_are_inert(key, func):
    """The page's sharpest public claim, held by AST.

    The methods page says methylation is computed, plotted, and read by
    nothing. Measured on 2026-11-13: the epigenetic module tracks ~2,000
    cells, costs ~2.5% of every engine step and takes ~31,000 random
    draws per step, and ``get_expression_modifier`` -- the one function
    through which methylation could reach transcription -- has no caller
    outside the module that defines it.

    This scans every call site in the package for that name. If it
    fails, the module was WIRED: flip its row to CHANGES and say what it
    drives. Do not delete this test to make the page's old sentence
    true again.
    """
    owner = REPO / "cognisom" / "modules" / f"{key}_module.py"
    assert owner.is_file(), owner

    callers = []
    for path in (REPO / "cognisom").rglob("*.py"):
        if path == owner:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                                  # noqa: PERF203
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = (fn.attr if isinstance(fn, ast.Attribute)
                    else fn.id if isinstance(fn, ast.Name) else None)
            if name == func:
                callers.append(f"{path.relative_to(REPO)}:{node.lineno}")

    assert not callers, (
        f"{key}.{func}() now has callers: {callers}. The methods page "
        f"says this module is computed and read by nothing, which is no "
        "longer true -- flip its row from REPORTS_ONLY to CHANGES and "
        "describe what it drives.")


# ── the tables are read, not retyped ─────────────────────────────────

def test_the_signal_table_is_the_engine_inventory():
    from cognisom.modules.signalling import inventory

    rows = signal_rows()
    inv = inventory()
    assert len(rows) == len(inv)
    assert [r["Field"] for r in rows] == [e.name for e in inv]
    # The claim: not every field the engine solves is consumed. The
    # number is pinned beside it -- if it moves because something was
    # wired, re-measure and update the page's sentence, do not re-pin.
    wired = sum(1 for r in rows if r["Changes a run"] == "yes")
    assert 0 < wired < len(rows), (
        "either nothing reads any field, or everything does; both mean "
        "the 'changes a run' column has stopped distinguishing anything")
    assert wired == 3, (
        f"{wired} of {len(rows)} fields are wired, not 3. Re-measure and "
        "update section 2's sentence on the page; do not re-pin.")


def test_the_kill_table_covers_every_calibrated_number():
    from cognisom.engine.py.immune.kill_calibration import calibration_report

    report = calibration_report()
    rows = kill_rows()
    assert len(rows) == len(report), (
        "the page's kill table and kill_calibration disagree about how "
        "many numbers are on the kill path")
    shown = {str(r["Parameter"]).replace(" ", "_") for r in rows}
    assert shown == set(report)


def test_every_kill_row_names_a_source_and_a_range():
    for row in kill_rows():
        assert str(row["Source"]).strip(), row["Parameter"]
        assert "-" in str(row["Plausible range"]), row["Parameter"]


def test_every_provenance_label_the_engine_uses_is_explained():
    """A label with no gloss reads as 'fine'.

    'calibration-choice' arrived on the kill path without a badge or a
    meaning on the page, and rendered as a grey word a reader would
    skim past. The page falls back rather than crashing, so only a test
    catches it.
    """
    page = METHODS.read_text()
    for status in status_counts():
        assert f'"{status}"' in page, (
            f"the kill path uses the provenance label {status!r} and the "
            "methods page neither badges nor explains it, so it renders "
            "as a grey word with no meaning attached")


def test_the_provenance_split_still_admits_estimates():
    counts = status_counts()
    assert sum(counts.values()) == len(kill_rows())
    assert counts.get("ESTIMATE", 0) > 0, (
        "no kill-path number is labelled an estimate. Either every one "
        "was sourced -- worth celebrating and worth checking -- or a "
        "label was quietly promoted.")


def test_the_validation_table_comes_from_capabilities():
    from cognisom.dashboard.capabilities import LAYERS

    rows = validation_rows()
    expected = [row for layer in LAYERS if layer.title == "Validation"
                for row in layer.rows]
    assert rows, "the Validation layer disappeared from capabilities.py"
    assert [r["Claim"] for r in rows] == [c.name for c in expected]


# ── rendered, not just parsed ────────────────────────────────────────

def _render(path: Path):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(path), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _all_text(at) -> str:
    parts = [m.value for m in at.markdown]
    parts += [c.value for c in at.caption]
    parts += [w.value for w in at.warning]
    parts += [i.value for i in at.info]
    return "\n".join(str(p) for p in parts)


@pytest.fixture(scope="module")
def rendered():
    return _render(METHODS)


def test_the_page_renders_every_module_with_its_code_link(rendered):
    text = _all_text(rendered)
    for row in MODULES:
        assert row.title in text, row.title
        assert f"cognisom/blob/main/{row.source}" in text, row.source


def test_the_page_renders_every_limitation_in_full(rendered):
    """Visible, not folded away.

    These were behind expanders first, and AppTest could not see them --
    which was the honest answer: nor could a reader who did not click.
    A page whose purpose is to say what is wrong must say it on the
    page, so the remedy is checked here too, not just the headline.
    """
    text = _all_text(rendered)
    for lim in LIMITATIONS:
        assert lim.what in text, lim.what
        assert lim.to_lift_it[:40] in text, (
            f"{lim.what!r} is listed without what would lift it")


def test_the_page_links_the_narrative_document(rendered):
    assert f"cognisom/blob/main/{NARRATIVE}" in _all_text(rendered), (
        "the page must link the narrative, since it is where the prose "
        "explanation lives")


def test_the_page_renders_every_kill_parameter_with_its_source(rendered):
    text = _all_text(rendered)
    for row in kill_rows():
        assert str(row["Parameter"]) in text, row["Parameter"]
        # The source sentence, not just the name: a table of numbers
        # without provenance is what this page exists to replace.
        assert str(row["Source"])[:40] in text, row["Parameter"]


def test_the_page_renders_every_signal_with_who_reads_it(rendered):
    """'Read by' is a sentence, and the section exists to carry it.

    It was an st.dataframe first, which clips a sentence to one line --
    hiding the column the section is for.
    """
    text = _all_text(rendered)
    for row in signal_rows():
        assert str(row["Field"]) in text, row["Field"]
        assert str(row["Read by"])[:50] in text, (
            f"{row['Field']} is listed without who reads it")
    assert "Read by nothing" in text, (
        "the page no longer marks any field as consumed by nothing; "
        "re-measure before assuming everything got wired")


def test_the_page_says_which_modules_drive_nothing(rendered):
    """The distinction the page is built around must survive rendering."""
    text = _all_text(rendered)
    assert "Reported only" in text
    for row in modules_by_wiring(REPORTS_ONLY):
        assert row.caveat[:40] in text, row.key


def test_the_page_states_the_one_external_validation_and_the_null(rendered):
    text = _all_text(rendered)
    assert "592 TCGA" in text, (
        "the page must name the one falsifiable test that passed")
    assert "p = 0.55" in text, (
        "the page must name the pre-specified outcome test that was null; "
        "a methods page that reports only its successes is a brochure")


def test_the_page_starts_no_simulation_and_reads_no_patient(rendered):
    source = METHODS.read_text()
    for forbidden in ("EngineRunner", "streamlit_page_gate", "run_simulation"):
        assert forbidden not in source, (
            f"{forbidden} on a public page: it must stay read-only")


def test_the_route_is_public_and_the_page_calls_no_gate():
    app = (REPO / "cognisom" / "dashboard" / "app.py").read_text()
    assert '"methods": ("47_methods.py"' in app
    assert '_query.get("page") in _PUBLIC_PAGES' in app


def test_the_page_survives_the_exec_that_actually_serves_it(tmp_path):
    """The only way a visitor reaches this page.

    ``app.py`` serves a public route with
    ``exec(compile(page.read_text(), ...))``, so the page runs with
    ``__file__`` pointing at app.py and not at itself. A page that
    resolves an asset from ``__file__`` works perfectly under AppTest
    and breaks for every real visitor -- which is why 44_roadmap.py
    resolves its video directory from the package instead.

    Rendering the page directly, as the tests above do, cannot catch
    that. This reproduces the exec.
    """
    shim = tmp_path / "exec_shim.py"
    shim.write_text(
        "from pathlib import Path\n"
        f"_p = Path({str(METHODS)!r})\n"
        'exec(compile(_p.read_text(), str(_p), "exec"))\n')
    at = _render(shim)
    text = _all_text(at)
    assert "How the simulation works" in str(at.title[0].value)
    for row in MODULES:
        assert row.title in text, (
            f"{row.title!r} is missing when the page is served the way "
            "app.py serves it, though it renders when run directly")


def test_the_page_stamps_the_commit_it_was_generated_from():
    page = METHODS.read_text()
    assert "COGNISOM_GIT_SHA" in page
    assert "/commit/" in page
