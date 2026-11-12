"""The public cells page says what the engine runs, and labels its guesses.

``/?page=cells`` tells a visitor how big each simulated cell is, what it
does and where every number comes from. A page like that is worth exactly
as much as its agreement with the code, so these tests re-read the sources
the catalogue claims to read and fail if a value on the page is a second
copy that has drifted, or if a guess is presented as a measurement.
"""

from __future__ import annotations

import ast
import contextlib
import io
from pathlib import Path

import pytest

from cognisom.dashboard import cell_catalog as cc

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "cognisom" / "dashboard" / "_pages" / "46_cells.py"
APP = ROOT / "cognisom" / "dashboard" / "app.py"


@pytest.fixture(scope="module")
def entries():
    return cc.build_catalog()


def _by_key(entries):
    return {e.key: e for e in entries}


def test_every_cell_type_the_engine_runs_has_an_entry(entries):
    from cognisom.modules.immune_module import LINEAGE_ENTITY
    assert set(_by_key(entries)) == {"normal", "cancer", *LINEAGE_ENTITY}


def test_the_sizes_are_the_ones_the_renderer_draws(entries):
    from cognisom.visualization import lineage_appearance as la
    looks = la.appearance_table()
    for e in entries:
        assert e.diameter_um == pytest.approx(looks[e.key]["diameter_um"])
        # An uncited size must not be shown as a measurement.
        assert (e.diameter_status == "measured") == bool(looks[e.key]["cited"])


def test_the_counts_are_the_default_runs(entries):
    from cognisom.dashboard.engine_runner import EngineRunner
    runner = EngineRunner()
    cells, immune = (runner.module_configs[k] for k in ("cellular", "immune"))
    got = {e.key: e.count for e in entries}
    assert got == {
        "normal": cells["n_normal_cells"], "cancer": cells["n_cancer_cells"],
        "T_cell": immune["n_t_cells"], "NK_cell": immune["n_nk_cells"],
        "macrophage": immune["n_macrophages"],
    }


def test_the_kill_numbers_carry_the_calibration_files_own_status(entries):
    """An ESTIMATE in kill_calibration must not read as measured here."""
    from cognisom.engine.py.immune import kill_calibration as kc
    report = kc.calibration_report()
    shown = {p.source: p for e in entries for p in e.parameters}
    for row in report.values():
        parameter = shown[row["source"]]
        assert parameter.status == cc._kill_status(row["status"])
    estimates = [r for r in report.values() if r["status"] == "ESTIMATE"]
    assert estimates, "the guard is vacuous if nothing is an estimate"


def test_the_contact_times_are_the_modules(entries):
    from cognisom.modules.immune_module import ImmuneModule
    with contextlib.redirect_stdout(io.StringIO()):
        immune = ImmuneModule({})
    for key in ("T_cell", "NK_cell", "macrophage"):
        values = {p.name: p for p in _by_key(entries)[key].parameters}
        lethal = values["Contact that kills, mean"]
        assert lethal.value == f"{immune._contact_minutes_for(key, True):g} min"
    # The macrophage has nothing in the library, and the page says so.
    mac = {p.name: p for p in _by_key(entries)["macrophage"].parameters}
    assert mac["Contact that kills, mean"].status == "placeholder"


def test_the_division_clock_is_shown_as_compressed(entries):
    """12 h and 90 h are a compressed scale. Shown bare they would read
    as a claim that a prostate tumour doubles twice a day."""
    for key in ("normal", "cancer"):
        row = next(p for p in _by_key(entries)[key].parameters
                   if p.name == "Time between divisions")
        assert row.status == "model choice"
        assert "Compressed" in row.source and "days" in row.source


def test_every_cell_states_what_is_missing(entries):
    for e in entries:
        assert e.missing, f"{e.name} claims no weakness"
        assert e.does and e.simulated


def test_every_status_is_explained(entries):
    used = {e.diameter_status for e in entries} | {
        p.status for e in entries for p in e.parameters}
    assert used <= set(cc.STATUS_MEANING)
    assert sum(cc.status_counts(entries).values()) == len(entries) + sum(
        len(e.parameters) for e in entries)


def test_the_page_types_no_numbers_of_its_own():
    """The page renders the catalogue. A digit in a string literal there
    would be a value that bypasses the engine."""
    tree = ast.parse(PAGE.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if node.lineno < 16:           # the module docstring
                continue
            stripped = (text.replace("0.85em", "").replace("0.45em", "")
                        .replace("50%", "").replace("255", ""))
            is_format_spec = len(text) <= 4 and text.startswith((".", "%"))
            if any(ch.isdigit() for ch in stripped) and "%." not in text \
                    and "{" not in text and "U0001" not in repr(text) \
                    and not is_format_spec:
                offenders.append((node.lineno, text[:60]))
    assert not offenders, offenders


def test_the_page_is_served_without_a_login():
    src = APP.read_text()
    assert '"cells": ("46_cells.py"' in src
    assert '_query.get("page") in _PUBLIC_PAGES' in src
    assert '"/?page=cells"' in src
