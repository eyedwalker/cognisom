"""T cells divide when they have seen their antigen, and not otherwise.

Until this, no T cell in the engine ever divided. That left out the
mechanism PD-1 blockade works through -- a CD28-dependent burst of CD8
proliferation (Kamphorst 2017, PMID 28280249, already cited by the
engine's CD28 relief term) -- so blockade could change how hard existing
cells killed and never how many there were.

What is pinned here is the SHAPE of the program rather than its outcome,
because the outcome is not measurable: every number behind it is a mouse
lymph node in acute infection, and this engine runs established tumour
tissue with no priming compartment. The shape is what the literature
actually settles:

* the trigger is the FIRST recognition and nothing after it, and division
  continues without further antigen (PMID 11323695, PMID 11323696);
* an exhausted clone is spent and does not divide (PMID 28514453);
* suppression does NOT slow division -- Chen 2005 (PMID 15623559), the
  source of this engine's Treg term, measured expansion unchanged while
  cytotoxicity fell, so a Treg gate here would contradict its own paper;
* the program belongs to the clone, so daughters inherit its progress.

And two properties that keep it from disturbing what is already measured:
it is off by default, and with it off a run is bit-identical.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pytest

from cognisom.core.event_bus import EventTypes
from cognisom.engine.py.immune.tcell_expansion import (
    DIVISION_HOURS, PROGRAM_DIVISIONS, divisions_due, expansion_report,
    fold_expansion,
)

CYCLE = DIVISION_HOURS.value
PROGRAM = int(PROGRAM_DIVISIONS.value)


# ── the program's shape ──────────────────────────────────────────────

def test_a_t_cell_that_never_recognised_anything_never_divides():
    """The whole point: this is antigen-driven, not a birth rate."""
    for now in (0.0, CYCLE, 10 * CYCLE, 1000.0):
        assert divisions_due(None, 0, now) == 0


def test_nothing_is_due_before_one_cycle_has_passed():
    assert divisions_due(0.0, 0, CYCLE - 1e-9) == 0
    assert divisions_due(0.0, 0, CYCLE) == 1


def test_one_division_falls_due_per_cycle():
    for k in range(1, PROGRAM + 1):
        assert divisions_due(0.0, 0, k * CYCLE) == k


def test_the_program_stops_at_its_cap():
    assert divisions_due(0.0, 0, 50 * CYCLE * PROGRAM) == PROGRAM
    # and a cell that has already done them all owes nothing
    assert divisions_due(0.0, PROGRAM, 1e6) == 0


def test_divisions_already_done_are_not_owed_again():
    assert divisions_due(0.0, 2, 3 * CYCLE) == 1
    assert divisions_due(0.0, 3, 3 * CYCLE) == 0


def test_the_program_needs_no_antigen_argument():
    """Kaech & Ahmed: the program 'could not be interrupted' and daughters
    'continued to divide ... in the absence of further antigenic
    stimulation'. A signature that took standing antigen would produce
    expansion only where killing was already happening, which is the
    opposite of what was measured."""
    import inspect
    params = set(inspect.signature(divisions_due).parameters)
    assert not params & {"antigen", "target", "in_contact", "engaged",
                         "presentation", "has_target"}, params


def test_the_clock_is_the_only_thing_that_advances_it():
    engaged_at, done = 3.0, 1
    a = divisions_due(engaged_at, done, 3.0 + 2 * CYCLE)
    b = divisions_due(engaged_at, done, 3.0 + 2 * CYCLE)
    assert a == b == 1


@pytest.mark.parametrize("divisions,fold", [(0, 1), (1, 2), (3, 8), (7, 128)])
def test_fold_expansion_is_a_doubling(divisions, fold):
    assert fold_expansion(divisions) == fold


# ── the numbers carry their provenance ───────────────────────────────

def test_both_numbers_are_labelled_estimates_with_a_source():
    """Neither is sourced FOR THIS SETTING, and saying otherwise would be
    the overclaim this repository exists to avoid."""
    for p in (DIVISION_HOURS, PROGRAM_DIVISIONS):
        assert p.status == "ESTIMATE", (p.status, p.source)
        assert "PMID" in p.source
        assert p.low <= p.value <= p.high


def test_the_default_cycle_is_the_slowest_measured_not_the_fastest():
    """Yoon measured 2 h for influenza and ~6 h for peptide+adjuvant. The
    engine's tissue is further from an acutely infected lymph node than
    either, so the slow end is the honest default; taking 2 h would make
    expansion three times brisker than any defensible reading."""
    assert DIVISION_HOURS.value == 6.0
    assert DIVISION_HOURS.low == 2.0


def test_the_report_says_suppression_does_not_apply_and_why():
    r = expansion_report(enabled=False)
    assert r["suppression_applies"] is False
    assert "15623559" in r["suppression_reason"]
    assert r["exhaustion_applies"] is True
    assert "no intratumoural" in r["caveat"]


# ── wiring: the module ───────────────────────────────────────────────

def _run(hours=12.0, scenario="Immunotherapy", seed=1, **immune):
    from cognisom.dashboard.engine_runner import EngineRunner
    np.random.seed(seed)
    runner = EngineRunner(dt=0.05, duration=hours, scenario=scenario,
                          record_interval=1000, snapshot_interval_hours=None,
                          overrides={"immune": immune} if immune else None)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        runner.build()
        runner.run()
    return runner


def _immune(runner):
    return runner.engine.modules["immune"]


def test_expansion_is_off_unless_asked_for():
    from cognisom.modules.immune_module import ImmuneModule
    assert ImmuneModule({}).tcell_expansion_enabled is False


def test_a_run_always_reports_whether_its_t_cells_could_divide():
    """An absent key would read as 'this build has no such mechanism'."""
    runner = _run(hours=0.5)
    report = _immune(runner).get_state()["tcell_expansion"]
    assert report["enabled"] is False
    assert report["divisions"] == 0
    assert report["division_hours"]["in_use"] == CYCLE


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_with_expansion_off_nothing_is_drawn_and_no_cell_is_added():
    """The bit-identity guarantee, checked at its mechanism: the
    expansion stream is never advanced and no daughter appears. This is
    what lets every pinned outcome in the suite stay valid."""
    runner = _run(hours=6.0, tcell_expansion_enabled=False)
    module = _immune(runner)
    fresh = np.random.default_rng(0x54434558)
    assert module._expansion_rng.bit_generator.state == \
        fresh.bit_generator.state, "the expansion stream was drawn from"
    assert module.total_tcell_divisions == 0
    assert all(c.divisions_done == 0 for c in module.immune_cells.values())


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_turning_it_on_multiplies_the_t_cells_that_recognised():
    """Measured 2026-11-08, 12 h Immunotherapy, seed 1: T cells 51 -> 99
    on 48 divisions from 48-49 founders. The ASSERTION is the direction
    and that divisions come only from founders; the numbers are recorded
    beside it. Re-measure if they move, do not re-pin."""
    off = _immune(_run(tcell_expansion_enabled=False))
    on = _immune(_run(tcell_expansion_enabled=True))

    def n_t(module):
        return sum(1 for c in module.immune_cells.values()
                   if c.cell_type == "T_cell")

    report = on.get_state()["tcell_expansion"]
    assert report["enabled"] is True
    assert report["divisions"] > 0
    assert n_t(on) > n_t(off), (n_t(on), n_t(off))
    # Every division came from a cell that had recognised an antigen.
    assert report["founders"] > 0
    assert report["divisions"] <= report["founders"] * PROGRAM
    # Only T cells divided.
    assert all(c.divisions_done == 0 for c in on.immune_cells.values()
               if c.cell_type != "T_cell")


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_more_t_cells_is_not_proportionally_more_killing():
    """Doubling the T cells did NOT double the killing: 100 -> 108 kills
    in the run above. That is the diluted-T-cell result this engine keeps
    reproducing -- NK cells and macrophages do most of the killing and
    cannot see the escape trait -- and it is why expansion is not a
    shortcut to a bigger treatment effect."""
    off = _immune(_run(tcell_expansion_enabled=False)).get_state()
    on = _immune(_run(tcell_expansion_enabled=True)).get_state()
    assert on["total_kills"] < 2 * off["total_kills"], (
        on["total_kills"], off["total_kills"])


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_suppressing_the_t_cells_does_not_slow_their_division():
    """Chen 2005 measured expansion UNCHANGED while cytotoxicity was
    suppressed, and that paper is where this engine's Treg term comes
    from. If a Treg gate ever reaches division, this fails."""
    plain = _immune(_run(tcell_expansion_enabled=True))
    suppressed = _immune(_run(tcell_expansion_enabled=True,
                              treg_fraction=0.5))
    a = plain.get_state()["tcell_expansion"]
    b = suppressed.get_state()["tcell_expansion"]
    # Founders can differ (suppression changes who survives to engage),
    # so the invariant is divisions PER FOUNDER, not the raw count.
    assert b["founders"] > 0 and a["founders"] > 0
    per_a = a["divisions"] / a["founders"]
    per_b = b["divisions"] / b["founders"]
    assert per_b == pytest.approx(per_a, rel=0.35), (per_a, per_b)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_a_division_is_not_announced_as_a_tissue_cell_division():
    """CELL_DIVIDED means a cell in CellularModule's dict divided, and
    ar_signaling, cell_mechanics, circadian, epigenetic and molecular all
    subscribe to it to inherit a clock or fork a view for the daughter.
    An immune cell has its own id space, so a T-cell division sent down
    that channel would have each of them look up a daughter that is not
    theirs."""
    runner = _run(tcell_expansion_enabled=True)
    kinds = [e for e in runner.event_log
             if e["event"] == EventTypes.TCELL_DIVIDED]
    assert kinds, "no T-cell division was announced at all"
    assert all(e["data"]["cell_type"] == "T_cell" for e in kinds)
    divided_as_tissue = [
        e for e in runner.event_log
        if e["event"] == EventTypes.CELL_DIVIDED
        and e["data"].get("cell_type") == "T_cell"]
    assert not divided_as_tissue, (
        f"{len(divided_as_tissue)} T-cell divisions went out as "
        "CELL_DIVIDED; five modules would treat them as tissue cells")


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_a_lineage_completes_the_program_rather_than_restarting_it():
    """The program belongs to the clone (Kaech & Ahmed), so a daughter
    inherits its progress. If each daughter started a fresh seven, one
    founder would expand without bound."""
    on = _immune(_run(hours=12.0, tcell_expansion_enabled=True))
    done = [c.divisions_done for c in on.immune_cells.values()
            if c.cell_type == "T_cell"]
    assert done and max(done) <= PROGRAM, max(done)
