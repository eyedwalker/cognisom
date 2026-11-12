"""A run can be put on the measured clock, in a tissue that is mostly tumour.

Why this file exists
--------------------
A viewer of the public demo asked whether it was realistic for everything
to happen in 24 hours. It was not, for two reasons that are both choices
of the presets rather than of the biology:

* cells divide on a clock compressed ~93x (cancer every 12 h; measured
  turnover is ~46 days), while killing and drug binding run in real time;
* the tumour is a nodule of ~4% of the tissue and the infiltrate is sized
  for the whole tissue, so about one effector per cancer cell converges on
  it and clears it in a day or two.

Measured on the demo patient (section, Baseline, seed 7, NK limit on):
cleared by hour 48 at the presets' values; on the measured clock with the
tumour at 74% of the tissue, 1,500 -> 1,315 untreated over a WEEK and
1,500 -> 819 on enzalutamide, tumour PSA down 61% in the first day.

Both are now named options, off by default so no preset moves.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from cognisom.dashboard.engine_runner import CLOCKS, EngineRunner
from cognisom.modules.cellular_module import MEASURED_DIVISION_HOURS


def _cells(runner):
    return runner.module_configs["cellular"]


def test_the_default_is_the_clock_and_tumour_every_preset_was_pinned_on():
    runner = EngineRunner()
    assert runner.clock == "compressed" and runner.tumour_fraction is None
    assert "division_time_cancer" not in _cells(runner)
    assert _cells(runner) == _cells(EngineRunner(clock="compressed"))


def test_the_measured_clock_keeps_the_measured_ratio():
    """7.5 to 1 is the sourced part (Meyer 1982, PMID 6185700). A measured
    clock that broke it would be a third, unsourced, timescale."""
    runner = EngineRunner(clock="measured")
    normal = _cells(runner)["division_time_normal"]
    cancer = _cells(runner)["division_time_cancer"]
    assert (normal, cancer) == (MEASURED_DIVISION_HOURS["normal"],
                                MEASURED_DIVISION_HOURS["cancer"])
    assert normal / cancer == pytest.approx(7.5, rel=0.001)
    assert cancer / 24.0 == pytest.approx(46.3, abs=0.1)    # days


def test_the_measured_clock_reaches_the_cells():
    runner = EngineRunner(clock="measured", duration=0.1)
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
    cellular = runner.engine.modules["cellular"]
    assert cellular.division_time_cancer == MEASURED_DIVISION_HOURS["cancer"]
    assert cellular.division_time_normal == MEASURED_DIVISION_HOURS["normal"]


def test_an_explicit_override_still_wins_over_the_clock():
    runner = EngineRunner(
        clock="measured", overrides={"cellular": {"division_time_cancer": 500.0}})
    assert _cells(runner)["division_time_cancer"] == 500.0


@pytest.mark.parametrize("tissue", ["section", "slab", "ring"])
def test_a_tumour_fraction_changes_who_the_cells_are_and_not_how_many(tissue):
    """The total is kept so the tissue box, vessels and infiltrate are the
    preset's: the one thing that changes is the effector-to-target ratio."""
    preset = EngineRunner(tissue=tissue)
    runner = EngineRunner(tissue=tissue, tumour_fraction=0.65)
    total = _cells(preset)["n_normal_cells"] + _cells(preset)["n_cancer_cells"]
    assert _cells(runner)["n_normal_cells"] + _cells(runner)["n_cancer_cells"] == total
    assert _cells(runner)["n_cancer_cells"] == round(total * 0.65)
    assert runner.module_configs["immune"] == preset.module_configs["immune"]
    assert runner.module_configs["vascular"] == preset.module_configs["vascular"]


def test_the_tumour_fraction_is_really_seeded():
    runner = EngineRunner(tumour_fraction=0.65, duration=0.1)
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
    cells = runner.engine.modules["cellular"].cells.values()
    cancer = sum(1 for c in cells if c.cell_type == "cancer")
    assert cancer / len(cells) == pytest.approx(0.65, abs=0.01)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
def test_a_tumour_fraction_that_is_not_a_fraction_is_refused(bad):
    with pytest.raises(ValueError, match="between 0 and 1"):
        EngineRunner(tumour_fraction=bad)


def test_an_unknown_clock_is_refused():
    assert CLOCKS == ("compressed", "measured")
    with pytest.raises(ValueError, match="clock must be one of"):
        EngineRunner(clock="fast")


def test_a_resumed_run_still_says_which_clock_it_was_on(tmp_path):
    from cognisom.core.checkpoint import load_checkpoint, save_checkpoint
    runner = EngineRunner(clock="measured", tumour_fraction=0.65,
                          duration=0.2, tissue="ring")
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
        path = str(tmp_path / "run.ckpt")
        save_checkpoint(runner, path)
        resumed = load_checkpoint(path)
    assert resumed.clock == "measured"
    assert resumed.tumour_fraction == 0.65
    assert (resumed.engine.modules["cellular"].division_time_cancer
            == MEASURED_DIVISION_HOURS["cancer"])


def test_the_recorder_can_ask_for_both():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "record_demo_run.py").read_text()
    assert "clock=args.clock" in src
    assert "tumour_fraction=args.tumour_fraction" in src
    assert '"clock": args.clock' in src
