"""A run can say that only some T cells recognise the tumour.

Why this file exists
--------------------
Every T cell consulted the whole receptor repertoire and took its best
clone, so in every run every T cell was tumour-specific. Inside human
tumours most are not: Scheper 2019 (PMID 30510250) tested the receptors of
intratumoral CD8 T cells against autologous tumour and found "a capacity
to recognize autologous tumor is limited to approximately 10%", with none
found in two of four patients (ovarian and colorectal; abstract read).

It matters for the course of a run. Measured on the demo patient (section,
1,500 of 2,040 cells cancer, measured clock, one week, seed 7), untreated:
1,500 -> 1,315 with every T cell reactive (107 T-cell kills) and
1,500 -> 1,413 at 10% (2 T-cell kills), flat by the last day.

Opt-in, because it is an estimate from other tumour types: no preset
changes, and the default is the behaviour every pin was taken on.
"""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from cognisom.engine.py.immune import kill_calibration as kc
from cognisom.modules.immune_module import ImmuneModule


def _module(**config) -> ImmuneModule:
    with contextlib.redirect_stdout(io.StringIO()):
        module = ImmuneModule(config)
    module.emit_event = lambda *a, **k: None
    return module


def _t(module):
    return module.immune_cells[module.add_immune_cell([50, 50, 50], "T_cell")]


def test_by_default_every_t_cell_is_reactive_as_it_always_was():
    module = _module()
    assert module.tcell_tumour_reactive_fraction is None
    assert all(_t(module).tumour_reactive for _ in range(200))


def test_the_stated_share_is_the_share_that_is_reactive():
    module = _module(tcell_tumour_reactive_fraction=0.10)
    reactive = [_t(module).tumour_reactive for _ in range(5_000)]
    assert np.mean(reactive) == pytest.approx(0.10, abs=0.015)
    state = module.get_state()
    assert state["n_tcells_tumour_reactive"] == sum(reactive)
    assert state["tcell_tumour_reactive_fraction"] == 0.10


def test_a_bystander_recognises_nothing_however_good_the_target():
    """The gate is before the repertoire is consulted at all."""
    module = _module(tcell_tumour_reactive_fraction=0.0)
    cell = _t(module)
    assert cell.tumour_reactive is False

    def boom(*a, **k):
        raise AssertionError("a bystander consulted the repertoire")
    module._tcr_repertoire = SimpleNamespace(best_match=boom)
    target = SimpleNamespace(mhc1_displayed_peptides=[object()],
                             mhc1_expression=1.0)
    assert module._recognize(cell, target) == (False, None)


def test_only_t_cells_have_bystanders():
    module = _module(tcell_tumour_reactive_fraction=0.0)
    for lineage in ("NK_cell", "macrophage"):
        cell = module.immune_cells[module.add_immune_cell([50, 50, 50], lineage)]
        assert cell.tumour_reactive is True


def test_a_daughter_inherits_reactivity_because_it_is_the_clones(monkeypatch):
    module = _module(tcell_tumour_reactive_fraction=0.5)
    src = __import__("inspect").getsource(ImmuneModule)
    assert "daughter.tumour_reactive = parent.tumour_reactive" in src


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_a_share_that_is_not_a_share_is_refused(bad):
    with pytest.raises(ValueError, match="between 0 and 1"):
        _module(tcell_tumour_reactive_fraction=bad)


def test_drawing_reactivity_does_not_touch_the_shared_random_stream():
    np.random.seed(9)
    expected = np.random.random()
    module = _module(tcell_tumour_reactive_fraction=0.1)
    np.random.seed(9)
    _t(module)
    assert np.random.random() == expected


def test_the_number_says_it_is_an_estimate_from_other_tumours():
    row = kc.calibration_report()["tcell_tumour_reactive_fraction"]
    assert row["status"] == "ESTIMATE"
    assert "30510250" in row["source"] and "not prostate" in row["source"]
    assert row["range"][0] == 0.0, "two of four patients had none"
