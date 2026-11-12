"""A macrophage does not eat a tumour cell that nothing has marked.

Why this file exists
--------------------
Macrophages phagocytosed on 5% of contacts, unconditionally. The number
had a source for its DIRECTION only -- CD47 engaging SIRP-alpha blocks
phagocytosis (Majeti 2009, Willingham 2012) -- and the engine applied it
to tumour cells with CD47 fully intact and no antibody anywhere. On the
realistic week-long run that came to roughly half a kill per macrophage
per day, 88-147 of some 150-200 kills, and it made an UNTREATED tumour
shrink. A simulation whose untreated arm cures the patient cannot be
used to ask what a drug adds.

Dooling 2023 (PMID 37095318, PMC10791169) measures the condition, in
full text: "Growth of WT tumouroids is likewise unaffected by anti-Tyrp1
with or without added macrophages", and "only for the maximal
phagocytosis condition of CD47 KO and anti-Tyrp1 opsonization is the
exponential growth of tumouroids reversed and tumouroids eradicated".
Both halves are needed -- something marking the target, and CD47 not
getting through -- so the licence is a PRODUCT of the two, and what is
left with neither is zero.

What these tests pin is the structure, not the magnitude: the product,
the two identities (nothing marked -> no phagocytosis; fully marked and
unblocked -> the calibrated rate back to the bit), and that the old
behaviour is recoverable from configuration. 0.05 per contact stays an
ESTIMATE and is asserted only where it is the thing under test.
"""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from cognisom.engine.py.immune import kill_calibration as kc
from cognisom.engine.py.immune.macrophage_phagocytosis import (
    phagocytic_licence,
)
from cognisom.modules.immune_module import ImmuneModule


def _module(**config) -> ImmuneModule:
    with contextlib.redirect_stdout(io.StringIO()):
        module = ImmuneModule(config)
    module.emit_event = lambda *a, **k: None      # no bus in a unit test
    return module


def _macrophage(module: ImmuneModule):
    cell_id = module.add_immune_cell([50, 50, 50], "macrophage")
    return module.immune_cells[cell_id]


def _tumour_cell(pdl1: float = 0.0):
    """A tumour cell as the engine has it: no CD47 state, PD-L1 untouched."""
    return SimpleNamespace(
        cell_id=10_001, position=np.array([50.0, 50.0, 50.0]),
        cell_type="cancer", pdl1_expression=pdl1, mhc1_expression=0.4,
        alive=True,
    )


# ── the licence is the product the paper describes ───────────────────

def test_neither_half_alone_licenses_phagocytosis():
    # Opsonised but CD47 intact: Dooling's anti-Tyrp1-on-WT arm, where
    # tumouroid growth was unaffected.
    assert phagocytic_licence(opsonisation=1.0, cd47_block=0.0) == 0.0
    # CD47 gone but nothing marking the cell: their CD47 KO without
    # anti-Tyrp1, which also did not reverse growth.
    assert phagocytic_licence(opsonisation=0.0, cd47_block=1.0) == 0.0
    # Both: the one condition that eradicated tumouroids.
    assert phagocytic_licence(opsonisation=1.0, cd47_block=1.0) == 1.0


def test_the_licence_is_a_product_and_not_a_sum():
    # Half-marked, half-blocked is a quarter of the rate. A sum would
    # make it the whole of it, which would say either half suffices.
    assert phagocytic_licence(opsonisation=0.5, cd47_block=0.5) == 0.25
    assert phagocytic_licence(
        opsonisation=0.5, cd47_block=0.5) < 0.5 + 0.5


def test_the_unprompted_share_is_the_floor_and_defaults_to_zero():
    assert kc.MACROPHAGE_UNPROMPTED_FRACTION.value == 0.0
    # With a floor, an unmarked target is eaten at exactly that share,
    # and a fully marked one still at the full rate.
    assert phagocytic_licence(unprompted=0.1) == pytest.approx(0.1)
    assert phagocytic_licence(
        opsonisation=1.0, cd47_block=1.0, unprompted=0.1) == 1.0


def test_the_floor_is_ours_and_says_so():
    row = kc.calibration_report()["macrophage_unprompted_fraction"]
    assert row["status"] == "ESTIMATE", (
        "the paper says 'unaffected', which supports the value but is "
        "not a measurement of it"
    )
    assert "37095318" in row["source"]
    assert row["range"] == (0.0, 0.1)
    # The per-contact rate is now defined against a MARKED target, so
    # its own description has to say so.
    rate = kc.calibration_report()["macrophage_kill_per_contact"]
    assert "CD47" in rate["source"]
    assert rate["status"] == "ESTIMATE"


# ── the engine ───────────────────────────────────────────────────────

def test_a_macrophage_offered_an_unmarked_tumour_cell_does_not_eat_it():
    module = _module()
    macrophage, target = _macrophage(module), _tumour_cell()
    assert module.macrophage_licence() == 0.0
    assert module._macrophage_kill_probability(target, macrophage) == 0.0
    # Recognition itself fails, so the cell does not even engage: it
    # keeps patrolling instead of standing on a target it cannot eat.
    recognised, _ = module._recognize(macrophage, target)
    assert recognised is False


def test_marking_the_tumour_restores_exactly_the_calibrated_rate():
    module = _module(tumour_opsonisation=1.0, cd47_block=1.0)
    macrophage, target = _macrophage(module), _tumour_cell()
    assert module.macrophage_licence() == 1.0
    # PD-L1 is 0 on an untouched cell and polarization 0.0 at M0, so
    # both existing gates are exact identities and what is left is the
    # calibrated per-contact number itself.
    assert module._macrophage_kill_probability(target, macrophage) == (
        pytest.approx(kc.MACROPHAGE_KILL_PER_CONTACT.value))
    assert module._recognize(macrophage, target)[0] is True


def test_the_old_behaviour_is_recoverable_from_configuration():
    # One setting returns the engine to what it did before this change,
    # for anyone re-running an earlier result.
    module = _module(macrophage_unprompted_fraction=1.0)
    macrophage, target = _macrophage(module), _tumour_cell()
    assert module._macrophage_kill_probability(target, macrophage) == (
        pytest.approx(kc.MACROPHAGE_KILL_PER_CONTACT.value))


def test_the_two_gates_still_compose_on_top_of_the_licence():
    # PD-L1 on a licensed target still suppresses phagocytosis
    # (Gordon 2017), and blockade still relieves it.
    marked = dict(tumour_opsonisation=1.0, cd47_block=1.0)
    plain = _module(**marked)
    blocked = _module(checkpoint_block=1.0, **marked)
    hot = _tumour_cell(pdl1=1.0)
    suppressed = plain._macrophage_kill_probability(hot, _macrophage(plain))
    rescued = blocked._macrophage_kill_probability(hot, _macrophage(blocked))
    assert suppressed < rescued == pytest.approx(
        kc.MACROPHAGE_KILL_PER_CONTACT.value)


def test_a_run_reports_what_licensed_its_macrophages():
    module = _module(tumour_opsonisation=0.5, cd47_block=0.5)
    _macrophage(module)
    state = module.get_state()
    assert state["tumour_opsonisation"] == 0.5
    assert state["cd47_block"] == 0.5
    assert state["macrophage_licence"] == pytest.approx(0.25)


@pytest.mark.parametrize("bad", [
    {"tumour_opsonisation": 1.5},
    {"cd47_block": -0.1},
    {"macrophage_unprompted_fraction": 2.0},
])
def test_a_fraction_outside_zero_to_one_is_refused(bad):
    with pytest.raises(ValueError):
        _module(**bad)
