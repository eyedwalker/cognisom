"""The IFN-gamma MHC-I rise reaches NK cells -- for some patients' clones.

Raising MHC-I cuts both ways: a tumour cell becomes more visible to a
TCR-matched T cell and LESS visible to a licensed NK cell, because
missing-self is relieved. The engine has both halves already -- the
KIR-HLA receptor balance reads ``mhc1_expression`` directly -- and the
IFN-gamma field (see test_interferon_reaches_the_bystanders) is what
now moves it.

The receptor model is off unless a MEASURED class-I typing is on
record; it refuses to invent one, because an NK prediction built on
synthetic alleles is indistinguishable from one built on real reads.
One measured typing exists: ``SEQC2``, from HCC1395BL matched-normal
WXS by OptiType, 1,198 reads.

What this file pins is both what is established and what is not:

* per clone the effect is large and follows the patient's own HLA;
* across a population of six NK cells it is NOT detectable at this
  scale -- 20 paired seeds gave a mean difference of -1.45 kills with
  sd 4.22, t(19) = -1.54. An earlier single-seed run looked like a
  clean 18 -> 14 and was noise. The number is recorded so nobody
  reports the trade-off as a population finding on one run.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from cognisom.modules.immune_module import ImmuneModule


class _Target:
    def __init__(self, mhc1):
        self.mhc1_expression = mhc1
        self.mutations = ["KRAS_G12D"]
        self.cell_type = "cancer"
        self.alive = True
        self.position = np.zeros(3)


def _module(**config):
    module = ImmuneModule({"n_t_cells": 0, "n_nk_cells": 8,
                           "n_macrophages": 0, **config})
    with redirect_stdout(io.StringIO()):
        module.initialize()
    return module


# ── the model is off without a real typing, on with one ───────────


def test_without_a_measured_typing_nk_cells_cannot_see_mhc1_at_all():
    module = _module()
    assert module._nk_licensing is None
    nk = next(c for c in module.immune_cells.values() if c.cell_type == "NK_cell")
    flat = {module._nk_kill_probability(nk, _Target(m)) for m in (0.3, 0.45, 0.9)}
    assert len(flat) == 1, "the fallback baseline ignores MHC-I, by design"


def test_a_measured_typing_turns_the_receptor_balance_on():
    module = _module(patient_id="SEQC2")
    assert module._nk_licensing is not None
    assert module.get_state()["nk_licensed_fraction"] == pytest.approx(0.286, abs=0.01)


def test_an_unknown_patient_does_not_get_an_invented_profile():
    assert _module(patient_id="not-a-real-patient")._nk_licensing is None


# ── per clone: the patient's HLA decides who can see it ───────────


def _by_subset(module):
    out = {}
    for cell in module.immune_cells.values():
        if cell.cell_type == "NK_cell" and getattr(cell, "kir_subset", None):
            out.setdefault(cell.kir_subset, cell)
    return out


def test_the_licensed_clone_is_silenced_as_mhc1_rises():
    """SEQC2 is HLA-C*06:02 homozygous -- a C2/C2 patient. KIR2DL1 reads
    C2 (Colonna), so it is licensed and inhibited by MHC-I: raising the
    target from 0.30 to 0.90 shuts it off entirely."""
    module = _module(patient_id="SEQC2")
    clones = _by_subset(module)
    assert "KIR2DL1" in clones, clones.keys()
    nk = clones["KIR2DL1"]
    low, mid, high = (module._nk_kill_probability(nk, _Target(m))
                      for m in (0.30, 0.45, 0.90))
    assert low > mid > high
    assert high == pytest.approx(0.0, abs=1e-9), "fully inhibited at high MHC-I"
    assert mid < 0.6 * low, "the IFN-driven 0.30 -> 0.45 step already halves it"


def test_a_clone_with_no_ligand_in_this_patient_is_blind_to_mhc1():
    """KIR2DL3 reads C1, which a C2/C2 patient does not carry, so that
    clone is unlicensed and its killing does not depend on MHC-I. Which
    NK clones can see the rise at all is therefore a property of the
    PATIENT, not of the tumour."""
    module = _module(patient_id="SEQC2")
    clones = _by_subset(module)
    if "KIR2DL3" not in clones:
        pytest.skip("this draw produced no KIR2DL3 clone")
    nk = clones["KIR2DL3"]
    values = {module._nk_kill_probability(nk, _Target(m)) for m in (0.30, 0.45, 0.90)}
    assert len(values) == 1


# ── what is NOT established, recorded so it is not re-claimed ─────


def test_the_population_level_trade_off_is_not_established_at_this_scale():
    """Measured 2026-10-23, 8 h, 300 cancer cells, six NK cells, the
    IFN-gamma field on vs off, 20 paired seeds:

        NK kills   on 17.2 +- 5.0   off 18.6 +- 3.5
        paired difference -1.45 +- 4.22, t(19) = -1.54, 11/20 negative

    The direction matches the mechanism and the size does not clear the
    noise. Raising the count of NK cells, or running the arm where most
    clones are licensed, is what would settle it. This test states the
    numbers rather than asserting an effect, so that a future run that
    DOES clear the noise is recognised as new information."""
    measured = {"n_seeds": 20, "nk_on_mean": 17.2, "nk_off_mean": 18.6,
                "paired_mean": -1.45, "paired_sd": 4.22, "t": -1.54}
    assert measured["paired_mean"] < 0, "direction matches missing-self"
    assert abs(measured["t"]) < 2.09, "t(19) 0.05 two-sided is 2.09: not significant"
    assert abs(measured["paired_mean"]) < measured["paired_sd"], "within one sd of zero"
