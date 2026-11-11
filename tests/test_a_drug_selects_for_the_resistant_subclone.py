"""A resistance variant in SOME cells, and a drug that selects for them.

Until this, a patient's mutation was in every cancer cell and a drug's
effect was one number for the whole tumour. So enzalutamide on the AR
F877L demo patient "failed" from the first step, everywhere at once.
That is not how resistance arrives: the variant is in some cells, the
drug blocks the rest, and the carriers outgrow them.

The claims tested are the mechanism's -- who carries the variant, what
the drug does to each kind of cell, and that the carrier share moves
under the one drug that selects and not under the two controls. The
measured numbers sit beside the claims; re-measure, do not re-pin.
"""

from __future__ import annotations

import contextlib
import io
import random
from types import SimpleNamespace

import numpy as np
import pytest

from cognisom.genomics.demo_patients import SU2C_AR_F877L_CLONALITY as CLONALITY
from cognisom.modules.cellular_module import CellularModule
from cognisom.workflow import parameter_bridge as pb
from cognisom.workflow.treatment import resolve_treatment

LABEL = "AR_F877L"
F877L = SimpleNamespace(variants=[
    SimpleNamespace(gene="AR", protein_change="p.F877L")])
PATIENT_MUTATIONS = [{"gene": "AR", "aa_change": "F877L", "label": LABEL,
                      "oncogenic": True}]

MODS = {"cellular": True, "spatial": True, "vascular": True,
        "ar_signaling": True, "immune": False, "lymphatic": False,
        "molecular": False, "epigenetic": False, "circadian": False,
        "morphogen": False, "cell_mechanics": False}


@pytest.fixture(scope="module")
def store():
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        return EntityStore()


# ── what the patient's own data allows ────────────────────────────


def test_the_allele_fraction_is_not_read_as_a_cell_fraction():
    """AR is amplified in this sample, so 0.233 counts gene copies. The
    recorded bound must be the one the copy number implies, and it must
    not be the allele fraction."""
    c = CLONALITY
    p, vaf = c["purity"], c["vaf"]
    copies = c["ar_copies_per_tumour_cell"]
    mutant = c["mutant_copies_per_tumour_cell"]
    # The two quantities are computed separately; they have to agree.
    assert vaf * (p * copies + 1 - p) / p == pytest.approx(mutant, abs=0.02)
    assert mutant / copies == pytest.approx(c["carrier_fraction_min"], abs=0.01)
    assert c["carrier_fraction_max"] == 1.0
    assert c["carrier_fraction_min"] > vaf, (
        "the lower bound on carriers is above the allele fraction; "
        "reading 0.233 as the subclone size understates it")
    assert c["measured_after_treatment"], (
        "the biopsy is post-exposure, so it bounds the fraction AFTER "
        "selection and not before")
    assert abs(c["purity"] - c["purity_from_autosomes"]) < 0.05


# ── who carries it ────────────────────────────────────────────────


def _cellular(fractions=None, mutations=PATIENT_MUTATIONS):
    config = {"patient_mutations": mutations}
    if fractions is not None:
        config["clonal_fractions"] = fractions
    with contextlib.redirect_stdout(io.StringIO()):
        return CellularModule(config)


@pytest.mark.parametrize("n,fraction", [(10, 0.1), (37, 0.26), (200, 0.05),
                                        (7, 0.5), (64, 1.0)])
def test_exactly_the_stated_share_of_founders_carry_it(n, fraction):
    module = _cellular({LABEL: fraction})
    carriers = sum(LABEL in module._founding_mutations() for _ in range(n))
    assert carriers == int(np.floor(n * fraction + 1e-9))


def test_without_a_fraction_every_founder_carries_it():
    module = _cellular()
    assert module.clonal_fractions == {}
    assert all(module._founding_mutations() == [LABEL] for _ in range(20))


def test_a_subclone_of_a_mutation_the_tumour_lacks_is_refused():
    with pytest.raises(ValueError, match="not one of this patient's"):
        _cellular({"AR_T878A": 0.2})


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_a_fraction_outside_zero_to_one_is_refused(bad):
    with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
        _cellular({LABEL: bad})


def test_a_subclone_needs_a_patient():
    from cognisom.dashboard.engine_runner import EngineRunner
    with pytest.raises(ValueError, match="no patient was given"):
        EngineRunner(tissue="ring", clonal_fractions={LABEL: 0.1})


# ── what the drug does to each kind of cell ───────────────────────


def test_the_drug_blocks_the_tumour_and_drives_the_carriers(store):
    t = resolve_treatment(["Enzalutamide"], F877L, store,
                          clonal_fractions={LABEL: 0.1})
    assert t.module_configs["ar_signaling"] == {
        "ar_transcriptional_block": 0.9,
        "ar_variant_activity": {
            "label": LABEL,
            "activity": pb.AR_AGONIST_TRANSCRIPTIONAL_ACTIVITY},
    }
    assert len(t.notes) == 1 and "10%" in t.notes[0]
    assert "NOT as a blockade" not in t.notes[0], (
        "the whole-tumour note is false once most cells are blocked")


def test_a_variant_in_every_cell_resolves_as_it_always_did(store):
    whole = resolve_treatment(["Enzalutamide"], F877L, store)
    stated = resolve_treatment(["Enzalutamide"], F877L, store,
                               clonal_fractions={LABEL: 1.0})
    assert stated.module_configs == whole.module_configs
    assert stated.notes == whole.notes


def test_darolutamide_blocks_carriers_and_the_rest_alike(store):
    t = resolve_treatment(["Darolutamide"], F877L, store,
                          clonal_fractions={LABEL: 0.1})
    assert t.module_configs["ar_signaling"] == {"ar_transcriptional_block": 0.9}


_PATIENT = []


def _demo_patient():
    """The real demo patient: the runner needs a full profile to check
    each mutation against its reference protein, which the two-field
    stand-in above cannot give it."""
    if not _PATIENT:
        import os
        os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
        from cognisom.genomics.demo_patients import (
            DEFAULT_DEMO_PATIENT, load_demo_patient)
        with contextlib.redirect_stdout(io.StringIO()):
            _PATIENT.append(load_demo_patient(DEFAULT_DEMO_PATIENT))
    return _PATIENT[0]


def _run(drug, fraction, duration):
    from cognisom.dashboard.engine_runner import EngineRunner
    np.random.seed(11)
    random.seed(11)
    runner = EngineRunner(
        tissue="ring", dt=0.05, duration=duration, scenario="Baseline",
        modules_enabled=MODS, patient=_demo_patient(),
        drugs=[drug] if drug else None,
        clonal_fractions={LABEL: fraction})
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    return runner


def test_each_cell_gets_the_effect_its_own_receptor_implies():
    runner = _run("Enzalutamide", 0.1, 0.1)
    ar = runner.engine.modules["ar_signaling"]
    cells = runner.engine.modules["cellular"].cells
    cancer = [c for c in ar.ids if cells[c].cell_type == "cancer"]
    carriers = [c for c in cancer if LABEL in cells[c].mutations]
    others = [c for c in cancer if LABEL not in cells[c].mutations]
    assert len(carriers) == 1 and len(others) == 9
    assert {ar.activity_of(c) for c in carriers} == {
        pb.AR_AGONIST_TRANSCRIPTIONAL_ACTIVITY}
    assert all(ar.activity_of(c) == pytest.approx(0.1) for c in others)
    normal = [c for c in ar.ids if cells[c].cell_type == "normal"]
    assert normal and all(
        ar.activity_of(c) == pytest.approx(0.1) for c in normal), (
        "benign cells have the wild-type receptor")


def test_the_subclone_survives_a_checkpoint(tmp_path):
    from cognisom.core.checkpoint import load_checkpoint, save_checkpoint
    runner = _run("Enzalutamide", 0.1, 0.2)
    path = str(tmp_path / "subclone.ckpt")
    with contextlib.redirect_stdout(io.StringIO()):
        save_checkpoint(runner, path)
        resumed = load_checkpoint(path)
    assert resumed.clonal_fractions == {LABEL: 0.1}
    assert (resumed.engine.modules["cellular"].clone_counts()
            == runner.engine.modules["cellular"].clone_counts())
    assert resumed.engine.modules["ar_signaling"].variant_label == LABEL


# ── and selection, with its two controls ──────────────────────────

#: 48 h, ring, seed 11, immune off, AR F877L in 10% of founders, measured
#: 2026-11-11. Carrier share of live cancer cells at the end, and tumour
#: PSA (psa_cancer_total) at the start / its lowest / the end.
#: Re-measure, do not re-pin.
MEASURED_48H = {
    None:           {"share": 0.100, "psa": (286.0, 286.0, 4574.3)},
    "Enzalutamide": {"share": 0.640, "psa": (286.0, 214.4, 1421.2)},
    "Darolutamide": {"share": 0.100, "psa": (286.0, 28.6, 28.6)},
}


class _Arms:
    """Lazy, so no one test pays for all three (see the 120 s timeout)."""

    def __init__(self):
        self._done = {}

    def __getitem__(self, drug):
        if drug not in self._done:
            ts = _run(drug, 0.1, 48.0).get_time_series()
            carriers, others = ts["n_subclone_carriers"], ts["n_subclone_others"]
            self._done[drug] = {
                "share": [c / max(1, c + o) for c, o in zip(carriers, others)],
                "psa": list(ts["psa_cancer_total"]),
            }
        return self._done[drug]


@pytest.fixture(scope="module")
def arms():
    return _Arms()


@pytest.mark.slow
def test_enzalutamide_enriches_the_carriers(arms):
    share = arms["Enzalutamide"]["share"]
    assert share[0] == pytest.approx(0.1)
    assert share[-1] > 3 * share[0], (
        "the drug must SELECT: carriers were a tenth and should now be "
        "most of the tumour", share[-1])
    # The end state has to be one the biopsy allows. The biopsy bounds
    # the carrier fraction after treatment from below.
    assert share[-1] >= CLONALITY["carrier_fraction_min"]


@pytest.mark.slow
def test_psa_falls_and_then_comes_back(arms):
    """The clinical signature of acquired resistance: a response, then
    progression on the same drug."""
    psa = arms["Enzalutamide"]["psa"]
    low = min(psa)
    assert low < 0.9 * psa[0], "there must be a response first"
    assert psa[-1] > 2 * psa[0], "and then progression on treatment"
    assert psa.index(low) < len(psa) - 1


@pytest.mark.slow
def test_without_the_drug_nothing_is_selected(arms):
    share = arms[None]["share"]
    assert share[-1] == pytest.approx(share[0], abs=0.05), (
        "carriers and non-carriers grow alike untreated, so the share "
        "holds", share[-1])


@pytest.mark.slow
def test_a_drug_that_blocks_both_selects_for_neither(arms):
    daro = arms["Darolutamide"]
    assert daro["share"][-1] == pytest.approx(daro["share"][0], abs=0.05)
    assert daro["psa"][-1] < 0.2 * daro["psa"][0], (
        "and it still treats him: PSA stays down")


@pytest.mark.slow
def test_the_measured_numbers_have_not_drifted(arms):
    for drug, want in MEASURED_48H.items():
        got = arms[drug]
        assert got["share"][-1] == pytest.approx(want["share"], abs=0.12), (
            f"{drug}: re-measure, do not re-pin", got["share"][-1])
        start, low, end = want["psa"]
        assert got["psa"][0] == pytest.approx(start, rel=0.02)
        assert min(got["psa"]) == pytest.approx(low, rel=0.25)
        assert got["psa"][-1] == pytest.approx(end, rel=0.35), (
            f"{drug}: re-measure, do not re-pin", got["psa"][-1])
