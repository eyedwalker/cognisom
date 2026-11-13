"""A drug can be given on the path the demo actually uses.

The agonist switch (enzalutamide drives AR F876L/F877L instead of
blocking it) lived behind ``EngineRunner.from_scenario``, which takes a
stored scenario and NO patient. ``EngineRunner(patient=...)`` -- the path
the demo recording and every page use -- had no drug argument. So the one
patient-specific drug response the engine models could not be run on the
patient it was written for, and nothing failed: the capability was built,
tested, and never connected.

``EngineRunner(drugs=[...])`` is the connection. These tests assert the
CLAIMS (the drug's effect depends on this patient's receptor; a drug the
library lacks is refused, not skipped) and keep the measured numbers
beside them.
"""

from __future__ import annotations

import contextlib
import io
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest

from cognisom.workflow import parameter_bridge as pb
from cognisom.workflow.treatment import resolve_treatment


def _profile(*changes):
    """The least a profile needs for the bridge to see its variants."""
    return SimpleNamespace(variants=[
        SimpleNamespace(gene=g, protein_change=c) for g, c in changes])


F877L = _profile(("AR", "p.F877L"))
WILD_TYPE = _profile(("TP53", "p.R175H"))


@pytest.fixture(scope="module")
def store():
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        return EntityStore()


# ── the join itself ───────────────────────────────────────────────


def test_the_same_drug_resolves_differently_for_two_patients(store):
    on_mutant = resolve_treatment(["Enzalutamide"], F877L, store)
    on_wild_type = resolve_treatment(["Enzalutamide"], WILD_TYPE, store)

    assert (on_mutant.module_configs["ar_signaling"]
            == {"ar_transcriptional_activity":
                pb.AR_AGONIST_TRANSCRIPTIONAL_ACTIVITY})
    assert any("AGONIST" in n and "F877L" in n for n in on_mutant.notes)

    assert on_wild_type.module_configs["ar_signaling"] == {
        "ar_transcriptional_block": 0.9}
    assert on_wild_type.notes == []


def test_darolutamide_keeps_its_blockade_on_this_patient(store):
    """The actionable half (Moilanen 2015, PMID 26137992)."""
    t = resolve_treatment(["Darolutamide"], F877L, store)
    assert t.module_configs["ar_signaling"] == {"ar_transcriptional_block": 0.9}
    assert t.notes == []


def test_no_patient_means_the_class_effect(store):
    t = resolve_treatment(["Enzalutamide"], None, store)
    assert t.variant_labels == []
    assert "ar_transcriptional_block" in t.module_configs["ar_signaling"]


def test_a_drug_the_library_lacks_is_refused_not_skipped(store):
    """Bicalutamide is not in the library. Skipping it would run an
    untreated patient under a treated label."""
    with pytest.raises(ValueError, match="not in the entity library"):
        resolve_treatment(["Bicalutamide"], F877L, store)


def test_a_lower_case_name_finds_the_drug(store):
    assert resolve_treatment("enzalutamide", None, store).drugs == ["Enzalutamide"]


# ── and it reaches the runner ─────────────────────────────────────


def test_no_drugs_changes_nothing():
    from cognisom.dashboard.engine_runner import EngineRunner
    plain = EngineRunner(tissue="ring")
    assert plain.treatment is None
    assert EngineRunner(tissue="ring", drugs=None).module_configs == plain.module_configs
    assert EngineRunner(tissue="ring", drugs=[]).module_configs == plain.module_configs


def test_the_runner_applies_the_treatment_and_says_what_it_dropped():
    from cognisom.dashboard.engine_runner import EngineRunner
    runner = EngineRunner(tissue="ring", drugs=["Enzalutamide"])
    assert runner.module_configs["ar_signaling"]["ar_transcriptional_block"] == 0.9
    # The bridge also configures the 'ode' solver module, which no preset
    # carries. It must be reported, not silently discarded.
    assert set(runner.treatment.module_configs) - set(runner.module_configs) \
        == set(runner.treatment.unapplied_modules)
    assert runner.treatment.summary()["drugs"] == ["Enzalutamide"]


def test_a_caller_override_still_wins():
    from cognisom.dashboard.engine_runner import EngineRunner
    runner = EngineRunner(
        tissue="ring", drugs=["Enzalutamide"],
        overrides={"ar_signaling": {"ar_transcriptional_block": 0.5}})
    assert runner.module_configs["ar_signaling"]["ar_transcriptional_block"] == 0.5


def test_the_demo_recorder_can_give_the_drug():
    """The recorder is what the public page is built from; the flag has
    to exist there and has to be passed to the runner."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "record_demo_run.py").read_text()
    assert '"--drug"' in src
    assert "seeded_runner(args.drug)" in src
    assert "drugs=drugs or None" in src
    assert '"treatment"' in src, "the sidecar must state the treatment"
    assert '"treatment_arms"' in src, (
        "one run cannot show what a drug did; the comparison arms must "
        "reach the sidecar the page reads")


# ── end to end, on the demo's own patient ─────────────────────────

MODS = {"cellular": True, "spatial": True, "vascular": True,
        "ar_signaling": True, "immune": False, "lymphatic": False,
        "molecular": False, "epigenetic": False, "circadian": False,
        "morphogen": False, "cell_mechanics": False}

#: SU2C SC_9184_T through EngineRunner(patient=..., drugs=...): 48 h,
#: ring, seed 11, immune off, measured 2026-11-10. Re-measure, do not
#: re-pin. They equal the override-driven arms in
#: test_an_agonist_is_applied_as_activation.SHIPPED_48H, which is the
#: point: the patient path now reaches the same physics.
MEASURED_48H = {
    None:           {"signalling": 0.9995, "cancer": 160},
    "Enzalutamide": {"signalling": 3.0485, "cancer": 160},
    "Darolutamide": {"signalling": 0.1000, "cancer": 10},
}


class _Arms:
    """Each arm is run the first time a test asks for it, so no single
    test pays for all three inside the suite's 120 s timeout -- on a
    loaded machine the eager fixture timed out without anything being
    wrong."""

    def __init__(self):
        self._done = {}
        self._patient = None

    def __getitem__(self, drug):
        if drug not in self._done:
            self._done[drug] = self._run(drug)
        return self._done[drug]

    def _run(self, drug):
        from cognisom.dashboard.engine_runner import EngineRunner
        from cognisom.genomics.demo_patients import (
            DEFAULT_DEMO_PATIENT, load_demo_patient)

        if self._patient is None:
            with contextlib.redirect_stdout(io.StringIO()):
                self._patient = load_demo_patient(DEFAULT_DEMO_PATIENT)
        np.random.seed(11)
        random.seed(11)
        runner = EngineRunner(
            tissue="ring", dt=0.05, duration=48.0, scenario="Baseline",
            modules_enabled=MODS, patient=self._patient,
            drugs=[drug] if drug else None)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.build()
            runner.run()
        ar = runner.engine.modules["ar_signaling"]
        return {
            "signalling": float(np.mean(
                [ar.signalling_fraction(c) for c in ar.ids])),
            "cancer": int(runner.get_time_series()["n_cancer"][-1]),
            "notes": list(runner.treatment.notes) if runner.treatment else [],
        }


@pytest.fixture(scope="module")
def arms():
    return _Arms()


@pytest.mark.slow
def test_enzalutamide_drives_this_patients_receptor(arms):
    untreated, enza = arms[None], arms["Enzalutamide"]
    assert enza["signalling"] > 2.5 * untreated["signalling"], (
        "enzalutamide must ACTIVATE AR in the F877L demo patient", enza)
    # Balbas 2013: growth on drug is equivalent to vehicle -- rescue,
    # not acceleration and not suppression.
    assert enza["cancer"] >= 0.9 * untreated["cancer"]
    assert any("AGONIST" in n for n in enza["notes"])


@pytest.mark.slow
def test_darolutamide_still_treats_him(arms):
    untreated, daro = arms[None], arms["Darolutamide"]
    assert daro["signalling"] < 0.2 * untreated["signalling"]
    assert daro["cancer"] < 0.5 * untreated["cancer"], (
        "there is a drug this patient can still be given", daro)
    assert daro["notes"] == []


@pytest.mark.slow
def test_the_measured_numbers_have_not_drifted(arms):
    for drug, want in MEASURED_48H.items():
        got = arms[drug]
        assert got["signalling"] == pytest.approx(want["signalling"], rel=0.02), (
            f"{drug}: re-measure, do not re-pin", got)
        assert got["cancer"] == pytest.approx(want["cancer"], rel=0.25, abs=3), (
            f"{drug}: re-measure, do not re-pin", got)


# ── and the record of it survives a checkpoint ────────────────────


def test_a_resumed_run_still_says_which_drug_was_given(tmp_path):
    """The drug's effect rides in module_configs and always survived.
    The record that a drug was given did not: a treated run came back
    with ``treatment is None``."""
    from cognisom.core.checkpoint import load_checkpoint, save_checkpoint
    from cognisom.dashboard.engine_runner import EngineRunner

    runner = EngineRunner(tissue="ring", dt=0.05, duration=0.2,
                          patient=F877L, drugs=["Enzalutamide"])
    path = str(tmp_path / "treated.ckpt")
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
        save_checkpoint(runner, path)
        resumed = load_checkpoint(path)

    assert resumed.treatment is not None, (
        "a treated run was resumed under an untreated label")
    assert resumed.treatment.summary() == runner.treatment.summary()
    assert any("AGONIST" in n for n in resumed.treatment.notes)
    # The label and the physics must describe the same run.
    assert (resumed.module_configs["ar_signaling"]
            == runner.module_configs["ar_signaling"])


def test_an_untreated_run_resumes_untreated(tmp_path):
    from cognisom.core.checkpoint import load_checkpoint, save_checkpoint
    from cognisom.dashboard.engine_runner import EngineRunner

    runner = EngineRunner(tissue="ring", dt=0.05, duration=0.2)
    path = str(tmp_path / "plain.ckpt")
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
        save_checkpoint(runner, path)
        assert load_checkpoint(path).treatment is None
