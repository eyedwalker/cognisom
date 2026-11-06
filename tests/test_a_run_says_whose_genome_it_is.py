"""A simulation the dashboard runs must say whose genome it is about.

``EngineRunner`` accepted ``patient=`` and ``tissue=`` for weeks and no
page passed either. Nothing failed: the engine fell back to a
placeholder ``KRAS_G12D`` driver on a generic HLA panel, on the
close-packed slab, and produced a perfectly ordinary result about
nobody. Personalisation is the platform's central claim, so the absence
of an error was the whole problem.

These tests cover the join -- ``cognisom.dashboard.sim_request`` -- and,
at the end, assert that each page which builds a runner still passes
both arguments. That last group is the one that would have caught the
original defect: everything else here can pass while no page calls it.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

from cognisom.dashboard.engine_runner import TISSUE_LAYOUTS
from cognisom.dashboard.sim_request import (
    GENERIC_CAVEAT, SESSION_PATIENT_KEY, TISSUE_CAPTIONS, TISSUE_CHOICES,
    TISSUE_LABELS, describe_subject, patient_from_session,
    presentation_lines, tissue_caption)


# ── A stand-in profile ───────────────────────────────────────────────
#
# Duck-typed, like engine_inputs itself. Using a real PatientProfile
# here would drag the VCF parser into a unit test; what matters is the
# attribute surface, and a fake that drifts from the real one is caught
# by the HCC1395 test at the bottom of this file.

@dataclass
class _Variant:
    gene: str
    protein_change: str
    consequence: str = "missense_variant"


@dataclass
class _Profile:
    patient_id: str = "TEST01"
    coding_variants: List[_Variant] = field(default_factory=list)
    cancer_driver_mutations: List[_Variant] = field(default_factory=list)
    hla_alleles: Optional[List[str]] = None
    hla_typing_method: str = "population-default"
    has_matched_normal: bool = True


def _profile(**kw) -> _Profile:
    base = dict(
        coding_variants=[_Variant("ARID1A", "p.N218T")],
        hla_alleles=["HLA-A*02:01", "HLA-B*07:02"],
    )
    base.update(kw)
    return _Profile(**base)


# ── 1. Finding the patient a page left behind ────────────────────────

def test_the_session_key_is_the_one_every_page_already_writes():
    """app.py, 26_genomic_twin, 28_digital_twin, 32_parabricks,
    35_orchestrator, 37_mad_agent and 38_real_patient all store under
    this name. If it changes here, the join silently finds nothing."""
    assert SESSION_PATIENT_KEY == "patient_profile"


def test_a_loaded_profile_is_found():
    prof = _profile()
    assert patient_from_session({SESSION_PATIENT_KEY: prof}) is prof


@pytest.mark.parametrize("session", [None, {}, {"other": 1},
                                     {SESSION_PATIENT_KEY: None}])
def test_no_profile_resolves_to_none_rather_than_raising(session):
    """A page renders on every interaction, so this runs constantly with
    nothing loaded. It must be quiet, and it must not guess."""
    assert patient_from_session(session) is None


def test_an_object_that_is_not_a_profile_is_refused():
    """A dict or a sentinel would sail through engine_inputs -- every
    read there is a getattr with a default -- and produce an empty
    personalisation that still called itself personalised."""
    assert patient_from_session({SESSION_PATIENT_KEY: {"patient_id": "x"}}) is None
    assert patient_from_session({SESSION_PATIENT_KEY: object()}) is None


# ── 2. Saying there is no patient ────────────────────────────────────

def test_with_no_patient_the_run_declares_itself_generic():
    subject = describe_subject(None)
    assert subject.personalised is False
    assert subject.patient is None
    assert GENERIC_CAVEAT in subject.caveats


def test_the_generic_caveat_names_the_placeholder_and_denies_being_a_twin():
    """The failure mode was a plausible-looking result nobody could
    attribute. The words that prevent that are the placeholder's name
    and an explicit denial, so both are asserted, not just a non-empty
    string."""
    assert "KRAS_G12D" in GENERIC_CAVEAT
    assert "not a personalised twin" in GENERIC_CAVEAT


# ── 3. Describing a patient that IS loaded ───────────────────────────

def test_a_loaded_patient_is_named_with_what_crosses_over():
    subject = describe_subject(_profile())
    assert subject.personalised is True
    assert subject.patient_id == "TEST01"
    assert "TEST01" in subject.headline
    assert subject.mutation_labels == ["ARID1A_N218T"]
    assert subject.hla_alleles == ["HLA-A*02:01", "HLA-B*07:02"]


def test_an_unmeasured_hla_typing_is_called_unmeasured():
    """NK licensing refuses a typing that was not derived from the
    patient's reads. A page that does not repeat that refusal lets a
    population default look like this patient's genotype."""
    subject = describe_subject(_profile(hla_typing_method="population-default"))
    assert subject.hla_measured is False
    assert any("not derived from this patient's reads" in c
               for c in subject.caveats)


def test_a_patient_whose_mutations_make_no_neoantigen_is_still_reported():
    """Zero presentable mutations is a result. Reporting it as a clean
    personalised run would be the same silence in a new place."""
    subject = describe_subject(_profile(
        coding_variants=[_Variant("TP53", "p.R213*", "stop_gained")]))
    assert subject.personalised is True
    assert subject.mutation_labels == []
    assert any("no canonical" in c or "nothing patient-specific" in c
               for c in subject.caveats)
    assert subject.skipped, "a variant that cannot cross must be listed"


def test_a_callset_with_no_matched_normal_says_so():
    """Without a matched normal a rare inherited variant and a tumour
    mutation are indistinguishable, so no neoantigen from this profile
    is established as tumour-specific."""
    subject = describe_subject(_profile(has_matched_normal=False))
    assert any("matched normal" in c for c in subject.caveats)


def test_a_matched_callset_does_not_invent_that_caveat():
    """Empty is a claim: the caveat list is only ever empty, or short,
    when nothing applies."""
    subject = describe_subject(_profile(has_matched_normal=True))
    assert not any("matched normal" in c for c in subject.caveats)


# ── 4. What the engine presented, once it ran ─────────────────────────

def test_presentation_lines_report_a_mutation_that_bound():
    lines = presentation_lines({
        "cancer_cells": 10, "cells_displaying": 7,
        "presented": ["ARID1A_N218T"], "unpresented": [],
    })
    assert any("ARID1A_N218T" in ln and "7 of 10" in ln for ln in lines)


def test_a_mutation_that_bound_nothing_gets_its_own_line():
    """An invisible mutation is a finding -- it predicts the tumour
    escapes T-cell recognition -- so it must not be dropped for looking
    like a failure."""
    lines = presentation_lines({
        "cancer_cells": 4, "cells_displaying": 0, "presented": [],
        "unpresented": [{"gene": "NOTCH1", "reason": "no peptide bound"}],
    })
    assert any("invisible to a T cell" in ln for ln in lines)
    assert any("NOTCH1" in ln for ln in lines)


def test_a_structural_reason_short_circuits_the_report():
    lines = presentation_lines({"reason": "no patient mutations"})
    assert lines == ["Nothing presented: no patient mutations."]


def test_no_report_is_no_lines():
    assert presentation_lines(None) == []
    assert presentation_lines({}) == []


# ── 5. Geometry ──────────────────────────────────────────────────────

def test_the_offered_layouts_are_exactly_the_ones_the_engine_accepts():
    """Two lists of the same thing. Asserting they AGREE rather than
    pinning either means adding a layout to the engine cannot leave the
    UI offering a value that raises, or hiding one that works."""
    assert set(TISSUE_CHOICES) == set(TISSUE_LAYOUTS)


def test_the_default_offered_is_the_engine_default():
    """The UI's first option must be what EngineRunner does with no
    argument, or the same scenario silently means two things."""
    sig = inspect.signature(
        __import__("cognisom.dashboard.engine_runner",
                   fromlist=["EngineRunner"]).EngineRunner.__init__)
    assert TISSUE_CHOICES[0] == sig.parameters["tissue"].default == "slab"


@pytest.mark.parametrize("name", TISSUE_CHOICES)
def test_every_layout_has_a_label_and_a_caption(name):
    assert TISSUE_LABELS.get(name)
    assert tissue_caption(name) == TISSUE_CAPTIONS[name]
    assert len(tissue_caption(name)) > 40


def test_the_slab_caption_admits_it_over_connects():
    """The default is the geometry that is wrong. A caption that does
    not say so turns the measurement into a secret."""
    caption = tissue_caption("slab")
    assert "1.6x" in caption
    assert "7.8" in caption and "5.0" in caption


def test_the_section_caption_says_where_its_occupancy_came_from():
    assert "Xenium" in tissue_caption("section")


def test_an_unknown_layout_gets_no_caption_rather_than_a_wrong_one():
    assert tissue_caption("blancmange") == ""


# ── 6. The pages actually pass it ────────────────────────────────────
#
# Everything above can pass while the dashboard ignores the module, which
# is the exact shape of the original defect. These read the page source,
# because a Streamlit page cannot be imported without a browser session.

PAGES_THAT_RUN_THE_ENGINE = [
    "3_simulation.py",
    "12_3d_visualization.py",
    "19_prostate_metastasis.py",
]


def _page_source(name: str) -> str:
    from pathlib import Path
    import cognisom.dashboard as dash
    path = Path(dash.__file__).parent / "_pages" / name
    assert path.exists(), f"{name} is gone; this list needs updating"
    return path.read_text()


@pytest.mark.parametrize("name", PAGES_THAT_RUN_THE_ENGINE)
def test_every_page_that_builds_a_runner_passes_a_patient(name):
    src = _page_source(name)
    assert "EngineRunner(" in src, (
        f"{name} no longer builds a runner; remove it from "
        "PAGES_THAT_RUN_THE_ENGINE rather than loosening this test")
    assert "patient=" in src, (
        f"{name} builds an EngineRunner without passing patient=, so every "
        "cancer cell in it carries the placeholder KRAS_G12D driver and "
        "nothing says so. This is the defect this file exists for.")


@pytest.mark.parametrize("name", PAGES_THAT_RUN_THE_ENGINE)
def test_every_page_that_builds_a_runner_chooses_its_geometry(name):
    src = _page_source(name)
    assert "tissue=" in src, (
        f"{name} does not pass tissue=, so it silently uses the slab, "
        "which over-connects every cell by about 1.6x")


@pytest.mark.parametrize("name", PAGES_THAT_RUN_THE_ENGINE)
def test_every_page_resolves_the_patient_through_the_shared_join(name):
    """Not by reading st.session_state["patient_profile"] by hand: the
    duck-type check and the caveat list are the parts that keep a
    non-profile from passing as a personalised run."""
    src = _page_source(name)
    assert "patient_from_session" in src and "describe_subject" in src, (
        f"{name} should resolve its patient through "
        "cognisom.dashboard.sim_request, not inline")
