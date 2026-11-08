"""Prostate tissue is more than half stroma. This engine models none of it.

The stromal barrier gates immune motility on every step, and until now the
only thing that carried it was the CANCER cell -- so normal prostate
tissue had no matrix at all, in a gland that is largely matrix, and no
cell existed whose job was to make it.

What this file does is pin the TISSUE FACT, from a primary open-access
measurement, so that the gap is a stated quantity rather than an
impression, and so that the day a stromal compartment is added, the
assertions below are what it has to satisfy.

Source, read in full: Selnaes KM, Vettukattil R, Bertilsson H, Wright AJ,
Heerschap A, Angelsen A, Tessem MB, Bathen TF. "Tissue Microstructure Is
Linked to MRI Parameters and Metabolite Levels in Prostate Cancer."
Front Oncol 2016;6:146. **PMID 27379208**, PMC4905954,
doi:10.3389/fonc.2016.00146, **CC BY**.

Whole-mount H&E from prostatectomy, colour-segmented by the method of
Langer 2010 -- whose own figures are paywalled, which is why this paper
is the source: it applies the same segmentation and reports it openly.

Why a test and not just a comment: the two numbers below each refute
something the code used to imply, and a comment cannot fail.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pytest

from cognisom.engine.py.spatial.ecm_barrier import (
    PROSTATE_AREA_FRACTION_CANCER, PROSTATE_AREA_FRACTION_NON_CANCER,
    PROSTATE_COMPOSITION_SOURCE, PROSTATE_EPITHELIUM_FRACTION_CANCER,
    PROSTATE_EPITHELIUM_FRACTION_NON_CANCER,
)

#: The difference in stromal area between cancer and non-cancer tissue
#: that the paper measured and found NOT significant (p = 0.3). It is the
#: bound a model of cancer-driven matrix deposition has to respect: a run
#: must not manufacture, inside itself, more stromal change than the
#: study failed to detect between a tumour and its own normal tissue.
MEASURED_NULL_STROMA_DIFFERENCE = (
    PROSTATE_AREA_FRACTION_NON_CANCER['stroma']
    - PROSTATE_AREA_FRACTION_CANCER['stroma'])


# ── the measurement ──────────────────────────────────────────────────

def test_stroma_is_the_majority_compartment_of_prostate_tissue():
    """54.8% of non-cancer section area, against 31.4% epithelium. Any
    statement that the engine simulates prostate tissue is a statement
    about the minority compartment."""
    assert PROSTATE_AREA_FRACTION_NON_CANCER['stroma'] == 0.548
    assert PROSTATE_AREA_FRACTION_NON_CANCER['stroma'] > 0.5
    assert (PROSTATE_AREA_FRACTION_NON_CANCER['stroma']
            > PROSTATE_EPITHELIUM_FRACTION_NON_CANCER)


def test_the_three_compartments_account_for_the_whole_section():
    """The paper tabulates lumen, stroma and nuclei, and those sum to
    91.9% -- the rest is epithelial cytoplasm, which it does not report
    separately. Epithelium is DERIVED as 1 - stroma - lumen, so the
    derivation is checked here rather than trusted."""
    for area, epi in ((PROSTATE_AREA_FRACTION_NON_CANCER,
                       PROSTATE_EPITHELIUM_FRACTION_NON_CANCER),
                      (PROSTATE_AREA_FRACTION_CANCER,
                       PROSTATE_EPITHELIUM_FRACTION_CANCER)):
        assert area['stroma'] + area['lumen'] + epi == pytest.approx(1.0)
        # and the nuclei the paper does report are inside the epithelium
        assert area['nuclei'] < epi


def test_the_epithelial_share_rises_with_cancer_and_the_lumen_closes():
    """The direction the paper measures, which is the one a tumour makes:
    glands fill in. Epithelium 31.4% -> 41.8%, lumen 13.8% -> 7.5%."""
    assert (PROSTATE_EPITHELIUM_FRACTION_CANCER
            > PROSTATE_EPITHELIUM_FRACTION_NON_CANCER)
    assert (PROSTATE_AREA_FRACTION_CANCER['lumen']
            < PROSTATE_AREA_FRACTION_NON_CANCER['lumen'])


def test_cancer_does_not_raise_the_stromal_fraction_in_prostate():
    """p = 0.3, and if anything it is LOWER in cancer (50.7% against
    54.8%). So cancer cells acting as net matrix depositors reproduces a
    PDAC phenomenon -- which is what ecm_barrier's own docstring cites --
    and not a prostate one."""
    assert (PROSTATE_AREA_FRACTION_CANCER['stroma']
            < PROSTATE_AREA_FRACTION_NON_CANCER['stroma'])
    assert abs(MEASURED_NULL_STROMA_DIFFERENCE) < 0.05


def test_the_source_is_named_and_is_the_open_one():
    assert "27379208" in PROSTATE_COMPOSITION_SOURCE
    assert "CC BY" in PROSTATE_COMPOSITION_SOURCE
    assert "p=0.3" in PROSTATE_COMPOSITION_SOURCE


# ── what the engine does with it ─────────────────────────────────────

def test_nothing_converts_an_area_fraction_into_a_blocking_density():
    """`local_ecm_density` is a dimensionless 0..1 blocking term and
    0.548 is a fraction of section area. Equating them would be the unit
    confusion this codebase keeps catching -- a field rate read as a pool
    rate, a diameter read as a radius. The mapping is unsourced, so no
    code may quietly assume one."""
    import inspect

    from cognisom.engine.py.spatial import ecm_barrier
    src = inspect.getsource(ecm_barrier)
    # the constants exist, but nothing arithmetically feeds them into the
    # attenuation functions
    for fn in (ecm_barrier.motility_attenuation,
               ecm_barrier.detection_attenuation,
               ecm_barrier.ecm_density_at):
        body = inspect.getsource(fn)
        assert "PROSTATE_" not in body, (
            f"{fn.__name__} reads a measured AREA FRACTION as if it were a "
            "blocking density; that conversion is not sourced")
    assert "PROSTATE_AREA_FRACTION" in src   # declared, for reference only


def test_the_engine_has_no_stromal_compartment_at_all():
    """THE GAP, as a checkable statement. The engine's cell types are
    epithelial and immune; there is no fibroblast, no smooth muscle, and
    nothing whose role is to hold the matrix that gates immune motility.

    When a stromal lineage is added this test FAILS, and that is its
    purpose: it then has to be rewritten to say what fraction of the
    tissue the new compartment occupies, against the 54.8% above.
    """
    from cognisom.dashboard.engine_runner import EngineRunner

    runner = EngineRunner(scenario="Baseline")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        runner.build()
    kinds = {c.cell_type for c in runner.engine.modules["cellular"].cells.values()}
    assert kinds == {"normal", "cancer"}, (
        f"cell types are now {sorted(kinds)}; if a stromal lineage has "
        "arrived, re-state this test as the fraction it occupies against "
        "the measured 54.8%")
    stromal = {"fibroblast", "smooth_muscle", "stromal"}
    assert not (kinds & stromal)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_a_run_cannot_manufacture_more_stroma_than_the_study_failed_to_find():
    """The bound the measurement puts on the deposition rate.

    Cancer cells raise `local_ecm_density` at `ecm_deposition_rate`. The
    paper found no significant stromal difference between a tumour and
    its own normal tissue -- 4.1 percentage points, p = 0.3 -- so a 24 h
    run must not move matrix density by more than that. At the shipped
    8e-4/h it moves 0.019, inside the bound; at the 0.05/h this defaulted
    to until 2026-11-08 it moved 1.0, saturating, which is 24 times the
    difference the study could not detect.
    """
    from cognisom.dashboard.engine_runner import EngineRunner

    np.random.seed(1)
    runner = EngineRunner(dt=0.05, duration=24.0, scenario="Baseline",
                          record_interval=1000, snapshot_interval_hours=None)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        runner.build()
        runner.run()
    ecm = runner.engine.modules["cellular"].get_state()["ecm"]
    assert ecm["max"] is not None
    moved = ecm["max"] - ecm["baseline_at_spawn"]
    assert moved <= abs(MEASURED_NULL_STROMA_DIFFERENCE), (
        f"matrix density moved {moved:.3f} in 24 h, more than the "
        f"{abs(MEASURED_NULL_STROMA_DIFFERENCE):.3f} stromal difference "
        "the study measured between cancer and normal tissue and found "
        "not significant")
    assert ecm["saturated_fraction"] == 0.0
