"""The drug reached every cell equally, and nothing said so.

Why this file exists
--------------------
``ARSignalingModule.activity_of`` returned one number for the whole
tumour. A cancer cell pressed against a capillary and one 200 um inside
a nodule saw exactly the same enzalutamide. That is not a modelling
choice anyone made -- it is the absence of one, and it means the engine
could not offer "the drug did not reach those cells" as a reason a
treatment failed, which is among the leading reasons real treatments do.

Primeau 2005 (PMID 16361566) measured the shape, in murine tumours and
human prostate PC-3 xenografts: doxorubicin "decreases exponentially
with distance from tumor blood vessels, decreasing to half its
perivascular concentration at a distance of about 40 to 50 um", while
hypoxic regions sit 90-140 um out, so "many viable tumor cells are not
exposed to detectable concentrations of drug following a single
injection".

What these tests pin is the STRUCTURE and the two identities, not a
penetration distance for enzalutamide -- there is no such measurement,
and the default stays uniform for reasons that are themselves measured
(Gibbons 2015: half-life 5.8 days, peak-to-trough 1.25). The point of
the mechanism is that uniformity became a claim with a switch behind it
instead of an unexamined silence.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from cognisom.engine.py.pharmacology import exposure as px


# ── the shape, against the paper that measured it ────────────────────

def test_the_drug_halves_over_the_measured_distance():
    half = px.BOLUS_HALF_DISTANCE_UM
    assert px.BOLUS_HALF_DISTANCE_RANGE == (40.0, 50.0)
    assert px.BOLUS_HALF_DISTANCE_RANGE[0] <= half <= px.BOLUS_HALF_DISTANCE_RANGE[1]
    # At the vessel wall the cell sees all of it.
    assert px.exposure_fraction(0.0, half) == 1.0
    # One half-distance halves it; two quarter it. Exponential, which is
    # the word the paper uses.
    assert px.exposure_fraction(half, half) == pytest.approx(0.5)
    assert px.exposure_fraction(2 * half, half) == pytest.approx(0.25)


def test_at_the_hypoxic_distance_the_drug_is_effectively_gone():
    """Primeau's own conclusion, recomputed rather than quoted.

    Hypoxia sat 90-140 um from a vessel in those tumours, and the paper
    reports that many viable cells there saw no detectable drug. At the
    measured half-distance that range leaves 7-25% of the perivascular
    level, so the two halves of the paper agree with each other.
    """
    low, high = px.HYPOXIA_DISTANCE_RANGE_UM
    assert (low, high) == (90.0, 140.0)
    at_near = px.exposure_fraction(low, px.BOLUS_HALF_DISTANCE_UM)
    at_far = px.exposure_fraction(high, px.BOLUS_HALF_DISTANCE_UM)
    assert 0.2 < at_near < 0.3, at_near
    assert 0.05 < at_far < 0.15, at_far
    assert at_far < at_near


def test_no_half_distance_is_exactly_uniform():
    """The identity that makes this safe to ship on by default.

    Not approximately 1.0 -- exactly, at every distance, so a run that
    declares no penetration limit is bit-identical to one from before
    this module existed.
    """
    for d in (0.0, 1.0, 45.0, 200.0, 1e6):
        assert px.exposure_fraction(d, None) == 1.0


def test_a_cell_no_vessel_serves_gets_nothing():
    # _nearest_capillary returns inf when a run has no vessels at all.
    # The oxygen path gives such a cell no oxygen; the drug path has to
    # agree with it rather than quietly dosing it.
    assert px.exposure_fraction(float('inf'), 45.0) == 0.0


# ── the scaling rule works on both signs of drug effect ──────────────

def test_exposure_scales_the_drug_and_not_the_cell():
    # A blockade: less drug means closer to untreated (1.0), never past
    # it, and never below the full-exposure effect.
    assert px.scale_activity(0.1, 1.0) == pytest.approx(0.1)
    assert px.scale_activity(0.1, 0.0) == pytest.approx(1.0)
    assert 0.1 < px.scale_activity(0.1, 0.5) < 1.0
    # The agonist case -- a receptor variant turning an anti-androgen
    # into a driver -- has to weaken the same way. One rule, both signs.
    assert px.scale_activity(3.05, 1.0) == pytest.approx(3.05)
    assert px.scale_activity(3.05, 0.0) == pytest.approx(1.0)
    assert 1.0 < px.scale_activity(3.05, 0.5) < 3.05


def test_exposure_is_clamped_so_a_bad_input_cannot_amplify_a_drug():
    assert px.scale_activity(0.1, 5.0) == pytest.approx(0.1)
    assert px.scale_activity(0.1, -1.0) == pytest.approx(1.0)


# ── why the default is uniform, computed rather than asserted ────────

def test_diffusion_alone_cannot_hold_a_gradient_for_this_drug():
    """The argument behind the uniform default, from the engine's own
    numbers rather than from prose.

    Enzalutamide's plasma profile is flat over a dosing interval
    (peak-to-trough 1.25) and sustained for weeks past steady state at
    day 28 (Gibbons 2015). Diffusion crosses the engine's own vessel
    spacing in minutes. So any steady-state gradient must be held by a
    SINK -- binding, sequestration, metabolism, efflux -- which is what
    doxorubicin has and what is unmeasured for enzalutamide.
    """
    from cognisom.modules.spatial_module import TESTOSTERONE_DIFFUSION_UM2_S

    pk = px.ENZALUTAMIDE_PK
    assert pk["half_life_days"] == 5.8
    assert pk["peak_to_trough_ratio"] == 1.25
    assert "25917876" in str(pk["source"])
    # Enzalutamide (MW 464) against testosterone (MW 288): same order of
    # molecular size, so the engine's own coefficient is the right scale
    # to make this argument at.
    hours = px.equilibration_hours(200.0, TESTOSTERONE_DIFFUSION_UM2_S)
    assert hours < 1.0, hours
    # Four orders of margin against a 24 h dosing interval.
    assert 24.0 / hours > 100.0
    # And the default therefore says uniform, as a DERIVED claim.
    assert px.ENZALUTAMIDE_PENETRATION.is_uniform
    assert px.ENZALUTAMIDE_PENETRATION.status == 'derived'
    assert "NOT a measurement of" in px.ENZALUTAMIDE_PENETRATION.source


def test_the_bolus_number_says_which_drug_it_belongs_to():
    """It is doxorubicin's, after one injection. The next reader must
    not be able to mistake it for a property of tumours."""
    row = px.penetration_report()['bolus_sequestered']
    assert row['status'] == 'sourced'
    assert "16361566" in row['source']
    assert "doxorubicin" in row['source'].lower()
    assert "SINGLE INJECTION" in row['source'].upper()


def test_a_negative_half_distance_is_refused():
    with pytest.raises(ValueError):
        px.DrugPenetration(half_distance_um=0.0, status='ESTIMATE', source='x')
    with pytest.raises(ValueError):
        px.exposure_fraction(10.0, -5.0)


# ── the engine ───────────────────────────────────────────────────────

def _runner(half_distance, duration=2.0, seed=5, n_capillaries=None):
    from cognisom.dashboard.engine_runner import EngineRunner

    overrides = {'ar_signaling': {'ar_transcriptional_block': 0.9}}
    if half_distance is not None:
        overrides['ar_signaling']['drug_half_distance_um'] = half_distance
    if n_capillaries is not None:
        overrides['vascular'] = {'n_capillaries': n_capillaries}
    np.random.seed(seed)
    runner = EngineRunner(tissue='ring', dt=0.05, duration=duration,
                          scenario='Baseline', record_interval=1000,
                          snapshot_interval_hours=None, overrides=overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    return runner


def _exposures(runner):
    """(distance, exposure, activity) per living mapped cell."""
    ar = runner.engine.modules['ar_signaling']
    vascular = runner.engine.modules['vascular']
    cells = runner.engine.modules['cellular'].cells
    rows = []
    for cell_id in ar.ids:
        cell = cells.get(cell_id)
        if cell is None or not cell.alive:
            continue
        _cap, d = vascular._nearest_capillary(cell.position)
        rows.append((d, ar.drug_exposure_of(cell_id), ar.activity_of(cell_id)))
    return sorted(rows)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_a_run_says_whether_the_drug_reached_everyone():
    uniform = _runner(None)
    limited = _runner(px.BOLUS_HALF_DISTANCE_UM)

    u = uniform.engine.modules['ar_signaling'].get_state()
    assert u['drug_exposure_uniform'] is True
    assert u['drug_half_distance_um'] is None
    # A uniform run reports no spread, because there is none to report.
    assert 'drug_exposure_mean' not in u
    assert "25917876" in u['drug_penetration_sources']['enzalutamide']['source']

    l = limited.engine.modules['ar_signaling'].get_state()
    assert l['drug_exposure_uniform'] is False
    assert l['drug_half_distance_um'] == px.BOLUS_HALF_DISTANCE_UM
    # THE CLAIM: the cells in one tissue do not all see the same drug.
    assert 0.0 < l['drug_exposure_min'] < l['drug_exposure_max'] <= 1.0
    assert l['drug_exposure_min'] < l['drug_exposure_mean'] < l['drug_exposure_max']
    # AND THE MEASUREMENT THAT MATTERS, which is not about the drug.
    # Primeau's tumours had viable cells 90-140 um from a vessel, seeing
    # no detectable drug. NO CELL IN ANY SHIPPED PRESET IS THAT FAR OUT.
    # Measured over ring/slab/section, Baseline and Hypoxia: the
    # furthest any cell sits from a capillary is 44.5 um (ring Hypoxia,
    # 3 vessels) and usually about 21 um, so even doxorubicin's steep
    # gradient leaves the worst-served cell 50-79% of the perivascular
    # level and NOT ONE cell below half.
    #
    # That is a property of the VASCULATURE, not of any drug, and it is
    # deliberate: the slab and section lay their vessels on a pitch
    # chosen against exchange_radius "so that no cell in the slab is
    # outside a Krogh cylinder" (vascular_module), and n_capillaries is
    # ignored in those layouts for that reason. Good perfusion is
    # therefore guaranteed by construction, and the consequence nobody
    # had traced is this one: drug penetration cannot limit anything in
    # a shipped preset, however steep the gradient. Using it as a
    # resistance mechanism needs a sparser vessel lattice, which is a
    # change to the vasculature with its own evidence to find --
    # see test_the_mechanism_bites_when_the_vessels_are_sparse below,
    # which proves the path is wired rather than vacuous.
    assert l['n_cells_below_10pct_exposure'] == 0, (
        'a shipped preset now starves some cell of drug; the vessel '
        'lattice has changed -- RE-MEASURE the spread over ring/slab/'
        'section and move the numbers in this comment with the claim')
    assert l['drug_exposure_min'] > 0.5, l['drug_exposure_min']


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_the_mechanism_bites_when_the_vessels_are_sparse():
    """Not vacuous: give the ring one vessel and the drug runs out.

    The shipped lattices leave no cell outside a Krogh cylinder, so the
    test above measures a mechanism that cannot fire there. The ring's
    radial layout DOES honour n_capillaries, so one vessel is the
    geometry Primeau's tumours have, and here is what the same code
    does in it. Measured, ring Baseline, seed 5, doxorubicin's
    half-distance:

        vessels   furthest cell   worst exposure   cells under half
           3          43.8 um         0.510              0 of 87
           2          53.9 um         0.436             25 of 84
           1          70.6 um         0.337             52 of 88

    At one vessel the blockade is seven times weaker at the worst-served
    cell than at the best (activity 0.70 against 0.19 on a configured
    0.1), which is a sanctuary site -- resistance with no mutation in it.
    """
    rows = _exposures(_runner(px.BOLUS_HALF_DISTANCE_UM, n_capillaries=1))
    assert len(rows) > 50
    distances = [d for d, _e, _a in rows]
    exposures = [e for _d, e, _a in rows]
    activities = [a for _d, _e, a in rows]
    # The geometry is the one the paper describes, not the shipped one.
    assert max(distances) > 60.0, max(distances)
    # THE CLAIM: most of the tissue is under half the perivascular dose.
    assert sum(1 for e in exposures if e < 0.5) > len(rows) // 2
    # and the blockade weakens with it, by a wide margin
    assert max(activities) > 3 * min(activities)
    assert min(activities) > 0.1, 'no cell may be blocked harder than the dose'
    assert max(activities) < 1.0, 'no cell may be fully untreated here'
    # and the numbers beside the claims
    assert min(exposures) == pytest.approx(0.337, abs=0.01)
    assert max(distances) == pytest.approx(70.6, abs=0.5)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_the_blockade_is_weaker_where_the_drug_is_thinner():
    """Per cell, not on average: the cell that sees least drug keeps
    most of its AR transcription."""
    runner = _runner(px.BOLUS_HALF_DISTANCE_UM)
    rows = _exposures(runner)
    assert len(rows) > 10
    nearest, farthest = rows[0], rows[-1]
    assert farthest[0] > nearest[0]
    # Exposure falls with distance and the blockade weakens with it:
    # activity runs from the configured 0.1 toward the untreated 1.0.
    assert farthest[1] < nearest[1]
    assert farthest[2] > nearest[2]
    for _d, exposure, activity in rows:
        assert 0.1 <= activity <= 1.0
        assert activity == pytest.approx(px.scale_activity(0.1, exposure))


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_without_a_vascular_module_there_is_no_geometry_to_dose_by():
    """A run with no vessels must not invent a gradient. Uniform is the
    honest answer there, and it is the one this returns."""
    from cognisom.modules.ar_signaling_module import ARSignalingModule

    with contextlib.redirect_stdout(io.StringIO()):
        module = ARSignalingModule({'ar_transcriptional_block': 0.9,
                                    'drug_half_distance_um': 45.0})
    assert module.vascular_module is None
    assert module.drug_exposure_of(0) == 1.0
    assert module.activity_of(0) == pytest.approx(0.1)
