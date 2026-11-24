"""Cancer cells can carry different intrinsic AR drive -- the structure
limitation 2 asks for, with the spread OFF by default.

Limitation 2 on ``?page=methods`` says every cell in a treatment arm
carries the same AR signalling to about one part in ten thousand, so
the engine has nothing graded across cells for the external
AR/proliferation association to correspond to. The magnitude was
sourced on 2026-11-24 from 24 hormone-therapy-naive prostate cancers
(``validation/ar_heterogeneity_singlecell.py``): CV ~0.46, sigma ~0.34.

This file covers the mechanism. Three things it has to get right, and
each one was a decision rather than a default:

1. OFF IS BIT-IDENTICAL. ``ar_drive_sigma`` defaults to 0, every cell
   sits at exactly 1.0, and a run matches one from before any of this
   existed. Without that, every pinned count in the suite moves for
   reasons unrelated to AR and the change cannot be separated from its
   own calibration.

2. DETERMINISTIC, NOT DRAWN. ``ar_dependent`` is assigned by
   ``cell_id % 3`` so that no random draw shifts the pinned runs; drive
   follows the same rule via ``ar_drive_from_id``. A draw here would
   consume the shared RNG stream and move every later draw.

3. DIVISION AND DEATH READ DIFFERENT NUMBERS. Division follows absolute
   signalling (drive included), because a more AR-driven cell should
   proliferate faster -- that is the association being reproduced. The
   withdrawal-death hazard follows ``suppression_fraction``, which
   divides drive out, because a cell that always ran at 70% of median
   drive is not withdrawing from anything. Keying the hazard on
   absolute signalling would invent androgen-withdrawal deaths in an
   UNTREATED tumour the moment drive was spread.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

pytest.importorskip("scipy")

from cognisom.modules import cellular_module as cm  # noqa: E402
from cognisom.validation import ar_heterogeneity_singlecell as sourced  # noqa: E402

MODS = {"cellular": True, "spatial": True, "vascular": True,
        "ar_signaling": True, "immune": False, "lymphatic": False,
        "molecular": False, "epigenetic": False, "circadian": False,
        "morphogen": False, "cell_mechanics": False}

#: Measured 2026-11-24, ring/untreated: the cancer count with the
#: spread on, as a fraction of the homogeneous run. Below 1 because the
#: coupling is one-sided, and falling with the horizon because the
#: population is a compounding mixture. This is the calibration gap
#: that keeps the spread off by default.
MEASURED_GROWTH_RATIO = {24.0: 0.85, 48.0: 0.80, 72.0: 0.75}


def _run(sigma, hours=24.0, scenario="Baseline", drugs=None, seed=11,
         override=True):
    from cognisom.dashboard.engine_runner import EngineRunner
    np.random.seed(seed)
    over = ({"cellular": {"ar_drive_sigma": sigma}} if override else None)
    runner = EngineRunner(tissue="ring", dt=0.05, duration=hours,
                          scenario=scenario, drugs=drugs,
                          modules_enabled=MODS, overrides=over)
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    return runner


def _counts(runner):
    s = runner.engine.modules["cellular"].get_state()
    return tuple(s[k] for k in ("n_cancer", "n_normal", "total_divisions",
                                "total_deaths", "androgen_withdrawal_deaths"))


# ── 1. off is bit-identical ────────────────────────────────────────


@pytest.mark.slow
def test_the_spread_is_off_by_default_and_changes_nothing():
    """The whole restructuring has to be inert at sigma 0."""
    assert _counts(_run(0.0)) == _counts(_run(None, override=False))


def test_the_default_really_is_zero():
    """A default that drifted to the sourced value would turn the
    feature on for every caller without anyone choosing it."""
    from cognisom.modules.cellular_module import CellularModule
    mod = CellularModule(config={})
    assert mod.ar_drive_sigma == 0.0
    assert cm.ar_drive_from_id(12345, 0.0) == 1.0
    assert cm.ar_drive_rate_correction(0.0) == 1.0


def test_a_cell_starts_at_the_population_median():
    from cognisom.modules.cellular_module import CellState
    assert CellState(cell_id=1, position=np.zeros(3),
                     cell_type="cancer", phase="G1").ar_drive == 1.0


# ── 2. deterministic, and the measured spread ──────────────────────


def test_the_drive_is_deterministic_and_takes_no_random_draw():
    """Same id, same value, regardless of the global RNG state."""
    np.random.seed(1)
    first = [cm.ar_drive_from_id(i, 0.34) for i in range(50)]
    np.random.seed(999)
    [np.random.random() for _ in range(100)]
    assert [cm.ar_drive_from_id(i, 0.34) for i in range(50)] == first


def test_the_drive_distribution_matches_what_was_sourced():
    """The claim is that the engine's spread IS the measured one; the
    numbers are pinned beside it. Re-measure, do not re-pin."""
    d = np.array([cm.ar_drive_from_id(i, cm.AR_DRIVE_SIGMA_SOURCED)
                  for i in range(20000)])
    assert np.median(d) == pytest.approx(1.0, abs=0.02)
    assert float(np.log(d).std()) == pytest.approx(
        cm.AR_DRIVE_SIGMA_SOURCED, abs=0.02)
    # the linear CV a lognormal of this sigma implies -- NOT the 0.456
    # measured linearly, which would want sigma 0.435. The gap is the
    # honest uncertainty and is recorded beside the constant.
    assert float(d.std() / d.mean()) == pytest.approx(0.35, abs=0.03)
    assert float((d < cm.AR_INTACT_ABOVE).mean()) == pytest.approx(
        0.44, abs=0.03), (
        "the fraction below the dead band is what makes the band "
        "load-bearing once drive is spread; the 2026-11-17 sweep found "
        "it inert only because every cell sat at the same value")


def test_the_engine_constant_points_at_what_sourced_it():
    """A constant whose provenance comment rots is an unsourced
    constant, which is the thing this whole exercise was avoiding. The
    comment lives in the module source, so that is what is read."""
    import inspect
    src = inspect.getsource(cm)
    head = src[:src.index("AR_DRIVE_SIGMA_SOURCED =")]
    assert cm.AR_DRIVE_SIGMA_SOURCED == pytest.approx(0.34, abs=0.001)
    assert "ar_heterogeneity_singlecell" in head, (
        "the sigma constant no longer names the module that measured "
        "it; a reader cannot tell whether 0.34 was sourced or guessed")
    assert "UPPER BOUND" in head, (
        "the caveat that this is a readout rather than occupancy has "
        "to travel with the constant")
    # and the module it names must still re-derive it
    assert sourced.AR_EXCESS_MIN > 1.0
    assert hasattr(sourced, "run")


# ── inheritance: a lineage property, not per-cell noise ────────────


@pytest.mark.slow
def test_a_daughter_inherits_its_parents_drive_exactly():
    """If drive were re-drawn per cell it would be noise, and no clone
    could out-grow another. Exact inheritance also means grouping the
    final population by drive recovers the founding clones."""
    runner = _run(0.34, hours=48.0)
    cel = runner.engine.modules["cellular"]
    cancer = [c for c in cel.cells.values() if c.cell_type == "cancer"]
    drives = {round(c.ar_drive, 12) for c in cancer}
    assert len(cancer) > len(drives), (
        f"{len(cancer)} cells carry {len(drives)} distinct drives: with "
        "exact inheritance there must be fewer clones than cells")
    assert len(drives) <= 40, "more clones than the ring seeds founders"


# ── 3. division and death read different numbers ───────────────────


@pytest.mark.slow
def test_an_untreated_run_has_no_withdrawal_deaths_despite_low_drive():
    """THE REASON the hazard was re-keyed. With drive spread, 44% of
    cells sit below the dead band in absolute signalling; none of them
    is withdrawing from anything."""
    runner = _run(0.34, hours=48.0)
    cel = runner.engine.modules["cellular"]
    ar = runner.engine.modules["ar_signaling"]
    cancer = [cid for cid, c in cel.cells.items() if c.cell_type == "cancer"]
    sig = np.array([ar.signalling_fraction(c) for c in cancer], dtype=float)
    sup = np.array([ar.suppression_fraction(c) for c in cancer], dtype=float)

    assert (sig < cm.AR_INTACT_ABOVE).any(), (
        "no cell is below the band, so this test cannot show the "
        "hazard ignoring intrinsic drive")
    assert float(sup.max() - sup.min()) < 1e-3, (
        "suppression should be uniform in an untreated arm: it is what "
        "TREATMENT did, and nothing was given")

    # COMPARATIVE, not absolute. "Zero withdrawal deaths untreated" is
    # a claim about the ring config, not about the engine: a section
    # run has a few, from cells sitting far enough from a vessel that
    # their androgen supply really is low -- measured at signalling
    # ~0.54 during the 2026-11-17 band sweep, and present at sigma 0
    # too. The claim that belongs here is that SPREADING DRIVE adds
    # none, which is what the re-keying buys and what would break if
    # the hazard read absolute signalling again.
    without = _run(0.0, hours=48.0).engine.modules["cellular"]
    assert (cel.androgen_withdrawal_deaths
            == without.androgen_withdrawal_deaths), (
        f"spreading drive changed androgen-withdrawal deaths from "
        f"{without.androgen_withdrawal_deaths} to "
        f"{cel.androgen_withdrawal_deaths} in an UNTREATED tumour. A "
        "cell with intrinsically low drive is not deprived; the hazard "
        "is reading absolute signalling instead of suppression.")


@pytest.mark.slow
def test_drive_spreads_signalling_but_not_suppression_under_a_drug():
    """Under blockade the two separate most visibly: every cell is
    suppressed to the same 0.10, while absolute signalling spreads."""
    runner = _run(0.34, hours=24.0, drugs=("Enzalutamide",))
    ar = runner.engine.modules["ar_signaling"]
    cel = runner.engine.modules["cellular"]
    cancer = [cid for cid, c in cel.cells.items() if c.cell_type == "cancer"]
    sig = np.array([ar.signalling_fraction(c) for c in cancer], dtype=float)
    sup = np.array([ar.suppression_fraction(c) for c in cancer], dtype=float)
    assert float(sig.max() - sig.min()) > 0.05, "drive did not spread"
    assert float(sup.max() - sup.min()) < 1e-3
    assert float(np.median(sup)) == pytest.approx(0.10, abs=0.02)


@pytest.mark.slow
def test_suppression_equals_signalling_while_every_cell_is_median():
    """At drive 1.0 the two must be the same number, which is what
    makes the sigma-0 run bit-identical."""
    runner = _run(0.0, hours=24.0)
    ar = runner.engine.modules["ar_signaling"]
    cel = runner.engine.modules["cellular"]
    for cid, cell in cel.cells.items():
        if cell.cell_type != "cancer":
            continue
        a, b = ar.signalling_fraction(cid), ar.suppression_fraction(cid)
        if a is None:
            continue
        assert a == pytest.approx(b, rel=1e-12), (cid, a, b)


# ── the calibration gap, measured and stated ───────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("hours", sorted(MEASURED_GROWTH_RATIO))
def test_the_spread_slows_population_growth_by_the_measured_amount(hours):
    """Not a bug and not correctable by a constant.

    The coupling is one-sided, so E[rate] <= 1 for any spread: adding
    heterogeneity MUST slow the population. A growing population is
    also a compounding mixture, so scaling every interval by the mean
    rate overshoots -- measured at 1.05 / 1.20 / 1.50 against the
    homogeneous run at 24 / 48 / 72 h, worse the longer it runs. That
    is why no correction is applied and why the spread is off by
    default: the base division time needs re-deriving against the
    measured doubling WITH heterogeneity first.
    """
    off = _counts(_run(0.0, hours=hours))[0]
    on = _counts(_run(0.34, hours=hours))[0]
    assert off > 0
    ratio = on / off
    assert ratio == pytest.approx(MEASURED_GROWTH_RATIO[hours], abs=0.08), (
        f"{hours:.0f} h: spread-on cancer count is {ratio:.3f} of the "
        f"homogeneous run, was {MEASURED_GROWTH_RATIO[hours]}. If the "
        "base division time has been re-derived this SHOULD move -- "
        "re-measure and update the table, and the limitation with it")
    assert ratio < 1.0, (
        "the spread is no longer slowing growth; under a one-sided gate "
        "it must, so either the gate was symmetrised (do not -- Balbas "
        "2013) or a correction was applied after all")


def test_the_mean_rate_factor_is_reported_but_not_applied():
    """The helper stays as a diagnostic. Applying it was measured to
    overshoot by 50% at 72 h, so a reader needs to find out why from
    the code rather than rediscover it."""
    assert cm.ar_drive_rate_correction(0.34) == pytest.approx(0.890, abs=0.01)
    doc = cm.ar_drive_rate_correction.__doc__ or ""
    assert "DIAGNOSTIC ONLY" in doc
    assert "compounding" in doc.lower()
    import inspect
    src = inspect.getsource(cm.CellularModule.update)
    assert "division_time" in src, (
        "the division decision has moved out of update(); point this "
        "check at wherever it went, or it guards nothing")
    assert "_ar_drive_rate_correction" not in src, (
        "the correction is being applied to division times again; it "
        "cannot preserve a compounding mixture's growth rate")
