"""AR_INTACT_ABOVE is not load-bearing, and the reason matters more than
the constant -- MEASURED, 2026-11-17.

The external AR/proliferation tests (493 TCGA prostate tumours, rho
+0.41; 24 paired pre/post-ADT biopsies, rho +0.49) put the 0.95 dead
band under tension: the data show AR activity and proliferation graded
together across the normal range, while the engine has a flat zone
covering most of it. The band was set to 0.95 to keep the Baseline run
bit-identical -- a convenience, not a measurement.

So the band was swept, 0.80/0.90/0.95/0.98/1.00 x both arms x three
seeds, in the shipped section configuration. What came back:

  1. Under a drug, the band is INERT. All five bands gave runs
     identical to every digit, mean PSA included -- while differing
     between SEEDS at the 7th decimal, which is what shows the
     comparison was live rather than a constant echoed back.

  2. Untreated, the band does perturb the random stream -- but by less
     than changing the seed does. Band range of the mean cancer count
     18.7 cells (0.70%) against a worst seed spread of 23; divisions
     0.40% against 25.

  3. THE REASON, and the thing this file pins: AR signalling has almost
     no cell-to-cell variation WITHIN an arm. Every untreated cancer
     cell sits within ~3e-4 of every other, just under 1.0. The
     variation is between arms (1.00 / 0.50 / 0.10), not between cells.

(3) is why the band is nearly irrelevant and why moving it would not
answer the external tension. A graded association needs cells that
DIFFER in AR drive; the engine gives them all the same number. Dialling
the band narrower would flatten nothing and grade nothing -- there is
nothing in the band's range to grade. It would, however, move every
pinned division count in the suite. So the fix is cell-to-cell
variation in AR drive, not a different threshold.

Two traps this had to get past, both recorded because each gave a
confident wrong answer first:

  - I predicted the UNTREATED arm would be the band-invariant one,
    since cellular_module says an intact run "takes no draw at all".
    It is the other way round: a drug drops signalling to 0.10, below
    every band, so every band takes the same branch, while untreated
    cells sit a few 1e-4 under 1.0 and a band at 1.00 therefore makes
    86% of the tumour roll a die every step.
  - I then suspected my own sweep harness of leaking state between
    runs in one process. It was not: re-running each cell in a FRESH
    process reproduced the single-process counts exactly, and two
    duplicate runs of one cell came back bit-identical.

The cheap ring configuration below is the same one
test_castration_slows_and_kills_the_dependent_tumour.py uses, because
a section run costs ~3.7 min against the suite's 120 s per-test
timeout. The near-delta shape was confirmed in BOTH: section gives
p01..p99 of 6e-4 over ~2,690 cells, ring 3.5e-4 over 40.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

pytest.importorskip("scipy")

from cognisom.modules import cellular_module as cm  # noqa: E402

# Same reduced set as the castration test: the AR path needs cellular,
# spatial, vascular and ar_signaling, and nothing else here reads it.
MODS = {"cellular": True, "spatial": True, "vascular": True,
        "ar_signaling": True, "immune": False, "lymphatic": False,
        "molecular": False, "epigenetic": False, "circadian": False,
        "morphogen": False, "cell_mechanics": False}

#: Measured 2026-11-17, ring/24 h/seed 11. The claim is the SHAPE --
#: one number per arm, no spread between cells -- and these are the
#: numbers beside it. Re-measure, do not re-pin, if they move.
EXPECTED = {
    "untreated": 1.00,
    "enzalutamide": 0.10,
    "castration": 0.50,
}
#: The widest p01..p99 spread any arm showed. Pinned an order of
#: magnitude loose: the claim is "no spread", not this exact value.
WIDEST_WITHIN_ARM_SPREAD = 1e-3


def _run(scenario, drugs=None, hours=24.0, seed=11):
    from cognisom.dashboard.engine_runner import EngineRunner
    np.random.seed(seed)
    runner = EngineRunner(tissue="ring", dt=0.05, duration=hours,
                          scenario=scenario, drugs=drugs,
                          modules_enabled=MODS)
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    return runner


def _signalling(runner) -> np.ndarray:
    cel = runner.engine.modules["cellular"]
    cancer = [cid for cid, c in cel.cells.items() if c.cell_type == "cancer"]
    return np.asarray(
        [s for s in (cel._ar_signalling(c) for c in cancer) if s is not None],
        dtype=float)


@pytest.fixture(scope="module")
def untreated():
    return _signalling(_run("Baseline"))


@pytest.fixture(scope="module")
def enzalutamide():
    return _signalling(_run("Baseline", drugs=("Enzalutamide",)))


@pytest.fixture(scope="module")
def castration():
    return _signalling(_run("Androgen Deprivation"))


# ── the shape: one number per arm ──────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("arm", sorted(EXPECTED))
def test_every_cell_in_an_arm_has_the_same_ar_signalling(
        arm, untreated, enzalutamide, castration):
    """The finding. Cell-to-cell spread within an arm is ~1e-4, so the
    engine has no graded AR signalling for a correlation to grade."""
    sig = {"untreated": untreated, "enzalutamide": enzalutamide,
           "castration": castration}[arm]
    assert sig.size, f"{arm}: no mapped cancer cells, nothing measured"
    spread = float(np.percentile(sig, 99) - np.percentile(sig, 1))
    assert spread <= WIDEST_WITHIN_ARM_SPREAD, (
        f"{arm}: AR signalling now spans {spread:.2e} between cells "
        f"(was <= {WIDEST_WITHIN_ARM_SPREAD:.0e}). If the engine has "
        "gained real cell-to-cell variation in AR drive this is GOOD "
        "news -- it is what limitation 2 on ?page=methods asks for. "
        "Re-measure, update the page, and do not simply widen this.")
    assert float(np.median(sig)) == pytest.approx(EXPECTED[arm], abs=0.01), (
        f"{arm}: median signalling moved off {EXPECTED[arm]}")


@pytest.mark.slow
def test_the_between_arm_difference_dwarfs_the_between_cell_spread(
        untreated, enzalutamide):
    """All of the engine's AR variation is between arms. 0.9 apart
    across the drug, ~1e-4 apart between cells in either."""
    between_arms = abs(float(np.median(untreated))
                       - float(np.median(enzalutamide)))
    within = max(
        float(np.percentile(a, 99) - np.percentile(a, 1))
        for a in (untreated, enzalutamide))
    assert between_arms > 100 * within, (
        f"between arms {between_arms:.4f}, within an arm {within:.2e}")


# ── why the band cannot matter under a drug ────────────────────────


@pytest.mark.slow
def test_a_blocked_cell_sits_far_below_every_plausible_dead_band(
        enzalutamide):
    """Enzalutamide puts every cell at ~0.10 of eugonadal -- below 0.80
    as surely as below 1.00. That is why the whole 0.80-1.00 sweep gave
    identical treated runs: every band takes the same branch."""
    assert float(enzalutamide.max()) < 0.80, (
        f"a blocked cell reached {enzalutamide.max():.4f}; the band "
        "sweep's conclusion that no band can change a treated run "
        "assumed every blocked cell was below 0.80")
    assert float(enzalutamide.max()) < cm.AR_INTACT_ABOVE


@pytest.mark.slow
def test_no_untreated_cell_sits_in_the_band_sensitive_window(untreated):
    """Below 0.98 is where a band of 0.95 differs from one of 0.98.
    Untreated cells are all above it, so only a band at 1.00 bites --
    and that is the drift the dead band exists to absorb."""
    assert float(untreated.min()) >= 0.98, (
        f"an untreated cell fell to {untreated.min():.4f}, inside the "
        "window where the band's value changes which branch is taken")


@pytest.mark.slow
def test_the_dead_band_does_not_change_a_treated_run(monkeypatch):
    """The sweep's headline, at its cheapest: the extreme bands give
    the same treated run. monkeypatch, so the module constant is
    restored for every other test in the session."""
    out = {}
    for band in (0.80, 1.00):
        monkeypatch.setattr(cm, "AR_INTACT_ABOVE", band)
        runner = _run("Baseline", drugs=("Enzalutamide",))
        state = runner.engine.modules["cellular"].get_state()
        out[band] = tuple(state[k] for k in
                          ("n_cancer", "n_normal", "total_divisions",
                           "total_deaths", "androgen_withdrawal_deaths"))
    assert out[0.80] == out[1.00], (
        f"the dead band now changes a treated run: {out}. The methods "
        "page says it cannot, because a blocked cell is below every "
        "band -- check that before changing this.")


class _CountingRandom:
    """Counts calls and delegates, so the stream is untouched."""

    def __init__(self, real):
        self._real = real
        self.n = 0

    def __call__(self, *args, **kwargs):
        self.n += 1
        return self._real(*args, **kwargs)


def _draws(monkeypatch, band, **run_kwargs) -> int:
    monkeypatch.setattr(cm, "AR_INTACT_ABOVE", band)
    counter = _CountingRandom(np.random.random)
    monkeypatch.setattr(np.random, "random", counter)
    _run("Baseline", **run_kwargs)
    return counter.n


@pytest.mark.slow
def test_the_dead_band_decides_whether_an_untreated_cell_draws(monkeypatch):
    """NON-VACUITY for the test above, proved at the mechanism instead
    of the outcome.

    The first attempt at this compared untreated OUTCOMES at 0.95 and
    1.00 and found them equal, which looked like a broken patch. It is
    not: in this cheap ring config the withdrawal hazard at 1.00 is
    rate * (1 - 0.9997), so the extra draws happen and then almost
    never kill anything. The band genuinely changes which BRANCH is
    taken while leaving the 40-cell outcome alone.

    So count the draws. At 0.95 an untreated cell is intact and takes
    none; at 1.00 every AR-dependent cell takes one per step. If those
    two counts were equal, the patch would not be reaching the code and
    this file's treated-arm equality would prove nothing.
    """
    intact = _draws(monkeypatch, 0.95)
    strict = _draws(monkeypatch, 1.00)
    assert strict > intact, (
        f"AR_INTACT_ABOVE=1.00 took {strict} draws against {intact} at "
        "0.95. Equal counts mean the patch is not reaching the hazard "
        "branch, and the treated-arm equality above is vacuous.")
