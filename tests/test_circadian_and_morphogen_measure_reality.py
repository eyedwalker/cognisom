"""Two modules whose published numbers were wrong rather than merely crude.

Both feed the dashboard, so these were plotted figures that did not mean what
they said.

**Circadian peak and trough counts.** ``total_troughs`` was structurally
unreachable -- the guard was ``old_phase < 0``, which can never hold because
phase is modulo 24 -- and ``total_peaks`` counted the phase-12 crossing, which
is the CLOCK/BMAL1 *trough* (``0.5 + 0.5*cos(pi) == 0``). Measured over 100
simulated hours with 20 cells: 80 "peaks", 0 troughs.

Detecting the sign change of ``d(CLOCK_BMAL1)/dt`` fixes the polarity but is
not sufficient on its own, and this is the part worth keeping a test on:
``dphase/dt = 1 + k*(master - phase)`` goes **negative** once a clock leads
the master by more than ``1/k``, which at the shipped ``coupling_strength``
of 0.1 is 10 hours. That reversal is a local extremum of CLOCK_BMAL1 at an
arbitrary point on the cycle, so a purely derivative-based detector counts it.
Measured on the real dashboard path: 6 of 15 "peaks" fired at
CLOCK_BMAL1 < 0.1, i.e. at the bottom of the cycle.

The midline guard closes it. The oscillation's mean is exactly 0.5 at any
amplitude, so a genuine maximum is always above it and a genuine minimum
always below.

Every test here runs at the coupling the dashboard actually passes. An earlier
test suite pinned ``coupling_strength=0.0``, which no caller uses --
``engine_runner`` passes ``{}`` so the dataclass default of 0.1 applies -- and
was therefore blind to the whole failure mode.

**Morphogen positions.** ``_sense_morphogens`` read ``identity.position``, a
copy taken at ``add_cell`` time and never refreshed, so every cell's morphogen
levels were frozen at its birth values however far it migrated. Measured: all
90 fates assigned at init, no change over 24 h.
"""
from __future__ import annotations

import contextlib
import io

import pytest

#: What cognisom/dashboard/engine_runner.py actually passes: {} -> the
#: dataclass default. Tests that pin 0.0 test a configuration nothing ships.
SHIPPED_COUPLING = 0.1


def _run_clocks(coupling, hours=100.0, n_cells=20, dt=0.05, bus=None):
    """Run n_cells clocks for `hours`, returning the module."""
    from cognisom.modules.circadian_module import CircadianModule

    mod = CircadianModule({'coupling_strength': coupling})
    if bus is not None:
        mod.event_bus = bus
    with contextlib.redirect_stdout(io.StringIO()):
        mod.initialize()
        for cid in range(n_cells):
            mod.add_cell(cid, initial_phase=(cid * 24.0 / n_cells))
        t = 0.0
        while t < hours:
            mod.update(dt)
            if bus is not None:
                bus.process_events()
            t += dt
    return mod


# ── circadian counters ───────────────────────────────────────────────────

def test_troughs_are_reachable_at_all():
    """total_troughs was structurally impossible to increment."""
    mod = _run_clocks(SHIPPED_COUPLING)

    assert mod.get_state()['total_troughs'] > 0


def test_peaks_and_troughs_both_land_near_the_expected_count():
    """100 h over a 24 h period is ~4 cycles per cell."""
    n_cells = 20
    mod = _run_clocks(SHIPPED_COUPLING, hours=100.0, n_cells=n_cells)
    state = mod.get_state()

    expected = (100.0 / 24.0) * n_cells  # ~83
    for key in ('total_peaks', 'total_troughs'):
        assert 0.6 * expected < state[key] < 1.5 * expected, (
            f"{key} = {state[key]}, expected roughly {expected:.0f}"
        )


def test_counts_are_roughly_balanced():
    """An oscillation has one trough per peak, give or take the ends."""
    state = _run_clocks(SHIPPED_COUPLING).get_state()
    peaks, troughs = state['total_peaks'], state['total_troughs']

    assert abs(peaks - troughs) < 0.3 * max(peaks, troughs), (
        f"{peaks} peaks against {troughs} troughs is not one cycle each"
    )


@pytest.mark.parametrize("coupling", [0.0, SHIPPED_COUPLING, 0.3])
def test_no_peak_fires_below_the_midline(coupling):
    """The failure mode the midline guard exists for.

    At any non-zero coupling the phase advance can reverse, producing a local
    extremum of CLOCK_BMAL1 partway up the cycle. Without the guard those are
    counted as peaks -- 6 of 15 on the shipped configuration.
    """
    from cognisom.core.event_bus import EventBus, EventTypes

    bus = EventBus()
    peaks, troughs = [], []
    bus.subscribe(EventTypes.CIRCADIAN_PEAK, lambda d: peaks.append(d['CLOCK_BMAL1']))
    bus.subscribe(EventTypes.CIRCADIAN_TROUGH, lambda d: troughs.append(d['CLOCK_BMAL1']))

    _run_clocks(coupling, bus=bus)

    assert peaks, "no peaks were emitted at all"
    assert troughs, "no troughs were emitted at all"

    misfired_peaks = [v for v in peaks if v <= 0.5]
    misfired_troughs = [v for v in troughs if v >= 0.5]
    assert not misfired_peaks, (
        f"{len(misfired_peaks)} of {len(peaks)} peaks fired at or below the "
        f"midline (lowest {min(peaks):.4f}) -- those are troughs"
    )
    assert not misfired_troughs, (
        f"{len(misfired_troughs)} of {len(troughs)} troughs fired at or above "
        f"the midline (highest {max(troughs):.4f})"
    )


def test_the_events_the_docstring_promises_exist():
    """CIRCADIAN_PEAK/TROUGH were documented but absent from EventTypes."""
    from cognisom.core.event_bus import EventTypes

    assert hasattr(EventTypes, 'CIRCADIAN_PEAK')
    assert hasattr(EventTypes, 'CIRCADIAN_TROUGH')


# ── morphogen positions ──────────────────────────────────────────────────

def test_morphogen_tracks_the_living_population_not_the_seeded_one():
    """It tracked only the cells present at init, frozen at birth values."""
    from cognisom.dashboard.engine_runner import EngineRunner

    with contextlib.redirect_stdout(io.StringIO()):
        runner = EngineRunner(dt=0.05, duration=24.0, scenario="Baseline")
        runner.build()
        runner.run()
        morph = runner.engine.modules['morphogen']
        cellular = runner.engine.modules['cellular']

    alive = sum(1 for c in cellular.cells.values() if c.alive)
    tracked = morph.get_state()['n_cells_tracked']

    assert tracked >= alive * 0.8, (
        f"morphogen tracks {tracked} cells against {alive} alive"
    )


def test_the_secreted_field_is_driven_by_somebody():
    """The module once assigned embryonic fates nothing consumed; now it
    is the source side of a field. 'Driven by nobody' -- identically
    zero, skipped as idle -- was the cytokine field's state before."""
    from cognisom.dashboard.engine_runner import EngineRunner

    with contextlib.redirect_stdout(io.StringIO()):
        runner = EngineRunner(dt=0.05, duration=6.0, scenario="Baseline")
        runner.build()
        runner.run()
        morph = runner.engine.modules['morphogen']
        spatial = runner.engine.modules['spatial']

    # Over the run, from the snapshots: by 6 h the immune compartment
    # has cleared Baseline's tumour, so nothing is secreting at the end.
    assert max(h.n_secreting for h in runner.history) > 0, "no cell ever secreted"
    assert max(h.chemokine_max for h in runner.history) > 0.0, "the field stayed zero"
    assert spatial.fields['cytokine'].last_solver != 'idle'
    assert morph.get_state()['decay_length_um'] is not None


def test_gradients_are_not_a_no_op_update():
    """_update_gradients_simple was a literal `pass`."""
    import inspect

    from cognisom.modules.morphogen_module import MorphogenModule

    source = inspect.getsource(MorphogenModule)
    # The method must do something other than pass.
    assert source.count("pass\n") < 3, (
        "morphogen still has stub method bodies"
    )
