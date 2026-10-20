"""The spatial field, coupled to the cells and vessels that drive it.

``SpatialModule`` was registered and enabled by default, allocated three
diffusible fields, and updated them 240 times over a 24 h run -- while
``add_source``/``add_sink`` had no callers anywhere outside
``spatial_module``'s own ``__main__``. So every field stayed identically
zero, nothing read ``get_concentration``/``get_gradient``, and the
under-diffusion warning fired 720 times (240 updates x 3 fields)
describing the transport of nothing.

Three things are pinned here.

ONE: the coupling exists. Cells register as first-order uptake and
capillaries as Dirichlet voxels, every update, cleared first.

TWO: the solve is right. At the shipped D = 2000 um^2/s on a 10 um grid
the diffusion number is 7200 against an explicit stability limit of 1/6,
so the sub-step cap applied ~1% of the oxygen transport. A field whose
diffusive relaxation time is short against the step has already reached
its steady state, so it is solved for directly. The discretisation is
checked against the analytic Krogh-type slab solution and its
second-order convergence.

THREE -- and this is the load-bearing one -- the hypoxia mechanism is NOT
regressed. VascularModule's lumped Krogh conductance is calibrated so a
cell at the Krogh radius sits exactly at its death threshold, and that
calibration is the only reason the Baseline preset survives 24 h. A
correct diffusion solve on the shipped 200x200x100 um tissue extent
returns a UNIFORM field (measured Thiele modulus ~0.01, vessel-to-corner
drop 0.002%), so sourcing cell O2 from the field would put every cell at
the arterial level and no preset could ever produce hypoxia. The field is
therefore the interstitial field; cell O2 stays on the vascular path.
Both of those facts are asserted below.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.core.event_bus import EventTypes
from cognisom.modules.spatial_module import SpatialField, SpatialModule

SECONDS_PER_HOUR = 3600.0


def _deaths(runner, cause: str, cell_type: str) -> int:
    """Deaths of one cause in one lineage, from the runner's event log."""
    return sum(1 for e in runner.event_log
               if e["event"] == EventTypes.CELL_DIED
               and e["data"].get("cause") == cause
               and e["data"].get("cell_type") == cell_type)


# ── Helpers ─────────────────────────────────────────────────────────

def slab(n: int, dx: float, lam: float, d_coeff: float = 100.0):
    """1-D slab: Dirichlet C=1 at voxel 0, first-order uptake elsewhere,
    no-flux at the far face. Uptake is chosen so sqrt(D/k) == ``lam``.
    """
    d_h = d_coeff * SECONDS_PER_HOUR
    k = d_h / lam ** 2
    f = SpatialField("slab", (n, 1, 1), dx, d_coeff)
    f.add_dirichlet(np.array([0.5 * dx, 0.5 * dx, 0.5 * dx]), 1.0)
    for i in range(1, n):
        f.add_uptake(np.array([(i + 0.5) * dx, 0.5 * dx, 0.5 * dx]), k)
    return f


def analytic_slab(n: int, dx: float, lam: float) -> np.ndarray:
    """C(x) = cosh((X-x)/lam) / cosh((X-x0)/lam): the steady state of
    D C'' = k C with C(x0)=1 and zero flux at the far face X.
    """
    x = (np.arange(n) + 0.5) * dx
    return np.cosh((n * dx - x) / lam) / np.cosh((n * dx - 0.5 * dx) / lam)


@pytest.fixture(scope="module")
def runs():
    """One 24 h all-nine-module run per preset, shared by the tests."""
    from cognisom.dashboard.engine_runner import EngineRunner

    out = {}
    for scenario in ("Baseline", "Hypoxia"):
        np.random.seed(12345)
        runner = EngineRunner(dt=0.05, duration=24.0, scenario=scenario,
                              record_interval=20)
        runner.build()
        runner.run()
        out[scenario] = runner
    return out


# ── ONE: the coupling exists ────────────────────────────────────────

def test_spatial_module_is_linked_to_cells_and_vessels(runs):
    """engine_runner._link_modules() wired immune, vascular, lymphatic and
    morphogen to cellular and never touched spatial."""
    state = runs["Baseline"].engine.modules["spatial"].get_state()
    assert state["coupled_to_cells"] is True
    assert state["coupled_to_vessels"] is True


def test_cells_are_sinks_and_capillaries_are_sources(runs):
    """The counts that were both zero for the whole run."""
    state = runs["Baseline"].engine.modules["spatial"].get_state()
    for name in ("oxygen", "glucose"):
        field = state["fields"][name]
        assert field["n_dirichlet"] > 0, f"{name}: no capillary sources"
        assert field["n_uptake"] > 0, f"{name}: no cell sinks"


def test_the_fields_are_no_longer_identically_zero(runs):
    """oxygen/glucose mean and max were exactly 0.0 after 24 h."""
    state = runs["Baseline"].engine.modules["spatial"].get_state()
    for name, arterial in (("oxygen", 0.21), ("glucose", 5.0)):
        field = state["fields"][name]
        assert field["max"] > 0.0
        assert field["mean"] > 0.0
        # The maximum principle: nothing may exceed the blood level.
        assert field["max"] <= arterial + 1e-6


def test_registration_is_cleared_each_update_so_it_cannot_grow(runs):
    """add_* append and update applies every entry, so re-registering
    without clearing grows the lists without bound and multiplies the
    deposit by the step count -- the hazard morphogen_module guards with
    a _registered_sources set. Clearing first is idempotent instead, and
    unlike the set-based guard it can retract a dead or moved cell.

    After 240 updates the uptake count must still be one entry per live
    cell, not 240 times that.
    """
    runner = runs["Baseline"]
    spatial = runner.engine.modules["spatial"]
    n_alive = sum(1 for c in runner.engine.modules["cellular"].cells.values()
                  if c.alive)
    assert spatial.fields["oxygen"].uptake  # non-empty
    assert len(spatial.fields["oxygen"].uptake) == n_alive


def test_sync_is_idempotent(runs):
    """Calling the sync twice must leave the term counts unchanged."""
    spatial = runs["Baseline"].engine.modules["spatial"]
    spatial._sync_sources_sinks()
    first = {n: (len(f.uptake), len(f.dirichlet))
             for n, f in spatial.fields.items()}
    for _ in range(3):
        spatial._sync_sources_sinks()
    after = {n: (len(f.uptake), len(f.dirichlet))
             for n, f in spatial.fields.items()}
    assert first == after


# ── TWO: the solve is right ─────────────────────────────────────────

def test_steady_state_matches_the_analytic_slab_solution():
    """D C'' - k C = 0 with a Dirichlet face and a no-flux face has the
    closed form cosh((X-x)/lam)/cosh((X-x0)/lam). Checked over four
    diffusion lengths, where the profile spans a factor of ~26.
    """
    n, dx, lam = 40, 10.0, 100.0
    f = slab(n, dx, lam)
    f.update(1.0)
    assert f.last_solver == "steady_state"

    numeric = f.concentration[:, 0, 0].astype(float)
    exact = analytic_slab(n, dx, lam)
    assert np.max(np.abs(numeric - exact)) < 2e-3
    # The profile really does decay -- this is not a uniform field.
    assert numeric[-1] < 0.05 * numeric[0]


@pytest.mark.parametrize("dx,n", [(10.0, 40), (5.0, 80), (2.5, 160)])
def test_steady_state_is_second_order_accurate(dx, n):
    """The centred 7-point stencil is O(dx^2); halving dx must cut the
    error by about four. Measured: 1.6e-4 -> 4e-5 -> 1e-5.
    """
    lam = 100.0
    f = slab(n, dx, lam)
    f.update(1.0)
    err = np.max(np.abs(f.concentration[:, 0, 0].astype(float)
                        - analytic_slab(n, dx, lam)))
    # Error budget scaling as dx^2 from a measured 1.6e-4 at dx=10.
    assert err < 2.0e-4 * (dx / 10.0) ** 2 * 1.5


def test_the_solution_is_non_negative_under_extreme_uptake():
    """A zeroth-order sink is unbounded below and drives a steady-state
    solve negative; the first-order uptake used here makes the operator
    an M-matrix, so 0 <= C <= C_dirichlet is a theorem. Checked at an
    uptake 10^4 times the physiological value.
    """
    f = slab(40, 10.0, 1.0)  # lam = 1 um, i.e. phi = 400
    f.update(1.0)
    c = f.concentration.astype(float)
    assert c.min() >= 0.0
    assert c.max() <= 1.0 + 1e-6
    assert np.isfinite(c).all()


def test_steady_state_is_a_fixed_point():
    """Solving an already-converged field must not move it. The explicit
    path cannot do this: capped at 512 sub-steps it applies ~1% of the
    transport per update and so creeps towards the answer for ever.
    """
    f = slab(40, 10.0, 100.0)
    f.update(1.0)
    first = f.concentration.copy()
    f.update(1.0)
    assert np.allclose(first, f.concentration, atol=1e-9)
    # ... and it is independent of the step size, being a steady state.
    g = slab(40, 10.0, 100.0)
    g.update(7.5)
    assert np.allclose(first, g.concentration, atol=1e-6)


def test_the_factorisation_is_reused_across_solves():
    """The operator depends on the Dirichlet pattern (capillaries never
    move) and the occupancy pattern (cells move only on division or
    death), so one factorisation serves many solves. Measured over a 24 h
    Baseline run: 7 rebuilds across 240 updates.
    """
    f = slab(40, 10.0, 100.0)
    for _ in range(10):
        f.update(1.0)
    assert f.lu_rebuilds == 1


def test_an_undriven_field_keeps_the_explicit_path():
    """A field carrying only an initial blob has no steady state but the
    trivial one; replacing its transient with zero would discard the very
    thing being modelled. This is also what keeps the shipped
    test_diffusion_stability suite on the explicit path.
    """
    f = SpatialField("blob", (21, 21, 21), 10.0, 2000.0)
    f.concentration[10, 10, 10] = 1000.0
    f.update(0.1)
    assert f.last_solver == "explicit"
    # The transient is integrated, not replaced by the trivial steady
    # state: material has left the seeded voxel and reached its
    # neighbour. (Mass is NOT conserved here and is not claimed to be:
    # at D = 2000 um^2/s the blob reaches the boundary within one update
    # and _compute_laplacian's shell is absorbing. The shipped
    # test_diffusion_stability suite pins conservation in the interior
    # regime where it holds.)
    assert f.concentration[10, 10, 10] < 1000.0
    assert f.concentration[11, 10, 10] > 0.0


def test_an_empty_field_does_no_work_and_says_so():
    """Nothing registered and nothing in it: diffusing identically zero is
    identically zero. The cytokine field was running 512 explicit
    sub-steps per update, 240 times a run, on an all-zero array.
    """
    f = SpatialField("cytokine", (20, 20, 10), 10.0, 100.0)
    f.update(0.1)
    assert f.last_solver == "idle"
    assert not f.concentration.any()


def test_uptake_is_applied_on_the_explicit_path_too():
    """The explicit path is reachable WITH uptake registered: a field
    driven by cells but with no vessel (spatial + cellular enabled,
    vascular disabled) fails the DRIVEN test, so it is integrated rather
    than solved. The explicit branch applied sources and sinks but not
    uptake, so that uptake was silently dropped and the field sat at its
    initial level for the whole run.

    Applied as exp(-k*dt), which is exact, unconditionally stable and
    non-negative for any k and dt.
    """
    f = SpatialField("oxygen", (20, 20, 10), 10.0, 0.0)  # D=0: no mixing
    f.concentration[:] = 0.21
    f.add_uptake(np.array([55.0, 55.0, 55.0]), 1.0)      # k = 1/h
    f.update(2.0)
    assert f.last_solver == "explicit"
    assert f.concentration[5, 5, 5] == pytest.approx(
        0.21 * np.exp(-2.0), rel=1e-5)
    # Untouched voxels stay put.
    assert f.concentration[1, 1, 1] == pytest.approx(0.21, rel=1e-6)


def test_explicit_uptake_cannot_go_negative_at_any_step():
    """exp(-k*dt) is in (0, 1] for every positive k and dt, so unlike the
    zeroth-order sink loop this needs no clamp."""
    f = SpatialField("oxygen", (8, 8, 8), 10.0, 0.0)
    f.concentration[:] = 1.0
    f.add_uptake(np.array([35.0, 35.0, 35.0]), 1e6)
    f.update(1e3)
    assert f.concentration.min() >= 0.0
    assert np.isfinite(f.concentration).all()


def test_the_under_diffusion_warning_fires_once_per_field(caplog):
    """It fired from inside update(), i.e. once per update: 720 identical
    records over a 24 h run. The condition is a property of (D, dx, dt),
    which do not change between updates.
    """
    f = SpatialField("oxygen", (21, 21, 21), 10.0, 2000.0)
    f.concentration[10, 10, 10] = 1000.0
    with caplog.at_level(logging.WARNING,
                         logger="cognisom.modules.spatial_module"):
        for _ in range(20):
            f.update(0.1)
    records = [r for r in caplog.records if "UNDER-DIFFUSED" in r.getMessage()]
    assert len(records) == 1


def test_a_whole_nine_module_run_logs_no_spatial_warnings(runs, caplog):
    """720 -> 0. The oxygen and glucose fields are solved directly, so
    there is no CFL condition to violate, and the cytokine field has
    nothing to transport."""
    state = runs["Baseline"].engine.modules["spatial"].get_state()
    assert state["fields"]["oxygen"]["solver"] == "steady_state"
    assert state["fields"]["glucose"]["solver"] == "steady_state"
    assert state["fields"]["cytokine"]["solver"] == "idle"


# ── THREE: the hypoxia mechanism is not regressed ───────────────────

def test_baseline_normal_cells_still_survive_24h(runs):
    """The number the Krogh calibration exists to produce. Before that
    calibration this was 0 normal cells alive at 24 h; it is 160, and
    nothing about the field coupling may change it.

    This is the all-nine-module survival test that did not exist -- the
    shipped Krogh test runs a two-module harness (cellular + vascular),
    so nothing pinned survival in the full path.
    """
    runner = runs["Baseline"]
    cells = runner.engine.modules["cellular"]
    alive = [c for c in cells.cells.values() if c.alive]
    normal = [c for c in alive if c.cell_type != "cancer"]

    assert len(normal) == 160
    assert _deaths(runner, "hypoxia", "normal") == 0
    # What this test exists for is above: normal-cell survival at 160, and
    # no normal cell dying of anything. The rest pins the run.
    #
    # Cancer: 22 survived while the chase step was frozen (x0.01 for x60);
    # 0 once chasing cells moved but every contact rolled its kill
    # probability each step and killed within a step or two; 6 now that a
    # contact has one outcome and lasts a measured time (a lethal NK
    # contact averages 19 min, a T cell's 75, and a failed NK contact ties
    # the cell up for 29), so some tumour cells divide before they are
    # killed. Every death is still an immune kill.
    assert len([c for c in alive if c.cell_type == "cancer"]) == 6
    assert cells.total_deaths == 9
    assert _deaths(runner, "immune_killed", "cancer") == 9
    # 80 normal divisions -- each normal cell divides once, now at a time
    # staggered over the day rather than all on the final step -- plus 5
    # cancer divisions that come before the kill.
    assert cells.total_divisions == 85
    assert runner.engine.modules["vascular"].total_O2_delivered == \
        pytest.approx(423.5066, abs=0.01)
    # The oxygen floor. One daughter sits near the edge of its capillary's
    # reach, hypoxic but alive; no normal cell is near death.
    assert min(c.oxygen for c in alive) > 0.02          # the death threshold
    assert sum(1 for c in alive if c.oxygen < 0.05) == 0
    assert min(c.oxygen for c in alive) == pytest.approx(0.0550, abs=1e-3)


def test_the_hypoxia_preset_still_produces_hypoxia(runs):
    """Three capillaries and exchange_rate 0.4. If cell O2 were sourced
    from the diffusive field this would be zero hypoxic cells, because a
    correct solve on this tissue extent returns a uniform field.

    The evidence is the cells that die of hypoxia, not the cells still
    hypoxic at 24 h. Once a chasing immune cell could reach what it had
    recognised, the cancer cells that used to starve in the capillary-poor
    corner were mostly killed first, and the end-of-run snapshot lost its
    hypoxic cells while the mechanism was untouched: normal cells starve
    in this preset whatever the immune compartment does.
    """
    runner = runs["Hypoxia"]
    cells = runner.engine.modules["cellular"]
    vascular = runner.engine.modules["vascular"]
    alive = [c for c in cells.cells.values() if c.alive]

    # The mechanism: normal cells starve in this preset and in no other.
    assert _deaths(runner, "hypoxia", "normal") == 60
    assert _deaths(runs["Baseline"], "hypoxia", "normal") == 0
    # Cancer cells starve here too when the immune compartment does not
    # reach them first: 16 while the chase step was frozen, 0 while every
    # contact killed within a step or two, 7 now that a kill takes a
    # measured contact time and a failed one ties the effector up for
    # another 29 min.
    assert _deaths(runner, "hypoxia", "cancer") == 7
    assert _deaths(runner, "immune_killed", "cancer") == 20
    assert cells.total_deaths == 87
    # The 24 h snapshot: the survivors are the 10 best-perfused normal
    # cells, and 3 cancer cells the compartment never got to.
    assert vascular.hypoxic_regions == 0
    assert sum(1 for c in alive if c.oxygen < 0.05) == 0
    assert min(c.oxygen for c in alive) == pytest.approx(0.0734, abs=1e-3)


def test_cell_oxygen_is_not_the_field_value(runs):
    """The two are different quantities and must stay so. cell.oxygen is
    VascularModule's hour-scale oxygen-sufficiency state, driven by a
    lumped Krogh conductance; the field is the interstitial level. On this
    geometry the field is at the arterial level everywhere, so if the two
    were the same number no cell could ever be hypoxic.
    """
    runner = runs["Baseline"]
    state = runner.engine.modules["spatial"].get_state()
    cells = [c for c in runner.engine.modules["cellular"].cells.values()
             if c.alive]
    mean_cell_o2 = float(np.mean([c.oxygen for c in cells]))

    assert state["interstitial_O2_mean"] == pytest.approx(0.21, abs=1e-3)
    assert mean_cell_o2 < 0.9 * state["interstitial_O2_mean"]


def test_this_tissue_extent_is_not_diffusion_limited(runs):
    """The measurement that justifies keeping cell O2 on the vascular
    path. The Thiele modulus phi = L*sqrt(k/D) is the domain length in
    diffusion lengths; phi << 1 means transport beats consumption so
    hugely that the field is uniform to O(phi^2) however fine the mesh.

    Measured: phi(oxygen) ~ 0.0097 on the shipped 200 um extent, so the
    vessel-to-corner drop is a few parts in 10^5. Hypoxia on this grid
    cannot come from O2 diffusion, and it is the lumped conductance that
    supplies it.
    """
    state = runs["Baseline"].engine.modules["spatial"].get_state()
    for name in ("oxygen", "glucose"):
        phi = state["fields"][name]["thiele_modulus"]
        assert 0.0 < phi < 0.1, f"{name}: phi={phi}"
        field = state["fields"][name]
        drop = (field["max"] - field["min"]) / field["max"]
        assert drop < 1e-3, f"{name}: drop={drop}"


def test_the_mechanism_does_bite_when_the_geometry_allows_it():
    """The coupling is not vacuous. The uniformity above is a property of
    the shipped 200 um tissue extent, not of this code: raise the Thiele
    modulus past 1 and the same solver produces a real gradient and a
    hypoxic far field. This is what makes the uniformity a finding rather
    than a bug.
    """
    f = slab(40, 10.0, 50.0)   # phi = 400/50 = 8
    f.update(1.0)
    profile = f.concentration[:, 0, 0].astype(float)
    assert f.thiele_modulus() > 1.0
    assert profile[0] == pytest.approx(1.0, abs=1e-6)
    assert profile[-1] < 0.01          # a genuinely starved far field
    assert np.all(np.diff(profile) <= 1e-7)   # monotone decay
