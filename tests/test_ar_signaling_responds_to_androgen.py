"""Androgen deprivation must collapse PSA, and a bad solver must not hide it.

This is the central claim of the prostate model: remove the androgen and the
AR-driven PSA output falls away. Nothing pinned it. ``DHT_input`` appeared in
no test, so the one behaviour the whole AR pathway exists to reproduce was
free to break silently.

The second half of this file matters more than the first. The AR system is
stiff: with AR near 4.3e4 and k_on 3.6 /nM/h, DHT turns over on a ~1.7 second
timescale while the slow species move over hundreds of hours. Integrated
explicitly at a step coarser than that, and with negative concentrations
clamped to zero, the clamp injects mass that was never supplied: it produces
AR_DHT near 8.5e4 nM against a supply bound of 20 nM, saturates the PSA
activation term 1.7e5-fold, and PSA then does not respond to androgen at all.
That failure looks exactly like a modelling error -- the numbers are smooth,
finite and plausible -- which is why it needs an invariant rather than an eye.

:func:`test_the_complex_cannot_exceed_what_the_androgen_supply_allows` is that
invariant. AR_DHT is fed only by DHT, which arrives at ``DHT_input`` per hour,
and leaves at ``k_complex_deg``, so no steady state above the ratio of the two
is reachable however the integration is done.
"""
from __future__ import annotations

import contextlib
import io
import logging

import numpy as np
import pytest

try:
    from scipy.integrate import solve_ivp
except ImportError:                                    # pragma: no cover
    solve_ivp = None

pytestmark = pytest.mark.skipif(solve_ivp is None, reason="scipy required")

#: Long enough for the slowest species (AR protein, k_deg 0.1 /h) to settle.
HOURS = 2000.0


@pytest.fixture(scope="module")
def builder():
    logging.disable(logging.INFO)
    from cognisom.genomics.reaction_builder import ReactionNetworkBuilder
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    yield ReactionNetworkBuilder(store)
    logging.disable(logging.NOTSET)


def _steady_state(builder, dht_input: float):
    """Integrate to steady state with a stiff solver.

    LSODA, not a fixed-step explicit method: see the module docstring.
    """
    system = builder.build_ar_signaling()
    system.parameters["DHT_input"] = dht_input
    rhs = system.get_rhs_func()
    params = dict(system.parameters)

    sol = solve_ivp(
        lambda t, y: rhs(t, y, params),
        (0.0, HOURS),
        np.array(system.species_initial, dtype=float),
        method="LSODA", rtol=1e-8, atol=1e-10,
    )
    assert sol.success, f"integration failed: {sol.message}"
    return dict(zip(system.species_names, sol.y[:, -1])), system


# ── the behaviour the pathway exists to reproduce ────────────────────────

def test_androgen_deprivation_collapses_psa(builder):
    intact, _ = _steady_state(builder, 1.0)
    deprived, _ = _steady_state(builder, 0.0)

    assert intact["PSA"] > 1.0, (
        f"PSA does not accumulate even with androgen present "
        f"({intact['PSA']:.4g}); the pathway is broken upstream of ADT")

    ratio = deprived["PSA"] / intact["PSA"]
    assert ratio < 0.05, (
        f"androgen deprivation left PSA at {ratio:.1%} of its intact level. "
        "This is the behaviour the AR pathway exists to reproduce")


def test_psa_is_monotonic_in_the_androgen_supply(builder):
    levels = [0.0, 0.05, 0.25, 1.0]
    psa = [_steady_state(builder, d)[0]["PSA"] for d in levels]

    assert all(b >= a - 1e-6 for a, b in zip(psa, psa[1:])), (
        f"PSA is not monotonic in androgen supply: {psa}")


def test_the_psa_induction_term_is_saturated_at_physiological_androgen(builder):
    """Records a live mis-scaling, so the cause is on file rather than rediscovered.

    ``K_psa`` is read from the AR gene's ``dna_binding_kd_nm``, which is the
    affinity of AR for the androgen response element on DNA. It is then used
    as the half-maximal constant for PSA induction *by the AR-DHT complex*.
    Those are different quantities, and the substitution is a category error
    rather than a unit error, so no conversion factor repairs it.

    The consequence is quantitative: AR_DHT settles at 20 nM against a Km of
    0.5 nM, so the activation term runs at 40/41 of Vmax. The model is in
    saturation across the whole physiological range, and the clinically
    meaningful behaviour -- PSA falling progressively under partial androgen
    suppression -- is unreachable. Full deprivation still works, which is why
    this went unnoticed.
    """
    state, system = _steady_state(builder, 1.0)
    km = system.parameters["K_psa"]
    occupancy = state["AR_DHT"] / (km + state["AR_DHT"])

    assert occupancy > 0.9, (
        f"the induction term now runs at {occupancy:.1%} of Vmax rather than "
        "in saturation. If the Km source was fixed, update this test and "
        "re-enable the dose-response assertion below")


@pytest.mark.xfail(strict=True, reason=(
    "K_psa is sourced from the AR-DNA binding affinity (0.5 nM) but used as "
    "the Km for PSA induction by the AR-DHT complex, which settles at 20 nM. "
    "The induction term is therefore 97.6% saturated and PSA moves only "
    "1.46x across a 20x change in androgen. Fixing it needs a defensible "
    "half-maximal constant for AR-driven KLK3 transcription, which the "
    "library does not currently hold."))
def test_psa_responds_to_partial_androgen_blockade(builder):
    """Partial suppression should move PSA substantially, not marginally."""
    psa_low = _steady_state(builder, 0.05)[0]["PSA"]
    psa_full = _steady_state(builder, 1.0)[0]["PSA"]

    assert psa_full > psa_low * 1.5, (
        f"PSA barely moves between 5% and full androgen ({psa_low:.4g} -> "
        f"{psa_full:.4g})")


def test_the_androgen_receptor_complex_forms_at_all(builder):
    intact, _ = _steady_state(builder, 1.0)
    deprived, _ = _steady_state(builder, 0.0)

    assert intact["AR_DHT"] > 1.0
    assert deprived["AR_DHT"] < 0.1 * intact["AR_DHT"]


# ── the invariant that catches a silent stiffness failure ────────────────

def test_the_complex_cannot_exceed_what_the_androgen_supply_allows(builder):
    """AR_DHT is fed only by DHT and cleared at k_complex_deg.

    No steady state above DHT_input / k_complex_deg is reachable, whatever
    the integrator does. An explicit step coarser than DHT's ~1.7 s
    timescale, with negative values clamped, breaches this by ~4000x while
    still returning smooth finite numbers.
    """
    for dht_input in (0.0, 0.25, 1.0):
        state, system = _steady_state(builder, dht_input)
        bound = dht_input / system.parameters["k_complex_deg"]

        assert state["AR_DHT"] <= bound + 1e-3, (
            f"AR_DHT reached {state['AR_DHT']:.4g} nM with androgen supplied "
            f"at {dht_input}/h and cleared at "
            f"{system.parameters['k_complex_deg']}/h, so it cannot exceed "
            f"{bound:.4g} nM. Mass is being created -- suspect the "
            "integrator, not the model")


def test_no_species_goes_meaningfully_negative(builder):
    state, _ = _steady_state(builder, 1.0)

    for name, value in state.items():
        assert value > -1e-6, f"{name} is negative at steady state: {value}"


# ── the parameters come from the library, and in the right units ─────────

def test_the_binding_rates_reproduce_the_declared_kd(builder):
    """k_off/k_on must equal the Kd the AR entity records, exactly.

    Choosing both rates independently breaks detailed balance and relaxes
    the network to the wrong equilibrium.
    """
    system = builder.build_ar_signaling()
    k_on = system.parameters["k_bind"]
    k_off = system.parameters["k_unbind"]

    declared = None
    ar = builder.store.find_entity_by_name("AR", "gene")
    for inter in (ar.interacts_with or []):
        if inter.get("target") == "DHT":
            declared = float(inter["kd_nm"])
    assert declared is not None, "the AR entity records no DHT interaction"

    assert k_off / k_on == pytest.approx(declared, rel=1e-9), (
        f"the system's effective Kd is {k_off / k_on:.4g} nM but the entity "
        f"declares {declared:.4g} nM")


def test_the_mrna_degradation_rate_comes_from_the_entity_half_life(builder):
    """Pins the resolver, not a direct physics_params read.

    The previous expression was `0.693/x if x else 0.5`, which read only one
    of the several unit-suffixed half-life names the library writes, and
    whose own stated 3.0 default was unreachable.
    """
    from cognisom.genomics.reaction_builder import degradation_rate_per_hour

    ar = builder.store.find_entity_by_name("AR", "gene")
    expected = degradation_rate_per_hour(ar.physics_params or {})
    assert expected is not None, "the AR entity records no half-life"

    system = builder.build_ar_signaling()
    assert system.parameters["k_ar_m_deg"] == pytest.approx(expected)


def test_a_half_life_in_minutes_is_not_read_as_hours(builder):
    """The 60x error this resolver exists to prevent."""
    from cognisom.genomics.reaction_builder import (
        degradation_rate_per_hour, resolve_half_life_hours,
    )

    assert resolve_half_life_hours({"half_life_min": 30.0}) == pytest.approx(0.5)
    assert degradation_rate_per_hour({"half_life_min": 30.0}) == pytest.approx(
        0.6931471805599453 / 0.5)


def test_the_system_declares_itself_stiff(builder):
    """If this is False the module wrapper picks rk45 and the run is wrong."""
    ode = builder.build_ar_signaling().to_ode_system()

    assert ode.stiff is True
