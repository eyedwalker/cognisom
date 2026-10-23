"""The anti-androgen class was mechanistically inert, and now is not.

Every anti-androgen in the library -- enzalutamide, apalutamide,
darolutamide, abiraterone -- mapped to one lever: a tenfold cut in
``k_bind``, the AR-DHT association rate. Measured on the system it
reaches, that lever does nothing. The AR-DHT complex is SUPPLY-bound,
not affinity-bound: AR sits at ~4.3e4 nM, so every DHT molecule that
arrives is captured whatever the affinity, and the complex settles at
``DHT_input / k_complex_deg``. Cutting k_bind 100-fold moves it 0.1%
and leaves PSA unchanged to five figures. The only reason the drug
appeared to work was a scripted ``division_time_cancer: 18.0`` sitting
beside it in the same mapping.

The real mechanism is downstream, and the library's own AR entry says
so: enzalutamide "inhibits AR signaling at three levels -- competitively
blocks androgen binding..., inhibits AR nuclear translocation..., and
prevents AR binding to DNA" -- which is what Tran et al. 2009 (Science
324:787, PMID 19359544) characterise for MDV3100. Two of the three act
after the complex has formed. So the drug now blocks the complex's
transcriptional OUTPUT (``ar_transcriptional_block``), which is both
the sourced mechanism and the one that moves anything.

The fraction blocked at therapeutic exposure is an ESTIMATE: the papers
characterise the mechanism, not a number.
"""

from __future__ import annotations

import contextlib
import io
import os

import numpy as np
import pytest

pytest.importorskip("scipy")

MODS = {"cellular": True, "spatial": True, "vascular": True, "ar_signaling": True,
        "immune": False, "lymphatic": False, "molecular": False, "epigenetic": False,
        "circadian": False, "morphogen": False, "cell_mechanics": False}


def _run(scenario, hours, overrides=None, seed=11):
    os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    from cognisom.dashboard.engine_runner import EngineRunner
    np.random.seed(seed)
    runner = EngineRunner(dt=0.05, duration=hours, scenario=scenario,
                          modules_enabled=MODS, overrides=overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    return runner


# ── the lever that does nothing ───────────────────────────────────


def test_cutting_the_binding_rate_does_not_reduce_ar_signalling():
    """Pin the shape of the defect so it cannot come back as a fix."""
    from scipy.integrate import solve_ivp
    from cognisom.genomics.reaction_builder import library_ar_system

    with contextlib.redirect_stdout(io.StringIO()):
        system = library_ar_system()
    rhs = system.get_rhs_func()

    def steady(**over):
        params = dict(system.parameters)
        params.setdefault("DHT_input", params["k_complex_deg"] * 13.7)
        params.update(over)
        sol = solve_ivp(lambda t, y: rhs(t, y, params), (0.0, 4000.0),
                        np.array(system.species_initial, dtype=float),
                        method="LSODA", rtol=1e-9, atol=1e-11)
        assert sol.success
        return dict(zip(system.species_names, sol.y[:, -1]))

    base = steady()
    blocked = steady(k_bind=system.parameters["k_bind"] / 100.0)
    assert blocked["AR_DHT"] == pytest.approx(base["AR_DHT"], rel=2e-3), (
        "a 100-fold affinity cut barely moves the complex: it is supply-bound")
    assert blocked["PSA"] == pytest.approx(base["PSA"], rel=1e-4)
    # and the reason: free DHT rises instead, by the same factor
    assert blocked["DHT"] > 50 * base["DHT"]


# ── the lever that works ──────────────────────────────────────────


@pytest.mark.slow
def test_blocking_transcription_lowers_signalling_without_touching_the_complex():
    treated = _run("Baseline", 24.0, {"ar_signaling": {"ar_transcriptional_block": 0.9}})
    untreated = _run("Baseline", 24.0)
    ar_t = treated.engine.modules["ar_signaling"]
    ar_u = untreated.engine.modules["ar_signaling"]
    # the complex is the same -- the drug is not competing for ligand
    assert (ar_t.get_state()["ar_dht_mean"]
            == pytest.approx(ar_u.get_state()["ar_dht_mean"], rel=0.01))
    # the signalling the cell acts on is a tenth of it
    sig_t = np.mean([ar_t.signalling_fraction(c) for c in ar_t.ids])
    sig_u = np.mean([ar_u.signalling_fraction(c) for c in ar_u.ids])
    assert sig_t == pytest.approx(0.1 * sig_u, rel=0.05)


@pytest.mark.slow
def test_the_drug_collapses_psa_where_castration_alone_cannot():
    """The clinically important difference. Castration leaves per-cell PSA
    at 0.908 of eugonadal, because the induction term is near-saturated
    at the androgen that remains. The drug acts on transcription itself,
    so it bypasses that saturation and takes PSA to a tenth."""
    castrate = _run("Androgen Deprivation", 24.0)
    drug = _run("Baseline", 24.0, {"ar_signaling": {"ar_transcriptional_block": 0.9}})
    psa_castrate = castrate.engine.modules["ar_signaling"].get_state()["psa_relative_to_eugonadal"]
    psa_drug = drug.engine.modules["ar_signaling"].get_state()["psa_relative_to_eugonadal"]
    assert psa_castrate > 0.85, f"castration alone: {psa_castrate:.3f}"
    assert psa_drug < 0.15, f"anti-androgen: {psa_drug:.3f}"


@pytest.mark.slow
@pytest.mark.timeout(400)   # three 48 h arms in one test
def test_the_drug_stops_the_tumour_and_the_two_treatments_compose():
    """48 h, immune off. Untreated doubles four times to 160; the drug
    holds the tumour at its seed of 10 (12 h / 0.1 signalling = a 120 h
    cycle, longer than the run); both together give the deepest
    suppression of signalling."""
    untreated = _run("Baseline", 48.0)
    drug = _run("Baseline", 48.0, {"ar_signaling": {"ar_transcriptional_block": 0.9}})
    both = _run("Androgen Deprivation", 48.0,
                {"ar_signaling": {"ar_transcriptional_block": 0.9}})

    def cancer(runner):
        return runner.engine.modules["cellular"].get_state()["n_cancer"]

    def signalling(runner):
        ar = runner.engine.modules["ar_signaling"]
        return float(np.mean([ar.signalling_fraction(c) for c in ar.ids]))

    assert cancer(untreated) >= 160 and cancer(drug) <= 12
    assert signalling(both) < signalling(drug) < signalling(untreated)
    assert signalling(both) < 0.05


@pytest.mark.slow
def test_an_untreated_run_is_untouched_by_the_new_parameter():
    assert _run("Baseline", 24.0).engine.modules["ar_signaling"].transcriptional_block == 0.0


# ── the mapping ───────────────────────────────────────────────────


def test_the_drug_class_now_names_the_mechanism_that_works():
    from cognisom.workflow.parameter_bridge import DRUG_CLASS_PARAM_MAP
    mapping = DRUG_CLASS_PARAM_MAP["anti_androgen"]
    assert mapping["ar_signaling"]["ar_transcriptional_block"] == 0.9
    # the inert lever is still mapped for the legacy ODE path, deliberately
    assert mapping["ode"]["k_bind"] == 10.0


def test_the_scripted_slowdown_does_not_double_count_the_mechanism():
    """With the AR module on, division time is computed from signalling:
    12 h / 0.1 = 120 h, far weaker than the scripted 18 h would be, so
    the scripted value is the one that does not bind. It stays for a run
    with no AR module, where it is the only slowdown there is."""
    from cognisom.workflow.parameter_bridge import DRUG_CLASS_PARAM_MAP
    scripted = DRUG_CLASS_PARAM_MAP["anti_androgen"]["cellular"]["division_time_cancer"]
    block = DRUG_CLASS_PARAM_MAP["anti_androgen"]["ar_signaling"]["ar_transcriptional_block"]
    mechanism = 12.0 / (1.0 - block)
    assert mechanism > scripted, (mechanism, scripted)
