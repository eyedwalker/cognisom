"""On an F877L receptor enzalutamide drives the tumour, and now it does.

The bridge already knew that enzalutamide is an agonist on AR F876L/F877L
-- Joseph 2013 (PMID 23779130) -- and it used that knowledge to SKIP the
blockade. Skipping is safe and wrong: the drug does not do nothing to
this patient, it activates the receptor it was prescribed to block, so a
skipped drug simulates a treated man as an untreated one and understates
the harm. Rising PSA on treatment is precisely the clinical picture of
enzalutamide resistance, and a model that reports an untreated tumour
cannot show it.

HOW MUCH it activates is MEASURED: Balbas et al., eLife 2013;2:e00499
(PMID 23580326, open access, read in full). One AR-dependent EGFP
reporter in LNCaP-Pb.PSE.EGFP, geometric-mean fluorescence, with the two
vehicle baselines matching each other within 1% so the rows compare
directly:

    AR          vehicle   enzalutamide   DHT
    wild type     348         66.4       1554
    F876L         345       1,051        1699

So 66.4/348 = 0.191 of baseline on the wild-type receptor and
1051/345 = 3.05 on F876L. It is PARTIAL agonism -- 1051/1699 is 62% of
what DHT itself reaches in the same assay -- so this must not be modelled
as "enzalutamide becomes androgen". The same paper's CV1 luciferase assay
gives ~50-fold induction; the reporter figures are used because they
carry a matched untreated baseline, which is what a ratio to untreated
has to be measured against.

AND GROWTH DOES NOT ACCELERATE. This is the part a plausible "fix" would
get wrong. Balbas's F876L xenografts on enzalutamide "grew with kinetics
roughly equivalent to either vehicle treatment arm" -- rescue to
baseline, not acceleration. The engine reproduces that because its AR
coupling is deliberately one-sided: ``cellular_module`` only divides the
division time when signalling falls BELOW ``AR_INTACT_ABOVE``, so
super-physiological AR drive leaves the cycle alone. That asymmetry
looks like an oversight and is the measured biology; the test below
exists so nobody symmetrises it.

The reporter and the growth rate are different quantities. PSA triples
while the tumour grows exactly as the untreated arm does, and that
separation is the finding.
"""

from __future__ import annotations

import contextlib
import io
import os

import numpy as np
import pytest

pytest.importorskip("scipy")

from cognisom.library.models import SimulationScenario  # noqa: E402
from cognisom.modules.cellular_module import AR_INTACT_ABOVE  # noqa: E402
from cognisom.workflow import parameter_bridge as pb  # noqa: E402


# ── the measured number, pinned beside the claim it came from ─────


def test_the_activation_is_the_balbas_ratio_and_is_partial():
    """3.05 is 1051/345, and 1051 is 62% of the DHT maximum."""
    f876l_enza, f876l_vehicle, f876l_dht = 1051.0, 345.0, 1699.0
    wt_enza, wt_vehicle = 66.4, 348.0

    assert pb.AR_AGONIST_TRANSCRIPTIONAL_ACTIVITY == pytest.approx(
        f876l_enza / f876l_vehicle, abs=0.01), (
        "the stored activity must be the ratio it is sourced to, "
        "re-measure from Balbas 2013 Figure 2, do not re-pin")
    # Partial, not full: it must sit strictly between "no effect" and
    # "as good as androgen".
    full_agonist = f876l_dht / f876l_vehicle
    assert 1.0 < pb.AR_AGONIST_TRANSCRIPTIONAL_ACTIVITY < full_agonist
    assert f876l_enza / f876l_dht == pytest.approx(0.62, abs=0.01)
    # The vehicle arms are what make the two rows comparable at all.
    assert abs(f876l_vehicle / wt_vehicle - 1.0) < 0.01
    # And the direction reverses between the two receptors: the same drug
    # suppresses one fivefold and activates the other threefold.
    assert wt_enza / wt_vehicle < 1.0 < f876l_enza / f876l_vehicle
    assert "23580326" in pb.AR_AGONIST_ACTIVITY_SOURCE


def test_the_source_string_names_a_measurement_not_a_mechanism():
    source = pb.AR_AGONIST_ACTIVITY_SOURCE
    assert "Balbas" in source and "reporter" in source
    assert "1051" in source and "345" in source, (
        "the source must carry the numbers the ratio is computed from, "
        "so a reader can check it without the paper")


# ── the module: the default is bit-identical ──────────────────────


def _module(**config):
    from cognisom.modules.ar_signaling_module import ARSignalingModule
    with contextlib.redirect_stdout(io.StringIO()):
        mod = ARSignalingModule(config)
        mod.initialize()
    return mod


def test_an_untreated_module_has_an_activity_of_exactly_one():
    assert _module().transcriptional_activity == 1.0


def test_the_activity_defaults_to_one_minus_the_block():
    """Every run that predates the agonist path is bit-identical.

    The factor is new, so it has to reduce to the old expression
    exactly -- not approximately -- for any block, or the whole existing
    suite of castration and anti-androgen pins would have moved.
    """
    for block in (0.0, 0.1, 0.5, 0.9, 1.0):
        mod = _module(ar_transcriptional_block=block)
        assert mod.transcriptional_activity == 1.0 - block, block


def test_an_explicit_activity_wins_over_the_block():
    mod = _module(ar_transcriptional_block=0.9,
                  ar_transcriptional_activity=3.05)
    assert mod.transcriptional_activity == 3.05, (
        "the agonist case is not a negative blockade; the factor is set "
        "directly and must not be re-derived from the block")


# ── the module: the ceiling rises only for an agonist ─────────────


def _eugonadal_cell(mod):
    """One cell sitting exactly at the untreated eugonadal steady state.

    ``add_cell`` with no state seeds precisely ``eugonadal_state``, so the
    complex equals its own reference and the fraction is the activity
    factor alone -- no integration, no drift, nothing else in the number.
    """
    mod.add_cell(1)
    return 1


def test_a_suppressed_cell_still_cannot_exceed_one():
    mod = _module(ar_transcriptional_block=0.9)
    cell = _eugonadal_cell(mod)
    frac = mod.signalling_fraction(cell)
    assert frac == pytest.approx(0.1, rel=1e-3), frac
    assert 0.0 <= frac <= 1.0


def test_only_an_agonist_reads_above_one():
    """The ceiling was a hard 1.0, which would have discarded the effect."""
    untreated = _module()
    assert untreated.signalling_fraction(_eugonadal_cell(untreated)) <= 1.0

    agonist = _module(ar_transcriptional_activity=3.05)
    frac = agonist.signalling_fraction(_eugonadal_cell(agonist))
    assert frac == pytest.approx(3.05, rel=1e-3), frac
    assert frac > 1.0, (
        "a hard 1.0 ceiling reports a treated patient as untreated and "
        "silently discards the entire agonist effect")


def test_the_fraction_is_still_clamped_at_the_activity():
    """Above 1 is allowed; unbounded is not."""
    mod = _module(ar_transcriptional_activity=3.05)
    _eugonadal_cell(mod)
    # A second cell whose complex is a hundred times its reference: the
    # fraction must stop at the activity factor, not run away with it.
    hundredfold = np.array(mod.eugonadal_state, dtype=float)
    hundredfold[mod._i["AR_DHT"]] *= 100.0
    mod.add_cell(2, state=hundredfold)
    assert mod.signalling_fraction(2) == pytest.approx(3.05, rel=1e-6)
    # and the floor is untouched
    floor = np.array(mod.eugonadal_state, dtype=float)
    floor[mod._i["AR_DHT"]] = -1.0
    mod.add_cell(3, state=floor)
    assert mod.signalling_fraction(3) == 0.0


# ── the bridge applies activation, not a blockade ─────────────────


@pytest.fixture(scope="module")
def store():
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        return EntityStore()


def _ids(store, kind, *names):
    out = []
    for name in names:
        ent = store.find_entity_by_name(name, kind)
        assert ent is not None, f"{kind} {name!r} not in the library"
        out.append(ent.entity_id)
    return out


def _scenario(store, drugs=(), variants=()):
    return SimulationScenario(
        name="t", display_name="t", scenario_type="drug_response",
        drug_ids=_ids(store, "drug", *drugs),
        tags=["drug_response"] + [f"variant:{v}" for v in variants])


def test_the_bridge_applies_the_measured_activation(store):
    bridge = pb.ParameterBridge(store)
    cfg = bridge.resolve_scenario(
        _scenario(store, drugs=("Enzalutamide",), variants=("AR:F877L",)),
        include_defaults=False)
    assert (cfg["ar_signaling"]["ar_transcriptional_activity"]
            == pb.AR_AGONIST_TRANSCRIPTIONAL_ACTIVITY)
    # and NOT as a blockade, which would cancel it
    assert "ar_transcriptional_block" not in cfg["ar_signaling"]
    assert "k_bind" not in cfg.get("ode", {})
    assert "division_time_cancer" not in cfg.get("cellular", {})


def test_the_run_says_what_it_did_and_why(store):
    bridge = pb.ParameterBridge(store)
    bridge.resolve_scenario(
        _scenario(store, drugs=("Enzalutamide",), variants=("AR:F877L",)),
        include_defaults=False)
    assert len(bridge.last_notes) == 1
    note = bridge.last_notes[0]
    for expected in ("AGONIST", "F877L", "3.05", "NOT as a blockade",
                     "23580326"):
        assert expected in note, (expected, note)


def test_a_wild_type_receptor_is_still_blocked(store):
    """The reversal is specific to the variant, not to the drug."""
    bridge = pb.ParameterBridge(store)
    cfg = bridge.resolve_scenario(
        _scenario(store, drugs=("Enzalutamide",), variants=("TP53:R175H",)),
        include_defaults=False)
    assert "ar_transcriptional_activity" not in cfg.get("ar_signaling", {})
    assert cfg["ar_signaling"]["ar_transcriptional_block"] == 0.9
    assert bridge.last_notes == []


def test_darolutamide_still_blocks_the_f876l_receptor(store):
    """Agonism is drug-specific, and this is the actionable case.

    Moilanen et al. 2015, Sci Rep 5:12007 (PMID 26137992, open access,
    read in full): in U2-OS transactivation assays "the F876L
    substitution in AR switched enzalutamide and ARN-509 from
    antagonists to agonists", while "only ODM-201 and its main
    metabolite ORM-15341 functioned as full antagonists for all tested
    mutant ARs". ODM-201 is darolutamide.

    So the reversal is a property of the bisaryl-thiohydantoins, not of
    AR blockade -- and the model that turned every anti-androgen into an
    agonist on this receptor would be easier to write, would be wrong,
    and would destroy the one clinically useful thing this path can say:
    that there is a drug this patient can still be given.
    """
    bridge = pb.ParameterBridge(store)
    cfg = bridge.resolve_scenario(
        _scenario(store, drugs=("Darolutamide",), variants=("AR:F876L",)),
        include_defaults=False)
    assert "ar_transcriptional_activity" not in cfg.get("ar_signaling", {})
    assert cfg["ar_signaling"]["ar_transcriptional_block"] == 0.9, (
        "darolutamide is a FULL antagonist on F876L and must still be "
        "applied as a blockade")
    assert bridge.last_notes == []


def test_apalutamide_reverses_on_f876l_the_same_way(store):
    """Same mutation, same switch -- Joseph 2013, confirmed by Moilanen."""
    bridge = pb.ParameterBridge(store)
    cfg = bridge.resolve_scenario(
        _scenario(store, drugs=("Apalutamide",), variants=("AR:F876L",)),
        include_defaults=False)
    assert (cfg["ar_signaling"]["ar_transcriptional_activity"]
            == pb.AR_AGONIST_TRANSCRIPTIONAL_ACTIVITY)
    assert "ar_transcriptional_block" not in cfg["ar_signaling"]


# ── and the engine does what the papers measured ──────────────────


MODS = {"cellular": True, "spatial": True, "vascular": True,
        "ar_signaling": True, "immune": False, "lymphatic": False,
        "molecular": False, "epigenetic": False, "circadian": False,
        "morphogen": False, "cell_mechanics": False}

#: 48 h, ring, seed 11, immune off, measured 2026-11-09. Re-measure, do
#: not re-pin: these are the numbers the claims above are made of.
SHIPPED_48H = {
    "untreated":  {"signalling": 0.9995, "psa": 0.9996, "cancer": 160},
    "wild_type":  {"signalling": 0.1000, "psa": 0.1001, "cancer": 10},
    "f877l":      {"signalling": 3.0485, "psa": 3.0487, "cancer": 160},
}


def _run(scenario, hours, overrides=None, seed=11):
    os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    from cognisom.dashboard.engine_runner import EngineRunner
    np.random.seed(seed)
    runner = EngineRunner(tissue="ring", dt=0.05, duration=hours,
                          scenario=scenario, modules_enabled=MODS,
                          overrides=overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    return runner


def _arms(hours=48.0, extra=None):
    base = dict(extra or {})

    def over(ar):
        cfg = {k: dict(v) for k, v in base.items()}
        cfg["ar_signaling"] = ar
        return cfg

    return (
        _run("Baseline", hours, base or None),
        _run("Baseline", hours, over({"ar_transcriptional_block": 0.9})),
        _run("Baseline", hours,
             over({"ar_transcriptional_activity":
                   pb.AR_AGONIST_TRANSCRIPTIONAL_ACTIVITY})),
    )


def _signalling(runner):
    ar = runner.engine.modules["ar_signaling"]
    return float(np.mean([ar.signalling_fraction(c) for c in ar.ids]))


@pytest.fixture(scope="module")
def three_arms():
    return _arms()


@pytest.mark.slow
@pytest.mark.timeout(600)   # three 48 h arms
def test_the_same_drug_suppresses_one_receptor_and_drives_the_other(
        three_arms):
    untreated, wild_type, f877l = three_arms
    s_u, s_w, s_f = (_signalling(r) for r in three_arms)
    assert s_w < s_u < s_f, (s_w, s_u, s_f)
    assert s_u == pytest.approx(SHIPPED_48H["untreated"]["signalling"],
                                rel=0.02)
    assert s_w == pytest.approx(SHIPPED_48H["wild_type"]["signalling"],
                                rel=0.02)
    assert s_f == pytest.approx(SHIPPED_48H["f877l"]["signalling"], rel=0.02)
    # The agonist arm is the only one the old 1.0 ceiling would have hidden.
    assert s_f > 1.0 and s_u <= 1.0


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_psa_rises_on_treatment_which_is_the_clinical_picture(three_arms):
    """The reporter is what the clinic follows, and it goes the wrong way."""
    def psa(runner):
        return runner.engine.modules["ar_signaling"].get_state()[
            "psa_relative_to_eugonadal"]

    psa_u, psa_w, psa_f = (psa(r) for r in three_arms)
    assert psa_w < psa_u < psa_f, (psa_w, psa_u, psa_f)
    assert psa_f / psa_u == pytest.approx(
        pb.AR_AGONIST_TRANSCRIPTIONAL_ACTIVITY, rel=0.02), (
        "PSA relative to untreated must equal the activation factor; the "
        "two are computed by different routes and have to agree")
    assert psa_u == pytest.approx(SHIPPED_48H["untreated"]["psa"], rel=0.02)
    assert psa_w == pytest.approx(SHIPPED_48H["wild_type"]["psa"], rel=0.02)
    assert psa_f == pytest.approx(SHIPPED_48H["f877l"]["psa"], rel=0.02)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_growth_is_rescued_to_the_untreated_arm_and_does_not_exceed_it(
        three_arms):
    """Balbas: "roughly equivalent to either vehicle treatment arm".

    DO NOT make the AR/division coupling symmetric. It is one-sided on
    purpose: ``cellular_module`` scales the division time only when
    signalling falls below AR_INTACT_ABOVE, so 3.05 leaves the cycle
    untouched. Symmetrising it would make the tumour grow three times
    faster than vehicle and would contradict the paper this activation
    factor is sourced to. The reporter triples; the growth rate does
    not. They are different quantities.
    """
    untreated, wild_type, f877l = three_arms

    def cancer(runner):
        return runner.engine.modules["cellular"].get_state()["n_cancer"]

    n_u, n_w, n_f = (cancer(r) for r in three_arms)
    # the drug works on the wild-type receptor
    assert n_w < n_u / 4, (n_w, n_u)
    assert n_w <= SHIPPED_48H["wild_type"]["cancer"]
    # and is wholly undone on this one -- equal, not merely close. Both
    # arms are above the AR_INTACT_ABOVE dead band, so neither takes the
    # slowdown branch and neither draws from the shared RNG stream; the
    # equality is the gate's doing, not a coincidence.
    assert _signalling(untreated) >= AR_INTACT_ABOVE
    assert _signalling(f877l) >= AR_INTACT_ABOVE
    assert n_f == n_u, (n_f, n_u)
    assert n_f == SHIPPED_48H["f877l"]["cancer"]
    # NOT acceleration
    assert n_f <= n_u, (
        "Balbas 2013 measured growth equivalent to vehicle, not faster; "
        "if this fails someone symmetrised the AR/division coupling")


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_an_agonist_driven_cell_does_not_die_of_androgen_withdrawal():
    """Every cancer cell AR-dependent, so the hazard can fire at all.

    Measured 48 h, ring, seed 11: the wild-type arm loses cells to
    withdrawal and the agonist arm loses none, because the hazard is
    gated on the same dead band the slowdown is.
    """
    dependent = {"cellular": {"ar_dependent_fraction": 1.0}}
    untreated, wild_type, f877l = _arms(extra=dependent)

    def deaths(runner):
        return runner.engine.modules["cellular"].get_state()[
            "androgen_withdrawal_deaths"]

    assert deaths(wild_type) > 0, (
        "with every cell AR-dependent the suppressed arm must lose some")
    assert deaths(untreated) == 0
    assert deaths(f877l) == 0, (
        "a cell driven above eugonadal AR cannot be dying of androgen "
        "withdrawal")
    cell_f = f877l.engine.modules["cellular"].get_state()
    assert cell_f["n_ar_dependent"] == cell_f["n_cancer"] > 0
