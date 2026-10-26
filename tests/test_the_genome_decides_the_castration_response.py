"""The patient's genome, not a population average, decides the ADT response.

Two facts in a genome change how a tumour answers castration, and both
now reach the engine through the parameter bridge:

* Who dies. Westin 1995 (PMID 7778676): the tumours whose apoptotic index
  rose after castration were all p53/Bcl-2/c-myc-negative; every tumour
  in which it did not rise carried at least one. A TP53, BCL2 or MYC
  variant makes a tumour that slows but does not die.
* Which drug helps. Veldscholte 1990/1992 (PMIDs 2260966, 1540595): on
  the LNCaP receptor (T877A) hydroxyflutamide, nilutamide and
  cyproterone are agonists and bicalutamide is not. Joseph 2013 (PMID
  23779130): on F876L, enzalutamide and apalutamide are agonists. An
  agonist must not be applied as a blockade, and the run must say so.
"""

from __future__ import annotations

import contextlib
import io
import os

import numpy as np
import pytest

pytest.importorskip("scipy")

from cognisom.library.models import SimulationScenario  # noqa: E402
from cognisom.workflow import parameter_bridge as pb  # noqa: E402
from cognisom.workflow.scenario_builder import variant_labels_from_profile  # noqa: E402


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


def _scenario(store, genes=(), drugs=(), variants=()):
    return SimulationScenario(name="t", display_name="t", scenario_type="drug_response",
                              gene_ids=_ids(store, "gene", *genes), drug_ids=_ids(store, "drug", *drugs),
                              tags=["drug_response"] + [f"variant:{v}" for v in variants])


# ── agonism is a pure function of drug and variant ─────────────────


def test_the_agonist_table_follows_the_papers():
    assert pb.anti_androgen_is_agonist("Enzalutamide", ["AR:F876L"]) == "F876L"
    assert pb.anti_androgen_is_agonist("apalutamide", ["AR:F877L"]) == "F877L"
    assert pb.anti_androgen_is_agonist("flutamide", ["AR:T877A"]) == "T877A"
    assert pb.anti_androgen_is_agonist("nilutamide", ["AR:T878A"]) == "T878A"
    assert pb.anti_androgen_is_agonist("bicalutamide", ["AR:T877A"]) is None, "Veldscholte 1992: ICI 176334 is not"
    assert pb.anti_androgen_is_agonist("Enzalutamide", ["AR:T877A"]) is None
    assert pb.anti_androgen_is_agonist("Enzalutamide", ["TP53:R175H", "PTEN:loss"]) is None
    assert pb.anti_androgen_is_agonist("Enzalutamide", []) is None


def test_variant_labels_come_from_the_profile_s_protein_changes():
    class V:
        def __init__(self, gene, change):
            self.gene, self.protein_change = gene, change

    class P:
        variants = [V("AR", "p.T877A"), V("TP53", "p.R175H"), V("BRCA2", None), V("", "p.X")]

    assert variant_labels_from_profile(P()) == ["AR:T877A", "TP53:R175H"]


# ── the bridge applies them ───────────────────────────────────────


def test_a_tp53_mutant_tumour_slows_but_does_not_die(store):
    cfg = pb.ParameterBridge(store).resolve_scenario(_scenario(store, genes=("TP53",)), include_defaults=False)
    assert cfg["cellular"]["ar_dependent_fraction"] == 0.0


def test_a_tumour_without_the_blockers_is_the_responding_kind(store):
    cfg = pb.ParameterBridge(store).resolve_scenario(_scenario(store, genes=("PTEN",)), include_defaults=False)
    assert cfg["cellular"]["ar_dependent_fraction"] == 1.0


def test_no_genome_keeps_the_population_prior(store):
    cfg = pb.ParameterBridge(store).resolve_scenario(_scenario(store), include_defaults=False)
    assert "ar_dependent_fraction" not in cfg.get("cellular", {})


def test_enzalutamide_blocks_a_wild_type_receptor_and_feeds_an_f876l_one(store):
    bridge = pb.ParameterBridge(store)
    blocked = bridge.resolve_scenario(_scenario(store, drugs=("Enzalutamide",), variants=("AR:T877A",)),
                                      include_defaults=False)
    assert blocked["ode"]["k_bind"] == 10.0 and bridge.last_notes == []
    fed = bridge.resolve_scenario(_scenario(store, drugs=("Enzalutamide",), variants=("AR:F876L",)),
                                  include_defaults=False)
    assert "k_bind" not in fed.get("ode", {})
    assert "division_time_cancer" not in fed.get("cellular", {})
    assert len(bridge.last_notes) == 1 and "AGONIST" in bridge.last_notes[0] and "F876L" in bridge.last_notes[0]


# ── and the engine does what the bridge says ──────────────────────


MODS = {"cellular": True, "spatial": True, "vascular": True, "ar_signaling": True,
        "immune": False, "lymphatic": False, "molecular": False, "epigenetic": False,
        "circadian": False, "morphogen": False, "cell_mechanics": False}


def _run(scenario, hours, overrides=None, seed=11):
    os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    from cognisom.dashboard.engine_runner import EngineRunner
    np.random.seed(seed)
    runner = EngineRunner(tissue="ring", dt=0.05, duration=hours, scenario=scenario, modules_enabled=MODS,
                          overrides=overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    return runner


@pytest.fixture(scope="module")
def responding():
    return _run("Androgen Deprivation", 72.0, {"cellular": {"ar_dependent_fraction": 1.0}})


@pytest.fixture(scope="module")
def p53_mutant():
    return _run("Androgen Deprivation", 72.0, {"cellular": {"ar_dependent_fraction": 0.0}})


# The module-scoped 72 h fixtures these request cost ~80 s against the
# suite's 120 s per-test timeout, and fixture setup counts toward the
# requesting test's budget. Explicit budget rather than a shorter run:
# the 72 h separation is what the test is about.
@pytest.mark.timeout(400)
@pytest.mark.slow
def test_the_responding_tumour_loses_cells_and_the_p53_mutant_does_not(responding, p53_mutant):
    r = responding.engine.modules["cellular"].get_state()
    m = p53_mutant.engine.modules["cellular"].get_state()
    assert r["androgen_withdrawal_deaths"] > 0 and m["androgen_withdrawal_deaths"] == 0
    assert r["n_ar_dependent"] == r["n_cancer"] and m["n_ar_dependent"] == 0
    assert m["n_cancer"] >= r["n_cancer"]
    # both still slowed: 10 cells at 12 h doubling would be 640 untouched at 72 h
    assert r["n_cancer"] < 100 and m["n_cancer"] < 100


# The module-scoped 72 h fixtures these request cost ~80 s against the
# suite's 120 s per-test timeout, and fixture setup counts toward the
# requesting test's budget. Explicit budget rather than a shorter run:
# the 72 h separation is what the test is about.
@pytest.mark.timeout(400)
@pytest.mark.slow
def test_the_clinic_s_number_is_the_tumour_total(responding):
    ts = responding.get_time_series()
    per_cell = ts["avg_psa"][-1] / ts["avg_psa"][0]
    total = ts["psa_cancer_total"][-1] / ts["psa_cancer_total"][0]
    # Per cell, PSA falls to about 0.30 -- it tracks the androgen through
    # an induction term that is no longer saturated (K_psa sourced to the
    # in-cell apparent Kd, PMID 26657626). This assertion used to read
    # "> 0.85, PSA barely moves", which was the broken Km talking.
    assert 0.2 < per_cell < 0.5, f"per cell {per_cell:.3f}"
    # The tumour's total barely moves, and for a reason worth seeing: the
    # cells that are left each make a third as much, and there are three
    # times as many of them by 72 h. Two effects of the same treatment
    # pulling opposite ways is exactly why the clinic follows the total.
    assert 0.8 < total < 1.5, f"tumour total {total:.3f}"
    assert ts["psa_total"][-1] >= ts["psa_cancer_total"][-1], "benign cells make PSA too"
