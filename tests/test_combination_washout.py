"""A stopped combination washes out component by component.

The four combination regimens have no Drug entity, so after a last dose they
used to stop acting at once -- a pembrolizumab + ipilimumab course looked
worse after stopping than pembrolizumab alone, purely for want of data. Each
combination now names the single-agent regimens it is made of; after the
last dose each component's share of the effect decays at that component's
own label half-life, and the shares add.

The shares are the components' single-agent effectiveness for the patient,
normalised. The effectiveness model adds a term per mechanism, so this is its
own split of the combined effect, and it is patient-specific. It is a
structural assumption and introduces no constant. A vaccine's lasting effect
is T-cell memory, not drug exposure; it is not modelled, so a vaccine's
share stops at the last dose, a lower bound that the result states.
"""
from __future__ import annotations

import contextlib
import io

import pytest

from cognisom.genomics.treatment_simulator import (
    TREATMENT_PROFILES, TreatmentSimulator, regimen_exposure, treatment_mechanisms,
)

STOP = 90
WINDOW = 365
COMBINATIONS = ("pembro_ipi_combo", "olaparib_pembro_combo", "neoantigen_vaccine_pembro")


def _twin(**kw):
    from cognisom.genomics.patient_profile import PatientProfile
    from cognisom.genomics.twin_config import DigitalTwinConfig
    base = dict(
        patient=PatientProfile(patient_id="PT-TEST"), tumor_mutational_burden=5.0,
        microsatellite_instability="MSS", immune_profile_measured=True,
        immune_score="suppressed", t_cell_exhaustion_fraction=0.4, mean_exhaustion=0.5,
        treg_fraction=0.2, immune_cell_fraction=0.2, pd_l1_expression=0.5,
        mhc1_downregulation=0.1, vaccine_candidate_count=8, strong_binder_count=6,
        neoantigen_vaccine_candidate=True,
    )
    base.update(kw)
    return DigitalTwinConfig(**base)


@pytest.fixture(scope="module")
def sim():
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        simulator = TreatmentSimulator(store=EntityStore())
    if not simulator._entity_profiles:
        pytest.skip("no entity profiles in this library")
    return simulator


def _without_components(profile):
    return {k: v for k, v in profile.items() if k != "components"}


def _progression(sim, profile, twin=None):
    curve = sim._simulate_tumor_dynamics(0.6, profile, twin or _twin(), WINDOW,
                                         last_dose_day=STOP)
    return sim._first_progression_day(curve)


# ── the links ───────────────────────────────────────────────────────────

def test_every_combination_names_the_regimens_it_is_made_of():
    multi = {k for k, p in TREATMENT_PROFILES.items() if len(treatment_mechanisms(p)) > 1}
    assert multi == set(COMBINATIONS)
    for key in COMBINATIONS:
        components = TREATMENT_PROFILES[key]["components"]
        assert all(c in TREATMENT_PROFILES for c in components), key
        union = frozenset().union(*(treatment_mechanisms(TREATMENT_PROFILES[c])
                                    for c in components))
        assert union == treatment_mechanisms(TREATMENT_PROFILES[key]), key


def test_an_unknown_component_is_refused():
    profile = dict(TREATMENT_PROFILES["pembro_ipi_combo"], components=("no_such_drug",))
    with pytest.raises(ValueError, match="no_such_drug"):
        TreatmentSimulator()._exposure_parts(profile, _twin())


# ── the exposure function ───────────────────────────────────────────────

def test_exposure_is_one_while_dosed_whatever_the_shares():
    parts = [(0.3, 2.0), (0.7, None)]
    assert all(regimen_exposure(d, STOP, parts) == 1.0 for d in range(STOP + 1))


def test_after_the_last_dose_each_share_washes_out_on_its_own():
    parts = [(0.5, 2.0), (0.5, None)]
    assert regimen_exposure(STOP + 2, STOP, parts) == pytest.approx(0.25)
    assert regimen_exposure(STOP + 4, STOP, parts) == pytest.approx(0.125)


# ── shares ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", COMBINATIONS)
def test_shares_sum_to_one(sim, key):
    parts = sim._exposure_parts(sim._profiles[key], _twin())
    assert len(parts) == 2
    assert sum(p["share"] for p in parts) == pytest.approx(1.0)


def test_a_component_that_works_for_this_patient_gets_the_larger_share(sim):
    """Olaparib's single-agent term is +0.35 with a DNA-repair defect and
    -0.3 without one, so its share of the combination follows the patient."""
    def olaparib_share(twin):
        parts = sim._exposure_parts(sim._profiles["olaparib_pembro_combo"], twin)
        return next(p["share"] for p in parts if p["name"].startswith("Olaparib"))
    assert olaparib_share(_twin(has_dna_repair_defect=True)) > \
        2 * olaparib_share(_twin(has_dna_repair_defect=False))


# ── what changes, and what does not ─────────────────────────────────────

@pytest.mark.parametrize("key", COMBINATIONS)
def test_dosed_throughout_a_combination_is_unchanged(sim, key):
    profile = sim._profiles[key]
    linked = sim._simulate_tumor_dynamics(0.6, profile, _twin(), WINDOW)
    unlinked = sim._simulate_tumor_dynamics(0.6, _without_components(profile), _twin(), WINDOW)
    assert linked == unlinked


@pytest.mark.parametrize("key", ("pembro_ipi_combo", "olaparib_pembro_combo"))
def test_a_stopped_antibody_combination_keeps_acting_for_weeks(sim, key):
    """Measured: progression 208 -> 261 and 198 -> 251 days."""
    profile = sim._profiles[key]
    gained = _progression(sim, profile) - _progression(sim, _without_components(profile))
    assert gained > 30


def test_without_the_library_a_stopped_combination_still_stops_at_once():
    hard = TreatmentSimulator()
    for key in COMBINATIONS:
        profile = hard._profiles[key]
        assert _progression(hard, profile) == _progression(hard, _without_components(profile)), key
        result = hard.simulate(key, _twin(), WINDOW, last_dose_day=STOP)
        assert all(c["half_life_days"] is None for c in result.exposure_components)


def test_a_vaccine_share_stops_at_the_last_dose_and_says_so(sim):
    result = sim.simulate("neoantigen_vaccine_pembro", _twin(), WINDOW, last_dose_day=STOP)
    vaccine = next(c for c in result.exposure_components if "Vaccine" in c["name"])
    assert vaccine["half_life_days"] is None
    assert "T-cell memory" in result.exposure_model
    assert "lower bound" in result.exposure_model
    # ... while its pembrolizumab share still washes out at the label value.
    pembro = next(c for c in result.exposure_components if c["name"].startswith("Pembrolizumab"))
    assert pembro["half_life_days"] == pytest.approx(22.0)


# ── the result says what it used ────────────────────────────────────────

def test_the_result_names_each_component_and_its_source(sim):
    result = sim.simulate("pembro_ipi_combo", _twin(), WINDOW, last_dose_day=STOP)
    halves = {c["name"].split()[0]: c["half_life_days"] for c in result.exposure_components}
    assert halves == {"Pembrolizumab": pytest.approx(22.0), "Ipilimumab": pytest.approx(15.4)}
    for drug in ("Pembrolizumab", "Ipilimumab"):
        # elimination_ since the half-life split: one name was carrying a
        # drug's clearance, a transcript's decay, a cytokine's time in
        # circulation and a protein's turnover.
        assert result.pk_sources[
            f"{drug}: elimination_half_life_hours"].startswith("measured:FDA-SPL:")
    assert "share" in result.exposure_model
    assert result.elimination_half_life_days is None      # no single value for a combination
