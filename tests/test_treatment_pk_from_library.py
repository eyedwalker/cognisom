"""A drug's label half-life now governs how its effect washes out.

The treatment simulator had no exposure variable: the kill rate ramped up
after an illustrative onset and stayed on until the end of the window. Each
library-backed regimen's profile now carries a ``pk`` block, built through
the resolver's evidence gate, and the label half-life (FDA prescribing
information, cited by SPL set id) decides how fast the effect fades after a
last dose.

What is pinned here:

- the label's number reaches the running code, converted hours -> days once;
- with no course end -- the default, and every caller today -- every curve
  is the same floats it was before: exposure is exactly 1.0 throughout and
  consumes no random draws;
- after a last dose, carry-over orders by half-life: weeks for the
  antibodies, about three weeks for enzalutamide, almost none for olaparib;
- a placeholder half-life is refused, field by field, so one stamped field
  cannot veto a measured one;
- volume of distribution and bioavailability travel with their provenance
  but cannot move a curve -- they need a dose and an interval the library
  does not hold;
- the result says where its half-life came from.

Exposure while dosed is held at steady state on purpose; see drug_exposure
for why a continuous-infusion rise would misstate a bolus antibody regimen.
"""
from __future__ import annotations

import contextlib
import copy
import io
import logging

import pytest

from cognisom.genomics.parameter_resolver import (
    PK_FIELDS, SimulationParameterResolver,
)
from cognisom.genomics import treatment_simulator as ts
from cognisom.genomics.treatment_simulator import (
    TREATMENT_PROFILES, TreatmentSimulator, drug_exposure,
    elimination_half_life_days,
)
from cognisom.library.parameters import PARAMETER_SPECS, _sidecar

STOP = 90
WINDOW = 365


def _twin():
    """The honesty suite's twin (test_treatment_predictions_are_honest), field
    for field, so the curve goldens below are the same numbers there."""
    from cognisom.genomics.patient_profile import PatientProfile
    from cognisom.genomics.twin_config import DigitalTwinConfig
    return DigitalTwinConfig(
        patient=PatientProfile(patient_id="PT-TEST"),
        tumor_mutational_burden=5.0,
        microsatellite_instability="MSS",
        immune_profile_measured=True,
        immune_score="suppressed",
        t_cell_exhaustion_fraction=0.4,
        mean_exhaustion=0.5,
        treg_fraction=0.2,
        immune_cell_fraction=0.2,
        pd_l1_expression=0.5,
        mhc1_downregulation=0.1,
        vaccine_candidate_count=8,
        strong_binder_count=6,
        neoantigen_vaccine_candidate=True,
    )


@pytest.fixture(scope="module")
def store():
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        return EntityStore()


@pytest.fixture(scope="module")
def sim(store):
    simulator = TreatmentSimulator(store=store)
    if not simulator._entity_profiles:
        pytest.skip("no entity profiles in this library")
    return simulator


def _without_pk(profile):
    return {k: v for k, v in profile.items() if k != "pk"}


def _progression_day(sim, profile, last_dose_day=STOP):
    curve = sim._simulate_tumor_dynamics(0.6, profile, _twin(), WINDOW,
                                         last_dose_day=last_dose_day)
    return sim._first_progression_day(curve)


class _PlantedStore:
    def __init__(self, *entities):
        self._rows = {(e.name, e.entity_type.value): e for e in entities}

    def find_entity_by_name(self, name, entity_type=None):
        return self._rows.get((name, entity_type))


# ── the label's number reaches the running code ─────────────────────────

def test_enzalutamide_carries_its_label_half_life(sim):
    pk = sim._profiles["enzalutamide"]["pk"]
    assert pk["elimination_half_life_hours"] == pytest.approx(139.2)
    assert pk["_sources"]["elimination_half_life_hours"].startswith("measured:FDA-SPL:")
    assert elimination_half_life_days(sim._profiles["enzalutamide"]) == pytest.approx(5.8)


def test_the_seam_converts_hours_to_days_once():
    assert elimination_half_life_days({"pk": {"elimination_half_life_hours": 24.0}}) == 1.0
    assert elimination_half_life_days({"pk": {"elimination_half_life_hours": None}}) is None
    assert elimination_half_life_days({"pk": None}) is None
    assert elimination_half_life_days(TREATMENT_PROFILES["pembrolizumab"]) is None


def test_the_pk_fields_are_sourced_parameters_in_the_units_assumed():
    units = {f: PARAMETER_SPECS[f].unit for f in PK_FIELDS}
    assert units == {"elimination_half_life_hours": "h", "volume_of_distribution_L": "L",
                     "bioavailability": "dimensionless"}
    assert not any(PARAMETER_SPECS[f].model_construct for f in PK_FIELDS)


# ── the exposure function ───────────────────────────────────────────────

def test_exposure_halves_every_half_life_after_the_last_dose():
    assert drug_exposure(STOP, STOP, 2.0) == 1.0
    assert drug_exposure(STOP + 2, STOP, 2.0) == pytest.approx(0.5)
    assert drug_exposure(STOP + 4, STOP, 2.0) == pytest.approx(0.25)
    assert drug_exposure(STOP + 30, STOP, 2.0) == pytest.approx(2.0 ** -15)


def test_exposure_is_exactly_one_while_dosed_and_zero_after_without_a_half_life():
    assert all(drug_exposure(d, STOP, h) == 1.0
               for d in range(STOP + 1) for h in (None, 0.5, 22.0))
    assert drug_exposure(STOP + 1, STOP, None) == 0.0


# ── with no course end nothing moves ────────────────────────────────────

def test_with_no_course_end_the_half_life_changes_no_curve(sim):
    """Every library-backed regimen, at several effectiveness levels: the
    profile with its pk block and without it give the same floats."""
    for key, profile in sim._profiles.items():
        for eff in (0.0, 0.4, 1.0):
            with_pk = sim._simulate_tumor_dynamics(eff, profile, _twin(), WINDOW)
            without = sim._simulate_tumor_dynamics(eff, _without_pk(profile), _twin(), WINDOW)
            assert with_pk == without, key


@pytest.mark.parametrize("key,c30,c90,c180", [
    ("pembrolizumab", 1.0541, 0.5935, 0.2488),
    ("enzalutamide", 1.1643, 0.7309, 0.2876),
    ("olaparib", 1.1489, 0.7108, 0.2944),
    ("pembro_ipi_combo", 1.06, 0.5781, 0.2296),
])
def test_hardcoded_curves_are_the_values_from_before_exposure_existed(key, c30, c90, c180):
    curve = TreatmentSimulator()._simulate_tumor_dynamics(
        0.6, TREATMENT_PROFILES[key], _twin(), 180)
    assert (curve[30], curve[90], curve[180]) == (c30, c90, c180)


def test_a_last_dose_on_or_after_the_window_end_is_dosing_throughout(sim):
    twin = _twin()
    base = sim.simulate("enzalutamide", twin, 180)
    for last in (180, 400):
        late = sim.simulate("enzalutamide", twin, 180, last_dose_day=last)
        assert late.tumor_response_curve == base.tumor_response_curve
        assert late.last_dose_day is None


def test_a_negative_last_dose_day_is_refused(sim):
    with pytest.raises(ValueError, match="last_dose_day"):
        sim.simulate("enzalutamide", _twin(), 180, last_dose_day=-1)


# ── after a last dose ───────────────────────────────────────────────────

def test_carry_over_after_a_course_orders_by_half_life(sim):
    """Measured: 59, 21 and 2 days of progression-free time gained."""
    gained = {}
    for key in ("pembrolizumab", "enzalutamide", "olaparib"):
        profile = sim._profiles[key]
        gained[key] = (_progression_day(sim, profile)
                       - _progression_day(sim, _without_pk(profile)))
    assert gained["pembrolizumab"] > gained["enzalutamide"] > gained["olaparib"] >= 0
    assert gained["pembrolizumab"] > 30        # weeks of carry-over, not days


def test_a_drug_with_a_half_life_still_acts_after_its_last_dose(sim):
    profile = sim._profiles["enzalutamide"]
    twin = _twin()
    washout = sim._simulate_tumor_dynamics(0.6, profile, twin, WINDOW, last_dose_day=STOP)
    step = sim._simulate_tumor_dynamics(0.6, _without_pk(profile), twin, WINDOW, last_dose_day=STOP)
    assert washout[:STOP + 1] == step[:STOP + 1]          # identical while dosed
    assert washout[STOP + 30] < step[STOP + 30]


def test_volume_and_bioavailability_cannot_move_a_curve(sim):
    """They need a dose and an interval to mean anything. Until then a
    profile that carries them and one that does not are the same run."""
    profile = sim._profiles["enzalutamide"]
    stripped = copy.deepcopy(profile)
    stripped["pk"]["volume_of_distribution_L"] = None
    stripped["pk"]["bioavailability"] = None
    loaded = copy.deepcopy(profile)
    loaded["pk"]["volume_of_distribution_L"] = 6.0
    loaded["pk"]["bioavailability"] = 0.3
    runs = [sim._simulate_tumor_dynamics(0.6, p, _twin(), WINDOW, last_dose_day=STOP)
            for p in (stripped, loaded)]
    assert runs[0] == runs[1]


def test_a_stop_without_a_half_life_warns_once(caplog):
    ts._NO_HALF_LIFE_REPORTED.clear()
    sim = TreatmentSimulator()
    with caplog.at_level(logging.WARNING, logger="cognisom.genomics.treatment_simulator"):
        sim.simulate("pembrolizumab", _twin(), 180)              # dosed throughout: silent
        sim.simulate("pembrolizumab", _twin(), 180, last_dose_day=60)
        sim.simulate("pembrolizumab", _twin(), 180, last_dose_day=60)
    lines = [r for r in caplog.records if "no half-life in the library" in r.getMessage()]
    assert len(lines) == 1
    ts._NO_HALF_LIFE_REPORTED.clear()


# ── the evidence gate, field by field ───────────────────────────────────

def test_a_placeholder_half_life_is_refused(store):
    row = copy.deepcopy(store.find_entity_by_name("Enzalutamide", "drug"))
    _sidecar(row).pop("elimination_half_life_hours")
    row.physics_params["elimination_half_life_hours"] = 12.0              # a bare stamp
    profile = SimulationParameterResolver(_PlantedStore(row)).get_treatment_profile("Enzalutamide")
    assert profile["pk"]["elimination_half_life_hours"] is None
    assert profile["pk"]["_sources"]["elimination_half_life_hours"] == "placeholder: not evidence"
    assert elimination_half_life_days(profile) is None


def test_a_stamped_field_does_not_veto_a_measured_one(store):
    """The worst-of aggregate would have refused the half-life here."""
    row = copy.deepcopy(store.find_entity_by_name("Enzalutamide", "drug"))
    _sidecar(row).pop("volume_of_distribution_L", None)
    row.physics_params["volume_of_distribution_L"] = 50.0     # the enrichment stamp
    pk = SimulationParameterResolver(_PlantedStore(row)).get_treatment_profile("Enzalutamide")["pk"]
    assert pk["volume_of_distribution_L"] is None
    assert pk["elimination_half_life_hours"] == pytest.approx(139.2)


# ── the result says where its number came from ──────────────────────────

def test_the_result_names_its_half_life_and_how_exposure_was_modelled(sim):
    r = sim.simulate("enzalutamide", _twin(), WINDOW, last_dose_day=STOP)
    assert r.last_dose_day == STOP
    assert r.elimination_half_life_days == pytest.approx(5.8)
    assert r.pk_sources["elimination_half_life_hours"].startswith("measured:FDA-SPL:")
    assert "5.8 days" in r.exposure_model
    summary = r.summary()
    assert summary["exposure_model"] == r.exposure_model
    assert summary["pk_sources"]["elimination_half_life_hours"] == r.pk_sources["elimination_half_life_hours"]


def test_a_regimen_without_library_pk_says_so():
    sim = TreatmentSimulator()
    stopped = sim.simulate("pembrolizumab", _twin(), 180, last_dose_day=60)
    assert stopped.pk_sources == {}
    assert "No library half-life" in stopped.exposure_model
    throughout = sim.simulate("pembrolizumab", _twin(), 180)
    assert "Dosed throughout" in throughout.exposure_model


def test_compare_treatments_threads_the_course_end(sim):
    results = sim.compare_treatments(["enzalutamide", "olaparib"], _twin(),
                                     WINDOW, last_dose_day=STOP)
    assert {r.last_dose_day for r in results} == {STOP}


def test_the_model_card_no_longer_says_pharmacokinetics_are_absent():
    from cognisom.mad.model_cards import TREATMENT_SIMULATOR_CARD as card
    limitations = " ".join(card.known_limitations)
    assert "Does not model drug-drug interactions or pharmacokinetics" not in limitations
    assert "half-life" in limitations
