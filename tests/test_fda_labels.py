"""Pharmacokinetics read out of regulatory prose, and the ways prose lies.

An FDA label is about as citable a source as exists, and it states a drug's
half-life, bioavailability and volume of distribution in sentences rather
than fields. Every one of the defects below was found on a real label during
the first pass and would have stored a confident, cited, wrong number:

- "4,864 L" read as 864 L, and "19,669 ± 13,358 L" as 13,358 L
- "12 ± 5 hours" read as 5 hours, the standard deviation
- "26 (39%) hours" and "90 (±58) hours" read as nothing at all
- "between 141 hours and 882 hours" read as 141, the low end
- "peak concentrations within 0.5 to 1 hour; the half-life is 3 hours" read
  as 0.5 hours, the Tmax
- pembrolizumab's label stating that hyaluronan has a half-life of half a day
- an antibody's 72% "absolute bioavailability", which only exists because a
  subcutaneous product does, written against an intravenous dose

Everything here runs offline against fixture labels.
"""
from __future__ import annotations

import pytest

from cognisom.library.fda_labels import (
    PKCandidate, accept, choose, extract, forget,
)
from cognisom.library.models import Drug
from cognisom.library.parameters import Provenance, get_parameter


def _label(text, brand="BRANDX", generic="drugname"):
    return {"set_id": "0000-test", "effective_time": "20240101",
            "openfda": {"brand_name": [brand], "generic_name": [generic]},
            "pharmacokinetics": [text]}


def _values(text, parameter, **kw):
    return sorted(c.value for c in extract(_label(text, **kw)) if c.parameter == parameter)


# ── numbers as labels print them ─────────────────────────────────────────

def test_a_thousands_separator_is_not_a_decimal_point():
    assert _values("The volume of distribution (Vss) was 4,864 L.", "volume_of_distribution_L") == [4864.0]


def test_the_mean_is_taken_not_the_standard_deviation():
    assert _values("The mean terminal half-life of drugname is 12 ± 5 hours.", "half_life_hours") == [12.0]
    assert _values("The volume of distribution is 19,669 ± 13,358 L.", "volume_of_distribution_L") == [19669.0]


def test_a_parenthetical_between_number_and_unit_is_skipped():
    assert _values("The mean terminal elimination half-life of drugname is 26 (39%) hours.", "half_life_hours") == [26.0]
    assert _values("The mean terminal plasma half-life (±SD) is 90 (±58) hours.", "half_life_hours") == [90.0]


def test_a_range_yields_its_midpoint():
    assert _values("The half-life of drugname ranged between 141 hours and 882 hours.", "half_life_hours") == [511.5]
    assert _values("The median terminal half-life of drugname is 7.7 to 9.5 days.", "half_life_hours") == [pytest.approx(8.6 * 24)]


def test_a_hyphenated_unit_is_read():
    assert _values("The 2-hour apparent half-life of drugname after dosing.", "half_life_hours") == [2.0]


def test_units_are_converted_to_hours():
    assert _values("The terminal half-life is 22 days.", "half_life_hours") == [528.0]
    assert _values("The elimination half-life is approximately 2 weeks.", "half_life_hours") == [336.0]
    assert _values("The elimination half-life was 85 minutes.", "half_life_hours") == [pytest.approx(85 / 60)]


# ── the wrong sentence ───────────────────────────────────────────────────

def test_tmax_in_the_same_sentence_is_not_a_half_life():
    text = ("Following oral administration, peak plasma concentrations were "
            "reached within 0.5 to 1 hour, and the terminal half-life of "
            "drugname is 3 hours.")
    assert _values(text, "half_life_hours") == [3.0]


def test_another_molecules_half_life_is_not_the_drugs():
    text = ("Unlike the stable components of the interstitial matrix, hyaluronan "
            "has a half-life of approximately 0.5 days. The terminal half-life "
            "of pembrolizumab is 22 days.")
    assert _values(text, "half_life_hours", generic="pembrolizumab") == [528.0]


def test_a_biosimilar_suffix_does_not_hide_the_drugs_own_sentences():
    text = "The median terminal half-life of infliximab, an IgG1 antibody, is 7.7 to 9.5 days."
    assert _values(text, "half_life_hours", brand="Renflexis", generic="infliximab-abda") == [pytest.approx(8.6 * 24)]


def test_a_per_kilogram_volume_is_not_a_volume():
    assert _values("The volume of distribution was 0.7 L/kg.", "volume_of_distribution_L") == []


# ── choosing among several ───────────────────────────────────────────────

def _cand(value, sentence, parameter="half_life_hours", generic="drugname"):
    return PKCandidate(parameter=parameter, value=value, unit="h", sentence=sentence,
                       raw_value=value, raw_unit="h", set_id="0000", generic=generic)


def test_a_special_population_figure_is_set_aside():
    pick, why = choose("half_life_hours", [
        _cand(12.0, "the mean terminal half-life of drugname is 12 hours"),
        _cand(18.0, "the half-life is prolonged to 18 hours in hepatic impairment"),
    ])
    assert pick is not None and pick.value == 12.0


def test_genuinely_different_values_are_surfaced_not_averaged():
    pick, why = choose("half_life_hours", [
        _cand(432.0, "terminal half-life in NHL patients was 18 days"),
        _cand(768.0, "terminal half-life in RA patients was 32 days"),
    ])
    assert pick is None
    assert "2 distinct" in why


def test_an_antibodys_sub_unity_bioavailability_is_the_subcutaneous_products():
    """Intravenous dosing is 100% by definition; 72% can only be the SC form."""
    pick, why = choose("bioavailability", [
        _cand(0.72, "the mean absolute bioavailability of atezolizumab was 72%",
              parameter="bioavailability", generic="atezolizumab"),
    ])
    assert pick is None
    assert "subcutaneous" in why


def test_a_small_molecules_bioavailability_is_fine():
    pick, _ = choose("bioavailability", [
        _cand(0.74, "the absolute oral bioavailability of tofacitinib is 74%",
              parameter="bioavailability", generic="tofacitinib"),
    ])
    assert pick is not None


# ── writing, and taking back ─────────────────────────────────────────────

def test_an_accepted_value_cites_the_label_and_quotes_the_sentence():
    drug = Drug(name="Enzalutamide")
    c = _cand(139.2, "The mean terminal half-life for enzalutamide is 5.8 days.")

    assert accept(drug, c) == []
    stored = get_parameter(drug, "half_life_hours")
    assert stored.provenance is Provenance.MEASURED
    assert stored.citation == "FDA-SPL:0000"
    assert "5.8 days" in stored.conditions


def test_a_wrong_write_can_be_taken_back():
    """The library otherwise only adds, and a mistake would be permanent."""
    drug = Drug(name="Atezolizumab")
    accept(drug, _cand(0.72, "absolute bioavailability was 72%",
                       parameter="bioavailability"))
    assert get_parameter(drug, "bioavailability") is not None

    assert forget(drug, "bioavailability")

    assert get_parameter(drug, "bioavailability") is None
    assert not forget(drug, "bioavailability"), "nothing left to forget"
