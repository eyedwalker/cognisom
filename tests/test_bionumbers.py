"""Fetching candidates is safe; choosing between them is not automatable.

BioNumbers indexes the quantities no protein database carries -- division
times, crawling speeds, cell volumes -- each with a primary reference. That
makes it the obvious source for most of what this library still lacks, and
also the easiest way to manufacture evidence at speed, because a text search
for "neutrophil migration speed" returns, in order: the general measurement at
17 um/min, the same paper's burn-patient arm at 9, a chemotaxis assay at 19, an
in-vitro tube measurement quoted in um per *second*, and a record of neutrophil
*volume* that matched on words alone.

Picking among those is a modelling judgement. So the module ranks and returns
them and writes nothing, and the tests below pin the two places where a wrong
answer would otherwise look like a right one: a value parsed out of its units,
and a citation naming a paper that does not say what the value claims.

Everything here runs offline against fixture pages.
"""
from __future__ import annotations

import pytest

from cognisom.library.bionumbers import (
    BioNumberRecord, _split_value, _title_matches, accept,
)
from cognisom.library.models import ImmuneCellEntity
from cognisom.library.parameters import Provenance, get_parameter


# ── reading a value out of a record ──────────────────────────────────────

@pytest.mark.parametrize("raw,value,unit,rng", [
    ("17 µm/min Range: ±6 µm/min", 17.0, "µm/min", "±6 µm/min"),
    ("320 μm^3 Range: ±37 μm^3", 320.0, "μm^3", "±37 μm^3"),
    ("1.4e-10 m^2/s", 1.4e-10, "m^2/s", ""),
    ("0.15 μm/sec", 0.15, "μm/sec", ""),
    ("24 hours", 24.0, "hours", ""),
])
def test_a_value_is_read_with_its_units(raw, value, unit, rng):
    got_value, got_unit, got_range = _split_value(raw)

    assert got_value == pytest.approx(value)
    assert got_unit == unit
    assert got_range == rng


def test_a_record_with_no_number_yields_no_value():
    """Some records carry only a prose range. They are leads, not values."""
    value, unit, _ = _split_value("see comments")

    assert value is None


def test_units_are_never_silently_dropped():
    """0.15 um/sec and 17 um/min differ by sixty.

    Both appear in the same search. The parser must surface the unit so a
    caller converts rather than assuming.
    """
    per_second = _split_value("0.15 μm/sec")
    per_minute = _split_value("17 µm/min")

    assert per_second[1] != per_minute[1]


# ── the reference must name the paper the value came from ────────────────

def test_a_matching_title_is_accepted():
    reference = ("Butler KL et al., Burn injury reduces neutrophil directional "
                 "migration speed in microfluidic devices. PLoS One")
    title = "Burn injury reduces neutrophil directional migration speed in microfluidic devices."

    assert _title_matches(reference, title)


def test_an_unrelated_title_is_rejected():
    """An esearch that returns something is not an esearch that was right."""
    reference = ("Butler KL et al., Burn injury reduces neutrophil directional "
                 "migration speed in microfluidic devices. PLoS One")
    title = "Structural basis of androgen receptor activation by a synthetic ligand."

    assert not _title_matches(reference, title)


def test_a_partially_overlapping_title_is_rejected():
    """Shared common words are not a match."""
    reference = "Smith J et al., Neutrophil migration in the lung. Nature"
    title = "Migration of dendritic cells in the skin."

    assert not _title_matches(reference, title)


# ── writing one, once a caller has chosen it ─────────────────────────────

def _record(**kw):
    base = dict(bnid="112247", quantity="Neutrophil speed", value=17.0,
                unit="µm/min", range_text="±6 µm/min",
                organism="Human Homo sapiens",
                reference="Butler KL et al., Burn injury reduces neutrophil "
                          "directional migration speed in microfluidic devices.",
                method="microfluidic device", comments="healthy controls",
                pmid="20689600")
    base.update(kw)
    return BioNumberRecord(**base)


def test_an_accepted_record_is_written_as_measured_with_its_pmid():
    cell = ImmuneCellEntity(name="Neutrophil")

    problems = accept(cell, "migration_speed_um_per_min", _record())

    assert problems == []
    stored = get_parameter(cell, "migration_speed_um_per_min")
    assert stored.value == pytest.approx(17.0)
    assert stored.provenance is Provenance.MEASURED
    assert stored.citation == "PMID:20689600"
    assert "BNID:112247" in stored.source
    assert "Homo sapiens" in stored.conditions


def test_a_record_whose_reference_did_not_resolve_is_refused():
    """Without a verified reference the number has no evidence behind it."""
    cell = ImmuneCellEntity(name="Neutrophil")

    problems = accept(cell, "migration_speed_um_per_min", _record(pmid=""))

    assert problems and "did not resolve" in problems[0]
    assert get_parameter(cell, "migration_speed_um_per_min") is None


def test_a_record_with_no_parsed_value_is_refused():
    cell = ImmuneCellEntity(name="Neutrophil")

    problems = accept(cell, "migration_speed_um_per_min", _record(value=None))

    assert problems and "no value parsed" in problems[0]


def test_a_converted_value_can_be_supplied_with_a_note():
    """The um/sec record needs a factor of 60, stated rather than hidden."""
    cell = ImmuneCellEntity(name="Neutrophil")
    record = _record(bnid="106809", value=0.15, unit="μm/sec", pmid="1234567")

    problems = accept(cell, "migration_speed_um_per_min", record,
                      value=9.0, note="converted from 0.15 um/sec, x60")

    assert problems == []
    stored = get_parameter(cell, "migration_speed_um_per_min")
    assert stored.value == pytest.approx(9.0)
    assert "x60" in stored.conditions


def test_the_spec_still_validates_an_accepted_value():
    """accept() goes through set_parameter, so the range check still applies."""
    cell = ImmuneCellEntity(name="Neutrophil")

    problems = accept(cell, "migration_speed_um_per_min",
                      _record(value=1e9))

    assert problems, "an absurd speed was accepted without complaint"


def test_citable_requires_both_a_value_and_a_reference():
    assert _record().citable
    assert not _record(pmid="").citable
    assert not _record(value=None).citable
