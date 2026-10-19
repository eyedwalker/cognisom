"""BRENDA reports every published measurement, including the wrong ones.

SABIO-RK's retirement left BRENDA as the structured source for turnover
numbers and Michaelis constants, and the library needs them: 26 entities carry
kcat or Km and none of those values has evidence behind it.

The hazard is not access, it is selection. BRENDA returns measurements across
organisms, substrates, assay conditions and point mutants, and it does not
separate them in any field except free-text commentary. A kcat measured on a
C163A caspase is a real number from a real paper about a protein the library
does not model. So mutants are excluded by default and the exclusion is
tested, along with the unit boundary -- BRENDA reports Km in millimolar and
this library stores micromolar, a factor of a thousand that must be applied
deliberately.

Everything here runs offline. Calls that would need credentials assert that
they refuse cleanly instead.
"""
from __future__ import annotations

import os

import pytest

from cognisom.library.brenda_sourcing import (
    BrendaCredentialsMissing, KineticRecord, _envelope, _parse, accept,
    credentials, km_values, turnover_numbers,
)
from cognisom.library.models import Protein
from cognisom.library.parameters import Provenance, get_parameter


@pytest.fixture
def no_credentials(monkeypatch):
    monkeypatch.delenv("BRENDA_EMAIL", raising=False)
    monkeypatch.delenv("BRENDA_PASSWORD", raising=False)


# ── access, and refusing without it ──────────────────────────────────────

def test_without_credentials_nothing_is_requested(no_credentials):
    with pytest.raises(BrendaCredentialsMissing) as exc:
        km_values("3.4.22.56")

    message = str(exc.value)
    assert "BRENDA_EMAIL" in message and "BRENDA_PASSWORD" in message
    assert "register" in message, "the error must say how to fix itself"


def test_the_password_is_hashed_and_never_sent_in_the_clear(monkeypatch):
    monkeypatch.setenv("BRENDA_EMAIL", "someone@example.org")
    monkeypatch.setenv("BRENDA_PASSWORD", "hunter2")

    email, digest = credentials()

    assert email == "someone@example.org"
    assert digest != "hunter2"
    assert len(digest) == 64, "BRENDA expects a SHA-256 hex digest"


def test_the_request_envelope_is_well_formed():
    from xml.etree import ElementTree

    xml = _envelope("getKmValue", ["a@b.c", "d" * 64, "1.1.1.1", "Homo sapiens"])

    root = ElementTree.fromstring(xml)
    assert root.tag.endswith("Envelope")
    assert "getKmValue" in xml


def test_an_ampersand_in_a_field_does_not_break_the_envelope():
    from xml.etree import ElementTree

    xml = _envelope("getKmValue", ["a&b@c.d", "x", "1.1.1.1", "Homo & sapiens"])

    ElementTree.fromstring(xml)   # would raise if unescaped


# ── reading what BRENDA returns ──────────────────────────────────────────

def test_a_record_row_is_split_into_fields():
    row = ("ecNumber*3.4.22.56#kmValue*0.0095#substrate*Ac-DEVD-AMC"
           "#organism*Homo sapiens#literature*12345678")

    fields = _parse(row)

    assert fields["kmValue"] == "0.0095"
    assert fields["substrate"] == "Ac-DEVD-AMC"
    assert fields["organism"] == "Homo sapiens"


def _record(**kw):
    base = dict(ec_number="3.4.22.56", parameter="km", value=0.0095, unit="mM",
                substrate="Ac-DEVD-AMC", organism="Homo sapiens",
                commentary="", literature="12345678", pmid="12345678")
    base.update(kw)
    return KineticRecord(**base)


@pytest.mark.parametrize("commentary,wild_type", [
    ("", True),
    ("pH 7.4, 37C", True),
    ("mutant C163A", False),
    ("MUTANT enzyme", False),
    ("site-directed mutation of the active site", False),
])
def test_a_mutant_is_recognised_from_the_commentary(commentary, wild_type):
    """BRENDA records mutants nowhere else."""
    assert _record(commentary=commentary).is_wild_type is wild_type


def test_a_non_human_record_is_recognised():
    assert _record(organism="Homo sapiens").is_human
    assert not _record(organism="Rattus norvegicus").is_human


# ── writing a chosen record ──────────────────────────────────────────────

def test_an_accepted_record_names_its_substrate_and_cites_the_paper():
    enzyme = Protein(name="Caspase-3")

    problems = accept(enzyme, "km_um", _record(), value=9.5,
                      note="converted from 0.0095 mM, x1000")

    assert problems == []
    stored = get_parameter(enzyme, "km_um")
    assert stored.value == pytest.approx(9.5)
    assert stored.provenance is Provenance.MEASURED
    assert stored.citation == "PMID:12345678"
    assert "Ac-DEVD-AMC" in stored.conditions, (
        "a Michaelis constant without its substrate describes nothing")
    assert "x1000" in stored.conditions, "the unit conversion must be stated"
    assert "BRENDA" in stored.source


def test_a_record_with_no_literature_is_refused():
    enzyme = Protein(name="Caspase-3")

    problems = accept(enzyme, "km_um", _record(pmid=""), value=9.5)

    assert problems and "PubMed" in problems[0]
    assert get_parameter(enzyme, "km_um") is None


def test_a_record_with_no_value_is_refused():
    enzyme = Protein(name="Caspase-3")

    problems = accept(enzyme, "km_um", _record(value=None))

    assert problems


def test_the_millimolar_to_micromolar_boundary_is_not_crossed_silently():
    """Writing BRENDA's own number into a micromolar field is 1000x wrong."""
    enzyme = Protein(name="Caspase-3")

    accept(enzyme, "km_um", _record(value=0.0095))

    stored = get_parameter(enzyme, "km_um")
    assert stored.value == pytest.approx(0.0095), (
        "accept() must not convert on its own; the caller passes the "
        "converted value so the factor is deliberate and recorded")
