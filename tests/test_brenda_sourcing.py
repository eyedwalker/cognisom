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
    """Neither the environment nor the project's .env supplies them.

    Both have to be neutralised: credentials() falls back to the .env file
    the library's other secrets live in, so clearing the environment alone
    leaves a developer's real credentials in play and the test passes or
    fails depending on whose machine it runs on.
    """
    import cognisom.library.brenda_sourcing as mod
    monkeypatch.delenv("BRENDA_EMAIL", raising=False)
    monkeypatch.delenv("BRENDA_PASSWORD", raising=False)
    monkeypatch.setattr(mod, "_from_env_file", lambda key: "")


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

    xml = _envelope("getKmValue",
                    "a@b.c," + "d" * 64 + ",ecNumber*1.1.1.1#organism*Homo sapiens#")

    root = ElementTree.fromstring(xml)
    assert root.tag.endswith("Envelope")
    assert "getKmValue" in xml
    assert "ecNumber*1.1.1.1" in xml, (
        "BRENDA reads one comma-separated string, not separate XML parts; "
        "sending parts made it answer that the credentials were wrong "
        "because it had read none")


def test_an_ampersand_in_a_field_does_not_break_the_envelope():
    from xml.etree import ElementTree

    xml = _envelope("getKmValue", "a&b@c.d,x,ecNumber*1.1.1.1#organism*Homo & sapiens#")

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


# ── what BRENDA says when it refuses ─────────────────────────────────────

class _Response:
    """Shaped like requests' response, for the fault paths."""

    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text
        self.content = text.encode("utf-8")


def _fault(code, message):
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/">'
        f"<SOAP-ENV:Body><SOAP-ENV:Fault><faultcode>{code}</faultcode>"
        f"<faultstring>{message}</faultstring>"
        "</SOAP-ENV:Fault></SOAP-ENV:Body></SOAP-ENV:Envelope>"
    )


def _call_with(monkeypatch, response):
    import cognisom.library.brenda_sourcing as mod

    class _Session:
        headers: dict = {}

        def post(self, *a, **kw):
            return response

    monkeypatch.setenv("BRENDA_EMAIL", "someone@example.org")
    monkeypatch.setenv("BRENDA_PASSWORD", "x")
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    return mod._call("getKmValue", "ecNumber*3.4.22.56#", _Session())


def test_a_rejected_password_is_reported_as_a_credentials_problem(monkeypatch):
    """BRENDA returns this inside an HTTP 500.

    Judging the status before parsing the body threw away the only line
    that says what to do, and reported a bad password as a server error.
    """
    response = _Response(500, _fault(
        "401", "Username or password is wrong, or account was not activated."))

    with pytest.raises(BrendaCredentialsMissing) as exc:
        _call_with(monkeypatch, response)

    message = str(exc.value)
    assert "not activated" in message
    assert "confirmation email" in message, (
        "the commonest cause is an unactivated account; the error should "
        "say so rather than leaving it to be guessed")


def test_another_fault_is_reported_with_its_own_message(monkeypatch):
    from cognisom.library.brenda_sourcing import BrendaError

    response = _Response(500, _fault("500", "EC number not found"))

    with pytest.raises(BrendaError) as exc:
        _call_with(monkeypatch, response)

    assert "EC number not found" in str(exc.value)


def test_a_status_error_with_no_fault_still_reports_the_body(monkeypatch):
    from cognisom.library.brenda_sourcing import BrendaError

    response = _Response(503, "<html>temporarily unavailable</html>")

    with pytest.raises(BrendaError) as exc:
        _call_with(monkeypatch, response)

    assert "503" in str(exc.value)


# ── the response is one blob of many records ─────────────────────────────

def test_many_records_are_split_out_of_one_response(monkeypatch):
    """BRENDA concatenates every record into a single <return> element.

    Records are joined by "!" and fields by "#". Reading the blob as one
    record -- which the element structure invites, since there is only ever
    one element -- returned a single row where caspase-3 has fifty-three,
    and the next record's first field showed up as a key called
    "!ecNumber", which is the tell.
    """
    blob = ("ecNumber*3.4.22.56#kmValue*0.0337#substrate*Ac-DEVD-AMC"
            "#organism*Homo sapiens#literature*700001"
            "!ecNumber*3.4.22.56#kmValue*0.0625#substrate*DEVD-pNA"
            "#organism*Homo sapiens#literature*700002")
    envelope = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/">'
        f"<SOAP-ENV:Body><return>{blob}</return></SOAP-ENV:Body></SOAP-ENV:Envelope>"
    )

    records = _call_with(monkeypatch, _Response(200, envelope))

    assert len(records) == 2
    assert records[0]["substrate"] == "Ac-DEVD-AMC"
    assert records[1]["kmValue"] == "0.0625"
    assert not any(k.startswith("!") for r in records for k in r), (
        "a leading '!' in a key means the record separator was not applied")


def test_the_no_value_sentinel_is_not_read_as_a_number():
    """BRENDA writes -999 where the paper gave no number."""
    from cognisom.library.brenda_sourcing import NO_VALUE, _records

    rows = [{"kmValue": "-999", "substrate": "more", "organism": "Homo sapiens",
             "literature": "1"},
            {"kmValue": "0.0337", "substrate": "Ac-DEVD-AMC",
             "organism": "Homo sapiens", "literature": "1"}]

    out = _records(rows, "3.4.22.56", "km", "kmValue", "mM")

    assert out[0].value is None, f"-999 was read as {out[0].value} mM"
    assert out[1].value == pytest.approx(0.0337)
    assert NO_VALUE == -999.0


def test_a_brenda_reference_id_is_not_treated_as_a_pubmed_id():
    """`literature` is BRENDA-internal; the PMID comes from getReference.

    Citing the reference number as a PMID would resolve to an unrelated
    paper, or to nothing at all.
    """
    from cognisom.library.brenda_sourcing import _records

    rows = [{"kmValue": "0.0337", "substrate": "Ac-DEVD-AMC",
             "organism": "Homo sapiens", "literature": "647471"}]

    without = _records(rows, "3.4.22.56", "km", "kmValue", "mM")
    assert without[0].pmid == "", "the BRENDA reference id leaked in as a PMID"
    assert not without[0].citable

    with_refs = _records(rows, "3.4.22.56", "km", "kmValue", "mM",
                         {"647471": {"pubmedId": "20580860",
                                     "title": "A paper", "year": "2010"}})
    assert with_refs[0].pmid == "20580860"
    assert with_refs[0].citable


# ── a shared EC number is not a protein ──────────────────────────────────

def test_a_shared_ec_number_is_refused(monkeypatch):
    """EC 2.7.10.2 is carried by 44 reviewed human proteins.

    JAK1, JAK2, SRC and BTK all map to it and BRENDA returns the same nine
    turnover numbers for each, every one measured on ATP rather than on the
    protein substrate the kinase acts on. Writing those as JAK1's kcat
    would attach four real papers to a number that is not JAK1's. This is
    the check that stops it, and it is the difference between mining a
    database and manufacturing citations from one.
    """
    import cognisom.library.brenda_sourcing as mod
    from cognisom.library.brenda_sourcing import BrendaError

    monkeypatch.setattr(mod, "proteins_sharing_ec", lambda ec, s=None: 44)

    with pytest.raises(BrendaError) as exc:
        mod.candidates("2.7.10.2", "kcat")

    assert "44" in str(exc.value)
    assert "class" in str(exc.value)


def test_a_single_protein_ec_is_allowed(monkeypatch):
    import cognisom.library.brenda_sourcing as mod

    monkeypatch.setattr(mod, "proteins_sharing_ec", lambda ec, s=None: 1)
    monkeypatch.setattr(mod, "km_values", lambda *a, **k: [])

    assert mod.candidates("3.4.22.56", "km") == []


# ── aggregating across papers ────────────────────────────────────────────

def _k(value, pmid, substrate="L-tryptophan"):
    return KineticRecord(ec_number="1.13.11.52", parameter="km", value=value,
                         unit="mM", substrate=substrate,
                         organism="Homo sapiens", literature="1", pmid=pmid)


def test_the_median_is_written_as_derived_with_the_spread_recorded():
    """One record out of forty-one would be an arbitrary pick."""
    from cognisom.library.brenda_sourcing import accept_median

    enzyme = Protein(name="IDO1")
    records = [_k(0.001, "111"), _k(0.02, "111"), _k(0.02, "222"),
               _k(0.5, "222"), _k(6.8, "333")]

    problems = accept_median(enzyme, "km_um", records, factor=1000.0)

    assert problems == []
    stored = get_parameter(enzyme, "km_um")
    assert stored.value == pytest.approx(20.0), "median of the five, in uM"
    assert stored.provenance is Provenance.DERIVED
    assert "median of 5" in stored.rationale
    assert "3 papers" in stored.rationale
    assert stored.citation in ("PMID:111", "PMID:222")


def test_a_single_record_is_measured_not_derived():
    """With one record there is no aggregation to describe."""
    from cognisom.library.brenda_sourcing import accept_median

    enzyme = Protein(name="TMPRSS2")

    accept_median(enzyme, "kcat_per_s", [_k(18.0, "35676539", "Boc-LGR-AMC")])

    stored = get_parameter(enzyme, "kcat_per_s")
    assert stored.provenance is Provenance.MEASURED
    assert stored.citation == "PMID:35676539"


def test_the_unit_factor_is_applied_to_the_median_not_assumed():
    from cognisom.library.brenda_sourcing import accept_median

    enzyme = Protein(name="IDO1")

    accept_median(enzyme, "km_um", [_k(0.02, "111")], factor=1000.0)

    assert get_parameter(enzyme, "km_um").value == pytest.approx(20.0), (
        "BRENDA reports mM and km_um is micromolar; without the factor the "
        "value is a thousandfold wrong")


def test_aggregating_nothing_is_refused():
    from cognisom.library.brenda_sourcing import accept_median

    assert accept_median(Protein(name="x"), "km_um", [])
