"""What a BioNumbers-sourced value must look like, whoever chose it.

The curation pass picks, for each library entity and parameter, one record
out of a pre-fetched candidate set. The picking is a judgement; these tests
are not about whether the judgement was good. They are about what every write
must carry so the judgement can be re-examined: a PubMed id that resolves to
the paper BioNumbers cited, the BNID so the record can be re-fetched, the
organism, the unit factor stated rather than assumed, and a value inside the
spec's plausible range.

They are written before the pass lands and pass vacuously on a library with
no BioNumbers-sourced values, so they cannot be satisfied by writing nothing;
``test_the_pass_actually_wrote_something`` is the one that fails in that case.

The live checks -- re-fetching a sample of BNIDs and confirming the stored
value is the record's value times the stated factor -- are marked ``network``
and skipped by default. They are the ones to run after a pass.
"""
from __future__ import annotations

import contextlib
import io
import math
import os
import re

import pytest

from cognisom.library.parameters import (
    PARAMETER_SPECS, Provenance, iter_parameters,
)

BNID_IN_SOURCE = re.compile(r"BNID:(\d+)")
PMID = re.compile(r"^PMID:\d{6,9}$")
#: Parameters the BioNumbers pass is allowed to write. A BioNumbers citation
#: on anything else means the pass wrote outside its remit.
BIONUMBERS_PARAMETERS = {
    "division_time_hours", "migration_speed_um_per_min", "radius_um",
    "apoptosis_rate_per_hour", "lifespan_hours", "replication_cycle_hours",
    "burst_size", "normal_concentration_mM",
    "oxygen_consumption_fmol_per_cell_per_hour",
}
#: Conversions a stated factor may legitimately be. Anything else is a
#: number somebody typed.
KNOWN_FACTORS = {1.0, 60.0, 1 / 60, 24.0, 1 / 24, 0.5, 1e-3, 1e3, 1e6, 1e-6,
                 3600.0, 1 / 3600}


@pytest.fixture(scope="module")
def sourced():
    """Every (entity, parameter, value) the BioNumbers pass wrote."""
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    ids = [r[0] for r in
           store._conn.execute("SELECT entity_id FROM entities").fetchall()]
    out = []
    for entity in (store.get_entity(i) for i in ids):
        if entity is None:
            continue
        for name, pv in iter_parameters(entity):
            if "BioNumbers" in (pv.source or ""):
                out.append((entity, name, pv))
    return out


def test_the_pass_actually_wrote_something(sourced):
    """Vacuous passes elsewhere in this file are caught here.

    One value -- the neutrophil speed, curated by hand before any pass --
    does not count as the pass having run.
    """
    if len(sourced) < 2:
        pytest.skip("no BioNumbers curation pass has run yet")
    assert len(sourced) >= 10, f"only {len(sourced)} values sourced"


def test_every_sourced_value_cites_a_pubmed_id(sourced):
    bad = [(e.name, n, pv.citation) for e, n, pv in sourced
           if not PMID.match(pv.citation or "")]
    assert not bad, f"BioNumbers writes without a resolvable PMID: {bad[:6]}"


def test_every_sourced_value_names_its_record(sourced):
    """Without the BNID the choice cannot be re-fetched and re-examined."""
    bad = [(e.name, n) for e, n, pv in sourced
           if not BNID_IN_SOURCE.search(pv.source or "")]
    assert not bad, f"no BNID recorded: {bad[:6]}"


def test_every_sourced_value_is_human(sourced):
    bad = [(e.name, n, pv.conditions[:60]) for e, n, pv in sourced
           if "sapiens" not in (pv.conditions or "").lower()]
    assert not bad, f"non-human (or organism unrecorded) records written: {bad[:6]}"


def test_every_sourced_value_is_measured_grade(sourced):
    """A BioNumbers record is one paper's measurement; it is never DERIVED."""
    bad = [(e.name, n, pv.provenance.value) for e, n, pv in sourced
           if pv.provenance is not Provenance.MEASURED]
    assert not bad, bad[:6]


def test_the_pass_stayed_within_its_remit(sourced):
    bad = [(e.name, n) for e, n, pv in sourced if n not in BIONUMBERS_PARAMETERS]
    assert not bad, f"BioNumbers cited on parameters it should not source: {bad}"


def test_every_sourced_value_is_inside_the_spec_range(sourced):
    for entity, name, pv in sourced:
        spec = PARAMETER_SPECS.get(name)
        if spec is None:
            continue
        assert spec.check(pv) == [], (
            f"{entity.name} {name}={pv.value} {pv.unit} fails its spec: "
            f"{spec.check(pv)}")
        assert pv.unit == spec.unit, (
            f"{entity.name} {name} stored in {pv.unit!r}, spec wants {spec.unit!r}")


def test_a_stated_unit_factor_is_a_known_conversion(sourced):
    """"x60" is a conversion; "x37" is a number somebody typed."""
    bad = []
    for entity, name, pv in sourced:
        for m in re.finditer(r"[x×]\s*([0-9.]+(?:e-?\d+)?)", pv.conditions or ""):
            factor = float(m.group(1))
            if not any(math.isclose(factor, k, rel_tol=1e-6) for k in KNOWN_FACTORS):
                bad.append((entity.name, name, factor))
    assert not bad, f"unexplained unit factors: {bad[:6]}"


def test_no_construct_was_sourced_from_bionumbers(sourced):
    """A model construct cannot be looked up; a BioNumbers hit for one is a
    word-match, e.g. a 'detection' record that is not a detection radius."""
    bad = [(e.name, n) for e, n, pv in sourced
           if PARAMETER_SPECS.get(n) and PARAMETER_SPECS[n].model_construct]
    assert not bad, bad


def test_the_neutrophil_speed_is_the_curated_one(sourced):
    """The first curation, done by hand earlier: 17 um/min, PMID:20689600."""
    hits = [pv for e, n, pv in sourced
            if e.name == "Neutrophil" and n == "migration_speed_um_per_min"]
    if not hits:
        pytest.skip("neutrophil speed not among BioNumbers-sourced values")
    assert hits[0].value == pytest.approx(17.0)
    assert hits[0].citation == "PMID:20689600"


def test_a_cell_lifespan_and_division_time_are_not_both_written(sourced):
    """Every cell owes one temporal parameter and no cell owes both."""
    by_entity = {}
    for entity, name, _ in sourced:
        by_entity.setdefault(entity.name, set()).add(name)
    both = [n for n, s in by_entity.items()
            if {"lifespan_hours", "division_time_hours"} <= s]
    assert not both, f"both temporal parameters written on: {both}"


# ── live checks: run after a pass with  pytest -m network ───────────────

@pytest.mark.network
def test_a_sample_of_records_refetch_to_the_stored_value(sourced):
    """The stored value must be the record's value times the stated factor.

    This is the check that makes a chosen record re-examinable rather than
    taken on trust: an agent can choose, but it cannot have invented the
    number, because the number is still on BioNumbers.
    """
    if not sourced:
        pytest.skip("nothing sourced")
    from cognisom.library.bionumbers import fetch
    sample = sourced[:: max(1, len(sourced) // 12)][:12]
    for entity, name, pv in sample:
        bnid = BNID_IN_SOURCE.search(pv.source).group(1)
        record = fetch(bnid)
        assert record.value is not None, f"BNID {bnid} no longer parses"
        m = re.search(r"[x×]\s*([0-9.]+(?:e-?\d+)?)", pv.conditions or "")
        factor = float(m.group(1)) if m else 1.0
        derived = re.search(r"ln\s*\(?2\)?\s*/", pv.conditions or "")
        if derived:
            expected = math.log(2) / (record.value * factor)
        else:
            expected = record.value * factor
        assert pv.value == pytest.approx(expected, rel=1e-3), (
            f"{entity.name} {name}: stored {pv.value}, BNID {bnid} says "
            f"{record.value} {record.unit} x {factor} = {expected}")


@pytest.mark.network
def test_a_sample_of_citations_name_the_cited_paper(sourced):
    if not sourced:
        pytest.skip("nothing sourced")
    from cognisom.library.bionumbers import fetch, resolve_pmid
    sample = sourced[:: max(1, len(sourced) // 8)][:8]
    for entity, name, pv in sample:
        bnid = BNID_IN_SOURCE.search(pv.source).group(1)
        pmid, _ = resolve_pmid(fetch(bnid).reference)
        assert f"PMID:{pmid}" == pv.citation, (
            f"{entity.name} {name} cites {pv.citation} but BNID {bnid}'s "
            f"reference resolves to PMID:{pmid}")
