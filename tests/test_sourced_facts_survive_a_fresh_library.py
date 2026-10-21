"""A fact that came off the network must survive leaving this machine.

The entity library's database is local state. ``curated_parameters.json``
exists so that evidence reaches a fresh clone, CI and production -- and
for a long time it carried parameters and three identity fields and
nothing else, which recreated the same problem one level down.

A sourcing pass also writes to ``entity.sourced``: PaxDb abundances with
the datasets they were measured in, AlphaFold accessions, HPA subcellular
locations, the leukocyte diameter convention. None of it was exported.
The numbers reached a fresh library; the record of where they came from
did not.

The test of what belongs in the file is reproducibility. A seeded fact is
rebuilt by a module in this repository. A fetched one is not, and losing
it means going back to the network -- or not knowing it is gone.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def fresh_store():
    """A library built from nothing but what is committed."""
    from cognisom.library.store import EntityStore

    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            yield EntityStore(db_path=str(Path(tmp) / "fresh.db"))


@pytest.fixture(scope="module")
def document():
    from cognisom.library.curated import CURATED_PATH

    return json.loads(CURATED_PATH.read_text())


def _sourced(store, name):
    entity = store.find_entity_by_name(name)
    assert entity is not None, f"{name} is not in a fresh library"
    return dict(getattr(entity, "sourced", None) or {})


# ── the four sources whose facts were being lost ─────────────────────

@pytest.mark.parametrize("entity_name,key", [
    ("Caspase-8", "abundance_ppm"),          # PaxDb
    ("Caspase-8", "abundance_measured_in"),  # ...and which datasets
    ("Monocyte", "diameter_convention"),     # PMID 2272970
])
def test_a_fetched_fact_reaches_a_fresh_library(fresh_store, entity_name, key):
    assert key in _sourced(fresh_store, entity_name), (
        f"{entity_name}.{key} exists only on the machine that fetched it")


def test_the_abundance_keeps_the_datasets_it_was_measured_in(fresh_store):
    """An abundance without its datasets cannot be re-checked, and the
    shared-peptide-pool detector works by comparing counts across every
    dataset where two proteins both appear."""
    measured_in = _sourced(fresh_store, "Caspase-8")["abundance_measured_in"]
    assert isinstance(measured_in, dict) and len(measured_in) >= 2


def test_the_diameter_convention_travels_with_the_diameter(fresh_store):
    """The number is useless on its own: this library holds leukocyte
    radii measured ruffle-INclusive and granule fractions measured
    ruffle-EXclusive, and only this string says so."""
    note = _sourced(fresh_store, "Monocyte")["diameter_convention"]
    assert "ruffle-inclusive" in note
    assert "PMID:2272970" in note and "PMID:6775712" in note


# ── the identity fields that let an accession be checked ─────────────

def test_a_mass_keeps_what_proves_it_is_the_right_protein(document):
    """``uniprot_id`` alone does not show the accession is the protein
    meant -- the recommended name and the sequence length are what a
    reader checks it against, and neither was exported.

    Asserted over the file rather than one entity: 29 of the 81 proteins
    carrying an accession predate the pass that fetched names, so a
    single example proves nothing either way.
    """
    from cognisom.library.curated import IDENTITY_FIELDS

    for field in ("uniprot_recommended_name", "sequence_length_aa",
                  "referenced_as"):
        assert field in IDENTITY_FIELDS

    identities = [e.get("identity") or {} for e in document["entities"]]
    named = [i for i in identities if i.get("uniprot_recommended_name")]
    lengths = [i for i in identities if i.get("sequence_length_aa")]
    # 108 and 56 at the time of writing. sequence_length_aa defaults to
    # the int 0, so a census that treats 0 as present counts 126 -- it
    # is not, and the export is right to skip it.
    assert len(named) >= 100, f"only {len(named)} recommended names exported"
    assert len(lengths) >= 50, f"only {len(lengths)} sequence lengths exported"

    # and the name must travel WITH its accession, or it proves nothing
    for identity in named:
        assert identity.get("uniprot_id"), identity


# ── the export is whole, and says what it wrote ──────────────────────

def test_the_export_carries_sourced_wholesale_not_by_allow_list(document):
    """An allow-list of key names would silently drop the next source to
    be added. The slot is exported whole for that reason."""
    with_sourced = [e for e in document["entities"] if e.get("sourced")]
    assert len(with_sourced) >= 150, (
        f"only {len(with_sourced)} entities carry sourced facts")
    keys = {k for e in with_sourced for k in e["sourced"]}
    assert {"abundance_ppm", "alphafold_model", "hpa_location"} <= keys


def test_applying_twice_changes_nothing(fresh_store):
    """Merged, not assigned: a second pass must not churn the store, and
    must not drop a key a later sourcing run added."""
    from cognisom.library import curated

    first = curated.apply_curated(fresh_store, force=True)
    second = curated.apply_curated(fresh_store, force=True)
    assert second.written == 0, (
        f"a second apply rewrote {second.written} records; "
        f"the first wrote {first.written}")
    assert not second.refused


def test_a_later_sourced_key_is_not_wiped_by_apply(fresh_store):
    """apply must merge. A wholesale assignment would delete whatever a
    sourcing pass wrote since the file was exported."""
    from cognisom.library import curated

    entity = fresh_store.find_entity_by_name("Monocyte")
    entity.sourced["a_later_pass"] = "must survive"
    fresh_store.update_entity(entity, changed_by="test")

    curated.apply_curated(fresh_store, force=True)
    assert _sourced(fresh_store, "Monocyte").get("a_later_pass") == \
        "must survive"
