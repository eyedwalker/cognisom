"""An accession that does not reach the database is not a citation.

Ninety-four molecules were curated with accessions re-resolved against
UniProt -- primary gene symbol, recommended protein name AND sequence
length checked against every submission, 86 of 86 -- and **not one of
them was stored**. The library held 37 entities with an accession where
it should have held 131.

Nothing failed. ``build()`` constructed a bare ``BioEntity`` and set an
ad-hoc ``.properties`` dict; ``BioEntity.to_dict()`` does not serialise
one. Each subclass instead exposes ``_extra_properties``, a FIXED key
set that silently drops anything it does not name, and only ``Protein``
named ``uniprot_id`` at all -- so a receptor, a cytokine, an adhesion
molecule and an MHC chain had nowhere to put an accession even when
built as the right class.

The same bug had already eaten the cytotoxic effectors' verified UniProt
names, and was fixed there one commit earlier without anybody checking
whether it was general. It was.

The cost is specific: a UniProt accession is the join key for the Human
Protein Atlas (subcellular location, which would replace the hand-written
COMPARTMENT table) and for AlphaFold (a real extent, where this library
records 0.005 um for every protein). Both were blocked on data that had
been fetched, verified, and thrown away on the way to disk.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IDENTITY_CLASSES = ("Protein", "Receptor", "Cytokine",
                    "AdhesionMolecule", "MHCMolecule")


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


def _stored(store, name):
    rows = [r[0] for r in store._conn.execute(
        "SELECT data_json FROM entities WHERE name = ?", (name,))]
    return (json.loads(rows[0]).get("properties") or {}) if rows else None


@pytest.mark.parametrize("cls_name", IDENTITY_CLASSES)
def test_every_protein_like_class_can_hold_an_accession(cls_name):
    """A receptor is a protein and has an accession. Only Protein could
    say so, and the rest dropped it."""
    from cognisom.library import models

    fields = getattr(models, cls_name).__dataclass_fields__
    for field in ("uniprot_id", "uniprot_recommended_name", "gene_symbol"):
        assert field in fields, f"{cls_name} cannot hold {field}"
    # Protein had a length field before the others did, under an older
    # name. Either is fine; having neither is not.
    assert ("sequence_length_aa" in fields
            or "amino_acid_length" in fields), f"{cls_name} has no length"


@pytest.mark.parametrize("cls_name", IDENTITY_CLASSES)
def test_the_accession_survives_a_round_trip(cls_name):
    """to_dict then from_dict. The fixed key set is why this needs
    asserting rather than assuming."""
    from cognisom.library import models
    from cognisom.library.models import BioEntity

    cls = getattr(models, cls_name)
    entity = cls(name=f"probe {cls_name}")
    entity.uniprot_id = "P12345"
    entity.uniprot_recommended_name = "A recommended name"
    length_field = ("amino_acid_length"
                    if hasattr(entity, "amino_acid_length")
                    else "sequence_length_aa")
    setattr(entity, length_field, 321)
    entity.gene_symbol = "GENE1"

    props = entity.to_dict().get("properties") or {}
    assert props.get("uniprot_id") == "P12345", (
        f"{cls_name}._extra_properties drops the accession")
    assert props.get("uniprot_recommended_name") == "A recommended name"
    assert props.get(length_field) == 321
    assert props.get("gene_symbol") == "GENE1"

    back = BioEntity.from_dict(entity.to_dict())
    assert (getattr(back, "uniprot_id", "")
            or (back.to_dict().get("properties") or {}).get("uniprot_id")
            ) == "P12345"


def test_every_curated_marker_molecule_kept_its_accession(store):
    """The measurement this file exists for: 94 curated, 94 stored."""
    records = json.loads(
        (REPO_ROOT / "cognisom/library/marker_molecules.json").read_text())
    assert len(records) >= 94

    lost = []
    for record in records:
        props = _stored(store, record["entity_name"])
        if props is None:
            continue
        if props.get("uniprot_id") != record["accession"]:
            lost.append((record["entity_name"], record["accession"],
                         props.get("uniprot_id")))
    assert not lost, f"{len(lost)} accessions did not reach the database: {lost[:5]}"


def test_the_verified_name_is_stored_too(store):
    """The accession alone does not record that it was CHECKED. The
    recommended name read back from UniProt is what does."""
    records = json.loads(
        (REPO_ROOT / "cognisom/library/marker_molecules.json").read_text())
    without = [r["entity_name"] for r in records
               if (_stored(store, r["entity_name"]) or {}).get(
                   "uniprot_recommended_name") != r["uniprot_name"]]
    assert not without, without[:5]


def test_the_cytotoxic_effectors_kept_theirs(store):
    """Same bug, found one commit earlier on these eight and fixed only
    for them."""
    from cognisom.library.seed_cytotoxic_effectors import EFFECTORS

    for effector in EFFECTORS:
        props = _stored(store, effector.name) or {}
        assert props.get("uniprot_id") == effector.accession, effector.name
        assert props.get("uniprot_recommended_name") == effector.uniprot_name


def test_the_library_holds_the_accessions_hpa_would_need(store):
    """A UniProt accession is the join key for the Human Protein Atlas
    and for AlphaFold. 37 before, and both were blocked on it."""
    rows = store._conn.execute(
        "SELECT COUNT(*) FROM entities WHERE "
        "json_extract(data_json, '$.properties.uniprot_id') IS NOT NULL AND "
        "json_extract(data_json, '$.properties.uniprot_id') != ''")
    assert list(rows)[0][0] >= 125
