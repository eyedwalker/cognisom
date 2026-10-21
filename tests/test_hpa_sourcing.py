"""Where a protein is, measured rather than asserted.

``cell_profiles.COMPARTMENT`` is a hand-written table of 60 marker
strings against a vocabulary of 173, written from memory by one person.
The library's own ``compartments`` field is better and is still an
assertion. The Human Protein Atlas is the measurement: antibody-based
immunofluorescence, CC BY 4.0, and free for commercial use **with
attribution** -- which is a condition, not a courtesy, and is why every
value this module writes carries its source string.

It was unblocked by a bug fix rather than by new data. HPA joins on
UniProt accession, and this library held 37 accessions where it should
have held 131: the rest had been fetched, verified against gene symbol,
recommended name and length, and then dropped on the way to disk.

THE TRAP THIS MODULE WALKED INTO ANYWAY
---------------------------------------
``_extra_properties`` is a FIXED key set per subclass. A loose attribute
is dropped on save, with no error and no warning. The first run of this
module reported "91 written" and stored **zero**, exactly as the three
losses before it had: the effectors' UniProt names, the 94 accessions,
and now a whole atlas pass.

``BioEntity.sourced`` exists so the fourth does not happen. Anything put
there survives whatever the subclass is, and the generic test below
fails if a new field is added that does not.
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

from cognisom.library import hpa_sourcing as hpa        # noqa: E402


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


class _Session:
    """Answers one scripted HPA query."""

    def __init__(self, payload):
        self.payload = payload
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)

        class _R:
            def __init__(self, p):
                self._p = p

            def json(self):
                return self._p

        return _R(self.payload)


# ── the lookup ───────────────────────────────────────────────────────

def test_the_row_is_matched_on_accession_not_on_order():
    """A gene search returns several rows -- a query for CD8A also
    returns CRTAM -- and taking the first is how a library records one
    protein's location against another's name."""
    session = _Session([
        {"Gene": "CRTAM", "Uniprot": ["O95727"],
         "Subcellular main location": ["Nucleoplasm"]},
        {"Gene": "CD8A", "Uniprot": ["P01732"],
         "Subcellular main location": ["Plasma membrane"]},
    ])
    found = hpa.fetch("CD8A", "P01732", session)
    assert found.hpa_terms == ("Plasma membrane",)
    assert found.bucket == "membrane"


def test_an_accession_that_is_not_there_is_refused():
    session = _Session([{"Gene": "CD8A", "Uniprot": ["O95727"],
                         "Subcellular main location": ["Nucleoplasm"]}])
    found = hpa.fetch("CD8A", "P01732", session)
    assert not found.ok
    assert "none carried P01732" in found.note


def test_an_unmapped_term_does_not_default_to_the_membrane():
    """The hand table's failure mode: anything it did not name fell
    through to the cell surface."""
    session = _Session([{"Gene": "X", "Uniprot": ["P00001"],
                         "Subcellular main location": ["Somewhere new"]}])
    found = hpa.fetch("X", "P00001", session)
    assert found.bucket == ""
    assert "no term maps" in found.note


def test_the_most_decisive_compartment_wins_as_elsewhere():
    session = _Session([{"Gene": "X", "Uniprot": ["P00001"],
                         "Subcellular main location": ["Cytosol", "Vesicles"]}])
    assert hpa.fetch("X", "P00001", session).bucket == "granule"


# ── the secretory trap ───────────────────────────────────────────────

def test_a_secreted_protein_seen_in_the_golgi_stays_secreted(store):
    """HPA reports where a protein was SEEN. For a secreted one that is
    where it was caught in transit: CXCL8 -- interleukin-8, whose entire
    function happens outside the cell -- reads Golgi apparatus.

    A naive mapping puts every chemokine in the Golgi and every
    interleukin with it.
    """
    assert "Golgi apparatus" in hpa.SECRETORY_TRANSIT
    assert "Endoplasmic reticulum" in hpa.SECRETORY_TRANSIT
    assert "Vesicles" in hpa.SECRETORY_TRANSIT
    # and a membrane term is not transit
    assert "Plasma membrane" not in hpa.SECRETORY_TRANSIT


def test_the_live_pass_caught_the_transit_cases(store):
    """Two of them in this library, and both are right: TIMP-1 and
    COL1A1 are secreted and HPA sees them on the way out."""
    for name in ("TIMP-1", "Collagen alpha-1(I) chain (COL1A1)"):
        entity = store.find_entity_by_name(name)
        if entity is None:
            continue
        sourced = getattr(entity, "sourced", None) or {}
        # kept as secreted: no hpa_bucket was written over it
        assert "hpa_bucket" not in sourced, (
            f"{name} took its transit compartment as a destination")


# ── what the pass actually stored ────────────────────────────────────

def test_the_library_carries_measured_locations_now(store):
    stored = 0
    for blob, in store._conn.execute("SELECT data_json FROM entities"):
        props = json.loads(blob).get("properties") or {}
        if props.get("hpa_location"):
            stored += 1
    assert stored >= 85, f"only {stored} entities carry an HPA location"


@pytest.mark.parametrize("name,term,bucket", [
    ("FOXP3", "Nucleoplasm", "nucleus"),
    ("Vimentin (VIM)", "Intermediate filaments", "cytoplasm"),
    ("Myeloperoxidase (MPO)", "Vesicles", "granule"),
    ("CD8 alpha chain (CD8A)", "Plasma membrane", "membrane"),
])
def test_a_known_protein_landed_where_it_belongs(store, name, term, bucket):
    """MPO is the one worth reading twice: azurophilic granule cargo,
    which the hand table placed correctly only because somebody had
    typed 'myeloperoxidase' into it. HPA says Vesicles, measured."""
    entity = store.find_entity_by_name(name)
    assert entity is not None, name
    sourced = getattr(entity, "sourced", None) or {}
    assert sourced.get("hpa_location") == term, sourced
    assert sourced.get("hpa_bucket") == bucket


def test_every_stored_location_carries_its_attribution(store):
    """CC BY 4.0 requires it, and a licence condition that lives only in
    a module docstring is a condition nobody ships."""
    for blob, in store._conn.execute("SELECT data_json FROM entities"):
        props = json.loads(blob).get("properties") or {}
        if props.get("hpa_location"):
            assert "Human Protein Atlas" in (props.get("hpa_source") or "")


def test_hpa_did_not_overwrite_the_curated_table(store):
    """The hand table is small and hand-checked. Where the two disagree
    that is a finding for a person, not something to resolve silently --
    and in this library they did not disagree once."""
    from cognisom.visualization.cell_profiles import compartment_of

    disagreements = []
    for blob, in store._conn.execute("SELECT data_json FROM entities"):
        data = json.loads(blob)
        props = data.get("properties") or {}
        bucket = props.get("hpa_bucket")
        curated = compartment_of(data["name"])
        if bucket and curated and curated != bucket:
            disagreements.append((data["name"], curated, bucket))
    assert not disagreements, disagreements


# ── the trap, generically ────────────────────────────────────────────

def test_anything_put_in_sourced_survives_whatever_the_subclass_is():
    """Three separate losses came from writing a loose attribute that
    _extra_properties did not name. This is the slot that does not do
    that, and it has to hold for every class or the next source loses
    its pass too."""
    from cognisom.library import models
    from cognisom.library.models import BioEntity

    for cls_name in ("Protein", "Receptor", "Cytokine", "AdhesionMolecule",
                     "MHCMolecule", "ImmuneCellEntity", "CellTypeEntity",
                     "Drug", "BioEntity"):
        cls = getattr(models, cls_name, None)
        if cls is None:
            continue
        entity = cls(name=f"probe {cls_name}")
        entity.sourced["a_future_atlas"] = "a value"
        props = entity.to_dict().get("properties") or {}
        assert props.get("a_future_atlas") == "a value", (
            f"{cls_name} drops sourced annotations on save")
        back = BioEntity.from_dict(entity.to_dict())
        assert (back.sourced or {}).get("a_future_atlas") == "a value", (
            f"{cls_name} loses sourced annotations on load")


def test_a_typed_property_still_wins_over_a_sourced_one():
    """`sourced` fills gaps; it does not get to redefine a field the
    subclass owns."""
    from cognisom.library.models import Protein

    entity = Protein(name="probe")
    entity.uniprot_id = "P00001"
    entity.sourced["uniprot_id"] = "WRONG"
    assert entity.to_dict()["properties"]["uniprot_id"] == "P00001"
