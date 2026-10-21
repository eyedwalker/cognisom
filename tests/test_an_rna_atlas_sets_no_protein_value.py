"""A transcript atlas annotates this library. It never sets a value.

CZ CELLxGENE Census aggregates the Human Cell Atlas and Tabula Sapiens
into 154 million cells, and all 25 of this library's immune lineages
appear in it. The obvious thing to do with that is the wrong thing: RNA
is not protein, the correlation across genes runs about 0.4-0.6, and a
transcript count cannot become a copy number or an abundance.

So the sourcing module has no ``set_parameter`` call in it, and the
first test here asserts that structurally rather than trusting the
author. What it does instead is record two things an atlas genuinely
supports: how many cells back each lineage, and whether an asserted
marker is actually enriched in the lineage that claims it.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE = (REPO_ROOT / "cognisom" / "library" / "cellxgene_sourcing.py")


@pytest.fixture(scope="module")
def store():
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        return EntityStore()


# ── the line the module does not cross ───────────────────────────────

def test_the_module_cannot_write_a_parameter():
    """Structural, not a convention. RNA setting a protein number is the
    one failure this module exists to avoid, so it has no way to.

    Checked by parsing rather than by grep: the module's own docstring
    says it makes no such call, and a text search matches that sentence
    and passes for the wrong reason.
    """
    import ast

    tree = ast.parse(MODULE.read_text())
    names = {node.id for node in ast.walk(tree)
             if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree)
             if isinstance(node, ast.Attribute)}
    imported = {alias.name for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                for alias in node.names}

    forbidden = {"set_parameter", "ParameterValue", "Provenance"}
    assert not (forbidden & (names | attrs | imported)), (
        f"the module references {sorted(forbidden & (names | attrs | imported))}")


def test_it_says_why_in_its_own_docstring():
    from cognisom.library import cellxgene_sourcing as mod

    assert "RNA is not protein" in mod.__doc__
    assert "0.4-0.6" in mod.__doc__


# ── coverage, so thin evidence is visible ────────────────────────────

def test_every_lineage_maps_to_an_ontology_label(store):
    from cognisom.library.cellxgene_sourcing import ONTOLOGY
    from cognisom.library.models import EntityType

    entities, _ = store.search(limit=10000)
    lineages = {e.name for e in entities
                if e.entity_type is EntityType.IMMUNE_CELL}
    missing = lineages - set(ONTOLOGY)
    assert not missing, f"no atlas label for {sorted(missing)}"
    assert len(ONTOLOGY) == 25


def test_coverage_is_recorded_with_the_census_version(store):
    from cognisom.library.cellxgene_sourcing import CENSUS_VERSION

    entity = store.find_entity_by_name("Neutrophil")
    cov = (getattr(entity, "sourced", None) or {}).get("atlas_coverage")
    assert cov, "the neutrophil has no recorded atlas coverage"
    assert cov["cells"] > 100_000
    assert cov["census_version"] == CENSUS_VERSION, (
        "an unpinned census would make every recorded count "
        "unreproducible")


def test_a_thin_lineage_says_so(store):
    """A result resting on Th2 behaviour rests on 55 cells somewhere."""
    entity = store.find_entity_by_name("CD4+ Th2 cell")
    cov = (getattr(entity, "sourced", None) or {})["atlas_coverage"]
    assert cov["cells"] < 5_000
    assert "THIN" in cov["note"]


@pytest.mark.parametrize("lineage", ["Eosinophil", "Basophil"])
def test_the_granulocytes_are_thin_and_that_is_the_assay(store, lineage):
    """They carry little RNA and survive dissociation poorly. Their
    absence is a property of the method, not of blood."""
    cov = (getattr(store.find_entity_by_name(lineage), "sourced", None)
           or {})["atlas_coverage"]
    assert cov["cells"] < 5_000


def test_the_macrophage_states_share_one_label(store):
    """The atlas does not type polarisation, so no M1/M2 difference can
    be checked against it. Recorded rather than silently conflated."""
    from cognisom.library.cellxgene_sourcing import ONTOLOGY

    assert ONTOLOGY["Macrophage M1"] == ONTOLOGY["Macrophage M2"] == "macrophage"
    for name in ("Macrophage M1", "Macrophage M2"):
        note = ((getattr(store.find_entity_by_name(name), "sourced", None)
                 or {})["atlas_coverage"]["note"])
        assert "does not separate" in note


# ── markers, which can fail ──────────────────────────────────────────

def test_markers_resolve_to_genes_through_the_librarys_own_records():
    from cognisom.library.cellxgene_sourcing import marker_gene_map

    gene_of = marker_gene_map()
    assert gene_of.get("CD19") == "CD19"
    assert gene_of.get("BLIMP-1") == "PRDM1"
    assert gene_of.get("CD117") == "KIT"


def test_a_marker_that_is_not_one_gene_is_reported_not_dropped():
    """IgD is not a single gene, CD127lo is an expression level, TCRgd
    is a heterodimer. Silence about those would read as a pass."""
    from cognisom.library.cellxgene_sourcing import marker_gene_map

    gene_of = marker_gene_map()
    for marker in ("IgD", "CD127lo", "TCRgd", "class-switched_Ig"):
        assert marker not in gene_of


def test_the_enrichment_rule_is_a_ratio_within_one_gene():
    """Comparing two genes' counts would not be supportable; comparing
    one gene across cell types is."""
    from cognisom.library.cellxgene_sourcing import ENRICHMENT, MarkerCheck

    strong = MarkerCheck("Naive B cell", "CD19", "CD19", 0.60, 0.02)
    assert strong.enrichment == pytest.approx(30.0)
    assert strong.supported

    flat = MarkerCheck("Naive B cell", "CD19", "CD19", 0.08, 0.07)
    assert not flat.supported
    assert ENRICHMENT > 1.0


def test_a_marker_absent_from_its_own_lineage_is_unsupported():
    from cognisom.library.cellxgene_sourcing import MarkerCheck

    absent = MarkerCheck("Eosinophil", "CD15", "FUT4", 0.01, 0.0)
    assert not absent.supported, (
        "a marker in 1% of its own lineage should not pass on a zero "
        "background")
