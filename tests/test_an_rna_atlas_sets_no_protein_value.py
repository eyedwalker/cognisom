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


def test_a_gene_cannot_test_a_distinction_between_its_own_isoforms():
    """The library had already written down why, in a field nothing read.

    CD45RA and CD45RO are splice isoforms of PTPRC, and both entity records
    carry an ``isoform_note`` saying so outright: "CD45RA and CD45RO are the
    two isoforms whose difference IS the marker". Transcript counts for PTPRC
    are therefore the SAME NUMBER for a naive cell's CD45RA claim and a memory
    cell's CD45RO claim, and the atlas run duly returned 1.14x for one and
    0.78x for the other -- two measurements of nothing, filed as two failures.

    PTPRC is the only gene in the library backing two marker names, so this
    is a small category; it is also the only one where a passing-looking
    number is actively misleading.
    """
    from cognisom.library.cellxgene_sourcing import _gene_siblings, MarkerCheck

    siblings, notes = _gene_siblings()
    assert siblings == {"PTPRC": {"CD45RA", "CD45RO"}}, siblings
    assert set(notes) == {"CD45RA", "CD45RO"}
    for name in ("CD45RA", "CD45RO"):
        assert "isoform" in notes[name].lower()

    collapsed = MarkerCheck(
        "CD8+ Naive T cell", "CD45RA", "PTPRC", 0.869, 0.760,
        gene_shared_with=("CD45RO",))
    assert collapsed.verdict == "COLLAPSED"
    assert not collapsed.supported
    # The enrichment ratio is computable and meaningless; the verdict must
    # not be derived from it.
    assert collapsed.enrichment == pytest.approx(0.869 / 0.760, rel=1e-6)


def test_a_true_attribute_that_is_not_distinctive_is_not_a_contradiction():
    """CD3 on an NKT cell: right about the cell, useless for telling it apart.

    The enrichment rule compares a lineage against the median of the others,
    and the others are mostly T cells, which all express CD3E. So the ratio
    comes out at 1.04x by construction -- the comparison set contains the
    thing being tested. Reading that as "the atlas disagrees" would be
    backwards: 53.5% of NKT cells express it, which is the claim.
    """
    from cognisom.library.cellxgene_sourcing import MarkerCheck, UBIQUITY

    shared = MarkerCheck("NKT cell", "CD3", "CD3E", 0.535, 0.512)
    assert shared.verdict == "SHARED"
    assert shared.pct_in_lineage > UBIQUITY
    assert shared.pct_median_elsewhere > UBIQUITY

    # Rare elsewhere and flat is a different thing entirely, and still counts
    # against the library.
    assert MarkerCheck("Naive B cell", "CD19", "CD19",
                       0.08, 0.07).verdict == "CONTRADICTED"


def test_an_induced_marker_under_the_dropout_floor_is_not_evidence():
    """iNOS and arginase-1 are what polarisation DOES, not what a cell is.

    All four markers that failed only the abundance floor pointed the right
    way -- 4.5x for ARG1 and unbounded for NOS2 and NCR2, since no other
    lineage expressed them at all -- at 0.1% to 4.7% of their own lineage.
    scRNA-seq dropout puts a floor under a low-abundance transcript, so the
    atlas is silent here, not contrary. CD80 at 4.7% against a 5% floor says
    how little this verdict can bear: it is "not measurable", and a
    threshold's near-misses must not be reported as refutations.
    """
    from cognisom.library.cellxgene_sourcing import (
        MarkerCheck, DETECTION_FLOOR, ENRICHMENT)

    for lineage, marker, gene, mine, other in (
            ("Macrophage M1", "iNOS", "NOS2", 0.001, 0.0),
            ("Macrophage M2", "arginase-1", "ARG1", 0.007, 0.002),
            ("ILC3", "NKp44", "NCR2", 0.022, 0.0),
            ("Macrophage M1", "CD80", "CD80", 0.047, 0.005)):
        check = MarkerCheck(lineage, marker, gene, mine, other)
        assert check.verdict == "BELOW_DETECTION", (marker, check.verdict)
        assert check.pct_in_lineage <= DETECTION_FLOOR
        assert check.enrichment >= ENRICHMENT, (
            f"{marker} must still point the right way, or it belongs in "
            "CONTRADICTED"
        )


def test_the_four_verdicts_that_are_not_evidence_are_kept_apart():
    """One boolean put four situations in one bucket read as disagreement.

    Of 85 markers tested against the census, 77 were confirmed and 8 were
    not. Those 8 were three isoform collapses, one shared lineage attribute
    and four markers under the dropout floor -- and nothing contradicted.
    "8 unsupported" and "0 contradicted" describe the same run, and only one
    of them is true about the library.
    """
    from cognisom.library.cellxgene_sourcing import MarkerCheck, Run

    run = Run(markers=[
        MarkerCheck("Eosinophil", "MBP", "PRG2", 0.706, 0.001),
        MarkerCheck("CD8+ Naive T cell", "CD45RA", "PTPRC", 0.869, 0.760,
                    gene_shared_with=("CD45RO",)),
        MarkerCheck("NKT cell", "CD3", "CD3E", 0.535, 0.512),
        MarkerCheck("Macrophage M1", "iNOS", "NOS2", 0.001, 0.0),
    ])
    assert len(run.unsupported) == 3
    assert run.contradicted == []
    assert set(run.by_verdict()) == {
        "CONFIRMED", "COLLAPSED", "SHARED", "BELOW_DETECTION"}
    text = run.summary()
    for word in ("confirmed", "collapsed", "shared", "below_detection"):
        assert word in text, text
