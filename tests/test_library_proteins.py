"""Protein accessions must be resolved by gene symbol, never by name.

UniProt name search is confidently wrong for exactly the entities this library
is full of, because a ligand and its receptor share a name. All four of these
were returned as top hits and accepted by a substring check before the
resolution strategy changed:

    "Androgen Receptor" -> P82094  TATA element modulatory factor
    "TNF-alpha"         -> P78536  ADAM17, the TNF-alpha CONVERTING ENZYME
    "IFN-gamma"         -> P15260  IFNGR1, the RECEPTOR
    "C3a"               -> Q16581  C3AR1, the C3a RECEPTOR

Each hit genuinely contains the query string in its own name, so no amount of
string matching separates them. The fix is not a better matcher: it is to stop
asking by name. A gene symbol is supplied, UniProt is asked for that exact
symbol, and the entry is kept only if the symbol it returns is the one asked
for -- which is what :func:`accepts` enforces.

The refusals matter as much as the resolutions. An entity naming a complex, a
cleavage fragment, a gene family or a polysaccharide has no single accession,
and quietly assigning one of its chains would put a number where there should
be a gap.
"""
from __future__ import annotations

import contextlib
import io
import json
import re

import pytest

from cognisom.library.proteins import (
    GENE_ALIASES, NOT_A_SINGLE_PROTEIN, accepts, candidate_symbols,
    gene_symbol_for,
)

ACCESSION = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")


# ── the four that motivated symbol-only resolution ───────────────────────

@pytest.mark.parametrize("name,symbol", [
    ("Androgen Receptor", "AR"),      # not TMF1
    ("TNF-alpha", "TNF"),             # not ADAM17
    ("IFN-gamma", "IFNG"),            # not IFNGR1
])
def test_a_ligand_maps_to_itself_not_to_its_receptor_or_processing_enzyme(name, symbol):
    got, how = gene_symbol_for(name)

    assert got == symbol, f"{name} must resolve to {symbol}, got {got}"
    assert how == "curated_alias"


def test_a_cleavage_fragment_is_refused_rather_than_sent_to_its_receptor():
    """C3a is a fragment of C3. Name search answers C3AR1, its receptor."""
    got, how = gene_symbol_for("C3a")

    assert got is None
    assert "cleavage fragment" in how


# ── refusals ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(NOT_A_SINGLE_PROTEIN))
def test_things_that_are_not_one_protein_are_refused_with_a_reason(name):
    got, how = gene_symbol_for(name)

    assert got is None, f"{name} should not resolve to a single accession"
    assert how and how.startswith("not a single protein:")
    assert len(how) > len("not a single protein: ") + 8, "the reason must say why"


def test_a_polysaccharide_is_not_given_a_protein_accession():
    got, how = gene_symbol_for("Hyaluronan")

    assert got is None
    assert "not a protein" in how


def test_heterodimers_are_refused_rather_than_resolved_to_one_chain():
    """IL-12 is IL12A + IL12B. Picking one would halve the molecule silently."""
    for name in ("IL-12", "IL-23", "LFA-1 (CD11a/CD18)", "Collagen I"):
        got, how = gene_symbol_for(name)
        assert got is None, f"{name} resolved to {got}"
        assert "heterodimer" in how or "heterotrimer" in how or "chains" in how


# ── HLA ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("allele,gene", [
    ("HLA-A*02:01", "HLA-A"), ("HLA-B*57:01", "HLA-B"),
    ("HLA-C*07:02", "HLA-C"), ("HLA-DRB1*04:01", "HLA-DRB1"),
])
def test_an_hla_allele_resolves_to_its_gene_and_says_so(allele, gene):
    """UniProt carries the gene, not the allele. The distinction is recorded."""
    got, how = gene_symbol_for(allele)

    assert got == gene
    assert how == "hla_gene_from_allele"


def test_a_serotype_spanning_two_genes_is_refused():
    """HLA-DQ2.5 is a DQA1/DQB1 pair, not one gene."""
    got, how = gene_symbol_for("HLA-DQ2.5")

    assert got is None
    assert "DQA1" in how and "DQB1" in how


# ── the acceptance gate ──────────────────────────────────────────────────

def test_an_entry_whose_gene_is_not_the_one_requested_is_rejected():
    """UniProt answers a gene query with related entries; this is the gate."""
    entry = {"genes": [{"geneName": {"value": "IFNGR1"}}]}

    assert not accepts(entry, "IFNG")
    assert accepts(entry, "IFNGR1")


def test_an_entry_with_no_genes_is_rejected():
    assert not accepts({}, "AR")
    assert not accepts({"genes": []}, "AR")


def test_a_symbol_among_several_gene_names_is_accepted():
    entry = {"genes": [{"geneName": {"value": "DDX58"}},
                       {"geneName": {"value": "RIGI"}}]}

    assert accepts(entry, "RIGI")


# ── symbol extraction ────────────────────────────────────────────────────

def test_a_parenthesised_symbol_is_offered_as_a_candidate():
    assert "MAP2K1" in candidate_symbols("MEK1 (MAP2K1)")


def test_a_plain_symbol_is_its_own_candidate():
    assert "TLR4" in candidate_symbols("TLR4")


def test_candidates_do_not_repeat():
    cands = candidate_symbols("ERK1/2 (MAPK3/1)")
    assert len(cands) == len(set(cands))


# ── the live library ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def resolved_entities():
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        store = EntityStore()
    rows = store._conn.execute("SELECT name, data_json FROM entities").fetchall()
    out = []
    for name, blob in rows:
        props = (json.loads(blob or "{}").get("properties") or {})
        if props.get("uniprot_id") or props.get("uniprot_status"):
            out.append((name, props))
    return out


def test_every_recorded_accession_is_well_formed(resolved_entities):
    if not resolved_entities:
        pytest.skip("no accessions in the library yet")

    bad = [f"{n}: {p['uniprot_id']}" for n, p in resolved_entities
           if p.get("uniprot_id") and not ACCESSION.match(p["uniprot_id"])]
    assert not bad, f"malformed UniProt accessions: {bad}"


def test_an_accession_always_comes_with_the_symbol_it_was_resolved_from(resolved_entities):
    """Without the symbol there is no way to re-verify the accession later."""
    missing = [n for n, p in resolved_entities
               if p.get("uniprot_id") and not p.get("gene_symbol")]
    assert not missing, f"accession with no gene_symbol: {missing}"


def test_nothing_is_both_resolved_and_refused(resolved_entities):
    both = [n for n, p in resolved_entities
            if p.get("uniprot_id") and p.get("uniprot_status") == "not_a_single_protein"]
    assert not both, f"entities carrying an accession AND a refusal: {both}"


def test_every_resolution_records_how_it_was_arrived_at(resolved_entities):
    """curated_alias is model-asserted and gene_exact_name is not; a reader has
    to be able to tell which."""
    allowed = {"curated_alias", "gene_exact_name", "hla_gene_from_allele"}
    bad = [f"{n}: {p.get('uniprot_resolved_by')}" for n, p in resolved_entities
           if p.get("uniprot_id") and p.get("uniprot_resolved_by") not in allowed]
    assert not bad, f"unrecorded or unknown provenance: {bad}"
