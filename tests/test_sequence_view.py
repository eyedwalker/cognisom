"""
Tests for engine.py.molecular.sequence_view.CellGenomeView.

The most important test in this module is
test_view_equivalence_to_naive_materialization: for any sequence of
substitutions, the view's per-base reads must equal the result of
deep-copying the reference and applying the same substitutions in
order. This is the correctness invariant the patent claim depends on.

Patent-evidence is also captured by tests of fork() semantics: daughter
views inherit deltas but later divergence does not contaminate the
parent. This is the cell-division operation the patent claim names.
"""

import random
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest

from engine.py.molecular.reference_genome import (
    ReferenceGenome,
    SubstitutionDelta,
    build_default_reference_genome,
)
from engine.py.molecular.sequence_view import (
    CellGenomeView,
    materialize_naive,
)


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def genome():
    return build_default_reference_genome()


@pytest.fixture
def view(genome):
    return CellGenomeView(genome)


# --- construction guards ----------------------------------------------------

def test_view_requires_frozen_genome():
    g = ReferenceGenome()
    g.add_gene("FOO", "ATCG")
    # Not frozen yet
    with pytest.raises(RuntimeError, match="frozen"):
        CellGenomeView(g)


# --- baseline reads ---------------------------------------------------------

def test_base_at_returns_reference_when_no_deltas(genome, view):
    """Empty view returns reference bases."""
    for gene in genome.gene_names():
        for pos in range(0, genome.length(gene), 200):
            assert view.base_at(gene, pos) == genome.get_reference_base(gene, pos)


def test_codon_at_returns_reference_codons_when_no_deltas(genome, view):
    """Empty view returns reference codons."""
    # KRAS codon 12 is GGT, codon 13 is GGC
    assert view.codon_at("KRAS", 11) == "GGT"
    assert view.codon_at("KRAS", 12) == "GGC"
    # TP53 codon 175 is CGC, codon 248 is CGG
    assert view.codon_at("TP53", 174) == "CGC"
    assert view.codon_at("TP53", 247) == "CGG"
    # BRAF codon 600 is GTG
    assert view.codon_at("BRAF", 599) == "GTG"


def test_materialize_with_no_deltas_returns_reference(genome, view):
    """Empty view materialize gives back the reference."""
    for gene in genome.gene_names():
        assert view.materialize(gene) == genome.get_reference_sequence(gene)


# --- writes ----------------------------------------------------------------

def test_add_substitution_reflected_in_base_at(genome, view):
    """A delta at position p must be visible at base_at(g, p)."""
    view.add_substitution("KRAS", 34, "A", mutation_id="G12D")
    assert view.base_at("KRAS", 34) == "A"
    # Adjacent positions still come from the reference
    assert view.base_at("KRAS", 33) == genome.get_reference_base("KRAS", 33)
    assert view.base_at("KRAS", 35) == genome.get_reference_base("KRAS", 35)


def test_add_substitution_reflected_in_codon_at(genome, view):
    """KRAS G12D: GGT -> GAT. codon_at(11) should yield GAT after the
    substitution at position 34."""
    view.add_substitution("KRAS", 34, "A", mutation_id="G12D")
    assert view.codon_at("KRAS", 11) == "GAT"


def test_add_substitution_overwrites_existing(genome, view):
    """Second add at same position replaces the first."""
    view.add_substitution("KRAS", 34, "A", mutation_id="first")
    view.add_substitution("KRAS", 34, "T", mutation_id="second")
    assert view.base_at("KRAS", 34) == "T"
    # But the log records both
    assert view.n_deltas() == 2


def test_add_substitution_rejects_out_of_range(genome, view):
    with pytest.raises(IndexError):
        view.add_substitution("KRAS", 999999, "A")


def test_add_substitution_rejects_unknown_gene(genome, view):
    with pytest.raises(KeyError):
        view.add_substitution("NOT_A_GENE", 0, "A")


# --- materialize -----------------------------------------------------------

def test_materialize_applies_all_deltas(genome, view):
    view.add_substitution("KRAS", 34, "A", "G12D")
    view.add_substitution("KRAS", 35, "C", "ad-hoc")
    full = view.materialize("KRAS")
    ref = genome.get_reference_sequence("KRAS")
    assert full[0:34] == ref[0:34]
    assert full[34] == "A"
    assert full[35] == "C"
    assert full[36:] == ref[36:]


def test_materialize_fast_path_returns_reference_for_clean_gene(genome, view):
    """When no deltas exist for a gene, materialize returns the reference
    string identity (or equal contents). Note: we don't test identity
    because the implementation is allowed to return a copy; equality is
    the contract."""
    view.add_substitution("KRAS", 34, "A")
    # KRAS has a delta; TP53 does not
    assert view.materialize("TP53") == genome.get_reference_sequence("TP53")


# --- iter_codons -----------------------------------------------------------

def test_iter_codons_applies_deltas(genome, view):
    view.add_substitution("KRAS", 34, "A", "G12D")
    codons = list(view.iter_codons("KRAS", start=0, end=51))
    assert codons[11] == "GAT"  # codon 12 with G12D applied


def test_iter_codons_does_not_overrun(genome, view):
    """iter_codons must not yield partial trailing codons."""
    codons = list(view.iter_codons("KRAS"))
    kras_len = genome.length("KRAS")
    # KRAS length must be divisible by 3 (it is: 153 = 51*3)
    assert len(codons) == kras_len // 3


# --- equivalence to naive materialization ---------------------------------

def _random_delta(rng, genome):
    """Pick a uniformly random (gene, position, new_base) triple, avoiding
    same-base no-ops."""
    gene = rng.choice(list(genome.gene_names()))
    length = genome.length(gene)
    pos = rng.randrange(length)
    ref_base = genome.get_reference_base(gene, pos)
    new = ref_base
    while new == ref_base:
        new = rng.choice("ACGT")
    return gene, pos, new


@pytest.mark.parametrize("seed", [11, 42, 256, 1024, 2026])
def test_view_equivalence_to_naive_materialization(genome, seed):
    """For 50 random substitutions, the view's base_at and the naive
    deep-copy+mutate approach must give identical answers at every position
    of every gene. This is the load-bearing correctness invariant."""
    rng = random.Random(seed)
    view = CellGenomeView(genome)
    applied = []

    for _ in range(50):
        gene, pos, new_base = _random_delta(rng, genome)
        d = SubstitutionDelta(gene, pos, new_base, mutation_id=f"r{len(applied)}")
        view.add_substitution(d.gene_name, d.position, d.new_base, d.mutation_id)
        applied.append(d)

    for gene in genome.gene_names():
        naive = materialize_naive(genome, gene, applied)
        # Compare via view's materialize()
        assert view.materialize(gene) == naive, f"materialize mismatch for {gene}"
        # And via base_at across the full sequence
        for i in range(genome.length(gene)):
            assert view.base_at(gene, i) == naive[i], (
                f"base_at mismatch at {gene}:{i}"
            )


# --- fork() semantics -----------------------------------------------------

def test_fork_creates_independent_view(genome, view):
    """After fork, mutations on parent do not affect daughter and vice versa."""
    view.add_substitution("KRAS", 34, "A", "parent-pre-fork")
    daughter = view.fork()
    # Daughter inherits parent's deltas
    assert daughter.base_at("KRAS", 34) == "A"
    # Parent mutates further
    view.add_substitution("KRAS", 100, "C", "parent-post-fork")
    assert view.base_at("KRAS", 100) == "C"
    assert daughter.base_at("KRAS", 100) == genome.get_reference_base("KRAS", 100)
    # Daughter mutates further
    daughter.add_substitution("TP53", 50, "T", "daughter-post-fork")
    assert daughter.base_at("TP53", 50) == "T"
    assert view.base_at("TP53", 50) == genome.get_reference_base("TP53", 50)


def test_fork_shares_reference_genome_by_identity(genome, view):
    """The key memory-architecture invariant: parent and daughter share the
    same ReferenceGenome instance (no copy)."""
    daughter = view.fork()
    assert daughter.reference is view.reference is genome


def test_fork_copies_delta_log_in_order(genome, view):
    view.add_substitution("KRAS", 34, "A", "first")
    view.add_substitution("KRAS", 100, "C", "second")
    daughter = view.fork()
    parent_log = view.deltas()
    daughter_log = daughter.deltas()
    assert daughter_log == parent_log


def test_repeated_fork_chain(genome, view):
    """Three generations of forks each carry the cumulative deltas of
    their ancestors but diverge independently after their own fork."""
    view.add_substitution("KRAS", 0, "T", "gen1")
    g2 = view.fork()
    g2.add_substitution("KRAS", 1, "G", "gen2")
    g3 = g2.fork()
    g3.add_substitution("KRAS", 2, "A", "gen3")

    # Gen 3 carries all three deltas
    assert g3.base_at("KRAS", 0) == "T"
    assert g3.base_at("KRAS", 1) == "G"
    assert g3.base_at("KRAS", 2) == "A"
    # Gen 2 has gen1 and gen2 but not gen3
    assert g2.base_at("KRAS", 0) == "T"
    assert g2.base_at("KRAS", 1) == "G"
    assert g2.base_at("KRAS", 2) != "A"
    # Gen 1 (the original view) has only its own delta
    assert view.base_at("KRAS", 0) == "T"
    assert view.base_at("KRAS", 1) != "G"


# --- provenance ----------------------------------------------------------

def test_deltas_returns_chronological_log(genome, view):
    view.add_substitution("KRAS", 34, "A", "first")
    view.add_substitution("TP53", 100, "C", "second")
    view.add_substitution("BRAF", 200, "T", "third")
    log = view.deltas()
    assert [d.mutation_id for d in log] == ["first", "second", "third"]


def test_deltas_for_gene_filters_correctly(genome, view):
    view.add_substitution("KRAS", 34, "A", "k1")
    view.add_substitution("TP53", 100, "C", "t1")
    view.add_substitution("KRAS", 100, "T", "k2")
    kras_deltas = view.deltas_for_gene("KRAS")
    assert [d.mutation_id for d in kras_deltas] == ["k1", "k2"]
