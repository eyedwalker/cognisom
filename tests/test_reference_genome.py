"""
Tests for engine.py.molecular.reference_genome.ReferenceGenome.

Patent-evidence: the immutability and shared-pointer semantics of
ReferenceGenome are the precondition for the per-cell-delta memory
architecture claim. These tests assert the invariants that make the
sharing safe.
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest

from engine.py.molecular.reference_genome import (
    GeneMetadata,
    ReferenceGenome,
    SubstitutionDelta,
    build_default_reference_genome,
)


# --- SubstitutionDelta --------------------------------------------------------

def test_delta_construction_normalizes_and_validates():
    d = SubstitutionDelta(gene_name="KRAS", position=34, new_base="A")
    assert d.new_base == "A"
    assert d.position == 34
    assert d.gene_name == "KRAS"


def test_delta_rejects_multi_char_base():
    with pytest.raises(ValueError):
        SubstitutionDelta("KRAS", 0, "AA")


def test_delta_rejects_non_canonical_base():
    with pytest.raises(ValueError):
        SubstitutionDelta("KRAS", 0, "X")


def test_delta_rejects_negative_position():
    with pytest.raises(ValueError):
        SubstitutionDelta("KRAS", -1, "A")


def test_delta_is_frozen_hashable():
    """Frozen dataclass is hashable so deltas can be set members."""
    d1 = SubstitutionDelta("KRAS", 34, "A", "uuid-1")
    d2 = SubstitutionDelta("KRAS", 34, "A", "uuid-1")
    assert hash(d1) == hash(d2)
    assert d1 == d2


# --- ReferenceGenome construction --------------------------------------------

def test_empty_genome():
    g = ReferenceGenome()
    assert len(g) == 0
    assert g.total_bases() == 0
    assert not g.is_frozen


def test_add_gene_normalizes_to_uppercase_and_treats_U_as_T():
    g = ReferenceGenome()
    g.add_gene("FOO", "atcuGAGu")
    assert g.get_reference_sequence("FOO") == "ATCTGAGT"


def test_add_gene_rejects_empty():
    g = ReferenceGenome()
    with pytest.raises(ValueError):
        g.add_gene("FOO", "")


def test_add_gene_rejects_non_acgt():
    g = ReferenceGenome()
    with pytest.raises(ValueError):
        g.add_gene("FOO", "ATCGN")


def test_add_gene_rejects_duplicate():
    g = ReferenceGenome()
    g.add_gene("FOO", "ATCG")
    with pytest.raises(ValueError):
        g.add_gene("FOO", "GGGG")


def test_freeze_blocks_add_gene():
    g = ReferenceGenome()
    g.add_gene("FOO", "ATCG")
    g.freeze()
    assert g.is_frozen
    with pytest.raises(RuntimeError):
        g.add_gene("BAR", "GGGG")


def test_freeze_returns_self_for_chaining():
    g = ReferenceGenome().add_gene  # not chained
    # Chained pattern test
    h = ReferenceGenome()
    h.add_gene("X", "A")
    chained = h.freeze()
    assert chained is h


# --- ReferenceGenome read access ---------------------------------------------

def test_get_reference_base_returns_single_char():
    g = ReferenceGenome()
    g.add_gene("FOO", "ATCG")
    assert g.get_reference_base("FOO", 0) == "A"
    assert g.get_reference_base("FOO", 3) == "G"


def test_get_reference_base_out_of_range_raises():
    g = ReferenceGenome()
    g.add_gene("FOO", "ATCG")
    with pytest.raises(IndexError):
        g.get_reference_base("FOO", 4)
    with pytest.raises(IndexError):
        g.get_reference_base("FOO", -1)


def test_get_reference_base_unknown_gene_raises():
    g = ReferenceGenome()
    with pytest.raises(KeyError):
        g.get_reference_base("MISSING", 0)


def test_length_and_total_bases():
    g = ReferenceGenome()
    g.add_gene("A", "AAA")
    g.add_gene("B", "GGGGG")
    assert g.length("A") == 3
    assert g.length("B") == 5
    assert g.total_bases() == 8


# --- Default genome from stock library --------------------------------------

def test_default_genome_has_KRAS_TP53_BRAF():
    g = build_default_reference_genome()
    assert g.is_frozen
    assert set(g.gene_names()) == {"KRAS", "TP53", "BRAF"}
    # Spot check hotspots survive the trip through ReferenceGenome
    kras = g.get_reference_sequence("KRAS")
    assert kras[33:36] == "GGT"  # codon 12
    tp53 = g.get_reference_sequence("TP53")
    assert tp53[522:525] == "CGC"  # codon 175
    braf = g.get_reference_sequence("BRAF")
    assert braf[1797:1800] == "GTG"  # codon 600


def test_default_genome_metadata():
    g = build_default_reference_genome()
    assert g.metadata("KRAS").gene_type == "protein_coding"
    assert g.metadata("TP53").is_baseline_tumor_suppressor is True
    assert g.metadata("BRAF").is_baseline_oncogene is False
