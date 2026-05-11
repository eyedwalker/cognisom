"""
Regression tests for the gene-library reference sequences in
modules/molecular_module.MolecularModule.

Reason for existence: the prior KRAS reference in the demo gene library
had an off-by-one base error (extra G at position 3) that produced a
premature stop at codon 3, truncating translation to 2 amino acids. The
fix was logged in DECISIONS.md on 2026-05-11. These tests assert that the
relevant biological invariants hold so the bug cannot reappear silently.

TP53 and BRAF placeholder sequences are NOT asserted to reach their
canonical oncogenic loci (R175H at codon 175 of TP53, V600E at codon 600
of BRAF) - those need real CDSes, deferred to a later sprint.
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import warnings

import pytest

from modules.molecular_module import MolecularModule
from engine.py.molecular.mutation_effect import MutationEffectClassifier


@pytest.fixture
def module():
    m = MolecularModule(config={})
    m.initialize()
    return m


def test_kras_codon_12_is_GGT(module):
    """Codon 12 of the KRAS reference must be GGT (Gly)."""
    seq = module.genes['KRAS'].dna.sequence
    assert seq[33:36] == "GGT", f"KRAS codon 12 is {seq[33:36]!r}, must be GGT"


def test_kras_codon_13_is_GGC(module):
    """Codon 13 of the KRAS reference must be GGC (Gly)."""
    seq = module.genes['KRAS'].dna.sequence
    assert seq[36:39] == "GGC", f"KRAS codon 13 is {seq[36:39]!r}, must be GGC"


def test_kras_no_premature_stop_before_codon_15(module):
    """The KRAS reference must not contain a stop codon (TAA/TAG/TGA) in
    frame before codon 15. This catches the prior bug where ATGGAC*TGA*
    put a stop at codon 3."""
    seq = module.genes['KRAS'].dna.sequence
    for codon_index in range(15):
        codon = seq[codon_index * 3 : codon_index * 3 + 3]
        assert codon not in ("TAA", "TAG", "TGA"), (
            f"KRAS has premature stop {codon} at codon {codon_index+1}; "
            f"reference is broken"
        )


def test_kras_translates_to_real_protein_prefix(module):
    """The KRAS reference, when transcribed and translated, must produce
    the canonical KRAS N-terminal protein sequence MTEYKLVVVG (codons 1-10)."""
    gene = module.genes['KRAS']
    mrna = gene.transcribe(rate_modifier=10.0)  # boost transcription rate so we get a transcript
    # If stochastic transcription failed this iteration, try a deterministic transcribe
    if mrna is None:
        mrna = gene.dna.transcribe()
    protein = mrna.translate()
    assert protein.startswith("MTEYKLVVVG"), (
        f"KRAS protein prefix is {protein[:20]!r}; expected MTEYKLVVVG..."
    )


def test_kras_g12d_via_named_mutation_does_not_warn(module):
    """Calling introduce_oncogenic_mutation('G12D') on KRAS with the
    classifier should NOT emit any warning - the resulting aa_change must
    match 'G12D' exactly."""
    clf = MutationEffectClassifier()
    gene = module.genes['KRAS']
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a test failure
        mutation = gene.introduce_oncogenic_mutation("G12D", classifier=clf)
    assert mutation is not None
    assert mutation.effect is not None
    assert mutation.effect.aa_change == "G12D"
    assert mutation.effect.category == "missense"


def test_kras_g12v_via_named_mutation_does_not_warn():
    """G12V should also produce the correct AA change with no warning.
    Fresh module to avoid the gene already being mutated from prior tests."""
    m = MolecularModule(config={})
    m.initialize()
    clf = MutationEffectClassifier()
    gene = m.genes['KRAS']
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mutation = gene.introduce_oncogenic_mutation("G12V", classifier=clf)
    assert mutation.effect.aa_change == "G12V"
    assert mutation.effect.category == "missense"


def test_kras_g13d_via_named_mutation_does_not_warn():
    """G13D should produce the correct AA change with no warning."""
    m = MolecularModule(config={})
    m.initialize()
    clf = MutationEffectClassifier()
    gene = m.genes['KRAS']
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mutation = gene.introduce_oncogenic_mutation("G13D", classifier=clf)
    assert mutation.effect.aa_change == "G13D"
    assert mutation.effect.category == "missense"
