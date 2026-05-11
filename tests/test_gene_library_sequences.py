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


# --- TP53 --------------------------------------------------------------------

def test_tp53_codon_175_is_CGC(module):
    seq = module.genes['TP53'].dna.sequence
    assert seq[522:525] == "CGC", f"TP53 codon 175 is {seq[522:525]!r}, must be CGC"


def test_tp53_codon_248_is_CGG(module):
    seq = module.genes['TP53'].dna.sequence
    assert seq[741:744] == "CGG", f"TP53 codon 248 is {seq[741:744]!r}, must be CGG"


def test_tp53_no_premature_stop(module):
    """TP53 must not contain in-frame stop codon before codon 393."""
    seq = module.genes['TP53'].dna.sequence
    for codon_index in range(393):
        codon = seq[codon_index * 3 : codon_index * 3 + 3]
        assert codon not in ("TAA", "TAG", "TGA"), (
            f"TP53 has premature stop {codon} at codon {codon_index+1}"
        )


def test_tp53_r175h_via_named_mutation_does_not_warn():
    """R175H must classify as missense R175H with no warning."""
    m = MolecularModule(config={})
    m.initialize()
    clf = MutationEffectClassifier()
    gene = m.genes['TP53']
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mutation = gene.introduce_oncogenic_mutation("R175H", classifier=clf)
    assert mutation.effect.aa_change == "R175H"
    assert mutation.effect.category == "missense"
    assert mutation.effect.blosum62_score == 0  # R->H is BLOSUM 0


def test_tp53_r248w_via_named_mutation_does_not_warn():
    """R248W must classify as missense R248W with no warning."""
    m = MolecularModule(config={})
    m.initialize()
    clf = MutationEffectClassifier()
    gene = m.genes['TP53']
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mutation = gene.introduce_oncogenic_mutation("R248W", classifier=clf)
    assert mutation.effect.aa_change == "R248W"
    assert mutation.effect.category == "missense"
    assert mutation.effect.blosum62_score == -3  # R->W is BLOSUM -3 (radical)


# --- BRAF --------------------------------------------------------------------

def test_braf_codon_600_is_GTG(module):
    seq = module.genes['BRAF'].dna.sequence
    assert seq[1797:1800] == "GTG", f"BRAF codon 600 is {seq[1797:1800]!r}, must be GTG"


def test_braf_no_premature_stop(module):
    """BRAF must not contain in-frame stop codon before codon 766."""
    seq = module.genes['BRAF'].dna.sequence
    for codon_index in range(766):
        codon = seq[codon_index * 3 : codon_index * 3 + 3]
        assert codon not in ("TAA", "TAG", "TGA"), (
            f"BRAF has premature stop {codon} at codon {codon_index+1}"
        )


def test_braf_v600e_via_named_mutation_does_not_warn():
    """V600E must classify as missense V600E with no warning."""
    m = MolecularModule(config={})
    m.initialize()
    clf = MutationEffectClassifier()
    gene = m.genes['BRAF']
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mutation = gene.introduce_oncogenic_mutation("V600E", classifier=clf)
    assert mutation.effect.aa_change == "V600E"
    assert mutation.effect.category == "missense"
    assert mutation.effect.blosum62_score == -2


# --- Cross-cutting ----------------------------------------------------------

def test_all_six_oncogenic_mutations_produce_correct_aa_change():
    """Smoke test: every entry in the ONCOGENIC_SUBSTITUTIONS table must
    classify with no warnings as the named AA change. This is the single
    most important regression test for the patent-claim integrity."""
    cases = [
        ('KRAS', 'G12D'), ('KRAS', 'G12V'), ('KRAS', 'G13D'),
        ('TP53', 'R175H'), ('TP53', 'R248W'),
        ('BRAF', 'V600E'),
    ]
    clf = MutationEffectClassifier()
    for gene_name, mut_name in cases:
        m = MolecularModule(config={})
        m.initialize()
        gene = m.genes[gene_name]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mutation = gene.introduce_oncogenic_mutation(mut_name, classifier=clf)
        assert mutation is not None, f"{gene_name} {mut_name}: returned None"
        assert mutation.effect.aa_change == mut_name, (
            f"{gene_name} {mut_name}: classifier produced {mutation.effect.aa_change}"
        )
