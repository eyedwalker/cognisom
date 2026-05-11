"""
Integration tests: Gene + MutationEffectClassifier (Sprint 1).

Patent-evidence layer: demonstrates that a mutation introduced through the
Gene API now carries a populated MutationEffect record. This is the
enablement evidence for the Upgrade-3-Stage-A patent claim: the simulator
derives a numerical impact score from an arbitrary substitution, not from
a hardcoded table.
"""

import sys
import warnings
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest

from engine.py.molecular.nucleic_acids import Gene
from engine.py.molecular.mutation_effect import MutationEffectClassifier


# Real KRAS CDS prefix (correct biology, codon 12 = GGT)
REAL_KRAS_CDS = (
    "ATGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGTGCCTTGACGATACAGC"
    "TAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAGGATTCCTACAGGAAGCAAGTAG"
)


@pytest.fixture
def clf():
    return MutationEffectClassifier()


def test_arbitrary_substitution_gets_classified(clf):
    """Pick a random off-table position and verify Mutation.effect is populated."""
    gene = Gene("KRAS", REAL_KRAS_CDS)
    # Position 20 is in codon 7 (KRAS LRys/Leu region). Pick a real change.
    mutation = gene.introduce_substitution(
        position=20, new_base="A", classifier=clf
    )
    assert mutation is not None
    assert mutation.effect is not None
    assert mutation.effect.category in {
        "synonymous", "missense", "nonsense", "stop_loss",
        "start_loss", "outside_coding",
    }
    assert 0.0 <= mutation.effect.impact_score <= 1.0


def test_named_oncogenic_mutation_with_real_kras_produces_correct_aa_change(clf):
    """G12D on the REAL KRAS CDS should classify as missense G12D with
    BLOSUM = -1. No warning should be emitted."""
    gene = Gene("KRAS", REAL_KRAS_CDS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn warnings into errors for this test
        mutation = gene.introduce_oncogenic_mutation("G12D", classifier=clf)
    assert mutation is not None
    assert mutation.effect is not None
    assert mutation.effect.category == "missense"
    assert mutation.effect.aa_change == "G12D"
    assert mutation.effect.blosum62_score == -1
    assert mutation.effect.impact_score > 0.4
    assert mutation.oncogenic is True
    assert gene.is_oncogene is True


def test_demo_kras_mismatch_warns_does_not_crash(clf):
    """The synthetic demo KRAS sequence has a premature stop at codon 3,
    so its codon 12 is TGG (Trp) not GGT (Gly). Introducing 'G12D' on this
    sequence should WARN that the AA change doesn't match the name, but
    must not crash."""
    demo_kras = (
        "ATGGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGTGCCTTGACGATACAGC"
        "TAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAGGATTCCTACAGGAAGCAAGTAGTA"
    )
    gene = Gene("KRAS", demo_kras)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mutation = gene.introduce_oncogenic_mutation("G12D", classifier=clf)
    # A warning should have been emitted
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 1
    assert "G12D" in str(runtime_warnings[0].message)
    assert mutation is not None
    assert mutation.effect is not None


def test_no_classifier_no_effect_preserves_legacy(clf):
    """Calling introduce_oncogenic_mutation WITHOUT a classifier preserves
    legacy behavior: the mutation is introduced but no effect is attached."""
    gene = Gene("KRAS", REAL_KRAS_CDS)
    mutation = gene.introduce_oncogenic_mutation("G12D")  # no classifier
    assert mutation is not None
    assert mutation.effect is None  # not classified
    assert mutation.oncogenic is True  # legacy flag still set


def test_synonymous_substitution_does_not_promote_oncogene(clf):
    """A synonymous mutation should NOT set gene.is_oncogene = True."""
    gene = Gene("KRAS", REAL_KRAS_CDS)
    # Codon 4 of REAL_KRAS_CDS: positions 9-11. Find a silent third-position
    # substitution. REAL_KRAS_CDS[9:12] = "TAT" (Tyr). TAT -> TAC = silent.
    assert REAL_KRAS_CDS[9:12] == "TAT"
    mutation = gene.introduce_substitution(
        position=11, new_base="C", classifier=clf
    )
    assert mutation.effect.category == "synonymous"
    assert mutation.effect.impact_score == 0.0
    assert mutation.oncogenic is False
    assert gene.is_oncogene is False


def test_classifier_is_optional_for_introduce_substitution(clf):
    """introduce_substitution without classifier still works (no effect attached)."""
    gene = Gene("KRAS", REAL_KRAS_CDS)
    mutation = gene.introduce_substitution(position=20, new_base="A")  # no classifier
    assert mutation is not None
    assert mutation.effect is None
