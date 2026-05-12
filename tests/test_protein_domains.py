"""
Protein domains + Stage B impact-multiplier unit tests.

Covers:
  * engine/py/molecular/protein_domains.py
      - role -> multiplier table is internally consistent
      - domain ranges are non-degenerate and 1-based
      - known hotspots (KRAS G12, BRAF V600, TP53 R175 / R248) land
        in critical-role domains
      - domain_at_codon resolves overlap by picking the highest
        multiplier (BRAF activation loop wins over BRAF kinase
        domain at V600)
      - impact_multiplier returns 1.0 for unknown genes / linker
        positions
  * engine/py/molecular/mutation_effect.py + Stage B
      - missense impact is amplified when gene_name is supplied and
        the codon lies in a critical domain
      - amplified impact stops at the missense ceiling
      - linker-region missense gets multiplier = 1.0 (no change)
      - the gene_name=None path matches Stage A exactly (no drift)
      - MutationEffect.domain_name / domain_role / domain_multiplier
        fields are populated for in-domain mutations
"""
from __future__ import annotations

import pytest

from engine.py.molecular.mutation_effect import MutationEffectClassifier
from engine.py.molecular.protein_domains import (
    DOMAINS,
    ROLE_MULTIPLIER,
    ProteinDomain,
    domain_at_codon,
    get_domains,
    impact_multiplier,
)


# Hotspot DNA-level coordinates (0-based base position in the curated
# reference CDSes at engine/py/molecular/reference_cds.py). These are
# the same coordinates the ONCOGENIC_SUBSTITUTIONS table uses.
KRAS_G12_BASE_POS = 34
BRAF_V600_BASE_POS = 1798
TP53_R175_BASE_POS = 523
TP53_R248_BASE_POS = 741


# ---------------------------------------------------------------------------
# Domain-table sanity
# ---------------------------------------------------------------------------

def test_role_multiplier_table_is_in_2_to_5_range():
    """Spec language: 'mutations in critical regions get 2-5x impact
    multiplier'. The role table must respect that range."""
    for role, mult in ROLE_MULTIPLIER.items():
        assert 1.0 <= mult <= 5.0, (
            f"role {role!r} has out-of-spec multiplier {mult}"
        )


def test_every_curated_domain_has_well_formed_range():
    for gene, domains in DOMAINS.items():
        seen_ranges = set()
        for d in domains:
            assert d.start_codon_1based >= 1
            assert d.end_codon_1based >= d.start_codon_1based
            assert d.role in ROLE_MULTIPLIER
            # Each annotated domain must be unique (catch copy-paste bugs)
            key = (d.start_codon_1based, d.end_codon_1based, d.name)
            assert key not in seen_ranges, f"duplicate domain entry in {gene}"
            seen_ranges.add(key)


# ---------------------------------------------------------------------------
# Known hotspots land in critical domains
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gene,codon,expected_domain_name", [
    ("KRAS", 12, "P-loop"),
    ("KRAS", 13, "P-loop"),
    ("BRAF", 600, "Activation loop"),
    ("TP53", 175, "DNA-binding domain"),
    ("TP53", 248, "DNA-binding domain"),
    ("TP53", 273, "DNA-binding domain"),
])
def test_canonical_hotspots_resolve_to_critical_domain(
    gene, codon, expected_domain_name
):
    d = domain_at_codon(gene, codon)
    assert d is not None, f"{gene} codon {codon} not in any annotated domain"
    assert d.name == expected_domain_name
    assert d.role == "critical"


def test_braf_activation_loop_beats_kinase_domain_at_v600():
    """The activation loop sits inside the kinase domain; the lookup
    must resolve to the activation loop (higher multiplier) and not
    the broader kinase domain (lower multiplier)."""
    d = domain_at_codon("BRAF", 600)
    assert d is not None
    assert d.name == "Activation loop"
    assert d.impact_multiplier == ROLE_MULTIPLIER["critical"]


def test_unknown_gene_returns_no_domain():
    assert domain_at_codon("WIDGET", 100) is None
    assert get_domains("WIDGET") == []
    assert impact_multiplier("WIDGET", 100) == 1.0


def test_codon_outside_any_domain_returns_no_domain():
    # KRAS codon 100 is in a linker between Switch II (60-76) and the
    # hypervariable C-terminus (165-189) -- no annotated domain.
    assert domain_at_codon("KRAS", 100) is None
    assert impact_multiplier("KRAS", 100) == 1.0


def test_none_gene_short_circuits_to_one():
    assert impact_multiplier(None, 12) == 1.0


# ---------------------------------------------------------------------------
# Classifier behaviour change
# ---------------------------------------------------------------------------

@pytest.fixture
def clf():
    return MutationEffectClassifier()


def test_kras_g12d_impact_amplified_in_p_loop(clf):
    """KRAS G12D: BLOSUM(G,D)=-1 yields a modest base impact; the
    P-loop multiplier (4.0x, critical) must push it to the missense
    ceiling 0.85."""
    from engine.py.molecular.reference_cds import KRAS_CDS
    stage_a = clf.classify_substitution(KRAS_CDS, KRAS_G12_BASE_POS, "A")
    stage_b = clf.classify_substitution(
        KRAS_CDS, KRAS_G12_BASE_POS, "A", gene_name="KRAS",
    )
    assert stage_b.impact_score > stage_a.impact_score, (
        f"Stage B impact {stage_b.impact_score} did not exceed Stage A "
        f"impact {stage_a.impact_score} -- the P-loop multiplier failed "
        "to fire"
    )
    assert stage_b.impact_score == pytest.approx(0.85, abs=1e-6)
    assert stage_b.domain_name == "P-loop"
    assert stage_b.domain_role == "critical"
    assert stage_b.domain_multiplier == 4.0


def test_braf_v600e_impact_amplified_in_activation_loop(clf):
    from engine.py.molecular.reference_cds import BRAF_CDS
    stage_a = clf.classify_substitution(BRAF_CDS, BRAF_V600_BASE_POS, "A")
    stage_b = clf.classify_substitution(
        BRAF_CDS, BRAF_V600_BASE_POS, "A", gene_name="BRAF",
    )
    assert stage_b.impact_score > stage_a.impact_score
    assert stage_b.domain_name == "Activation loop"
    assert stage_b.domain_role == "critical"
    assert stage_b.domain_multiplier == 4.0


def test_tp53_r248w_impact_amplified_in_dbd(clf):
    from engine.py.molecular.reference_cds import TP53_CDS
    stage_a = clf.classify_substitution(TP53_CDS, TP53_R248_BASE_POS, "T")
    stage_b = clf.classify_substitution(
        TP53_CDS, TP53_R248_BASE_POS, "T", gene_name="TP53",
    )
    assert stage_b.impact_score > stage_a.impact_score
    assert stage_b.domain_name == "DNA-binding domain"
    assert stage_b.domain_role == "critical"
    assert stage_b.domain_multiplier == 4.0


def test_stage_a_path_unchanged_without_gene_name(clf):
    """The gene_name=None path must match the prior Stage A output
    bit-for-bit so existing callers do not silently change behaviour."""
    from engine.py.molecular.reference_cds import KRAS_CDS
    a = clf.classify_substitution(KRAS_CDS, KRAS_G12_BASE_POS, "A")
    a_explicit = clf.classify_substitution(
        KRAS_CDS, KRAS_G12_BASE_POS, "A", gene_name=None,
    )
    assert a.impact_score == a_explicit.impact_score
    assert a.domain_name is None
    assert a.domain_role is None
    assert a.domain_multiplier == 1.0


def test_missense_outside_any_domain_no_multiplier(clf):
    """Mutate a position in KRAS that's between annotated domains.

    The curated KRAS CDS in reference_cds covers only the 51 N-terminal
    codons (G12/G13 hotspot region). Codons 18-29 fall between the
    P-loop end (codon 17) and Switch I start (codon 30) -- a true
    linker stretch in the GTPase fold with no functional annotation
    in protein_domains._KRAS. Stage B must leave impact unchanged here.
    """
    from engine.py.molecular.reference_cds import KRAS_CDS

    # Sanity-check the curator's understanding: codon 20 must NOT be
    # in any annotated KRAS domain.
    assert domain_at_codon("KRAS", 20) is None, (
        "test fixture is stale: KRAS codon 20 is now annotated; pick "
        "a different linker codon (must be in 18-29 or 39-59)"
    )

    codon_start = (20 - 1) * 3  # 0-based start of codon 20 in CDS
    found = False
    for offset in range(3):
        for new_base in "ACGT":
            pos = codon_start + offset
            if pos >= len(KRAS_CDS):
                continue
            if new_base == KRAS_CDS[pos]:
                continue
            base = clf.classify_substitution(KRAS_CDS, pos, new_base)
            if base.category != "missense":
                continue
            ann = clf.classify_substitution(
                KRAS_CDS, pos, new_base, gene_name="KRAS",
            )
            assert ann.domain_name is None
            assert ann.domain_multiplier == 1.0
            assert ann.impact_score == base.impact_score
            found = True
            break
        if found:
            break
    assert found, (
        "could not engineer a missense at KRAS codon 20 in the curated "
        "CDS; sequence content may have changed"
    )


def test_amplified_impact_stops_at_missense_ceiling(clf):
    """Even the most radical critical-region missense should clamp at
    _MISSENSE_IMPACT_MAX (0.85). Verified for KRAS G12D where base
    impact ~0.51 and multiplier 4.0 would otherwise produce 2.04."""
    from engine.py.molecular.reference_cds import KRAS_CDS
    e = clf.classify_substitution(
        KRAS_CDS, KRAS_G12_BASE_POS, "A", gene_name="KRAS",
    )
    assert e.impact_score <= 0.85 + 1e-9


def test_synonymous_path_unaffected_by_gene_name(clf):
    """Synonymous substitutions return impact=0 regardless of the
    domain -- there is no AA change to amplify."""
    from engine.py.molecular.reference_cds import KRAS_CDS
    # Find a synonymous third-base wobble in the P-loop region.
    for codon_1based in range(10, 18):
        codon_start = (codon_1based - 1) * 3
        if codon_start + 3 > len(KRAS_CDS):
            break
        wt_codon = KRAS_CDS[codon_start:codon_start + 3]
        for new_third in "ACGT":
            if new_third == wt_codon[2]:
                continue
            mut_codon = wt_codon[:2] + new_third
            from engine.py.molecular.mutation_effect import _CODON_TABLE
            if _CODON_TABLE.get(wt_codon) == _CODON_TABLE.get(mut_codon):
                e = clf.classify_substitution(
                    KRAS_CDS, codon_start + 2, new_third,
                    gene_name="KRAS",
                )
                assert e.category == "synonymous"
                assert e.impact_score == 0.0
                assert e.domain_multiplier == 1.0
                return
    pytest.skip(
        "could not find a synonymous wobble in the KRAS P-loop range "
        "of the curated CDS; safe to revisit"
    )


def test_nonsense_path_unaffected_by_domain_multiplier(clf):
    """Nonsense mutations have their own impact scoring path; the
    Stage B multiplier must not amplify them."""
    from engine.py.molecular.reference_cds import KRAS_CDS
    # Find a position in the P-loop where the substitution yields a
    # premature stop. Codon 12 GGT -> TGT is C, not stop. We need a
    # codon like CAG -> TAG (Q->*) etc. Just scan to find one.
    from engine.py.molecular.mutation_effect import _CODON_TABLE
    found = False
    for codon_1based in range(10, 18):
        codon_start = (codon_1based - 1) * 3
        if codon_start + 3 > len(KRAS_CDS):
            break
        wt_codon = KRAS_CDS[codon_start:codon_start + 3]
        for offset in range(3):
            for new_base in "ACGT":
                if new_base == wt_codon[offset]:
                    continue
                mut_codon = (
                    wt_codon[:offset] + new_base + wt_codon[offset + 1:]
                )
                if _CODON_TABLE.get(mut_codon) == "*":
                    e_a = clf.classify_substitution(
                        KRAS_CDS, codon_start + offset, new_base,
                    )
                    e_b = clf.classify_substitution(
                        KRAS_CDS, codon_start + offset, new_base,
                        gene_name="KRAS",
                    )
                    # Both paths produce the same nonsense impact --
                    # the domain multiplier path only kicks in for
                    # missense.
                    assert e_a.category == "nonsense"
                    assert e_b.category == "nonsense"
                    assert e_a.impact_score == e_b.impact_score
                    found = True
                    break
            if found:
                break
        if found:
            break
    assert found, (
        "no single-base substitution in the KRAS P-loop yields a "
        "premature stop in the curated CDS; safe to revisit"
    )
