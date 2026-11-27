"""Each neoantigen record describes the mutation it came from.

The predictor scores in two passes: one walks the variants and collects
peptide windows, the other walks those windows and builds records. The
variant was not carried between them, so the record builder read the
first pass's leaked loop variable and every record in a run reported the
LAST variant's allele frequency. Measured on 2026-11-27 with two genes
at 0.80 and 0.05: both genes' records said 0.05.

That field is also the ranking tie-break, so the defect reordered
candidates as well as mislabelling them, and it would have fed the wrong
frequency into every cancer-cell fraction.
"""

import pytest

from cognisom.genomics.cancer_cell_fraction import (
    CLONALITY_POSSIBLY_CLONAL,
    CLONALITY_SUBCLONAL,
    CopyState,
    SampleCopyProfile,
)
from cognisom.genomics.gene_protein_mapper import ProteinInfo
from cognisom.genomics.neoantigen_predictor import NeoantigenPredictor
from cognisom.genomics.vcf_parser import Variant

SEQ_A = "MKVLAAGIVGLNLGGSAYMKFDTRAWQPVLYHCNDEFGHIKLMNPQRSTVWYACDEFGHIKL"
SEQ_B = "MTTYAQDLFGKRSAVVNPQWECMLGHIKFDTRAWQPVLYHCNDEFGHIKLMNPQRSTVWYAC"

#: Deliberately far apart, so a leak is unmistakable.
AF_A = 0.80
AF_B = 0.05

ALLELES = ["HLA-A*02:01"]
LENGTHS = [9]


def _variant(gene, pos, ref, alt, af, alt_count=None, depth=None):
    v = Variant(chrom="1", pos=1000 + pos, id=".", ref="C", alt="T",
                qual=100.0, filter_status="PASS")
    v.gene = gene
    v.consequence = "missense_variant"
    v.protein_change = f"p.{ref}{pos}{alt}"
    v.tumor_af = af
    v.tumor_alt_count = alt_count
    v.tumor_depth = depth
    return v


@pytest.fixture(scope="module")
def inputs():
    variants = [
        _variant("GENEA", 30, "L", "P", AF_A),
        _variant("GENEB", 30, "D", "N", AF_B),
    ]
    proteins = {
        "GENEA": ProteinInfo(gene="GENEA", uniprot_id="X1", protein_name="A",
                             sequence=SEQ_A, length=len(SEQ_A)),
        "GENEB": ProteinInfo(gene="GENEB", uniprot_id="X2", protein_name="B",
                             sequence=SEQ_B, length=len(SEQ_B)),
    }
    return variants, proteins


@pytest.fixture(scope="module")
def plain(inputs):
    """A run with no sample profile: the default path."""
    variants, proteins = inputs
    p = NeoantigenPredictor()
    neos = p.predict(variants, proteins, ALLELES, peptide_lengths=LENGTHS,
                     max_neoantigens=500)
    return p, neos


def diploid(gene):
    return CopyState(total_tumour_copies=2.0, source=f"assumed diploid, {gene}")


@pytest.fixture(scope="module")
def both_genes_typed(inputs):
    variants, proteins = inputs
    p = NeoantigenPredictor()
    sample = SampleCopyProfile(
        purity=1.0,
        purity_source="set to 1.0 by this test, not a measurement",
        copy_states={"GENEA": diploid("GENEA"), "GENEB": diploid("GENEB")},
    )
    neos = p.predict(variants, proteins, ALLELES, peptide_lengths=LENGTHS,
                     max_neoantigens=500, sample=sample)
    return p, neos


def by_gene(neos, attr):
    out = {}
    for n in neos:
        out.setdefault(n.source_gene, set()).add(getattr(n, attr))
    return out


# ── the regression ────────────────────────────────────────────────

def test_a_record_reports_its_own_variants_allele_frequency(plain):
    _, neos = plain
    freqs = by_gene(neos, "variant_allele_frequency")
    assert freqs["GENEA"] == {AF_A}
    assert freqs["GENEB"] == {AF_B}


def test_the_two_genes_both_produced_records(plain):
    """Non-vacuity: with records from only one gene a leak is invisible."""
    _, neos = plain
    genes = {n.source_gene for n in neos}
    assert genes == {"GENEA", "GENEB"}
    assert len(neos) > 2


def test_the_two_allele_frequencies_really_differ():
    """Non-vacuity: equal frequencies would make the test above pass always."""
    assert AF_A != AF_B


# ── without a sample profile, nothing is claimed ──────────────────

def test_no_sample_profile_means_no_fraction_and_no_clonality(plain):
    _, neos = plain
    assert all(n.cancer_cell_fraction is None for n in neos)
    assert all(n.clonality is None for n in neos)


def test_the_diagnostics_say_why_no_fraction_was_computed(plain):
    p, _ = plain
    assert p.diagnostics.variants_with_a_cell_fraction == 0
    assert "no SampleCopyProfile" in p.diagnostics.cell_fraction_unavailable_reason


# ── with one, every record carries the interval ───────────────────

def test_every_record_gets_a_fraction_when_the_sample_is_typed(both_genes_typed):
    _, neos = both_genes_typed
    assert neos
    assert all(n.cancer_cell_fraction is not None for n in neos)


def test_a_low_frequency_variant_is_labelled_subclonal(both_genes_typed):
    """0.05 of reads in a pure diploid tumour is 5-10% of cells."""
    _, neos = both_genes_typed
    labels = by_gene(neos, "clonality")
    assert labels["GENEB"] == {CLONALITY_SUBCLONAL}
    b = next(n for n in neos if n.source_gene == "GENEB")
    assert b.cancer_cell_fraction.lower == pytest.approx(0.05, abs=1e-9)
    assert b.cancer_cell_fraction.upper == pytest.approx(0.10, abs=1e-9)


def test_a_high_frequency_variant_is_only_called_possibly_clonal(
        both_genes_typed):
    """Never plain "clonal": bulk data from one sample does not prove it."""
    _, neos = both_genes_typed
    labels = by_gene(neos, "clonality")
    assert labels["GENEA"] == {CLONALITY_POSSIBLY_CLONAL}
    assert CLONALITY_POSSIBLY_CLONAL != "clonal"


def test_the_fraction_counter_counts_variants_not_peptides(both_genes_typed):
    """Two mutations, many peptides, two fractions."""
    p, neos = both_genes_typed
    assert p.diagnostics.variants_with_a_cell_fraction == 2
    assert len(neos) > 2


def test_an_untyped_gene_gets_no_fraction_even_beside_a_typed_one(inputs):
    """A gene with no copy state is skipped, not assumed diploid."""
    variants, proteins = inputs
    p = NeoantigenPredictor()
    sample = SampleCopyProfile(
        purity=0.7,
        purity_source="set by this test",
        copy_states={"GENEA": diploid("GENEA")},
    )
    neos = p.predict(variants, proteins, ALLELES, peptide_lengths=LENGTHS,
                     max_neoantigens=500, sample=sample)
    fractions = {g: {f is not None for f in v}
                 for g, v in by_gene(neos, "cancer_cell_fraction").items()}
    assert fractions["GENEA"] == {True}
    assert fractions["GENEB"] == {False}
    assert p.diagnostics.variants_with_a_cell_fraction == 1


def test_a_purity_with_no_provenance_is_refused():
    with pytest.raises(ValueError, match="purity_source is required"):
        SampleCopyProfile(purity=0.7, copy_states={"X": diploid("X")})


def test_a_profile_without_purity_computes_nothing():
    sample = SampleCopyProfile(copy_states={"GENEA": diploid("GENEA")})
    assert not sample.can_compute_fractions
    assert sample.fraction_for_gene("GENEA", 0.3) is None


# ── it is an output: supplying a sample changes nothing else ──────

def test_the_sample_profile_adds_a_column_and_moves_nothing(
        plain, both_genes_typed):
    """The coverage work must not perturb the binding predictions."""
    _, without = plain
    _, with_sample = both_genes_typed
    assert [n.peptide for n in without] == [n.peptide for n in with_sample]
    assert [n.best_hla_allele for n in without] == [
        n.best_hla_allele for n in with_sample]
    assert [n.binding_affinity_nm for n in without] == [
        n.binding_affinity_nm for n in with_sample]
    assert [n.vaccine_priority for n in without] == [
        n.vaccine_priority for n in with_sample]


def test_the_serialised_record_carries_the_fraction(both_genes_typed):
    _, neos = both_genes_typed
    d = next(n for n in neos if n.source_gene == "GENEB").to_dict()
    assert d["clonality"] == CLONALITY_SUBCLONAL
    assert d["cancer_cell_fraction"]["upper"] == pytest.approx(0.10, abs=1e-9)
    assert d["cancer_cell_fraction"]["status"] == "derived"


def test_a_plain_record_serialises_the_absence_as_null(plain):
    _, neos = plain
    d = neos[0].to_dict()
    assert d["clonality"] is None
    assert d["cancer_cell_fraction"] is None
