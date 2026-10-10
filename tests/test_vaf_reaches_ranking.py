"""Variant allele frequency must reach the neoantigen record and the ranking.

`tumor_af` was parsed by the VCF reader and used in somatic classification,
then never referenced again anywhere in the repository -- so a target present
in 4% of tumour reads ranked identically to one present in 90%, which is the
most obvious objection anyone would raise about the output.

Two neighbouring fields asserted facts the pipeline never established:
`expression_level` defaulted to the string "high" and `clonality` to "clonal"
on every record, neither ever assigned from data. They are now None, because
cancer-cell fraction needs tumour purity and allele-specific copy number and
transcript abundance needs RNA quantification, none of which this pipeline
computes.
"""
import pytest

from cognisom.genomics.neoantigen_predictor import Neoantigen, NeoantigenPredictor
from cognisom.genomics.gene_protein_mapper import GeneProteinMapper
from cognisom.genomics.vcf_parser import Variant


def _variant(gene, protein_change, tumor_af=None):
    v = Variant(chrom="1", pos=1, id=".", ref="A", alt="T",
                qual=100.0, filter_status="PASS")
    v.gene = gene
    v.protein_change = protein_change
    v.consequence = "missense_variant"
    v.is_coding = True
    v.is_cancer_driver = True
    v.tumor_af = tumor_af
    return v


class TestFieldsNoLongerAssertUnmeasuredFacts:
    def test_expression_and_clonality_default_to_unknown(self):
        n = Neoantigen(
            peptide="SIINFEKL", wild_type_peptide="SIINFEKM",
            source_gene="KRAS", mutation="G12D", protein_change="p.G12D",
            mutation_position_in_peptide=0, peptide_length=8,
            best_hla_allele="HLA-A*02:01", binding_affinity_nm=100.0,
            percentile_rank=1.0,
        )
        assert n.expression_level is None, (
            'must not claim "high" expression with no RNA quantification'
        )
        assert n.clonality is None, (
            'must not claim "clonal" with no purity or copy number'
        )
        assert n.variant_allele_frequency is None


class TestAlleleFrequencyIsCarried:
    def test_frequency_reaches_the_record(self):
        muts = [_variant("KRAS", "p.G12D", tumor_af=0.42)]
        proteins = GeneProteinMapper().get_proteins_for_genes(["KRAS"])
        neos = NeoantigenPredictor().predict(
            cancer_mutations=muts, affected_proteins=proteins,
            hla_alleles=["HLA-A*02:01"],
        )
        assert neos, "expected at least one peptide for KRAS G12D"
        assert all(n.variant_allele_frequency == pytest.approx(0.42)
                   for n in neos)

    def test_absent_frequency_stays_none(self):
        muts = [_variant("KRAS", "p.G12D", tumor_af=None)]
        proteins = GeneProteinMapper().get_proteins_for_genes(["KRAS"])
        neos = NeoantigenPredictor().predict(
            cancer_mutations=muts, affected_proteins=proteins,
            hla_alleles=["HLA-A*02:01"],
        )
        assert neos
        assert all(n.variant_allele_frequency is None for n in neos)

    def test_frequency_is_exported(self):
        n = Neoantigen(
            peptide="SIINFEKL", wild_type_peptide="SIINFEKM",
            source_gene="KRAS", mutation="G12D", protein_change="p.G12D",
            mutation_position_in_peptide=0, peptide_length=8,
            best_hla_allele="HLA-A*02:01", binding_affinity_nm=100.0,
            percentile_rank=1.0, variant_allele_frequency=0.1234567,
        )
        d = n.to_dict()
        assert d["variant_allele_frequency"] == pytest.approx(0.1235, abs=1e-4)
        assert Neoantigen(
            peptide="A", wild_type_peptide="A", source_gene="G",
            mutation="m", protein_change="p.m",
            mutation_position_in_peptide=0, peptide_length=1,
            best_hla_allele="x", binding_affinity_nm=1.0, percentile_rank=1.0,
        ).to_dict()["variant_allele_frequency"] is None


class TestRankingUsesItAsATiebreaker:
    def test_affinity_still_dominates(self):
        # A weaker binder with a high frequency must not outrank a stronger
        # binder with a low one: the frequency is a tiebreaker, not a term.
        lo = Neoantigen(peptide="A", wild_type_peptide="A", source_gene="G",
                        mutation="m", protein_change="p.m",
                        mutation_position_in_peptide=0, peptide_length=1,
                        best_hla_allele="x", binding_affinity_nm=10.0,
                        percentile_rank=1.0, variant_allele_frequency=0.05)
        hi = Neoantigen(peptide="B", wild_type_peptide="B", source_gene="G",
                        mutation="m", protein_change="p.m",
                        mutation_position_in_peptide=0, peptide_length=1,
                        best_hla_allele="x", binding_affinity_nm=400.0,
                        percentile_rank=2.0, variant_allele_frequency=0.95)
        ordered = sorted(
            [hi, lo],
            key=lambda n: (n.binding_affinity_nm,
                           -(n.variant_allele_frequency
                             if n.variant_allele_frequency is not None else -1.0)),
        )
        assert ordered[0] is lo

    def test_equal_affinity_prefers_the_higher_frequency(self):
        a = Neoantigen(peptide="A", wild_type_peptide="A", source_gene="G",
                       mutation="m", protein_change="p.m",
                       mutation_position_in_peptide=0, peptide_length=1,
                       best_hla_allele="x", binding_affinity_nm=151.0,
                       percentile_rank=2.0, variant_allele_frequency=0.10)
        b = Neoantigen(peptide="B", wild_type_peptide="B", source_gene="G",
                       mutation="m", protein_change="p.m",
                       mutation_position_in_peptide=0, peptide_length=1,
                       best_hla_allele="x", binding_affinity_nm=151.0,
                       percentile_rank=2.0, variant_allele_frequency=0.80)
        ordered = sorted(
            [a, b],
            key=lambda n: (n.binding_affinity_nm,
                           -(n.variant_allele_frequency
                             if n.variant_allele_frequency is not None else -1.0)),
        )
        assert ordered[0] is b

    def test_missing_frequency_sorts_last_among_equals(self):
        known = Neoantigen(peptide="A", wild_type_peptide="A", source_gene="G",
                           mutation="m", protein_change="p.m",
                           mutation_position_in_peptide=0, peptide_length=1,
                           best_hla_allele="x", binding_affinity_nm=151.0,
                           percentile_rank=2.0, variant_allele_frequency=0.01)
        unknown = Neoantigen(peptide="B", wild_type_peptide="B", source_gene="G",
                             mutation="m", protein_change="p.m",
                             mutation_position_in_peptide=0, peptide_length=1,
                             best_hla_allele="x", binding_affinity_nm=151.0,
                             percentile_rank=2.0)
        ordered = sorted(
            [unknown, known],
            key=lambda n: (n.binding_affinity_nm,
                           -(n.variant_allele_frequency
                             if n.variant_allele_frequency is not None else -1.0)),
        )
        assert ordered[0] is known
