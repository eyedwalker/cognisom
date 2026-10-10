"""Neoantigen generation must not be confined to the curated driver panel.

The bundled FASTA holds 37 proteins. A variant in any other gene had no
sequence to generate a peptide from, so gating antigen generation on the
driver panel and gating it on the proteome amounted to the same restriction --
and a tumour that should yield one to three hundred protein-altering somatic
variants yielded a handful of candidates.

`scripts/fetch_mane_proteome.py` builds the MANE Select proteome (~19,300
genes) into `data/mane_select_proteins.faa.gz`, which the mapper layers
underneath the curated panel. These tests pin the layering and the numbering;
the ones needing the generated file skip when it has not been fetched.
"""
import pytest

from cognisom.genomics.gene_protein_mapper import (
    BUILTIN_PROTEINS,
    MANE_FASTA,
    GeneProteinMapper,
    NUMBERING_OFFSETS,
    _load_reference_proteome,
)

needs_mane = pytest.mark.skipif(
    not MANE_FASTA.exists(),
    reason="MANE proteome not fetched; run scripts/fetch_mane_proteome.py",
)


class TestCuratedPanelIsUnaffected:
    def test_curated_entries_take_precedence(self):
        # The curated sequences are Swiss-Prot canonical and their numbering
        # has been checked against documented hotspots. Layering must not
        # replace them.
        curated = _load_reference_proteome()
        assert curated, "the curated FASTA should always be present"
        for gene, info in curated.items():
            assert BUILTIN_PROTEINS[gene].sequence == info.sequence, (
                f"{gene} was overwritten by the MANE layer"
            )

    def test_ar_keeps_its_verified_offset(self):
        # AR's +1 offset reconciles UniProt canonical numbering with the
        # numbering used throughout the androgen-receptor literature. It was
        # verified against eight documented mutations.
        assert NUMBERING_OFFSETS["AR"] == 1
        assert BUILTIN_PROTEINS["AR"].numbering_offset == 1


@needs_mane
class TestExomeWideCoverage:
    def test_many_more_genes_resolve(self):
        assert len(BUILTIN_PROTEINS) > 15_000, (
            f"expected the MANE layer to be loaded, got {len(BUILTIN_PROTEINS)}"
        )

    @pytest.mark.parametrize("gene,pos,wt", [
        # Recurrent hotspots in genes outside the 37-gene curated panel.
        ("CTNNB1", 45, "S"),
        ("SF3B1", 700, "K"),
        ("MTOR", 2215, "S"),
        ("RNF43", 117, "R"),
        ("SMARCA4", 1157, "R"),
    ])
    def test_non_panel_hotspot_residues_are_correct(self, gene, pos, wt):
        # If the numbering were off, these would silently disagree -- which is
        # the failure mode that makes a mis-built proteome worse than none.
        p = GeneProteinMapper().get_protein(gene)
        assert p is not None, f"{gene} should resolve from the MANE layer"
        assert p.sequence[pos - 1] == wt

    def test_mane_entries_carry_no_numbering_offset(self):
        # MANE Select numbering matches what the annotator reports, because
        # vep_annotator pins the same transcripts.
        p = GeneProteinMapper().get_protein("CTNNB1")
        assert p.numbering_offset == 0


class TestGracefulWithoutTheGeneratedFile:
    def test_curated_panel_alone_is_still_usable(self):
        # A fresh checkout has no MANE file. The curated genes must still work
        # rather than the loader raising.
        curated = _load_reference_proteome()
        assert len(curated) >= 37
        for gene in ("KRAS", "TP53", "BRAF"):
            assert gene in curated
