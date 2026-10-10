"""
VariantAnnotator must never invent protein-level annotations.

Background. ``_predict_protein_change`` used to synthesize an HGVS
protein change for any SNV in a cancer driver gene, using a codon
position derived as ``(pos - gene_start) // 3`` and amino acids drawn
from two small lookup tables keyed on the DNA bases. It ignored exon
structure, UTRs, CDS offset and strand, so the output was syntactically
valid and biologically meaningless.

That output was not quarantined. It reached the neoantigen predictor,
the OncoKB annotator, and -- through
``mutation_adapter.variant_to_patent_mutation`` -- the simulation
pipeline, where it is indistinguishable from a real VEP/SnpEff call.

These tests pin the abstention. A protein change may only come from the
input VCF's own annotation; where that is absent the field stays None.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.genomics.variant_annotator import VariantAnnotator
from cognisom.genomics.vcf_parser import Variant


def _variant(**kw):
    fields = dict(
        chrom="chr12", pos=25245350, id=".", ref="C", alt="T",
        qual=100.0, filter_status="PASS",
        gene=None, protein_change=None, consequence=None,
    )
    fields.update(kw)
    return Variant(**fields)


def test_predict_protein_change_always_abstains():
    """The generator must return None for every input, including the
    driver-gene SNVs it used to fabricate for."""
    ann = VariantAnnotator()
    for gene in ("KRAS", "TP53", "BRAF", "AR", "PTEN"):
        for ref, alt in (("A", "C"), ("C", "T"), ("G", "A"), ("T", "G")):
            v = _variant(ref=ref, alt=alt, gene=gene)
            assert ann._predict_protein_change(v, gene) is None, (
                f"annotator fabricated a protein change for {gene} "
                f"{ref}>{alt}; it has no transcript model and must abstain"
            )


def test_indels_are_not_given_invented_consequences():
    """Indels previously produced 'p.del<n>' / 'p.ins<n>' strings, which
    assert a protein consequence that equally requires a transcript
    model to determine."""
    ann = VariantAnnotator()
    deletion = _variant(ref="CTGA", alt="C", gene="TP53")
    insertion = _variant(ref="C", alt="CTGA", gene="TP53")
    assert ann._predict_protein_change(deletion, "TP53") is None
    assert ann._predict_protein_change(insertion, "TP53") is None


def test_unannotated_driver_variant_keeps_protein_change_none():
    """End-to-end through the public annotate path: a raw VCF row in a
    driver gene is still flagged as a driver, but gains no protein-level
    annotation it did not arrive with."""
    ann = VariantAnnotator()
    v = _variant(gene="KRAS", protein_change=None)
    annotated = ann.annotate([v])[0]
    assert annotated.protein_change is None, (
        "an unannotated VCF row must not acquire a protein change"
    )


def test_existing_annotation_is_preserved():
    """The abstention must not clobber a real upstream annotation."""
    ann = VariantAnnotator()
    v = _variant(gene="KRAS", protein_change="p.G12D")
    annotated = ann.annotate([v])[0]
    assert annotated.protein_change == "p.G12D"


def test_annotator_is_stateless_across_reuse():
    """Regression: the fabrication path throttled itself with per-gene
    counters stashed on the instance as ``_pchange_count_<GENE>`` and
    never reset them, so a reused annotator degraded on each subsequent
    patient. No such state should exist now."""
    ann = VariantAnnotator()
    batch = [_variant(gene="KRAS", pos=25245350 + i) for i in range(10)]
    first = ann.annotate(list(batch))
    second = ann.annotate(list(batch))

    leaked = [a for a in vars(ann) if a.startswith("_pchange_count_")]
    assert not leaked, f"annotator leaked per-gene counters: {leaked}"

    assert [v.is_cancer_driver for v in first] == \
           [v.is_cancer_driver for v in second], (
        "annotating the same batch twice must give the same result"
    )
