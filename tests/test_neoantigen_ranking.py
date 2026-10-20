"""What puts the candidate list in order, and what a peptide's failure means.

Two outputs the trained scorer returns on every call were being thrown
away, and the one that was kept was not the one being ranked on.

**Ranking.** The list was sorted on raw IC50. That is not comparable
across alleles -- the same nanomolar figure is a good binder for one
allele's groove and a poor one for another's -- so ordering a
mixed-allele list by it ranks the alleles as much as the peptides. The
presentation score is on one scale for all of them and additionally
folds in whether the peptide would be produced at all, which affinity
cannot see.

**Processing.** ``processing_score`` was never read. It is what separates
a peptide that fails because nothing would ever cut it out of the
protein from one that is produced and simply does not fit the groove.
The first is not a candidate at any affinity; the second may be one on
another allele.

Neither branch can be exercised on a machine without MHCflurry, because
the approximate path produces neither number -- so these tests supply
the columns the trained scorer returns and check what the pipeline does
with them. Measured through this repo's code on the real predictor, over
five peptides x three alleles:

- every row carried a processing score; none were missing;
- the affinity ordering and the presentation ordering of those fifteen
  pairs diverge, first at rank 8 (affinity puts LVVVGADGVG/A*02:01 at
  24,893 nM ahead of KLVVVGADGV/A*24:02 at 28,564 nM; presentation
  reverses them), which is the cross-allele incomparability this change
  is about;
- the processing score is per peptide, not per pair -- NLVPMVATV comes
  back 0.5330 on all three alleles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.genomics import mhcflurry_binding as mb
from cognisom.genomics.gene_protein_mapper import BUILTIN_PROTEINS
from cognisom.genomics.neoantigen_predictor import NeoantigenPredictor
from cognisom.genomics.vcf_parser import Variant

HLA = ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"]


def _variant(gene="KRAS", protein_change="p.G12D", af=0.4):
    return Variant(
        chrom="chr12", pos=25245350, id=".", ref="C", alt="T", qual=99.0,
        filter_status="PASS", gene=gene, protein_change=protein_change,
        is_coding=True, is_cancer_driver=True, tumor_af=af,
    )


class _Scored:
    """What one row of the trained scorer's frame amounts to."""

    method = "mhcflurry"

    def __init__(self, affinity, presentation, processing):
        self.affinity_nm = affinity
        self.presentation_score = presentation
        self.processing_score = processing
        self.percentile_rank = presentation and (1.0 - presentation) * 10


def _install(monkeypatch, score):
    """Score every pair through ``score(peptide) -> _Scored``."""
    monkeypatch.setattr(mb, "is_mhcflurry_available", lambda: True)
    monkeypatch.setattr(mb, "predict_binding", lambda p, a: score(p))
    monkeypatch.setattr(mb, "predict_binding_batch",
                        lambda peps, als: [score(p) for p in peps])


def _disagreeing(peptides):
    """Affinity and presentation that rank the same peptides oppositely.

    The alphabetically first peptide gets the best affinity and the worst
    presentation score, so the two orderings cannot both be satisfied and
    the test can tell which one ran.
    """
    order = {p: i for i, p in enumerate(sorted(peptides))}

    def score(peptide):
        i = order[peptide]
        return _Scored(affinity=100.0 + i,
                       presentation=0.05 * (i + 1),
                       processing=0.5)
    return score


def _run(predictor, variant, gene="KRAS"):
    return predictor.predict(
        cancer_mutations=[variant],
        affected_proteins={gene: BUILTIN_PROTEINS[gene]},
        hla_alleles=HLA,
    )


def _peptides(variant, gene="KRAS"):
    """The peptides this variant yields, under whatever scorer is in place."""
    return sorted({n.peptide for n in _run(NeoantigenPredictor(), variant, gene)})


def test_the_list_is_ranked_on_presentation_not_affinity(monkeypatch):
    variant = _variant()
    peptides = _peptides(variant)
    assert len(peptides) > 1, "need several peptides to have an ordering"

    _install(monkeypatch, _disagreeing(peptides))
    predictor = NeoantigenPredictor()
    result = _run(predictor, variant)

    assert result, "the variant yielded nothing to rank"
    assert predictor.diagnostics.ranking_basis == "presentation_score"

    scores = [n.presentation_score for n in result]
    assert scores == sorted(scores, reverse=True), scores
    # The affinity ordering is the reverse, so this is not a coincidence.
    affinities = [n.binding_affinity_nm for n in result]
    assert affinities != sorted(affinities), (
        "ranked by affinity after all", affinities)
    assert result[0].presentation_score == max(scores)


def test_the_approximate_path_still_ranks_on_affinity():
    """Unchanged behaviour where there is no presentation score: the
    fallback has to be exactly what it was, not a new ordering."""
    predictor = NeoantigenPredictor()
    result = _run(predictor, _variant())

    assert result
    assert predictor.diagnostics.ranking_basis == "affinity_nm"
    assert all(n.presentation_score is None for n in result)
    affinities = [n.binding_affinity_nm for n in result]
    assert affinities == sorted(affinities), affinities


def test_the_two_orderings_are_never_mixed(monkeypatch):
    """One candidate without a presentation score sends the whole list to
    the affinity ordering, because a list sorted on two different scales
    is not sorted at all."""
    variant = _variant()
    peptides = _peptides(variant)
    full = _disagreeing(peptides)
    odd_one = peptides[0]

    def partial(peptide):
        scored = full(peptide)
        if peptide == odd_one:                 # the approximate path for one
            scored.presentation_score = None
            scored.processing_score = None
        return scored

    _install(monkeypatch, partial)
    predictor = NeoantigenPredictor()
    result = _run(predictor, variant)

    assert predictor.diagnostics.ranking_basis == "affinity_nm"
    affinities = [n.binding_affinity_nm for n in result]
    assert affinities == sorted(affinities), affinities


def test_the_processing_score_reaches_the_candidate(monkeypatch):
    """Returned on every call and previously discarded."""
    variant = _variant()
    peptides = _peptides(variant)
    order = {p: i for i, p in enumerate(peptides)}
    _install(monkeypatch, lambda p: _Scored(
        affinity=200.0, presentation=0.5, processing=0.1 * (order[p] + 1)))

    result = _run(NeoantigenPredictor(), variant)
    assert result
    assert all(n.processing_score is not None for n in result)
    carried = {n.peptide: n.processing_score for n in result}
    for peptide, score in carried.items():
        assert score == pytest.approx(0.1 * (order[peptide] + 1))


def test_a_processing_score_is_not_invented_on_the_approximate_path():
    """A motif score says nothing about proteasomal cleavage, so there is
    nothing honest to put here. A number would read as a measurement."""
    result = _run(NeoantigenPredictor(), _variant())
    assert result
    assert all(n.processing_score is None for n in result)


def test_the_ranking_basis_is_recorded_either_way(monkeypatch):
    """A caller comparing two runs' top candidates needs to know which
    ordering it is looking at; the two are not interchangeable."""
    variant = _variant()
    assert NeoantigenPredictor().diagnostics.ranking_basis == ""

    plain = NeoantigenPredictor()
    _run(plain, variant)
    assert plain.diagnostics.ranking_basis == "affinity_nm"

    _install(monkeypatch, lambda p: _Scored(300.0, 0.4, 0.4))
    trained = NeoantigenPredictor()
    _run(trained, variant)
    assert trained.diagnostics.ranking_basis == "presentation_score"
