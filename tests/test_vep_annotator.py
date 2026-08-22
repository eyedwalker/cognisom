"""
The VEP annotation stage.

Protein consequences cannot be derived from a genomic coordinate alone,
so `_predict_protein_change` was made to return None rather than invent
them -- which left raw callsets yielding no neoantigens at all. This
stage is what fills that gap, and its correctness turns almost entirely
on one decision: which transcript to read.

Residue numbering is a property of a transcript, not of a gene. VEP
returns every transcript it knows; for BRAF V600E it offers 24, reporting
the same substitution at protein position 600, 640, 299 and 157. Read the
wrong one and the position disagrees with the bundled reference proteome,
so the wild-type residue check refuses a real mutation.

MANE Select is the anchor: for BRAF that is ENST00000646891 -> position
600 -> SwissProt P15056, the same entry in reference_proteins.fasta.

These tests run offline against recorded response shapes. The one test
that talks to Ensembl is marked `slow` and skips without opt-in.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.genomics.vcf_parser import Variant
from cognisom.genomics.vep_annotator import (
    VEPAnnotator,
    VEPUnavailableError,
    _protein_change,
    _shorten_hgvsp,
)


def variant(chrom="chr7", pos=140753336, ref="A", alt="T") -> Variant:
    return Variant(chrom=chrom, pos=pos, id=".", ref=ref, alt=alt,
                   qual=99.0, filter_status="PASS")


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    """Stands in for requests.Session, recording what was submitted."""

    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status
        self.posted = []

    def post(self, url, headers=None, params=None, json=None, timeout=None):
        self.posted.append({"url": url, "params": params, "json": json})
        return FakeResponse(self.payload, self.status)


# The BRAF V600E response, reduced to the transcripts that matter. This
# is the real shape: several coding transcripts, disagreeing on position,
# exactly one of them MANE Select.
BRAF_RESPONSE = [{
    "input": "7 140753336 . A T . . .",
    "most_severe_consequence": "missense_variant",
    "transcript_consequences": [
        {"transcript_id": "ENST00000288602", "gene_symbol": "BRAF",
         "consequence_terms": ["missense_variant"], "amino_acids": "V/E",
         "protein_end": 640},
        {"transcript_id": "ENST00000496384", "gene_symbol": "BRAF",
         "consequence_terms": ["missense_variant"], "amino_acids": "V/E",
         "protein_end": 600, "canonical": 1},
        {"transcript_id": "ENST00000646891", "gene_symbol": "BRAF",
         "consequence_terms": ["missense_variant"], "amino_acids": "V/E",
         "protein_end": 600, "mane_select": "NM_004333.6",
         "swissprot": ["P15056.266"], "hgvsp": "ENSP00000493543.1:p.Val600Glu"},
    ],
}]


# ── Transcript selection ────────────────────────────────────────────

def test_mane_select_is_preferred_over_every_other_transcript():
    """The whole correctness of this stage rests on this choice."""
    session = FakeSession(BRAF_RESPONSE)
    annotator = VEPAnnotator(allow_remote=True, session=session)

    v = variant()
    annotator.annotate([v])

    assert v.gene == "BRAF"
    assert v.transcript == "ENST00000646891"
    assert v.protein_change == "p.V600E"      # not 640, not 157
    assert annotator.stats.mane_transcript_used == 1
    assert annotator.stats.fallback_transcript_used == 0


def test_canonical_is_used_when_no_mane_transcript_exists():
    payload = [{
        "input": "7 140753336 . A T . . .",
        "transcript_consequences": [
            {"transcript_id": "ENST0000AAA", "gene_symbol": "BRAF",
             "consequence_terms": ["missense_variant"], "amino_acids": "V/E",
             "protein_end": 640},
            {"transcript_id": "ENST0000BBB", "gene_symbol": "BRAF",
             "consequence_terms": ["missense_variant"], "amino_acids": "V/E",
             "protein_end": 600, "canonical": 1},
        ],
    }]
    annotator = VEPAnnotator(allow_remote=True, session=FakeSession(payload))
    v = variant()
    annotator.annotate([v])

    assert v.transcript == "ENST0000BBB"
    assert v.protein_change == "p.V600E"
    assert annotator.stats.fallback_transcript_used == 1


def test_the_swissprot_accession_is_carried_through():
    """It is what lets a caller check the proteome agrees."""
    v = variant()
    VEPAnnotator(allow_remote=True,
                 session=FakeSession(BRAF_RESPONSE)).annotate([v])
    assert v.info["SWISSPROT"] == "P15056"   # version suffix stripped


def test_coding_status_comes_from_the_shared_consequence_rule():
    v = variant()
    VEPAnnotator(allow_remote=True,
                 session=FakeSession(BRAF_RESPONSE)).annotate([v])
    assert v.consequence == "missense"
    assert v.is_coding is True


def test_a_non_coding_variant_is_labelled_not_left_blank():
    payload = [{"input": "1 100 . A T . . .",
                "most_severe_consequence": "intron_variant",
                "transcript_consequences": [
                    {"transcript_id": "T1", "gene_symbol": "X",
                     "consequence_terms": ["intron_variant"]}]}]
    annotator = VEPAnnotator(allow_remote=True, session=FakeSession(payload))
    v = variant(chrom="1", pos=100)
    annotator.annotate([v])

    assert v.consequence == "intron"
    assert v.is_coding is False
    assert annotator.stats.variants_no_consequence == 1


# ── What gets submitted ─────────────────────────────────────────────

def test_chr_prefix_is_stripped_for_ensembl():
    session = FakeSession(BRAF_RESPONSE)
    VEPAnnotator(allow_remote=True, session=session).annotate([variant()])
    assert session.posted[0]["json"]["variants"] == ["7 140753336 . A T . . ."]


def test_indels_are_not_submitted():
    """They need left-alignment against the reference to be expressed
    correctly here, and getting that subtly wrong is worse than not
    annotating them."""
    session = FakeSession([])
    annotator = VEPAnnotator(allow_remote=True, session=session)
    annotator.annotate([
        variant(ref="ACTG", alt="A"),
        variant(ref="A", alt="ATTG"),
    ])
    assert annotator.stats.variants_submitted == 0
    assert session.posted == []


def test_mane_and_uniprot_flags_are_requested():
    session = FakeSession(BRAF_RESPONSE)
    VEPAnnotator(allow_remote=True, session=session).annotate([variant()])
    params = session.posted[0]["params"]
    assert params["mane"] == 1 and params["uniprot"] == 1


def test_grch37_uses_the_separate_ensembl_host():
    session = FakeSession(BRAF_RESPONSE)
    VEPAnnotator(assembly="GRCh37", allow_remote=True,
                 session=session).annotate([variant()])
    assert "grch37" in session.posted[0]["url"]


def test_an_unknown_assembly_is_rejected():
    with pytest.raises(ValueError):
        VEPAnnotator(assembly="hg19")


# ── Failure is loud ─────────────────────────────────────────────────

def test_remote_use_must_be_opted_into():
    """Ensembl receives the coordinates; that is a decision, not a default."""
    with pytest.raises(VEPUnavailableError, match="patient data"):
        VEPAnnotator().annotate([variant()])


def test_an_http_failure_raises_rather_than_returning_nothing():
    """'No coding variants' and 'nothing annotated them' are opposite
    findings that must not look identical."""
    annotator = VEPAnnotator(allow_remote=True,
                             session=FakeSession("gateway timeout", status=502))
    with pytest.raises(VEPUnavailableError, match="502"):
        annotator.annotate([variant()])


def test_an_error_body_raises():
    annotator = VEPAnnotator(
        allow_remote=True,
        session=FakeSession({"error": "sequence region not found"}))
    with pytest.raises(VEPUnavailableError, match="sequence region"):
        annotator.annotate([variant()])


def test_empty_input_is_not_an_error():
    assert VEPAnnotator().annotate([]) == []


# ── HGVS conversion ─────────────────────────────────────────────────

@pytest.mark.parametrize("three,one", [
    ("p.Val600Glu", "p.V600E"),
    ("p.Arg248Trp", "p.R248W"),
    ("p.Gly12Val", "p.G12V"),
    ("p.Glu178Ter", "p.E178*"),
])
def test_three_letter_hgvs_is_converted(three, one):
    """The rest of the pipeline parses one-letter HGVS; VEP emits three."""
    assert _shorten_hgvsp(three) == one


@pytest.mark.parametrize("hgvsp", ["p.Val600", "p.Xyz600Glu", "c.1799T>A", ""])
def test_unparseable_hgvs_is_declined(hgvsp):
    assert _shorten_hgvsp(hgvsp) is None


def test_protein_change_falls_back_to_amino_acids_when_hgvsp_is_absent():
    assert _protein_change(
        {"amino_acids": "R/W", "protein_end": 248}) == "p.R248W"
    assert _protein_change({"amino_acids": "R", "protein_end": 248}) is None
    assert _protein_change({"amino_acids": "R/W"}) is None


# ── Against the live service ────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("COGNISOM_ALLOW_VEP_REMOTE") != "1",
    reason="set COGNISOM_ALLOW_VEP_REMOTE=1 to query Ensembl",
)
def test_live_vep_agrees_with_the_bundled_proteome():
    """The property that makes this stage safe: MANE positions line up
    with the UniProt sequences shipped in reference_proteins.fasta."""
    from cognisom.genomics.gene_protein_mapper import BUILTIN_PROTEINS

    variants = [
        variant("chr7", 140753336, "A", "T"),    # BRAF V600E
        variant("chr17", 7674220, "C", "A"),     # TP53 R248L
        variant("chr12", 25245350, "C", "A"),    # KRAS G12V
    ]
    VEPAnnotator(allow_remote=True).annotate(variants)

    import re
    for v in variants:
        assert v.protein_change, f"{v.location_str} not annotated"
        wt, pos, _ = re.match(r"p\.([A-Z])(\d+)([A-Z*])",
                              v.protein_change).groups()
        protein = BUILTIN_PROTEINS[v.gene]
        assert protein.residue_at(int(pos)) == wt, (
            f"{v.gene} {v.protein_change}: proteome disagrees with VEP"
        )
