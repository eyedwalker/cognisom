"""
VEP Annotation Stage
====================

Adds real protein consequences to a raw variant callset.

Why this exists
---------------
A protein consequence cannot be derived from a genomic coordinate alone:
it needs the reference sequence and the transcript's exon structure and
reading frame. ``VariantAnnotator._predict_protein_change`` used to
produce one anyway, from a single DNA base and an offset from the start
of the gene, and every field of the result was invented. It now returns
None, which is correct -- and leaves raw callsets yielding no
neoantigens until something real annotates them. This is that something.

Transcript selection is the whole problem
-----------------------------------------
Residue numbering is a property of a transcript, not of a gene. VEP
returns every transcript it knows: for BRAF V600E it offers 24, reporting
the same substitution at protein position 600, 640, 299 and 157 depending
on which one you read. Take the wrong one and the position disagrees with
the reference proteome this package bundles, so the wild-type residue
check refuses a perfectly real mutation.

This module pins **MANE Select**, which for BRAF is ENST00000646891 ->
position 600 -> SwissProt P15056, the same entry in
``data/reference_proteins.fasta``. MANE is the transcript set RefSeq and
Ensembl agreed on precisely so that clinical reporting has one answer, so
it is the right anchor and it lines up with UniProt canonical by design.

The SwissProt accession VEP reports is carried through and checked
against the bundled proteome, and the existing wild-type residue
validation in the neoantigen path is the backstop: a transcript mismatch
that slips through shows up as a refused variant with a recorded reason,
never as a silently wrong peptide.

Privacy
-------
The REST backend sends variant coordinates to Ensembl's public service.
That is fine for cell lines and public cohorts and is NOT appropriate for
identifiable patient data, so it must be opted into explicitly. Run VEP
locally (the offline cache, via Docker) for anything clinical.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import requests

from .vcf_parser import Variant

logger = logging.getLogger(__name__)

# Ensembl serves GRCh38 from the main host and GRCh37 from a separate one.
REST_HOSTS = {
    "GRCh38": "https://rest.ensembl.org",
    "GRCh37": "https://grch37.rest.ensembl.org",
}

#: Ensembl asks for <=15 requests/second and caps POST bodies; 200 is
#: comfortably inside both and keeps the round trips low.
BATCH_SIZE = 200
REQUEST_PAUSE_SEC = 0.2

#: Three-letter to one-letter, for HGVS protein notation.
AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*",
}


class VEPUnavailableError(RuntimeError):
    """Raised when annotation was requested and could not be performed.

    Deliberately an error rather than an empty result: "this callset has
    no coding variants" and "nothing annotated it" are opposite findings
    that would otherwise look identical downstream.
    """


@dataclass
class AnnotationStats:
    """What the annotation stage did, and did not, manage to do."""
    variants_submitted: int = 0
    variants_annotated: int = 0
    variants_no_consequence: int = 0
    variants_not_returned: int = 0
    coding_annotated: int = 0
    mane_transcript_used: int = 0
    fallback_transcript_used: int = 0
    genes_seen: List[str] = field(default_factory=list)
    assembly: str = ""
    backend: str = ""

    def to_dict(self) -> Dict:
        return {
            "backend": self.backend,
            "assembly": self.assembly,
            "variants_submitted": self.variants_submitted,
            "variants_annotated": self.variants_annotated,
            "variants_no_consequence": self.variants_no_consequence,
            "variants_not_returned": self.variants_not_returned,
            "coding_annotated": self.coding_annotated,
            "mane_transcript_used": self.mane_transcript_used,
            "fallback_transcript_used": self.fallback_transcript_used,
            "distinct_genes": len(set(self.genes_seen)),
        }


class VEPAnnotator:
    """Annotate variants with Ensembl VEP.

    Example:
        annotator = VEPAnnotator(assembly="GRCh38", allow_remote=True)
        annotator.annotate(variants)
        print(annotator.stats.to_dict())
    """

    def __init__(self, assembly: str = "GRCh38",
                 allow_remote: bool = False,
                 timeout: int = 120,
                 session: Optional[requests.Session] = None):
        if assembly not in REST_HOSTS:
            raise ValueError(
                f"Unsupported assembly {assembly!r}; expected one of "
                f"{sorted(REST_HOSTS)}"
            )
        self.assembly = assembly
        self.allow_remote = allow_remote
        self.timeout = timeout
        self._session = session or requests.Session()
        self.stats = AnnotationStats(assembly=assembly, backend="ensembl-rest")

    # ── Public API ─────────────────────────────────────────────────

    def annotate(self, variants: Sequence[Variant]) -> List[Variant]:
        """Fill gene, consequence and protein_change in place.

        Returns the same list, so it composes with VariantAnnotator.
        """
        if not variants:
            return list(variants)

        if not self.allow_remote:
            raise VEPUnavailableError(
                "VEP annotation was requested but the remote backend is not "
                "enabled. The Ensembl REST service receives variant "
                "coordinates, which is acceptable for cell lines and public "
                "cohorts and not for identifiable patient data. Pass "
                "allow_remote=True to use it, or run VEP locally against an "
                "offline cache for clinical work."
            )

        self.stats = AnnotationStats(assembly=self.assembly,
                                     backend="ensembl-rest")
        by_key: Dict[str, Variant] = {}
        payload: List[str] = []
        for variant in variants:
            key = self._region_string(variant)
            if key is None:
                continue
            by_key[key] = variant
            payload.append(key)

        self.stats.variants_submitted = len(payload)
        returned = 0

        for chunk in _chunks(payload, BATCH_SIZE):
            for record in self._post(chunk):
                returned += 1
                variant = by_key.get(record.get("input", ""))
                if variant is not None:
                    self._apply(variant, record)

        self.stats.variants_not_returned = max(
            0, self.stats.variants_submitted - returned
        )
        logger.info("VEP annotation: %s", self.stats.to_dict())
        return list(variants)

    # ── Internals ──────────────────────────────────────────────────

    @staticmethod
    def _region_string(variant: Variant) -> Optional[str]:
        """Format a variant as VEP's `region` input.

        Ensembl uses bare chromosome names, so a `chr` prefix is stripped.
        Only substitutions are submitted: an indel needs left-alignment
        against the reference to be expressed correctly here, and getting
        that subtly wrong is worse than not annotating it.
        """
        ref, alt = (variant.ref or "").upper(), (variant.alt or "").upper()
        if len(ref) != 1 or len(alt) != 1:
            return None
        if ref not in "ACGT" or alt not in "ACGT":
            return None
        chrom = variant.chrom[3:] if variant.chrom.lower().startswith("chr") \
            else variant.chrom
        return f"{chrom} {variant.pos} . {ref} {alt} . . ."

    def _post(self, variants: List[str]) -> List[Dict]:
        url = f"{REST_HOSTS[self.assembly]}/vep/human/region"
        try:
            response = self._session.post(
                url,
                headers={"Content-Type": "application/json",
                         "Accept": "application/json"},
                params={"uniprot": 1, "mane": 1, "canonical": 1},
                json={"variants": variants},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise VEPUnavailableError(f"VEP request failed: {e}") from e

        if response.status_code != 200:
            raise VEPUnavailableError(
                f"VEP returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        time.sleep(REQUEST_PAUSE_SEC)
        body = response.json()
        if isinstance(body, dict) and "error" in body:
            raise VEPUnavailableError(f"VEP error: {body['error'][:200]}")
        return body if isinstance(body, list) else []

    def _apply(self, variant: Variant, record: Dict) -> None:
        consequences = record.get("transcript_consequences") or []
        chosen = self._select_transcript(consequences)
        if chosen is None:
            self.stats.variants_no_consequence += 1
            # Still record the top-level worst consequence, so an intronic
            # or intergenic call is labelled rather than left blank.
            terms = record.get("most_severe_consequence")
            if terms:
                variant.consequence = _normalise(terms)
            return

        self.stats.variants_annotated += 1
        if chosen.get("mane_select"):
            self.stats.mane_transcript_used += 1
        else:
            self.stats.fallback_transcript_used += 1

        gene = chosen.get("gene_symbol")
        if gene:
            variant.gene = gene
            self.stats.genes_seen.append(gene)

        variant.transcript = chosen.get("transcript_id") or variant.transcript
        terms = chosen.get("consequence_terms") or []
        if terms:
            variant.consequence = _normalise(terms[0])

        protein_change = _protein_change(chosen)
        if protein_change:
            variant.protein_change = protein_change

        # Coding status comes from the consequence term, via the same
        # rule TMB uses, so the two cannot disagree.
        from .vcf_parser import is_protein_altering
        if is_protein_altering(variant.consequence):
            variant.is_coding = True
            self.stats.coding_annotated += 1

        swissprot = chosen.get("swissprot") or []
        if swissprot:
            # e.g. "P15056.266" -> "P15056"
            variant.info = dict(variant.info)
            variant.info["SWISSPROT"] = str(swissprot[0]).split(".")[0]

    @staticmethod
    def _select_transcript(consequences: Iterable[Dict]) -> Optional[Dict]:
        """Pick one transcript, preferring MANE Select.

        Residue numbering is a property of the transcript, so this choice
        decides whether reported positions line up with the bundled
        reference proteome. MANE Select is the RefSeq/Ensembl consensus
        transcript and matches UniProt canonical, which is what that
        proteome holds.
        """
        coding = [c for c in consequences if c.get("protein_end")]
        if not coding:
            return None
        for predicate in (lambda c: c.get("mane_select"),
                          lambda c: c.get("canonical"),
                          lambda c: True):
            for consequence in coding:
                if predicate(consequence):
                    return consequence
        return None


def _protein_change(consequence: Dict) -> Optional[str]:
    """Build `p.R248W` from VEP's amino_acids/protein_end pair."""
    hgvsp = consequence.get("hgvsp")
    if hgvsp and ":" in hgvsp:
        short = _shorten_hgvsp(hgvsp.split(":", 1)[1])
        if short:
            return short

    amino_acids = consequence.get("amino_acids") or ""
    position = consequence.get("protein_end")
    if not position or "/" not in amino_acids:
        return None
    ref_aa, _, alt_aa = amino_acids.partition("/")
    if len(ref_aa) != 1 or len(alt_aa) != 1:
        return None
    return f"p.{ref_aa}{position}{alt_aa}"


def _shorten_hgvsp(hgvsp: str) -> Optional[str]:
    """Convert `p.Arg248Trp` to `p.R248W`.

    The rest of the pipeline parses one-letter HGVS; VEP emits three.
    """
    import re

    match = re.fullmatch(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", hgvsp)
    if not match:
        return None
    ref, pos, alt = match.groups()
    if ref not in AA3_TO_1 or alt not in AA3_TO_1:
        return None
    return f"p.{AA3_TO_1[ref]}{pos}{AA3_TO_1[alt]}"


def _normalise(term: str) -> str:
    """VEP terms to the vocabulary vcf_parser already understands."""
    mapping = {
        "missense_variant": "missense",
        "stop_gained": "nonsense",
        "frameshift_variant": "frameshift",
        "splice_acceptor_variant": "splice_acceptor",
        "splice_donor_variant": "splice_donor",
        "splice_region_variant": "splice_region",
        "start_lost": "start_lost",
        "stop_lost": "stop_lost",
        "inframe_deletion": "inframe_deletion",
        "inframe_insertion": "inframe_insertion",
        "synonymous_variant": "synonymous",
    }
    return mapping.get(term, term.replace("_variant", ""))


def _chunks(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]
