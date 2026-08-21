"""
VCF Parser
==========

Parse Variant Call Format (VCF) files into structured variant records.
Supports VCF 4.1/4.2/4.3 format with annotation fields from common
variant callers (GATK HaplotypeCaller, DeepVariant, Strelka2, Mutect2).

Pure Python implementation — no external VCF library required.

Somatic (tumor/normal) support
------------------------------
A matched tumor-normal callset is not "a VCF with an extra column": which
column is the tumor is declared in the header, not by position. Mutect2
writes ``##tumor_sample=`` / ``##normal_sample=`` and then orders the sample
columns however the BAMs were supplied. The SEQC2 HCC1395 callset this repo
ships (``seqc2_demo.vcf``) is ordered ``normal`` first, so reading "the first
sample column" reads the *normal* genotype — the germline — which is the
exact opposite of what neoantigen calling needs.

This parser therefore resolves tumor/normal by name and exposes both sides
of the comparison (:attr:`Variant.tumor_af`, :attr:`Variant.normal_af`, ...)
plus the caller's own somatic evidence statistics, so downstream stages can
tell a somatic mutation from an inherited one instead of assuming.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Somatic calling thresholds ──────────────────────────────────────
#
# These are the high-confidence thresholds from the original MuTect paper
# (Cibulskis et al., "Sensitive detection of somatic point mutations in
# impure and heterogeneous cancer samples", Nat Biotechnol 31:213-219,
# 2013). They are named constants rather than inline numbers so that a
# reader can check them against the citation, and so that callers can
# override them explicitly rather than by editing code.
#
# Mutect2 *emits* at TLOD >= 3.0 (``--tumor-lod-to-emit``), which is far
# below the confidence bar for calling a mutation somatic. A raw Mutect2
# VCF is therefore dominated by low-evidence sites; treating every emitted
# record as a real somatic mutation inflates TMB and floods neoantigen
# prediction with artifacts.
TUMOR_LOD_THRESHOLD = 6.3      # LOD_T: variant is real in the tumor
NORMAL_LOD_THRESHOLD = 2.2     # LOD_N: normal is homozygous reference

# MuTect's "alt allele in normal" rule is a conjunction, and it matters that
# it stays one: a *single* alt read in a 30x normal is 3.3% by arithmetic, so
# a fraction test on its own rejects most true somatic calls as germline.
# Both the count and the fraction must be exceeded before the normal is
# treated as carrying the allele. (MuTect defaults
# --max_alt_alleles_in_normal_count 2, --max_alt_allele_in_normal_fraction 0.03.)
MAX_NORMAL_ALT_COUNT = 2
MAX_NORMAL_AF = 0.03
NORMAL_ARTIFACT_LOD_THRESHOLD = 0.0   # NALOD <= 0 => looks like an artifact present in normal

# INFO/FORMAT keys whose Number is "A" (one value per ALT allele) or "R"
# (one value per allele including REF). Multi-allelic sites are split into
# one Variant each, so these have to be indexed per-allele rather than
# copied wholesale — otherwise allele 2 inherits allele 1's depth and
# allele fraction.
_PER_ALT_INFO_KEYS = ("TLOD", "NLOD", "NALOD", "POPAF")
_PER_ALT_FORMAT_KEYS = ("AF",)
_PER_ALLELE_FORMAT_KEYS = ("AD", "FAD", "F1R2", "F2R1")


# Consequences that count toward tumor mutational burden.
#
# This set is not a matter of taste: TMB is only comparable across
# platforms if everyone counts the same categories, and the reference
# definition here is cBioPortal's TMB_NONSYNONYMOUS, since that is what
# the SU2C/PCF cohort reports and what this pipeline is validated
# against.
#
# It was derived from that cohort rather than assumed. Solving
# `mutation_count / reported_TMB` for the implied megabase denominator
# across 427 patients gives a median of exactly 30.00 Mb with the
# tightest spread (CV 0.028) for precisely this set. Narrower sets are
# visibly wrong: dropping splice, in-frame indels and nonstop moves the
# implied denominator to 28.38 Mb and more than doubles its variance,
# which is what a missing category looks like when you fit for it.
#
# Splice-region variants are included because the reference definition
# includes them, not because they are reliably protein-altering. They do
# not leak into neoantigen prediction: that path needs a parseable
# `p.XnnnY` protein change, which these do not have.
PROTEIN_ALTERING_CONSEQUENCES = frozenset({
    "missense",
    "nonsense", "stop_gained",
    "frameshift",
    "splice", "splice_site", "splice_donor", "splice_acceptor",
    "splice_region",
    "start_lost", "start_gained",
    "stop_lost", "nonstop",
    "inframe_deletion", "inframe_insertion",
    "protein_altering",
})


def is_protein_altering(consequence: Optional[str]) -> bool:
    """True when a consequence term should count toward TMB."""
    if not consequence:
        return False
    return consequence.strip().lower() in PROTEIN_ALTERING_CONSEQUENCES


@dataclass
class SomaticEvidence:
    """Caller-reported statistics bearing on whether a call is somatic.

    All fields are optional: they are only present if the caller emitted
    them. ``None`` means "not reported", which is different from zero.
    """
    tumor_lod: Optional[float] = None        # TLOD
    normal_lod: Optional[float] = None       # NLOD
    normal_artifact_lod: Optional[float] = None  # NALOD
    germline_qual: Optional[float] = None    # GERMQ (phred)
    population_af_phred: Optional[float] = None  # POPAF (-log10 population AF)
    strand_qual: Optional[float] = None      # STRANDQ
    seq_qual: Optional[float] = None         # SEQQ

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "tumor_lod": self.tumor_lod,
            "normal_lod": self.normal_lod,
            "normal_artifact_lod": self.normal_artifact_lod,
            "germline_qual": self.germline_qual,
            "population_af_phred": self.population_af_phred,
            "strand_qual": self.strand_qual,
            "seq_qual": self.seq_qual,
        }


@dataclass
class Variant:
    """A single genomic variant from a VCF file."""
    chrom: str
    pos: int
    id: str
    ref: str
    alt: str
    qual: float
    filter_status: str
    info: Dict[str, str] = field(default_factory=dict)
    genotype: str = "."  # e.g. "0/1", "1/1" — the TUMOR genotype when known

    # Annotation fields (populated by VariantAnnotator)
    gene: Optional[str] = None
    consequence: Optional[str] = None  # missense, nonsense, synonymous, etc.
    protein_change: Optional[str] = None  # e.g. "p.T877A"
    transcript: Optional[str] = None
    cosmic_id: Optional[str] = None
    clinvar_significance: Optional[str] = None
    dbsnp_id: Optional[str] = None

    # ── Tumor / normal evidence ────────────────────────────────────
    # Populated when the VCF declares a tumor (and optionally a normal)
    # sample. `None` means the caller did not report the field.
    tumor_af: Optional[float] = None
    tumor_depth: Optional[int] = None
    tumor_ref_count: Optional[int] = None
    tumor_alt_count: Optional[int] = None
    tumor_genotype: Optional[str] = None

    normal_af: Optional[float] = None
    normal_depth: Optional[int] = None
    normal_ref_count: Optional[int] = None
    normal_alt_count: Optional[int] = None
    normal_genotype: Optional[str] = None

    somatic_evidence: SomaticEvidence = field(default_factory=SomaticEvidence)
    # SOMATIC | GERMLINE | ARTIFACT | LOW_EVIDENCE | UNKNOWN
    somatic_status: str = "UNKNOWN"
    somatic_filter_reasons: List[str] = field(default_factory=list)

    # Computed fields
    is_coding: bool = False
    is_cancer_driver: bool = False
    impact: str = "UNKNOWN"  # HIGH, MODERATE, LOW, MODIFIER

    @property
    def is_snv(self) -> bool:
        return len(self.ref) == 1 and len(self.alt) == 1

    @property
    def is_indel(self) -> bool:
        return len(self.ref) != len(self.alt)

    @property
    def is_insertion(self) -> bool:
        return len(self.alt) > len(self.ref)

    @property
    def is_deletion(self) -> bool:
        return len(self.ref) > len(self.alt)

    @property
    def is_somatic(self) -> bool:
        """True only when the tumor/normal comparison supports somatic origin.

        Deliberately strict: ``UNKNOWN`` (no normal to compare against)
        is not somatic. A tumor-only VCF cannot distinguish a somatic
        mutation from a rare inherited one, and silently treating it as
        somatic is how germline variants end up in a vaccine design.
        """
        return self.somatic_status == "SOMATIC"

    @property
    def variant_type(self) -> str:
        if self.is_snv:
            return "SNV"
        elif self.is_insertion:
            return "INS"
        elif self.is_deletion:
            return "DEL"
        return "COMPLEX"

    @property
    def location_str(self) -> str:
        return f"{self.chrom}:{self.pos}"

    def summary(self) -> str:
        parts = [f"{self.chrom}:{self.pos} {self.ref}>{self.alt}"]
        if self.gene:
            parts.append(f"gene={self.gene}")
        if self.protein_change:
            parts.append(self.protein_change)
        if self.consequence:
            parts.append(self.consequence)
        if self.somatic_status != "UNKNOWN":
            parts.append(self.somatic_status.lower())
        if self.tumor_af is not None:
            parts.append(f"tumor_af={self.tumor_af:.3f}")
        return " ".join(parts)


class VCFParser:
    """Parse VCF files into structured Variant records.

    Supports:
    - Standard VCF 4.1/4.2/4.3 format
    - Multi-allelic sites (splits into separate Variant records, with
      per-allele INFO/FORMAT values indexed correctly)
    - INFO field parsing (ANN/CSQ for functional annotations)
    - Genotype (GT) extraction from FORMAT/sample columns
    - Matched tumor/normal callsets: resolves which column is the tumor
      from ``##tumor_sample`` / ``##normal_sample`` rather than by position

    After a parse, the header facts are available on the instance:
    :attr:`sample_names`, :attr:`tumor_sample`, :attr:`normal_sample`,
    :attr:`is_somatic_vcf` and :attr:`filters_applied`.

    Example:
        parser = VCFParser()
        variants = parser.parse_file("patient.vcf")
        coding = parser.filter_coding(variants)
        print(f"{len(coding)} coding variants out of {len(variants)} total")
    """

    def __init__(self):
        self.sample_names: List[str] = []
        self.tumor_sample: Optional[str] = None
        self.normal_sample: Optional[str] = None
        self.caller: Optional[str] = None
        # True once a record is seen with a FILTER value other than "."
        # i.e. the callset has actually been through a filtering step.
        self.filters_applied: bool = False

    @property
    def is_somatic_vcf(self) -> bool:
        """True when the header declares both a tumor and a normal sample."""
        return self.tumor_sample is not None and self.normal_sample is not None

    def parse_file(self, vcf_path: str) -> List[Variant]:
        """Parse a VCF file from disk.

        Args:
            vcf_path: Path to .vcf or .vcf.gz file.

        Returns:
            List of Variant records.
        """
        path = Path(vcf_path)
        if not path.exists():
            raise FileNotFoundError(f"VCF file not found: {vcf_path}")

        if path.suffix == ".gz":
            import gzip
            with gzip.open(path, "rt") as f:
                text = f.read()
        else:
            with open(path, "r") as f:
                text = f.read()

        return self.parse_text(text)

    def parse_text(self, vcf_text: str) -> List[Variant]:
        """Parse VCF content from a string.

        Args:
            vcf_text: VCF file content as text.

        Returns:
            List of Variant records.
        """
        variants: List[Variant] = []
        self.sample_names = []
        self.tumor_sample = None
        self.normal_sample = None
        self.caller = None
        self.filters_applied = False

        tumor_idx: Optional[int] = None
        normal_idx: Optional[int] = None

        for line in vcf_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue

            # Meta-information lines. These carry the tumor/normal
            # declaration, so they cannot simply be skipped.
            if line.startswith("##"):
                self._parse_meta(line)
                continue

            # Column header — establishes sample name -> column index.
            if line.startswith("#CHROM"):
                header_columns = line[1:].split("\t")
                self.sample_names = header_columns[9:]
                tumor_idx, normal_idx = self._resolve_sample_columns()
                continue

            # Data line
            fields = line.split("\t")
            if len(fields) < 8:
                continue

            chrom = fields[0]
            pos = int(fields[1])
            var_id = fields[2]
            ref = fields[3]
            alts = fields[4].split(",")
            qual = float(fields[5]) if fields[5] != "." else 0.0
            filt = fields[6]
            info_str = fields[7]

            if filt not in (".", ""):
                self.filters_applied = True

            # Parse INFO field
            info = self._parse_info(info_str)

            # Parse per-sample FORMAT values once per record.
            tumor_fields: Dict[str, str] = {}
            normal_fields: Dict[str, str] = {}
            if len(fields) >= 10:
                fmt_keys = fields[8].split(":")
                if tumor_idx is not None and 9 + tumor_idx < len(fields):
                    tumor_fields = self._parse_sample(fmt_keys, fields[9 + tumor_idx])
                if normal_idx is not None and 9 + normal_idx < len(fields):
                    normal_fields = self._parse_sample(fmt_keys, fields[9 + normal_idx])

            # Split multi-allelic sites. `allele_index` is the 0-based
            # index into the ALT list, used to pick this allele's value
            # out of Number=A / Number=R fields.
            allele_index = -1
            for alt in alts:
                alt = alt.strip()
                allele_index += 1
                if alt == "." or alt == "*":
                    continue

                variant = Variant(
                    chrom=chrom,
                    pos=pos,
                    id=var_id,
                    ref=ref,
                    alt=alt,
                    qual=qual,
                    filter_status=filt,
                    info=info,
                )

                self._apply_sample_evidence(
                    variant, tumor_fields, normal_fields, allele_index
                )
                self._apply_somatic_evidence(variant, info, allele_index)
                self._classify_somatic(variant)

                # Extract annotations from INFO if present
                self._extract_annotations(variant, info)

                # Extract dbSNP ID
                if var_id.startswith("rs"):
                    variant.dbsnp_id = var_id

                variants.append(variant)

        if self.is_somatic_vcf and not self.filters_applied:
            logger.warning(
                "VCF declares tumor '%s' / normal '%s' but every record has an "
                "empty FILTER column: this callset has not been through "
                "FilterMutectCalls. Raw Mutect2 output is emitted down to "
                "TLOD>=3.0 and is dominated by low-evidence sites. Use "
                "filter_somatic() before treating these as real mutations.",
                self.tumor_sample, self.normal_sample,
            )

        logger.info(
            "Parsed %d variants from VCF (samples=%s, tumor=%s, normal=%s, "
            "filtered=%s)",
            len(variants), self.sample_names or "none",
            self.tumor_sample, self.normal_sample, self.filters_applied,
        )
        return variants

    # ── Header handling ────────────────────────────────────────────

    def _parse_meta(self, line: str) -> None:
        """Extract the meta-information lines we act on."""
        if line.startswith("##tumor_sample="):
            self.tumor_sample = line.split("=", 1)[1].strip()
        elif line.startswith("##normal_sample="):
            self.normal_sample = line.split("=", 1)[1].strip()
        elif line.startswith("##source="):
            self.caller = line.split("=", 1)[1].strip()

    def _resolve_sample_columns(self) -> Tuple[Optional[int], Optional[int]]:
        """Map the tumor/normal declarations onto sample column offsets.

        Returns (tumor_offset, normal_offset), each 0-based within the
        sample columns (i.e. column 9 + offset in the data line), or None
        when that side is not present.

        Resolution is by NAME, never by position: Mutect2 orders sample
        columns by the order the BAMs were given, so the tumor is not
        reliably first. ``seqc2_demo.vcf`` is ordered normal-then-tumor.
        """
        names = self.sample_names
        if not names:
            return None, None

        tumor_idx: Optional[int] = None
        normal_idx: Optional[int] = None

        if self.tumor_sample and self.tumor_sample in names:
            tumor_idx = names.index(self.tumor_sample)
        if self.normal_sample and self.normal_sample in names:
            normal_idx = names.index(self.normal_sample)

        # Two columns and only one side named: the other column is the
        # other side.
        if len(names) == 2:
            if tumor_idx is None and normal_idx is not None:
                tumor_idx = 1 - normal_idx
            elif normal_idx is None and tumor_idx is not None:
                normal_idx = 1 - tumor_idx

        # Single-sample VCF (germline callset, or tumor-only): that one
        # column is the sample of interest. There is no normal to compare
        # against, which _classify_somatic reflects as UNKNOWN.
        if tumor_idx is None and len(names) == 1:
            tumor_idx = 0

        if tumor_idx is None and names:
            logger.warning(
                "VCF has %d sample columns %s but no ##tumor_sample header; "
                "defaulting to the first column. If this is a matched "
                "tumor/normal callset the wrong sample may be read.",
                len(names), names,
            )
            tumor_idx = 0

        return tumor_idx, normal_idx

    # ── Per-sample / per-allele extraction ─────────────────────────

    @staticmethod
    def _parse_sample(fmt_keys: List[str], sample_field: str) -> Dict[str, str]:
        """Zip a FORMAT key list against one sample's colon-delimited values."""
        values = sample_field.split(":")
        return {k: values[i] for i, k in enumerate(fmt_keys) if i < len(values)}

    @staticmethod
    def _nth(raw: Optional[str], index: int) -> Optional[str]:
        """Pick element `index` out of a comma-delimited VCF value."""
        if raw is None or raw in (".", ""):
            return None
        parts = raw.split(",")
        if index < 0 or index >= len(parts):
            return None
        val = parts[index].strip()
        return None if val in (".", "") else val

    @classmethod
    def _float_at(cls, raw: Optional[str], index: int) -> Optional[float]:
        val = cls._nth(raw, index)
        try:
            return float(val) if val is not None else None
        except ValueError:
            return None

    @classmethod
    def _int_at(cls, raw: Optional[str], index: int) -> Optional[int]:
        val = cls._nth(raw, index)
        try:
            return int(val) if val is not None else None
        except ValueError:
            return None

    def _apply_sample_evidence(
        self,
        variant: Variant,
        tumor_fields: Dict[str, str],
        normal_fields: Dict[str, str],
        allele_index: int,
    ) -> None:
        """Populate tumor/normal depth, counts and allele fraction."""
        for fields_, prefix in ((tumor_fields, "tumor"), (normal_fields, "normal")):
            if not fields_:
                continue

            genotype = fields_.get("GT", ".")
            # AD is Number=R: [ref, alt1, alt2, ...]
            ref_count = self._int_at(fields_.get("AD"), 0)
            alt_count = self._int_at(fields_.get("AD"), allele_index + 1)
            # AF is Number=A: [alt1, alt2, ...]
            reported_af = self._float_at(fields_.get("AF"), allele_index)
            depth = self._int_at(fields_.get("DP"), 0)

            # Observed fraction straight from the read counts. This is the
            # only trustworthy fraction for the NORMAL column: Mutect2
            # defines FORMAT/AF as "allele fractions of alternate alleles
            # in the tumor" and emits that same tumor-derived value in the
            # normal's column, where it is ~0.03 even at sites with zero
            # alt reads in the normal. Believing it there manufactures
            # germline evidence out of nothing.
            observed_af = None
            if ref_count is not None and alt_count is not None:
                total = ref_count + alt_count
                observed_af = (alt_count / total) if total > 0 else None

            if prefix == "normal":
                af = observed_af if observed_af is not None else reported_af
            else:
                af = reported_af if reported_af is not None else observed_af

            setattr(variant, f"{prefix}_genotype", genotype)
            setattr(variant, f"{prefix}_ref_count", ref_count)
            setattr(variant, f"{prefix}_alt_count", alt_count)
            setattr(variant, f"{prefix}_af", af)
            setattr(variant, f"{prefix}_depth", depth)

        # `genotype` is the long-standing public field; keep it pointing at
        # the tumor so existing callers get the clinically relevant side.
        if variant.tumor_genotype is not None:
            variant.genotype = variant.tumor_genotype

    def _apply_somatic_evidence(
        self, variant: Variant, info: Dict[str, str], allele_index: int
    ) -> None:
        """Pull the caller's somatic statistics out of INFO."""
        ev = variant.somatic_evidence
        ev.tumor_lod = self._float_at(info.get("TLOD"), allele_index)
        ev.normal_lod = self._float_at(info.get("NLOD"), allele_index)
        ev.normal_artifact_lod = self._float_at(info.get("NALOD"), allele_index)
        ev.population_af_phred = self._float_at(info.get("POPAF"), allele_index)
        ev.germline_qual = self._float_at(info.get("GERMQ"), 0)
        ev.strand_qual = self._float_at(info.get("STRANDQ"), 0)
        ev.seq_qual = self._float_at(info.get("SEQQ"), 0)

    def _classify_somatic(self, variant: Variant) -> None:
        """Assign a somatic status, recording why.

        This is an approximation of GATK FilterMutectCalls using the
        thresholds from Cibulskis et al. 2013 — it is NOT a replacement
        for it, and does not model contamination, orientation bias or the
        panel of normals. It exists so that an unfiltered callset is
        triaged explicitly rather than accepted wholesale.
        """
        # An explicit caller FILTER wins: it came from a tool that had
        # more information than we do here.
        if variant.filter_status not in (".", "", "PASS"):
            variant.somatic_status = "ARTIFACT"
            variant.somatic_filter_reasons = [
                f"caller_filter:{variant.filter_status}"
            ]
            return

        # No normal to compare against => the tumor/normal question is
        # simply unanswerable from this file.
        if variant.normal_genotype is None and variant.normal_af is None:
            variant.somatic_status = "UNKNOWN"
            variant.somatic_filter_reasons = ["no_matched_normal"]
            return

        germline_reasons: List[str] = []
        artifact_reasons: List[str] = []
        weak_reasons: List[str] = []
        ev = variant.somatic_evidence

        # ── Is the allele present in the germline? ─────────────────
        # LOD_N is the primary evidence: it is the caller's own likelihood
        # ratio that the normal is homozygous reference.
        if ev.normal_lod is not None and ev.normal_lod < NORMAL_LOD_THRESHOLD:
            germline_reasons.append(f"normal_lod<{NORMAL_LOD_THRESHOLD}")

        # Direct read support in the normal — count AND fraction together.
        if (variant.normal_alt_count is not None
                and variant.normal_af is not None
                and variant.normal_alt_count >= MAX_NORMAL_ALT_COUNT
                and variant.normal_af >= MAX_NORMAL_AF):
            germline_reasons.append("alt_reads_in_normal")

        # An alt-carrying genotype call in the normal is explicit.
        if variant.normal_genotype and re.search(r"[1-9]", variant.normal_genotype):
            germline_reasons.append("alt_allele_in_normal")

        # ── Is it an artifact? ─────────────────────────────────────
        if (ev.normal_artifact_lod is not None
                and ev.normal_artifact_lod <= NORMAL_ARTIFACT_LOD_THRESHOLD):
            artifact_reasons.append("normal_artifact")

        # ── Is there enough evidence it is real in the tumor? ──────
        if ev.tumor_lod is not None and ev.tumor_lod < TUMOR_LOD_THRESHOLD:
            weak_reasons.append(f"tumor_lod<{TUMOR_LOD_THRESHOLD}")

        variant.somatic_filter_reasons = (
            germline_reasons + artifact_reasons + weak_reasons
        )

        # Order matters: germline is the finding that most changes what a
        # caller should do with the variant (never put it in a vaccine), so
        # it is reported even when the call is also weak.
        if germline_reasons:
            variant.somatic_status = "GERMLINE"
        elif artifact_reasons:
            variant.somatic_status = "ARTIFACT"
        elif weak_reasons:
            variant.somatic_status = "LOW_EVIDENCE"
        else:
            variant.somatic_status = "SOMATIC"

    # ── Filters ────────────────────────────────────────────────────

    def filter_pass(self, variants: List[Variant]) -> List[Variant]:
        """Keep variants the caller marked PASS.

        Unfiltered records (FILTER ".") are retained for backward
        compatibility — many callers emit "." for everything and dropping
        them would empty the callset — but this is logged, because
        "unfiltered" is being treated as "passed" and that is an
        assumption, not a fact.
        """
        unfiltered = sum(1 for v in variants if v.filter_status in (".", ""))
        if unfiltered:
            logger.warning(
                "filter_pass(): %d/%d records have no FILTER value and are "
                "being kept as if they passed. Run FilterMutectCalls, or use "
                "filter_somatic(), before relying on these.",
                unfiltered, len(variants),
            )
        return [v for v in variants if v.filter_status in ("PASS", ".", "")]

    def filter_somatic(self, variants: List[Variant],
                       include_unknown: bool = False) -> List[Variant]:
        """Keep only variants supported as somatic by the tumor/normal comparison.

        Args:
            variants: Parsed variants.
            include_unknown: If True, also keep variants with no matched
                normal (status UNKNOWN). Off by default — a tumor-only
                callset cannot separate somatic from rare germline, and
                including those in neoantigen design risks targeting a
                self antigen.
        """
        keep = {"SOMATIC"}
        if include_unknown:
            keep.add("UNKNOWN")
        kept = [v for v in variants if v.somatic_status in keep]
        logger.info(
            "filter_somatic(): kept %d/%d variants (%s)",
            len(kept), len(variants), ", ".join(sorted(keep)),
        )
        return kept

    def filter_coding(self, variants: List[Variant]) -> List[Variant]:
        """Keep only coding variants (those with gene/protein annotations)."""
        return [v for v in variants if v.is_coding]

    def filter_cancer_genes(self, variants: List[Variant],
                            cancer_type: str = "prostate") -> List[Variant]:
        """Keep variants in known cancer driver genes."""
        return [v for v in variants if v.is_cancer_driver]

    def filter_by_impact(self, variants: List[Variant],
                         min_impact: str = "MODERATE") -> List[Variant]:
        """Filter by predicted impact level."""
        levels = {"HIGH": 3, "MODERATE": 2, "LOW": 1, "MODIFIER": 0, "UNKNOWN": -1}
        threshold = levels.get(min_impact, 0)
        return [v for v in variants if levels.get(v.impact, -1) >= threshold]

    def variant_summary(self, variants: List[Variant]) -> Dict:
        """Generate summary statistics for a variant list."""
        total = len(variants)
        snvs = sum(1 for v in variants if v.is_snv)
        indels = sum(1 for v in variants if v.is_indel)
        coding = sum(1 for v in variants if v.is_coding)
        drivers = sum(1 for v in variants if v.is_cancer_driver)
        het = sum(1 for v in variants if "0/1" in v.genotype or "0|1" in v.genotype)
        hom = sum(1 for v in variants if "1/1" in v.genotype or "1|1" in v.genotype)

        # Chromosome distribution
        by_chrom = {}
        for v in variants:
            by_chrom[v.chrom] = by_chrom.get(v.chrom, 0) + 1

        # Consequence distribution
        by_consequence = {}
        for v in variants:
            c = v.consequence or "unknown"
            by_consequence[c] = by_consequence.get(c, 0) + 1

        # Somatic status distribution — the tumor/normal verdict, so a
        # caller can see how much of the callset actually survives it.
        by_somatic_status = {}
        for v in variants:
            by_somatic_status[v.somatic_status] = (
                by_somatic_status.get(v.somatic_status, 0) + 1
            )

        return {
            "total_variants": total,
            "snvs": snvs,
            "indels": indels,
            "coding_variants": coding,
            "cancer_drivers": drivers,
            "heterozygous": het,
            "homozygous": hom,
            "by_chromosome": by_chrom,
            "by_consequence": by_consequence,
            "by_somatic_status": by_somatic_status,
            "somatic_variants": by_somatic_status.get("SOMATIC", 0),
            "tumor_sample": self.tumor_sample,
            "normal_sample": self.normal_sample,
            "filters_applied": self.filters_applied,
        }

    @staticmethod
    def _parse_info(info_str: str) -> Dict[str, str]:
        """Parse VCF INFO field into key-value dict."""
        info = {}
        if info_str == ".":
            return info
        for item in info_str.split(";"):
            if "=" in item:
                key, val = item.split("=", 1)
                info[key] = val
            else:
                info[item] = "true"
        return info

    def _extract_annotations(self, variant: Variant, info: Dict):
        """Extract functional annotations from INFO field.

        Supports:
        - ANN field (SnpEff format)
        - CSQ field (VEP/Ensembl format)
        - GENE, AA_CHANGE, CONSEQUENCE direct fields
        """
        # Direct annotation fields (simple annotated VCFs)
        if "GENE" in info:
            variant.gene = info["GENE"]
        if "AA_CHANGE" in info:
            variant.protein_change = info["AA_CHANGE"]
        if "CONSEQUENCE" in info:
            variant.consequence = info["CONSEQUENCE"]
            variant.is_coding = is_protein_altering(variant.consequence)

        # SnpEff ANN field
        if "ANN" in info:
            self._parse_snpeff_ann(variant, info["ANN"])

        # VEP CSQ field
        if "CSQ" in info:
            self._parse_vep_csq(variant, info["CSQ"])

        # ClinVar
        if "CLNSIG" in info:
            variant.clinvar_significance = info["CLNSIG"]

        # COSMIC
        if "COSMIC_ID" in info:
            variant.cosmic_id = info["COSMIC_ID"]

    def _parse_snpeff_ann(self, variant: Variant, ann_str: str):
        """Parse SnpEff ANN field.

        Format: Allele|Annotation|Annotation_Impact|Gene_Name|Gene_ID|
                Feature_Type|Feature_ID|Transcript_Biotype|Rank|
                HGVS.c|HGVS.p|cDNA.pos|CDS.pos|AA.pos|Distance|...
        """
        # Take first annotation (highest impact)
        first = ann_str.split(",")[0]
        parts = first.split("|")
        if len(parts) >= 11:
            variant.consequence = parts[1].lower().replace("_variant", "")
            variant.impact = parts[2]
            variant.gene = parts[3]
            variant.transcript = parts[6]
            hgvs_p = parts[10]
            if hgvs_p:
                variant.protein_change = hgvs_p
            # Impact is SnpEff's own summary; the consequence term is the
            # authority when the two disagree.
            variant.is_coding = (variant.impact in ("HIGH", "MODERATE")
                                 or is_protein_altering(variant.consequence))

    def _parse_vep_csq(self, variant: Variant, csq_str: str):
        """Parse VEP CSQ field (simplified — takes first consequence)."""
        first = csq_str.split(",")[0]
        parts = first.split("|")
        # VEP format varies, but common fields:
        # Consequence|Impact|SYMBOL|Gene|Feature_type|Feature|...
        if len(parts) >= 4:
            variant.consequence = parts[0].lower().replace("_variant", "")
            variant.impact = parts[1] if len(parts) > 1 else "UNKNOWN"
            variant.gene = parts[2] if len(parts) > 2 else None
            variant.is_coding = (variant.impact in ("HIGH", "MODERATE")
                                 or is_protein_altering(variant.consequence))
