"""
VCF Parser
==========

Parse Variant Call Format (VCF) files into structured variant records.
Supports VCF 4.1/4.2/4.3 format with annotation fields from common
variant callers (GATK HaplotypeCaller, DeepVariant, Strelka2).

Pure Python implementation — no external VCF library required.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


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
    genotype: str = "."  # e.g. "0/1", "1/1"

    # Annotation fields (populated by VariantAnnotator)
    gene: Optional[str] = None
    consequence: Optional[str] = None  # missense, nonsense, synonymous, etc.
    protein_change: Optional[str] = None  # e.g. "p.T877A"
    transcript: Optional[str] = None
    cosmic_id: Optional[str] = None
    clinvar_significance: Optional[str] = None
    dbsnp_id: Optional[str] = None

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
        return " ".join(parts)


class VCFParser:
    """Parse VCF files into structured Variant records.

    Supports:
    - Standard VCF 4.1/4.2/4.3 format
    - Multi-allelic sites (splits into separate Variant records)
    - INFO field parsing (ANN/CSQ for functional annotations)
    - Genotype (GT) extraction from FORMAT/sample columns

    Example:
        parser = VCFParser()
        variants = parser.parse_file("patient.vcf")
        coding = parser.filter_coding(variants)
        print(f"{len(coding)} coding variants out of {len(variants)} total")
    """

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
        variants = []
        header_columns = None

        for line in vcf_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue

            # Header lines
            if line.startswith("##"):
                continue

            # Column header
            if line.startswith("#CHROM"):
                header_columns = line[1:].split("\t")
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

            # Parse INFO field
            info = self._parse_info(info_str)

            # Extract genotype if FORMAT + sample columns present
            genotype = "."
            if len(fields) >= 10:
                fmt = fields[8].split(":")
                sample = fields[9].split(":")
                gt_idx = fmt.index("GT") if "GT" in fmt else -1
                if gt_idx >= 0 and gt_idx < len(sample):
                    genotype = sample[gt_idx]

            # Split multi-allelic sites
            for alt in alts:
                alt = alt.strip()
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
                    genotype=genotype,
                )

                # Extract annotations from INFO if present
                self._extract_annotations(variant, info)

                # Extract dbSNP ID
                if var_id.startswith("rs"):
                    variant.dbsnp_id = var_id

                variants.append(variant)

        logger.info(f"Parsed {len(variants)} variants from VCF")
        return variants

    def filter_pass(self, variants: List[Variant]) -> List[Variant]:
        """Keep only variants that PASS filters."""
        return [v for v in variants if v.filter_status in ("PASS", ".")]

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
            variant.is_coding = variant.consequence in (
                "missense", "nonsense", "frameshift",
                "splice_donor", "splice_acceptor", "start_lost", "stop_lost",
            )

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
            variant.is_coding = variant.impact in ("HIGH", "MODERATE")

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
            variant.is_coding = variant.impact in ("HIGH", "MODERATE")
