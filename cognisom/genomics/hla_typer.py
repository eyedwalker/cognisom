"""
HLA Typer
=========

Extract HLA alleles from patient VCF data for neoantigen prediction.

HLA (Human Leukocyte Antigen) alleles determine which peptides a patient's
immune system can present to T-cells. Accurate HLA typing is essential for
predicting which tumor neoantigens will trigger an immune response.

Supports two modes:
1. VCF-based: Extract HLA alleles from chromosome 6 variant annotations
2. Default: Use common population alleles when VCF data is insufficient

References:
- NetMHCpan 4.1 (Reynisson et al., NAR 2020)
- OptiType (Szolek et al., Bioinformatics 2014)
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Common HLA alleles by locus with population frequencies (Caucasian)
# Used as defaults when VCF-based typing is unavailable
COMMON_HLA_ALLELES: Dict[str, List[Tuple[str, float]]] = {
    "A": [
        ("A*02:01", 0.29),  # Most common globally
        ("A*01:01", 0.16),
        ("A*03:01", 0.13),
        ("A*24:02", 0.10),
        ("A*11:01", 0.07),
        ("A*68:01", 0.05),
        ("A*32:01", 0.04),
        ("A*31:01", 0.03),
        ("A*26:01", 0.03),
        ("A*29:02", 0.03),
    ],
    "B": [
        ("B*07:02", 0.13),
        ("B*08:01", 0.10),
        ("B*44:02", 0.09),
        ("B*44:03", 0.07),
        ("B*35:01", 0.06),
        ("B*15:01", 0.06),
        ("B*51:01", 0.05),
        ("B*40:01", 0.05),
        ("B*18:01", 0.04),
        ("B*57:01", 0.03),
    ],
    "C": [
        ("C*07:01", 0.15),
        ("C*07:02", 0.13),
        ("C*04:01", 0.11),
        ("C*05:01", 0.09),
        ("C*06:02", 0.08),
        ("C*03:04", 0.07),
        ("C*03:03", 0.06),
        ("C*12:03", 0.05),
        ("C*01:02", 0.04),
        ("C*02:02", 0.04),
    ],
}

# Predefined HLA profiles for synthetic/demo patients
SYNTHETIC_HLA_PROFILES: Dict[str, List[str]] = {
    "COGNISOM-DEMO-001": [
        "HLA-A*02:01", "HLA-A*24:02",
        "HLA-B*07:02", "HLA-B*44:03",
        "HLA-C*05:01", "HLA-C*07:02",
    ],
    "default": [
        "HLA-A*02:01", "HLA-A*03:01",
        "HLA-B*07:02", "HLA-B*08:01",
        "HLA-C*07:01", "HLA-C*07:02",
    ],
}

# MHC-I binding groove properties per allele (simplified)
# peptide_length: preferred peptide length for binding
# anchor_residues: positions with strong amino acid preferences
HLA_BINDING_PROPERTIES: Dict[str, Dict] = {
    "HLA-A*02:01": {
        "peptide_lengths": [9, 10],
        "anchor_positions": {2: ["L", "M", "V", "I"], 9: ["V", "L", "I"]},
        "binding_threshold_nm": 500,
    },
    "HLA-A*24:02": {
        "peptide_lengths": [9, 10],
        "anchor_positions": {2: ["Y", "F"], 9: ["F", "L", "I"]},
        "binding_threshold_nm": 500,
    },
    "HLA-A*03:01": {
        "peptide_lengths": [9, 10],
        "anchor_positions": {2: ["L", "V", "M"], 9: ["K", "R"]},
        "binding_threshold_nm": 500,
    },
    "HLA-B*07:02": {
        "peptide_lengths": [9, 10],
        "anchor_positions": {2: ["P"], 9: ["L", "M"]},
        "binding_threshold_nm": 500,
    },
    "HLA-B*44:03": {
        "peptide_lengths": [9, 10, 11],
        "anchor_positions": {2: ["E"], 9: ["Y", "F", "W"]},
        "binding_threshold_nm": 500,
    },
    "HLA-C*07:02": {
        "peptide_lengths": [9],
        "anchor_positions": {2: ["R", "K"], 9: ["L", "F"]},
        "binding_threshold_nm": 500,
    },
    "HLA-C*05:01": {
        "peptide_lengths": [9],
        "anchor_positions": {2: ["A", "V"], 9: ["L", "V"]},
        "binding_threshold_nm": 500,
    },
}


class HLATyper:
    """Extract or assign HLA alleles for a patient.

    In a full clinical pipeline, HLA typing would use specialized tools
    (OptiType, HLA-HD, or Illumina TruSight HLA) on raw sequencing reads.
    This implementation provides two approaches:

    1. VCF-based: Extract HLA gene variants from chromosome 6 annotations
    2. Synthetic/default: Assign representative HLA alleles for simulation

    Example:
        typer = HLATyper()
        alleles = typer.type_from_variants(variants, patient_id="MAYO-001")
        print(alleles)
        # ['HLA-A*02:01', 'HLA-A*24:02', 'HLA-B*07:02', ...]
    """

    def type_from_variants(
        self,
        variants: list,
        patient_id: str = "anonymous",
    ) -> List[str]:
        """Extract HLA alleles from VCF variants.

        Looks for variants on chromosome 6 in HLA genes (HLA-A, HLA-B, HLA-C).
        If insufficient HLA data is found in the VCF, falls back to a
        representative profile.

        Args:
            variants: List of Variant objects from VCFParser.
            patient_id: Patient identifier for synthetic profile lookup.

        Returns:
            List of HLA allele strings, e.g. ["HLA-A*02:01", "HLA-A*24:02", ...]
        """
        # Check for predefined synthetic profiles
        if patient_id in SYNTHETIC_HLA_PROFILES:
            alleles = SYNTHETIC_HLA_PROFILES[patient_id]
            logger.info(f"Using predefined HLA profile for {patient_id}: {alleles}")
            return alleles

        # Try to extract HLA info from VCF variants on chr6
        hla_variants = [
            v for v in variants
            if (getattr(v, "chrom", "") in ("chr6", "6") and
                getattr(v, "gene", "") and
                getattr(v, "gene", "").startswith("HLA"))
        ]

        if hla_variants:
            alleles = self._infer_from_hla_variants(hla_variants)
            if len(alleles) >= 4:  # At least 2 loci typed
                logger.info(f"Inferred HLA alleles from VCF: {alleles}")
                return alleles

        # Fallback: assign common population alleles
        alleles = self._assign_default_alleles()
        logger.info(f"Using default HLA alleles for {patient_id}: {alleles}")
        return alleles

    def _infer_from_hla_variants(self, hla_variants: list) -> List[str]:
        """Attempt to infer HLA alleles from VCF variant annotations.

        This is a simplified inference — clinical HLA typing requires
        specialized algorithms on raw reads. We extract what we can from
        the INFO field annotations.
        """
        alleles = []
        seen_loci = set()

        for v in hla_variants:
            gene = getattr(v, "gene", "")
            # Extract locus (A, B, C)
            match = re.match(r"HLA-([ABC])", gene)
            if not match:
                continue

            locus = match.group(1)
            if locus in seen_loci:
                continue

            # Check if INFO contains specific allele annotation
            info = getattr(v, "info", {})
            allele_info = info.get("HLA_ALLELE", "")

            if allele_info and "*" in allele_info:
                alleles.append(f"HLA-{allele_info}")
            else:
                # Assign most common alleles for this locus
                common = COMMON_HLA_ALLELES.get(locus, [])
                if len(common) >= 2:
                    alleles.append(f"HLA-{common[0][0]}")
                    alleles.append(f"HLA-{common[1][0]}")
                    seen_loci.add(locus)

        return alleles

    def _assign_default_alleles(self) -> List[str]:
        """Assign default HLA alleles based on population frequencies."""
        return list(SYNTHETIC_HLA_PROFILES["default"])

    @staticmethod
    def get_binding_properties(allele: str) -> Optional[Dict]:
        """Get binding properties for an HLA allele.

        Returns dict with peptide_lengths, anchor_positions, and
        binding_threshold_nm, or None if allele is not characterized.
        """
        # Normalize format: "A*02:01" → "HLA-A*02:01"
        if not allele.startswith("HLA-"):
            allele = f"HLA-{allele}"
        return HLA_BINDING_PROPERTIES.get(allele)

    @staticmethod
    def format_alleles(alleles: List[str]) -> str:
        """Format alleles for display.

        Returns a human-readable string like:
        "HLA-A: *02:01 / *24:02, HLA-B: *07:02 / *44:03, HLA-C: *05:01 / *07:02"
        """
        by_locus: Dict[str, List[str]] = {}
        for allele in alleles:
            # Parse "HLA-A*02:01" → locus="A", spec="*02:01"
            match = re.match(r"HLA-([ABC])\*(.+)", allele)
            if match:
                locus = match.group(1)
                spec = f"*{match.group(2)}"
                by_locus.setdefault(locus, []).append(spec)

        parts = []
        for locus in ["A", "B", "C"]:
            specs = by_locus.get(locus, [])
            if specs:
                parts.append(f"HLA-{locus}: {' / '.join(specs)}")

        return ", ".join(parts)
