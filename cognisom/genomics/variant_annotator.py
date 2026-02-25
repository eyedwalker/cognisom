"""
Variant Annotator
=================

Annotate genomic variants with gene names, protein consequences,
cancer driver status, and clinical significance.

Uses a built-in database of prostate cancer driver genes and
the standard codon table for amino acid change prediction.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from .vcf_parser import Variant

logger = logging.getLogger(__name__)


# Prostate cancer driver genes with known roles and hotspot mutations
PROSTATE_CANCER_DRIVERS = {
    "AR": {
        "full_name": "Androgen Receptor",
        "role": "oncogene",
        "chromosome": "chrX",
        "description": "Primary driver of prostate cancer growth. Ligand-activated "
                       "transcription factor in androgen signaling pathway.",
        "hotspot_mutations": {
            "T877A": "Broadens ligand specificity — responds to anti-androgens as agonists",
            "L702H": "Glucocorticoid-responsive — activated by cortisol",
            "H875Y": "Increased transactivation in castrate conditions",
            "F877L": "Enzalutamide resistance mutation",
            "W742C": "Bicalutamide agonist mutation",
        },
        "clinical_significance": "Target of ADT (androgen deprivation therapy). "
                                "Mutations drive castration-resistant disease (CRPC).",
        "therapies": ["enzalutamide", "abiraterone", "darolutamide"],
    },
    "TP53": {
        "full_name": "Tumor Protein P53",
        "role": "tumor_suppressor",
        "chromosome": "chr17",
        "description": "Guardian of the genome. Activates DNA repair, cell cycle arrest, "
                       "and apoptosis in response to DNA damage.",
        "hotspot_mutations": {
            "R248W": "DNA contact mutant — loss of DNA binding",
            "R273H": "DNA contact mutant — loss of DNA binding",
            "R175H": "Structural mutant — protein misfolding",
            "G245S": "Structural mutant — altered protein conformation",
            "Y220C": "Druggable pocket — target of APR-246/eprenetapopt",
        },
        "clinical_significance": "Mutated in ~50% of mCRPC. Associated with aggressive disease, "
                                "poor prognosis, and therapy resistance.",
        "therapies": ["eprenetapopt (APR-246)", "adavosertib (Wee1 inhibitor)"],
    },
    "PTEN": {
        "full_name": "Phosphatase and Tensin Homolog",
        "role": "tumor_suppressor",
        "chromosome": "chr10",
        "description": "Negative regulator of PI3K/AKT/mTOR pathway. "
                       "Lipid phosphatase that antagonizes cell growth signaling.",
        "hotspot_mutations": {
            "R130*": "Premature stop — complete loss of function",
            "R233*": "Premature stop — complete loss of function",
        },
        "clinical_significance": "Deleted/mutated in ~40% of mCRPC. Loss activates PI3K/AKT "
                                "pathway → cell survival and proliferation.",
        "therapies": ["ipatasertib (AKT inhibitor)", "alpelisib (PI3K inhibitor)"],
    },
    "BRCA2": {
        "full_name": "BRCA2 DNA Repair Associated",
        "role": "tumor_suppressor",
        "chromosome": "chr13",
        "description": "Homologous recombination DNA repair. Essential for error-free "
                       "repair of double-strand breaks.",
        "hotspot_mutations": {},
        "clinical_significance": "Mutated in ~12% of mCRPC. Confers sensitivity to PARP "
                                "inhibitors (synthetic lethality) and platinum chemotherapy.",
        "therapies": ["olaparib", "rucaparib", "niraparib", "talazoparib"],
    },
    "BRCA1": {
        "full_name": "BRCA1 DNA Repair Associated",
        "role": "tumor_suppressor",
        "chromosome": "chr17",
        "description": "Homologous recombination DNA repair and tumor suppression.",
        "hotspot_mutations": {},
        "clinical_significance": "Mutated in ~5% of mCRPC. PARP inhibitor sensitive.",
        "therapies": ["olaparib", "rucaparib"],
    },
    "SPOP": {
        "full_name": "Speckle-type POZ Protein",
        "role": "tumor_suppressor",
        "chromosome": "chr17",
        "description": "E3 ubiquitin ligase substrate adaptor. Targets AR, ERG, "
                       "and other oncoproteins for degradation.",
        "hotspot_mutations": {
            "Y87C": "Substrate binding domain — impaired degradation of AR",
            "F102C": "Substrate binding domain — impaired degradation of AR",
            "F133V": "Substrate binding domain — stabilizes oncoproteins",
            "W131G": "Substrate binding domain",
        },
        "clinical_significance": "Mutated in ~10% of primary prostate cancer. Mutually "
                                "exclusive with TMPRSS2-ERG fusions.",
        "therapies": ["BET inhibitors (experimental)"],
    },
    "FOXA1": {
        "full_name": "Forkhead Box A1",
        "role": "oncogene",
        "chromosome": "chr14",
        "description": "Pioneer transcription factor that opens chromatin for AR binding. "
                       "Mutations alter AR transcriptional program.",
        "hotspot_mutations": {
            "I176M": "Wing2 region — altered DNA binding specificity",
        },
        "clinical_significance": "Mutated in ~10% of mCRPC. Gain-of-function mutations "
                                "reprogram AR binding and drive castration resistance.",
        "therapies": [],
    },
    "CDK12": {
        "full_name": "Cyclin Dependent Kinase 12",
        "role": "tumor_suppressor",
        "chromosome": "chr17",
        "description": "Regulates transcription elongation and RNA splicing. "
                       "Loss causes focal tandem duplications (FTDs).",
        "hotspot_mutations": {},
        "clinical_significance": "Biallelic loss in ~7% of mCRPC. Creates neoantigen-rich "
                                "tumors — potentially immunotherapy responsive.",
        "therapies": ["checkpoint inhibitors (experimental)"],
    },
    "RB1": {
        "full_name": "Retinoblastoma Transcriptional Corepressor 1",
        "role": "tumor_suppressor",
        "chromosome": "chr13",
        "description": "Cell cycle checkpoint. Prevents entry into S-phase by "
                       "sequestering E2F transcription factors.",
        "hotspot_mutations": {},
        "clinical_significance": "Lost in ~10% of mCRPC. Associated with neuroendocrine "
                                "differentiation and aggressive disease.",
        "therapies": ["CDK4/6 inhibitors (experimental in CRPC)"],
    },
    "MYC": {
        "full_name": "MYC Proto-Oncogene",
        "role": "oncogene",
        "chromosome": "chr8",
        "description": "Master transcriptional regulator. Amplification drives "
                       "cell proliferation and metabolic reprogramming.",
        "hotspot_mutations": {},
        "clinical_significance": "Amplified in ~30% of mCRPC. Associated with aggressive "
                                "disease and therapy resistance.",
        "therapies": ["BET inhibitors (experimental)", "omomyc (experimental)"],
    },
    "ERG": {
        "full_name": "ETS Transcription Factor ERG",
        "role": "oncogene",
        "chromosome": "chr21",
        "description": "ETS family transcription factor. TMPRSS2-ERG fusion "
                       "is the most common genomic alteration in prostate cancer.",
        "hotspot_mutations": {},
        "clinical_significance": "TMPRSS2-ERG fusion in ~50% of prostate cancers. "
                                "Androgen-driven overexpression of ERG.",
        "therapies": ["PARP inhibitors (synergistic in ERG+ tumors)"],
    },
    "ATM": {
        "full_name": "ATM Serine/Threonine Kinase",
        "role": "tumor_suppressor",
        "chromosome": "chr11",
        "description": "DNA damage response kinase. Activates checkpoint signaling "
                       "in response to double-strand breaks.",
        "hotspot_mutations": {},
        "clinical_significance": "Mutated in ~5% of mCRPC. May respond to PARP inhibitors "
                                "and platinum chemotherapy.",
        "therapies": ["olaparib (some evidence)", "platinum chemotherapy"],
    },
    "PIK3CA": {
        "full_name": "PI3K Catalytic Subunit Alpha",
        "role": "oncogene",
        "chromosome": "chr3",
        "description": "Catalytic subunit of PI3K. Gain-of-function mutations "
                       "hyperactivate PI3K/AKT/mTOR signaling.",
        "hotspot_mutations": {
            "E545K": "Helical domain — constitutive PI3K activation",
            "H1047R": "Kinase domain — constitutive PI3K activation",
        },
        "clinical_significance": "Mutated in ~5% of mCRPC. Activating mutations "
                                "may respond to PI3K/AKT inhibitors.",
        "therapies": ["alpelisib", "ipatasertib"],
    },
    "APC": {
        "full_name": "APC Regulator of WNT Signaling",
        "role": "tumor_suppressor",
        "chromosome": "chr5",
        "description": "Negative regulator of Wnt/beta-catenin pathway. "
                       "Loss activates Wnt signaling.",
        "hotspot_mutations": {},
        "clinical_significance": "Mutated in ~10% of mCRPC. Wnt pathway activation "
                                "drives lineage plasticity.",
        "therapies": [],
    },
}

# All known cancer driver gene symbols for quick lookup
CANCER_DRIVER_GENES = set(PROSTATE_CANCER_DRIVERS.keys())


class VariantAnnotator:
    """Annotate variants with gene information and cancer relevance.

    Uses built-in prostate cancer driver gene database and
    optional VCF INFO field annotations.

    Example:
        from cognisom.genomics import VCFParser
        from cognisom.genomics.variant_annotator import VariantAnnotator

        parser = VCFParser()
        variants = parser.parse_file("patient.vcf")

        annotator = VariantAnnotator()
        annotated = annotator.annotate(variants)

        drivers = [v for v in annotated if v.is_cancer_driver]
        for d in drivers:
            print(d.summary())
    """

    def __init__(self, cancer_type: str = "prostate"):
        self.cancer_type = cancer_type
        self.driver_db = PROSTATE_CANCER_DRIVERS

    def annotate(self, variants: List[Variant]) -> List[Variant]:
        """Annotate a list of variants with cancer gene information.

        Fills in: gene, consequence, is_coding, is_cancer_driver, impact.
        Returns the same list (modified in place).
        """
        for v in variants:
            self._annotate_variant(v)
        annotated_count = sum(1 for v in variants if v.gene)
        driver_count = sum(1 for v in variants if v.is_cancer_driver)
        logger.info(
            f"Annotated {annotated_count}/{len(variants)} variants with gene info, "
            f"{driver_count} cancer drivers"
        )
        return variants

    def _annotate_variant(self, variant: Variant):
        """Annotate a single variant."""
        # If gene already set (from VCF INFO), check driver status
        if variant.gene:
            gene_upper = variant.gene.upper()
            if gene_upper in CANCER_DRIVER_GENES:
                variant.is_cancer_driver = True
                self._annotate_driver_details(variant, gene_upper)
            return

        # Try to extract gene from INFO fields
        gene = self._extract_gene_from_info(variant.info)
        if gene:
            variant.gene = gene
            gene_upper = gene.upper()
            if gene_upper in CANCER_DRIVER_GENES:
                variant.is_cancer_driver = True
                self._annotate_driver_details(variant, gene_upper)

    def _extract_gene_from_info(self, info: Dict[str, str]) -> Optional[str]:
        """Try to extract gene name from various INFO field formats."""
        # Direct gene field
        for key in ("GENE", "Gene", "gene", "SYMBOL", "Gene_Name"):
            if key in info:
                return info[key]

        # From ANN field (already parsed in VCF parser, but check raw)
        if "ANN" in info:
            parts = info["ANN"].split("|")
            if len(parts) >= 4 and parts[3]:
                return parts[3]

        return None

    def _annotate_driver_details(self, variant: Variant, gene: str):
        """Add cancer driver details from our built-in database."""
        driver_info = self.driver_db.get(gene, {})
        if not driver_info:
            return

        # Check for hotspot mutation
        if variant.protein_change:
            # Extract amino acid change (e.g. "p.T877A" → "T877A")
            aa_change = variant.protein_change.replace("p.", "")
            hotspots = driver_info.get("hotspot_mutations", {})
            if aa_change in hotspots:
                variant.clinvar_significance = (
                    f"Hotspot: {hotspots[aa_change]}"
                )
                variant.impact = "HIGH"

    def get_driver_info(self, gene: str) -> Optional[Dict]:
        """Get cancer driver information for a gene."""
        return self.driver_db.get(gene.upper())

    def get_all_drivers(self) -> Dict:
        """Get the full cancer driver gene database."""
        return dict(self.driver_db)

    def compute_tmb(self, variants: List[Variant],
                    exome_size_mb: float = 30.0) -> float:
        """Compute Tumor Mutational Burden (mutations per megabase).

        Args:
            variants: List of coding variants.
            exome_size_mb: Size of the captured exome in megabases.

        Returns:
            TMB value (variants/Mb). >10 is considered TMB-high.
        """
        coding = [v for v in variants if v.is_coding]
        tmb = len(coding) / exome_size_mb
        logger.info(f"TMB: {tmb:.1f} mutations/Mb ({len(coding)} coding variants)")
        return tmb

    def classify_msi(self, variants: List[Variant]) -> str:
        """Classify microsatellite instability status.

        Simple heuristic: MSI-H if indel count in microsatellite regions
        is significantly elevated. For accurate MSI, use MSIsensor or similar.

        Returns: "MSI-H", "MSS", or "unknown"
        """
        indels = [v for v in variants if v.is_indel]
        # Simplified heuristic — real MSI calling needs microsatellite loci
        if len(indels) > 50:
            return "MSI-H"
        return "MSS"
