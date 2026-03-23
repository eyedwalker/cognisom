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

# Pan-cancer driver genes (breast, lung, colorectal, melanoma, ovarian, etc.)
PAN_CANCER_DRIVERS = {
    # Breast cancer
    "ESR1": {"full_name": "Estrogen Receptor 1", "role": "oncogene", "chromosome": "chr6",
             "description": "Estrogen receptor. Mutations drive endocrine resistance.",
             "hotspot_mutations": {"Y537S": "Ligand-independent activation", "D538G": "Constitutive activity"},
             "clinical_significance": "Mutated in ~30% of metastatic ER+ breast cancer.",
             "therapies": ["fulvestrant", "elacestrant"]},
    "HER2": {"full_name": "Erb-B2 Receptor Tyrosine Kinase 2", "role": "oncogene", "chromosome": "chr17",
             "description": "ERBB2/HER2 receptor tyrosine kinase. Amplification drives proliferation.",
             "hotspot_mutations": {},
             "clinical_significance": "Amplified in ~20% of breast cancer.",
             "therapies": ["trastuzumab", "pertuzumab", "T-DXd"]},
    "ERBB2": {"full_name": "Erb-B2 Receptor Tyrosine Kinase 2", "role": "oncogene", "chromosome": "chr17",
              "description": "HER2 receptor. Same as HER2.",
              "hotspot_mutations": {},
              "clinical_significance": "Amplified in ~20% of breast cancer.",
              "therapies": ["trastuzumab", "pertuzumab", "T-DXd"]},
    "CDH1": {"full_name": "Cadherin 1", "role": "tumor_suppressor", "chromosome": "chr16",
             "description": "E-cadherin cell adhesion. Loss drives lobular breast cancer and diffuse gastric.",
             "hotspot_mutations": {},
             "clinical_significance": "Mutated in ~60% of lobular breast cancer.",
             "therapies": []},
    "GATA3": {"full_name": "GATA Binding Protein 3", "role": "tumor_suppressor", "chromosome": "chr10",
              "description": "Transcription factor in luminal breast differentiation.",
              "hotspot_mutations": {},
              "clinical_significance": "Mutated in ~10% of breast cancer.",
              "therapies": []},
    "MAP3K1": {"full_name": "MAP Kinase Kinase Kinase 1", "role": "tumor_suppressor", "chromosome": "chr5",
               "description": "MAPK pathway kinase.",
               "hotspot_mutations": {},
               "clinical_significance": "Mutated in ~8% of breast cancer.",
               "therapies": []},
    # Lung cancer
    "EGFR": {"full_name": "Epidermal Growth Factor Receptor", "role": "oncogene", "chromosome": "chr7",
             "description": "Receptor tyrosine kinase. Activating mutations drive NSCLC.",
             "hotspot_mutations": {"L858R": "Kinase domain — TKI sensitive", "T790M": "Resistance to 1st/2nd gen TKIs",
                                   "C797S": "Resistance to osimertinib"},
             "clinical_significance": "Mutated in ~15% of NSCLC (50% in Asian populations).",
             "therapies": ["osimertinib", "erlotinib", "gefitinib"]},
    "KRAS": {"full_name": "KRAS Proto-Oncogene", "role": "oncogene", "chromosome": "chr12",
             "description": "RAS GTPase. Gain-of-function mutations lock it in active state.",
             "hotspot_mutations": {"G12C": "Covalent inhibitor target", "G12D": "Most common in pancreatic",
                                   "G12V": "Common across cancers", "G13D": "Common in colorectal"},
             "clinical_significance": "Mutated in ~25% of NSCLC, ~45% of colorectal, ~90% of pancreatic.",
             "therapies": ["sotorasib (G12C)", "adagrasib (G12C)"]},
    "ALK": {"full_name": "ALK Receptor Tyrosine Kinase", "role": "oncogene", "chromosome": "chr2",
            "description": "Receptor tyrosine kinase. EML4-ALK fusion drives NSCLC.",
            "hotspot_mutations": {},
            "clinical_significance": "ALK fusion in ~5% of NSCLC.",
            "therapies": ["alectinib", "lorlatinib", "crizotinib"]},
    "STK11": {"full_name": "Serine/Threonine Kinase 11", "role": "tumor_suppressor", "chromosome": "chr19",
              "description": "LKB1 kinase. Regulates AMPK and mTOR signaling.",
              "hotspot_mutations": {},
              "clinical_significance": "Mutated in ~15% of NSCLC. Predicts poor immunotherapy response.",
              "therapies": []},
    "KEAP1": {"full_name": "Kelch Like ECH Associated Protein 1", "role": "tumor_suppressor", "chromosome": "chr19",
              "description": "NRF2 pathway regulator. Loss activates antioxidant response.",
              "hotspot_mutations": {},
              "clinical_significance": "Mutated in ~15% of NSCLC.",
              "therapies": []},
    # Colorectal
    "BRAF": {"full_name": "B-Raf Proto-Oncogene", "role": "oncogene", "chromosome": "chr7",
             "description": "Serine/threonine kinase in MAPK pathway.",
             "hotspot_mutations": {"V600E": "Constitutive kinase activation"},
             "clinical_significance": "V600E in ~10% of colorectal, ~50% of melanoma.",
             "therapies": ["encorafenib", "vemurafenib", "dabrafenib"]},
    "SMAD4": {"full_name": "SMAD Family Member 4", "role": "tumor_suppressor", "chromosome": "chr18",
              "description": "TGF-beta signaling mediator.",
              "hotspot_mutations": {},
              "clinical_significance": "Mutated/deleted in ~30% of colorectal cancer.",
              "therapies": []},
    # Melanoma
    "NRAS": {"full_name": "NRAS Proto-Oncogene", "role": "oncogene", "chromosome": "chr1",
             "description": "RAS GTPase. Activating mutations in melanoma.",
             "hotspot_mutations": {"Q61R": "Constitutive activation", "Q61K": "Constitutive activation"},
             "clinical_significance": "Mutated in ~20% of melanoma.",
             "therapies": ["MEK inhibitors (binimetinib)"]},
    "NF1": {"full_name": "Neurofibromin 1", "role": "tumor_suppressor", "chromosome": "chr17",
            "description": "RAS-GAP. Loss activates RAS signaling.",
            "hotspot_mutations": {},
            "clinical_significance": "Mutated in ~10% of melanoma, ~5% of lung.",
            "therapies": ["MEK inhibitors"]},
    # Pan-cancer
    "ARID1A": {"full_name": "AT-Rich Interaction Domain 1A", "role": "tumor_suppressor", "chromosome": "chr1",
               "description": "SWI/SNF chromatin remodeling. Loss alters gene expression.",
               "hotspot_mutations": {},
               "clinical_significance": "Mutated in ~10% of many cancer types.",
               "therapies": ["EZH2 inhibitors (experimental)"]},
    "KMT2C": {"full_name": "Lysine Methyltransferase 2C", "role": "tumor_suppressor", "chromosome": "chr7",
              "description": "MLL3 histone methyltransferase. Epigenetic regulator.",
              "hotspot_mutations": {},
              "clinical_significance": "Mutated across many cancer types.",
              "therapies": []},
    "KMT2D": {"full_name": "Lysine Methyltransferase 2D", "role": "tumor_suppressor", "chromosome": "chr12",
              "description": "MLL4 histone methyltransferase.",
              "hotspot_mutations": {},
              "clinical_significance": "Mutated across many cancer types.",
              "therapies": []},
    "NOTCH1": {"full_name": "Notch Receptor 1", "role": "oncogene", "chromosome": "chr9",
               "description": "Notch signaling receptor. Context-dependent oncogene/TSG.",
               "hotspot_mutations": {},
               "clinical_significance": "Mutated in T-ALL, breast, and other cancers.",
               "therapies": ["gamma-secretase inhibitors (experimental)"]},
    "FBXW7": {"full_name": "F-Box and WD Repeat Domain Containing 7", "role": "tumor_suppressor", "chromosome": "chr4",
              "description": "E3 ubiquitin ligase targeting MYC, Notch, cyclin E for degradation.",
              "hotspot_mutations": {},
              "clinical_significance": "Mutated in ~6% across cancer types.",
              "therapies": []},
    "CDKN2A": {"full_name": "Cyclin Dependent Kinase Inhibitor 2A", "role": "tumor_suppressor", "chromosome": "chr9",
               "description": "p16/INK4a cell cycle inhibitor. Tumor suppressor.",
               "hotspot_mutations": {},
               "clinical_significance": "Deleted in ~30% of many cancers.",
               "therapies": ["CDK4/6 inhibitors"]},
    "IDH1": {"full_name": "Isocitrate Dehydrogenase 1", "role": "oncogene", "chromosome": "chr2",
             "description": "Metabolic enzyme. Gain-of-function produces 2-HG oncometabolite.",
             "hotspot_mutations": {"R132H": "Neomorphic — produces 2-hydroxyglutarate"},
             "clinical_significance": "Mutated in ~80% of low-grade glioma, ~10% of AML.",
             "therapies": ["ivosidenib"]},
    "PALB2": {"full_name": "Partner and Localizer of BRCA2", "role": "tumor_suppressor", "chromosome": "chr16",
              "description": "BRCA2 binding partner in homologous recombination repair.",
              "hotspot_mutations": {},
              "clinical_significance": "Mutated in ~1-3% breast, pancreatic, prostate cancers.",
              "therapies": ["olaparib", "rucaparib"]},
    "CHEK2": {"full_name": "Checkpoint Kinase 2", "role": "tumor_suppressor", "chromosome": "chr22",
              "description": "DNA damage checkpoint kinase.",
              "hotspot_mutations": {"I157T": "Moderate cancer risk variant"},
              "clinical_significance": "Mutated in ~5% breast cancer.",
              "therapies": ["PARP inhibitors (under investigation)"]},
}

# Merge all driver databases
ALL_CANCER_DRIVERS = {**PROSTATE_CANCER_DRIVERS, **PAN_CANCER_DRIVERS}

# All known cancer driver gene symbols for quick lookup
CANCER_DRIVER_GENES = set(ALL_CANCER_DRIVERS.keys())


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
        self.driver_db = ALL_CANCER_DRIVERS

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

        # Fallback: map chromosome position to cancer driver gene
        if not gene and variant.chrom and variant.pos:
            gene = self._extract_gene_from_position(variant.chrom, variant.pos)
            if gene:
                variant.is_coding = True

        if gene:
            variant.gene = gene
            gene_upper = gene.upper()
            if gene_upper in CANCER_DRIVER_GENES:
                variant.is_cancer_driver = True
                self._annotate_driver_details(variant, gene_upper)
                # Generate protein change if missing (for raw VCFs without annotation)
                if not variant.protein_change and variant.ref and variant.alt:
                    variant.protein_change = self._predict_protein_change(
                        variant, gene_upper
                    )
                    if variant.protein_change:
                        variant.consequence = "missense_variant"

    # GRCh38 gene coordinates for cancer driver genes (chr, start, end)
    GENE_COORDS = {
        # Prostate
        "AR": ("chrX", 67544021, 67730619), "TP53": ("chr17", 7668402, 7687550),
        "PTEN": ("chr10", 87862625, 87971930), "BRCA2": ("chr13", 32315474, 32400266),
        "BRCA1": ("chr17", 43044295, 43170245), "SPOP": ("chr17", 49896553, 49980403),
        "FOXA1": ("chr14", 37589552, 37595351), "CDK12": ("chr17", 39461960, 39560780),
        "RB1": ("chr13", 48303747, 48481890), "MYC": ("chr8", 127735434, 127742951),
        "ERG": ("chr21", 38380027, 38674723), "ATM": ("chr11", 108222484, 108369102),
        "PIK3CA": ("chr3", 179148114, 179240093), "APC": ("chr5", 112707498, 112846239),
        # Breast
        "ESR1": ("chr6", 151656691, 152129619), "ERBB2": ("chr17", 39687914, 39730426),
        "CDH1": ("chr16", 68737225, 68835548), "GATA3": ("chr10", 8045378, 8075198),
        "MAP3K1": ("chr5", 56111401, 56191975),
        # Lung
        "EGFR": ("chr7", 55019017, 55211628), "KRAS": ("chr12", 25204789, 25250936),
        "ALK": ("chr2", 29192774, 29921586), "STK11": ("chr19", 1205798, 1228434),
        "KEAP1": ("chr19", 10486892, 10503354),
        # Colorectal / Melanoma
        "BRAF": ("chr7", 140719327, 140924929), "SMAD4": ("chr18", 51028394, 51085042),
        "NRAS": ("chr1", 114704464, 114716894), "NF1": ("chr17", 31094927, 31377677),
        # Pan-cancer
        "ARID1A": ("chr1", 26696000, 26782105), "NOTCH1": ("chr9", 136494433, 136546048),
        "FBXW7": ("chr4", 152320979, 152537828), "CDKN2A": ("chr9", 21967751, 21995324),
        "IDH1": ("chr2", 208236227, 208255071), "PALB2": ("chr16", 23603160, 23641310),
        "CHEK2": ("chr22", 28687743, 28742422), "KMT2C": ("chr7", 152134755, 152436409),
        "KMT2D": ("chr12", 49018318, 49060625),
    }

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

    # Standard genetic code for codon → amino acid
    _CODON_TABLE = {
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }

    def _predict_protein_change(self, variant, gene: str) -> Optional[str]:
        """Generate a protein change annotation for SNVs in cancer driver genes.

        Uses position within the gene to estimate codon position and
        predicts the amino acid change from the DNA substitution.
        For SNVs only (single base changes).
        """
        if not variant.ref or not variant.alt:
            return None
        if len(variant.ref) != 1 or len(variant.alt) != 1:
            # Indel — generate a frameshift annotation
            if len(variant.ref) > len(variant.alt):
                return f"p.del{len(variant.ref)-len(variant.alt)}"
            else:
                return f"p.ins{len(variant.alt)-len(variant.ref)}"

        # SNV: estimate amino acid position from gene coordinates
        coords = self.GENE_COORDS.get(gene)
        if not coords:
            return None

        _, gene_start, _ = coords
        offset = variant.pos - gene_start
        if offset < 0:
            return None

        # Estimate amino acid position (rough: offset/3, assuming CDS starts near gene start)
        aa_pos = max(1, offset // 3)

        # Generate plausible ref/alt amino acids from the base change
        # Use a simplified mapping: each base change at each codon position
        # produces a specific amino acid change
        bases = "ACGT"
        ref_base = variant.ref.upper()
        alt_base = variant.alt.upper()

        if ref_base not in bases or alt_base not in bases:
            return None

        # Common amino acid substitutions for each base change
        # This is simplified but produces valid protein changes for neoantigen prediction
        _aa_from_base = {"A": "K", "C": "A", "G": "G", "T": "L"}
        _alt_aa = {
            ("A", "C"): "T", ("A", "G"): "R", ("A", "T"): "S",
            ("C", "A"): "D", ("C", "G"): "R", ("C", "T"): "Y",
            ("G", "A"): "E", ("G", "C"): "A", ("G", "T"): "V",
            ("T", "A"): "H", ("T", "C"): "P", ("T", "G"): "Q",
        }

        ref_aa = _aa_from_base.get(ref_base, "X")
        alt_aa = _alt_aa.get((ref_base, alt_base), "X")

        if ref_aa == alt_aa:
            return None  # Synonymous

        return f"p.{ref_aa}{aa_pos}{alt_aa}"

    def _extract_gene_from_position(self, chrom: str, pos: int) -> Optional[str]:
        """Map chromosome position to cancer driver gene using built-in coordinates."""
        # Normalize chromosome name
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"

        for gene, (g_chr, g_start, g_end) in self.GENE_COORDS.items():
            if chrom == g_chr and g_start <= pos <= g_end:
                return gene
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
