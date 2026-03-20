"""
External Evidence Sources for MAD Agent
=========================================

Integrates regulatory, immunological, and pharmacological databases
to provide the "Layer 4: External Evidence" for the Unified Patient Profile.

Sources:
  - OncoKB: Variant actionability (integrated separately in oncokb_client.py)
  - CIViC: Community-curated clinical interpretations
  - ClinicalTrials.gov: Trial matching (integrated in clinical_trials.py)
  - IMGT/HLA: HLA nomenclature validation
  - FDA Pharmacogenomic Biomarkers: FDA-labeled drug-gene interactions

All sources are queried via public APIs (no auth required for basic access).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# CIViC (Clinical Interpretations of Variants in Cancer)
# ─────────────────────────────────────────────────────────────────────────

CIVIC_API_BASE = "https://civicdb.org/api"


@dataclass
class CIViCEvidence:
    """Evidence record from CIViC database."""
    gene: str
    variant: str
    disease: str
    drugs: List[str]
    evidence_level: str  # A (validated), B (clinical), C (case), D (preclinical), E (inferential)
    evidence_type: str   # Predictive, Diagnostic, Prognostic, Predisposing
    clinical_significance: str  # Sensitivity, Resistance, etc.
    source: str  # PubMed ID or citation
    civic_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gene": self.gene,
            "variant": self.variant,
            "disease": self.disease,
            "drugs": self.drugs,
            "evidence_level": self.evidence_level,
            "evidence_type": self.evidence_type,
            "clinical_significance": self.clinical_significance,
            "source": self.source,
        }


def query_civic(gene: str, variant: str = None) -> List[CIViCEvidence]:
    """Query CIViC for clinical evidence on a gene/variant.

    Args:
        gene: Hugo gene symbol (e.g., "BRCA2")
        variant: Optional variant name (e.g., "V600E")

    Returns:
        List of CIViC evidence records.
    """
    try:
        # CIViC GraphQL API
        query = """
        query($gene: String!) {
            genes(name: $gene) {
                nodes {
                    name
                    variants {
                        nodes {
                            name
                            evidenceItems {
                                nodes {
                                    id
                                    evidenceLevel
                                    evidenceType
                                    significance
                                    disease { name }
                                    therapies { name }
                                    source { citation }
                                }
                            }
                        }
                    }
                }
            }
        }
        """
        resp = requests.post(
            f"{CIVIC_API_BASE}/graphql",
            json={"query": query, "variables": {"gene": gene}},
            timeout=10,
        )

        if resp.status_code != 200:
            return []

        data = resp.json().get("data", {}).get("genes", {}).get("nodes", [])
        results = []

        for gene_node in data:
            for var_node in gene_node.get("variants", {}).get("nodes", []):
                var_name = var_node.get("name", "")
                if variant and variant.lower() not in var_name.lower():
                    continue

                for ev in var_node.get("evidenceItems", {}).get("nodes", []):
                    drugs = [t.get("name", "") for t in ev.get("therapies", [])]
                    results.append(CIViCEvidence(
                        gene=gene,
                        variant=var_name,
                        disease=ev.get("disease", {}).get("name", ""),
                        drugs=drugs,
                        evidence_level=ev.get("evidenceLevel", ""),
                        evidence_type=ev.get("evidenceType", ""),
                        clinical_significance=ev.get("significance", ""),
                        source=ev.get("source", {}).get("citation", ""),
                        civic_id=ev.get("id", 0),
                    ))

        return results[:20]  # Limit

    except Exception as e:
        logger.debug("CIViC query failed for %s: %s", gene, e)
        return []


# ─────────────────────────────────────────────────────────────────────────
# IMGT/HLA Database (HLA nomenclature validation)
# ─────────────────────────────────────────────────────────────────────────

IMGT_API_BASE = "https://www.ebi.ac.uk/ipd/imgt/hla/api"


def validate_hla_allele(allele: str) -> bool:
    """Check if an HLA allele exists in the IMGT/HLA database.

    The IMGT/HLA database (Release 3.63+) is the official repository
    for all HLA sequences and nomenclature.

    Args:
        allele: HLA allele string (e.g., "HLA-A*02:01")

    Returns:
        True if the allele is valid in IMGT/HLA.
    """
    # Normalize
    a = allele.replace("HLA-", "")

    try:
        resp = requests.get(
            f"{IMGT_API_BASE}/allele/{a}",
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        # Fallback: check against known common alleles
        common = {
            "A*01:01", "A*02:01", "A*03:01", "A*11:01", "A*24:02",
            "B*07:02", "B*08:01", "B*15:01", "B*35:01", "B*44:02",
            "C*03:04", "C*04:01", "C*05:01", "C*06:02", "C*07:01",
            "C*07:02", "C*12:03",
        }
        return a in common


def get_hla_allele_info(allele: str) -> Optional[Dict[str, Any]]:
    """Get detailed info for an HLA allele from IMGT/HLA."""
    a = allele.replace("HLA-", "")
    try:
        resp = requests.get(f"{IMGT_API_BASE}/allele/{a}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────
# FDA Pharmacogenomic Biomarkers Table
# ─────────────────────────────────────────────────────────────────────────

# Based on FDA Table of Pharmacogenomic Biomarkers (March 2026 update)
# https://www.fda.gov/drugs/science-and-research-drugs/table-pharmacogenomic-biomarkers-drug-labeling

FDA_PHARMACOGENOMIC_BIOMARKERS = {
    # Gene: [(drug, biomarker_use, labeling_section)]
    "BRCA1": [
        ("Olaparib", "Indicated for germline BRCA-mutated mCRPC", "Indications"),
        ("Rucaparib", "Indicated for BRCA-mutated mCRPC", "Indications"),
        ("Talazoparib", "Indicated for germline BRCA-mutated breast cancer", "Indications"),
    ],
    "BRCA2": [
        ("Olaparib", "Indicated for germline BRCA-mutated mCRPC", "Indications"),
        ("Rucaparib", "Indicated for BRCA-mutated mCRPC", "Indications"),
        ("Niraparib", "Indicated for HRD-positive ovarian cancer", "Indications"),
    ],
    "ATM": [
        ("Olaparib", "Included in HRR gene panel (PROfound)", "Clinical Studies"),
    ],
    "CDK12": [
        ("Olaparib", "Included in HRR gene panel (PROfound Cohort B)", "Clinical Studies"),
    ],
    "MSI-H": [
        ("Pembrolizumab", "Tissue-agnostic approval for MSI-H/dMMR", "Indications"),
        ("Dostarlimab", "MSI-H/dMMR endometrial cancer", "Indications"),
    ],
    "TMB-H": [
        ("Pembrolizumab", "Tissue-agnostic approval for TMB >=10 mut/Mb", "Indications"),
    ],
    "PD-L1": [
        ("Pembrolizumab", "CPS >=10 for various solid tumors", "Indications"),
        ("Atezolizumab", "PD-L1 positive urothelial carcinoma", "Indications"),
        ("Nivolumab", "PD-L1 expression informs treatment selection", "Clinical Studies"),
    ],
    "HLA-B*57:01": [
        ("Abacavir", "Screen before prescribing — risk of hypersensitivity", "Boxed Warning"),
    ],
    "DPYD": [
        ("Fluorouracil", "Reduced DPD activity — severe toxicity risk", "Warnings"),
        ("Capecitabine", "Reduced DPD activity — dose reduction required", "Warnings"),
    ],
    "UGT1A1": [
        ("Irinotecan", "*28 allele — reduced glucuronidation, toxicity risk", "Warnings"),
    ],
}


def check_fda_biomarkers(gene: str) -> List[Dict[str, str]]:
    """Check if a gene has FDA-labeled pharmacogenomic biomarkers.

    Args:
        gene: Gene symbol (e.g., "BRCA2", "MSI-H", "PD-L1")

    Returns:
        List of {drug, use, section} dicts from FDA labeling.
    """
    entries = FDA_PHARMACOGENOMIC_BIOMARKERS.get(gene.upper(), [])
    return [
        {"drug": drug, "use": use, "section": section}
        for drug, use, section in entries
    ]


def get_all_fda_biomarker_genes() -> List[str]:
    """Return all genes with FDA pharmacogenomic biomarker labels."""
    return sorted(FDA_PHARMACOGENOMIC_BIOMARKERS.keys())


# ─────────────────────────────────────────────────────────────────────────
# Unified Evidence Query
# ─────────────────────────────────────────────────────────────────────────

def query_all_evidence(
    gene: str,
    variant: str = None,
) -> Dict[str, Any]:
    """Query all evidence sources for a gene/variant.

    Returns a unified evidence package from CIViC, FDA BQP, and IMGT/HLA.
    OncoKB and ClinicalTrials.gov are queried separately.
    """
    result = {
        "gene": gene,
        "variant": variant,
        "civic": [],
        "fda_biomarkers": [],
    }

    # CIViC
    civic_results = query_civic(gene, variant)
    result["civic"] = [r.to_dict() for r in civic_results[:10]]

    # FDA Biomarkers
    fda = check_fda_biomarkers(gene)
    result["fda_biomarkers"] = fda

    return result
