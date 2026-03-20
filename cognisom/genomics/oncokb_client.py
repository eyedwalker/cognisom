"""
OncoKB API Client
==================

Queries the OncoKB precision oncology knowledge base for variant
actionability, treatment evidence, and biomarker annotations.

OncoKB provides:
  - 700+ genes, 5,000+ variants annotated
  - Evidence levels 1-4 (FDA-approved → investigational)
  - Drug associations for each actionable variant
  - Oncogenic/likely-oncogenic/VUS classification

API: https://www.oncokb.org/api/v1/
Academic license: Free (requires API token registration)

References:
  Chakravarty et al., JCO Precision Oncology 2017
  https://www.oncokb.org
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

ONCOKB_API_BASE = "https://www.oncokb.org/api/v1"
ONCOKB_TOKEN = os.environ.get("ONCOKB_API_TOKEN", "")


@dataclass
class OncoKBAnnotation:
    """Annotation result from OncoKB for a single variant."""

    gene: str
    variant: str  # e.g., "V600E", "T877A"
    oncogenic: str  # "Oncogenic", "Likely Oncogenic", "Inconclusive", "Unknown"
    mutation_effect: str  # "Gain-of-function", "Loss-of-function", "Switch-of-function"
    highest_sensitive_level: str  # "LEVEL_1", "LEVEL_2", "LEVEL_3A", etc.
    highest_resistance_level: str  # "LEVEL_R1", "LEVEL_R2", ""

    # Treatments
    treatments: List[Dict[str, Any]] = field(default_factory=list)
    """Each: {drugs: [str], level: str, cancer_type: str, description: str}."""

    # Diagnostic / prognostic
    diagnostic_level: str = ""
    prognostic_level: str = ""

    # Raw response
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.highest_sensitive_level != ""

    @property
    def evidence_level_numeric(self) -> float:
        """Convert OncoKB level to numeric (lower = stronger evidence)."""
        level_map = {
            "LEVEL_1": 1.0, "LEVEL_2": 2.0,
            "LEVEL_3A": 3.0, "LEVEL_3B": 3.5,
            "LEVEL_4": 4.0,
        }
        return level_map.get(self.highest_sensitive_level, 5.0)

    @property
    def drugs(self) -> List[str]:
        """All drugs across all treatments."""
        all_drugs = []
        for tx in self.treatments:
            all_drugs.extend(tx.get("drugs", []))
        return list(set(all_drugs))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gene": self.gene,
            "variant": self.variant,
            "oncogenic": self.oncogenic,
            "mutation_effect": self.mutation_effect,
            "highest_sensitive_level": self.highest_sensitive_level,
            "highest_resistance_level": self.highest_resistance_level,
            "treatments": self.treatments,
            "is_actionable": self.is_actionable,
            "drugs": self.drugs,
        }


class OncoKBClient:
    """Client for the OncoKB precision oncology API.

    Requires an API token (free for academic use).
    Register at https://www.oncokb.org/account/register

    Set the token via:
      - Environment variable: ONCOKB_API_TOKEN
      - Constructor parameter: api_token
    """

    def __init__(self, api_token: str = None):
        self.token = api_token or ONCOKB_TOKEN
        self.base_url = ONCOKB_API_BASE
        self._cache: Dict[str, OncoKBAnnotation] = {}
        self._available = None

    @property
    def is_available(self) -> bool:
        """Check if OncoKB API is configured and reachable."""
        if self._available is not None:
            return self._available

        if not self.token:
            logger.info("OncoKB: No API token configured (set ONCOKB_API_TOKEN)")
            self._available = False
            return False

        try:
            resp = requests.get(
                f"{self.base_url}/info",
                headers=self._headers(),
                timeout=5,
            )
            self._available = resp.status_code == 200
        except Exception:
            self._available = False

        return self._available

    def annotate_mutation(
        self,
        gene: str,
        protein_change: str,
        tumor_type: str = "Prostate Cancer",
    ) -> OncoKBAnnotation:
        """Annotate a single mutation via OncoKB.

        Args:
            gene: Hugo gene symbol (e.g., "BRAF", "BRCA2")
            protein_change: Protein change (e.g., "V600E", "T877A")
            tumor_type: OncoKB tumor type (e.g., "Prostate Cancer")

        Returns:
            OncoKBAnnotation with actionability data.
        """
        cache_key = f"{gene}:{protein_change}:{tumor_type}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # If API not available, return from built-in knowledge base
        if not self.is_available:
            return self._fallback_annotation(gene, protein_change, tumor_type)

        try:
            resp = requests.get(
                f"{self.base_url}/annotate/mutations/byProteinChange",
                params={
                    "hugoSymbol": gene,
                    "alteration": protein_change,
                    "tumorType": tumor_type,
                },
                headers=self._headers(),
                timeout=10,
            )

            if resp.status_code == 200:
                data = resp.json()
                annotation = self._parse_response(gene, protein_change, data)
                self._cache[cache_key] = annotation
                return annotation
            else:
                logger.warning("OncoKB API returned %d for %s %s",
                               resp.status_code, gene, protein_change)
                return self._fallback_annotation(gene, protein_change, tumor_type)

        except Exception as e:
            logger.warning("OncoKB query failed for %s %s: %s", gene, protein_change, e)
            return self._fallback_annotation(gene, protein_change, tumor_type)

    def annotate_variants(
        self,
        variants: List[Dict[str, str]],
        tumor_type: str = "Prostate Cancer",
    ) -> List[OncoKBAnnotation]:
        """Annotate multiple variants.

        Args:
            variants: List of {gene, protein_change} dicts
            tumor_type: Cancer type for all variants
        """
        results = []
        for v in variants:
            gene = v.get("gene", "")
            change = v.get("protein_change", "")
            if gene and change:
                results.append(self.annotate_mutation(gene, change, tumor_type))
            else:
                results.append(OncoKBAnnotation(
                    gene=gene, variant=change,
                    oncogenic="Unknown", mutation_effect="Unknown",
                    highest_sensitive_level="", highest_resistance_level="",
                ))
        return results

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

    def _parse_response(
        self, gene: str, variant: str, data: Dict
    ) -> OncoKBAnnotation:
        """Parse OncoKB API response into structured annotation."""
        mutation_effect = data.get("mutationEffect", {})
        treatments = []

        for tx in data.get("treatments", []):
            drugs = [d.get("drugName", "") for d in tx.get("drugs", [])]
            treatments.append({
                "drugs": drugs,
                "level": tx.get("level", ""),
                "cancer_type": tx.get("levelAssociatedCancerType", {}).get("mainType", {}).get("name", ""),
                "description": tx.get("description", "")[:200],
            })

        return OncoKBAnnotation(
            gene=gene,
            variant=variant,
            oncogenic=data.get("oncogenic", "Unknown"),
            mutation_effect=mutation_effect.get("knownEffect", "Unknown"),
            highest_sensitive_level=data.get("highestSensitiveLevel", "") or "",
            highest_resistance_level=data.get("highestResistanceLevel", "") or "",
            treatments=treatments,
            diagnostic_level=data.get("highestDiagnosticImplicationLevel", "") or "",
            prognostic_level=data.get("highestPrognosticImplicationLevel", "") or "",
            raw=data,
        )

    def _fallback_annotation(
        self, gene: str, protein_change: str, tumor_type: str
    ) -> OncoKBAnnotation:
        """Built-in fallback knowledge base when API is unavailable.

        Covers the most clinically significant prostate cancer variants
        based on NCCN guidelines and published evidence.
        """
        key = f"{gene}:{protein_change}".upper()
        gene_upper = gene.upper()

        # --- Prostate cancer driver gene knowledge base ---
        # Based on NCCN Prostate Cancer v2.2024 + published trials

        annotation = OncoKBAnnotation(
            gene=gene,
            variant=protein_change,
            oncogenic="Unknown",
            mutation_effect="Unknown",
            highest_sensitive_level="",
            highest_resistance_level="",
        )

        # BRCA1/2 — PARP inhibitors (Level 1)
        if gene_upper in ("BRCA1", "BRCA2"):
            annotation.oncogenic = "Oncogenic"
            annotation.mutation_effect = "Loss-of-function"
            annotation.highest_sensitive_level = "LEVEL_1"
            annotation.treatments = [{
                "drugs": ["Olaparib", "Rucaparib"],
                "level": "LEVEL_1",
                "cancer_type": "Prostate Cancer",
                "description": "FDA-approved: olaparib for BRCA1/2 mCRPC (PROfound, NCT02987543)",
            }]

        # ATM — PARP inhibitors (Level 1, Cohort B)
        elif gene_upper == "ATM":
            annotation.oncogenic = "Oncogenic"
            annotation.mutation_effect = "Loss-of-function"
            annotation.highest_sensitive_level = "LEVEL_1"
            annotation.treatments = [{
                "drugs": ["Olaparib"],
                "level": "LEVEL_1",
                "cancer_type": "Prostate Cancer",
                "description": "FDA-approved: olaparib for ATM-mutated mCRPC (PROfound Cohort B)",
            }]

        # CDK12 — Checkpoint inhibitors (Level 3A)
        elif gene_upper == "CDK12":
            annotation.oncogenic = "Oncogenic"
            annotation.mutation_effect = "Loss-of-function"
            annotation.highest_sensitive_level = "LEVEL_3A"
            annotation.treatments = [{
                "drugs": ["Pembrolizumab", "Nivolumab + Ipilimumab"],
                "level": "LEVEL_3A",
                "cancer_type": "Prostate Cancer",
                "description": "CDK12 biallelic loss creates neoantigen-rich TME; checkpoint response observed",
            }]

        # AR — Resistance to AR-targeted therapy
        elif gene_upper == "AR":
            annotation.oncogenic = "Oncogenic"
            annotation.mutation_effect = "Gain-of-function"
            annotation.highest_resistance_level = "LEVEL_R1"
            annotation.treatments = [{
                "drugs": ["Enzalutamide", "Abiraterone"],
                "level": "LEVEL_R1",
                "cancer_type": "Prostate Cancer",
                "description": "AR mutations (T877A, L702H, etc.) confer resistance to AR-targeted therapy",
            }]

        # TP53 — Oncogenic but no direct targeted therapy
        elif gene_upper == "TP53":
            annotation.oncogenic = "Oncogenic"
            annotation.mutation_effect = "Loss-of-function"
            annotation.highest_sensitive_level = ""
            annotation.treatments = []

        # PTEN — PI3K/AKT pathway (Level 3A)
        elif gene_upper == "PTEN":
            annotation.oncogenic = "Oncogenic"
            annotation.mutation_effect = "Loss-of-function"
            annotation.highest_sensitive_level = "LEVEL_3A"
            annotation.treatments = [{
                "drugs": ["Ipatasertib", "AZD5363"],
                "level": "LEVEL_3A",
                "cancer_type": "Prostate Cancer",
                "description": "PTEN loss activates PI3K/AKT pathway; AKT inhibitors in trials",
            }]

        # PIK3CA — PI3K inhibitors (Level 3A)
        elif gene_upper == "PIK3CA":
            annotation.oncogenic = "Oncogenic"
            annotation.mutation_effect = "Gain-of-function"
            annotation.highest_sensitive_level = "LEVEL_3A"
            annotation.treatments = [{
                "drugs": ["Alpelisib"],
                "level": "LEVEL_3A",
                "cancer_type": "Prostate Cancer",
                "description": "PIK3CA activating mutations; PI3K inhibitors under investigation",
            }]

        # PALB2, CHEK2 — PARP (Level 2)
        elif gene_upper in ("PALB2", "CHEK2"):
            annotation.oncogenic = "Oncogenic"
            annotation.mutation_effect = "Loss-of-function"
            annotation.highest_sensitive_level = "LEVEL_2"
            annotation.treatments = [{
                "drugs": ["Olaparib"],
                "level": "LEVEL_2",
                "cancer_type": "Prostate Cancer",
                "description": f"{gene} mutation — HRD, PARP inhibitor candidate",
            }]

        # SPOP — BET inhibitor sensitivity (Level 4)
        elif gene_upper == "SPOP":
            annotation.oncogenic = "Oncogenic"
            annotation.mutation_effect = "Loss-of-function"
            annotation.highest_sensitive_level = "LEVEL_4"
            annotation.treatments = [{
                "drugs": ["BET inhibitors (investigational)"],
                "level": "LEVEL_4",
                "cancer_type": "Prostate Cancer",
                "description": "SPOP mutations sensitize to BET inhibition; early clinical data",
            }]

        # RB1 — CDK4/6 inhibitors (Level 4)
        elif gene_upper == "RB1":
            annotation.oncogenic = "Oncogenic"
            annotation.mutation_effect = "Loss-of-function"
            annotation.highest_sensitive_level = "LEVEL_4"
            annotation.treatments = [{
                "drugs": ["Palbociclib (investigational in prostate)"],
                "level": "LEVEL_4",
                "cancer_type": "Prostate Cancer",
                "description": "RB1 loss → neuroendocrine differentiation risk; CDK4/6 inhibitors in trials",
            }]

        # MSI-H / TMB-H — Pembrolizumab (Level 1, tissue-agnostic)
        elif gene_upper in ("MLH1", "MSH2", "MSH6", "PMS2"):
            annotation.oncogenic = "Oncogenic"
            annotation.mutation_effect = "Loss-of-function"
            annotation.highest_sensitive_level = "LEVEL_1"
            annotation.treatments = [{
                "drugs": ["Pembrolizumab"],
                "level": "LEVEL_1",
                "cancer_type": "All Solid Tumors",
                "description": "FDA-approved: pembrolizumab for MSI-H/dMMR (tissue-agnostic)",
            }]

        return annotation


def get_oncokb_client() -> OncoKBClient:
    """Get a configured OncoKB client."""
    return OncoKBClient()
