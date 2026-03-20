"""
ClinicalTrials.gov API Client
================================

Matches patient biomarkers to active clinical trials using the
ClinicalTrials.gov V2 API (launched 2024, replaces legacy API).

Adds "trial matching" to the MAD Board evidence chain — showing
the clinician which trials the patient may be eligible for.

API: https://clinicaltrials.gov/api/v2/studies
No authentication required. Rate limit: 3 requests/second.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

CT_API_BASE = "https://clinicaltrials.gov/api/v2/studies"


@dataclass
class ClinicalTrial:
    """A matching clinical trial from ClinicalTrials.gov."""

    nct_id: str
    title: str
    status: str  # "RECRUITING", "ACTIVE_NOT_RECRUITING", etc.
    phase: str  # "PHASE1", "PHASE2", "PHASE3"
    conditions: List[str]
    interventions: List[str]
    sponsor: str = ""
    start_date: str = ""
    locations_count: int = 0
    match_reason: str = ""  # Why this trial matched

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nct_id": self.nct_id,
            "title": self.title,
            "status": self.status,
            "phase": self.phase,
            "conditions": self.conditions,
            "interventions": self.interventions,
            "sponsor": self.sponsor,
            "match_reason": self.match_reason,
        }


class ClinicalTrialsClient:
    """Query ClinicalTrials.gov for matching immunotherapy trials."""

    def __init__(self):
        self.base_url = CT_API_BASE
        self._last_request = 0.0

    def find_trials_for_patient(
        self,
        cancer_type: str = "prostate cancer",
        biomarkers: Optional[List[str]] = None,
        drugs: Optional[List[str]] = None,
        max_results: int = 10,
    ) -> List[ClinicalTrial]:
        """Find matching trials based on cancer type and biomarkers.

        Args:
            cancer_type: Cancer type (e.g., "prostate cancer")
            biomarkers: Biomarker keywords to search (e.g., ["BRCA2", "TMB-H"])
            drugs: Drug keywords (e.g., ["pembrolizumab", "olaparib"])
            max_results: Maximum trials to return
        """
        trials = []

        # Search by biomarkers
        if biomarkers:
            for biomarker in biomarkers[:3]:  # Limit API calls
                query = f"{cancer_type} AND {biomarker}"
                results = self._search(query, max_results=5)
                for trial in results:
                    trial.match_reason = f"Biomarker: {biomarker}"
                    trials.append(trial)

        # Search by drugs
        if drugs:
            for drug in drugs[:3]:
                query = f"{cancer_type} AND {drug}"
                results = self._search(query, max_results=5)
                for trial in results:
                    trial.match_reason = f"Drug: {drug}"
                    trials.append(trial)

        # Default: immunotherapy in cancer type
        if not trials:
            query = f"{cancer_type} AND immunotherapy"
            trials = self._search(query, max_results=max_results)
            for t in trials:
                t.match_reason = "Immunotherapy"

        # Deduplicate by NCT ID
        seen = set()
        unique = []
        for trial in trials:
            if trial.nct_id not in seen:
                seen.add(trial.nct_id)
                unique.append(trial)

        return unique[:max_results]

    def _search(self, query: str, max_results: int = 10) -> List[ClinicalTrial]:
        """Search ClinicalTrials.gov V2 API."""
        # Rate limiting (3 req/sec)
        elapsed = time.time() - self._last_request
        if elapsed < 0.35:
            time.sleep(0.35 - elapsed)

        try:
            resp = requests.get(
                self.base_url,
                params={
                    "query.cond": query,
                    "filter.overallStatus": "RECRUITING",
                    "pageSize": max_results,
                    "fields": (
                        "NCTId,BriefTitle,OverallStatus,Phase,"
                        "Condition,InterventionName,LeadSponsorName,"
                        "StartDate,LocationCountry"
                    ),
                },
                timeout=10,
            )
            self._last_request = time.time()

            if resp.status_code != 200:
                logger.warning("ClinicalTrials.gov returned %d", resp.status_code)
                return []

            data = resp.json()
            studies = data.get("studies", [])
            trials = []

            for study in studies:
                protocol = study.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                design = protocol.get("designModule", {})
                conditions_module = protocol.get("conditionsModule", {})
                arms_module = protocol.get("armsInterventionsModule", {})
                sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
                contacts_module = protocol.get("contactsLocationsModule", {})

                interventions = []
                for arm in arms_module.get("interventions", []):
                    interventions.append(arm.get("name", ""))

                locations = contacts_module.get("locations", [])

                trials.append(ClinicalTrial(
                    nct_id=id_module.get("nctId", ""),
                    title=id_module.get("briefTitle", ""),
                    status=status_module.get("overallStatus", ""),
                    phase=", ".join(design.get("phases", [])),
                    conditions=conditions_module.get("conditions", []),
                    interventions=interventions,
                    sponsor=sponsor_module.get("leadSponsor", {}).get("name", ""),
                    start_date=status_module.get("startDateStruct", {}).get("date", ""),
                    locations_count=len(locations),
                ))

            return trials

        except Exception as e:
            logger.warning("ClinicalTrials.gov search failed: %s", e)
            return []


def match_patient_to_trials(
    profile,  # PatientProfile
    twin,     # DigitalTwinConfig
    max_results: int = 10,
) -> List[ClinicalTrial]:
    """Match a patient to clinical trials based on their biomarkers.

    Builds search queries from the patient's actionable biomarkers
    and searches ClinicalTrials.gov for recruiting trials.
    """
    client = ClinicalTrialsClient()

    cancer_type = getattr(profile, "cancer_type", "prostate cancer")
    if cancer_type == "prostate":
        cancer_type = "prostate cancer"

    # Build biomarker keywords
    biomarkers = []
    if getattr(profile, "is_tmb_high", False):
        biomarkers.append("TMB-high")
    if getattr(profile, "has_dna_repair_defect", False):
        biomarkers.append("BRCA")
        biomarkers.append("homologous recombination deficiency")
    if getattr(profile, "msi_status", "") == "MSI-H":
        biomarkers.append("MSI-H")
    if getattr(twin, "neoantigen_vaccine_candidate", False):
        biomarkers.append("neoantigen vaccine")

    # Build drug keywords from profile
    drugs = []
    if getattr(profile, "has_dna_repair_defect", False):
        drugs.append("olaparib")
    if getattr(profile, "is_tmb_high", False) or getattr(profile, "msi_status", "") == "MSI-H":
        drugs.append("pembrolizumab")

    return client.find_trials_for_patient(
        cancer_type=cancer_type,
        biomarkers=biomarkers if biomarkers else None,
        drugs=drugs if drugs else None,
        max_results=max_results,
    )
