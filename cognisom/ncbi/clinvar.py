"""
NCBI ClinVar Integration
=========================

Search ClinVar for known clinical variants by gene, position, or variant name.
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional

from .client import NCBIClient, http_get

log = logging.getLogger(__name__)


@dataclass
class ClinVarVariant:
    """A ClinVar variant record."""
    uid: str
    title: str
    gene: str
    variant_type: str
    clinical_significance: str
    condition: str
    review_status: str
    accession: str
    position: str


def search_clinvar(gene: str, significance: str = "",
                   max_results: int = 20) -> List[ClinVarVariant]:
    """Search ClinVar for variants in a gene.

    Args:
        gene: Gene symbol (e.g. "BRCA1").
        significance: Filter by clinical significance
                      ("pathogenic", "benign", "uncertain", or "" for all).
        max_results: Max variants to return.

    Returns:
        List of ClinVarVariant.
    """
    client = NCBIClient()

    # Build search term
    parts = [f"{gene}[Gene]"]
    if significance:
        parts.append(f"{significance}[Clinical significance]")
    term = " AND ".join(parts)

    result = client.esearch(db="clinvar", term=term, retmax=max_results)
    ids = result.get("idlist", [])
    if not ids:
        return []

    # Fetch summaries
    summaries = client.esummary(db="clinvar", ids=ids)

    variants = []
    for uid in ids:
        doc = summaries.get(uid, {})
        if not isinstance(doc, dict):
            continue

        # Extract fields from summary
        genes_list = doc.get("genes", [])
        gene_name = genes_list[0].get("symbol", gene) if genes_list else gene

        # Clinical significance
        clin_sig_obj = doc.get("clinical_significance", {})
        if isinstance(clin_sig_obj, dict):
            clin_sig = clin_sig_obj.get("description", "")
            review = clin_sig_obj.get("review_status", "")
        else:
            clin_sig = str(clin_sig_obj)
            review = ""

        # Variation set
        var_set = doc.get("variation_set", [])
        var_name = ""
        var_type = ""
        if var_set and isinstance(var_set, list):
            vs = var_set[0] if var_set else {}
            if isinstance(vs, dict):
                var_name = vs.get("variation_name", "")
                var_type = vs.get("variation_type", "")

        # Traits/conditions
        trait_set = doc.get("trait_set", [])
        conditions = []
        if isinstance(trait_set, list):
            for trait in trait_set:
                if isinstance(trait, dict):
                    tname = trait.get("trait_name", "")
                    if tname:
                        conditions.append(tname)

        # Accession
        accession = doc.get("accession", "")

        # Location
        loc = doc.get("variation_loc", [])
        position = ""
        if isinstance(loc, list) and loc:
            first_loc = loc[0]
            if isinstance(first_loc, dict):
                chrom = first_loc.get("chr", "")
                start = first_loc.get("start", "")
                if chrom and start:
                    position = f"chr{chrom}:{start}"

        variants.append(ClinVarVariant(
            uid=uid,
            title=doc.get("title", var_name),
            gene=gene_name,
            variant_type=var_type or doc.get("obj_type", ""),
            clinical_significance=clin_sig,
            condition="; ".join(conditions[:3]),
            review_status=review,
            accession=accession,
            position=position,
        ))

    return variants


def get_variant_details(accession: str) -> Optional[str]:
    """Get full ClinVar record XML by accession (e.g. 'VCV000017599')."""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=clinvar&id={accession}&rettype=vcv&retmode=xml"
    try:
        return http_get(url)
    except Exception as e:
        log.error(f"ClinVar fetch failed for {accession}: {e}")
        return None
