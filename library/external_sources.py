"""
External Data Source Integrations
=================================

Connectors for external biological databases:
- KEGG: Pathway maps, gene-pathway relationships
- PubChem: Compound properties, targets, assays
- STRING: Protein-protein interaction networks
- Reactome: Pathway databases, interaction networks
- DrugBank: Drug mechanisms, interactions
- BioGRID: Genetic interactions

Usage:
    from cognisom.library.external_sources import KEGGClient, PubChemClient

    # KEGG pathways
    kegg = KEGGClient()
    pathways = kegg.search_pathways("prostate cancer")
    genes = kegg.get_pathway_genes("hsa05215")

    # PubChem compounds
    pubchem = PubChemClient()
    compound = pubchem.get_compound("enzalutamide")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from functools import lru_cache

import requests

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# KEGG Integration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class KEGGPathway:
    """KEGG pathway entry."""
    pathway_id: str = ""
    name: str = ""
    organism: str = "hsa"  # Human
    description: str = ""
    genes: List[str] = field(default_factory=list)
    compounds: List[str] = field(default_factory=list)
    url: str = ""


@dataclass
class KEGGGene:
    """KEGG gene entry."""
    gene_id: str = ""
    symbol: str = ""
    name: str = ""
    definition: str = ""
    pathways: List[str] = field(default_factory=list)


class KEGGClient:
    """Client for KEGG REST API.

    KEGG (Kyoto Encyclopedia of Genes and Genomes) provides pathway
    maps and gene-pathway relationships.

    API docs: https://www.kegg.jp/kegg/rest/keggapi.html
    """

    BASE_URL = "https://rest.kegg.jp"

    def __init__(self, cache_ttl: int = 3600):
        self._cache_ttl = cache_ttl
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Cognisom/1.0 (cognisom.com; research)"
        })

    def search_pathways(self, query: str, organism: str = "hsa") -> List[KEGGPathway]:
        """Search for pathways matching query.

        Args:
            query: Search term (e.g., "prostate cancer", "PI3K")
            organism: Organism code (hsa=human, mmu=mouse)

        Returns:
            List of matching pathways
        """
        try:
            # Search in pathway database
            url = f"{self.BASE_URL}/find/pathway/{query}"
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()

            pathways = []
            for line in resp.text.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    pathway_id = parts[0].replace("path:", "")
                    name = parts[1]

                    # Filter by organism
                    if organism and not pathway_id.startswith(organism):
                        continue

                    pathways.append(KEGGPathway(
                        pathway_id=pathway_id,
                        name=name,
                        organism=organism,
                        url=f"https://www.kegg.jp/pathway/{pathway_id}"
                    ))

            log.info("KEGG search '%s' found %d pathways", query, len(pathways))
            return pathways

        except Exception as e:
            log.error("KEGG search failed: %s", e)
            return []

    def get_pathway(self, pathway_id: str) -> Optional[KEGGPathway]:
        """Get detailed pathway information.

        Args:
            pathway_id: KEGG pathway ID (e.g., "hsa05215")

        Returns:
            Pathway with genes and compounds
        """
        try:
            url = f"{self.BASE_URL}/get/{pathway_id}"
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()

            pathway = KEGGPathway(
                pathway_id=pathway_id,
                url=f"https://www.kegg.jp/pathway/{pathway_id}"
            )

            # Parse KEGG flat file format
            current_section = None
            for line in resp.text.split("\n"):
                if line.startswith("NAME"):
                    pathway.name = line[12:].strip()
                elif line.startswith("DESCRIPTION"):
                    pathway.description = line[12:].strip()
                elif line.startswith("GENE"):
                    current_section = "GENE"
                    gene_part = line[12:].strip()
                    if gene_part:
                        gene_id = gene_part.split()[0]
                        pathway.genes.append(gene_id)
                elif line.startswith("COMPOUND"):
                    current_section = "COMPOUND"
                elif line.startswith(" ") and current_section == "GENE":
                    gene_part = line.strip()
                    if gene_part:
                        gene_id = gene_part.split()[0]
                        if gene_id.isdigit():
                            pathway.genes.append(gene_id)
                elif line.startswith(" ") and current_section == "COMPOUND":
                    compound_part = line.strip().split()[0]
                    pathway.compounds.append(compound_part)
                elif not line.startswith(" "):
                    current_section = None

            return pathway

        except Exception as e:
            log.error("KEGG get pathway failed: %s", e)
            return None

    def get_pathway_genes(self, pathway_id: str) -> List[KEGGGene]:
        """Get genes in a pathway.

        Args:
            pathway_id: KEGG pathway ID

        Returns:
            List of genes with their info
        """
        try:
            # Get pathway gene list
            url = f"{self.BASE_URL}/link/genes/{pathway_id}"
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()

            gene_ids = []
            for line in resp.text.strip().split("\n"):
                if "\t" in line:
                    gene_id = line.split("\t")[1]
                    gene_ids.append(gene_id)

            # Get gene details (batch)
            genes = []
            for gene_id in gene_ids[:50]:  # Limit for API courtesy
                gene = self._get_gene(gene_id)
                if gene:
                    genes.append(gene)
                time.sleep(0.1)  # Rate limit

            return genes

        except Exception as e:
            log.error("KEGG get pathway genes failed: %s", e)
            return []

    def _get_gene(self, gene_id: str) -> Optional[KEGGGene]:
        """Get single gene info."""
        try:
            url = f"{self.BASE_URL}/get/{gene_id}"
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()

            gene = KEGGGene(gene_id=gene_id)

            for line in resp.text.split("\n"):
                if line.startswith("SYMBOL"):
                    gene.symbol = line[12:].strip().split(",")[0]
                elif line.startswith("NAME"):
                    gene.name = line[12:].strip()
                elif line.startswith("DEFINITION"):
                    gene.definition = line[12:].strip()

            return gene

        except:
            return None

    def list_human_pathways(self) -> List[KEGGPathway]:
        """List all human pathways."""
        try:
            url = f"{self.BASE_URL}/list/pathway/hsa"
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()

            pathways = []
            for line in resp.text.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 2:
                    pathway_id = parts[0].replace("path:", "")
                    name = parts[1].replace(" - Homo sapiens (human)", "")
                    pathways.append(KEGGPathway(
                        pathway_id=pathway_id,
                        name=name,
                        organism="hsa",
                    ))

            return pathways

        except Exception as e:
            log.error("KEGG list pathways failed: %s", e)
            return []


# ═══════════════════════════════════════════════════════════════════════
# PubChem Integration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PubChemCompound:
    """PubChem compound entry."""
    cid: int = 0
    name: str = ""
    iupac_name: str = ""
    smiles: str = ""
    inchi: str = ""
    inchi_key: str = ""
    molecular_formula: str = ""
    molecular_weight: float = 0.0
    xlogp: Optional[float] = None
    hbd: int = 0  # H-bond donors
    hba: int = 0  # H-bond acceptors
    tpsa: float = 0.0  # Topological polar surface area
    rotatable_bonds: int = 0
    synonyms: List[str] = field(default_factory=list)
    url: str = ""


class PubChemClient:
    """Client for PubChem PUG REST API.

    PubChem provides compound properties, bioassay data, and target info.

    API docs: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
    """

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Cognisom/1.0 (cognisom.com; research)"
        })

    def search_compounds(self, query: str, limit: int = 10) -> List[PubChemCompound]:
        """Search for compounds by name.

        Args:
            query: Compound name (e.g., "enzalutamide", "docetaxel")
            limit: Maximum results

        Returns:
            List of matching compounds
        """
        try:
            # Search by name
            url = f"{self.BASE_URL}/compound/name/{query}/cids/JSON"
            resp = self._session.get(url, timeout=30)

            if resp.status_code == 404:
                return []

            resp.raise_for_status()
            data = resp.json()

            cids = data.get("IdentifierList", {}).get("CID", [])[:limit]

            compounds = []
            for cid in cids:
                compound = self.get_compound_by_cid(cid)
                if compound:
                    compounds.append(compound)
                time.sleep(0.2)  # Rate limit

            return compounds

        except Exception as e:
            log.error("PubChem search failed: %s", e)
            return []

    def get_compound_by_cid(self, cid: int) -> Optional[PubChemCompound]:
        """Get compound by CID.

        Args:
            cid: PubChem Compound ID

        Returns:
            Compound with properties
        """
        try:
            # Get properties
            props = "MolecularFormula,MolecularWeight,CanonicalSMILES,InChI,InChIKey,IUPACName,XLogP,HBondDonorCount,HBondAcceptorCount,TPSA,RotatableBondCount"
            url = f"{self.BASE_URL}/compound/cid/{cid}/property/{props}/JSON"
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            props_data = data.get("PropertyTable", {}).get("Properties", [{}])[0]

            # Get title/name
            name_url = f"{self.BASE_URL}/compound/cid/{cid}/description/JSON"
            name_resp = self._session.get(name_url, timeout=30)
            name = ""
            if name_resp.status_code == 200:
                name_data = name_resp.json()
                info = name_data.get("InformationList", {}).get("Information", [{}])[0]
                name = info.get("Title", "")

            compound = PubChemCompound(
                cid=cid,
                name=name,
                iupac_name=props_data.get("IUPACName", ""),
                smiles=props_data.get("CanonicalSMILES", ""),
                inchi=props_data.get("InChI", ""),
                inchi_key=props_data.get("InChIKey", ""),
                molecular_formula=props_data.get("MolecularFormula", ""),
                molecular_weight=props_data.get("MolecularWeight", 0.0),
                xlogp=props_data.get("XLogP"),
                hbd=props_data.get("HBondDonorCount", 0),
                hba=props_data.get("HBondAcceptorCount", 0),
                tpsa=props_data.get("TPSA", 0.0),
                rotatable_bonds=props_data.get("RotatableBondCount", 0),
                url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            )

            return compound

        except Exception as e:
            log.error("PubChem get compound failed: %s", e)
            return None

    def get_compound(self, name: str) -> Optional[PubChemCompound]:
        """Get compound by name (convenience method)."""
        compounds = self.search_compounds(name, limit=1)
        return compounds[0] if compounds else None

    def get_compound_image_url(self, cid: int, size: int = 300) -> str:
        """Get 2D structure image URL."""
        return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?image_size={size}x{size}"


# ═══════════════════════════════════════════════════════════════════════
# STRING Integration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class STRINGInteraction:
    """Protein-protein interaction from STRING."""
    protein_a: str = ""
    protein_b: str = ""
    gene_a: str = ""
    gene_b: str = ""
    combined_score: int = 0  # 0-1000
    experimental_score: int = 0
    database_score: int = 0
    textmining_score: int = 0
    coexpression_score: int = 0


@dataclass
class STRINGProtein:
    """Protein from STRING database."""
    string_id: str = ""
    preferred_name: str = ""
    annotation: str = ""
    taxon_id: int = 9606  # Human


class STRINGClient:
    """Client for STRING protein interaction database.

    STRING provides protein-protein interaction networks.

    API docs: https://string-db.org/help/api/
    """

    BASE_URL = "https://string-db.org/api"

    def __init__(self, species: int = 9606):
        """
        Args:
            species: NCBI taxonomy ID (9606=human, 10090=mouse)
        """
        self._species = species
        self._session = requests.Session()

    def get_interactions(
        self,
        proteins: List[str],
        score_threshold: int = 400,
        limit: int = 100
    ) -> List[STRINGInteraction]:
        """Get protein-protein interactions.

        Args:
            proteins: List of protein/gene names
            score_threshold: Minimum combined score (0-1000)
            limit: Maximum interactions to return

        Returns:
            List of interactions
        """
        try:
            proteins_str = "%0d".join(proteins)
            url = f"{self.BASE_URL}/json/network"
            params = {
                "identifiers": proteins_str,
                "species": self._species,
                "required_score": score_threshold,
                "limit": limit,
            }

            resp = self._session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            interactions = []
            for item in data:
                interaction = STRINGInteraction(
                    protein_a=item.get("stringId_A", ""),
                    protein_b=item.get("stringId_B", ""),
                    gene_a=item.get("preferredName_A", ""),
                    gene_b=item.get("preferredName_B", ""),
                    combined_score=int(item.get("score", 0) * 1000),
                    experimental_score=int(item.get("escore", 0) * 1000),
                    database_score=int(item.get("dscore", 0) * 1000),
                    textmining_score=int(item.get("tscore", 0) * 1000),
                )
                interactions.append(interaction)

            return interactions

        except Exception as e:
            log.error("STRING get interactions failed: %s", e)
            return []

    def get_network_image(
        self,
        proteins: List[str],
        network_type: str = "physical"
    ) -> str:
        """Get network visualization URL.

        Args:
            proteins: List of protein/gene names
            network_type: "physical" or "functional"

        Returns:
            URL to network image
        """
        proteins_str = "%0d".join(proteins)
        return (
            f"{self.BASE_URL}/image/network?"
            f"identifiers={proteins_str}&"
            f"species={self._species}&"
            f"network_type={network_type}&"
            f"add_white_nodes=5"
        )

    def get_enrichment(
        self,
        proteins: List[str],
        category: str = "Process"
    ) -> List[Dict[str, Any]]:
        """Get functional enrichment for protein set.

        Args:
            proteins: List of protein/gene names
            category: GO category (Process, Function, Component) or KEGG, Reactome

        Returns:
            List of enriched terms
        """
        try:
            proteins_str = "%0d".join(proteins)
            url = f"{self.BASE_URL}/json/enrichment"
            params = {
                "identifiers": proteins_str,
                "species": self._species,
            }

            resp = self._session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            # Filter by category
            results = []
            for item in data:
                if category.lower() in item.get("category", "").lower():
                    results.append({
                        "term": item.get("term", ""),
                        "description": item.get("description", ""),
                        "p_value": item.get("p_value", 1.0),
                        "fdr": item.get("fdr", 1.0),
                        "genes": item.get("inputGenes", "").split(","),
                    })

            return sorted(results, key=lambda x: x["fdr"])

        except Exception as e:
            log.error("STRING enrichment failed: %s", e)
            return []


# ═══════════════════════════════════════════════════════════════════════
# Reactome Integration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ReactomePathway:
    """Reactome pathway entry."""
    stable_id: str = ""
    name: str = ""
    species: str = "Homo sapiens"
    diagram_url: str = ""
    is_disease: bool = False


class ReactomeClient:
    """Client for Reactome pathway database.

    Reactome provides detailed pathway information and analysis.

    API docs: https://reactome.org/ContentService/
    """

    BASE_URL = "https://reactome.org/ContentService"

    def __init__(self):
        self._session = requests.Session()

    def search_pathways(self, query: str, species: str = "Homo sapiens") -> List[ReactomePathway]:
        """Search for pathways."""
        try:
            url = f"{self.BASE_URL}/search/query"
            params = {
                "query": query,
                "species": species,
                "types": "Pathway",
                "cluster": "true",
            }

            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            pathways = []
            for group in data.get("results", []):
                for entry in group.get("entries", []):
                    pathway = ReactomePathway(
                        stable_id=entry.get("stId", ""),
                        name=entry.get("name", ""),
                        species=entry.get("species", [species])[0] if entry.get("species") else species,
                        is_disease=entry.get("isDisease", False),
                    )
                    pathways.append(pathway)

            return pathways

        except Exception as e:
            log.error("Reactome search failed: %s", e)
            return []

    def get_pathway_diagram_url(self, pathway_id: str) -> str:
        """Get pathway diagram image URL."""
        return f"https://reactome.org/ContentService/exporter/diagram/{pathway_id}.png?quality=7"

    def analyze_genes(self, genes: List[str]) -> List[Dict[str, Any]]:
        """Analyze gene list for pathway enrichment.

        Args:
            genes: List of gene symbols

        Returns:
            Enriched pathways with p-values
        """
        try:
            url = f"{self.BASE_URL}/identifiers/projection"
            data = "\n".join(genes)

            resp = self._session.post(
                url,
                data=data,
                headers={"Content-Type": "text/plain"},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()

            # Get enrichment token
            token = result.get("summary", {}).get("token")
            if not token:
                return []

            # Get enrichment results
            enrich_url = f"https://reactome.org/AnalysisService/token/{token}"
            enrich_resp = self._session.get(enrich_url, timeout=30)
            enrich_resp.raise_for_status()
            enrich_data = enrich_resp.json()

            pathways = []
            for pw in enrich_data.get("pathways", [])[:50]:
                pathways.append({
                    "pathway_id": pw.get("stId", ""),
                    "name": pw.get("name", ""),
                    "p_value": pw.get("entities", {}).get("pValue", 1.0),
                    "fdr": pw.get("entities", {}).get("fdr", 1.0),
                    "found_genes": pw.get("entities", {}).get("found", 0),
                    "total_genes": pw.get("entities", {}).get("total", 0),
                })

            return sorted(pathways, key=lambda x: x["fdr"])

        except Exception as e:
            log.error("Reactome analysis failed: %s", e)
            return []


# ═══════════════════════════════════════════════════════════════════════
# Unified Data Source Manager
# ═══════════════════════════════════════════════════════════════════════

class ExternalDataManager:
    """Unified manager for all external data sources."""

    def __init__(self):
        self._kegg = KEGGClient()
        self._pubchem = PubChemClient()
        self._string = STRINGClient()
        self._reactome = ReactomeClient()

    @property
    def kegg(self) -> KEGGClient:
        return self._kegg

    @property
    def pubchem(self) -> PubChemClient:
        return self._pubchem

    @property
    def string(self) -> STRINGClient:
        return self._string

    @property
    def reactome(self) -> ReactomeClient:
        return self._reactome

    def search_all(self, query: str) -> Dict[str, Any]:
        """Search across all data sources.

        Args:
            query: Search term

        Returns:
            Results from each source
        """
        results = {
            "kegg_pathways": [],
            "pubchem_compounds": [],
            "reactome_pathways": [],
        }

        try:
            results["kegg_pathways"] = self._kegg.search_pathways(query)[:5]
        except:
            pass

        try:
            results["pubchem_compounds"] = self._pubchem.search_compounds(query, limit=3)
        except:
            pass

        try:
            results["reactome_pathways"] = self._reactome.search_pathways(query)[:5]
        except:
            pass

        return results


# Singleton instance
_data_manager: Optional[ExternalDataManager] = None


def get_data_manager() -> ExternalDataManager:
    """Get the global external data manager."""
    global _data_manager
    if _data_manager is None:
        _data_manager = ExternalDataManager()
    return _data_manager
