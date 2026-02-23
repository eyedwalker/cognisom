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
# PDB / RCSB Integration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PDBEntry:
    """RCSB PDB entry with structural metadata for visualization."""
    pdb_id: str = ""
    title: str = ""
    resolution: float = 0.0
    method: str = ""  # X-RAY DIFFRACTION, ELECTRON MICROSCOPY, NMR
    chain_count: int = 0
    molecular_weight: float = 0.0
    organism: str = ""
    ligand_ids: List[str] = field(default_factory=list)
    deposition_date: str = ""
    doi: str = ""
    url: str = ""


class PDBClient:
    """Client for RCSB PDB Search and Data APIs.

    Provides structural metadata needed for RTX-accelerated
    molecular visualization (resolution, method, chain count).

    API docs: https://search.rcsb.org/ and https://data.rcsb.org/
    """

    SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
    DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Cognisom/1.0 (cognisom.com; research)",
            "Content-Type": "application/json",
        })

    def search_by_gene(self, gene_name: str, limit: int = 5) -> List[str]:
        """Search PDB for structures of a gene/protein.

        Args:
            gene_name: Gene symbol (e.g., "TP53", "BRCA1")
            limit: Maximum PDB IDs to return

        Returns:
            List of PDB IDs sorted by resolution (best first)
        """
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "full_text",
                        "parameters": {"value": gene_name}
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entity_source_organism.ncbi_scientific_name",
                            "operator": "exact_match",
                            "value": "Homo sapiens"
                        }
                    }
                ]
            },
            "return_type": "entry",
            "request_options": {
                "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
                "paginate": {"start": 0, "rows": limit}
            }
        }

        try:
            resp = self._session.post(self.SEARCH_URL, json=query, timeout=30)
            if resp.status_code == 204:
                return []
            resp.raise_for_status()
            data = resp.json()
            return [hit["identifier"] for hit in data.get("result_set", [])]
        except Exception as e:
            log.error("PDB search for %s failed: %s", gene_name, e)
            return []

    def get_entry_info(self, pdb_id: str) -> Optional[PDBEntry]:
        """Get structural metadata for a PDB entry.

        Args:
            pdb_id: PDB ID (e.g., "1TUP", "6GGF")

        Returns:
            PDBEntry with resolution, method, chain count
        """
        try:
            url = f"{self.DATA_URL}/{pdb_id}"
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            info = data.get("rcsb_entry_info", {})
            container = data.get("rcsb_entry_container_identifiers", {})

            # Get ligand IDs
            ligands = []
            for comp_id in container.get("non_polymer_entity_ids", []):
                ligands.append(comp_id)

            # Get citation DOI
            doi = ""
            citations = data.get("rcsb_primary_citation", {})
            if citations:
                doi = citations.get("pdbx_database_id_DOI", "")

            entry = PDBEntry(
                pdb_id=pdb_id,
                title=data.get("struct", {}).get("title", ""),
                resolution=info.get("resolution_combined", [0.0])[0] if info.get("resolution_combined") else 0.0,
                method=info.get("experimental_method", ""),
                chain_count=info.get("polymer_entity_count", 0),
                molecular_weight=info.get("molecular_weight", 0.0),
                ligand_ids=ligands,
                deposition_date=info.get("deposit_date", ""),
                doi=doi,
                url=f"https://www.rcsb.org/structure/{pdb_id}",
            )

            return entry

        except Exception as e:
            log.error("PDB get entry %s failed: %s", pdb_id, e)
            return None

    def get_best_structure(self, gene_name: str) -> Optional[PDBEntry]:
        """Get the best (highest resolution) structure for a gene."""
        pdb_ids = self.search_by_gene(gene_name, limit=1)
        if pdb_ids:
            return self.get_entry_info(pdb_ids[0])
        return None


# ═══════════════════════════════════════════════════════════════════════
# ChEBI Integration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ChEBIEntity:
    """ChEBI chemical entity."""
    chebi_id: str = ""
    name: str = ""
    definition: str = ""
    formula: str = ""
    mass: float = 0.0
    inchi: str = ""
    inchi_key: str = ""
    smiles: str = ""


class ChEBIClient:
    """Client for ChEBI (Chemical Entities of Biological Interest).

    API docs: https://www.ebi.ac.uk/chebi/webServices.do
    """

    BASE_URL = "https://www.ebi.ac.uk/chebi/searchId.do"
    SEARCH_URL = "https://www.ebi.ac.uk/chebi/advancedSearchFwd.do"
    REST_URL = "https://www.ebi.ac.uk/webservices/chebi/2.0/test"

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Cognisom/1.0 (cognisom.com; research)"
        })

    def get_entity(self, chebi_id: str) -> Optional[ChEBIEntity]:
        """Get ChEBI entity by ID.

        Args:
            chebi_id: ChEBI ID (e.g., "CHEBI:15377" for water)

        Returns:
            ChEBI entity with formula, mass, InChI
        """
        clean_id = chebi_id.replace("CHEBI:", "")
        try:
            url = f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:{clean_id}"
            # Use the OLS API for structured data
            ols_url = f"https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms?iri=http://purl.obolibrary.org/obo/CHEBI_{clean_id}"
            resp = self._session.get(ols_url, timeout=30)
            if resp.status_code != 200:
                return None

            data = resp.json()
            terms = data.get("_embedded", {}).get("terms", [])
            if not terms:
                return None

            term = terms[0]
            annot = term.get("annotation", {})

            return ChEBIEntity(
                chebi_id=f"CHEBI:{clean_id}",
                name=term.get("label", ""),
                definition=term.get("description", [""])[0] if term.get("description") else "",
                formula=annot.get("formula", [""])[0] if annot.get("formula") else "",
                mass=float(annot.get("mass", ["0"])[0]) if annot.get("mass") else 0.0,
                inchi=annot.get("inchi", [""])[0] if annot.get("inchi") else "",
                inchi_key=annot.get("inchikey", [""])[0] if annot.get("inchikey") else "",
                smiles=annot.get("smiles", [""])[0] if annot.get("smiles") else "",
            )
        except Exception as e:
            log.error("ChEBI get entity %s failed: %s", chebi_id, e)
            return None

    def search(self, query: str, limit: int = 10) -> List[ChEBIEntity]:
        """Search ChEBI by name.

        Args:
            query: Chemical name (e.g., "testosterone", "glucose")
            limit: Max results

        Returns:
            List of matching ChEBI entities
        """
        try:
            url = f"https://www.ebi.ac.uk/ols4/api/search?q={query}&ontology=chebi&rows={limit}"
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for doc in data.get("response", {}).get("docs", []):
                obo_id = doc.get("obo_id", "")
                if obo_id.startswith("CHEBI:"):
                    results.append(ChEBIEntity(
                        chebi_id=obo_id,
                        name=doc.get("label", ""),
                        definition=doc.get("description", [""])[0] if doc.get("description") else "",
                    ))

            return results
        except Exception as e:
            log.error("ChEBI search '%s' failed: %s", query, e)
            return []


# ═══════════════════════════════════════════════════════════════════════
# AlphaFold DB Integration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AlphaFoldEntry:
    """AlphaFold predicted structure entry."""
    uniprot_id: str = ""
    model_url: str = ""      # .pdb file URL
    cif_url: str = ""        # .cif file URL
    pae_url: str = ""        # predicted aligned error
    confidence_url: str = ""  # per-residue confidence
    mean_plddt: float = 0.0  # overall confidence 0-100


class AlphaFoldClient:
    """Client for AlphaFold Protein Structure Database.

    Provides predicted structures for proteins without experimental PDB entries.
    These are visualization-ready PDB files for RTX ray tracing.
    """

    BASE_URL = "https://alphafold.ebi.ac.uk/api"

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Cognisom/1.0 (cognisom.com; research)"
        })

    def get_prediction(self, uniprot_id: str) -> Optional[AlphaFoldEntry]:
        """Get AlphaFold prediction for a UniProt ID.

        Args:
            uniprot_id: UniProt accession (e.g., "P04637" for TP53)

        Returns:
            AlphaFoldEntry with model URLs and confidence
        """
        try:
            url = f"{self.BASE_URL}/prediction/{uniprot_id}"
            resp = self._session.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()

            if not data:
                return None

            entry_data = data[0] if isinstance(data, list) else data
            return AlphaFoldEntry(
                uniprot_id=uniprot_id,
                model_url=entry_data.get("pdbUrl", ""),
                cif_url=entry_data.get("cifUrl", ""),
                pae_url=entry_data.get("paeImageUrl", ""),
                confidence_url=entry_data.get("confidenceUrl", ""),
                mean_plddt=entry_data.get("globalMetrics", {}).get("globalMetricValue", 0.0)
                           if entry_data.get("globalMetrics") else 0.0,
            )
        except Exception as e:
            log.error("AlphaFold prediction for %s failed: %s", uniprot_id, e)
            return None

    def get_model_url(self, uniprot_id: str) -> str:
        """Get direct URL to AlphaFold PDB model file."""
        return f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"


# ═══════════════════════════════════════════════════════════════════════
# UniProt REST Integration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class UniProtEntry:
    """UniProt protein entry with rich annotation."""
    accession: str = ""
    gene_name: str = ""
    protein_name: str = ""
    organism: str = ""
    sequence: str = ""
    length: int = 0
    mass: float = 0.0
    function_summary: str = ""
    subcellular_location: str = ""
    tissue_specificity: str = ""
    go_terms: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    disease_associations: List[str] = field(default_factory=list)
    pdb_ids: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


class UniProtClient:
    """Client for UniProt REST API.

    Provides comprehensive protein annotation including function,
    structure, location, and disease associations.

    API docs: https://rest.uniprot.org/
    """

    BASE_URL = "https://rest.uniprot.org/uniprotkb"

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Cognisom/1.0 (cognisom.com; research)",
            "Accept": "application/json",
        })

    def get_entry(self, accession: str) -> Optional[UniProtEntry]:
        """Get full UniProt entry by accession.

        Args:
            accession: UniProt accession (e.g., "P04637")

        Returns:
            UniProtEntry with comprehensive annotation
        """
        try:
            url = f"{self.BASE_URL}/{accession}"
            resp = self._session.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()

            # Gene name
            gene_name = ""
            genes = data.get("genes", [])
            if genes:
                gene_name = genes[0].get("geneName", {}).get("value", "")

            # Protein name
            protein_name = ""
            prot_desc = data.get("proteinDescription", {})
            rec_name = prot_desc.get("recommendedName", {})
            if rec_name:
                protein_name = rec_name.get("fullName", {}).get("value", "")

            # Function from comments
            function_summary = ""
            subcellular = ""
            tissue_spec = ""
            diseases = []
            for comment in data.get("comments", []):
                ctype = comment.get("commentType", "")
                if ctype == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        function_summary = texts[0].get("value", "")
                elif ctype == "SUBCELLULAR LOCATION":
                    locs = comment.get("subcellularLocations", [])
                    parts = []
                    for loc in locs:
                        loc_val = loc.get("location", {}).get("value", "")
                        if loc_val:
                            parts.append(loc_val)
                    subcellular = "; ".join(parts[:5])
                elif ctype == "TISSUE SPECIFICITY":
                    texts = comment.get("texts", [])
                    if texts:
                        tissue_spec = texts[0].get("value", "")[:500]
                elif ctype == "DISEASE":
                    disease = comment.get("disease", {})
                    dname = disease.get("diseaseId", "")
                    if dname:
                        diseases.append(dname)

            # GO terms
            go_terms = []
            for xref in data.get("uniProtKBCrossReferences", []):
                if xref.get("database") == "GO":
                    go_id = xref.get("id", "")
                    props = xref.get("properties", [])
                    term_name = ""
                    for p in props:
                        if p.get("key") == "GoTerm":
                            term_name = p.get("value", "")
                    go_terms.append(f"{go_id}: {term_name}" if term_name else go_id)

            # Domains
            domains = []
            for feat in data.get("features", []):
                if feat.get("type") == "Domain":
                    desc = feat.get("description", "")
                    if desc:
                        domains.append(desc)

            # PDB cross-refs
            pdb_ids = []
            for xref in data.get("uniProtKBCrossReferences", []):
                if xref.get("database") == "PDB":
                    pdb_ids.append(xref.get("id", ""))

            # Keywords
            keywords = [kw.get("name", "") for kw in data.get("keywords", []) if kw.get("name")]

            # Sequence
            seq_data = data.get("sequence", {})

            return UniProtEntry(
                accession=accession,
                gene_name=gene_name,
                protein_name=protein_name,
                organism=data.get("organism", {}).get("scientificName", ""),
                sequence=seq_data.get("value", "")[:100],
                length=seq_data.get("length", 0),
                mass=seq_data.get("molWeight", 0.0),
                function_summary=function_summary[:1000],
                subcellular_location=subcellular,
                tissue_specificity=tissue_spec,
                go_terms=go_terms[:50],
                domains=domains[:20],
                disease_associations=diseases[:10],
                pdb_ids=pdb_ids[:20],
                keywords=keywords[:30],
            )

        except Exception as e:
            log.error("UniProt get %s failed: %s", accession, e)
            return None

    def search_by_gene(self, gene_name: str, organism: str = "Homo sapiens") -> Optional[str]:
        """Search UniProt by gene name and return the best accession.

        Args:
            gene_name: Gene symbol (e.g., "TP53")
            organism: Organism name

        Returns:
            UniProt accession or None
        """
        try:
            url = f"{self.BASE_URL}/search"
            params = {
                "query": f"gene_exact:{gene_name} AND organism_name:\"{organism}\" AND reviewed:true",
                "format": "json",
                "size": 1,
                "fields": "accession",
            }
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                return results[0].get("primaryAccession", "")
            return None
        except Exception as e:
            log.error("UniProt search for %s failed: %s", gene_name, e)
            return None


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
        self._pdb = PDBClient()
        self._chebi = ChEBIClient()
        self._alphafold = AlphaFoldClient()
        self._uniprot = UniProtClient()

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

    @property
    def pdb(self) -> PDBClient:
        return self._pdb

    @property
    def chebi(self) -> ChEBIClient:
        return self._chebi

    @property
    def alphafold(self) -> AlphaFoldClient:
        return self._alphafold

    @property
    def uniprot(self) -> UniProtClient:
        return self._uniprot

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
