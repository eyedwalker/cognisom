"""
Data Loaders — Populate the entity library from public databases
================================================================

Loaders query public APIs (NCBI Gene, UniProt, PDB, Reactome, ChEMBL)
and create BioEntity objects that are inserted into the EntityStore.

Usage:
    store = EntityStore()
    loader = EntityLoader(store)
    loader.load_gene("TP53")
    loader.load_protein("TP53")
    loader.load_prostate_cancer_catalog()  # batch load essential entities
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Any, List, Optional

from .models import (
    BioEntity,
    CellTypeEntity,
    Drug,
    EntityType,
    Gene,
    Ligand,
    Metabolite,
    Mutation,
    OrganEntity,
    Pathway,
    Protein,
    Receptor,
    Relationship,
    RelationshipType,
    TissueTypeEntity,
)
from .store import EntityStore

log = logging.getLogger(__name__)


def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Cognisom/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def _http_get_json(url: str, timeout: int = 30) -> Any:
    return json.loads(_http_get(url, timeout=timeout))


def _nested_get(d: dict, *keys: str) -> str:
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, {})
        else:
            return ""
    return d if isinstance(d, str) else ""


class EntityLoader:
    """Load biological entities from public APIs into the EntityStore."""

    def __init__(self, store: EntityStore):
        self.store = store

    # ── Gene (NCBI Gene) ─────────────────────────────────────────────

    def load_gene(self, query: str, gene_type: str = "") -> Optional[Gene]:
        """Load a gene from NCBI Gene by name or ID."""
        try:
            search_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=gene&term={urllib.parse.quote(query)}[Gene Name]+AND+Homo+sapiens[Organism]"
                "&retmax=1&retmode=json"
            )
            sr = _http_get_json(search_url)
            ids = sr.get("esearchresult", {}).get("idlist", [])
            if not ids:
                log.warning("No NCBI Gene found for '%s'", query)
                return None

            gene_id = ids[0]

            sum_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=gene&id={gene_id}&retmode=json"
            )
            sd = _http_get_json(sum_url)
            doc = sd.get("result", {}).get(gene_id, {})

            symbol = doc.get("name", query)
            gene = Gene(
                name=symbol,
                display_name=symbol,
                description=doc.get("summary", "")[:500],
                symbol=symbol,
                full_name=doc.get("description", ""),
                chromosome=doc.get("chromosome", ""),
                gene_type=gene_type or doc.get("genetype", ""),
                map_location=doc.get("maplocation", ""),
                organism=doc.get("organism", {}).get("scientificname", "Homo sapiens"),
                synonyms=[s.strip() for s in doc.get("otheraliases", "").split(",") if s.strip()],
                external_ids={"ncbi_gene": gene_id},
                source="ncbi_gene",
                source_url=f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}",
                tags=["prostate_cancer"] if gene_type else [],
            )

            self.store.add_entity(gene)
            log.info("Loaded gene: %s (NCBI:%s)", symbol, gene_id)
            return gene
        except Exception as e:
            log.error("Failed to load gene '%s': %s", query, e)
            return None

    # ── Protein (UniProt) ────────────────────────────────────────────

    def load_protein(self, gene_name: str) -> Optional[Protein]:
        """Load a protein from UniProt by gene name."""
        try:
            search_url = (
                "https://rest.uniprot.org/uniprotkb/search"
                f"?query={urllib.parse.quote(gene_name)}+AND+organism_id:9606"
                "&format=json&fields=accession,gene_names,protein_name,sequence,"
                "go_p,go_f,go_c,cc_function,cc_pathway,length&size=1"
            )
            req = urllib.request.Request(
                search_url,
                headers={"User-Agent": "Cognisom/1.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())

            results = data.get("results", [])
            if not results:
                log.warning("No UniProt protein found for '%s'", gene_name)
                return None

            entry = results[0]
            accession = entry.get("primaryAccession", "")
            prot_name = _nested_get(
                entry, "proteinDescription", "recommendedName", "fullName", "value"
            )

            # GO terms
            go_terms = []
            for section in ("go_p", "go_f", "go_c"):
                for go in entry.get(section, []):
                    if isinstance(go, dict):
                        go_terms.append(f"{go.get('id', '')} - {go.get('name', '')}")

            # Function
            functions = []
            for comment in entry.get("cc_function", []) if isinstance(entry.get("cc_function"), list) else []:
                if isinstance(comment, dict):
                    for txt in comment.get("texts", []):
                        if isinstance(txt, dict):
                            functions.append(txt.get("value", ""))

            seq_obj = entry.get("sequence", {})
            seq_val = seq_obj.get("value", "")

            pathways = []
            for p in entry.get("cc_pathway", []) if isinstance(entry.get("cc_pathway"), list) else []:
                if isinstance(p, dict):
                    pathways.append(p.get("name", ""))

            protein = Protein(
                name=prot_name or gene_name,
                display_name=prot_name or gene_name,
                description="; ".join(functions)[:500],
                gene_source=gene_name,
                uniprot_id=accession,
                amino_acid_length=seq_obj.get("length", 0),
                sequence_preview=seq_val[:100] if seq_val else "",
                function_summary="; ".join(functions)[:300],
                go_terms=go_terms[:15],
                pathways=pathways,
                external_ids={"uniprot": accession},
                ontology_ids=[g.split(" - ")[0] for g in go_terms[:10] if g.startswith("GO:")],
                source="uniprot",
                source_url=f"https://www.uniprot.org/uniprot/{accession}",
                tags=["prostate_cancer"],
            )

            self.store.add_entity(protein)
            log.info("Loaded protein: %s (UniProt:%s)", prot_name, accession)
            return protein
        except Exception as e:
            log.error("Failed to load protein '%s': %s", gene_name, e)
            return None

    # ── PDB structures ───────────────────────────────────────────────

    def load_pdb_ids(self, gene_name: str, protein_entity_id: str = "") -> List[str]:
        """Load PDB IDs for a gene and optionally link to a protein entity."""
        try:
            search_body = json.dumps({
                "query": {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {"value": f"{gene_name} Homo sapiens"},
                },
                "return_type": "entry",
                "request_options": {"paginate": {"start": 0, "rows": 5}},
            })
            req = urllib.request.Request(
                "https://search.rcsb.org/rcsbsearch/v2/query",
                data=search_body.encode(),
                headers={"Content-Type": "application/json", "User-Agent": "Cognisom/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())

            pdb_ids = [r["identifier"] for r in data.get("result_set", [])]
            log.info("Found %d PDB structures for %s", len(pdb_ids), gene_name)

            # If we have a protein entity, update its pdb_ids
            if protein_entity_id and pdb_ids:
                entity = self.store.get_entity(protein_entity_id)
                if entity and isinstance(entity, Protein):
                    entity.pdb_ids = pdb_ids
                    self.store.update_entity(entity)

            return pdb_ids
        except Exception as e:
            log.error("PDB search failed for '%s': %s", gene_name, e)
            return []

    # ── Batch: Gene + Protein + Relationship ─────────────────────────

    def load_gene_protein_pair(
        self, gene_name: str, gene_type: str = ""
    ) -> tuple:
        """Load a gene, its protein, and create an 'encodes' relationship."""
        gene = self.load_gene(gene_name, gene_type=gene_type)
        protein = self.load_protein(gene_name)

        if gene and protein:
            rel = Relationship(
                source_id=gene.entity_id,
                target_id=protein.entity_id,
                rel_type=RelationshipType.ENCODES,
                confidence=1.0,
                evidence="NCBI Gene + UniProt cross-reference",
            )
            self.store.add_relationship(rel)
            log.info("Created encodes relationship: %s -> %s", gene.name, protein.name)

            # Try to get PDB structures
            self.load_pdb_ids(gene_name, protein.entity_id)

        return gene, protein

    # ── Manual entity creators ───────────────────────────────────────

    def add_drug(
        self, name: str, drug_class: str, mechanism: str,
        targets: List[str], approval_status: str = "approved",
        smiles: str = "", tags: Optional[List[str]] = None,
    ) -> Drug:
        """Manually add a drug entity."""
        drug = Drug(
            name=name,
            display_name=name,
            description=mechanism,
            drug_class=drug_class,
            mechanism=mechanism,
            targets=targets,
            smiles=smiles,
            approval_status=approval_status,
            source="manual",
            tags=tags or ["prostate_cancer"],
        )
        self.store.add_entity(drug)
        return drug

    def add_mutation(
        self, gene_symbol: str, position: str, mutation_type: str,
        consequence: str, clinical_significance: str = "",
        frequency: float = 0.0, tags: Optional[List[str]] = None,
    ) -> Mutation:
        """Manually add a mutation entity."""
        name = f"{gene_symbol}_{position}"
        mutation = Mutation(
            name=name,
            display_name=f"{gene_symbol} {position}",
            description=f"{mutation_type} mutation in {gene_symbol} at {position}; {consequence}",
            gene_symbol=gene_symbol,
            mutation_type=mutation_type,
            position=position,
            consequence=consequence,
            frequency=frequency,
            clinical_significance=clinical_significance,
            source="manual",
            tags=tags or ["prostate_cancer"],
        )
        self.store.add_entity(mutation)
        return mutation

    def add_cell_type(
        self, name: str, description: str, tissue_origin: str,
        markers: List[str], lineage: str = "",
        cell_ontology_id: str = "", tags: Optional[List[str]] = None,
    ) -> CellTypeEntity:
        """Manually add a cell type entity."""
        ct = CellTypeEntity(
            name=name,
            display_name=name,
            description=description,
            cell_ontology_id=cell_ontology_id,
            tissue_origin=tissue_origin,
            markers=markers,
            lineage=lineage,
            source="manual",
            tags=tags or ["prostate_cancer"],
        )
        self.store.add_entity(ct)
        return ct

    def add_pathway(
        self, name: str, description: str, pathway_type: str,
        genes: List[str], proteins: Optional[List[str]] = None,
        reactome_id: str = "", tags: Optional[List[str]] = None,
    ) -> Pathway:
        """Manually add a pathway entity."""
        pw = Pathway(
            name=name,
            display_name=name,
            description=description,
            pathway_type=pathway_type,
            genes=genes,
            proteins=proteins or [],
            reactome_id=reactome_id,
            source="manual",
            tags=tags or ["prostate_cancer"],
        )
        self.store.add_entity(pw)
        return pw

    def add_metabolite(
        self, name: str, description: str, molecular_formula: str = "",
        molecular_weight: float = 0.0, chebi_id: str = "",
        tags: Optional[List[str]] = None,
    ) -> Metabolite:
        """Manually add a metabolite entity."""
        met = Metabolite(
            name=name,
            display_name=name,
            description=description,
            molecular_formula=molecular_formula,
            molecular_weight=molecular_weight,
            chebi_id=chebi_id,
            source="manual",
            tags=tags or ["metabolism"],
        )
        self.store.add_entity(met)
        return met

    def add_tissue_type(
        self, name: str, description: str, organ: str,
        cell_types: List[str], uberon_id: str = "",
        tags: Optional[List[str]] = None,
    ) -> TissueTypeEntity:
        """Manually add a tissue type."""
        tt = TissueTypeEntity(
            name=name,
            display_name=name,
            description=description,
            organ=organ,
            cell_types=cell_types,
            uberon_id=uberon_id,
            source="manual",
            tags=tags or ["prostate_cancer"],
        )
        self.store.add_entity(tt)
        return tt

    def add_receptor(
        self, name: str, description: str, receptor_type: str,
        ligands: List[str], signaling_pathway: str = "",
        gene_source: str = "", tags: Optional[List[str]] = None,
    ) -> Receptor:
        """Manually add a receptor entity."""
        rec = Receptor(
            name=name,
            display_name=name,
            description=description,
            receptor_type=receptor_type,
            ligands=ligands,
            signaling_pathway=signaling_pathway,
            gene_source=gene_source,
            source="manual",
            tags=tags or ["prostate_cancer"],
        )
        self.store.add_entity(rec)
        return rec

    def add_organ(
        self, name: str, description: str, system: str,
        tissue_types: List[str], uberon_id: str = "",
        tags: Optional[List[str]] = None,
    ) -> OrganEntity:
        """Manually add an organ entity."""
        org = OrganEntity(
            name=name,
            display_name=name,
            description=description,
            system=system,
            tissue_types=tissue_types,
            uberon_id=uberon_id,
            source="manual",
            tags=tags or ["anatomy"],
        )
        self.store.add_entity(org)
        return org
