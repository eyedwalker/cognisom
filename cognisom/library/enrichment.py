"""
Multi-Source Enrichment Engine
===============================

Orchestrates API calls across NCBI Gene, UniProt, PDB, AlphaFold,
KEGG, STRING, Reactome, PubChem, ChEBI, and ClinVar to build
richly-annotated entities with visualization-ready structural data.

Each entity is enriched with:
  - Shape: PDB 3D coordinates, AlphaFold predictions
  - Composition: sequences, SMILES, chemical properties
  - Interactions: KEGG/Reactome/STRING networks → Relationships
  - Lifecycles: expression, disease associations
  - Human interpretation: summaries, function, mechanism text

Usage:
    store = EntityStore()
    enricher = EntityEnricher(store)
    report = enricher.enrich_gene("TP53")
    report = enricher.enrich_drug("Enzalutamide")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .models import (
    BioEntity,
    Drug,
    EntityType,
    Gene,
    Mutation,
    Pathway,
    Protein,
    Relationship,
    RelationshipType,
)
from .store import EntityStore
from .external_sources import (
    AlphaFoldClient,
    ChEBIClient,
    ExternalDataManager,
    KEGGClient,
    PDBClient,
    PubChemClient,
    ReactomeClient,
    STRINGClient,
    UniProtClient,
)

log = logging.getLogger(__name__)


# ── Enrichment Report ────────────────────────────────────────────────

@dataclass
class EnrichmentReport:
    """Report from an enrichment run."""
    entities_created: int = 0
    entities_updated: int = 0
    entities_skipped: int = 0
    relationships_created: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    by_type: Dict[str, int] = field(default_factory=dict)

    def merge(self, other: EnrichmentReport):
        self.entities_created += other.entities_created
        self.entities_updated += other.entities_updated
        self.entities_skipped += other.entities_skipped
        self.relationships_created += other.relationships_created
        self.errors.extend(other.errors)
        self.duration_seconds += other.duration_seconds
        for k, v in other.by_type.items():
            self.by_type[k] = self.by_type.get(k, 0) + v


# ── Entity Enricher ─────────────────────────────────────────────────

class EntityEnricher:
    """Multi-source enrichment engine for biological entities.

    Orchestrates API calls to populate entity fields with data from
    NCBI, UniProt, PDB, AlphaFold, KEGG, STRING, Reactome, PubChem,
    ClinVar, and ChEBI.
    """

    def __init__(self, store: EntityStore):
        self.store = store
        self._dm = ExternalDataManager()
        self._ncbi = None  # lazy init

    @property
    def ncbi(self):
        if self._ncbi is None:
            from cognisom.ncbi.client import NCBIClient
            self._ncbi = NCBIClient()
        return self._ncbi

    # ── Gene Enrichment Pipeline ──────────────────────────────────────

    def enrich_gene(
        self, symbol: str,
        create_protein: bool = True,
        create_relationships: bool = True,
        tags: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> EnrichmentReport:
        """Full gene enrichment pipeline.

        1. NCBI Gene → symbol, full_name, chromosome, summary, aliases
        2. UniProt → uniprot_id, function, GO terms, domains
        3. PDB → experimental structures for visualization
        4. AlphaFold → predicted structure URL
        5. KEGG → pathway memberships
        6. STRING → interaction partners → Relationships
        7. ClinVar → clinical variants → Mutation entities

        Returns:
            EnrichmentReport with counts of created/updated entities
        """
        report = EnrichmentReport()
        t0 = time.time()

        def _progress(msg: str):
            if progress_callback:
                progress_callback(msg)
            log.info("Enrich %s: %s", symbol, msg)

        # Check for existing entity
        existing = self.store.find_entity_by_name(symbol, "gene")

        # Step 1: NCBI Gene
        _progress("Querying NCBI Gene...")
        gene = self._enrich_gene_ncbi(symbol, existing, tags)
        if gene is None:
            report.errors.append(f"NCBI Gene lookup failed for {symbol}")
            report.duration_seconds = time.time() - t0
            return report

        if existing:
            report.entities_updated += 1
        else:
            report.entities_created += 1
        report.by_type["gene"] = report.by_type.get("gene", 0) + 1

        # Step 2: UniProt — find accession for this gene
        uniprot_accession = ""
        _progress("Querying UniProt...")
        try:
            uniprot_accession = self._dm.uniprot.search_by_gene(symbol) or ""
            if uniprot_accession:
                gene.external_ids["uniprot"] = uniprot_accession
        except Exception as e:
            report.errors.append(f"UniProt search: {e}")

        # Step 3: Create Protein entity if UniProt found
        protein = None
        if create_protein and uniprot_accession:
            _progress("Enriching protein from UniProt...")
            protein = self._create_protein_from_uniprot(
                uniprot_accession, symbol, tags
            )
            if protein:
                report.entities_created += 1
                report.by_type["protein"] = report.by_type.get("protein", 0) + 1

                if create_relationships:
                    rel = Relationship(
                        source_id=gene.entity_id,
                        target_id=protein.entity_id,
                        rel_type=RelationshipType.ENCODES,
                        confidence=1.0,
                        evidence="NCBI Gene + UniProt cross-reference",
                    )
                    if self.store.add_relationship(rel):
                        report.relationships_created += 1

        # Step 4: PDB structures
        if protein and uniprot_accession:
            _progress("Querying PDB...")
            try:
                pdb_ids = self._dm.pdb.search_by_gene(symbol, limit=5)
                if pdb_ids:
                    protein.pdb_ids = pdb_ids
                    best = self._dm.pdb.get_entry_info(pdb_ids[0])
                    if best:
                        protein.structure_method = best.method
                        protein.resolution_angstrom = best.resolution
                        protein.structure_url = best.url
                        protein.chain_count = best.chain_count
            except Exception as e:
                report.errors.append(f"PDB: {e}")

        # Step 5: AlphaFold predicted structure
        if protein and uniprot_accession:
            _progress("Querying AlphaFold...")
            try:
                af = self._dm.alphafold.get_prediction(uniprot_accession)
                if af:
                    protein.alphafold_id = f"AF-{uniprot_accession}-F1"
                    protein.alphafold_url = af.model_url
                    # If no experimental structure, use AlphaFold
                    if not protein.structure_url:
                        protein.structure_url = af.model_url
                        protein.structure_method = "AlphaFold"
            except Exception as e:
                report.errors.append(f"AlphaFold: {e}")

        # Save protein updates
        if protein:
            self.store.upsert_entity(protein)

        # Step 6: KEGG pathways
        _progress("Querying KEGG pathways...")
        try:
            ncbi_id = gene.ncbi_gene_id or gene.external_ids.get("ncbi_gene", "")
            if ncbi_id:
                kegg_pathways = self._get_kegg_pathways_for_gene(ncbi_id)
                for pw_id, pw_name in kegg_pathways[:10]:
                    pw_entity = self._ensure_pathway(
                        pw_name, kegg_id=pw_id, tags=tags
                    )
                    if pw_entity and create_relationships:
                        rel = Relationship(
                            source_id=gene.entity_id,
                            target_id=pw_entity.entity_id,
                            rel_type=RelationshipType.PART_OF,
                            confidence=0.9,
                            evidence=f"KEGG {pw_id}",
                        )
                        if self.store.add_relationship(rel):
                            report.relationships_created += 1
                            report.by_type["pathway"] = report.by_type.get("pathway", 0) + 1
        except Exception as e:
            report.errors.append(f"KEGG: {e}")

        # Step 7: STRING interactions
        if create_relationships:
            _progress("Querying STRING interactions...")
            try:
                interactions = self._dm.string.get_interactions([symbol], score_threshold=700, limit=10)
                for inter in interactions:
                    partner = inter.gene_b if inter.gene_a == symbol else inter.gene_a
                    if partner and partner != symbol:
                        partner_entity = self.store.find_entity_by_name(partner, "gene")
                        if partner_entity:
                            rel = Relationship(
                                source_id=gene.entity_id,
                                target_id=partner_entity.entity_id,
                                rel_type=RelationshipType.BINDS_TO,
                                confidence=inter.combined_score / 1000.0,
                                evidence=f"STRING score={inter.combined_score}",
                            )
                            if self.store.add_relationship(rel):
                                report.relationships_created += 1
            except Exception as e:
                report.errors.append(f"STRING: {e}")

        # Step 8: ClinVar variants
        _progress("Querying ClinVar...")
        try:
            from cognisom.ncbi.clinvar import search_clinvar
            variants = search_clinvar(symbol, max_results=5)
            for var in variants:
                mut = Mutation(
                    name=f"{symbol}_{var.title[:30]}",
                    display_name=var.title[:80],
                    description=f"{var.clinical_significance} variant in {symbol}",
                    gene_symbol=symbol,
                    mutation_type=var.variant_type,
                    position=var.position,
                    clinical_significance=var.clinical_significance,
                    external_ids={"clinvar": var.accession},
                    source="clinvar",
                    source_url=f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{var.uid}/",
                    tags=tags or [],
                )
                if self.store.add_entity(mut):
                    report.entities_created += 1
                    report.by_type["mutation"] = report.by_type.get("mutation", 0) + 1
                    if create_relationships:
                        rel = Relationship(
                            source_id=mut.entity_id,
                            target_id=gene.entity_id,
                            rel_type=RelationshipType.MUTATED_IN,
                            confidence=0.95,
                            evidence=f"ClinVar {var.accession}",
                        )
                        if self.store.add_relationship(rel):
                            report.relationships_created += 1
        except Exception as e:
            report.errors.append(f"ClinVar: {e}")

        # Final save of gene
        self.store.upsert_entity(gene)

        report.duration_seconds = time.time() - t0
        _progress(f"Done ({report.entities_created} created, {report.relationships_created} rels)")
        return report

    def _enrich_gene_ncbi(
        self, symbol: str, existing: Optional[BioEntity], tags: Optional[List[str]]
    ) -> Optional[Gene]:
        """Fetch gene data from NCBI and create/update Gene entity."""
        try:
            result = self.ncbi.esearch(db="gene", term=f"{symbol}[Gene Name] AND Homo sapiens[Organism]", retmax=1)
            ids = result.get("idlist", [])
            if not ids:
                return None

            gene_id = ids[0]
            summaries = self.ncbi.esummary(db="gene", ids=[gene_id])
            doc = summaries.get(gene_id, {})
            if not isinstance(doc, dict):
                return None

            if existing and isinstance(existing, Gene):
                gene = existing
            else:
                gene = Gene()

            gene.name = doc.get("name", symbol)
            gene.display_name = gene.name
            gene.symbol = gene.name
            gene.full_name = doc.get("description", "")
            gene.chromosome = doc.get("chromosome", "")
            gene.gene_type = doc.get("genetype", "")
            gene.map_location = doc.get("maplocation", "")
            gene.organism = doc.get("organism", {}).get("scientificname", "Homo sapiens") if isinstance(doc.get("organism"), dict) else "Homo sapiens"
            gene.summary = doc.get("summary", "")[:2000]
            gene.ncbi_gene_id = gene_id
            gene.description = gene.summary[:500]
            gene.external_ids["ncbi_gene"] = gene_id
            gene.source = "ncbi_gene"
            gene.source_url = f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}"

            # Aliases
            aliases = doc.get("otheraliases", "")
            if aliases:
                gene.synonyms = [s.strip() for s in aliases.split(",") if s.strip()]

            # Other designations
            other_desig = doc.get("otherdesignations", "")
            if other_desig and not gene.full_name:
                gene.full_name = other_desig.split("|")[0].strip()

            if tags:
                gene.tags = list(set(gene.tags + tags))

            return gene

        except Exception as e:
            log.error("NCBI Gene enrichment for %s failed: %s", symbol, e)
            return None

    def _create_protein_from_uniprot(
        self, accession: str, gene_name: str, tags: Optional[List[str]]
    ) -> Optional[Protein]:
        """Create a Protein entity from UniProt data."""
        try:
            entry = self._dm.uniprot.get_entry(accession)
            if entry is None:
                return None

            existing = self.store.find_entity_by_external_id("uniprot", accession)
            if existing and isinstance(existing, Protein):
                protein = existing
            else:
                protein = Protein()

            protein.name = entry.protein_name or gene_name
            protein.display_name = protein.name
            protein.gene_source = gene_name
            protein.uniprot_id = accession
            protein.amino_acid_length = entry.length
            protein.sequence_preview = entry.sequence[:100]
            protein.function_summary = entry.function_summary[:1000]
            protein.go_terms = entry.go_terms[:30]
            protein.mass_daltons = entry.mass
            protein.subcellular_location = entry.subcellular_location
            protein.domains = entry.domains[:20]
            protein.tissue_specificity = entry.tissue_specificity[:500]
            protein.disease_associations = entry.disease_associations[:10]
            protein.pdb_ids = entry.pdb_ids[:20]
            protein.description = entry.function_summary[:500]
            protein.external_ids["uniprot"] = accession
            protein.source = "uniprot"
            protein.source_url = f"https://www.uniprot.org/uniprot/{accession}"
            protein.ontology_ids = [g.split(":")[0] + ":" + g.split(":")[1].split(" ")[0]
                                    for g in entry.go_terms[:10]
                                    if g.startswith("GO:")]

            if tags:
                protein.tags = list(set(protein.tags + tags))

            self.store.upsert_entity(protein)
            return protein

        except Exception as e:
            log.error("UniProt protein creation for %s failed: %s", accession, e)
            return None

    def _get_kegg_pathways_for_gene(self, ncbi_gene_id: str) -> List[tuple]:
        """Get KEGG pathways for a gene using NCBI elink or KEGG link."""
        try:
            url = f"https://rest.kegg.jp/link/pathway/hsa:{ncbi_gene_id}"
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Cognisom/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                text = resp.read().decode()

            pathways = []
            for line in text.strip().split("\n"):
                if "\t" in line:
                    _, pw_id = line.split("\t")
                    pw_id = pw_id.replace("path:", "")
                    pathways.append((pw_id, ""))

            # Get names
            result = []
            for pw_id, _ in pathways[:10]:
                try:
                    info = self._dm.kegg.get_pathway(pw_id)
                    if info:
                        result.append((pw_id, info.name))
                    else:
                        result.append((pw_id, pw_id))
                    time.sleep(0.1)
                except Exception:
                    result.append((pw_id, pw_id))

            return result
        except Exception as e:
            log.debug("KEGG pathway lookup for gene %s: %s", ncbi_gene_id, e)
            return []

    def _ensure_pathway(
        self, name: str, kegg_id: str = "", reactome_id: str = "",
        tags: Optional[List[str]] = None,
    ) -> Optional[Pathway]:
        """Get or create a Pathway entity."""
        if kegg_id:
            existing = self.store.find_entity_by_external_id("kegg", kegg_id)
            if existing:
                return existing if isinstance(existing, Pathway) else None

        existing = self.store.find_entity_by_name(name, "pathway")
        if existing:
            return existing if isinstance(existing, Pathway) else None

        pw = Pathway(
            name=name,
            display_name=name,
            kegg_id=kegg_id,
            reactome_id=reactome_id,
            source="kegg" if kegg_id else "reactome",
            tags=tags or [],
        )
        if kegg_id:
            pw.external_ids["kegg"] = kegg_id
        if reactome_id:
            pw.external_ids["reactome"] = reactome_id

        self.store.add_entity(pw)
        return pw

    # ── Drug Enrichment Pipeline ──────────────────────────────────────

    def enrich_drug(
        self, name: str,
        create_relationships: bool = True,
        tags: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> EnrichmentReport:
        """Full drug enrichment from PubChem.

        1. PubChem → SMILES, InChI, molecular properties, 2D/3D URLs
        2. Link to target proteins/genes → Relationships

        Returns:
            EnrichmentReport
        """
        report = EnrichmentReport()
        t0 = time.time()

        def _progress(msg: str):
            if progress_callback:
                progress_callback(msg)

        _progress(f"Searching PubChem for {name}...")
        existing = self.store.find_entity_by_name(name, "drug")

        try:
            compound = self._dm.pubchem.get_compound(name)
            if compound is None:
                report.errors.append(f"PubChem: no compound found for '{name}'")
                report.duration_seconds = time.time() - t0
                return report

            if existing and isinstance(existing, Drug):
                drug = existing
            else:
                drug = Drug()

            drug.name = name
            drug.display_name = name
            drug.smiles = compound.smiles
            drug.inchi = compound.inchi
            drug.inchi_key = compound.inchi_key
            drug.molecular_weight = compound.molecular_weight
            drug.molecular_formula = compound.molecular_formula
            drug.pubchem_cid = compound.cid
            drug.logp = compound.xlogp or 0.0
            drug.tpsa = compound.tpsa
            drug.hbd = compound.hbd
            drug.hba = compound.hba
            drug.rotatable_bonds = compound.rotatable_bonds
            drug.description = f"{compound.iupac_name or name}; MW={compound.molecular_weight:.1f}"
            drug.external_ids["pubchem_cid"] = str(compound.cid)
            drug.source = "pubchem"
            drug.source_url = compound.url

            # 2D/3D structure URLs for visualization
            drug.structure_2d_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound.cid}/PNG?image_size=300x300"
            drug.conformer_3d_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound.cid}/record/SDF/?record_type=3d&response_type=save"

            if tags:
                drug.tags = list(set(drug.tags + tags))

            self.store.upsert_entity(drug)

            if existing:
                report.entities_updated += 1
            else:
                report.entities_created += 1
            report.by_type["drug"] = 1

            # Link to target genes/proteins
            if create_relationships and drug.targets:
                for target_name in drug.targets:
                    target = self.store.find_entity_by_name(target_name, "gene")
                    if not target:
                        target = self.store.find_entity_by_name(target_name, "protein")
                    if target:
                        rel = Relationship(
                            source_id=drug.entity_id,
                            target_id=target.entity_id,
                            rel_type=RelationshipType.TARGETS,
                            confidence=0.9,
                            evidence="Drug target annotation",
                        )
                        if self.store.add_relationship(rel):
                            report.relationships_created += 1

        except Exception as e:
            report.errors.append(f"Drug enrichment for {name}: {e}")

        report.duration_seconds = time.time() - t0
        return report

    # ── Pathway Enrichment Pipeline ──────────────────────────────────

    def enrich_kegg_pathway(
        self, pathway_id: str,
        import_genes: bool = True,
        tags: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> EnrichmentReport:
        """Import a KEGG pathway with all its genes.

        1. KEGG get_pathway → pathway details, gene members
        2. Create Gene entities for each member
        3. Create REGULATES relationships

        Returns:
            EnrichmentReport
        """
        report = EnrichmentReport()
        t0 = time.time()

        def _progress(msg: str):
            if progress_callback:
                progress_callback(msg)

        _progress(f"Fetching KEGG pathway {pathway_id}...")

        try:
            pw_data = self._dm.kegg.get_pathway(pathway_id)
            if pw_data is None:
                report.errors.append(f"KEGG pathway {pathway_id} not found")
                report.duration_seconds = time.time() - t0
                return report

            # Create/update pathway entity
            pw_entity = self._ensure_pathway(
                pw_data.name, kegg_id=pathway_id, tags=tags
            )
            if pw_entity:
                pw_entity.description = pw_data.description
                pw_entity.diagram_url = pw_data.url
                self.store.upsert_entity(pw_entity)
                report.entities_created += 1
                report.by_type["pathway"] = 1

            # Import gene members
            if import_genes:
                _progress(f"Importing genes from {pathway_id}...")
                kegg_genes = self._dm.kegg.get_pathway_genes(pathway_id)

                for kg in kegg_genes:
                    if not kg.symbol:
                        continue

                    _progress(f"Enriching gene {kg.symbol}...")
                    gene_report = self.enrich_gene(
                        kg.symbol,
                        create_protein=True,
                        create_relationships=False,
                        tags=tags,
                    )
                    report.merge(gene_report)

                    # Link gene to pathway
                    if pw_entity:
                        gene_entity = self.store.find_entity_by_name(kg.symbol, "gene")
                        if gene_entity:
                            rel = Relationship(
                                source_id=gene_entity.entity_id,
                                target_id=pw_entity.entity_id,
                                rel_type=RelationshipType.PART_OF,
                                confidence=0.9,
                                evidence=f"KEGG {pathway_id}",
                            )
                            if self.store.add_relationship(rel):
                                report.relationships_created += 1

                    time.sleep(0.3)  # Rate limiting

        except Exception as e:
            report.errors.append(f"Pathway enrichment: {e}")

        report.duration_seconds = time.time() - t0
        return report

    # ── STRING Network Import ────────────────────────────────────────

    def enrich_string_network(
        self, seed_genes: List[str],
        score_threshold: int = 700,
        import_partners: bool = True,
        tags: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> EnrichmentReport:
        """Import interaction network from STRING.

        1. Query STRING for interactions among seed genes
        2. Create BINDS_TO relationships
        3. Optionally import partner genes not yet in DB

        Returns:
            EnrichmentReport
        """
        report = EnrichmentReport()
        t0 = time.time()

        def _progress(msg: str):
            if progress_callback:
                progress_callback(msg)

        _progress(f"Querying STRING for {len(seed_genes)} genes...")

        try:
            interactions = self._dm.string.get_interactions(
                seed_genes, score_threshold=score_threshold, limit=50
            )

            all_genes = set(seed_genes)
            for inter in interactions:
                all_genes.add(inter.gene_a)
                all_genes.add(inter.gene_b)

            # Ensure seed genes exist
            for symbol in seed_genes:
                existing = self.store.find_entity_by_name(symbol, "gene")
                if not existing:
                    _progress(f"Enriching seed gene {symbol}...")
                    gene_report = self.enrich_gene(
                        symbol, create_protein=True, tags=tags
                    )
                    report.merge(gene_report)
                    time.sleep(0.3)

            # Import partner genes
            if import_partners:
                partner_genes = all_genes - set(seed_genes)
                for symbol in list(partner_genes)[:20]:
                    existing = self.store.find_entity_by_name(symbol, "gene")
                    if not existing:
                        _progress(f"Importing partner {symbol}...")
                        gene_report = self.enrich_gene(
                            symbol, create_protein=False,
                            create_relationships=False, tags=tags,
                        )
                        report.merge(gene_report)
                        time.sleep(0.3)

            # Create relationships
            for inter in interactions:
                gene_a = self.store.find_entity_by_name(inter.gene_a, "gene")
                gene_b = self.store.find_entity_by_name(inter.gene_b, "gene")
                if gene_a and gene_b:
                    rel = Relationship(
                        source_id=gene_a.entity_id,
                        target_id=gene_b.entity_id,
                        rel_type=RelationshipType.BINDS_TO,
                        confidence=inter.combined_score / 1000.0,
                        evidence=f"STRING score={inter.combined_score}",
                    )
                    if self.store.add_relationship(rel):
                        report.relationships_created += 1

        except Exception as e:
            report.errors.append(f"STRING network: {e}")

        report.duration_seconds = time.time() - t0
        return report

    # ── Batch Enrichment ─────────────────────────────────────────────

    def enrich_gene_list(
        self, symbols: List[str],
        create_proteins: bool = True,
        tags: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> EnrichmentReport:
        """Enrich a list of genes sequentially.

        Args:
            symbols: Gene symbols to enrich
            create_proteins: Also create Protein entities
            tags: Tags to apply
            progress_callback: Called with (current_index, total, symbol)

        Returns:
            Combined EnrichmentReport
        """
        report = EnrichmentReport()
        t0 = time.time()

        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(f"[{i+1}/{len(symbols)}] {symbol}")

            try:
                gene_report = self.enrich_gene(
                    symbol,
                    create_protein=create_proteins,
                    tags=tags,
                )
                report.merge(gene_report)
            except Exception as e:
                report.errors.append(f"{symbol}: {e}")

            # Rate limit between genes
            time.sleep(0.5)

        report.duration_seconds = time.time() - t0
        return report

    def enrich_drug_list(
        self, names: List[str],
        tags: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> EnrichmentReport:
        """Enrich a list of drugs from PubChem."""
        report = EnrichmentReport()
        t0 = time.time()

        for i, name in enumerate(names):
            if progress_callback:
                progress_callback(f"[{i+1}/{len(names)}] {name}")

            try:
                drug_report = self.enrich_drug(name, tags=tags)
                report.merge(drug_report)
            except Exception as e:
                report.errors.append(f"{name}: {e}")

            time.sleep(0.3)

        report.duration_seconds = time.time() - t0
        return report

    # ── Enrich Existing Entities ─────────────────────────────────────

    def enrich_all_genes(
        self, progress_callback: Optional[Callable] = None
    ) -> EnrichmentReport:
        """Re-enrich all existing Gene entities with latest API data."""
        report = EnrichmentReport()
        t0 = time.time()

        genes, total = self.store.search(entity_type="gene", limit=500)
        for i, entity in enumerate(genes):
            if not isinstance(entity, Gene):
                continue
            symbol = entity.symbol or entity.name
            if not symbol:
                continue

            if progress_callback:
                progress_callback(f"[{i+1}/{total}] Re-enriching {symbol}")

            try:
                gene_report = self.enrich_gene(symbol, create_protein=True)
                report.merge(gene_report)
            except Exception as e:
                report.errors.append(f"Re-enrich {symbol}: {e}")

            time.sleep(0.5)

        report.duration_seconds = time.time() - t0
        return report

    def enrich_all_drugs(
        self, progress_callback: Optional[Callable] = None
    ) -> EnrichmentReport:
        """Re-enrich all existing Drug entities with latest PubChem data."""
        report = EnrichmentReport()
        t0 = time.time()

        drugs, total = self.store.search(entity_type="drug", limit=500)
        for i, entity in enumerate(drugs):
            if not isinstance(entity, Drug):
                continue
            name = entity.name
            if not name:
                continue

            if progress_callback:
                progress_callback(f"[{i+1}/{total}] Re-enriching {name}")

            try:
                drug_report = self.enrich_drug(name)
                report.merge(drug_report)
            except Exception as e:
                report.errors.append(f"Re-enrich {name}: {e}")

            time.sleep(0.3)

        report.duration_seconds = time.time() - t0
        return report
