"""
Bulk Import Pipeline
=====================

High-level orchestration for bulk entity population:
- Import gene lists with full enrichment
- Import KEGG pathways with all member genes
- Import STRING interaction networks
- Import drug lists from PubChem
- Run full enrichment on existing entities
- Build cross-reference relationships

Usage:
    store = EntityStore()
    importer = BulkImporter(store)
    report = importer.import_preset("Prostate Cancer (80 genes + 20 drugs)")
    report = importer.import_gene_list(["TP53", "BRCA1", "PTEN"])
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

from .enrichment import EntityEnricher, EnrichmentReport
from .models import Relationship, RelationshipType
from .store import EntityStore

log = logging.getLogger(__name__)


class BulkImporter:
    """High-level import orchestrator with progress tracking."""

    def __init__(self, store: EntityStore):
        self.store = store
        self.enricher = EntityEnricher(store)

    def import_gene_list(
        self,
        symbols: List[str],
        enrich: bool = True,
        tags: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> EnrichmentReport:
        """Import a list of genes with optional enrichment.

        Args:
            symbols: Gene symbols to import
            enrich: If True, run full enrichment (NCBI, UniProt, PDB, etc.)
            tags: Tags to apply to all created entities
            progress_callback: Called with status string

        Returns:
            EnrichmentReport with counts
        """
        report = EnrichmentReport()
        t0 = time.time()

        # Deduplicate
        unique = list(dict.fromkeys(symbols))
        total = len(unique)

        for i, symbol in enumerate(unique):
            pct = int((i / total) * 100) if total else 0

            if progress_callback:
                progress_callback(f"[{i+1}/{total}] ({pct}%) Importing {symbol}...")

            try:
                if enrich:
                    gene_report = self.enricher.enrich_gene(
                        symbol, create_protein=True, tags=tags
                    )
                    report.merge(gene_report)
                else:
                    # Quick import — NCBI Gene only, no enrichment cascade
                    gene_report = self.enricher.enrich_gene(
                        symbol, create_protein=False,
                        create_relationships=False, tags=tags,
                    )
                    report.merge(gene_report)
            except Exception as e:
                report.errors.append(f"{symbol}: {e}")

            time.sleep(0.3)

        report.duration_seconds = time.time() - t0
        log.info(
            "Gene import done: %d created, %d updated, %d errors in %.1fs",
            report.entities_created, report.entities_updated,
            len(report.errors), report.duration_seconds,
        )
        return report

    def import_drug_list(
        self,
        names: List[str],
        tags: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> EnrichmentReport:
        """Import drugs from PubChem.

        Args:
            names: Drug names (e.g., ["Enzalutamide", "Olaparib"])
            tags: Tags to apply
            progress_callback: Status callback

        Returns:
            EnrichmentReport
        """
        return self.enricher.enrich_drug_list(
            names, tags=tags, progress_callback=progress_callback
        )

    def import_kegg_pathway(
        self,
        pathway_id: str,
        import_genes: bool = True,
        tags: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> EnrichmentReport:
        """Import a KEGG pathway with all member genes.

        Args:
            pathway_id: KEGG pathway ID (e.g., "hsa05215")
            import_genes: Whether to import and enrich all genes
            tags: Tags to apply
            progress_callback: Status callback

        Returns:
            EnrichmentReport
        """
        return self.enricher.enrich_kegg_pathway(
            pathway_id,
            import_genes=import_genes,
            tags=tags,
            progress_callback=progress_callback,
        )

    def import_string_network(
        self,
        seed_genes: List[str],
        score_threshold: int = 700,
        import_partners: bool = True,
        tags: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> EnrichmentReport:
        """Import STRING protein interaction network.

        Args:
            seed_genes: Seed gene symbols
            score_threshold: Minimum STRING score (0-1000)
            import_partners: Import interaction partner genes
            tags: Tags to apply
            progress_callback: Status callback

        Returns:
            EnrichmentReport
        """
        return self.enricher.enrich_string_network(
            seed_genes,
            score_threshold=score_threshold,
            import_partners=import_partners,
            tags=tags,
            progress_callback=progress_callback,
        )

    def import_preset(
        self,
        preset_name: str,
        progress_callback: Optional[Callable] = None,
    ) -> EnrichmentReport:
        """Import a predefined gene/drug set.

        Args:
            preset_name: Name from gene_sets.IMPORT_SETS
            progress_callback: Status callback

        Returns:
            EnrichmentReport
        """
        from .gene_sets import IMPORT_SETS

        preset = IMPORT_SETS.get(preset_name)
        if preset is None:
            report = EnrichmentReport()
            report.errors.append(f"Unknown preset: {preset_name}")
            return report

        report = EnrichmentReport()
        t0 = time.time()

        genes = preset.get("genes", [])
        drugs = preset.get("drugs", [])
        pathways = preset.get("pathways", [])

        # Import genes
        if genes:
            if progress_callback:
                progress_callback(f"Importing {len(genes)} genes...")
            gene_report = self.import_gene_list(
                genes,
                enrich=True,
                tags=["preset", preset_name.split("(")[0].strip().lower().replace(" ", "_")],
                progress_callback=progress_callback,
            )
            report.merge(gene_report)

        # Import drugs
        if drugs:
            if progress_callback:
                progress_callback(f"Importing {len(drugs)} drugs...")
            drug_report = self.import_drug_list(
                drugs,
                tags=["preset"],
                progress_callback=progress_callback,
            )
            report.merge(drug_report)

        # Import pathways
        if pathways:
            for pw_id in pathways:
                if progress_callback:
                    progress_callback(f"Importing pathway {pw_id}...")
                pw_report = self.import_kegg_pathway(
                    pw_id,
                    import_genes=False,  # already imported above
                    tags=["preset"],
                    progress_callback=progress_callback,
                )
                report.merge(pw_report)
                time.sleep(0.3)

        report.duration_seconds = time.time() - t0
        return report

    def run_full_enrichment(
        self,
        entity_types: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> EnrichmentReport:
        """Re-enrich all existing entities.

        Args:
            entity_types: Types to enrich (default: ["gene", "drug"])
            progress_callback: Status callback

        Returns:
            EnrichmentReport
        """
        types = entity_types or ["gene", "drug"]
        report = EnrichmentReport()

        if "gene" in types:
            if progress_callback:
                progress_callback("Re-enriching all genes...")
            gene_report = self.enricher.enrich_all_genes(progress_callback)
            report.merge(gene_report)

        if "drug" in types:
            if progress_callback:
                progress_callback("Re-enriching all drugs...")
            drug_report = self.enricher.enrich_all_drugs(progress_callback)
            report.merge(drug_report)

        return report

    def build_cross_references(
        self, progress_callback: Optional[Callable] = None,
    ) -> int:
        """Build relationships from existing entity cross-references.

        Scans all entities for external_ids and creates relationships
        between entities that share references (e.g., Gene→Protein via UniProt).

        Returns:
            Number of relationships created
        """
        rels_created = 0
        t0 = time.time()

        if progress_callback:
            progress_callback("Scanning entities for cross-references...")

        # Get all entities
        genes, _ = self.store.search(entity_type="gene", limit=1000)
        proteins, _ = self.store.search(entity_type="protein", limit=1000)
        drugs, _ = self.store.search(entity_type="drug", limit=1000)
        pathways, _ = self.store.search(entity_type="pathway", limit=1000)

        # Gene → Protein (ENCODES) via shared UniProt
        gene_uniprot = {}
        for g in genes:
            uid = g.external_ids.get("uniprot")
            if uid:
                gene_uniprot[uid] = g

        for p in proteins:
            uid = p.external_ids.get("uniprot")
            if uid and uid in gene_uniprot:
                gene = gene_uniprot[uid]
                rel = Relationship(
                    source_id=gene.entity_id,
                    target_id=p.entity_id,
                    rel_type=RelationshipType.ENCODES,
                    confidence=1.0,
                    evidence="UniProt cross-reference",
                )
                if self.store.add_relationship(rel):
                    rels_created += 1

        # Drug → Gene/Protein (TARGETS) via target names
        for d in drugs:
            if hasattr(d, "targets"):
                for target_name in d.targets:
                    target = self.store.find_entity_by_name(target_name, "gene")
                    if not target:
                        target = self.store.find_entity_by_name(target_name, "protein")
                    if target:
                        rel = Relationship(
                            source_id=d.entity_id,
                            target_id=target.entity_id,
                            rel_type=RelationshipType.TARGETS,
                            confidence=0.8,
                            evidence="Drug target cross-reference",
                        )
                        if self.store.add_relationship(rel):
                            rels_created += 1

        # Gene → Pathway (PART_OF) via KEGG IDs
        pw_by_kegg = {}
        for pw in pathways:
            kid = pw.external_ids.get("kegg")
            if kid:
                pw_by_kegg[kid] = pw

        elapsed = time.time() - t0
        if progress_callback:
            progress_callback(f"Built {rels_created} cross-references in {elapsed:.1f}s")

        log.info("Cross-references: %d relationships created in %.1fs", rels_created, elapsed)
        return rels_created

    def get_data_quality_report(self) -> Dict:
        """Get data completeness metrics.

        Returns dict with:
            - total_entities: int
            - total_relationships: int
            - by_type: Dict[str, int]
            - quality: Dict[str, float] (% of entities with key fields)
        """
        stats = self.store.stats()

        # Quality checks
        quality = {}

        # Gene quality
        genes, total_genes = self.store.search(entity_type="gene", limit=500)
        if total_genes > 0:
            has_ncbi = sum(1 for g in genes if g.external_ids.get("ncbi_gene"))
            has_chrom = sum(1 for g in genes if hasattr(g, "chromosome") and g.chromosome)
            has_summary = sum(1 for g in genes if hasattr(g, "summary") and g.summary)
            quality["genes_with_ncbi_id"] = has_ncbi / total_genes
            quality["genes_with_chromosome"] = has_chrom / total_genes
            quality["genes_with_summary"] = has_summary / total_genes

        # Protein quality
        proteins, total_proteins = self.store.search(entity_type="protein", limit=500)
        if total_proteins > 0:
            has_uniprot = sum(1 for p in proteins if p.external_ids.get("uniprot"))
            has_pdb = sum(1 for p in proteins if hasattr(p, "pdb_ids") and p.pdb_ids)
            has_structure = sum(1 for p in proteins if hasattr(p, "structure_url") and p.structure_url)
            quality["proteins_with_uniprot"] = has_uniprot / total_proteins
            quality["proteins_with_pdb"] = has_pdb / total_proteins
            quality["proteins_with_structure"] = has_structure / total_proteins

        # Drug quality
        drugs, total_drugs = self.store.search(entity_type="drug", limit=500)
        if total_drugs > 0:
            has_smiles = sum(1 for d in drugs if hasattr(d, "smiles") and d.smiles)
            has_cid = sum(1 for d in drugs if hasattr(d, "pubchem_cid") and d.pubchem_cid)
            quality["drugs_with_smiles"] = has_smiles / total_drugs
            quality["drugs_with_pubchem_cid"] = has_cid / total_drugs

        return {
            "total_entities": stats["total_entities"],
            "total_relationships": stats["total_relationships"],
            "by_type": stats["by_type"],
            "quality": quality,
        }
