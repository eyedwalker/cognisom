"""
Seed Data — Prostate Cancer Entity Catalog
==========================================

Pre-built catalog of essential biological entities for prostate cancer
research. Includes genes, drugs, cell types, pathways, metabolites,
mutations, and their relationships.

API-sourced entities (genes, proteins) are fetched live from NCBI/UniProt.
Manually curated entities (drugs, cell types, pathways) use expert knowledge.

Usage:
    from cognisom.library.seed_data import seed_prostate_cancer_catalog
    from cognisom.library.store import EntityStore

    store = EntityStore()
    seed_prostate_cancer_catalog(store, fetch_remote=True)
"""

from __future__ import annotations

import logging
import time
from typing import List

from .loaders import EntityLoader
from .models import Relationship, RelationshipType
from .store import EntityStore

log = logging.getLogger(__name__)


# ── Key prostate cancer genes ────────────────────────────────────────

PROSTATE_CANCER_GENES = [
    ("AR", "oncogene"),
    ("TP53", "tumor_suppressor"),
    ("PTEN", "tumor_suppressor"),
    ("BRCA2", "tumor_suppressor"),
    ("MYC", "oncogene"),
    ("RB1", "tumor_suppressor"),
    ("ERG", "oncogene"),
    ("TMPRSS2", "signaling"),
    ("FOXA1", "signaling"),
    ("SPOP", "tumor_suppressor"),
    ("CDK12", "tumor_suppressor"),
    ("PIK3CA", "oncogene"),
    ("AKT1", "oncogene"),
    ("HOXB13", "oncogene"),
    ("KMT2D", "tumor_suppressor"),
    ("ATM", "tumor_suppressor"),
    ("CHD1", "tumor_suppressor"),
    ("NKX3-1", "tumor_suppressor"),
    ("EZH2", "oncogene"),
    ("MDM2", "oncogene"),
]


def seed_prostate_cancer_catalog(
    store: EntityStore,
    fetch_remote: bool = False,
    rate_limit: float = 0.4,
) -> dict:
    """Seed the entity library with a prostate cancer catalog.

    Args:
        store: EntityStore to populate
        fetch_remote: If True, fetch from NCBI/UniProt APIs (slower but richer).
                     If False, use curated manual data only.
        rate_limit: Seconds between API calls (to avoid rate limiting)

    Returns:
        dict with counts of entities created
    """
    loader = EntityLoader(store)
    counts = {
        "genes": 0, "proteins": 0, "drugs": 0, "cell_types": 0,
        "pathways": 0, "metabolites": 0, "mutations": 0,
        "tissues": 0, "receptors": 0, "organs": 0, "relationships": 0,
    }

    # ── Genes & Proteins (from API or manual) ────────────────────────
    if fetch_remote:
        log.info("Fetching genes and proteins from NCBI/UniProt...")
        for gene_name, gene_type in PROSTATE_CANCER_GENES:
            try:
                gene, protein = loader.load_gene_protein_pair(gene_name, gene_type)
                if gene:
                    counts["genes"] += 1
                if protein:
                    counts["proteins"] += 1
                    counts["relationships"] += 1
                time.sleep(rate_limit)
            except Exception as e:
                log.warning("Failed to load %s: %s", gene_name, e)
    else:
        log.info("Seeding genes from curated data (no API calls)...")
        counts["genes"] += _seed_manual_genes(loader)

    # ── Drugs ────────────────────────────────────────────────────────
    log.info("Seeding drugs...")
    counts["drugs"] += _seed_drugs(loader)

    # ── Cell types ───────────────────────────────────────────────────
    log.info("Seeding cell types...")
    counts["cell_types"] += _seed_cell_types(loader)

    # ── Pathways ─────────────────────────────────────────────────────
    log.info("Seeding pathways...")
    counts["pathways"] += _seed_pathways(loader)

    # ── Metabolites ──────────────────────────────────────────────────
    log.info("Seeding metabolites...")
    counts["metabolites"] += _seed_metabolites(loader)

    # ── Mutations ────────────────────────────────────────────────────
    log.info("Seeding mutations...")
    counts["mutations"] += _seed_mutations(loader)

    # ── Tissue types ─────────────────────────────────────────────────
    log.info("Seeding tissue types...")
    counts["tissues"] += _seed_tissues(loader)

    # ── Receptors ────────────────────────────────────────────────────
    log.info("Seeding receptors...")
    counts["receptors"] += _seed_receptors(loader)

    # ── Organs ───────────────────────────────────────────────────────
    log.info("Seeding organs...")
    counts["organs"] += _seed_organs(loader)

    # ── Cross-entity relationships ───────────────────────────────────
    log.info("Creating cross-entity relationships...")
    counts["relationships"] += _seed_relationships(store)

    total = sum(counts.values())
    log.info("Seed complete: %d total items created", total)
    return counts


# ── Manual genes (no API) ────────────────────────────────────────────

def _seed_manual_genes(loader: EntityLoader) -> int:
    """Seed genes with curated descriptions (no API calls)."""
    from .models import Gene
    gene_data = [
        ("AR", "oncogene", "Xq12", "Androgen Receptor — master regulator of prostate development and cancer progression. Activated by testosterone/DHT binding."),
        ("TP53", "tumor_suppressor", "17p13.1", "Tumor protein p53 — guardian of the genome. Mutated in ~50% of all cancers. Regulates apoptosis, senescence, DNA repair."),
        ("PTEN", "tumor_suppressor", "10q23.31", "Phosphatase and tensin homolog — negative regulator of PI3K/AKT pathway. Lost in ~40% of prostate cancers."),
        ("BRCA2", "tumor_suppressor", "13q13.1", "BRCA2 DNA repair associated — homologous recombination repair. Mutations confer sensitivity to PARP inhibitors."),
        ("MYC", "oncogene", "8q24.21", "MYC proto-oncogene — transcription factor controlling cell growth and proliferation. Amplified in aggressive prostate cancer."),
        ("RB1", "tumor_suppressor", "13q14.2", "Retinoblastoma protein — cell cycle gatekeeper. Loss drives neuroendocrine differentiation in prostate cancer."),
        ("ERG", "oncogene", "21q22.2", "ETS-related gene — transcription factor. TMPRSS2-ERG fusion found in ~50% of prostate cancers."),
        ("TMPRSS2", "signaling", "21q22.3", "Transmembrane serine protease 2 — androgen-regulated. TMPRSS2-ERG fusion is the most common gene fusion in prostate cancer."),
        ("FOXA1", "signaling", "14q21.1", "Forkhead box A1 — pioneer transcription factor. Cooperates with AR to regulate prostate-specific gene expression."),
        ("SPOP", "tumor_suppressor", "17q21.33", "Speckle-type POZ protein — E3 ubiquitin ligase. Mutated in ~15% of prostate cancers. Regulates AR stability."),
        ("CDK12", "tumor_suppressor", "17q12", "Cyclin-dependent kinase 12 — DNA damage response. Loss causes tandem duplications and neoantigen generation."),
        ("PIK3CA", "oncogene", "3q26.32", "PI3K catalytic subunit alpha — key signaling kinase. Activating mutations drive the PI3K/AKT/mTOR pathway."),
        ("AKT1", "oncogene", "14q32.33", "AKT serine/threonine kinase 1 — central node of PI3K signaling. Promotes cell survival and proliferation."),
        ("HOXB13", "oncogene", "17q21.32", "Homeobox B13 — prostate-specific transcription factor. G84E germline variant increases prostate cancer risk."),
        ("KMT2D", "tumor_suppressor", "12q13.12", "Lysine methyltransferase 2D — histone H3K4 methyltransferase. Mutations disrupt enhancer function."),
        ("ATM", "tumor_suppressor", "11q22.3", "ATM serine/threonine kinase — DNA damage checkpoint. Loss impairs double-strand break repair."),
        ("CHD1", "tumor_suppressor", "5q21.1", "Chromodomain helicase DNA binding 1 — chromatin remodeler. Deletion correlates with SPOP mutations."),
        ("NKX3-1", "tumor_suppressor", "8p21.2", "NK3 homeobox 1 — prostate-specific tumor suppressor. Haploinsufficiency in early prostate cancer."),
        ("EZH2", "oncogene", "7q36.1", "Enhancer of zeste homolog 2 — polycomb repressive complex. Overexpressed in metastatic prostate cancer."),
        ("MDM2", "oncogene", "12q15", "MDM2 proto-oncogene — E3 ubiquitin ligase for p53. Amplification inactivates p53 pathway."),
    ]
    count = 0
    for symbol, gtype, chrom, desc in gene_data:
        gene = Gene(
            name=symbol,
            display_name=symbol,
            description=desc,
            symbol=symbol,
            full_name=desc.split("—")[0].strip() if "—" in desc else symbol,
            chromosome=chrom.split("p")[0].split("q")[0] if "p" in chrom or "q" in chrom else chrom,
            gene_type=gtype,
            map_location=chrom,
            source="curated",
            tags=["prostate_cancer", gtype],
        )
        if loader.store.add_entity(gene):
            count += 1
    return count


# ── Drugs ────────────────────────────────────────────────────────────

def _seed_drugs(loader: EntityLoader) -> int:
    drugs = [
        ("Enzalutamide", "anti-androgen", "Competitive AR inhibitor; blocks nuclear translocation and DNA binding of the androgen receptor",
         ["AR"], "approved", "CC1(C2CCC3(C(=O)NC(=C3C2CC(C1)(F)F)C#N)C)C4=CC(=C(C=C4)F)C(=O)NC5=CC=C(C=N5)C(F)(F)F"),
        ("Abiraterone", "anti-androgen", "CYP17A1 inhibitor; blocks androgen biosynthesis in testes, adrenals, and tumor",
         ["CYP17A1"], "approved", ""),
        ("Docetaxel", "chemotherapy", "Taxane microtubule stabilizer; inhibits mitotic cell division",
         ["TUBB"], "approved", ""),
        ("Cabazitaxel", "chemotherapy", "Semi-synthetic taxane; overcomes docetaxel resistance",
         ["TUBB"], "approved", ""),
        ("Olaparib", "PARP_inhibitor", "PARP1/2 inhibitor; synthetic lethality with BRCA1/2 and HRR deficiency",
         ["PARP1", "PARP2", "BRCA2"], "approved", ""),
        ("Rucaparib", "PARP_inhibitor", "PARP inhibitor for BRCA-mutated mCRPC",
         ["PARP1", "BRCA2"], "approved", ""),
        ("Pembrolizumab", "immunotherapy", "Anti-PD-1 checkpoint inhibitor; activates T cell anti-tumor immunity",
         ["PDCD1"], "approved", ""),
        ("Ipatasertib", "PI3K_pathway", "AKT inhibitor; blocks PI3K/AKT/mTOR survival signaling",
         ["AKT1", "AKT2"], "clinical_trial", ""),
        ("Darolutamide", "anti-androgen", "Structurally distinct AR inhibitor with low CNS penetration",
         ["AR"], "approved", ""),
        ("Lutetium-177 PSMA", "radiotherapy", "Radioligand targeting PSMA; delivers beta radiation to PSMA-expressing cells",
         ["FOLH1"], "approved", ""),
        ("Talazoparib", "PARP_inhibitor", "Potent PARP trapping agent for HRR-deficient prostate cancer",
         ["PARP1", "BRCA2"], "approved", ""),
        ("Apalutamide", "anti-androgen", "AR inhibitor for non-metastatic castration-resistant prostate cancer",
         ["AR"], "approved", ""),
    ]
    count = 0
    for name, dclass, mech, targets, status, smiles in drugs:
        loader.add_drug(name, dclass, mech, targets, status, smiles)
        count += 1
    return count


# ── Cell types ───────────────────────────────────────────────────────

def _seed_cell_types(loader: EntityLoader) -> int:
    cell_types = [
        ("Luminal epithelial cell", "Main secretory cell of the prostate. Expresses AR, PSA, CK8, CK18.",
         "prostate", ["AR+", "PSA+", "CK8+", "CK18+"], "epithelial", "CL:0002327"),
        ("Basal epithelial cell", "Progenitor cell layer of the prostate. AR-low, p63+, CK5+.",
         "prostate", ["p63+", "CK5+", "CK14+", "AR-low"], "epithelial", "CL:0000646"),
        ("Neuroendocrine cell", "Rare endocrine cell in prostate. Synaptophysin+, chromogranin A+.",
         "prostate", ["SYP+", "CHGA+", "NSE+"], "neuroendocrine", "CL:0000165"),
        ("Stromal fibroblast", "Mesenchymal cell in prostate stroma. Vimentin+, alpha-SMA+.",
         "prostate", ["VIM+", "ACTA2+", "FAP+"], "mesenchymal", "CL:0000057"),
        ("Cancer-associated fibroblast", "Activated fibroblast in tumor microenvironment. Promotes tumor growth.",
         "prostate", ["FAP+", "PDPN+", "alpha-SMA+"], "mesenchymal", ""),
        ("Endothelial cell", "Lines blood vessels in prostate tissue. CD31+, CD34+.",
         "prostate", ["CD31+", "CD34+", "VEGFR2+"], "endothelial", "CL:0000115"),
        ("CD8+ T cell", "Cytotoxic T lymphocyte. Key effector of anti-tumor immunity.",
         "blood", ["CD8+", "CD3+", "TCR+"], "lymphoid", "CL:0000625"),
        ("CD4+ T cell", "Helper T cell. Coordinates immune responses.",
         "blood", ["CD4+", "CD3+", "TCR+"], "lymphoid", "CL:0000624"),
        ("NK cell", "Natural killer cell. Innate cytotoxic lymphocyte.",
         "blood", ["CD56+", "CD16+", "NKp46+"], "lymphoid", "CL:0000623"),
        ("Macrophage", "Phagocytic myeloid cell in tumor microenvironment. M1 (anti-tumor) vs M2 (pro-tumor).",
         "blood", ["CD68+", "CD163+", "CD14+"], "myeloid", "CL:0000235"),
        ("Prostate cancer stem cell", "Self-renewing cancer cell with high tumorigenicity. CD44+, ALDH+.",
         "prostate", ["CD44+", "ALDH+", "CD133+", "integrin-a2b1+"], "epithelial", ""),
        ("Regulatory T cell", "Immunosuppressive CD4+ T cell subset. Suppresses anti-tumor immunity in TME.",
         "blood", ["CD4+", "CD25+", "FOXP3+"], "lymphoid", "CL:0000815"),
    ]
    count = 0
    for name, desc, origin, markers, lineage, cl_id in cell_types:
        loader.add_cell_type(name, desc, origin, markers, lineage, cl_id)
        count += 1
    return count


# ── Pathways ─────────────────────────────────────────────────────────

def _seed_pathways(loader: EntityLoader) -> int:
    pathways = [
        ("AR Signaling", "Androgen receptor signaling pathway — master regulator of prostate biology",
         "signaling", ["AR", "FOXA1", "HOXB13", "NKX3-1", "KLK3"],
         "R-HSA-9009391"),
        ("PI3K/AKT/mTOR", "Phosphoinositide 3-kinase survival pathway — PTEN loss activates this cascade",
         "signaling", ["PIK3CA", "AKT1", "PTEN", "MTOR", "TSC1", "TSC2"],
         "R-HSA-1257604"),
        ("DNA Damage Response", "Homologous recombination repair and checkpoint signaling",
         "signaling", ["BRCA2", "ATM", "CDK12", "CHEK2", "RAD51", "TP53"],
         "R-HSA-73894"),
        ("p53 Pathway", "TP53-mediated tumor suppression — apoptosis, senescence, cell cycle arrest",
         "signaling", ["TP53", "MDM2", "CDKN1A", "BAX", "BBC3"],
         "R-HSA-6806003"),
        ("Wnt/beta-Catenin", "Canonical Wnt signaling — drives stemness and treatment resistance",
         "signaling", ["CTNNB1", "APC", "RNF43", "RSPO2"],
         "R-HSA-195721"),
        ("RB/E2F Cell Cycle", "Retinoblastoma-mediated cell cycle control",
         "signaling", ["RB1", "CDK4", "CDK6", "CCND1", "E2F1"],
         "R-HSA-69236"),
        ("TMPRSS2-ERG Fusion", "Oncogenic gene fusion driving ~50% of prostate cancers",
         "signaling", ["TMPRSS2", "ERG", "ETV1", "ETV4"],
         ""),
        ("Glycolysis", "Glucose catabolism — Warburg effect in cancer cells",
         "metabolic", ["HK2", "PKM", "LDHA", "PFKL"],
         "R-HSA-70171"),
        ("TCA Cycle", "Tricarboxylic acid cycle — central metabolic hub",
         "metabolic", ["CS", "IDH1", "IDH2", "SDHA", "FH"],
         "R-HSA-71403"),
        ("Oxidative Phosphorylation", "Mitochondrial electron transport chain and ATP synthesis",
         "metabolic", ["MT-ND1", "MT-CO1", "MT-ATP6", "NDUFS1"],
         "R-HSA-163200"),
        ("PD-1/PD-L1 Immune Checkpoint", "Programmed death signaling — suppresses anti-tumor immunity",
         "signaling", ["PDCD1", "CD274", "PDCD1LG2"],
         "R-HSA-389948"),
        ("Epigenetic Regulation", "Histone modification and chromatin remodeling in prostate cancer",
         "regulatory", ["EZH2", "KMT2D", "CHD1", "KDM5A", "BRD4"],
         ""),
    ]
    count = 0
    for name, desc, ptype, genes, reactome in pathways:
        loader.add_pathway(name, desc, ptype, genes, reactome_id=reactome)
        count += 1
    return count


# ── Metabolites ──────────────────────────────────────────────────────

def _seed_metabolites(loader: EntityLoader) -> int:
    metabolites = [
        ("Glucose", "Primary energy source for cellular metabolism", "C6H12O6", 180.16, "CHEBI:17234"),
        ("Oxygen", "Terminal electron acceptor in oxidative phosphorylation", "O2", 32.0, "CHEBI:15379"),
        ("ATP", "Universal energy currency of the cell", "C10H16N5O13P3", 507.18, "CHEBI:15422"),
        ("Lactate", "End product of anaerobic glycolysis (Warburg effect marker)", "C3H6O3", 90.08, "CHEBI:24996"),
        ("Pyruvate", "Central metabolite linking glycolysis to TCA cycle", "C3H4O3", 88.06, "CHEBI:15361"),
        ("Citrate", "TCA cycle intermediate; elevated in prostate secretions", "C6H8O7", 192.12, "CHEBI:16947"),
        ("Testosterone", "Primary androgen; substrate for 5-alpha reductase", "C19H28O2", 288.42, "CHEBI:17347"),
        ("DHT", "Dihydrotestosterone — most potent androgen; binds AR", "C19H30O2", 290.44, "CHEBI:16330"),
        ("PSA", "Prostate-specific antigen (KLK3 product) — serum biomarker", "", 0.0, ""),
        ("NADH", "Electron carrier from TCA cycle to ETC", "C21H29N7O14P2", 663.43, "CHEBI:16908"),
        ("Glutamine", "Major anaplerotic fuel for cancer cells", "C5H10N2O3", 146.14, "CHEBI:28300"),
        ("Acetyl-CoA", "Central metabolic intermediate connecting multiple pathways", "C23H38N7O17P3S", 809.57, "CHEBI:15351"),
    ]
    count = 0
    for name, desc, formula, mw, chebi in metabolites:
        loader.add_metabolite(name, desc, formula, mw, chebi, tags=["metabolism", "prostate_cancer"])
        count += 1
    return count


# ── Mutations ────────────────────────────────────────────────────────

def _seed_mutations(loader: EntityLoader) -> int:
    mutations = [
        ("TP53", "R175H", "missense", "loss_of_function", "pathogenic", 0.03),
        ("TP53", "R248W", "missense", "loss_of_function", "pathogenic", 0.02),
        ("TP53", "R273H", "missense", "loss_of_function", "pathogenic", 0.02),
        ("PTEN", "del exon 5-6", "deletion", "loss_of_function", "pathogenic", 0.15),
        ("PTEN", "R130*", "nonsense", "loss_of_function", "pathogenic", 0.05),
        ("BRCA2", "del exon 13", "deletion", "loss_of_function", "pathogenic", 0.03),
        ("SPOP", "F133V", "missense", "gain_of_function", "pathogenic", 0.06),
        ("SPOP", "W131G", "missense", "gain_of_function", "pathogenic", 0.04),
        ("FOXA1", "R219S", "missense", "gain_of_function", "pathogenic", 0.03),
        ("PIK3CA", "H1047R", "missense", "gain_of_function", "pathogenic", 0.02),
        ("AR", "T878A", "missense", "gain_of_function", "pathogenic", 0.05),
        ("AR", "amplification", "amplification", "gain_of_function", "pathogenic", 0.20),
        ("RB1", "del", "deletion", "loss_of_function", "pathogenic", 0.10),
        ("MYC", "amplification", "amplification", "gain_of_function", "pathogenic", 0.08),
        ("ATM", "truncation", "frameshift", "loss_of_function", "pathogenic", 0.04),
        ("CDK12", "biallelic loss", "deletion", "loss_of_function", "pathogenic", 0.05),
        ("HOXB13", "G84E", "missense", "gain_of_function", "pathogenic", 0.01),
    ]
    count = 0
    for gene, pos, mtype, cons, clin, freq in mutations:
        loader.add_mutation(gene, pos, mtype, cons, clin, freq)
        count += 1
    return count


# ── Tissue types ─────────────────────────────────────────────────────

def _seed_tissues(loader: EntityLoader) -> int:
    tissues = [
        ("Prostatic epithelium", "Glandular epithelium of the prostate", "prostate",
         ["Luminal epithelial cell", "Basal epithelial cell", "Neuroendocrine cell"], "UBERON:0002367"),
        ("Prostatic stroma", "Fibromuscular stroma of the prostate", "prostate",
         ["Stromal fibroblast", "Endothelial cell"], "UBERON:0004179"),
        ("Bone metastasis site", "Common metastatic site for prostate cancer", "bone",
         ["Osteoblast", "Osteoclast", "Cancer cell"], "UBERON:0002481"),
        ("Lymph node", "Secondary lymphoid organ; first site of metastatic spread", "lymphatic",
         ["CD8+ T cell", "CD4+ T cell", "Macrophage"], "UBERON:0000029"),
    ]
    count = 0
    for name, desc, organ, cts, uberon in tissues:
        loader.add_tissue_type(name, desc, organ, cts, uberon)
        count += 1
    return count


# ── Receptors ────────────────────────────────────────────────────────

def _seed_receptors(loader: EntityLoader) -> int:
    receptors = [
        ("Androgen Receptor", "Nuclear hormone receptor activated by testosterone/DHT",
         "nuclear", ["Testosterone", "DHT", "Enzalutamide"], "AR Signaling", "AR"),
        ("PSMA", "Prostate-specific membrane antigen (FOLH1); target for Lu-177 therapy",
         "type_II_membrane", [], "Folate metabolism", "FOLH1"),
        ("PD-1", "Programmed death receptor 1; immune checkpoint",
         "type_I_membrane", ["PD-L1", "PD-L2", "Pembrolizumab"], "PD-1/PD-L1 Immune Checkpoint", "PDCD1"),
        ("EGFR", "Epidermal growth factor receptor; RTK signaling",
         "RTK", ["EGF", "TGF-alpha"], "RAS/MAPK", "EGFR"),
        ("VEGFR2", "Vascular endothelial growth factor receptor; angiogenesis",
         "RTK", ["VEGF-A"], "Angiogenesis", "KDR"),
    ]
    count = 0
    for name, desc, rtype, ligs, pathway, gene in receptors:
        loader.add_receptor(name, desc, rtype, ligs, pathway, gene)
        count += 1
    return count


# ── Organs ───────────────────────────────────────────────────────────

def _seed_organs(loader: EntityLoader) -> int:
    organs = [
        ("Prostate", "Male reproductive gland; primary site of prostate cancer",
         "reproductive", ["Prostatic epithelium", "Prostatic stroma"], "UBERON:0002367"),
        ("Bone", "Common metastatic site for prostate cancer (osteoblastic lesions)",
         "skeletal", ["Bone metastasis site"], "UBERON:0002481"),
        ("Liver", "Occasional metastatic site for advanced prostate cancer",
         "digestive", [], "UBERON:0002107"),
        ("Lung", "Occasional metastatic site; hematogenous spread",
         "respiratory", [], "UBERON:0002048"),
        ("Lymph node", "First site of regional metastatic spread",
         "lymphatic", ["Lymph node"], "UBERON:0000029"),
    ]
    count = 0
    for name, desc, system, tts, uberon in organs:
        loader.add_organ(name, desc, system, tts, uberon)
        count += 1
    return count


# ── Cross-entity relationships ───────────────────────────────────────

def _seed_relationships(store: EntityStore) -> int:
    """Create relationships between entities that were just seeded."""
    # Find entities by name for linking
    entities_by_name = {}
    all_entities, _ = store.search(limit=500, status="active")
    for e in all_entities:
        entities_by_name[e.name.lower()] = e.entity_id

    def _link(src: str, tgt: str, rtype: RelationshipType, evidence: str = "curated") -> bool:
        src_id = entities_by_name.get(src.lower())
        tgt_id = entities_by_name.get(tgt.lower())
        if src_id and tgt_id:
            rel = Relationship(
                source_id=src_id, target_id=tgt_id,
                rel_type=rtype, evidence=evidence,
            )
            return store.add_relationship(rel)
        return False

    count = 0

    # Drug-target relationships
    drug_targets = [
        ("Enzalutamide", "Androgen Receptor", RelationshipType.INHIBITS),
        ("Abiraterone", "AR", RelationshipType.INHIBITS),
        ("Olaparib", "BRCA2", RelationshipType.TARGETS),
        ("Pembrolizumab", "PD-1", RelationshipType.INHIBITS),
        ("Docetaxel", "Luminal epithelial cell", RelationshipType.TARGETS),
    ]
    for src, tgt, rt in drug_targets:
        if _link(src, tgt, rt, "drug-target"):
            count += 1

    # Pathway-gene relationships
    pathway_genes = [
        ("AR Signaling", "AR", RelationshipType.REGULATES),
        ("PI3K/AKT/mTOR", "PTEN", RelationshipType.REGULATES),
        ("DNA Damage Response", "BRCA2", RelationshipType.REGULATES),
        ("p53 Pathway", "TP53", RelationshipType.REGULATES),
    ]
    for src, tgt, rt in pathway_genes:
        if _link(src, tgt, rt, "pathway-gene"):
            count += 1

    # Mutation-gene relationships
    mutations_to_genes = [
        ("TP53_R175H", "TP53"), ("PTEN_del exon 5-6", "PTEN"),
        ("BRCA2_del exon 13", "BRCA2"), ("SPOP_F133V", "SPOP"),
        ("AR_T878A", "AR"), ("PIK3CA_H1047R", "PIK3CA"),
    ]
    for mut_name, gene_name in mutations_to_genes:
        if _link(mut_name, gene_name, RelationshipType.MUTATED_IN, "mutation-gene"):
            count += 1

    # Cell type - tissue relationships
    cell_tissue = [
        ("Luminal epithelial cell", "Prostatic epithelium", RelationshipType.PART_OF),
        ("Basal epithelial cell", "Prostatic epithelium", RelationshipType.PART_OF),
        ("Stromal fibroblast", "Prostatic stroma", RelationshipType.PART_OF),
        ("Endothelial cell", "Prostatic stroma", RelationshipType.PART_OF),
    ]
    for src, tgt, rt in cell_tissue:
        if _link(src, tgt, rt, "cell-tissue"):
            count += 1

    # Tissue - organ relationships
    tissue_organ = [
        ("Prostatic epithelium", "Prostate", RelationshipType.PART_OF),
        ("Prostatic stroma", "Prostate", RelationshipType.PART_OF),
    ]
    for src, tgt, rt in tissue_organ:
        if _link(src, tgt, rt, "tissue-organ"):
            count += 1

    return count
