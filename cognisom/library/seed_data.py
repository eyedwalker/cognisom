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
    """Seed genes with research-grade descriptions and simulation parameters."""
    from .models import Gene

    # Each entry: (symbol, gene_type, map_location, description,
    #              physics_params, compartments, interacts_with, usd_prim_type)
    gene_data = [
        ("AR", "oncogene", "Xq12",
         "Androgen Receptor (AR) is a 919-amino acid ligand-activated nuclear "
         "transcription factor of the steroid hormone receptor superfamily. In "
         "prostate epithelium, AR exists in the cytoplasm bound to heat shock "
         "proteins (HSP90/HSP70). Upon binding dihydrotestosterone (DHT, Kd ~0.1 nM) "
         "or testosterone (Kd ~0.4 nM), AR undergoes conformational change, dissociates "
         "from HSPs, homodimerizes, and translocates to the nucleus. Nuclear AR "
         "recruits coactivators (SRC-1, p300/CBP) and pioneer factors (FOXA1, HOXB13) "
         "to bind androgen response elements (AREs) in promoter/enhancer regions, "
         "driving transcription of target genes including KLK3 (PSA), TMPRSS2, NKX3-1, "
         "and FKBP5. In castration-resistant prostate cancer (CRPC), AR signaling "
         "persists through multiple resistance mechanisms: AR gene amplification (~30% "
         "mCRPC), point mutations in the ligand-binding domain (T878A broadens ligand "
         "specificity; L702H enables glucocorticoid activation; F877L converts "
         "enzalutamide from antagonist to agonist), constitutively active splice "
         "variants (AR-V7 lacks LBD entirely), and intratumoral androgen synthesis "
         "via CYP17A1. AR is the primary therapeutic target in prostate cancer: "
         "enzalutamide and darolutamide competitively inhibit LBD binding, while "
         "abiraterone blocks upstream androgen biosynthesis.",
         {"nuclear_translocation_time_min": 15, "dna_binding_kd_nm": 0.5,
          "half_life_hours": 3.0, "transcription_rate_mrna_per_hour": 50,
          "degradation_rate_ubiquitin": 0.1},
         ["cytoplasm", "nucleus"],
         [{"target": "DHT", "type": "binds_to", "kd_nm": 0.1},
          {"target": "FOXA1", "type": "binds_to", "kd_nm": 50},
          {"target": "HSP90", "type": "binds_to", "kd_nm": 10},
          {"target": "SPOP", "type": "activates", "note": "ubiquitination target"}],
         "BioProtein"),

        ("TP53", "tumor_suppressor", "17p13.1",
         "Tumor protein p53 (TP53) encodes a 393-amino acid transcription factor "
         "that functions as the central hub of cellular stress response, earning its "
         "designation as 'guardian of the genome.' Under normal conditions, p53 is "
         "maintained at low levels through continuous MDM2-mediated ubiquitination "
         "and proteasomal degradation (half-life ~20 minutes). DNA damage, oncogene "
         "activation, or hypoxia triggers phosphorylation of p53 at Ser15/Ser20 by "
         "ATM/ATR and CHK1/CHK2 kinases, disrupting the MDM2 interaction and "
         "stabilizing p53. Stabilized p53 tetramerizes and binds response elements "
         "to transactivate target genes controlling cell cycle arrest (CDKN1A/p21 "
         "inhibits CDK2/CDK4 complexes), apoptosis (BAX, PUMA/BBC3, NOXA "
         "permeabilize mitochondrial outer membrane), senescence (via sustained "
         "p21 and RB hypophosphorylation), and DNA repair (GADD45, XPC). p53 also "
         "represses anti-apoptotic genes (BCL2, survivin) and metabolic genes "
         "(GLUT1, TIGAR rewires glycolysis). In prostate cancer, TP53 mutations "
         "occur in ~25% of primary tumors but >50% of metastatic castration-resistant "
         "disease, concentrated at hotspot residues in the DNA-binding domain: R175H "
         "(structural mutant, disrupts zinc coordination), R248W (DNA contact mutant, "
         "abolishes sequence-specific binding), R273H (DNA contact mutant), and Y220C "
         "(destabilizes beta-sandwich). Many TP53 mutations confer gain-of-function "
         "properties including enhanced invasion, chemoresistance, and genomic "
         "instability through interaction with p63/p73 family members.",
         {"half_life_normal_min": 20, "half_life_stabilized_hours": 6,
          "tetramerization_kd_nm": 50, "dna_binding_kd_nm": 5,
          "transactivation_fold": 100},
         ["nucleus", "cytoplasm", "mitochondria"],
         [{"target": "MDM2", "type": "binds_to", "kd_nm": 0.6},
          {"target": "BAX", "type": "activates", "note": "apoptosis"},
          {"target": "CDKN1A", "type": "activates", "note": "cell cycle arrest"},
          {"target": "BCL2", "type": "inhibits", "note": "anti-apoptotic"}],
         "BioProtein"),

        ("PTEN", "tumor_suppressor", "10q23.31",
         "Phosphatase and tensin homolog (PTEN) is a 403-amino acid dual-specificity "
         "phosphatase that functions as the principal negative regulator of the "
         "PI3K/AKT/mTOR signaling axis. PTEN dephosphorylates phosphatidylinositol "
         "(3,4,5)-trisphosphate (PIP3) at the D3 position, converting it to PIP2 and "
         "directly antagonizing PI3K activity. This lipid phosphatase activity is "
         "essential: loss of PTEN causes constitutive PIP3 accumulation, hyperactivation "
         "of AKT (via PDK1-mediated phosphorylation at Thr308 and mTORC2-mediated "
         "phosphorylation at Ser473), and downstream activation of mTORC1, S6K, and "
         "4E-BP1, promoting protein synthesis, cell growth, survival, and metabolic "
         "reprogramming. PTEN also possesses protein phosphatase activity, "
         "dephosphorylating FAK to suppress cell migration and invasion. Nuclear PTEN "
         "maintains chromosome stability through interaction with CENP-C and promotes "
         "p53 stability by antagonizing MDM2. PTEN loss occurs in ~40% of primary "
         "prostate cancers and >60% of metastatic disease, through homozygous deletion "
         "(most common), inactivating point mutations (C124S abolishes catalytic "
         "activity; R130* creates truncated protein), promoter methylation, or "
         "post-translational modification. PTEN haploinsufficiency (single-copy loss) "
         "is sufficient to promote tumorigenesis through dosage-dependent effects on "
         "PIP3 levels. PTEN loss also causes MHC-I downregulation, reducing immune "
         "surveillance, and cooperates with TP53 loss to drive lethal metastatic "
         "prostate cancer in mouse models.",
         {"catalytic_kcat_per_s": 22, "km_pip3_um": 25,
          "membrane_association_fraction": 0.3, "nuclear_fraction": 0.15,
          "half_life_hours": 4},
         ["cytoplasm", "cell_membrane", "nucleus"],
         [{"target": "PIP3", "type": "catalyzes", "km_um": 25},
          {"target": "AKT1", "type": "inhibits", "note": "via PIP3 depletion"},
          {"target": "FAK", "type": "inhibits", "note": "protein phosphatase"},
          {"target": "MDM2", "type": "inhibits", "note": "nuclear PTEN"}],
         "BioProtein"),

        ("BRCA2", "tumor_suppressor", "13q13.1",
         "Breast cancer type 2 susceptibility protein (BRCA2) is a 3,418-amino acid "
         "scaffolding protein essential for homologous recombination (HR) DNA repair. "
         "BRCA2 directly binds RAD51 recombinase through eight BRC repeats (BRC1-8) and "
         "a C-terminal domain, mediating RAD51 loading onto single-stranded DNA (ssDNA) "
         "at resected double-strand breaks (DSBs). This displaces RPA from ssDNA and "
         "forms the RAD51 nucleoprotein filament required for homology search and strand "
         "invasion. BRCA2 also stabilizes stalled replication forks by preventing MRE11 "
         "nuclease-mediated degradation. BRCA2 is recruited to DSBs via PALB2, which "
         "bridges BRCA1 and BRCA2, forming the BRCA1-PALB2-BRCA2 complex. Loss of BRCA2 "
         "causes HR deficiency (HRD), forcing cells to rely on error-prone repair "
         "mechanisms (NHEJ, MMEJ, theta-mediated end joining), leading to genomic "
         "instability, characteristic mutational signatures (SBS3, large-scale state "
         "transitions), and sensitivity to DNA-damaging agents. This creates a "
         "therapeutic vulnerability: PARP inhibitors (olaparib, rucaparib, talazoparib) "
         "trap PARP1/2 on DNA, generating replication-associated DSBs that require HR "
         "for resolution, causing synthetic lethality in BRCA2-deficient cells. BRCA2 "
         "mutations occur in ~6-8% of metastatic prostate cancers (germline ~3%, "
         "somatic ~4%) and are associated with aggressive disease, higher Gleason scores, "
         "and worse prognosis. Common mutations include frameshift deletions in exon 11 "
         "(E1143fs), splice site mutations, and large deletions. BRCA2-mutant prostate "
         "cancers also show enhanced sensitivity to platinum chemotherapy and may "
         "benefit from immune checkpoint inhibitors due to increased neoantigen load.",
         {"rad51_loading_rate_per_min": 5, "fork_protection_length_kb": 2,
          "half_life_hours": 6, "foci_formation_time_min": 30},
         ["nucleus"],
         [{"target": "RAD51", "type": "binds_to", "kd_nm": 15},
          {"target": "PALB2", "type": "binds_to", "kd_nm": 5},
          {"target": "RPA", "type": "inhibits", "note": "displacement from ssDNA"},
          {"target": "PARP1", "type": "activates", "note": "synthetic lethality when lost"}],
         "BioProtein"),

        ("MYC", "oncogene", "8q24.21",
         "MYC proto-oncogene protein is a 439-amino acid basic helix-loop-helix "
         "leucine zipper (bHLH-LZ) transcription factor that heterodimerizes with MAX "
         "to bind E-box sequences (CACGTG) in the promoters of ~15% of all human genes. "
         "MYC is a master regulator of cell growth, controlling ribosome biogenesis "
         "(rRNA, tRNA, ribosomal proteins), nucleotide biosynthesis (CAD, DHFR, TS), "
         "glycolytic metabolism (HK2, PKM2, LDHA, GLUT1), glutaminolysis (GLS1, SLC1A5), "
         "and mitochondrial biogenesis. MYC also promotes cell cycle progression by "
         "inducing cyclins D1/D2 and CDK4 while repressing CDK inhibitors p21 and p27. "
         "Critically, MYC expression is tightly regulated in normal cells through mRNA "
         "instability (half-life ~30 min) and rapid protein turnover (half-life ~20-30 "
         "min via FBXW7-mediated ubiquitination). In prostate cancer, MYC overexpression "
         "occurs through 8q24 amplification (~10-30% of cases), enhancer hijacking, and "
         "transcriptional upregulation by AR. MYC cooperates with AR to establish a "
         "feed-forward loop promoting castration resistance. MYC amplification is "
         "associated with transition to aggressive, AR-indifferent disease and correlates "
         "with high Gleason grade and poor outcomes.",
         {"dna_binding_kd_nm": 10, "half_life_protein_min": 25,
          "half_life_mrna_min": 30, "target_gene_count": 3000,
          "transactivation_fold": 5},
         ["nucleus"],
         [{"target": "MAX", "type": "binds_to", "kd_nm": 2},
          {"target": "FBXW7", "type": "binds_to", "note": "ubiquitination/degradation"},
          {"target": "CDK4", "type": "activates"},
          {"target": "LDHA", "type": "activates", "note": "Warburg effect"}],
         "BioProtein"),

        ("RB1", "tumor_suppressor", "13q14.2",
         "Retinoblastoma protein (RB1) is a 928-amino acid tumor suppressor that serves "
         "as the central gatekeeper of the G1/S cell cycle transition. In its "
         "hypophosphorylated state, RB1 binds and sequesters E2F transcription factors "
         "(E2F1-3), repressing transcription of genes required for S-phase entry "
         "including cyclin A, cyclin E, CDK2, DHFR, TK1, MCM helicases, and ORC "
         "components. RB1 also recruits chromatin-modifying complexes (HDAC1/2, SWI/SNF, "
         "HP1) to E2F-responsive promoters, establishing repressive chromatin. Mitogenic "
         "signals activate cyclin D-CDK4/6 complexes, which monophosphorylate RB1, "
         "followed by cyclin E-CDK2 hyperphosphorylation that fully inactivates RB1, "
         "releasing E2F and committing the cell to division. RB1 loss occurs in ~10-15% "
         "of primary prostate cancers and ~25% of metastatic CRPC, primarily through "
         "biallelic deletion. RB1 loss has profound consequences: it removes the cell "
         "cycle brake, enables E2F-driven proliferation independent of mitogenic signals, "
         "and critically drives lineage plasticity and neuroendocrine transdifferentiation "
         "(NEPC) in cooperation with TP53 loss. Combined RB1/TP53 loss enables "
         "SOX2/EZH2-mediated epigenetic reprogramming from luminal adenocarcinoma to "
         "aggressive small cell/neuroendocrine phenotype, which is AR-negative and "
         "resistant to hormonal therapies.",
         {"phosphorylation_sites": 16, "e2f_binding_kd_nm": 1,
          "half_life_hours": 12, "cdk4_km_um": 5},
         ["nucleus"],
         [{"target": "E2F1", "type": "inhibits", "kd_nm": 1},
          {"target": "CDK4", "type": "binds_to", "note": "substrate"},
          {"target": "HDAC1", "type": "activates", "note": "chromatin repression"},
          {"target": "EZH2", "type": "inhibits", "note": "when lost, EZH2 upregulated"}],
         "BioProtein"),

        ("ERG", "oncogene", "21q22.2",
         "ETS-related gene (ERG) is a 462-amino acid ETS family transcription factor "
         "that binds the core GGA(A/T) motif through its ETS domain. ERG is not normally "
         "expressed in prostate epithelium. However, the TMPRSS2-ERG gene fusion — "
         "created by interstitial deletion or translocation on chromosome 21 — places "
         "ERG under control of the androgen-responsive TMPRSS2 promoter, making it the "
         "most common genomic alteration in prostate cancer (~50% of cases). The fusion "
         "typically joins TMPRSS2 exon 1 or 2 to ERG exon 4, producing a truncated but "
         "transcriptionally active ERG protein. Ectopic ERG expression reprograms the "
         "AR cistrome by redirecting AR binding to ERG/ETS sites genome-wide, activating "
         "invasion-associated programs (MMP3, MMP9, PLAU, ADAM19), epithelial-mesenchymal "
         "transition (VIM, ZEB1/2), and WNT signaling while repressing prostate "
         "differentiation genes (NKX3-1, SLC45A3). ERG also disrupts SPOP-mediated "
         "protein degradation by directly binding SPOP, stabilizing oncoproteins including "
         "AR itself. Importantly, TMPRSS2-ERG is an early event in prostate carcinogenesis "
         "(present in ~20% of PIN lesions) but is insufficient for malignant transformation "
         "alone — it cooperates with PTEN loss, PI3K activation, or TP53 mutation for "
         "full oncogenic progression.",
         {"dna_binding_kd_nm": 15, "half_life_hours": 4,
          "transactivation_fold": 8},
         ["nucleus"],
         [{"target": "TMPRSS2", "type": "binds_to", "note": "gene fusion partner"},
          {"target": "SPOP", "type": "inhibits", "note": "blocks SPOP degradation"},
          {"target": "AR", "type": "activates", "note": "cistrome reprogramming"},
          {"target": "NKX3-1", "type": "inhibits", "note": "represses differentiation"}],
         "BioProtein"),

        ("TMPRSS2", "signaling", "21q22.3",
         "Transmembrane serine protease 2 (TMPRSS2) is a 492-amino acid type II "
         "transmembrane serine protease expressed predominantly in prostate epithelium "
         "under direct androgen receptor transcriptional control. TMPRSS2 contains an "
         "extracellular domain with LDLR class A, scavenger receptor cysteine-rich, and "
         "serine protease domains. Its normal function involves proteolytic activation of "
         "substrates at the cell surface, including matriptase and hepatocyte growth "
         "factor (HGF). TMPRSS2 gained clinical notoriety for two reasons: First, "
         "TMPRSS2-ERG gene fusions are the most common structural rearrangement in "
         "prostate cancer (~50%), placing ERG oncogene under androgen-responsive "
         "TMPRSS2 regulatory elements. The fusion occurs through a 3-Mb interstitial "
         "deletion on chromosome 21 or balanced translocation, and its frequency is "
         "higher in Caucasian than Asian populations. Second, TMPRSS2 cleaves the "
         "SARS-CoV-2 spike protein at the S2' site, facilitating viral membrane fusion "
         "after ACE2 binding. Its high expression in prostate may explain certain "
         "COVID-19 epidemiological observations. The TMPRSS2 promoter contains multiple "
         "AREs, making its transcription exquisitely sensitive to androgen levels, which "
         "is why the TMPRSS2-ERG fusion renders ERG expression androgen-dependent.",
         {"protease_kcat_per_s": 5, "half_life_hours": 8},
         ["cell_membrane"],
         [{"target": "ERG", "type": "activates", "note": "fusion drives expression"},
          {"target": "AR", "type": "binds_to", "note": "promoter regulation"},
          {"target": "HGF", "type": "catalyzes", "note": "proteolytic activation"}],
         "BioProtein"),

        ("FOXA1", "signaling", "14q21.1",
         "Forkhead box A1 (FOXA1) is a 472-amino acid winged-helix pioneer transcription "
         "factor that binds nucleosomal DNA and opens compacted chromatin to enable "
         "subsequent binding by other transcription factors, most critically the androgen "
         "receptor. FOXA1 binds its forkhead motif (TGTTTAC) in the context of "
         "nucleosome-wrapped DNA, displacing linker histones H1 and creating DNase I "
         "hypersensitive sites. This pioneering activity is essential for AR binding "
         "to ~50% of its genomic targets. FOXA1 establishes a prostate-specific "
         "enhancer landscape during development and maintains luminal cell identity "
         "in adult prostate. FOXA1 is mutated in ~10-15% of prostate cancers, "
         "predominantly in the Wing2 domain (affecting DNA binding) and the "
         "transactivation domain. Mutations are classified as Class 1 (loss of "
         "function, reduced chromatin opening, luminal-to-basal shift) or Class 2 "
         "(gain of function, expanded AR cistrome, hyperactivation of AR program). "
         "FOXA1 mutations are enriched in metastatic CRPC and are mutually exclusive "
         "with SPOP mutations. FOXA1 cooperates with HOXB13 to define AR binding "
         "at prostate-specific enhancers, and the FOXA1/AR/HOXB13 triad is considered "
         "the master regulatory axis of prostate luminal cell identity.",
         {"dna_binding_kd_nm": 5, "chromatin_opening_time_min": 30,
          "half_life_hours": 6, "nucleosome_displacement_fraction": 0.8},
         ["nucleus"],
         [{"target": "AR", "type": "activates", "note": "pioneer factor for AR binding"},
          {"target": "HOXB13", "type": "binds_to", "note": "cooperates at enhancers"},
          {"target": "H1", "type": "inhibits", "note": "displaces linker histone"}],
         "BioProtein"),

        ("SPOP", "tumor_suppressor", "17q21.33",
         "Speckle-type POZ protein (SPOP) is a 374-amino acid substrate adaptor for "
         "the CUL3-RBX1 E3 ubiquitin ligase complex. SPOP recognizes substrates through "
         "its N-terminal MATH domain and bridges them to CUL3 via its BTB/POZ domain, "
         "targeting them for K48-linked polyubiquitination and proteasomal degradation. "
         "Key SPOP substrates in prostate include AR (promotes AR turnover), SRC-3/NCOA3 "
         "(AR coactivator), ERG (drives degradation of fusion protein), TRIM24 (AR "
         "coactivator), DEK (chromatin remodeler), and BRD4 (BET protein). SPOP "
         "mutations occur in ~15% of primary prostate cancers, exclusively in the "
         "MATH domain at substrate-binding residues (F133V, W131G, F102C, Y87C), "
         "impairing substrate recognition. This stabilizes AR, SRC-3, ERG, and BRD4, "
         "collectively amplifying AR signaling and oncogenic transcription. Notably, "
         "SPOP mutations define a distinct molecular subtype of prostate cancer: "
         "they are mutually exclusive with TMPRSS2-ERG fusions, enriched for CHD1 "
         "co-deletion, associated with homogeneous AR-driven biology, and paradoxically "
         "confer sensitivity to BET inhibitors (through BRD4 stabilization) and "
         "androgen deprivation therapy. SPOP mutations are early clonal events, "
         "occurring in high-grade prostatic intraepithelial neoplasia (HGPIN) and "
         "maintained through metastatic progression.",
         {"ubiquitination_rate_per_min": 2, "substrate_kd_nm": 100,
          "half_life_hours": 8},
         ["nucleus"],
         [{"target": "AR", "type": "inhibits", "note": "ubiquitination/degradation"},
          {"target": "ERG", "type": "inhibits", "note": "ubiquitination/degradation"},
          {"target": "BRD4", "type": "inhibits", "note": "ubiquitination/degradation"},
          {"target": "CUL3", "type": "binds_to", "note": "E3 ligase complex"}],
         "BioProtein"),

        ("CDK12", "tumor_suppressor", "17q12",
         "Cyclin-dependent kinase 12 (CDK12) is a 1,490-amino acid serine/threonine "
         "kinase that pairs with cyclin K to phosphorylate the C-terminal domain (CTD) "
         "of RNA polymerase II at Ser2, promoting transcriptional elongation and "
         "3'-end processing of pre-mRNAs encoding DNA damage response (DDR) proteins. "
         "CDK12 is required for expression of BRCA1, ATR, FANCI, FANCD2, and other "
         "HR repair genes. Biallelic CDK12 loss occurs in ~5-7% of metastatic prostate "
         "cancers and produces a distinctive genomic phenotype: widespread tandem "
         "duplications (>100 per genome, median size ~0.5 Mb) that generate numerous "
         "gene fusions and expressed neoantigens. This makes CDK12-loss tumors one of "
         "the most immunogenic subtypes of prostate cancer, with high predicted "
         "neoantigen burden rivaling microsatellite-unstable cancers. CDK12-loss tumors "
         "show increased T-cell infiltration and may respond to immune checkpoint "
         "inhibitors, representing a biomarker for immunotherapy in prostate cancer. "
         "CDK12 loss also creates a functional HRD-like state through downregulation "
         "of DDR genes, potentially sensitizing cells to PARP inhibitors and platinum "
         "chemotherapy, though clinical responses to PARPi have been modest.",
         {"kinase_kcat_per_s": 0.5, "ctd_km_um": 2, "half_life_hours": 10},
         ["nucleus"],
         [{"target": "POLR2A", "type": "phosphorylates", "note": "CTD Ser2"},
          {"target": "Cyclin K", "type": "binds_to", "note": "kinase activator"},
          {"target": "BRCA1", "type": "activates", "note": "transcriptional elongation"},
          {"target": "ATR", "type": "activates", "note": "DDR gene expression"}],
         "BioProtein"),

        ("PIK3CA", "oncogene", "3q26.32",
         "Phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic subunit alpha "
         "(PIK3CA) encodes the p110alpha catalytic subunit of class IA PI3K, a lipid "
         "kinase that phosphorylates PIP2 to generate PIP3, the critical second messenger "
         "that recruits and activates AKT and PDK1 at the plasma membrane. PI3K is "
         "activated by receptor tyrosine kinases (EGFR, ERBB2, IGFR1) and RAS GTPase, "
         "coupling growth factor signaling to the AKT/mTOR survival axis. PIK3CA forms "
         "a heterodimer with the p85alpha regulatory subunit (PIK3R1), which inhibits "
         "catalytic activity in the basal state; receptor binding relieves this "
         "autoinhibition. Activating PIK3CA mutations occur in ~5-10% of prostate "
         "cancers, concentrated at three hotspots: E545K and E542K in the helical domain "
         "(disrupt p85 inhibitory contact) and H1047R in the kinase domain (alter "
         "substrate binding). These mutations constitutively activate PI3K signaling "
         "even in the presence of functional PTEN. PIK3CA mutations and PTEN loss are "
         "partially redundant but can co-occur in advanced disease, and their combined "
         "effect maximally activates the PI3K/AKT/mTOR axis.",
         {"catalytic_kcat_per_s": 8, "km_pip2_um": 20,
          "half_life_hours": 24, "membrane_recruitment_time_s": 30},
         ["cytoplasm", "cell_membrane"],
         [{"target": "PIP2", "type": "catalyzes", "km_um": 20},
          {"target": "AKT1", "type": "activates", "note": "via PIP3 generation"},
          {"target": "PTEN", "type": "inhibits", "note": "antagonistic relationship"},
          {"target": "PIK3R1", "type": "binds_to", "note": "regulatory subunit"}],
         "BioProtein"),

        ("AKT1", "oncogene", "14q32.33",
         "AKT serine/threonine kinase 1 (AKT1, also known as protein kinase B/PKB) is "
         "a 480-amino acid AGC family kinase that serves as the central node of the "
         "PI3K survival signaling pathway. AKT1 contains an N-terminal PH domain that "
         "binds PIP3 at the plasma membrane, a kinase domain, and a C-terminal "
         "regulatory domain. Full activation requires dual phosphorylation: PDK1 "
         "phosphorylates Thr308 in the activation loop, and mTORC2 phosphorylates "
         "Ser473 in the hydrophobic motif. Activated AKT phosphorylates >100 substrates "
         "controlling cell survival (BAD, caspase-9, FOXO1/3), growth (TSC2 inhibition "
         "releases mTORC1), metabolism (GLUT4 translocation, GSK3beta inhibition of "
         "glycogen synthase), and cell cycle (p21/p27 nuclear exclusion). AKT1 is "
         "constitutively activated in ~40% of prostate cancers through upstream PTEN "
         "loss or PIK3CA mutation. The E17K activating mutation in the PH domain "
         "enables PIP3-independent membrane recruitment and occurs in ~1-2% of prostate "
         "cancers. AKT inhibitors (ipatasertib, capivasertib) are in clinical trials "
         "for PTEN-loss prostate cancer.",
         {"kinase_kcat_per_s": 15, "half_life_hours": 18,
          "membrane_dwell_time_s": 120, "phosphorylation_targets": 100},
         ["cytoplasm", "cell_membrane", "nucleus"],
         [{"target": "mTORC1", "type": "activates", "note": "via TSC2 inhibition"},
          {"target": "BAD", "type": "phosphorylates", "note": "pro-survival"},
          {"target": "FOXO3", "type": "phosphorylates", "note": "nuclear exclusion"},
          {"target": "GSK3B", "type": "phosphorylates", "note": "inhibition"}],
         "BioProtein"),

        ("HOXB13", "oncogene", "17q21.32",
         "Homeobox B13 (HOXB13) is a 270-amino acid homeodomain transcription factor "
         "with highly restricted expression in the prostate, where it cooperates with "
         "FOXA1 and AR to define prostate-specific enhancer landscapes. HOXB13 binds "
         "its cognate homeodomain motif at AR/FOXA1 co-occupied enhancers and is "
         "essential for androgen-dependent transcription of prostate differentiation "
         "genes. The G84E germline variant (rs138213197) is the strongest known genetic "
         "risk factor for prostate cancer in men of European descent, conferring a "
         "~3-5 fold increased risk. G84E disrupts a conserved MEIS interaction domain, "
         "altering HOXB13's ability to recruit chromatin modifiers and potentially "
         "converting it from a differentiation factor to a proliferation driver. "
         "HOXB13 also directly interacts with AR corepressor HDAC3 and with MEIS1/2 "
         "homeodomain proteins. In CRPC, HOXB13 can drive AR-independent transcriptional "
         "programs and is essential for AR-V7 activity, making it a potential therapeutic "
         "target for treatment-resistant disease.",
         {"dna_binding_kd_nm": 8, "half_life_hours": 6},
         ["nucleus"],
         [{"target": "AR", "type": "activates", "note": "enhancer cooperativity"},
          {"target": "FOXA1", "type": "binds_to", "note": "pioneer factor triad"},
          {"target": "MEIS1", "type": "binds_to", "note": "G84E disrupts this"},
          {"target": "HDAC3", "type": "binds_to", "note": "corepressor recruitment"}],
         "BioProtein"),

        ("KMT2D", "tumor_suppressor", "12q13.12",
         "Lysine methyltransferase 2D (KMT2D, also MLL4) is a 5,537-amino acid histone "
         "H3 lysine 4 (H3K4) methyltransferase and the largest component of the COMPASS "
         "(Complex of Proteins Associated with Set1)-like complex. KMT2D catalyzes mono- "
         "and dimethylation of H3K4 at enhancers, establishing the H3K4me1 mark that "
         "defines enhancer identity. Active enhancers are further marked by H3K27ac "
         "(deposited by p300/CBP recruited to KMT2D-primed sites). KMT2D mutations "
         "occur in ~8% of prostate cancers, typically truncating or frameshift mutations "
         "that abolish catalytic activity. Loss of KMT2D collapses enhancer landscapes: "
         "H3K4me1 is globally reduced, leading to failure of lineage-specific gene "
         "activation. In prostate cancer, KMT2D loss disrupts AR-dependent enhancers "
         "and sensitizes cells to EZH2 inhibitors, since the unopposed H3K27me3 "
         "deposited by PRC2 at former enhancer sites becomes the dominant chromatin "
         "mark. KMT2D cooperates with the BAF (SWI/SNF) chromatin remodeling complex "
         "and with UTX (KDM6A) H3K27 demethylase within the same complex.",
         {"methyltransferase_kcat_per_min": 0.5, "half_life_hours": 18},
         ["nucleus"],
         [{"target": "H3K4", "type": "catalyzes", "note": "mono/dimethylation"},
          {"target": "p300", "type": "activates", "note": "enhancer priming"},
          {"target": "UTX", "type": "binds_to", "note": "COMPASS complex"},
          {"target": "EZH2", "type": "inhibits", "note": "functional antagonism"}],
         "BioProtein"),

        ("ATM", "tumor_suppressor", "11q22.3",
         "Ataxia-telangiectasia mutated (ATM) is a 3,056-amino acid serine/threonine "
         "kinase of the PI3K-related kinase (PIKK) family that serves as the apical "
         "kinase in the DNA double-strand break (DSB) response. ATM exists as an "
         "inactive homodimer in unperturbed cells. Upon DSB detection by the MRN complex "
         "(MRE11-RAD50-NBS1), ATM is recruited to break sites, undergoes autophosphorylation "
         "at Ser1981, and dissociates into active monomers. Activated ATM phosphorylates "
         ">700 substrates including H2AX (gamma-H2AX, marks DSBs), CHK2 (effector kinase), "
         "p53 (Ser15, stabilization), BRCA1 (Ser1387, HR activation), and NBS1. ATM also "
         "phosphorylates KAP1/TRIM28 to relax heterochromatin at DSB sites, enabling "
         "access for repair factors. ATM mutations occur in ~5-8% of metastatic prostate "
         "cancer through truncating mutations, missense mutations in the kinase domain, "
         "and large deletions. ATM loss impairs DSB repair, causing genomic instability "
         "but also creating therapeutic vulnerabilities. ATM-deficient tumors show "
         "sensitivity to PARP inhibitors (partial HR deficiency), ATR inhibitors "
         "(synthetic lethal with ATM loss), and platinum chemotherapy.",
         {"kinase_kcat_per_s": 2, "recruitment_time_s": 30,
          "substrates": 700, "half_life_hours": 24},
         ["nucleus"],
         [{"target": "H2AX", "type": "phosphorylates", "note": "gamma-H2AX at DSBs"},
          {"target": "CHK2", "type": "phosphorylates", "note": "effector kinase"},
          {"target": "TP53", "type": "phosphorylates", "note": "Ser15 stabilization"},
          {"target": "BRCA1", "type": "phosphorylates", "note": "HR activation"}],
         "BioProtein"),

        ("CHD1", "tumor_suppressor", "5q21.1",
         "Chromodomain-helicase-DNA binding protein 1 (CHD1) is a 1,710-amino acid "
         "ATP-dependent chromatin remodeler that recognizes H3K4me2/3 marks through its "
         "tandem chromodomains and uses its SNF2-like ATPase/helicase domain to slide, "
         "eject, and reposition nucleosomes. CHD1 maintains open chromatin at active "
         "promoters and transcriptionally active gene bodies, facilitating RNA Pol II "
         "elongation. CHD1 also plays a critical role in homologous recombination repair "
         "by promoting chromatin decompaction at DSB sites, enabling RAD51 loading and "
         "strand invasion. CHD1 homozygous deletion occurs in ~6-8% of prostate cancers "
         "and defines a molecular subtype that is virtually always co-mutant with SPOP "
         "and lacks ETS fusions. CHD1-deleted tumors exhibit a characteristic chromatin "
         "state with reduced H3K4me2 at promoters, impaired transcriptional elongation, "
         "and a partial HR deficiency that may sensitize to PARP inhibitors. CHD1 loss "
         "also alters the AR cistrome, shifting AR binding to new genomic loci and "
         "contributing to treatment resistance.",
         {"atpase_rate_per_s": 3, "nucleosome_sliding_speed_bp_per_s": 10,
          "half_life_hours": 12},
         ["nucleus"],
         [{"target": "H3K4me2", "type": "binds_to", "note": "chromodomain recognition"},
          {"target": "RAD51", "type": "activates", "note": "HR repair chromatin opening"},
          {"target": "SPOP", "type": "binds_to", "note": "co-deleted in same subtype"}],
         "BioProtein"),

        ("NKX3-1", "tumor_suppressor", "8p21.2",
         "NK3 homeobox 1 (NKX3-1) is a 234-amino acid homeodomain transcription factor "
         "and the earliest known marker of prostate epithelial differentiation. NKX3-1 "
         "is androgen-regulated and expressed exclusively in prostate luminal cells, where "
         "it maintains glandular differentiation and suppresses proliferation. NKX3-1 "
         "binds a specific DNA motif (TAAGT(G/A)) and activates prostate differentiation "
         "genes while repressing genes associated with invasion and stemness. NKX3-1 also "
         "functions as a guardian against oxidative DNA damage by activating the ATM-dependent "
         "DNA damage response and directly binding topoisomerase I to promote DNA repair. "
         "NKX3-1 resides on chromosome 8p21, a region of frequent allelic loss in prostate "
         "cancer. Hemizygous loss of NKX3-1 is one of the earliest events in prostate "
         "carcinogenesis, present in >80% of HGPIN lesions. NKX3-1 is haploinsufficient: "
         "single-copy loss reduces NKX3-1 protein below the threshold needed for tumor "
         "suppression. Biallelic loss is uncommon, suggesting that complete NKX3-1 "
         "absence may actually impair cancer cell viability by disrupting AR signaling.",
         {"dna_binding_kd_nm": 10, "half_life_hours": 4},
         ["nucleus"],
         [{"target": "AR", "type": "activates", "note": "AR target gene, feedforward"},
          {"target": "TOP1", "type": "binds_to", "note": "DNA repair"},
          {"target": "ATM", "type": "activates", "note": "DDR activation"}],
         "BioProtein"),

        ("EZH2", "oncogene", "7q36.1",
         "Enhancer of zeste homolog 2 (EZH2) is a 746-amino acid SET domain-containing "
         "histone methyltransferase and the catalytic subunit of Polycomb Repressive "
         "Complex 2 (PRC2). EZH2 trimethylates histone H3 at lysine 27 (H3K27me3), "
         "establishing a transcriptionally repressive chromatin state that silences "
         "tumor suppressors, differentiation genes, and developmental regulators. PRC2 "
         "contains EED (reads existing H3K27me3 for spreading), SUZ12 (nucleosome "
         "binding), and RBBP4/7 (histone chaperones). In prostate cancer, EZH2 is "
         "overexpressed through multiple mechanisms: MYC-driven transcription, "
         "microRNA-101 loss (miR-101 targets EZH2 mRNA), RB1 loss (E2F-driven "
         "transcription), and gene amplification. Elevated EZH2 silences a constellation "
         "of tumor suppressors (DAB2IP, ADRB2, SLIT2, CDH1) and promotes lineage "
         "plasticity by repressing luminal differentiation programs. EZH2 is particularly "
         "important in neuroendocrine prostate cancer (NEPC), where it cooperates with "
         "RB1/TP53 loss to drive epigenetic reprogramming. EZH2 also has PRC2-independent "
         "oncogenic functions: it acts as a transcriptional activator for AR, NF-kappaB, "
         "and STAT3 targets through direct methylation of non-histone substrates. EZH2 "
         "inhibitors (tazemetostat, CPI-1205) are in clinical trials for prostate cancer.",
         {"methyltransferase_kcat_per_min": 1.5, "h3k27me3_spreading_rate_kb_per_hour": 2,
          "half_life_hours": 8},
         ["nucleus"],
         [{"target": "H3K27", "type": "catalyzes", "note": "trimethylation"},
          {"target": "EED", "type": "binds_to", "note": "PRC2 complex"},
          {"target": "DAB2IP", "type": "inhibits", "note": "silencing"},
          {"target": "AR", "type": "activates", "note": "PRC2-independent"}],
         "BioProtein"),

        ("MDM2", "oncogene", "12q15",
         "Mouse double minute 2 homolog (MDM2) is a 491-amino acid RING finger E3 "
         "ubiquitin ligase that is the master negative regulator of p53 stability and "
         "activity. MDM2 binds the N-terminal transactivation domain of p53 (residues "
         "18-26) with high affinity (Kd ~0.6 microM), blocking p53's ability to recruit "
         "transcriptional coactivators. MDM2 also mono-ubiquitinates p53 at C-terminal "
         "lysines, promoting nuclear export, and catalyzes p53 polyubiquitination for "
         "proteasomal degradation. This creates an autoregulatory feedback loop: p53 "
         "transcriptionally activates MDM2, and MDM2 destroys p53, maintaining low "
         "steady-state p53 levels. DNA damage disrupts this loop via ATM/ATR-mediated "
         "phosphorylation of p53 (disrupts MDM2 binding) and MDM2 (promotes "
         "self-ubiquitination). MDMX (MDM4), a structural homolog lacking E3 ligase "
         "activity, heterodimerizes with MDM2 and enhances p53 degradation. MDM2 "
         "amplification occurs in ~5% of prostate cancers (enriched in TP53-wild-type "
         "tumors) and functionally inactivates the p53 pathway without mutation. MDM2 "
         "also has p53-independent oncogenic functions including ubiquitination of RB1, "
         "E-cadherin, FOXO3, and androgen receptor. Nutlins (idasanutlin, milademetan) "
         "are small-molecule MDM2-p53 interaction inhibitors in clinical trials.",
         {"ubiquitin_ligase_kcat_per_min": 3, "p53_binding_kd_um": 0.6,
          "half_life_min": 30, "autoubiquitination_rate": 0.1},
         ["nucleus", "cytoplasm"],
         [{"target": "TP53", "type": "inhibits", "kd_um": 0.6},
          {"target": "MDM4", "type": "binds_to", "note": "heterodimer"},
          {"target": "RB1", "type": "inhibits", "note": "p53-independent"},
          {"target": "FOXO3", "type": "inhibits", "note": "ubiquitination"}],
         "BioProtein"),
    ]

    count = 0
    for (symbol, gtype, chrom, desc, physics, compartments, interactions,
         usd_type) in gene_data:
        gene = Gene(
            name=symbol,
            display_name=symbol,
            description=desc,
            symbol=symbol,
            full_name=desc.split("(")[0].strip() if "(" in desc else (
                desc.split("is a")[0].strip() if "is a" in desc else symbol
            ),
            chromosome=chrom.split("p")[0].split("q")[0] if "p" in chrom or "q" in chrom else chrom,
            gene_type=gtype,
            map_location=chrom,
            source="curated",
            tags=["prostate_cancer", gtype],
            physics_params=physics,
            compartments=compartments,
            interacts_with=interactions,
            usd_prim_type=usd_type,
            color_rgb=[0.2, 0.6, 1.0] if gtype == "tumor_suppressor" else [1.0, 0.3, 0.3],
            mesh_type="protein_structure",
            scale_um=0.005,
        )
        if loader.store.add_entity(gene):
            count += 1

    # Add simulation-specific immune effects to gene entities
    # These values are used by twin_config.py via parameter_resolver
    _gene_immune_effects = {
        "PTEN": {"mhc1_downregulation_effect": 0.3, "tumor_growth_multiplier": 1.3,
                 "pi3k_pathway_activation": 0.8},
        "TP53": {"mhc1_downregulation_effect": 0.2, "tumor_growth_multiplier": 1.2,
                 "apoptosis_evasion": 0.5},
        "AR": {"mhc1_downregulation_effect": 0.0, "tumor_growth_multiplier": 1.1,
               "ar_signaling_boost": 1.5},
        "RB1": {"mhc1_downregulation_effect": 0.1, "tumor_growth_multiplier": 1.15,
                "neuroendocrine_risk": 0.3},
        "MYC": {"mhc1_downregulation_effect": 0.1, "tumor_growth_multiplier": 1.4,
                "metabolic_reprogramming": 0.6},
        "BRCA2": {"mhc1_downregulation_effect": 0.0, "tumor_growth_multiplier": 1.0,
                  "hrd_score": 0.9, "parp_sensitivity": 0.85},
        "ATM": {"mhc1_downregulation_effect": 0.0, "tumor_growth_multiplier": 1.05,
                "hrd_score": 0.5, "parp_sensitivity": 0.5},
        "CDK12": {"mhc1_downregulation_effect": 0.0, "tumor_growth_multiplier": 1.1,
                  "neoantigen_generation": 0.7, "immunotherapy_sensitivity": 0.6},
        "SPOP": {"mhc1_downregulation_effect": 0.0, "tumor_growth_multiplier": 1.1,
                 "ar_stability_increase": 0.4},
        "ERG": {"mhc1_downregulation_effect": 0.0, "tumor_growth_multiplier": 1.1,
                "invasion_boost": 0.5},
        "PIK3CA": {"mhc1_downregulation_effect": 0.1, "tumor_growth_multiplier": 1.2,
                   "pi3k_pathway_activation": 0.7},
        "EZH2": {"mhc1_downregulation_effect": 0.15, "tumor_growth_multiplier": 1.15,
                 "epigenetic_silencing": 0.6, "neuroendocrine_risk": 0.2},
    }
    for gene_name, effects in _gene_immune_effects.items():
        entity = loader.store.find_entity_by_name(gene_name, "gene")
        if entity:
            pp = entity.physics_params if hasattr(entity, "physics_params") else {}
            pp.update(effects)
            entity.physics_params = pp
            try:
                loader.store.update_entity(entity, changed_by="seed_immune_effects")
            except Exception:
                pass

    return count


# ── Drugs ────────────────────────────────────────────────────────────

def _seed_drugs(loader: EntityLoader) -> int:
    # Each entry: (name, drug_class, mechanism, targets, status, smiles)
    # Descriptions are research-grade with PK/PD parameters
    drugs = [
        ("Enzalutamide", "anti-androgen",
         "Second-generation androgen receptor (AR) antagonist that inhibits AR signaling "
         "at three levels: competitively blocks androgen binding to the ligand-binding domain "
         "(IC50 ~21 nM for AR binding), inhibits AR nuclear translocation by disrupting the "
         "importin-alpha interaction, and prevents AR binding to DNA by blocking the "
         "N/C-terminal interaction required for transcriptional activation. Enzalutamide "
         "binds AR with 5-8 fold greater affinity than first-generation antiandrogens "
         "(bicalutamide). Oral bioavailability >80%, steady-state Cmax ~16.6 mcg/mL, "
         "terminal half-life ~5.8 days. Metabolized by CYP2C8 and CYP3A4 to active "
         "metabolite N-desmethyl enzalutamide. Crosses blood-brain barrier (causes ~1% "
         "seizure risk). Resistance mechanisms include AR-F877L mutation (converts "
         "enzalutamide to agonist), AR-V7 splice variant (lacks LBD), AR amplification, "
         "glucocorticoid receptor bypass, and lineage plasticity to AR-null NEPC.",
         ["AR"], "approved",
         "CC1(C2CCC3(C(=O)NC(=C3C2CC(C1)(F)F)C#N)C)C4=CC(=C(C=C4)F)C(=O)NC5=CC=C(C=N5)C(F)(F)F"),

        ("Abiraterone", "anti-androgen",
         "Irreversible inhibitor of cytochrome P450 17A1 (CYP17A1), the enzyme catalyzing "
         "both 17-alpha-hydroxylase and 17,20-lyase reactions in androgen biosynthesis. "
         "Blocks conversion of pregnenolone to DHEA and progesterone to androstenedione "
         "in the testes, adrenal glands, and intratumoral steroidogenic pathway. "
         "Administered as abiraterone acetate prodrug, rapidly deacetylated to active "
         "abiraterone. IC50 for CYP17A1 ~2.5 nM, >10-fold selectivity over CYP17A1 "
         "hydroxylase. Requires co-administration with prednisone (5 mg BID) to prevent "
         "mineralocorticoid excess from upstream steroid accumulation. Terminal half-life "
         "~12 hours. Resistance via CYP17A1 point mutations (T208A), AR mutations "
         "enabling activation by progesterone metabolites, and intratumoral AKR1C3 "
         "upregulation (alternative DHT synthesis).",
         ["CYP17A1"], "approved", ""),

        ("Docetaxel", "chemotherapy",
         "Semi-synthetic taxane that binds beta-tubulin subunit at a site distinct from "
         "vinca alkaloids, stabilizing microtubule polymers and preventing depolymerization. "
         "This arrests cells in mitosis by blocking spindle dynamics required for chromosome "
         "segregation, activating the spindle assembly checkpoint and triggering mitotic "
         "catastrophe and apoptosis via the intrinsic (mitochondrial) pathway. Docetaxel "
         "also inhibits BCL-2 phosphorylation and induces BCL-2 degradation. In prostate "
         "cancer, docetaxel disrupts AR nuclear trafficking via microtubule-dependent "
         "transport, providing an additional mechanism against AR signaling. Standard "
         "dosing: 75 mg/m2 IV q3w. Terminal half-life ~11 hours. >95% protein-bound. "
         "CYP3A4 metabolized. Dose-limiting toxicities: neutropenia, peripheral neuropathy.",
         ["TUBB"], "approved", ""),

        ("Cabazitaxel", "chemotherapy",
         "Semi-synthetic taxane derivative with dimethyl substitutions at C7 and C10 that "
         "reduce affinity for P-glycoprotein (MDR1/ABCB1) efflux pump, overcoming the "
         "primary mechanism of docetaxel resistance. Cabazitaxel stabilizes microtubules "
         "with similar potency to docetaxel but maintains activity in docetaxel-resistant "
         "cell lines. Greater CNS penetration than docetaxel. Approved for mCRPC post-docetaxel. "
         "Standard dosing: 25 mg/m2 IV q3w with prednisone. Terminal half-life ~95 hours. "
         "CYP3A4/5 metabolized. Higher febrile neutropenia risk than docetaxel; G-CSF "
         "prophylaxis recommended.",
         ["TUBB"], "approved", ""),

        ("Olaparib", "PARP_inhibitor",
         "Potent inhibitor of poly(ADP-ribose) polymerase 1 and 2 (PARP1/2) that exploits "
         "synthetic lethality in tumors with homologous recombination repair deficiency "
         "(HRD). Olaparib inhibits PARP catalytic activity (IC50 ~5 nM for PARP1) and, "
         "critically, traps PARP-DNA complexes at single-strand break sites, converting "
         "them into cytotoxic double-strand breaks during replication. HR-deficient cells "
         "(BRCA1/2, ATM, PALB2, CDK12 mutations) cannot repair these DSBs by high-fidelity "
         "HR and rely on error-prone NHEJ, leading to chromosomal instability and cell "
         "death. FDA approved for mCRPC with BRCA1/2 mutations (PROfound trial: 7.4 mo "
         "rPFS improvement). Also active in ATM-mutant and CDK12-loss tumors, though "
         "with lower response rates. Oral, 300 mg BID. Half-life ~11 hours. "
         "Resistance: secondary BRCA2 reversion mutations restoring reading frame, "
         "53BP1/RIF1/SHIELDIN loss (restoring end resection), PARP1 mutations.",
         ["PARP1", "PARP2", "BRCA2"], "approved", ""),

        ("Rucaparib", "PARP_inhibitor",
         "PARP1/2/3 inhibitor with broader PARP family coverage than olaparib. "
         "Inhibits PARP1 (IC50 ~1.4 nM) and effectively traps PARP-DNA complexes. "
         "Approved for BRCA-mutated mCRPC (TRITON2/3 trials). Also inhibits tankyrase "
         "1/2 (TNKS1/2), which may contribute to WNT pathway modulation. Oral, 600 mg "
         "BID. Half-life ~17 hours. Notable for robust activity in genomic LOH-high "
         "tumors beyond BRCA1/2 mutations, suggesting utility in a broader HRD population.",
         ["PARP1", "BRCA2"], "approved", ""),

        ("Pembrolizumab", "immunotherapy",
         "Humanized IgG4-kappa monoclonal antibody that blocks the interaction between "
         "programmed death receptor 1 (PD-1) on T cells and its ligands PD-L1 (CD274) "
         "and PD-L2 (PDCD1LG2) on tumor cells and antigen-presenting cells. PD-1 is an "
         "inhibitory co-receptor that, upon ligand engagement, recruits SHP-2 phosphatase "
         "to dephosphorylate TCR signaling molecules (ZAP70, CD3-zeta), attenuating T cell "
         "activation, proliferation, and cytokine production. Pembrolizumab blockade "
         "restores anti-tumor T cell effector function. FDA approved for MSI-H/dMMR or "
         "TMB-H (>=10 mut/Mb) solid tumors including prostate cancer (KEYNOTE-158). "
         "In combination with enzalutamide (KEYNOTE-641) and olaparib (KEYNOTE-365) for "
         "broader mCRPC populations. IV 200 mg q3w or 400 mg q6w. Half-life ~26 days. "
         "Immune-related adverse events (irAEs) in ~15-20%: colitis, hepatitis, "
         "thyroiditis, pneumonitis, dermatitis.",
         ["PDCD1"], "approved", ""),

        ("Ipatasertib", "PI3K_pathway",
         "Highly selective ATP-competitive inhibitor of all three AKT isoforms (AKT1/2/3) "
         "with Ki ~0.3-4.6 nM. Blocks AKT-mediated phosphorylation of downstream substrates "
         "including TSC2 (mTORC1 activation), GSK3-beta (glycogen metabolism), FOXO3 "
         "(transcription), and BAD (anti-apoptotic). In PTEN-loss prostate cancer, "
         "ipatasertib + abiraterone showed improved rPFS (18.5 vs 16.5 months) in the "
         "IPATential150 trial, particularly in the PTEN-loss biomarker subgroup. Oral, "
         "400 mg QD. Half-life ~26 hours. Main toxicities: diarrhea, hyperglycemia, rash.",
         ["AKT1", "AKT2"], "clinical_trial", ""),

        ("Darolutamide", "anti-androgen",
         "Structurally distinct AR antagonist with a unique pyrazole core that differs "
         "from the thiohydantoin scaffold of enzalutamide and apalutamide. Binds AR LBD "
         "with high affinity (IC50 ~26 nM) but exhibits low blood-brain barrier penetration "
         "(P-gp substrate), resulting in significantly lower rates of CNS side effects "
         "(seizures, fatigue, falls) compared to other AR inhibitors. Darolutamide also "
         "inhibits mutant AR forms including F877L, W742C, and T878A that confer "
         "resistance to enzalutamide. ARAMIS trial demonstrated 40% reduction in risk of "
         "metastasis or death in non-metastatic CRPC. Active metabolite keto-darolutamide "
         "has similar AR affinity. Oral, 600 mg BID with food. Half-life ~20 hours.",
         ["AR"], "approved", ""),

        ("Lutetium-177 PSMA", "radiotherapy",
         "Radioligand therapy consisting of Lutetium-177 (beta-emitting radionuclide, "
         "half-life 6.7 days, max beta energy 0.5 MeV, tissue penetration ~2 mm) "
         "conjugated to PSMA-617, a small-molecule ligand targeting prostate-specific "
         "membrane antigen (PSMA/FOLH1). PSMA is a type II transmembrane glycoprotein "
         "overexpressed on >90% of prostate cancer cells (10-1000x higher than normal "
         "prostate). Upon binding, Lu-177 PSMA-617 is internalized and delivers "
         "cytotoxic beta radiation to tumor cells and neighboring cells (crossfire effect). "
         "VISION trial: improved OS by 4 months and rPFS by 5.3 months in mCRPC. "
         "Requires PSMA-PET confirmation of target expression. 7.4 GBq IV q6w x 6 cycles. "
         "Main toxicities: xerostomia, bone marrow suppression.",
         ["FOLH1"], "approved", ""),

        ("Talazoparib", "PARP_inhibitor",
         "Potent PARP1/2 inhibitor and the strongest PARP trapper in the clinical class "
         "(PARP trapping potency ~100x greater than olaparib). IC50 for PARP1 ~0.57 nM. "
         "The enhanced trapping ability generates more cytotoxic PARP-DNA complexes at "
         "lower concentrations, but also increases myelosuppression risk. TALAPRO-2 trial: "
         "talazoparib + enzalutamide improved rPFS in mCRPC regardless of HRR status "
         "(though HRR-deficient patients benefited most). Oral, 0.5 mg QD (when combined "
         "with enzalutamide). Half-life ~90 hours. Dose-limiting: thrombocytopenia, anemia.",
         ["PARP1", "BRCA2"], "approved", ""),

        ("Apalutamide", "anti-androgen",
         "Second-generation AR antagonist structurally related to enzalutamide (shared "
         "thiohydantoin core). Binds AR LBD with similar affinity to enzalutamide, "
         "inhibits nuclear translocation, DNA binding, and coactivator recruitment. "
         "SPARTAN trial: 24.3-month improvement in metastasis-free survival in "
         "non-metastatic CRPC. TITAN trial: 35% reduction in risk of death in metastatic "
         "castration-sensitive prostate cancer. Oral, 240 mg QD. Half-life ~72 hours "
         "(active metabolite N-desmethyl apalutamide has similar half-life and activity). "
         "CYP2C8/3A4 metabolized. Notable side effects: rash (24%), hypothyroidism (8%), "
         "seizure risk (~0.2%).",
         ["AR"], "approved", ""),
    ]
    count = 0
    for name, dclass, mech, targets, status, smiles in drugs:
        loader.add_drug(name, dclass, mech, targets, status, smiles)
        count += 1

    # Add simulation-specific physics_params to drug entities
    # These values transfer the TREATMENT_PROFILES data into the entity library
    _drug_sim_params = {
        "Enzalutamide": {
            "exhaustion_reversal": 0.0, "treg_effect": 0.0, "irae_base_risk": 0.05,
            "effect_onset_days": 30, "effect_ramp_days": 21,
            "requires_ar_sensitivity": True, "best_for": ["any"],
        },
        "Abiraterone": {
            "exhaustion_reversal": 0.0, "treg_effect": 0.0, "irae_base_risk": 0.05,
            "effect_onset_days": 28, "effect_ramp_days": 21,
            "requires_ar_sensitivity": True, "best_for": ["any"],
        },
        "Olaparib": {
            "exhaustion_reversal": 0.0, "treg_effect": 0.0, "irae_base_risk": 0.05,
            "effect_onset_days": 28, "effect_ramp_days": 28,
            "requires_dna_repair_defect": True, "best_for": ["any"],
        },
        "Rucaparib": {
            "exhaustion_reversal": 0.0, "treg_effect": 0.0, "irae_base_risk": 0.05,
            "effect_onset_days": 28, "effect_ramp_days": 28,
            "requires_dna_repair_defect": True, "best_for": ["any"],
        },
        "Talazoparib": {
            "exhaustion_reversal": 0.0, "treg_effect": 0.0, "irae_base_risk": 0.08,
            "effect_onset_days": 21, "effect_ramp_days": 21,
            "requires_dna_repair_defect": True, "best_for": ["any"],
        },
        "Pembrolizumab": {
            "exhaustion_reversal": 0.55, "treg_effect": 0.0, "irae_base_risk": 0.15,
            "effect_onset_days": 14, "effect_ramp_days": 21,
            "best_for": ["hot", "suppressed"],
        },
        "Docetaxel": {
            "exhaustion_reversal": 0.0, "treg_effect": 0.0, "irae_base_risk": 0.20,
            "effect_onset_days": 7, "effect_ramp_days": 14,
            "best_for": ["any"],
        },
        "Cabazitaxel": {
            "exhaustion_reversal": 0.0, "treg_effect": 0.0, "irae_base_risk": 0.25,
            "effect_onset_days": 7, "effect_ramp_days": 14,
            "best_for": ["any"],
        },
        "Darolutamide": {
            "exhaustion_reversal": 0.0, "treg_effect": 0.0, "irae_base_risk": 0.03,
            "effect_onset_days": 30, "effect_ramp_days": 21,
            "requires_ar_sensitivity": True, "best_for": ["any"],
        },
        "Apalutamide": {
            "exhaustion_reversal": 0.0, "treg_effect": 0.0, "irae_base_risk": 0.05,
            "effect_onset_days": 28, "effect_ramp_days": 21,
            "requires_ar_sensitivity": True, "best_for": ["any"],
        },
        "Lutetium-177 PSMA": {
            "exhaustion_reversal": 0.05, "treg_effect": 0.0, "irae_base_risk": 0.10,
            "effect_onset_days": 14, "effect_ramp_days": 28,
            "best_for": ["any"],
        },
        "Ipatasertib": {
            "exhaustion_reversal": 0.0, "treg_effect": 0.0, "irae_base_risk": 0.10,
            "effect_onset_days": 21, "effect_ramp_days": 21,
            "best_for": ["any"],
        },
    }

    for drug_name, sim_params in _drug_sim_params.items():
        entity = loader.store.find_entity_by_name(drug_name, "drug")
        if entity:
            existing_pp = entity.physics_params if hasattr(entity, "physics_params") else {}
            existing_pp.update(sim_params)
            entity.physics_params = existing_pp
            try:
                loader.store.update_entity(entity, changed_by="seed_simulation_params")
            except Exception:
                pass

    return count


# ── Cell types ───────────────────────────────────────────────────────

def _seed_cell_types(loader: EntityLoader) -> int:
    cell_types = [
        ("Luminal epithelial cell",
         "The predominant secretory cell of the prostate gland, comprising ~60% of the "
         "epithelial compartment. Luminal cells express high levels of androgen receptor "
         "(AR) and are the primary source of prostate-specific antigen (PSA/KLK3), "
         "prostatic acid phosphatase (PAP), and other secretory proteins in the seminal "
         "fluid. Characterized by cytokeratin 8/18 (CK8/CK18) expression and apical "
         "E-cadherin. AR-dependent transcription drives luminal cell survival and "
         "proliferation; androgen deprivation therapy induces apoptosis preferentially "
         "in this compartment. Most prostate adenocarcinomas originate from luminal cells, "
         "maintaining AR expression and luminal markers throughout disease progression. "
         "Division time ~24-48 hours in cancer; quiescent in normal tissue.",
         "prostate", ["AR+", "PSA+", "CK8+", "CK18+", "NKX3-1+", "FOXA1+"],
         "epithelial", "CL:0002327"),

        ("Basal epithelial cell",
         "Multipotent progenitor cells forming the outer layer of prostate acini, resting "
         "on the basement membrane. Basal cells express p63 (TP63), CK5, CK14, and low "
         "levels of AR. They serve as the stem/progenitor compartment, giving rise to "
         "luminal, neuroendocrine, and intermediate transit-amplifying cells. Basal cells "
         "are resistant to androgen deprivation and may serve as cells of origin for "
         "certain prostate cancers, particularly basal-type tumors. They secrete hedgehog "
         "ligands that maintain stromal homeostasis and produce components of the basement "
         "membrane (laminin, collagen IV). Loss of the basal cell layer is the histological "
         "hallmark of invasive adenocarcinoma (vs. benign HGPIN which retains patchy basal cells).",
         "prostate", ["p63+", "CK5+", "CK14+", "AR-low", "BCL2+"],
         "epithelial", "CL:0000646"),

        ("Neuroendocrine cell",
         "Rare (<1%) post-mitotic endocrine cells scattered within prostate acini that "
         "secrete neuropeptides including chromogranin A (CHGA), synaptophysin (SYP), "
         "neuron-specific enolase (NSE), serotonin, and bombesin/GRP. These paracrine "
         "signals promote proliferation and survival of neighboring luminal cells. "
         "Neuroendocrine cells are AR-negative and androgen-independent, providing a "
         "survival niche during ADT. Treatment-emergent neuroendocrine prostate cancer "
         "(NEPC) arises when luminal adenocarcinoma cells undergo transdifferentiation "
         "(driven by RB1/TP53 loss + EZH2/SOX2 upregulation) to acquire neuroendocrine "
         "features: AR-negative, high proliferation, visceral metastases, poor prognosis "
         "(median OS ~7 months). NEPC represents ~15-20% of treatment-resistant mCRPC.",
         "prostate", ["SYP+", "CHGA+", "NSE+", "AR-", "CD56+"],
         "neuroendocrine", "CL:0000165"),

        ("Stromal fibroblast",
         "Mesenchymal cells embedded in the extracellular matrix (ECM) of the prostate "
         "stroma, producing collagen I/III, fibronectin, and proteoglycans. Normal stromal "
         "fibroblasts maintain epithelial homeostasis through paracrine signaling: "
         "AR-dependent production of growth factors (FGF7/10, IGF1, HGF) acts on epithelial "
         "cells. In the tumor microenvironment, fibroblasts become activated (alpha-SMA+, "
         "vimentin+) and remodel ECM through MMP2/9 secretion, creating invasion-permissive "
         "tracks. Stromal AR signaling is essential for prostate development; stromal-epithelial "
         "crosstalk through DHT-dependent andromedins drives normal glandular growth.",
         "prostate", ["VIM+", "ACTA2+", "FAP-", "COL1A1+"],
         "mesenchymal", "CL:0000057"),

        ("Cancer-associated fibroblast",
         "Activated fibroblasts in the tumor microenvironment (TME) that promote cancer "
         "progression through ECM remodeling, pro-angiogenic signaling, immune suppression, "
         "and metabolic support. CAFs express FAP (fibroblast activation protein), PDPN "
         "(podoplanin), alpha-SMA, and S100A4. They secrete CXCL12 (attracts immune cells), "
         "TGF-beta (immunosuppressive), IL-6 (activates STAT3 in cancer cells), and VEGF "
         "(promotes angiogenesis). CAFs also provide metabolic fuel to cancer cells via "
         "reverse Warburg effect: CAFs undergo aerobic glycolysis, secreting lactate and "
         "pyruvate that feed cancer cell oxidative phosphorylation. CAF-derived exosomes "
         "transfer miRNAs that promote treatment resistance. CAFs represent 20-80% of "
         "tumor stromal cells and are associated with poor prognosis.",
         "prostate", ["FAP+", "PDPN+", "alpha-SMA+", "S100A4+", "IL6+"],
         "mesenchymal", ""),

        ("Endothelial cell",
         "Specialized squamous cells lining blood vessels (CD31+/PECAM1, CD34+, VEGFR2+) "
         "that form a selective barrier between blood and tissue. In the prostate, "
         "endothelial cells regulate oxygen and nutrient delivery, leukocyte trafficking, "
         "and angiogenesis. Tumor-associated endothelial cells are phenotypically distinct: "
         "they express elevated VEGFR2, DLL4 (Notch ligand), and angiopoietin-2, and form "
         "irregular, leaky vessels with poor pericyte coverage. This chaotic vasculature "
         "creates hypoxic zones that promote tumor aggression and treatment resistance. "
         "Endothelial cells also express PD-L1 in the TME and present antigens to T cells, "
         "serving as non-professional APCs that can either activate or tolerize immune cells.",
         "prostate", ["CD31+", "CD34+", "VEGFR2+", "VE-cadherin+", "vWF+"],
         "endothelial", "CL:0000115"),

        ("CD8+ T cell",
         "Cytotoxic T lymphocyte (CTL) and the primary effector of adaptive anti-tumor "
         "immunity. CD8+ T cells recognize tumor neoantigens presented on MHC class I "
         "molecules through their T cell receptor (TCR). Upon activation (TCR signal + "
         "CD28 co-stimulation + IL-2), CTLs clonally expand and kill target cells through "
         "two mechanisms: perforin/granzyme pathway (perforin pores allow granzyme B entry, "
         "activating caspase-3/7 apoptosis) and Fas/FasL interaction. In the prostate TME, "
         "CD8+ T cells become progressively exhausted (PD-1hi, TIM-3+, LAG-3+, TOX+) "
         "through chronic antigen stimulation, losing effector function in a hierarchical "
         "manner: IL-2 production lost first, then TNF-alpha, then IFN-gamma, and finally "
         "cytotoxic capacity. Checkpoint inhibitors (anti-PD-1) can reinvigorate a "
         "progenitor-exhausted (TCF1+, PD-1int) subset but not terminally exhausted "
         "(TCF1-, PD-1hi) cells. Migration speed ~10-15 um/min in tissue.",
         "blood", ["CD8+", "CD3+", "TCR+", "perforin+", "granzyme B+"],
         "lymphoid", "CL:0000625"),

        ("CD4+ T cell",
         "Helper T lymphocyte that orchestrates adaptive immune responses through cytokine "
         "production and cell-cell contact. CD4+ T cells recognize antigens on MHC class II "
         "molecules and differentiate into functional subsets: Th1 (IFN-gamma, anti-tumor), "
         "Th2 (IL-4/IL-13, anti-helminth), Th17 (IL-17, mucosal immunity), Tfh (IL-21, "
         "B cell help), and Treg (IL-10/TGF-beta, immunosuppression). In the prostate TME, "
         "the Th1/Treg balance is critical: high Th1 infiltration correlates with favorable "
         "prognosis, while Treg accumulation suppresses anti-tumor immunity. CD4+ T cells "
         "also provide 'help' to CD8+ CTLs through CD40L-CD40 interactions with dendritic "
         "cells, licensing them for optimal CTL priming. CD4+ T cells can also directly "
         "kill MHC-II+ tumor cells through granzyme B, though this is less common.",
         "blood", ["CD4+", "CD3+", "TCR+", "CD40L+"],
         "lymphoid", "CL:0000624"),

        ("NK cell",
         "Innate lymphoid cell that provides rapid cytotoxicity against tumor cells without "
         "requiring prior antigen sensitization or MHC restriction. NK cells integrate "
         "activating signals (NKG2D binding MICA/B on stressed cells, DNAM-1, natural "
         "cytotoxicity receptors NKp46/NKp30/NKp44) and inhibitory signals (KIR receptors "
         "binding self-MHC-I). When activating signals exceed inhibitory, NK cells degranulate "
         "(perforin/granzyme) and produce IFN-gamma and TNF-alpha. NK cells are particularly "
         "effective against MHC-I-low tumor cells (the 'missing self' hypothesis), which "
         "escape CD8+ T cell recognition. CD56bright NK cells (blood) produce cytokines; "
         "CD56dim NK cells (tissue) are primarily cytotoxic. In prostate cancer, NK cell "
         "infiltration is associated with better outcomes but is often limited by TGF-beta "
         "and IL-10 in the immunosuppressive TME.",
         "blood", ["CD56+", "CD16+", "NKp46+", "NKG2D+", "perforin+"],
         "lymphoid", "CL:0000623"),

        ("Macrophage",
         "Phagocytic myeloid cell with remarkable plasticity, existing on a spectrum between "
         "classically activated M1 (anti-tumor) and alternatively activated M2 (pro-tumor) "
         "states. M1 macrophages are induced by IFN-gamma + LPS/TNF-alpha, express iNOS "
         "(produces anti-microbial NO), IL-12, TNF-alpha, and MHC-II, and perform "
         "phagocytosis and antigen presentation. M2 macrophages are induced by IL-4/IL-13 "
         "(M2a), immune complexes (M2b), or IL-10/TGF-beta/glucocorticoids (M2c), express "
         "CD163, CD206 (mannose receptor), arginase-1, and secrete IL-10, TGF-beta, VEGF, "
         "and MMPs. Tumor-associated macrophages (TAMs) in prostate cancer are predominantly "
         "M2-polarized, promoting tumor growth through angiogenesis (VEGF), ECM remodeling "
         "(MMP2/9), immunosuppression (IL-10, PD-L1), and direct growth factor production "
         "(EGF, FGF). High CD163+ TAM density correlates with biochemical recurrence and "
         "metastasis. TAMs represent the most abundant immune cell in the prostate TME.",
         "blood", ["CD68+", "CD163+", "CD14+", "CSF1R+", "CD206+"],
         "myeloid", "CL:0000235"),

        ("Prostate cancer stem cell",
         "Self-renewing subpopulation (~0.1-5% of tumor cells) with enhanced tumorigenicity, "
         "chemoresistance, and metastatic potential. Prostate CSCs are enriched by CD44+/"
         "CD24-/ALDH+ phenotype, express pluripotency factors (OCT4, SOX2, NANOG), and "
         "demonstrate asymmetric division generating both CSC daughters and differentiated "
         "bulk tumor cells. CSCs resist standard therapies through quiescence (slow cycling, "
         "evading anti-mitotic drugs), ABC transporter efflux (ABCG2), enhanced DNA repair, "
         "anti-apoptotic programs (BCL2, MCL1), and immune evasion (low MHC-I). They express "
         "integrin alpha2-beta1 for basement membrane adhesion and CD133 for stem identity. "
         "CSCs are enriched after ADT, chemotherapy, and radiation, driving treatment "
         "resistance and tumor recurrence.",
         "prostate", ["CD44+", "ALDH+", "CD133+", "integrin-a2b1+", "SOX2+", "NANOG+"],
         "epithelial", ""),

        ("Regulatory T cell",
         "Immunosuppressive CD4+ T cell subset defined by the master transcription factor "
         "FOXP3 and surface markers CD25 (IL-2 receptor alpha chain) and CTLA-4. Tregs "
         "suppress anti-tumor immunity through multiple mechanisms: IL-10 and TGF-beta "
         "secretion (inhibit effector T cells and NK cells), CTLA-4-mediated trans-endocytosis "
         "of CD80/CD86 from dendritic cells (removing co-stimulatory signals), IL-2 "
         "consumption (metabolic disruption of effector T cells), and granzyme B-mediated "
         "killing of effector cells. In the prostate TME, Tregs are recruited by CCL22 "
         "and CXCL12 secreted by tumor cells and are expanded by TGF-beta and IDO-producing "
         "myeloid-derived suppressor cells. High Treg infiltration correlates with poor "
         "outcomes. Anti-CTLA-4 antibodies (ipilimumab) deplete Tregs in the TME through "
         "Fc-mediated ADCC while simultaneously blocking CTLA-4 checkpoint on effector T cells.",
         "blood", ["CD4+", "CD25hi", "FOXP3+", "CTLA-4+", "GITR+", "CD127lo"],
         "lymphoid", "CL:0000815"),
    ]
    count = 0
    for name, desc, origin, markers, lineage, cl_id in cell_types:
        loader.add_cell_type(name, desc, origin, markers, lineage, cl_id)
        count += 1
    return count


# ── Pathways ─────────────────────────────────────────────────────────

def _seed_pathways(loader: EntityLoader) -> int:
    pathways = [
        ("AR Signaling",
         "Androgen receptor (AR) signaling is the master regulatory axis of prostate "
         "development, homeostasis, and cancer progression. The canonical pathway begins "
         "with testicular testosterone (T) production, conversion to dihydrotestosterone "
         "(DHT) by 5-alpha-reductase type 2 in prostate stroma, and DHT binding to AR "
         "in luminal epithelial cells. Activated AR recruits pioneer factors FOXA1 and "
         "HOXB13, p300/CBP histone acetyltransferase, and the Mediator complex to "
         "androgen response elements (AREs), driving expression of PSA (KLK3), NKX3-1, "
         "TMPRSS2, FKBP5, and hundreds of other prostate-specific genes. In CRPC, AR "
         "signaling persists through AR amplification, activating mutations (T878A, "
         "L702H), splice variants (AR-V7), intratumoral steroidogenesis, and crosstalk "
         "with PI3K/AKT, MAPK, and glucocorticoid receptor bypass pathways.",
         "signaling", ["AR", "FOXA1", "HOXB13", "NKX3-1", "KLK3"],
         "R-HSA-9009391"),

        ("PI3K/AKT/mTOR",
         "The PI3K/AKT/mTOR pathway is the principal survival signaling cascade in "
         "prostate cancer, activated in >40% of cases through PTEN loss (most common), "
         "PIK3CA activating mutations, or AKT1 overexpression. Upon growth factor "
         "receptor activation (EGFR, IGF1R, ERBB2), PI3K phosphorylates PIP2 to "
         "generate PIP3, which recruits AKT to the plasma membrane for activation "
         "by PDK1 (Thr308) and mTORC2 (Ser473). Activated AKT phosphorylates TSC2 "
         "to relieve inhibition of mTORC1, driving cap-dependent translation (via S6K "
         "and 4E-BP1), ribosome biogenesis, lipid synthesis (via SREBP), and autophagy "
         "suppression. PTEN antagonizes this cascade by dephosphorylating PIP3. "
         "Reciprocal cross-inhibition exists between AR and PI3K pathways: PI3K/AKT "
         "suppresses AR transcriptional output, and vice versa. This means inhibiting "
         "one pathway reactivates the other, necessitating dual targeting.",
         "signaling", ["PIK3CA", "AKT1", "PTEN", "MTOR", "TSC1", "TSC2"],
         "R-HSA-1257604"),

        ("DNA Damage Response",
         "The DNA damage response (DDR) encompasses the sensing, signaling, and repair "
         "of DNA lesions, particularly double-strand breaks (DSBs). DSBs are detected "
         "by the MRN complex (MRE11-RAD50-NBS1), which recruits ATM kinase. ATM "
         "phosphorylates H2AX (gamma-H2AX), MDC1, and effector kinases CHK1/CHK2, "
         "establishing a signaling cascade that activates cell cycle checkpoints (G1/S "
         "via p53/p21, intra-S via CDC25A, G2/M via CDC25C). DSBs are repaired by "
         "two main pathways: homologous recombination (HR, high-fidelity, requires "
         "BRCA1-PALB2-BRCA2-RAD51 and a sister chromatid template) or non-homologous "
         "end joining (NHEJ, error-prone, 53BP1-RIF1-SHIELDIN pathway). HR deficiency "
         "(BRCA2, ATM, CDK12, PALB2 mutations) creates the therapeutic target for "
         "PARP inhibitors through synthetic lethality.",
         "signaling", ["BRCA2", "ATM", "CDK12", "CHEK2", "RAD51", "TP53"],
         "R-HSA-73894"),

        ("p53 Pathway",
         "The p53 tumor suppression pathway integrates diverse stress signals — DNA "
         "damage, oncogene activation, hypoxia, nucleotide depletion, ribosomal "
         "stress — into a unified transcriptional response. Under normal conditions, "
         "MDM2 maintains p53 at low levels through ubiquitin-mediated degradation "
         "(half-life ~20 min). Stress signals disrupt MDM2-p53 interaction via "
         "phosphorylation (ATM/ATR on p53 Ser15, CHK2 on Ser20) and ARF binding to "
         "MDM2. Stabilized p53 activates cell cycle arrest (p21/CDKN1A, 14-3-3-sigma), "
         "apoptosis (BAX, PUMA/BBC3, NOXA, death receptors DR5/TRAIL-R2), senescence "
         "(p21 sustained expression + RB engagement), DNA repair (GADD45, XPC), "
         "and metabolic reprogramming (TIGAR, SCO2, GLS2). p53 is the most frequently "
         "mutated gene in mCRPC (>50%), and its loss is required for neuroendocrine "
         "transdifferentiation alongside RB1.",
         "signaling", ["TP53", "MDM2", "CDKN1A", "BAX", "BBC3"],
         "R-HSA-6806003"),

        ("Wnt/beta-Catenin",
         "Canonical Wnt signaling drives stemness, self-renewal, and treatment resistance "
         "in prostate cancer. In the absence of Wnt ligands, cytoplasmic beta-catenin "
         "is phosphorylated by the destruction complex (APC, Axin, GSK3-beta, CK1-alpha) "
         "and targeted for proteasomal degradation. Wnt ligand binding to Frizzled/LRP5/6 "
         "co-receptors recruits Dishevelled, disrupting the destruction complex and "
         "allowing beta-catenin to accumulate and translocate to the nucleus, where it "
         "binds TCF/LEF transcription factors to activate MYC, cyclin D1, AXIN2, and "
         "stemness genes. APC mutations (~5% of prostate cancers), RNF43 inactivation, "
         "and RSPO2 fusions activate Wnt signaling. Wnt also crosstalk with AR: "
         "beta-catenin directly binds AR and enhances its transcriptional activity.",
         "signaling", ["CTNNB1", "APC", "RNF43", "RSPO2"],
         "R-HSA-195721"),

        ("RB/E2F Cell Cycle",
         "The RB1/E2F pathway is the central gatekeeper of G1-to-S cell cycle transition. "
         "Hypophosphorylated RB1 binds E2F1-3 transcription factors and recruits HDAC/SWI-SNF "
         "chromatin remodelers to repress S-phase genes. Mitogenic signals (via cyclin D-CDK4/6) "
         "initiate RB1 monophosphorylation, followed by cyclin E-CDK2 hyperphosphorylation, "
         "fully inactivating RB1 and releasing E2F to drive S-phase entry. CDK4/6 inhibitors "
         "(palbociclib, ribociclib) maintain RB1 in the hypophosphorylated state. RB1 loss "
         "(~15-25% mCRPC) removes this checkpoint entirely, enabling constitutive E2F-driven "
         "proliferation and, critically, enabling lineage plasticity and neuroendocrine "
         "transdifferentiation when combined with TP53 loss.",
         "signaling", ["RB1", "CDK4", "CDK6", "CCND1", "E2F1"],
         "R-HSA-69236"),

        ("TMPRSS2-ERG Fusion",
         "The TMPRSS2-ERG gene fusion is the most common structural genomic alteration in "
         "prostate cancer, present in ~50% of cases. It results from an interstitial "
         "deletion of ~3 Mb on chromosome 21 (or less commonly, balanced translocation) "
         "that juxtaposes the androgen-responsive TMPRSS2 promoter to the ERG oncogene "
         "coding sequence. This places ERG expression under androgen control, driving "
         "constitutive ERG transcription in luminal prostate cells. The fusion typically "
         "joins TMPRSS2 exon 1 or 2 to ERG exon 4, producing a truncated but "
         "transcriptionally active ERG protein. ERG reprograms the AR cistrome, activates "
         "invasion programs (MMP3/9, PLAU), suppresses differentiation (NKX3-1), and "
         "blocks SPOP-mediated protein degradation. TMPRSS2-ERG is an early event but "
         "insufficient alone for malignancy — requires cooperating lesions (PTEN loss, "
         "PI3K activation).",
         "signaling", ["TMPRSS2", "ERG", "ETV1", "ETV4"],
         ""),

        ("Glycolysis",
         "Glycolysis converts glucose to pyruvate through 10 enzymatic steps, generating "
         "2 ATP and 2 NADH per glucose molecule. In cancer cells, glycolysis is "
         "constitutively upregulated even in the presence of oxygen (aerobic glycolysis "
         "or Warburg effect), driven by HIF-1alpha, MYC, and PI3K/AKT signaling that "
         "upregulate glucose transporters (GLUT1/GLUT3), hexokinase 2 (HK2, first "
         "committed step), phosphofructokinase (PFK, rate-limiting), and pyruvate kinase "
         "M2 (PKM2, cancer-specific splice variant). PKM2 exists as a less-active dimer "
         "in cancer, diverting glycolytic intermediates into biosynthetic pathways "
         "(pentose phosphate pathway for nucleotides, serine biosynthesis for one-carbon "
         "metabolism). Lactate dehydrogenase A (LDHA) converts pyruvate to lactate, "
         "which is exported by MCT4, acidifying the TME and suppressing anti-tumor immunity.",
         "metabolic", ["HK2", "PKM", "LDHA", "PFKL"],
         "R-HSA-70171"),

        ("TCA Cycle",
         "The tricarboxylic acid (TCA/Krebs) cycle operates in the mitochondrial matrix, "
         "oxidizing acetyl-CoA from glycolysis, fatty acid beta-oxidation, and amino acid "
         "catabolism to generate NADH, FADH2, and GTP. Normal prostate epithelium has a "
         "uniquely truncated TCA cycle: high zinc accumulation (~10-fold above other "
         "tissues) inhibits aconitase (ACO2), blocking citrate oxidation and enabling "
         "citrate secretion into seminal fluid. Prostate cancer restores TCA function "
         "by downregulating zinc transporters (SLC39A1/ZIP1), reactivating aconitase, "
         "and enabling complete citrate oxidation for energy production — representing "
         "a metabolic transformation from citrate-producing to citrate-oxidizing phenotype. "
         "IDH1/2 mutations (rare in prostate) produce the oncometabolite 2-hydroxyglutarate "
         "that inhibits alpha-ketoglutarate-dependent dioxygenases (TET2, KDM, PHDs).",
         "metabolic", ["CS", "IDH1", "IDH2", "SDHA", "FH"],
         "R-HSA-71403"),

        ("Oxidative Phosphorylation",
         "Oxidative phosphorylation (OXPHOS) in the inner mitochondrial membrane couples "
         "electron transport (Complexes I-IV) to proton gradient-driven ATP synthesis "
         "(Complex V/ATP synthase), producing ~30-32 ATP per glucose. Complex I (NADH "
         "dehydrogenase, 45 subunits) and Complex II (succinate dehydrogenase) feed "
         "electrons to ubiquinone, Complex III (cytochrome bc1) transfers to cytochrome c, "
         "and Complex IV (cytochrome c oxidase) reduces O2 to H2O. While cancer was "
         "historically thought to be glycolysis-dependent, metastatic prostate cancer "
         "relies heavily on OXPHOS and lipid oxidation. Circulating tumor cells show "
         "elevated OXPHOS gene expression, and AR signaling promotes mitochondrial "
         "biogenesis through PGC-1alpha activation. Electron leakage from Complexes I "
         "and III generates superoxide (O2-), contributing to ROS signaling and DNA damage.",
         "metabolic", ["MT-ND1", "MT-CO1", "MT-ATP6", "NDUFS1"],
         "R-HSA-163200"),

        ("PD-1/PD-L1 Immune Checkpoint",
         "The PD-1/PD-L1 axis is a critical immune checkpoint that normally prevents "
         "autoimmunity but is co-opted by tumors to evade immune destruction. PD-1 "
         "(PDCD1) on activated T cells engages PD-L1 (CD274) on tumor cells, recruiting "
         "SHP-2 phosphatase to the TCR signaling complex, dephosphorylating ZAP70 and "
         "CD3-zeta, and attenuating T cell proliferation, cytokine production, and "
         "cytotoxicity. PD-L1 is upregulated on tumor cells by IFN-gamma (adaptive "
         "immune resistance), oncogenic signaling (PTEN loss, MYC amplification), and "
         "hypoxia (HIF-1alpha). PD-L2 (PDCD1LG2) is a second PD-1 ligand expressed "
         "mainly on dendritic cells. Prostate cancer is generally considered an "
         "immunologically cold tumor with low PD-L1 expression (~5-20%), limiting "
         "checkpoint inhibitor efficacy. Exceptions: MSI-H/dMMR tumors (~3%), TMB-high "
         "tumors, CDK12-loss tumors (neoantigen-rich), and tumors following PARP "
         "inhibitor treatment (which increases PD-L1 via cGAS-STING pathway activation).",
         "signaling", ["PDCD1", "CD274", "PDCD1LG2"],
         "R-HSA-389948"),

        ("Epigenetic Regulation",
         "Epigenetic mechanisms — DNA methylation, histone modification, chromatin "
         "remodeling, and non-coding RNA — control gene expression without altering DNA "
         "sequence and are profoundly dysregulated in prostate cancer. EZH2 (H3K27me3 "
         "writer, PRC2 catalytic subunit) is overexpressed in mCRPC and silences tumor "
         "suppressors (DAB2IP, CDH1). KMT2D (H3K4me1 writer, enhancer activator) is "
         "mutated in ~8%, collapsing AR-dependent enhancers. CHD1 (ATP-dependent chromatin "
         "remodeler, reads H3K4me2/3) is deleted in ~6%, impairing transcriptional "
         "elongation and HR repair. BRD4 (BET protein, reads H3K27ac at super-enhancers) "
         "drives MYC transcription and is stabilized by SPOP mutations. DNA methylation "
         "at CpG islands silences GSTP1 (~90% of prostate cancers, used as biomarker), "
         "APC, RASSF1A, and RARbeta2. Therapeutic targets: EZH2 inhibitors (tazemetostat), "
         "BET inhibitors (JQ1, birabresib), DNMT inhibitors (decitabine), and HDAC "
         "inhibitors (vorinostat, panobinostat).",
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
        ("Glucose",
         "D-glucose (dextrose) is the primary carbon and energy source for mammalian cells, "
         "entering via GLUT1-4 transporters and phosphorylated by hexokinase to glucose-6-phosphate "
         "(G6P), committing it to glycolysis. Normal prostate tissue consumes relatively little "
         "glucose due to truncated TCA cycle. Cancer cells dramatically upregulate glucose uptake "
         "(GLUT1 overexpression driven by HIF-1alpha and MYC) for aerobic glycolysis (Warburg "
         "effect) and biosynthetic pathways (PPP for NADPH and ribose-5-phosphate). Plasma "
         "glucose: ~5 mM; tumor interstitial glucose: ~0.5-2 mM (depleted). FDG-PET exploits "
         "this differential uptake for tumor imaging.",
         "C6H12O6", 180.16, "CHEBI:17234"),

        ("Oxygen",
         "Molecular oxygen serves as the terminal electron acceptor in mitochondrial Complex IV "
         "(cytochrome c oxidase), essential for oxidative phosphorylation. Tissue oxygenation "
         "in normal prostate: ~40-60 mmHg (pO2). Tumor hypoxia (<10 mmHg) is common due to "
         "chaotic vasculature and high consumption. Hypoxia stabilizes HIF-1alpha, driving "
         "VEGF (angiogenesis), GLUT1/LDHA (glycolytic switch), BNIP3/BNIP3L (autophagy), "
         "and LOX (collagen crosslinking, premetastatic niche). Oxygen diffusion limit from "
         "capillary: ~100-200 um. Diffusion coefficient in tissue: ~2x10^-5 cm^2/s.",
         "O2", 32.0, "CHEBI:15379"),

        ("ATP",
         "Adenosine 5'-triphosphate is the universal energy currency, coupling exergonic "
         "reactions (hydrolysis: DeltaG = -30.5 kJ/mol under standard conditions, ~-54 kJ/mol "
         "in vivo) to endergonic cellular processes including motor proteins, ion pumps, "
         "biosynthesis, and signaling. Intracellular ATP: ~1-10 mM; cytoplasmic concentration "
         "maintained by glycolysis (~2 ATP/glucose), OXPHOS (~30 ATP/glucose), and creatine "
         "kinase shuttle. ATP/ADP ratio is a key metabolic sensor: AMPK activates when ratio "
         "drops, inhibiting mTOR and activating catabolic pathways. Cancer cells maintain "
         "ATP through a combination of glycolysis, glutaminolysis, and fatty acid oxidation.",
         "C10H16N5O13P3", 507.18, "CHEBI:15422"),

        ("Lactate",
         "L-lactate is the end product of anaerobic glycolysis, produced from pyruvate by "
         "lactate dehydrogenase A (LDHA). In the Warburg effect, cancer cells convert >80% "
         "of glucose to lactate even in the presence of oxygen, exporting it via MCT4 "
         "monocarboxylate transporter. Extracellular lactate (tumor TME: ~10-40 mM vs "
         "normal ~1.5 mM) acidifies the microenvironment (pH 6.5-6.9), suppressing T cell "
         "and NK cell cytotoxicity, promoting M2 macrophage polarization, inhibiting dendritic "
         "cell antigen presentation, and stabilizing HIF-1alpha. Lactate also serves as a fuel "
         "for oxidative cancer cells (reverse Warburg/metabolic symbiosis) and promotes "
         "angiogenesis through VEGF upregulation.",
         "C3H6O3", 90.08, "CHEBI:24996"),

        ("Pyruvate",
         "Pyruvate sits at the metabolic crossroads between glycolysis, TCA cycle, "
         "gluconeogenesis, and amino acid metabolism. In the cytoplasm, pyruvate kinase "
         "(PKM2 in cancer) converts phosphoenolpyruvate to pyruvate. Pyruvate fate: "
         "(1) mitochondrial import via MPC1/2 transporter and oxidative decarboxylation "
         "by pyruvate dehydrogenase (PDH) to acetyl-CoA for TCA cycle; (2) reduction to "
         "lactate by LDHA (regenerating NAD+ for glycolysis); (3) carboxylation to "
         "oxaloacetate by pyruvate carboxylase (anaplerosis). Cancer cells shunt pyruvate "
         "toward lactate via LDHA upregulation and PDK1/3-mediated PDH inhibition.",
         "C3H4O3", 88.06, "CHEBI:15361"),

        ("Citrate",
         "Citrate has a unique role in prostate biology. Normal prostate epithelial cells "
         "accumulate extraordinarily high zinc levels (~10x other soft tissues), which "
         "inhibits mitochondrial aconitase (m-aconitase/ACO2), blocking citrate oxidation "
         "in the TCA cycle. This causes citrate to accumulate (intracellular: ~10 mM) and "
         "be secreted into seminal fluid (concentration: ~100-200 mM). Prostate cancer cells "
         "downregulate zinc transporters (ZIP1/SLC39A1), restoring aconitase activity and "
         "enabling full TCA cycle flux — a metabolic transformation from 'citrate factory' "
         "to 'citrate consumer.' This metabolic switch is so fundamental that some consider "
         "it the earliest biochemical change in prostate carcinogenesis.",
         "C6H8O7", 192.12, "CHEBI:16947"),

        ("Testosterone",
         "Testosterone (T) is the primary circulating androgen (normal male serum: ~10-35 nM), "
         "produced by Leydig cells in the testes under LH stimulation. In prostate tissue, "
         "T is converted to the more potent dihydrotestosterone (DHT) by 5-alpha-reductase "
         "type 2 (SRD5A2). T itself binds AR with Kd ~0.4 nM (weaker than DHT at ~0.1 nM). "
         "Androgen deprivation therapy (surgical or chemical castration) reduces serum T from "
         "~15 nM to <1 nM, but intratumoral T levels remain detectable (~1-5 nM in CRPC) due "
         "to adrenal androgens (DHEA, androstenedione) and de novo steroidogenesis within "
         "tumor cells. Testosterone is also converted to estradiol by aromatase (CYP19A1), "
         "which has both proliferative (ERalpha) and anti-proliferative (ERbeta) effects.",
         "C19H28O2", 288.42, "CHEBI:17347"),

        ("DHT",
         "5-alpha-Dihydrotestosterone is the most potent natural androgen, formed from "
         "testosterone by 5-alpha-reductase type 2 (SRD5A2) in prostate stroma and "
         "converted back to less active metabolites by 3-alpha-HSD. DHT binds AR with "
         "~3-fold higher affinity than testosterone (Kd ~0.1 nM) and ~5-fold slower "
         "dissociation rate, making it the primary AR agonist in prostate tissue. "
         "Intraprostatic DHT: ~5-10 nM in normal prostate, reduced to ~0.5-2 nM during "
         "ADT but sufficient to activate amplified/mutant AR in CRPC. 5-alpha-reductase "
         "inhibitors (finasteride, dutasteride) block DHT synthesis and reduce prostate "
         "cancer risk by ~25% (PCPT trial). In CRPC, alternative DHT synthesis via AKR1C3 "
         "(bypassing T) from adrenal precursors contributes to treatment resistance.",
         "C19H30O2", 290.44, "CHEBI:16330"),

        ("PSA",
         "Prostate-specific antigen (PSA, also KLK3) is a 237-amino acid kallikrein-family "
         "serine protease secreted by prostate luminal epithelial cells under AR transcriptional "
         "control. PSA's normal function is to cleave semenogelin I/II in seminal fluid, "
         "liquefying the seminal coagulum. Serum PSA (normal: <4 ng/mL) leaks from prostate "
         "into blood due to disrupted glandular architecture. PSA exists as free PSA (~10-30%) "
         "and complexed PSA (bound to alpha-1-antichymotrypsin and alpha-2-macroglobulin). "
         "%free PSA <10% raises cancer suspicion; >25% suggests BPH. PSA velocity >0.75 "
         "ng/mL/year and PSA density >0.15 ng/mL/cc are additional diagnostic criteria. "
         "PSA is the most widely used cancer biomarker but lacks specificity: elevated by "
         "BPH, prostatitis, ejaculation, and digital rectal examination.",
         "", 0.0, ""),

        ("NADH",
         "Nicotinamide adenine dinucleotide (reduced form) is the principal electron carrier "
         "from catabolic pathways (glycolysis: 2 NADH, pyruvate dehydrogenase: 2 NADH, TCA "
         "cycle: 6 NADH per glucose) to Complex I of the mitochondrial electron transport "
         "chain. Each NADH donates 2 electrons, driving proton pumping and generating ~2.5 ATP "
         "per NADH. Cytoplasmic NADH cannot cross the inner mitochondrial membrane and is "
         "shuttled via malate-aspartate shuttle (yields NADH in matrix) or glycerol-3-phosphate "
         "shuttle (yields FADH2, less efficient). NAD+/NADH ratio (~700:1 in cytoplasm) is a "
         "key metabolic regulator: sirtuins (SIRT1-7) require NAD+ for histone deacetylation, "
         "linking metabolism to epigenetics. Cancer cells increase NAD+ salvage via NAMPT.",
         "C21H29N7O14P2", 663.43, "CHEBI:16908"),

        ("Glutamine",
         "L-glutamine is the most abundant amino acid in plasma (~0.5-0.8 mM) and a critical "
         "fuel for rapidly proliferating cells. Glutaminase (GLS1, upregulated by MYC) converts "
         "glutamine to glutamate in mitochondria, which is then converted to alpha-ketoglutarate "
         "by glutamate dehydrogenase (GLUD1) or transaminases, feeding the TCA cycle "
         "(anaplerosis). Glutamine also provides nitrogen for purine and pyrimidine biosynthesis "
         "(via CAD and PPAT), hexosamine pathway (UDP-GlcNAc for O-GlcNAcylation), and "
         "glutathione synthesis (antioxidant defense). MYC-driven cancers are 'glutamine-addicted' "
         "and sensitive to GLS inhibitors (CB-839/telaglenastat). AR signaling also upregulates "
         "glutamine uptake in prostate cancer via SLC1A5/ASCT2 transporter.",
         "C5H10N2O3", 146.14, "CHEBI:28300"),

        ("Acetyl-CoA",
         "Acetyl-coenzyme A is the central metabolic intermediate linking carbohydrate, fatty "
         "acid, and amino acid metabolism. Generated in mitochondria by pyruvate dehydrogenase "
         "(from glycolysis), fatty acid beta-oxidation, and amino acid catabolism. Mitochondrial "
         "acetyl-CoA enters TCA cycle via citrate synthase. Cytoplasmic acetyl-CoA (generated "
         "by ATP-citrate lyase/ACLY from exported citrate or by ACSS2 from acetate) is the "
         "substrate for de novo lipogenesis (FASN, ACC) and histone acetylation (p300/CBP, "
         "GCN5, PCAF). Cancer cells upregulate cytoplasmic acetyl-CoA production to fuel "
         "membrane lipid synthesis for rapid proliferation and to maintain the acetylation "
         "landscape. ACLY inhibitors (bempedoic acid) and FASN inhibitors (TVB-2640) are "
         "under investigation in cancer.",
         "C23H38N7O17P3S", 809.57, "CHEBI:15351"),
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
