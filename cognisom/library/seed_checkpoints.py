"""
Seed Data — Expanded Molecular Biology Catalog
================================================

Immune checkpoints, growth factors, signaling kinases, transcription
factors, apoptosis regulators, and ECM components with PhD-level
descriptions, physics parameters, and interaction networks.
"""

from __future__ import annotations

import logging
from typing import Dict

from .models import BioEntity, EntityType, Relationship, RelationshipType
from .store import EntityStore

log = logging.getLogger(__name__)


def seed_expanded_catalog(store: EntityStore) -> Dict[str, int]:
    """Seed expanded molecular biology entities."""
    counts = {"checkpoints": 0, "growth_factors": 0, "kinases": 0,
              "transcription_factors": 0, "apoptosis": 0, "ecm": 0}

    counts["checkpoints"] += _seed_immune_checkpoints(store)
    counts["growth_factors"] += _seed_growth_factors(store)
    counts["kinases"] += _seed_signaling_kinases(store)
    counts["transcription_factors"] += _seed_transcription_factors(store)
    counts["apoptosis"] += _seed_apoptosis_regulators(store)
    counts["ecm"] += _seed_ecm_components(store)

    total = sum(counts.values())
    log.info("Expanded catalog seed complete: %d total items", total)
    return counts


def _add(store, name, etype, desc, tags, physics=None, compartments=None,
         interactions=None, color=None):
    """Helper to create and add an entity."""
    entity = BioEntity(
        name=name, display_name=name, description=desc,
        entity_type=etype, source="curated", tags=tags,
        physics_params=physics or {},
        compartments=compartments or [],
        interacts_with=interactions or [],
        usd_prim_type="BioProtein",
        color_rgb=color or [0.5, 0.5, 0.5],
    )
    return 1 if store.add_entity(entity) else 0


# ══════════════════════════════════════════════════════════════════════
# IMMUNE CHECKPOINTS
# ══════════════════════════════════════════════════════════════════════

def _seed_immune_checkpoints(store: EntityStore) -> int:
    R = EntityType.RECEPTOR
    c = 0

    c += _add(store, "CTLA-4 (CD152)", R,
        "Cytotoxic T-lymphocyte-associated protein 4 is a 223-amino acid inhibitory "
        "receptor and the first immune checkpoint targeted therapeutically. CTLA-4 "
        "competes with the costimulatory receptor CD28 for binding to B7-1 (CD80) and "
        "B7-2 (CD86) on antigen-presenting cells, but with ~20-fold higher affinity "
        "(Kd ~0.4 uM vs ~4 uM for CD28). CTLA-4 also removes B7 ligands from APC "
        "surfaces via trans-endocytosis, physically stripping costimulatory molecules. "
        "Constitutively expressed on Tregs (essential for their suppressive function) "
        "and upregulated on activated effector T cells. CTLA-4 recruits SHP-2 and "
        "PP2A phosphatases to the TCR signaling complex, attenuating T cell activation "
        "at the priming stage in lymph nodes. Ipilimumab (anti-CTLA-4) depletes "
        "intratumoral Tregs via FcgammaRIIIA-mediated ADCC and blocks CTLA-4 checkpoint "
        "on effector T cells. First checkpoint inhibitor to show survival benefit in "
        "melanoma (2011). High irAE rate (~15-30%) reflects broad immune activation.",
        ["immunology", "checkpoint", "immunotherapy"],
        {"cd28_competition_kd_ratio": 20, "treg_expression": "constitutive"},
        ["cell_membrane"],
        [{"target": "CD80", "type": "binds_to", "kd_um": 0.4},
         {"target": "CD86", "type": "binds_to", "kd_um": 0.4},
         {"target": "SHP-2", "type": "activates", "note": "inhibitory signaling"}],
        [0.8, 0.2, 0.2])

    c += _add(store, "LAG-3 (CD223)", R,
        "Lymphocyte-activation gene 3 is a 498-amino acid CD4 structural homolog that "
        "binds MHC class II with higher affinity than CD4, functioning as a negative "
        "regulator of T cell expansion. LAG-3 also binds FGL1 (fibrinogen-like protein 1, "
        "secreted by liver and tumor cells) as a major inhibitory ligand independent of "
        "MHC-II. LAG-3 is expressed on exhausted CD8+ T cells (co-expressed with PD-1), "
        "Tregs (contributes to suppressive function), and NK cells. LAG-3 signaling "
        "inhibits T cell proliferation and cytokine production through an incompletely "
        "characterized mechanism involving the KIEELE motif in its cytoplasmic tail. "
        "Relatlimab (anti-LAG-3) combined with nivolumab (Opdualag) was FDA-approved "
        "for melanoma in 2022, demonstrating that dual checkpoint blockade (PD-1+LAG-3) "
        "improves outcomes beyond anti-PD-1 alone with less toxicity than PD-1+CTLA-4.",
        ["immunology", "checkpoint", "immunotherapy"],
        {"mhc_ii_affinity_fold_vs_cd4": 100},
        ["cell_membrane"],
        [{"target": "MHC-II", "type": "binds_to", "note": "higher affinity than CD4"},
         {"target": "FGL1", "type": "binds_to", "note": "tumor-derived ligand"}],
        [0.8, 0.3, 0.3])

    c += _add(store, "TIM-3 (HAVCR2)", R,
        "T cell immunoglobulin and mucin domain-containing protein 3 is a 301-amino acid "
        "receptor that marks terminally exhausted T cells when co-expressed with PD-1. "
        "Unlike PD-1 and CTLA-4, TIM-3 lacks a classical ITIM motif; instead it signals "
        "through phosphorylation of Y256/Y263 in its cytoplasmic tail, releasing BAT3 "
        "and recruiting FYN kinase. TIM-3 binds four distinct ligands: galectin-9 "
        "(induces apoptosis of TIM-3+ Th1 cells), CEACAM1 (cis and trans interactions "
        "on T cells and tumor cells), phosphatidylserine (on apoptotic cells, mediating "
        "cross-presentation by dendritic cells), and HMGB1 (alarmin, blocks nucleic acid "
        "sensing). PD-1+TIM-3+ CD8+ T cells represent the terminally exhausted subset "
        "that CANNOT be rescued by anti-PD-1 alone, driving interest in anti-TIM-3 "
        "combination therapy. Sabatolimab and cobolimab are anti-TIM-3 antibodies in "
        "clinical trials for MDS and solid tumors.",
        ["immunology", "checkpoint", "exhaustion"],
        {},
        ["cell_membrane"],
        [{"target": "galectin-9", "type": "binds_to", "note": "Th1 apoptosis"},
         {"target": "CEACAM1", "type": "binds_to"},
         {"target": "PtdSer", "type": "binds_to", "note": "apoptotic cells"}],
        [0.8, 0.3, 0.5])

    c += _add(store, "TIGIT", R,
        "T cell immunoreceptor with Ig and ITIM domains is a 244-amino acid inhibitory "
        "receptor on T cells and NK cells that competes with the activating receptor "
        "DNAM-1 (CD226) for binding to CD155 (PVR/nectin-like 5) and CD112 (nectin-2) "
        "on tumor cells and APCs. TIGIT binds CD155 with much higher affinity than "
        "DNAM-1 (Kd ~1 nM vs ~119 nM), outcompeting the activating signal — analogous "
        "to the CTLA-4/CD28 paradigm. TIGIT also directly delivers inhibitory signals "
        "through its cytoplasmic ITIM domain, recruiting SHIP-1 phosphatase. On NK cells, "
        "TIGIT suppresses cytotoxicity and IFN-gamma production. TIGIT+ Tregs are highly "
        "suppressive and enriched in tumors. Tiragolumab (anti-TIGIT) showed promise in "
        "combination with atezolizumab in NSCLC (CITYSCAPE trial) but failed confirmatory "
        "SKYSCRAPER-01 trial, raising questions about patient selection and biomarkers.",
        ["immunology", "checkpoint", "NK_cell"],
        {"dnam1_competition_kd_ratio": 100},
        ["cell_membrane"],
        [{"target": "CD155", "type": "binds_to", "kd_nm": 1},
         {"target": "DNAM-1", "type": "inhibits", "note": "competitive antagonism"},
         {"target": "SHIP-1", "type": "activates", "note": "ITIM signaling"}],
        [0.7, 0.3, 0.6])

    c += _add(store, "CD47", R,
        "CD47 ('don't eat me' signal) is a 323-amino acid ubiquitously expressed "
        "transmembrane protein that binds SIRPalpha (signal regulatory protein alpha) "
        "on macrophages and dendritic cells, delivering a dominant anti-phagocytic signal. "
        "SIRPalpha contains ITIMs that recruit SHP-1/SHP-2 phosphatases, inhibiting "
        "myosin IIA accumulation at the phagocytic synapse and blocking engulfment. "
        "CD47 is overexpressed on virtually all human cancers (2-5 fold above normal), "
        "enabling immune evasion from the innate phagocytic checkpoint. Also serves as "
        "a self-marker on red blood cells (prevents splenic macrophage clearance; loss "
        "of CD47 marks aged RBCs for removal). Magrolimab (anti-CD47) blocks SIRPalpha "
        "interaction, enabling macrophage phagocytosis of tumor cells — particularly "
        "effective when combined with tumor-opsonizing antibodies (rituximab) that provide "
        "the 'eat me' signal via FcgammaR. On-target anemia is the major toxicity.",
        ["immunology", "checkpoint", "phagocytosis", "innate"],
        {"sirpa_kd_nm": 10, "rbc_expression": True},
        ["cell_membrane"],
        [{"target": "SIRPalpha", "type": "binds_to", "kd_nm": 10},
         {"target": "SHP-1", "type": "activates", "note": "anti-phagocytic"}],
        [0.6, 0.3, 0.8])

    c += _add(store, "IDO1", R,
        "Indoleamine 2,3-dioxygenase 1 is a 403-amino acid heme-containing enzyme that "
        "catalyzes the rate-limiting step of tryptophan catabolism via the kynurenine "
        "pathway: L-tryptophan + O2 → N-formylkynurenine. IDO1 is induced by IFN-gamma "
        "in dendritic cells, macrophages, and tumor cells, creating local tryptophan "
        "depletion and kynurenine accumulation. Dual immunosuppressive mechanism: (1) "
        "Tryptophan starvation activates GCN2 kinase in T cells, inhibiting mTOR and "
        "inducing T cell anergy/apoptosis; (2) Kynurenine activates the aryl hydrocarbon "
        "receptor (AhR) in T cells, promoting Treg differentiation and suppressing Th17. "
        "IDO1 is a major immunosuppressive mechanism in the TME. However, epacadostat "
        "(IDO1 inhibitor) spectacularly failed the ECHO-301 trial (with pembrolizumab in "
        "melanoma), casting doubt on IDO1 as a monotherapy target. The failure may reflect "
        "redundancy with IDO2 and TDO2 (alternative tryptophan catabolic enzymes).",
        ["immunology", "checkpoint", "metabolism", "immunosuppression"],
        {"km_tryptophan_um": 20, "kcat_per_s": 1.5},
        ["cytoplasm"],
        [{"target": "tryptophan", "type": "catalyzes", "km_um": 20},
         {"target": "AhR", "type": "activates", "note": "via kynurenine"},
         {"target": "GCN2", "type": "activates", "note": "via Trp depletion"}],
        [0.5, 0.7, 0.3])

    c += _add(store, "CD73 (NT5E)", R,
        "Ecto-5'-nucleotidase (574 amino acids, GPI-anchored) that converts extracellular "
        "AMP to adenosine — the terminal step of the immunosuppressive purinergic pathway. "
        "CD73 works in tandem with CD39 (NTPDase1), which converts immunostimulatory ATP "
        "(released from dying cells as a DAMP) to AMP. The resulting adenosine binds A2A "
        "receptor (A2AR) on T cells and NK cells, elevating cAMP and suppressing effector "
        "functions: reduced IFN-gamma, granzyme B, and proliferation. CD73 is overexpressed "
        "on tumor cells, Tregs, and myeloid-derived suppressor cells, creating an "
        "adenosine-rich immunosuppressive TME. CD73 expression correlates with poor "
        "prognosis across multiple cancers. Oleclumab (anti-CD73) is in clinical trials "
        "combined with durvalumab for NSCLC and TNBC.",
        ["immunology", "checkpoint", "purinergic", "metabolism"],
        {"km_amp_um": 50, "kcat_per_s": 15},
        ["cell_membrane"],
        [{"target": "AMP", "type": "catalyzes", "note": "→ adenosine"},
         {"target": "A2AR", "type": "activates", "note": "via adenosine product"},
         {"target": "CD39", "type": "binds_to", "note": "sequential pathway"}],
        [0.5, 0.8, 0.5])

    c += _add(store, "4-1BB (CD137)", R,
        "TNF receptor superfamily member 9 (255 amino acids) that functions as a potent "
        "T cell COSTIMULATORY receptor — unlike the inhibitory checkpoints above. 4-1BB "
        "is upregulated on activated CD8+ T cells, CD4+ T cells, NK cells, and dendritic "
        "cells. 4-1BB ligand (4-1BBL/TNFSF9) on APCs engages 4-1BB, recruiting TRAF1/2 "
        "and activating NF-kappaB and PI3K-AKT survival signaling. 4-1BB costimulation "
        "enhances T cell proliferation, cytokine production, and critically promotes "
        "long-term CD8+ T cell survival and memory formation by upregulating BCL-2/BCL-XL "
        "anti-apoptotic proteins. 4-1BB signaling also rescues exhausted T cells. "
        "Urelumab (strong agonist, hepatotoxicity at high doses) and utomilumab (weaker "
        "agonist, safer) are anti-4-1BB antibodies. 4-1BB costimulatory domains are "
        "incorporated into second-generation CAR-T cells (e.g., tisagenlecleucel/Kymriah) "
        "for enhanced persistence.",
        ["immunology", "costimulatory", "immunotherapy", "CAR-T"],
        {},
        ["cell_membrane"],
        [{"target": "4-1BBL", "type": "binds_to"},
         {"target": "TRAF1", "type": "activates"},
         {"target": "BCL-XL", "type": "activates", "note": "anti-apoptotic"}],
        [0.2, 0.8, 0.2])

    c += _add(store, "OX40 (CD134)", R,
        "TNF receptor superfamily member 4 (277 amino acids) that is a costimulatory "
        "receptor transiently upregulated on activated CD4+ T cells (peak at 24-48h). "
        "OX40 ligand (OX40L/TNFSF4) on APCs and endothelial cells engages OX40, "
        "recruiting TRAF2/3/5 and activating NF-kappaB and PI3K-AKT. OX40 signaling "
        "promotes CD4+ T cell clonal expansion, survival (BCL-2/BCL-XL upregulation), "
        "Th1/Th2 cytokine production, and memory T cell generation. Critically, OX40 "
        "agonism also inhibits Treg suppressive function and can convert Tregs into "
        "effector cells in the TME. Combined OX40/4-1BB costimulation is synergistic. "
        "Pogalizumab and ivuxolimab (anti-OX40 agonists) are in clinical trials. "
        "OX40-OX40L interaction is also important in allergic airway inflammation, and "
        "anti-OX40L (amlitelimab) treats atopic dermatitis.",
        ["immunology", "costimulatory", "immunotherapy"],
        {},
        ["cell_membrane"],
        [{"target": "OX40L", "type": "binds_to"},
         {"target": "TRAF2", "type": "activates"},
         {"target": "Treg", "type": "inhibits", "note": "suppressive function"}],
        [0.2, 0.7, 0.3])

    c += _add(store, "VISTA (B7-H5)", R,
        "V-domain immunoglobulin suppressor of T cell activation (311 amino acids) is a "
        "B7 family checkpoint that is uniquely pH-dependent — its inhibitory activity "
        "increases dramatically at acidic pH (6.0) found in the tumor microenvironment, "
        "making it a TME-selective checkpoint. VISTA binds PSGL-1 as its receptor on "
        "T cells at low pH but not at physiological pH (7.4). VISTA is constitutively "
        "expressed at high levels on myeloid cells (monocytes, macrophages, DCs) and at "
        "lower levels on T cells. VISTA suppresses T cell activation, proliferation, and "
        "cytokine production. VISTA expression is associated with immune-cold tumors. "
        "Anti-VISTA antibodies (CI-8993, HMBD-002) are in early clinical trials and may "
        "complement anti-PD-1 by targeting a non-redundant pathway active specifically "
        "in the acidic TME.",
        ["immunology", "checkpoint", "TME", "pH_dependent"],
        {"ph_optimum": 6.0},
        ["cell_membrane"],
        [{"target": "PSGL-1", "type": "binds_to", "note": "pH-dependent, stronger at pH 6"}],
        [0.6, 0.4, 0.7])

    log.info("Seeded %d immune checkpoint entities", c)
    return c


# ══════════════════════════════════════════════════════════════════════
# GROWTH FACTORS
# ══════════════════════════════════════════════════════════════════════

def _seed_growth_factors(store: EntityStore) -> int:
    C = EntityType.CYTOKINE
    c = 0

    c += _add(store, "EGF", C,
        "Epidermal growth factor (6 kDa, 53 amino acids) is the prototypic ligand for "
        "the EGFR/ERBB1 receptor tyrosine kinase. EGF binding induces EGFR "
        "homodimerization and heterodimerization (with ERBB2/HER2, ERBB3, ERBB4), "
        "activating intrinsic tyrosine kinase activity and autophosphorylation of "
        "C-terminal tyrosines. These phosphotyrosines recruit SH2-domain adaptor proteins "
        "(GRB2-SOS→RAS-RAF-MEK-ERK; GAB1→PI3K-AKT; PLCgamma→PKC) driving proliferation, "
        "survival, migration, and differentiation. EGFR is overexpressed or mutated in "
        "many cancers: EGFR amplification in GBM, L858R/exon 19 del activating mutations "
        "in NSCLC (sensitive to erlotinib/gefitinib/osimertinib TKIs), T790M resistance "
        "mutation (overcome by osimertinib). Cetuximab (anti-EGFR mAb) treats colorectal "
        "and head/neck cancers (requires KRAS wild-type).",
        ["growth_factor", "RTK", "cancer"],
        {"egfr_kd_nm": 2, "molecular_weight_kda": 6},
        ["extracellular"],
        [{"target": "EGFR", "type": "binds_to", "kd_nm": 2},
         {"target": "RAS", "type": "activates", "note": "via GRB2-SOS"},
         {"target": "PI3K", "type": "activates", "note": "via GAB1"}],
        [1.0, 0.8, 0.2])

    c += _add(store, "VEGF-A", C,
        "Vascular endothelial growth factor A (45 kDa homodimer) is the master regulator "
        "of angiogenesis — the formation of new blood vessels from existing vasculature. "
        "VEGF-A binds VEGFR2 (KDR, primary signaling receptor, Kd ~75 pM) and VEGFR1 "
        "(Flt-1, decoy/modulator). VEGF-A is transcriptionally induced by HIF-1alpha "
        "under hypoxic conditions via HRE in the VEGF promoter, creating a hypoxia→"
        "angiogenesis axis that tumors exploit for blood supply. VEGFR2 activation "
        "triggers PLCgamma-PKC (proliferation), PI3K-AKT (survival), SRC-VE-cadherin "
        "phosphorylation (permeability), and FAK (migration) in endothelial cells. "
        "Bevacizumab (anti-VEGF-A) and ramucirumab (anti-VEGFR2) treat colorectal, lung, "
        "renal, and other cancers. VEGF TKIs: sunitinib, sorafenib, axitinib, lenvatinib.",
        ["growth_factor", "angiogenesis", "cancer"],
        {"vegfr2_kd_pm": 75, "molecular_weight_kda": 45, "half_life_hours": 0.5},
        ["extracellular"],
        [{"target": "VEGFR2", "type": "binds_to", "kd_pm": 75},
         {"target": "HIF-1alpha", "type": "activates", "note": "induced by hypoxia"},
         {"target": "endothelial_cell", "type": "activates", "note": "proliferation/migration"}],
        [0.9, 0.3, 0.3])

    c += _add(store, "HGF", C,
        "Hepatocyte growth factor (90 kDa) is the sole ligand for the c-MET receptor "
        "tyrosine kinase, also known as scatter factor due to its ability to induce "
        "epithelial cell dispersal and migration. HGF is secreted as an inactive "
        "single-chain precursor by mesenchymal cells and activated by HGF activator "
        "(HGFA) serine protease cleavage into alpha/beta heterodimer. c-MET activation "
        "drives invasive growth program: epithelial-mesenchymal transition, cell "
        "scattering, branching morphogenesis, and survival. In cancer, MET amplification, "
        "MET exon 14 skipping mutations, and paracrine HGF from CAFs drive tumor "
        "invasion and metastasis. MET is also a resistance mechanism to EGFR inhibitors "
        "(MET amplification bypasses EGFR blockade via ERBB3-PI3K). Capmatinib and "
        "tepotinib (MET TKIs) treat MET exon 14 NSCLC. Cabozantinib (MET/VEGFR2/AXL).",
        ["growth_factor", "invasion", "metastasis"],
        {"met_kd_nm": 0.3, "molecular_weight_kda": 90},
        ["extracellular"],
        [{"target": "c-MET", "type": "binds_to", "kd_nm": 0.3},
         {"target": "RAS", "type": "activates", "note": "via GRB2"},
         {"target": "PI3K", "type": "activates", "note": "via GAB1"}],
        [0.8, 0.6, 0.2])

    c += _add(store, "VEGF-A splice variants", C,
        "VEGF-A gene produces multiple splice variants with distinct properties: VEGF121 "
        "(freely diffusible, no heparin binding), VEGF165 (most abundant, binds heparan "
        "sulfate and neuropilin-1 co-receptor, optimal signaling), VEGF189 (matrix-bound, "
        "released by MMP cleavage), and VEGF206 (matrix-sequestered). Alternative "
        "splicing at exon 8 produces anti-angiogenic VEGFxxxb isoforms (VEGF165b) that "
        "bind VEGFR2 but fail to activate it fully. The VEGF165/VEGF165b ratio shifts "
        "toward pro-angiogenic in cancer. Neuropilin-1 (NRP1) acts as a VEGFR2 "
        "co-receptor for VEGF165, enhancing signaling. The VEGF gradient created by "
        "differential heparan sulfate binding of splice variants guides endothelial tip "
        "cell migration during sprouting angiogenesis.",
        ["growth_factor", "angiogenesis", "splicing"],
        {},
        ["extracellular", "ECM"],
        [{"target": "VEGFR2", "type": "binds_to"},
         {"target": "NRP1", "type": "binds_to", "note": "VEGF165 co-receptor"}],
        [0.9, 0.4, 0.3])

    c += _add(store, "IGF-1", C,
        "Insulin-like growth factor 1 (7.6 kDa, 70 amino acids) is a major endocrine "
        "and paracrine growth factor with ~50% sequence homology to insulin. IGF-1 binds "
        "IGF-1R (receptor tyrosine kinase, Kd ~0.1 nM) activating IRS-1→PI3K-AKT "
        "(survival, glucose uptake, protein synthesis via mTOR) and GRB2-SOS→RAS-MAPK "
        "(proliferation). Circulating IGF-1 (produced primarily by liver under GH "
        "stimulation) is bound to IGF-binding proteins (IGFBP1-6) that regulate "
        "bioavailability; IGFBP3 carries ~75% of circulating IGF-1. Elevated IGF-1 "
        "levels are epidemiologically associated with increased cancer risk (prostate, "
        "breast, colorectal). IGF-1R inhibitors (linsitinib, ganitumab) have failed in "
        "clinical trials, possibly due to compensatory insulin receptor signaling.",
        ["growth_factor", "metabolism", "cancer"],
        {"igf1r_kd_nm": 0.1, "molecular_weight_kda": 7.6, "half_life_hours": 12},
        ["extracellular"],
        [{"target": "IGF-1R", "type": "binds_to", "kd_nm": 0.1},
         {"target": "PI3K", "type": "activates", "note": "via IRS-1"},
         {"target": "mTOR", "type": "activates", "note": "via AKT-TSC2"}],
        [0.3, 0.8, 0.5])

    c += _add(store, "IL-15", C,
        "Interleukin-15 (15 kDa) is essential for NK cell development, CD8+ memory T cell "
        "homeostasis, and ILC1 survival. Unlike most cytokines, IL-15 does not function "
        "as a soluble factor but is trans-presented: IL-15 binds IL-15Ralpha on the "
        "surface of dendritic cells and macrophages, and this complex is presented in "
        "trans to IL-2Rbeta/gamma-c on NK and CD8+ T cells. Trans-presentation is "
        "required because free IL-15 has low affinity for IL-2Rbeta (Kd ~1 nM for "
        "complex vs ~1 uM for IL-15 alone). IL-15 signals through JAK1/JAK3-STAT5, "
        "promoting survival (BCL-2 upregulation), proliferation, and effector function. "
        "ALT-803 (IL-15 superagonist/IL-15Ralpha-Fc fusion) enhances NK and CD8+ T cell "
        "anti-tumor activity and is in clinical trials for multiple cancers.",
        ["cytokine", "NK_cell", "memory_T_cell", "immunotherapy"],
        {"il2rb_kd_nm": 1, "molecular_weight_kda": 15},
        ["extracellular", "cell_membrane"],
        [{"target": "IL-15Ralpha", "type": "binds_to", "note": "trans-presentation"},
         {"target": "JAK1", "type": "activates"},
         {"target": "STAT5", "type": "activates"}],
        [1.0, 0.7, 0.2])

    c += _add(store, "FGF2 (bFGF)", C,
        "Basic fibroblast growth factor (18 kDa, 155 amino acids) is a potent mitogen for "
        "endothelial cells, fibroblasts, and smooth muscle cells. Lacks a signal peptide and "
        "is released non-classically (cell damage, exosome secretion, or membrane translocation). "
        "FGF2 binds FGFR1-4 receptor tyrosine kinases with heparan sulfate proteoglycan (HSPG) "
        "as an obligate co-receptor — the ternary FGF2-FGFR-HSPG complex is required for "
        "signaling. Activates RAS-MAPK (proliferation), PLCgamma-PKC, and PI3K-AKT pathways. "
        "In cancer: pro-angiogenic (synergizes with VEGF), promotes chemoresistance, and drives "
        "fibroblast activation in the TME. FGF2 is also essential for embryonic stem cell "
        "self-renewal (maintaining pluripotency). FGFR inhibitors (erdafitinib, futibatinib) "
        "treat FGFR-altered urothelial and cholangiocarcinoma.",
        ["growth_factor", "angiogenesis", "FGFR"],
        {"fgfr1_kd_nm": 5, "molecular_weight_kda": 18},
        ["extracellular", "cytoplasm", "nucleus"],
        [{"target": "FGFR1", "type": "binds_to", "kd_nm": 5},
         {"target": "HSPG", "type": "binds_to", "note": "obligate co-receptor"},
         {"target": "VEGF-A", "type": "activates", "note": "synergistic angiogenesis"}],
        [0.8, 0.5, 0.2])

    c += _add(store, "PDGF-BB", C,
        "Platelet-derived growth factor BB (30 kDa homodimer of two B chains) signals through "
        "PDGFRalpha (binds all PDGF dimers) and PDGFRbeta (binds BB and AB). PDGF-BB is the "
        "most potent activator of PDGFRbeta, which is critical for pericyte and smooth muscle "
        "cell recruitment during blood vessel maturation (stabilizing nascent VEGF-induced "
        "vessels). PDGFRbeta activation signals through RAS-MAPK, PI3K-AKT, and PLCgamma-PKC. "
        "In fibrosis: PDGF-BB drives hepatic stellate cell and myofibroblast proliferation, "
        "contributing to liver/lung/kidney fibrosis. In cancer: autocrine PDGF loops drive "
        "glioblastoma and dermatofibrosarcoma protuberans (DFSP, harboring COL1A1-PDGFB fusion). "
        "Imatinib inhibits PDGFRalpha/beta (along with ABL and KIT) and treats DFSP and "
        "hypereosinophilic syndrome with PDGFR fusions.",
        ["growth_factor", "PDGFR", "pericyte", "fibrosis"],
        {"pdgfrb_kd_nm": 0.5, "molecular_weight_kda": 30},
        ["extracellular"],
        [{"target": "PDGFRbeta", "type": "binds_to", "kd_nm": 0.5},
         {"target": "pericyte", "type": "activates", "note": "vessel maturation"},
         {"target": "RAS", "type": "activates", "note": "via GRB2-SOS"}],
        [0.7, 0.4, 0.8])

    c += _add(store, "SCF (Kit ligand)", C,
        "Stem cell factor (36 kDa, exists as soluble and membrane-bound forms) is the ligand "
        "for the c-KIT (CD117) receptor tyrosine kinase. SCF-KIT signaling is essential for "
        "hematopoietic stem cell maintenance, mast cell development and survival, melanocyte "
        "migration and survival, spermatogenesis, and interstitial cells of Cajal (gut pacemaker "
        "cells). KIT activation triggers PI3K-AKT, RAS-MAPK, JAK-STAT, and PLCgamma pathways. "
        "Gain-of-function KIT mutations (D816V in mastocytosis, exon 11 in GIST) cause "
        "constitutive activation. Imatinib treats KIT-mutant GIST (except D816V, which requires "
        "avapritinib). KIT is a diagnostic marker for GIST (>95% KIT+), mast cell neoplasms, "
        "and seminoma. CD117 on hematopoietic stem cells is a target for antibody-drug "
        "conjugates in conditioning for bone marrow transplant.",
        ["growth_factor", "KIT", "stem_cell", "GIST"],
        {"kit_kd_nm": 0.1, "molecular_weight_kda": 36},
        ["extracellular", "cell_membrane"],
        [{"target": "c-KIT", "type": "binds_to", "kd_nm": 0.1},
         {"target": "PI3K", "type": "activates"},
         {"target": "mast_cell", "type": "activates", "note": "development/survival"}],
        [0.6, 0.8, 0.3])

    c += _add(store, "BMP4", C,
        "Bone morphogenetic protein 4 (47 kDa homodimer) is a TGF-beta superfamily member "
        "that plays critical roles in embryonic development (ventral mesoderm specification, "
        "neural crest formation) and adult tissue homeostasis. BMP4 binds type I receptors "
        "(BMPR1A/ALK3, BMPR1B/ALK6) and type II receptors (BMPR2, ActRIIA/B), forming a "
        "heterotetrameric complex. Type I receptor phosphorylates SMAD1/5/8, which complex "
        "with SMAD4 and translocate to the nucleus to regulate target genes (ID1-4, RUNX2). "
        "BMP4 is antagonized by noggin, chordin, and follistatin (extracellular sequestration). "
        "In cancer, BMP4 has context-dependent roles: promotes differentiation (tumor-suppressive "
        "in some cancers) but also drives epithelial-mesenchymal transition and cancer stem cell "
        "maintenance in others. In prostate cancer, BMP4 promotes osteoblastic bone metastasis.",
        ["growth_factor", "TGF_beta_superfamily", "SMAD", "bone"],
        {"bmpr1a_kd_nm": 5, "molecular_weight_kda": 47},
        ["extracellular"],
        [{"target": "BMPR1A", "type": "binds_to", "kd_nm": 5},
         {"target": "SMAD1", "type": "activates", "note": "via BMPR1"},
         {"target": "noggin", "type": "binds_to", "note": "antagonist"}],
        [0.5, 0.7, 0.6])

    log.info("Seeded %d growth factor entities", c)
    return c


# ══════════════════════════════════════════════════════════════════════
# SIGNALING KINASES
# ══════════════════════════════════════════════════════════════════════

def _seed_signaling_kinases(store: EntityStore) -> int:
    P = EntityType.PROTEIN
    c = 0

    c += _add(store, "JAK1", P,
        "Janus kinase 1 (1154 amino acids, 130 kDa) is a non-receptor tyrosine kinase "
        "associated with type I and type II cytokine receptor signaling. JAK1 pairs with "
        "JAK2 (IFN-gamma receptor), JAK3 (common gamma-chain cytokines: IL-2/4/7/9/15/21), "
        "or TYK2 (type I IFN, IL-12/23). Upon cytokine binding, JAK1 trans-phosphorylates "
        "its kinase partner and receptor cytoplasmic tails, creating STAT docking sites. "
        "JAK1 mutations are found in ALL and hepatocellular carcinoma. JAK1 selective "
        "inhibitors: filgotinib (JAK1-selective, RA); baricitinib and tofacitinib are "
        "less selective (JAK1/2 and JAK1/3). JAK inhibitors suppress inflammatory cytokine "
        "signaling and treat RA, psoriatic arthritis, atopic dermatitis, alopecia areata, "
        "and vitiligo.",
        ["kinase", "JAK-STAT", "signaling"],
        {"kinase_kcat_per_s": 5, "half_life_hours": 12},
        ["cytoplasm", "cell_membrane"],
        [{"target": "STAT1", "type": "phosphorylates"},
         {"target": "STAT3", "type": "phosphorylates"},
         {"target": "JAK2", "type": "binds_to", "note": "trans-phosphorylation"}],
        [0.3, 0.5, 0.9])

    c += _add(store, "JAK2", P,
        "Janus kinase 2 (1132 amino acids) mediates signaling from hematopoietic cytokine "
        "receptors (EPOR, THPOR, G-CSFR) and type II cytokine receptors (IFN-gamma, IL-10). "
        "The JAK2 V617F gain-of-function mutation (valine to phenylalanine at position 617 "
        "in the pseudokinase domain, disrupting autoinhibition) is the driver of "
        "myeloproliferative neoplasms: present in ~95% of polycythemia vera, ~50-60% of "
        "essential thrombocythemia, and ~50% of primary myelofibrosis. V617F causes "
        "constitutive JAK2-STAT5 activation and cytokine-independent growth. Ruxolitinib "
        "(JAK1/2 inhibitor) is FDA-approved for myelofibrosis and PV; fedratinib "
        "(JAK2-selective) for myelofibrosis. JAK2 also mediates erythropoietin signaling "
        "for red blood cell production.",
        ["kinase", "JAK-STAT", "myeloproliferative"],
        {"kinase_kcat_per_s": 8},
        ["cytoplasm"],
        [{"target": "STAT5", "type": "phosphorylates"},
         {"target": "EPOR", "type": "binds_to", "note": "erythropoiesis"},
         {"target": "THPOR", "type": "binds_to", "note": "thrombopoiesis"}],
        [0.4, 0.5, 0.9])

    c += _add(store, "mTOR", P,
        "Mechanistic target of rapamycin (2549 amino acids, 289 kDa) is an atypical "
        "serine/threonine kinase of the PIKK family that serves as the central integrator "
        "of growth factor, nutrient, energy, and stress signals. mTOR exists in two "
        "complexes: mTORC1 (with Raptor, rapamycin-sensitive) senses amino acids (via "
        "Ragulator/Rag GTPases at lysosomes) and growth factors (via PI3K-AKT-TSC2), "
        "phosphorylating S6K1 (ribosome biogenesis) and 4E-BP1 (cap-dependent translation "
        "initiation); mTORC2 (with Rictor, rapamycin-resistant initially) phosphorylates "
        "AKT at Ser473 (full activation), SGK1, and PKCa. mTORC1 also inhibits autophagy "
        "(ULK1 phosphorylation) and promotes lipid synthesis (SREBP). Rapamycin/sirolimus "
        "and analogs (everolimus, temsirolimus) inhibit mTORC1 and treat RCC, breast cancer "
        "(with exemestane), and TSC-associated tumors.",
        ["kinase", "mTOR", "metabolism", "growth"],
        {"kinase_kcat_per_s": 2},
        ["cytoplasm", "lysosome_membrane"],
        [{"target": "S6K1", "type": "phosphorylates", "note": "mTORC1"},
         {"target": "4E-BP1", "type": "phosphorylates", "note": "mTORC1, translation"},
         {"target": "AKT1", "type": "phosphorylates", "note": "mTORC2, Ser473"}],
        [0.5, 0.5, 0.8])

    c += _add(store, "CDK4", P,
        "Cyclin-dependent kinase 4 (303 amino acids) partners with D-type cyclins "
        "(cyclin D1/D2/D3) to initiate RB1 phosphorylation at the G1/S cell cycle "
        "transition. CDK4 monophosphorylates RB1 (at up to 14 sites), partially "
        "inactivating E2F repression and allowing cyclin E-CDK2 to complete RB1 "
        "hyperphosphorylation (irreversible commitment to S-phase). CDK4 is inhibited by "
        "INK4 family members (p16INK4a/CDKN2A, p15INK4b, p18INK4c, p19INK4d) which "
        "compete with cyclin D for CDK4 binding. CDKN2A deletion (removes p16 brake) or "
        "cyclin D1 amplification constitutively activates CDK4 in many cancers. CDK4/6 "
        "inhibitors (palbociclib, ribociclib, abemaciclib) maintain RB1 in "
        "hypophosphorylated state and are transformative in HR+/HER2- breast cancer "
        "(PALOMA, MONALEESA, MONARCH trials). Require intact RB1 for efficacy.",
        ["kinase", "cell_cycle", "cancer"],
        {"rb_km_um": 5, "half_life_hours": 8},
        ["nucleus"],
        [{"target": "RB1", "type": "phosphorylates"},
         {"target": "Cyclin D1", "type": "binds_to", "note": "activating partner"},
         {"target": "p16/CDKN2A", "type": "binds_to", "note": "inhibitor"}],
        [0.6, 0.4, 0.8])

    c += _add(store, "BTK", P,
        "Bruton's tyrosine kinase (659 amino acids) is a non-receptor tyrosine kinase "
        "essential for B cell receptor (BCR) signaling, B cell development, and mature "
        "B cell survival. BTK contains PH, TH, SH3, SH2, and kinase domains. BCR "
        "activation recruits SYK kinase, which phosphorylates BLNK/SLP-65, creating "
        "docking sites for BTK. BTK then phosphorylates PLCgamma2, generating IP3 "
        "(calcium release from ER) and DAG (PKC activation, NF-kappaB). BTK mutations "
        "cause X-linked agammaglobulinemia (XLA, Bruton's disease) with absence of "
        "mature B cells. Ibrutinib (irreversible BTK inhibitor, binds C481) "
        "revolutionized treatment of CLL, mantle cell lymphoma, and Waldenstrom's. "
        "Acalabrutinib and zanubrutinib are more selective second-generation BTK "
        "inhibitors with fewer off-target effects (less atrial fibrillation).",
        ["kinase", "B_cell", "CLL", "lymphoma"],
        {"kinase_kcat_per_s": 3},
        ["cytoplasm"],
        [{"target": "PLCgamma2", "type": "phosphorylates"},
         {"target": "SYK", "type": "binds_to", "note": "upstream activator"},
         {"target": "NF-kappaB", "type": "activates", "note": "via PKC"}],
        [0.4, 0.6, 0.8])

    c += _add(store, "MEK1 (MAP2K1)", P,
        "Mitogen-activated protein kinase kinase 1 (393 amino acids) is a dual-specificity "
        "kinase that phosphorylates ERK1/2 at both threonine (Thr202) and tyrosine (Tyr204) "
        "residues in the activation loop — the only known substrates of MEK1. Activated by "
        "RAF1/BRAF phosphorylation at Ser218/Ser222 in the MEK1 activation segment. MEK1 is "
        "the central bottleneck of the RAS-RAF-MEK-ERK cascade, making it an ideal therapeutic "
        "target: trametinib (highly selective MEK1/2 inhibitor, Ki ~0.7 nM) is FDA-approved "
        "for BRAF V600E melanoma (with dabrafenib), NSCLC, and anaplastic thyroid cancer. "
        "Cobimetinib (with vemurafenib) and binimetinib (with encorafenib) are alternative "
        "MEK inhibitors. MEK inhibitor resistance: MAPK pathway reactivation via MEK2 "
        "mutations, RAF amplification, or receptor tyrosine kinase bypass.",
        ["kinase", "MAPK", "cancer"],
        {"kinase_ki_nm": 0.7},
        ["cytoplasm"],
        [{"target": "ERK1", "type": "phosphorylates", "note": "Thr202/Tyr204"},
         {"target": "ERK2", "type": "phosphorylates"},
         {"target": "RAF1", "type": "binds_to", "note": "activated by"}],
        [0.4, 0.6, 0.8])

    c += _add(store, "ERK1/2 (MAPK3/1)", P,
        "Extracellular signal-regulated kinases 1 and 2 (379/360 amino acids) are the "
        "terminal kinases of the canonical RAS-RAF-MEK-ERK mitogenic signaling cascade — "
        "the most frequently mutated pathway in human cancer (~40% of all cancers). Activated "
        "ERK1/2 phosphorylate >200 substrates in the cytoplasm (RSK, MNK, cPLA2) and "
        "nucleus (ELK1, c-FOS, c-MYC, ETS factors), driving proliferation, differentiation, "
        "survival, and migration. ERK nuclear translocation is mediated by importin-7. "
        "ERK also phosphorylates SOS1 and RAF (negative feedback), creating oscillatory "
        "signaling dynamics. ERK pathway activity can be read as phospho-ERK by "
        "immunohistochemistry. Ulixertinib is a first-in-class ERK1/2 inhibitor in trials "
        "for MAPK-driven cancers resistant to upstream RAF/MEK inhibitors.",
        ["kinase", "MAPK", "proliferation"],
        {"substrates": 200, "kinase_kcat_per_s": 10},
        ["cytoplasm", "nucleus"],
        [{"target": "ELK1", "type": "phosphorylates", "note": "nuclear TF"},
         {"target": "c-FOS", "type": "phosphorylates", "note": "AP-1 component"},
         {"target": "RSK", "type": "phosphorylates", "note": "cytoplasmic substrate"}],
        [0.5, 0.5, 0.8])

    c += _add(store, "SRC", P,
        "Proto-oncogene tyrosine-protein kinase SRC (536 amino acids) is the prototypic "
        "non-receptor tyrosine kinase and the first identified oncogene (v-src from Rous "
        "sarcoma virus, 1976 Nobel Prize). SRC contains SH3 (proline-rich binding), SH2 "
        "(phosphotyrosine binding), and kinase domains. Autoinhibited by intramolecular "
        "SH2 binding to C-terminal pTyr527 (phosphorylated by CSK); activated by "
        "dephosphorylation (PTPalpha) or displacement (receptor binding). SRC localizes "
        "to focal adhesions, integrin complexes, and receptor tyrosine kinases. "
        "Phosphorylates: FAK (cell migration), p130CAS (invasion), VE-cadherin "
        "(vascular permeability), STAT3 (survival). Overactive SRC drives invasion and "
        "metastasis in colon, breast, and prostate cancers. Dasatinib (SRC/ABL inhibitor) "
        "treats CML and Ph+ ALL.",
        ["kinase", "oncogene", "focal_adhesion", "invasion"],
        {"kinase_kcat_per_s": 8},
        ["cytoplasm", "cell_membrane", "focal_adhesions"],
        [{"target": "FAK", "type": "phosphorylates", "note": "migration"},
         {"target": "VE-cadherin", "type": "phosphorylates", "note": "permeability"},
         {"target": "CSK", "type": "binds_to", "note": "negative regulator"}],
        [0.5, 0.4, 0.9])

    c += _add(store, "Aurora A (AURKA)", P,
        "Aurora kinase A (403 amino acids) is a mitotic serine/threonine kinase essential "
        "for centrosome maturation, mitotic spindle assembly, and chromosome alignment. "
        "Aurora A is activated at centrosomes in late G2 by TPX2 binding (allosteric "
        "activation and protection from PP1 phosphatase). Substrates: TACC3 (centrosome "
        "maturation), PLK1 (mitotic entry), BRCA1 (checkpoint override), p53 (MDM2-mediated "
        "degradation). Aurora A amplification occurs in breast, ovarian, and colorectal "
        "cancers, driving centrosome amplification, supernumerary spindles, and chromosomal "
        "instability (CIN). Aurora A also stabilizes N-MYC in neuroblastoma (preventing "
        "FBXW7-mediated degradation). Alisertib (MLN8237, Aurora A inhibitor) showed "
        "activity in PTCL and neuroblastoma trials.",
        ["kinase", "mitosis", "centrosome", "CIN"],
        {"peak_activity": "mitosis", "kinase_kcat_per_s": 5},
        ["centrosome", "spindle"],
        [{"target": "TPX2", "type": "binds_to", "note": "allosteric activator"},
         {"target": "PLK1", "type": "phosphorylates", "note": "mitotic entry"},
         {"target": "N-MYC", "type": "activates", "note": "stabilization"}],
        [0.3, 0.9, 0.3])

    c += _add(store, "WEE1", P,
        "WEE1 kinase (646 amino acids) is the gatekeeper of the G2/M cell cycle checkpoint "
        "that prevents premature mitotic entry by phosphorylating CDK1 (CDC2) at inhibitory "
        "Tyr15. This keeps the CDK1/cyclin B complex inactive until all DNA damage is repaired "
        "and replication is complete. CDC25 phosphatases remove the Tyr15 phosphorylation to "
        "trigger mitotic entry. TP53-mutant cancers (which lack the G1/S checkpoint) are "
        "heavily dependent on the G2/M checkpoint for DNA repair, creating a therapeutic "
        "vulnerability: WEE1 inhibition (adavosertib/AZD1775) forces TP53-mutant cells into "
        "mitosis with unrepaired DNA damage, causing mitotic catastrophe and cell death. "
        "Adavosertib has shown activity in TP53-mutant ovarian cancer, SCLC, and AML, "
        "particularly in combination with DNA-damaging agents (carboplatin, gemcitabine).",
        ["kinase", "cell_cycle", "G2_M_checkpoint", "synthetic_lethality"],
        {"cdk1_km_um": 2},
        ["nucleus"],
        [{"target": "CDK1", "type": "phosphorylates", "note": "Tyr15 inhibitory"},
         {"target": "CDC25", "type": "inhibits", "note": "antagonistic relationship"},
         {"target": "TP53", "type": "activates", "note": "synthetic lethal when lost"}],
        [0.4, 0.7, 0.7])

    log.info("Seeded %d signaling kinase entities", c)
    return c


# ══════════════════════════════════════════════════════════════════════
# TRANSCRIPTION FACTORS
# ══════════════════════════════════════════════════════════════════════

def _seed_transcription_factors(store: EntityStore) -> int:
    P = EntityType.PROTEIN
    c = 0

    c += _add(store, "NF-kappaB (p65/RELA)", P,
        "Nuclear factor kappa-B is a family of dimeric transcription factors (p65/RELA, "
        "p50, p52, c-REL, RELB) that control ~500 genes involved in inflammation, "
        "immunity, cell survival, and proliferation. In the canonical pathway: TNF/IL-1/"
        "TLR ligands activate IKK complex (IKKalpha/IKKbeta/NEMO) → IKKbeta "
        "phosphorylates IkappaBalpha → ubiquitination and proteasomal degradation → "
        "p65/p50 nuclear translocation → transcription of TNF-alpha, IL-6, IL-8, COX-2, "
        "iNOS, BCL-2, BCL-XL, XIAP, cyclin D1, MMP-9, VEGF. Constitutive NF-kappaB "
        "activation drives many cancers through anti-apoptotic and pro-proliferative "
        "gene expression. Therapeutic targeting is challenging due to essential role in "
        "immunity; IKKbeta inhibitors and proteasome inhibitors (bortezomib, which blocks "
        "IkappaB degradation) are used in myeloma.",
        ["transcription_factor", "inflammation", "NF-kB"],
        {"nuclear_translocation_time_min": 15, "target_genes": 500},
        ["cytoplasm", "nucleus"],
        [{"target": "IkappaB", "type": "binds_to", "note": "sequestered in cytoplasm"},
         {"target": "IKKbeta", "type": "activates", "note": "canonical pathway"},
         {"target": "TNF", "type": "activates", "note": "target gene"}],
        [0.9, 0.5, 0.2])

    c += _add(store, "STAT3", P,
        "Signal transducer and activator of transcription 3 (770 amino acids) is an "
        "oncogenic transcription factor constitutively activated in ~70% of solid tumors "
        "and most hematologic malignancies. In normal signaling: cytokine receptor "
        "activation (IL-6, IL-10, IL-21, EGF) → JAK phosphorylation of receptor → STAT3 "
        "SH2 domain docks on phosphotyrosine → JAK phosphorylates STAT3 at Y705 → STAT3 "
        "dimerization (reciprocal SH2-pY705 interaction) → nuclear translocation → binds "
        "GAS (gamma-activated sequence) elements. STAT3 target genes: BCL-2, BCL-XL, MCL-1 "
        "(survival), cyclin D1 (proliferation), VEGF (angiogenesis), MMP-2/9 (invasion), "
        "IL-10, TGF-beta (immunosuppression). In cancer, constitutive STAT3 creates an "
        "immunosuppressive TME by inducing PD-L1 on tumor cells and driving M2 macrophage "
        "polarization. STAT3 inhibitors (napabucasin, TTI-101) are in clinical trials.",
        ["transcription_factor", "oncogene", "STAT"],
        {"y705_phosphorylation_time_min": 5, "target_genes": 200},
        ["cytoplasm", "nucleus"],
        [{"target": "JAK1", "type": "binds_to", "note": "phosphorylation at Y705"},
         {"target": "BCL-2", "type": "activates", "note": "anti-apoptotic"},
         {"target": "VEGF", "type": "activates", "note": "angiogenesis"}],
        [0.6, 0.3, 0.9])

    c += _add(store, "HIF-1alpha", P,
        "Hypoxia-inducible factor 1 alpha (826 amino acids) is the master transcriptional "
        "regulator of cellular oxygen homeostasis. Under normoxia (>5% O2): prolyl "
        "hydroxylases (PHD1/2/3, requiring O2, Fe2+, alpha-ketoglutarate) hydroxylate "
        "HIF-1alpha at Pro402 and Pro564 → von Hippel-Lindau (VHL) E3 ubiquitin ligase "
        "recognizes hydroxylated prolines → polyubiquitination → proteasomal degradation "
        "(half-life ~5 minutes). Under hypoxia (<1-2% O2): PHDs are inactive → HIF-1alpha "
        "stabilizes → dimerizes with HIF-1beta (ARNT) → binds hypoxia response elements "
        "(HREs, core 5'-RCGTG-3') → transcribes ~200 target genes: VEGF (angiogenesis), "
        "GLUT1/HK2/LDHA/PDK1 (glycolytic switch), EPO (erythropoiesis), LOX (ECM "
        "remodeling). Belzutifan (HIF-2alpha inhibitor) treats VHL disease-associated RCC.",
        ["transcription_factor", "hypoxia", "Warburg", "angiogenesis"],
        {"half_life_normoxia_min": 5, "half_life_hypoxia_hours": 4, "target_genes": 200},
        ["cytoplasm", "nucleus"],
        [{"target": "VHL", "type": "binds_to", "note": "ubiquitination under normoxia"},
         {"target": "VEGF", "type": "activates", "note": "HRE in promoter"},
         {"target": "GLUT1", "type": "activates", "note": "glycolytic switch"}],
        [0.7, 0.2, 0.9])

    c += _add(store, "STAT1", P,
        "Signal transducer and activator of transcription 1 (750 amino acids) is the "
        "principal mediator of interferon-gamma signaling and a key anti-tumor transcription "
        "factor. IFN-gamma binding to IFNGR1/IFNGR2 activates JAK1/JAK2, which phosphorylate "
        "STAT1 at Y701. Phosphorylated STAT1 homodimerizes (gamma-activated factor/GAF) and "
        "translocates to the nucleus, binding gamma-activated sequences (GAS) in promoters "
        "of immunostimulatory genes: IRF1, CIITA (MHC-II transactivator), TAP1/2 (antigen "
        "processing), CXCL9/10/11 (T cell recruitment), and iNOS. STAT1 also forms the "
        "ISGF3 complex with STAT2 and IRF9 for type I IFN signaling (binds ISREs). STAT1 "
        "loss in tumors impairs MHC-I expression and antigen presentation, causing resistance "
        "to checkpoint immunotherapy. STAT1 is functionally antagonistic to STAT3 — STAT1 "
        "promotes anti-tumor immunity while STAT3 promotes immunosuppression.",
        ["transcription_factor", "STAT", "IFN_gamma", "anti_tumor"],
        {"y701_phosphorylation_time_min": 5, "target_genes": 150},
        ["cytoplasm", "nucleus"],
        [{"target": "JAK1", "type": "binds_to", "note": "phosphorylation at Y701"},
         {"target": "IRF1", "type": "activates", "note": "target gene"},
         {"target": "STAT3", "type": "inhibits", "note": "functional antagonism"}],
        [0.3, 0.6, 0.9])

    c += _add(store, "NFAT (NFATC1)", P,
        "Nuclear factor of activated T cells (943 amino acids) is a calcium-responsive "
        "transcription factor essential for T cell activation. In resting T cells, NFAT is "
        "heavily phosphorylated (by CK1, GSK3, DYRK kinases) and retained in the cytoplasm. "
        "TCR stimulation → PLCgamma1 → IP3 → ER calcium release → sustained Ca2+ entry via "
        "CRAC/ORAI1 channels → calmodulin activation → calcineurin (serine/threonine "
        "phosphatase) dephosphorylates NFAT → nuclear translocation → binds NFAT response "
        "elements, cooperating with AP-1 (FOS/JUN) to activate IL-2, IL-4, TNF-alpha, and "
        "IFN-gamma transcription. Cyclosporine A and tacrolimus (FK506) are calcineurin "
        "inhibitors that block NFAT dephosphorylation — the basis of transplant "
        "immunosuppression. In T cell exhaustion, NFAT activation without AP-1 partner "
        "drives expression of exhaustion genes (PD-1, TIM-3, LAG-3, TOX).",
        ["transcription_factor", "T_cell", "calcium", "immunosuppression"],
        {"dephosphorylation_time_min": 2, "nuclear_translocation_min": 5},
        ["cytoplasm", "nucleus"],
        [{"target": "calcineurin", "type": "binds_to", "note": "dephosphorylation"},
         {"target": "AP-1", "type": "binds_to", "note": "cooperative at IL-2 promoter"},
         {"target": "IL-2", "type": "activates", "note": "target gene"}],
        [0.4, 0.7, 0.6])

    c += _add(store, "beta-catenin (CTNNB1)", P,
        "Beta-catenin (781 amino acids) has dual roles as a structural component of "
        "adherens junctions (binding E-cadherin cytoplasmic tail) and as the transcriptional "
        "effector of canonical Wnt signaling. In the absence of Wnt ligands, cytoplasmic "
        "beta-catenin is continuously phosphorylated by the destruction complex "
        "(APC-Axin-GSK3beta-CK1alpha): CK1 phosphorylates Ser45, then GSK3beta "
        "phosphorylates Thr41/Ser37/Ser33 → recognized by beta-TrCP E3 ubiquitin ligase → "
        "proteasomal degradation. Wnt binding to Frizzled/LRP5/6 → Dishevelled recruitment → "
        "destruction complex disruption → beta-catenin accumulates → nuclear translocation → "
        "binds TCF/LEF transcription factors → activates MYC, cyclin D1, AXIN2, LGR5 "
        "(stemness). APC mutations (truncating, lost beta-catenin binding) occur in ~80% of "
        "colorectal cancers and ~5% of prostate cancers.",
        ["transcription_factor", "Wnt", "adherens_junction", "stemness"],
        {"half_life_without_wnt_min": 30, "half_life_with_wnt_hours": 6},
        ["cytoplasm", "nucleus", "adherens_junction"],
        [{"target": "TCF/LEF", "type": "binds_to", "note": "nuclear co-activator"},
         {"target": "APC", "type": "binds_to", "note": "destruction complex"},
         {"target": "E-cadherin", "type": "binds_to", "note": "adherens junction"}],
        [0.5, 0.8, 0.4])

    c += _add(store, "IRF3", P,
        "Interferon regulatory factor 3 (427 amino acids) is the key transcription factor "
        "for innate antiviral type I interferon production. IRF3 is activated by the "
        "cGAS-STING-TBK1 pathway (cytoplasmic DNA) and RIG-I/MDA5-MAVS-TBK1 pathway "
        "(cytoplasmic RNA). TBK1 phosphorylates IRF3 at C-terminal serine cluster "
        "(Ser386/Ser396) → IRF3 dimerization → nuclear translocation → binds IRF-E/ISRE "
        "elements in IFN-beta, CXCL10, and ISG15 promoters. IRF3 is the initial IFN-beta "
        "inducer; IFN-beta then upregulates IRF7 (the IFN-alpha amplification factor) "
        "through autocrine IFNAR signaling. In anti-tumor immunity, tumor DNA in the "
        "cytoplasm of dendritic cells activates cGAS-STING→TBK1→IRF3, producing IFN-beta "
        "that is essential for CD8+ T cell cross-priming against tumor antigens.",
        ["transcription_factor", "innate_immunity", "interferon", "cGAS_STING"],
        {"phosphorylation_time_min": 15},
        ["cytoplasm", "nucleus"],
        [{"target": "TBK1", "type": "binds_to", "note": "phosphorylation at Ser386"},
         {"target": "IFN-beta", "type": "activates", "note": "primary target gene"},
         {"target": "IRF7", "type": "activates", "note": "amplification loop"}],
        [0.3, 0.5, 0.9])

    log.info("Seeded %d transcription factor entities", c)
    return c


# ══════════════════════════════════════════════════════════════════════
# APOPTOSIS REGULATORS
# ══════════════════════════════════════════════════════════════════════

def _seed_apoptosis_regulators(store: EntityStore) -> int:
    P = EntityType.PROTEIN
    c = 0

    c += _add(store, "BCL-2", P,
        "B-cell lymphoma 2 (239 amino acids, 26 kDa) is the founding member of the BCL-2 "
        "family of apoptosis regulators and the prototypic anti-apoptotic protein. BCL-2 "
        "resides on the mitochondrial outer membrane (OMM), where it sequesters "
        "pro-apoptotic BH3-only proteins (BIM, BID, PUMA, BAD) in its hydrophobic BH3-"
        "binding groove, preventing them from activating BAX/BAK pore formation. BCL-2 "
        "also directly inhibits BAX/BAK activation. The t(14;18) translocation in "
        "follicular lymphoma places BCL-2 under IgH enhancer control, causing "
        "overexpression and resistance to apoptosis. Venetoclax (ABT-199) is a BH3 "
        "mimetic that binds the BCL-2 groove with sub-nanomolar affinity (Ki <0.01 nM), "
        "displacing BH3-only proteins and triggering BAX/BAK-dependent apoptosis. "
        "FDA-approved for CLL and AML. Tumor lysis syndrome is the major acute risk.",
        ["apoptosis", "anti-apoptotic", "BCL-2_family", "cancer"],
        {"bh3_groove_kd_nm": 0.01, "half_life_hours": 24},
        ["mitochondrial_outer_membrane"],
        [{"target": "BAX", "type": "inhibits", "note": "sequesters"},
         {"target": "BIM", "type": "binds_to", "note": "BH3 groove"},
         {"target": "venetoclax", "type": "binds_to", "kd_nm": 0.01}],
        [0.2, 0.8, 0.3])

    c += _add(store, "BAX", P,
        "BCL-2-associated X protein (192 amino acids, 21 kDa) is the principal "
        "pro-apoptotic effector of the intrinsic (mitochondrial) apoptosis pathway. In "
        "healthy cells, BAX exists as an inactive monomer in the cytoplasm with its "
        "C-terminal transmembrane domain tucked into the BH3-binding groove "
        "(autoinhibited). Apoptotic signals → BH3-only proteins (BIM, tBID, PUMA) bind "
        "BAX, triggering conformational change → N-terminal exposure → mitochondrial "
        "translocation → BAX oligomerization into large pores in the OMM (mitochondrial "
        "outer membrane permeabilization/MOMP) → cytochrome c, SMAC/Diablo, and AIF "
        "release → apoptosome formation → caspase cascade. BAX pores can be 28-100 nm "
        "diameter. BAX and BAK are functionally redundant; loss of both confers "
        "resistance to intrinsic apoptosis.",
        ["apoptosis", "pro-apoptotic", "BCL-2_family", "MOMP"],
        {"pore_diameter_nm": 50, "oligomerization_time_min": 30},
        ["cytoplasm", "mitochondrial_outer_membrane"],
        [{"target": "BCL-2", "type": "binds_to", "note": "inhibited by"},
         {"target": "BIM", "type": "binds_to", "note": "activator"},
         {"target": "cytochrome_c", "type": "activates", "note": "via MOMP"}],
        [0.9, 0.2, 0.2])

    c += _add(store, "Caspase-3", P,
        "Executioner caspase (277 amino acids, 32 kDa pro-form) that cleaves ~500 "
        "substrates during apoptosis, demolishing the cell from within. Caspase-3 exists "
        "as an inactive zymogen (procaspase-3) that is activated by initiator caspases "
        "(caspase-8 from death receptor pathway, caspase-9 from apoptosome) via cleavage "
        "at Asp175, generating p17/p12 heterodimer. Active caspase-3 (cysteine protease, "
        "Asp-Glu-Val-Asp specificity) cleaves: PARP (prevents DNA repair), ICAD/DFF45 "
        "(releases CAD endonuclease for DNA fragmentation), lamin A/C (nuclear envelope "
        "collapse), gelsolin (actin cytoskeleton dismantling), and ROCK1 (membrane "
        "blebbing). Cleaved caspase-3 is the gold standard immunohistochemistry marker "
        "of apoptosis. Caspase-3 also mediates immunogenic cell death, releasing DAMPs "
        "that activate anti-tumor immunity.",
        ["apoptosis", "executioner_caspase", "protease"],
        {"kcat_per_s": 3.5, "km_devd_um": 10, "substrates": 500},
        ["cytoplasm"],
        [{"target": "PARP", "type": "catalyzes", "note": "cleavage, prevents repair"},
         {"target": "ICAD", "type": "catalyzes", "note": "releases CAD DNase"},
         {"target": "Caspase-9", "type": "binds_to", "note": "activated by"}],
        [0.9, 0.3, 0.3])

    c += _add(store, "Cytochrome c", P,
        "Small heme protein (12.4 kDa, 104 amino acids) that has dual roles: (1) "
        "Essential component of the mitochondrial electron transport chain, shuttling "
        "electrons from Complex III (cytochrome bc1) to Complex IV (cytochrome c oxidase) "
        "in the intermembrane space; (2) Key initiator of intrinsic apoptosis when "
        "released from mitochondria during MOMP. Released cytochrome c binds APAF-1 "
        "(apoptotic protease-activating factor 1) in the cytoplasm, triggering dATP-"
        "dependent conformational change and APAF-1 oligomerization into the heptameric "
        "wheel-shaped apoptosome (~700 kDa). The apoptosome recruits and activates "
        "procaspase-9 via CARD-CARD interaction, initiating the caspase cascade. The "
        "amount of cytochrome c released is switch-like (all-or-none) due to positive "
        "feedback between caspases and additional MOMP.",
        ["apoptosis", "ETC", "apoptosome", "mitochondria"],
        {"electron_transfer_rate_per_s": 300, "molecular_weight_kda": 12.4},
        ["mitochondrial_intermembrane_space", "cytoplasm"],
        [{"target": "APAF-1", "type": "binds_to", "note": "apoptosome formation"},
         {"target": "Complex III", "type": "binds_to", "note": "electron shuttle"},
         {"target": "Complex IV", "type": "binds_to", "note": "electron delivery"}],
        [0.8, 0.3, 0.3])

    c += _add(store, "Caspase-8", P,
        "Initiator caspase (479 amino acids, 55 kDa pro-form) that triggers the extrinsic "
        "(death receptor) apoptosis pathway. Death ligands (FasL, TRAIL, TNF) bind their "
        "receptors (Fas, DR4/5, TNFR1) → FADD adaptor recruitment → procaspase-8 binds via "
        "DED-DED interaction → DISC (death-inducing signaling complex) formation → proximity-"
        "induced dimerization and auto-proteolysis → active caspase-8 (p18/p10 heterodimer). "
        "Active caspase-8 directly cleaves and activates executioner caspases-3/7 (type I cells, "
        "e.g., lymphocytes) or cleaves BID to tBID (type II cells, e.g., hepatocytes), which "
        "amplifies the signal through mitochondrial MOMP. Caspase-8 also has non-apoptotic "
        "roles: prevents necroptosis by cleaving RIPK1/RIPK3, regulates inflammasome "
        "activation, and is required for lymphocyte proliferation.",
        ["apoptosis", "initiator_caspase", "death_receptor", "DISC"],
        {"kcat_per_s": 1.5},
        ["cytoplasm", "DISC"],
        [{"target": "Caspase-3", "type": "activates", "note": "direct cleavage"},
         {"target": "BID", "type": "catalyzes", "note": "→ tBID"},
         {"target": "RIPK1", "type": "catalyzes", "note": "prevents necroptosis"}],
        [0.8, 0.4, 0.2])

    c += _add(store, "Caspase-9", P,
        "Initiator caspase (416 amino acids, 46 kDa pro-form) of the intrinsic "
        "(mitochondrial) apoptosis pathway. After MOMP releases cytochrome c, cytochrome c "
        "binds APAF-1 → dATP-dependent conformational change → APAF-1 oligomerization into "
        "the heptameric apoptosome (wheel-shaped, ~1.4 MDa) → procaspase-9 CARD domain binds "
        "APAF-1 CARD hub → proximity-induced dimerization activates caspase-9 (no "
        "auto-cleavage required). Active caspase-9 cleaves and activates executioner "
        "caspases-3 and -7. Caspase-9 activity is directly inhibited by XIAP (BIR3 domain "
        "binds caspase-9 dimerization interface). SMAC/Diablo released during MOMP displaces "
        "XIAP, allowing caspase-9 to function. Caspase-9 is the rate-limiting step of "
        "apoptosome-mediated cell death.",
        ["apoptosis", "initiator_caspase", "apoptosome", "intrinsic_pathway"],
        {"kcat_per_s": 0.5},
        ["cytoplasm", "apoptosome"],
        [{"target": "APAF-1", "type": "binds_to", "note": "CARD interaction"},
         {"target": "Caspase-3", "type": "activates", "note": "cleavage at Asp175"},
         {"target": "XIAP", "type": "binds_to", "note": "direct inhibition"}],
        [0.7, 0.4, 0.3])

    c += _add(store, "MCL-1", P,
        "Myeloid cell leukemia 1 (350 amino acids, 40 kDa) is an anti-apoptotic BCL-2 "
        "family member with the unique property of extremely short half-life (~30 minutes "
        "to 3 hours), making its levels acutely responsive to stress signals. MCL-1 is "
        "regulated at every level: transcription (STAT3, NF-kappaB), mRNA stability "
        "(microRNAs), translation (mTORC1/4E-BP1), and protein stability (GSK3beta "
        "phosphorylation at Ser159 → FBXW7/beta-TrCP ubiquitination → proteasomal "
        "degradation). MCL-1 sequesters pro-apoptotic BH3-only proteins (BIM, PUMA, NOXA) "
        "and directly inhibits BAK on the OMM. MCL-1 is amplified in ~10% of all cancers "
        "(most frequently in breast, lung, and AML) and is a major resistance factor to "
        "venetoclax (BCL-2 inhibitor). MCL-1-selective inhibitors (AMG-176, S64315/MIK665) "
        "are in clinical trials for hematologic malignancies.",
        ["apoptosis", "anti-apoptotic", "BCL-2_family", "resistance"],
        {"half_life_hours": 1, "degradation_rate_high": True},
        ["mitochondrial_outer_membrane"],
        [{"target": "BAK", "type": "inhibits"},
         {"target": "BIM", "type": "binds_to", "note": "BH3 groove"},
         {"target": "NOXA", "type": "binds_to", "note": "selective BH3-only"}],
        [0.3, 0.8, 0.4])

    c += _add(store, "PUMA (BBC3)", P,
        "p53-upregulated modulator of apoptosis (193 amino acids) is a BH3-only "
        "BCL-2 family member and the principal mediator of p53-dependent apoptosis. PUMA "
        "is transcriptionally induced by p53 (p53 response elements in promoter) within "
        "hours of DNA damage and by FOXO3 during growth factor withdrawal. PUMA binds "
        "all five anti-apoptotic BCL-2 family members (BCL-2, BCL-XL, BCL-W, MCL-1, A1) "
        "with high affinity via its BH3 domain, displacing sequestered BAX and BAK and "
        "enabling MOMP. PUMA can also directly activate BAX through transient BH3 domain "
        "interaction. PUMA is the dominant p53 effector for apoptosis in most cell types: "
        "PUMA knockout mice are resistant to irradiation-induced lymphocyte death and "
        "p53-dependent intestinal stem cell apoptosis.",
        ["apoptosis", "BH3_only", "p53", "pro-apoptotic"],
        {"binding_all_antiapoptotic": True},
        ["cytoplasm", "mitochondrial_outer_membrane"],
        [{"target": "BCL-2", "type": "binds_to", "note": "BH3 displacement"},
         {"target": "MCL-1", "type": "binds_to", "note": "BH3 displacement"},
         {"target": "BAX", "type": "activates", "note": "indirect + direct"}],
        [0.9, 0.2, 0.3])

    c += _add(store, "XIAP (BIRC4)", P,
        "X-linked inhibitor of apoptosis protein (497 amino acids) is the only IAP family "
        "member that directly inhibits caspases through physical binding. XIAP BIR2 domain "
        "binds the active site of executioner caspases-3 and -7 (blocking substrate access), "
        "while BIR3 domain binds caspase-9 dimerization interface (preventing apoptosome "
        "activation). XIAP RING domain has E3 ubiquitin ligase activity, targeting caspases "
        "for degradation. XIAP is antagonized by SMAC/Diablo and Omi/HtrA2, both released "
        "from mitochondria during MOMP: SMAC N-terminal AVPI motif competes with caspases "
        "for XIAP BIR groove binding. SMAC mimetics (birinapant, LCL161) are synthetic "
        "peptidomimetics that displace XIAP from caspases and also induce degradation of "
        "cIAP1/2 (activating non-canonical NF-kappaB and TNF-dependent apoptosis).",
        ["apoptosis", "IAP", "caspase_inhibitor"],
        {},
        ["cytoplasm"],
        [{"target": "Caspase-3", "type": "inhibits", "note": "BIR2 blocks active site"},
         {"target": "Caspase-9", "type": "inhibits", "note": "BIR3 blocks dimerization"},
         {"target": "SMAC", "type": "binds_to", "note": "antagonist"}],
        [0.2, 0.9, 0.3])

    log.info("Seeded %d apoptosis regulator entities", c)
    return c


# ══════════════════════════════════════════════════════════════════════
# ECM COMPONENTS
# ══════════════════════════════════════════════════════════════════════

def _seed_ecm_components(store: EntityStore) -> int:
    P = EntityType.PROTEIN
    c = 0

    c += _add(store, "Collagen I", P,
        "Type I collagen (COL1A1/COL1A2) is the most abundant protein in the human body "
        "(~25% of total protein), forming the structural framework of bone, skin, tendon, "
        "cornea, and interstitial connective tissue. Triple helix of two alpha-1(I) and "
        "one alpha-2(I) chains, each ~1000 amino acids with characteristic Gly-X-Y repeat "
        "(glycine at every third position, X often proline, Y often hydroxyproline). "
        "Procollagen is secreted and processed by N/C-terminal propeptidases, then "
        "self-assembles into fibrils (67 nm D-period banding) cross-linked by lysyl "
        "oxidase. In cancer, tumor-associated collagen signatures (TACS-3: aligned "
        "perpendicular fibers at tumor boundary) promote invasion along collagen tracks. "
        "Excessive collagen deposition (desmoplasia, fibrosis) in tumor stroma creates "
        "a physical barrier to drug delivery and T cell infiltration.",
        ["ECM", "collagen", "structural", "fibrosis"],
        {"fibril_tensile_strength_MPa": 50, "d_period_nm": 67},
        ["extracellular", "ECM"],
        [{"target": "integrin_a1b1", "type": "binds_to"},
         {"target": "integrin_a2b1", "type": "binds_to"},
         {"target": "MMP-1", "type": "binds_to", "note": "collagenase cleavage"}],
        [0.8, 0.7, 0.6])

    c += _add(store, "MMP-9 (Gelatinase B)", P,
        "Matrix metalloproteinase 9 (707 amino acids, 92 kDa) is a zinc-dependent "
        "endopeptidase stored in tertiary (gelatinase) granules of neutrophils and "
        "produced by macrophages, fibroblasts, and tumor cells. MMP-9 degrades type IV "
        "collagen (major component of basement membranes), gelatin, elastin, and "
        "fibrillin, facilitating cell migration through ECM barriers. In cancer: MMP-9 "
        "promotes invasion (basement membrane degradation), angiogenesis (liberates "
        "matrix-bound VEGF), and immune evasion (cleaves IL-2Ralpha from T cells). "
        "MMP-9 is secreted as pro-MMP-9 (92 kDa), activated by MMP-3, plasmin, or "
        "neutrophil elastase to 82 kDa active form. Neutrophil MMP-9 is stored without "
        "TIMP-1 (unlike other cell sources), allowing rapid unregulated activity upon "
        "degranulation. Broad-spectrum MMP inhibitors (marimastat, batimastat) failed "
        "in cancer trials due to musculoskeletal toxicity.",
        ["ECM", "MMP", "invasion", "angiogenesis"],
        {"kcat_per_s": 10, "km_gelatin_um": 5},
        ["extracellular"],
        [{"target": "collagen_IV", "type": "catalyzes", "note": "degradation"},
         {"target": "VEGF", "type": "activates", "note": "liberation from ECM"},
         {"target": "TIMP-1", "type": "binds_to", "note": "endogenous inhibitor"}],
        [0.7, 0.5, 0.4])

    c += _add(store, "Hyaluronan", P,
        "Hyaluronic acid is a non-sulfated glycosaminoglycan (GAG) composed of repeating "
        "disaccharides of N-acetylglucosamine and glucuronic acid, reaching molecular "
        "weights of 10^6 to 10^7 Da and lengths up to 25 um. Synthesized by hyaluronan "
        "synthases (HAS1-3) at the plasma membrane and extruded directly into the "
        "extracellular space. Hyaluronan binds CD44 (principal receptor, cell adhesion, "
        "migration, signaling) and RHAMM (intracellular signaling, mitotic spindle). "
        "In the tumor microenvironment, hyaluronan is massively overproduced (particularly "
        "in pancreatic cancer), creating a viscous matrix that compresses blood vessels, "
        "impairs drug delivery, and promotes cancer stemness via CD44 signaling. "
        "Hyaluronidase (PEGPH20/PVHA) degrades tumor hyaluronan and was tested in "
        "pancreatic cancer (HALO trial) to improve drug penetration.",
        ["ECM", "GAG", "CD44", "tumor_stroma"],
        {"molecular_weight_da": 5000000, "length_um": 10},
        ["extracellular", "ECM"],
        [{"target": "CD44", "type": "binds_to", "note": "cell adhesion/signaling"},
         {"target": "RHAMM", "type": "binds_to", "note": "migration"},
         {"target": "hyaluronidase", "type": "binds_to", "note": "degradation"}],
        [0.5, 0.8, 0.9])

    c += _add(store, "Collagen IV", P,
        "Type IV collagen is the primary structural component of all basement membranes — "
        "thin, specialized ECM sheets underlying epithelia, endothelia, and surrounding muscle "
        "and fat cells. Unlike fibrillar collagens (I, II, III), collagen IV forms a "
        "non-fibrillar, sheet-like network through end-to-end and lateral associations of its "
        "triple-helical protomers (alpha1-alpha6 chains). The network provides mechanical "
        "support, cell attachment (via integrin alpha1beta1, alpha2beta1), and serves as a "
        "size-selective filtration barrier (e.g., glomerular basement membrane in kidney). "
        "Basement membrane degradation by MMP-2 and MMP-9 (type IV collagenases) is the "
        "critical step in tumor invasion and metastasis — cancer cells must breach the "
        "basement membrane to invade stroma and enter blood/lymphatic vessels. Goodpasture "
        "syndrome: autoantibodies against the alpha3(IV) NC1 domain cause glomerulonephritis.",
        ["ECM", "collagen", "basement_membrane", "invasion"],
        {},
        ["extracellular", "basement_membrane"],
        [{"target": "integrin_a1b1", "type": "binds_to"},
         {"target": "MMP-2", "type": "binds_to", "note": "type IV collagenase"},
         {"target": "MMP-9", "type": "binds_to", "note": "gelatinase B cleavage"}],
        [0.7, 0.7, 0.5])

    c += _add(store, "Laminin", P,
        "Large (400-900 kDa) heterotrimeric glycoprotein (alpha/beta/gamma chains) that is the "
        "first ECM component deposited during embryonic development and the organizing foundation "
        "of all basement membranes. Laminin self-polymerizes through LN domain interactions to "
        "form a network that is subsequently reinforced by collagen IV crosslinking. Cell binding "
        "occurs through integrin receptors: alpha6beta1 (primary laminin receptor on epithelial "
        "cells), alpha6beta4 (hemidesmosomes in skin, strong adhesion), alpha3beta1 (kidney, "
        "lung), and alpha7beta1 (muscle). Laminin-332 (formerly laminin-5) is the critical "
        "adhesion substrate for epidermal keratinocytes; its loss causes junctional epidermolysis "
        "bullosa (lethal skin blistering). In cancer, laminin fragments generated by MMP cleavage "
        "can promote migration and angiogenesis.",
        ["ECM", "basement_membrane", "laminin", "cell_adhesion"],
        {"molecular_weight_kda": 800},
        ["extracellular", "basement_membrane"],
        [{"target": "integrin_a6b1", "type": "binds_to", "note": "epithelial adhesion"},
         {"target": "integrin_a6b4", "type": "binds_to", "note": "hemidesmosomes"},
         {"target": "collagen_IV", "type": "binds_to", "note": "BM network"}],
        [0.6, 0.5, 0.7])

    c += _add(store, "Fibronectin", P,
        "Large (440 kDa homodimer) multidomain glycoprotein that connects cells to the "
        "extracellular matrix and plays essential roles in wound healing, embryonic development, "
        "and blood clotting. Fibronectin contains the RGD (Arg-Gly-Asp) tripeptide motif in its "
        "type III10 repeat — the prototypic integrin-binding sequence recognized by alpha5beta1 "
        "(primary FN receptor), alphavbeta3, and alphavbeta1 integrins. Exists in two forms: "
        "plasma fibronectin (soluble dimer produced by hepatocytes, circulates at ~300 ug/mL) "
        "and cellular fibronectin (insoluble fibrils assembled by fibroblasts into the ECM "
        "via integrin-mediated tension). Fibronectin fibrillogenesis requires cell-generated "
        "mechanical force to expose cryptic self-assembly sites. In wound healing, fibronectin "
        "provides the provisional matrix for cell migration. In cancer, fibronectin in the "
        "pre-metastatic niche attracts bone marrow-derived cells that prepare distant sites "
        "for metastatic colonization.",
        ["ECM", "fibronectin", "RGD", "wound_healing", "integrin"],
        {"molecular_weight_kda": 440, "serum_concentration_ug_ml": 300},
        ["extracellular", "ECM", "blood"],
        [{"target": "integrin_a5b1", "type": "binds_to", "note": "RGD recognition"},
         {"target": "integrin_avb3", "type": "binds_to", "note": "RGD"},
         {"target": "fibrin", "type": "binds_to", "note": "wound clot matrix"}],
        [0.7, 0.5, 0.4])

    c += _add(store, "MMP-2 (Gelatinase A)", P,
        "Matrix metalloproteinase 2 (72 kDa, 660 amino acids) is a zinc-dependent gelatinase "
        "that degrades type IV collagen — the principal structural component of basement "
        "membranes. MMP-2 is constitutively expressed by fibroblasts, endothelial cells, and "
        "many tumor cells. Pro-MMP-2 (72 kDa) is activated at the cell surface by a unique "
        "mechanism: MT1-MMP (MMP-14, membrane-type MMP) cleaves pro-MMP-2 in complex with "
        "TIMP-2, generating the 62 kDa active form. This cell-surface activation focuses "
        "MMP-2 activity at the leading edge of migrating cells. MMP-2 also cleaves laminin, "
        "fibronectin, elastin, and growth factors (TGF-beta activation, VEGF liberation from "
        "ECM). In invasion: MMP-2 at invadopodia (actin-rich membrane protrusions) degrades "
        "basement membrane, the rate-limiting step of metastatic dissemination.",
        ["ECM", "MMP", "invasion", "basement_membrane"],
        {"kcat_per_s": 8, "km_collagen_iv_um": 3},
        ["extracellular", "invadopodia"],
        [{"target": "collagen_IV", "type": "catalyzes", "note": "BM degradation"},
         {"target": "MT1-MMP", "type": "binds_to", "note": "cell-surface activation"},
         {"target": "TIMP-2", "type": "binds_to", "note": "inhibitor + activator"}],
        [0.6, 0.5, 0.4])

    c += _add(store, "TIMP-1", P,
        "Tissue inhibitor of metalloproteinases 1 (28 kDa, 207 amino acids) is the primary "
        "endogenous inhibitor of MMP-9 (gelatinase B) and most other MMPs. TIMP-1 inhibits "
        "MMPs by binding the active site zinc in a 1:1 stoichiometric complex with Ki values "
        "in the sub-nanomolar range (Ki ~0.1-1 nM for most MMPs). TIMP-1 also has "
        "MMP-independent functions: it binds CD63 on the cell surface, activating FAK/PI3K "
        "survival signaling, and serves as a growth factor for erythroid progenitors. Elevated "
        "serum TIMP-1 is a poor prognostic marker in colorectal, breast, and gastric cancers — "
        "paradoxically, because TIMP-1 promotes tumor cell survival through MMP-independent "
        "anti-apoptotic signaling rather than inhibiting invasion. TIMP-2 has a unique dual "
        "role: inhibits most MMPs but is required for MT1-MMP-mediated pro-MMP-2 activation.",
        ["ECM", "TIMP", "MMP_inhibitor"],
        {"mmp9_ki_nm": 0.5},
        ["extracellular"],
        [{"target": "MMP-9", "type": "inhibits", "kd_nm": 0.5},
         {"target": "CD63", "type": "binds_to", "note": "survival signaling"},
         {"target": "FAK", "type": "activates", "note": "MMP-independent"}],
        [0.5, 0.6, 0.5])

    log.info("Seeded %d ECM component entities", c)
    return c
