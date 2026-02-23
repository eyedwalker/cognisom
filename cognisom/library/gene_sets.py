"""
Curated Gene Sets and Entity Lists
====================================

Pre-built lists for one-click population of the entity database.
Each set targets a specific biological domain with genes, drugs,
and pathways that are well-characterized in public databases.

Usage:
    from cognisom.library.gene_sets import PROSTATE_CANCER_COMPREHENSIVE
    from cognisom.library.bulk_import import BulkImporter

    importer = BulkImporter(store)
    report = importer.import_gene_list(PROSTATE_CANCER_COMPREHENSIVE)
"""

# ── Prostate Cancer ──────────────────────────────────────────────────

PROSTATE_CANCER_COMPREHENSIVE = [
    # AR signaling axis
    "AR", "FOXA1", "HOXB13", "NKX3-1", "KLK3",  # PSA
    "TMPRSS2", "ERG", "ETV1", "ETV4", "ETV5",
    # Tumor suppressors
    "TP53", "RB1", "PTEN", "BRCA1", "BRCA2",
    "ATM", "CDH1", "CDKN1B", "APC", "SMAD4",
    # Oncogenes
    "MYC", "PIK3CA", "PIK3CB", "AKT1", "BRAF",
    "KRAS", "NRAS", "HRAS", "RAF1", "MTOR",
    # DNA repair
    "MSH2", "MSH6", "MLH1", "PMS2", "PALB2",
    "RAD51", "CHEK2", "NBN", "CDK12",
    # Cell cycle
    "CDK4", "CDK6", "CCND1", "CCNE1", "E2F1",
    "MDM2", "MDM4", "CDKN2A", "CDKN2B",
    # Epigenetic
    "EZH2", "KDM1A", "DNMT3A", "DNMT3B", "KMT2A",
    # Immune
    "CD274",  # PD-L1
    "PDCD1",  # PD-1
    "CTLA4", "LAG3", "TIGIT",
    # Metabolism
    "FASN", "ACLY", "SCD", "HK2", "LDHA",
    # Growth factors
    "EGFR", "ERBB2", "FGFR1", "FGFR2", "MET",
    "IGF1R", "VEGFA", "KDR",
    # WNT/Notch
    "CTNNB1", "WNT5A", "NOTCH1", "DLL3",
    # Neuroendocrine
    "SYP", "CHGA", "NCAM1", "SOX2", "ASCL1",
    "INSM1", "POU3F2",
    # Steroidogenesis
    "CYP17A1", "SRD5A1", "SRD5A2", "HSD3B1",
    "CYP11A1", "STAR",
]

PROSTATE_CANCER_DRUGS = [
    "Enzalutamide", "Abiraterone", "Docetaxel", "Cabazitaxel",
    "Apalutamide", "Darolutamide", "Olaparib", "Rucaparib",
    "Sipuleucel-T", "Radium-223", "Lutetium-177 PSMA",
    "Pembrolizumab", "Bicalutamide", "Flutamide",
    "Leuprolide", "Goserelin", "Degarelix",
    "Cabozantinib", "Talazoparib", "Niraparib",
]

PROSTATE_KEGG_PATHWAYS = [
    "hsa05215",  # Prostate cancer
    "hsa04115",  # p53 signaling pathway
    "hsa04151",  # PI3K-Akt signaling pathway
    "hsa04010",  # MAPK signaling pathway
    "hsa04110",  # Cell cycle
    "hsa04310",  # Wnt signaling pathway
    "hsa04330",  # Notch signaling pathway
    "hsa04350",  # TGF-beta signaling pathway
]

# ── Pan-Cancer Drivers ──────────────────────────────────────────────

PAN_CANCER_DRIVERS = [
    # Tumor suppressors (COSMIC Cancer Gene Census Tier 1)
    "TP53", "RB1", "PTEN", "APC", "BRCA1", "BRCA2",
    "VHL", "WT1", "NF1", "NF2", "SMAD4", "SMARCB1",
    "STK11", "TSC1", "TSC2", "CDH1", "CDKN2A",
    "BAP1", "ARID1A", "ARID2", "SETD2",
    # Oncogenes
    "KRAS", "NRAS", "HRAS", "BRAF", "PIK3CA",
    "MYC", "MYCN", "MYCL", "EGFR", "ERBB2",
    "ALK", "ROS1", "RET", "MET", "FGFR1", "FGFR2",
    "FGFR3", "KIT", "PDGFRA", "ABL1", "JAK2",
    "FLT3", "NOTCH1", "IDH1", "IDH2",
    "AKT1", "MTOR", "CCND1", "CDK4", "CDK6",
    "MDM2", "MDM4", "BCL2", "MCL1",
    # Epigenetic regulators
    "EZH2", "DNMT3A", "TET2", "ASXL1", "KMT2A",
    "KMT2D", "CREBBP", "EP300",
    # DNA damage repair
    "ATM", "ATR", "CHEK1", "CHEK2", "PALB2",
    "RAD51", "FANCA", "FANCC",
    # Immune evasion
    "CD274", "PDCD1LG2", "B2M", "JAK1",
    # Splicing
    "SF3B1", "U2AF1", "SRSF2",
]

# ── Breast Cancer ──────────────────────────────────────────────────

BREAST_CANCER_CORE = [
    "BRCA1", "BRCA2", "TP53", "ERBB2", "ESR1",
    "PGR", "PIK3CA", "AKT1", "PTEN", "CDH1",
    "GATA3", "MAP3K1", "MAP2K4", "CBFB", "RUNX1",
    "TBX3", "CDK4", "CDK6", "CCND1", "RB1",
    "MYC", "FGFR1", "NOTCH1", "SF3B1", "PALB2",
    "ATM", "CHEK2", "FOXA1", "KMT2C", "NF1",
    "ARID1A", "CDKN2A", "MDM2", "EGFR",
    "MET", "VEGFA", "MTOR", "RAD51",
]

BREAST_CANCER_DRUGS = [
    "Tamoxifen", "Letrozole", "Anastrozole", "Exemestane",
    "Trastuzumab", "Pertuzumab", "Lapatinib", "Tucatinib",
    "Palbociclib", "Ribociclib", "Abemaciclib",
    "Olaparib", "Talazoparib",
    "Capecitabine", "Doxorubicin", "Paclitaxel",
    "Atezolizumab", "Pembrolizumab",
    "Alpelisib", "Everolimus",
]

# ── DNA Repair ──────────────────────────────────────────────────────

DNA_REPAIR_GENES = [
    # Homologous recombination
    "BRCA1", "BRCA2", "RAD51", "RAD51B", "RAD51C",
    "RAD51D", "PALB2", "BARD1", "XRCC2", "XRCC3",
    # Mismatch repair
    "MSH2", "MSH3", "MSH6", "MLH1", "MLH3",
    "PMS1", "PMS2", "EPCAM",
    # Nucleotide excision repair
    "ERCC1", "ERCC2", "ERCC3", "ERCC4", "XPA", "XPC",
    # Base excision repair
    "OGG1", "MUTYH", "APEX1", "XRCC1", "POLB",
    # DDR signaling
    "ATM", "ATR", "CHEK1", "CHEK2", "NBN", "MRE11",
    "RAD50", "TP53BP1", "RNF168", "CDK12",
]

# ── Immune Checkpoint ────────────────────────────────────────────────

IMMUNE_CHECKPOINT_GENES = [
    "CD274",    # PD-L1
    "PDCD1",    # PD-1
    "PDCD1LG2", # PD-L2
    "CTLA4",
    "LAG3",
    "TIGIT",
    "TIM3",     # HAVCR2
    "HAVCR2",
    "VISTA",    # VSIR
    "VSIR",
    "BTLA",
    "CD47",
    "SIRPA",
    "CD80",
    "CD86",
    "ICOS",
    "TNFRSF18", # GITR
    "TNFRSF4",  # OX40
    "TNFRSF9",  # 4-1BB
    "IDO1",
]

# ── Kinase Targets (Druggable) ──────────────────────────────────────

KINASE_TARGETS = [
    # Receptor tyrosine kinases
    "EGFR", "ERBB2", "ERBB3", "ERBB4",
    "FGFR1", "FGFR2", "FGFR3", "FGFR4",
    "MET", "ALK", "ROS1", "RET", "NTRK1",
    "KIT", "PDGFRA", "PDGFRB", "FLT3",
    "KDR", "FLT1", "IGF1R", "INSR",
    "AXL", "MERTK",
    # Cytoplasmic kinases
    "ABL1", "ABL2", "JAK1", "JAK2", "JAK3",
    "SRC", "BTK", "SYK", "LCK",
    # Serine/threonine kinases
    "BRAF", "RAF1", "MAP2K1", "MAP2K2",
    "MAPK1", "MAPK3",
    "AKT1", "AKT2", "MTOR",
    "CDK4", "CDK6", "CDK2", "CDK1",
    "PLK1", "AURKA", "AURKB",
    "CHEK1", "CHEK2", "WEE1",
    "PIK3CA", "PIK3CB", "PIK3CD",
]

# ── Metabolic Enzymes ───────────────────────────────────────────────

METABOLIC_ENZYMES = [
    # Glycolysis
    "HK1", "HK2", "GPI", "PFKM", "ALDOA",
    "TPI1", "GAPDH", "PGK1", "ENO1", "PKM",
    # TCA cycle
    "CS", "ACO2", "IDH1", "IDH2", "OGDH",
    "SUCLA2", "SDHA", "FH", "MDH2",
    # Fatty acid synthesis
    "FASN", "ACACA", "ACLY", "SCD",
    # Amino acid metabolism
    "GLS", "GLUD1", "GOT1", "GOT2",
    # Nucleotide synthesis
    "DHFR", "TYMS", "RRM1", "RRM2",
    # Oxidative phosphorylation
    "NDUFS1", "UQCRC1", "COX4I1", "ATP5F1A",
    # Pentose phosphate
    "G6PD", "PGD", "TKT",
]

# ── FDA-Approved Cancer Drugs ───────────────────────────────────────

FDA_APPROVED_CANCER_DRUGS = [
    # Immunotherapy
    "Pembrolizumab", "Nivolumab", "Atezolizumab",
    "Durvalumab", "Avelumab", "Ipilimumab",
    "Cemiplimab",
    # Targeted therapy
    "Imatinib", "Erlotinib", "Gefitinib", "Osimertinib",
    "Crizotinib", "Alectinib", "Lorlatinib",
    "Vemurafenib", "Dabrafenib", "Trametinib",
    "Sotorasib", "Adagrasib",
    "Palbociclib", "Ribociclib", "Abemaciclib",
    "Olaparib", "Rucaparib", "Niraparib", "Talazoparib",
    "Venetoclax", "Ibrutinib", "Acalabrutinib",
    "Trastuzumab", "Bevacizumab", "Cetuximab",
    "Lenvatinib", "Sorafenib", "Sunitinib",
    "Cabozantinib", "Regorafenib", "Axitinib",
    "Alpelisib", "Everolimus", "Temsirolimus",
    "Larotrectinib", "Entrectinib",
    # Chemotherapy
    "Cisplatin", "Carboplatin", "Oxaliplatin",
    "Docetaxel", "Paclitaxel", "Cabazitaxel",
    "Doxorubicin", "Epirubicin",
    "5-Fluorouracil", "Capecitabine", "Gemcitabine",
    "Irinotecan", "Topotecan", "Etoposide",
    "Cyclophosphamide", "Temozolomide",
    "Methotrexate", "Pemetrexed",
    # Hormonal
    "Tamoxifen", "Letrozole", "Anastrozole",
    "Enzalutamide", "Abiraterone", "Apalutamide",
]

# ── KEGG Cancer Pathways ───────────────────────────────────────────

KEGG_CANCER_PATHWAYS = [
    "hsa05200",  # Pathways in cancer
    "hsa05215",  # Prostate cancer
    "hsa05224",  # Breast cancer
    "hsa05210",  # Colorectal cancer
    "hsa05225",  # Hepatocellular carcinoma
    "hsa05226",  # Gastric cancer
    "hsa05222",  # Small cell lung cancer
    "hsa05223",  # Non-small cell lung cancer
    "hsa05214",  # Glioma
    "hsa05221",  # Acute myeloid leukemia
    "hsa05220",  # Chronic myeloid leukemia
    "hsa05218",  # Melanoma
    "hsa05211",  # Renal cell carcinoma
    "hsa05219",  # Bladder cancer
    "hsa05213",  # Endometrial cancer
    "hsa05216",  # Thyroid cancer
    "hsa05217",  # Basal cell carcinoma
    "hsa05212",  # Pancreatic cancer
    # Signaling pathways
    "hsa04010",  # MAPK signaling
    "hsa04151",  # PI3K-Akt signaling
    "hsa04110",  # Cell cycle
    "hsa04115",  # p53 signaling
    "hsa04310",  # Wnt signaling
    "hsa04330",  # Notch signaling
    "hsa04064",  # NF-kappa B signaling
    "hsa04210",  # Apoptosis
    "hsa04370",  # VEGF signaling
    "hsa04620",  # Toll-like receptor signaling
    "hsa04630",  # JAK-STAT signaling
]

# ── Predefined Import Sets ──────────────────────────────────────────

IMPORT_SETS = {
    "Prostate Cancer (80 genes + 20 drugs)": {
        "genes": PROSTATE_CANCER_COMPREHENSIVE,
        "drugs": PROSTATE_CANCER_DRUGS,
        "pathways": PROSTATE_KEGG_PATHWAYS,
        "description": "Comprehensive prostate cancer gene panel including AR signaling, tumor suppressors, oncogenes, DNA repair, and neuroendocrine markers.",
    },
    "Pan-Cancer Drivers (80 genes)": {
        "genes": PAN_CANCER_DRIVERS,
        "drugs": [],
        "pathways": [],
        "description": "COSMIC Cancer Gene Census Tier 1 genes — the most commonly mutated driver genes across all cancer types.",
    },
    "Breast Cancer (38 genes + 20 drugs)": {
        "genes": BREAST_CANCER_CORE,
        "drugs": BREAST_CANCER_DRUGS,
        "pathways": [],
        "description": "Core breast cancer gene panel with hormonal, HER2, and triple-negative pathway genes.",
    },
    "DNA Repair (35 genes)": {
        "genes": DNA_REPAIR_GENES,
        "drugs": [],
        "pathways": [],
        "description": "Homologous recombination, mismatch repair, NER, BER, and DDR signaling genes.",
    },
    "Immune Checkpoints (20 genes)": {
        "genes": IMMUNE_CHECKPOINT_GENES,
        "drugs": [],
        "pathways": [],
        "description": "Immune checkpoint and co-stimulatory receptors/ligands for immunotherapy.",
    },
    "Kinase Targets (50 genes)": {
        "genes": KINASE_TARGETS,
        "drugs": [],
        "pathways": [],
        "description": "Druggable kinases (RTKs, cytoplasmic, Ser/Thr) that are therapeutic targets.",
    },
    "FDA Cancer Drugs (70 drugs)": {
        "genes": [],
        "drugs": FDA_APPROVED_CANCER_DRUGS,
        "pathways": [],
        "description": "FDA-approved cancer drugs including immunotherapy, targeted, chemo, and hormonal agents.",
    },
    "Full Catalog (all genes + drugs)": {
        "genes": list(set(
            PROSTATE_CANCER_COMPREHENSIVE + PAN_CANCER_DRIVERS +
            BREAST_CANCER_CORE + DNA_REPAIR_GENES +
            IMMUNE_CHECKPOINT_GENES + KINASE_TARGETS +
            METABOLIC_ENZYMES
        )),
        "drugs": list(set(
            PROSTATE_CANCER_DRUGS + BREAST_CANCER_DRUGS +
            FDA_APPROVED_CANCER_DRUGS
        )),
        "pathways": KEGG_CANCER_PATHWAYS,
        "description": "All curated gene sets combined — ~350 unique genes, ~80 drugs, 30 pathways.",
    },
}
